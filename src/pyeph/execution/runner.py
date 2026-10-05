"""Compile a method's pure step and stream bounded blocks of observations."""

from dataclasses import dataclass, field, replace
from typing import Any

import jax
import numpy as np

from pyeph.core._configuration import boolean_scalar, integer_scalar
from pyeph.execution import _blocks, _validation
from pyeph.integrators.krylov import LanczosOptions
from pyeph.observables.population import ElectronicPopulation


@dataclass(frozen=True)
class Execution:
    jit: bool = True
    chunk_size: int = 128
    save_every: int = 1
    check_finite: bool = True
    allow_host_callbacks: bool = False
    verify_external_gradients: bool = False

    def __post_init__(self):
        for name in ("jit", "check_finite", "allow_host_callbacks", "verify_external_gradients"):
            object.__setattr__(self, name, boolean_scalar(getattr(self, name), name))
        for name in ("chunk_size", "save_every"):
            object.__setattr__(self, name, integer_scalar(getattr(self, name), name))
        if self.chunk_size < 1 or self.save_every < 1:
            raise ValueError("chunk_size and save_every must be positive integers")


@dataclass
class RunResult:
    final_state: Any
    times: np.ndarray
    observables: Any
    metadata: dict = field(default_factory=dict)


class SimulationError(RuntimeError):
    """A failed evaluation/chunk with a retained state for checkpointing.

    ``failed_state`` is None when execution raised before returning a trustworthy
    candidate. The original exception remains available as ``__cause__``.
    Checked propagation supplies host numerical ``diagnostics``; its failed
    state is the retained macrostep input, possibly beyond the published chunk.
    """

    def __init__(self, message, *, last_valid_state, failed_state, diagnostics=None):
        super().__init__(message)
        self.last_valid_state = last_valid_state
        self.failed_state = failed_state
        self.diagnostics = diagnostics


class Runner:
    """One runner can reuse compiled blocks across repeated runs of the same shape.

    `observer(times, values)` receives host arrays once per output chunk. Set
    collect=False with an observer for bounded-memory output. When output is
    requested, initial and final values are recorded; intermediate sampling is
    tied to the absolute step. With no observer and collect=False, observations
    are not evaluated.

    Static configuration is read-only once assembled. Construct a new runner
    when changing the model, method, integrator, execution or measurement. Use
    update_parameters for validated changes to the dynamic numerical parameters.
    """

    def __init__(self, problem, integrator, execution=None):
        self._problem = problem.validate()
        self._integrator = integrator
        self._execution = execution or Execution()
        validate_integrator = getattr(problem.method, "validate_integrator", None)
        if callable(validate_integrator):
            validate_integrator(integrator)
        self._checked = isinstance(integrator.electronic, LanczosOptions)
        if problem.geometry_guard is not None and not self._checked:
            raise ValueError("coordinate guards require scalar checked Lanczos propagation")
        if self._checked:
            if not callable(getattr(problem.method, "build_checked_step", None)):
                raise ValueError("checked Lanczos propagation requires build_checked_step")
            if not problem.model.spec.native_jax:
                raise ValueError("checked Lanczos propagation currently requires a native JAX model")
        if not problem.model.spec.native_jax:
            if not (self.execution.allow_host_callbacks and
                    getattr(problem.model, "execution_mode", None) == "host_callback"):
                raise ValueError("external models require a host-callback adapter and explicit "
                                 "Execution(allow_host_callbacks=True)")
        self._measurement = problem.measurement or ElectronicPopulation()
        self.measurement.validate(problem)
        self._compiled = {}
        self._running = False

    @property
    def problem(self):
        return self._problem

    @property
    def integrator(self):
        return self._integrator

    @property
    def execution(self):
        return self._execution

    @property
    def measurement(self):
        return self._measurement

    def describe(self):
        """Summarize declared physical and execution scope without evaluating a model."""
        from pyeph.core.description import describe_simulation

        return describe_simulation(self)

    def update_parameters(self, params):
        """Validate and replace dynamic parameters, retaining compiled blocks.

        The model, method and other static configuration remain unchanged. A
        failed validation leaves the current problem intact. Parameters are
        runtime arguments, so cached kernels and initial measurements both use
        their new values. Caller-owned mutable providers must remain unchanged;
        replacing an artifact identity does not invalidate compiled constants.
        Updates are allowed only between runs; observers cannot change them.
        """
        if self._running:
            raise RuntimeError("cannot update parameters while this runner is running")
        candidate = replace(self.problem, params=params).validate()
        self.measurement.validate(candidate)
        self._problem = candidate

    def _validate_state(self, state):
        return _validation.validate_state(self.problem, self.measurement, self.execution, state,
                              checked=self._checked)

    def save_checkpoint(self, path, state, *, artifact_ids=None):
        """Save with a complete model/config/source identity, including parameter content."""
        from pyeph.io.checkpoint import save_checkpoint
        from pyeph.io.provenance import problem_manifest, validate_manifest

        manifest = problem_manifest(self.problem, self.integrator, artifact_ids=artifact_ids)
        validate_manifest(manifest, require_complete=True)
        self._validate_state(state)
        save_checkpoint(path, state, metadata={"simulation_manifest": manifest})

    def load_checkpoint(self, path, *, artifact_ids=None):
        """Restore only when this simulation matches the saved scientific identity."""
        from pyeph.io.checkpoint import load_checkpoint
        from pyeph.io.provenance import assert_matching_manifest, problem_manifest

        state, metadata = load_checkpoint(path)
        if "simulation_manifest" not in metadata:
            raise ValueError("checkpoint has no simulation manifest; use low-level load for manual migration")
        expected = problem_manifest(self.problem, self.integrator, artifact_ids=artifact_ids)
        assert_matching_manifest(metadata["simulation_manifest"], expected)
        self._validate_state(state)
        return state

    def _block(self, batch, nsteps, sample_indices=None):
        """Build a block with compact output at sorted zero-based step indices.

        The private default retains the all-step scan used by kernel benchmarks.
        An empty selection performs no measurement tracing or evaluation and
        returns empty times plus an empty dict. Sparse output uses fixed-size
        buffers, so save_every reduces device output storage as well as transfers.
        """
        indices = _blocks.sample_schedule(nsteps, sample_indices)
        key = batch, nsteps, indices
        if key not in self._compiled:
            block = _blocks.build_block(self.problem, self.integrator, self.measurement,
                                batch=batch, nsteps=nsteps, indices=indices,
                                set_time=self._set_time)
            self._compiled[key] = jax.jit(block) if self.execution.jit else block
        return self._compiled[key]

    def _observation(self, batch):
        """Reuse a pure initial measurement without caching parameters or values."""
        key = "observation", batch
        if key not in self._compiled:
            observe = _blocks.build_observer(self.problem, self.measurement, batch=batch)
            self._compiled[key] = jax.jit(observe) if self.execution.jit else observe
        return self._compiled[key]

    def _set_time(self, state, time, batch):
        # A method may stop within a macrostep (for example a failed MASH
        # event). Its failed state retains the actual last accepted time.
        succeeded = getattr(self.problem.method, "step_succeeded", None)
        if callable(succeeded):
            accepted = jax.vmap(succeeded)(state) if batch else succeeded(state)
            time = jax.numpy.where(accepted, time, state.time)
        return state._replace(time=time)

    def _checked_block(self, batch, nsteps, sample_indices=None):
        """Stop at the first rejected macrostep without publishing partial output.

        Each method handles its batch stages explicitly. Mapping an entire
        checked step would turn scalar acceptance gates into lane-wise selects.
        The transient diagnostics never enter the physical checkpoint state.
        """
        indices = _blocks.sample_schedule(nsteps, sample_indices)
        key = "checked", batch, nsteps, indices
        if key not in self._compiled:
            block = _blocks.build_checked_block(self.problem, self.integrator, self.measurement,
                                        batch=batch, nsteps=nsteps, indices=indices)
            self._compiled[key] = jax.jit(block) if self.execution.jit else block
        return self._compiled[key]

    def run(self, initial, steps, *, observer=None, collect=True):
        if self._running:
            raise RuntimeError("this runner is already running; reentrant runs are unsupported")
        self._running = True
        try:
            return self._run(initial, steps, observer=observer, collect=collect)
        finally:
            self._running = False

    def _validate_observations(self, values, *, last_valid_state, failed_state, phase):
        """Let a measurement reject a complete host block before publication.

        This optional hook owns observable-specific validity rules. Some valid
        diagnostics are undefined by construction, so the runner does not apply
        a blanket finite-value policy to arbitrary user measurements.
        """
        if self.problem.geometry_guard is not None:
            if any(not np.isfinite(np.asarray(value)).all() for value in jax.tree.leaves(values)):
                raise SimulationError(f"{phase} observation contains nonfinite values",
                                      last_valid_state=last_valid_state, failed_state=failed_state)
        validate = getattr(self.measurement, "validate_observations", None)
        if callable(validate):
            try:
                validate(values)
            except Exception as exc:
                raise SimulationError(f"{phase} observation validation failed",
                                      last_valid_state=last_valid_state,
                                      failed_state=failed_state) from exc

    def _validate_chunk_result(self, state, previous_state, block_info):
        """Reject numerical or method failures before observations are published."""
        if self._checked and int(block_info["failed_macro_index"]) >= 0:
            from pyeph.dynamics.checked import PHASE_NAMES
            from pyeph.integrators.krylov import STATUS

            diagnostics = jax.device_get({**block_info,
                "trajectory_ids": state.trajectory_id,
                "attempted_time": state.time})
            info = diagnostics["step_info"]
            reason = {1: "electronic action rejected", 2: "nonfinite physical stage",
                      3: "accumulated action estimate exceeds macrostep budget",
                      4: "coordinate outside declared CoordinateBox domain"}.get(
                          int(info.code), f"unknown method status {int(info.code)}")
            if int(info.code) == 1:
                reason += ": " + "; ".join(STATUS.get(int(code), f"action status {int(code)}")
                    for code in np.unique(info.action.status) if code != 0)
            phase = PHASE_NAMES.get(int(info.phase), f"phase {int(info.phase)}")
            raise SimulationError(
                f"checked electronic propagation failed during {phase} "
                f"(substep {int(info.substep)}): {reason}",
                last_valid_state=previous_state, failed_state=state,
                diagnostics=diagnostics)
        if self.execution.check_finite:
            import jax.numpy as jnp

            finite = jnp.all(jnp.stack([jnp.all(jnp.isfinite(x))
                for x in (state.q, state.p, state.electronic, state.time)]))
            if not bool(finite):
                raise SimulationError("nonfinite state produced by a simulation chunk",
                                      last_valid_state=previous_state, failed_state=state)
        validate_result = getattr(self.problem.method, "validate_result", None)
        if callable(validate_result):
            try:
                validate_result(state)
            except SimulationError as exc:
                if exc.last_valid_state is None:
                    exc.last_valid_state = previous_state
                raise

    def _run(self, initial, steps, *, observer=None, collect=True):
        if not isinstance(steps, int) or steps < 0:
            raise ValueError("steps must be a nonnegative integer")
        batch = self._validate_state(initial)
        _validation.validate_run_span(self.problem.nuclear_treatment, self.integrator, initial, steps)
        needs_output = collect or observer is not None
        if needs_output:
            try:
                initial_values = jax.device_get(
                    self._observation(batch)(self.problem.params, initial))
            except Exception as exc:
                raise SimulationError("initial observation execution failed",
                                      last_valid_state=initial, failed_state=None) from exc
            self._validate_observations(initial_values, last_valid_state=initial,
                                        failed_state=None, phase="initial")
        times = [np.asarray(initial.time)[None]] if collect else []
        values = [jax.tree.map(lambda x: np.asarray(x)[None], initial_values)] if collect else []
        if observer is not None:
            observer(np.asarray(initial.time)[None],
                     jax.tree.map(lambda x: np.asarray(x)[None], initial_values))
        state = initial
        done = 0
        initial_step = int(np.asarray(initial.step).reshape(-1)[0])
        while done < steps:
            count = min(self.execution.chunk_size, steps - done)
            if needs_output:
                absolute = initial_step + done + np.arange(1, count + 1)
                mask = absolute % self.execution.save_every == 0
                if done + count == steps:
                    mask[-1] = True
                indices = np.flatnonzero(mask)
            else:
                indices = np.empty(0, dtype=int)
            previous_state = state
            state = self._set_time(state, initial.time + done * self.integrator.dt, batch)
            try:
                if self._checked:
                    state, (block_time, block_values), block_info = self._checked_block(
                        batch, count, indices)(self.problem.params, state)
                else:
                    state, (block_time, block_values) = self._block(
                        batch, count, indices)(self.problem.params, state)
                    block_info = None
                # Callbacks/device failures may arrive asynchronously. A run
                # must not report success merely because output and optional
                # finite-value checks are disabled. No failed chunk is published.
                jax.block_until_ready((state, block_time, block_values, block_info))
            except Exception as exc:
                raise SimulationError("simulation chunk execution failed",
                                      last_valid_state=previous_state, failed_state=None) from exc
            self._validate_chunk_result(state, previous_state, block_info)
            if indices.size:
                # The compiled block already contains only selected observations.
                sampled_time, sampled_values = jax.device_get((block_time, block_values))
                self._validate_observations(sampled_values, last_valid_state=previous_state,
                                            failed_state=state, phase="simulation chunk")
                if observer is not None:
                    observer(sampled_time, sampled_values)
                if collect:
                    times.append(sampled_time)
                    values.append(sampled_values)
            done += count
        result_values = jax.tree.map(lambda *x: np.concatenate(x, axis=0), *values) if collect else {}
        result_times = np.concatenate(times, axis=0) if collect else np.empty((0,))
        return RunResult(state, result_times, result_values, {
            "dt": self.integrator.dt, "steps": steps, "batched": batch,
            "model": self.problem.model.spec.name,
            "method": type(self.problem.method).__name__,
            "electronic_integrator": self.integrator.electronic_name,
            "x64": bool(jax.config.x64_enabled),
        })
