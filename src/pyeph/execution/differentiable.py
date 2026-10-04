"""Pure fixed-length sensitivity rollouts for smooth native CPA and Ehrenfest.

Operational validation, output and checkpoints remain the Runner's job. This
module exposes the same method steps to JAX transformations without performing
host validation or device-to-host conversion inside the differentiable call.
"""

from dataclasses import dataclass, replace
import hashlib
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from pyeph.core._configuration import boolean_scalar, integer_scalar
from pyeph.core.state import TrajectoryState
from pyeph.dynamics.cpa import CPA
from pyeph.dynamics.ehrenfest import Ehrenfest
from pyeph.execution._validation import validate_run_span, validate_state
from pyeph.execution.runner import Execution
from pyeph.io.checkpoint import array_fingerprint
from pyeph.observables.population import ElectronicPopulation


class DifferentiableResult(NamedTuple):
    """Device PyTree with the initial sample and every accepted fixed step."""

    final_state: Any
    times: Any
    observables: Any


def _reject_callbacks(jaxpr):
    """Inspect nested traced programs without executing callbacks."""
    visited = set()

    def visit(value):
        if id(value) in visited:
            return
        visited.add(id(value))
        nested = getattr(value, "jaxpr", None)
        if nested is not None:
            visit(nested)
        for equation in getattr(value, "eqns", ()):
            if equation.primitive.name in {"pure_callback", "io_callback", "debug_callback"}:
                raise ValueError("differentiable rollout does not permit host callbacks")
            visit(equation.params)
        if isinstance(value, dict):
            for item in value.values():
                visit(item)
        elif isinstance(value, (tuple, list)):
            for item in value:
                visit(item)

    visit(jaxpr)


def _trace_programs(function, params, initial):
    """Fresh wrappers prevent cached tracing from concealing changed static data."""
    def primal(p, state):
        return function(p, state)

    def tangent(value):
        if jnp.issubdtype(jnp.asarray(value).dtype, jnp.inexact):
            return jnp.ones_like(value)
        return np.zeros(np.shape(value), dtype=jax.dtypes.float0)

    directions = jax.tree.map(tangent, (params, initial))

    def forward(p, state):
        return jax.jvp(primal, (p, state), directions)

    def probe(p, state):
        # Trace all inexact outputs. This is a derivative-purity check, not a
        # physical loss or a finite-difference correctness certificate.
        return sum(jnp.sum(jnp.real(value)) + jnp.sum(jnp.imag(value))
                   for value in jax.tree.leaves(primal(p, state))
                   if jnp.issubdtype(jnp.asarray(value).dtype, jnp.inexact))

    return tuple(jax.make_jaxpr(f)(params, initial) for f in (
        primal, forward, jax.grad(probe, argnums=(0, 1), allow_int=True)))


def _program_identity(programs):
    """Bind representative traced operations and closed-over numerical constants."""
    digest = hashlib.sha256()
    visited = set()

    def visit(value):
        if id(value) in visited:
            return
        visited.add(id(value))
        if hasattr(value, "consts"):
            digest.update(array_fingerprint(value.consts).encode())
        nested = getattr(value, "jaxpr", None)
        if nested is not None:
            visit(nested)
        for equation in getattr(value, "eqns", ()):
            visit(equation.params)
        if isinstance(value, dict):
            for key in sorted(value):
                visit(value[key])
        elif isinstance(value, (list, tuple)):
            for item in value:
                visit(item)

    for program in programs:
        digest.update(str(program).encode())
        visit(program)
    return digest.hexdigest()


def _kernel(problem, integrator, measurement, *, steps, batch, rematerialize):
    def rollout(params, initial):
        runtime = replace(problem, params=params)
        step = runtime.method.build_step(runtime, integrator)

        def observe(state):
            return measurement.evaluate(runtime, state)

        if batch:
            step = jax.vmap(step)
            observe = jax.vmap(observe)

        def advance(state, index):
            # Match the operational runner's clock anchoring, including a
            # nonzero initial time and different origins in independent lanes.
            state = state._replace(time=initial.time + index * integrator.dt)
            state = step(state)
            state = state._replace(time=initial.time + (index + 1) * integrator.dt)
            return state, (state.time, observe(state))

        body = jax.checkpoint(advance) if rematerialize else advance
        final, (times, values) = jax.lax.scan(body, initial, jnp.arange(steps))
        first = observe(initial)
        values = jax.tree.map(lambda a, b: jnp.concatenate((jnp.asarray(a)[None], b), axis=0),
                              first, values)
        times = jnp.concatenate((initial.time[None], times), axis=0)
        return DifferentiableResult(final, times, values)

    return rollout


@dataclass(frozen=True, eq=False, init=False)
class DifferentiableRollout:
    """Build and preflight once, then differentiate a pure ``(params, state)`` call.

    Only the native CPA/Ehrenfest implementations with fixed-step electronic
    RK4 are supported. A representative initial state determines whether the
    kernel vmaps independent trajectories. All changing model parameters are
    explicit runtime PyTrees. Nuclear masses/path definitions, timestep, step
    count, method and measurement are static configuration.

    Call ``preflight`` on concrete new inputs outside AD/JIT when needed. The
    pure call deliberately has no host checks or event/failure handling; it
    differentiates the discrete smooth integrator, not exact continuum dynamics.
    Smoothness and higher-derivative correctness of custom providers remain
    the provider author's responsibility.
    """

    problem: Any
    integrator: Any
    measurement: Any
    steps: int
    batch: bool
    rematerialize: bool
    _call: Any
    _reference_inputs: Any
    _static_identity: Any

    def __init__(self, problem, integrator, initial, *, steps, rematerialize=False):
        steps = integer_scalar(steps, "steps")
        rematerialize = boolean_scalar(rematerialize, "rematerialize")
        if steps < 0:
            raise ValueError("differentiable rollout steps must be nonnegative")
        if not isinstance(initial, TrajectoryState):
            raise TypeError("differentiable geometry dynamics requires a TrajectoryState")
        if type(problem.method) not in (CPA, Ehrenfest):
            raise ValueError("differentiable rollout supports native CPA and Ehrenfest only")
        if not problem.model.spec.native_jax:
            raise ValueError("differentiable rollout requires a native JAX model without callbacks")
        if integrator.electronic != "rk4":
            raise ValueError("differentiable rollout requires RK4; checked solvers and eigendecomposition "
                             "exponentials are not qualified for this sensitivity interface")
        problem.validate()
        measurement = problem.measurement or ElectronicPopulation()
        measurement.validate(problem)
        batch = np.ndim(initial.time) == 1
        for name, value in dict(problem=problem, integrator=integrator, measurement=measurement,
                                steps=steps, batch=batch, rematerialize=rematerialize).items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "_call", _kernel(problem, integrator, measurement,
                                                steps=steps, batch=batch,
                                                rematerialize=rematerialize))
        object.__setattr__(self, "_reference_inputs", jax.tree.map(
            lambda value: jnp.array(value, copy=True), (problem.params, initial)))
        object.__setattr__(self, "_static_identity", None)
        self.preflight(problem.params, initial)

    def preflight(self, params, initial):
        """Validate concrete inputs and traced callback-free execution outside AD.

        This checks starting-state compatibility and traceability, not every
        future geometry or the accuracy of a custom derivative. Dynamic inputs
        used later must retain their model's physical constraints.
        """
        runtime = replace(self.problem, params=params).validate()
        self.measurement.validate(runtime)
        if not isinstance(initial, TrajectoryState):
            raise TypeError("differentiable geometry dynamics requires a TrajectoryState")
        if (np.ndim(initial.time) == 1) != self.batch:
            raise ValueError("initial state batching differs from the preflight configuration")
        validate_run_span(runtime.nuclear_treatment, self.integrator, initial, self.steps)
        # Inspect derivative programs as well as the primal: custom derivative
        # rules can otherwise introduce callbacks only when AD is requested.
        programs = _trace_programs(self._call, *self._reference_inputs)
        for program in programs:
            _reject_callbacks(program)
        identity = _program_identity(programs)
        if self._static_identity is not None and identity != self._static_identity:
            raise ValueError("static rollout computation changed; construct a new DifferentiableRollout")
        if self._static_identity is None:
            object.__setattr__(self, "_static_identity", identity)
        if (jax.tree.structure((params, initial)) != jax.tree.structure(self._reference_inputs)
                or any(np.shape(a) != np.shape(b) or jnp.asarray(a).dtype != jnp.asarray(b).dtype
                       for a, b in zip(jax.tree.leaves((params, initial)),
                                       jax.tree.leaves(self._reference_inputs), strict=True))):
            for program in _trace_programs(self._call, params, initial):
                _reject_callbacks(program)
        validate_state(runtime, self.measurement, Execution(), initial, checked=False)

    def __call__(self, params, initial):
        """Evaluate a device-only PyTree; valid inside jit, jvp, grad and vmap."""
        return self._call(params, initial)
