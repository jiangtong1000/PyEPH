"""CPA for recorded electronic frames, without a fictitious force-capable model."""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from pyeph.core._configuration import real_scalar
from pyeph.core.state import ElectronicPathState
from pyeph.execution.runner import Execution, RunResult, SimulationError
from pyeph.integrators.electronic import propagate
from pyeph.integrators.krylov import LanczosOptions
from pyeph.paths.electronic import AdiabaticElectronicPath, FixedBasisElectronicPath


@dataclass(frozen=True)
class RecordedCPA:
    """Prescribed electronic dynamics under an explicit interpolation policy.

    Fixed-basis paths use their interpolated H(t) with a chosen Integrator.
    Adiabatic frames use endpoint phase halves around a declared basis transport:
    c1 = exp(-i E1 dt/2) T01 exp(-i E0 dt/2) c0. This defines an interval
    splitting, not spatial NACs or an arbitrarily sampleable intermediate basis.
    For raw nonunitary overlaps the projection loss remains in the amplitudes.
    No renormalization is applied. `max_subspace_loss` bounds the raw diagnostic
    even when a polar transport has been explicitly requested by the dataset.

    Electronic columns are independent vectors sharing this same recorded path.
    There are no nuclear feedback or hopping operations in this profile.
    Configuration is immutable; construct a new instance for a different path,
    integrator, execution policy or subspace-loss bound.
    """

    path: object
    integrator: object = None
    execution: object = None
    max_subspace_loss: float = 1e-8

    def __post_init__(self):
        if not isinstance(self.path, (FixedBasisElectronicPath, AdiabaticElectronicPath)):
            raise TypeError("RecordedCPA requires a fixed-basis or adiabatic electronic path")
        object.__setattr__(self, "execution", self.execution or Execution())
        if isinstance(getattr(self.integrator, "electronic", None), LanczosOptions):
            raise ValueError("RecordedCPA does not yet support checked Lanczos propagation")
        if self.max_subspace_loss is not None:
            loss = real_scalar(self.max_subspace_loss, "max_subspace_loss")
            if not 0 <= loss <= 1:
                raise ValueError("max_subspace_loss must lie in [0,1], or be explicitly None")
            object.__setattr__(self, "max_subspace_loss", loss)
        if isinstance(self.path, FixedBasisElectronicPath):
            if self.integrator is None:
                raise ValueError("fixed-basis interpolation needs an explicit integrator")
            if self.path.interpolation != "linear":
                raise ValueError("continuous propagation requires a declared matrix interpolation")
        else:
            if self.integrator is not None:
                raise ValueError("adiabatic interval durations come from the recorded grid")
            if self.path.overlaps is None:
                raise ValueError("adiabatic energies alone do not define basis transport")
        object.__setattr__(self, "_compiled", {})

    def initialize(self, electronic, *, time=None, frame_index=0):
        c = jnp.array(electronic, dtype=jnp.result_type(1j), copy=True)
        if c.ndim not in (1, 2) or c.shape[0] != self.path.nstates or not np.isfinite(c).all():
            raise ValueError("electronic data must be a finite vector or column block in the path basis")
        if not isinstance(frame_index, int) or not 0 <= frame_index < len(self.path.times):
            raise ValueError("initial frame index is out of range")
        t = float(self.path.times[frame_index]) if time is None else float(time)
        self.path.validate_sample_time(t)
        if isinstance(self.path, AdiabaticElectronicPath) and t != float(self.path.times[frame_index]):
            raise ValueError("initial time and adiabatic frame index disagree")
        return ElectronicPathState(c, jnp.asarray(t), jnp.asarray(0), jnp.asarray(frame_index))

    def _manifest(self, artifact_ids=None):
        from pyeph.io.provenance import recorded_path_manifest

        return recorded_path_manifest(self.path, self.integrator, artifact_ids=artifact_ids,
                                      method={"name": "recorded_cpa",
                                              "max_subspace_loss": self.max_subspace_loss})

    def save_checkpoint(self, path, state, *, artifact_ids=None):
        from pyeph.io.checkpoint import save_checkpoint
        from pyeph.io.provenance import validate_manifest

        manifest = self._manifest(artifact_ids)
        validate_manifest(manifest, require_complete=True)
        self.run(state, 0, collect=False)
        save_checkpoint(path, state, metadata={"simulation_manifest": manifest})

    def load_checkpoint(self, path, *, artifact_ids=None):
        from pyeph.io.checkpoint import load_checkpoint
        from pyeph.io.provenance import assert_matching_manifest

        state, metadata = load_checkpoint(path)
        if "simulation_manifest" not in metadata:
            raise ValueError("checkpoint has no simulation manifest")
        assert_matching_manifest(metadata["simulation_manifest"], self._manifest(artifact_ids))
        self.run(state, 0, collect=False)
        return state

    def _block(self, count):
        if count in self._compiled:
            return self._compiled[count]
        path = self.path

        def step(state, _):
            if isinstance(path, FixedBasisElectronicPath):
                c = propagate(path.apply, state.time, state.electronic, self.integrator.dt,
                              algorithm=self.integrator.electronic,
                              substeps=self.integrator.electronic_substeps)
                state = state._replace(electronic=c, time=state.time + self.integrator.dt,
                                       step=state.step + 1)
                loss = jnp.asarray(0., dtype=state.time.dtype)
                valid = jnp.asarray(True)
            else:
                i = state.frame_index
                dt = path.times[i + 1] - path.times[i]
                phase0 = jnp.exp(-0.5j * dt * path.energies[i])
                phase1 = jnp.exp(-0.5j * dt * path.energies[i + 1])
                if state.electronic.ndim == 2:
                    phase0, phase1 = phase0[:, None], phase1[:, None]
                transport = path.transport_at(i)
                c = phase1 * (transport.matrix @ (phase0 * state.electronic))
                state = state._replace(electronic=c, time=path.times[i + 1],
                                       step=state.step + 1, frame_index=i + 1)
                loss = transport.diagnostics.maximum_norm_loss
                valid = transport.valid
            values = {"population": jnp.abs(state.electronic)**2,
                      "norm": jnp.sum(jnp.abs(state.electronic)**2, axis=0),
                      "maximum_subspace_loss": loss, "transport_valid": valid}
            return state, (state.time, values)

        def block(state):
            def body(s, index):
                if isinstance(path, FixedBasisElectronicPath):
                    s = s._replace(time=state.time + index * self.integrator.dt)
                s, (time, values) = step(s, None)
                if isinstance(path, FixedBasisElectronicPath):
                    s = s._replace(time=state.time + (index + 1) * self.integrator.dt)
                return s, (s.time, values)
            return jax.lax.scan(body, state, xs=jnp.arange(count))

        self._compiled[count] = jax.jit(block) if self.execution.jit else block
        return self._compiled[count]

    def run(self, initial, steps, *, observer=None, collect=True):
        if not isinstance(initial, ElectronicPathState):
            raise TypeError("initialize RecordedCPA or load an electronic-path checkpoint")
        if not isinstance(steps, int) or steps < 0:
            raise ValueError("steps must be a nonnegative integer")
        c = np.asarray(initial.electronic)
        if c.ndim not in (1, 2) or c.shape[0] != self.path.nstates or not np.isfinite(c).all():
            raise ValueError("initial electronic state must be a finite vector/block in the path basis")
        if np.ndim(initial.time) or not np.isfinite(initial.time):
            raise ValueError("initial recorded time must be a finite scalar")
        for name in ("step", "frame_index"):
            a = np.asarray(getattr(initial, name))
            if a.ndim or not np.issubdtype(a.dtype, np.integer) or a < 0:
                raise ValueError("recorded state counters must be nonnegative scalar integers")
            advances = name == "step" or isinstance(self.path, AdiabaticElectronicPath)
            if advances and steps > int(np.iinfo(a.dtype).max) - int(a):
                raise ValueError(f"requested steps would overflow recorded {name}")
        if isinstance(self.path, FixedBasisElectronicPath):
            self.path.validate_span(float(initial.time), float(initial.time + steps * self.integrator.dt))
        else:
            start = int(initial.frame_index)
            if start < 0 or start + steps >= len(self.path.times):
                raise ValueError("requested propagation exceeds the available recorded intervals")
            if float(initial.time) != float(self.path.times[start]):
                raise ValueError("checkpoint time and recorded frame index disagree")
            for i in range(start, start + steps):
                transport = self.path.transport_at(i)
                if not bool(transport.valid):
                    raise ValueError(f"invalid overlap transport at interval {i}")
                if (self.max_subspace_loss is not None and
                        float(transport.diagnostics.maximum_norm_loss) > self.max_subspace_loss):
                    raise ValueError(f"raw subspace loss exceeds the declared bound at interval {i}")
        values0 = {"population": np.abs(initial.electronic)**2,
                   "norm": np.sum(np.abs(initial.electronic)**2, axis=0),
                   "maximum_subspace_loss": np.asarray(0.), "transport_valid": np.asarray(True)}
        values0 = jax.tree.map(lambda x: np.asarray(x)[None], values0)
        times, values = ([np.asarray(initial.time)[None]], [values0]) if collect else ([], [])
        if observer:
            observer(np.asarray(initial.time)[None], values0)
        state, done = initial, 0
        while done < steps:
            count = min(self.execution.chunk_size, steps - done)
            previous = state
            if isinstance(self.path, FixedBasisElectronicPath):
                state = state._replace(time=initial.time + done * self.integrator.dt)
            state, (t, observed) = self._block(count)(state)
            if self.execution.check_finite and not bool(jnp.all(jnp.isfinite(state.electronic))):
                raise SimulationError("nonfinite electronic path state", last_valid_state=previous,
                                      failed_state=state)
            mask = (int(initial.step) + done + np.arange(1, count + 1)) % self.execution.save_every == 0
            if done + count == steps:
                mask[-1] = True
            indices = np.flatnonzero(mask)
            if len(indices):
                t, observed = jax.device_get((t[indices], jax.tree.map(lambda x: x[indices], observed)))
                if observer:
                    observer(t, observed)
                if collect:
                    times.append(t)
                    values.append(observed)
            done += count
        observations = jax.tree.map(lambda *x: np.concatenate(x), *values) if collect else {}
        return RunResult(state, np.concatenate(times) if collect else np.array([]), observations,
                         {"method": "recorded_cpa", "basis_id": self.path.basis_id,
                          "interpolation": self.path.interpolation, "steps": steps})
