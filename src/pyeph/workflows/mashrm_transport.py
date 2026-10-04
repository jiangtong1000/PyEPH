"""Single-origin RM velocity correlations on independent canonical trajectories.

The per-trajectory product is formed before reduction. The origin is separate
from the method's physical state and survives continuation unchanged. A workflow
checkpoint resumes future output segments; it does not checkpoint an output
writer or previously accumulated statistics. Adjacent segments share a boundary
sample, which callers must deduplicate when concatenating along time.
"""

from dataclasses import dataclass, field
import hashlib
import json
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from pyeph.core.problem import CoupledClassical, Problem
from pyeph.core.state import TrajectoryState
from pyeph.dynamics.mashrm import MASHRM
from pyeph.execution.ensemble import EnsembleResult, _ids, _same_grid
from pyeph.execution.runner import Runner
from pyeph.io.provenance import assert_matching_manifest, problem_manifest, validate_manifest
from pyeph.observables.statistics import EnsembleMoments
from pyeph.observables.transport.mashrm import RMVelocity
from pyeph.workflows.mashrm_equilibrium import LinearEPCCanonical


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value):
    return hashlib.sha256(_json(value).encode()).hexdigest()


@dataclass(frozen=True, init=False, eq=False)
class RMCorrelationOrigin:
    """Owned initial velocities with stable IDs and a common physical origin.

    ``preparation_metadata`` returns a fresh dictionary. Numerical arrays are
    immutable JAX snapshots. Reordering or discarding trajectories requires a
    corresponding explicit origin subset; a continuation never remeasures v0.
    """

    trajectory_ids: Any
    velocity0: Any
    time0: float
    workflow_fingerprint: str
    _preparation_json: str = field(repr=False)

    def __init__(self, trajectory_ids, velocity0, time0, workflow_fingerprint,
                 preparation_metadata):
        ids = _ids(trajectory_ids)
        velocity = np.asarray(velocity0)
        time = np.asarray(time0)
        if (velocity.ndim != 2 or velocity.shape[0] != len(ids) or not velocity.shape[1]
                or velocity.dtype.kind != "f" or not np.isfinite(velocity).all()):
            raise ValueError("origin velocities must be finite real (trajectory, probe) data")
        if time.ndim != 0 or time.dtype.kind not in "iuf" or not np.isfinite(time):
            raise ValueError("origin time must be one finite real scalar")
        if not isinstance(workflow_fingerprint, str) or not workflow_fingerprint:
            raise ValueError("origin must identify its complete workflow")
        metadata = _json(preparation_metadata)
        if (not isinstance(preparation_metadata.get("preparation_id"), str)
                or not preparation_metadata["preparation_id"]):
            raise ValueError("origin must identify its canonical preparation")
        if velocity.dtype.itemsize > 4 and not jax.config.x64_enabled:
            raise ValueError("enable JAX x64 before constructing a double-precision correlation origin")
        object.__setattr__(self, "trajectory_ids", jnp.array(ids, copy=True))
        object.__setattr__(self, "velocity0", jnp.array(velocity, copy=True))
        object.__setattr__(self, "time0", float(time))
        object.__setattr__(self, "workflow_fingerprint", workflow_fingerprint)
        object.__setattr__(self, "_preparation_json", metadata)

    @property
    def preparation_metadata(self):
        return json.loads(self._preparation_json)

    @property
    def preparation_id(self):
        return self.preparation_metadata["preparation_id"]


@dataclass(frozen=True)
class RMTransportState:
    """Physical trajectory state plus immutable correlation-origin context."""

    state: TrajectoryState
    origin: RMCorrelationOrigin


@dataclass(frozen=True)
class RMTransportResult:
    """One time segment of independent-trajectory correlation statistics.

    ``statistics.times`` are lag times relative to the original preparation.
    Mean/M2 have shape (saved_time, origin_probe, current_probe). The SEM uses
    independent trajectories, never time origins, as samples. ``statistics``
    is None for collect=False; streaming observers still receive products.
    """

    final_state: RMTransportState
    statistics: EnsembleResult | None
    metadata: dict


@dataclass(frozen=True, init=False, eq=False)
class RMTransport:
    """Compose canonical preparation, MASHRM and validated physical velocities.

    This first workflow accepts LinearEPCCanonical and complete real spectra.
    Physical velocity providers remain separate from the Hamiltonian model.
    Configuration and numerical parameters are fixed for the workflow's life;
    construct another workflow when changing the scientific problem.

    ``observer(lag_times, values)`` receives host arrays in bounded chunks.
    Values contain the ordinary RMVelocity output plus velocity_correlation
    with axes (time, trajectory, origin_probe, current_probe). With collect=True
    only O(saved_times*probes**2) correlation moments are retained on the host,
    in addition to the current trajectory batch and temporary output chunk.
    collect=False retains no output history. No mobility prefactor, integration,
    charge conversion, overlapping origins, or quantum-nuclear correction is
    implied by a velocity correlation.
    """

    sampler: LinearEPCCanonical
    _runner: Runner = field(repr=False)
    _identity_json: str = field(repr=False)
    _artifact_ids: tuple = field(repr=False)

    def __init__(self, sampler, integrator, measurement, *, method=None, execution=None,
                 artifact_ids=None):
        if type(sampler) is not LinearEPCCanonical:
            raise TypeError("RMTransport requires a LinearEPCCanonical sampler")
        if type(measurement) is not RMVelocity:
            raise TypeError("RMTransport requires an RMVelocity measurement")
        method = MASHRM() if method is None else method
        if type(method) is not MASHRM:
            raise TypeError("RMTransport requires MASHRM")
        if not sampler.gap_tolerance == method.gap_tolerance == measurement.gap_tolerance:
            raise ValueError("sampler, method and measurement gap tolerances must agree")
        if method.real_tolerance != measurement.real_tolerance:
            raise ValueError("method and measurement real tolerances must agree")
        problem = Problem(sampler.model, sampler.params, CoupledClassical(sampler.masses),
                          method, measurement)
        runner = Runner(problem, integrator, execution)
        manifest = problem_manifest(problem, integrator, artifact_ids=artifact_ids)
        validate_manifest(manifest, require_complete=True)
        identity = dict(schema=1, simulation=manifest, preparation=dict(
            type="LinearEPCCanonical", beta=sampler.beta, kappa=sampler.kappa,
            max_trials=sampler.max_trials, gap_tolerance=sampler.gap_tolerance))
        object.__setattr__(self, "sampler", sampler)
        object.__setattr__(self, "_runner", runner)
        object.__setattr__(self, "_identity_json", _json(identity))
        object.__setattr__(self, "_artifact_ids", tuple((artifact_ids or {}).items()))

    @property
    def problem(self):
        return self._runner.problem

    @property
    def integrator(self):
        return self._runner.integrator

    @property
    def execution(self):
        return self._runner.execution

    @property
    def measurement(self):
        return self._runner.measurement

    @property
    def identity(self):
        return json.loads(self._identity_json)

    @property
    def fingerprint(self):
        return _digest(self.identity)

    def _check_identity(self):
        current = problem_manifest(self.problem, self.integrator,
                                   artifact_ids=dict(self._artifact_ids))
        assert_matching_manifest(self.identity["simulation"], current)

    def prepare(self, trajectory_ids, *, seed=0):
        """Draw independent canonical trajectories and measure their initial v0."""
        self._check_identity()
        ensemble = self.sampler.sample(trajectory_ids, seed=seed)
        # The ordinary runner owns both physical preflight and observable
        # validation, including every trajectory of the preparation batch.
        observed = self._runner.run(ensemble.state, 0)
        origin = RMCorrelationOrigin(
            ensemble.state.trajectory_id, observed.observables["velocity"][0],
            np.asarray(ensemble.state.time)[0], self.fingerprint, ensemble.metadata)
        return RMTransportState(ensemble.state, origin)

    def _validate_context(self, context):
        if not isinstance(context, RMTransportState) or not isinstance(context.origin, RMCorrelationOrigin):
            raise TypeError("continuation requires an RMTransportState with its original origin")
        if not isinstance(context.state, TrajectoryState):
            raise TypeError("continuation requires a physical TrajectoryState")
        state, origin = context.state, context.origin
        if origin.workflow_fingerprint != self.fingerprint:
            raise ValueError("correlation origin belongs to a different workflow")
        preparation = origin.preparation_metadata
        expected = self.sampler.preparation_metadata(seed=preparation.get("seed"))
        if preparation != expected:
            raise ValueError("correlation origin canonical preparation metadata mismatch")
        if (np.ndim(state.time) != 1 or not np.array_equal(
                np.asarray(state.trajectory_id), np.asarray(origin.trajectory_ids))):
            raise ValueError("trajectory IDs must exactly match the correlation origin in order")
        if origin.velocity0.shape != (state.time.size, len(self.measurement.probes)):
            raise ValueError("origin velocity shape does not match trajectory and probe axes")
        times = np.asarray(state.time)
        if (not np.isfinite(times).all() or not np.all(times == times[0])
                or np.any(times < origin.time0)):
            raise ValueError("RM correlation trajectories require a common time at or after origin")

    def run(self, initial, steps, *, observer=None, collect=True):
        """Propagate one segment, preserving each trajectory's original velocity.

        Every collected segment includes its entry sample, even on restart.
        Statistics from disjoint trajectory partitions can be combined with
        merge_ensembles. Adjacent time segments share their boundary and must
        be concatenated after explicitly removing that duplicate time sample.
        Runner/measurement failures raise SimulationError with physical states;
        wrap a retained state with initial.origin to checkpoint continuation.
        Host correlation-product/moment representability errors are ValueError
        before workflow observer publication, without a retained chunk state.
        External observer exceptions likewise keep their original type.
        """
        self._check_identity()
        self._validate_context(initial)
        times, means, squares = [], [], []
        origin = initial.origin
        velocity0 = np.asarray(origin.velocity0)

        def consume(absolute_times, values):
            clock = np.asarray(absolute_times)
            if clock.ndim != 2 or clock.shape[1] != len(origin.trajectory_ids):
                raise ValueError("correlation observations must have time and trajectory axes")
            lag = clock[:, 0]-origin.time0
            if any(not _same_grid(lag, clock[:, i]-origin.time0) for i in range(clock.shape[1])):
                raise ValueError("correlation trajectories must share a lag-time grid")
            velocity = np.asarray(values["velocity"])
            if velocity.shape != clock.shape+(len(self.measurement.probes),):
                raise ValueError("velocity observations do not match time, trajectory and probe axes")
            with np.errstate(over="ignore", invalid="ignore"):
                products = velocity0[None, :, :, None]*velocity[:, :, None, :]
            if not np.isfinite(products).all():
                raise ValueError("per-trajectory velocity products must be finite")
            # Snapshot moments before invoking user code, which may reuse or
            # modify its received arrays while writing output.
            with np.errstate(over="ignore", invalid="ignore"):
                moments = EnsembleMoments().update(products, axis=1) if collect else None
            if collect and not (np.isfinite(moments.mean).all() and np.isfinite(moments.m2).all()):
                raise ValueError("correlation moments are not representable in the output dtype")
            if observer is not None:
                observer(lag.copy(), {**values, "velocity_correlation": products})
            if collect:
                times.append(lag)
                means.append(moments.mean)
                squares.append(moments.m2)

        raw = self._runner.run(initial.state, steps, collect=False,
                               observer=consume if collect or observer is not None else None)
        statistics = None
        if collect:
            statistics = EnsembleResult(
                np.concatenate(times), np.array(origin.trajectory_ids),
                np.concatenate(means), np.concatenate(squares),
                self.identity["simulation"], origin.preparation_id).validate()
        return RMTransportResult(RMTransportState(raw.final_state, origin), statistics,
                                 {**raw.metadata, "origin_time": origin.time0,
                                  "probes": self.measurement.probes,
                                  "correlation_order": "initial_i_times_current_j",
                                  "workflow_fingerprint": self.fingerprint})

    def save_checkpoint(self, path, context):
        """Atomically save physical state and origin; output history is external."""
        from pyeph.io.checkpoint import save_checkpoint

        self._check_identity()
        self._validate_context(context)
        self._runner._validate_state(context.state)
        origin = context.origin
        auxiliary = dict(trajectory_ids=origin.trajectory_ids, velocity0=origin.velocity0,
                         time0=jnp.asarray(origin.time0, dtype=context.state.time.dtype))
        metadata = dict(rm_transport_schema=1, workflow_identity=self.identity,
                        workflow_fingerprint=self.fingerprint,
                        preparation=origin.preparation_metadata)
        save_checkpoint(path, context.state, metadata=metadata, auxiliary=auxiliary)

    def load_checkpoint(self, path):
        """Restore a segment continuation only under the same scientific identity."""
        from pyeph.io.checkpoint import load_checkpoint

        self._check_identity()
        state, metadata, auxiliary = load_checkpoint(path, with_auxiliary=True)
        if metadata.get("rm_transport_schema") != 1:
            raise ValueError("checkpoint does not contain an RM transport origin")
        if (metadata.get("workflow_identity") != self.identity
                or metadata.get("workflow_fingerprint") != self.fingerprint):
            raise ValueError("RM transport checkpoint workflow identity mismatch")
        if not isinstance(auxiliary, dict) or set(auxiliary) != {"trajectory_ids", "velocity0", "time0"}:
            raise ValueError("RM transport checkpoint has invalid origin fields")
        origin = RMCorrelationOrigin(**auxiliary, workflow_fingerprint=self.fingerprint,
                                     preparation_metadata=metadata["preparation"])
        result = RMTransportState(state, origin)
        self._validate_context(result)
        self._runner._validate_state(state)
        return result


__all__ = ["RMCorrelationOrigin", "RMTransportState", "RMTransportResult", "RMTransport"]
