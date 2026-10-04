"""Action-only CPA current correlations from an explicit density factor.

For rho0=L L†, propagate [U L, U J_b(0) L for each origin probe b].
The resulting C[a,b]=sum_k (UL)_k† J_a(t) (U J_b(0)L)_k is the complex,
unsymmetrized trace. No full Hamiltonian, current, density or propagator is
constructed here. A caller may nevertheless choose a full-rank factor.
"""

from dataclasses import dataclass
import hashlib
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from pyeph.core._configuration import integer_scalar
from pyeph.core.contracts import ProbeContext
from pyeph.core.problem import Problem
from pyeph.core.state import make_state
from pyeph.core.validation import validate_model_at
from pyeph.dynamics.cpa import CPA
from pyeph.io.provenance import problem_manifest


_PAYLOAD = "column_transport"
_SCHEMA = 1
_PHASE_STREAM = 0x43545243
_FIELDS = {"schema", "rank", "nprobes", "time0", "q0", "p0", "trajectory_id0",
           "norms_squared0", "configuration_digest", "factor_digest",
           "preparation_code", "trace_ids", "trace_seed"}


def _velocity(problem, state):
    treatment = problem.nuclear_treatment
    if hasattr(treatment, "path") and callable(getattr(treatment.path, "velocity", None)):
        return treatment.path.velocity(state.time)
    if hasattr(treatment, "masses"):
        return state.p / jnp.asarray(treatment.masses)
    if callable(getattr(treatment, "velocity", None)):
        return treatment.velocity(state)
    return None


def _trajectory_key(seed, trajectory_id):
    seed = integer_scalar(seed, "seed")
    trajectory_id = integer_scalar(trajectory_id, "trajectory_id")
    if not 0 <= seed < 2**32 or not 0 <= trajectory_id < 2**32:
        raise ValueError("seed and trajectory_id must be representable as uint32")
    key = jax.random.fold_in(jax.random.key(seed, impl="threefry2x32"), np.uint32(trajectory_id))
    return seed, trajectory_id, key


def _configuration_digest(problem):
    # Integrator changes do not redefine the physical correlation origin.
    # Strict checkpoint provenance separately includes the actual integrator
    # and requires identities for opaque providers. This digest alone cannot
    # discover mutations to unencoded captured data inside custom callbacks.
    fingerprint = problem_manifest(problem, None)["fingerprint"]
    return np.frombuffer(bytes.fromhex(fingerprint), dtype=np.uint8).copy()


def _tolerance(dtype):
    dtype = np.dtype(dtype)
    real = np.empty((), dtype=dtype).real.dtype
    return 64 * np.finfo(real).eps


def _shape_payload(state, nstates, nprobes):
    if (state.electronic.ndim != 2 or state.electronic.shape[0] != nstates
            or state.electronic.shape[1] == 0 or state.electronic.shape[1] % (nprobes + 1)):
        raise ValueError("column transport needs an (nstates, rank*(1+nprobes)) electronic block")
    if not isinstance(state.method_state, dict) or _PAYLOAD not in state.method_state:
        raise ValueError("initialize column transport with its density-factor initializer")
    payload = state.method_state[_PAYLOAD]
    if not isinstance(payload, dict) or set(payload) != _FIELDS:
        raise ValueError("column transport origin payload has an unsupported schema")
    rank = state.electronic.shape[1] // (nprobes + 1)
    if payload["norms_squared0"].shape != (state.electronic.shape[1],):
        raise ValueError("origin norms must identify every propagated column")
    return payload, rank


@dataclass(frozen=True)
class ColumnTransportMeasurement:
    """Physical action probes for prescribed-path CPA, with explicit origins.

    ``probe_callback(params, ProbeContext, name, vectors)`` optionally replaces
    ``model.probe_apply``. Both must be pure linear actions of a physical
    Hermitian current in the model's fixed basis and consistent units. No
    charge/legacy factor is inferred. An explicit callback may convert imported
    legacy data, but this workflow does not guess a probe's physical meaning.

    Shape/finiteness checks do not prove linearity or global Hermiticity from a
    few columns. Providers must establish those contracts independently; this
    workflow never constructs an identity matrix to audit them.

    Output ``current_correlation`` has axes (current_probe, origin_probe).
    ``column_norm_squared_drift`` compares each column with its own initial
    squared norm; arbitrary density/insertion columns are not orthonormal.
    The returned correlation is one nuclear-path sample. For stochastic
    infinite-temperature preparation it also contains trace-sampling noise;
    this class neither estimates nor conflates the two uncertainty sources.
    """

    probes: tuple[str, ...] = ("current_x",)
    probe_callback: Any = None

    def __post_init__(self):
        if isinstance(self.probes, str):
            raise ValueError("probes must be a collection of unique names")
        probes = tuple(self.probes)
        if (not probes or any(not isinstance(p, str) or not p for p in probes)
                or len(set(probes)) != len(probes)):
            raise ValueError("probes must be nonempty unique names")
        if self.probe_callback is not None and not callable(self.probe_callback):
            raise TypeError("probe_callback must be an action callable or None")
        object.__setattr__(self, "probes", probes)

    def validate(self, problem):
        if (not isinstance(problem.method, CPA)
                or not getattr(problem.nuclear_treatment, "prescribed", False)
                or problem.model.spec.basis_kind != "fixed_orthonormal"):
            raise ValueError("column transport requires CPA on prescribed fixed-orthonormal paths")
        if self.probe_callback is None:
            missing = set(self.probes) - set(problem.model.spec.probes)
            if missing:
                raise ValueError(f"model does not define required physical probes: {sorted(missing)}")

    def action(self, problem, state, probe, vectors):
        context = ProbeContext(state.q, _velocity(problem, state), state.time)
        if self.probe_callback is None:
            value = problem.model.probe_apply(problem.params, context, probe, vectors)
        else:
            value = self.probe_callback(problem.params, context, probe, vectors)
        value = jnp.asarray(value)
        if value.shape != vectors.shape or value.dtype.kind not in "iufc":
            raise ValueError("current action must return numeric arrays preserving vector/block shape")
        return value

    def evaluate(self, problem, state):
        n, p = problem.model.spec.system.nstates, len(self.probes)
        payload, rank = _shape_payload(state, n, p)
        left = state.electronic[:, :rank]
        inserted = state.electronic[:, rank:]
        rows = []
        for probe in self.probes:
            acted = self.action(problem, state, probe, inserted).reshape(n, p, rank)
            rows.append(jnp.einsum("sk,sbk->b", left.conj(), acted))
        norms = jnp.sum(jnp.abs(state.electronic)**2, axis=0)
        drift = norms - payload["norms_squared0"]
        return {"current_correlation": jnp.stack(rows), "lag_time": state.time-payload["time0"],
                "column_norm_squared_drift": drift,
                "max_column_norm_squared_drift": jnp.max(jnp.abs(drift))}

    def validate_observations(self, values):
        if not isinstance(values, dict) or "current_correlation" not in values:
            raise ValueError("column transport observations are missing their correlation")
        correlation = np.asarray(values["current_correlation"])
        if correlation.shape[-2:] != (len(self.probes), len(self.probes)):
            raise ValueError("correlation axes must be (current_probe, origin_probe)")
        for name, value in values.items():
            value = np.asarray(value)
            if value.dtype.kind not in "iufc" or not np.isfinite(value).all():
                raise ValueError(f"nonfinite or nonnumeric column transport observation: {name}")

    def validate_initial_state(self, problem, state, *, batch=False):
        """Host origin validation before every run/checkpoint, even without output.

        Numerical params, encoded static configuration and source must match
        preparation. Opaque captured data remains the caller's immutability
        responsibility; strict Runner checkpoints additionally require complete
        external artifact identities. No parameter quench is inferred.
        """
        self.validate(problem)
        n, p = problem.model.spec.system.nstates, len(self.probes)
        electronic = np.asarray(state.electronic)
        prefix = (len(state.time),) if batch else ()
        if (electronic.ndim != 2 + int(batch) or electronic.shape[-2] != n
                or electronic.shape[-1] == 0 or electronic.shape[-1] % (p+1)):
            raise ValueError("column transport electronic block has an invalid column layout")
        rank = electronic.shape[-1] // (p+1)
        if not isinstance(state.method_state, dict) or _PAYLOAD not in state.method_state:
            raise ValueError("column transport state lacks its original preparation")
        raw = state.method_state[_PAYLOAD]
        if not isinstance(raw, dict) or set(raw) != _FIELDS:
            raise ValueError("column transport origin payload has an unsupported schema")
        payload = {name: np.asarray(value) for name, value in raw.items()}
        shapes = {name: prefix for name in ("schema", "rank", "nprobes", "time0", "trajectory_id0", "preparation_code", "trace_seed")}
        shapes.update(q0=prefix+problem.model.spec.system.q_shape,
                      p0=prefix+problem.model.spec.system.q_shape,
                      norms_squared0=prefix+(electronic.shape[-1],),
                      configuration_digest=prefix+(32,), factor_digest=prefix+(32,), trace_ids=prefix+(rank,))
        for name, shape in shapes.items():
            array = payload[name]
            if array.shape != shape or array.dtype.kind not in "iuf" or not np.isfinite(array).all():
                raise ValueError(f"invalid column origin field {name}: expected finite shape {shape}")
        for name in ("schema", "rank", "nprobes", "preparation_code"):
            if payload[name].dtype != np.int32:
                raise ValueError(f"column origin {name} must have int32 dtype")
        for name in ("trace_ids", "trace_seed", "trajectory_id0"):
            if payload[name].dtype != np.uint32:
                raise ValueError(f"column origin {name} must have uint32 dtype")
        for name in ("configuration_digest", "factor_digest"):
            if payload[name].dtype != np.uint8:
                raise ValueError(f"column origin {name} must have uint8 dtype")
        if (np.any(payload["schema"] != _SCHEMA) or np.any(payload["rank"] != rank)
                or np.any(payload["nprobes"] != p)):
            raise ValueError("column origin schema/rank/probe count does not match the state")
        if not np.array_equal(payload["trajectory_id0"], np.asarray(state.trajectory_id)):
            raise ValueError("column correlation origin belongs to a different trajectory ID")
        if np.any(np.asarray(state.time) < payload["time0"]):
            raise ValueError("a correlation state cannot precede its origin time")
        norms = payload["norms_squared0"]
        tolerance = _tolerance(electronic.dtype)
        if np.any(norms < 0) or not np.allclose(norms[..., :rank].sum(axis=-1), 1., atol=tolerance, rtol=0):
            raise ValueError("origin density-factor columns must have unit total squared norm")
        modes = payload["preparation_code"]
        if np.any((modes != 0) & (modes != 1)):
            raise ValueError("unknown column preparation code")
        if rank > 1 and np.any(np.diff(np.sort(payload["trace_ids"], axis=-1), axis=-1) == 0):
            raise ValueError("trace/column IDs must be unique within each preparation")
        if not np.all(payload["configuration_digest"] == _configuration_digest(problem)):
            raise ValueError("column correlation origin model/parameters/probes/configuration changed; prepare a new origin")


def make_column_transport_problem(model, params, nuclear_treatment, *,
                                  probes=("current_x",), probe_callback=None):
    """Assemble the action-based correlation workflow on ordinary native CPA."""
    measurement = ColumnTransportMeasurement(probes, probe_callback)
    return Problem(model, params, nuclear_treatment, CPA(), measurement).validate()


def initialize_column_transport_state(problem, q, p, factor, *, time=0.,
                                      trajectory_id=0, seed=0):
    """Initialize an exact supplied density rho=L L† with Tr(rho)=1.

    ``factor`` has shape (nstates, rank), rank>=1. Columns need not be orthogonal,
    equal-weight, or individually normalized; zero columns are allowed. No
    normalization or finite-temperature approximation is silently applied.
    The caller owns the physical preparation represented by this factor.
    The stored state key uses an explicit threefry2x32 trajectory stream; CPA
    carries it unchanged. seed and trajectory_id are nonnegative uint32 values.
    """
    seed, trajectory_id, trajectory_key = _trajectory_key(seed, trajectory_id)
    problem.validate()
    if not isinstance(problem.measurement, ColumnTransportMeasurement):
        raise TypeError("problem must use ColumnTransportMeasurement")
    factor = np.asarray(factor)
    n = problem.model.spec.system.nstates
    if (factor.ndim != 2 or factor.shape[0] != n or not factor.shape[1]
            or factor.dtype.kind not in "iufc" or not np.isfinite(factor).all()):
        raise ValueError("factor must be finite numeric (nstates, rank) data with rank>=1")
    state = make_state(q, p, factor, time=time, trajectory_id=trajectory_id, seed=seed)
    if state.q.shape != problem.model.spec.system.q_shape:
        raise ValueError("initial coordinates do not match the model")
    factor = state.electronic
    trace = float(np.sum(np.abs(np.asarray(factor))**2, dtype=np.float64))
    if not np.isfinite(trace) or abs(trace-1.) > _tolerance(factor.dtype):
        raise ValueError("density factor must have unit trace; normalization is explicit")
    prescribed_q = np.asarray(problem.nuclear_treatment.point(state, 0.)[0])
    if (prescribed_q.shape != state.q.shape or np.iscomplexobj(prescribed_q)
            or not np.isfinite(prescribed_q).all()
            or not np.allclose(prescribed_q, state.q, atol=_tolerance(state.q.dtype), rtol=_tolerance(state.q.dtype))):
        raise ValueError("initial coordinates must match the prescribed path at origin time")
    validate_model_at(problem.model, problem.params, state.q)
    pieces = [factor]
    for probe in problem.measurement.probes:
        value = problem.measurement.action(problem, state, probe, factor)
        if not np.isfinite(np.asarray(value)).all():
            raise ValueError("initial physical current action contains nonfinite values")
        pieces.append(value)
    block = jnp.concatenate(pieces, axis=1)
    norms = jnp.sum(jnp.abs(block)**2, axis=0)
    if not np.isfinite(np.asarray(norms)).all():
        raise ValueError("initial column squared norms are not representable")
    rank = factor.shape[1]
    if rank > np.iinfo(np.int32).max:
        raise ValueError("density factor rank must fit in int32")
    payload = {"schema": jnp.int32(_SCHEMA), "rank": jnp.int32(rank),
               "nprobes": jnp.int32(len(problem.measurement.probes)),
               "time0": state.time, "q0": state.q, "p0": state.p,
               "trajectory_id0": state.trajectory_id, "norms_squared0": norms,
               "configuration_digest": jnp.asarray(_configuration_digest(problem)),
               "factor_digest": jnp.asarray(np.frombuffer(hashlib.sha256(np.asarray(factor).tobytes(order="C")).digest(), dtype=np.uint8)),
               "preparation_code": jnp.int32(0), "trace_ids": jnp.arange(rank, dtype=jnp.uint32),
               "trace_seed": jnp.uint32(0)}
    state = make_state(state.q, state.p, block, time=time, trajectory_id=trajectory_id,
                       seed=seed, method_state={_PAYLOAD: payload})
    state = state._replace(key=jax.random.key_data(trajectory_key))
    problem.measurement.validate_initial_state(problem, state)
    return state


def initialize_infinite_temperature_columns(problem, q, p, *, trace_ids, seed=0,
                                            trajectory_id=0, time=0.):
    """Prepare an unbiased beta=0 random-phase trace estimate, not finite T.

    Independent z_k have unit-magnitude components and E[z_k z_k†]=I. With K
    supplied IDs, L[:,k]=z_k/sqrt(N*K), so E[L L†]=I/N and its trace is exactly
    one mathematically. Returned C is the average over K independent trace
    estimates conditional on this one nuclear path; columns are not additional
    independent nuclear trajectories. No thermal ratio or finite-T claim is made.

    Explicit threefry2x32 keys fold in the trajectory ID, a fixed workflow stream tag, then each
    trace ID. Permuting/partitioning IDs preserves the underlying phases;
    factor scaling changes explicitly with K. Nuclear preparation remains the
    caller's separate responsibility. There is deliberately no beta argument.
    """
    ids = np.asarray(trace_ids)
    if (ids.ndim != 1 or not ids.size or ids.dtype.kind not in "iu"
            or np.any(ids < 0) or np.any(ids >= 2**32) or np.unique(ids).size != ids.size):
        raise ValueError("trace_ids must be unique nonnegative uint32 integer IDs")
    seed, trajectory_id, trajectory_key = _trajectory_key(seed, trajectory_id)
    n = problem.model.spec.system.nstates
    dtype = jnp.result_type(1.)
    key = jax.random.fold_in(trajectory_key, _PHASE_STREAM)
    keys = jax.vmap(lambda index: jax.random.fold_in(key, index))(jnp.asarray(ids, dtype=jnp.uint32))
    phases = jax.vmap(lambda draw: jax.random.uniform(draw, (n,), dtype=dtype))(keys)
    factor = jnp.exp(2j*jnp.pi*phases).T / jnp.sqrt(jnp.asarray(n*len(ids), dtype=dtype))
    state = initialize_column_transport_state(problem, q, p, factor, time=time,
                                               trajectory_id=trajectory_id, seed=seed)
    payload = dict(state.method_state[_PAYLOAD])
    payload.update(preparation_code=jnp.int32(1), trace_ids=jnp.asarray(ids, dtype=jnp.uint32),
                   trace_seed=jnp.uint32(seed))
    return state._replace(method_state={_PAYLOAD: payload})


__all__ = ["ColumnTransportMeasurement", "make_column_transport_problem",
           "initialize_column_transport_state", "initialize_infinite_temperature_columns"]
