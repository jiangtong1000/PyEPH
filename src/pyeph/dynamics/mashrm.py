"""Real finite-state Runeson--Manolopoulos MASH.

The fixed-basis complex mapping vector is propagated with the carrier h; the
scalar reference contributes to forces and conserved energy, but its removable
global electronic phase is omitted. This method uses the conditional-sphere RM
preparation/estimator, including at N=2, and is distinct from original MASH2.

The first implementation deliberately requires a complete isolated real
spectrum. Endpoint subdivision cannot discover arbitrarily fast paired
recrossings; timestep/subdivision convergence remains necessary.
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from pyeph.core._configuration import boolean_scalar, integer_scalar, real_scalar
from pyeph.core.problem import CoupledClassical
from pyeph.core.state import make_state
from pyeph.dynamics._mashrm_step import build_step as _build_step
from pyeph.dynamics.mashrm_mapping import (
    mapping_density,
    mapping_populations,
    sample_population_conditional,
)
from pyeph.execution.random import as_threefry_key
from pyeph.execution.runner import SimulationError
from pyeph.representations.adiabatic import (
    diagonalize,
    model_surfaces,
    surface_force,
    validate_surfaces,
)


STATUS_MESSAGES = {
    1: "invalid, complex, non-Hermitian, or degenerate Hamiltonian/force",
    2: "active surface is inconsistent with the largest mapping population",
    3: "event localization failed its residual or time-bracket requirement",
    4: "invalid, grazing, or unresolved outgoing impulse direction",
    5: "event capacity exhausted; reduce dt or increase max_events_per_step",
    6: "unresolved competing events or more than two largest populations tied",
    7: "integer diagnostic capacity exhausted; counters cannot advance safely",
}


class MASHRMError(SimulationError):
    """A failed bounded step retaining the last accepted physical state."""

    def __init__(self, message, state):
        super().__init__(message, last_valid_state=None, failed_state=state)


def _basis(value):
    if not isinstance(value, str) or value not in ("fixed", "adiabatic"):
        raise ValueError("basis must be 'fixed' or 'adiabatic'")
    return value


def _validate_model(model):
    spec = model.spec
    if (spec.system.nstates < 2 or spec.complex_valued or not spec.native_jax
            or spec.basis_kind != "fixed_orthonormal"):
        raise ValueError("MASHRM requires a real native fixed orthonormal model with N>=2")
    if not spec.force_support or any(not callable(getattr(model, name, None))
                                    for name in ("reference_gradient", "contract_gradient")):
        raise ValueError("MASHRM requires complete reference and electronic forces")


def _method_state(active, dtype):
    result = {name: jnp.int32(0) for name in (
        "status", "events", "event_attempts", "accepted", "frustrated", "localization_iterations")}
    result.update(active=jnp.int32(active))
    result.update({name: jnp.zeros((), dtype) for name in (
        "max_event_residual", "max_event_bracket_width", "max_impulse_energy_error")})
    return result


def mapping_state(model, params, q, p, mapping, *, basis="fixed", active=None,
                  time=0., step=0, trajectory_id=0, seed=0, gap_tolerance=1e-10):
    """Construct one deterministic mapping state, without sampling an ensemble.

    The vector is supplied in the declared preparation basis and stored in the
    fixed model basis. Active always denotes a sorted adiabatic surface. A tied
    largest adiabatic population requires an explicit incoming active surface.
    """
    _validate_model(model)
    _basis(basis)
    mapping = np.asarray(mapping)
    n = model.spec.system.nstates
    if mapping.shape != (n,) or not np.isfinite(mapping).all():
        raise ValueError("mapping must be one finite N-component vector")
    if not np.isclose(np.vdot(mapping, mapping).real, 1., atol=1e-10, rtol=1e-10):
        raise ValueError("mapping must have unit norm")
    state = make_state(q, p, mapping, time=time, step=step, trajectory_id=trajectory_id, seed=seed)
    if state.q.dtype != jnp.float64 or state.electronic.dtype != jnp.complex128:
        raise ValueError("MASHRM currently requires JAX x64 and double-precision states")
    if state.q.shape != model.spec.system.q_shape:
        raise ValueError("coordinate shape must match the model")
    h = model.apply(params, state.q, jnp.eye(n, dtype=state.q.dtype))
    if not np.allclose(np.imag(np.asarray(h)), 0., atol=1e-12, rtol=0.):
        raise ValueError("MASHRM requires an actually real Hamiltonian")
    data = validate_surfaces(diagonalize(h, gap_tolerance=gap_tolerance))
    c = data.vectors @ state.electronic if basis == "adiabatic" else state.electronic
    populations = np.abs(np.asarray(data.vectors.conj().T @ c))**2
    largest = populations.max()
    if active is None:
        if np.count_nonzero(largest-populations <= 1e-12) != 1:
            raise ValueError("tied largest populations require an explicit active surface")
        active = int(populations.argmax())
    active = integer_scalar(active, "active")
    if not 0 <= active < n or largest-populations[active] > 1e-12:
        raise ValueError("active must select a largest adiabatic mapping population")
    return state._replace(electronic=c, method_state=_method_state(active, state.q.dtype))


def sample_population(model, params, q, p, *, population=0, basis="adiabatic",
                      time=0., step=0, trajectory_id=0, seed=0, gap_tolerance=1e-10):
    """Prepare a conditional-sphere RM population in a declared basis.

    Nuclear sampling is independent and supplied via q,p. A stable trajectory
    ID determines the draw, independent of batching. Fixed-basis preparation
    does not prescribe the active adiabatic surface.
    """
    # make_state validates ID/seed before converting either to an RNG argument.
    base = make_state(q, p, np.eye(1, model.spec.system.nstates)[0],
                      time=time, step=step, trajectory_id=trajectory_id, seed=seed)
    next_key, sample_key = jax.random.split(as_threefry_key(base.key))
    mapping = sample_population_conditional(sample_key, model.spec.system.nstates,
                                           population=population)
    state = mapping_state(model, params, q, p, mapping, basis=basis, time=time, step=step,
                          trajectory_id=trajectory_id, seed=seed, gap_tolerance=gap_tolerance)
    return state._replace(key=jax.random.key_data(next_key))


def total_energy(model, params, state, masses):
    """Kinetic plus scalar reference plus the active adiabatic carrier energy."""
    data = model_surfaces(model, params, state.q)
    return (jnp.sum(state.p**2/(2*jnp.asarray(masses)))
            + model.reference_energy(params, state.q)
            + data.energies[state.method_state["active"]])


@dataclass(frozen=True)
class MASHRMPopulation:
    """One-time RM population/density estimators in an explicit output basis.

    Individual estimates may be negative. The active surface is a separate
    diagnostic, not the RM population estimator. No correlation prescription
    follows from multiplying these one-time estimators.
    """

    basis: str = "fixed"
    include_density: bool = False
    include_nuclei: bool = False
    supports_mashrm = True

    def __post_init__(self):
        _basis(self.basis)
        for name in ("include_density", "include_nuclei"):
            object.__setattr__(self, name, boolean_scalar(getattr(self, name), name))

    def validate(self, problem):
        if not isinstance(problem.method, MASHRM):
            raise ValueError("MASHRMPopulation requires MASHRM")

    def evaluate(self, problem, state):
        c = state.electronic
        if self.basis == "adiabatic":
            c = model_surfaces(problem.model, problem.params, state.q).vectors.conj().T @ c
        result = dict(population=mapping_populations(c),
                      mapping_norm=jnp.vdot(c, c).real,
                      energy=total_energy(problem.model, problem.params, state,
                                          problem.nuclear_treatment.masses),
                      **state.method_state)
        if self.include_density:
            result["density"] = mapping_density(c)
        if self.include_nuclei:
            result.update(q=state.q, p=state.p)
        return result


@dataclass(frozen=True)
class MASHRM:
    """Real RM mapping dynamics with bounded, isolated pair events.

    Each smooth segment uses velocity Verlet and a midpoint electronic matrix
    exponential. Every endpoint-bracketed competitor is localized; overlapping
    time brackets and top triple ties fail explicitly. q and c are never
    projected at an event. The all-state RM direction determines the impulse.
    """

    event_substeps: int = 4
    max_events_per_step: int = 8
    event_tolerance: float = 1e-10
    event_time_tolerance: float = 1e-10
    bisection_iterations: int = 48
    gap_tolerance: float = 1e-10
    real_tolerance: float = 1e-12
    direction_tolerance: float = 1e-14
    rate_tolerance: float = 1e-12
    uses_mapping_estimator = True
    numerical_scheme = "rm_fixed_diabatic_vv_midpoint"

    def __post_init__(self):
        for name in ("event_substeps", "max_events_per_step", "bisection_iterations"):
            value = integer_scalar(getattr(self, name), name)
            if not 1 <= value <= np.iinfo(np.int32).max:
                raise ValueError(f"{name} must be a positive int32-representable integer")
            object.__setattr__(self, name, value)
        for name in ("event_tolerance", "event_time_tolerance", "gap_tolerance", "real_tolerance",
                     "direction_tolerance", "rate_tolerance"):
            value = real_scalar(getattr(self, name), name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
            object.__setattr__(self, name, value)

    def validate(self, problem):
        _validate_model(problem.model)
        if not isinstance(problem.nuclear_treatment, CoupledClassical):
            raise ValueError("MASHRM requires coupled canonical classical nuclei")
        if not getattr(problem.measurement, "supports_mashrm", False):
            raise ValueError("MASHRM requires an explicit RM mapping measurement")

    def validate_state(self, state, *, batch=False):
        if (state.q.dtype != jnp.float64 or state.p.dtype != jnp.float64
                or state.electronic.dtype != jnp.complex128 or state.time.dtype != jnp.float64):
            raise ValueError("MASHRM currently requires double-precision states and JAX x64")
        if self.event_tolerance < 8*np.finfo(state.q.dtype).eps:
            raise ValueError("event_tolerance is below usable coordinate precision")
        shape = state.electronic.shape
        if len(shape) != (2 if batch else 1) or shape[-1] < 2:
            raise ValueError("MASHRM requires one N-component mapping vector per trajectory")
        norm = np.sum(abs(np.asarray(state.electronic))**2, axis=-1)
        if not np.allclose(norm, 1., atol=1e-8, rtol=1e-8):
            raise ValueError("MASHRM mapping vectors must have unit norm")
        expected = _method_state(0, state.q.dtype)
        m = state.method_state
        if not isinstance(m, dict) or set(m) != set(expected):
            raise ValueError("use a MASHRM state constructor to initialize method_state")
        target_shape = (state.time.size,) if batch else ()
        for name, value in m.items():
            array = np.asarray(value)
            if array.dtype != np.dtype(expected[name].dtype):
                raise ValueError(f"MASHRM {name} must use constructor dtype {expected[name].dtype}")
            if array.shape != target_shape or not np.isfinite(array).all() or np.any(array < 0):
                raise ValueError(f"invalid MASHRM method_state field {name}")
        if np.any(np.asarray(m["active"]) >= shape[-1]):
            raise ValueError("active surface is outside the mapping vector")
        if (np.any(np.asarray(m["events"]) != np.asarray(m["accepted"]+m["frustrated"]))
                or np.any(np.asarray(m["event_attempts"]) < np.asarray(m["events"]))):
            raise ValueError("inconsistent MASHRM event counters")
        self.validate_result(state)

    def validate_initial_state(self, problem, state, *, batch=False):
        """Check the prepared active surface against the current model parameters.

        This host preflight also applies to zero-step runs and checkpoints.
        Native batches evaluate spectra and forces with one array-mapped call,
        without retaining parameters between calls. Explicit pair boundaries
        remain admissible; the event kernel determines their direction.
        """
        model, params = problem.model, problem.params
        n = model.spec.system.nstates

        def inspect(q, c, active):
            h = model.apply(params, q, jnp.eye(n, dtype=q.dtype))
            data = diagonalize(h, gap_tolerance=self.gap_tolerance)
            reference = jnp.asarray(model.reference_energy(params, q))
            if reference.ndim or jnp.iscomplexobj(reference):
                raise ValueError("MASHRM initial reference energy must be a finite real scalar")
            force = surface_force(model, params, q, data, active)
            if force.value.shape != q.shape or jnp.iscomplexobj(force.value):
                raise ValueError("MASHRM initial active force must be finite and real")
            spectral = (jnp.all(jnp.isfinite(data.energies))
                        & jnp.all(jnp.isfinite(data.vectors))
                        & jnp.all(jnp.isfinite(data.gaps)) & ~jnp.any(data.near_degenerate)
                        & jnp.allclose(data.vectors.conj().T@data.vectors,
                                       jnp.eye(n), rtol=1e-8, atol=1e-8))
            populations = jnp.abs(data.vectors.conj().T@c)**2
            ownership = jnp.max(populations)-populations[active] <= self.event_tolerance
            return jnp.stack((jnp.all(jnp.abs(jnp.imag(h)) <= self.real_tolerance),
                              spectral, jnp.isfinite(reference),
                              force.valid & jnp.all(jnp.isfinite(force.value)), ownership))

        checks = (jax.vmap(inspect)(state.q, state.electronic, state.method_state["active"])
                  if batch else inspect(state.q, state.electronic, state.method_state["active"]))
        checks = np.asarray(jax.device_get(checks)).reshape(-1, 5)
        messages = (
            "initial Hamiltonian must be actually real",
            "initial Hamiltonian must have a finite Hermitian, isolated spectrum",
            "initial reference energy must be a finite real scalar",
            "initial active force must be finite and real",
            "initial active surface must select a largest mapping population for the current model parameters",
        )
        for index, message in enumerate(messages):
            invalid = np.flatnonzero(~checks[:, index])
            if invalid.size:
                raise ValueError(f"MASHRM {message}; invalid trajectory lanes {invalid[:8].tolist()}")

    def validate_integrator(self, integrator):
        if integrator.electronic != "exponential_midpoint" or integrator.electronic_substeps != 1:
            raise ValueError("MASHRM requires Integrator(electronic='exponential_midpoint', "
                             "electronic_substeps=1); use event_substeps for subdivision")

    def step_succeeded(self, state):
        return state.method_state["status"] == 0

    def validate_result(self, state):
        status = np.asarray(state.method_state["status"])
        if np.any(status):
            details = "; ".join(f"{int(c)}: {STATUS_MESSAGES.get(int(c), 'unknown failure')}"
                                for c in np.unique(status[status != 0]))
            raise MASHRMError(f"MASHRM failed ({details}); inspect failed_state diagnostics", state)

    def build_step(self, problem, integrator):
        """Bind the RM smooth propagator and isolated-pair event controller."""
        self.validate_integrator(integrator)
        return _build_step(self, problem, integrator)
