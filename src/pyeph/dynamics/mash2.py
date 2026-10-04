"""Two-state MASH with fixed-diabatic, piecewise VV/midpoint propagation.

Mapping amplitudes encode a unit spin. They are NOT wavefunction estimators.
Only population-to-population preparation/measurement is implemented here.
See docs/MASH2.md for the numerical algorithm, scientific scope and references.
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from pyeph.core._configuration import boolean_scalar, integer_scalar, real_scalar
from pyeph.core.problem import CoupledClassical
from pyeph.core.state import make_state
from pyeph.execution.random import trajectory_keys
from pyeph.execution.runner import SimulationError
from pyeph.integrators.electronic import exponential_action
from pyeph.integrators.events import bisect_crossing
from pyeph.integrators.impulses import (
    MomentumImpulse as MomentumImpulse,
    energy_conserving_impulse,
)
from pyeph.representations.adiabatic import (
    derivative_coupling,
    diagonalize,
    model_surfaces,
    surface_force,
    validate_surfaces,
)


STATUS_MESSAGES = {
    1: "invalid, complex, non-Hermitian, or degenerate Hamiltonian/force",
    2: "active surface disagrees with the mapping hemisphere",
    3: "crossing localization did not converge within its iteration budget",
    4: "crossing has an invalid or vanishing NAC/momentum direction",
    5: "event capacity exhausted; reduce dt or increase max_events_per_step",
}


class MASHError(SimulationError):
    """Failed bounded step, retaining the finite state and diagnostics."""

    def __init__(self, message, state):
        super().__init__(message, last_valid_state=None, failed_state=state)


def momentum_impulse(momentum, masses, nac, energy_change, *, direction_tolerance=1e-14):
    """Conserve energy by rescaling only the mass-weighted NAC component.

    ``energy_change`` is the proposed final-minus-initial adiabatic energy.
    An energetically forbidden upward hop reflects this component. Canonical
    p and positive diagonal masses are required; ``nac`` has the shape of p.
    A zero incident component is an unresolved grazing event, not assigned an
    arbitrary downhill direction. Inputs are validated inside JIT via ``valid``.
    """
    return energy_conserving_impulse(momentum, masses, nac, energy_change,
                                      direction_tolerance=direction_tolerance)


def sample_population_spin(key, active=0, *, shape=()):
    """Importance sample a population-prepared unit spin: z=±sqrt(U).

    This absorbs the population-population MASH weight 2|Sz| into the initial
    distribution. ``active=0`` selects the lower hemisphere; 1 the upper.
    ``shape`` is useful for statistical tests or vectorized initial preparation.
    """
    if active not in (0, 1):
        raise ValueError("active must be 0 (lower) or 1 (upper)")
    z_key, phi_key = jax.random.split(key)
    dtype = jnp.result_type(1.0)
    # Exclude the exactly equatorial floating-point sample to define its side.
    uniform = jax.random.uniform(z_key, shape, dtype=dtype,
                                 minval=jnp.finfo(dtype).eps, maxval=1.0)
    z = (2 * active - 1) * jnp.sqrt(uniform)
    phi = 2 * jnp.pi * jax.random.uniform(phi_key, shape, dtype=dtype)
    radius = jnp.sqrt(jnp.maximum(0, 1 - z*z))
    return jnp.stack((radius*jnp.cos(phi), radius*jnp.sin(phi), z), axis=-1)


def _method_state(active, dtype):
    return dict(active=jnp.int32(active), status=jnp.int32(0), events=jnp.int32(0),
                event_attempts=jnp.int32(0),
                accepted=jnp.int32(0), frustrated=jnp.int32(0),
                localization_iterations=jnp.int32(0),
                max_event_residual=jnp.zeros((), dtype),
                max_impulse_energy_error=jnp.zeros((), dtype))


def _spinor(spin):
    """Sorted lower/upper amplitudes: Sy=2 Im(conj(c_upper)*c_lower)."""
    phi = jnp.arctan2(spin[1], spin[0])
    lower = jnp.sqrt(jnp.maximum(0, (1 - spin[2]) / 2))
    upper = jnp.sqrt(jnp.maximum(0, (1 + spin[2]) / 2)) * jnp.exp(-1j * phi)
    return jnp.stack((lower, upper))


def mapping_state(model, params, q, p, spin, *, active=None, time=0.0, step=0,
                  trajectory_id=0, seed=0, gap_tolerance=1e-10):
    """Construct a deterministic spin trajectory; this is not a physical sampler.

    Spin components refer to real adiabatic eigenvectors returned by ``eigh``.
    A change of their relative sign changes Sx and Sy. An equatorial initial
    spin requires an explicit active surface to specify the incoming side.
    """
    spin = np.asarray(spin, dtype=float)
    if spin.shape != (3,) or not np.isfinite(spin).all():
        raise ValueError("spin must be a finite three-vector")
    if not np.isclose(np.dot(spin, spin), 1, atol=1e-10, rtol=1e-10):
        raise ValueError("mapping spin must have unit length")
    if model.spec.system.nstates != 2 or model.spec.complex_valued:
        raise ValueError("MASH2 requires a real two-state Hamiltonian")
    if active is None:
        if spin[2] == 0:
            raise ValueError("an equatorial initial spin requires an explicit active surface")
        active = int(spin[2] > 0)
    if active not in (0, 1) or (2 * active - 1) * spin[2] < -1e-12:
        raise ValueError("active surface must match the spin hemisphere")
    state = make_state(q, p, [1, 0], time=time, step=step, trajectory_id=trajectory_id, seed=seed)
    h = model.apply(params, state.q, jnp.eye(2, dtype=state.q.dtype))
    if not np.allclose(np.imag(np.asarray(h)), 0, atol=1e-12, rtol=0):
        raise ValueError("MASH2 requires an actually real Hamiltonian")
    surfaces = validate_surfaces(diagonalize(h, gap_tolerance=gap_tolerance))
    c = surfaces.vectors @ _spinor(jnp.asarray(spin, dtype=state.q.dtype))
    return state._replace(electronic=c, method_state=_method_state(active, state.q.dtype))


def sample_adiabatic_population(model, params, q, p, *, active=0, time=0.0,
                                trajectory_id=0, seed=0, gap_tolerance=1e-10):
    """Prepare one adiabatic population for the MASH population estimator.

    Nuclear sampling is supplied separately by q,p. Keys are derived from the
    global seed and stable trajectory ID, so batching does not change samples.
    """
    key = trajectory_keys(seed, [trajectory_id], typed=True)[0]
    next_key, sample_key = jax.random.split(key)
    state = mapping_state(model, params, q, p, sample_population_spin(sample_key, active),
                          active=active, time=time, trajectory_id=trajectory_id, seed=seed,
                          gap_tolerance=gap_tolerance)
    return state._replace(key=jax.random.key_data(next_key))


def spin_z(surfaces, electronic):
    """Gauge-independent hemisphere coordinate in the sorted adiabatic basis."""
    amplitudes = surfaces.vectors.conj().T @ electronic
    return jnp.abs(amplitudes[1])**2 - jnp.abs(amplitudes[0])**2


def total_energy(model, params, state, masses):
    """MASH conserved Hamiltonian: kinetic + reference + active eigenvalue."""
    surfaces = model_surfaces(model, params, state.q)
    return (jnp.sum(state.p**2 / (2 * jnp.asarray(masses)))
            + model.reference_energy(params, state.q)
            + surfaces.energies[state.method_state["active"]])


@dataclass(frozen=True)
class MASHPopulation:
    """Active-state population estimator for |Sz|-weighted preparation.

    The same samples/weight do not define coherence or arbitrary correlation
    estimators. At an exactly localized event this uses the outgoing active side.
    """

    include_nuclei: bool = False
    supports_mash2 = True

    def __post_init__(self):
        object.__setattr__(self, "include_nuclei", boolean_scalar(self.include_nuclei, "include_nuclei"))

    def validate(self, problem):
        if not isinstance(problem.method, MASH2):
            raise ValueError("MASHPopulation requires MASH2")

    def evaluate(self, problem, state):
        surfaces = model_surfaces(problem.model, problem.params, state.q)
        result = dict(population=jax.nn.one_hot(state.method_state["active"], 2),
                      mapping_norm=jnp.vdot(state.electronic, state.electronic).real,
                      spin_z=spin_z(surfaces, state.electronic),
                      energy=total_energy(problem.model, problem.params, state,
                                          problem.nuclear_treatment.masses),
                      **{k: state.method_state[k] for k in (
                          "events", "event_attempts", "accepted", "frustrated", "status",
                          "localization_iterations", "max_event_residual",
                          "max_impulse_energy_error")})
        if self.include_nuclei:
            result.update(q=state.q, p=state.p)
        return result


@dataclass(frozen=True)
class MASH2:
    """Real two-state MASH with bounded event-resolved fixed-basis propagation.

    ``event_substeps`` subdivides each nuclear step; sign changes are sought in
    each interval. It cannot guarantee discovery of arbitrarily fast paired
    recrossings. Reduce dt/increase subdivision to establish convergence.
    """

    event_substeps: int = 4
    max_events_per_step: int = 8
    event_tolerance: float = 1e-10
    bisection_iterations: int = 40
    gap_tolerance: float = 1e-10
    real_tolerance: float = 1e-12
    direction_tolerance: float = 1e-14
    uses_mapping_estimator = True
    numerical_scheme = "fixed_diabatic_pc_midpoint"

    def __post_init__(self):
        for name in ("event_substeps", "max_events_per_step", "bisection_iterations"):
            object.__setattr__(self, name, integer_scalar(getattr(self, name), name))
            if not isinstance(getattr(self, name), int) or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("event_tolerance", "gap_tolerance", "real_tolerance", "direction_tolerance"):
            object.__setattr__(self, name, real_scalar(getattr(self, name), name))
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")

    def validate(self, problem):
        spec = problem.model.spec
        if (spec.system.nstates != 2 or spec.complex_valued
                or spec.basis_kind != "fixed_orthonormal"):
            raise ValueError("MASH2 requires a real two-state fixed orthonormal model")
        if not isinstance(problem.nuclear_treatment, CoupledClassical):
            raise ValueError("MASH2 requires coupled canonical classical nuclei")
        if not spec.force_support or any(not callable(getattr(problem.model, n, None))
                                        for n in ("reference_gradient", "contract_gradient")):
            raise ValueError("MASH2 requires complete reference and electronic forces")
        if not getattr(problem.measurement, "supports_mash2", False):
            raise ValueError("MASH2 requires an explicit MASHPopulation measurement")

    def validate_state(self, state, *, batch=False):
        if self.event_tolerance < 8*np.finfo(state.q.dtype).eps:
            raise ValueError("event_tolerance is below usable coordinate precision; "
                             "enable JAX x64 or increase the event tolerance")
        shape = state.electronic.shape
        if len(shape) != (2 if batch else 1) or shape[-1] != 2:
            raise ValueError("MASH2 requires one two-component mapping vector per trajectory")
        norm = np.sum(np.abs(np.asarray(state.electronic))**2, axis=-1)
        if not np.allclose(norm, 1, atol=1e-8, rtol=1e-8):
            raise ValueError("MASH2 mapping vectors must be normalized")
        expected = _method_state(0, state.q.dtype)
        if not isinstance(state.method_state, dict) or set(state.method_state) != set(expected):
            raise ValueError("use a MASH2 state constructor to initialize method_state")
        target_shape = (state.time.size,) if batch else ()
        for name, x in state.method_state.items():
            array = np.asarray(x)
            if array.dtype != np.dtype(expected[name].dtype):
                raise ValueError(f"MASH2 {name} must use constructor dtype {expected[name].dtype}")
            if array.shape != target_shape or not np.isfinite(array).all():
                raise ValueError(f"invalid MASH2 method_state field {name}")
            if name in ("active", "status", "events", "event_attempts", "accepted", "frustrated",
                        "localization_iterations"):
                if not np.issubdtype(array.dtype, np.integer) or np.any(array < 0):
                    raise ValueError(f"MASH2 {name} must contain nonnegative integers")
        if np.any(np.asarray(state.method_state["active"]) > 1):
            raise ValueError("active must be 0 or 1")
        m = state.method_state
        if (np.any(np.asarray(m["events"]) != np.asarray(m["accepted"] + m["frustrated"]))
                or np.any(np.asarray(m["event_attempts"]) < np.asarray(m["events"]))):
            raise ValueError("inconsistent MASH2 event counters")
        self.validate_result(state)

    def validate_initial_state(self, problem, state, *, batch=False):
        """Validate the current Hamiltonian and prepared hemisphere before output.

        Native batches evaluate spectra and forces with one array-mapped call;
        opaque providers retain per-lane evaluation. No model parameters are
        retained between calls. Equatorial states remain admissible because
        their incoming/outgoing direction belongs to the event kernel.
        """
        model, params = problem.model, problem.params

        def inspect(q, c, active):
            h = model.apply(params, q, jnp.eye(2, dtype=q.dtype))
            data = diagonalize(h, gap_tolerance=self.gap_tolerance)
            reference = jnp.asarray(model.reference_energy(params, q))
            if reference.ndim or jnp.iscomplexobj(reference):
                raise ValueError("MASH2 initial reference energy must be a finite real scalar")
            force = surface_force(model, params, q, data, active)
            if force.value.shape != q.shape or jnp.iscomplexobj(force.value):
                raise ValueError("MASH2 initial active force must be finite and real")
            spectral = (jnp.all(jnp.isfinite(data.energies))
                        & jnp.all(jnp.isfinite(data.vectors))
                        & jnp.all(jnp.isfinite(data.gaps)) & ~jnp.any(data.near_degenerate)
                        & jnp.allclose(data.vectors.conj().T@data.vectors,
                                       jnp.eye(2), rtol=1e-8, atol=1e-8))
            ownership = (2*active-1)*spin_z(data, c) >= -self.event_tolerance
            return jnp.stack((jnp.all(jnp.abs(jnp.imag(h)) <= self.real_tolerance),
                              spectral, jnp.isfinite(reference),
                              force.valid & jnp.all(jnp.isfinite(force.value)), ownership))

        if batch and model.spec.native_jax:
            checks = jax.vmap(inspect)(state.q, state.electronic, state.method_state["active"])
        elif batch:
            checks = jnp.stack([inspect(q, c, active) for q, c, active in
                                zip(state.q, state.electronic, state.method_state["active"])])
        else:
            checks = inspect(state.q, state.electronic, state.method_state["active"])
        checks = np.asarray(jax.device_get(checks)).reshape(-1, 5)
        messages = (
            "initial Hamiltonian must be actually real",
            "initial Hamiltonian must have a finite Hermitian, isolated spectrum",
            "initial reference energy must be a finite real scalar",
            "initial active force must be finite and real",
            "initial active surface must match the mapping hemisphere for the current model parameters",
        )
        for index, message in enumerate(messages):
            invalid = np.flatnonzero(~checks[:, index])
            if invalid.size:
                raise ValueError(f"MASH2 {message}; invalid trajectory lanes {invalid[:8].tolist()}")

    def step_succeeded(self, state):
        """Runner hook: preserve the last physical event time after failure."""
        return state.method_state["status"] == 0

    def validate_result(self, state):
        status = np.asarray(state.method_state["status"])
        if np.any(status):
            codes = np.unique(status[status != 0])
            details = "; ".join(f"{int(c)}: {STATUS_MESSAGES.get(int(c), 'unknown failure')}"
                                for c in codes)
            raise MASHError(f"MASH2 failed ({details}); inspect failed_state diagnostics", state)

    def validate_integrator(self, integrator):
        """Reject a mismatched numerical policy before initial observations."""
        if integrator.electronic != "exponential_midpoint" or integrator.electronic_substeps != 1:
            raise ValueError("MASH2 requires Integrator(electronic='exponential_midpoint', "
                             "electronic_substeps=1); use MASH2.event_substeps for subdivision")

    def build_step(self, problem, integrator):
        self.validate_integrator(integrator)
        model, params = problem.model, problem.params
        mass = jnp.asarray(problem.nuclear_treatment.masses)
        width = integrator.dt / self.event_substeps

        def surfaces_at(q):
            h = model.apply(params, q, jnp.eye(2, dtype=q.dtype))
            real = jnp.max(jnp.abs(jnp.imag(h))) <= self.real_tolerance
            surfaces = diagonalize(h, gap_tolerance=self.gap_tolerance)
            valid = (real & jnp.all(jnp.isfinite(surfaces.energies))
                     & ~jnp.any(surfaces.near_degenerate)
                     & jnp.isfinite(model.reference_energy(params, q)))
            return h, surfaces, valid

        def mark(state, code):
            return state._replace(method_state={**state.method_state, "status": jnp.int32(code)})

        def smooth(state, duration):
            active = state.method_state["active"]
            _, start, valid_start = surfaces_at(state.q)
            f0 = surface_force(model, params, state.q, start, active)
            p_half = state.p + duration / 2 * f0.value
            q = state.q + duration * p_half / mass
            _, end, valid_end = surfaces_at(q)
            f1 = surface_force(model, params, q, end, active)
            p = p_half + duration / 2 * f1.value
            h_mid, _, valid_mid = surfaces_at((state.q + q) / 2)
            c = exponential_action(h_mid, state.electronic, duration)
            valid = (valid_start & valid_end & valid_mid & f0.valid & f1.valid
                     & jnp.all(jnp.isfinite(q)) & jnp.all(jnp.isfinite(p))
                     & jnp.all(jnp.isfinite(c)))
            return state._replace(q=q, p=p, electronic=c, time=state.time + duration), end, valid

        def equator(state, surfaces):
            c = surfaces.vectors.conj().T @ state.electronic
            # Preserve relative phase while removing only localization error.
            c = c / jnp.abs(c) / jnp.sqrt(2.0)
            return state._replace(electronic=surfaces.vectors @ c)

        def hop(start, duration, events_this_step):
            side = 2 * start.method_state["active"] - 1

            def value_at(t):
                candidate, surfaces, valid = smooth(start, t)
                return jnp.where(valid, side * spin_z(surfaces, candidate.electronic), jnp.nan)

            root = bisect_crossing(value_at, duration, iterations=self.bisection_iterations,
                                    tolerance=self.event_tolerance)
            event, surfaces, valid = smooth(start, root.time)
            nac = derivative_coupling(model, params, event.q, surfaces, 0, 1)
            energy_change = (1 - 2 * start.method_state["active"]) * surfaces.gaps[0, 1]
            impulse = momentum_impulse(event.p, mass, jnp.real(nac.value), energy_change,
                                       direction_tolerance=self.direction_tolerance)
            valid_impulse = (valid & nac.valid & impulse.valid
                             & (jnp.max(jnp.abs(jnp.imag(nac.value))) <= self.real_tolerance))
            code = jnp.where(~root.converged, 3, jnp.where(~valid_impulse, 4, 0))
            event = equator(event, surfaces)
            active = jnp.where(impulse.accepted, 1 - start.method_state["active"],
                               start.method_state["active"])
            old = start.method_state
            diagnostics = {**old, "active": active,
                "events": old["events"] + 1,
                "event_attempts": old["event_attempts"] + 1,
                "accepted": old["accepted"] + impulse.accepted.astype(jnp.int32),
                "frustrated": old["frustrated"] + (~impulse.accepted).astype(jnp.int32),
                "localization_iterations": old["localization_iterations"] + root.iterations,
                "max_event_residual": jnp.maximum(old["max_event_residual"], jnp.abs(root.residual)),
                "max_impulse_energy_error": jnp.maximum(old["max_impulse_energy_error"],
                                                         jnp.abs(impulse.energy_error))}
            updated = event._replace(p=impulse.momentum, method_state=diagnostics)
            # Failed proposals never contaminate the last finite accepted state.
            failed = mark(start, code)
            failed = failed._replace(method_state={**failed.method_state,
                "event_attempts": old["event_attempts"] + 1,
                "max_event_residual": jnp.maximum(old["max_event_residual"],
                    jnp.where(jnp.isfinite(root.residual), jnp.abs(root.residual), 0.0)),
                "localization_iterations": old["localization_iterations"] + root.iterations})
            return jax.lax.cond(code == 0,
                lambda: (updated, jnp.maximum(0, duration-root.time), events_this_step + 1),
                lambda: (failed, jnp.zeros_like(duration), events_this_step))

        def interval(state, events_this_step):
            remaining = jnp.asarray(width, dtype=state.q.dtype)

            def condition(carry):
                s, left, _, attempts = carry
                return ((left > 0) & (s.method_state["status"] == 0)
                        & (attempts <= self.max_events_per_step))

            def body(carry):
                s, left, count, attempts = carry
                trial, end, valid = smooth(s, left)
                side = 2*s.method_state["active"] - 1
                crossing = side * spin_z(end, trial.electronic) < 0

                def handle_crossing():
                    def exhausted():
                        failed = mark(s, 5)
                        failed = failed._replace(method_state={**failed.method_state,
                            "event_attempts": s.method_state["event_attempts"] + 1})
                        return failed, jnp.zeros_like(left), count

                    return jax.lax.cond(count < self.max_events_per_step,
                        lambda: hop(s, left, count),
                        exhausted)

                result = jax.lax.cond(valid,
                    lambda: jax.lax.cond(crossing, handle_crossing,
                                        lambda: (trial, jnp.zeros_like(left), count)),
                    lambda: (mark(s, 1), jnp.zeros_like(left), count))
                return *result, attempts + 1

            state, remaining, count, _ = jax.lax.while_loop(
                condition, body, (state, remaining, events_this_step, jnp.int32(0)))
            state = jax.lax.cond((remaining > 0) & (state.method_state["status"] == 0),
                                 lambda: mark(state, 5), lambda: state)
            return state, count

        def step(state):
            _, surfaces, valid = surfaces_at(state.q)
            side = 2*state.method_state["active"] - 1
            consistent = side*spin_z(surfaces, state.electronic) >= -10*self.event_tolerance
            state = jax.lax.cond((state.method_state["status"] == 0) & ~valid,
                                 lambda: mark(state, 1), lambda: state)
            state = jax.lax.cond((state.method_state["status"] == 0) & ~consistent,
                                 lambda: mark(state, 2), lambda: state)
            original_time, original_step = state.time, state.step

            def body(_, carry):
                s, count = carry
                return jax.lax.cond(s.method_state["status"] == 0,
                                    lambda: interval(s, count), lambda: (s, count))

            state, _ = jax.lax.fori_loop(0, self.event_substeps, body, (state, jnp.int32(0)))
            success = state.method_state["status"] == 0
            return state._replace(time=jnp.where(success, original_time + integrator.dt, state.time),
                                  step=jnp.where(success, original_step + 1, state.step))

        return step
