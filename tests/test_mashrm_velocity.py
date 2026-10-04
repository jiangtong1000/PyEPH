"""Independent conditional moments and exact static-H RM velocity correlations."""

from dataclasses import dataclass, replace
from fractions import Fraction
from math import fsum

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.linalg import expm

from pyeph import CoupledClassical, Execution, Integrator, Problem, Simulation
from pyeph.core.contracts import ProbeContext
from pyeph.dynamics.mashrm import MASHRM, mapping_state
from pyeph.models.epc import LinearEPCModel
from pyeph.observables.transport.mashrm import (
    FixedPositionVelocity,
    RMVelocity,
    rm_gamma,
    rm_velocity,
)


def sector_quadrature(active=0):
    """Independent exact polynomial/phase quadrature for a three-state sector.

    Uniform simplex density conditional on P0 being the maximum is 6. Its
    polygon splits at P0=1/2. Two Gauss points per coordinate exactly integrate
    the required fourth moments after the independent phase sum. Three phase
    roots kill every nonzero harmonic through degree two; global phase is set
    to zero because the measured algebra is invariant under it.
    """
    nodes, weights = np.polynomial.legendre.leggauss(2)
    samples, quadrature_weights = [], []
    for left, right in ((1/3, .5), (.5, 1.)):
        for point, weight_x in zip(nodes, weights, strict=True):
            x = (left+right)/2+point*(right-left)/2
            lower, upper = max(0., 1-2*x), min(x, 1-x)
            for point_y, weight_y in zip(nodes, weights, strict=True):
                y = (lower+upper)/2+point_y*(upper-lower)/2
                amplitude = np.sqrt([x, y, 1-x-y])
                for phi1 in 2*np.pi*np.arange(3)/3:
                    for phi2 in 2*np.pi*np.arange(3)/3:
                        c = amplitude*np.exp(1j*np.array([0., phi1, phi2]))
                        c[[0, active]] = c[[active, 0]]
                        samples.append(c)
                        quadrature_weights.append(6*weight_x*(right-left)/2*weight_y*(upper-lower)/2/9)
    return np.asarray(samples), np.asarray(quadrature_weights)


@pytest.mark.parametrize("n,expected", [(2, Fraction(1, 6)), (3, Fraction(47, 432)),
                                        (4, Fraction(67, 864))])
def test_gamma_exact_rational_values(n, expected):
    assert rm_gamma(n) == pytest.approx(float(expected), abs=3e-17)


def test_gamma_against_independent_conditional_simplex_integration():
    c, weights = sector_quadrature()
    populations = abs(c)**2
    np.testing.assert_allclose(weights.sum(), 1., atol=3e-15)
    assert np.all(populations[:, 0] > populations[:, 1:].max(axis=1))
    np.testing.assert_allclose(weights@populations[:, 0], float(Fraction(11, 18)), atol=3e-15)
    np.testing.assert_allclose(weights@populations[:, 0]**2, float(Fraction(85, 216)), atol=3e-15)
    for b in (1, 2):
        np.testing.assert_allclose(weights@(populations[:, 0]*populations[:, b]), rm_gamma(3), atol=2e-15)
    assert abs(rm_gamma(3)-1/(3*4)) > .02  # Unconditional fourth moment is wrong.
    # Focused first-moment matching does not supply the conditional fourth moment.
    assert abs((11/18)*(7/36)-rm_gamma(3)) > .005


def static_problem(*, callback=None, probes=("velocity",), include_nuclei=False):
    model = LinearEPCModel(3, 1)
    rotation, _ = np.linalg.qr(np.array([[1., .2, .4], [-.3, 1.2, .1], [.4, -.2, 1.]]))
    h = rotation@np.diag([-.4, .1, .8])@rotation.T
    params = model.create_params(h, np.zeros((1, 3, 3)), omega=[.3])
    params["reference_offset"] = .7
    callback = callback or FixedPositionVelocity({"velocity": [0., 1.3, 3.]})
    measurement = RMVelocity(probes=probes, probe_callback=callback, include_nuclei=include_nuclei)
    problem = Problem(model, params, CoupledClassical([2.]), MASHRM(event_substeps=1), measurement)
    return problem, rotation


def test_action_algebra_matches_anticommutator_and_is_phase_basis_covariant():
    rng = np.random.default_rng(1782)
    c = rng.normal(size=4)+1j*rng.normal(size=4)
    c /= np.linalg.norm(c)
    u = rng.normal(size=4)+1j*rng.normal(size=4)
    u /= np.linalg.norm(u)
    raw = rng.normal(size=(4, 4))+1j*rng.normal(size=(4, 4))
    velocity = (raw+raw.conj().T)/2
    projector = np.outer(u, u.conj())
    expected = np.vdot(c, (projector@velocity+velocity@projector)@c).real/np.sqrt(2*rm_gamma(4))
    actual = rm_velocity(c, u, velocity@c)
    np.testing.assert_allclose(actual, expected, atol=2e-15)
    np.testing.assert_allclose(jax.jit(rm_velocity)(c, u, velocity@c), expected, atol=2e-15)
    unitary, _ = np.linalg.qr(rng.normal(size=(4, 4))+1j*rng.normal(size=(4, 4)))
    np.testing.assert_allclose(rm_velocity(unitary@c, unitary@u, unitary@velocity@c), actual, atol=4e-15)
    np.testing.assert_allclose(rm_velocity(np.exp(.9j)*c, np.exp(-.7j)*u,
                                          np.exp(.9j)*(velocity@c)), actual, atol=3e-15)


def test_fixed_position_provider_action_and_rotated_full_matrix_agree():
    problem, rotation = static_problem()
    model, params = problem.model, problem.params
    context = ProbeContext(jnp.array([.2]), jnp.array([.3]), jnp.array(.7))
    h, x = np.asarray(params["h0"]), np.diag([0., 1.3, 3.])
    velocity = 1j*(h@x-x@h)
    provider = problem.measurement.probe_callback
    vectors = np.array([[1.+.4j, .2], [.3, -.7j], [.1, .9]])
    np.testing.assert_allclose(provider(model, params, context, "velocity", vectors), velocity@vectors, atol=5e-16)
    np.testing.assert_allclose(provider(model, params, context, "velocity", vectors[:, 0]),
                               velocity@vectors[:, 0], atol=5e-16)
    rotated = FixedPositionVelocity({"velocity": rotation.T@x@rotation})
    diagonal_params = {**params, "h0": rotation.T@h@rotation}
    np.testing.assert_allclose(rotated(model, diagonal_params, context, "velocity", rotation.T@vectors),
                               rotation.T@velocity@vectors, atol=1e-15)


def test_fixed_position_arrays_are_snapshotted_before_cached_jit():
    x = np.array([0., 1.3, 3.])
    provider = FixedPositionVelocity({"velocity": x})
    problem, _ = static_problem(callback=provider)
    context = ProbeContext(jnp.array([0.]), None, jnp.array(0.))
    action = jax.jit(lambda v: provider(problem.model, problem.params, context, "velocity", v))
    before = action(jnp.eye(3))
    x[:] = 400.
    np.testing.assert_array_equal(action(jnp.eye(3)), before)
    np.testing.assert_array_equal(provider.operators[0][1], [0., 1.3, 3.])
    with pytest.raises(ValueError, match="no fixed position"):
        provider(problem.model, problem.params, context, "unknown", jnp.eye(3))


@dataclass(frozen=True)
class VelocityWithMapping:
    """Test-local recording; physical measurements retain their own validation."""

    velocity: RMVelocity
    supports_mashrm = True

    def validate(self, problem):
        self.velocity.validate(problem)

    def evaluate(self, problem, state):
        return {**self.velocity.evaluate(problem, state), "electronic": state.electronic}

    def validate_observations(self, values):
        self.velocity.validate_observations({key: value for key, value in values.items()
                                             if key != "electronic"})
        if not np.isfinite(np.asarray(values["electronic"])).all():
            raise ValueError("recorded mapping vectors must be finite")


@pytest.fixture(scope="module")
def static_correlation():
    problem, _ = static_problem(include_nuclei=True)
    problem = replace(problem, measurement=VelocityWithMapping(problem.measurement))
    h = np.asarray(problem.params["h0"])
    energies, vectors = np.linalg.eigh(h)
    sector_c, weights, active = [], [], []
    for a in range(3):
        c, w = sector_quadrature(a)
        sector_c.append(c@vectors.T)
        weights.extend(w)
        active.extend([a]*len(c))
    carrier = np.concatenate(sector_c)
    count = len(carrier)
    initial = mapping_state(problem.model, problem.params, [0.], [0.], carrier[0], active=0)
    batch = jax.tree.map(lambda value: jnp.broadcast_to(value, (count,)+value.shape), initial)
    batch = batch._replace(electronic=jnp.asarray(carrier),
        trajectory_id=jnp.arange(count, dtype=jnp.uint32),
        key=jax.vmap(lambda i: jax.random.fold_in(initial.key, i))(jnp.arange(count, dtype=jnp.uint32)),
        method_state={**batch.method_state, "active": jnp.asarray(active, dtype=jnp.int32)})
    result = Simulation(problem, Integrator(.125, "exponential_midpoint"),
                        Execution(chunk_size=8)).run(batch, 16)
    return problem, batch, result, np.asarray(weights), np.asarray(active), energies, vectors


@pytest.mark.parametrize("beta", [0., .7, 4.])
def test_static_h_exact_symmetrized_vacf_through_public_mashrm(static_correlation, beta):
    problem, initial, result, sector_weights, active, energies, vectors = static_correlation
    boltzmann = np.exp(-beta*(energies-energies.min()))
    boltzmann /= boltzmann.sum()
    weights = sector_weights*boltzmann[active]
    velocity_samples = np.asarray(result.observables["velocity"])[..., 0]
    correlation = (velocity_samples[0]*velocity_samples)@weights
    h, x = np.asarray(problem.params["h0"]), np.diag([0., 1.3, 3.])
    v = 1j*(h@x-x@h)
    rho = (vectors*boltzmann)@vectors.T
    exact, exact_carriers = [], []
    times = np.asarray(result.times)
    np.testing.assert_array_equal(times, np.broadcast_to(times[:, :1], times.shape))
    for time in times[:, 0]:
        unitary = expm(-1j*h*time)
        exact.append(np.trace(rho@v@unitary.conj().T@v@unitary).real)
        exact_carriers.append(np.asarray(initial.electronic)@unitary.T)
    np.testing.assert_allclose(correlation, exact, atol=2e-14, rtol=0)
    exact_carriers = np.asarray(exact_carriers)
    component_budget = 8e-15
    np.testing.assert_allclose(result.observables["electronic"], exact_carriers,
                               atol=component_budget, rtol=0)
    # This analytical zero is evaluated after 16 propagation steps and a
    # 216-point quadrature. Its former fixed 2e-15 gate was smaller than the
    # effect of allowed state roundoff on Linux, even with compensated sums.
    # Enforce the SAME component budget at EVERY time above, then use
    # |(c+d)^* A (c+d)-c^* A c| <= ||A|| (2||c|| ||d|| + ||d||^2).
    # The estimator matrices and conditional moment are independent NumPy
    # algebra. Tight VACF, final-state and operator covariance checks remain.
    projectors = np.stack([np.outer(vectors[:,a], vectors[:,a]) for a in range(3)])
    operators = (projectors@v+v@projectors)/np.sqrt(2*float(Fraction(47,432)))
    operator_norms = np.linalg.norm(operators, ord=2, axis=(-2,-1))[active]
    delta = np.sqrt(h.shape[0])*component_budget
    state_budget = ((2*np.linalg.norm(exact_carriers, axis=-1)*delta+delta**2)
                    * operator_norms)@abs(weights)
    reference_velocity = np.einsum("tbi,bij,tbj->tb", exact_carriers.conj(),
                                    operators[active], exact_carriers).real
    reference_mean = np.array([fsum(float(a)*float(b) for a, b in zip(row, weights, strict=True))
                               for row in reference_velocity])
    # Include the finite quadrature residual and the standard n-term dot
    # rounding allowance gamma_n=sum_count*eps/(1-sum_count*eps). Reference
    # fsum still rounds its products and final result, covered by 2*eps.
    eps = np.finfo(velocity_samples.dtype).eps
    gamma = len(weights)*eps/(1-len(weights)*eps)
    reduction_budget = (gamma*np.sum(abs(velocity_samples*weights), axis=1)
                        + 2*eps*np.sum(abs(reference_velocity*weights), axis=1))
    mean_budget = state_budget+abs(reference_mean)+reduction_budget
    assert np.all(abs(velocity_samples@weights) <= mean_budget)
    np.testing.assert_array_equal(result.observables["velocity_status"], 0)
    np.testing.assert_array_equal(result.observables["events"], 0)
    np.testing.assert_array_equal(result.final_state.method_state["active"], active)
    np.testing.assert_allclose(result.final_state.electronic,
                               np.asarray(initial.electronic)@expm(-2j*h).T, atol=8e-15, rtol=0)
    problem.measurement.validate_observations(result.observables)


def test_measurement_action_fallback_and_charge_scaling_are_explicit():
    class ProbeModel(LinearEPCModel):
        def probe_apply(self, params, context, probe, vectors):
            return params["velocity"]@vectors

    base, _ = static_problem()
    model = ProbeModel(3, 1)
    object.__setattr__(model, "spec", replace(model.spec, probes=("current_named",)))
    h = np.asarray(base.params["h0"])
    x = np.diag([0., 1.3, 3.])
    v = 1j*(h@x-x@h)
    c, _ = sector_quadrature()
    initial = mapping_state(model, base.params, [0.], [0.], c[0]*np.exp(1j*np.array([.1, .9, -.7])),
                            basis="adiabatic", active=0)
    measurement = RMVelocity(probes=("current_named",))
    problem = Problem(model, {**base.params, "velocity": v}, base.nuclear_treatment, base.method, measurement)
    problem.validate()
    ordinary = measurement.evaluate(problem, initial)
    charge = -2.5
    charged = measurement.evaluate(replace(problem, params={**problem.params, "velocity": charge*v}), initial)
    np.testing.assert_allclose(charged["velocity"], charge*ordinary["velocity"], atol=2e-15)
    # A name containing current does not trigger division by an assumed charge.
    assert abs(float(ordinary["velocity"][0])) > .01


@pytest.mark.parametrize("kind,status", [("identity", 3), ("site_offdiagonal", 3),
                                         ("nonhermitian", 2), ("nonfinite", 1)])
def test_bad_physical_probes_produce_explicit_status_and_host_rejection(kind, status):
    matrix = {"identity": np.eye(3),
              "site_offdiagonal": np.array([[0., 1., 0.], [1., 0., 0.], [0., 0., 0.]]),
              "nonhermitian": np.array([[0., 1., 0.], [0., 0., 0.], [0., 0., 0.]]),
              "nonfinite": np.full((3, 3), np.nan)}[kind]

    def callback(model, params, context, probe, vectors):
        return jnp.asarray(matrix)@vectors

    problem, _ = static_problem(callback=callback)
    state = mapping_state(problem.model, problem.params, [0.], [0.], [1., 0., 0.], basis="adiabatic")
    values = jax.jit(lambda current: problem.measurement.evaluate(problem, current))(state)
    assert int(values["velocity_status"][0]) == status
    assert np.isnan(values["velocity"][0])
    assert int(state.method_state["status"]) == 0  # Measurement never mutates dynamics.
    with pytest.raises(ValueError, match="RM velocity validation failed"):
        problem.measurement.validate_observations(values)


def test_multiple_probes_and_host_nonfinite_output_guards():
    provider = FixedPositionVelocity({"x": [0., 1., 3.], "y": [1., -.2, 2.]})
    problem, _ = static_problem(callback=provider, probes=("y", "x"), include_nuclei=True)
    state = mapping_state(problem.model, problem.params, [.2], [.3],
                          np.sqrt([.6, .3, .1])*np.exp(1j*np.array([.2, .6, -.3])), basis="adiabatic")
    values = problem.measurement.evaluate(problem, state)
    assert values["velocity"].shape == values["velocity_status"].shape == (2,)
    problem.measurement.validate_observations(values)
    for name in ("velocity", "energy", "mapping_norm", "max_event_residual", "q"):
        changed = {**values, name: np.full(np.asarray(values[name]).shape, np.nan)}
        with pytest.raises(ValueError):
            problem.measurement.validate_observations(changed)
    with pytest.raises(ValueError, match="inconsistent trajectory shape"):
        problem.measurement.validate_observations({**values, "energy": np.array([0.])})


def test_direct_measurement_rejects_wrong_active_ownership_without_state_mutation():
    problem, _ = static_problem()
    state = mapping_state(problem.model, problem.params, [0.], [0.], [1., 0., 0.], basis="adiabatic")
    wrong = state._replace(method_state={**state.method_state, "active": jnp.int32(1)})
    values = problem.measurement.evaluate(problem, wrong)
    assert int(values["velocity_status"][0]) == 5
    assert int(wrong.method_state["status"]) == 0
    with pytest.raises(ValueError, match="active surface"):
        problem.measurement.validate_observations(values)


@pytest.mark.parametrize("positions", [{}, {"v": [0.]}, {"v": [0., np.nan]},
                                       {"v": [0., 1j]}, {"v": [[0., 1.], [0., 0.]]}])
def test_invalid_position_operators_are_rejected(positions):
    with pytest.raises(ValueError):
        FixedPositionVelocity(positions)


@pytest.mark.parametrize("n", [0, 1, True, 2.5])
def test_invalid_gamma_state_counts(n):
    with pytest.raises(ValueError):
        rm_gamma(n)


def test_measurement_configuration_and_problem_guards():
    with pytest.raises(ValueError):
        RMVelocity(probes=("v", "v"))
    with pytest.raises(ValueError):
        RMVelocity(diagonal_tolerance=np.inf)
    with pytest.raises(TypeError):
        RMVelocity(probe_callback=3)
    problem, _ = static_problem()
    with pytest.raises(ValueError, match="does not define"):
        replace(problem, measurement=RMVelocity()).validate()
    with pytest.raises(ValueError, match="missing"):
        replace(problem, measurement=RMVelocity(probe_callback=FixedPositionVelocity({"x": [0., 1., 2.]}))).validate()
    with pytest.raises(ValueError, match="match the model"):
        replace(problem, measurement=RMVelocity(probe_callback=FixedPositionVelocity({"velocity": [0., 1.]}))).validate()
    with pytest.raises(ValueError, match="equally sized"):
        rm_velocity(jnp.ones(3), jnp.ones(2), jnp.ones(3))
