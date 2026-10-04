"""Physical and numerical references for the restricted two-state MASH method."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.linalg import expm
from scipy.optimize import brentq

from pyeph.core.contracts import ModelSpec
from pyeph.core.problem import CoupledClassical, PrescribedPath, Problem
from pyeph.core.state import make_state, stack_states
from pyeph.core.system import SystemSpec
from pyeph.dynamics.mash2 import (
    MASH2,
    MASHError,
    MASHPopulation,
    mapping_state,
    momentum_impulse,
    sample_adiabatic_population,
    sample_population_spin,
)
from pyeph.execution.runner import Execution
from pyeph.integrators.electronic import Integrator
from pyeph.integrators.events import bisect_crossing
from pyeph.models.base import AutoDiffModel
from pyeph.observables.population import ElectronicPopulation
from pyeph.paths.harmonic import ConstantPath
from pyeph.simulation import Simulation


class RotatingModel(AutoDiffModel):
    """Constant ±delta surfaces; position-dependent eigenvectors R(q)."""

    spec = ModelSpec(SystemSpec(2, (1,), coordinate_kind="canonical"), name="rotating")

    def dense(self, params, q):
        delta = 0.02 if params is None else params
        x = 2*q[0]
        return delta*jnp.array([[-jnp.cos(x), -jnp.sin(x)],
                                [-jnp.sin(x), jnp.cos(x)]])

    def apply(self, params, q, vectors):
        return self.dense(params, q) @ vectors

    def reference_energy(self, params, q):
        return jnp.zeros((), dtype=q.dtype)


class UncoupledOscillator(AutoDiffModel):
    spec = ModelSpec(SystemSpec(2, (1,), coordinate_kind="canonical"), name="uncoupled")

    def apply(self, params, q, vectors):
        return jnp.diag(jnp.array([-0.1, 0.1])) @ vectors

    def reference_energy(self, params, q):
        return 0.5*jnp.sum(q*q)


def rotating_problem(*, method=None, measurement=None):
    return Problem(RotatingModel(), None, CoupledClassical(1.0), method or MASH2(),
                   measurement or MASHPopulation(include_nuclei=True))


def rotating_state(momentum=0.3, *, trajectory_id=0):
    return mapping_state(RotatingModel(), None, [0.0], [momentum], [-np.sqrt(0.75), 0, -0.5],
                         trajectory_id=trajectory_id)


def run_rotating(dt, *, momentum=0.3, duration=4, method=None, jit=True):
    return Simulation(rotating_problem(method=method), Integrator(dt, "exponential_midpoint"),
                      Execution(jit=jit, chunk_size=64)).run(
                          rotating_state(momentum), round(duration/dt))


def exact_single_event(momentum, duration=4):
    """Independent analytic adiabatic propagator for constant speed/surface gap.

    R(q).T dR/dq=[[0,-1],[1,0]], so the exact moving-basis Hamiltonian
    is diag(-delta,delta) - i*p*R.T*dR/dq. Locate its first event with SciPy.
    """
    def generator(p):
        return np.array([[-0.02, 1j*p], [-1j*p, 0.02]])

    c0 = np.array([np.sqrt(0.75), -0.5], dtype=complex)

    def z(t):
        c = expm(-1j*generator(momentum)*t) @ c0
        return abs(c[1])**2 - abs(c[0])**2

    event_time = brentq(z, 0, 2, xtol=1e-14)
    accepted = momentum**2 >= 0.08
    outgoing = np.sqrt(momentum**2 - 0.08) if accepted else -momentum
    q = momentum*event_time + outgoing*(duration-event_time)
    adiabatic = (expm(-1j*generator(outgoing)*(duration-event_time))
                 @ expm(-1j*generator(momentum)*event_time) @ c0)
    rotation = np.array([[np.cos(q), -np.sin(q)], [np.sin(q), np.cos(q)]])
    # Both test trajectories have only one event in [0,4].
    assert (2*int(accepted)-1)*(abs(adiabatic[1])**2 - abs(adiabatic[0])**2) > 0
    return q, outgoing, rotation @ adiabatic


@pytest.mark.parametrize("active", [0, 1])
def test_population_weighted_sampler_moments(active):
    spin = np.asarray(sample_population_spin(jax.random.PRNGKey(71), active, shape=(200_000,)))
    np.testing.assert_allclose(np.sum(spin*spin, axis=-1), 1, atol=5e-16)
    assert np.all(spin[:, 2]*(2*active-1) > 0)
    assert abs(np.mean(abs(spin[:, 2])) - 2/3) < 0.003
    np.testing.assert_allclose(np.mean(spin*spin, axis=0), [0.25, 0.25, 0.5], atol=0.003)
    np.testing.assert_allclose(np.mean(spin[:, :2], axis=0), 0, atol=0.003)


def test_sampler_stable_ids_and_active_estimator_not_squared_amplitude():
    model = RotatingModel()
    a = sample_adiabatic_population(model, None, [0], [0.3], seed=92, trajectory_id=7)
    b = sample_adiabatic_population(model, None, [0], [0.3], seed=92, trajectory_id=8)
    repeat = sample_adiabatic_population(model, None, [0], [0.3], seed=92, trajectory_id=7)
    np.testing.assert_array_equal(a.electronic, repeat.electronic)
    assert not np.allclose(a.electronic, b.electronic)
    measurement = MASHPopulation().evaluate(rotating_problem(), a)
    np.testing.assert_array_equal(measurement["population"], [1, 0])
    assert not np.allclose(np.abs(a.electronic)**2, measurement["population"])
    assert float(measurement["energy"]) == pytest.approx(0.3**2/2 - 0.02)


@pytest.mark.parametrize("delta", [0.1, -0.3, 10.0])
def test_impulse_energy_and_mass_weighted_perpendicular_component(delta):
    p, mass, nac = np.array([1.2, -0.7, 2.3]), np.array([1., 3., 7.]), np.array([0.3, -0.5, 0.4])
    impulse = momentum_impulse(p, mass, nac, delta)
    assert bool(impulse.valid)
    direction = (nac/np.sqrt(mass))/np.linalg.norm(nac/np.sqrt(mass))
    before, after = p/np.sqrt(mass), np.asarray(impulse.momentum)/np.sqrt(mass)
    def perpendicular(vector):
        return vector - np.dot(vector, direction)*direction

    np.testing.assert_allclose(perpendicular(after), perpendicular(before), atol=1e-14)
    energy_change = np.sum(after*after-before*before)/2
    if delta == 10:
        assert not bool(impulse.accepted)
        assert energy_change == pytest.approx(0, abs=1e-14)
        assert np.dot(after, direction) == pytest.approx(-np.dot(before, direction))
    else:
        assert bool(impulse.accepted)
        assert energy_change == pytest.approx(-delta, abs=1e-14)
    assert abs(float(impulse.energy_error)) < 1e-14


def test_impulse_grazing_and_invalid_data_are_explicit():
    assert not bool(momentum_impulse([0., 1.], [1., 1.], [1., 0.], -0.2).valid)
    assert not bool(momentum_impulse([1.], [1.], [0.], -0.2).valid)
    assert not bool(momentum_impulse([1.], [-1.], [1.], -0.2).valid)
    with pytest.raises(ValueError, match="real"):
        momentum_impulse([1.], [1.], [1j], 0.2)


def test_bisection_bounded_status_and_equatorial_start_recrossing():
    root = jax.jit(lambda t: bisect_crossing(lambda x: 0.3-x, t))(1.)
    assert bool(root.converged)
    assert float(root.time) == pytest.approx(0.3, abs=1e-10)
    assert int(root.iterations) <= 40
    failed = bisect_crossing(lambda x: 0.3-x, 1., iterations=1, tolerance=1e-12)
    assert not bool(failed.converged)
    unbracketed = bisect_crossing(lambda x: 1+x, 1.)
    assert not bool(unbracketed.converged)
    # A spin starts at the equator, leaves into the correct hemisphere, then
    # returns. Returning t=0 here would cause repeated spurious immediate hops.
    recross = bisect_crossing(lambda t: t*(0.7-t), 1.)
    assert bool(recross.converged)
    assert float(recross.time) == pytest.approx(0.7, abs=1e-9)


@pytest.mark.parametrize("momentum", [0.3, 0.2])
def test_accepted_and_frustrated_trajectory_match_independent_analytic_solution(momentum):
    result = run_rotating(0.1, momentum=momentum, method=MASH2(event_substeps=2))
    state, expected = result.final_state, exact_single_event(momentum)
    np.testing.assert_allclose(state.q, [expected[0]], atol=3e-7)
    np.testing.assert_allclose(state.p, [expected[1]], atol=1e-14)
    np.testing.assert_allclose(state.electronic, expected[2], atol=3e-6)
    assert int(state.method_state["events"]) == 1
    assert int(state.method_state["accepted"]) == int(momentum == 0.3)
    assert int(state.method_state["frustrated"]) == int(momentum == 0.2)
    assert int(state.method_state["event_attempts"]) == 1
    assert float(state.method_state["max_event_residual"]) <= 1e-10
    np.testing.assert_allclose(result.observables["energy"], momentum**2/2-0.02, atol=1e-14)
    np.testing.assert_allclose(result.observables["mapping_norm"], 1, atol=1e-13)


def test_event_trajectory_timestep_and_subinterval_convergence():
    expected = exact_single_event(0.3)
    errors = []
    for dt in (0.4, 0.2, 0.1):
        state = run_rotating(dt, method=MASH2(event_substeps=1)).final_state
        actual = np.concatenate((state.q, state.electronic.real, state.electronic.imag))
        exact = np.concatenate(([expected[0]], expected[2].real, expected[2].imag))
        errors.append(np.linalg.norm(actual-exact))
    assert errors[0]/errors[1] > 3.5
    assert errors[1]/errors[2] > 3.5
    subdivided = run_rotating(0.4, method=MASH2(event_substeps=4)).final_state
    fine = run_rotating(0.1, method=MASH2(event_substeps=1)).final_state
    np.testing.assert_allclose(subdivided.q, fine.q, atol=2e-12)
    np.testing.assert_allclose(subdivided.electronic, fine.electronic, atol=2e-12)


def test_no_coupling_born_oppenheimer_limit_and_smooth_map_reversal():
    model = UncoupledOscillator()
    state = sample_adiabatic_population(model, None, [0.7], [0.3], seed=1)
    problem = Problem(model, None, CoupledClassical(1), MASH2(event_substeps=1), MASHPopulation())
    sim = Simulation(problem, Integrator(0.02, "exponential_midpoint"))
    result = sim.run(state, 100)
    end = result.final_state
    np.testing.assert_allclose(end.q, [0.7*np.cos(2)+0.3*np.sin(2)], atol=5e-5)
    np.testing.assert_allclose(end.p, [-0.7*np.sin(2)+0.3*np.cos(2)], atol=5e-5)
    assert int(end.method_state["events"]) == 0
    np.testing.assert_array_equal(result.observables["population"], np.tile([1, 0], (101, 1)))
    reverse = sim.run(end._replace(p=-end.p, electronic=end.electronic.conj()), 100).final_state
    np.testing.assert_allclose(reverse.q, state.q, atol=3e-14)
    np.testing.assert_allclose(reverse.p, -state.p, atol=3e-14)
    np.testing.assert_allclose(reverse.electronic, state.electronic.conj(), atol=4e-14)


def test_jit_eager_batch_and_restart_agree():
    method = MASH2(event_substeps=1)
    problem = rotating_problem(method=method)
    integrator = Integrator(0.2, "exponential_midpoint")
    states = [rotating_state(0.3), rotating_state(0.2, trajectory_id=1)]
    batch = stack_states(states)
    result = Simulation(problem, integrator).run(batch, 20)
    for i, s in enumerate(states):
        single = Simulation(problem, integrator).run(s, 20).final_state
        np.testing.assert_allclose(result.final_state.q[i], single.q, atol=1e-12)
        np.testing.assert_allclose(result.final_state.electronic[i], single.electronic, atol=1e-12)
    step = method.build_step(problem, integrator)
    eager, compiled = step(states[0]), jax.jit(step)(states[0])
    np.testing.assert_allclose(eager.electronic, compiled.electronic, atol=1e-14)
    first = Simulation(problem, integrator).run(batch, 7)
    second = Simulation(problem, integrator).run(first.final_state, 13)
    jax.tree.map(lambda a, b: np.testing.assert_allclose(a, b, atol=1e-13),
                 result.final_state, second.final_state)


def test_failures_keep_last_finite_state_and_raise_after_compiled_chunk():
    method = MASH2(event_substeps=1, bisection_iterations=1, event_tolerance=1e-14)
    with pytest.raises(MASHError, match="localization") as caught:
        run_rotating(1., method=method)
    state = caught.value.failed_state
    assert int(state.method_state["status"]) == 3
    assert int(state.method_state["event_attempts"]) == 1
    assert int(state.method_state["localization_iterations"]) == 1
    assert np.isfinite(np.asarray(state.electronic)).all()
    assert float(state.time) < 4
    capacity = MASH2(event_substeps=32, max_events_per_step=1)
    with pytest.raises(MASHError, match="capacity") as caught:
        run_rotating(8., momentum=1., duration=8, method=capacity)
    assert int(caught.value.failed_state.method_state["status"]) == 5
    assert int(caught.value.failed_state.method_state["events"]) == 1


def test_refining_subintervals_resolves_recrossings_hidden_from_endpoints():
    # Endpoint sign tests cannot rule out a pair of crossings. This example
    # records that limitation and demonstrates the explicit refinement knob.
    coarse = run_rotating(6., momentum=1., duration=6,
                          method=MASH2(event_substeps=1)).final_state
    resolved = run_rotating(6., momentum=1., duration=6,
                            method=MASH2(event_substeps=32)).final_state
    fine = run_rotating(6/32, momentum=1., duration=6,
                        method=MASH2(event_substeps=1)).final_state
    assert int(resolved.method_state["events"]) > int(coarse.method_state["events"])
    assert int(resolved.method_state["events"]) >= 2
    np.testing.assert_allclose(resolved.q, fine.q, atol=2e-11)
    np.testing.assert_allclose(resolved.electronic, fine.electronic, atol=2e-11)


def test_capability_initialization_and_integrator_rejections():
    problem = rotating_problem()
    with pytest.raises(ValueError, match="explicit MASHPopulation"):
        replace(problem, measurement=None).validate()
    with pytest.raises(ValueError, match="mapping methods"):
        ElectronicPopulation().validate(problem)
    with pytest.raises(ValueError, match="coupled"):
        replace(problem, nuclear_treatment=PrescribedPath(ConstantPath([0]))).validate()
    with pytest.raises(ValueError, match="constructor"):
        MASH2().validate_state(make_state([0], [1], [1, 0]))
    with pytest.raises(ValueError, match="unit length"):
        mapping_state(RotatingModel(), None, [0], [1], [0, 0, -2])
    with pytest.raises(ValueError, match="exponential_midpoint"):
        Simulation(problem, Integrator(0.1)).run(rotating_state(), 1)
    with pytest.raises(ValueError, match="degenerate"):
        mapping_state(RotatingModel(), 0., [0], [1], [0, 0, -1])
    step = MASH2().build_step(replace(problem, params=0.), Integrator(0.1, "exponential_midpoint"))
    failed = jax.jit(step)(rotating_state())
    assert int(failed.method_state["status"]) == 1
    with pytest.raises(MASHError, match="degenerate"):
        MASH2().validate_result(failed)
    inconsistent = rotating_state()._replace(method_state={**rotating_state().method_state,
                                                          "active": jnp.int32(1)})
    with pytest.raises(ValueError, match="hemisphere"):
        Simulation(problem, Integrator(0.1, "exponential_midpoint")).run(inconsistent, 1)
