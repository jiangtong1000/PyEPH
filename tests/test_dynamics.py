"""Independent references for physical propagation, not implementation snapshots."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.integrate import solve_ivp
from scipy.linalg import expm

from pyeph.core.contracts import ModelSpec
from pyeph.core.problem import CoupledClassical, PrescribedPath, Problem
from pyeph.core.state import make_state, stack_states
from pyeph.core.system import SystemSpec
from pyeph.dynamics.cpa import CPA
from pyeph.dynamics.ehrenfest import Ehrenfest, total_energy
from pyeph.execution.runner import Execution
from pyeph.integrators.electronic import Integrator
from pyeph.models.base import AutoDiffModel
from pyeph.observables.population import FunctionalMeasurement
from pyeph.paths.harmonic import ConstantPath, HarmonicBath, HarmonicPath
from pyeph.simulation import Simulation


class ReferenceModel(AutoDiffModel):
    spec = ModelSpec(SystemSpec(2, (1,), coordinate_kind="canonical"), name="reference")

    def apply(self, params, q, vectors):
        matrix = jnp.array([[params[0] * q[0], params[1]],
                            [params[1], -params[0] * q[0]]])
        return matrix @ vectors

    def reference_energy(self, params, q):
        return 0.5 * params[2] * jnp.sum(q**2)


def energy_measurement(mass=1.0):
    return FunctionalMeasurement(lambda p, s: {
        "energy": total_energy(p.model, p.params, s, mass),
        "norm": jnp.vdot(s.electronic, s.electronic).real,
    })


@pytest.mark.parametrize("algorithm", ["rk4", "exponential_midpoint"])
def test_static_cpa_matches_exact_complex_exponential(algorithm):
    model, params = ReferenceModel(), jnp.array([0.3, 0.17, 1.0])
    q = jnp.array([0.8])
    initial = make_state(q, [0], [1 / np.sqrt(2), 1j / np.sqrt(2)])
    problem = Problem(model, params, PrescribedPath(ConstantPath(q)), CPA())
    run = Simulation(problem, Integrator(0.01, electronic=algorithm), Execution(chunk_size=31))
    result = run.run(initial, 200)
    h = np.array([[0.24, 0.17], [0.17, -0.24]])
    expected = expm(-2j * h) @ np.asarray(initial.electronic)
    np.testing.assert_allclose(result.final_state.electronic, expected, atol=3e-12)
    np.testing.assert_allclose(result.observables["norm"], 1, atol=3e-12)


def test_time_dependent_cpa_against_scipy_ode():
    params = jnp.array([0.4, 0.23, 0.6])
    path = HarmonicPath([0.7], [0.3], [1.3])
    problem = Problem(ReferenceModel(), params, PrescribedPath(path), CPA())
    initial = make_state([0.7], [0.3], [1, 0])
    result = Simulation(problem, Integrator(0.02)).run(initial, 150)

    def rhs(t, c):
        q = 0.7 * np.cos(1.3 * t) + 0.3 / 1.3 * np.sin(1.3 * t)
        h = np.array([[0.4 * q, 0.23], [0.23, -0.4 * q]])
        return -1j * h @ c

    exact = solve_ivp(rhs, (0, 3), [1 + 0j, 0j], rtol=1e-12, atol=1e-13).y[:, -1]
    np.testing.assert_allclose(result.final_state.electronic, exact, atol=2e-9)


def test_ehrenfest_converges_to_independent_coupled_ode():
    params = jnp.array([0.4, 0.23, 0.6])
    initial = make_state([0.7], [0.3], [1, 0])
    problem = Problem(ReferenceModel(), params, CoupledClassical(1.0), Ehrenfest(),
                      energy_measurement())

    def rhs(t, y):
        q, p = y[:2]
        c = y[2:4] + 1j * y[4:6]
        h = np.array([[0.4 * q, 0.23], [0.23, -0.4 * q]])
        dc = -1j * h @ c
        dp = -0.6 * q - 0.4 * (abs(c[0])**2 - abs(c[1])**2)
        return np.concatenate(([p, dp], dc.real, dc.imag))

    exact = solve_ivp(rhs, (0, 2), [0.7, 0.3, 1, 0, 0, 0], rtol=1e-12, atol=1e-13).y[:, -1]
    errors, energy_errors = [], []
    for dt in (0.04, 0.02, 0.01):
        result = Simulation(problem, Integrator(dt, "exponential_midpoint")).run(
            initial, round(2 / dt))
        s = result.final_state
        actual = np.concatenate((s.q, s.p, s.electronic.real, s.electronic.imag))
        errors.append(np.linalg.norm(actual - exact))
        energy = result.observables["energy"]
        energy_errors.append(np.max(abs(energy - energy[0])))
        np.testing.assert_allclose(result.observables["norm"], 1, atol=2e-13)
    assert errors[0] / errors[1] > 3.8
    assert errors[1] / errors[2] > 3.8
    assert energy_errors[0] / energy_errors[1] > 3.7
    assert errors[-1] < 3e-5


def test_independent_batched_harmonic_cpa_and_chunking():
    problem = Problem(ReferenceModel(), jnp.array([0.4, 0.2, 0.6]),
                      HarmonicBath([0.9]), CPA())
    states = [make_state([q], [p], [1, 0], trajectory_id=i, seed=12)
              for i, (q, p) in enumerate([(0.7, 0.2), (-0.3, 0.5), (0.1, -0.2)])]
    batch = stack_states(states)
    batched = Simulation(problem, Integrator(0.02), Execution(chunk_size=13)).run(batch, 37)
    single = [Simulation(problem, Integrator(0.02), Execution(chunk_size=19)).run(s, 37)
              for s in states]
    for i, result in enumerate(single):
        np.testing.assert_allclose(batched.final_state.electronic[i], result.final_state.electronic,
                                   atol=2e-14)
        np.testing.assert_allclose(batched.final_state.q[i], result.final_state.q, atol=2e-14)
    assert not np.allclose(batched.final_state.electronic[0], batched.final_state.electronic[1])


def test_streamed_output_restart_and_absolute_sampling():
    problem = Problem(ReferenceModel(), jnp.array([0.4, 0.2, 0.6]), HarmonicBath([0.9]), CPA())
    initial = make_state([0.7], [0.2], [1, 0])
    sim = Simulation(problem, Integrator(0.02), Execution(chunk_size=7, save_every=4))
    whole = sim.run(initial, 31)
    first = sim.run(initial, 13)
    second = sim.run(first.final_state, 18)
    np.testing.assert_allclose(whole.final_state.electronic, second.final_state.electronic, atol=1e-14)
    np.testing.assert_allclose(second.times, np.array([13, 16, 20, 24, 28, 31]) * 0.02)
    chunks = []
    streamed = sim.run(initial, 31, observer=lambda t, x: chunks.append((t, x)), collect=False)
    assert streamed.observables == {} and streamed.times.size == 0
    np.testing.assert_allclose(np.concatenate([x[0] for x in chunks]), whole.times)


def test_compatibility_and_state_rejections():
    model = ReferenceModel()
    params = jnp.array([0.4, 0.2, 0.6])
    with pytest.raises(ValueError, match="coupled"):
        Problem(model, params, PrescribedPath(ConstantPath([0.])), Ehrenfest()).validate()
    with pytest.raises(ValueError, match="prescribed"):
        Problem(model, params, CoupledClassical(1), CPA()).validate()
    with pytest.raises(ValueError, match="positive"):
        Problem(model, params, CoupledClassical(-1), Ehrenfest()).validate()
    problem = Problem(model, params, CoupledClassical(1), Ehrenfest())
    with pytest.raises(ValueError, match="normalized"):
        Simulation(problem, Integrator(0.01)).run(make_state([0], [0], [1, 1]), 1)
    with pytest.raises(ValueError, match="pure electronic"):
        Simulation(problem, Integrator(0.01)).run(make_state([0], [0], np.eye(2)), 1)


def test_custom_method_needs_no_runner_changes():
    class Drift:
        def validate(self, problem):
            pass

        def build_step(self, problem, integrator):
            def step(s):
                return s._replace(q=s.q + integrator.dt * s.p, time=s.time + integrator.dt,
                                  step=s.step + 1)
            return step

    problem = Problem(ReferenceModel(), None, CoupledClassical(1), Drift())
    result = Simulation(problem, Integrator(0.1)).run(make_state([1], [2], [1, 0]), 10)
    np.testing.assert_allclose(result.final_state.q, [3.0])


def test_free_and_oscillator_path_limits():
    path = HarmonicPath([1, 2], [3, 4], [0, 2], [2, 3])
    np.testing.assert_allclose(path.position(0), [1, 2])
    np.testing.assert_allclose(path.velocity(0), [1.5, 4 / 3])
    assert float(path.position(2)[0]) == 4
    derivative = jax.jacfwd(path.position)(0.4)
    np.testing.assert_allclose(derivative, path.velocity(0.4), atol=1e-14)
