import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pyeph.core.contracts import pure_state_weight
from pyeph.core.coordinates import MassWeightedModes
from pyeph.core.problem import CoupledClassical, Problem
from pyeph.core.state import make_state
from pyeph.core.units import UnitSystem
from pyeph.dynamics.ehrenfest import Ehrenfest
from pyeph.execution.runner import Execution, SimulationError
from pyeph.integrators.electronic import Integrator
from pyeph.models.analytic import SpinBosonModel
from pyeph.observables.statistics import EnsembleMoments
from pyeph.simulation import Simulation


def test_mass_weighted_coordinate_energy_and_gradient_consistency():
    rng = np.random.default_rng(24)
    modes, _ = np.linalg.qr(rng.normal(size=(6, 4)))
    transform = MassWeightedModes(np.zeros((2, 3)), [2, 3], modes)
    q, p = jnp.array(rng.normal(size=4)), jnp.array(rng.normal(size=4))
    r = transform.to_cartesian(q)
    np.testing.assert_allclose(transform.to_modes(r), q, atol=1e-14)
    velocity = transform.velocity(p)
    np.testing.assert_allclose(0.5 * np.sum(np.array([2, 3])[:, None] * velocity**2),
                               0.5 * p @ p, atol=1e-14)
    gradient_r = jnp.sin(r) + 0.3 * r
    reference = jax.grad(lambda x: jnp.sum(-jnp.cos(transform.to_cartesian(x))
                         + 0.15 * transform.to_cartesian(x)**2))(q)
    np.testing.assert_allclose(transform.pullback_gradient(gradient_r), reference, atol=1e-14)
    cartesian_momentum = jnp.array([2, 3])[:, None] * velocity
    np.testing.assert_allclose(transform.momentum(cartesian_momentum), p, atol=1e-14)


def test_same_geometry_different_electronic_weights_have_different_force():
    model = SpinBosonModel(1)
    params = model.default_params()
    q = jnp.array([0.3])
    a = model.contract_gradient(params, q, pure_state_weight(jnp.array([1., 0.])))
    b = model.contract_gradient(params, q, pure_state_weight(jnp.array([0., 1.])))
    np.testing.assert_allclose(a, [1.])
    np.testing.assert_allclose(b, [-1.])


def test_reduced_unit_energy_and_time_scales():
    units = UnitSystem.from_ev_angstrom(energy_ev=0.1, length_angstrom=7.2)
    assert units.time_fs == pytest.approx(6.582119569, rel=1e-8)
    assert units.temperature_from_kelvin(300) == pytest.approx(0.2585199978, rel=1e-8)
    # E_red = p_red²/(2m_red), with p_red=p_atomic * length_bohr (hbar=1).
    mass, momentum = 1836., 0.03
    p_reduced = momentum * units.length_bohr
    np.testing.assert_allclose(p_reduced**2 / (2 * units.mass_from_electron_masses(mass)),
                               momentum**2 / (2 * mass) / units.energy_hartree)


def test_partition_invariant_complex_ensemble_statistics():
    rng = np.random.default_rng(19)
    data = rng.normal(size=(71, 5, 2)) + 1j * rng.normal(size=(71, 5, 2))
    accumulated = EnsembleMoments().update(data[:12])
    accumulated.merge(EnsembleMoments().update(data[12:43]))
    accumulated.update(data[43:])
    np.testing.assert_allclose(accumulated.mean, data.mean(axis=0), atol=4e-16)
    np.testing.assert_allclose(accumulated.variance, data.var(axis=0, ddof=1), rtol=1e-14)
    np.testing.assert_allclose(accumulated.standard_error,
                               np.sqrt(data.var(axis=0, ddof=1) / len(data)), rtol=1e-14)
    assert accumulated.count == len(data)
    assert np.isnan(EnsembleMoments().update(data[:1]).standard_error).all()


def test_invalid_runtime_keeps_last_accepted_state():
    class BrokenMethod:
        def validate(self, problem):
            pass

        def build_step(self, problem, integrator):
            def step(state):
                return state._replace(q=state.q * jnp.nan)
            return step

    model = SpinBosonModel(1)
    initial = make_state([0.2], [0.3], [1, 0])
    problem = Problem(model, model.default_params(), CoupledClassical(1.), BrokenMethod())
    with pytest.raises(SimulationError) as info:
        Simulation(problem, Integrator(0.01), Execution(chunk_size=2)).run(initial, 5)
    np.testing.assert_array_equal(info.value.last_valid_state.q, initial.q)
    assert np.isnan(info.value.failed_state.q).all()


def test_runtime_parameters_do_not_become_stale_compilation_constants():
    model = SpinBosonModel(1)
    params = model.default_params()
    initial = make_state([0.2], [0.3], [1, 0])
    problem = Problem(model, params, CoupledClassical(1.), Ehrenfest())
    sim = Simulation(problem, Integrator(0.01))
    first = sim.run(initial, 4)
    params["delta"] = 0.7
    second = sim.run(initial, 4)
    independent = Simulation(problem, Integrator(0.01)).run(initial, 4)
    assert not np.allclose(first.final_state.electronic, second.final_state.electronic)
    np.testing.assert_allclose(second.final_state.electronic, independent.final_state.electronic)


@pytest.mark.parametrize("counter,value", [("step", 1.5), ("trajectory_id", 1.9),
                                          ("step", 2**63), ("trajectory_id", 2**32)])
def test_state_counter_identity_is_never_silently_truncated(counter, value):
    with pytest.raises(ValueError):
        make_state([0], [0], [1, 0], **{counter: value})


def test_complex_coordinate_map_rejected():
    with pytest.raises(ValueError, match="real"):
        MassWeightedModes(np.zeros((1, 3)), [1], np.eye(3, dtype=complex))
