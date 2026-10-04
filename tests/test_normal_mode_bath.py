"""Physical-coordinate harmonic motion checked against independent ODE mechanics."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.linalg import expm

from pyeph import CPA, Integrator, Problem, Simulation, make_state, stack_states
from pyeph.models.epc import LinearEPCModel
from pyeph.paths.normal_modes import NormalModeBath, sample_normal_modes


def coupled_fixture():
    mass = np.array([[1.3], [2.7]])
    equilibrium = np.array([[.1, .2, -.1], [.4, -.2, .3]])
    rng = np.random.default_rng(409)
    rotation, _ = np.linalg.qr(rng.normal(size=(6, 6)))
    frequencies = np.array([.23, .51, .8, 1.1, 1.6, 2.])
    root = np.sqrt(np.broadcast_to(mass, equilibrium.shape).reshape(-1))
    hessian = root[:, None]*(rotation @ np.diag(frequencies**2) @ rotation.T)*root[None, :]
    return mass, equilibrium, hessian


@pytest.mark.parametrize("time", [0., .1, 1.7, -2.3])
def test_coupled_cartesian_motion_matches_independent_matrix_exponential(time):
    mass, equilibrium, hessian = coupled_fixture()
    bath = NormalModeBath.from_hessian(hessian, mass, equilibrium)
    q = equilibrium + np.array([[.2, -.1, .3], [-.2, .1, .15]])
    p = np.array([[.1, .3, -.2], [.4, -.2, .1]])
    state = make_state(q, p, [1, 0])
    actual_q, actual_p = jax.jit(bath.point)(state, time)
    masses = np.broadcast_to(mass, q.shape).reshape(-1)
    generator = np.block([[np.zeros((6, 6)), np.diag(1/masses)],
                          [-hessian, np.zeros((6, 6))]])
    expected = expm(time*generator) @ np.r_[(q-equilibrium).reshape(-1), p.reshape(-1)]
    np.testing.assert_allclose(actual_q, equilibrium+expected[:6].reshape(q.shape), atol=2e-14)
    np.testing.assert_allclose(actual_p, expected[6:].reshape(p.shape), atol=2e-14)
    before_energy = .5*np.sum(p.reshape(-1)**2/masses)+.5*(q-equilibrium).reshape(-1)@hessian@(q-equilibrium).reshape(-1)
    dx = np.asarray(actual_q-equilibrium).reshape(-1)
    after_energy = .5*np.sum(np.asarray(actual_p).reshape(-1)**2/masses)+.5*dx@hessian@dx
    assert after_energy == pytest.approx(before_energy, rel=2e-14)


def test_time_derivative_is_physical_velocity_including_free_mode():
    bath = NormalModeBath.from_hessian(np.diag([0., 2.]), [3., 5.], [0., 0.])
    state = make_state([.4, -.1], [.3, .7], [1, 0])
    for time in (0., .7):
        q, derivative = jax.jvp(lambda t: bath.point(state, t)[0], (jnp.array(time),), (jnp.array(1.),))
        expected_p = bath.point(state, time)[1]
        np.testing.assert_allclose(derivative, expected_p/bath.masses, atol=1e-14)
        assert q[0] == pytest.approx(.4+.1*time)


def test_instability_is_explicit_and_constraints_keep_incoming_position():
    with pytest.raises(ValueError, match="unstable harmonic modes"):
        NormalModeBath.from_hessian(np.diag([-.3, 0., 2.]), 1., np.zeros(3))
    bath = NormalModeBath.from_hessian(np.diag([-.3, 0., 2.]), 1., np.zeros(3), frozen_modes=(0,))
    state = make_state([.17, .8, .3], [0., .11, .9], [1, 0])
    bath.validate_initial_state(state)
    q, p = jax.jit(bath.point)(state, 1.2)
    assert q[0] == pytest.approx(.17)
    assert p[0] == 0
    assert q[1] == pytest.approx(.8+.11*1.2)
    assert p[1] == pytest.approx(.11)
    assert bath.squared_frequencies[0] == pytest.approx(-.3)


def test_public_runner_rejects_constraint_violation_before_run_or_checkpoint(tmp_path):
    bath = NormalModeBath.from_hessian(np.diag([-.3, 2.]), 1., np.zeros(2), frozen_modes=(0,))
    model = LinearEPCModel(2, 2)
    problem = Problem(model, model.default_params(), bath, CPA())
    runner = Simulation(problem, Integrator(.1))
    bad = make_state([0., 0.], [.1, 0.], [1, 0])
    with pytest.raises(ValueError, match="zero initial modal momentum"):
        runner.run(bad, 0, collect=False)
    with pytest.raises(ValueError, match="zero initial modal momentum"):
        runner.save_checkpoint(tmp_path/"invalid.h5", bad)
    assert not (tmp_path/"invalid.h5").exists()
    good = make_state([0., .3], [0., .1], [1, 0])
    result = runner.run(good, 10)
    np.testing.assert_allclose(result.final_state.q, bath.point(good, 1.)[0], atol=2e-14)
    np.testing.assert_allclose(result.final_state.p, bath.point(good, 1.)[1], atol=2e-14)


def test_thermal_covariance_and_partition_independent_sampling():
    mass, equilibrium, hessian = coupled_fixture()
    bath = NormalModeBath.from_hessian(hessian, mass, equilibrium)
    temperature = .4
    q, p = sample_normal_modes(bath, temperature, np.arange(12000), seed=915)
    displacement = np.asarray(q-equilibrium).reshape(-1, 6)
    momentum = np.asarray(p).reshape(-1, 6)
    expected_q = temperature*np.linalg.inv(hessian)
    expected_p = temperature*np.diag(np.broadcast_to(mass, equilibrium.shape).reshape(-1))
    for actual, expected in ((np.cov(displacement.T), expected_q), (np.cov(momentum.T), expected_p)):
        scale = np.sqrt(np.diag(expected)[:, None]*np.diag(expected)[None, :])
        np.testing.assert_allclose(actual/scale, expected/scale, atol=.055, rtol=0)
    subset_q, subset_p = sample_normal_modes(bath, temperature, np.array([7, 12, 24]), seed=915)
    np.testing.assert_allclose(subset_q, q[np.array([7, 12, 24])], atol=1e-14, rtol=0)
    np.testing.assert_allclose(subset_p, p[np.array([7, 12, 24])], atol=1e-14, rtol=0)


def test_wigner_variance_and_explicit_free_position_policy():
    bath = NormalModeBath.from_hessian(np.diag([0., 4.]), 1., np.zeros(2))
    with pytest.raises(ValueError, match="explicit free_positions"):
        sample_normal_modes(bath, .3, np.arange(10))
    q, p = sample_normal_modes(bath, 0., np.arange(12000), distribution="wigner", free_positions=[.7])
    np.testing.assert_array_equal(q[:, 0], .7)
    np.testing.assert_array_equal(p[:, 0], 0.)
    assert np.var(q[:, 1]) == pytest.approx(.25, rel=.055)
    assert np.var(p[:, 1]) == pytest.approx(1., rel=.055)


def test_configuration_snapshots_and_explicit_numerical_zero_tolerance():
    values, vectors, mass, equilibrium = np.array([-1e-14, 2.]), np.eye(2), np.ones(2), np.zeros(2)
    bath = NormalModeBath(values, vectors, mass, equilibrium, zero_tolerance=1e-13)
    values[:] = -5.
    vectors[:] = 0.
    mass[:] = -1.
    equilibrium[:] = 4.
    np.testing.assert_array_equal(bath.frequencies, [0., np.sqrt(2.)])
    np.testing.assert_array_equal(bath.equilibrium, [0., 0.])
    np.testing.assert_array_equal(bath.masses, [1., 1.])
    with pytest.raises(ValueError, match="orthonormal"):
        NormalModeBath([1., 2.], np.ones((2, 2)), 1., np.zeros(2))
    with pytest.raises(ValueError, match="symmetric"):
        NormalModeBath.from_hessian([[1., 2.], [0., 1.]], 1., np.zeros(2))


def test_single_precision_source_and_purely_free_or_frozen_ensembles():
    rng = np.random.default_rng(4)
    matrix = rng.normal(size=(3, 3)).astype(np.float32)
    hessian = matrix @ matrix.T
    bath = NormalModeBath.from_hessian(hessian, np.ones(3, np.float32), np.zeros(3))
    reconstructed = bath.eigenvectors @ jnp.diag(bath.squared_frequencies) @ bath.eigenvectors.T
    np.testing.assert_allclose(reconstructed, hessian, atol=1e-13)
    with pytest.raises(ValueError, match="selected execution precision"):
        NormalModeBath(np.asarray(bath.squared_frequencies, np.float32),
                        np.asarray(bath.eigenvectors, np.float32), 1., np.zeros(3))
    ids = np.arange(4000)
    frozen = NormalModeBath.from_hessian(-np.eye(3), 1., np.ones(3), frozen_modes=(0, 1, 2))
    q, p = sample_normal_modes(frozen, .4, ids)
    np.testing.assert_array_equal(q, 1.)
    np.testing.assert_array_equal(p, 0.)
    free = NormalModeBath.from_hessian(np.zeros((3, 3)), [2., 3., 4.], np.ones(3))
    q, p = sample_normal_modes(free, .4, ids, free_positions=np.zeros(3))
    np.testing.assert_array_equal(q, 1.)
    np.testing.assert_allclose(np.var(p, axis=0), .4*np.array([2., 3., 4.]), rtol=.07)


@pytest.mark.parametrize("enable_x64", [True, False])
def test_accepted_basis_roundoff_cannot_reject_its_own_constrained_samples(enable_x64):
    before = jax.config.x64_enabled
    jax.config.update("jax_enable_x64", enable_x64)
    try:
        dtype = np.float64 if enable_x64 else np.float32
        epsilon = np.finfo(dtype).eps
        direction = np.r_[0., np.full(16, .25)].astype(dtype)
        vectors = np.eye(17, dtype=dtype)
        vectors[0] += 16*epsilon*direction
        bath = NormalModeBath(np.r_[0., np.ones(16)], vectors, 1., np.zeros(17),
                              frozen_modes=(0,))
        q, p = bath.from_modes(jnp.zeros(17), jnp.asarray(direction))
        bath.validate_initial_state(make_state(q, p, [1., 0.]))
        q, p = sample_normal_modes(bath, .4, np.arange(8), seed=615)
        batch = stack_states([make_state(q[i], p[i], [1., 0.], trajectory_id=i)
                              for i in range(8)])
        bath.validate_initial_state(batch, batch=True)
        # Every individual Gram entry would pass an elementwise128eps test,
        # but the combined leakage into the frozen mode is too large.
        vectors = np.eye(17, dtype=dtype)
        vectors[0, 1:] = 64*epsilon
        with pytest.raises(ValueError, match="orthonormal"):
            NormalModeBath(np.r_[0., np.ones(16)], vectors, 1., np.zeros(17), frozen_modes=(0,))
    finally:
        jax.config.update("jax_enable_x64", before)
