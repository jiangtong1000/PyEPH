"""Independent finite-coordinate references for FFT harmonic trajectories."""

from itertools import product

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.linalg import expm

from pyeph import Execution, Integrator, Simulation, make_state, stack_states
from pyeph.adapters.real_space_epc import RealSpaceEPC
from pyeph.paths.normal_modes import NormalModeBath
from pyeph.paths.periodic_harmonic import PeriodicHarmonicBath, sample_periodic_harmonic
from pyeph.workflows.transport import initialize_transport_state, make_transport_problem


def stencil_fixture(mesh=(3, 2, 1)):
    rng = np.random.default_rng(328)
    onsite = rng.normal(size=(6, 6))
    onsite = onsite @ onsite.T + 5*np.eye(6)
    bond = rng.normal(size=(6, 6))*.13
    stencils = {(0, 0, 0): onsite, (1, 0, 0): bond, (-1, 0, 0): bond.T,
                (0, 1, 0): .7*bond, (0, -1, 0): .7*bond.T}
    atoms, cells, values = [], [], []
    for cell, matrix in stencils.items():
        for a, b in product(range(2), repeat=2):
            atoms.append((a, b))
            cells.append(cell)
            values.append(matrix[3*a:3*a+3, 3*b:3*b+3])
    masses = np.array([1.3, 2.7])
    q_shape = (int(np.prod(mesh))*2, 3)
    equilibrium = rng.normal(size=q_shape)*.1
    return mesh, masses, np.array(atoms), np.array(cells), np.array(values), equilibrium


def finite_hessian(mesh, masses, atoms, cells, values, equilibrium):
    ncell, nat = int(np.prod(mesh)), len(masses)
    hessian = np.zeros((ncell*nat*3, ncell*nat*3))
    for cell in product(*(range(n) for n in mesh)):
        i = np.ravel_multi_index(cell, mesh)
        for (a, b), shift, value in zip(atoms, cells, values):
            target = tuple((np.array(cell)+shift) % mesh)
            j = np.ravel_multi_index(target, mesh)
            ia, ib = (i*nat+a)*3, (j*nat+b)*3
            hessian[ia:ia+3, ib:ib+3] += value
    return hessian


@pytest.mark.parametrize("time", [0., .17, 2.8, -1.4])
def test_fft_motion_matches_independent_canonical_matrix_exponential(time):
    mesh, masses, atoms, cells, values, equilibrium = fixture = stencil_fixture()
    bath = PeriodicHarmonicBath(mesh, masses, atoms, cells, values, equilibrium=equilibrium)
    hessian = finite_hessian(*fixture)
    full_masses = np.broadcast_to(np.tile(masses, int(np.prod(mesh)))[:, None], equilibrium.shape)
    rng = np.random.default_rng(529)
    q, p = equilibrium+rng.normal(size=equilibrium.shape)*.03, rng.normal(size=equilibrium.shape)*.02
    state = make_state(q, p, [1, 0])
    n = q.size
    generator = np.block([[np.zeros((n, n)), np.diag(1/full_masses.ravel())],
                          [-hessian, np.zeros((n, n))]])
    expected = expm(time*generator) @ np.r_[(q-equilibrium).ravel(), p.ravel()]
    actual_q, actual_p = jax.jit(bath.point)(state, time)
    np.testing.assert_allclose(actual_q, equilibrium+expected[:n].reshape(q.shape), atol=2e-14)
    np.testing.assert_allclose(actual_p, expected[n:].reshape(q.shape), atol=2e-14)
    dense = NormalModeBath.from_hessian(hessian, full_masses, equilibrium)
    np.testing.assert_allclose(bath.point(state, time), dense.point(state, time), atol=3e-14)
    velocity = jax.jvp(lambda t: bath.point(state, t)[0], (jnp.array(time),), (jnp.array(1.),))[1]
    np.testing.assert_allclose(velocity, actual_p/full_masses, atol=3e-14)


def test_sampler_cartesian_covariance_and_stable_identity_partitions():
    mesh, masses, atoms, cells, values, equilibrium = fixture = stencil_fixture((3, 1, 1))
    bath = PeriodicHarmonicBath(mesh, masses, atoms, cells, values, equilibrium=equilibrium)
    hessian = finite_hessian(*fixture)
    temperature, ids = .7, np.arange(14000)
    q, p = sample_periodic_harmonic(bath, temperature, ids, seed=732)
    expected_q = temperature*np.linalg.inv(hessian)
    expected_p = temperature*np.diag(np.broadcast_to(bath.masses, equilibrium.shape).ravel())
    for actual, expected in ((np.cov(np.asarray(q-equilibrium).reshape(len(ids), -1).T), expected_q),
                             (np.cov(np.asarray(p).reshape(len(ids), -1).T), expected_p)):
        scale = np.sqrt(expected.diagonal()[:, None]*expected.diagonal()[None, :])
        np.testing.assert_allclose(actual/scale, expected/scale, atol=.045, rtol=0)
    selected = np.array([8, 113, 7000])
    subset = sample_periodic_harmonic(bath, temperature, selected, seed=732)
    np.testing.assert_allclose(subset, (q[selected], p[selected]), atol=2e-14)


def test_acoustic_free_motion_and_explicit_constraint_policy():
    atoms, cells = np.zeros((3, 2), int), np.array([[0, 0, 0], [1, 0, 0], [-1, 0, 0]])
    blocks = np.array([2*np.eye(3), -np.eye(3), -np.eye(3)])
    bath = PeriodicHarmonicBath((5, 1, 1), [2.], atoms, cells, blocks, zero_tolerance=1e-14)
    uniform = make_state(np.full((5, 3), .3), np.full((5, 3), .7), [1, 0])
    q, p = jax.jit(bath.point)(uniform, 2.)
    np.testing.assert_allclose(q, 1., atol=1e-14)
    np.testing.assert_allclose(p, .7, atol=1e-14)
    with pytest.raises(ValueError, match="explicit free_positions"):
        sample_periodic_harmonic(bath, .4, np.arange(3))
    q, p = sample_periodic_harmonic(bath, .4, np.arange(3), free_positions=np.full((5, 3), .3))
    np.testing.assert_allclose(q.mean(axis=1), .3, atol=1e-14)
    with pytest.raises(ValueError, match="zero-mode subspace"):
        sample_periodic_harmonic(bath, .4, np.arange(3), free_positions=np.arange(15).reshape(5, 3))
    constrained = PeriodicHarmonicBath((5, 1, 1), [2.], atoms, cells, blocks, frozen_below=1e-6)
    with pytest.raises(ValueError, match="zero initial modal momentum"):
        constrained.validate_initial_state(uniform)
    q, p = sample_periodic_harmonic(constrained, .4, np.arange(3))
    constrained.validate_initial_state(stack_states([make_state(q[i], p[i], [1, 0], trajectory_id=i)
                                                    for i in range(3)]), batch=True)
    np.testing.assert_allclose(q.mean(axis=1), 0., atol=1e-14)
    np.testing.assert_allclose(p.mean(axis=1), 0., atol=1e-14)


def test_wigner_zero_temperature_variances_and_retained_unstable_spectrum():
    atoms, cells, values = np.array([[0, 0]]), np.array([[0, 0, 0]]), np.array([np.diag([1., 4., 9.])])
    bath = PeriodicHarmonicBath((1, 1, 1), [1.], atoms, cells, values)
    q, p = sample_periodic_harmonic(bath, 0., np.arange(14000), distribution="wigner")
    np.testing.assert_allclose(np.var(q[:, 0, :], axis=0), 1/(2*np.array([1., 2., 3.])), rtol=.04)
    np.testing.assert_allclose(np.var(p[:, 0, :], axis=0), np.array([1., 2., 3.])/2, rtol=.04)
    values[0, 0, 0] = -.7
    with pytest.raises(ValueError, match="unstable periodic"):
        PeriodicHarmonicBath((1, 1, 1), [1.], atoms, cells, values)
    bath = PeriodicHarmonicBath((1, 1, 1), [1.], atoms, cells, values, frozen_below=0.)
    assert bath.squared_frequencies.ravel()[0] == pytest.approx(-.7)
    state = make_state([[.3, .2, .1]], [[0., .4, .2]], [1, 0])
    q, p = bath.point(state, .6)
    assert q[0, 0] == pytest.approx(.3)
    assert p[0, 0] == pytest.approx(0.)


def test_unpaired_images_rejected_even_if_they_alias_on_the_finite_grid():
    # On a one-cell grid +R and -R alias, so checking only D(k) would miss
    # this malformed directed stencil.
    with pytest.raises(ValueError, match="reverse-cell blocks"):
        PeriodicHarmonicBath((1, 1, 1), [1.], [[0, 0]], [[1, 0, 0]], [np.eye(3)])


def test_periodic_bath_native_thermal_run_restart_and_constraint_preflight(tmp_path):
    data = RealSpaceEPC(
        cell=5*np.eye(3), atom_positions=[[0., 0., 0.]], masses=[1.7],
        wannier_centers=[[0., 0., 0.]], hopping_orbitals=np.zeros((3, 2), int),
        hopping_cells=[[0, 0, 0], [1, 0, 0], [-1, 0, 0]], hopping_values=[.1, -.2, -.2],
        epc_channels=[0], epc_atoms=[0], epc_cells=[[0, 0, 0]], epc_values=[[0., .05, 0.]],
        ifc_atoms=[[0, 0]], ifc_cells=[[0, 0, 0]], ifc_values=[np.diag([0., .6, .9])],
    )
    compiled = data.compile_supercell((3, 1, 1), real_tolerance=0.)
    bath = PeriodicHarmonicBath((3, 1, 1), data.masses, data.ifc_atoms, data.ifc_cells,
                                data.ifc_values, frozen_below=1e-6)
    problem = make_transport_problem(compiled.model, compiled.params, bath)
    ids = np.array([14, 19])
    q, p = sample_periodic_harmonic(bath, .4, ids, seed=712)
    initial = stack_states([initialize_transport_state(problem, q[i], p[i], 2.,
                                                       trajectory_id=int(identity), seed=712)
                            for i, identity in enumerate(ids)])
    simulation = Simulation(problem, Integrator(.02), Execution(chunk_size=4))
    full = simulation.run(initial, 18)
    first = simulation.run(initial, 7)
    path = tmp_path/"periodic_checkpoint.h5"
    simulation.save_checkpoint(path, first.final_state)
    resumed = simulation.run(simulation.load_checkpoint(path), 11)
    for actual, expected in zip(jax.tree_util.tree_leaves(resumed.final_state),
                                 jax.tree_util.tree_leaves(full.final_state)):
        np.testing.assert_allclose(actual, expected, atol=2e-13, rtol=0.)
    np.testing.assert_array_equal(resumed.final_state.trajectory_id, ids)
    np.testing.assert_array_equal(resumed.final_state.key, full.final_state.key)
    np.testing.assert_array_equal(resumed.final_state.step, 18)
    np.testing.assert_allclose(resumed.observables["current_correlation"],
                               full.observables["current_correlation"][7:], atol=2e-13)
    expected_q, expected_p = jax.vmap(bath.point, in_axes=(0, None))(initial, .36)
    np.testing.assert_allclose(full.final_state.q, expected_q, atol=2e-13)
    np.testing.assert_allclose(full.final_state.p, expected_p, atol=2e-13)
    bad = initial._replace(p=initial.p.at[1, 0, 0].set(.1))
    with pytest.raises(ValueError, match="zero initial modal momentum"):
        simulation.run(bad, 0, collect=False)
    invalid = tmp_path/"invalid.h5"
    with pytest.raises(ValueError, match="zero initial modal momentum"):
        simulation.save_checkpoint(invalid, bad)
    assert not invalid.exists()
