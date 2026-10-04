"""Material-parameter and complete atomistic checks, distinct from model accuracy."""

import importlib.util
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pyeph import Execution, Integrator, Simulation
from pyeph.core.contracts import ProbeContext, pure_state_weight
from pyeph.core.state import stack_states
from pyeph.core.units import BOHR_ANGSTROM, HARTREE_EV

_PATH = Path(__file__).resolve().parents[1]/"examples/perovskite.py"
_SPEC = importlib.util.spec_from_file_location("perovskite_example", _PATH)
example = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = example
_SPEC.loader.exec_module(example)


def test_published_equilibrium_r_point_gap_and_spin_orbit_multiplets():
    model, params, equilibrium, _ = example.build_cspbi3()
    a = example.SOURCE_PARAMETERS["lattice"]/BOHR_ANGSTROM
    h = model.apply_peierls(params, equilibrium, jnp.full(3, np.pi/a), jnp.eye(model.nstates))
    energies = np.linalg.eigvalsh(np.asarray(h))*HARTREE_EV
    # Table I parameters are printed to 0.001 eV. Paper reports a 1.017 eV DFT gap.
    np.testing.assert_allclose(energies[26]-energies[25], 1.017, atol=.002, rtol=0.)
    np.testing.assert_allclose(energies[25:28:2], [-.0010675594, 1.0171708793], atol=2e-10)
    np.testing.assert_allclose(np.diff(energies)[::2], 0., atol=2e-14)
    assert model.spec.complex_valued
    assert model.spec.system.q_shape == (5, 3)


@pytest.mark.parametrize("mesh", [(1, 1, 1), (2, 1, 1), (2, 2, 1)])
def test_disordered_supercell_matches_independent_matrix_force_and_current(mesh):
    problem, initial = example.fixture(mesh)
    model = problem.model.models[0]
    params, neutral = problem.params
    q, c = initial.q, initial.electronic
    h, gradient, current = example.numpy_quantities(model, params, q, np.asarray(c))
    np.testing.assert_allclose(model.dense(params, q), h, atol=2e-16)
    np.testing.assert_allclose(model.apply(params, q, c), h@c, atol=2e-16)
    actual = model.contract_gradient(params, q, pure_state_weight(c))
    np.testing.assert_allclose(actual, gradient, atol=2e-16)
    total = -(problem.model.reference_gradient(problem.params, q)+actual)
    eps = 2e-5
    finite = np.zeros(q.shape)
    for index in np.ndindex(q.shape):
        d = np.zeros(q.shape)
        d[index] = eps

        def energy(x):
            neutral_energy = .5*np.sum(np.asarray(neutral["spring"])*(x-neutral["equilibrium"])**2)
            return neutral_energy+np.vdot(c, example.numpy_quantities(model, params, x)[0]@c).real

        finite[index] = -(energy(q+d)-energy(q-d))/(2*eps)
    np.testing.assert_allclose(total, finite, atol=3e-11, rtol=2e-6)
    for i, axis in enumerate("xyz"):
        np.testing.assert_allclose(model.probe_apply(params, ProbeContext(q), f"current_{axis}", c),
                                   current[i]@c, atol=3e-16)
    assert np.linalg.norm(np.asarray(actual)[1::5]) > 1e-6  # iodine carrier force is active
    np.testing.assert_array_equal(np.asarray(actual)[4::5], 0.)  # declared Cs spectator
    assert np.linalg.norm(np.asarray(total)[4::5]) > 1e-6  # neutral force still acts on Cs


def test_octahedral_tilt_changes_hamiltonian_and_has_complete_directional_force():
    model, params, q, _ = example.build_cspbi3((2, 1, 1))
    direction = np.zeros(q.shape)
    # Move only two ligand atoms transversely; every Pb centre stays fixed.
    direction[1, 1], direction[6, 1] = 1., -1.
    rng = np.random.default_rng(942)
    c = rng.normal(size=model.nstates)+1j*rng.normal(size=model.nstates)
    c /= np.linalg.norm(c)
    h0 = example.numpy_quantities(model, params, q)[0]
    perturbed = q+.03*direction
    h1 = example.numpy_quantities(model, params, perturbed)[0]
    assert np.linalg.norm(h1-h0) > 1e-4
    actual = model.contract_gradient(params, perturbed, pure_state_weight(jnp.asarray(c)))
    eps = 1e-5
    finite = np.vdot(c, (example.numpy_quantities(model, params, perturbed+eps*direction)[0]
                         -example.numpy_quantities(model, params, perturbed-eps*direction)[0])@c).real/(2*eps)
    np.testing.assert_allclose(np.sum(actual*direction), finite, atol=5e-12, rtol=2e-7)
    assert abs(finite) > 1e-7


@pytest.mark.parametrize("method", ["cpa", "ehrenfest"])
def test_public_dynamics_refines_against_independent_ode_and_batch_partition(method):
    problem, initial = example.fixture(method=method)
    values = []
    for dt, steps, stride in ((3., 4, 1), (1.5, 8, 2)):
        runner = Simulation(problem, Integrator(dt, "rk4", electronic_substeps=2),
                            Execution(chunk_size=2, save_every=stride))
        values.append(runner.run(initial, steps))
    reference, _ = example.scipy_reference(problem, initial, values[0].times)
    for name, target in reference.items():
        errors = [np.max(abs(result.observables[name]-target)) for result in values]
        assert errors[1] < .45*errors[0]+2e-10
    assert np.max(abs(values[1].observables["electronic"]-reference["electronic"])) < 1e-6
    # Distinct IDs/initial states exercise batch independence, not duplicate lanes.
    _, second = example.fixture(method=method, seed=445)
    second = second._replace(trajectory_id=jnp.asarray(20, dtype=second.trajectory_id.dtype))
    batch = stack_states((initial, second))
    runner = Simulation(problem, Integrator(1.5, "rk4", electronic_substeps=2),
                        Execution(chunk_size=2, save_every=1))
    combined = runner.run(batch, 4)
    scalar = [runner.run(state, 4) for state in (initial, second)]
    for i, result in enumerate(scalar):
        for actual, expected in zip(jax.tree.leaves(combined.final_state), jax.tree.leaves(result.final_state), strict=True):
            np.testing.assert_allclose(actual[i], expected, atol=3e-14, rtol=3e-14)


@pytest.mark.parametrize("mesh", [(0, 1, 1), (True, 1, 1), (1, 1), (1.5, 1, 1)])
def test_bad_mesh_rejected(mesh):
    with pytest.raises((TypeError, ValueError)):
        example.build_cspbi3(mesh)
