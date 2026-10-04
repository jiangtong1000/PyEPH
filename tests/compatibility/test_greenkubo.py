"""Legacy constructors on the shared native engine, with pinned old-engine data.

The fixture generator runs original source in a separate environment and saves
actual samples. These tests import only the new package and need no numba,
MPI installation or old PyEPH checkout.
"""

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import h5py
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.linalg

from pyeph.core.contracts import ProbeContext
from pyeph.core.problem import Problem
from pyeph.core.units import UnitSystem
from pyeph.dynamics.cpa import CPA
from pyeph.execution.runner import Execution
from pyeph.greenkubo.analysis import merge_outputs
from pyeph.greenkubo.hamiltonian import ElectronPhononHamiltonian
from pyeph.greenkubo.lattice import BravaisLattice2D
from pyeph.greenkubo.phonon import ClassicPhononBath, QuantumPhononBath
from pyeph.greenkubo.propagator import DensityMatrixUnitaryPropagator
from pyeph.greenkubo.simulation import GreenKuboSimulation, _StructuredLFModel, _assemble
from pyeph.greenkubo.utils import (
    get_hdf5_matrix_list_shape, load_hdf5_matrix, load_hdf5_matrix_list_item,
    write_hdf5_csr_matrix, write_hdf5_csr_matrix_list,
)
from pyeph.observables.transport.greenkubo import current_correlation
from pyeph.integrators.electronic import Integrator
from pyeph.io.provenance import problem_manifest
from pyeph.models.lattice_epc import LatticeEPCModel
from pyeph.paths.harmonic import HarmonicBath

DATA = Path(__file__).parent/"data/greenkubo"
CASES = ["1D_CPA"] + [f"2D_{method}_{model}Peierls_zz{zigzag}"
    for method in ("CPA", "PT_CPA") for model in ("bond", "optical") for zigzag in (True, False)]
CASES += ["band_narrow_only"] + [f"nonlocal_{distribution}_{gauge}"
    for distribution in ("Boltzmann", "Wigner") for gauge in (False, True)]

spec = importlib.util.spec_from_file_location("greenkubo_fixture_configuration", DATA/"generate_reference.py")
fixture_configuration = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixture_configuration)


@pytest.fixture(autouse=True)
def serial(monkeypatch):
    monkeypatch.setenv("USE_MPI", "false")


def test_native_lattice_dimensions_snapshot_mutable_scalar_inputs():
    mutable = dict(nstates_config=np.array(2), nmodes=np.array(1), ncells=np.array(2),
                   nhalf=np.array(0), nonlocal_phonons=np.array(False))
    model = LatticeEPCModel(**mutable)
    params, q = {"frequencies": jnp.array([.3])}, jnp.array([.2, -.4])
    problem = Problem(model, params, HarmonicBath([.3, .3]), CPA())
    before = problem_manifest(problem, Integrator(.01))["payload"]["model"]
    compiled = jax.jit(lambda position: model.fields(params, position))
    expected = compiled(q)
    for value in mutable.values():
        value[...] = 7
    assert type(model.nstates_config) is int and type(model.nonlocal_phonons) is bool
    assert problem_manifest(problem, Integrator(.01))["payload"]["model"] == before
    np.testing.assert_array_equal(compiled(q), expected)
    np.testing.assert_allclose(model.fields(params, q), expected, atol=1e-15)
    with pytest.raises(ValueError, match="integer"):
        LatticeEPCModel(2.5, 1, 2)


@pytest.fixture(scope="module", params=CASES)
def native_case(request):
    # Explicitly serial even when a developer happens to have mpi4py installed.
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("USE_MPI", "false")
        with h5py.File(DATA/"reference.h5") as archive:
            reference = {key: value[...] for key, value in archive[request.param].items()}
        lattice, ham, classical, quantum, propagator = fixture_configuration.make_case(request.param)
        simulation = GreenKuboSimulation(
            lattice, ham, classical, quantum, propagator,
            initial_samples=(reference["q0"], reference["p0"]),
        )
        result = simulation.run()
        yield request.param, simulation, result, reference


def test_original_engine_trajectory_with_identical_saved_samples(native_case):
    name, simulation, result, reference = native_case
    np.testing.assert_array_equal(simulation.classic_ph.initial_samples()[0], reference["q0"])
    np.testing.assert_array_equal(simulation.classic_ph.initial_samples()[1], reference["p0"])
    model = getattr(simulation.problem.model, "base_model", simulation.problem.model)
    h0 = model.apply(simulation.problem.params, simulation.initial_state.q[0], jnp.eye(simulation.lattice.nsites))
    np.testing.assert_allclose(h0, reference["h0_first"], rtol=1e-10, atol=1e-11)
    np.testing.assert_allclose(simulation.initial_state.method_state["transport"]["rho0"][0],
                               reference["rho0_first"], rtol=1e-9, atol=1e-11)
    np.testing.assert_allclose(result.times[:, 0], reference["times"], rtol=0, atol=1e-14)
    np.testing.assert_allclose(result.observables["current_correlation"], reference["correlation"],
                               rtol=2e-8, atol=1e-9)
    np.testing.assert_allclose(result.final_state.electronic[0], reference["unitary_final_first"],
                               rtol=1e-9, atol=2e-11)
    if name.startswith("nonlocal_"):
        q0, p0 = np.asarray(simulation.initial_state.q), np.asarray(simulation.initial_state.p)
        w = np.asarray(simulation.problem.params["canonical_frequencies"])
        fields = []
        for t in reference["times"]:
            q = q0*np.cos(w*t)+p0*np.sin(w*t)/w
            field = jax.vmap(lambda x: model.fields(simulation.problem.params, x))(jnp.asarray(q))
            fields.append(np.asarray(field).transpose(1, 0, 2))
        np.testing.assert_allclose(fields, reference["real_space_fields"], rtol=1e-11, atol=1e-11)


def test_unchanged_historical_expected_current_files(native_case):
    name, _simulation, result, reference = native_case
    file = DATA/f"expected_{name}.h5"
    if not file.exists():
        return  # These are additional nonlocal and band-narrow-only source oracles.
    manifest = json.loads((DATA/"provenance.json").read_text())
    with h5py.File(file) as archive:
        for axis in ("x", "y"):
            key = f"current_{axis}"
            if key not in archive:
                continue
            recorded = manifest["comparison_to_historical"][name][key]
            column = ("x", "y").index(axis)
            assert np.allclose(reference["correlation"][..., column].mean(axis=1), archive[key][...]) == recorded["allclose"]
            if not recorded["allclose"]:
                pytest.xfail("archival golden differs from the pinned original engine; see provenance.json")
            np.testing.assert_allclose(result.observables["current_correlation"][..., column].mean(axis=1),
                                       archive[key][...], rtol=1e-5, atol=1e-8)


def test_source_and_fixture_provenance():
    manifest = json.loads((DATA/"provenance.json").read_text())
    assert manifest["source_revision"] == fixture_configuration.REVISION
    assert hashlib.sha256((DATA/"reference.h5").read_bytes()).hexdigest() == manifest["reference_sha256"]
    assert len(manifest["historical_expected"]) == 9
    for name, digest in manifest["historical_expected"].items():
        assert hashlib.sha256((DATA/name).read_bytes()).hexdigest() == digest


def complex_small_case(*, quantum=False, band_only=False, policy="legacy_full", execution=None):
    lattice = BravaisLattice2D(3, 1, 2, [[0., 0.], [.35, .2]], unit_system=UnitSystem(.01, 3.))
    hcell = np.array([[.25, .12+.04j], [.12-.04j, -.17]])
    hopping = np.array([[-.3, .05j], [.03-.02j, -.2]])
    onsite_epc = np.zeros((2, 2, 1), complex)
    onsite_epc[:, :, 0] = [[.08, .02j], [-.02j, -.06]]
    inter_epc = np.array([[[.01j], [.02]], [[-.03j], [.04]]])
    g = {(0, 0): {(0, 0): onsite_epc}, (1, 0): {(0, 0): inter_epc}}
    ham = ElectronPhononHamiltonian({(0, 0): hcell, (1, 0): hopping}, g, lattice)
    bath = ClassicPhononBath([[.7]], .8, g)
    q = QuantumPhononBath([1.8], [.5], .8, band_narrow_only=band_only) if quantum else None
    propagator = DensityMatrixUnitaryPropagator(lattice.nsites, 2, .01, .06, .8)
    initial = (np.array([[[.2, -.3, .1], [.4, .1, -.2]]]),
               np.array([[[-.1, .2, .4], [.1, -.2, .3]]]))
    return GreenKuboSimulation(lattice, ham, bath, q, propagator, initial_samples=initial,
                               thermal_policy=policy, execution=execution)


def explicit_dense_hamiltonian(simulation, fields):
    lattice, ham = simulation.lattice, simulation.ham
    matrix = np.zeros((lattice.nsites, lattice.nsites), complex)
    for cell_y in range(lattice.ny):
        for cell_x in range(lattice.nx):
            cell = cell_y*lattice.nx+cell_x
            for (dx, dy) in set(ham.tmat) | set(ham.gmat):
                target = ((cell_y+dy) % lattice.ny)*lattice.nx+(cell_x+dx) % lattice.nx
                block = ham.tmat.get((dx, dy), np.zeros((lattice.ncenter, lattice.ncenter))).copy()
                for (px, py), epc in ham.gmat.get((dx, dy), {}).items():
                    phonon = ((cell_y+py) % lattice.ny)*lattice.nx+(cell_x+px) % lattice.nx
                    for mode in range(ham.nmodes):
                        block = block + epc[..., mode]*fields[mode, phonon]
                for i, j in np.ndindex(lattice.ncenter, lattice.ncenter):
                    row, col = cell*lattice.ncenter+i, target*lattice.ncenter+j
                    matrix[row, col] += block[i, j]
                    if (dx, dy) != (0, 0):
                        matrix[col, row] += block[i, j].conjugate()
    return matrix


def test_complex_sparse_action_probe_and_diagonal_lf_preservation():
    simulation = complex_small_case()
    model, params = simulation.problem.model, simulation.problem.params
    q = simulation.initial_state.q[0]
    field = np.asarray(model.fields(params, q))
    expected = explicit_dense_hamiltonian(simulation, field)
    matrix = np.asarray(jax.jit(model.apply)(params, q, jnp.eye(model.nstates)))
    np.testing.assert_allclose(matrix, expected, atol=1e-13)
    np.testing.assert_allclose(matrix, matrix.conj().T, atol=1e-13)
    displacement = np.array([[simulation.lattice.minimum_displacement(i, j)
                               for j in range(model.nstates)] for i in range(model.nstates)])
    for axis, probe in enumerate(("current_x", "current_y")):
        current = np.asarray(model.probe_apply(params, ProbeContext(q), probe, jnp.eye(model.nstates)))
        np.testing.assert_allclose(current, 1j*displacement[..., axis]*expected, atol=1e-13)
        np.testing.assert_allclose(current, current.conj().T, atol=1e-13)
    dressed = _StructuredLFModel(model, .37)
    narrowed = np.asarray(jax.jit(dressed.apply)(params, q, jnp.eye(model.nstates)))
    reference = .37*expected
    np.fill_diagonal(reference, expected.diagonal())
    np.testing.assert_allclose(narrowed, reference, atol=1e-13)
    np.testing.assert_allclose(model.diagonal(params, q), expected.diagonal())
    # The operator remains differentiable through the sparse field gather.
    target = jnp.arange(model.nstates, dtype=float)+1j
    derivative = jax.grad(lambda x: jnp.real(jnp.vdot(target, model.apply(params, x, target))))(q)
    step = 1e-5
    numeric = []
    for k in range(len(q)):
        direction = np.eye(len(q))[k]*step
        def value(x):
            return np.vdot(target, model.apply(params, x, target)).real
        numeric.append((value(q+direction)-value(q-direction))/(2*step))
    np.testing.assert_allclose(derivative, numeric, rtol=1e-7, atol=1e-8)


def test_epc_only_edges_and_zero_crossings_keep_current_indices():
    lattice = BravaisLattice2D(3, 1, 1, [[0, 0]])
    g = {(1, 0): {(0, 0): np.ones((1, 1, 1))}}
    ham = ElectronPhononHamiltonian({(0, 0): np.array([[.4]])}, g, lattice)
    for values in ([0., 1., -1.], [1., 0., -1.], [0., 0., 0.]):
        h = ham.build_ep_variation_matrix(np.array(values).reshape(1, 1, 3))[0]
        current = ham.build_jx_jy([h])[0][0].toarray()
        displacement = np.array([[lattice.minimum_displacement(i, j)[0] for j in range(3)] for i in range(3)])
        np.testing.assert_allclose(current, displacement*h.toarray())
        np.testing.assert_allclose(current, -current.conj().T)


@pytest.mark.parametrize("policy", ["legacy_full", "offdiagonal"])
@pytest.mark.parametrize("band_only", [False, True])
def test_complex_lf_policy_and_band_narrow_current_formula(policy, band_only):
    simulation = complex_small_case(quantum=True, band_only=band_only, policy=policy)
    model = simulation.problem.model
    q0 = simulation.initial_state.q[0]
    h = np.asarray(model.bare_hamiltonian(simulation.problem.params, q0))
    factor = simulation.quantum_ph.polaron_prefactor
    expected_h = h*factor
    if policy == "offdiagonal":
        np.fill_diagonal(expected_h, h.diagonal())
    rho = scipy.linalg.expm(-simulation.propagator.beta*expected_h)
    rho /= rho.trace()
    np.testing.assert_allclose(simulation.initial_state.method_state["transport"]["rho0"][0], rho, atol=1e-12)
    result = simulation.run()
    if band_only:
        final = result.final_state
        for traj in range(2):
            currents = model.base_model.probe_apply(simulation.problem.params, ProbeContext(final.q[traj]),
                                                    "current_x", jnp.eye(model.spec.system.nstates))
            initial = model.base_model.probe_apply(simulation.problem.params, ProbeContext(simulation.initial_state.q[traj]),
                                                    "current_x", jnp.eye(model.spec.system.nstates))
            expected = factor**2*current_correlation(final.electronic[traj],
                simulation.initial_state.method_state["transport"]["rho0"][traj], currents, initial)
            np.testing.assert_allclose(result.observables["current_correlation"][-1, traj, 0], expected, atol=1e-12)


def test_serial_output_saved_samples_chunking_and_checkpoint(tmp_path):
    simulation = complex_small_case(execution=Execution(chunk_size=2))
    result = simulation.run(tmp_path/"stream", dump_interval=2, collect=True, keep_rank_files=True)
    mean = result.observables["current_correlation"].mean(axis=1)[:, 0]
    with h5py.File(tmp_path/"stream/collected_current_autocorr.h5") as handle:
        np.testing.assert_allclose(handle["current_x"], mean)
        np.testing.assert_array_equal(handle["current_x_std"], np.zeros(len(mean)))
        assert handle.attrs["total_time"] == simulation.time_step*len(mean)
    samples = simulation.load_initial_samples(tmp_path/"stream/initial_samples_0.h5")
    for actual, expected in zip(samples, simulation.classic_ph.initial_samples()):
        np.testing.assert_array_equal(actual, expected)
    with h5py.File(tmp_path/"stream/initial_samples_0.h5") as handle:
        assert handle.attrs["physical_unit_scale_known"]
        assert handle.attrs["energy_hartree"] == .01
    restarted = complex_small_case()
    restarted.run(steps=2)
    restarted.save_checkpoint(tmp_path/"checkpoint.h5")
    restored = complex_small_case()
    restored.load_checkpoint(tmp_path/"checkpoint.h5")
    remaining = restored.run(tmp_path/"resumed", collect=True)
    np.testing.assert_allclose(remaining.final_state.electronic, result.final_state.electronic, atol=1e-12)
    np.testing.assert_allclose(remaining.observables["current_correlation"],
                               result.observables["current_correlation"][2:], atol=1e-12)
    with h5py.File(tmp_path/"resumed/collected_current_autocorr.h5") as handle:
        assert handle.attrs["initial_time"] == .02
        np.testing.assert_allclose(handle["time"], remaining.times[:, 0])
    recovered = restored.load_initial_samples(tmp_path/"resumed/initial_samples_0.h5")
    for actual, expected in zip(recovered, simulation.classic_ph.initial_samples()):
        np.testing.assert_array_equal(actual, expected)


def test_sparse_hdf5_compatibility_helpers(tmp_path):
    simulation = complex_small_case()
    matrix = simulation.ham.h_static
    with h5py.File(tmp_path/"sparse.h5", "w") as handle:
        write_hdf5_csr_matrix(handle, "matrix", matrix)
        write_hdf5_csr_matrix_list(handle, "batch", [matrix, 2*matrix])
    with h5py.File(tmp_path/"sparse.h5") as handle:
        np.testing.assert_allclose(load_hdf5_matrix(handle["matrix"], dense=True), matrix.toarray())
        assert get_hdf5_matrix_list_shape(handle["batch"]) == (2, *matrix.shape)
        np.testing.assert_allclose(load_hdf5_matrix_list_item(handle["batch"], 1, dense=True), 2*matrix.toarray())


def test_rank_mean_merge_validates_steps_and_truncates_safe_prefix(tmp_path):
    for rank, count in enumerate((3, 2)):
        with h5py.File(tmp_path/f"currents_{rank}.h5", "w") as handle:
            handle.attrs["current_step"], handle.attrs["time_step"] = count, .1
            handle["current_x"] = np.arange(count)+1j*rank
    with pytest.raises(ValueError, match="same step"):
        merge_outputs(tmp_path, ["x"], 2, safe_mode=False)
    output = merge_outputs(tmp_path, ["x"], 2, safe_mode=True)
    with h5py.File(output) as handle:
        np.testing.assert_allclose(handle["current_x"], [0+.5j, 1+.5j])
        np.testing.assert_allclose(handle["current_x_std"], [.5, .5])


@pytest.mark.parametrize("enabled", [False, True])
def test_legacy_precision_is_explicit_in_fresh_process(enabled):
    code = """
import jax
before = jax.config.x64_enabled
from pyeph.greenkubo.typical_model_helper import build_1d_Holstein_Peierls_model
from pyeph.greenkubo.propagator import DensityMatrixUnitaryPropagator
from pyeph.greenkubo.simulation import GreenKuboSimulation
assert jax.config.x64_enabled == before
ham, bath, quantum, lattice, temperature = build_1d_Holstein_Peierls_model(
    1., .1, [], [], .2, .3, 3, 1.)
propagator = DensityMatrixUnitaryPropagator(3, 1, .01, .02, temperature)
if not before:
    try:
        GreenKuboSimulation(lattice, ham, bath, quantum, propagator)
    except ValueError as error:
        assert 'JAX_ENABLE_X64=true' in str(error)
        assert 'configure_precision' in str(error)
    else:
        raise AssertionError('legacy float64 was silently downcast')
else:
    simulation = GreenKuboSimulation(lattice, ham, bath, quantum, propagator)
    result = simulation.run()
    assert str(result.final_state.q.dtype) == 'float64'
    assert str(result.final_state.electronic.dtype) == 'complex128'
assert jax.config.x64_enabled == before
"""
    environment = os.environ.copy()
    environment.update(JAX_ENABLE_X64=str(enabled).lower(), USE_MPI="false")
    output = subprocess.run([sys.executable, "-c", code], env=environment, text=True,
                            capture_output=True, timeout=60)
    assert output.returncode == 0, output.stdout+output.stderr


def test_empty_classical_bath_and_zero_narrowing_remain_finite():
    lattice = BravaisLattice2D(3, 1, 1, [[0., 0.]])
    empty_epc = {(0, 0): {(0, 0): np.zeros((1, 1, 0))}}
    ham = ElectronPhononHamiltonian({(1, 0): np.array([[-.3]])}, empty_epc, lattice)
    bath = ClassicPhononBath(np.empty((0, 1)), .4, empty_epc)
    quantum = QuantumPhononBath([1.], [100.], .4, band_narrow_only=True)
    assert quantum.polaron_prefactor == 0.
    propagator = DensityMatrixUnitaryPropagator(3, 2, .01, .03, .4)
    simulation = GreenKuboSimulation(lattice, ham, bath, quantum, propagator)
    result = simulation.run()
    assert np.isfinite(result.final_state.electronic).all()
    assert np.isfinite(simulation.propagator.jx_0).all()
    np.testing.assert_array_equal(result.observables["current_correlation"], 0.)


def test_nonhermitian_trial_epc_is_rejected_before_native_dynamics():
    lattice = BravaisLattice2D(2, 1, 2, [[0., 0.], [.2, 0.]])
    g = {(0, 0): {(0, 0): np.array([[[0.], [1.]], [[0.], [0.]]])}}
    # Host construction remains available for historical isolated estimator
    # tests carrying unused random trial EPC data.
    ham = ElectronPhononHamiltonian({(0, 0): np.eye(2)}, g, lattice)
    bath = ClassicPhononBath([[1.]], .4, g)
    bath.initialize_position_and_momentum(2, 1, 1, np.random.default_rng(0))
    with pytest.raises(ValueError, match="Hermitian"):
        ham.native(bath)


def test_legacy_layout_rejects_sparse_observation_grid():
    with pytest.raises(ValueError, match="save_every=1"):
        complex_small_case(execution=Execution(save_every=2))


def test_rank_merge_rejects_shifted_time_origins(tmp_path):
    for rank in (0, 1):
        with h5py.File(tmp_path/f"currents_{rank}.h5", "w") as handle:
            handle.attrs["current_step"], handle.attrs["time_step"] = 2, .1
            handle.attrs["initial_time"] = rank*.2
            handle["time"] = np.array([0., .1])+rank*.2
            handle["current_x"] = np.ones(2)
    with pytest.raises(ValueError, match="time grids"):
        merge_outputs(tmp_path, ["x"], 2, safe_mode=True)


def test_global_rank_identity_offsets_without_changing_samples():
    simulation = complex_small_case()
    _, other = _assemble(simulation.ham, simulation.classic_ph, None, simulation.propagator,
                         thermal_policy="legacy_full", trajectory_id_start=8, seed=1120)
    np.testing.assert_array_equal(other.trajectory_id, [8, 9])
    np.testing.assert_array_equal(other.q, simulation.initial_state.q)
    np.testing.assert_array_equal(other.method_state["transport"]["legacy_q0"],
                                   simulation.initial_state.method_state["transport"]["legacy_q0"])
    with pytest.raises(ValueError, match="uint32"):
        _assemble(simulation.ham, simulation.classic_ph, None, simulation.propagator,
                  thermal_policy="legacy_full", trajectory_id_start=2**32-1)
