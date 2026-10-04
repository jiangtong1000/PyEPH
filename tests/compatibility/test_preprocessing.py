"""Native QE2PERT regression checks; no old PyEPH installation is imported.

DNTT inputs/reference values and skew-cell cases originate in PyEPH revision
6c4693acbb69a06a5bc8b0593abde2170ff38843 (BSD-3-Clause). See data provenance.
Formula checks below are independent NumPy contractions.
"""

import importlib
import json
import hashlib
import logging
import os
from pathlib import Path
import sys

import h5py
import numpy as np
import pytest

from pyeph.post_qe2pert import (
    CalcEphMatReciprocal, ElectronBands, PhononDispersion, parse_qpoint_path,
)
from pyeph.post_qe2pert import unwrap_epc
from pyeph.post_qe2pert._support import get_mpi_info
from pyeph.post_qe2pert.eph_mat_mixed import CalcEphMatMixed
from pyeph.post_qe2pert.linalg import unpack_dyn_matrix

DATA = Path(__file__).parent / "data/preprocessing"
PATH_TEXT = """11
0 0 0 50
0 .5 0 50
0 .5 .5 50
0 0 .5 50
0 0 0 50
-.5 0 .5 50
-.5 .5 .5 50
0 .5 0 50
-.5 .5 0 50
-.5 0 0 50
0 0 0 1
"""


@pytest.fixture(autouse=True)
def serial_preprocessing(monkeypatch):
    monkeypatch.setenv("USE_MPI", "false")


@pytest.fixture(scope="module")
def references():
    with h5py.File(DATA / "references.h5") as handle:
        return {key: handle[key][...] for key in handle}


def test_bundled_input_and_reference_provenance():
    manifest = json.loads((DATA / "provenance.json").read_text())
    assert hashlib.sha256((DATA / "DNTT_epr.h5").read_bytes()).hexdigest() == (
        manifest["originals"]["DNTT_epr.h5"]["sha256"]
    )
    assert hashlib.sha256((DATA / "references.h5").read_bytes()).hexdigest() == (
        manifest["derived"]["references.h5"]["sha256"]
    )


def test_path_preserves_original_interpolation_convention(references):
    path = parse_qpoint_path(PATH_TEXT)
    assert path.shape == (511, 3)
    np.testing.assert_allclose(path[references["path_indices"]], references["points"])
    # ninterp means points strictly between vertices, so 2 gives 4 total points.
    path = parse_qpoint_path("2\n0 0 0 2\n1 0 0 1")
    np.testing.assert_allclose(path[:, 0], [0, 1 / 3, 2 / 3, 1])


def test_packed_hermitian_matrix_keeps_complex_conjugation():
    packed = [2, 1 + 2j, 3, -2j, 4 - 3j, -1]
    expected = [[2, 1 + 2j, -2j], [1 - 2j, 3, 4 - 3j], [2j, 4 + 3j, -1]]
    np.testing.assert_array_equal(unpack_dyn_matrix(packed, 3), expected)


def test_dntt_electron_bands_and_fourier_matrix(references):
    bands = ElectronBands(DATA / "DNTT_epr.h5")
    actual = bands.calc_band_structure(references["points"])
    np.testing.assert_allclose(actual, references["band_energies"], rtol=1e-7, atol=1e-10)
    kpoint = references["points"][1]
    expected = np.zeros((bands.num_wann, bands.num_wann), complex)
    for j in range(bands.num_wann):
        for i in range(j + 1):
            info = bands.ham_r_info[f"H_{i+1}{j+1}"]
            expected[i, j] = sum(
                hop * np.exp(2j * np.pi * np.dot(bands.rvec_set[index], kpoint))
                for hop, index in zip(info["hopping_element"], info["rvec_indices"])
            )
            expected[j, i] = expected[i, j].conjugate()
    vals, vectors = bands.solve_eigenvalue_vector(kpoint)
    np.testing.assert_allclose(expected @ vectors, vectors * vals, atol=1e-12)


def test_dntt_polar_phonon_reference_and_mass_weighting(references):
    phonons = PhononDispersion(DATA / "DNTT_epr.h5", polar=True)
    constants = phonons.extract_force_constants()
    # Reference files store frequencies in Ry, including signed acoustic values.
    frequencies, modes = phonons.compute_phonon_dispersion(
        references["points"], constants, mass_weight=False,
    )
    np.testing.assert_allclose(
        frequencies, references["phonon_frequencies"], rtol=1e-5, atol=1e-9,
    )
    identity = np.eye(3 * phonons.nat)
    for matrix in modes:
        np.testing.assert_allclose(matrix.conj().T @ matrix, identity, atol=1e-10)
    w_mass, weighted = phonons.solve_phonon_modes(constants, references["points"][1])
    np.testing.assert_allclose(w_mass, frequencies[1], atol=1e-12)
    mass_diagonal = np.repeat(phonons.mass, 3)
    np.testing.assert_allclose(
        weighted.conj().T @ (mass_diagonal[:, None] * weighted), identity, atol=1e-10,
    )


def test_dntt_one_reciprocal_epc_pair(references):
    epc = CalcEphMatReciprocal(DATA / "DNTT_epr.h5")
    actual = epc.calc_ephmat(references["epc_kpoints"], references["epc_qpoints"])
    for key in ("deformation_potential", "eph_matrix_elements"):
        np.testing.assert_allclose(actual[key], references[key], rtol=1e-5, atol=1e-8)


def test_mixed_inverse_fourier_formula_retains_upper_triangle():
    # The original mixed-space routine uses an unnormalized inverse transform
    # and fills only the stored electronic upper triangle.
    calculator = object.__new__(CalcEphMatMixed)
    calculator.logger = logging.getLogger("mixed_formula_test")
    calculator.num_wann = 2
    calculator.nat = 1
    calculator.rvec_set_el = np.array([[0, 0, 0], [1, 0, 0]])
    qpoints = np.array([[.2, .1, 0], [-.2, -.1, 0]])
    rph = np.array([[0, 0, 0], [1, 0, 0], [-1, 1, 0]])
    rng = np.random.default_rng(93)
    g = rng.normal(size=(2, 2, 3, 2, 2)) + 1j * rng.normal(size=(2, 2, 3, 2, 2))
    actual = calculator.ifft_to_real_space(g, qpoints, rph)
    expected = np.zeros_like(actual)
    for i in range(2):
        for j in range(i, 2):
            for ir, r in enumerate(rph):
                for iq, q in enumerate(qpoints):
                    expected[i, j, :, :, ir] += g[i, j, :, :, iq] * np.exp(-2j*np.pi*r.dot(q))
    np.testing.assert_allclose(actual, expected, atol=1e-12)


def test_polar_qmesh_singleton_axis_is_zeroed_before_correction(tmp_path, monkeypatch):
    # Migration repairs a legacy NameError when a polar q-mesh axis has length 1.
    file = tmp_path / "polar_parameters.h5"
    with h5py.File(file, "w") as handle:
        handle["basic_data/epsil"] = np.eye(3)
        handle["basic_data/zstar"] = np.zeros((1, 3, 3))
    phonons = object.__new__(PhononDispersion)
    phonons.epr_file = file
    phonons.lpolar = True
    phonons.logger = logging.getLogger("polar_formula_test")
    phonons.bg = np.eye(3)
    phonons.qc_dim = np.array([1, 2, 1])
    monkeypatch.setattr(phonons, "init_onsite_polar_correction", lambda: None)
    phonons.setup_polar_correction()
    for key in ("nrx", "nrx_ph"):
        np.testing.assert_array_equal(phonons.polar_params[key][[0, 2]], [0, 0])
        assert phonons.polar_params[key][1] > 0


def test_skew_cell_minimum_image_and_intact_molecules():
    lattice = np.array([[8., 0, 0], [3.9, 7, 0], [1.7, 2.2, 6.5]])
    difference = np.array([[-1.12715017, .51187324, .44156853]])
    shift = unwrap_epc._nearest_integer_shift(difference, lattice)
    np.testing.assert_array_equal(shift, [[-1, 0, 1]])
    assert not np.array_equal(shift, np.rint(difference))
    frac = np.array([[.97, .22, .18], [.03, .22, .18], [.42, .96, .63], [.42, .04, .63]])
    tau = frac @ lattice
    pairs = unwrap_epc.assign_atoms_to_mol(tau, ["H"] * 4, lattice)
    assert len(pairs.mol1_indices) == len(pairs.mol2_indices) == 2
    unwrapped = tau + pairs.base_shifts @ lattice
    for indices in (pairs.mol1_indices, pairs.mol2_indices):
        assert np.linalg.norm(unwrapped[indices[1]] - unwrapped[indices[0]]) < .75


def test_localization_import_does_not_require_qcpbc(monkeypatch):
    import jax

    backend_environment = {key: os.environ.get(key) for key in ("XLA_FLAGS", "JAX_PLATFORM_NAME")}
    use_x64 = jax.config.jax_enable_x64
    monkeypatch.setitem(sys.modules, "pyqcpbc", None)
    module = importlib.import_module("pyeph.post_qe2pert.localize_eph_pyqcpbc")
    importlib.reload(module)
    assert jax.config.jax_enable_x64 == use_x64
    assert {key: os.environ.get(key) for key in backend_environment} == backend_environment
    with pytest.raises(ImportError, match="optional pyqcpbc"):
        module.compute_eph_mat_real_space_pyqcpbc(*([None] * 11))


def test_localization_density_matches_independent_formula():
    from pyeph.post_qe2pert.localize_eph import compute_density, zero_out_negative_freqs

    rng = np.random.default_rng(9)
    epc = rng.normal(size=(2, 2, 3, 4, 5)) + 1j * rng.normal(size=(2, 2, 3, 4, 5))
    positions = rng.normal(size=(4, 3))
    density, radius = compute_density(epc, positions)
    reference = np.array([np.sum(np.abs(epc[:, :, :, ir, :]) ** 2) for ir in range(4)])
    np.testing.assert_allclose(density, reference / reference.sum(), atol=1e-12)
    np.testing.assert_allclose(radius, np.linalg.norm(positions, axis=1))
    frequencies = np.array([[-.1, .5, 1.0]])
    modes, frequencies = zero_out_negative_freqs(frequencies, np.eye(3)[None].copy(), .2)
    np.testing.assert_array_equal(frequencies, [[.2, .5, 1]])
    np.testing.assert_array_equal(modes[0], np.diag([0, 1, 1]))


def test_optional_mpi_serial_contract():
    assert get_mpi_info() == {"has_mpi": False, "rank": 0, "size": 1, "comm": None}


def test_public_qcpbc_log_parser(tmp_path):
    from pyeph.utils.analyze_qcpbc_opt import extract_loss_history_qcpbc

    log = tmp_path / "localization.log"
    log.write_text("Initialization\nCycle Loss Gradient\n--------------------\n"
                   "0 1.5e-2 2.0\n1 0.0025 0.1\n\n2 0.0001 0.01\n"
                   "Converged successfully\n3 99.0 ignored\n")
    assert extract_loss_history_qcpbc(log) == [.015, .0025, .0001]
    log.write_text("No optimization table\n")
    with pytest.raises(ValueError, match="Cycle table"):
        extract_loss_history_qcpbc(log)


def test_public_cif_helper_keeps_pymatgen_optional(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "pymatgen", None)
    module = importlib.import_module("pyeph.utils.cifio")
    importlib.reload(module)
    output = tmp_path / "geometry.in"
    with pytest.raises(ValueError, match="positions_unit"):
        module.cif_to_qe_blocks("unused.cif", output, positions_unit="nm")
    with pytest.raises(ImportError, match="optional pymatgen"):
        module.cif_to_qe_blocks("unused.cif", output)
    assert not output.exists()


def test_native_dntt_extraction_entrypoint(tmp_path):
    from pyeph.post_qe2pert.extract import main

    output = tmp_path / "dntt_extracted.h5"
    args = ["--epr-file", str(DATA / "DNTT_epr.h5"), "--nx", "2", "--ny", "2",
            "--output", str(output)]
    main(args)
    with h5py.File(output) as handle, h5py.File(DATA / "DNTT_epr.h5") as source:
        assert handle["freq_full"].shape == (4, 216)
        assert handle["mode_full"].shape == (4, 216, 216)
        np.testing.assert_array_equal(handle["mass"], source["basic_data/mass"])
        for key in ("gmat_raw", "freq_full", "mode_full", "mass"):
            assert np.isfinite(handle[key][...]).all()
        modes = handle["mode_full"][...]
        partners = handle["partner_hbz_for_minus"][...]
        half = len(handle["q_hbz"])
        np.testing.assert_allclose(modes[half:], modes[partners].conj())
        np.testing.assert_allclose(handle["q_minus"][...], -handle["q_hbz"][...][partners])
        for matrix in modes:
            np.testing.assert_allclose(matrix.conj().T @ matrix, np.eye(216), atol=1e-10)
        assert handle["freq_full"].attrs["units"] == "Ry"
    before = hashlib.sha256(output.read_bytes()).hexdigest()
    with pytest.raises(SystemExit):
        main(args)
    assert hashlib.sha256(output.read_bytes()).hexdigest() == before
