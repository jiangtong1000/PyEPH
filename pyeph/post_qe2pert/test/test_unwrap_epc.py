"""Unit tests for MIC-related logic in unwrap_epc."""

import importlib.util
import sys
from pathlib import Path

import numpy as np


def _load_unwrap_module():
    module_path = Path(__file__).resolve().parents[1] / "unwrap_epc.py"
    spec = importlib.util.spec_from_file_location("unwrap_epc_under_test", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


unwrap_epc = _load_unwrap_module()


def test_nearest_shift_matches_rounding_in_orthogonal_cells():
    rng = np.random.default_rng(123)
    lattice_vectors = np.diag([4.1, 5.2, 6.3])
    delta_frac = rng.uniform(-1.49, 1.49, size=(200, 3))

    shifts = unwrap_epc._nearest_integer_shift(delta_frac, lattice_vectors)
    assert np.array_equal(shifts, np.rint(delta_frac).astype(int))


def test_nearest_shift_finds_cartesian_minimum_for_skewed_cell():
    lattice_vectors = np.array(
        [
            [8.0, 0.0, 0.0],
            [3.9, 7.0, 0.0],
            [1.7, 2.2, 6.5],
        ],
        dtype=float,
    )
    delta_frac = np.array([-1.12715017, 0.51187324, 0.44156853], dtype=float)

    got = unwrap_epc._nearest_integer_shift(delta_frac.reshape(1, 3), lattice_vectors)[0]
    expected = np.array([-1, 0, 1], dtype=int)
    assert np.array_equal(got, expected)
    assert not np.array_equal(np.rint(delta_frac).astype(int), expected)


def test_assign_atoms_to_mol_on_skewed_cell_uses_metric_mic():
    lattice_vectors = np.array(
        [
            [8.0, 0.0, 0.0],
            [3.9, 7.0, 0.0],
            [1.7, 2.2, 6.5],
        ],
        dtype=float,
    )
    frac_coords = np.array(
        [
            [0.97, 0.22, 0.18],
            [0.03, 0.22, 0.18],
            [0.42, 0.96, 0.63],
            [0.42, 0.04, 0.63],
        ],
        dtype=float,
    )
    tau = frac_coords @ lattice_vectors
    symbols = np.array(["H", "H", "H", "H"])

    pair_map = unwrap_epc.assign_atoms_to_mol(tau, symbols, lattice_vectors)
    assert len(pair_map.mol1_indices) == 2
    assert len(pair_map.mol2_indices) == 2

    # Rebuild pre-centroid shifts to verify the centroid MIC shift selection.
    connectivity_matrix = unwrap_epc.get_bond_connectivity(tau, symbols, lattice_vectors)
    components = unwrap_epc._find_connected_components(connectivity_matrix)
    assert len(components) == 2
    mol1_indices, mol2_indices = components

    delta_cart = tau[None, :, :] - tau[:, None, :]
    delta_frac = unwrap_epc._cart_to_frac(delta_cart, lattice_vectors)
    delta_shifts = -unwrap_epc._nearest_integer_shift(delta_frac, lattice_vectors)
    base_shifts_pre_pair = np.zeros((len(tau), 3), dtype=int)
    unwrap_epc._unwrap_component(
        mol1_indices, connectivity_matrix, delta_shifts, base_shifts_pre_pair
    )
    unwrap_epc._unwrap_component(
        mol2_indices, connectivity_matrix, delta_shifts, base_shifts_pre_pair
    )

    pre_pair_coords = tau + unwrap_epc._frac_to_cart(base_shifts_pre_pair, lattice_vectors)
    centroid1 = np.mean(pre_pair_coords[mol1_indices], axis=0)
    centroid2 = np.mean(pre_pair_coords[mol2_indices], axis=0)
    centroid1_frac = unwrap_epc._cart_to_frac(centroid1.reshape(1, 3), lattice_vectors)[0]
    centroid2_frac = unwrap_epc._cart_to_frac(centroid2.reshape(1, 3), lattice_vectors)[0]
    expected_pair_shift = unwrap_epc._nearest_integer_shift(
        (centroid1_frac - centroid2_frac).reshape(1, 3), lattice_vectors
    )[0]

    applied_shift = pair_map.base_shifts[mol2_indices] - base_shifts_pre_pair[mol2_indices]
    assert np.all(applied_shift == applied_shift[0])
    assert np.array_equal(applied_shift[0], expected_pair_shift)

    unwrapped_tau = tau + unwrap_epc._frac_to_cart(pair_map.base_shifts, lattice_vectors)
    for component in (pair_map.mol1_indices, pair_map.mol2_indices):
        bond_length = np.linalg.norm(unwrapped_tau[component[0]] - unwrapped_tau[component[1]])
        assert bond_length < 0.75


def test_get_bond_connectivity_recognizes_skewed_boundary_bond():
    lattice_vectors = 0.4195 * np.array(
        [
            [8.0, 0.0, 0.0],
            [3.9, 7.0, 0.0],
            [1.7, 2.2, 6.5],
        ],
        dtype=float,
    )
    symbols = np.array(["C", "C"])
    delta_frac = np.array([-1.12715017, 0.51187324, 0.44156853], dtype=float)
    frac_coords = np.array([[0.0, 0.0, 0.0], delta_frac], dtype=float)
    tau = frac_coords @ lattice_vectors

    connectivity_matrix = unwrap_epc.get_bond_connectivity(tau, symbols, lattice_vectors)
    assert connectivity_matrix[0, 1]
    assert connectivity_matrix[1, 0]

    # Regression guard: component-wise fractional rounding misses this bond.
    rounded_mic_frac = delta_frac - np.rint(delta_frac)
    rounded_dist = np.linalg.norm(rounded_mic_frac @ lattice_vectors)
    cutoff = (2.0 * unwrap_epc.COVALENT_RADII["C"]) * unwrap_epc.BOND_TOLERANCE
    assert rounded_dist > cutoff
