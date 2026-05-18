"""
Visualize the atomic-resolved EPC for selected molecular clusters in a supercell.
The input unit cell contains all atoms that belong to two interacting molecules (mol-1 and mol-2).
Although the molecules are chopped up across the box boundaries, depending on the starting CIF input.
"""

import numpy as np
from collections import deque
from dataclasses import dataclass
from itertools import product

# constants for bond connectivity to assign atoms to molecules
COVALENT_RADII = {
    'H': 0.31, 'O': 0.66, 'N': 0.71, 'C': 0.76, 'S': 1.05
}
DEFAULT_COVALENT_RADIUS = 0.77
BOND_TOLERANCE = 1.20

@dataclass
class IntactPairMap:
    """
    mapping original cells to new cells for atoms.
    we only need to know the shifts for the base cell, since other cells are translations
    
    base_shifts: 
        integer unwrapping shifts for the base cell
        shape: (natoms, 3)
        the i-th atom now sits at the new cell (i',j',k') = base_shifts[i].
    mol1_indices: 
        atom indices for mol-1.
        shape: (n_mol1,)
    mol2_indices: 
        atom indices for mol-2.
        shape: (n_mol2,)
    """
    base_shifts: np.ndarray
    mol1_indices: np.ndarray
    mol2_indices: np.ndarray
    mol1_center: np.ndarray


def _validate_positions(tau: np.ndarray, lattice_vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    tau = np.asarray(tau, dtype=float)
    lattice_vectors = np.asarray(lattice_vectors, dtype=float)
    assert tau.ndim == 2 and tau.shape[1] == 3, f"`tau` must have shape (natoms, 3), got {tau.shape}."
    assert lattice_vectors.shape == (3, 3), f"`lattice_vectors` must have shape (3, 3), got {lattice_vectors.shape}."
    return tau, lattice_vectors


def _cart_to_frac(vectors_cart: np.ndarray, lattice_vectors: np.ndarray) -> np.ndarray:
    vectors_cart = np.asarray(vectors_cart, dtype=float)
    flat = vectors_cart.reshape(-1, 3)
    flat_frac = np.linalg.solve(lattice_vectors.T, flat.T).T
    return flat_frac.reshape(vectors_cart.shape)


def _frac_to_cart(vectors_frac: np.ndarray, lattice_vectors: np.ndarray) -> np.ndarray:
    vectors_frac = np.asarray(vectors_frac, dtype=float)
    flat = vectors_frac.reshape(-1, 3)
    flat_cart = flat @ lattice_vectors
    return flat_cart.reshape(vectors_frac.shape)


def _find_connected_components(connectivity_matrix: np.ndarray) -> list[np.ndarray]:
    natoms = connectivity_matrix.shape[0]
    unseen = np.ones(natoms, dtype=bool)
    components = []

    for start in range(natoms):
        if not unseen[start]:
            continue
        queue = deque([start])
        unseen[start] = False
        component = []
        while queue:
            i = queue.popleft()
            component.append(i)
            for j in np.where(connectivity_matrix[i])[0]:
                if unseen[j]:
                    unseen[j] = False
                    queue.append(j)
        components.append(np.array(sorted(component), dtype=int))

    components.sort(key=lambda comp: int(comp[0]))
    return components


def _unwrap_component(
    component: np.ndarray,
    connectivity_matrix: np.ndarray,
    delta_shifts: np.ndarray,
    base_shifts: np.ndarray,
) -> None:
    comp_mask = np.zeros(connectivity_matrix.shape[0], dtype=bool)
    comp_mask[component] = True

    root = int(component[0])
    visited = np.zeros(connectivity_matrix.shape[0], dtype=bool)
    queue = deque([root])
    visited[root] = True
    base_shifts[root] = 0

    while queue:
        i = queue.popleft()
        neighbors = np.where(connectivity_matrix[i] & comp_mask)[0]
        for j in neighbors:
            candidate = base_shifts[i] + delta_shifts[i, j]
            if not visited[j]:
                base_shifts[j] = candidate
                visited[j] = True
                queue.append(j)

    if not np.all(visited[component]):
        raise RuntimeError("Failed to unwrap one molecular component; graph traversal did not reach all atoms.")

def get_bond_connectivity(tau, symbols, lattice_vectors):
    """
    Computes a PBC-aware boolean connectivity matrix using MIC 
    and element-specific covalent radii.
    
    tau: original cell's atomic positions in Cartesian coordinates (natoms, 3)
    symbols: atomic symbols (natoms,)
    lattice_vectors: lattice vectors in Cartesian coordinates (nvecs=3, 3)
    
    """
    tau, lattice_vectors = _validate_positions(tau, lattice_vectors)
    symbols = np.asarray(symbols)
    natoms = tau.shape[0]
    if symbols.shape[0] != natoms:
        raise ValueError(f"`symbols` must have length {natoms}, got {symbols.shape[0]}.")

    radii = np.array([COVALENT_RADII.get(str(sym), DEFAULT_COVALENT_RADIUS) for sym in symbols], dtype=float)

    # Pairwise displacement i->j with MIC.
    delta_cart = tau[None, :, :] - tau[:, None, :]
    delta_frac = _cart_to_frac(delta_cart, lattice_vectors)
    delta_frac_mic = delta_frac - np.rint(delta_frac)
    delta_mic_cart = _frac_to_cart(delta_frac_mic, lattice_vectors)
    distances = np.linalg.norm(delta_mic_cart, axis=-1)

    cutoff = (radii[:, None] + radii[None, :]) * BOND_TOLERANCE
    connectivity_matrix = (distances <= cutoff) & (distances > 1e-12)
    np.fill_diagonal(connectivity_matrix, False)
    return connectivity_matrix

def assign_atoms_to_mol(tau, symbols, lattice_vectors) -> IntactPairMap:
    """
    Finds mol-1, mol-2, and pairs them using a 27-shift global search.

    Logic:
    In the original unit cell, all atoms to form mol-1 and mol-2 are exisiting, 
    it's just they are fragmented, and some atoms need to unwrap across cells to form a complete molecule.
    so the atomic indices will be fixed, but mapping original cell indices to new cell indices to make the new cell include complete molecule.
    
    Parameters:
        tau: atomic positions in Cartesian coordinates (natoms, 3)
        lattice_vectors: lattice vectors in Cartesian coordinates (nvecs=3, 3)
    
    Returns:
        IntactPairMap: A dataclass containing:
            - base_shifts: ndarray (natoms, 3), the integer unwrapping shifts for the base cell.
            - mol1_indices: ndarray (n_mol1,), indices of atoms belonging to mol-1.
            - mol2_indices: ndarray (n_mol2,), indices of atoms belonging to mol-2.
    """
    tau, lattice_vectors = _validate_positions(tau, lattice_vectors)
    natoms = len(tau)
    symbols = np.asarray(symbols)
    if symbols.shape[0] != natoms:
        raise ValueError(f"`symbols` must have length {natoms}, got {symbols.shape[0]}.")

    base_shifts = np.zeros((natoms, 3), dtype=int)
    
    # 1. Build topology
    connectivity_matrix = get_bond_connectivity(tau, symbols, lattice_vectors)
    components = _find_connected_components(connectivity_matrix)
    if len(components) != 2:
        raise ValueError(
            f"Expected exactly 2 molecular components in the unit cell, found {len(components)}. "
            "Check covalent radii, bond tolerance, or input geometry."
        )
    mol1_indices, mol2_indices = components

    # Pairwise integer image offsets from i->j under MIC:
    # shift[j] = shift[i] - round(frac(tau[j]-tau[i])).
    delta_cart = tau[None, :, :] - tau[:, None, :]
    delta_frac = _cart_to_frac(delta_cart, lattice_vectors)
    delta_shifts = -np.rint(delta_frac).astype(int) # is the sign consistent

    # 2/3. BFS unwrap both molecules independently.
    _unwrap_component(mol1_indices, connectivity_matrix, delta_shifts, base_shifts)
    _unwrap_component(mol2_indices, connectivity_matrix, delta_shifts, base_shifts)

    # 4. Robust dimer pairing via 27 integer image offsets of mol-2.
    unwrapped_tau = tau + _frac_to_cart(base_shifts, lattice_vectors)
    mol1_coords = unwrapped_tau[mol1_indices]
    mol2_coords = unwrapped_tau[mol2_indices]

    best_shift = np.zeros(3, dtype=int)
    best_contact = np.inf
    for shift_tuple in product((-1, 0, 1), repeat=3):
        trial_shift = np.array(shift_tuple, dtype=int)
        trial_coords = mol2_coords + (trial_shift @ lattice_vectors)
        pairwise_dist = np.linalg.norm(mol1_coords[:, None, :] - trial_coords[None, :, :], axis=-1)
        closest_contact = np.min(pairwise_dist)
        if closest_contact < best_contact:
            best_contact = closest_contact
            best_shift = trial_shift

    base_shifts[mol2_indices] += best_shift
    return IntactPairMap(base_shifts, mol1_indices, mol2_indices)


def merge_cell_epc(tau: np.ndarray, lattice_vectors: np.ndarray, pair_map: IntactPairMap, cells: np.ndarray, EPC: np.ndarray) -> tuple[dict, float]:
    """
        Vectorized mapping of the EPC data onto the supercell grid, 
        complete with the fractional leakage sanity check.
            
        Uses the base cell integer shifts to instantly project all atoms into their new 
        (i', j', k') cells. Bins containing exactly `natoms` represent complete, 
        intact dimer pairs. Bins containing fewer atoms are boundary fragments.
        
        Parameters:
            tau: ndarray, (natoms, 3), atomic positions in Cartesian coordinates.
            lattice_vectors: ndarray, (3, 3), lattice vectors in Cartesian coordinates.
            pair_map: IntactPairMap, the fundamental unwrapping shifts from assign_atoms_to_mol.
            cells: ndarray, (ncell, 3), the integer supercell grid vectors.
            EPC: ndarray, (ncell, natoms, 3), the EPC vectors for each atom in the original supercell.
        
        Returns:
            complete_cells_dict: dict, keys are new cell tuples (i', j', k'), values are dicts 
                                containing 'atoms_coord' and 'epc' arrays for the intact pairs.
            boundary_leakage_fraction: float, the ratio of EPC magnitude on boundary fragments 
                                    to the total EPC magnitude in the supercell.
    """
    
    tau, lattice_vectors = _validate_positions(tau, lattice_vectors)
    cells = np.asarray(cells, dtype=int)
    EPC = np.asarray(EPC, dtype=float)
    ncell = len(cells)
    natoms = len(tau)
    if cells.ndim != 2 or cells.shape[1] != 3:
        raise ValueError(f"`cells` must have shape (ncell, 3), got {cells.shape}.")
    if EPC.ndim != 3 or EPC.shape != (ncell, natoms, 3):
        raise ValueError(
            f"`EPC` must have shape (ncell, natoms, 3) = ({ncell}, {natoms}, 3), got {EPC.shape}."
        )
    if pair_map.base_shifts.shape != (natoms, 3):
        raise ValueError(
            f"`pair_map.base_shifts` must have shape ({natoms}, 3), got {pair_map.base_shifts.shape}."
        )
    
    # 1. Array Vectorization: Calculate ALL new cell indices instantly
    # cells shape: (ncell, 3). base_shifts shape: (natoms, 3).
    # R_new shape becomes (ncell, natoms, 3)
    R_new = cells[:, None, :] - pair_map.base_shifts[None, :, :]
    
    # 2. Flatten and group by the new cell tuples
    flat_new_cells = R_new.reshape(-1, 3)
    flat_atom_idx = np.broadcast_to(np.arange(natoms), (ncell, natoms)).reshape(-1)
    flat_epc = EPC.reshape(-1, 3)

    unique_cells, inverse, counts = np.unique(
        flat_new_cells, axis=0, return_inverse=True, return_counts=True
    )
    
    # 3. Separate Complete Pairs vs. Boundary Fragments; Those complete cells have exactly two complete molecules, the boundary cells could have fragments
    complete_cells_dict = {}
    total_epc_magnitude = np.sum(np.linalg.norm(flat_epc, axis=1))
    fragment_epc_magnitude = 0.0
    full_atom_order = np.arange(natoms, dtype=int)
    unwrapped_coords = tau + _frac_to_cart(pair_map.base_shifts, lattice_vectors)

    sorted_member_idx = np.argsort(inverse, kind="stable")
    sorted_groups = inverse[sorted_member_idx]
    group_split = np.flatnonzero(
        np.r_[True, sorted_groups[1:] != sorted_groups[:-1], True]
    )

    for start, end in zip(group_split[:-1], group_split[1:]):
        members = sorted_member_idx[start:end]
        gid = int(sorted_groups[start])
        cell_tuple = tuple(int(x) for x in unique_cells[gid])
        member_atoms = flat_atom_idx[members]
        member_epc = flat_epc[members]

        is_complete = (
            counts[gid] == natoms and
            np.array_equal(np.sort(member_atoms), full_atom_order)
        )
        if is_complete:
            order = np.argsort(member_atoms)
            atom_order = member_atoms[order]
            cell_shift_cart = _frac_to_cart(np.array(cell_tuple, dtype=float), lattice_vectors)
            complete_cells_dict[cell_tuple] = {
                "atoms_coord": (unwrapped_coords[atom_order] + cell_shift_cart).copy(),
                "epc": member_epc[order].copy(),
                "atom_indices": atom_order.copy(),
            }
        else:
            fragment_epc_magnitude += float(np.sum(np.linalg.norm(member_epc, axis=1)))
    
    # 4. The Fractional Leakage Sanity Check
    # eta = sum(abs(fragment_EPC)) / sum(abs(all_EPC))
    boundary_leakage_fraction = (
        fragment_epc_magnitude / total_epc_magnitude if total_epc_magnitude > 0 else 0.0
    )
    print(f"Supercell boundary leakage fraction: {boundary_leakage_fraction:.4f}")
    
    return complete_cells_dict, boundary_leakage_fraction