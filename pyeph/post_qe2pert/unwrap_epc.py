"""
Visualize the atomic-resolved EPC for selected molecular clusters in a supercell.
The input unit cell contains all atoms that belong to two interacting molecules (mol-1 and mol-2).
Although the molecules are chopped up across the box boundaries, depending on the starting CIF input.
"""

from __future__ import annotations

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
    mol2_center: np.ndarray


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


def _nearest_integer_shift_single(
    d_frac: np.ndarray,
    metric: np.ndarray,
    lambda_min: float,
) -> np.ndarray:
    """
    Find integer n minimizing ||(d_frac - n) @ lattice||_2^2 using the lattice metric.
    """
    d_frac = np.asarray(d_frac, dtype=float).reshape(3)
    n0 = np.rint(d_frac).astype(int)
    diff0 = d_frac - n0
    upper_bound = float(diff0 @ metric @ diff0)
    radius = int(np.ceil(np.sqrt(max(upper_bound, 0.0) / lambda_min))) + 1

    lower = np.floor(d_frac - radius).astype(int)
    upper = np.ceil(d_frac + radius).astype(int)

    best_n = n0.copy()
    best_val = upper_bound
    best_key = tuple(int(x) for x in best_n)
    tol = 1e-14

    for i, j, k in product(
        range(int(lower[0]), int(upper[0]) + 1),
        range(int(lower[1]), int(upper[1]) + 1),
        range(int(lower[2]), int(upper[2]) + 1),
    ):
        n = np.array([i, j, k], dtype=int)
        diff = d_frac - n
        val = float(diff @ metric @ diff)
        key = (i, j, k)
        if (val < best_val - tol) or (abs(val - best_val) <= tol and key < best_key):
            best_val = val
            best_n = n
            best_key = key

    return best_n


def _nearest_integer_shift(d_frac: np.ndarray, lattice_vectors: np.ndarray) -> np.ndarray:
    """
    Vectorized nearest-image integer shift for shape (..., 3) fractional displacements.
    """
    d_frac = np.asarray(d_frac, dtype=float)
    if d_frac.shape[-1] != 3:
        raise ValueError(f"`d_frac` must have shape (..., 3), got {d_frac.shape}.")

    metric = lattice_vectors @ lattice_vectors.T
    eigenvalues = np.linalg.eigvalsh(metric)
    lambda_min = float(np.min(eigenvalues))
    if lambda_min <= 0:
        raise ValueError("`lattice_vectors` must define a positive-definite metric.")

    flat = d_frac.reshape(-1, 3)
    flat_shifts = np.empty_like(flat, dtype=int)
    for idx, disp in enumerate(flat):
        flat_shifts[idx] = _nearest_integer_shift_single(disp, metric, lambda_min)
    return flat_shifts.reshape(d_frac.shape)


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
            else:
                if not np.array_equal(base_shifts[j], candidate):
                    raise RuntimeError(
                        f"Topology Error: Cycle inconsistency detected!\n"
                        f"  Atom {j} was previously assigned shift {base_shifts[j]}.\n"
                        f"  But path via atom {i} demands shift {candidate}.\n"
                        f"  Check your adjacency distance cutoffs; intermolecular "
                        f"contacts are likely being mistaken for covalent bonds."
                    )

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
    delta_frac_mic = delta_frac - _nearest_integer_shift(delta_frac, lattice_vectors)
    delta_mic_cart = _frac_to_cart(delta_frac_mic, lattice_vectors)
    distances = np.linalg.norm(delta_mic_cart, axis=-1)

    cutoff = (radii[:, None] + radii[None, :]) * BOND_TOLERANCE
    connectivity_matrix = (distances <= cutoff) & (distances > 1e-12)
    np.fill_diagonal(connectivity_matrix, False)
    return connectivity_matrix

def assign_atoms_to_mol(tau, symbols, lattice_vectors) -> IntactPairMap:
    """
    Finds mol-1, mol-2, unwrap them, and pairs them using Centroid MIC.
    Returns the mapping shifts and the final Cartesian centers of both molecules.
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
    delta_cart = tau[None, :, :] - tau[:, None, :]
    delta_frac = _cart_to_frac(delta_cart, lattice_vectors)
    delta_shifts = -_nearest_integer_shift(delta_frac, lattice_vectors)

    # 2/3. BFS unwrap both molecules independently (using the cycle-safe version)
    _unwrap_component(mol1_indices, connectivity_matrix, delta_shifts, base_shifts)
    _unwrap_component(mol2_indices, connectivity_matrix, delta_shifts, base_shifts)

    # 4. Centroid MIC Pairing
    unwrapped_tau = tau + _frac_to_cart(base_shifts, lattice_vectors)
    mol1_coords = unwrapped_tau[mol1_indices]
    mol2_coords = unwrapped_tau[mol2_indices]

    # Calculate geometric centroids in Cartesian space (safe because they are unwrapped)
    centroid1_cart = np.mean(mol1_coords, axis=0)
    centroid2_cart = np.mean(mol2_coords, axis=0)

    # Convert to fractional to easily find the MIC integer shift
    centroid1_frac = _cart_to_frac(centroid1_cart.reshape(1, 3), lattice_vectors)[0]
    centroid2_frac = _cart_to_frac(centroid2_cart.reshape(1, 3), lattice_vectors)[0]

    # The exact integer shift needed to bring Mol2's centroid to Mol1's centroid
    delta_frac = centroid1_frac - centroid2_frac
    best_shift = _nearest_integer_shift(delta_frac.reshape(1, 3), lattice_vectors)[0]

    # Apply the shift to Mol-2's atoms
    base_shifts[mol2_indices] += best_shift

    # Calculate the final shifted center for Mol-2 in Cartesian space
    shift_cart = _frac_to_cart(best_shift.reshape(1, 3), lattice_vectors)[0]
    final_centroid2_cart = centroid2_cart + shift_cart

    return IntactPairMap(
        base_shifts=base_shifts,
        mol1_indices=mol1_indices,
        mol2_indices=mol2_indices,
        mol1_center=centroid1_cart,
        mol2_center=final_centroid2_cart
    )

def merge_cell_epc(tau: np.ndarray, lattice_vectors: np.ndarray, pair_map: IntactPairMap, cells: np.ndarray, EPC: np.ndarray) -> tuple[dict, dict, float]:
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
                             containing 'atoms_coord', 'epc', and 'atom_indices' arrays for intact pairs.
        frag_cells_dict: dict, keys are new cell tuples (i', j', k'), values are dicts
                         containing 'atoms_coord', 'epc', and 'atom_indices' arrays for boundary fragments.
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
        raise ValueError(f"`EPC` must have shape (ncell, natoms, 3) = ({ncell}, {natoms}, 3), got {EPC.shape}.")
    if pair_map.base_shifts.shape != (natoms, 3):
        raise ValueError(f"`pair_map.base_shifts` must have shape ({natoms}, 3), got {pair_map.base_shifts.shape}.")
    
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
    frag_cells_dict = {}  # NEW: Store the fragments

    total_epc_magnitude = (np.abs(flat_epc)**2).sum()
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

        # Sort atoms to maintain consistent ordering in both dicts
        order = np.argsort(member_atoms)
        atom_order = member_atoms[order]
        cell_shift_cart = _frac_to_cart(np.array(cell_tuple, dtype=float), lattice_vectors)
        shifted_coords = (unwrapped_coords[atom_order] + cell_shift_cart).copy()

        if is_complete:
            complete_cells_dict[cell_tuple] = {
                "atoms_coord": shifted_coords,
                "epc": member_epc[order].copy(),
                "atom_indices": atom_order.copy(),
            }
        else:
            fragment_epc_magnitude += (np.abs(member_epc)**2).sum()
            # NEW: Save the fragment data
            frag_cells_dict[cell_tuple] = {
                "atoms_coord": shifted_coords,
                "epc": member_epc[order].copy(),
                "atom_indices": atom_order.copy(),
            }
    
    # 4. The Fractional Leakage Sanity Check
    boundary_leakage_fraction = (
        fragment_epc_magnitude / total_epc_magnitude if total_epc_magnitude > 0 else 0.0
    )
    print(f"Supercell boundary leakage fraction: {boundary_leakage_fraction:.4f}")
    
    return complete_cells_dict, frag_cells_dict, boundary_leakage_fraction

def get_unwrapped_base_xyz(
    tau: np.ndarray,
    symbols: list[str] | np.ndarray,
    lattice_vectors: np.ndarray,
    pair_map: IntactPairMap
) -> str:
    """
    Applies the base shifts to the original cell and formats the result as an .xyz string.
    Groups mol-1 and mol-2 atoms together sequentially for easy visualization.

    Parameters:
        tau: ndarray, (natoms, 3), original fractional/broken coordinates.
        symbols: list or ndarray of atomic symbols.
        lattice_vectors: ndarray, (3, 3), lattice vectors in Cartesian coordinates.
        pair_map: IntactPairMap generated by assign_atoms_to_mol.

    Returns:
        A string formatted in standard .xyz format containing the unwrapped base cell.
    """
    tau, lattice_vectors = _validate_positions(tau, lattice_vectors)
    symbols = np.asarray(symbols)

    # 1. Apply the integer shifts to unwrap the base cell
    unwrapped_tau = tau + _frac_to_cart(pair_map.base_shifts, lattice_vectors)

    # 2. Start building the standard XYZ format
    natoms = len(tau)
    lines = [
        str(natoms),
        "Unwrapped Base Cell Sanity Check | Mol-1 followed by Mol-2"
    ]

    # 3. Append Mol-1 atoms
    for idx in pair_map.mol1_indices:
        coords = unwrapped_tau[idx]
        lines.append(f"{symbols[idx]:2s} {coords[0]:12.6f} {coords[1]:12.6f} {coords[2]:12.6f}")

    # 4. Append Mol-2 atoms
    for idx in pair_map.mol2_indices:
        coords = unwrapped_tau[idx]
        lines.append(f"{symbols[idx]:2s} {coords[0]:12.6f} {coords[1]:12.6f} {coords[2]:12.6f}")

    return "\n".join(lines)

def write_supercell_xyz(
    filename: str,
    complete_cells_dict: dict,
    frag_cells_dict: dict,
    symbols: list[str] | np.ndarray,
    mode: str = "both"
) -> None:
    """
    Writes the coordinates of the supercell mapping to a standard .xyz file.

    Parameters:
        filename: The output path for the .xyz file.
        complete_cells_dict: Dictionary of complete cells.
        frag_cells_dict: Dictionary of boundary fragments.
        symbols: List or 1D array of atomic symbols corresponding to the base unit cell.
        mode: String determining what to write. Options are 'complete', 'frag', or 'both'.
    """
    mode = mode.lower()
    if mode not in ["complete", "frag", "both"]:
        raise ValueError("The 'mode' argument must be 'complete', 'frag', or 'both'.")

    symbols = np.asarray(symbols)

    # 1. Determine which dictionaries to process based on the mode
    write_complete = mode in ["complete", "both"]
    write_frag = mode in ["frag", "both"]

    # 2. Dynamically count total atoms and cell stats
    total_atoms = 0
    num_complete = len(complete_cells_dict) if write_complete else 0
    num_frag = len(frag_cells_dict) if write_frag else 0

    if write_complete:
        total_atoms += sum(len(data["atoms_coord"]) for data in complete_cells_dict.values())
    if write_frag:
        total_atoms += sum(len(data["atoms_coord"]) for data in frag_cells_dict.values())

    if total_atoms == 0:
        print(f"Warning: No atoms found for mode '{mode}'. Nothing to write.")
        return

    # 3. Setup the header line
    header_comment = f"Supercell mapping | Mode: {mode} | {num_complete} complete pairs, {num_frag} boundary fragments"
    lines = [str(total_atoms), header_comment]

    # 4. Conditionally append complete cells
    if write_complete:
        for cell_tuple, cell_data in complete_cells_dict.items():
            coords = cell_data["atoms_coord"]
            indices = cell_data["atom_indices"]
            for i in range(len(coords)):
                sym = symbols[indices[i]]
                x, y, z = coords[i]
                lines.append(f"{sym:2s} {x:12.6f} {y:12.6f} {z:12.6f}")

    # 5. Conditionally append fragment cells
    if write_frag:
        for cell_tuple, cell_data in frag_cells_dict.items():
            coords = cell_data["atoms_coord"]
            indices = cell_data["atom_indices"]
            for i in range(len(coords)):
                sym = symbols[indices[i]]
                x, y, z = coords[i]
                lines.append(f"{sym:2s} {x:12.6f} {y:12.6f} {z:12.6f}")

    # 6. Write to disk
    with open(filename, 'w') as f:
        f.write("\n".join(lines))
        f.write("\n")

    print(f"Successfully wrote {total_atoms} atoms to '{filename}' (Mode: {mode})")
