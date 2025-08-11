# extract eph matrix elements from the ephmat.h5 file

import h5py
import numpy as np
from itertools import product


def get_length(r_cryst, at):
    """
    Calculate length of a crystal vector in Cartesian coordinates
    r_cryst: R vector in crystal coordinates (3,)
    at: lattice vectors (3,3)
        the i-th row of at is the i-th lattice vector
        the column is the x, y, z components of the lattice vector
    """
    r_cart = at.T @ r_cryst
    return np.linalg.norm(r_cart)

def set_cutoff_small(rdim, at):
    """
    Cutoff distance for Wigner-Seitz cell vector search
    rdim: k-mesh dimensions [nk1, nk2, nk3]
    at: lattice vectors (3,3)
    """
    ndim = np.array(rdim) // 2 + 1
    cutoff = 0.0
    
    for i, j, k in product([-1, 1], repeat=3): # corner points of the k-mesh
        r_cryst = ndim * np.array([i, j, k], dtype=float)
        dist = get_length(r_cryst, at)
        if dist > cutoff:
            cutoff = dist
    
    return cutoff

class PostQE2Pert():
    def __init__(self, epr_file):
        self.epr_file = epr_file
        self.read_hdf5_data()

    def read_hdf5_data(self):
        """
        at: lattice vectors in unit of lattice constant
        kc_dim: k-mesh dimensions
        num_wann: number of Wannier functions
        wannier_center_cryst: Wannier centers in crystal coordinates
        """
        with h5py.File(self.epr_file, 'r') as f:
            self.at = f['basic_data/at'][:]
            self.alat = f['basic_data/alat'][()]
            self.kc_dim = f['basic_data/kc_dim'][:]
            self.qc_dim = f['basic_data/qc_dim'][:]
            self.num_wann = f['basic_data/num_wann'][()]
            self.nat = f['basic_data/nat'][()]
            self.wannier_center_cryst = f['basic_data/wannier_center_cryst'][:]
            self.tau = f['basic_data/tau'][:]  # atomic positions in cart coordinates (unit of alat). (nat, 3)
            self.bg = f['basic_data/bg'][:]    # reciprocal lattice vectors in unit of 2pi / alat
            self.mass = f['basic_data/mass'][:]  # real, (nat,) atomic masses in atomic unit
            assert self.wannier_center_cryst.shape == (self.num_wann, 3), f"wannier_center_cryst shape mismatch: {self.wannier_center_cryst.shape} != ({self.num_wann}, 3)"

    def cryst_to_cart(self, positions, direction=1):
        """
        Convert between crystal and Cartesian coordinates
        positions: array of positions (nvec or nat, 3)
        direction: 1 for cryst->cart, -1 for cart->cryst
        """
        if direction == 1:
            # Crystal to Cartesian: r_cart = at.T @ r_cryst
            return np.array([self.at.T @ pos for pos in positions])
        else:
            # Cartesian to Crystal: r_cryst = bg.T @ r_cart # bg.T here is inverse of at.T
            return np.array([self.bg.T @ pos for pos in positions])

    def init_rvec_images(self, kdim=None, ws_search_range=3):
        """
        Initialize all possible R vectors (in crystal coordinates)
        kdim: k-mesh dimensions [nk1, nk2, nk3], defaults to self.kc_dim
        Returns: 
        idx_accum_grid: index of the first R vector in vec_cryst for
        nimag_per_grid: number of R vectors in vec_cryst for each grid point
        vec_cryst: union of all R vectors
        """
        if kdim is None:
            kdim = self.kc_dim
        nr1, nr2, nr3 = kdim
        ws_dim = np.array([nr1, nr2, nr3])
        cutoff = set_cutoff_small(ws_dim, self.at)

        idx_accum_grid = np.zeros(nr1*nr2*nr3, dtype=int)
        nimag_per_grid = np.zeros(nr1*nr2*nr3, dtype=int)
        rvecs_list = []
        tot = 0

        for ir, (k, j, i) in enumerate(product(range(nr3), range(nr2), range(nr1))):
            base_vec = np.array([i, j, k])
            nimg = 0
            # Pre-filtering before computing actual hopping distance
            for (mi, mj, mk) in product(range(-ws_search_range, ws_search_range+1), repeat=3):
                # supercell translations of base_vec, all physically equivalent positions due to PBC
                rvec = np.array([mi, mj, mk]) * ws_dim + base_vec
                if get_length(rvec, self.at) < cutoff:
                    # a point distance from origin will also not contribute to any hopping
                    # since the hopping relies on |R+tau_a - tau_b|
                    nimg += 1
                    rvecs_list.append(rvec)
                            
            if nimg < 1:
                raise ValueError('nim < 1 for grid point ({},{},{})'.format(i, j, k))
            idx_accum_grid[ir] = tot
            nimag_per_grid[ir] = nimg
            tot += nimg
        
        rvecs = np.array(rvecs_list).reshape(tot, 3)
        return {'idx_accum_grid': idx_accum_grid, 'nimag_per_grid': nimag_per_grid, 'vec_cryst': rvecs}

    def set_wigner_seitz_cell(self, rvec_images, tau_a, tau_b, eps=1e-6):
        """
        Find R vectors for Wigner-Seitz cell connecting tau_a (origin) and tau_b (lattice vector R)
        
        rvec_images: dict with 'idx_accum_grid', 'nimag_per_grid', 'vec_cryst'
        tau_a, tau_b: (3,) array, Wannier centers in crystal coordinates
        Returns:
            ws_indices: indices in vec_cryst of the selected R-vectors
            (select which R are used for hopping)
            ws_degeneracy: degeneracy for each selected R-vector 
            (tell how much weight selected R has, get rid of overcounting for interpolation)
        """
        vec_cryst = rvec_images['vec_cryst']
        idx = rvec_images['idx_accum_grid']
        nim = rvec_images['nimag_per_grid']
        tot_r = self.kc_dim[0] * self.kc_dim[1] * self.kc_dim[2]
        # the no. potential rvecs, nvec >> the total number of grid points, tot_r
        # only a tiny subset connects two wannier centers
        nvec = vec_cryst.shape[0]

        itmp = np.zeros(nvec, dtype=int)
        ndeg = np.zeros(tot_r, dtype=int)

        # For each grid point
        for ir in range(tot_r):
            idx0 = idx[ir]
            nimg = nim[ir]
            dist = np.zeros(nimg)
            # loop over possible rvecs at this grid point
            for n in range(nimg):
                vec_b = vec_cryst[idx0 + n] + tau_b
                dist[n] = get_length(vec_b - tau_a, self.at)
            # Find rvecs with minimal distances (degenerate)
            dist -= dist.min()
            ndeg[ir] = np.count_nonzero(dist < eps)
            # Mark equivalences in itmp
            for n in range(nimg):
                if dist[n] < eps:
                    itmp[idx0 + n] = ndeg[ir]

        if np.any(ndeg < 1):
            raise ValueError("ndeg < 1 found in set_wigner_seitz_cell")

        ws_indices = np.flatnonzero(itmp > 0)
        ws_degeneracy = itmp[ws_indices]

        return ws_indices, ws_degeneracy

    def get_rvec_set(self):
        """
        Perturbo's set_ws_cell_el to get R vector set in crystal coordinates
        Returns:
        rvec_set: unique R vectors (nrvec, 3) in crystal coordinates
        ham_r_info: info about each matrix element's Wigner-Seitz cell
        """
        nr1, nr2, nr3 = self.kc_dim
        
        # [1] Generate all possible R vector images
        print("Generating R vector images...")
        rvec_images = self.init_rvec_images()
        print(f"Generated {len(rvec_images['vec_cryst'])} R vector images")
        
        # [2] For each matrix element, find its Wigner-Seitz cell
        ham_r_ws_indices = []
        ham_r_info = {}
        
        print("Finding Wigner-Seitz cells for each matrix element...")
        ws_x = 1
        for jb in range(self.num_wann):
            for ib in range(jb + 1):
                ws_indices, ws_degeneracy = self.set_wigner_seitz_cell(
                    rvec_images,
                    self.wannier_center_cryst[ib], 
                    self.wannier_center_cryst[jb]
                )
                ham_r_ws_indices.append(ws_indices)
                with h5py.File(self.epr_file, 'r') as fa:
                    ham_r_x = fa[f"electron_wannier/hopping_r{ws_x}"][()] + 1j * fa[f"electron_wannier/hopping_i{ws_x}"][()]
                ham_r_info[f'H_{ib+1}{jb+1}'] = {
                    "key": f'H_{ib+1}{jb+1}',
                    'hopping_element': ham_r_x,
                    'ib': ib, 'jb': jb,
                    'nr': len(ws_indices),
                    'rvec_indices': ws_indices,
                    'degeneracy': ws_degeneracy
                }
                assert len(ham_r_x) == len(ws_indices), f"hopping_element shape mismatch no. rvecs: {ham_r_x.shape} != {ws_indices.shape}"
                print(f"  H_{ib+1}{jb+1}: {len(ws_indices)} R vectors")
                ws_x += 1
        
        # [3] Collect all unique R vectors
        print("Collecting unique R vectors...")
        all_indices = set()
        for ws_indices in ham_r_ws_indices:
            all_indices.update(ws_indices)
        
        # Sort indices for consistent output
        unique_indices = sorted(all_indices)
        rvec_set = rvec_images['vec_cryst'][unique_indices]
        
        print(f"Final result: {len(rvec_set)} unique R vectors for all matrix wannier hopping pairs")
        
        return rvec_set, ham_r_info

    def extract_eph_in_real_space(self, ham_r_info):
        """
        Extract electron-phonon matrix elements in real space
        Following Perturbo's init_elph_mat_wann and read_elph_mat_wann
        
        Returns:
        eph_data: dict containing:
            - 'matrix_elements': real-space e-ph matrix elements
            - 'rvec_set_el': electron R-vectors  
            - 'rvec_set_ph': phonon R-vectors
            - 'eph_info': information about each matrix element
        """
        print("Extracting electron-phonon matrix elements...")
        
        eph_info = []
        matrix_elements = {}
        
        for ia in range(self.nat):
            for jw in range(self.num_wann):
                for iw in range(self.num_wann):
                    with h5py.File(self.epr_file, 'r') as f:
                        group = f['eph_matrix_wannier']
                        
                        dset_r = f"ep_hop_r_{ia+1}_{jw+1}_{iw+1}"
                        dset_i = f"ep_hop_i_{ia+1}_{jw+1}_{iw+1}"
                        assert dset_r in group and dset_i in group, f"Dataset {dset_r} or {dset_i} not found in H5"
                        
                        r_val = group[dset_r][:]
                        i_val = group[dset_i][:]
                        ep_hop = r_val + 1j * i_val # (nrp, nre, 3)
                        
                        key = (iw, jw, ia)
                        matrix_elements[key] = {
                            'ep_hop': ep_hop
                        }
                        
                        from_key = min(iw, jw)
                        to_key = max(iw, jw)
                        len_re = ham_r_info[f'H_{from_key+1}{to_key+1}']['nr']
                        assert ep_hop.shape[1] == len_re, f"ep_hop shape mismatch: {ep_hop.shape[1]} != {len_re}"
        
        return matrix_elements

    def extract_force_constants(self):
        """
        Extract real-space interatomic force constants (IFCs)
        Following Perturbo's init_lattice_ifc from phonon_dispersion.f90, and force_constant.f90
        
        Returns:
        force_constants: dict containing:
            - 'ifc_data': real-space force constants Φ_ij(R)
            - 'rvec_set_ph': phonon R-vectors
            - 'mass': atomic masses
            - 'ifc_info': information about each force constant matrix
        """
        print("Extracting real-space interatomic force constants...")
        
        mass = self.mass
        atom_pos_cart = self.tau # (nat, 3)
        atom_pos_cryst = self.cryst_to_cart(atom_pos_cart, direction=-1) # (nat, 3)
        rvec_ph_images = self.init_rvec_images(kdim=self.qc_dim)
        
        # Set up Wigner-Seitz cells for each atom pair (ia, ja)
        print("Setting up Wigner-Seitz cells for IFCs...")
        ifc_info = []
        ifc_data = {}
        all_ph_indices = set()
        
        m = 0
        for ja in range(self.nat):
            for ia in range(ja + 1):  # ia <= ja (upper triangular)
                m += 1
                
                # Phonon WS cell: connecting cryst_tau(ia) to cryst_tau(ja)
                ws_ph_indices, ws_ph_degeneracy = self.set_wigner_seitz_cell(
                    rvec_ph_images,
                    atom_pos_cryst[ia],
                    atom_pos_cryst[ja]
                )
                all_ph_indices.update(ws_ph_indices)
                
                with h5py.File(self.epr_file, 'r') as f:
                    group = f['force_constants']
                    dset_name = f"ifc{m}"
                    assert dset_name in group, f"Dataset {dset_name} not found in HDF5"
                    ifc_matrix = group[dset_name][:]  # (nr, 3, 3) complex
                    assert ifc_matrix.shape == (np.prod(self.qc_dim), 3, 3), f"ifc_matrix shape mismatch: {ifc_matrix.shape} != {(np.prod(self.qc_dim), 3, 3)}"
                    key = (ia, ja)
                    ifc_data[key] = {
                        'ifc_matrix': ifc_matrix,  # (nr, 3, 3)
                        'ws_ph_indices': ws_ph_indices,
                        'ws_ph_degeneracy': ws_ph_degeneracy,
                        'nrp': len(ws_ph_indices),
                        'mass_factor': 1.0 / np.sqrt(mass[ia] * mass[ja])
                    }
        
        # Collect unique phonon R-vectors
        unique_ph_indices = sorted(all_ph_indices)
        rvec_set_ph = rvec_ph_images['vec_cryst'][unique_ph_indices]
        
        print(f"Final result: {len(rvec_set_ph)} unique phonon R-vectors")
        print(f"Total force constant matrices: {len(ifc_data)}")
        
        return {
            'ifc_data': ifc_data,
            'rvec_set_ph': rvec_set_ph,
            'mass': mass,
            'ifc_info': ifc_info,
            'atom_pos_cryst': atom_pos_cryst
        }

    def solve_phonon_modes(self, force_constants, qpoint):
        """
        Solve phonon eigenvalue problem at a specific q-point
        Following Perturbo's solve_phonon_modes
        
        Args:
            force_constants: output from extract_force_constants()
            qpoint: q-point in crystal coordinates (3,)
            
        Returns:
            frequencies: phonon frequencies (3*nat,)
            modes: phonon eigenvectors (3*nat, 3*nat) - polarization vectors
        """
        ifc_data = force_constants['ifc_data']
        rvec_set_ph = force_constants['rvec_set_ph']
        mass = force_constants['mass']
        
        nmodes = 3 * self.nat
        
        # [1] Compute phase factors e^(iq·R)
        phase_factors = np.exp(1j * 2 * np.pi * (rvec_set_ph @ qpoint))
        
        # [2] Fourier transform force constants to q-space
        # Create mapping from R-vector indices to phase factors
        rvec_to_phase = {}
        for i, rvec in enumerate(rvec_set_ph):
            rvec_key = tuple(rvec.astype(int))
            rvec_to_phase[rvec_key] = phase_factors[i]
        
        # Initialize dynamical matrix
        dyn_matrix = np.zeros((nmodes, nmodes), dtype=complex)
        
        # [3] Build dynamical matrix
        for (ia, ja), data in ifc_data.items():
            ifc_matrix = data['ifc_matrix']  # (nr, 3, 3)
            ws_ph_indices = data['ws_ph_indices']
            ws_ph_degeneracy = data['ws_ph_degeneracy']
            
            # Fourier transform this force constant
            dmat_q = np.zeros((3, 3), dtype=complex)
            for ir, rvec_idx in enumerate(ws_ph_indices):
                rvec = rvec_set_ph[rvec_idx]
                rvec_key = tuple(rvec.astype(int))
                phase = rvec_to_phase[rvec_key]
                
                # Add contribution with proper degeneracy weighting
                dmat_q += (phase / ws_ph_degeneracy[ir]) * ifc_matrix[ir]
            
            # Mass normalization
            mass_factor = 1.0 / np.sqrt(mass[ia] * mass[ja])
            dmat_q *= mass_factor
            
            # Fill dynamical matrix (both upper and lower triangular)
            for i in range(3):
                for j in range(3):
                    ii = ia * 3 + i
                    jj = ja * 3 + j
                    
                    dyn_matrix[ii, jj] = dmat_q[i, j]
                    if ia != ja:  # Fill symmetric part
                        dyn_matrix[jj, ii] = np.conj(dmat_q[j, i])
        
        # [4] Ensure Hermiticity
        dyn_matrix = (dyn_matrix + dyn_matrix.conj().T) / 2
        
        # [5] Diagonalize
        eigenvalues, eigenvectors = np.linalg.eigh(dyn_matrix)
        
        # [6] Compute frequencies
        frequencies = np.zeros(nmodes)
        for i in range(nmodes):
            if eigenvalues[i] >= 0:
                frequencies[i] = np.sqrt(eigenvalues[i])
            else:
                frequencies[i] = -np.sqrt(-eigenvalues[i])  # Negative for imaginary frequencies
        
        # [7] Normalize eigenvectors by mass
        modes = np.zeros_like(eigenvectors)
        for i in range(nmodes):
            for j in range(nmodes):
                ia = j // 3
                modes[j, i] = eigenvectors[j, i] / np.sqrt(mass[ia])
        
        return frequencies, modes

    def couple_eph_to_phonon_modes(self, eph_data, force_constants, qpoint):
        """
        Transform electron-phonon coupling from atomic displacements to phonon modes
        Following the algorithm in notes.md
        
        Args:
            eph_data: output from extract_eph_in_real_space()
            force_constants: output from extract_force_constants()
            qpoint: q-point in crystal coordinates (3,)
            
        Returns:
            eph_phonon_modes: electron-phonon coupling to phonon modes in mixed representation
        """
        print(f"Coupling electron-phonon matrix elements to phonon modes at q = {qpoint}")
        
        # [1] Solve phonon modes at this q-point
        frequencies, modes = self.solve_phonon_modes(force_constants, qpoint)
        nmodes = len(frequencies)
        
        print(f"  Found {nmodes} phonon modes, {frequencies}")

        # Get phonon R-vectors for Fourier transform
        rvec_set_ph = force_constants['rvec_set_ph']
        nrp = len(rvec_set_ph)
        
        # Compute phase factors for Fourier transform: e^{iq·R_p}
        phase_factors = np.exp(1j * 2 * np.pi * (rvec_set_ph @ qpoint))
        print(f"  Computed phase factors for {nrp} phonon R-vectors")
        
        eph_phonon_modes = {}
        
        for (iw, jw, ia), data in eph_data.items():
            ep_hop = data['ep_hop']  # (nrp, nre, 3) - Python/HDF5 convention
            nrp_data, nre, _ = ep_hop.shape
            
            # Step 2: Transform to phonon mode coordinates
            # g_{ij}^{n}(R_e, R_p, q) = Σ_{α} g_{ij}^{α,a}(R_e, R_p) * u_{α,a}^{n}(q)
            ep_phonon_modes_temp = np.zeros((nmodes, nre, nrp_data), dtype=complex)
            
            for mu in range(nmodes):
                # Get polarization vector for this mode
                e_mu = modes[:, mu]  # (3*nat,)
                
                # Transform: sum over displacement directions for this atom
                for alpha in range(3):
                    mode_component = e_mu[ia * 3 + alpha]  # Polarization for atom ia, direction alpha
                    # ep_hop[:, :, alpha] has shape (nrp, nre)
                    ep_phonon_modes_temp[mu] += mode_component * ep_hop[:, :, alpha].T  # Now (nre, nrp)
            
            # Step 3: Fourier transform phononic part
            # g_{ij}^{n}(R_e, q) = Σ_{R_p} g_{ij}^{n}(R_e, R_p, q) * e^{iq·R_p}
            ep_final = np.zeros((nmodes, nre), dtype=complex)
            
            for mu in range(nmodes):
                # Sum over phonon R-vectors: (nre, nrp) @ (nrp,) -> (nre,)
                ep_final[mu] = ep_phonon_modes_temp[mu] @ phase_factors[:nrp_data]
            
            # Store final result: g_{ij}^n(R, R', q) where R=0, R'=R_e
            eph_phonon_modes[(iw, jw, ia)] = {
                'coupling_matrix': ep_final,  # (nmodes, nre) - final mixed representation
                'frequencies': frequencies,
                'modes': modes,
                'qpoint': qpoint,
                'shape_info': f'({nmodes} modes, {nre} R_e vectors)'
            }
            
            if (iw, jw, ia) == list(eph_data.keys())[0]:  # Print info for first element
                print(f"  Transformed {(iw, jw, ia)}: {ep_hop.shape} -> {ep_final.shape}")
                print(f"    Step 2: (nrp, nre, 3) -> (nmodes, nre, nrp)")  
                print(f"    Step 3: (nmodes, nre, nrp) -> (nmodes, nre) via Fourier transform")
        
        print(f"  Transformed {len(eph_phonon_modes)} electron-phonon matrix elements")
        
        return eph_phonon_modes

    def compute_phonon_dispersion(self, qpath, force_constants):
        """
        Compute phonon dispersion along a q-path
        
        Args:
            qpath: array of q-points (nq, 3) in crystal coordinates
            force_constants: output from extract_force_constants()
            
        Returns:
            frequencies: phonon frequencies (nq, 3*nat)
            modes: phonon eigenvectors (nq, 3*nat, 3*nat)
        """
        nq = len(qpath)
        nmodes = 3 * self.nat
        
        frequencies = np.zeros((nq, nmodes))
        modes = np.zeros((nq, nmodes, nmodes), dtype=complex)
        
        print(f"Computing phonon dispersion for {nq} q-points...")
        for iq, qpoint in enumerate(qpath):
            freq, mode = self.solve_phonon_modes(force_constants, qpoint)
            frequencies[iq] = freq
            modes[iq] = mode
            
            if (iq + 1) % 100 == 0:
                print(f"  Completed {iq + 1}/{nq} q-points")
        
        return frequencies, modes


if __name__ == "__main__":
    
    def extract_hopping(post_qe2pert):
        print("# DNTT System Parameters:")
        print(f"# k-mesh: {post_qe2pert.kc_dim}")
        print(f"# Number of Wannier functions: {post_qe2pert.num_wann}")
        print(f"# Wannier centers:\n{post_qe2pert.wannier_center_cryst}")
        print(f"# Lattice vectors:\n{post_qe2pert.at}")
        print(f"# Lattice constant (Bohr): {post_qe2pert.alat}")
            
        rvec_set_cryst, ham_r_info = post_qe2pert.get_rvec_set()
        print(f"# Number of R vectors: {len(rvec_set_cryst)}")
        print(f"# Number of (ij) matrix elements (upper triangular, H_ij(R)): {len(ham_r_info)}")
        print(f"# R vectors (crystal coordinates):")
        for i, rvec in enumerate(rvec_set_cryst):
            print(f"R{i+1:2d}: [{rvec[0]:6.0f}, {rvec[1]:6.0f}, {rvec[2]:6.0f}]")
        
        rvec_set_cart = post_qe2pert.cryst_to_cart(rvec_set_cryst)
        print(f"# R vectors (Cartesian coordinates, bohr):")
        for i, rvec in enumerate(rvec_set_cart):
            print(f"R{i+1:2d}: [{rvec[0]:8.4f}, {rvec[1]:8.4f}, {rvec[2]:8.4f}]")
            
        print(f"# Matrix element information:")
        for key, info in ham_r_info.items():
            print(f"{info['key']}: {info['nr']} R vectors")
        return ham_r_info