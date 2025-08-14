# extract eph matrix elements from the ephmat.h5 file

import h5py
import numpy as np
from itertools import product
import scipy.linalg

def unpack_dyn_matrix(dyn_upper, nmodes):
    dyn_matrix = np.zeros((nmodes, nmodes), dtype=np.complex128)
    idx = 0
    for j in range(nmodes):        # Column index (0-based)
        for i in range(j + 1):   # Row index, upper triangular (0-based)
            dyn_matrix[i, j] = dyn_upper[idx]
            if i != j:
                dyn_matrix[j, i] = np.conj(dyn_upper[idx])
            idx += 1
    return dyn_matrix

def get_length(r_cryst, at):
    r_cryst = np.asarray(r_cryst, dtype=np.float64)
    at = np.asarray(at, dtype=np.float64)
    r_cart = r_cryst @ at
    return np.sqrt(np.einsum('...i,...i->...', r_cart, r_cart))


def set_cutoff_small(rdim, at):
    """
    Retrun the cutoff radius in real space for Wigner-Seitz cell vector search
    (the edge of the first Brillouin zone in reciprocal space)
    Define a sphere in real space containing all R-vectors within approx half of the Brillouin zone
    This ensures we capture all relevant R-vectors for Wannier hopping while avoiding unnecessary computations
    for R-vectors far away from the Brillouin zone.
    rdim: k-mesh dimensions [nk1, nk2, nk3]
    at: lattice vectors (3,3)
    """
    ndim = np.array(rdim) // 2 + 1 # half width of the k-mesh plus 1
    cutoff = 0.0
    
    for i, j, k in product([-1, 1], repeat=3): # corner points of the k-mesh
        r_cryst = ndim * np.array([i, j, k], dtype=np.float64)
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
            self.kc_dim = f['basic_data/kc_dim'][:] # number of kc points in each direction
            self.qc_dim = f['basic_data/qc_dim'][:]
            self.num_wann = f['basic_data/num_wann'][()]
            self.nat = f['basic_data/nat'][()]
            self.wannier_center_cryst = f['basic_data/wannier_center_cryst'][:]
            self.tau = f['basic_data/tau'][:]  # atomic positions in cart coordinates (unit of alat). (nat, 3)
            self.bg = f['basic_data/bg'][:]    # reciprocal lattice vectors in unit of 2pi / alat
            self.mass = f['basic_data/mass'][:]  # real, (nat,) atomic masses in atomic unit
            assert self.wannier_center_cryst.shape == (self.num_wann, 3), f"wannier_center_cryst shape mismatch: {self.wannier_center_cryst.shape} != ({self.num_wann}, 3)"

    def convert_coordinates(self, positions, direction='crys_to_cart'):
        """
        Convert between crystal and Cartesian coordinates
        positions: array of positions (nvec or nat, 3)
        direction: 'crys_to_cart' or 'cart_to_crys'
        """
        if direction == 'crys_to_cart':
            # Crystal to Cartesian: r_cart = at.T @ r_cryst
            return np.array([self.at.T @ pos for pos in positions])
        elif direction == 'cart_to_crys':
            # Cartesian to Crystal: r_cryst = bg.T @ r_cart # bg.T here is inverse of at.T
            return np.array([self.bg.T @ pos for pos in positions])
        else:
            raise ValueError(f"Invalid direction: {direction}")

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
            idx_accum_grid[ir] = tot
            base_vec = np.array([i, j, k])
            nimg = 0
            for (mk, mj, mi) in product(range(-ws_search_range, ws_search_range+1), repeat=3):
                # supercell translations of base_vec, all physically equivalent positions due to PBC
                # images are lattice vectors in supercell, just physical copies of base_vec 
                # base_vec is the primitive index of the supercell
                rvec = np.array([mi, mj, mk]) * ws_dim + base_vec
                if get_length(rvec, self.at) < cutoff:
                    # a point distance from origin will also not contribute to any hopping
                    # since the hopping relies on |R+tau_a - tau_b|
                    nimg += 1
                    rvecs_list.append(rvec)
                            
            if nimg < 1:
                raise ValueError('nim < 1 for grid point ({},{},{})'.format(i, j, k))
            nimag_per_grid[ir] = nimg
            tot += nimg
        
        rvecs = np.array(rvecs_list).reshape(tot, 3)
        return {
            'idx_accum_grid': idx_accum_grid, # (nr1*nr2*nr3,)
            'nimag_per_grid': nimag_per_grid, # (nr1*nr2*nr3,)
            'vec_cryst': rvecs # (tot, 3)
            }

    def set_wigner_seitz_cell(self, kdim, rvec_images, tau_a, tau_b, eps=1e-6):
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
        
        tot_r = int(np.prod(kdim))
        nvec  = vec_cryst.shape[0]
        itmp = np.zeros(nvec, dtype=int) # per-image degeneracy (0 if not selected)
        
        dtau = (tau_b - tau_a)
    
        for ir in range(tot_r):
            start = idx[ir]
            nimg  = nim[ir]
            stop  = start + nimg

            # Block of candidate R (crystal coords), shift by tau_b - tau_a
            R_block_cryst = vec_cryst[start:stop] + dtau  # (nimg, 3)
            dist = get_length(R_block_cryst, self.at)  # (nimg,)

            # Find degenerate images with minimal distance
            dmin = dist.min()
            sel  = np.isclose(dist, dmin, atol=eps, rtol=0) # (nimg,)

            g    = int(sel.sum())
            if g < 1:
                raise ValueError(f"ndeg < 1 at grid point {ir}")

            # Write degeneracy for selected images in this block with mask sel
            itmp[start:stop][sel] = g

        ws_indices    = np.flatnonzero(itmp)
        ws_degeneracy = itmp[ws_indices]
        return ws_indices, ws_degeneracy

    def extract_force_constants(self):
        print("Extracting real-space interatomic force constants...")
        
        mass = self.mass
        atom_pos_cart = self.tau # (nat, 3)
        atom_pos_cryst = self.convert_coordinates(atom_pos_cart, direction='cart_to_crys') # (nat, 3)
        rvec_ph_images = self.init_rvec_images(kdim=self.qc_dim)
        
        print("Setting up Wigner-Seitz cells for IFCs...")
        ws_cells = {}
        unique_ph_indices = set()

        for ja in range(self.nat):
            for ia in range(ja + 1):
                ws_ph_indices, _ = self.set_wigner_seitz_cell(
                    self.qc_dim, rvec_ph_images, atom_pos_cryst[ia], atom_pos_cryst[ja]
                )
                ws_cells[(ia, ja)] = ws_ph_indices
                unique_ph_indices.update(ws_ph_indices)

        # Create compact R-vector set and remapping (for memory efficiency)
        unique_ph_indices = sorted(unique_ph_indices)
        rvec_set_ph = rvec_ph_images['vec_cryst'][unique_ph_indices]
        index_mapping = {orig_idx: new_idx for new_idx, orig_idx in enumerate(unique_ph_indices)}

        # Update WS indices to compact indices
        for key in ws_cells:
            old_indices = ws_cells[key]
            new_indices = np.array([index_mapping[idx] for idx in old_indices])
            ws_cells[key] = new_indices

        # Read IFC data
        m = 0
        ifc_data = {}
        with h5py.File(self.epr_file, 'r') as f:
            group = f['force_constant']
            for ja in range(self.nat):
                for ia in range(ja + 1):
                    m += 1
                    ws_ph_indices = ws_cells[(ia, ja)]
                    nr = len(ws_ph_indices)

                    dset_name = f"ifc{m}"
                    ifc_matrix = group[dset_name][:]
                    assert ifc_matrix.shape == (nr, 3, 3), f"ifc_matrix shape mismatch: {ifc_matrix.shape} != ({nr}, 3, 3)"
                    ifc_matrix = ifc_matrix.transpose(0, 2, 1) # to match Perturbo's column-major convention
                    # ifc_matrix = (ifc_matrix + ifc_matrix.transpose(0, 2, 1)) * 0.5 # This makes things worse
                
                    ifc_data[(ia, ja)] = {
                        'ifc_matrix': ifc_matrix,
                        'ws_ph_indices': ws_ph_indices,
                        'nrp': len(ws_ph_indices),
                        'mass_factor': 1.0 / np.sqrt(mass[ia] * mass[ja])
                    }

        return {
            'ifc_data': ifc_data,
            'rvec_set_ph': rvec_set_ph,
            'mass': mass,
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
        masses = force_constants['mass']
        
        nmodes = 3 * self.nat
        phase_factors = np.exp(1j * 2 * np.pi * (rvec_set_ph @ qpoint))
        
        # Build dynamical matrix blocks
        num_atom_pairs = self.nat * (self.nat + 1) // 2
        dmat = np.zeros((3, 3, num_atom_pairs), dtype=np.complex128)
        
        pair_idx = 0
        for ja in range(self.nat):
            for ia in range(ja + 1):
                ifc_matrix = ifc_data[(ia, ja)]['ifc_matrix']
                ws_ph_indices = ifc_data[(ia, ja)]['ws_ph_indices']
                mass_factor = 1.0 / np.sqrt(masses[ia] * masses[ja])
                
                for ir, rvec_idx in enumerate(ws_ph_indices):
                    phase = phase_factors[rvec_idx]
                    dmat[:, :, pair_idx] += phase * ifc_matrix[ir] * mass_factor
                pair_idx += 1
                
        # Pack upper triangular dynamical matrix
        num_elements = nmodes * (nmodes + 1) // 2
        dyn_upper = np.zeros(num_elements, dtype=np.complex128)
        
        for jj in range(nmodes):
            for ii in range(jj + 1):
                elem_idx = (jj * (jj + 1)) // 2 + ii
                
                ia, i = divmod(ii, 3)
                ja, j = divmod(jj, 3)
                
                pair_idx = (ja * (ja + 1)) // 2 + ia
                
                if ia != ja:
                    dyn_upper[elem_idx] = dmat[i, j, pair_idx]
                else:
                    dyn_upper[elem_idx] = (dmat[i, j, pair_idx] + np.conj(dmat[j, i, pair_idx])) * 0.5
        
        # Diagonalize dynamical matrix
        dyn_matrix = unpack_dyn_matrix(dyn_upper, nmodes)
        eigenvalues, eigenvectors = scipy.linalg.eigh(dyn_matrix)
        
        # Compute frequencies with proper handling of negative eigenvalues
        frequencies = np.sign(eigenvalues) * np.sqrt(np.abs(eigenvalues))
        
        # Normalize eigenvectors by mass
        mass_sqrt_inv = np.repeat(1.0 / np.sqrt(masses), 3)
        modes = eigenvectors * mass_sqrt_inv[:, np.newaxis]
        
        return frequencies, modes

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
                    self.kc_dim,
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
        
        # [3] Collect all unique R vectors and create remapping
        print("Collecting unique R vectors...")
        all_indices = set()
        for ws_indices in ham_r_ws_indices:
            all_indices.update(ws_indices)
        
        # Sort indices for consistent output
        unique_indices = sorted(all_indices)
        rvec_set = rvec_images['vec_cryst'][unique_indices]
        
        # Create mapping from original indices to compact indices
        # Following Fortran's setup_rvec_set_el logic
        index_mapping = {}
        for new_idx, orig_idx in enumerate(unique_indices):
            index_mapping[orig_idx] = new_idx
        
        # Update matrix element indices to point into compact rvec_set
        # This is crucial - matches Fortran line 146: ptr%ws_el%rvec(ir) = rvec_label(ire)
        for key, info in ham_r_info.items():
            old_indices = info['rvec_indices']
            new_indices = np.array([index_mapping[idx] for idx in old_indices])
            info['rvec_indices'] = new_indices
        
        print(f"Final result: {len(rvec_set)} unique R vectors for all matrix wannier hopping pairs")
        print(f"Updated matrix element indices to point into compact rvec_set")
        
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
        
        # Compute phase factors for Fourier transform: e^{iqR}, q (3, ), R (nrp, 3)
        phase_factors = np.exp(1j * 2 * np.pi * (rvec_set_ph @ qpoint))
        print(f"  Computed phase factors for {nrp} phonon R-vectors")
        
        eph_phonon_modes = {}
        
        for (iw, jw, ia), data in eph_data.items():
            ep_hop = data['ep_hop']  # (nrp, nre, 3) - Python/HDF5 convention
            nrp_data, nre, _ = ep_hop.shape
            
            # Step 2: Transform to phonon mode coordinates
            # g_{ij}^{n}(R_e, R_p, q) = Σ_{α} g_{ij}^{α,a}(R_e, R_p) * u_{α,a}^{n}(q)
            ep_phonon_modes_temp = np.zeros((nmodes, nre, nrp_data), dtype=np.complex128)
            
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
            ep_final = np.zeros((nmodes, nre), dtype=np.complex128)
            
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
        modes = np.zeros((nq, nmodes, nmodes), dtype=np.complex128)
        
        print(f"Computing phonon dispersion for {nq} q-points...")
        for iq, qpoint in enumerate(qpath):
            freq, mode = self.solve_phonon_modes(force_constants, qpoint)
            frequencies[iq] = freq
            modes[iq] = mode
            
            if (iq + 1) % 100 == 0:
                print(f"  Completed {iq + 1}/{nq} q-points")
        
        return frequencies, modes
