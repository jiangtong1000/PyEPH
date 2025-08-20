# extract eph matrix elements from the ephmat.h5 file

import h5py
import numpy as np
from itertools import product
import scipy.linalg

from pyeph.post_qe2pert.linalg import unpack_dyn_matrix
from pyeph.post_qe2pert.utils import get_length, set_cutoff_small
from pyeph.utils.logger import setup_logger


class PostQE2Pert():
    def __init__(self, epr_file, verbose=False):
        self.epr_file = epr_file
        self.verbose = verbose
        self.logger = setup_logger("post_qe2pert", level="DEBUG" if verbose else "INFO")
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
            self.bg = f['basic_data/bg'][:]    # reciprocal lattice vectors in unit of 2pi / alat (3, 3) <-> (spatial, reciprocal)
            self.mass = f['basic_data/mass'][:]  # real, (nat,) atomic masses in atomic unit
            self.volume = f['basic_data/volume'][()]  # unit cell volume
            self.tpiba = 2.0 * np.pi / self.alat
            
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
            return np.array([self.bg @ pos for pos in positions]) # Very careful
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
            sel = dist - dmin < eps

            g    = int(sel.sum())
            if g < 1:
                raise ValueError(f"ndeg < 1 at grid point {ir}")

            # Write degeneracy for selected images in this block with mask sel
            itmp[start:stop][sel] = g

        ws_indices    = np.flatnonzero(itmp)
        ws_degeneracy = itmp[ws_indices]
        return ws_indices, ws_degeneracy

    def get_rvec_set(self):
        """
        TODO: Tong: this part has some redundancy with the phonon dispersion code
        Perturbo's set_ws_cell_el to get R vector set in crystal coordinates
        Returns:
        rvec_set: unique R vectors (nrvec, 3) in crystal coordinates
        ham_r_info: info about each matrix element's Wigner-Seitz cell
        """
        nr1, nr2, nr3 = self.kc_dim
        
        # [1] Generate all possible R vector images
        self.logger.debug("Generating R vector images...")
        rvec_images = self.init_rvec_images()
        self.logger.debug(f"Generated {len(rvec_images['vec_cryst'])} R vector images")
        
        # [2] For each matrix element, find its Wigner-Seitz cell
        ham_r_ws_indices = []
        ham_r_info = {}
        
        self.logger.debug("Finding Wigner-Seitz cells for each matrix element...")
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
                self.logger.debug(f"H_{ib+1}{jb+1}: {len(ws_indices)} R vectors")
                ws_x += 1
        
        # [3] Collect all unique R vectors and create remapping
        self.logger.debug("Collecting unique R vectors...")
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
        
        self.logger.info(f"Final result: {len(rvec_set)} unique R vectors for all matrix wannier hopping pairs")
        self.logger.debug("Updated matrix element indices to point into compact rvec_set")
        
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
        self.logger.info("Extracting electron-phonon matrix elements...")
        
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