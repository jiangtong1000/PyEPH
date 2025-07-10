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
            self.num_wann = f['basic_data/num_wann'][()]
            self.wannier_center_cryst = f['basic_data/wannier_center_cryst'][:]
            assert self.wannier_center_cryst.shape == (self.num_wann, 3), f"wannier_center_cryst shape mismatch: {self.wannier_center_cryst.shape} != ({self.num_wann}, 3)"

    def init_rvec_images(self, ws_search_range=3):
        """
        Initialize all possible R vectors (in crystal coordinates)
        Returns: 
        idx_accum_grid: index of the first R vector in vec_cryst for
        nimag_per_grid: number of R vectors in vec_cryst for each grid point
        vec_cryst: union of all R vectors
        """
        nr1, nr2, nr3 = self.kc_dim
        ws_dim = np.array([nr1, nr2, nr3])
        cutoff = set_cutoff_small(ws_dim, self.at)

        idx_accum_grid = np.zeros(nr1*nr2*nr3, dtype=int)
        nimag_per_grid = np.zeros(nr1*nr2*nr3, dtype=int)
        rvecs_list = []
        tot = 0
        ir = 0

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
            idx_accum_grid[ir-1] = tot
            nimag_per_grid[ir-1] = nimg
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
        ham_r_ws_degeneracy = []
        ham_r_info = []
        
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
                ham_r_ws_degeneracy.append(ws_degeneracy)
                with h5py.File(self.epr_file, 'r') as fa:
                    ham_r_x = fa[f"electron_wannier/hopping_r{ws_x}"][()] + 1j * fa[f"electron_wannier/hopping_i{ws_x}"][()]
                ham_r_info.append({
                    'hopping_key': f'H_{ib+1}{jb+1}',
                    'hopping_element': ham_r_x,
                    'ib': ib, 'jb': jb,
                    'nr': len(ws_indices),
                    'rvec_indices': ws_indices,
                    'degeneracy': ws_degeneracy
                })
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

    def cryst_to_cart(self, rvec_set):
        """
        Convert R vectors from crystal to Cartesian coordinates
        rvec_set: R vectors (nrvec, 3) in crystal coordinates
        Returns: R vectors in Cartesian coordinates
        """
        return np.array([self.at.T @ rvec for rvec in rvec_set])

    def test_rvec_extraction(self):
        """
        Test the R vector extraction with DNTT data
        """
        print("DNTT System Parameters:")
        print(f"k-mesh: {self.kc_dim}")
        print(f"Number of Wannier functions: {self.num_wann}")
        print(f"Wannier centers:\n{self.wannier_center_cryst}")
        print(f"Lattice vectors:\n{self.at}")
        print(f"Lattice constant: {self.alat}")
        
        # Get R vectors
        rvec_set_cryst, ham_r_info = self.get_rvec_set()
        
        # Convert to Cartesian
        rvec_set_cart = self.cryst_to_cart(rvec_set_cryst)
        
        print("\nR vectors (crystal coordinates):")
        for i, rvec in enumerate(rvec_set_cryst):
            print(f"R{i+1:2d}: [{rvec[0]:6.0f}, {rvec[1]:6.0f}, {rvec[2]:6.0f}]")
        
        print("\nR vectors (Cartesian coordinates, bohr):")
        for i, rvec in enumerate(rvec_set_cart):
            print(f"R{i+1:2d}: [{rvec[0]:8.4f}, {rvec[1]:8.4f}, {rvec[2]:8.4f}]")
        
        print("\nMatrix element information:")
        for info in ham_r_info:
            print(f"{info['matrix_element']}: {info['nr']} R vectors")
        
        return rvec_set_cryst, rvec_set_cart, ham_r_info


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
        for info in ham_r_info:
            print(f"{info['hopping_key']}: {info['nr']} R vectors")
    
    epr_file = "DNTT_epr.h5"
    post_qe2pert = PostQE2Pert(epr_file)
    extract_hopping(post_qe2pert)