# extract eph matrix elements from the ephmat.h5 file

import h5py
import numpy as np
from itertools import product


def get_length(r_cryst, at):
    """
    Calculate length of vector in Cartesian coordinates
    r_cryst: R vector in crystal coordinates (3,)
    at: lattice vectors (3,3) TODO: figure out the axis order of at.
    """
    r_cart = at @ r_cryst
    return np.linalg.norm(r_cart)


def set_cutoff_small(rdim, at):
    """
    Set cutoff distance for Wigner-Seitz cell
    rdim: k-mesh dimensions [nk1, nk2, nk3]
    at: lattice vectors (3,3)
    """
    ndim = np.array(rdim) // 2 + 1
    cutoff = 0.0
    
    for i, j, k in product([-1, 1], repeat=3):
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
            self.kc_dim = f['basic_data/kc_dim'][:]
            self.num_wann = f['basic_data/num_wann'][()]
            self.wannier_center_cryst = f['basic_data/wannier_center_cryst'][:]

    def init_rvec_images(self, ws_search_range=3):
        """
        Initialize all possible R vector images within cutoff
        Returns: list of R vectors in crystal coordinates
        """
        nr1, nr2, nr3 = self.kc_dim
        ws_dim = np.array([nr1, nr2, nr3])
        cutoff = set_cutoff_small(ws_dim, self.at)

        idx_accum_grid = np.zeros(nr1*nr2*nr3, dtype=int)
        nimag_per_grid = np.zeros(nr1*nr2*nr3, dtype=int)
        rvecs_list = []
        tot = 0
        ir = 0
        
        # First pass: count number of vectors per grid point, set up idx and nim
        for k in range(nr3):
            for j in range(nr2):
                for i in range(nr1):
                    ir += 1
                    vec = np.array([i, j, k])
                    nimg = 0
                    # Search
                    for mk in range(-ws_search_range, ws_search_range+1):
                        for mj in range(-ws_search_range, ws_search_range+1):
                            for mi in range(-ws_search_range, ws_search_range+1):
                                rvec = np.array([mi, mj, mk]) * ws_dim + vec
                                if get_length(rvec, self.at) < cutoff:
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
            ws_degeneracy: degeneracy for each selected R-vector
        """
        vec_cryst = rvec_images['vec_cryst']
        idx = rvec_images['idx_accum_grid']
        nim = rvec_images['nimag_per_grid']
        vec_cryst = rvec_images['vec_cryst']
        tot_r = self.kc_dim[0] * self.kc_dim[1] * self.kc_dim[2]
        nvec = vec_cryst.shape[0]

        itmp = np.zeros(nvec, dtype=int)
        ndeg = np.zeros(tot_r, dtype=int)

        # For each grid point
        for ir in range(tot_r):
            idx0 = idx[ir]
            nimg = nim[ir]
            dist = np.zeros(nimg)
            # For each image at this grid point
            for n in range(nimg):
                vec_b = vec_cryst[idx0 + n].astype(float) + tau_b
                dist[n] = get_length(vec_b - tau_a, self.at)
            # Find the minimum distance
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
        Efficient Perturbo's set_ws_cell_el to get R vector set in crystal coordinates
        Returns:
        rvec_set: unique R vectors (nrvec, 3) in crystal coordinates
        ham_r_info: info about each matrix element's Wigner-Seitz cell
        """
        nr1, nr2, nr3 = self.kc_dim
        
        # Step 1: Generate all possible R vector images (vectorized)
        print("Generating R vector images...")
        rvec_images = self.init_rvec_images()
        print(f"Generated {len(rvec_images['vec_cryst'])} R vector images")
        
        # Step 2: For each matrix element, find its Wigner-Seitz cell (vectorized)
        nelem = self.num_wann * (self.num_wann + 1) // 2
        ham_r_ws_indices = []
        ham_r_ws_degeneracy = []
        ham_r_info = []
        
        print("Finding Wigner-Seitz cells for each matrix element...")
        print(self.wannier_center_cryst.shape)
        m = 0
        for jb in range(self.num_wann):
            for ib in range(jb + 1):  # ib <= jb (upper triangular)
                ws_indices, ws_degeneracy = self.set_wigner_seitz_cell(
                    rvec_images,
                    self.wannier_center_cryst[ib], 
                    self.wannier_center_cryst[jb]
                )
                ham_r_ws_indices.append(ws_indices)
                ham_r_ws_degeneracy.append(ws_degeneracy)
                ham_r_info.append({
                    'matrix_element': f'H_{ib+1}{jb+1}',
                    'ib': ib, 'jb': jb,
                    'nr': len(ws_indices),
                    'rvec_indices': ws_indices,
                    'degeneracy': ws_degeneracy
                })
                print(f"  H_{ib+1}{jb+1}: {len(ws_indices)} R vectors")
                m += 1
        
        # Step 3: Collect all unique R vectors (vectorized)
        print("Collecting unique R vectors...")
        all_indices = set()
        for ws_indices in ham_r_ws_indices:
            all_indices.update(ws_indices)
        
        # Sort indices for consistent output
        unique_indices = sorted(all_indices)
        rvec_set = rvec_images['vec_cryst'][unique_indices]
        
        print(f"Final result: {len(rvec_set)} unique R vectors")
        
        return rvec_set, ham_r_info

    def cryst_to_cart(self, rvec_set):
        """
        Convert R vectors from crystal to Cartesian coordinates
        rvec_set: R vectors (nrvec, 3) in crystal coordinates
        Returns: R vectors in Cartesian coordinates
        """
        return np.array([self.at @ rvec for rvec in rvec_set])

    def test_rvec_extraction(self):
        """
        Test the R vector extraction with DNTT data
        """
        print("DNTT System Parameters:")
        print(f"k-mesh: {self.kc_dim}")
        print(f"Number of Wannier functions: {self.num_wann}")
        print(f"Wannier centers:\n{self.wannier_center_cryst.T}")
        print(f"Lattice vectors:\n{self.at}")
        
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
    epr_file = "../7_QE2PERT/DNTT_epr.h5"
    post_qe2pert = PostQE2Pert(epr_file)
    rvec_cryst, rvec_cart, info = post_qe2pert.test_rvec_extraction()