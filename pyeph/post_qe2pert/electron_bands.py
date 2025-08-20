import numpy as np
import scipy.linalg
from pyeph.post_qe2pert import PostQE2Pert
from pyeph.post_qe2pert.linalg import unpack_dyn_matrix


class ElectronBands(PostQE2Pert):
    def __init__(self, epr_file, verbose=False):
        super().__init__(epr_file, verbose=verbose)
        self.logger = self.logger.getChild("electron_bands")
        
        # Initialize electron Wannier data
        self.rvec_set, self.ham_r_info = self.get_rvec_set()
        self.nrvec = len(self.rvec_set)
        self.logger.info(f"Initialized with {self.nrvec} R-vectors and {self.num_wann} Wannier functions")
    
    def solve_eigenvalue_vector(self, kpt):
        """
        Args:
            kpt: (3,) array, k-point in crystal coordinates
            
        Returns:
            eigenvalues: (num_wann,) array of band energies
            eigenvectors: (num_wann, num_wann) array of eigenvectors
        """
        exp_ikr = np.exp(1j * 2.0 * np.pi * np.dot(self.rvec_set, kpt))
        hamk = self._fourier_transform_hamiltonian(exp_ikr)
        eigenvalues, eigenvectors = scipy.linalg.eigh(hamk)
        return eigenvalues, eigenvectors
    
    def _fourier_transform_hamiltonian(self, exp_ikr):
        """
        Fourier transform real-space Hamiltonian to k-space
        
        Args:
            exp_ikr: (nrvec,) complex array of phase factors
            
        Returns:
            hamk: (num_wann, num_wann) complex Hamiltonian matrix at k
        """
        nelem = self.num_wann * (self.num_wann + 1) // 2
        hamk_upper = np.zeros(nelem, dtype=np.complex128)
        
        m = 0
        for jb in range(self.num_wann):
            for ib in range(jb + 1):
                # Matrix element index in upper triangular storage
                key = f'H_{ib+1}{jb+1}'
                info = self.ham_r_info[key]
                
                rvec_indices = info['rvec_indices']
                degeneracy = info['degeneracy']
                hopping_element = info['hopping_element']
                
                # Sum over R-vectors for this matrix element
                hamk_element = 0.0 + 0.0j
                for ir, rvec_idx in enumerate(rvec_indices):
                    hamk_element += hopping_element[ir] * exp_ikr[rvec_idx]
                
                hamk_upper[m] = hamk_element
                m += 1
        
        hamk = unpack_dyn_matrix(hamk_upper, self.num_wann)
        
        return hamk
    
    def calc_band_structure(self, kpoints):
        """
        Compute electronic band structure for given k-points
        
        Args:
            kpoints: (nk, 3) array of k-points in crystal coordinates
            
        Returns:
            band_energies: (num_wann, nk) array of band energies in Rydberg
        """
        nk = len(kpoints)
        band_energies = np.zeros((self.num_wann, nk))
        
        self.logger.info(f"Computing band structure for {nk} k-points...")
        
        for ik, kpt in enumerate(kpoints):
            eigenvalues, _ = self.solve_eigenvalue_vector(kpt)
            band_energies[:, ik] = eigenvalues
            
            if (ik + 1) % 100 == 0 or ik == nk - 1:
                self.logger.debug(f"Processed {ik + 1}/{nk} k-points")
        
        return band_energies