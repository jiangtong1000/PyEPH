import numpy as np
import h5py
from .post_qe2pert import PostQE2Pert
from .phonon_disp import PhononDispersion
from pyeph.utils.logger import setup_logger

class CalcEphMat(PostQE2Pert):
    def __init__(self, epr_file, verbose=False):
        super().__init__(epr_file, verbose)
        self.logger = setup_logger("calc_eph_mat", level="DEBUG" if verbose else "INFO")
        self.phonon_calc = PhononDispersion(epr_file, verbose)

    def transform_to_phonon_mode_basis(self, matrix_elements, qpoints, force_constants):
        """
        Step 2: Transform to Phonon Mode Basis
        Apply phonon eigenvectors to convert from atomic displacements to phonon modes:
        g_ij^n(Re, Rp, q) = sum_{a,alpha} g_ij^{alpha,a}(Re, Rp) * u_{alpha,a}^n(q)
        
        Args:
            matrix_elements: output from extract_eph_in_real_space()
            qpoints: q-points in crystal coordinates (nq, 3)
            force_constants: output from phonon_calc.extract_force_constants()
            
        Returns:
            g_mode_data: dict with transformed matrix elements
        """
        self.logger.info("Step 2: Transforming to phonon mode basis...")
        
        nq = len(qpoints)
        nmodes = 3 * self.nat
        g_mode_data = {}
        
        for iq, qpoint in enumerate(qpoints):
            if self.verbose:
                self.logger.debug(f"Processing q-point {iq+1}/{nq}: {qpoint}")
            
            # Solve phonon modes at this q-point
            frequencies, modes = self.phonon_calc.solve_phonon_modes(force_constants, qpoint)
            
            # Transform each (iw, jw, ia) combination
            for key, data in matrix_elements.items():
                iw, jw, ia = key
                ep_hop = data['ep_hop']  # Shape: (nrp, nre, 3)
                nrp, nre, _ = ep_hop.shape
                
                # Initialize mode-transformed matrix
                g_mode = np.zeros((nrp, nre, nmodes), dtype=np.complex128)
                
                # Apply transformation: g_ij^n = sum_{alpha} g_ij^{alpha,a} * u_{alpha,a}^n
                for n in range(nmodes):  # phonon mode index
                    for alpha in range(3):  # Cartesian direction
                        # Extract phonon eigenvector component for this atom and direction
                        mode_component = modes[ia*3 + alpha, n]
                        
                        # Add contribution to mode n
                        g_mode[:, :, n] += ep_hop[:, :, alpha] * mode_component
                
                # Store transformed matrix
                result_key = (iq, iw, jw, ia)
                g_mode_data[result_key] = {
                    'g_mode': g_mode,
                    'qpoint': qpoint,
                    'frequencies': frequencies,
                    'modes': modes
                }
        
        self.logger.info(f"Transformed to phonon mode basis for {len(g_mode_data)} combinations")
        return g_mode_data

    def fourier_transform_phononic_part(self, g_mode_data, rvec_set_ph):
        """
        Step 3: Fourier Transform Phononic Part
        Convert from real-space phonon coordinates to reciprocal space:
        g_ij^n(Re, q) = sum_Rp g_ij^n(Re, Rp, q) * exp(i*q*Rp)
        
        Args:
            g_mode_data: output from transform_to_phonon_mode_basis()
            rvec_set_ph: phononic R-vectors from force_constants
            
        Returns:
            g_final_data: dict with final transformed matrix elements
        """
        self.logger.info("Step 3: Fourier transforming phononic part...")
        
        g_final_data = {}
        
        # Group by (iw, jw, ia) and q-point
        grouped_data = {}
        for key, data in g_mode_data.items():
            iq, iw, jw, ia = key
            qpoint = tuple(data['qpoint'])
            group_key = (iw, jw, ia, qpoint)
            
            if group_key not in grouped_data:
                grouped_data[group_key] = data
        
        for (iw, jw, ia, qpoint_tuple), data in grouped_data.items():
            qpoint = np.array(qpoint_tuple)
            g_mode = data['g_mode']  # Shape: (nrp, nre, nmodes)
            frequencies = data['frequencies']
            modes = data['modes']
            
            nrp, nre, nmodes = g_mode.shape
            
            # Compute Fourier transform: g_ij^n(Re, q) = sum_Rp g_ij^n(Re, Rp, q) * exp(i*q*Rp)
            g_final = np.zeros((nre, nmodes), dtype=np.complex128)
            
            for ire in range(nre):
                for n in range(nmodes):
                    # Sum over phononic R-vectors
                    for irp in range(nrp):
                        if irp < len(rvec_set_ph):
                            rp_vec = rvec_set_ph[irp]
                            phase = np.exp(1j * 2 * np.pi * np.dot(qpoint, rp_vec))
                            g_final[ire, n] += g_mode[irp, ire, n] * phase
            
            # Store final result
            result_key = (iw, jw, ia, tuple(qpoint))
            g_final_data[result_key] = {
                'g_final': g_final,
                'qpoint': qpoint,
                'frequencies': frequencies,
                'modes': modes
            }
        
        self.logger.info(f"Fourier transformed {len(g_final_data)} matrix elements")
        return g_final_data

    def create_mixed_representation(self, g_final_data, rvec_set_el):
        """
        Step 4: Final Mixed Representation
        The result gives the desired form: g_ij^n(R, R', q) = g_ij^n(Re, q) where R = 0, R' = Re
        
        Args:
            g_final_data: output from fourier_transform_phononic_part()
            rvec_set_el: electronic R-vectors
            
        Returns:
            eph_coupling: final electron-phonon coupling matrix in mixed representation
        """
        self.logger.info("Step 4: Creating final mixed representation...")
        
        eph_coupling = {}
        
        for key, data in g_final_data.items():
            iw, jw, ia, qpoint_tuple = key
            g_final = data['g_final']
            qpoint = data['qpoint']
            frequencies = data['frequencies']
            modes = data['modes']
            
            nre, nmodes = g_final.shape
            
            # Create mixed representation: g_ij^n(R=0, R'=Re, q)
            mixed_g = np.zeros((len(rvec_set_el), nmodes), dtype=np.complex128)
            
            # Map electronic R-vectors
            for ire in range(min(nre, len(rvec_set_el))):
                for n in range(nmodes):
                    mixed_g[ire, n] = g_final[ire, n]
            
            # Store in final format
            final_key = (iw, jw, tuple(qpoint))
            if final_key not in eph_coupling:
                eph_coupling[final_key] = {}
            
            eph_coupling[final_key][ia] = {
                'matrix': mixed_g,
                'frequencies': frequencies,
                'modes': modes,
                'rvec_set_el': rvec_set_el
            }
        
        self.logger.info(f"Created mixed representation for {len(eph_coupling)} (orbital, q-point) combinations")
        return eph_coupling

    def compute_eph_coupling_matrix(self, qpoints):
        """
        Main function to compute electron-phonon coupling matrix in mixed real-reciprocal space
        Following the complete transformation algorithm from notes.md
        
        Args:
            qpoints: q-points in crystal coordinates (nq, 3)
            
        Returns:
            eph_coupling: electron-phonon coupling matrix in desired mixed representation
        """
        self.logger.info("Computing electron-phonon coupling matrix in mixed representation...")
        
        # Extract force constants for phonon calculations
        force_constants = self.phonon_calc.extract_force_constants()
        rvec_set_ph = force_constants['rvec_set_ph']
        
        # Get electronic R-vectors
        rvec_set_el, ham_r_info = self.get_rvec_set()
        
        # Step 1: Extract ep_hop from epr.h5
        matrix_elements = self.extract_eph_in_real_space(ham_r_info)
        
        # Step 2: Transform to phonon mode basis
        g_mode_data = self.transform_to_phonon_mode_basis(matrix_elements, qpoints, force_constants)
        
        # Step 3: Fourier transform phononic part
        g_final_data = self.fourier_transform_phononic_part(g_mode_data, rvec_set_ph)
        
        # Step 4: Create final mixed representation
        eph_coupling = self.create_mixed_representation(g_final_data, rvec_set_el)
        
        self.logger.info("Successfully computed electron-phonon coupling matrix!")
        return eph_coupling