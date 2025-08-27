import numpy as np
import h5py
from .post_qe2pert import PostQE2Pert
from .phonon_disp import PhononDispersion
from .eph_mat_reciprocal import CalcEphMatReciprocal
from pyeph.utils.logger import setup_logger, get_mpi_info

class CalcEphMatMixed(CalcEphMatReciprocal):
    def __init__(self, epr_file, verbose=False):
        super().__init__(epr_file, verbose)
        self.logger = setup_logger("calc_eph_mat_mixed", level="DEBUG" if verbose else "INFO")
        self.phonon_calc = PhononDispersion(epr_file, verbose)
    
    def extract_gmat_raw(self, nre_vec_tot, nrph_vec_tot):
        """
        Extract raw e-ph matrix elements in real space
        g_{ij}^{a\alpha}(R_e, R_q)
        """
        gmat_raw = np.zeros((self.num_wann, self.num_wann, self.nat, 3, nre_vec_tot, nrph_vec_tot), dtype=np.complex128)
        for key, ephmat in self.eph_matrix_elements.items():
            iw, jw, ia = key
            ep_hop = ephmat['ep_hop']
            nrp, nre, _ = ep_hop.shape
            ep_hop = ep_hop.transpose(2, 1, 0) # (3, nre, nrp)
            
            from_key = min(iw, jw)
            to_key = max(iw, jw)
            info = self.ham_r_info[f'H_{from_key+1}{to_key+1}']
            re_indices = info['rvec_indices']
            assert nre == len(re_indices)

            rp_indices = ephmat['ws_ph_indices']
            assert nrp == len(rp_indices)
            
            # this holds: ep_hop for (iw, jw) is conj of ep_hop at (jw, iw)
            gmat_raw[iw, jw, ia][:, re_indices[:, None], rp_indices] = ep_hop
        return gmat_raw

    def calc_ephmat_mixed(self, qpoints):
        """
        Compute e-ph coupling matrix in mixed real-reciprocal space representation.
        Electronics in real space (R_e), phonons in reciprocal space (q).
        
        Args:
            qpoints: array of q-points in crystal coordinates (nq, 3)
            
        Returns:
            gmat: e-ph matrix elements with shape (num_wann, num_wann, nmodes, nre_vec_tot, nq)
        """
        # Input validation
        self.logger.info(f"Computing e-ph matrix in mixed real-reciprocal space for {len(qpoints)} q-points")
        
        nmodes = 3 * self.nat
        nq = len(qpoints)
        nre_vec_tot = len(self.rvec_set_el)
        nrph_vec_tot = len(self.rvec_set_ph_eph)
        
        # Pre-allocate final array
        gmat = np.zeros((self.num_wann, self.num_wann, nmodes, nre_vec_tot, nq), dtype=np.complex128)

        # (num_wann, num_wann, natoms, 3, nre_vec_tot, nrph_vec_tot)
        gmat_atoms = self.extract_gmat_raw(nre_vec_tot, nrph_vec_tot)
        phonon_freqs = np.zeros((nmodes, nq), dtype=np.float64)
        
        # these Rq is only for iw <= jw (so we need phase conj for iw>jw later)
        exp_iqr = np.exp(1j * 2.0 * np.pi * self.rvec_set_ph_eph @ qpoints.T) # (nrp, nq)

        for iq, qpt in enumerate(qpoints):
            if self.verbose:
                self.logger.debug(f"Processing q-point {iq+1}/{nq}: {qpt}")
            
            # Transform the phonon basis (q)
            wqt, mq = self.phonon_calc.solve_phonon_modes(self.force_constants, qpt)
            phonon_freqs[:, iq] = wqt
            mq = mq.reshape((self.nat, 3, nmodes)) # took care of mass denom
            # (nwan, nwan, nmodes, nre, nrp)
            gmat_modes = np.einsum("ijakep, akn->ijnep", gmat_atoms, mq, optimize=True)

            # Fourier transform phononic part
            phase = exp_iqr[:, iq]
            for iw in range(self.num_wann):
                for jw in range(self.num_wann):
                    if iw <= jw:
                        gmat[iw, jw, :, :, iq] = np.einsum("nep,p->ne", gmat_modes[iw, jw], phase)
                    else:
                        gmat[iw, jw, :, :, iq] = np.einsum("nep,p->ne", gmat_modes[iw, jw], phase.conj())
            
            gmat = np.einsum("ijnep, n->ijnep", gmat, np.sqrt(0.5 / wqt))
        return gmat, phonon_freqs, self.rvec_set_el