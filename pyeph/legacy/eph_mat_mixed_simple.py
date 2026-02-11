import numpy as np
from .post_qe2pert import PostQE2Pert
from .phonon_disp import PhononDispersion
from .eph_mat_reciprocal import CalcEphMatReciprocal
from .electron_bands import ElectronBands
from pyeph.utils.logger import setup_logger, get_mpi_info
from pyeph.utils.constants import ryd_to_mev

class CalcEphMatMixed(CalcEphMatReciprocal):
    def __init__(self, epr_file, polar=False, verbose=False):
        PostQE2Pert.__init__(self, epr_file, verbose)
        self.logger = setup_logger("calc_eph_mat_mixed", level="DEBUG" if verbose else "INFO")
        
        # Initialize phonon and electron calculations
        self.phonon_calc = PhononDispersion(epr_file, polar=polar, verbose=verbose)
        self.electron_calc = ElectronBands(epr_file, verbose=verbose)
        
        # Extract e-ph matrix elements and setup
        self.rvec_set_el, self.ham_r_info = self.get_rvec_set()
        self.force_constants = self.phonon_calc.extract_force_constants()
        self.eph_matrix_elements = self.extract_eph_in_real_space_with_ws(self.ham_r_info)
        
        self.logger.info(f"Initialized with {len(self.rvec_set_el)} electron R-vectors, "
                        f"{len(self.rvec_set_ph_eph)} phonon R-vectors")
    
    def extract_gmat_raw(self, nre_vec_tot, nrph_vec_tot):
        """
        Extract raw e-ph matrix elements in real space
        g_{ij}^{a\alpha}(R_e, R_q)
        """
        gmat_raw = np.zeros((self.num_wann, self.num_wann, self.nat, 3, nre_vec_tot, nrph_vec_tot), dtype=np.complex128)
        for key, ephmat in self.eph_matrix_elements.items():
            iw, jw, ia = key
            if iw > jw:
                continue
            ep_hop = ephmat['ep_hop'] # (nrp, nre, 3)
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
            # gmat_raw[iw, jw, ia][:, re_indices[:, None], rp_indices] = ep_hop
            
            # let's do this carefully for now, and we can optimize later
            for re_idx in range(nre):
                for rp_idx in range(nrp):
                    for i in range(3):
                        gmat_raw[iw, jw, ia, i, re_indices[re_idx], rp_indices[rp_idx]] = ep_hop[i, re_idx, rp_idx]
        
        assert np.allclose(gmat_raw.imag, 0)
        gmat_raw = gmat_raw.real
        return gmat_raw

    def calc_ephmat_mixed(self, qpoints, Nx, phfreq_cutoff=1.5/ryd_to_mev):
        """
        Compute e-ph coupling matrix in mixed real-reciprocal space representation with MPI support.
        Electronics in real space (R_e), phonons in reciprocal space (q).
        
        Args:
            qpoints: array of q-points in crystal coordinates (nq, 3)
            
        Returns:
            gmat: e-ph matrix elements with shape (num_wann, num_wann, nmodes, nre_vec_tot, nq)
            phonon_freqs: phonon frequencies with shape (nmodes, nq)
            rvec_set_el: electronic R-vectors
        """
        nq = len(qpoints)
        nmodes = 3 * self.nat
        nre_vec_tot = len(self.rvec_set_el)
        nrph_vec_tot = len(self.rvec_set_ph_eph)
        
        gmat = np.zeros((self.num_wann, self.num_wann, nmodes, nre_vec_tot, nq), dtype=np.complex128)
        import h5py
        with h5py.File(f"eph_data_{Nx}.h5", "r") as fa:
            freqs = fa["freq_full"][:]
            eigvecs = fa["mode_full"][:]

        gmat_atoms = self.extract_gmat_raw(nre_vec_tot, nrph_vec_tot)
        exp_iqr = np.exp(1j * 2.0 * np.pi * self.rvec_set_ph_eph @ qpoints.T) # (nrp, nq)
        
        eph_raw_rot = np.einsum("ijkaep, pq->ijkaeq", gmat_atoms, exp_iqr)
        rotated_eigvecs = np.empty((nq, nmodes, nmodes), dtype=np.complex128)

        for iq in range(nq):
            wqt, mq = freqs[iq].copy(), eigvecs[iq].copy()
            sqrt_mass_per_component = 1.0 / np.sqrt(np.repeat(self.mass, 3))
            mq = np.einsum("ij, i->ij", mq, sqrt_mass_per_component)
            mq[:, wqt < phfreq_cutoff] = 0
            wqt[wqt < phfreq_cutoff] = phfreq_cutoff # these part will never be used anyway
            rotated_eigvecs[iq] = np.einsum("ij, j->ij", mq, np.sqrt(0.5 / wqt))
        
        for iq, qpt in enumerate(qpoints):
            gmat_atoms_q = np.zeros((self.num_wann, self.num_wann, self.nat, 3, nre_vec_tot), dtype=np.complex128)
            
            # Transform the phonon basis (q)
            wqt, mq = freqs[iq], eigvecs[iq]
            sqrt_mass_per_component = 1.0 / np.sqrt(np.repeat(self.mass, 3))
            mq = np.einsum("ij, i->ij", mq, sqrt_mass_per_component)
            mq[:, wqt < phfreq_cutoff] = 0
            wqt[wqt < phfreq_cutoff] = phfreq_cutoff # these part will never be used anyway
            mq = np.einsum("ij, j->ij", mq, np.sqrt(0.5 / wqt))
            assert np.allclose(mq, rotated_eigvecs[iq])
            phase = exp_iqr[:, iq]
            for jw in range(self.num_wann):
                for iw in range(jw + 1):
                    gmat_atoms_q[iw, jw] = np.einsum("akep, p->ake", gmat_atoms[iw, jw], phase)
                    print(gmat_atoms_q[iw, jw].shape, eph_raw_rot[iw, jw, :, :, :, iq].shape)
                    assert np.allclose(gmat_atoms_q[iw, jw], eph_raw_rot[iw, jw, :, :, :, iq])
                    ## else:
                    ## gmat_atoms_q[iw, jw] = np.einsum("akep, p->ake", gmat_atoms[iw, jw], phase.conj())
            gmat_atoms_q = gmat_atoms_q.reshape((self.num_wann, self.num_wann, self.nat * 3, nre_vec_tot))
            gmat[:, :, :, :, iq] = np.einsum("ijme, mn->ijne", gmat_atoms_q, mq, optimize=True)
            
        return gmat, freqs, self.rvec_set_el, self.rvec_set_ph_eph, eph_raw_rot, rotated_eigvecs
    
    def ifft_to_real_space(self, gmat, qpoints, rph):
        """
        gmat (num_wann, num_wann, nmodes, nre_vec_tot, nq)
        qpoints (nq, 3)
        return gmat_real (num_wann, num_wann, nmodes, nre_vec_tot, nph_vec_tot)
        """
        self.logger.info(f"Converting e-ph matrix from reciprocal space to real space")
        self.logger.debug(f"gmat shape: {gmat.shape}")
        self.logger.debug(f"qpoints shape: {qpoints.shape}")
        nmodes = 3 * self.nat
        nre_vec_tot = len(self.rvec_set_el)
        nrph_vec_tot = len(rph)
        exp_iqr = np.exp(-1j * 2.0 * np.pi * rph @ qpoints.T) # (nrp, nq)
        gmat_real = np.zeros((self.num_wann, self.num_wann, nmodes, nre_vec_tot, nrph_vec_tot), dtype=np.complex128)
        
        # gmat = gmat.transpose(0)
        for jw in range(self.num_wann):
            for iw in range(jw + 1):
                gmat_real[iw, jw, :, :, :] = np.einsum("neq, pq->nep", gmat[iw, jw], exp_iqr)
                # else:
                    # gmat_real[iw, jw, :, :, :] = np.einsum("neq, pq->nep", gmat[iw, jw], exp_iqr.conj())
        return gmat_real