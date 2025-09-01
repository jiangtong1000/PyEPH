import numpy as np
import h5py
from .post_qe2pert import PostQE2Pert
from .phonon_disp import PhononDispersion
from .eph_mat_reciprocal import CalcEphMatReciprocal
from pyeph.utils.logger import setup_logger, get_mpi_info

class CalcEphMatMixed(CalcEphMatReciprocal):
    def __init__(self, epr_file, polar=False, verbose=False):
        super().__init__(epr_file, verbose)
        self.logger = setup_logger("calc_eph_mat_mixed", level="DEBUG" if verbose else "INFO")
        self.phonon_calc = PhononDispersion(epr_file, polar, verbose)
    
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

    def calc_ephmat_mixed(self, qpoints, eps=1e-5):
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
        mpi_info = get_mpi_info()
        rank = mpi_info['rank']
        nprocs = mpi_info['size']
        comm = mpi_info['comm']
        has_mpi = mpi_info['has_mpi']
        
        nq = len(qpoints)
        nmodes = 3 * self.nat
        nre_vec_tot = len(self.rvec_set_el)
        nrph_vec_tot = len(self.rvec_set_ph_eph)
        
        # Distribute q-points across MPI ranks
        if has_mpi and nprocs > 1:
            qpts_per_rank, remainder = divmod(nq, nprocs)
            start = rank * qpts_per_rank + min(rank, remainder)
            end = start + qpts_per_rank + (1 if rank < remainder else 0)
            local_qpoints = qpoints[start:end]
            self.logger.info(f"MPI enabled: {nprocs} ranks processing {nq} q-points total")
            self.logger.info(f"Rank {rank} is processing q-points {start} to {end}")
        else:
            local_qpoints = qpoints
            start, end = 0, nq
            self.logger.info(f"Single process mode: processing {nq} q-points total")
        
        # Input validation
        self.logger.info(f"Computing e-ph matrix in mixed real-reciprocal space for {len(local_qpoints)} local q-points")
        
        local_nq = len(local_qpoints)
        
        # Pre-allocate local arrays
        local_gmat = np.zeros((self.num_wann, self.num_wann, nmodes, nre_vec_tot, local_nq), dtype=np.complex128)
        local_phonon_freqs = np.zeros((nmodes, local_nq), dtype=np.float64)

        # (num_wann, num_wann, natoms, 3, nre_vec_tot, nrph_vec_tot)
        gmat_atoms = self.extract_gmat_raw(nre_vec_tot, nrph_vec_tot)
        
        # these Rq is only for iw <= jw (so we need phase conj for iw>jw later)
        exp_iqr = np.exp(1j * 2.0 * np.pi * self.rvec_set_ph_eph @ local_qpoints.T) # (nrp, local_nq)

        for iq, qpt in enumerate(local_qpoints):
            if self.verbose:
                self.logger.debug(f"Processing local q-point {iq+1}/{local_nq}: {qpt}")
                
            gmat_atoms_q = np.zeros((self.num_wann, self.num_wann, self.nat, 3, nre_vec_tot), dtype=np.complex128)
            
            # Transform the phonon basis (q)
            wqt, mq = self.phonon_calc.solve_phonon_modes(self.force_constants, qpt)
            mq = np.einsum("ij, j->ij", mq, np.sqrt(0.5 / wqt))
            mask = wqt > eps
            local_phonon_freqs[:, iq] = wqt
            mq = mq.reshape((self.nat, 3, nmodes)) # took care of mass denom
            
            phase = exp_iqr[:, iq]
            for iw in range(self.num_wann):
                for jw in range(self.num_wann):
                    if iw <= jw:
                        gmat_atoms_q[iw, jw] = np.einsum("akep, p->ake", gmat_atoms[iw, jw], phase)
                    else:
                        gmat_atoms_q[iw, jw] = np.einsum("akep, p->ake", gmat_atoms[iw, jw], phase.conj())
            
            # (nwan, nwan, nmodes, nre, nq)
            local_gmat[:, :, :, :, iq] = np.einsum("ijake, akn->ijne", gmat_atoms_q, mq, optimize=True)
            
            if (iq + 1) % 100 == 0:
                self.logger.debug(f"Completed {iq + 1}/{local_nq} local q-points")
        
        # Gather results from all ranks
        if has_mpi and nprocs > 1:
            raise NotImplementedError("MPI is not supported for mixed e-ph matrix")
        else:
            gmat = local_gmat
            phonon_freqs = local_phonon_freqs
        
        return gmat, phonon_freqs, self.rvec_set_el, self.rvec_set_ph_eph
    
    def ifft_to_real_space(self, gmat, qpoints):
        """
        gmat (num_wann, num_wann, nmodes, nre_vec_tot, nq)
        qpoints (nq, 3)
        return gmat_real (num_wann, num_wann, nmodes, nre_vec_tot, nph_vec_tot)
        """
        
        nq = len(qpoints)
        nmodes = 3 * self.nat
        nre_vec_tot = len(self.rvec_set_el)
        nrph_vec_tot = len(self.rvec_set_ph_eph)
        exp_iqr = np.exp(-1j * 2.0 * np.pi * self.rvec_set_ph_eph @ qpoints.T) # (nrp, nq)
        gmat_real = np.zeros((self.num_wann, self.num_wann, nmodes, nre_vec_tot, nrph_vec_tot), dtype=np.complex128)
        for iw in range(self.num_wann):
            for jw in range(self.num_wann):
                if iw <= jw:
                    gmat_real[iw, jw, :, :, :] = np.einsum("neq, pq->nep", gmat[iw, jw], exp_iqr)
                else:
                    gmat_real[iw, jw, :, :, :] = np.einsum("neq, pq->nep", gmat[iw, jw], exp_iqr.conj())
        return gmat_real / nq