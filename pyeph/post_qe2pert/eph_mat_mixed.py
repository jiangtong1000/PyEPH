import numpy as np
import h5py
from .post_qe2pert import PostQE2Pert
from .phonon_disp import PhononDispersion
from .eph_mat_reciprocal import CalcEphMatReciprocal
from .electron_bands import ElectronBands
from pyeph.utils.logger import setup_logger, get_mpi_info

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
        self.logger.info(f"Extracting raw e-ph matrix elements for {nre_vec_tot} electronic and {nrph_vec_tot} phononic R-vectors")
        gmat_atoms = self.extract_gmat_raw(nre_vec_tot, nrph_vec_tot)
        self.logger.debug(f"Raw gmat_atoms shape: {gmat_atoms.shape}")
        
        # these Rq is only for iw <= jw (so we need phase conj for iw>jw later)
        self.logger.debug(f"Computing phase factors for {local_nq} local q-points")
        exp_iqr = np.exp(1j * 2.0 * np.pi * self.rvec_set_ph_eph @ local_qpoints.T) # (nrp, local_nq)

        for iq, qpt in enumerate(local_qpoints):
            if self.verbose:
                self.logger.debug(f"Processing local q-point {iq+1}/{local_nq}: {qpt}")
                
            gmat_atoms_q = np.zeros((self.num_wann, self.num_wann, self.nat, 3, nre_vec_tot), dtype=np.complex128)
            
            # Transform the phonon basis (q)
            self.logger.debug(f"Solving phonon modes for q-point {iq+1} / {len(local_qpoints)}: {qpt}")
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
            gmat = self._gather_mpi_results(local_gmat, (self.num_wann, self.num_wann, nmodes, nre_vec_tot, nq), comm, rank, nprocs, is_complex=True)
            phonon_freqs = self._gather_mpi_results(local_phonon_freqs, (nmodes, nq), comm, rank, nprocs, is_complex=False)
            if rank != 0:
                # Non-root ranks don't have complete data
                return None, None, None, None
        else:
            gmat = local_gmat
            phonon_freqs = local_phonon_freqs
        
        return gmat, phonon_freqs, self.rvec_set_el, self.rvec_set_ph_eph
    
    def _gather_mpi_results(self, local_data, global_shape, comm, rank, nprocs, is_complex=False):
        """
        Gather results from all MPI ranks for mixed e-ph matrix calculations.
        
        Args:
            local_data: Local data array from current rank
            global_shape: Shape of the global result array  
            comm: MPI communicator
            rank: Current MPI rank
            nprocs: Number of MPI processes
            is_complex: Whether data is complex
            
        Returns:
            Gathered data array on rank 0, local_data on other ranks
        """
        if rank == 0:
            # Allocate global array
            dtype = local_data.dtype if is_complex else np.float64
            global_data = np.zeros(global_shape, dtype=dtype)
            
            # Determine q-point distribution
            nq = global_shape[-1]  # q-points are always the last dimension
            
            # Place rank 0's data first
            qpts_per_rank, remainder = divmod(nq, nprocs)
            start = 0
            end = qpts_per_rank + (1 if remainder > 0 else 0)
            global_data[..., start:end] = local_data
            
            # Receive data from other ranks
            for source_rank in range(1, nprocs):
                start = source_rank * qpts_per_rank + min(source_rank, remainder)
                end = start + qpts_per_rank + (1 if source_rank < remainder else 0)
                
                received_data = comm.recv(source=source_rank, tag=source_rank)
                global_data[..., start:end] = received_data
            
            return global_data
        else:
            # Send local data to rank 0
            comm.send(local_data, dest=0, tag=rank)
            return None
    
    def ifft_to_real_space(self, gmat, qpoints):
        """
        gmat (num_wann, num_wann, nmodes, nre_vec_tot, nq)
        qpoints (nq, 3)
        return gmat_real (num_wann, num_wann, nmodes, nre_vec_tot, nph_vec_tot)
        """
        self.logger.info(f"Converting e-ph matrix from reciprocal space to real space")
        self.logger.debug(f"gmat shape: {gmat.shape}")
        self.logger.debug(f"qpoints shape: {qpoints.shape}")
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