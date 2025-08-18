# extract eph matrix elements from the ephmat.h5 file

import h5py
import numpy as np
from itertools import product
import scipy.linalg
from pyeph.post_qe2pert import PostQE2Pert
from pyeph.post_qe2pert.linalg import unpack_dyn_matrix
from pyeph.utils.logger import get_mpi_info, get_mpi_rank, get_mpi_size, get_mpi_comm


class PhononDispersion(PostQE2Pert):
    def __init__(self, epr_file, verbose=False):
        super().__init__(epr_file, verbose)
        self.logger = self.logger.getChild("phonon_disp")

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
        dmat_without_mass = np.zeros((3, 3, num_atom_pairs), dtype=np.complex128)
        
        pair_idx = 0
        for ja in range(self.nat):
            for ia in range(ja + 1):
                ifc_matrix = ifc_data[(ia, ja)]['ifc_matrix']
                ws_ph_indices = ifc_data[(ia, ja)]['ws_ph_indices']
                
                for ir, rvec_idx in enumerate(ws_ph_indices):
                    phase = phase_factors[rvec_idx]
                    dmat_without_mass[:, :, pair_idx] += phase * ifc_matrix[ir]
                pair_idx += 1
        
        # Apply polar correction if needed (following Fortran phonon_dispersion.f90:100-106)
        if self.lpolar:
            if self.verbose:
                print(f"  Applying polar correction for q = {qpoint}")
            dmat_lr = self.dyn_mat_longrange(qpoint)
            if dmat_lr is not None:
                # Add long-range correction to short-range part
                dmat_without_mass += dmat_lr
        
        # Pack upper triangular dynamical matrix
        num_elements = nmodes * (nmodes + 1) // 2
        dyn_upper = np.zeros(num_elements, dtype=np.complex128)
        
        for jj in range(nmodes):
            for ii in range(jj + 1):
                elem_idx = (jj * (jj + 1)) // 2 + ii
                
                ia, i = divmod(ii, 3)
                ja, j = divmod(jj, 3)
                mass_factor = 1.0 / np.sqrt(masses[ia] * masses[ja])
                
                pair_idx = (ja * (ja + 1)) // 2 + ia
                
                if ia != ja:
                    dyn_upper[elem_idx] = dmat_without_mass[i, j, pair_idx] * mass_factor
                else:
                    dyn_upper[elem_idx] = (dmat_without_mass[i, j, pair_idx] + np.conj(dmat_without_mass[j, i, pair_idx])) * 0.5 * mass_factor
        
        # Diagonalize dynamical matrix
        dyn_matrix = unpack_dyn_matrix(dyn_upper, nmodes)
        eigenvalues, eigenvectors = scipy.linalg.eigh(dyn_matrix)
        
        # Compute frequencies with proper handling of negative eigenvalues
        frequencies = np.sign(eigenvalues) * np.sqrt(np.abs(eigenvalues))
        
        # Normalize eigenvectors by mass
        mass_sqrt_inv = np.repeat(1.0 / np.sqrt(masses), 3)
        modes = eigenvectors * mass_sqrt_inv[:, np.newaxis]
        
        return frequencies, modes

    def compute_phonon_dispersion(self, qpath, force_constants):
        """
        Compute phonon dispersion along a q-path with MPI support
        
        Args:
            qpath: array of q-points (nq, 3) in crystal coordinates
            force_constants: output from extract_force_constants()
            
        Returns:
            frequencies: phonon frequencies (nq, 3*nat)
            modes: phonon eigenvectors (nq, 3*nat, 3*nat)
        """
        mpi_info = get_mpi_info()
        rank = mpi_info['rank']
        nprocs = mpi_info['size']
        comm = mpi_info['comm']
        has_mpi = mpi_info['has_mpi']
        
        nq = len(qpath)
        nmodes = 3 * self.nat
        
        # Distribute q-points across MPI ranks
        if has_mpi and nprocs > 1:
            qpts_per_rank, remainder = divmod(nq, nprocs)
            start = rank * qpts_per_rank + min(rank, remainder)
            end = start + qpts_per_rank + (1 if rank < remainder else 0)
            local_qpoints = qpath[start:end]
            self.logger.info(f"MPI enabled: {nprocs} ranks processing {nq} q-points total")
            self.logger.info(f"Rank {rank} is processing q-points {start} to {end}")
        else:
            local_qpoints = qpath
            start, end = 0, nq
            self.logger.info(f"Single process mode: processing {nq} q-points total")
        
        # Compute phonon dispersion for local q-points
        local_nq = len(local_qpoints)
        local_frequencies = np.zeros((local_nq, nmodes))
        local_modes = np.zeros((local_nq, nmodes, nmodes), dtype=np.complex128)
        
        self.logger.info(f"Computing phonon dispersion for {local_nq} q-points...")
        for iq, qpoint in enumerate(local_qpoints):
            freq, mode = self.solve_phonon_modes(force_constants, qpoint)
            local_frequencies[iq] = freq
            local_modes[iq] = mode
            
            if (iq + 1) % 100 == 0:
                self.logger.debug(f"Completed {iq + 1}/{local_nq} q-points")
        
        # Gather results from all ranks
        if has_mpi and nprocs > 1:
            frequencies = self._gather_mpi_results(local_frequencies, nq, nmodes, comm, rank)
            modes = self._gather_mpi_results(local_modes, nq, (nmodes, nmodes), comm, rank, is_complex=True)
        else:
            frequencies = local_frequencies
            modes = local_modes
        
        return frequencies, modes

    def _gather_mpi_results(self, local_data, total_size, mode_shape, comm, rank, is_complex=False):
        """
        Args:
            local_data: Local data array
            total_size: Total number of q-points
            mode_shape: Shape of modes (nmodes,) or (nmodes, nmodes)
            comm: MPI communicator
            rank: Current rank
            is_complex: Whether data is complex
            
        Returns:
            Gathered data array on rank 0, local_data on other ranks
        """
        if rank == 0:
            if isinstance(mode_shape, tuple):
                gathered_data = np.zeros((total_size,) + mode_shape, dtype=local_data.dtype)
            else:
                gathered_data = np.zeros((total_size, mode_shape), dtype=local_data.dtype)
            
            nprocs = comm.Get_size()
            for source_rank in range(nprocs):
                qpts_per_rank, remainder = divmod(total_size, nprocs)
                start = source_rank * qpts_per_rank + min(source_rank, remainder)
                end = start + qpts_per_rank + (1 if source_rank < remainder else 0)
                
                if source_rank == 0:
                    gathered_data[start:end] = local_data
                else:
                    received_data = comm.recv(source=source_rank, tag=source_rank)
                    gathered_data[start:end] = received_data
            
            return gathered_data
        else:
            comm.send(local_data, dest=0, tag=rank)
            return local_data
