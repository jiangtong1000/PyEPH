# extract eph matrix elements from the ephmat.h5 file

import h5py
import numpy as np
from itertools import product
import scipy.linalg
from pyeph.post_qe2pert import PostQE2Pert
from pyeph.post_qe2pert.linalg import unpack_dyn_matrix
from pyeph.utils.logger import get_mpi_info, get_mpi_rank, get_mpi_size, get_mpi_comm


class PhononDispersion(PostQE2Pert):
    def __init__(self, epr_file, polar=None, verbose=False):
        super().__init__(epr_file, verbose)
        self.lpolar = polar
        self.logger = self.logger.getChild("phonon_disp")
        self.setup_polar_correction()
    
    def setup_polar_correction(self):
        with h5py.File(self.epr_file, 'r') as f:
            if self.lpolar is None:
                self.lpolar = f['basic_data/lpolar'][()] if 'basic_data/lpolar' in f else False
            else:
                assert isinstance(self.lpolar, bool), "polar must be a boolean"
            if self.lpolar:
                self.logger.info("Polar correction is enabled")
                self.polar_params = {}
                self.polar_params['epsil'] = f['basic_data/epsil'][:]  # dielectric tensor (3,3)
                self.polar_params['zstar'] = f['basic_data/zstar'][:]  # Born effective charges (nat,3,3)
                self.polar_params['polar_alpha'] = f['basic_data/polar_alpha'][()] if 'basic_data/polar_alpha' in f else 1.0
                assert self.polar_params['polar_alpha'] > 1e-12, "polar_alpha is too small"
                self.polar_params['loto_alpha'] = f['basic_data/loto_alpha'][()] if 'basic_data/loto_alpha' in f else 1.0
                
                self.polar_params['system_2d'] = f['basic_data/system_2d'][()] if 'basic_data/system_2d' in f else False
                assert not self.polar_params['system_2d'], "2D system is not supported yet"
                
                self.polar_params['gmax'] = 14.0
                ggmax = self.polar_params['gmax'] * 4.0 * self.polar_params['polar_alpha']
                
                def estimate_nrx(bg_vec, epsil):
                    return int(np.ceil(np.sqrt(ggmax / np.dot(bg_vec, np.dot(epsil, bg_vec)))))
                def estimate_nrx_ph(bg_vec):
                    return int(np.ceil(np.sqrt(ggmax / np.dot(bg_vec, bg_vec))))

                nrx1 = estimate_nrx(self.bg[:, 0], self.polar_params['epsil'])
                nrx2 = estimate_nrx(self.bg[:, 1], self.polar_params['epsil'])
                nrx3 = estimate_nrx(self.bg[:, 2], self.polar_params['epsil'])
                nrx1_ph = estimate_nrx_ph(self.bg[:, 0])
                nrx2_ph = estimate_nrx_ph(self.bg[:, 1])
                nrx3_ph = estimate_nrx_ph(self.bg[:, 2])
                for i in range(3):
                    if self.qc_dim[i] < 2:
                        nrx[i] = 0
                        nrx_ph[i] = 0
                self.polar_params['nrx'] = np.array([nrx1, nrx2, nrx3])
                self.polar_params['nrx_ph'] = np.array([nrx1_ph, nrx2_ph, nrx3_ph])
                
                self.init_onsite_polar_correction()

    def init_onsite_polar_correction(self):
        """
        Initialize onsite correction for polar systems
        Following Perturbo's init_onsite_correction in polar_correction.f90
        
        The onsite correction cancels the long-range contribution at q=0 to enforce
        the acoustic sum rule for phonons.
        """
        self.logger.debug("Initializing onsite correction for polar system...")
        
        nat = self.nat
        nelem = nat * (nat + 1) // 2
        
        # Compute long-range dynamical matrix at Gamma point (q=0)
        q_gamma = np.array([0.0, 0.0, 0.0])
        dyn_gamma = self.dyn_mat_longrange_3d(q_gamma)
        
        # Check that result is real (should be for q=0)
        if np.any(np.abs(np.imag(dyn_gamma)) > 1e-16):
            print("Warning: Onsite polar correction is not real!")
        
        # Unpack upper triangular matrix to full matrix
        dd = np.zeros((3, 3, nat, nat))
        n = 0
        for ja in range(nat):
            for ia in range(ja + 1):
                dd0 = np.real(dyn_gamma[:, :, n])
                
                dd[:, :, ia, ja] = dd0
                if ia != ja:
                    dd[:, :, ja, ia] = dd0.T
                else:
                    # Impose Hermiticity for diagonal terms
                    dd[:, :, ia, ia] = (dd0 + dd0.T) * 0.5
                n += 1
        
        # Compute onsite correction: negative sum over all atom pairs
        self.onsite_correction = np.zeros((3, 3, nat))
        for ia in range(nat):
            dd0 = np.zeros((3, 3))
            for ja in range(nat):
                dd0 += dd[:, :, ia, ja]
            self.onsite_correction[:, :, ia] = -dd0
        
        self.logger.debug(f"Onsite correction computed for {nat} atoms")

    def dyn_mat_longrange_3d(self, qpoint):
        """
        Compute long-range polar correction to dynamical matrix for 3D systems
        Following Perturbo's dyn_mat_longrange_3d in polar_correction.f90
        
        Args:
            qpoint: q-point in crystal coordinates (3,)
            
        Returns:
            dmat_lr: long-range correction (3, 3, nelem) where nelem = nat*(nat+1)//2
        """
        if not self.lpolar:
            return None
            
        nelem = self.nat * (self.nat + 1) // 2
        nrx1, nrx2, nrx3 = self.polar_params['nrx_ph']
        
        # Ewald parameters
        falph = 4.0 * self.polar_params['polar_alpha']
        ggmax = self.polar_params['gmax'] * falph
        fac = 8.0 * np.pi / self.volume # 4pi * e^2 / Volume
        
        dmat_lr = np.zeros((3, 3, nelem), dtype=np.complex128)
        
        # Sum over G-vectors
        for m1 in range(-nrx1, nrx1 + 1):
            for m2 in range(-nrx2, nrx2 + 1):
                for m3 in range(-nrx3, nrx3 + 1):
                    qG_cryst = qpoint + np.array([m1, m2, m3])
                    qG_cart = self.bg.T @ qG_cryst

                    qeq = qG_cart @ self.polar_params['epsil'] @ qG_cart
                    # Skip if too small or too large
                    if qeq < 1e-14 or qeq > ggmax:
                        continue
                        
                    qfac = np.exp(-qeq / falph) / qeq
                    
                    # Sum over atom pairs
                    n = 0
                    for ja in range(self.nat):
                        for ia in range(ja + 1):
                            phase_arg = 2 * np.pi * np.dot(qG_cart, self.tau[ia] - self.tau[ja])
                            phase = np.exp(1j * phase_arg)
                            for i in range(3):
                                for j in range(3):
                                    contrib = qG_cart[i] * qG_cart[j] * qfac * phase
                                    dmat_lr[i, j, n] += contrib
                            n += 1
        
        # Apply Born effective charge tensors
        for ja in range(self.nat):
            for ia in range(ja + 1):
                n = ja * (ja + 1) // 2 + ia
                temp = dmat_lr[:, :, n] @ self.polar_params['zstar'][ja].T
                dmat_lr[:, :, n] = self.polar_params['zstar'][ia] @ temp
    
        dmat_lr *= fac
        return dmat_lr

    def dyn_mat_longrange_2d(self, qpoint):
        raise NotImplementedError("2D long-range polar correction not implemented")

    def dyn_mat_longrange(self, qpoint):
        """
        Compute long-range polar correction to dynamical matrix
        Dispatches to 2D or 3D implementation based on system_2d flag
        Includes onsite correction as per Perturbo's dyn_mat_longrange
        
        Args:
            qpoint: q-point in crystal coordinates (3,)
            
        Returns:
            dmat_lr: long-range correction (3, 3, nelem) where nelem = nat*(nat+1)//2
        """
        if not self.lpolar:
            return None
            
        # Get base long-range correction
        if self.polar_params['system_2d']:
            dmat_lr = self.dyn_mat_longrange_2d(qpoint)
        else:
            dmat_lr = self.dyn_mat_longrange_3d(qpoint)
            
        for ia in range(self.nat):
            idx = (ia + 1) * (ia + 2) // 2 - 1
            dmat_lr[:, :, idx] += self.onsite_correction[:, :, ia]
        
        return dmat_lr

    def extract_force_constants(self):
        self.logger.info("Extracting real-space interatomic force constants...")
        
        mass = self.mass
        atom_pos_cart = self.tau # (nat, 3)
        atom_pos_cryst = self.convert_coordinates(atom_pos_cart, direction='cart_to_crys') # (nat, 3)
        rvec_ph_images = self.init_rvec_images(kdim=self.qc_dim)
        
        self.logger.debug("Setting up Wigner-Seitz cells for IFCs...")
        ws_cells = {}
        unique_ph_indices = set()

        for ja in range(self.nat):
            for ia in range(ja + 1):
                ws_ph_indices, _ = self.set_wigner_seitz_cell(
                    self.qc_dim, rvec_ph_images, atom_pos_cryst[ia], atom_pos_cryst[ja]
                )
                ws_cells[(ia, ja)] = ws_ph_indices
                unique_ph_indices.update(ws_ph_indices)

        # Create compact R-vector set and remapping (for memory efficiency)
        unique_ph_indices = sorted(unique_ph_indices) #TODO: Tong, test if removing it, what happens?
        rvec_set_ph = rvec_ph_images['vec_cryst'][unique_ph_indices]
        index_mapping = {orig_idx: new_idx for new_idx, orig_idx in enumerate(unique_ph_indices)}

        # Update WS indices to compact indices
        for key in ws_cells:
            old_indices = ws_cells[key]
            new_indices = np.array([index_mapping[idx] for idx in old_indices])
            ws_cells[key] = new_indices

        # Read IFC data
        m = 0
        ifc_data = {}
        with h5py.File(self.epr_file, 'r') as f:
            group = f['force_constant']
            for ja in range(self.nat):
                for ia in range(ja + 1):
                    m += 1
                    ws_ph_indices = ws_cells[(ia, ja)]
                    nr = len(ws_ph_indices)

                    dset_name = f"ifc{m}"
                    ifc_matrix = group[dset_name][:]
                    assert ifc_matrix.shape == (nr, 3, 3), f"ifc_matrix shape mismatch: {ifc_matrix.shape} != ({nr}, 3, 3)"
                    ifc_matrix = ifc_matrix.transpose(0, 2, 1) # to match Perturbo's column-major convention
                    # ifc_matrix = (ifc_matrix + ifc_matrix.transpose(0, 2, 1)) * 0.5 # This makes things worse
                
                    ifc_data[(ia, ja)] = {
                        'ifc_matrix': ifc_matrix,
                        'ws_ph_indices': ws_ph_indices,
                        'nrp': len(ws_ph_indices),
                        'mass_factor': 1.0 / np.sqrt(mass[ia] * mass[ja])
                    }

        return {
            'ifc_data': ifc_data,
            'rvec_set_ph': rvec_set_ph,
            'mass': mass,
            'atom_pos_cryst': atom_pos_cryst
        }

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
        
        # orthogonality check
        assert np.allclose(np.dot(eigenvectors, eigenvectors.T.conj()), np.eye(nmodes))
        # Compute frequencies with proper handling of negative eigenvalues
        frequencies = np.sign(eigenvalues) * np.sqrt(np.abs(eigenvalues))
        
        # Normalize eigenvectors by mass
        mass_sqrt_inv = np.repeat(1.0 / np.sqrt(masses), 3)
        modes = np.einsum("ij, i->ij", eigenvectors, mass_sqrt_inv)
        
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
