import numpy as np
import h5py
from .post_qe2pert import PostQE2Pert
from .phonon_disp import PhononDispersion
from .electron_bands import ElectronBands
from pyeph.utils.constants import ryd_to_mev, ryd_to_ev, bohr_to_ang
from pyeph.utils.logger import setup_logger

class CalcEphMatReciprocal(PostQE2Pert):
    """
    Compute electron-phonon coupling matrix
    """
    
    def __init__(self, epr_file, verbose=False):
        super().__init__(epr_file, verbose)
        self.logger = setup_logger("calc_eph_mat_reciprocal", level="DEBUG" if verbose else "INFO")
        
        # Initialize phonon and electron calculations
        self.phonon_calc = PhononDispersion(epr_file, verbose)
        self.electron_calc = ElectronBands(epr_file, verbose)
        
        # Extract e-ph matrix elements and setup
        self.rvec_set_el, self.ham_r_info = self.get_rvec_set()
        self.force_constants = self.phonon_calc.extract_force_constants()
        self.rvec_set_ph = self.force_constants['rvec_set_ph']
        
        # Extract e-ph matrix elements with proper WS cell calculation
        self.eph_matrix_elements = self.extract_eph_in_real_space_with_ws(self.ham_r_info)
        
        self.logger.info(f"Initialized with {len(self.rvec_set_el)} electron R-vectors, "
                        f"{len(self.rvec_set_ph)} phonon R-vectors")

    def eph_fourier_el_para(self, kpt):
        """
        Fourier transform electronic part of e-ph matrix elements
        Following Perturbo's eph_fourier_el_para subroutine
        
        Args:
            kpt: k-point in crystal coordinates (3,)
            
        Returns:
            g_kerp: complex array (3, max_nrp, nb, nb, na)
        """

        # the rvecs for phonon can be different, we are picking the maximal one
        max_nrp = max(data['ep_hop'].shape[0] for data in self.eph_matrix_elements.values())
        # g_k(electronic)_r(phononic)
        g_kerp = np.zeros((3, max_nrp, self.num_wann, self.num_wann, self.nat), dtype=np.complex128)
        
        # Compute phase factors for electronic R-vectors
        exp_ikr = np.exp(1j * 2.0 * np.pi * np.dot(self.rvec_set_el, kpt))
        
        for key, data in self.eph_matrix_elements.items():
            iw, jw, ia = key
            ep_hop = data['ep_hop']  # Shape: (nrp, nre, 3)
            nrp, nre, _ = ep_hop.shape
            
            # Get electronic R-vector indices for this matrix element
            from_key = min(iw, jw)
            to_key = max(iw, jw)
            info = self.ham_r_info[f'H_{from_key+1}{to_key+1}']
            rvec_indices = info['rvec_indices']
            
            # Fourier transform over electronic R-vectors
            for irp in range(nrp):
                for i in range(3):
                    g_element = 0.0 + 0.0j
                    for ire in range(nre):
                        rvec_idx = rvec_indices[ire]
                        g_element += ep_hop[irp, ire, i] * exp_ikr[rvec_idx]
                    g_kerp[i, irp, iw, jw, ia] = g_element
        
        return g_kerp
    
    def extract_eph_in_real_space_with_ws(self, ham_r_info):
        """
        Extract electron-phonon matrix elements with WS cell calculation
        
        Each (iw, jw, ia) matrix element gets its own phononic WS cell between:
        - Wannier center of orbital iw (TODO: why jw center is not used here)
        - Atomic position of atom ia
        """
        self.logger.info("Extracting electron-phonon matrix elements with WS cells...")
        
        # Get atomic positions in crystal coordinates
        atom_pos_cart = self.tau
        atom_pos_cryst = self.convert_coordinates(atom_pos_cart, direction='cart_to_crys')
        
        # Initialize phononic R-vector images
        rvec_ph_images = self.init_rvec_images(kdim=self.qc_dim)
        
        # First pass: collect all unique phononic R-vector indices
        self.logger.debug("Collecting unique phononic R-vector indices from all WS cells...")
        ws_cells_eph = {}
        unique_ph_indices = set()
        
        for ia in range(self.nat):
            for jw in range(self.num_wann):
                for iw in range(self.num_wann):
                    # Compute phononic WS cell for this matrix element
                    # Between wannier center of iw and atomic position of ia
                    # wannier_center = self.wannier_center_cryst[iw] if iw <= jw else self.wannier_center_cryst[jw]
                    wannier_center = self.wannier_center_cryst[iw]
                    ws_ph_indices, _ = self.set_wigner_seitz_cell(
                        self.qc_dim, rvec_ph_images, 
                        wannier_center, atom_pos_cryst[ia]
                    )
                    ws_cells_eph[(iw, jw, ia)] = ws_ph_indices
                    unique_ph_indices.update(ws_ph_indices)
        
        # Create compact unified R-vector set for e-ph matrix (similar to extract_force_constants)
        unique_ph_indices = sorted(unique_ph_indices)
        self.rvec_set_ph_eph = rvec_ph_images['vec_cryst'][unique_ph_indices]
        index_mapping = {orig_idx: new_idx for new_idx, orig_idx in enumerate(unique_ph_indices)}
        
        # Remap WS indices to compact unified set
        for key in ws_cells_eph:
            old_indices = ws_cells_eph[key]
            new_indices = np.array([index_mapping[idx] for idx in old_indices])
            ws_cells_eph[key] = new_indices
        
        # Second pass: load matrix elements with remapped indices
        matrix_elements = {}
        
        for ia in range(self.nat):
            for jw in range(self.num_wann):
                for iw in range(self.num_wann):

                    from_key = min(iw, jw)
                    to_key = max(iw, jw)
                    key = (iw, jw, ia)
                    # ws_ph_key = (from_key, to_key, ia)
                    ws_ph_key = key
                    ws_ph_indices = ws_cells_eph[ws_ph_key]
                    
                    # Load matrix element from HDF5
                    with h5py.File(self.epr_file, 'r') as f:
                        group = f['eph_matrix_wannier']
                        
                        dset_r = f"ep_hop_r_{ia+1}_{jw+1}_{iw+1}"
                        dset_i = f"ep_hop_i_{ia+1}_{jw+1}_{iw+1}"
                        assert dset_r in group and dset_i in group, f"Dataset {dset_r} or {dset_i} not found in H5"
                        
                        r_val = group[dset_r][:]
                        i_val = group[dset_i][:]
                        ep_hop = r_val + 1j * i_val  # (nrp, nre, 3)

                        if iw > jw: # TODO:  tong: I think this is correct, but to be carefully checked.
                            ep_hop = ep_hop.conj()
                        
                        # Verify dimensions match WS cell (using original count before remapping)
                        nrp_expected = len(ws_ph_indices)
                        assert ep_hop.shape[0] == nrp_expected, f"Phononic R-vector count mismatch: {ep_hop.shape[0]} != {nrp_expected}"
                        
                        # Verify electronic R-vectors match
                        nre_expected = ham_r_info[f'H_{from_key+1}{to_key+1}']['nr']
                        assert ep_hop.shape[1] == nre_expected, f"Electronic R-vector count mismatch: {ep_hop.shape[1]} != {nre_expected}"
                        
                        matrix_elements[key] = {
                            'ep_hop': ep_hop,
                            'ws_ph_indices': ws_ph_indices,  # Remapped indices
                            'nrp': len(ws_ph_indices)
                        }
        
        self.logger.info(f"Extracted {len(matrix_elements)} electron-phonon matrix elements with unified WS cells")
        self.logger.info(f"Created unified phononic R-vector set with {len(unique_ph_indices)} vectors")
        return matrix_elements

    def eph_fourier_elph(self, qpt, g_kerp):
        """
        Fourier transform phononic part of e-ph matrix elements
        Following Perturbo's eph_fourier_elph subroutine
        
        Args:
            qpt: q-point in crystal coordinates (3,)
            g_kerp: output from eph_fourier_el_para
            
        Returns:
            gkq: complex array (nb, nb, 3*na) - e-ph matrix in k,q space
        """
        nmodes = 3 * self.nat
        gkq = np.zeros((self.num_wann, self.num_wann, nmodes), dtype=np.complex128)
        
        # Compute phase factors for unified phononic R-vectors (from e-ph WS cells)
        exp_iqr = np.exp(1j * 2.0 * np.pi * np.dot(self.rvec_set_ph_eph, qpt))

        for ia in range(self.nat):
            for jw in range(self.num_wann):
                for iw in range(self.num_wann):
                    key = (iw, jw, ia)
                    ep_hop = self.eph_matrix_elements[key]['ep_hop']  # Shape: (nrp, nre, 3)
                    nrp = ep_hop.shape[0]
                    
                    # Get WS cell indices for this matrix element
                    ws_ph_indices = self.eph_matrix_elements[key]['ws_ph_indices']
                    assert nrp == len(ws_ph_indices)
                    
                    # Fourier transform over phononic R-vectors
                    gkq_tmp = np.zeros(3, dtype=np.complex128)
                    for ir, rvec_idx in enumerate(ws_ph_indices):
                        # Use rvec_idx (global index) for phase, ir (local index) for g_kerp
                        phase = exp_iqr[rvec_idx]
                        for i in range(3):
                            if iw > jw:
                                gkq_tmp[i] += g_kerp[i, ir, iw, jw, ia] * phase.conj() # Why???
                            else:
                                gkq_tmp[i] += g_kerp[i, ir, iw, jw, ia] * phase
                    
                    # Store in phononic mode index format
                    for i in range(3):
                        idx = ia * 3 + i
                        gkq[iw, jw, idx] = gkq_tmp[i]
        
        return gkq

    def eph_transform(self, qpt, phonon_modes, uk, ukq, gkq):
        """
        Transform e-ph matrix elements to eigenstate basis
        Following Perturbo's eph_transform subroutine
        
        Args:
            qpt: q-point in crystal coordinates
            phonon_modes: phonon eigenvectors (3*nat, 3*nat)
            uk: electron eigenvectors at k (nb, nb)
            ukq: electron eigenvectors at k+q (nb, nb)
            gkq: e-ph matrix in Wannier/Cartesian basis (nb, nb, 3*nat)
            
        Returns:
            gkq: transformed e-ph matrix in eigenstate basis (nb, nb, 3*nat)
        """
        nmodes = 3 * self.nat
        
        # Transform to phonon mode coordinates: gkq'(j) = sum_i gkq(i)*mq(i,j)
        for jw in range(self.num_wann):
            for iw in range(self.num_wann):
                ctmp = gkq[iw, jw, :].copy()
                gkq[iw, jw, :] = np.dot(ctmp, phonon_modes)
        
        # TODO: Add polar correction if needed
        # if self.phonon_calc.lpolar:
        #     # Add long-range polar correction
        #     pass
        
        # Transform from Wannier to Bloch gauge: g^(H) = U_kq^  g^(W) U_k
        ukq_h = ukq.T.conj()
        for i in range(nmodes):
            gtmp = np.dot(ukq_h, gkq[:, :, i])
            gkq[:, :, i] = np.dot(gtmp, uk)
        
        return gkq

    def calc_ephmat(self, kpoints, qpoints, phfreq_cutoff=1.5/ryd_to_mev):
        """
        Compute electron-phonon coupling matrix with MPI support over both k and q points
        
        Args:
            kpoints: array of k-points (nk, 3) in crystal coordinates
            qpoints: array of q-points (nq, 3) in crystal coordinates
            phfreq_cutoff: phonon frequency cutoff in Rydberg
            
        Returns:
            results: dict containing computed e-ph quantities
        """
        from pyeph.utils.logger import get_mpi_info
        
        mpi_info = get_mpi_info()
        rank = mpi_info['rank']
        nprocs = mpi_info['size']
        comm = mpi_info['comm']
        has_mpi = mpi_info['has_mpi']
        
        nk = len(kpoints)
        nq = len(qpoints)
        nbands = self.num_wann
        nmodes = 3 * self.nat
        tot_mass = np.sum(self.mass)
        total_kq_pairs = nk * nq

        self.logger.info(f"Computing e-ph matrix for {nk} k-points, {nq} q-points, {nbands} bands, {nmodes} modes")
        self.logger.info(f"Total (k,q) pairs: {total_kq_pairs}")

        # Create list of all (k,q) pairs with their indices
        kq_pairs = []
        for ik, xk in enumerate(kpoints):
            for iq, xq in enumerate(qpoints):
                kq_pairs.append((ik, iq, xk, xq))

        # Distribute (k,q) pairs across MPI ranks
        if has_mpi and nprocs > 1:
            pairs_per_rank, remainder = divmod(total_kq_pairs, nprocs)
            start_pair = rank * pairs_per_rank + min(rank, remainder)
            end_pair = start_pair + pairs_per_rank + (1 if rank < remainder else 0)
            local_kq_pairs = kq_pairs[start_pair:end_pair]
            self.logger.info(f"MPI enabled: {nprocs} ranks processing {total_kq_pairs} (k,q) pairs total")
            self.logger.info(f"Rank {rank} is processing {len(local_kq_pairs)} (k,q) pairs")
        else:
            local_kq_pairs = kq_pairs
            self.logger.info(f"Single process mode: processing {total_kq_pairs} (k,q) pairs total")

        # Initialize local storage for results
        local_results = {}  # Store results by (ik, iq) indices

        # Pre-compute phonon frequencies using existing MPI parallelization
        wq_full, _ = self.phonon_calc.compute_phonon_dispersion(qpoints, self.force_constants)
        
        # Extract frequencies (only rank 0 has full results in MPI mode)
        if rank == 0 or not has_mpi:
            wq = wq_full
        else:
            wq = None
        
        # Broadcast phonon frequencies to all ranks
        if has_mpi:
            wq = comm.bcast(wq, root=0)
        
        # Process local (k,q) pairs
        for pair_idx, (ik, iq, xk, xq) in enumerate(local_kq_pairs):
            if self.verbose:
                self.logger.debug(f"Processing pair {pair_idx+1}/{len(local_kq_pairs)}: k={ik}, q={iq}")
            
            xkq = xk + xq
            
            # Electronic wavefunctions
            enk, uk = self.electron_calc.solve_eigenvalue_vector(xk)
            ekq, ukq = self.electron_calc.solve_eigenvalue_vector(xkq)
            
            # Phonon modes at q
            wqt, mq = self.phonon_calc.solve_phonon_modes(self.force_constants, xq)
            
            # Fourier transform electronic part
            g_kerp = self.eph_fourier_el_para(xk)
            
            # Get e-ph matrix elements
            gkq = self.eph_fourier_elph(xq, g_kerp)
            
            # Transform to phonon modes and Bloch gauge
            gkq = self.eph_transform(xq, mq, uk, ukq, gkq)
            
            # Compute |g|^2
            g2 = np.abs(gkq)**2

            # Compute deformation potential and |g| for each mode
            dp2 = np.zeros(nmodes)
            gm2 = np.zeros(nmodes)

            for im in range(nmodes):
                for jb in range(nbands):
                    for ib in range(nbands):
                        dp2[im] += g2[ib, jb, im]
                        if wqt[im] > phfreq_cutoff:
                            gm2[im] += g2[ib, jb, im] * 0.5 / wqt[im]

            # Handle degenerate phonon modes
            im = 0
            while im < nmodes:
                i = 0
                for j in range(im+1, nmodes):
                    if abs(wqt[j] - wqt[im]) > 1e-12:
                        break
                    i += 1
                
                if i > 0:  # Found degenerate modes
                    avg_dp2 = np.sum(dp2[im:im+i+1]) / (i+1)
                    avg_gm2 = np.sum(gm2[im:im+i+1]) / (i+1)
                    dp2[im:im+i+1] = avg_dp2
                    gm2[im:im+i+1] = avg_gm2
                
                im = im + i + 1

            # Store results for this (k,q) pair
            gmod_kq = np.zeros(nmodes)
            dpot_kq = np.zeros(nmodes)
            for im in range(nmodes):
                gmod_kq[im] = np.sqrt(gm2[im] / nbands)
                dpot_kq[im] = np.sqrt(dp2[im] * tot_mass / nbands)
            
            local_results[(ik, iq)] = {
                'gmod': gmod_kq,
                'dpot': dpot_kq
            }

        # Gather all results at rank 0
        if has_mpi and nprocs > 1:
            all_results = comm.gather(local_results, root=0)
            
            if rank == 0:
                # Merge all local results
                merged_results = {}
                for rank_results in all_results:
                    merged_results.update(rank_results)
                
                # Reconstruct full arrays
                dpot = np.zeros((nmodes, nq, nk))
                gmod = np.zeros((nmodes, nq, nk))
                
                for (ik, iq), data in merged_results.items():
                    dpot[:, iq, ik] = data['dpot']
                    gmod[:, iq, ik] = data['gmod']
                
                results = {
                    'kpoints': kpoints,
                    'qpoints': qpoints,
                    'phonon_frequencies': wq,
                    'deformation_potential': dpot * ryd_to_ev / bohr_to_ang,
                    'eph_matrix_elements': gmod * ryd_to_mev,
                    'phfreq_cutoff': phfreq_cutoff
                }
                self.logger.info("Successfully computed e-ph coupling matrix in all-reciprocal space!")
                return results
            else:
                return None
        else:
            # Single process - reconstruct arrays directly
            dpot = np.zeros((nmodes, nq, nk))
            gmod = np.zeros((nmodes, nq, nk))
            
            for (ik, iq), data in local_results.items():
                dpot[:, iq, ik] = data['dpot']
                gmod[:, iq, ik] = data['gmod']
            
            results = {
                'kpoints': kpoints,
                'qpoints': qpoints,
                'phonon_frequencies': wq,
                'deformation_potential': dpot * ryd_to_ev / bohr_to_ang,
                'eph_matrix_elements': gmod * ryd_to_mev,
                'phfreq_cutoff': phfreq_cutoff
            }
            self.logger.info("Successfully computed e-ph coupling matrix in all-reciprocal space!")
            return results