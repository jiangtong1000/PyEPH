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
                    ws_ph_indices, _ = self.set_wigner_seitz_cell(
                        self.qc_dim, rvec_ph_images, 
                        self.wannier_center_cryst[iw], atom_pos_cryst[ia]
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
                    ws_ph_key = (from_key, to_key, ia)
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

                        if iw > jw: # TODO: tong: this does not matter now because this isn't used. but we ned to think abt this
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

    def calc_ephmat(self, kpoints, qpoints, phfreq_cutoff=0.0):
        """
        Args:
            kpoints: array of k-points (nk, 3) in crystal coordinates
            qpoints: array of q-points (nq, 3) in crystal coordinates
            phfreq_cutoff: phonon frequency cutoff in Rydberg
            
        Returns:
            results: dict containing computed e-ph quantities
        """
        nk = len(kpoints)
        nq = len(qpoints)
        nbands = self.num_wann
        nmodes = 3 * self.nat
        tot_mass = np.sum(self.mass)

        self.logger.info(f"Computing e-ph matrix for {nk} k-points, {nq} q-points, {nbands} bands, {nmodes} modes")

        # Initialize output arrays
        dpot = np.zeros((nmodes, nq, nk))  # deformation potential
        gmod = np.zeros((nmodes, nq, nk))  # |g| matrix elements
        wq = np.zeros((nmodes, nq))        # phonon frequencies
        
        for ik, xk in enumerate(kpoints):
            if self.verbose:
                self.logger.debug(f"Processing k-point {ik+1}/{nk}: {xk}")
            
            # Electronic wavefunction at k
            enk, uk = self.electron_calc.solve_eigenvalue_vector(xk) # uk carry phases
            # Fourier transform electronic part (bottleneck)
            g_kerp = self.eph_fourier_el_para(xk)
            
            # q-point loop
            for iq, xq in enumerate(qpoints):
                xkq = xk + xq
                
                # Electronic wavefunction at k+q
                ekq, ukq = self.electron_calc.solve_eigenvalue_vector(xkq)
                
                # Phonon frequencies and eigenvectors at q
                wqt, mq = self.phonon_calc.solve_phonon_modes(self.force_constants, xq)

                np.save("ekq.npy", ekq)
                np.save("ukq.npy", ukq)
                np.save('wqt.npy', wqt)
                np.save('mq.npy', mq)

                # Store phonon frequencies (only once)
                if ik == 0:
                    wq[:, iq] = wqt
                
                # Get e-ph matrix elements in Wannier gauge and Cartesian coords
                gkq = self.eph_fourier_elph(xq, g_kerp) # TODO: tong, this is different.
                np.save("gkq.npy", gkq)
                exit()

                # Transform to phonon modes and Bloch gauge
                # Could have a sign difference for each elements across different runs, fine
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
                
                np.save('gm2.npy', gm2)
                np.save('dp2.npy', dp2)
                np.save('g2.npy', g2)

                # Handle degenerate phonon modes (average over degenerate modes)
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
                
                # Store final results
                for im in range(nmodes):
                    gmod[im, iq, ik] = np.sqrt(gm2[im] / nbands)
                    dpot[im, iq, ik] = np.sqrt(dp2[im] * tot_mass / nbands)
                


        results = {
            'kpoints': kpoints,
            'qpoints': qpoints,
            'phonon_frequencies': wq,
            'deformation_potential': dpot,
            'eph_matrix_elements': gmod,
            'phfreq_cutoff': phfreq_cutoff
        }
        self.logger.info("Successfully computed e-ph coupling matrix in all-reciprocal space!")
        return results
    
    def output_ephmat_text(self, results, output_file):
        """
        Output e-ph matrix results to text file (original Perturbo format)
        
        Args:
            results: output from calc_ephmat
            output_file: output filename (.ephmat)
        """
        kpoints = results['kpoints']
        qpoints = results['qpoints']
        wq = results['phonon_frequencies']
        dpot = results['deformation_potential']
        gmod = results['eph_matrix_elements']
        
        nk = len(kpoints)
        nq = len(qpoints)
        nmodes = wq.shape[0]
        
        self.logger.info(f"Writing e-ph matrix results to {output_file}")
        
        with open(output_file, 'w') as f:
            f.write("#  ik      xk     iq      xq   imod    omega(meV)    deform. pot.(eV/A)    |g|(meV)\n")
            
            for ik in range(nk):
                for iq in range(nq):
                    for im in range(nmodes):
                        f.write(f"{ik+1:4d} {0.0:9.5f} {iq+1:4d} {0.0:9.5f} {im+1:3d} "
                               f"{wq[im,iq]*ryd_to_mev:12.6f} "
                               f"{dpot[im,iq,ik]*ryd_to_ev/bohr_to_ang:22.12E} "
                               f"{gmod[im,iq,ik]*ryd_to_mev:22.12E}\n")
                    f.write("  \n")
        
        self.logger.info(f"Successfully wrote e-ph matrix results to {output_file}")