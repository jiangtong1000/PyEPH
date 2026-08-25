# SVD of the real-space e-ph tensor g(Re, Rp) and dump of svd_kq.h5 for the
# variational polaron code. Follows the modified Perturbo: init_eph_svd and
# cal_uk_formf/cal_vq_formf in eph_svd.f90, dump_svd_on_kq_h5 in calc_ephmat.f90.

import numpy as np
import h5py

from .eph_mat_mixed import CalcEphMatMixed
from .polar_eph import eph_wan_longrange
from pyeph.utils.constants import ryd_to_mev
from pyeph.utils.logger import setup_logger, get_mpi_info
from pyeph.utils.grid import uniform_grid_zfast, minus_index_map


def write_svd_kq(path, Uk, Vq, Uk_bloch, longrange, phmodes, phfreq_mev, bands_mev):
    """
    Write svd_kq.h5 the way read_svd_kq in mini_dd2 expects it:
        Uk        (Nc, nb, nb, Nk)
        Vq        (nph, Nc, nb, nb, Nq)
        Uk_bloch  (Nk, nb, nb)    [ik, band, wannier]
        longrange (Nq, nph)
        phmodes   (Nq, nph, nph)  [iq, mode, cart]
        phfreq    (Nq, nph)  meV, NOT zeroed below the cutoff
        bands     (Nk, nb)   meV
    """
    with h5py.File(path, 'w') as f:
        for name, arr in (('Uk', Uk), ('Vq', Vq), ('Uk_bloch', Uk_bloch),
                          ('longrange', longrange), ('phmodes', phmodes)):
            f.create_dataset(name + '_real', data=np.ascontiguousarray(arr.real))
            f.create_dataset(name + '_imag', data=np.ascontiguousarray(arr.imag))
        f.create_dataset('phfreq', data=np.ascontiguousarray(phfreq_mev))
        f.create_dataset('bands', data=np.ascontiguousarray(bands_mev))


class EphSVD(CalcEphMatMixed):
    """
    Per-band-pair SVD of g(Re, Rp) and evaluation of the k/q form factors,
    like Perturbo's svd-elph mode.
    """

    def __init__(self, epr_file, polar=False, verbose=False):
        super().__init__(epr_file, polar=polar, verbose=verbose)
        self.logger = setup_logger("eph_svd", level="DEBUG" if verbose else "INFO")
        self._build_svd_el_rvec_set()
        self.Ur = None   # (nre, Nc, nb, nb)
        self.Vr = None   # (nb, nb, Nc, nmod, nrp), singular values absorbed
        self.S2 = None   # sigma^2, descending (this is what Perturbo's S_g holds)
        self.Nc = None
        self.svd_rel_error = None

    def _build_svd_el_rvec_set(self):
        # Perturbo builds the electron WS cell of epr(iw,jw,ia) from the centers
        # (iw, jw) IN THAT ORDER (set_ws_cell_eph), so for iw > jw the R set is
        # the mirror of the Hamiltonian one. We need our own unified set here,
        # the ham rvec_set_el is not enough.
        images = self.init_rvec_images(kdim=self.kc_dim)
        cells = {}
        uniq = set()
        for iw in range(self.num_wann):
            for jw in range(self.num_wann):
                ws_idx, _ = self.set_wigner_seitz_cell(
                    self.kc_dim, images,
                    self.wannier_center_cryst[iw], self.wannier_center_cryst[jw])
                cells[(iw, jw)] = ws_idx
                uniq.update(ws_idx.tolist())
        uniq = sorted(uniq)
        self.rvec_set_el_svd = images['vec_cryst'][uniq]
        mapping = {orig: new for new, orig in enumerate(uniq)}
        self.svd_el_indices = {key: np.array([mapping[i] for i in idx])
                               for key, idx in cells.items()}

        # for iw <= jw this must reproduce the Hamiltonian WS cell exactly
        for jw in range(self.num_wann):
            for iw in range(jw + 1):
                ham_vecs = self.rvec_set_el[
                    self.ham_r_info[f'H_{iw+1}{jw+1}']['rvec_indices']]
                svd_vecs = self.rvec_set_el_svd[self.svd_el_indices[(iw, jw)]]
                assert np.array_equal(ham_vecs, svd_vecs), (iw, jw)
        for (iw, jw, ia), d in self.eph_matrix_elements.items():
            assert d['ep_hop'].shape[1] == len(self.svd_el_indices[(iw, jw)]), (iw, jw, ia)
        self.logger.info(f"SVD electron R set: {len(self.rvec_set_el_svd)} vectors "
                         f"(Hamiltonian set: {len(self.rvec_set_el)})")

    def assemble_g_rerp(self, ib, jb):
        """
        Raw complex g for band pair (ib, jb) as the matrix G(nre, nmod*nrp),
        column (3*ia + ix)*nrp + irp like in eph_svd.f90.
        """
        nre = len(self.rvec_set_el_svd)
        nrp = len(self.rvec_set_ph_eph)
        nmod = 3 * self.nat
        G = np.zeros((nre, nmod * nrp), dtype=np.complex128)
        el_idx = self.svd_el_indices[(ib, jb)]
        for ia in range(self.nat):
            d = self.eph_matrix_elements[(ib, jb, ia)]
            ep = d['ep_hop']  # (nrp_a, nre_a, 3)
            if ib > jb:
                # undo the conj applied in extract_eph_in_real_space_with_ws,
                # the svd wants the raw dataset (Fortran reads it without conj)
                ep = ep.conj()
            ph_idx = d['ws_ph_indices']
            for ix in range(3):
                im = 3 * ia + ix
                G[np.ix_(el_idx, im * nrp + ph_idx)] = ep[:, :, ix].T
        return G

    def build_svd_factors(self, nsvd=None):
        """
        SVD via the Gram matrix G G^H per band pair, exactly like the Fortran:
        V = U^H G, so V carries the singular values. nsvd=None keeps full rank.
        """
        nb, nat = self.num_wann, self.nat
        nmod = 3 * nat
        nre = len(self.rvec_set_el_svd)
        nrp = len(self.rvec_set_ph_eph)
        Nc = nre if nsvd is None else min(int(nsvd), nre)
        if nsvd is not None and int(nsvd) > nre:
            self.logger.warning(f"nsvd={nsvd} exceeds nre={nre}, clamped")
        self.Nc = Nc

        self.Ur = np.zeros((nre, Nc, nb, nb), dtype=np.complex128)
        self.Vr = np.zeros((nb, nb, Nc, nmod, nrp), dtype=np.complex128)
        self.S2 = np.zeros((nre, nb, nb))
        err = np.zeros(2)  # num, den

        comm, rank, nprocs, use_mpi = self._mpi_setup()
        iwork = 0
        for ib in range(nb):
            for jb in range(nb):
                iwork += 1
                if use_mpi and (iwork % nprocs) != rank:
                    continue
                G = self.assemble_g_rerp(ib, jb)
                W = G @ G.conj().T
                evals, evecs = np.linalg.eigh(W)
                U = evecs[:, ::-1]  # descending
                self.S2[:, ib, jb] = evals[::-1]
                V = U.conj().T @ G
                self.Ur[:, :, ib, jb] = U[:, :Nc]
                self.Vr[ib, jb] = V[:Nc].reshape(Nc, nmod, nrp)
                err[0] += np.sum(np.abs(G - U[:, :Nc] @ V[:Nc])**2)
                err[1] += np.sum(np.abs(G)**2)
                self.logger.debug(f"SVD pair ({ib},{jb}) done ({iwork}/{nb*nb})")

        self._allreduce([self.Ur, self.Vr, self.S2, err], comm, use_mpi)
        self.svd_rel_error = err[0] / max(err[1], 1.0e-300)
        self.logger.info(f"SVD low-rank relative error (Nc={Nc}) = "
                         f"{self.svd_rel_error:.5e}")

    def cal_uk_formf(self, kpts, chunk=4096):
        # Uk[g, ib, jb, ik] = sum_Re Ur[Re, g, ib, jb] exp(+i 2pi k.Re)
        nk = len(kpts)
        nb = self.num_wann
        Uk = np.zeros((self.Nc, nb, nb, nk), dtype=np.complex128)
        comm, rank, nprocs, use_mpi = self._mpi_setup()
        lo, hi = self._local_range(nk, rank, nprocs) if use_mpi else (0, nk)
        for s in range(lo, hi, chunk):
            e = min(s + chunk, hi)
            expkr = np.exp(2j * np.pi * (kpts[s:e] @ self.rvec_set_el_svd.T))
            Uk[..., s:e] = np.einsum('ke,egij->gijk', expkr, self.Ur, optimize=True)
        self._allreduce([Uk], comm, use_mpi)
        return Uk

    def assemble_vq(self, qpts, Freq, Modes, phfreq_cutoff_ry, chunk=256):
        """
        Vq[nu, g, ib, jb, iq]: FT over Rp (phase +i 2pi q.Rp), rotate cartesian
        -> eigenmode with the (gauge fixed) modes, then scale by
        sqrt(0.5/w) * ryd2mev / sqrt(Nq). Modes at or below the cutoff get 0.
        """
        nq = len(qpts)
        nb, nmod = self.num_wann, 3 * self.nat
        keep = Freq > phfreq_cutoff_ry  # strict >, like the Fortran
        scale = np.zeros_like(Freq)
        scale[keep] = np.sqrt(0.5 / Freq[keep]) * ryd_to_mev / np.sqrt(float(nq))

        Vq = np.zeros((nmod, self.Nc, nb, nb, nq), dtype=np.complex128)
        comm, rank, nprocs, use_mpi = self._mpi_setup()
        lo, hi = self._local_range(nq, rank, nprocs) if use_mpi else (0, nq)
        for s in range(lo, hi, chunk):
            e = min(s + chunk, hi)
            expqr = np.exp(2j * np.pi * (qpts[s:e] @ self.rvec_set_ph_eph.T))
            Vq_cart = np.einsum('ijgap,qp->ijgaq', self.Vr, expqr, optimize=True)
            # no conj in the mode rotation
            Vq_mode = np.einsum('ijgaq,qan->ngijq', Vq_cart, Modes[s:e], optimize=True)
            Vq[..., s:e] = Vq_mode * scale[s:e].T[:, None, None, None, :]
        self._allreduce([Vq], comm, use_mpi)
        return Vq

    def assemble_longrange(self, qpts, Freq, Modes, phfreq_cutoff_ry):
        # band-diagonal polar term, (Nq, nmod) in meV, zeros if not polar
        nq = len(qpts)
        nmod = 3 * self.nat
        L = np.zeros((nq, nmod), dtype=np.complex128)
        if not self.phonon_calc.lpolar:
            return L
        pp = self.phonon_calc.polar_params
        comm, rank, nprocs, use_mpi = self._mpi_setup()
        lo, hi = self._local_range(nq, rank, nprocs) if use_mpi else (0, nq)
        for iq in range(lo, hi):
            lr = eph_wan_longrange(pp, self.bg, self.tau, self.volume,
                                   self.tpiba, self.nat, qpts[iq], uf=Modes[iq])
            keep = Freq[iq] > phfreq_cutoff_ry
            sc = np.zeros(nmod)
            sc[keep] = np.sqrt(0.5 / Freq[iq][keep]) * ryd_to_mev / np.sqrt(float(nq))
            L[iq] = lr * sc
        self._allreduce([L], comm, use_mpi)
        return L

    def solve_bloch(self, kpts):
        # Ek in Ry, Ub[ik] is the eigenvector matrix (wannier rows, band columns)
        nk, nb = len(kpts), self.num_wann
        Ek = np.zeros((nk, nb))
        Ub = np.zeros((nk, nb, nb), dtype=np.complex128)
        comm, rank, nprocs, use_mpi = self._mpi_setup()
        lo, hi = self._local_range(nk, rank, nprocs) if use_mpi else (0, nk)
        for ik in range(lo, hi):
            enk, uk = self.electron_calc.solve_eigenvalue_vector(kpts[ik])
            Ek[ik] = enk
            Ub[ik] = uk
        self._allreduce([Ek, Ub], comm, use_mpi)
        return Ek, Ub

    def solve_phonons(self, qpts):
        # Freq in Ry, Modes[iq] mass-divided eigen-displacements (cart, mode)
        nq, nmod = len(qpts), 3 * self.nat
        Freq = np.zeros((nq, nmod))
        Modes = np.zeros((nq, nmod, nmod), dtype=np.complex128)
        comm, rank, nprocs, use_mpi = self._mpi_setup()
        lo, hi = self._local_range(nq, rank, nprocs) if use_mpi else (0, nq)
        for iq in range(lo, hi):
            wqt, mq = self.phonon_calc.solve_phonon_modes(
                self.force_constants, qpts[iq], mass_weight=True)
            Freq[iq] = wqt
            Modes[iq] = mq
        self._allreduce([Freq, Modes], comm, use_mpi)
        return Freq, Modes

    @staticmethod
    def trs_fix_electrons(Ek, Ub, minus_map, etol=1.0e-3):
        """
        Align the Bloch eigenvectors between k and -k, in place. Port of the
        electron block in calc_ephmat.f90 (dump_svd_on_kq_h5): degenerate blocks
        need |dE| < etol (Ry) at both k and -k, 1x1 blocks get a scalar phase
        from the plain (no conj) overlap, bigger blocks a Procrustes rotation
        W = u.vh of O = U(-k)^T U(k), applied as U(-k) <- U(-k) conj(W).
        TRIM points are skipped.
        """
        nk, nb = Ek.shape
        for ik in range(nk):
            jk = minus_map[ik]
            if jk <= ik:  # each pair once, skip TRIM
                continue
            bstart = 0
            while bstart < nb:
                bend = bstart
                while (bend < nb - 1
                       and abs(Ek[ik, bend + 1] - Ek[ik, bend]) < etol
                       and abs(Ek[jk, bend + 1] - Ek[jk, bend]) < etol):
                    bend += 1
                if bend == bstart:
                    c = np.sum(Ub[jk][:, bstart] * Ub[ik][:, bstart])
                    if abs(c) > 1.0e-8:
                        Ub[jk][:, bstart] *= np.conj(c / abs(c))
                else:
                    blk = slice(bstart, bend + 1)
                    O = Ub[jk][:, blk].T @ Ub[ik][:, blk]
                    u, _, vh = np.linalg.svd(O)
                    Ub[jk][:, blk] = Ub[jk][:, blk] @ np.conj(u @ vh)
                bstart = bend + 1

    @staticmethod
    def trs_fix_phonons(Freq, Modes, minus_map, masses, wtol=1.0e-10,
                        symmetrize=True):
        """
        Same for the phonon eigenvectors but with the mass metric
        O = E(-q)^T M E(q), then hard symmetrization E(-q) = conj(E(q)) and
        pairwise averaged frequencies. In place, TRIM untouched.
        """
        nq, nmod = Freq.shape
        mcart = np.repeat(np.asarray(masses, dtype=float), 3)
        for iq in range(nq):
            jq = minus_map[iq]
            if jq <= iq:
                continue
            bstart = 0
            while bstart < nmod:
                bend = bstart
                while (bend < nmod - 1
                       and abs(Freq[iq, bend + 1] - Freq[iq, bend]) < wtol
                       and abs(Freq[jq, bend + 1] - Freq[jq, bend]) < wtol):
                    bend += 1
                if bend == bstart:
                    c = np.sum(Modes[jq][:, bstart] * (mcart * Modes[iq][:, bstart]))
                    if abs(c) > 1.0e-12:
                        Modes[jq][:, bstart] *= np.conj(c / abs(c))
                else:
                    blk = slice(bstart, bend + 1)
                    O = Modes[jq][:, blk].T @ (mcart[:, None] * Modes[iq][:, blk])
                    u, _, vh = np.linalg.svd(O)
                    Modes[jq][:, blk] = Modes[jq][:, blk] @ np.conj(u @ vh)
                bstart = bend + 1
            if symmetrize:
                m_sym = 0.5 * (Modes[iq] + np.conj(Modes[jq]))
                Modes[iq] = m_sym
                Modes[jq] = np.conj(m_sym)
                f = 0.5 * (Freq[iq] + Freq[jq])
                Freq[iq] = f
                Freq[jq] = f

    @staticmethod
    def apply_mode_phases(Modes, phases, minus_map, tol=1.0e-8):
        # optional per-(q,nu) gauge (localization hook), needs |phase|=1 and
        # phases[-q] = conj(phases[q]) so E(-q) = conj(E(q)) survives
        phases = np.asarray(phases, dtype=np.complex128)
        assert phases.shape == Modes.shape[:2]
        assert np.allclose(np.abs(phases), 1.0, atol=tol)
        assert np.allclose(phases[minus_map], np.conj(phases), atol=tol), \
            "phases must satisfy TRS: phases[-q] = conj(phases[q])"
        Modes *= phases[:, None, :]

    def dump_svd_kq(self, path, kmesh, nsvd=None, phfreq_cutoff_mev=1.0,
                    mode_phases=None):
        """
        Produce svd_kq.h5 on the uniform Gamma-first z-fastest mesh (k grid =
        q grid, that's what mini_dd2 needs). nsvd=None keeps full rank.
        """
        kpts, _ = uniform_grid_zfast(kmesh)
        minus = minus_index_map(kmesh)
        cutoff_ry = float(phfreq_cutoff_mev) / ryd_to_mev

        if self.Ur is None or (nsvd is not None and self.Nc != min(
                int(nsvd), len(self.rvec_set_el_svd))):
            self.build_svd_factors(nsvd=nsvd)

        self.logger.info(f"Solving Bloch states and phonons on {len(kpts)} mesh points")
        Ek, Ub = self.solve_bloch(kpts)
        Freq, Modes = self.solve_phonons(kpts)

        self.trs_fix_electrons(Ek, Ub, minus)
        self.trs_fix_phonons(Freq, Modes, minus, self.mass)
        if mode_phases is not None:
            self.apply_mode_phases(Modes, mode_phases, minus)

        self.logger.info("Evaluating SVD form factors on the mesh")
        Uk = self.cal_uk_formf(kpts)
        Vq = self.assemble_vq(kpts, Freq, Modes, cutoff_ry)
        L = self.assemble_longrange(kpts, Freq, Modes, cutoff_ry)

        _, rank, _, use_mpi = self._mpi_setup()
        if not use_mpi or rank == 0:
            write_svd_kq(path, Uk, Vq,
                         Ub.transpose(0, 2, 1),     # [ik, band, wannier]
                         L,
                         Modes.transpose(0, 2, 1),  # [iq, mode, cart]
                         Freq * ryd_to_mev,
                         Ek * ryd_to_mev)
            self.logger.info(f"Wrote {path}")
        return dict(kpts=kpts, Uk=Uk, Vq=Vq, Uk_bloch=Ub, longrange=L,
                    phfreq=Freq, phmodes=Modes, bands=Ek)

    @staticmethod
    def _mpi_setup():
        info = get_mpi_info()
        use_mpi = info['has_mpi'] and info['size'] > 1
        return info['comm'], info['rank'], info['size'], use_mpi

    @staticmethod
    def _local_range(n, rank, nprocs):
        per, rem = divmod(n, nprocs)
        lo = rank * per + min(rank, rem)
        return lo, lo + per + (1 if rank < rem else 0)

    @staticmethod
    def _allreduce(arrays, comm, use_mpi):
        if not use_mpi:
            return
        from mpi4py import MPI
        for a in arrays:
            comm.Allreduce(MPI.IN_PLACE, a)
