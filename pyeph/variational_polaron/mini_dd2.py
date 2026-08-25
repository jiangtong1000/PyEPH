#!/usr/bin/env python
"""Toyozawa / Davydov polaron ansaetze (dD2, D2) for ab initio eph Hamiltonians.

Notation follows Baumgarten, Wu, Jiang & Lee, arXiv:2605.05675:
``A_nk`` electronic and ``B_nuq`` coherent-state amplitudes, projection
intermediate ``D_k``, bands ``eps_nk``, phonon frequencies ``omega_nuq``, and
the low-rank eph kernel  g = U^dag (Sigma V) U + U^dag L U  with Wannier-to-
Bloch matrices ``U_in(k)``, singular vectors ``Sigma_ij^g(k)`` / ``V_ijnu^g(q)``
(g = 1..Nc) and the Wannier-diagonal long-range term ``L_nu(q)``.
"""
from types import SimpleNamespace

import numpy as np
from scipy import fft as _sfft
from scipy.sparse.linalg import LinearOperator, eigsh

__all__ = ["Lattice", "Hamiltonian", "dD2", "D2", "optimize",
           "read_svd_kq", "zero_acoustic_modes"]


def _abs2(x):
    return x.real * x.real + x.imag * x.imag


class Lattice:
    """Uniform k-mesh over the unit cube: index maps and batched 3-D FFTs.

    ``a`` holds the real-space lattice vectors as rows (in units of ``alat``);
    it only enters through ``exp(i K . R_j)``, so it is irrelevant at K = 0.
    Points are ``(ix/nkx, iy/nky, iz/nkz)`` in C order with z fastest, Gamma at
    flat index 0.  ``fft_forward`` is unnormalised; ``fft_backward`` has 1/Nk.
    """

    def __init__(self, a, kmesh, alat=1.0, nthreads=None):
        self.kmesh = np.asarray(kmesh, dtype=np.int64).ravel()
        self.Nk = int(np.prod(self.kmesh))
        self.nthreads = -1 if nthreads is None else int(nthreads)
        nkx, nky, nkz = (int(v) for v in self.kmesh)
        ix, iy, iz = np.meshgrid(np.arange(nkx), np.arange(nky),
                                 np.arange(nkz), indexing="ij")
        self.grid_index = np.stack([ix.ravel(), iy.ravel(), iz.ravel()],
                                   axis=1).astype(np.int64)
        self.Rj = self.grid_index @ (np.asarray(a, float) * float(alat))
        self.minus_k = self._flat(-self.grid_index)
        self._kpq = self._kmq = None

    def _flat(self, idx):
        m = self.kmesh
        idx = np.mod(idx, m)
        return (idx[..., 0] * m[1] + idx[..., 1]) * m[2] + idx[..., 2]

    @property
    def kpq_map(self):
        """Dense (Nk, Nk) k+q table, needed only in the dense path."""
        if self._kpq is None:
            g = self.grid_index
            self._kpq = self._flat(g[:, None, :] + g[None, :, :])
        return self._kpq

    @property
    def kmq_map(self):
        if self._kmq is None:
            g = self.grid_index
            self._kmq = self._flat(g[:, None, :] - g[None, :, :])
        return self._kmq

    def kcoeffs(self, K):
        """``exp(i K . R_j)`` for a cartesian K (scalar means the x axis)."""
        K = np.asarray(K, dtype=np.float64).ravel()
        K = np.array([K[0], 0.0, 0.0]) if K.size == 1 else K
        return np.exp(1j * (self.Rj @ K))

    def _fft(self, x, fn):
        x = np.ascontiguousarray(x)
        shape = x.shape[:-1] + tuple(int(v) for v in self.kmesh)
        return fn(x.reshape(shape), axes=(-3, -2, -1),
                  workers=self.nthreads).reshape(x.shape)

    def fft_forward(self, x):
        return self._fft(x, _sfft.fftn)

    def fft_backward(self, x):
        return self._fft(x, _sfft.ifftn)


def _read_complex(f, stem):
    out = np.empty(f[f"{stem}_real"].shape, dtype=np.complex128)
    out.real[:] = f[f"{stem}_real"][()]
    out.imag[:] = f[f"{stem}_imag"][()]
    return out


def read_svd_kq(path):
    """Read an ``svd_kq.h5``; per-k quantities come back with k/q last."""
    import h5py

    with h5py.File(path, "r") as f:
        eps = np.ascontiguousarray(np.array(f["bands"], dtype=float).T)
        omega = np.ascontiguousarray(np.array(f["phfreq"], dtype=float).T)
        inp = SimpleNamespace(
            eps=eps, omega=omega,                    # (nb, Nk), (nph, Nq)
            Sigma=_read_complex(f, "Uk"),            # (Nc, nb, nb, Nk)
            V=_read_complex(f, "Vq"),                # (nph, Nc, nb, nb, Nq)
            U=_read_complex(f, "Uk_bloch"),          # (Nk, nb, nb)
            L=np.ascontiguousarray(_read_complex(f, "longrange").T),  # (nph, Nq)
        )
    inp.nb, inp.Nk = inp.eps.shape
    inp.nph, inp.Nc = inp.omega.shape[0], inp.Sigma.shape[0]
    return inp


def zero_acoustic_modes(omega, V, L=None, cutoff=1.0e-3):
    """Zero modes with ``omega < cutoff`` in-place (frequencies and coupling)."""
    mask = omega.real < cutoff
    omega[mask] = 0.0
    inu, iq = np.nonzero(mask)
    V[inu, :, :, :, iq] = 0.0
    if L is not None:
        L[mask] = 0.0
    return mask


class Hamiltonian:
    """Bands ``eps_nk``, frequencies ``omega_nuq`` and the eph coupling.

    Supply the low-rank factors of arXiv:2605.05675 Eq. (5) --
    ``Sigma (Nc,nb,nb,Nk)``, ``V (nph,Nc,nb,nb,Nq)``, the Wannier-to-Bloch
    matrices ``U (Nk,nb,nb)`` [+ Wannier-diagonal long-range ``L (nph,Nq)``] --
    and/or a dense ``g[ik, iq, inu, jb, ib]``; ``use_svd`` picks the path.

    ``L_nu(q)`` is Wannier-diagonal and k-independent, so it is itself one
    extra singular vector: ``Sigma -> delta_ij``, ``V -> L * delta_ij``.  The
    constructor absorbs it that way (``Nc += 1``, one transient copy of ``V``),
    so downstream there is only the pure ``Sigma (x) V`` contraction.
    """

    def __init__(self, eps, omega, lattice, Sigma=None, V=None, U=None,
                 L=None, g=None):
        self.lattice = lattice
        self.eps = np.ascontiguousarray(eps, dtype=np.complex128)
        self.omega = np.ascontiguousarray(omega, dtype=np.complex128)
        (self.nb, Nk), self.nph = self.eps.shape, self.omega.shape[0]
        if Nk != lattice.Nk or self.omega.shape[1] != lattice.Nk:
            raise ValueError("eps/omega k-dimension does not match the lattice")
        self.Sigma = self.V = self.U = self.g = None
        self.Nc = 0

        if Sigma is not None:
            Sigma = np.ascontiguousarray(Sigma, dtype=np.complex128)
            self.Nc = Sigma.size // (self.nb * self.nb * Nk)
            Sigma = Sigma.reshape(self.Nc, self.nb, self.nb, Nk)
            V = np.ascontiguousarray(V, np.complex128).reshape(
                self.nph, self.Nc, self.nb, self.nb, Nk)

            if L is not None:
                eye = np.eye(self.nb, dtype=np.complex128)
                L = np.asarray(L, np.complex128)
                Sigma = np.concatenate(
                    [Sigma, np.tile(eye[None, :, :, None], (1, 1, 1, Nk))])
                V = np.concatenate(
                    [V, L[:, None, None, None, :]
                        * eye[None, None, :, :, None]], axis=1)
                self.Nc += 1

            self.Sigma, self.V = Sigma, V
            self.U = np.ascontiguousarray(U, np.complex128).reshape(
                Nk, self.nb, self.nb)

        if g is not None:
            self.g = np.ascontiguousarray(g, np.complex128).reshape(
                Nk, Nk, self.nph, self.nb, self.nb)

        if self.Sigma is None and self.g is None:
            raise ValueError("supply either the low-rank factors or a dense g")

        self.use_svd = self.Sigma is not None

    def to_wannier(self, A_conj):
        """``Aw[i,k] = sum_n U[k,n,i] A_conj[n,k]`` -- conj(A) to Wannier."""
        return np.einsum("kli,lk->ik", self.U, A_conj, optimize=True)

    def from_wannier(self, w, conjugate=False):
        """Adjoint of :meth:`to_wannier` (optionally conjugated rotation)."""
        B = np.conj(self.U) if conjugate else self.U
        return np.einsum("kli,ik->lk", B, w, optimize=True)

    def reconstruct_g_from_svd(self):
        """Dense ``g`` from the low-rank factors."""
        lat, nb, nph, Nk = self.lattice, self.nb, self.nph, self.lattice.Nk
        g_wann = np.einsum("ngijq,gijk->nqkij", self.V, self.Sigma,
                           optimize=True)          # [inu, iq, ik, iw, jw]
        M = np.ascontiguousarray(self.U.transpose(0, 2, 1))
        Mdag = np.conj(M).transpose(0, 2, 1)

        g = np.empty((Nk, Nk, nph, nb, nb), dtype=np.complex128)
        for iq in range(Nk):
            blk = np.einsum("kai,nkij,kjb->nkab", Mdag[lat.kpq_map[:, iq]],
                            g_wann[:, iq], M, optimize=True)
            g[:, iq] = blk.transpose(1, 0, 3, 2)   # -> [ik, inu, jb, ib]
        self.g, self.use_svd = g, False

        return g


class dD2:
    """Momentum-projected Toyozawa (dD2) ansatz at crystal momentum ``K``.

    Variational parameters (arXiv:2605.05675 Eqs. (3)-(4)): coherent-state
    amplitudes ``B (Nk, nph)`` and electronic amplitudes ``A (Nk, nb)``, passed
    in the public shapes and stored transposed (k last).  ``herm`` averages in
    the hermitian twin of the truncated-SVD eph term (the production setting).
    """

    def __init__(self, shift, electron, ham, lattice=None, K=0.0, herm=True):
        self.ham = ham
        self.lattice = ham.lattice if lattice is None else lattice
        self.nb, self.nph, self.Nk = ham.nb, ham.nph, self.lattice.Nk
        self.herm = herm
        self.eiKR = self.lattice.kcoeffs(K)        # exp(i K . R_j)
        self.nB, self.nA = self.Nk * self.nph, self.Nk * self.nb
        self.nparams = 2 * (self.nB + self.nA)
        self.set_params(shift, electron)

    def set_params(self, shift, electron):
        """Set ``B`` (shift) and ``A`` (electron) from the public shapes."""
        self.B = np.asarray(shift, np.complex128).reshape(self.Nk, self.nph).T.copy()
        self.A = np.asarray(electron, np.complex128).reshape(self.Nk, self.nb).T.copy()

    def get_params(self):
        """``(B, A)`` back in the public ``(Nk, ...)`` shapes."""
        return (np.ascontiguousarray(self.B.T), np.ascontiguousarray(self.A.T))

    def _intermediates(self, A=None):
        """FFT objects shared by energy and gradient (D_k = FFT[cs_overlap_K])."""
        lat, A = self.lattice, self.A if A is None else A
        A_sq = _abs2(A)
        B_sq_j = _abs2(self.B).sum(0)
        # exp(sum |B|^2 e^{-iq.Rj}); the subtracted norm cancels in E = F/S
        cs_overlap = np.exp(lat.fft_forward(B_sq_j.astype(np.complex128))
                            - B_sq_j.sum())
        cs_overlap_K = cs_overlap * self.eiKR
        D = lat.fft_forward(cs_overlap_K)
        A_sq_ft = lat.fft_forward(A_sq.sum(0).astype(np.complex128))
        return SimpleNamespace(A=A, A_sq=A_sq, A_sq_ft=A_sq_ft,
                               cs_overlap=cs_overlap,
                               cs_overlap_K=cs_overlap_K, D=D,
                               ovlp=np.sum(cs_overlap_K * A_sq_ft).real)

    def energy_terms(self):
        """Unnormalised ``(kinetic, eph, phonon, overlap)`` at the state."""
        lat, ham, inter = self.lattice, self.ham, self._intermediates()
        kin = np.sum(inter.cs_overlap_K * lat.fft_forward(
            (inter.A_sq * ham.eps).sum(0))).real
        ph = np.sum(lat.fft_forward((_abs2(self.B) * ham.omega).sum(0))
                    * inter.A_sq_ft * self.eiKR * inter.cs_overlap).real
        return kin, self._eph_energy(inter), ph, inter.ovlp

    def get_energy(self):
        kin, eph, ph, ovlp = self.energy_terms()
        return (kin + eph + ph) / ovlp

    def _eph_energy(self, inter):
        return (self._eph_energy_svd(inter) if self.ham.use_svd
                else self._eph_energy_dense(inter))

    def _eph_energy_dense(self, inter):
        """Dense eph energy (the ``+ h.c.`` factor of 2 included)."""
        lat = self.lattice
        Bc_mq = np.conj(self.B)[:, lat.minus_k]    # B*_{nu,-q}
        AcD = np.conj(inter.A) * inter.D[None, :]  # A*_{nk} D_k
        A_kpq = inter.A[:, lat.kpq_map]
        return 2.0 * np.einsum("kqnji,nq,jk,ikq->", self.ham.g, Bc_mq, AcD,
                               A_kpq, optimize=True).real

    def _g_qnu(self, Aw_in, Aw_ft, conj_factors=False):
        """``sum_gij V[nu,g,i,j,q] IFFT[FFT[Sigma Aw_in] Aw_ft]|_{-q}`` --
        the coefficient of ``B*`` in the eph energy."""
        lat, ham = self.lattice, self.ham
        X = (np.conj(ham.Sigma) * np.conj(Aw_in)[None, None] if conj_factors
             else ham.Sigma * Aw_in[None, None])
        X = lat.fft_forward(X) * Aw_ft[None, :, None, :]
        conv = lat.fft_backward(X)[..., lat.minus_k]
        V = np.conj(ham.V) if conj_factors else ham.V
        return np.einsum("ngijq,gijq->nq", V, conv, optimize=True)

    def _eph_energy_svd(self, inter):
        """SVD eph energy; with ``herm`` the hermitian twin is averaged in."""
        lat, ham, mk = self.lattice, self.ham, self.lattice.minus_k
        Aw = ham.to_wannier(np.conj(inter.A))      # conj(A) in Wannier basis
        AwD = Aw * inter.D[None, :]                # Aw_ik D_k
        Aw_ft = lat.fft_forward(np.conj(Aw[:, mk]))
        val = np.sum(np.conj(self.B)[:, mk] * self._g_qnu(AwD, Aw_ft))
        if not self.herm:
            return 2.0 * val.real
        AwD_mq_ft = lat.fft_forward(AwD[:, mk])
        val_h = np.sum(np.conj(self.B)
                       * self._g_qnu(Aw, AwD_mq_ft, conj_factors=True))
        return (val + val_h).real

    def _acc(self):
        """Wirtinger derivative accumulators: ``dF_dB = dF/dB*`` etc."""
        z = lambda r: np.zeros((r, self.Nk), np.complex128)  # noqa: E731
        return SimpleNamespace(dF_dB=z(self.nph), dF_dA=z(self.nb),
                               dS_dB=z(self.nph), dS_dA=z(self.nb))

    def components(self, A=None, elec_only=False):
        """``(acc, F, S)`` -- Wirtinger derivative blocks plus the values.
        With ``elec_only`` only the A blocks are filled and ``F`` is None."""
        inter = self._intermediates(A)
        acc = self._acc()
        E_kin, E_ph = self._fft_terms(inter, acc, elec_only)
        E_eph = self._eph_grad(inter, acc, elec_only, projected=True)
        return acc, None if E_eph is None else E_kin + E_eph + E_ph, inter.ovlp

    def gradient(self):
        """Gradient of the energy, packed as [dRe B | dIm B | dRe A | dIm A]."""
        acc, F, S = self.components()
        wB = (acc.dF_dB - (F / S) * acc.dS_dB) / S
        wA = (acc.dF_dA - (F / S) * acc.dS_dA) / S
        return 2.0 * np.concatenate([wB.real.ravel(), wB.imag.ravel(),
                                     wA.real.ravel(), wA.imag.ravel()])

    def _fft_terms(self, inter, acc, elec_only):
        """Overlap, kinetic and phonon derivatives; returns (E_kin, E_ph)."""
        lat, ham, A = self.lattice, self.ham, inter.A

        acc.dS_dA += inter.D.real * A                                # overlap
        kin_ft = lat.fft_forward((inter.A_sq * ham.eps).sum(0))      # kinetic
        kin_j = self.eiKR * inter.cs_overlap * kin_ft
        acc.dF_dA += (ham.eps * inter.D[None, :]).real * A
        ph_ft = lat.fft_forward((_abs2(self.B) * ham.omega).sum(0))  # phonon
        ph_j = inter.A_sq_ft * inter.cs_overlap_K * ph_ft
        acc.dF_dA += lat.fft_forward(inter.cs_overlap * self.eiKR * ph_ft).real * A

        if not elec_only:
            ovlp_ft = lat.fft_forward(inter.A_sq_ft * inter.cs_overlap_K)
            acc.dS_dB += ovlp_ft.real * self.B
            acc.dF_dB += lat.fft_forward(kin_j).real * self.B
            omega_ovlp = (ham.omega * ovlp_ft[None, :]).real
            acc.dF_dB += (lat.fft_forward(ph_j).real[None, :] + omega_ovlp) * self.B
        return kin_j.sum().real, ph_j.sum().real

    def _eph_grad(self, inter, acc, elec_only, projected):
        """Dispatch the eph derivative terms; returns the eph energy."""
        if not self.ham.use_svd:
            return self._eph_grad_dense(inter, acc, elec_only, projected)
        E = None if elec_only else self._eph_energy_svd(inter)
        dF_dB, dF_dA = self._grad_svd_main(inter, elec_only, projected)
        if self.herm:
            hB, hA = self._grad_svd_herm(inter, elec_only, projected)
            dF_dB = None if dF_dB is None else 0.5 * (dF_dB + hB)
            dF_dA = 0.5 * (dF_dA + hA)
        if not elec_only:
            acc.dF_dB += dF_dB
        acc.dF_dA += dF_dA
        return E

    def _eph_grad_dense(self, inter, acc, elec_only, projected):
        """Dense-path eph derivatives; ``projected=False`` is the D2 variant
        (no ``dD/dB`` chain).  Returns the eph energy."""
        lat, g = self.lattice, self.ham.g
        mk, kpq, kmq = lat.minus_k, lat.kpq_map, lat.kmq_map
        A, D = inter.A, inter.D
        A_conj = np.conj(A)
        Bc_mq = np.conj(self.B)[:, mk]             # B*_{nu,-q}

        # gBA[j,k] = sum_{q nu i} g B*_{-q} A_{k+q};  gBAD adds A* D on (j,k)
        gBA = np.einsum("kqnji,nq,ikq->jk", g, Bc_mq, A[:, kpq],
                        optimize=True)
        gBAD = np.einsum("kqnji,nq,jk,k->ikq", g, Bc_mq, A_conj, D,
                         optimize=True)
        acc.dF_dA += gBA * D[None, :] + np.conj(
            gBAD[:, kmq, np.arange(self.Nk)[None, :]].sum(-1))
        E = 2.0 * np.einsum("jk,jk,k->", gBA, A_conj, D, optimize=True).real
        if elec_only:
            return E

        acc.dF_dB += np.einsum("kqnji,jk,k,ikq->nq", g[:, mk], A_conj, D,
                               A[:, kmq], optimize=True)
        if projected:                              # dD/dB chain
            pre = lat.fft_forward((gBA * A_conj).sum(0))
            acc.dF_dB += (2.0 * lat.fft_forward(
                inter.cs_overlap * self.eiKR * pre).real * self.B)
        return E

    def _grad_svd_main(self, inter, elec_only, projected):
        """Main-variant SVD eph gradient -> Wirtinger ``(dF_dB, dF_dA)``.

        B is contracted into V before the k-convolution (``VB_ft``) and reused
        by every term.  ``projected=False`` (D2) keeps the explicit-B* term
        alone -- the ``dD/dB`` chains do not exist there.
        """
        lat, ham, mk = self.lattice, self.ham, self.lattice.minus_k
        Aw = ham.to_wannier(np.conj(inter.A))      # conj(A) in Wannier basis
        AwD = Aw * inter.D[None, :]
        Aw_ft = lat.fft_forward(np.conj(Aw[:, mk]))
        Bc_mq = np.conj(self.B)[:, mk]

        VB_ft = lat.fft_forward(np.einsum("ngijq,nq->gijq", ham.V, Bc_mq, optimize=True))
        SigAD_ft = lat.fft_forward(ham.Sigma * AwD[None, None, :, :])
        VBA_k = lat.fft_backward(VB_ft * Aw_ft[None, :, None, :])[..., mk]
        SigVBA = np.einsum("gijk,gijk->jk", VBA_k, ham.Sigma, optimize=True)
        SigAVB_j = np.einsum("gijr,gijr->ir", SigAD_ft, VB_ft, optimize=True)

        dF_dA = (ham.from_wannier(SigVBA) * inter.D[None, :]
                 + np.conj(ham.from_wannier(lat.fft_backward(SigAVB_j), conjugate=True)))
        if elec_only:
            return None, dF_dA

        SigA_conv = lat.fft_backward(SigAD_ft * Aw_ft[None, :, None, :])[..., mk]
        dF_dB = np.einsum("ngijq,gijq->nq", ham.V, SigA_conv, optimize=True)[:, mk]
        if projected:                              # dD/dB chain
            dF_dB = dF_dB + (2.0 * self._dD_chain((SigVBA * Aw).sum(0), inter).real * self.B)
        return dF_dB, dF_dA

    def _grad_svd_herm(self, inter, elec_only, projected):
        """Hermitian-twin SVD eph gradient (not a plain conjugation of the
        main variant: the -k reflections and the absorbed factors differ)."""
        lat, ham, mk = self.lattice, self.ham, self.lattice.minus_k
        Aw = ham.to_wannier(np.conj(inter.A))
        AwD = Aw * inter.D[None, :]
        Bc_mq = np.conj(self.B)[:, mk]
        AwD_ft = lat.fft_forward(AwD)

        VB_ft = lat.fft_forward(np.einsum("ngijq,nq->gijq", np.conj(ham.V[..., mk]), Bc_mq, optimize=True))
        SigA_ft = lat.fft_forward((np.conj(ham.Sigma) * np.conj(Aw)[None, None, :, :])[..., mk])
        SigVB_j = np.einsum("gijr,gijr->ir", SigA_ft, VB_ft, optimize=True)
        VBA_p = lat.fft_backward(VB_ft * AwD_ft[None, :, None, :])
        SigVBA_p = np.einsum("gijp,gijp->jp", np.conj(ham.Sigma), VBA_p, optimize=True)
        SigVB_k = lat.fft_backward(SigVB_j)[..., mk]

        dF_dA = (ham.from_wannier(SigVB_k) * inter.D[None, :]
                 + np.conj(ham.from_wannier(SigVBA_p, conjugate=True)))

        if elec_only:
            return None, dF_dA

        SigA_conv = lat.fft_backward(SigA_ft * AwD_ft[None, :, None, :])  # no -q
        dF_dB = np.einsum("ngijq,gijq->nq", np.conj(ham.V), SigA_conv, optimize=True)

        if projected:                              # dD/dB chain
            dF_dB = dF_dB + (2.0 * self._dD_chain((SigVB_k * Aw).sum(0), inter).real * self.B)

        return dF_dB, dF_dA

    def _dD_chain(self, X_k, inter):
        """The ``dD/dB`` chain: FFT -> * cs_overlap e^{iK.Rj} -> FFT."""
        lat = self.lattice
        return lat.fft_forward(lat.fft_forward(X_k) * inter.cs_overlap * self.eiKR)

    def solve_electron(self, v0=None, trunc_eps=1.0e-12, tol=1.0e-10):
        """Exact electronic ground state at the current shifts ``B``.

        Solves ``H(B) A = E S A`` with the diagonal metric ``S_k = Re(D_k)``
        (identically 1 for D2): the operator is the Wirtinger block
        ``H A = dF/dA*`` from ``components``.  This is exact, because F is
        quadratic in A.  The operator is applied matrix-free on the active set
        ``S > trunc_eps * max(S)`` after the ``S^{-1/2}`` symmetrisation.
        Returns ``(E, A, n_matvec)`` with ``A (nb, Nk)`` normalised to
        ``A^H S A = 1``.
        """
        nb, Nk = self.nb, self.Nk
        S = self._intermediates(np.zeros_like(self.A)).D.real

        if not S.max() > 0.0:
            raise RuntimeError("electron metric max(Re D_k) is non-positive")

        active = np.flatnonzero(S > trunc_eps * S.max())
        sqrtS = np.sqrt(S[active])
        dim, n_mv = active.size * nb, [0]

        def embed(phi):
            A = np.zeros((nb, Nk), np.complex128)
            A[:, active] = phi.reshape(nb, -1) / sqrtS[None, :]
            return A

        def matvec(phi):
            n_mv[0] += 1
            acc = self.components(embed(np.asarray(phi, np.complex128)), elec_only=True)[0]
            return (acc.dF_dA[:, active] / sqrtS[None, :]).ravel()

        if v0 is not None:
            v0 = (np.asarray(v0, np.complex128)[:, active] * sqrtS[None, :]).ravel()
            v0 = v0 if np.linalg.norm(v0) > 1e-12 else None

        op = LinearOperator((dim, dim), matvec=matvec, dtype=complex)
        w, V = eigsh(op, k=1, which="SA", v0=v0, tol=tol)

        return float(w[0]), embed(V[:, 0]), n_mv[0]


class D2(dD2):
    """Unprojected Davydov D2 ansatz (arXiv:2605.05675 Eq. (3)) -- NOT dD2 at
    K = 0: the single j = 0 term, no cs_overlap, D_k = 1, S = sum |A|^2."""

    def __init__(self, shift, electron, ham, lattice=None, herm=True):
        super().__init__(shift, electron, ham, lattice, K=0.0, herm=herm)
        self.eiKR = np.ones(self.Nk)               # K is meaningless for D2

    def _intermediates(self, A=None):
        A = self.A if A is None else A
        A_sq = _abs2(A)
        cs_overlap = np.zeros(self.Nk, np.complex128)
        cs_overlap[0] = 1.0                        # -> delta_{j0}
        A_sq_ft = self.lattice.fft_forward(A_sq.sum(0).astype(complex))
        return SimpleNamespace(A=A, A_sq=A_sq, A_sq_ft=A_sq_ft,
                               cs_overlap=cs_overlap,
                               cs_overlap_K=cs_overlap,
                               D=np.ones(self.Nk, np.complex128),
                               ovlp=float(A_sq.sum()))

    def components(self, A=None, elec_only=False):
        """As for dD2; kinetic/phonon/overlap terms are pointwise (no FFTs)."""
        inter = self._intermediates(A)
        acc = self._acc()
        eps, omega = np.real(self.ham.eps), np.real(self.ham.omega)
        S, E_ph = inter.ovlp, float((omega * _abs2(self.B)).sum())

        acc.dS_dA += inter.A
        acc.dF_dA += (eps + E_ph) * inter.A
        if not elec_only:
            acc.dF_dB += omega * S * self.B
        E_eph = self._eph_grad(inter, acc, elec_only, projected=False)
        F = (None if E_eph is None
             else float((eps * inter.A_sq).sum()) + E_eph + S * E_ph)
        return acc, F, S


def optimize(prob, gtol=1.0e-8, maxiter=500, inner_tol=1.0e-10, verbose=False):
    from scipy.optimize import minimize

    nB = prob.nB
    state = {"key": None, "grad": None, "E": None,
             "warm": prob.A.copy(), "n_eig": 0, "n_mv": 0}

    def ensure(x):
        x = np.ascontiguousarray(x)
        if state["key"] == x.tobytes():
            return
        prob.B.real.flat[:] = x[:nB]
        prob.B.imag.flat[:] = x[nB:]
        E, A, n_mv = prob.solve_electron(v0=state["warm"], tol=inner_tol)
        prob.A = A
        state.update(key=x.tobytes(), E=E, warm=A, grad=None)
        state["n_eig"] += 1
        state["n_mv"] += n_mv
        if verbose:
            print(f"    eigensolve {state['n_eig']:4d}   E = {E:+.12f}")

    def fun(x):
        ensure(x)
        return state["E"]

    def jac(x):
        ensure(x)
        if state["grad"] is None:
            state["grad"] = prob.gradient()[:2 * nB]
        return state["grad"]

    x0 = np.concatenate([prob.B.real.ravel(), prob.B.imag.ravel()])
    res = minimize(fun, x0, jac=jac, method="L-BFGS-B",
                   options={"maxiter": int(maxiter), "gtol": float(gtol),
                            "ftol": 1.0e-14})
    ensure(res.x)                                  # leave prob at the solution
    prob.result, prob.n_eigensolves, prob.n_matvecs = res, state["n_eig"], state["n_mv"]
    return prob


if __name__ == "__main__":

    H5 = "/n/home01/mbaumgarten/Software/qcpbc/libpbc/samples/svd_kq.h5"
    inp = read_svd_kq(H5)
    zero_acoustic_modes(inp.omega, inp.V, cutoff=1e-3)

    lat = Lattice(np.eye(3), [5, 5, 5], 1.0)
    ham = Hamiltonian(inp.eps, inp.omega, lat, inp.Sigma, inp.V, inp.U, inp.L)

    rng = np.random.default_rng(125)

    shift0 = rng.standard_normal((inp.Nk, inp.nph)) * (1 + 0j) + 1j * rng.standard_normal((inp.Nk, inp.nph))
    shift0[inp.omega.real.T < 1e-3] = 0.0
    shift0 *= np.sqrt(1.5 / np.sum(np.abs(shift0) ** 2))  # keep cs_overlap bounded

    electron0 = rng.standard_normal((inp.Nk, inp.nb)) + 1j * rng.standard_normal((inp.Nk, inp.nb))
    electron0 /= np.linalg.norm(electron0)

    prob_dd2 = dD2(shift0, electron0, ham, lat, K=0.0, herm=True)
    obj_dd2 = optimize(prob_dd2)
    print(f"dD2: E = {obj_dd2.get_energy():+.12f}, {obj_dd2.result.nit} outer its")

    prob_d2 = D2(shift0, electron0, ham, lat, herm=True)
    obj_d2 = optimize(prob_d2)
    print(f"D2: E = {obj_d2.get_energy():+.12f}, {obj_d2.result.nit} outer its")
