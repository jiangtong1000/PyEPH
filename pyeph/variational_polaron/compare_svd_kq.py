#!/usr/bin/env python
# compare two svd_kq.h5 files (pyeph vs perturbo svd-elph) on gauge invariants.
# Raw complex datasets will NOT match: eigensolver phases and rotations inside
# degenerate subspaces (bands, phonon modes, degenerate sigma^2 channels) are
# arbitrary. What has to match: bands/phfreq elementwise, |longrange|, and the
# reconstructed sum_ij |g^H|^2 grouped over degenerate phonon modes.
import numpy as np
import h5py

from pyeph.utils.grid import uniform_grid_zfast

FILE_A = "svd_kq_pyeph.h5"
FILE_B = "gkq/svd_kq.h5"   # reference
KMESH = [5, 5, 5]
SAMPLES = 300              # random (k, q) pairs for the |g|^2 check
SEED = 0
WTOL_MEV = 1.0e-3          # phonon degeneracy grouping


def load(fname):
    d = {}
    with h5py.File(fname, "r") as f:
        for k in f:
            d[k] = f[k][()]
    for stem in ("Uk", "Vq", "Uk_bloch", "longrange", "phmodes"):
        d[stem] = d.pop(stem + "_real") + 1j * d.pop(stem + "_imag")
    return d


def grouped_g2(d, iks, iqs, ikqs, wtol_mev):
    # sum_ij |g^H|^2 per mode, averaged over degenerate mode groups
    Uk, Vq, Ub, L, w = d["Uk"], d["Vq"], d["Uk_bloch"], d["longrange"], d["phfreq"]
    nb = Ub.shape[1]
    out = np.zeros((len(iks), Vq.shape[0]))
    for n, (ik, iq, ikq) in enumerate(zip(iks, iqs, ikqs)):
        gW = np.einsum("gij,ngij->nij", Uk[..., ik], Vq[..., iq], optimize=True)
        gW = gW + L[iq][:, None, None] * np.eye(nb)[None]
        uk = Ub[ik].T   # columns are the eigenvectors
        ukq = Ub[ikq].T
        gH = np.einsum("ai,nij,jb->nab", np.conj(ukq).T, gW, uk, optimize=True)
        s = np.sum(np.abs(gH) ** 2, axis=(1, 2))
        grp = np.concatenate([[0], np.cumsum(np.abs(np.diff(w[iq])) > wtol_mev)])
        for g_ in range(grp[-1] + 1):
            m = grp == g_
            s[m] = s[m].sum() / m.sum()
        out[n] = s
    return out


A, B = load(FILE_A), load(FILE_B)
m = np.asarray(KMESH)
_, grid = uniform_grid_zfast(m)
Nk = len(grid)
assert A["bands"].shape[0] == Nk == B["bands"].shape[0], "KMESH mismatch"

for k in ("bands", "phfreq"):
    print(f"{k:12s}: max |dev| = {np.max(np.abs(A[k] - B[k])):.3e} meV")
dev = np.max(np.abs(np.abs(A["longrange"]) - np.abs(B["longrange"])))
print(f"|longrange| : max |dev| = {dev:.3e} meV")

rng = np.random.default_rng(SEED)
iks = rng.integers(0, Nk, size=SAMPLES)
iqs = rng.integers(0, Nk, size=SAMPLES)
idx = np.mod(grid[iks] + grid[iqs], m)
ikqs = (idx[:, 0] * m[1] + idx[:, 1]) * m[2] + idx[:, 2]

sA = grouped_g2(A, iks, iqs, ikqs, WTOL_MEV)
sB = grouped_g2(B, iks, iqs, ikqs, WTOL_MEV)
scale = max(sB.max(), 1e-300)
print(f"grouped band-summed |g|^2 on {SAMPLES} sampled (k,q): "
      f"max dev = {np.max(np.abs(sA - sB)):.4e} "
      f"(scale {scale:.4e}, rel {np.max(np.abs(sA - sB))/scale:.3e})")
