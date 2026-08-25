#!/usr/bin/env python
# run dD2/D2 on an svd_kq.h5 (from perturbo svd-elph or make_svd_kq.py).
# Copy next to your run, edit the parameters, run. mini_dd2.py is untouched,
# the preprocessing lives in utils.py.
import numpy as np

from pyeph.variational_polaron.mini_dd2 import (
    Lattice, Hamiltonian, dD2, D2, optimize, read_svd_kq)
from pyeph.variational_polaron.utils import (
    read_epr_cell, zero_acoustic_modes, hole_transformation)

H5 = "gkq/svd_kq.h5"
KMESH = [5, 5, 5]
EPR = "lif_epwan.h5"     # for the lattice vectors, only matters at K != 0;
                         # set to None for the identity lattice
HOLE = True              # hole picture (eps -> -eps etc.)
CUTOFF_MEV = 1.0e-3
K = 0.0
SEED = 125
SHIFT_NORM = 1.5         # initial sum |B|^2, keeps cs_overlap bounded
MAXITER = 500
GTOL = 1.0e-8
SAVE = None              # e.g. "result.npz"

inp = read_svd_kq(H5)

zero_acoustic_modes(inp, cutoff=CUTOFF_MEV)
if HOLE:
    hole_transformation(inp, KMESH)

at, alat = read_epr_cell(EPR) if EPR is not None else (np.eye(3), 1.0)
lat = Lattice(at, KMESH, alat)
ham = Hamiltonian(inp.eps, inp.omega, lat, inp.Sigma, inp.V, inp.U, inp.L)

rng = np.random.default_rng(SEED)

shift0 = rng.standard_normal((inp.Nk, inp.nph)) + 1j * rng.standard_normal((inp.Nk, inp.nph))
shift0[inp.omega.real.T < CUTOFF_MEV] = 0.0
shift0 *= np.sqrt(SHIFT_NORM / np.sum(np.abs(shift0) ** 2))

electron0 = rng.standard_normal((inp.Nk, inp.nb)) + 1j * rng.standard_normal((inp.Nk, inp.nb))
electron0 /= np.linalg.norm(electron0)

out = {}
for name, prob in (("dD2", dD2(shift0, electron0, ham, lat, K=K, herm=True)),
                   ("D2", D2(shift0, electron0, ham, lat, herm=True))):
    obj = optimize(prob, gtol=GTOL, maxiter=MAXITER)
    E = obj.get_energy()
    print(f"{name}: E = {E:+.12f} meV ({obj.result.nit} outer its, "
          f"converged={obj.result.success})")
    B, A = obj.get_params()
    out[f"E_{name}"] = E
    out[f"B_{name}"], out[f"A_{name}"] = B, A

if SAVE:
    np.savez(SAVE, **out)
    print(f"saved to {SAVE}")
