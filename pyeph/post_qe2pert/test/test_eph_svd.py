"""Tests for the SVD e-ph pipeline (eph_svd.py) on the DNTT test data."""
from pathlib import Path

import numpy as np
import pytest

from pyeph.post_qe2pert.eph_svd import EphSVD
from pyeph.utils.constants import ryd_to_mev
from pyeph.utils.grid import uniform_grid_zfast, minus_index_map

EPR = Path(__file__).resolve().parent / "DNTT_epr.h5"
KMESH = [2, 2, 2]


@pytest.fixture(scope="module")
def ep():
    obj = EphSVD(str(EPR), polar=False, verbose=False)
    obj.build_svd_factors(nsvd=None)  # full rank
    return obj


@pytest.fixture(scope="module")
def dump(ep, tmp_path_factory):
    fname = tmp_path_factory.mktemp("svd") / "svd_kq.h5"
    out = ep.dump_svd_kq(str(fname), KMESH, nsvd=None, phfreq_cutoff_mev=1.0)
    return fname, out


def _vq_cart(ep, qpts):
    expqr = np.exp(2j * np.pi * (qpts @ ep.rvec_set_ph_eph.T))
    return np.einsum('ijgap,qp->ijgaq', ep.Vr, expqr, optimize=True)


def test_svd_reconstruction_and_singular_values(ep):
    # full rank U (x) V must give back G, and sigma must match lapack
    nb = ep.num_wann
    for ib in range(nb):
        for jb in range(nb):
            G = ep.assemble_g_rerp(ib, jb)
            nre = G.shape[0]
            U = ep.Ur[:, :, ib, jb]
            V = ep.Vr[ib, jb].reshape(nre, -1)
            rel = np.linalg.norm(G - U @ V) / np.linalg.norm(G)
            assert rel < 1e-10, (ib, jb, rel)

            s_ref = np.linalg.svd(G, compute_uv=False)
            s_ours = np.sqrt(np.clip(ep.S2[:, ib, jb], 0.0, None))
            well = s_ref > 1e-7 * s_ref[0]
            assert np.allclose(s_ours[well], s_ref[well], rtol=1e-6), (ib, jb)


def test_formfactor_vs_direct_ft(ep):
    # sum_g Uk(g) Vq_cart(g) == direct double FT of the raw tensor
    kpts, _ = uniform_grid_zfast(KMESH)
    xk, xq = kpts[3], kpts[5]
    Uk = ep.cal_uk_formf(np.array([xk]))[..., 0]
    Vqc = _vq_cart(ep, np.array([xq]))[..., 0]
    gW = np.einsum('gij,ijga->ija', Uk, Vqc, optimize=True)

    expk = np.exp(2j * np.pi * (ep.rvec_set_el_svd @ xk))
    expq = np.exp(2j * np.pi * (ep.rvec_set_ph_eph @ xq))
    nrp = len(ep.rvec_set_ph_eph)
    nmod = 3 * ep.nat
    for ib in range(ep.num_wann):
        for jb in range(ep.num_wann):
            G = ep.assemble_g_rerp(ib, jb).reshape(-1, nmod, nrp)
            direct = np.einsum('e,eap,p->a', expk, G, expq, optimize=True)
            assert np.allclose(gW[ib, jb], direct, atol=1e-10 * np.abs(direct).max())


def test_wannier_hermiticity_on_mesh(ep):
    # conj(gW_ij,a(k,q)) = gW_ji,a(k+q,-q) on the commensurate mesh; this
    # tests the mirrored electron WS cells / conj handling for iw > jw
    kpts, grid = uniform_grid_zfast(KMESH)
    minus = minus_index_map(KMESH)
    nk = len(kpts)
    m = np.asarray(KMESH)
    kpq = (grid[:, None, :] + grid[None, :, :]) % m
    kpq = (kpq[..., 0] * m[1] + kpq[..., 1]) * m[2] + kpq[..., 2]

    Uk = ep.cal_uk_formf(kpts)
    Vqc = _vq_cart(ep, kpts)
    gW = np.einsum('gijk,ijgaq->kqija', Uk, Vqc, optimize=True)

    scale = np.abs(gW).max()
    for ik in range(nk):
        for iq in range(nk):
            lhs = np.conj(gW[ik, iq])
            rhs = gW[kpq[ik, iq], minus[iq]].transpose(1, 0, 2)
            assert np.allclose(lhs, rhs, atol=1e-9 * scale), (ik, iq)


def test_svd_kq_file_contents(ep, dump):
    from pyeph.variational_polaron.mini_dd2 import read_svd_kq

    fname, out = dump
    inp = read_svd_kq(str(fname))
    nmod = 3 * ep.nat
    Nk = int(np.prod(KMESH))
    assert (inp.nb, inp.Nk, inp.nph, inp.Nc) == (ep.num_wann, Nk, nmod, ep.Nc)

    # units, meV
    enk, _ = ep.electron_calc.solve_eigenvalue_vector(np.zeros(3))
    assert np.allclose(inp.eps[:, 0], enk * ryd_to_mev, atol=1e-8)
    wqt, _ = ep.phonon_calc.solve_phonon_modes(ep.force_constants, np.zeros(3))
    assert np.allclose(inp.omega[:, 0], wqt * ryd_to_mev, atol=1e-6)

    # E(-q) = conj(E(q)) after the gauge fix
    minus = minus_index_map(KMESH)
    modes = out['phmodes']
    for iq in range(Nk):
        jq = minus[iq]
        if jq == iq:
            continue
        assert np.allclose(modes[jq], np.conj(modes[iq]), atol=1e-10)
    assert np.allclose(out['phfreq'][minus], out['phfreq'], atol=1e-12)


def test_bloch_hermiticity_and_dense_reference(ep, dump):
    from pyeph.variational_polaron.mini_dd2 import (
        Lattice, Hamiltonian, read_svd_kq)

    fname, out = dump
    inp = read_svd_kq(str(fname))
    lat = Lattice(np.eye(3), KMESH, 1.0)
    ham = Hamiltonian(inp.eps, inp.omega, lat, inp.Sigma, inp.V, inp.U, inp.L)
    g = ham.reconstruct_g_from_svd()  # [ik, iq, nu, jb, ib]
    Nk = lat.Nk

    # conj(g[k, q, nu, b, a]) = g[k+q, -q, nu, a, b]
    scale = np.abs(g).max()
    assert scale > 0
    for ik in range(Nk):
        for iq in range(Nk):
            lhs = np.conj(g[ik, iq])
            rhs = g[lat.kpq_map[ik, iq], lat.minus_k[iq]].transpose(0, 2, 1)
            assert np.allclose(lhs, rhs, atol=1e-8 * scale), (ik, iq)

    # band-summed |g|^2 per mode against the dense path (that sum is invariant
    # under the iw>jw conj convention of the dense path, so it's comparable)
    kpts, _ = uniform_grid_zfast(KMESH)
    cutoff_ry = 1.0 / ryd_to_mev
    for ik, iq in [(1, 3), (2, 6), (0, 5)]:
        ikq = lat.kpq_map[ik, iq]
        g_kerp = ep.eph_fourier_el_para(kpts[ik])
        gkq = ep.eph_fourier_elph(kpts[iq], g_kerp)
        gkq = ep.eph_transform(kpts[iq], out['phmodes'][iq],
                               out['Uk_bloch'][ik], out['Uk_bloch'][ikq], gkq)
        w = out['phfreq'][iq]
        fac = np.where(w > cutoff_ry,
                       np.sqrt(0.5 / np.where(w > cutoff_ry, w, 1.0))
                       * ryd_to_mev / np.sqrt(Nk), 0.0)
        gkq = gkq * fac[None, None, :]
        ref = np.sum(np.abs(gkq)**2, axis=(0, 1))
        ours = np.sum(np.abs(g[ik, iq])**2, axis=(1, 2))
        assert np.allclose(ours, ref, rtol=1e-8, atol=1e-10 * max(ref.max(), 1)), (ik, iq)


def test_mode_phase_hook(ep, dump, tmp_path):
    # per-(q,nu) gauge must leave |g| invariants alone and keep TRS
    from pyeph.variational_polaron.mini_dd2 import (
        Lattice, Hamiltonian, read_svd_kq)

    _, out = dump
    Nk = int(np.prod(KMESH))
    nmod = 3 * ep.nat
    minus = minus_index_map(KMESH)

    rng = np.random.default_rng(3)
    theta = rng.uniform(-np.pi, np.pi, size=(Nk, nmod))
    for iq in range(Nk):
        jq = minus[iq]
        if jq == iq:
            theta[iq] = 0.0
        elif jq > iq:
            theta[jq] = -theta[iq]
    phases = np.exp(1j * theta)

    fname2 = tmp_path / "svd_kq_phased.h5"
    out2 = ep.dump_svd_kq(str(fname2), KMESH, nsvd=None,
                          phfreq_cutoff_mev=1.0, mode_phases=phases)

    modes = out2['phmodes']
    for iq in range(Nk):
        jq = minus[iq]
        if jq != iq:
            assert np.allclose(modes[jq], np.conj(modes[iq]), atol=1e-10)

    assert np.allclose(np.abs(out2['Vq']), np.abs(out['Vq']), atol=1e-10)
    lat = Lattice(np.eye(3), KMESH, 1.0)
    inp2 = read_svd_kq(str(fname2))
    g2 = Hamiltonian(inp2.eps, inp2.omega, lat, inp2.Sigma, inp2.V, inp2.U,
                     inp2.L).reconstruct_g_from_svd()
    s2 = np.sum(np.abs(g2)**2, axis=(3, 4))
    inp1 = read_svd_kq(str(dump[0]))
    g1 = Hamiltonian(inp1.eps, inp1.omega, lat, inp1.Sigma, inp1.V, inp1.U,
                     inp1.L).reconstruct_g_from_svd()
    s1 = np.sum(np.abs(g1)**2, axis=(3, 4))
    assert np.allclose(s1, s2, rtol=1e-10, atol=1e-10 * max(s1.max(), 1.0))


def test_utils_reconstruct_hermitize_hole(dump):
    import copy
    from pyeph.variational_polaron import utils
    from pyeph.variational_polaron import mini_dd2
    from pyeph.variational_polaron.mini_dd2 import (
        Lattice, Hamiltonian, read_svd_kq)

    fname, _ = dump
    inp = read_svd_kq(str(fname))
    g_utils = utils.reconstruct_g_from_svd(inp, KMESH)

    inp2 = read_svd_kq(str(fname))
    lat = Lattice(np.eye(3), KMESH, 1.0)
    g_ref = Hamiltonian(inp2.eps, inp2.omega, lat, inp2.Sigma, inp2.V,
                        inp2.U, inp2.L).reconstruct_g_from_svd()
    assert np.allclose(g_utils, g_ref, atol=1e-12 * np.abs(g_ref).max())

    # full rank -> already hermitian, hermitianize is a no-op and idempotent
    g_h = utils.hermitianize_g(g_utils, KMESH)
    assert np.allclose(g_h, g_utils, atol=1e-9 * np.abs(g_utils).max())
    assert np.allclose(utils.hermitianize_g(g_h, KMESH), g_h,
                       atol=1e-12 * np.abs(g_h).max())

    # hole transformation is an involution and keeps hermiticity
    orig = copy.deepcopy(inp)
    utils.hole_transformation(inp, KMESH)
    g_hole = utils.reconstruct_g_from_svd(inp, KMESH)
    assert np.allclose(utils.hermitianize_g(g_hole, KMESH), g_hole,
                       atol=1e-9 * np.abs(g_hole).max())
    utils.hole_transformation(inp, KMESH)
    for attr in ("eps", "Sigma", "V", "U", "L"):
        assert np.allclose(getattr(inp, attr), getattr(orig, attr), atol=1e-14)

    # namespace zero_acoustic_modes vs the mini_dd2 original (with L passed
    # there too, the utils one always zeroes the longrange)
    a, b = read_svd_kq(str(fname)), read_svd_kq(str(fname))
    mask_a = utils.zero_acoustic_modes(a, cutoff=1e-3)
    mask_b = mini_dd2.zero_acoustic_modes(b.omega, b.V, L=b.L, cutoff=1e-3)
    assert np.array_equal(mask_a, mask_b)
    assert np.allclose(a.omega, b.omega) and np.allclose(a.V, b.V)
    assert np.allclose(a.L, b.L)
    assert np.all(a.L[mask_a] == 0.0)


def test_mini_dd2_svd_vs_dense_energy(dump):
    from pyeph.variational_polaron.mini_dd2 import (
        Lattice, Hamiltonian, dD2, D2, read_svd_kq)
    from pyeph.variational_polaron.utils import zero_acoustic_modes

    fname, _ = dump
    inp = read_svd_kq(str(fname))
    zero_acoustic_modes(inp, cutoff=1e-3)
    lat = Lattice(np.eye(3), KMESH, 1.0)
    ham_svd = Hamiltonian(inp.eps, inp.omega, lat, inp.Sigma, inp.V, inp.U, inp.L)
    ham_dense = Hamiltonian(inp.eps, inp.omega, lat, inp.Sigma, inp.V, inp.U, inp.L)
    ham_dense.reconstruct_g_from_svd()
    assert ham_svd.use_svd and not ham_dense.use_svd

    rng = np.random.default_rng(7)
    shift0 = (rng.standard_normal((inp.Nk, inp.nph))
              + 1j * rng.standard_normal((inp.Nk, inp.nph)))
    shift0[inp.omega.real.T < 1e-3] = 0.0
    shift0 *= np.sqrt(1.5 / np.sum(np.abs(shift0)**2))
    electron0 = (rng.standard_normal((inp.Nk, inp.nb))
                 + 1j * rng.standard_normal((inp.Nk, inp.nb)))
    electron0 /= np.linalg.norm(electron0)

    for cls in (dD2, D2):
        e_svd = cls(shift0, electron0, ham_svd).get_energy()
        e_dense = cls(shift0, electron0, ham_dense).get_energy()
        assert np.isfinite(e_svd)
        assert abs(e_svd - e_dense) < 1e-8 * max(1.0, abs(e_dense)), cls.__name__
