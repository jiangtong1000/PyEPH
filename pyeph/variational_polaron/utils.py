# pre/post-processing for the variational polaron workflow. Everything works
# on the SimpleNamespace from mini_dd2.read_svd_kq:
#   eps (nb, Nk), omega (nph, Nq), Sigma (Nc, nb, nb, Nk),
#   V (nph, Nc, nb, nb, Nq), U (Nk, nb, nb), L (nph, Nq)
# dense g is indexed g[ik, iq, inu, jb, ib], jb = k side, ib = k+q side.
import numpy as np

from pyeph.utils.grid import uniform_grid_zfast, minus_index_map

__all__ = [
    "read_epr_cell",
    "zero_acoustic_modes",
    "hole_transformation",
    "reconstruct_g_from_svd",
    "hermitianize_g",
    "kpq_index_map",
]


def read_epr_cell(epr_file):
    """Lattice vectors (rows, units of alat) and alat from a perturbo epr.h5."""
    import h5py
    with h5py.File(epr_file, "r") as f:
        at = f["basic_data/at"][:]
        alat = float(f["basic_data/alat"][()])
    return at, alat


def kpq_index_map(kmesh):
    """(Nk, Nk) table of the flat index of k+q on the z-fastest grid."""
    m = np.asarray(kmesh, dtype=np.int64).ravel()
    _, grid = uniform_grid_zfast(m)
    idx = np.mod(grid[:, None, :] + grid[None, :, :], m)
    return (idx[..., 0] * m[1] + idx[..., 1]) * m[2] + idx[..., 2]


def zero_acoustic_modes(inp, cutoff=1.0e-3):
    """
    Zero out modes with omega < cutoff: frequency, Vq slice AND the long-range
    term, so they contribute neither elastic nor coupling energy. (L is usually
    already zero on this mask if the file was made with the standard 1 meV
    phfreq_cutoff, but it must never survive on a zeroed mode.)
    In place, returns the (nph, Nq) mask.
    """
    mask = inp.omega.real < cutoff
    inp.omega[mask] = 0.0
    inu, iq = np.nonzero(mask)
    inp.V[inu, :, :, :, iq] = 0.0
    inp.L[mask] = 0.0
    return mask


def hole_transformation(inp, kmesh):
    """
    Electron -> hole picture, in place. Same as hole_transformation_SVD in the
    production drivers plus the ekj *= -1 they do next to it:

        eps'   = -eps
        Sigma' = -conj(Sigma[..., -k])
        V'     =  conj(V[..., -q])
        U'     =  conj(U[-k])
        L'     = -conj(L[:, -q])

    omega is even in q, untouched. Applying this twice gives back the input.
    """
    mk = minus_index_map(kmesh)
    inp.eps = -inp.eps
    inp.Sigma = -np.conj(inp.Sigma[..., mk])
    inp.V = np.conj(inp.V[..., mk])
    inp.U = np.conj(inp.U[mk])
    inp.L = -np.conj(inp.L[:, mk])
    return inp


def reconstruct_g_from_svd(inp, kmesh):
    """
    Dense Bloch-gauge g from the SVD factors, long-range included:
        g^W_ij,nu(k,q) = sum_g Sigma[g,i,j,k] V[nu,g,i,j,q] + delta_ij L[nu,q]
        g^H = U_{k+q}^dag g^W U_k
    Returns g[ik, iq, inu, jb, ib]. Memory goes as Nk^2, only for small meshes.
    """
    nb, Nk = inp.eps.shape
    nph = inp.omega.shape[0]
    kpq = kpq_index_map(kmesh)

    g_wann = np.einsum("ngijq,gijk->nqkij", inp.V, inp.Sigma, optimize=True)
    for i in range(nb):
        g_wann[:, :, :, i, i] += inp.L[:, :, None]

    # columns of uk are the eigenvectors: uk[wann, band] = U[k].T
    M = np.ascontiguousarray(inp.U.transpose(0, 2, 1))
    Mdag = np.conj(M).transpose(0, 2, 1)

    g = np.empty((Nk, Nk, nph, nb, nb), dtype=np.complex128)
    for iq in range(Nk):
        blk = np.einsum("kai,nkij,kjb->nkab", Mdag[kpq[:, iq]],
                        g_wann[:, iq], M, optimize=True)
        g[:, iq] = blk.transpose(1, 0, 3, 2)  # -> [ik, inu, jb, ib]
    return g


def hermitianize_g(g, kmesh):
    """
    g <- (g + hermitian twin)/2 with twin = conj(g[k+q, -q, nu, ib, jb]).
    Exact g fulfills g == twin, a truncated-SVD reconstruction only up to the
    truncation error; this restores it (dense-path analogue of herm=True).
    Returns a new array.
    """
    mk = minus_index_map(kmesh)
    kpq = kpq_index_map(kmesh)
    twin = np.conj(g[kpq, mk[None, :]]).transpose(0, 1, 2, 4, 3)
    return 0.5 * (g + twin)
