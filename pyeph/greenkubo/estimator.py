import scipy.sparse
import numpy
from numba import njit
# from line_profiler import profile

@njit
def sum_current_jit(u_t, w, jt_data, indices, indptr, n_rows):
    s = 0.0
    for i in range(n_rows):
        row_start = indptr[i]
        row_end = indptr[i+1]
        for k in range(row_start, row_end):
            j = indices[k]
            s = s + jt_data[k] * numpy.dot(u_t[j], w[i])
    return s

def current_from_density_no_polaron(
    jrho0T: numpy.ndarray,
    u_t: numpy.ndarray,
    j_t: scipy.sparse.csr_matrix,
) -> numpy.ndarray:
    """
    Compute the current without polaron transformation.
    C(t) = -Tr[J(t) U J(0) rho_0 U^dagger]
    """
    # Debugging purpose
    # j_t_dense = j_t.toarray()
    # c_t = -numpy.einsum("ij, jk, ki->", j_t_dense, u_t, jrho0T.T, optimize=True)
    w = u_t.conj() @ jrho0T
    c_t = -sum_current_jit(u_t, w, j_t.data, j_t.indices, j_t.indptr, j_t.shape[0])
    return c_t

def _prepare_sector_arrays(sectors, F0):
    """
    Flatten sector dict into arrays friendly to Numba.
    Returns quad_idx (N,4), F0_vals (N,), sector_offsets (6,)
    with sectors ordered as [-2, -1, 0, 1, 2].
    """
    quad_list = []
    f0_vals = []
    offsets = [0]
    for n in (-2, -1, 0, 1, 2):
        quad_n = sectors[n]
        quad_list.extend(quad_n)
        for q in quad_n:
            f0_vals.append(F0[q])
        offsets.append(len(quad_list))
    quad_idx = numpy.asarray(quad_list, dtype=numpy.int32)
    F0_vals = numpy.asarray(f0_vals, dtype=numpy.float64)
    sector_offsets = numpy.asarray(offsets, dtype=numpy.int32)
    return quad_idx, F0_vals, sector_offsets

def _stack_to_dense(mats, dtype=numpy.complex128):
    """
    Stack a list of csr/array matrices into a dense (n, nsites, nsites) array.
    """
    dense_list = []
    for M in mats:
        arr = M.toarray() if hasattr(M, "toarray") else numpy.asarray(M)
        dense_list.append(numpy.asarray(arr, dtype=dtype))
    return numpy.ascontiguousarray(numpy.stack(dense_list), dtype=dtype)

@njit
def _current_from_density_polaron_kernel(u_t, G, jt_dense, j0_dense, quad_idx, sec_offsets, sec_weights, F0_vals):
    ntraj = u_t.shape[0]
    ct = numpy.empty(ntraj, dtype=numpy.complex128)
    for itraj in range(ntraj):
        um = u_t[itraj]
        gm = G[itraj]
        jt = jt_dense[itraj]
        j0 = j0_dense[itraj]
        total_itraj = 0.0 + 0.0j
        for s in range(5):
            start = sec_offsets[s]
            end = sec_offsets[s + 1]
            w_n = sec_weights[s]
            accu_n = 0.0 + 0.0j
            for idx in range(start, end):
                i = quad_idx[idx, 0]
                j = quad_idx[idx, 1]
                k = quad_idx[idx, 2]
                l = quad_idx[idx, 3]
                accu_n += jt[i, j] * j0[k, l] * um[j, k] * gm[i, l] * F0_vals[idx]
            total_itraj += accu_n * w_n
        ct[itraj] = -total_itraj
    return ct

def _current_from_density_polaron_python(u_t, rho0, j_t_list, j_0_list, F0, sectors, sec_weights):
    u_t_conj = numpy.conjugate(u_t)
    G = numpy.einsum("min, mln->mil", u_t_conj, rho0)
    ntraj = u_t.shape[0]
    ct = numpy.zeros(ntraj, dtype=numpy.complex128)
    for itraj in range(ntraj):
        jt = j_t_list[itraj]
        j0 = j_0_list[itraj]
        um = u_t[itraj]
        gm = G[itraj]
        
        total_itraj = 0.0 + 0.0j
        for n, quad_list in sectors.items():
            w_n = sec_weights[n+2]
            accu_n = 0.0 + 0.0j
            for (i, j, k, l) in quad_list:
                jt_ij = jt[i, j]
                j0_kl = j0[k, l]
                accu_n += jt_ij * j0_kl * um[j, k] * gm[i, l] * F0[(i, j, k, l)]
            total_itraj += accu_n * w_n
        ct[itraj] = -total_itraj
    return ct

# @profile
def current_from_density_polaron(
    u_t,
    rho0,
    j_t_list,
    j_0_list,
    F0,
    sectors,
    sec_weights,
    use_python=False,
    quad_idx=None,
    F0_vals=None,
    sector_offsets=None,
):
    """
    u_t: (ntraj, nsites, nsites)
    rho0: (ntraj, nsites, nsites)
    j_t_list: list of scipy.sparse.csr_matrix, length ntraj
    j_0_list: list of scipy.sparse.csr_matrix, length ntraj
    F0: dict of float, keyed by (i, j, k, l)
    sectors: dict of lists of tuples, length 5
    sec_weights: numpy.ndarray, (5,)
    use_python: force the original Python implementation for debugging
    quad_idx, F0_vals, sector_offsets: optional pre-flattened sector data
    """
    if use_python:
        return _current_from_density_polaron_python(u_t, rho0, j_t_list, j_0_list, F0, sectors, sec_weights)

    if quad_idx is None or F0_vals is None or sector_offsets is None:
        quad_idx, F0_vals, sector_offsets = _prepare_sector_arrays(sectors, F0)
    # Preserve possible complex weights; do not downcast.
    sec_w = numpy.asarray(sec_weights, dtype=numpy.complex128)

    j0_dense = _stack_to_dense(j_0_list)
    jt_dense = _stack_to_dense(j_t_list)

    u_t_c = numpy.ascontiguousarray(u_t)
    rho0_c = numpy.ascontiguousarray(rho0)
    G = numpy.einsum("min, mln->mil", numpy.conjugate(u_t_c), rho0_c)

    return _current_from_density_polaron_kernel(
        u_t_c, G, jt_dense, j0_dense, quad_idx, sector_offsets, sec_w, F0_vals
    )
    
def get_sectors_for_polaron_transform(nzidx_hopping):
    """
    nzidx_hopping : array_like of shape (n_hop, 2)
        List/array of (i, j) integer indices for nonzero hoppings.

    """
    nzidx_array = numpy.asarray(nzidx_hopping, dtype=numpy.int32)
    if nzidx_array.ndim != 2 or nzidx_array.shape[1] != 2:
        raise ValueError("nzidx_hopping must be of shape (n_hop, 2)")
    
    n_hop = len(nzidx_array)
    i_vals = nzidx_array[:, 0]
    j_vals = nzidx_array[:, 1]
    
    sectors = {-2: [], -1: [], 0: [], 1: [], 2: []}
    
    for idx1 in range(n_hop):
        i, j = int(i_vals[idx1]), int(j_vals[idx1])
        for idx2 in range(n_hop):
            k, l = int(i_vals[idx2]), int(j_vals[idx2])
            val = (i == k) - (j == k) - (i == l) + (j == l)
            if val not in sectors:
                raise ValueError(f"Invalid value: {val}")
            sectors[val].append((i, j, k, l))
    
    return sectors
