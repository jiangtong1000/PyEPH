"""Legacy current-sign adapters around JAX trace contractions.

Historical current arrays omit i. This module consequently returns the minus
sign from multiplying two such currents, while native transport uses physical
Hermitian currents and no extra minus sign.
"""

import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp

from pyeph.observables.transport.polaron import build_lf_sectors
from ._precision import require_legacy_precision


def _dense(matrix):
    return np.asarray(matrix.toarray() if sp.issparse(matrix) else matrix)


def _stack_to_dense(mats, dtype=np.complex128):
    return np.asarray([_dense(matrix) for matrix in mats], dtype=dtype)


def sum_current_jit(u_t, w, jt_data, indices, indptr, n_rows):
    """Historical name; conversion is host-only, contraction is native JAX."""
    require_legacy_precision()
    matrix = sp.csr_matrix((jt_data, indices, indptr), shape=(n_rows, n_rows))
    return jnp.einsum("ij,jk,ik->", jnp.asarray(matrix.toarray()), jnp.asarray(u_t), jnp.asarray(w))


def current_from_density_no_polaron(jrho0T, u_t, j_t):
    require_legacy_precision()
    unitary, insertion, current = map(jnp.asarray, (u_t, jrho0T, _dense(j_t)))
    return -jnp.einsum("ij,jk,ik->", current, unitary, unitary.conj() @ insertion)


def get_sectors_for_polaron_transform(nzidx_hopping):
    quads, sector = build_lf_sectors(nzidx_hopping)
    quads, sector = np.asarray(quads), np.asarray(sector)
    return {value: [tuple(int(x) for x in q) for q in quads[sector == value]]
            for value in range(-2, 3)}


def _prepare_sector_arrays(sectors, F0):
    quads, weights, offsets = [], [], [0]
    for sector in range(-2, 3):
        block = sectors[sector]
        quads.extend(block)
        weights.extend(F0[tuple(q)] for q in block)
        offsets.append(len(quads))
    return (np.array(quads, dtype=np.int32).reshape(-1, 4), np.asarray(weights),
            np.array(offsets, dtype=np.int32))


def current_from_density_polaron(u_t, rho0, j_t_list, j_0_list, F0, sectors,
                                sec_weights, use_python=False, quad_idx=None,
                                F0_vals=None, sector_offsets=None):
    """Preserve the general historical sector-weight API without a second engine.

    ``use_python`` remains accepted for old callers; both values use the same
    JAX contraction. Independent four-index reference tests live in the suite.
    """
    require_legacy_precision()
    if quad_idx is None or F0_vals is None or sector_offsets is None:
        quad_idx, F0_vals, sector_offsets = _prepare_sector_arrays(sectors, F0)
    quads = jnp.asarray(quad_idx)
    if not len(quads):
        return jnp.zeros(np.shape(u_t)[0], dtype=jnp.result_type(1j))
    sector = np.repeat(np.arange(5), np.diff(np.asarray(sector_offsets)))
    weights = jnp.asarray(F0_vals)*jnp.asarray(sec_weights)[sector]
    unitary, rho = jnp.asarray(u_t), jnp.asarray(rho0)
    current, initial = jnp.asarray(_stack_to_dense(j_t_list)), jnp.asarray(_stack_to_dense(j_0_list))
    contraction = unitary.conj() @ rho.swapaxes(-1, -2)
    i, j, k, ell = quads.T
    return -jnp.sum(current[:, i, j]*initial[:, k, ell]*unitary[:, j, k]
                    * contraction[:, i, ell]*weights, axis=-1)


_current_from_density_polaron_python = current_from_density_polaron
