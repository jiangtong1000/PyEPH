"""Lang--Firsov current correlations for a local identical-site quantum bath.

Sector decomposition follows the PyEPH contributors' 2026 BSD-3-Clause
implementation (https://github.com/jiangtong1000/PyEPH), re-expressed as pure
JAX array contractions.  Currents use the physical Hermitian convention; use
``legacy_current_to_physical`` for historical matrices lacking ``1j``.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from .greenkubo import _matrix_inputs


def build_lf_sectors(hopping_pairs, nstates=None):
    """Prepare ``(quad_indices, sector_indices)`` outside compiled propagation.

    ``hopping_pairs`` lists the unique *directed* current-operator edges
    ``(i,j)``.  Its Cartesian square yields quadruples ``(i,j,k,l)`` with
    sector ``delta_ik-delta_jk-delta_il+delta_jl`` in ``[-2,2]``.  Both current
    components must be supported on these edges; include their union for a
    cross-correlation.  Duplicate or negative edges are rejected because they
    would silently change the correlation.  This host-side preprocessing has
    quadratic cost in the number of edges and is not itself JIT compatible.
    """
    pairs = np.asarray(hopping_pairs)
    if pairs.size == 0:
        pairs = np.empty((0, 2), dtype=np.int32)
    if pairs.ndim != 2 or pairs.shape[1] != 2:
        raise ValueError("hopping_pairs must have shape (nedges, 2)")
    if not np.issubdtype(pairs.dtype, np.integer):
        raise ValueError("hopping_pairs must contain integer state indices")
    if np.any(pairs < 0):
        raise ValueError("hopping_pairs cannot contain negative indices")
    if nstates is not None and np.any(pairs >= nstates):
        raise ValueError("hopping_pairs contain an out-of-range state index")
    if len(np.unique(pairs, axis=0)) != len(pairs):
        raise ValueError("hopping_pairs must not contain duplicate directed edges")
    nedges = len(pairs)
    quads = np.concatenate(
        [np.repeat(pairs, nedges, axis=0), np.tile(pairs, (nedges, 1))], axis=1
    ).astype(np.int32, copy=False)
    i, j, k, ell = quads.T
    sectors = (
        (i == k).astype(np.int32)
        - (j == k).astype(np.int32)
        - (i == ell).astype(np.int32)
        + (j == ell).astype(np.int32)
    )
    return jnp.asarray(quads), jnp.asarray(sectors)


def lf_current_correlation(
    unitary,
    density,
    current_t,
    current_0,
    quad_indices,
    sector_indices,
    phi0,
    phit,
):
    r"""Evaluate a bath-dressed current correlation using prepared sectors.

    ``F_ijkl(t) = exp[(-2+delta_ij+delta_kl)*phi0 - n_ijkl*phit]``.
    The returned complex correlation sums ``Jt_ij J0_kl U_jk
    (U.conj() @ rho.T)_il F_ijkl`` over the supplied quadruples.  All matrix,
    ``phi0`` and ``phit`` batch axes must be broadcastable.  The edge index
    arrays are static topology data shared across the batch.

    This observable must accompany the specified local LF model dressing; it
    is neither a generic quantum-bath estimator nor a feedback-dynamics rule.
    """
    unitary, density, current_t, current_0 = _matrix_inputs(
        unitary, density, current_t, current_0
    )
    quads = jnp.asarray(quad_indices)
    sectors = jnp.asarray(sector_indices)
    if quads.ndim != 2 or quads.shape[-1] != 4:
        raise ValueError("quad_indices must have shape (nquadruples, 4)")
    if sectors.shape != (quads.shape[0],):
        raise ValueError("sector_indices must have shape (nquadruples,)")
    i, j, k, ell = quads.T
    diagonal_count = (i == j).astype(jnp.int32) + (k == ell).astype(jnp.int32)
    bath_factor = jnp.exp(
        (-2 + diagonal_count) * jnp.asarray(phi0)[..., None]
        - sectors * jnp.asarray(phit)[..., None]
    )
    density_factor = jnp.conj(unitary) @ jnp.swapaxes(density, -1, -2)
    terms = (
        current_t[..., i, j]
        * current_0[..., k, ell]
        * unitary[..., j, k]
        * density_factor[..., i, ell]
        * bath_factor
    )
    return jnp.sum(terms, axis=-1)


def lf_current_correlation_dense(unitary, density, current_t, current_0, phi0, phit):
    """Evaluate all state-index quadruples; an O(n^4) small-system oracle.

    Production runs should prepare the union of supported current edges with
    ``build_lf_sectors`` and call ``lf_current_correlation`` instead.
    """
    unitary, density, current_t, current_0 = _matrix_inputs(
        unitary, density, current_t, current_0
    )
    nstates = unitary.shape[-1]
    i, j, k, ell = jnp.indices((nstates,) * 4)
    diagonal_count = (i == j).astype(jnp.int32) + (k == ell).astype(jnp.int32)
    sectors = (
        (i == k).astype(jnp.int32)
        - (j == k).astype(jnp.int32)
        - (i == ell).astype(jnp.int32)
        + (j == ell).astype(jnp.int32)
    )
    bath_factor = jnp.exp(
        (-2 + diagonal_count) * jnp.asarray(phi0)[..., None, None, None, None]
        - sectors * jnp.asarray(phit)[..., None, None, None, None]
    )
    density_factor = jnp.conj(unitary) @ jnp.swapaxes(density, -1, -2)
    return jnp.einsum(
        "...ij,...kl,...jk,...il,...ijkl->...",
        current_t,
        current_0,
        unitary,
        density_factor,
        bath_factor,
    )


__all__ = ["build_lf_sectors", "lf_current_correlation", "lf_current_correlation_dense"]
