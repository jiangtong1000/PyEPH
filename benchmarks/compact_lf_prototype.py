"""Isolated exact compact Lang--Firsov estimator, not a runtime API.

The support is a unique directed edge set S. This representation means the
complete Cartesian square S x S; it does not describe arbitrary subsets of
quadruples accepted by the existing general LF estimator. Currents outside S
are ignored, including during the ordinary-trace baseline contraction.

For V=conj(U)@rho.T and D(J)_ij=exp[(-1+delta_ij)*phi0] J_ij,

  C0 = sum_(k,l in S) D(J0)_kl sum_i (D(Jt) U)_ik V_il.

This accounts for all edge pairs with their sector-zero bath weight. The
remaining correction sums only pairs with nonzero
n=delta_ik-delta_jk-delta_il+delta_jl, using weight
exp(-2*phi0-n*phit)-exp(-2*phi0). A nonzero n requires both edges to be
offdiagonal. The combined exponents avoid the 0*inf failure of multiplying
exp(-2*phi0) by expm1(-n*phit) at strong coupling. Near phit=0 the subtraction
can lose relative accuracy in a tiny correction; tests compare absolute and
scale-aware error. No roundoff-inclusive certificate is claimed.

The host builder uses shared-vertex incidence lists, requiring
O(N+E+sum_v degree(v)^2) work/storage. For bounded-degree graphs the topology is
linear in state count; dense/high-degree graphs retain a worst-case quadratic
edge-pair cost. Forward numerical cost is O(N^3+E*N+M), where M is the number
of correction pairs. The existing dense full-U/density representation remains;
this is not a linear-memory electronic propagator or a stochastic trace.
"""

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np


@dataclass(frozen=True, eq=False)
class CompactSectors:
    """Validated host topology returned by build_compact_sectors."""

    nstates: int
    support_pairs: object
    quad_indices: object
    sector_indices: object

    @property
    def storage_bytes(self):
        return sum(array.nbytes for array in
                   (self.support_pairs, self.quad_indices, self.sector_indices))


def build_compact_sectors(support_pairs, nstates):
    if (isinstance(nstates, bool) or not isinstance(nstates, (int, np.integer))
            or not 1 <= nstates <= np.iinfo(np.int32).max):
        raise ValueError("nstates must be a positive int32-compatible integer")
    pairs = np.asarray(support_pairs)
    if pairs.size == 0:
        pairs = np.empty((0, 2), dtype=np.int32)
    if pairs.ndim != 2 or pairs.shape[1] != 2 or pairs.dtype.kind not in "iu":
        raise ValueError("support_pairs must have integer shape (nedges,2)")
    if np.any(pairs < 0) or np.any(pairs >= nstates):
        raise ValueError("support_pairs contain out-of-range indices")
    if len(np.unique(pairs, axis=0)) != len(pairs):
        raise ValueError("support_pairs must not contain duplicates")
    pairs = np.array(pairs, dtype=np.int32, copy=True)
    rows = pairs.tolist()
    incident = [[] for _ in range(nstates)]
    for edge, (i, j) in enumerate(rows):
        if i != j:
            incident[i].append(edge)
            incident[j].append(edge)
    quads, sectors = [], []
    for i, j in rows:
        if i == j:
            continue
        for other in sorted(set(incident[i]).union(incident[j])):
            k, ell = rows[other]
            sector = (i == k)-(j == k)-(i == ell)+(j == ell)
            if sector:
                quads.append((i, j, k, ell))
                sectors.append(sector)
    quads = np.asarray(quads, dtype=np.int32).reshape(-1, 4)
    sectors = np.asarray(sectors, dtype=np.int32)
    for array in (pairs, quads, sectors):
        array.setflags(write=False)
    return CompactSectors(int(nstates), pairs, quads, sectors)


def compact_lf_correlation(unitary, density, current_t, current_0, topology, phi0, phit):
    """Exact S x S LF contraction using a masked baseline plus corrections.

    Matrix and bath leading axes must broadcast, just as for the existing
    full-sector estimator. No Hermiticity or unitarity simplification is used.
    The caller supplies topology from build_compact_sectors; current values
    outside its support never enter either contraction.
    """
    unitary, density, current_t, current_0 = map(jnp.asarray,
                                               (unitary, density, current_t, current_0))
    arrays = (unitary, density, current_t, current_0)
    if any(a.ndim < 2 or a.shape[-2:] != (topology.nstates, topology.nstates) for a in arrays):
        raise ValueError("electronic matrices must have shape (...,nstates,nstates)")
    phi0, phit = jnp.asarray(phi0), jnp.asarray(phit)
    pairs = jnp.asarray(topology.support_pairs)
    rows, columns = pairs.T
    dressing = jnp.exp((-1+(rows == columns).astype(jnp.int32))*phi0[..., None])
    jt = dressing*current_t[..., rows, columns]
    j0 = dressing*current_0[..., rows, columns]
    leading = jnp.broadcast_shapes(unitary.shape[:-2], jt.shape[:-1])
    applied = jnp.zeros(leading+unitary.shape[-2:], dtype=jnp.result_type(unitary, jt))
    applied = applied.at[..., rows, :].add(jt[..., :, None]*unitary[..., columns, :])
    density_factor = jnp.conj(unitary) @ jnp.swapaxes(density, -1, -2)
    baseline = jnp.sum(j0*jnp.sum(applied[..., :, rows]*density_factor[..., :, columns], axis=-2), axis=-1)

    i, j, k, ell = jnp.asarray(topology.quad_indices).T
    sectors = jnp.asarray(topology.sector_indices)
    correction_weight = (jnp.exp(-2*phi0[..., None]-sectors*phit[..., None])
                         - jnp.exp(-2*phi0[..., None]))
    # An explicit multiway contraction also handles all leading broadcast axes
    # consistently on the supported older JAX compiler. Its constant-folded
    # chained complex multiplication can otherwise miscompile this expression.
    correction = jnp.einsum("...m,...m,...m,...m,...m->...",
        current_t[..., i, j], current_0[..., k, ell], unitary[..., j, k],
        density_factor[..., i, ell], correction_weight, optimize=False)
    return baseline+correction
