"""Exact local Lang--Firsov correlations on a declared current-edge support.

The support represents its complete Cartesian square. Sector-zero pairs are
contracted as an ordinary trace; only nonzero sectors require explicit
quadruples. Arbitrary quadruple subsets retain the separate ``polaron`` API.
This uses the same local, independent, identical-site bath convention as that
module and preserves diagonal currents and complex matrix algebra.

Topology construction visits O(N+E+sum_v degree(v)^2) candidate data and
sorts each incident candidate set; bounded-degree graphs remain linear.
The full-U correlation still costs O(N^3+E*N+M), with M correction quadruples;
this is neither a stochastic trace nor a linear-memory electronic propagator.
"""

from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
import numpy as np

from .greenkubo import _matrix_inputs


@dataclass(frozen=True, eq=False, init=False)
class CompactLFSectors:
    """Immutable support and complete nonzero-sector corrections.

    Construction validates and copies the unique directed support, then builds
    every required correction. Derived quadruples cannot be supplied or
    silently omitted. Keep this topology static when compiling a standalone
    contraction; transport workflows capture it as problem configuration.
    """

    nstates: int
    support_pairs: Any
    quad_indices: Any
    sector_indices: Any

    def __init__(self, support_pairs, nstates):
        nstates, pairs, quads, sectors = _sector_arrays(support_pairs, nstates)
        object.__setattr__(self, "nstates", nstates)
        for name, value in (("support_pairs", pairs), ("quad_indices", quads),
                            ("sector_indices", sectors)):
            object.__setattr__(self, name, jnp.array(value, copy=True))

    @property
    def storage_bytes(self):
        return sum(array.nbytes for array in
                   (self.support_pairs, self.quad_indices, self.sector_indices))


def build_compact_lf_sectors(hopping_pairs, nstates):
    """Build the complete support-square topology without enumerating E squared pairs."""
    return CompactLFSectors(hopping_pairs, nstates)


def _sector_arrays(support_pairs, nstates):
    if (isinstance(nstates, bool) or not isinstance(nstates, (int, np.integer))
            or not 1 <= nstates <= np.iinfo(np.int32).max):
        raise ValueError("nstates must be a positive int32-compatible integer")
    pairs = np.asarray(support_pairs)
    if pairs.shape == (0,):
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
    return int(nstates), pairs, quads, sectors


def lf_current_correlation_compact(unitary, density, current_t, current_0, topology, phi0, phit):
    """Exact S x S LF contraction using a masked baseline plus corrections.

    Matrix and bath leading axes must broadcast, just as for the existing
    full-sector estimator. No Hermiticity or unitarity simplification is used.
    The caller supplies topology from build_compact_lf_sectors; current values
    outside its support never enter either contraction.
    """
    if not isinstance(topology, CompactLFSectors):
        raise TypeError("topology must be built as CompactLFSectors")
    unitary, density, current_t, current_0 = _matrix_inputs(unitary, density, current_t, current_0)
    if unitary.shape[-1] != topology.nstates:
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


__all__ = ["CompactLFSectors", "build_compact_lf_sectors", "lf_current_correlation_compact"]
