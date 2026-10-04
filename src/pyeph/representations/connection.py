"""Cross-time overlap diagnostics and explicitly chosen projection transport.

O[a,b] = <phi_a(t0)|phi_b(t1)>. Thus projection of old coefficients into
the new basis is c(t1) = O† c(t0). This is basis transport only, not electronic
time evolution. No AO overlap/connection is inferred from H or same-time S.
"""

from typing import Any, NamedTuple

import jax.numpy as jnp
import numpy as np


class OverlapDiagnostics(NamedTuple):
    singular_values: Any
    maximum_norm_loss: Any
    rank_deficient: Any
    is_isometry: Any
    is_contraction: Any


class OverlapTransport(NamedTuple):
    matrix: Any
    raw_overlap: Any
    diagnostics: OverlapDiagnostics
    projection_applied: Any
    valid: Any


def analyze_overlap(overlap, *, tolerance=1e-10, rank_tolerance=1e-12):
    """Diagnose equal-dimensional orthonormal subspaces without renormalizing.

    maximum_norm_loss = max(0, 1 - sigma_min**2) is the largest projected
    norm-squared loss. Values above one in the singular spectrum instead flag
    an invalid expanding overlap through is_contraction=False.
    """
    o = jnp.asarray(overlap)
    if o.ndim != 2 or o.shape[0] != o.shape[1] or o.shape[0] == 0:
        raise ValueError("this transport profile requires a nonempty square overlap")
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be finite and nonnegative")
    if not np.isfinite(rank_tolerance) or rank_tolerance < 0:
        raise ValueError("rank_tolerance must be finite and nonnegative")
    singular = jnp.linalg.svd(o.astype(jnp.result_type(o, 1.0)), compute_uv=False)
    finite = jnp.all(jnp.isfinite(singular))
    return OverlapDiagnostics(
        singular, jnp.maximum(0, 1 - jnp.min(singular)**2),
        ~finite | (jnp.min(singular) <= rank_tolerance),
        finite & jnp.all(jnp.abs(singular - 1) <= tolerance),
        finite & jnp.all(singular <= 1 + tolerance),
    )


def transport_from_overlap(overlap, *, mode="raw", tolerance=1e-10, rank_tolerance=1e-12):
    """Return raw projection or the explicitly requested polar transport.

    ``raw`` preserves actual subspace loss. ``polar`` replaces O by its unitary
    polar factor and therefore discards that loss, which remains in diagnostics.
    A singular overlap has no unique unitary polar transport and is invalid.
    """
    if mode not in {"raw", "polar"}:
        raise ValueError("transport mode must be 'raw' or 'polar'")
    o = jnp.asarray(overlap)
    o = o.astype(jnp.result_type(o, 1.0))
    diagnostics = analyze_overlap(o, tolerance=tolerance, rank_tolerance=rank_tolerance)
    valid = diagnostics.is_contraction
    if mode == "polar":
        left, _, right = jnp.linalg.svd(o, full_matrices=False)
        matrix = (left @ right).conj().T
        valid = valid & ~diagnostics.rank_deficient
    else:
        matrix = o.conj().T
    matrix = jnp.where(valid, matrix, jnp.nan)
    return OverlapTransport(matrix, o, diagnostics, jnp.asarray(mode == "polar"), valid)
