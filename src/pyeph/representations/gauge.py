"""Explicit column-phase alignment, without state permutation or subspace guessing."""

from typing import Any, NamedTuple

import jax.numpy as jnp


class PhaseAlignment(NamedTuple):
    vectors: Any
    overlaps: Any
    valid: Any


def align_phases(reference, candidate, *, overlap_tolerance=1e-12):
    """Align each candidate column to the corresponding reference column.

    This handles real signs and complex U(1) phases. It does not track reordered
    states or rotate a degenerate subspace. A zero-overlap column is unchanged
    and marked invalid, so a caller cannot mistake it for successful tracking.
    """
    reference, candidate = jnp.asarray(reference), jnp.asarray(candidate)
    if reference.ndim != 2 or reference.shape != candidate.shape:
        raise ValueError("reference and candidate must have equal matrix shapes")
    if overlap_tolerance < 0:
        raise ValueError("overlap_tolerance must be nonnegative")
    overlaps = jnp.sum(reference.conj() * candidate, axis=0)
    magnitudes = jnp.abs(overlaps)
    valid = jnp.isfinite(magnitudes) & (magnitudes > overlap_tolerance)
    phase = jnp.where(valid, overlaps.conj() / jnp.where(valid, magnitudes, 1), 1)
    return PhaseAlignment(candidate * phase[None, :], overlaps, valid)


def transform_operator(operator, vectors):
    """Operator in the column basis ``vectors``: U† operator U."""
    return vectors.conj().T @ operator @ vectors


def transform_state(state, vectors):
    """Coefficients in the column basis ``vectors``: U† state (vector or block)."""
    return vectors.conj().T @ state
