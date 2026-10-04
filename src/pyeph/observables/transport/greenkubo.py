"""Thermal preparation and dense electronic current correlations.

The public current convention is a physical Hermitian operator.  Historical
PyEPH ``build_jx_jy`` output omitted a factor of ``1j``; use the explicit legacy
conversion when importing that data.  Neither function inserts charge or unit
conversion factors.  Arrays may have broadcastable leading batch dimensions.

The equations and legacy convention follow PyEPH's Green--Kubo implementation
by the PyEPH contributors (2026, BSD-3-Clause).  These are independent JAX
implementations; the original implementation lives at
https://github.com/jiangtong1000/PyEPH .
"""

from __future__ import annotations

import jax.numpy as jnp


def _square_matrix(value, name):
    value = jnp.asarray(value)
    if value.ndim < 2 or value.shape[-2] != value.shape[-1]:
        raise ValueError(f"{name} must have shape (..., n, n)")
    return value


def _matrix_inputs(unitary, density, current_t, current_0):
    arrays = tuple(
        _square_matrix(value, name)
        for value, name in (
            (unitary, "unitary"),
            (density, "density"),
            (current_t, "current_t"),
            (current_0, "current_0"),
        )
    )
    if len({array.shape[-1] for array in arrays}) != 1:
        raise ValueError("electronic matrix dimensions must agree")
    return arrays


def thermal_density_matrix(hamiltonian, beta):
    """Return ``exp(-beta H) / Tr(exp(-beta H))`` for Hermitian ``H``.

    ``beta`` is inverse energy, not inverse Kelvin.  Subtracting the lowest
    eigenvalue before exponentiation avoids overflow at low temperature and
    makes arbitrary scalar energy offsets harmless.  ``beta`` must be
    nonnegative; positive infinity selects the lowest eigenspace.  Input
    validation belongs at problem construction, outside compiled numerical
    kernels.  Complex eigenvectors use a conjugate transpose.  A
    zero-dimensional electronic space is rejected.

    This eigen-decomposition implementation is intended for exact small-system
    transport.  Derivatives of eigenspaces at degeneracy require additional
    care; differentiating this routine through a degeneracy is not promised.
    """
    hamiltonian = _square_matrix(hamiltonian, "hamiltonian")
    if hamiltonian.shape[-1] == 0:
        raise ValueError("hamiltonian must have at least one electronic state")
    energies, vectors = jnp.linalg.eigh(hamiltonian)
    shifted = energies - energies[..., :1]
    # The ground-state log weight is exactly zero even when beta is infinity.
    log_weights = jnp.where(shifted == 0, 0, -jnp.asarray(beta)[..., None] * shifted)
    weights = jnp.exp(log_weights)
    weights = weights / jnp.sum(weights, axis=-1, keepdims=True)
    return (vectors * weights[..., None, :]) @ jnp.swapaxes(
        jnp.conj(vectors), -1, -2
    )


def current_correlation(unitary, density, current_t, current_0):
    r"""Evaluate ``Tr[J(t) U(t) J(0) rho(0) U(t)^dagger]``.

    All inputs use the same fixed orthonormal electronic basis.  The two
    currents may be different components, permitting cross-correlations.
    The complex unsymmetrized correlation is returned without taking its real
    part, performing an ensemble average, or integrating a mobility.  This
    estimator assumes a prescribed electronic path; it is not automatically
    a response estimator for trajectories with state-dependent feedback.
    """
    unitary, density, current_t, current_0 = _matrix_inputs(
        unitary, density, current_t, current_0
    )
    evolved_insertion = unitary @ current_0 @ density
    evolved_insertion = evolved_insertion @ jnp.swapaxes(jnp.conj(unitary), -1, -2)
    return jnp.einsum("...ij,...ji->...", current_t, evolved_insertion)


def legacy_current_to_physical(current_without_i):
    """Restore the ``1j`` omitted by historical PyEPH current builders.

    Input already contains its displacement, hopping, charge and unit factors
    as applicable.  No additional sign, length or charge convention is inferred.
    """
    return 1j * jnp.asarray(current_without_i)


def legacy_current_correlation(unitary, density, current_t, current_0):
    """Evaluate the old ``-Tr[J_t U J_0 rho U^dagger]`` convention explicitly."""
    return current_correlation(
        unitary,
        density,
        legacy_current_to_physical(current_t),
        legacy_current_to_physical(current_0),
    )


__all__ = [
    "thermal_density_matrix",
    "current_correlation",
    "legacy_current_to_physical",
    "legacy_current_correlation",
]
