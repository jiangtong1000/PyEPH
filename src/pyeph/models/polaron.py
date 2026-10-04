"""Local independent harmonic-bath Lang--Firsov dressing.

This deliberately reproduces the local, identical-site quantum-bath
approximation used by the PyEPH contributors (2026, BSD-3-Clause), with an
independent JAX implementation.  It does not derive a general nonlocal
Fröhlich transformation or an Ehrenfest force for integrated-out modes.
See https://github.com/jiangtong1000/PyEPH .
"""

from __future__ import annotations

import jax.numpy as jnp


def lf_phi(frequencies, couplings, beta, time):
    r"""Compute the dimensionless local bath correlation ``phi(time)``.

    ``phi(t) = sum_m |g_m / w_m|^2 [coth(beta*w_m/2) cos(w_m*t)
    - 1j*sin(w_m*t)]``.  Frequencies and couplings have the same energy unit,
    time has its inverse unit (hbar=1), and beta is inverse energy.  The last
    input axis labels modes; all leading axes follow NumPy broadcasting.
    Frequencies and beta must be strictly positive for nonempty baths.  Empty
    mode axes are valid and produce zero.  Site-diagonal Hermitian couplings
    are normally real; the modulus retains phase independence if encoded
    as complex amplitudes.
    """
    frequencies = jnp.asarray(frequencies)
    couplings = jnp.asarray(couplings)
    if frequencies.ndim == 0 or couplings.ndim == 0:
        raise ValueError("frequencies and couplings require a final mode axis")
    if frequencies.shape[-1] != couplings.shape[-1]:
        raise ValueError("frequencies and couplings must contain the same modes")
    phase = frequencies * jnp.asarray(time)[..., None]
    thermal_argument = frequencies * jnp.asarray(beta)[..., None] / 2
    coupling_squared = jnp.abs(couplings / frequencies) ** 2
    terms = coupling_squared * (
        jnp.cos(phase) / jnp.tanh(thermal_argument) - 1j * jnp.sin(phase)
    )
    return jnp.sum(terms, axis=-1)


def lf_band_narrowing(frequencies, couplings, beta):
    """Return ``exp(-phi(0))`` for the declared local harmonic bath."""
    phi0 = jnp.real(lf_phi(frequencies, couplings, beta, 0.0))
    return jnp.exp(-phi0)


def band_narrow_hamiltonian(hamiltonian, factor):
    """Scale off-diagonal elements by ``factor`` and preserve the diagonal.

    ``factor`` is a real scalar or has broadcastable batch axes.  A scalar
    factor describes the identical-site local-bath approximation.  This
    function intentionally does not reproduce historical initialization that
    scaled the entire matrix, including site energies.  Legacy parity callers
    must choose that alternative explicitly when preparing their initial state.
    """
    hamiltonian = jnp.asarray(hamiltonian)
    if hamiltonian.ndim < 2 or hamiltonian.shape[-2] != hamiltonian.shape[-1]:
        raise ValueError("hamiltonian must have shape (..., n, n)")
    diagonal_mask = jnp.eye(hamiltonian.shape[-1], dtype=bool)
    narrowed = hamiltonian * jnp.asarray(factor)[..., None, None]
    return jnp.where(diagonal_mask, hamiltonian, narrowed)


__all__ = ["lf_phi", "lf_band_narrowing", "band_narrow_hamiltonian"]
