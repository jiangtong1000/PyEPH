"""Small dense spectral helpers for fixed orthonormal effective models.

These are optional operations, not prerequisites for diabatic propagation.
Kernels return validity flags and NaNs for undefined individual surfaces or
couplings. Call ``validate_surfaces`` on the host when rejection is desired.
No arbitrary gap floor is used to define a coupling at a degeneracy.
"""

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from pyeph.core.contracts import LowRankWeight, SurfaceData


class SurfaceQuantity(NamedTuple):
    value: Any
    valid: Any


def diagonalize(matrix, *, gap_tolerance=1e-10, hermiticity_tolerance=1e-10):
    """Return sorted energies, column eigenvectors and pairwise gap status.

    ``gaps[a,b] = energies[b] - energies[a]``. ``near_degenerate`` has
    shape (states, states), with a false diagonal. Tolerances use the model's
    energy units. Nonfinite/non-Hermitian input produces nonfinite results;
    it is never silently replaced by its Hermitian part.
    """
    if not np.isfinite(gap_tolerance) or gap_tolerance < 0:
        raise ValueError("gap_tolerance must be finite and nonnegative")
    if not np.isfinite(hermiticity_tolerance) or hermiticity_tolerance < 0:
        raise ValueError("hermiticity_tolerance must be finite and nonnegative")
    h = jnp.asarray(matrix)
    if h.ndim != 2 or h.shape[0] != h.shape[1] or h.shape[0] == 0:
        raise ValueError("the electronic Hamiltonian must be nonempty and square")
    h = h.astype(jnp.result_type(h, 1.0))
    valid = jnp.all(jnp.isfinite(h)) & (
        jnp.max(jnp.abs(h - h.conj().T))
        <= hermiticity_tolerance * (1 + jnp.max(jnp.abs(h)))
    )
    energies, vectors = jnp.linalg.eigh(h, symmetrize_input=False)
    energies = jnp.where(valid, energies, jnp.nan)
    vectors = jnp.where(valid, vectors, jnp.nan)
    gaps = energies[None, :] - energies[:, None]
    near = (jnp.abs(gaps) <= gap_tolerance) & ~jnp.eye(len(energies), dtype=bool)
    return SurfaceData(energies, vectors, gaps, near)


def model_surfaces(model, params, q, **kwargs):
    """Materialize a small model explicitly for a requested spectral operation."""
    if model.spec.basis_kind != "fixed_orthonormal":
        raise ValueError("these spectral helpers require a fixed orthonormal basis")
    q = jnp.asarray(q)
    n = model.spec.system.nstates
    matrix = model.apply(params, q, jnp.eye(n, dtype=jnp.result_type(q, 1.0)))
    return diagonalize(matrix, **kwargs)


def validate_surfaces(data, *, states=None):
    """Host validation, rejecting undefined individual surfaces.

    With ``states=None``, all states must be isolated. Otherwise only the
    requested states must be separated from the rest of the spectrum. This
    function is intentionally not called inside a JIT-compiled step.
    """
    e, u, gaps, near = map(np.asarray, data)
    n = len(e)
    if e.shape != (n,) or u.shape != (n, n) or gaps.shape != (n, n):
        raise ValueError("inconsistent spectral data shapes")
    if near.shape != (n, n):
        raise ValueError("near_degenerate must be a pairwise state mask")
    if not all(np.isfinite(x).all() for x in (e, u, gaps)):
        raise ValueError("nonfinite spectral data: check Hamiltonian Hermiticity and finiteness")
    if np.iscomplexobj(e) or np.any(np.diff(e) < 0):
        raise ValueError("surface energies must be real and sorted in increasing order")
    if not np.allclose(gaps, e[None, :] - e[:, None], rtol=1e-10, atol=0.):
        raise ValueError("surface gaps must equal energies[None,:] - energies[:,None]")
    if not np.allclose(u.conj().T @ u, np.eye(n), rtol=1e-8, atol=1e-8):
        raise ValueError("surface vectors must form an orthonormal basis")
    indices = np.arange(n) if states is None else np.atleast_1d(states)
    if not np.issubdtype(indices.dtype, np.integer) or np.any((indices < 0) | (indices >= n)):
        raise ValueError("surface indices are out of range")
    if np.any(near[indices]):
        raise ValueError("requested individual surface is degenerate or near-degenerate")
    return data


def _state_valid(data, index):
    index = jnp.asarray(index)
    if index.ndim or not jnp.issubdtype(index.dtype, jnp.integer):
        raise ValueError("a surface index must be a scalar integer")
    n = len(data.energies)
    return ((index >= 0) & (index < n) & ~jnp.any(data.near_degenerate[index])
            & jnp.all(jnp.isfinite(data.energies))
            & jnp.all(jnp.isfinite(data.vectors[:, index])))


def surface_force(model, params, q, data, state):
    """Total force on one isolated surface; never differentiate its eigenvector.

    ``data`` must have been evaluated for this same model, parameter set and q.
    ``valid`` also requires finite selected vectors and returned derivatives;
    spectral isolation alone does not certify a force provider's output.
    """
    if not model.spec.force_support or model.spec.basis_kind != "fixed_orthonormal":
        raise ValueError("surface forces require a fixed-basis force-capable model")
    valid = _state_valid(data, state)
    u = jax.lax.stop_gradient(data.vectors[:, state][:, None])
    value = -model.reference_gradient(params, q) - model.contract_gradient(
        params, q, LowRankWeight(u, u)
    )
    valid = valid & jnp.all(jnp.isfinite(value))
    return SurfaceQuantity(jnp.where(valid, value, jnp.nan), valid)


def derivative_coupling(model, params, q, data, bra, ket):
    """Selected spatial NAC ``<u_bra|grad u_ket>`` for distinct isolated states.

    Complex off-diagonal derivatives require two real contractions. The
    diagonal gauge connection is not determined by this gap formula. Invalid
    indices, a diagonal request, near-degeneracy or nonfinite vectors/derivatives
    yield valid=False. The selected gap must also be finite and nonzero.
    """
    if not model.spec.force_support or model.spec.basis_kind != "fixed_orthonormal":
        raise ValueError("spatial couplings require a fixed-basis force-capable model")
    gap = data.gaps[bra, ket]
    valid = (_state_valid(data, bra) & _state_valid(data, ket) & (bra != ket)
             & jnp.isfinite(gap) & (gap != 0))
    left = jax.lax.stop_gradient(data.vectors[:, bra][:, None])
    right = jax.lax.stop_gradient(data.vectors[:, ket][:, None])
    real = model.contract_gradient(params, q, LowRankWeight(left, right))
    imag = model.contract_gradient(params, q, LowRankWeight(1j * left, right))
    denominator = jnp.where(valid, gap, 1.0)
    value = (real + 1j * imag) / denominator
    valid = valid & jnp.all(jnp.isfinite(value))
    return SurfaceQuantity(jnp.where(valid, value, jnp.nan), valid)
