"""All-state Runeson--Manolopoulos impulse algebra in a fixed orthonormal basis.

These helpers use a complete isolated spectrum and one mapping vector. They do
not implement hopping, normalize that vector, or supply an initial ensemble.
The algebra permits complex Hermitian matrices; this alone does not validate a
complex/SOC hopping formulation. The dynamics method must enforce its scope.
"""

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

from pyeph.core._configuration import real_scalar
from pyeph.core.contracts import LowRankWeight


RM_SUCCESS = 0
RM_NONFINITE = 1
RM_INVALID_SPECTRUM = 2
RM_DEGENERATE = 3
RM_INVALID_PAIR = 4
RM_NONFINITE_GRADIENT = 5

STATUS_NAMES = (
    "success", "nonfinite spectral or mapping data", "invalid spectral basis or gaps",
    "nonisolated spectrum", "invalid or identical surface pair", "nonfinite gradient",
)


class RMWeightResult(NamedTuple):
    weight: LowRankWeight
    populations: Any
    status: Any

    @property
    def valid(self):
        return self.status == RM_SUCCESS


class RMImpulseResult(NamedTuple):
    value: Any
    populations: Any
    status: Any

    @property
    def valid(self):
        return self.status == RM_SUCCESS


def _nonnegative_tolerance(value, name):
    value = real_scalar(value, name)
    if value < 0:
        raise ValueError(f"{name} must be nonnegative")
    return value


def rm_impulse_weight(data, c, active, competitor, *, gap_tolerance=1e-10,
                      orthonormality_tolerance=1e-8):
    """Return the rank-two weight for the full RM impulse direction.

    With ``c_k = u_k.conj() @ c``, define
    ``w_a = sum(k != a, u_k*c_k/(epsilon_a-epsilon_k))``. The weight has left
    columns ``(w_a, w_b)`` and right columns ``(c_a*u_a, -c_b*u_b)``. A single
    ``contract_gradient`` then gives ``delta = 0.5*grad[c†(P_a-P_b)c]`` with
    the *fixed-basis* c held fixed in this partial coordinate derivative.
    Outer derivatives through these factors remain available.

    ``data`` must contain the full spectrum at the same model/params/q as any
    subsequent gradient contraction; this identity cannot be checked here.
    Eigenvalues need not be sorted. Both its near-degenerate mask and the
    explicit gap tolerance are respected, including spectator degeneracies.
    There is no denominator floor. Invalid numerical data returns NaN factors
    and a status; malformed static shapes/dtypes raise ValueError.

    Populations are raw ``abs(U†c)**2``. No normalization, mapping-estimator
    alpha, scalar-reference or mass factor enters this operation. Scaling c by
    z scales its populations and contracted direction by abs(z)**2. A zero
    vector is valid algebra; a dynamics method must enforce its own norm and
    nonzero event-direction conditions. Tolerances are static configuration.
    """
    gap_tolerance = _nonnegative_tolerance(gap_tolerance, "gap_tolerance")
    orthonormality_tolerance = _nonnegative_tolerance(
        orthonormality_tolerance, "orthonormality_tolerance")
    energies, vectors, gaps, near = map(jnp.asarray, data)
    c, active, competitor = map(jnp.asarray, (c, active, competitor))
    if energies.ndim != 1 or len(energies) < 2:
        raise ValueError("RM impulse requires at least two complete surface energies")
    n = len(energies)
    if vectors.shape != (n, n) or gaps.shape != (n, n) or near.shape != (n, n):
        raise ValueError("RM spectral arrays must have complete (nstates,nstates) shapes")
    if c.shape != (n,):
        raise ValueError("RM impulse expects one fixed-basis mapping vector")
    if not jnp.issubdtype(energies.dtype, jnp.floating) or not jnp.issubdtype(gaps.dtype, jnp.floating):
        raise ValueError("RM energies and gaps must be real floating arrays")
    if not all(jnp.issubdtype(x.dtype, jnp.inexact) for x in (vectors, c)):
        raise ValueError("RM vectors and mapping coefficients must be floating arrays")
    if near.dtype != jnp.bool_:
        raise ValueError("near_degenerate must be a boolean mask")
    for index in (active, competitor):
        if index.ndim or not jnp.issubdtype(index.dtype, jnp.integer):
            raise ValueError("RM surface indices must be scalar integers")

    finite = (jnp.all(jnp.isfinite(energies)) & jnp.all(jnp.isfinite(vectors))
              & jnp.all(jnp.isfinite(gaps)) & jnp.all(jnp.isfinite(c)))
    expected_gaps = energies[None, :] - energies[:, None]
    diagonal = jnp.eye(n, dtype=bool)
    basis_valid = (
        (jnp.max(jnp.abs(vectors.conj().T @ vectors-jnp.eye(n)))
         <= orthonormality_tolerance)
        & jnp.all(jnp.isclose(gaps, expected_gaps, rtol=1e-8, atol=0.))
        & ~jnp.any(near & diagonal) & jnp.all(near == near.T)
    )
    degenerate = jnp.any(near) | jnp.any((jnp.abs(expected_gaps) <= gap_tolerance) & ~diagonal)
    pair_valid = ((active >= 0) & (active < n) & (competitor >= 0)
                  & (competitor < n) & (active != competitor))
    status = jnp.select((~finite, ~basis_valid, degenerate, ~pair_valid),
                        (RM_NONFINITE, RM_INVALID_SPECTRUM, RM_DEGENERATE, RM_INVALID_PAIR),
                        default=RM_SUCCESS).astype(jnp.int32)
    valid = status == RM_SUCCESS
    # Clipping only makes invalid indices safe to evaluate under JIT/vmap; the
    # unmodified indices determine validity and invalid outputs remain NaN.
    a, b = jnp.clip(active, 0, n-1), jnp.clip(competitor, 0, n-1)
    coefficients = vectors.conj().T @ c
    populations = jnp.abs(coefficients)**2
    indices = jnp.arange(n)

    def response(index):
        selected = (indices != index) & valid
        # The self term is exactly omitted. Invalid spectra get safe arithmetic
        # only, never a finite physical coupling defined by a gap floor.
        denominator = jnp.where(selected, energies[index]-energies, 1.)
        return vectors @ jnp.where(selected, coefficients/denominator, 0.)

    left = jnp.stack((response(a), response(b)), axis=-1)
    right = jnp.stack((coefficients[a]*vectors[:, a], -coefficients[b]*vectors[:, b]), axis=-1)
    all_finite = (finite & jnp.all(jnp.isfinite(left)) & jnp.all(jnp.isfinite(right))
                  & jnp.all(jnp.isfinite(populations)))
    status = jnp.where(valid & ~all_finite, RM_NONFINITE, status)
    valid = status == RM_SUCCESS
    weight = LowRankWeight(jnp.where(valid, left, jnp.nan), jnp.where(valid, right, jnp.nan))
    return RMWeightResult(weight, jnp.where(valid, populations, jnp.nan), status)


def rm_impulse_direction(model, params, q, data, c, active, competitor, *,
                         gap_tolerance=1e-10, orthonormality_tolerance=1e-8):
    """Contract one rank-two weight to give the mass-unweighted RM direction.

    Uses exactly one real gradient contraction on a valid input; no full
    Hamiltonian derivative tensor is formed. A scalar conditional skips the
    provider on invalid input. As usual, batching this conditional with vmap
    does not guarantee cancellation of work for individual invalid lanes.
    Native fixed-basis force-capable models are required. The gradient has q's
    shape; reference forces do not enter this population-projector derivative.

    For valid results the pair margin is ``populations[a]-populations[b]`` and
    its instantaneous rate is ``2*sum(value*velocity)`` under fixed-basis
    Schrödinger evolution. Mass-weight an impulse separately as delta/sqrt(m).
    """
    if (not model.spec.native_jax or not model.spec.force_support
            or model.spec.basis_kind != "fixed_orthonormal"):
        raise ValueError("RM impulse requires a native fixed-basis force-capable model")
    q = jnp.asarray(q)
    if q.shape != model.spec.system.q_shape or not jnp.issubdtype(q.dtype, jnp.floating):
        raise ValueError("RM coordinates must be real floating arrays with the model's q_shape")
    if len(data.energies) != model.spec.system.nstates:
        raise ValueError("RM spectrum must cover the model's complete electronic space")
    result = rm_impulse_weight(data, c, active, competitor,
        gap_tolerance=gap_tolerance, orthonormality_tolerance=orthonormality_tolerance)
    status = jnp.where(jnp.all(jnp.isfinite(q)), result.status, RM_NONFINITE)

    def evaluate(_):
        value = jnp.asarray(model.contract_gradient(params, q, result.weight))
        if value.shape != q.shape or not jnp.issubdtype(value.dtype, jnp.floating):
            raise ValueError("contract_gradient must return a real gradient with q's shape")
        # Partial derivatives of native JAX models have the coordinate dtype.
        # Preserve that contract across the invalid conditional branch too.
        if value.dtype != q.dtype:
            raise ValueError("contract_gradient must return the coordinate floating dtype")
        valid = jnp.all(jnp.isfinite(value))
        return (jnp.where(valid, value, jnp.nan),
                jnp.where(valid, RM_SUCCESS, RM_NONFINITE_GRADIENT).astype(jnp.int32))

    value, status = jax.lax.cond(status == RM_SUCCESS, evaluate,
        lambda _: (jnp.full_like(q, jnp.nan), status), operand=None)
    return RMImpulseResult(value, result.populations, status)
