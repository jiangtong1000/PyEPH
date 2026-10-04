"""Checked Hermitian exponential action for a frozen, declared linear operator.

The kernel uses only apply(vector), fixed-capacity arrays and a small symmetric
eigensolve. The error estimate includes the exact-arithmetic Lanczos bound,
the computed three-term recurrence defect, and an explicit roundoff allowance.
The last term is a heuristic diagnostic, NOT a certified floating-point bound.
See docs/CHECKED_PROPAGATION_DESIGN.md for acceptance and finite-arithmetic limits.
"""

from dataclasses import dataclass
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from pyeph.core._configuration import integer_scalar, real_scalar


STATUS = {
    0: "success",
    1: "maximum Krylov dimension did not meet the error budget",
    2: "invalid budget or nonfinite input, operator action, or projected calculation",
    3: "projected operator failed the Hermiticity check",
    4: "Krylov basis failed the orthogonality check",
    5: "requested tolerance is below the roundoff allowance",
    6: "near-breakdown occurred before the error budget was met",
}


@dataclass(frozen=True)
class LanczosOptions:
    """Immutable compilation choices; budget is atol + rtol * input norm."""

    max_dimension: int = 32
    atol: float = 1e-12
    rtol: float = 1e-10
    breakdown_rtol: float = 64 * np.finfo(np.float64).eps
    orthogonality_tolerance: float = 1e-10
    hermiticity_tolerance: float = 1e-10

    def __post_init__(self):
        dimension = integer_scalar(self.max_dimension, "max_dimension")
        if dimension < 1:
            raise ValueError("max_dimension must be a positive integer")
        object.__setattr__(self, "max_dimension", dimension)
        for name in ("atol", "rtol", "breakdown_rtol", "orthogonality_tolerance",
                     "hermiticity_tolerance"):
            value = real_scalar(getattr(self, name), name)
            if value < 0:
                raise ValueError(f"{name} must be a nonnegative scalar")
            object.__setattr__(self, name, value)
        if self.atol == self.rtol == 0:
            raise ValueError("the error budget must be positive")


class LanczosResult(NamedTuple):
    """Candidate action and diagnostics; any nonzero status forbids acceptance."""

    value: Any
    error_estimate: Any
    truncation_bound: Any
    recurrence_bound: Any
    roundoff_allowance: Any
    orthogonality_error: Any
    hermiticity_error: Any
    iterations: Any
    status: Any


class LanczosError(RuntimeError):
    def __init__(self, result):
        self.result = result
        codes = np.unique(np.asarray(result.status))
        details = "; ".join(STATUS[int(code)] for code in codes if code != 0)
        super().__init__(f"checked Lanczos action failed: {details}")


def require_success(result):
    """Host-side acceptance gate; retain the complete failed result for inspection."""
    if np.any(np.asarray(result.status) != 0):
        raise LanczosError(result)
    return result.value


def _norm(value):
    """Scaled Euclidean/Frobenius norm, with an exact zero branch."""
    scale = jnp.max(jnp.abs(value))
    divisor = jnp.where(scale > 0, scale, 1.0)
    return scale * jnp.sqrt(jnp.sum(jnp.abs(value / divisor)**2))


def _vector_action(apply, vector, duration, options, absolute_budget=None):
    n, capacity = len(vector), min(len(vector), options.max_dimension)
    real_dtype = vector.real.dtype
    eps, tiny = jnp.finfo(real_dtype).eps, jnp.finfo(real_dtype).tiny
    valid_input = jnp.all(jnp.isfinite(vector)) & jnp.isfinite(duration)
    safe_vector = jnp.where(jnp.isfinite(vector), vector, 0)
    length = _norm(safe_vector)
    budget = (options.atol + options.rtol * length
              if absolute_budget is None else absolute_budget)
    valid_input = valid_input & jnp.isfinite(length) & jnp.isfinite(budget) & (budget >= 0)
    trivial = valid_input & ((length == 0) | (duration == 0))

    def unchanged(_):
        zero = jnp.zeros((), real_dtype)
        return LanczosResult(vector, zero, zero, zero, zero, zero, zero,
                            jnp.int32(0), jnp.int32(0))

    def calculate(_):
        q0 = safe_vector / jnp.where(length > 0, length, 1.0)
        q = jnp.zeros((n, capacity + 1), vector.dtype).at[:, 0].set(q0)
        hq = jnp.zeros((n, capacity), vector.dtype)
        alpha = jnp.zeros(capacity, real_dtype)
        beta = jnp.zeros(capacity, real_dtype)
        # Keep Hq as an independent record of each applied basis vector. This
        # deliberately spends another O(n*m) buffer to check the recurrence.
        initial = (q, hq, alpha, beta, jnp.zeros(n, vector.dtype), jnp.int32(0),
                   valid_input, valid_input, jnp.asarray(False))

        def body(index, carry):
            def advance(carry):
                q, hq, alpha, beta, tail, count, alive, finite, near = carry
                current = q[:, index]
                applied = jnp.asarray(apply(current), dtype=vector.dtype)
                if applied.shape != current.shape:
                    raise ValueError("apply(vector) must return the same vector shape")
                action_finite = jnp.all(jnp.isfinite(applied))
                applied = jnp.where(action_finite, applied, jnp.zeros_like(applied))
                diagonal = jnp.real(jnp.vdot(current, applied))
                previous = q[:, jnp.maximum(index-1, 0)] * beta[jnp.maximum(index-1, 0)]
                previous = jnp.where(index > 0, previous, jnp.zeros_like(previous))
                residual = applied - diagonal * current - previous
                # Two full reorthogonalization passes. Future columns are zero.
                for _ in range(2):
                    residual = residual - q @ (q.conj().T @ residual)
                residual_norm = _norm(residual)
                scale = jnp.maximum(_norm(applied), jnp.abs(diagonal) + _norm(previous))
                near = residual_norm <= options.breakdown_rtol * scale
                near = near | (residual_norm == 0)
                finite = finite & action_finite & jnp.isfinite(residual_norm) & jnp.isfinite(scale)
                next_vector = residual / jnp.where(residual_norm > 0, residual_norm, 1.0)
                q = q.at[:, index+1].set(jnp.where(near | ~finite, 0, next_vector))
                hq = hq.at[:, index].set(applied)
                alpha = alpha.at[index].set(diagonal)
                beta = beta.at[index].set(residual_norm)
                return (q, hq, alpha, beta, residual, jnp.asarray(index+1, jnp.int32),
                        finite & ~near, finite, near)
            return jax.lax.cond(carry[6], advance, lambda x: x, carry)

        q, hq, alpha, beta, tail, count, _, finite, near = jax.lax.fori_loop(
            0, capacity, body, initial)
        used = jnp.arange(capacity) < count
        basis = q[:, :capacity] * used[None, :]
        off_diagonal = beta[:-1] * (jnp.arange(capacity-1) < count-1)
        projected = (jnp.diag(alpha) + jnp.diag(off_diagonal, 1)
                     + jnp.diag(off_diagonal, -1))
        # Padded inactive modes are uncoupled; the shape never depends on count.
        energies, rotation = jnp.linalg.eigh(projected)
        weights = rotation @ (jnp.exp(-1j * duration * energies) * rotation[0, :])
        value = length * (basis @ weights)

        gram = basis.conj().T @ basis
        orthogonality = _norm(gram - jnp.diag(used.astype(real_dtype)))
        actual_projection = basis.conj().T @ hq
        hermiticity = _norm(actual_projection - actual_projection.conj().T)
        hermiticity /= jnp.maximum(_norm(actual_projection), tiny)

        last = jax.nn.one_hot(jnp.maximum(count-1, 0), capacity, dtype=real_dtype)
        defect = hq - basis @ projected - tail[:, None] * last[None, :]
        recurrence_bound = length * jnp.abs(duration) * _norm(defect)
        # Product beta[0]...beta[m-1] includes the final residual. Do not erase
        # its contribution at a near-breakdown. The log product avoids overflow.
        logs = jnp.where(used, jnp.log(beta), 0.0)
        order = jnp.maximum(count, 1)
        log_bound = (jnp.log(length) + jnp.sum(logs)
                     + order * jnp.log(jnp.abs(duration))
                     - jax.scipy.special.gammaln(order + 1.0))
        truncation_bound = jnp.exp(log_bound)
        # Heuristic allowance for normalization, matvec/reorthogonalization,
        # the small eigensolve and reconstruction. This is NOT directed rounding
        # or a proof of the accuracy of an arbitrary provider's apply operation.
        roundoff = (32 * eps * (count + 1) * length
                    * (1 + jnp.abs(duration) * _norm(hq)))
        bound = truncation_bound + recurrence_bound + roundoff
        status = jnp.where(bound <= budget, 0, jnp.where(near, 6, 1))
        status = jnp.where(roundoff > budget, 5, status)
        status = jnp.where(orthogonality > options.orthogonality_tolerance, 4, status)
        status = jnp.where(hermiticity > options.hermiticity_tolerance, 3, status)
        finite = (finite & jnp.all(jnp.isfinite(value)) & jnp.isfinite(recurrence_bound)
                  & jnp.isfinite(roundoff) & jnp.isfinite(orthogonality)
                  & jnp.isfinite(hermiticity))
        status = jnp.where(finite, status, 2).astype(jnp.int32)
        bound = jnp.where(finite, bound, jnp.inf)
        return LanczosResult(value, bound, truncation_bound, recurrence_bound, roundoff,
                            orthogonality, hermiticity, count, status)

    return jax.lax.cond(trivial, unchanged, calculate, operand=None)


def lanczos_action(apply, vectors, duration, options=LanczosOptions(), *, absolute_budget=None):
    """Approximate exp(-i*duration*H) vectors for a declared Hermitian H.

    Only float64/complex128 is validated by this implementation.
    Columns use independent Krylov spaces: diagnostics are scalar for a vector,
    or shaped (ncolumns,) for a column block. The returned value is a candidate
    even on failure. Call require_success on the host, or consume status in a
    bounded compiled caller before advancing any physical state.

    absolute_budget, when supplied, replaces atol + rtol * input norm. It may
    be a scalar, or a (ncolumns,) array for a block, and is a dynamic runtime
    argument. It must be finite and nonnegative. Zero permits exact bypasses
    such as a zero vector; nontrivial actions retain their roundoff allowance.

    Projected Hermiticity is a diagnostic, not a proof about unexplored space.
    apply must be pure and linear. Under vmap, lane conditionals can execute
    masked operator actions; no cancellation of in-flight batched work is
    promised. The total error_estimate includes a heuristic roundoff allowance,
    not a certified floating-point bound. No automatic renormalization is used.
    This function does not JIT itself or capture changing model parameters.
    """
    vectors = jnp.asarray(vectors)
    if vectors.ndim not in (1, 2) or not vectors.shape[0] or (
            vectors.ndim == 2 and not vectors.shape[1]):
        raise ValueError("vectors must have shape (states,) or (states,columns), with nonzero sizes")
    if vectors.dtype not in (jnp.dtype("float64"), jnp.dtype("complex128")):
        raise ValueError("the checked action requires float64 or complex128; enable JAX x64")
    vectors = vectors.astype(jnp.complex128)
    duration = jnp.asarray(duration)
    if duration.ndim or jnp.issubdtype(duration.dtype, jnp.complexfloating):
        raise ValueError("duration must be one real scalar")
    duration = duration.astype(vectors.real.dtype)
    if absolute_budget is not None:
        absolute_budget = jnp.asarray(absolute_budget)
        allowed_shapes = ((),) if vectors.ndim == 1 else ((), (vectors.shape[1],))
        if (absolute_budget.shape not in allowed_shapes
                or not jnp.issubdtype(absolute_budget.dtype, jnp.number)
                or jnp.issubdtype(absolute_budget.dtype, jnp.complexfloating)):
            raise ValueError("absolute_budget must be a real scalar or one real value per column")
        absolute_budget = absolute_budget.astype(vectors.real.dtype)
    if vectors.ndim == 1:
        return _vector_action(apply, vectors, duration, options, absolute_budget)
    if absolute_budget is not None:
        budgets = jnp.broadcast_to(absolute_budget, (vectors.shape[1],))
        result = jax.vmap(lambda column, budget: _vector_action(
            apply, column, duration, options, budget), in_axes=(1, 0), out_axes=0)(vectors, budgets)
        return result._replace(value=result.value.T)
    result = jax.vmap(lambda column: _vector_action(apply, column, duration, options),
                      in_axes=1, out_axes=0)(vectors)
    return result._replace(value=result.value.T)
