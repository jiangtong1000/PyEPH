"""Opt-in finite-temperature preparation from a declared Hermitian action.

This module constructs density factors, not dynamics methods. The caller owns
the operator's linearity, Hermiticity and full spectral enclosure; opaque
action probes cannot certify these contracts. See docs/THERMAL_COLUMNS.md.
"""

from dataclasses import dataclass
import hashlib
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from scipy.special import ive

from pyeph.core._configuration import integer_scalar, real_scalar
from pyeph.execution.random import trajectory_keys


STATUS = {
    0: "accepted conditional truncation bound and heuristic precision screen",
    1: "nonfinite input, action, recurrence or normalization",
    2: "zero filtered norm; improve spectral bounds or execution precision",
    3: "relative polynomial truncation budget exceeded; tighten the plan or spectral bounds",
    4: "heuristic precision screen exceeded; improve bounds or precision, or relax the target",
}


def _log_tail_bound(z, degree):
    if z == 0:
        return -np.inf
    n = float(degree+1)
    ratio = n/z
    t = np.arcsinh(ratio) if np.isfinite(ratio) else np.log(n+np.hypot(n, z))-np.log(z)
    return min(0., float(np.log(2.) + n*(n/(np.hypot(z, n)+z)) - n*t))


@dataclass(frozen=True, eq=False, init=False)
class ThermalFilterPlan:
    """Fixed host plan for F=exp[-beta*(H-lower*I)/2].

    Bounds and beta are finite host scalars and remain fixed during AD. A flat
    interval declares H=lower*I; beta=0 also gives the identity without calling
    the action. ``action_id`` and ``bounds_id`` are caller-owned identities for
    the operator artifacts and spectral-enclosure argument, not certificates
    inferred by this class. Coefficients are owned, read-only host float64 data.
    """

    beta: float
    lower: float
    upper: float
    polynomial_atol: float
    max_degree: int
    action_id: str
    bounds_id: str
    coefficients: Any
    log_truncation_bound: float

    def __init__(self, beta, lower, upper, *, action_id, bounds_id,
                 polynomial_atol=1e-12, max_degree=100000):
        beta, lower, upper, tolerance = (real_scalar(value, name) for value, name in (
            (beta, "beta"), (lower, "lower"), (upper, "upper"),
            (polynomial_atol, "polynomial_atol")))
        max_degree = integer_scalar(max_degree, "max_degree")
        if beta < 0 or lower > upper or not 1e-300 <= tolerance < 1 or max_degree < 0:
            raise ValueError("require beta>=0, lower<=upper, 1e-300<=polynomial_atol<1, "
                             "and max_degree>=0")
        for name, value in (("action_id", action_id), ("bounds_id", bounds_id)):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a nonempty caller-owned artifact identity")
        z = beta*((upper-lower)/4)
        if not np.isfinite(z) or not np.isfinite(beta*lower):
            raise ValueError("scaled spectral width and beta*lower must be representable")
        if z == 0 and beta > 0 and lower < upper:
            raise ValueError("positive scaled spectral width underflows host precision")
        if z == 0:
            coefficients, log_bound = np.array([1.]), -np.inf
        else:
            low, high, target = -1, 0, np.log(tolerance)
            while _log_tail_bound(z, high) > target and high < max_degree:
                low, high = high, min(max_degree, max(1, 2*high))
            if _log_tail_bound(z, high) > target:
                raise ValueError("requested polynomial truncation bound exceeds max_degree")
            while high-low > 1:
                middle = (low+high)//2
                if _log_tail_bound(z, middle) <= target:
                    high = middle
                else:
                    low = middle
            orders = np.arange(high+1)
            coefficients = ive(orders, z)*np.where(orders == 0, 1., 2*(-1.)**orders)
            if not np.isfinite(coefficients).all():
                raise FloatingPointError("scaled Bessel coefficients are nonfinite")
            log_bound = _log_tail_bound(z, high)
        coefficients.setflags(write=False)
        values = dict(beta=beta, lower=lower, upper=upper, polynomial_atol=tolerance,
                      max_degree=max_degree, action_id=action_id, bounds_id=bounds_id,
                      coefficients=coefficients, log_truncation_bound=log_bound)
        for name, value in values.items():
            object.__setattr__(self, name, value)

    @property
    def degree(self):
        return len(self.coefficients)-1

    @property
    def truncation_bound(self):
        if self.log_truncation_bound == -np.inf:
            return 0.
        return max(float(np.nextafter(0., 1.)), float(np.exp(self.log_truncation_bound)))

    def metadata(self):
        """JSON-ready recipe; save with column IDs, diagnostics and factor digest."""
        return dict(beta=self.beta, spectral_interval=[self.lower, self.upper],
                    polynomial_atol=self.polynomial_atol, degree=self.degree,
                    max_degree=self.max_degree, action_id=self.action_id, bounds_id=self.bounds_id,
                    coefficients_sha256=hashlib.sha256(self.coefficients.tobytes()).hexdigest(),
                    coefficient_dtype=str(self.coefficients.dtype),
                    conditional_exact_arithmetic_truncation_bound=self.truncation_bound,
                    log_truncation_bound=(None if self.log_truncation_bound == -np.inf
                                          else self.log_truncation_bound),
                    coefficient_method="scipy.special.ive",
                    bound_evaluation="ordinary host float64, not interval arithmetic",
                    floating_error_certificate=None)


def thermal_random_columns(nstates, column_ids, *, seed=0, trajectory_id=0, kind="complex"):
    """Identity-indexed unit phases/signs for one nuclear geometry, shape (N,K).

    A scalar column ID requests K=1; otherwise IDs must be a nonempty unique
    one-dimensional uint32-compatible array. Generate separate geometry blocks
    with their explicit trajectory IDs, then vmap preparation over those blocks.
    Permuting or partitioning IDs preserves columns before normalization.
    """
    nstates = integer_scalar(nstates, "nstates")
    seed = integer_scalar(seed, "seed")
    trajectory_id = integer_scalar(trajectory_id, "trajectory_id")
    if nstates < 1 or not 0 <= seed < 2**32 or not 0 <= trajectory_id < 2**32:
        raise ValueError("require positive nstates and uint32 seed/trajectory_id")
    ids = np.atleast_1d(np.asarray(column_ids))
    if (ids.ndim != 1 or not ids.size or ids.dtype.kind not in "iu"
            or np.any(ids < 0) or np.any(ids >= 2**32) or len(np.unique(ids)) != len(ids)):
        raise ValueError("column_ids must be nonempty unique uint32-compatible integers")
    if kind not in ("real", "complex"):
        raise ValueError("kind must be real or complex")
    key = trajectory_keys(seed, [trajectory_id], typed=True)[0]
    key = jax.random.fold_in(key, np.uint32(0x54484D46))
    keys = jax.vmap(lambda identity: jax.random.fold_in(key, identity))(jnp.asarray(ids, jnp.uint32))
    dtype = jnp.result_type(1.)

    def one(key):
        if kind == "real":
            return jnp.where(jax.random.bernoulli(key, shape=(nstates,)), 1., -1.).astype(dtype)
        return jnp.exp(2j*jnp.pi*jax.random.uniform(key, (nstates,), dtype=dtype))

    return jax.vmap(one)(keys).T


class ThermalColumnsResult(NamedTuple):
    """Factor and separate polynomial/normalization/precision diagnostics."""

    factor: Any
    status: Any
    relative_truncation_bound: Any
    normalized_factor_error_bound: Any
    density_trace_norm_error_bound: Any
    heuristic_precision_screen: Any
    norm_ratio: Any
    log_norm_squared: Any
    scaled_partition_estimate: Any
    scaled_partition_underflow: Any
    scaled_partition_overflow: Any
    log_partition_estimate: Any
    rtol: Any

    def diagnostics(self):
        """JSON-ready diagnostics after the host acceptance gate."""
        require_success(self)
        values = {name: np.asarray(getattr(self, name)).tolist()
                  for name in self._fields if name != "factor"}
        # Arbitrarily rescaled caller columns may overflow the raw squared
        # norm while the accepted normalized factor and log estimate are fine.
        raw = np.asarray(self.scaled_partition_estimate)
        values["scaled_partition_estimate"] = np.where(np.isfinite(raw), raw, None).tolist()
        return values


class ThermalPreparationError(RuntimeError):
    def __init__(self, result):
        self.result = result
        codes = np.unique(np.asarray(result.status))
        super().__init__("thermal preparation rejected: " + "; ".join(STATUS[int(c)] for c in codes))


def require_success(result):
    """Host acceptance gate; failures retain diagnostics in exception.result."""
    if np.any(np.asarray(result.status) != 0):
        raise ThermalPreparationError(result)
    return result.factor


def _scaled_norm_parts(value):
    scale = jnp.max(jnp.abs(value))
    divisor = jnp.where(scale > 0, scale, 1.)
    scaled = value/divisor
    length = jnp.sqrt(jnp.sum(jnp.abs(scaled)**2))
    return scaled, length, jnp.log(scale)+jnp.log(length)


def prepare_thermal_columns(action, columns, plan, *, rtol=1e-6):
    """Pure single-geometry JAX recurrence and pooled normalization.

    ``action(block)`` must preserve a numeric (N,K) shape and represent the
    declared Hermitian H. No identity/dense Hamiltonian/density is constructed.
    The returned factor is NaN on rejection; call ``require_success`` before
    a host dynamics initializer. Passing is conditional on caller contracts,
    not a roundoff-inclusive or physical thermal-accuracy certificate.

    The heuristic precision screen is eps*(degree+1)**2*||Omega||F/||Y||F,
    using actual execution dtype. It rejects severe normalization amplification
    but does not bound arbitrary action, coefficient or recurrence error.
    AD concerns action parameters at fixed beta, bounds, degree and columns;
    reverse mode may retain O(degree*N*K) temporary storage.
    """
    if not isinstance(plan, ThermalFilterPlan):
        raise TypeError("plan must be a ThermalFilterPlan")
    rtol = real_scalar(rtol, "rtol")
    if not 0 < rtol < 1:
        raise ValueError("rtol must lie strictly between zero and one")
    columns = jnp.asarray(columns)
    if columns.ndim != 2 or min(columns.shape) < 1 or columns.dtype.kind not in "iufc":
        raise ValueError("columns must be a nonempty numeric (nstates,rank) block")
    coefficients = jnp.asarray(plan.coefficients)
    previous = columns.astype(jnp.result_type(columns, coefficients))
    filtered = coefficients[0]*previous
    finite = jnp.all(jnp.isfinite(previous)) & jnp.all(jnp.isfinite(coefficients))
    if plan.degree:
        center, radius = plan.lower/2+plan.upper/2, (plan.upper-plan.lower)/2

        def scaled_action(value):
            acted = jnp.asarray(action(value))
            if acted.shape != value.shape or acted.dtype.kind not in "iufc":
                raise ValueError("Hamiltonian action must preserve numeric column-block shape")
            return (acted-center*value)/radius, jnp.all(jnp.isfinite(acted))

        current, action_finite = scaled_action(previous)
        previous = previous.astype(current.dtype)
        filtered = (filtered+coefficients[1]*current).astype(current.dtype)
        finite = finite & action_finite & jnp.all(jnp.isfinite(current))

        def advance(index, state):
            previous, current, filtered, finite = state
            acted, action_finite = scaled_action(current)
            following = 2*acted-previous
            filtered = filtered+coefficients[index]*following
            finite = (finite & action_finite & jnp.all(jnp.isfinite(following))
                      & jnp.all(jnp.isfinite(filtered)))
            return current, following, filtered, finite

        _, _, filtered, finite = jax.lax.fori_loop(
            2, plan.degree+1, advance, (previous, current, filtered, finite))
    scaled, length, log_norm = _scaled_norm_parts(filtered)
    _, input_length, log_input_norm = _scaled_norm_parts(columns.astype(filtered.dtype))
    log_ratio = log_norm-log_input_norm
    relative = jnp.exp(jnp.asarray(plan.log_truncation_bound)-log_ratio)
    if plan.log_truncation_bound != -np.inf:
        # Keep positive mathematical tails distinct from the exact identity;
        # the smallest normal survives backends that flush subnormal values.
        relative = jnp.maximum(relative, jnp.finfo(filtered.real.dtype).tiny)
    epsilon = jnp.finfo(filtered.real.dtype).eps
    screen = jnp.exp(jnp.log(epsilon)+2*jnp.log(plan.degree+1.)-log_ratio)
    factor = scaled/jnp.where(length > 0, length, 1.)
    log_scaled_partition = 2*log_norm-jnp.log(columns.shape[1])
    partition = jnp.exp(log_scaled_partition)
    log_partition = log_scaled_partition-plan.beta*plan.lower
    zero_norm = (length == 0) | (input_length == 0)
    finite = (finite & jnp.all(jnp.isfinite(filtered)) & jnp.all(jnp.isfinite(factor))
              & jnp.isfinite(log_partition) & jnp.isfinite(log_ratio))
    status = jnp.where(screen > rtol, 4, 0)
    status = jnp.where(relative > rtol, 3, status)
    status = jnp.where(~finite, 1, status)
    status = jnp.where(zero_norm, 2, status).astype(jnp.int32)
    factor = jnp.where(status == 0, factor, jnp.nan)
    return ThermalColumnsResult(
        factor, status, relative, jnp.minimum(2., 2*relative), jnp.minimum(2., 4*relative),
        screen, jnp.exp(log_ratio), 2*log_norm, partition,
        (partition == 0) & jnp.isfinite(log_scaled_partition),
        jnp.isinf(partition) & jnp.isfinite(log_scaled_partition), log_partition,
        jnp.asarray(rtol, dtype=filtered.real.dtype))


__all__ = ["ThermalFilterPlan", "ThermalColumnsResult", "ThermalPreparationError",
           "thermal_random_columns", "prepare_thermal_columns", "require_success"]
