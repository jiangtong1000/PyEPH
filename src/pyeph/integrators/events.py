"""Bounded scalar crossing localization for pure JAX dynamics kernels."""

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

from pyeph.core._configuration import boolean_scalar, real_scalar


class CrossingRoot(NamedTuple):
    time: Any
    residual: Any
    bracket_width: Any
    iterations: Any
    converged: Any


class BracketedCrossing(NamedTuple):
    time: Any
    residual: Any
    lower: Any
    upper: Any
    iterations: Any
    converged: Any


def bisect_bracketed_crossing(value_at, duration, *, iterations=48,
                              population_tolerance=1e-10, time_tolerance=1e-10,
                              both_endpoints=False):
    """Localize a sign bracket with both residual and time-width requirements.

    This stricter variant keeps the returned point inside the final bracket,
    even for a nonmonotone function. It does not discover hidden recrossings or
    guarantee that the bracket contains only one root. Exact terminal zeros
    are accepted; an initial zero is not automatically returned because it may
    be an outgoing boundary followed by a later incoming crossing.
    ``both_endpoints=True`` requires both retained endpoint residuals to meet
    the population tolerance, allowing a caller to choose either physical side
    of the event. The returned point still has the smaller absolute residual.
    """
    if not isinstance(iterations, int) or isinstance(iterations, bool) or iterations < 1:
        raise ValueError("iterations must be a positive integer")
    population_tolerance = real_scalar(population_tolerance, "population_tolerance")
    time_tolerance = real_scalar(time_tolerance, "time_tolerance")
    both_endpoints = boolean_scalar(both_endpoints, "both_endpoints")
    for value in (population_tolerance, time_tolerance):
        if value <= 0:
            raise ValueError("crossing tolerances must be finite and positive")
    duration = jnp.asarray(duration)
    if duration.ndim or not (jnp.issubdtype(duration.dtype, jnp.floating)
                            or jnp.issubdtype(duration.dtype, jnp.integer)):
        raise ValueError("crossing duration must be one real scalar")
    duration = duration.astype(jnp.result_type(duration, 1.0))
    zero = jnp.zeros_like(duration)
    f0, f1 = jnp.asarray(value_at(zero)), jnp.asarray(value_at(duration))
    if (f0.ndim or f1.ndim or any(not (jnp.issubdtype(f.dtype, jnp.floating)
                                      or jnp.issubdtype(f.dtype, jnp.integer)) for f in (f0, f1))):
        raise ValueError("a crossing function must return one real scalar")
    f0, f1 = (f.astype(jnp.result_type(f0, f1, duration)) for f in (f0, f1))
    valid = (jnp.isfinite(duration) & (duration > 0) & jnp.isfinite(f0)
             & jnp.isfinite(f1) & (f0 >= 0) & (f1 <= 0))
    terminal = f1 == 0
    lower, fl = jnp.where(terminal, duration, zero), jnp.where(terminal, f1, f0)

    def residual(fl, fu):
        return jnp.where(jnp.abs(fl) < jnp.abs(fu), fl, fu)

    def population_error(fl, fu):
        return (jnp.maximum(jnp.abs(fl), jnp.abs(fu)) if both_endpoints
                else jnp.abs(residual(fl, fu)))

    def condition(carry):
        lo, hi, fl, fu, count, ok = carry
        unresolved = ((hi-lo > time_tolerance)
                      | (population_error(fl, fu) > population_tolerance))
        return ok & (count < iterations) & unresolved

    def body(carry):
        lo, hi, fl, fu, count, ok = carry
        middle = lo+(hi-lo)/2
        fm = jnp.asarray(value_at(middle), dtype=fl.dtype)
        exact = fm == 0
        next_lo = jnp.where(fm >= 0, middle, lo)
        next_hi = jnp.where((fm < 0) | exact, middle, hi)
        next_fl = jnp.where(fm >= 0, fm, fl)
        next_fu = jnp.where((fm < 0) | exact, fm, fu)
        progressed = (middle > lo) & (middle < hi)
        return (next_lo, next_hi, next_fl, next_fu, count+1,
                ok & jnp.isfinite(fm) & (progressed | exact))

    lo, hi, fl, fu, count, ok = jax.lax.while_loop(
        condition, body, (lower, duration, fl, f1, jnp.int32(0), valid))
    use_lower = jnp.abs(fl) < jnp.abs(fu)
    time, value = jnp.where(use_lower, lo, hi), residual(fl, fu)
    converged = ok & (hi-lo <= time_tolerance) & (population_error(fl, fu) <= population_tolerance)
    return BracketedCrossing(time, value, lo, hi, count, converged)


def bisect_crossing(value_at, duration, *, iterations=40, tolerance=1e-10):
    """Locate a positive-to-negative crossing in ``[0,duration]``.

    ``value_at(0) >= 0`` (up to ``tolerance``) and ``value_at(duration) < 0``
    are required. This localizer cannot discover unbracketed recrossings. The
    caller must subdivide the trajectory and check convergence with subdivision.
    The returned residual is signed. A finite iteration budget never silently
    accepts an unconverged root. Endpoint roots are allowed.
    """
    if not isinstance(iterations, int) or iterations < 1:
        raise ValueError("iterations must be a positive integer")
    if tolerance <= 0:
        raise ValueError("tolerance must be positive")
    duration = jnp.asarray(duration)
    zero = jnp.zeros_like(duration)
    f0, f1 = value_at(zero), value_at(duration)
    bracketed = (jnp.isfinite(f0) & jnp.isfinite(f1) & (duration > 0)
                 & (f0 >= -tolerance) & (f1 < 0))
    # Never return the initial equator merely because its residual is zero:
    # following a hop the spin can leave that boundary and cross it again.
    # A crossing genuinely at t=0 is still approached by bisection.
    root, residual = duration, f1

    def condition(carry):
        _, _, _, residual, count, valid = carry
        return valid & (count < iterations) & (jnp.abs(residual) > tolerance)

    def body(carry):
        lower, upper, best, residual, count, valid = carry
        midpoint = (lower + upper) / 2
        value = value_at(midpoint)
        improve = jnp.abs(value) < jnp.abs(residual)
        best = jnp.where(improve, midpoint, best)
        residual = jnp.where(improve, value, residual)
        lower = jnp.where(value >= 0, midpoint, lower)
        upper = jnp.where(value < 0, midpoint, upper)
        return lower, upper, best, residual, count + 1, valid & jnp.isfinite(value)

    lower, upper, root, residual, count, valid = jax.lax.while_loop(
        condition, body, (zero, duration, root, residual, jnp.int32(0), bracketed)
    )
    return CrossingRoot(root, residual, upper - lower, count,
                        valid & (jnp.abs(residual) <= tolerance))
