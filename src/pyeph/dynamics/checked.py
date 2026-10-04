"""Diagnostics and scalar acceptance gates for checked electronic macrosteps.

These values are numerical diagnostics, separate from the physical method state.
The gates act between completed actions/stages; they cannot cancel work already
in flight inside a batched operator action.
"""

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

from pyeph.core.contracts import prepared_action
from pyeph.integrators.krylov import LanczosOptions, LanczosResult, lanczos_action

PHASE_NONE = 0
PHASE_CPA = 1
PHASE_EHRENFEST_FIRST = 2
PHASE_FORCE_FIRST = 3
PHASE_FORCE_SECOND = 4
PHASE_EHRENFEST_SECOND = 5
PHASE_ENDPOINT = 6
PHASE_NAMES = {
    PHASE_NONE: "initial state",
    PHASE_CPA: "CPA electronic action",
    PHASE_EHRENFEST_FIRST: "Ehrenfest first electronic half",
    PHASE_FORCE_FIRST: "Ehrenfest first force and drift",
    PHASE_FORCE_SECOND: "Ehrenfest second force and kick",
    PHASE_EHRENFEST_SECOND: "Ehrenfest second electronic half",
    PHASE_ENDPOINT: "macrostep endpoint",
}


class CheckedStepInfo(NamedTuple):
    """First failure and latest attempted action, with fixed per-column budgets.

    code: 0 success, 1 failed action, 2 nonfinite physical stage, 3 accumulated
    action estimate exceeds the macrostep budget. The action's own status gives
    the reason for code 1. Batch failures roll back the entire macrostep.
    """

    code: Any
    phase: Any
    substep: Any
    failed_trajectories: Any
    action: LanczosResult
    accumulated_error_estimate: Any
    macrostep_budget: Any


def empty_checked_info(state, *, batch=False):
    """Allocate shape-only diagnostics without invoking a model or measurement."""
    axis = 1 if batch else 0
    shape = state.electronic.shape[:axis] + state.electronic.shape[axis + 1:]
    zero = jnp.zeros(shape, state.electronic.real.dtype)
    count = jnp.zeros(shape, jnp.int32)
    action = LanczosResult(state.electronic, zero, zero, zero, zero, zero, zero,
                           count, count)
    trajectories = (state.electronic.shape[0],) if batch else ()
    return CheckedStepInfo(jnp.int32(0), jnp.int32(PHASE_NONE), jnp.int32(-1),
                           jnp.zeros(trajectories, bool), action, zero, zero)


def validate_checked_model(problem, integrator):
    if not isinstance(integrator.electronic, LanczosOptions):
        raise ValueError("a checked electronic step requires LanczosOptions")
    if not problem.model.spec.native_jax:
        raise ValueError("checked Lanczos propagation currently requires a native JAX model")


def _trajectory_mask(mask, *, batch):
    """Reduce coordinate or column failures, preserving only a batch axis."""
    mask = jnp.asarray(mask, bool)
    if batch:
        return jnp.any(mask, axis=tuple(range(1, mask.ndim)))
    return jnp.any(mask)


def check_finite(info, *values, phase, batch):
    """Check an already completed physical stage; keep earlier failures intact."""
    bad = jnp.zeros_like(info.failed_trajectories)
    for value in values:
        bad = bad | _trajectory_mask(~jnp.isfinite(value), batch=batch)
    failed = (info.code == 0) & jnp.any(bad)
    return info._replace(
        code=jnp.where(failed, jnp.int32(2), info.code),
        phase=jnp.where(failed, jnp.int32(phase), info.phase),
        substep=jnp.where(failed, jnp.int32(-1), info.substep),
        failed_trajectories=jnp.where(failed, bad, info.failed_trajectories),
    )


def initial_checked_info(state, options, *, batch):
    if state.electronic.dtype != jnp.dtype(jnp.complex128):
        raise ValueError("checked propagation requires complex128 electronic state")
    if any(jnp.asarray(value).dtype != jnp.dtype(jnp.float64)
           for value in (state.q, state.p, state.time)):
        raise ValueError("checked propagation requires float64 q, p, and time")
    info = empty_checked_info(state, batch=batch)
    axis = 1 if batch else 0
    magnitude = jnp.abs(state.electronic)
    scale = jnp.max(magnitude, axis=axis, keepdims=True)
    normalized = magnitude / jnp.where(scale > 0, scale, 1.0)
    norm = jnp.squeeze(scale, axis=axis) * jnp.sqrt(jnp.sum(normalized**2, axis=axis))
    budget = options.atol + options.rtol * norm
    info = info._replace(macrostep_budget=budget)
    return check_finite(info, state.q, state.p, state.electronic, state.time, budget,
                        phase=PHASE_NONE, batch=batch)


def electronic_action(model, params, q, electronic, duration, options, budget, *, batch):
    def action(qi, ci, bi):
        # Geometry work stays outside the kernel's vector/column iterations.
        # The closure is consumed here, never cached across states or parameters.
        apply = prepared_action(model, params, qi)
        return lanczos_action(apply, ci, duration, options, absolute_budget=bi)
    return jax.vmap(action)(q, electronic, budget) if batch else action(q, electronic, budget)


def record_action(info, action, *, phase, substep, batch):
    bad = _trajectory_mask(action.status != 0, batch=batch)
    return info._replace(
        code=jnp.where(jnp.any(bad), jnp.int32(1), jnp.int32(0)),
        phase=jnp.int32(phase), substep=jnp.asarray(substep, jnp.int32),
        failed_trajectories=bad, action=action,
        accumulated_error_estimate=info.accumulated_error_estimate + action.error_estimate,
    )


def fixed_geometry_actions(model, params, q, electronic, info, duration, options,
                           substeps, *, phase, batch):
    """Advance one frozen-geometry half, respecting the original macro budget."""
    action_duration = duration / substeps
    # This helper is the Ehrenfest half; both halves share one original budget.
    allocation = info.macrostep_budget / (2 * substeps)

    def body(index, carry):
        def advance(carry):
            c, diagnostic = carry
            action = electronic_action(model, params, q, c, action_duration, options,
                                       allocation, batch=batch)
            diagnostic = record_action(diagnostic, action, phase=phase,
                                       substep=index, batch=batch)
            return action.value, diagnostic
        return jax.lax.cond(carry[1].code == 0, advance, lambda x: x, carry)

    return jax.lax.fori_loop(0, substeps, body, (electronic, info))


def finish_checked_step(initial, candidate, info, *, batch):
    """Commit every physical field together, only after the final budget gate."""
    info = check_finite(info, candidate.q, candidate.p, candidate.electronic, candidate.time,
                        phase=PHASE_ENDPOINT, batch=batch)
    bad = _trajectory_mask(
        ~jnp.isfinite(info.accumulated_error_estimate)
        | (info.accumulated_error_estimate > info.macrostep_budget), batch=batch)
    failed = (info.code == 0) & jnp.any(bad)
    info = info._replace(
        code=jnp.where(failed, jnp.int32(3), info.code),
        phase=jnp.where(failed, jnp.int32(PHASE_ENDPOINT), info.phase),
        substep=jnp.where(failed, jnp.int32(-1), info.substep),
        failed_trajectories=jnp.where(failed, bad, info.failed_trajectories),
    )
    state = jax.lax.cond(info.code == 0, lambda _: candidate, lambda _: initial, operand=None)
    return state, info
