"""Pure trajectory blocks with dense, sparse, or disabled observation output.

The runner owns validation, caching, compilation, and publication. Builders here
capture its static configuration; numerical parameters remain runtime arguments.
Ordinary and checked propagation keep separate control flow because a checked
step can reject the entire batch and must suppress all subsequent stages.
"""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np


def build_observer(problem, measurement, *, batch):
    """Build an initial measurement with dynamic parameters and trajectory state."""
    def observe(params, state):
        runtime_problem = replace(problem, params=params)

        def evaluate(single):
            return measurement.evaluate(runtime_problem, single)

        return jax.vmap(evaluate)(state) if batch else evaluate(state)

    return observe


def sample_schedule(nsteps, sample_indices):
    """Normalize a private block's sample selection to its immutable cache key."""
    if not isinstance(nsteps, int) or isinstance(nsteps, bool) or nsteps < 0:
        raise ValueError("block nsteps must be a nonnegative integer")
    if sample_indices is None:
        return tuple(range(nsteps))
    selected = np.asarray(sample_indices)
    if (selected.ndim != 1 or (selected.size and
            (not np.issubdtype(selected.dtype, np.integer)
             or np.any(selected < 0) or np.any(selected >= nsteps)
             or np.any(selected[1:] <= selected[:-1])))):
        raise ValueError("sample_indices must be sorted unique in-range integer indices")
    return tuple(int(i) for i in selected)


def _observation_buffers(observe, state, count):
    # Shape evaluation is abstract: no measurements or callbacks run.
    output_shape = jax.eval_shape(observe, state)
    values = jax.tree.map(lambda x: jnp.zeros((count,) + x.shape, dtype=x.dtype), output_shape)
    times = jnp.zeros((count,) + state.time.shape, dtype=state.time.dtype)
    return times, values


def _save_mask(nsteps, indices):
    mask = np.zeros(nsteps, dtype=bool)
    mask[list(indices)] = True
    return jnp.asarray(mask)


def _save_observation(observe, state, buffers):
    cursor, times, values = buffers
    observation = observe(state)
    times = times.at[cursor].set(state.time)
    values = jax.tree.map(lambda buffer, value: buffer.at[cursor].set(value),
                          values, observation)
    return cursor + 1, times, values


def build_block(problem, integrator, measurement, *, batch, nsteps, indices, set_time):
    """Build an ordinary block; dense output avoids carried scatter buffers."""
    def block(params, state):
        # Never capture NN/array parameters as constants in the compiled step.
        runtime_problem = replace(problem, params=params)
        step = runtime_problem.method.build_step(runtime_problem, integrator)
        if batch:
            step = jax.vmap(step)

        def advance(s, index):
            s = set_time(s, state.time + index * integrator.dt, batch)
            s = step(s)
            s = set_time(s, state.time + (index + 1) * integrator.dt, batch)
            return s

        if not indices:
            # Do not even construct/trace the observer when output is disabled.
            def body(s, index):
                return advance(s, index), None
            final, _ = jax.lax.scan(body, state, xs=jnp.arange(nsteps))
            empty_time = jnp.empty((0,) + state.time.shape, dtype=state.time.dtype)
            return final, (empty_time, {})

        def observe(s):
            return measurement.evaluate(runtime_problem, s)

        if batch:
            observe = jax.vmap(observe)

        if len(indices) == nsteps:
            def body(s, index):
                s = advance(s, index)
                return s, (s.time, observe(s))
            return jax.lax.scan(body, state, xs=jnp.arange(nsteps))

        times, values = _observation_buffers(observe, state, len(indices))
        save_mask = _save_mask(nsteps, indices)

        def body(carry, index):
            s, cursor, times, values = carry
            s = advance(s, index)
            cursor, times, values = jax.lax.cond(
                save_mask[index], lambda buffers: _save_observation(observe, s, buffers),
                lambda buffers: buffers, (cursor, times, values))
            return (s, cursor, times, values), None

        (final, _, times, values), _ = jax.lax.scan(
            body, (state, jnp.int32(0), times, values), xs=jnp.arange(nsteps))
        return final, (times, values)

    return block


def build_checked_block(problem, integrator, measurement, *, batch, nsteps, indices):
    """Build scalar acceptance gates around explicitly batched method stages."""
    from pyeph.dynamics.checked import empty_checked_info

    def block(params, state):
        runtime_problem = replace(problem, params=params)
        step = runtime_problem.method.build_checked_step(runtime_problem, integrator, batch=batch)
        info = {"step_info": empty_checked_info(state, batch=batch),
                "failed_macro_index": jnp.int32(-1)}
        if indices:
            def observe(s):
                return measurement.evaluate(runtime_problem, s)
            if batch:
                observe = jax.vmap(observe)
            times, values = _observation_buffers(observe, state, len(indices))
            save_mask = _save_mask(nsteps, indices)
        else:
            times = jnp.zeros((0,) + state.time.shape, dtype=state.time.dtype)
            values = {}

        def attempt(carry, index):
            before, block_info, cursor, times, values = carry
            before = before._replace(time=state.time + index * integrator.dt)
            candidate, step_info = step(before)

            def accept(_):
                accepted = candidate._replace(time=state.time + (index + 1) * integrator.dt)
                buffers = (cursor, times, values)
                if indices:
                    buffers = jax.lax.cond(
                        save_mask[index],
                        lambda buffers: _save_observation(observe, accepted, buffers),
                        lambda buffers: buffers, buffers)
                return (accepted, block_info, *buffers)

            def reject(_):
                failed = {"step_info": step_info, "failed_macro_index": index}
                return before, failed, cursor, times, values

            return jax.lax.cond(step_info.code == 0, accept, reject, operand=None)

        def body(carry, index):
            result = jax.lax.cond(carry[1]["failed_macro_index"] < 0,
                                  lambda x: attempt(x, index), lambda x: x, carry)
            return result, None

        (final, info, _, times, values), _ = jax.lax.scan(
            body, (state, info, jnp.int32(0), times, values), xs=jnp.arange(nsteps, dtype=jnp.int32))
        return final, (times, values), info

    return block
