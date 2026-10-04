"""Trajectory state is a PyTree of arrays; batching adds one leading axis."""

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np


class TrajectoryState(NamedTuple):
    q: Any
    p: Any
    electronic: Any
    time: Any
    step: Any
    trajectory_id: Any
    key: Any
    method_state: Any = ()


class ElectronicPathState(NamedTuple):
    """State for recorded electronic data without invented nuclear coordinates."""

    electronic: Any
    time: Any
    step: Any
    frame_index: Any


def _snapshot_state_leaf(value):
    """Own array storage without narrowing host method-state integer counters."""
    if isinstance(value, np.ndarray):
        snapshot = np.array(value, copy=True)
        snapshot.flags.writeable = False
        return snapshot
    if isinstance(value, jax.Array):
        return value.copy()
    return value


def make_state(q, p, electronic, *, time=0.0, step=0, trajectory_id=0, seed=0, method_state=()):
    """Build and validate one state; electronic columns may be propagated together.

    This constructor does not normalize electronic data: full propagators and
    response blocks need their actual column weights preserved.
    Input arrays and method-state containers are snapshotted, so later caller
    mutations do not change the initial state. Host method-state arrays retain
    their exact dtype, including wide integer counters when JAX x64 is disabled.
    """
    if np.iscomplexobj(q) or np.iscomplexobj(p):
        raise ValueError("nuclear coordinates and momenta must be real")
    q = jnp.array(q, dtype=jnp.result_type(1.0), copy=True)
    p = jnp.array(p, dtype=q.dtype, copy=True)
    electronic = jnp.array(electronic, dtype=jnp.result_type(q, 1j), copy=True)
    if q.shape != p.shape or q.ndim < 1:
        raise ValueError("q and p must have the same non-scalar shape")
    if electronic.ndim not in (1, 2):
        raise ValueError("electronic state must be a vector or column block")
    if not all(np.isfinite(np.asarray(x)).all() for x in (q, p, electronic)):
        raise ValueError("initial state contains nonfinite values")
    step_limit = np.iinfo(np.int64 if jax.config.x64_enabled else np.int32).max
    if (np.ndim(time) != 0 or np.iscomplexobj(time) or not np.isfinite(time)
            or not isinstance(step, (int, np.integer)) or isinstance(step, bool)
            or not 0 <= step <= step_limit):
        raise ValueError("time must be finite and step a nonnegative representable integer")
    if (not isinstance(trajectory_id, (int, np.integer)) or isinstance(trajectory_id, bool)
            or not 0 <= trajectory_id < 2**32):
        raise ValueError("trajectory_id must be a nonnegative uint32 integer")
    key = jax.random.key_data(jax.random.fold_in(
        jax.random.key(seed, impl="threefry2x32"), np.uint32(trajectory_id)))
    return TrajectoryState(
        q, p, electronic, jnp.array(time, dtype=q.dtype, copy=True),
        jnp.asarray(step, dtype=jnp.int64 if jax.config.x64_enabled else jnp.int32),
        jnp.asarray(trajectory_id, dtype=jnp.uint32), key,
        jax.tree.map(_snapshot_state_leaf, method_state),
    )


def stack_states(states):
    if not states:
        raise ValueError("cannot create an empty trajectory batch")
    ids = [int(s.trajectory_id) for s in states]
    if len(ids) != len(set(ids)):
        raise ValueError("trajectory IDs must be unique within a batch")
    return jax.tree.map(lambda *x: jnp.stack(x), *states)
