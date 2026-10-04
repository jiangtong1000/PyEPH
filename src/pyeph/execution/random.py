"""Stable random streams indexed by trajectory identity, not batch partition."""

import jax
import jax.numpy as jnp
import numpy as np


def as_threefry_key(key):
    """Interpret a stored uint32 pair as Threefry, independent of JAX defaults."""
    if jax.dtypes.issubdtype(key.dtype, jax.dtypes.prng_key):
        key = jax.random.key_data(key)
    return jax.random.wrap_key_data(key, impl="threefry2x32")


def trajectory_keys(seed, trajectory_ids, *, typed=False):
    """Identity-indexed Threefry keys; default output retains raw uint32 pairs.

    Set ``typed=True`` for keys consumed by JAX random operations, so an
    ambient PRNG implementation cannot reinterpret the stored two-word format.
    """
    ids = np.asarray(trajectory_ids)
    if ids.ndim != 1 or not np.issubdtype(ids.dtype, np.integer):
        raise ValueError("trajectory_ids must be a one-dimensional integer array")
    if np.any(ids < 0) or np.any(ids >= 2**32) or np.unique(ids).size != ids.size:
        raise ValueError("trajectory IDs must be unique nonnegative uint32 values")
    keys = jax.vmap(lambda i: jax.random.fold_in(
        jax.random.key(seed, impl="threefry2x32"), i))(
        jnp.asarray(ids, dtype=jnp.uint32))
    return keys if typed else jax.random.key_data(keys)


def event_key(key, step, event=0, *, typed=False):
    """Stateless Threefry stream for a declared (step,event).

    Retries/chunks do not change the stream. Raw uint32 pairs remain the
    default return format; use ``typed=True`` for subsequent random draws.
    """
    # Two words preserve step identity beyond the uint32 wrap in x64 runs.
    step = jnp.asarray(step)
    low = step.astype(jnp.uint32)
    high = (step >> 32).astype(jnp.uint32) if step.dtype.itemsize > 4 else jnp.uint32(0)
    key = as_threefry_key(key)
    key = jax.random.fold_in(jax.random.fold_in(jax.random.fold_in(key, high), low), event)
    return key if typed else jax.random.key_data(key)
