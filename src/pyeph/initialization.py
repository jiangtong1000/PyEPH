"""Initial ensembles with explicit coordinate and temperature conventions."""

import jax
import jax.numpy as jnp
import numpy as np

from pyeph.execution.random import trajectory_keys


def sample_harmonic(frequencies, masses, temperature, trajectory_ids, *, seed=0,
                    distribution="classical"):
    """Canonical oscillator q,p, with kB=hbar=1 and batch axis first.

    Frequencies and masses broadcast to the nuclear coordinate shape. A zero
    frequency has no normalizable position distribution and is rejected; sample
    free translations separately. Wigner sampling includes zero-point variance.
    """
    w, m = np.broadcast_arrays(np.asarray(frequencies, float), np.asarray(masses, float))
    if w.ndim == 0:
        w, m = w.reshape(1), m.reshape(1)
    if not np.isfinite(w).all() or np.any(w <= 0):
        raise ValueError("thermal oscillator frequencies must be finite and strictly positive")
    if not np.isfinite(m).all() or np.any(m <= 0):
        raise ValueError("masses must be finite and positive")
    if not np.isfinite(temperature) or temperature < 0:
        raise ValueError("temperature must be finite and nonnegative in energy units")
    if distribution not in {"classical", "wigner"}:
        raise ValueError("distribution must be classical or wigner")
    w, m = jnp.asarray(w), jnp.asarray(m)
    if distribution == "classical":
        qvar, pvar = temperature / (m * w**2), m * temperature
    else:
        occupation = jnp.ones_like(w) if temperature == 0 else 1 / jnp.tanh(w / (2 * temperature))
        qvar, pvar = occupation / (2 * m * w), occupation * m * w / 2
    keys = trajectory_keys(seed, trajectory_ids, typed=True)

    def one(key):
        qkey, pkey = jax.random.split(key)
        return (jax.random.normal(qkey, w.shape, dtype=w.dtype) * jnp.sqrt(qvar),
                jax.random.normal(pkey, w.shape, dtype=w.dtype) * jnp.sqrt(pvar))

    return jax.vmap(one)(keys)
