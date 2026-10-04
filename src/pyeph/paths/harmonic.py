"""Analytic prescribed oscillator paths with canonical coordinate conventions."""

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np


@dataclass(frozen=True)
class HarmonicPath:
    q0: object
    p0: object
    frequencies: object
    masses: object = 1.0
    origin: float = 0.0

    def __post_init__(self):
        q, p, w, m = map(np.asarray, (self.q0, self.p0, self.frequencies, self.masses))
        if q.shape != p.shape or q.ndim < 1:
            raise ValueError("q0 and p0 must have the same non-scalar shape")
        for array in (q, p, w, m):
            if np.iscomplexobj(array) or not np.isfinite(array).all():
                raise ValueError("harmonic path data must be real and finite")
        if np.any(w < 0) or np.any(m <= 0):
            raise ValueError("frequencies must be nonnegative and masses positive")
        origin = np.asarray(self.origin)
        if origin.ndim or np.iscomplexobj(origin) or not np.isfinite(origin):
            raise ValueError("harmonic path origin must be a finite real scalar")
        object.__setattr__(self, "origin", float(origin))
        np.broadcast_to(w, q.shape)
        np.broadcast_to(m, q.shape)
        for name in ("q0", "p0", "frequencies", "masses"):
            object.__setattr__(self, name, jnp.array(
                getattr(self, name), dtype=jnp.result_type(1.0), copy=True))

    def position(self, time):
        t = time - self.origin
        w = self.frequencies
        # sinc implements the free-particle limit at zero frequency.
        return self.q0 * jnp.cos(w * t) + self.p0 / self.masses * t * jnp.sinc(w * t / jnp.pi)

    def velocity(self, time):
        t = time - self.origin
        w = self.frequencies
        return -self.q0 * w * jnp.sin(w * t) + self.p0 / self.masses * jnp.cos(w * t)


@dataclass(frozen=True)
class ConstantPath:
    q: object

    def __post_init__(self):
        q = np.asarray(self.q)
        if q.ndim < 1 or np.iscomplexobj(q) or not np.isfinite(q).all():
            raise ValueError("constant path coordinates must be a finite real nonscalar array")
        object.__setattr__(self, "q", jnp.array(q, dtype=jnp.result_type(1.0), copy=True))

    def position(self, time):
        return jnp.asarray(self.q, dtype=jnp.result_type(1.0))

    def velocity(self, time):
        return jnp.zeros_like(self.position(time))


@dataclass(frozen=True)
class HarmonicBath:
    """Prescribed harmonic evolution with independent initial data in each state.

    Unlike a shared recorded path, this schedule advances each trajectory's own
    canonical q,p. It remains CPA: electronic amplitudes never enter its motion.
    """

    frequencies: object
    masses: object = 1.0
    prescribed = True

    def __post_init__(self):
        # These arrays are static kernel inputs; freeze their values at ingestion.
        for name in ("frequencies", "masses"):
            value = np.asarray(getattr(self, name))
            if np.iscomplexobj(value) or not np.isfinite(value).all():
                raise ValueError("harmonic bath frequencies and masses must be real and finite")
            object.__setattr__(self, name, jnp.array(value, copy=True))

    def validate(self, q_shape):
        w, m = np.asarray(self.frequencies), np.asarray(self.masses)
        np.broadcast_to(w, q_shape)
        np.broadcast_to(m, q_shape)
        if not np.isfinite(w).all() or np.any(w < 0):
            raise ValueError("frequencies must be finite and nonnegative")
        if not np.isfinite(m).all() or np.any(m <= 0):
            raise ValueError("masses must be finite and positive")

    def point(self, state, elapsed):
        w, m = jnp.asarray(self.frequencies), jnp.asarray(self.masses)
        phase = w * elapsed
        q = state.q * jnp.cos(phase) + state.p / m * elapsed * jnp.sinc(phase / jnp.pi)
        p = state.p * jnp.cos(phase) - m * w * state.q * jnp.sin(phase)
        return q, p


def legacy_quadratures_to_canonical(x, y, frequencies):
    """Convert PyEPH's dimensionless oscillator fields to unit-mass Q, P."""
    w = jnp.asarray(frequencies)
    return jnp.asarray(x) / jnp.sqrt(2 * w), jnp.asarray(y) * jnp.sqrt(w / 2)
