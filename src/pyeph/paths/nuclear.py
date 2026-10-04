"""Recorded Cartesian/canonical paths with a consistent position/velocity interpolant."""

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np


def _validated_times(times):
    times = np.asarray(times)
    if times.ndim != 1 or len(times) < 2:
        raise ValueError("times must be a one-dimensional array with at least two frames")
    if np.iscomplexobj(times) or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("recorded times must be finite and strictly increasing")
    converted = jnp.array(times, dtype=jnp.result_type(1.0), copy=True)
    if not np.isfinite(np.asarray(converted)).all() or np.any(np.diff(np.asarray(converted)) <= 0):
        raise ValueError("the time grid loses finiteness or ordering at the configured precision")
    return converted


class RecordedTimeDomain:
    """Host checks separate from pure scalar-time interpolation kernels."""

    @property
    def start_time(self):
        return float(self.times[0])

    @property
    def end_time(self):
        return float(self.times[-1])

    @property
    def domain(self):
        return self.start_time, self.end_time

    def contains(self, time):
        t = jnp.asarray(time)
        tolerance = self._time_tolerance(t)
        return jnp.isfinite(t) & (t >= self.times[0] - tolerance) & (t <= self.times[-1] + tolerance)

    def _time_tolerance(self, time):
        scale = jnp.maximum(1., jnp.maximum(jnp.max(jnp.abs(self.times)), jnp.abs(time)))
        return 16 * jnp.finfo(self.times.dtype).eps * scale

    def validate_time(self, time):
        """Raise on the host for any requested time outside the closed domain."""
        t = np.asarray(time)
        if np.iscomplexobj(t) or not np.isfinite(t).all():
            raise ValueError("requested times must be real and finite")
        scale = np.maximum(1., np.maximum(max(abs(self.start_time), abs(self.end_time)), np.abs(t)))
        tolerance = 16 * np.finfo(self.times.dtype).eps * scale
        if np.any((t < self.start_time - tolerance) | (t > self.end_time + tolerance)):
            raise ValueError(f"requested time is outside recorded domain {self.domain}")

    def validate_span(self, start, stop):
        self.validate_time([start, stop])
        if stop < start:
            raise ValueError("a simulation span must have stop >= start")

    def _interval(self, time):
        t = jnp.asarray(time)
        if t.ndim or jnp.issubdtype(t.dtype, jnp.complexfloating):
            raise ValueError("interpolation expects one real scalar time; use vmap for a batch")
        valid = self.contains(t)
        # Snap roundoff outside endpoints, but retain the original validity flag
        # so material extrapolation returns NaN rather than a clipped answer.
        # Strict comparisons preserve the interpolant's one-sided derivative
        # at an exact endpoint (jnp.clip has a half derivative at equality).
        t = jnp.where(t < self.times[0], self.times[0],
                      jnp.where(t > self.times[-1], self.times[-1], t))
        index = jnp.clip(jnp.searchsorted(self.times, t, side="right") - 1,
                         0, len(self.times) - 2)
        width = self.times[index + 1] - self.times[index]
        fraction = (t - self.times[index]) / width
        return index, width, fraction, valid


@dataclass(frozen=True, eq=False)
class RecordedNuclearPath(RecordedTimeDomain):
    """Piecewise cubic Hermite path through supplied q and dq/dt samples.

    Velocities are coordinate velocities, not canonical momenta. Position and
    velocity are continuous; acceleration can jump at a knot. No extrapolation
    is performed: JIT kernels return NaN outside the recorded domain, while
    validate_time/validate_span provide explicit host exceptions before a run.
    """

    times: object
    positions: object
    velocities: object
    interpolation = "cubic_hermite"

    def __post_init__(self):
        times = _validated_times(self.times)
        q, v = np.asarray(self.positions), np.asarray(self.velocities)
        if q.ndim < 2 or q.shape != v.shape or q.shape[0] != len(times) or 0 in q.shape:
            raise ValueError("positions and velocities must have equal (frames, *q_shape) shapes")
        if np.iscomplexobj(q) or np.iscomplexobj(v) or not all(
            np.isfinite(x).all() for x in (q, v)
        ):
            raise ValueError("recorded nuclear data must be real and finite")
        object.__setattr__(self, "times", times)
        object.__setattr__(self, "positions", jnp.array(q, dtype=times.dtype, copy=True))
        object.__setattr__(self, "velocities", jnp.array(v, dtype=times.dtype, copy=True))

    def _evaluate(self, time):
        i, h, s, valid = self._interval(time)
        q0, q1 = self.positions[i], self.positions[i + 1]
        v0, v1 = self.velocities[i], self.velocities[i + 1]
        q = ((2*s**3 - 3*s**2 + 1)*q0 + (s**3 - 2*s**2 + s)*h*v0
             + (-2*s**3 + 3*s**2)*q1 + (s**3 - s**2)*h*v1)
        v = ((6*s**2 - 6*s)*q0/h + (3*s**2 - 4*s + 1)*v0
             + (-6*s**2 + 6*s)*q1/h + (3*s**2 - 2*s)*v1)
        return jnp.where(valid, q, jnp.nan), jnp.where(valid, v, jnp.nan)

    def position(self, time):
        return self._evaluate(time)[0]

    def velocity(self, time):
        return self._evaluate(time)[1]
