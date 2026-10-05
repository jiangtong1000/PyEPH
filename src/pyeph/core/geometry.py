"""Explicit coordinate domains for scalar checked dynamics.

A box certifies membership of the supplied float64 coordinates only. It is not
neighbor-list coverage, provider eligibility, a bound on internal arithmetic,
or evidence of physical model accuracy. No graph is rebuilt automatically.
"""
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np

_SIGN = np.uint64(1 << 63)
_MAGNITUDE = np.uint64((1 << 63) - 1)
_EXPONENT = np.uint64(0x7FF0000000000000)


def _keys(array):
    raw = array.view(np.uint64)
    raw = np.where((raw & _MAGNITUDE) == 0, np.uint64(0), raw)
    return np.where((raw & _SIGN) != 0, ~raw, raw ^ _SIGN)


@dataclass(frozen=True)
class CoordinateBox:
    """Closed, immutable user-declared domain in the model's coordinate units.

    Bounds have exactly the model's coordinate shape. Membership uses integer
    ordering of IEEE binary64 inputs, including signed zero and subnormals;
    it performs no floating-point subtraction or distance calculation.
    """
    lower: object
    upper: object
    shape: tuple = field(init=False)
    _lower_keys: tuple = field(init=False, repr=False)
    _upper_keys: tuple = field(init=False, repr=False)

    def __post_init__(self):
        if np.iscomplexobj(self.lower) or np.iscomplexobj(self.upper):
            raise ValueError("coordinate bounds must be real")
        lower, upper = (np.asarray(x, dtype=np.float64) for x in (self.lower, self.upper))
        if lower.shape != upper.shape or not lower.ndim or not lower.size:
            raise ValueError("coordinate bounds require matching nonempty shapes")
        if not np.isfinite(lower).all() or not np.isfinite(upper).all():
            raise ValueError("coordinate bounds must be finite")
        low, high = _keys(lower), _keys(upper)
        if np.any(low > high):
            raise ValueError("lower coordinate bounds must not exceed upper bounds")
        object.__setattr__(self, "shape", lower.shape)
        def immutable(array):
            return tuple(immutable(x) for x in array) if array.ndim else float(array)
        object.__setattr__(self, "lower", immutable(lower))
        object.__setattr__(self, "upper", immutable(upper))
        object.__setattr__(self, "_lower_keys", tuple(int(x) for x in low.flat))
        object.__setattr__(self, "_upper_keys", tuple(int(x) for x in high.flat))

    def contains(self, q):
        """Pure scalar predicate; batching an entire guarded step is unsupported."""
        if q.shape != self.shape or q.dtype != jnp.dtype(jnp.float64):
            raise ValueError("coordinate guard requires its exact shape and float64 inputs")
        raw = jax.lax.bitcast_convert_type(q, jnp.uint64)
        finite = (raw & jnp.uint64(_EXPONENT)) != jnp.uint64(_EXPONENT)
        raw = jnp.where((raw & jnp.uint64(_MAGNITUDE)) == 0, jnp.uint64(0), raw)
        keys = jnp.where((raw & jnp.uint64(_SIGN)) != 0, ~raw, raw ^ jnp.uint64(_SIGN))
        low = jnp.asarray(self._lower_keys, dtype=jnp.uint64).reshape(self.shape)
        high = jnp.asarray(self._upper_keys, dtype=jnp.uint64).reshape(self.shape)
        return jnp.all(finite & (keys >= low) & (keys <= high))

    def require(self, q):
        """Host entry check before any model or measurement preflight."""
        q = np.asarray(q)
        if q.shape != self.shape or q.dtype != np.dtype(np.float64):
            raise ValueError("coordinate guard requires one scalar trajectory with float64 coordinates")
        low = np.asarray(self._lower_keys, np.uint64).reshape(self.shape)
        high = np.asarray(self._upper_keys, np.uint64).reshape(self.shape)
        if not np.isfinite(q).all() or np.any((_keys(q) < low) | (_keys(q) > high)):
            raise ValueError("coordinates are outside the declared CoordinateBox domain")
