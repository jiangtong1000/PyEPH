"""Host-only scalar snapshots for named immutable configuration fields.

These helpers never touch dynamic parameter trees or set JAX precision. A NumPy
zero-dimensional array is mutable; retaining it in a frozen dataclass can make
the declared configuration disagree with constants captured by compiled code.
"""

import math

import numpy as np


def real_scalar(value, name, *, allow_infinite=False):
    array = np.asarray(value)
    if array.ndim or array.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be one real scalar")
    result = float(array)
    if math.isnan(result) or (not allow_infinite and not math.isfinite(result)):
        raise ValueError(f"{name} must be finite" if not allow_infinite else f"{name} cannot be NaN")
    return result


def integer_scalar(value, name):
    array = np.asarray(value)
    if array.ndim or array.dtype.kind not in "iu":
        raise ValueError(f"{name} must be one exact integer scalar")
    return int(array)


def boolean_scalar(value, name):
    array = np.asarray(value)
    if array.ndim or array.dtype.kind != "b":
        raise ValueError(f"{name} must be one boolean scalar")
    return bool(array)
