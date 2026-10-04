"""Prescribed electronic frames, with declared interpolation and basis transport.

These datasets do not provide nuclear feedback forces. In particular, temporal
overlaps cannot supply the spatial direction needed for momentum rescaling.
"""

from dataclasses import dataclass, field
from typing import Any, NamedTuple

import jax.numpy as jnp
import numpy as np

from pyeph.paths.nuclear import RecordedTimeDomain, _validated_times
from pyeph.representations.connection import transport_from_overlap
from pyeph.core.units import UnitSystem


class FixedBasisFrame(NamedTuple):
    hamiltonian: Any
    time: Any
    valid: Any


class AdiabaticFrame(NamedTuple):
    energies: Any
    time: Any
    valid: Any


def _requested(request, allowed):
    if request not in (None, allowed):
        raise ValueError(f"this electronic path provides {allowed}, not {request!r}")


@dataclass(frozen=True, eq=False)
class FixedBasisElectronicPath(RecordedTimeDomain):
    """Hermitian H(t) in one fixed basis; linear interpolation is an explicit model.

    The interpolated matrix is continuous but its time derivative can have
    kinks. ``frames_only`` instead rejects intermediate sampling via the frame
    validity flag. Complex matrix elements are preserved.
    """

    times: object
    hamiltonians: object
    basis_id: str = "fixed"
    interpolation: str = "linear"
    unit_system: UnitSystem = field(default_factory=UnitSystem)
    force_support = False
    basis_kind = "fixed_orthonormal"

    def __post_init__(self):
        times = _validated_times(self.times)
        h = np.asarray(self.hamiltonians)
        if h.ndim != 3 or h.shape[0] != len(times) or h.shape[1] != h.shape[2] or not h.shape[1]:
            raise ValueError("Hamiltonians must have shape (frames, states, states)")
        if not np.isfinite(h).all() or not np.allclose(h, h.conj().swapaxes(-1, -2),
                                                     rtol=1e-10, atol=1e-12):
            raise ValueError("recorded Hamiltonians must be finite and Hermitian")
        if not self.basis_id or self.interpolation not in {"linear", "frames_only"}:
            raise ValueError("declare a basis_id and 'linear' or 'frames_only' interpolation")
        object.__setattr__(self, "times", times)
        object.__setattr__(self, "hamiltonians", jnp.array(h, dtype=jnp.result_type(h, 1.0), copy=True))

    @property
    def nstates(self):
        return self.hamiltonians.shape[1]

    def validate_sample_time(self, time):
        self.validate_time(time)
        if self.interpolation == "frames_only" and not np.isin(time, np.asarray(self.times)).all():
            raise ValueError("this dataset only defines recorded frame times")

    def sample_frame(self, time, request=None):
        _requested(request, "hamiltonian")
        i, _, s, valid = self._interval(time)
        h = (1 - s)*self.hamiltonians[i] + s*self.hamiltonians[i + 1]
        if self.interpolation == "frames_only":
            valid = valid & jnp.any(self.times == time)
        return FixedBasisFrame(jnp.where(valid, h, jnp.nan), jnp.asarray(time), valid)

    def apply(self, time, vectors):
        return self.sample_frame(time).hamiltonian @ vectors


@dataclass(frozen=True, eq=False)
class AdiabaticElectronicPath(RecordedTimeDomain):
    """Instantaneous energies and optional adjacent-frame temporal overlaps.

    ``overlaps[k,a,b] = <phi_a(t[k])|phi_b(t[k+1])>``. Energies are only
    defined at recorded frames: this class does not invent an intermediate
    adiabatic basis. ``interval_transport`` requires exact adjacent forward
    endpoints. ``transport_at`` is its JIT-compatible index-based counterpart.
    """

    times: object
    energies: object
    overlaps: object = None
    basis_id: str = "instantaneous_adiabatic"
    transport_mode: str = "raw"
    unit_system: UnitSystem = field(default_factory=UnitSystem)
    interpolation = "frames_only"
    force_support = False
    basis_kind = "instantaneous_orthonormal"

    def __post_init__(self):
        times = _validated_times(self.times)
        e = np.asarray(self.energies)
        if e.ndim != 2 or e.shape[0] != len(times) or not e.shape[1]:
            raise ValueError("energies must have shape (frames, states)")
        if np.iscomplexobj(e) or not np.isfinite(e).all():
            raise ValueError("adiabatic energies must be real and finite")
        if not self.basis_id or self.transport_mode not in {"raw", "polar"}:
            raise ValueError("declare a basis_id and 'raw' or 'polar' transport_mode")
        object.__setattr__(self, "times", times)
        object.__setattr__(self, "energies", jnp.array(e, dtype=times.dtype, copy=True))
        if self.overlaps is not None:
            o = np.asarray(self.overlaps)
            if o.shape != (len(times) - 1, e.shape[1], e.shape[1]):
                raise ValueError("overlaps must have shape (intervals, states, states)")
            if not np.isfinite(o).all() or np.any(np.linalg.svd(o, compute_uv=False) > 1 + 1e-10):
                raise ValueError("orthonormal-basis overlaps must be finite contractions")
            object.__setattr__(self, "overlaps", jnp.array(o, dtype=jnp.result_type(o, 1.0), copy=True))

    @property
    def nstates(self):
        return self.energies.shape[1]

    def validate_sample_time(self, time):
        self.validate_time(time)
        if not np.isin(time, np.asarray(self.times)).all():
            raise ValueError("an intermediate adiabatic basis is not defined; use recorded times")

    def sample_frame(self, time, request=None):
        _requested(request, "energies")
        self._interval(time)  # Validate scalar shape for the compiled interface.
        index = jnp.clip(jnp.searchsorted(self.times, time), 0, len(self.times) - 1)
        valid = self.contains(time) & (self.times[index] == time)
        return AdiabaticFrame(jnp.where(valid, self.energies[index], jnp.nan),
                              jnp.asarray(time), valid)

    def interval_transport(self, time0, time1):
        self.validate_sample_time([time0, time1])
        indices = np.searchsorted(np.asarray(self.times), [time0, time1])
        if indices[1] != indices[0] + 1:
            raise ValueError("transport requires adjacent forward recorded endpoints")
        return self.transport_at(int(indices[0]))

    def transport_at(self, index):
        if self.overlaps is None:
            raise ValueError("energies alone do not define transport between instantaneous bases")
        i = jnp.asarray(index)
        if i.ndim or not jnp.issubdtype(i.dtype, jnp.integer):
            raise ValueError("an interval index must be a scalar integer")
        valid_index = (i >= 0) & (i < len(self.times) - 1)
        result = transport_from_overlap(self.overlaps[i], mode=self.transport_mode)
        return result._replace(matrix=jnp.where(valid_index, result.matrix, jnp.nan),
                               valid=result.valid & valid_index)
