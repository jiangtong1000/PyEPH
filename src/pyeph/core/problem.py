"""Validate physical compatibility once, before compiling a trajectory loop."""

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np


@dataclass(frozen=True)
class CoupledClassical:
    masses: Any
    prescribed = False

    def __post_init__(self):
        import jax.numpy as jnp

        # The nuclear treatment is captured as static configuration by JIT.
        # Retaining a caller's mutable array would let provenance see a new
        # mass while an already compiled propagator still used the old one.
        object.__setattr__(self, "masses", jnp.array(self.masses, copy=True))

    def validate(self, q_shape):
        m = np.asarray(self.masses)
        try:
            np.broadcast_to(m, q_shape)
        except ValueError as exc:
            raise ValueError("masses must broadcast to the coordinate shape; use (atoms,1)") from exc
        if np.iscomplexobj(m) or not np.isfinite(m).all() or np.any(m <= 0):
            raise ValueError("all canonical masses must be finite and positive")


@dataclass(frozen=True)
class PrescribedPath:
    path: Any
    prescribed = True

    def validate(self, q_shape):
        if not callable(getattr(self.path, "position", None)):
            raise TypeError("a nuclear path must provide position(time)")
        if not callable(getattr(self.path, "velocity", None)):
            raise TypeError("a nuclear path must provide velocity(time)")
        representative_time = getattr(self.path, "start_time", 0.0)
        if np.shape(self.path.position(representative_time)) != q_shape:
            raise ValueError("path and model coordinate shapes differ")

    def point(self, state, elapsed):
        import jax.numpy as jnp

        return self.path.position(state.time + elapsed), jnp.zeros_like(state.p)


class DynamicsMethod(Protocol):
    """A new method validates its scope and builds a pure single-trajectory step."""

    def validate(self, problem): ...

    def build_step(self, problem, integrator): ...


@dataclass(frozen=True)
class Problem:
    model: Any
    params: Any
    nuclear_treatment: Any
    method: DynamicsMethod
    measurement: Any = None

    def validate(self):
        spec = self.model.spec
        if spec.basis_kind != "fixed_orthonormal":
            raise ValueError("native geometry dynamics requires a fixed orthonormal effective basis")
        if spec.energy_convention != "reference_plus_carrier":
            raise ValueError("the model must declare a consistent reference-plus-carrier total energy")
        if callable(getattr(self.model, "validate_params", None)):
            self.model.validate_params(self.params)
        self.nuclear_treatment.validate(spec.system.q_shape)
        self.method.validate(self)
        if self.measurement is not None:
            self.measurement.validate(self)
        return self
