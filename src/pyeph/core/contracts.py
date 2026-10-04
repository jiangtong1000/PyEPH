"""Structural model protocols and small, explicit operation data types."""

from dataclasses import dataclass, field
from typing import Any, NamedTuple, Protocol, runtime_checkable

import jax.numpy as jnp

from pyeph.core.system import SystemSpec
from pyeph.core.units import UnitSystem
from pyeph.core._configuration import boolean_scalar


@dataclass(frozen=True)
class ModelSpec:
    system: SystemSpec
    name: str = "model"
    force_support: bool = True
    native_jax: bool = True
    complex_valued: bool = False
    probes: tuple[str, ...] = ()
    energy_convention: str = "reference_plus_carrier"
    basis_kind: str = "fixed_orthonormal"
    electronic_sector: str = "single_carrier"
    unit_system: UnitSystem = field(default_factory=UnitSystem)

    def __post_init__(self):
        for name in ("force_support", "native_jax", "complex_valued"):
            object.__setattr__(self, name, boolean_scalar(getattr(self, name), name))
        if isinstance(self.probes, str):
            raise ValueError("probes must be a collection of names, not one string")
        probes = tuple(self.probes)
        if any(not isinstance(p, str) or not p for p in probes):
            raise ValueError("probes must contain nonempty strings")
        object.__setattr__(self, "probes", probes)


class LowRankWeight(NamedTuple):
    """W = left @ right.conj().T, each factor shaped (nstates, rank)."""

    left: Any
    right: Any


def pure_state_weight(state) -> LowRankWeight:
    state = jnp.asarray(state)
    if state.ndim != 1:
        raise ValueError("pure_state_weight expects one electronic state vector")
    return LowRankWeight(state[:, None], state[:, None])


class ProbeContext(NamedTuple):
    """Numerical context; its basis identity is declared by the model spec.

    velocity is dq/dt in the same coordinate convention as q. A probe requiring
    it must reject None rather than silently substituting zero motion.
    """

    q: Any
    velocity: Any = None
    time: Any = None


class SurfaceData(NamedTuple):
    energies: Any
    vectors: Any
    gaps: Any
    near_degenerate: Any


@runtime_checkable
class GeometryModel(Protocol):
    spec: ModelSpec

    def reference_energy(self, params, q): ...

    def apply(self, params, q, vectors): ...


@runtime_checkable
class ForceModel(GeometryModel, Protocol):
    def reference_gradient(self, params, q): ...

    def contract_gradient(self, params, q, weight): ...


@runtime_checkable
class NuclearPath(Protocol):
    def position(self, time): ...

    def velocity(self, time): ...


def prepared_action(model, params, q):
    """Create a pure operator action for one geometry, without persistent caches.

    An optional ``model.prepare_action(params, q)`` can close over dense,
    sparse or block coefficients computed once for this action. Its callable
    must equal ``model.apply(params, q, vectors)`` for vectors and column blocks.
    Create and consume it inside the numerical function being JIT-compiled or
    differentiated; the callable itself is not a JAX output or checkpoint value.
    Models without this hook retain their regular apply operation. A subclass
    or instance overriding only apply also retains that operation: an inherited
    preparation must not silently bypass its changed Hamiltonian.
    """
    prepare = getattr(model, "prepare_action", None)
    if prepare is None:
        return lambda vectors: model.apply(params, q, vectors)
    if not callable(prepare):
        raise TypeError("model.prepare_action must be callable or None")

    def owner(name):
        if name in getattr(model, "__dict__", {}):
            return -1
        for index, cls in enumerate(type(model).__mro__):
            if name in cls.__dict__:
                return index
        return None

    apply_owner, prepare_owner = owner("apply"), owner("prepare_action")
    if apply_owner is None or prepare_owner is None or apply_owner < prepare_owner:
        return lambda vectors: model.apply(params, q, vectors)
    action = prepare(params, q)
    if not callable(action):
        raise TypeError("model.prepare_action(params, q) must return a callable")
    return action


def contracted_value(apply, weight):
    """Re Tr(W† H), without dense H for a factored weight.

    `apply` is a closure applying H to a vector or a block of column vectors.
    This also permits complex off-diagonal derivatives by choosing W or i W.
    """
    if isinstance(weight, LowRankWeight):
        if weight.left.ndim != 2 or weight.right.shape != weight.left.shape:
            raise ValueError("weight factors must have equal (nstates, rank) shapes")
        return jnp.real(jnp.vdot(weight.left, apply(weight.right)))
    weight = jnp.asarray(weight)
    if weight.ndim != 2 or weight.shape[0] != weight.shape[1]:
        raise ValueError("dense weight must be square; use pure_state_weight for a vector")
    return jnp.real(jnp.vdot(weight, apply(jnp.eye(weight.shape[0], dtype=weight.dtype))))
