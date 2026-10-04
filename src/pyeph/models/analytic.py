"""Small analytic Hamiltonians in atomic units and canonical coordinates.

The Tully models are the three scattering models of J. C. Tully,
J. Chem. Phys. 93, 1061 (1990), doi:10.1063/1.459170.
Parameters, coordinates and nuclear masses are separate: these classes do not
silently select the mass used in a dynamics calculation.
"""

from dataclasses import dataclass, field

import jax.numpy as jnp

from pyeph.core._configuration import integer_scalar
from pyeph.core.contracts import ModelSpec
from pyeph.core.system import SystemSpec
from pyeph.models.base import AutoDiffModel


@dataclass(frozen=True)
class TullyModel(AutoDiffModel):
    """Tully's simple, dual, or extended coupling model (``kind=1,2,3``)."""

    kind: int = 1
    spec: ModelSpec = field(init=False)

    def __post_init__(self):
        object.__setattr__(self, "kind", integer_scalar(self.kind, "kind"))
        if self.kind not in (1, 2, 3):
            raise ValueError("Tully kind must be 1, 2, or 3")
        object.__setattr__(self, "spec", ModelSpec(
            system=SystemSpec(nstates=2, q_shape=(1,), coordinate_kind="canonical"),
            name=f"tully_{self.kind}"))

    def default_params(self):
        if self.kind == 1:
            return dict(a=0.01, b=1.6, c=0.005, d=1.0)
        if self.kind == 2:
            return dict(a=0.10, b=0.28, c=0.015, d=0.06, e0=0.05)
        return dict(a=0.0006, b=0.1, c=0.9)

    def dense(self, params, q):
        p = self.default_params() if params is None else params
        x = jnp.asarray(q)[0]
        if self.kind == 1:
            # This sign convention has the correct slope at x=0, unlike
            # sign(x), which is zero there. The exponential never overflows
            # even when JAX batches both sides of the piecewise expression.
            z = (p["a"] * jnp.where(x >= 0, 1., -1.)
                 * -jnp.expm1(-p["b"] * jnp.abs(x)))
            v = p["c"] * jnp.exp(-p["d"] * x * x)
            return jnp.stack((jnp.stack((z, v)), jnp.stack((v, -z))))
        if self.kind == 2:
            z = -p["a"] * jnp.exp(-p["b"] * x * x) + p["e0"]
            v = p["c"] * jnp.exp(-p["d"] * x * x)
            return jnp.stack((jnp.stack((jnp.zeros_like(x), v)),
                              jnp.stack((v, z))))
        tail = jnp.exp(-p["c"] * jnp.abs(x))
        v = p["b"] * jnp.where(x < 0, tail, 2.0 - tail)
        z = jnp.asarray(p["a"], dtype=x.dtype)
        return jnp.stack((jnp.stack((z, v)), jnp.stack((v, -z))))

    def apply(self, params, q, vectors):
        return self.dense(params, q) @ vectors

    def reference_energy(self, params, q):
        return jnp.zeros((), dtype=jnp.asarray(q).dtype)


@dataclass(frozen=True)
class SpinBosonModel(AutoDiffModel):
    """Mass-weighted classical spin-boson model.

    V_ref = sum(omega**2 * (Q-Q_eq)**2)/2 and
    h = (bias + coupling dot Q) sigma_z + delta sigma_x.
    This is the explicit classical-bath model, not an implicit quantum bath.
    """

    nmodes: int = 1
    spec: ModelSpec = field(init=False)

    def __post_init__(self):
        object.__setattr__(self, "nmodes", integer_scalar(self.nmodes, "nmodes"))
        if self.nmodes < 1:
            raise ValueError("nmodes must be positive")
        object.__setattr__(self, "spec", ModelSpec(
            system=SystemSpec(nstates=2, q_shape=(self.nmodes,),
                              coordinate_kind="normal_mode"),
            name="spin_boson"))

    def default_params(self):
        return dict(omega=jnp.ones(self.nmodes), coupling=jnp.ones(self.nmodes),
                    q_eq=jnp.zeros(self.nmodes), bias=0.0, delta=0.1,
                    reference_offset=0.0)

    def dense(self, params, q):
        p = self.default_params() if params is None else params
        z = p["bias"] + jnp.dot(p["coupling"], q)
        d = jnp.asarray(p["delta"], dtype=z.dtype)
        return jnp.stack((jnp.stack((z, d)), jnp.stack((d, -z))))

    def apply(self, params, q, vectors):
        return self.dense(params, q) @ vectors

    def reference_energy(self, params, q):
        p = self.default_params() if params is None else params
        return (0.5 * jnp.sum((p["omega"] * (q - p["q_eq"])) ** 2)
                + p.get("reference_offset", 0.0))
