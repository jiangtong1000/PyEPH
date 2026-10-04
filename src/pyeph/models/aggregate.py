"""Smooth nonlinear single-carrier aggregate on a fixed site graph.

This is a geometry-dependent infrastructure fixture, not a fitted molecular
transfer-integral parametrization. Every unordered edge is retained at every
step, including edges whose smooth envelope currently vanishes.
"""

from dataclasses import dataclass, field

import jax.numpy as jnp
import numpy as np

from pyeph.core._configuration import boolean_scalar, integer_scalar, real_scalar
from pyeph.core.contracts import ModelSpec
from pyeph.core.system import SystemSpec
from pyeph.models.base import AutoDiffModel


def smooth_switch(distance, switch_on, cutoff):
    """C2 switching envelope, exactly zero with zero slope at the cutoff."""
    u = jnp.clip((distance - switch_on) / (cutoff - switch_on), 0.0, 1.0)
    return 1.0 - 10.0 * u**3 + 15.0 * u**4 - 6.0 * u**5


def edge_action(onsite, hopping, edges, vectors):
    scalar = vectors.ndim == 1
    v = vectors[:, None] if scalar else vectors
    out = (onsite[:, None] * v).astype(jnp.result_type(onsite, hopping, v))
    if edges:
        e = jnp.asarray(edges)
        i, j = e[:, 0], e[:, 1]
        out = out.at[i].add(hopping[:, None] * v[j])
        out = out.at[j].add(hopping.conj()[:, None] * v[i])
    return out[:, 0] if scalar else out


@dataclass(frozen=True)
class AggregateModel(AutoDiffModel):
    """One fixed orthonormal effective state per site with Cartesian centers.

    All units are atomic. Hopping current uses diagonal site positions and
    hbar=1. ``lab_current_*`` additionally needs ProbeContext.velocity and
    includes motion of the carrier's site centers. It is not an ionic current.
    """

    nsites: int
    edges: tuple
    cutoff: float = 8.0
    switch_on: float = 6.0
    charge: float = -1.0
    complex_valued: bool = False
    spec: ModelSpec = field(init=False)

    def __post_init__(self):
        object.__setattr__(self, "nsites", integer_scalar(self.nsites, "nsites"))
        object.__setattr__(self, "complex_valued", boolean_scalar(self.complex_valued, "complex_valued"))
        for name in ("cutoff", "switch_on", "charge"):
            object.__setattr__(self, name, real_scalar(getattr(self, name), name))
        raw_edges = tuple(tuple(e) for e in self.edges)
        if any(x != int(x) for e in raw_edges for x in e):
            raise ValueError("edge indices must be exact integers")
        edges = tuple(tuple(int(x) for x in e) for e in raw_edges)
        if self.nsites < 1:
            raise ValueError("nsites must be positive")
        if not 0 <= self.switch_on < self.cutoff:
            raise ValueError("require 0 <= switch_on < cutoff")
        if any(len(e) != 2 or not 0 <= e[0] < e[1] < self.nsites for e in edges):
            raise ValueError("edges must have canonical indices 0 <= i < j < nsites")
        if len(set(edges)) != len(edges):
            raise ValueError("duplicate edges would double-count couplings")
        object.__setattr__(self, "edges", edges)
        object.__setattr__(self, "spec", ModelSpec(
            system=SystemSpec(nstates=self.nsites, q_shape=(self.nsites, 3)),
            name="nonlinear_aggregate", complex_valued=self.complex_valued,
            probes=tuple(f"{p}_{a}" for p in ("position", "current", "lab_current")
                         for a in "xyz")))

    def default_params(self):
        dtype = jnp.complex128 if self.complex_valued else jnp.float64
        positions = jnp.zeros((self.nsites, 3)).at[:, 0].set(jnp.arange(self.nsites) * 2.0)
        return dict(onsite=jnp.zeros(self.nsites), hopping=jnp.full(len(self.edges), 0.02, dtype),
                    decay=jnp.ones(len(self.edges)), reference_distance=jnp.full(len(self.edges), 2.0),
                    environment=jnp.zeros((len(self.edges), 2)),
                    reference_positions=positions, spring=jnp.zeros(self.nsites),
                    reference_offset=0.0)

    def displacements(self, q):
        if not self.edges:
            return jnp.zeros((0, 3), dtype=q.dtype)
        e = jnp.asarray(self.edges)
        return q[e[:, 1]] - q[e[:, 0]]

    def elements(self, params, q):
        p = self.default_params() if params is None else params
        r = jnp.linalg.norm(self.displacements(q), axis=-1)
        radial = jnp.exp(-p["decay"] * (r - p["reference_distance"]))
        radial = radial * smooth_switch(r, self.switch_on, self.cutoff)
        onsite = jnp.asarray(p["onsite"])
        if self.edges:
            e = jnp.asarray(self.edges)
            onsite = onsite.at[e[:, 0]].add(p["environment"][:, 0] * radial)
            onsite = onsite.at[e[:, 1]].add(p["environment"][:, 1] * radial)
        return onsite, p["hopping"] * radial

    def apply(self, params, q, vectors):
        onsite, hopping = self.elements(params, q)
        return edge_action(onsite, hopping, self.edges, vectors)

    def prepare_action(self, params, q):
        """Evaluate the smooth geometry dependence once, retaining sparse edges."""
        onsite, hopping = self.elements(params, q)
        return lambda vectors: edge_action(onsite, hopping, self.edges, vectors)

    def reference_energy(self, params, q):
        p = self.default_params() if params is None else params
        return (0.5 * jnp.sum(p["spring"][:, None] * (q - p["reference_positions"]) ** 2)
                + p.get("reference_offset", 0.0))

    def probe_apply(self, params, context, probe, vectors):
        if probe not in self.spec.probes:
            return super().probe_apply(params, context, probe, vectors)
        axis = "xyz".index(probe[-1])
        if probe.startswith("position_"):
            d = context.q[:, axis]
            return d * vectors if vectors.ndim == 1 else d[:, None] * vectors
        _, hopping = self.elements(params, context.q)
        j = 1j * self.charge * self.displacements(context.q)[:, axis] * hopping
        diagonal = jnp.zeros(self.nsites)
        if probe.startswith("lab_current_"):
            if context.velocity is None:
                raise ValueError("laboratory carrier current requires coordinate velocities")
            diagonal = self.charge * context.velocity[:, axis]
        return edge_action(diagonal, j, self.edges, vectors)

    def validate_geometry(self, q):
        q = np.asarray(q)
        if q.shape != (self.nsites, 3) or not np.all(np.isfinite(q)):
            raise ValueError("aggregate coordinates must be finite with shape (nsites, 3)")
        if self.edges and np.any(np.linalg.norm(
                q[np.asarray(self.edges)[:, 1]] - q[np.asarray(self.edges)[:, 0]], axis=-1) == 0):
            raise ValueError("coupled site centers must not coincide")

    def validate_params(self, params):
        p = self.default_params() if params is None else params
        shapes = dict(onsite=(self.nsites,), hopping=(len(self.edges),),
                      decay=(len(self.edges),), reference_distance=(len(self.edges),),
                      environment=(len(self.edges), 2),
                      reference_positions=(self.nsites, 3), spring=(self.nsites,))
        for key, shape in shapes.items():
            a = np.asarray(p[key])
            if a.shape != shape or not np.all(np.isfinite(a)):
                raise ValueError(f"{key} must be finite with shape {shape}")
            if np.iscomplexobj(a) and (not self.complex_valued or key != "hopping"):
                raise ValueError(f"{key} must be real under the declared model")
        if any(np.any(np.asarray(p[key]) < 0) for key in ("decay", "reference_distance", "spring")):
            raise ValueError("decay, reference_distance, and spring must be nonnegative")
