"""Fixed orthonormal orbital blocks with explicit periodic image edges.

Cell vectors are rows: d_abn = q_b + n @ cell - q_a. Each Hermitian pair
appears exactly once. Bloch sums use the lattice gauge exp(i k dot n@cell),
while a physical uniform Peierls probe uses the full displacement d_abn.
The default action is the Gamma-point real-space simulation supercell.
"""

from dataclasses import dataclass, field

import jax.numpy as jnp
import numpy as np

from pyeph.core._configuration import boolean_scalar, integer_scalar, real_scalar
from pyeph.core.contracts import ModelSpec
from pyeph.core.system import SystemSpec
from pyeph.models.aggregate import smooth_switch
from pyeph.models._block import block_action
from pyeph.models.base import AutoDiffModel


@dataclass(frozen=True)
class PeriodicBlockModel(AutoDiffModel):
    """Geometry-dependent Hermitian orbital blocks, including complex ones.

    An edge is (a,b,nx,ny,nz), with a<b, or a==b and the first nonzero image
    component positive. Onsite blocks are separate. The fixed graph is never
    pruned. This is an effective carrier potential, not a raw KS/AO adapter.
    """

    nsites: int
    norbitals: int
    edges: tuple
    cell: tuple
    cutoff: float = 8.0
    switch_on: float = 6.0
    charge: float = -1.0
    complex_valued: bool = True
    spec: ModelSpec = field(init=False)

    def __post_init__(self):
        for name in ("nsites", "norbitals"):
            object.__setattr__(self, name, integer_scalar(getattr(self, name), name))
        object.__setattr__(self, "complex_valued", boolean_scalar(self.complex_valued, "complex_valued"))
        for name in ("cutoff", "switch_on", "charge"):
            object.__setattr__(self, name, real_scalar(getattr(self, name), name))
        raw_edges = tuple(tuple(e) for e in self.edges)
        if any(x != int(x) for e in raw_edges for x in e):
            raise ValueError("orbital indices and cell images must be exact integers")
        edges = tuple(tuple(int(x) for x in e) for e in raw_edges)
        cell = np.asarray(self.cell, dtype=float)
        if self.nsites < 1 or self.norbitals < 1:
            raise ValueError("nsites and norbitals must be positive")
        if cell.shape != (3, 3) or not np.all(np.isfinite(cell)) or abs(np.linalg.det(cell)) < 1e-12:
            raise ValueError("cell must be a finite nonsingular row-vector matrix")
        if not 0 <= self.switch_on < self.cutoff:
            raise ValueError("require 0 <= switch_on < cutoff")
        for e in edges:
            if len(e) != 5 or not 0 <= e[0] <= e[1] < self.nsites:
                raise ValueError("edge must be (a,b,nx,ny,nz), 0 <= a <= b < nsites")
            if e[0] == e[1] and tuple(e[2:]) <= (0, 0, 0):
                raise ValueError("self-image edges require a lexicographically positive nonzero image")
        if len(set(edges)) != len(edges):
            raise ValueError("duplicate periodic edges would double-count blocks")
        object.__setattr__(self, "edges", edges)
        object.__setattr__(self, "cell", tuple(tuple(x for x in row) for row in cell))
        object.__setattr__(self, "spec", ModelSpec(
            system=SystemSpec(nstates=self.nsites * self.norbitals, q_shape=(self.nsites, 3)),
            name="periodic_blocks", complex_valued=self.complex_valued,
            probes=tuple(f"current_{a}" for a in "xyz")))

    def default_params(self):
        b, e = self.norbitals, len(self.edges)
        dtype = jnp.complex128 if self.complex_valued else jnp.float64
        return dict(onsite=jnp.zeros((self.nsites, b, b), dtype),
                    onsite_derivative=jnp.zeros((self.nsites, 3, b, b), dtype),
                    hopping=jnp.zeros((e, b, b), dtype), decay=jnp.zeros(e),
                    reference_distance=jnp.ones(e),
                    reference_positions=jnp.zeros((self.nsites, 3)),
                    spring=jnp.zeros(self.nsites), reference_offset=0.0)

    def displacements(self, q):
        if not self.edges:
            return jnp.zeros((0, 3), dtype=q.dtype)
        e = jnp.asarray(self.edges)
        return q[e[:, 1]] + e[:, 2:] @ jnp.asarray(self.cell) - q[e[:, 0]]

    def blocks(self, params, q):
        p = self.default_params() if params is None else params
        onsite = p["onsite"] + jnp.einsum(
            "ic,icab->iab", q - p["reference_positions"], p["onsite_derivative"])
        # Construction is Hermitian for arbitrary real NN residual weights too.
        onsite = (onsite + onsite.conj().swapaxes(-1, -2)) * 0.5
        r = jnp.linalg.norm(self.displacements(q), axis=-1)
        radial = jnp.exp(-p["decay"] * (r - p["reference_distance"]))
        radial = radial * smooth_switch(r, self.switch_on, self.cutoff)
        return onsite, p["hopping"] * radial[:, None, None]

    def _action(self, onsite, hopping, vectors):
        return block_action(onsite, hopping, self.edges, vectors)

    def apply(self, params, q, vectors):
        return self._action(*self.blocks(params, q), vectors)

    def prepare_action(self, params, q):
        """Prepare onsite and image-edge blocks, preserving their block structure."""
        onsite, hopping = self.blocks(params, q)
        return lambda vectors: self._action(onsite, hopping, vectors)

    def apply_bloch(self, params, q, k, vectors):
        """Apply a translationally repeated cell's lattice-gauge H(k)."""
        onsite, hopping = self.blocks(params, q)
        if self.edges:
            image = jnp.asarray(self.edges)[:, 2:] @ jnp.asarray(self.cell)
            hopping = hopping * jnp.exp(1j * (image @ k))[:, None, None]
        return self._action(onsite, hopping, vectors)

    def dense_bloch(self, params, q, k):
        return self.apply_bloch(params, q, k, jnp.eye(self.nstates))

    def apply_peierls(self, params, q, wavevector, vectors):
        """H(kappa) with exp(i kappa dot d); current=q_charge*dH/dkappa."""
        onsite, hopping = self.blocks(params, q)
        hopping = hopping * jnp.exp(1j * (self.displacements(q) @ wavevector))[:, None, None]
        return self._action(onsite, hopping, vectors)

    def probe_apply(self, params, context, probe, vectors):
        if probe not in self.spec.probes:
            return super().probe_apply(params, context, probe, vectors)
        axis = "xyz".index(probe[-1])
        onsite, hopping = self.blocks(params, context.q)
        current = (1j * self.charge * self.displacements(context.q)[:, axis, None, None] * hopping)
        return self._action(jnp.zeros_like(onsite), current, vectors)

    def reference_energy(self, params, q):
        p = self.default_params() if params is None else params
        return (0.5 * jnp.sum(p["spring"][:, None] * (q - p["reference_positions"]) ** 2)
                + p.get("reference_offset", 0.0))

    def rewrapped(self, shifts):
        """New static graph for q'=q+shifts@cell (integer shifts per site).

        The caller also wraps reference_positions by the same shifts. Edge
        images change to n'=n+shift_a-shift_b, preserving each displacement.
        """
        shifts = np.asarray(shifts)
        if shifts.shape != (self.nsites, 3) or not np.all(shifts == np.round(shifts)):
            raise ValueError("shifts must be integer translations with shape (nsites,3)")
        edges = tuple((e[0], e[1], *(np.asarray(e[2:]) + shifts[e[0]] - shifts[e[1]]).astype(int))
                      for e in self.edges)
        return PeriodicBlockModel(self.nsites, self.norbitals, edges, self.cell,
                                  self.cutoff, self.switch_on, self.charge, self.complex_valued)

    def validate_geometry(self, q):
        q = np.asarray(q)
        if q.shape != (self.nsites, 3) or not np.all(np.isfinite(q)):
            raise ValueError("periodic coordinates must be finite with shape (nsites,3)")
        if self.edges:
            e = np.asarray(self.edges)
            d = q[e[:, 1]] + e[:, 2:] @ np.asarray(self.cell) - q[e[:, 0]]
            if np.any(np.linalg.norm(d, axis=-1) == 0):
                raise ValueError("coupled image centers must not coincide")

    def validate_params(self, params):
        p = self.default_params() if params is None else params
        b, e = self.norbitals, len(self.edges)
        shapes = dict(onsite=(self.nsites, b, b), onsite_derivative=(self.nsites, 3, b, b),
                      hopping=(e, b, b), decay=(e,), reference_distance=(e,),
                      reference_positions=(self.nsites, 3), spring=(self.nsites,))
        for key, shape in shapes.items():
            a = np.asarray(p[key])
            if a.shape != shape or not np.all(np.isfinite(a)):
                raise ValueError(f"{key} must be finite with shape {shape}")
            if np.iscomplexobj(a) and (not self.complex_valued or key not in {"onsite", "onsite_derivative", "hopping"}):
                raise ValueError(f"{key} must be real under the declared model")
            if key in {"onsite", "onsite_derivative"} and not np.allclose(a, a.conj().swapaxes(-1, -2), atol=1e-12):
                raise ValueError(f"{key} must be Hermitian")
        if any(np.any(np.asarray(p[key]) < 0) for key in ("decay", "reference_distance", "spring")):
            raise ValueError("decay, reference_distance, and spring must be nonnegative")
