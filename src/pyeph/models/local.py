"""Native local carrier blocks on an explicit finite or periodic graph.

This is one concrete fixed-basis representation, not an AO/overlap interface.
Providers return onsite and raw hopping coefficients in the declared basis;
the model applies the graph's final hopping support once. Providers themselves
own the smoothness, locality, symmetry, units, and physical meaning of their
features and coefficients. A scalar-feature network is not orbital equivariant.
"""

from dataclasses import dataclass, field, replace
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from pyeph.core._configuration import boolean_scalar, integer_scalar, real_scalar
from pyeph.core.contracts import ModelSpec
from pyeph.core.system import SystemSpec
from pyeph.models._block import block_action
from pyeph.models.aggregate import smooth_switch
from pyeph.models.base import AutoDiffModel


class LocalCoefficients(NamedTuple):
    """Onsite (N,b,b) and unique-edge (E,b,b) blocks.

    Providers return RAW hopping blocks; LocalBlockModel multiplies support
    exactly once, including after a provider sums baseline and residual blocks.
    Onsite must already be Hermitian. Offsite blocks need not be Hermitian.
    """

    onsite: Any
    hopping: Any


class LocalGeometry(NamedTuple):
    """Differentiable per-geometry data supplied to the coefficient provider.

    Providers receive all atomic coordinates separately, including atoms with
    zero center weight. ``support`` must also gate provider-side neighbor
    messages if their contribution to onsite terms should disappear at cutoff.
    No implicit neighbor search, pruning, internal descriptor, or feature
    normalization is performed.
    """

    centers: Any
    pairs: Any
    displacements: Any
    distances: Any
    support: Any
    atom_site: Any
    atom_weights: Any


@dataclass(frozen=True)
class LocalBlockGraph:
    """Static unique Hermitian pairs and uniformly sized site blocks.

    Finite edges may be (a,b); they normalize to (a,b,0,0,0). Periodic edges are
    (a,b,nx,ny,nz), with a<b or a==b and a lexicographically positive image.
    Cell vectors are rows: d = R_b + image @ cell - R_a. The action represents
    the finite graph or Gamma-point simulation supercell. Onsite zero-image
    self-edges are forbidden because onsite blocks are supplied separately.

    Optional cutoff/switch_on must be supplied together. Their quintic support
    does not make an arbitrary discontinuous/singular provider smooth, and the
    graph must already contain every required candidate edge.
    """

    nsites: int
    norbitals: int
    edges: tuple
    cell: tuple | None = None
    switch_on: float | None = field(default=None, kw_only=True)
    cutoff: float | None = field(default=None, kw_only=True)

    def __post_init__(self):
        for name in ("nsites", "norbitals"):
            value = integer_scalar(getattr(self, name), name)
            if not 1 <= value <= np.iinfo(np.int32).max:
                raise ValueError(f"{name} must be positive and fit in int32")
            object.__setattr__(self, name, value)
        try:
            edges = tuple(tuple(integer_scalar(x, "edge index/image") for x in e)
                          for e in self.edges)
        except TypeError as exc:
            raise ValueError("edges must be collections of integer indices/images") from exc
        normalized = []
        for edge in edges:
            if len(edge) == 2:
                edge = (*edge, 0, 0, 0)
            if len(edge) != 5 or not 0 <= edge[0] <= edge[1] < self.nsites:
                raise ValueError("edges must have canonical sites 0 <= a <= b < nsites")
            if edge[0] == edge[1] and edge[2:] <= (0, 0, 0):
                raise ValueError("self-image edges need a lexicographically positive image")
            if self.cell is None and edge[2:] != (0, 0, 0):
                raise ValueError("nonzero images require an explicit cell")
            if any(not np.iinfo(np.int32).min <= x <= np.iinfo(np.int32).max for x in edge):
                raise ValueError("edge indices/images must fit in int32")
            normalized.append(edge)
        if len(set(normalized)) != len(normalized):
            raise ValueError("duplicate edges would double-count coefficients")
        object.__setattr__(self, "edges", tuple(normalized))
        if self.cell is not None:
            cell = np.asarray(self.cell)
            if (cell.shape != (3, 3) or cell.dtype.kind not in "iuf"
                    or not np.all(np.isfinite(cell))):
                raise ValueError("cell must be a finite real 3x3 row-vector matrix")
            cell = np.asarray(cell, dtype=float)
            with np.errstate(over="ignore", invalid="ignore"):
                determinant = np.linalg.det(cell)
            if not np.isfinite(determinant) or abs(determinant) < 1e-12:
                raise ValueError("cell must be nonsingular with finite determinant")
            object.__setattr__(self, "cell", tuple(tuple(float(x) for x in row) for row in cell))
        if (self.switch_on is None) != (self.cutoff is None):
            raise ValueError("switch_on and cutoff must be supplied together")
        if self.cutoff is not None:
            switch_on = real_scalar(self.switch_on, "switch_on")
            cutoff = real_scalar(self.cutoff, "cutoff")
            if not 0 <= switch_on < cutoff:
                raise ValueError("require 0 <= switch_on < cutoff")
            object.__setattr__(self, "switch_on", switch_on)
            object.__setattr__(self, "cutoff", cutoff)

    @property
    def nstates(self):
        return self.nsites * self.norbitals

    def rewrapped(self, shifts):
        """Change images for R'=R+shifts@cell, preserving physical displacements."""
        if self.cell is None:
            raise ValueError("rewrapping requires a periodic cell")
        shifts = np.asarray(shifts)
        if shifts.shape != (self.nsites, 3) or shifts.dtype.kind not in "iu":
            raise ValueError("shifts must have integer shape (nsites,3)")
        # Python integers avoid wraparound while adding images to large shifts.
        rows = tuple(tuple(int(x) for x in row) for row in shifts)
        edges = tuple((a, b, *(image[k] + rows[a][k] - rows[b][k] for k in range(3)))
                      for a, b, *image in self.edges)
        return replace(self, edges=edges)


@dataclass(frozen=True)
class AtomCenterMap:
    """Fixed sparse atom-to-center assignment with nonnegative weights.

    Each atomic coordinate belongs to one center; weights sum to one per center.
    Zero-weight atoms remain present in q and available to the provider. This
    linear map neither infers masses nor performs molecular periodic unwrapping.
    """

    atom_site: tuple
    weights: tuple
    nsites: int

    def __post_init__(self):
        nsites = integer_scalar(self.nsites, "nsites")
        if nsites < 1:
            raise ValueError("nsites must be positive")
        sites = np.asarray(self.atom_site)
        weights = np.asarray(self.weights)
        if (sites.ndim != 1 or not sites.size or sites.dtype.kind not in "iu"
                or np.any(sites < 0) or np.any(sites >= nsites)):
            raise ValueError("atom_site must contain valid integer site assignments")
        if (weights.shape != sites.shape or weights.dtype.kind not in "iuf"
                or not np.all(np.isfinite(weights)) or np.any(weights < 0)):
            raise ValueError("weights must be finite nonnegative real values, one per atom")
        weights = np.asarray(weights, dtype=float)
        totals = np.bincount(sites.astype(int), weights=weights, minlength=nsites)
        if not np.allclose(totals, 1.0, atol=1e-14, rtol=0):
            raise ValueError("center weights must sum to one for every site")
        object.__setattr__(self, "nsites", nsites)
        object.__setattr__(self, "atom_site", tuple(int(x) for x in sites))
        object.__setattr__(self, "weights", tuple(float(x) for x in weights))

    @property
    def natoms(self):
        return len(self.atom_site)

    def apply(self, q):
        """Map one atomic Cartesian array to centers without a dense map matrix."""
        q = jnp.asarray(q)
        if q.shape != (self.natoms, 3) or q.dtype.kind != "f":
            raise ValueError("q must be real floating coordinates with shape (natoms,3)")
        sites = jnp.asarray(self.atom_site, dtype=jnp.int32)
        weights = jnp.asarray(self.weights, dtype=q.dtype)
        return jnp.zeros((self.nsites, 3), q.dtype).at[sites].add(weights[:, None] * q)


@dataclass(frozen=True)
class LocalBlockModel(AutoDiffModel):
    """Complete native carrier operator from a structural coefficient provider.

    ``coefficient_provider(params, q_atoms, geometry)`` returns LocalCoefficients
    in a fixed orthonormal basis. params remain dynamic; the provider and graph
    are static, pure configuration. A provider may combine baseline and neural
    raw coefficients before returning them. It must preserve basis conventions,
    own all geometry/parameter derivatives, and return already-Hermitian onsite
    blocks. This adapter never silently symmetrizes or discards imaginary parts.

    Vref=0. Compose a separately declared physical nuclear reference in the SAME
    atomic coordinate space when needed. Current probes are instantaneous
    electronic hopping currents. In a finite graph these equal the commutator
    with X=diag(center positions), repeated over block channels. Periodic graphs
    instead use full image-resolved Peierls displacements, and may have nonzero
    self-image current even when the finite cell-position commutator vanishes.
    They include every returned hopping coefficient, but no
    moving-center convective term, intra-center dipoles, ionic current, or AO
    connection. Complex blocks are supported algebraically; this does not
    establish a complex/SOC hopping method.

    validate_at checks representative actual outputs before compilation. Later
    finite/Hermitian/smooth behavior is a trusted-provider numerical contract;
    shape and declared dtype checks still apply during tracing. New provider
    implementations/artifacts require new model/Runner instances and explicit
    identities for strict provenance when their captured state is opaque.
    """

    graph: LocalBlockGraph
    centers: AtomCenterMap
    coefficient_provider: Any
    charge: float = field(default=-1.0, kw_only=True)
    complex_valued: bool = field(default=False, kw_only=True)
    basis_id: str = field(default="fixed", kw_only=True)
    spec: ModelSpec = field(init=False)

    def __post_init__(self):
        if not isinstance(self.graph, LocalBlockGraph) or not isinstance(self.centers, AtomCenterMap):
            raise TypeError("graph and centers must be LocalBlockGraph and AtomCenterMap")
        if self.graph.nsites != self.centers.nsites:
            raise ValueError("graph and center map disagree on nsites")
        if not callable(self.coefficient_provider):
            raise TypeError("coefficient_provider must be callable")
        if not isinstance(self.basis_id, str) or not self.basis_id:
            raise ValueError("basis_id must be a nonempty string")
        object.__setattr__(self, "charge", real_scalar(self.charge, "charge"))
        object.__setattr__(self, "complex_valued", boolean_scalar(self.complex_valued, "complex_valued"))
        object.__setattr__(self, "spec", ModelSpec(
            SystemSpec(self.graph.nstates, (self.centers.natoms, 3), self.basis_id),
            name="local_blocks", complex_valued=self.complex_valued,
            probes=tuple(f"current_{axis}" for axis in "xyz")))

    def geometry(self, q):
        q = jnp.asarray(q)
        centers = self.centers.apply(q)
        edges = jnp.asarray(self.graph.edges, dtype=jnp.int32).reshape(-1, 5)
        pairs = edges[:, :2]
        displacement = centers[pairs[:, 1]] - centers[pairs[:, 0]]
        if self.graph.cell is not None:
            displacement = displacement + edges[:, 2:] @ jnp.asarray(self.graph.cell, dtype=q.dtype)
        distances = jnp.linalg.norm(displacement, axis=-1)
        support = (jnp.ones_like(distances) if self.graph.cutoff is None else
                   smooth_switch(distances, self.graph.switch_on, self.graph.cutoff))
        return LocalGeometry(centers, pairs, displacement, distances, support,
                             jnp.asarray(self.centers.atom_site, dtype=jnp.int32),
                             jnp.asarray(self.centers.weights, dtype=q.dtype))

    def _coefficients(self, params, q, geometry):
        result = self.coefficient_provider(params, q, geometry)
        if not isinstance(result, LocalCoefficients):
            raise TypeError("coefficient_provider must return LocalCoefficients")
        arrays = tuple(jnp.asarray(x) for x in result)
        n, b, e = self.graph.nsites, self.graph.norbitals, len(self.graph.edges)
        for name, value, shape in zip(("onsite", "hopping"), arrays, ((n, b, b), (e, b, b))):
            if value.shape != shape:
                raise ValueError(f"{name} coefficients must have shape {shape}")
            if value.dtype.kind not in "iufc":
                raise ValueError(f"{name} coefficients must be numeric real or complex arrays")
            if value.dtype.kind == "c" and not self.complex_valued:
                raise ValueError("complex coefficients contradict complex_valued=False")
        return LocalCoefficients(arrays[0], arrays[1] * geometry.support[:, None, None])

    def coefficients(self, params, q):
        """Physical onsite/hopping blocks, with final support applied once."""
        q = jnp.asarray(q)
        return self._coefficients(params, q, self.geometry(q))

    def _action(self, coefficients, vectors):
        vectors = jnp.asarray(vectors)
        if vectors.ndim not in (1, 2) or vectors.shape[0] != self.nstates:
            raise ValueError("vectors must have shape (nstates,) or (nstates,ncolumns)")
        if vectors.dtype.kind not in "iufc":
            raise ValueError("vectors must be numeric real or complex arrays")
        return block_action(*coefficients, self.graph.edges, vectors)

    def apply(self, params, q, vectors):
        return self._action(self.coefficients(params, q), vectors)

    def prepare_action(self, params, q):
        coefficients = self.coefficients(params, q)
        return lambda vectors: self._action(coefficients, vectors)

    def reference_energy(self, params, q):
        return jnp.zeros((), dtype=jnp.asarray(q).dtype)

    def probe_apply(self, params, context, probe, vectors):
        if probe not in self.spec.probes:
            return super().probe_apply(params, context, probe, vectors)
        q = jnp.asarray(context.q)
        geometry = self.geometry(q)
        onsite, hopping = self._coefficients(params, q, geometry)
        axis = "xyz".index(probe[-1])
        current = 1j * self.charge * geometry.displacements[:, axis, None, None] * hopping
        return self._action(LocalCoefficients(jnp.zeros_like(onsite), current), vectors)

    def apply_peierls(self, params, q, wavevector, vectors):
        """Full-displacement uniform phase action; this is not a Bloch-sum API.

        H(kappa) has exp(i*kappa·d) on each forward edge. The current is charge
        times its derivative at kappa=0, with hbar=1.
        """
        q, wavevector = jnp.asarray(q), jnp.asarray(wavevector)
        if wavevector.shape != (3,) or wavevector.dtype.kind not in "iuf":
            raise ValueError("wavevector must have real shape (3,)")
        geometry = self.geometry(q)
        onsite, hopping = self._coefficients(params, q, geometry)
        phase = jnp.exp(1j * (geometry.displacements @ wavevector))
        return self._action(LocalCoefficients(onsite, hopping * phase[:, None, None]), vectors)

    def rewrapped(self, shifts):
        """Preserve provider/map while changing coherent fragment image labels.

        The caller must shift every atom A by shifts[atom_site[A]] @ cell.
        Providers must be invariant to this relabeling or transform their own
        extra references/descriptor graphs consistently. This method changes
        only the declared graph and cannot update opaque provider internals.
        """
        return replace(self, graph=self.graph.rewrapped(shifts))

    def validate_params(self, params):
        validate = getattr(self.coefficient_provider, "validate_params", None)
        if validate is not None:
            if not callable(validate):
                raise TypeError("provider.validate_params must be callable or None")
            validate(params)

    def _validate_geometry(self, q, batch):
        q = np.asarray(q)
        shape = self.spec.system.q_shape
        shape_invalid = q.shape != shape
        if batch:
            shape_invalid = q.ndim != 3 or q.shape[1:] != shape or q.shape[0] < 1
        if shape_invalid:
            raise ValueError("atomic coordinates must have declared shape, with one leading batch axis if requested")
        if q.dtype.kind != "f" or not np.all(np.isfinite(q)):
            raise ValueError("atomic coordinates must be finite real floating values")
        values = q if batch else q[None]
        mapped = np.zeros((self.graph.nsites, len(values), 3), dtype=q.dtype)
        weights = np.asarray(self.centers.weights, dtype=q.dtype)
        with np.errstate(over="ignore", invalid="ignore"):
            np.add.at(mapped, np.asarray(self.centers.atom_site), (values * weights[None, :, None]).transpose(1, 0, 2))
            mapped = mapped.transpose(1, 0, 2)
            if not np.all(np.isfinite(mapped)):
                raise ValueError("mapped centers must be finite")
            if self.graph.edges:
                edges = np.asarray(self.graph.edges)
                displacement = mapped[:, edges[:, 1]] - mapped[:, edges[:, 0]]
                if self.graph.cell is not None:
                    cell = np.asarray(self.graph.cell, dtype=q.dtype)
                    if not np.all(np.isfinite(cell)):
                        raise ValueError("cell must be representable in the coordinate dtype")
                    displacement = displacement + edges[:, 2:].astype(q.dtype) @ cell
                distances = np.linalg.norm(displacement, axis=-1)
                if not np.all(np.isfinite(distances)) or np.any(distances == 0):
                    raise ValueError("coupled image centers must have finite nonzero separations")
        return q

    def validate_geometry(self, q):
        self._validate_geometry(q, False)

    def validate_at(self, params, q, *, batch=False):
        """Host preflight of actual outputs; batch uses one native vmap call.

        This checks the supplied geometries, not all future configurations, and
        never repairs a provider's invalid output. It is intentionally separate
        from ordinary compiled action/force evaluation.
        """
        batch = boolean_scalar(batch, "batch")
        self.validate_params(params)
        q = self._validate_geometry(q, batch)
        native_q = jnp.asarray(q)
        if native_q.dtype != q.dtype:
            # Respect the actual enabled JAX precision without setting it here.
            # A host float64 array may become float32 when x64 is disabled.
            self._validate_geometry(np.asarray(native_q), batch)
        evaluate = jax.vmap(lambda coordinate: self.coefficients(params, coordinate)) if batch else self.coefficients
        result = evaluate(native_q) if batch else evaluate(params, native_q)
        onsite, hopping = map(np.asarray, result)
        if not np.all(np.isfinite(onsite)) or not np.all(np.isfinite(hopping)):
            raise ValueError("local coefficients must be finite at supplied geometries")
        real_dtype = np.asarray(onsite.real).dtype
        eps = np.finfo(real_dtype).eps if real_dtype.kind == "f" else np.finfo(float).eps
        # Avoid overflowing abs(complex) or H-H† even when every component is
        # finite. Scale real/imaginary components BEFORE subtracting/adding.
        real, imag = np.asarray(onsite.real, dtype=float), np.asarray(onsite.imag, dtype=float)
        scale = np.maximum(1.0, np.maximum(np.max(np.abs(real), axis=(-2, -1), keepdims=True),
                                          np.max(np.abs(imag), axis=(-2, -1), keepdims=True)))
        real, imag = real/scale, imag/scale
        if (np.any(np.abs(real-real.swapaxes(-1, -2)) > 64*eps)
                or np.any(np.abs(imag+imag.swapaxes(-1, -2)) > 64*eps)):
            raise ValueError("onsite coefficients must be Hermitian at supplied geometries")
