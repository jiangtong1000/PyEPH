"""Explicit host lifecycle for finite and periodic local-block candidate graphs.

Each snapshot is immutable and has a fixed edge shape. Search and rebuilding
never occur inside a differentiated model or a compiled dynamics step.
"""

from dataclasses import dataclass, field, replace
from copy import copy
import hashlib
from itertools import product
from fractions import Fraction
from math import ceil, floor, isqrt
from struct import pack, unpack
import json

import numpy as np

from pyeph.core._configuration import integer_scalar, real_scalar
from pyeph.models.local import AtomCenterMap, LocalBlockGraph


def _coordinates(value, count=None):
    array = np.asarray(value)
    if (array.ndim != 2 or array.shape[1] != 3 or not len(array)
            or (count is not None and len(array) != count)
            or array.dtype.kind not in "iuf" or not np.isfinite(array).all()):
        raise ValueError("coordinates must have finite real shape (n,3)")
    return np.asarray(array, dtype=float)


def _cell(value):
    # Reuse the graph's row-vector convention and cell validation.
    graph = LocalBlockGraph(1, 1, (), value)
    if graph.cell is None:
        raise ValueError("an explicit periodic cell is required")
    return np.asarray(graph.cell)


def _images(value, count):
    array = np.asarray(value)
    if array.shape != (count, 3) or array.dtype.kind not in "iu":
        raise ValueError("images must have integer shape (n,3)")
    if np.any(array >= 2**52) or np.any(array <= -(2**52)):
        raise ValueError("image magnitude must be below 2**52 for float64 coordinates")
    return np.asarray(array, dtype=np.int64)


def _center_positions(q, centers):
    result = np.zeros((centers.nsites, 3))
    np.add.at(result, np.asarray(centers.atom_site), np.asarray(centers.weights)[:, None] * q)
    if not np.isfinite(result).all():
        raise ValueError("center coordinates overflowed")
    return result


def _translated(q, offset, cell):
    result = q + offset
    # Error-free TwoSum residual detects lost subcell information even when
    # subtracting the same huge shift appears to recover the old coordinate.
    recovered_offset = result - q
    residual = (q - (result - recovered_offset)) + (offset - recovered_offset)
    tolerance = 64 * np.finfo(float).eps * max(1., float(np.max(np.abs(cell))))
    if (not np.isfinite(result).all() or not np.isfinite(residual).all()
            or np.any(np.abs(residual) > tolerance)):
        raise ValueError("lattice translation loses coordinate precision; use a nearer origin")
    return result


def _same_vectors(before, after, cell):
    if not np.isfinite(before).all() or not np.isfinite(after).all():
        raise ValueError("lattice translation produced nonfinite geometry")
    scale = max(1., float(np.max(np.abs(cell))),
                float(np.max(np.abs(before))) if np.size(before) else 0.)
    tolerance = 64 * np.finfo(float).eps * scale
    if not np.allclose(before, after, rtol=0, atol=tolerance):
        raise ValueError("lattice translation loses geometry precision; use a nearer origin")


def _fragment_vectors(q, centers):
    sites = np.asarray(centers.atom_site)
    _, anchors = np.unique(sites, return_index=True)
    return q - q[anchors[sites]]


def wrap_positions(positions, cell):
    """Return primary-cell positions and explicit removed integer images.

    ``positions = wrapped + images @ cell``; cell vectors are rows. This
    operation does not infer molecular connectivity or trajectory windings.
    Use :func:`wrap_atoms` to shift every atom of a fragment together.
    """
    positions, cell = _coordinates(positions), _cell(cell)
    fractional = positions @ np.linalg.inv(cell)
    if not np.isfinite(fractional).all() or np.any(np.abs(fractional) >= 2**52):
        raise ValueError("fractional coordinates exceed reliable integer image resolution")
    images = np.floor(fractional).astype(np.int64)
    wrapped = _translated(positions, -(images @ cell), cell)
    return wrapped, images


def unwrap_positions(positions, images, cell):
    """Restore positions using supplied winding images, never a minimum-image guess."""
    positions, cell = _coordinates(positions), _cell(cell)
    images = _images(images, len(positions))
    return _translated(positions, images @ cell, cell)


def wrap_atoms(q, centers, cell):
    """Wrap centers while preserving all intra-fragment atomic differences.

    Input fragments must already be coherently unwrapped. Returned images have
    shape ``(nsites,3)`` and satisfy ``q = wrapped + images[atom_site] @ cell``.
    A fragment can extend outside the primary cell after this operation.
    """
    if not isinstance(centers, AtomCenterMap):
        raise TypeError("centers must be an AtomCenterMap")
    q, cell = _coordinates(q, centers.natoms), _cell(cell)
    wrapped_centers, images = wrap_positions(_center_positions(q, centers), cell)
    wrapped = _translated(q, -(images[np.asarray(centers.atom_site)] @ cell), cell)
    _same_vectors(wrapped_centers, _center_positions(wrapped, centers), cell)
    _same_vectors(_fragment_vectors(q, centers), _fragment_vectors(wrapped, centers), cell)
    return wrapped, images


def unwrap_atoms(q, images, centers, cell):
    """Undo :func:`wrap_atoms` using its explicit per-fragment images."""
    if not isinstance(centers, AtomCenterMap):
        raise TypeError("centers must be an AtomCenterMap")
    q, cell = _coordinates(q, centers.natoms), _cell(cell)
    images = _images(images, centers.nsites)
    unwrapped = unwrap_positions(q, images[np.asarray(centers.atom_site)], cell)
    expected = unwrap_positions(_center_positions(q, centers), images, cell)
    # Compare translation residuals rather than huge absolute coordinates.
    _same_vectors(np.zeros_like(expected), _center_positions(unwrapped, centers) - expected, cell)
    _same_vectors(_fragment_vectors(q, centers), _fragment_vectors(unwrapped, centers), cell)
    return unwrapped


class NeighborCapacityError(ValueError):
    """A complete search exceeded the declared edge allocation budget."""

    def __init__(self, required, capacity):
        self.required, self.capacity = required, capacity
        super().__init__(f"neighbor graph needs {required} edges; capacity is {capacity}")


class NeighborCoverageError(ValueError):
    """The reference skin certificate has expired; the old snapshot is retained."""

    def __init__(self, report):
        self.report = report
        super().__init__("neighbor skin exhausted; explicitly rebuild before propagation "
                         f"(maximum center displacement {report.maximum_displacement:g})")


def _fraction(value):
    """Decode float64 bits without floating arithmetic, including subnormals."""
    bits = unpack(">Q", pack(">d", value))[0]
    exponent = (bits >> 52) & 0x7ff
    mantissa = bits & ((1 << 52) - 1)
    if exponent == 0x7ff:
        raise ValueError("exact host geometry requires finite float64 inputs")
    power = -1074 if exponent == 0 else exponent - 1075
    if exponent:
        mantissa += 1 << 52
    if bits >> 63:
        mantissa = -mantissa
    return (Fraction(mantissa << power) if power >= 0
            else Fraction(mantissa, 1 << -power))


class _HostGeometry:
    """Approximate display coordinates and cached exact float-input centers."""

    def __init__(self, q, centers):
        self.q, self.centers = q, centers
        self.positions = _center_positions(q, centers)
        self._atoms = [[] for _ in range(centers.nsites)]
        for atom, site in enumerate(centers.atom_site):
            self._atoms[site].append(atom)
        self._exact = {}

    def exact(self, site):
        if site not in self._exact:
            atoms = self._atoms[site]
            if len(atoms) == 1 and self.centers.weights[atoms[0]] == 1.:
                result = tuple(_fraction(x) for x in self.q[atoms[0]])
            else:
                terms = []
                for atom in atoms:
                    weight = _fraction(self.centers.weights[atom])
                    if weight:
                        terms.append((weight, tuple(_fraction(x) for x in self.q[atom])))
                result = tuple(sum((weight*position[axis] for weight, position in terms),
                                   Fraction(0)) for axis in range(3))
            self._exact[site] = result
        return self._exact[site]


def _integer_geometry(points, radius, cell=()):
    """Scale dyadic float-input geometry once; distance comparisons use integers.

    Exact weighted centers, cell entries and radius are dyadic rationals. Their
    denominators divide the largest power of two, including subnormal inputs.
    The periodic inverse is generally not dyadic and stays rational separately.
    """
    rows = (*points, *cell, (radius,))
    denominator = max(value.denominator for row in rows for value in row)

    def scale(rows):
        return tuple(tuple(value.numerator * (denominator // value.denominator)
                           for value in row) for row in rows)

    scaled_radius = radius.numerator * (denominator // radius.denominator)
    return scale(points), scaled_radius, scale(cell), denominator


def _inverse_exact(cell):
    """Exact 3x3 inverse for complete periodic image boxes, once per search."""
    rows = [[_fraction(cell[i][j]) for j in range(3)]
            + [Fraction(int(i == j)) for j in range(3)] for i in range(3)]
    for column in range(3):
        pivot = next((i for i in range(column, 3) if rows[i][column]), None)
        if pivot is None:
            raise ValueError("periodic cell is exactly singular")
        rows[column], rows[pivot] = rows[pivot], rows[column]
        factor = rows[column][column]
        rows[column] = [x / factor for x in rows[column]]
        for i in range(3):
            if i != column:
                factor = rows[i][column]
                rows[i] = [x - factor*y for x, y in zip(rows[i], rows[column])]
    return tuple(tuple(row[3:]) for row in rows)


def _ceil_sqrt(value):
    lower = isqrt(value.numerator // value.denominator)
    return lower + (lower*lower*value.denominator != value.numerator)


def _edges(geometry, cutoff, cell, max_image_checks, *, skin=0.):
    """Complete exact float-input search, intentionally host-only and bounded."""
    radius = _fraction(cutoff) + _fraction(skin)
    inverse = None if cell is None else _inverse_exact(cell)
    limits = None if inverse is None else tuple(_ceil_sqrt(
        radius**2 * sum((inverse[k][j]**2 for k in range(3)), Fraction(0))) for j in range(3))
    exact_cell = () if cell is None else tuple(tuple(_fraction(x) for x in row) for row in cell)
    exact_positions = tuple(geometry.exact(site) for site in range(len(geometry.positions)))
    positions, scaled_radius, scaled_cell, denominator = _integer_geometry(
        exact_positions, radius, exact_cell)
    radius_squared = scaled_radius**2
    edges, checked = [], 0
    for a in range(len(geometry.positions)):
        for b in range(a if cell is not None else a + 1, len(geometry.positions)):

            delta = tuple(y-x for x, y in zip(positions[a], positions[b]))
            if inverse is None:
                images, count = ((0, 0, 0),), 1
            else:
                fractional = tuple(sum(
                    (Fraction(delta[k], denominator) * inverse[k][j] for k in range(3)),
                    Fraction(0)) for j in range(3))
                bounds = tuple((floor(-f)-limit, ceil(-f)+limit)
                               for f, limit in zip(fractional, limits))
                if any(lo < np.iinfo(np.int32).min or hi > np.iinfo(np.int32).max
                       for lo, hi in bounds):
                    raise ValueError("candidate image bounds must fit in int32")
                count = 1
                for lo, hi in bounds:
                    count *= hi-lo+1
            checked += count
            if checked > max_image_checks:
                raise ValueError(f"candidate search exceeds max_image_checks={max_image_checks}; "
                                 "increase the explicit host search budget")
            if inverse is not None:
                # product pools its input iterables eagerly. Check the full
                # Python-integer allocation budget BEFORE constructing it.
                images = product(*(range(lo, hi+1) for lo, hi in bounds))
            for image in images:
                if a == b and image <= (0, 0, 0):
                    continue
                vector = delta
                if scaled_cell:
                    vector = tuple(delta[j] + sum(image[k]*scaled_cell[k][j] for k in range(3))
                                   for j in range(3))
                if sum(x*x for x in vector) <= radius_squared:
                    edges.append((a, b, *image))
    return tuple(edges)


@dataclass(frozen=True)
class NeighborCoverage:
    """Per-geometry coverage evidence; an expired skin is not an omitted edge.

    ``maximum_displacement`` is an approximate display value; ``within_skin``
    uses exact float-input host geometry. It need not equal a floating comparison
    of that display value against skin/2 at a rounding boundary.
    ``missing_edges=None`` means no exhaustive search was requested. An empty
    tuple certifies coverage at this geometry only, even if the skin expired.
    """

    maximum_displacement: float
    within_skin: bool
    missing_edges: tuple | None

    @property
    def covered(self):
        if self.missing_edges is not None:
            return not self.missing_edges
        return True if self.within_skin else None

    @property
    def rebuild_required(self):
        return not self.within_skin


@dataclass(frozen=True)
class NeighborGraph:
    """Candidate snapshot for an existing :class:`LocalBlockModel`.

    Edges include every center/image pair within the exact sum of the float64
    ``cutoff`` and ``skin`` inputs at the
    reference geometry. A displacement of at most ``skin/2`` for every center
    certifies that every pair within the physical cutoff remains represented.
    The host certificate interprets float64 inputs as exact real geometry; it
    does not certify device rounding or hidden dynamics stages. It assumes
    fixed cell, fixed center map and continuous,
    consistently unwrapped coordinates. It is deliberately conservative.

    ``capacity`` caps the number of unique Hermitian edges; no edge is silently
    dropped and no dummy edge is inserted. A rebuild can change the edge shape
    and requires a new model and execution object. This host object does not
    attach an automatic guard to Runner or inspect intermediate compiled steps.
    """

    centers: AtomCenterMap
    reference_coordinates: tuple
    norbitals: int
    switch_on: float
    cutoff: float
    skin: float
    cell: tuple | None = None
    capacity: int | None = None
    max_image_checks: int = 1_000_000
    generation: int = 0
    parent_identity: str | None = None
    graph: LocalBlockGraph = field(init=False)
    reference_centers: tuple = field(init=False)

    def __post_init__(self):
        if not isinstance(self.centers, AtomCenterMap):
            raise TypeError("centers must be an AtomCenterMap")
        q = _coordinates(self.reference_coordinates, self.centers.natoms)
        empty = LocalBlockGraph(self.centers.nsites, self.norbitals, (), self.cell,
                                switch_on=self.switch_on, cutoff=self.cutoff)
        skin = real_scalar(self.skin, "skin")
        if skin < 0 or not np.isfinite(empty.cutoff + skin):
            raise ValueError("skin must be nonnegative with finite cutoff + skin")
        capacity = self.capacity
        if capacity is not None:
            capacity = integer_scalar(capacity, "capacity")
            if capacity < 0:
                raise ValueError("capacity must be nonnegative")
        checks = integer_scalar(self.max_image_checks, "max_image_checks")
        generation = integer_scalar(self.generation, "generation")
        if checks < 1 or generation < 0:
            raise ValueError("max_image_checks must be positive and generation nonnegative")
        if (self.parent_identity is not None and
                (not isinstance(self.parent_identity, str) or not self.parent_identity)):
            raise ValueError("parent_identity must be a nonempty string or None")
        geometry = _HostGeometry(q, self.centers)
        positions = geometry.positions
        edges = _edges(geometry, empty.cutoff, empty.cell, checks, skin=skin)
        if capacity is not None and len(edges) > capacity:
            raise NeighborCapacityError(len(edges), capacity)
        for key, value in dict(reference_coordinates=tuple(map(tuple, q.tolist())),
                               reference_centers=tuple(map(tuple, positions.tolist())),
                               norbitals=empty.norbitals, switch_on=empty.switch_on,
                               cutoff=empty.cutoff, skin=skin, cell=empty.cell,
                               capacity=capacity, max_image_checks=checks,
                               generation=generation, graph=replace(empty, edges=edges)).items():
            object.__setattr__(self, key, value)

    def check(self, q, *, exhaustive=False):
        """Check skin displacement, optionally searching for omitted physical pairs.

        No minimum-image reduction is applied to displacement from the reference:
        wrapping a site without rewrapping the snapshot must invalidate its skin.
        Decisions use exact rational float-input geometry;
        maximum_displacement is an approximate diagnostic. This
        checks one float-input geometry, not device arithmetic or unsaved stages.
        """
        if not isinstance(exhaustive, (bool, np.bool_)):
            raise ValueError("exhaustive must be boolean")
        geometry = _HostGeometry(_coordinates(q, self.centers.natoms), self.centers)
        reference = _HostGeometry(np.asarray(self.reference_coordinates), self.centers)
        with np.errstate(over="ignore", invalid="ignore"):
            displacement = np.hypot.reduce(geometry.positions - self.reference_centers, axis=1)
        maximum = float(np.max(displacement))
        count = self.centers.nsites
        points = tuple(reference.exact(site) for site in range(count)) + tuple(
            geometry.exact(site) for site in range(count))
        scaled, limit, _, _ = _integer_geometry(points, _fraction(self.skin) / 2)
        limit_squared = limit**2
        within = all(sum((y-x)**2 for x, y in zip(scaled[site], scaled[count+site]))
                     <= limit_squared for site in range(count))
        missing = None
        if exhaustive:
            required = _edges(geometry, self.cutoff, self.cell, self.max_image_checks)
            existing = set(self.graph.edges)
            missing = tuple(edge for edge in required if edge not in existing)
        return NeighborCoverage(maximum, within, missing)

    def require_coverage(self, q, *, exhaustive=False):
        """Raise with diagnostics when an explicit rebuild is required."""
        report = self.check(q, exhaustive=exhaustive)
        if report.rebuild_required or report.covered is False:
            raise NeighborCoverageError(report)
        return report

    def rebuild(self, q, *, capacity=None):
        """Create the next fixed-cell generation; failed searches leave this one intact."""
        return replace(self, reference_coordinates=q,
                       capacity=self.capacity if capacity is None else capacity,
                       generation=self.generation + 1, parent_identity=self.identity)

    def rewrapped(self, shifts):
        """Use R'=R+shifts@cell for all atoms of each center and its graph images.

        With ``wrapped, images = wrap_atoms(q, centers, cell)``, use
        ``snapshot.rewrapped(-images)`` to check/evaluate ``wrapped``. The
        Image relabeling preserves the intended physical displacements. Rounded
        translated coordinates must also retain complete host candidates; if
        not, this operation rejects and requires an explicit rebuild/remap.
        """
        if self.cell is None:
            raise ValueError("rewrapping requires a periodic cell")
        shifts = _images(shifts, self.centers.nsites)
        q = np.asarray(self.reference_coordinates)
        shifted = _translated(q, shifts[np.asarray(self.centers.atom_site)] @ self.cell, self.cell)
        expected_centers = _translated(np.asarray(self.reference_centers), shifts @ self.cell,
                                       self.cell)
        positions = _center_positions(shifted, self.centers)
        _same_vectors(np.zeros_like(positions), positions - expected_centers, self.cell)
        _same_vectors(_fragment_vectors(q, self.centers),
                      _fragment_vectors(shifted, self.centers), self.cell)
        # Preserve candidate membership/order, then verify completeness for the
        # translated float-input geometry without silently adding interactions.
        graph = self.graph.rewrapped(shifts)
        for old, new in zip(self.graph.edges, graph.edges):
            a, b = old[:2]
            before = (np.asarray(self.reference_centers[b]) - self.reference_centers[a]
                      + np.asarray(old[2:]) @ self.cell)
            after = positions[b] - positions[a] + np.asarray(new[2:]) @ self.cell
            _same_vectors(before, after, self.cell)
        # Rounded translations change the exact float-input reference geometry.
        # Preserve edge order/membership only if it is still a complete list.
        required = _edges(_HostGeometry(shifted, self.centers), self.cutoff, self.cell,
                          self.max_image_checks, skin=self.skin)
        if not set(required).issubset(graph.edges):
            raise ValueError("rewrapping loses candidate coverage; explicitly rebuild and "
                             "remap parameters at the translated coordinates")
        result = copy(self)
        for key, value in dict(reference_coordinates=tuple(map(tuple, shifted.tolist())),
                               reference_centers=tuple(map(tuple, positions.tolist())),
                               graph=graph, generation=self.generation + 1,
                               parent_identity=self.identity).items():
            object.__setattr__(result, key, value)
        return result

    def metadata(self):
        """Fresh JSON-compatible construction evidence for a campaign manifest.

        The model graph already enters normal model provenance. Save this record
        separately when reference geometry, skin budget and rebuild lineage are
        needed; it does not identify provider weights or physical model accuracy.
        """
        return {"schema": 1, "kind": "pyeph.models.neighbors.NeighborGraph",
                "centers": {"atom_site": list(self.centers.atom_site),
                            "weights": list(self.centers.weights),
                            "nsites": self.centers.nsites},
                "reference_coordinates": [list(row) for row in self.reference_coordinates],
                "norbitals": self.norbitals, "switch_on": self.switch_on,
                "cutoff": self.cutoff, "skin": self.skin,
                "cell": None if self.cell is None else [list(row) for row in self.cell],
                "capacity": self.capacity, "max_image_checks": self.max_image_checks,
                "generation": self.generation, "parent_identity": self.parent_identity,
                "edges": [list(row) for row in self.graph.edges]}

    @property
    def identity(self):
        payload = json.dumps(self.metadata(), sort_keys=True, separators=(",", ":"), allow_nan=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()
