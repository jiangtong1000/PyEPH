"""Oriented one-orbital fragment coefficients for a declared effective basis.

The two-centre angular form follows Slater and Koster, Phys. Rev. 94, 1498
(1954), doi:10.1103/PhysRev.94.1498. Applying this axial-orbital approximation
to a molecular fragment, its exponential radial law, and its deformation
potentials are modeling choices, not a parameterization of any material.
"""

from dataclasses import dataclass, field
from collections.abc import Mapping

import jax.numpy as jnp
import numpy as np

from pyeph.core._configuration import integer_scalar, real_scalar
from pyeph.models.local import LocalCoefficients


@dataclass(frozen=True)
class OrientedFragmentCoefficients:
    """One real, orthonormal, effective carrier state per fragment.

    An ordered atom triplet (o,a,b) fixes each oriented normal
    n = phase * ((q[a]-q[o]) x (q[b]-q[o])) / |cross|. No eigensolver or
    coordinate-dependent sign fixing is used. A constant phase flip changes
    the corresponding Hamiltonian row/column sign; electronic states and
    labels must be transformed by the same gauge. Coordinates are coherently
    unwrapped, including when LocalBlockGraph has periodic image edges.

    Runtime params are a mapping with onsite (N,), deformation (N,2),
    bond_lengths (N,2), and scalar pp_sigma, pp_pi, decay, reference_distance.
    All are real. The onsite energy is onsite + sum(deformation*(l-l_ref));
    both radial integrals are pp_* exp[-decay*(r-reference_distance)].
    LocalBlockModel applies the final hopping cutoff exactly once.

    Units are caller-owned and consistent (normally Hartree and bohr).
    This is an effective fixed-basis potential, not a moving AO basis: no
    overlap metric, basis-connection force, or charged-state calibration is
    inferred. A scalar nuclear reference potential must be supplied separately.

    Zero/short anchor bonds, nearly collinear anchors, and mismatched fragment
    assignments yield nonfinite coefficients, so LocalBlockModel.validate_at
    and checked dynamics reject them. Thresholds define an excluded domain;
    they do not regularize singular frame physics. All future geometries must
    stay strictly inside the admitted domain.
    """

    anchors: tuple
    phases: tuple | None = None
    min_bond_length: float = field(default=1e-10, kw_only=True)
    min_sine: float = field(default=1e-6, kw_only=True)

    def __post_init__(self):
        try:
            anchors = tuple(tuple(integer_scalar(x, "anchor index") for x in row)
                            for row in self.anchors)
        except TypeError as exc:
            raise ValueError("anchors must contain ordered atom triplets") from exc
        if (not anchors or any(len(row) != 3 or len(set(row)) != 3
                               or min(row) < 0 for row in anchors)):
            raise ValueError("each fragment needs three distinct nonnegative anchor indices")
        phases = ((1,) * len(anchors) if self.phases is None else
                  tuple(integer_scalar(x, "phase") for x in self.phases))
        if len(phases) != len(anchors) or any(x not in (-1, 1) for x in phases):
            raise ValueError("phases must contain one fixed +1 or -1 per fragment")
        length = real_scalar(self.min_bond_length, "min_bond_length")
        sine = real_scalar(self.min_sine, "min_sine")
        if length <= 0 or not 0 < sine < 1:
            raise ValueError("require min_bond_length > 0 and 0 < min_sine < 1")
        object.__setattr__(self, "anchors", anchors)
        object.__setattr__(self, "phases", phases)
        object.__setattr__(self, "min_bond_length", length)
        object.__setattr__(self, "min_sine", sine)

    def validate_params(self, params):
        n = len(self.anchors)
        shapes = dict(onsite=(n,), deformation=(n, 2), bond_lengths=(n, 2),
                      pp_sigma=(), pp_pi=(), decay=(), reference_distance=())
        if not isinstance(params, Mapping) or set(params) != set(shapes):
            raise ValueError(f"fragment params must have exactly these keys: {tuple(shapes)}")
        for name, shape in shapes.items():
            value = np.asarray(params[name])
            if value.shape != shape or value.dtype.kind not in "iuf" or not np.isfinite(value).all():
                raise ValueError(f"{name} must be finite real values with shape {shape}")
        if (np.any(np.asarray(params["bond_lengths"]) <= 0)
                or float(params["decay"]) < 0 or float(params["reference_distance"]) <= 0):
            raise ValueError("reference lengths must be positive and decay nonnegative")

    def __call__(self, params, q, geometry):
        q = jnp.asarray(q)
        n = len(self.anchors)
        if q.ndim != 2 or q.shape[1] != 3 or q.dtype.kind != "f":
            raise ValueError("fragment coordinates must have real floating shape (natoms,3)")
        if max(max(row) for row in self.anchors) >= q.shape[0]:
            raise ValueError("anchor index lies outside the supplied atomic coordinates")
        if geometry.centers.shape != (n, 3):
            raise ValueError("one anchor triplet is required per electronic fragment")
        indices = jnp.asarray(self.anchors, dtype=jnp.int32)
        first = q[indices[:, 1]] - q[indices[:, 0]]
        second = q[indices[:, 2]] - q[indices[:, 0]]
        lengths = jnp.stack((jnp.linalg.norm(first, axis=1),
                             jnp.linalg.norm(second, axis=1)), axis=1)
        # Normalize each bond first: the sine guard is dimensionless and does
        # not spuriously reject a uniformly rescaled nonsingular triangle.
        safe_lengths = jnp.where(lengths > self.min_bond_length, lengths, 1.)
        cross = jnp.cross(first/safe_lengths[:, :1], second/safe_lengths[:, 1:])
        sine = jnp.linalg.norm(cross, axis=1)
        valid = (jnp.all(lengths > self.min_bond_length, axis=1)
                 & (sine > self.min_sine)
                 & jnp.all(geometry.atom_site[indices] == jnp.arange(n)[:, None], axis=1))
        normal = cross/jnp.where(valid, sine, 1.)[:, None]
        normal = normal*jnp.asarray(self.phases, dtype=q.dtype)[:, None]
        normal = jnp.where(valid[:, None], normal, jnp.nan)
        onsite = params["onsite"] + jnp.sum(
            params["deformation"]*(lengths-params["bond_lengths"]), axis=1)
        onsite = jnp.where(valid, onsite, jnp.nan)
        i, j = geometry.pairs[:, 0], geometry.pairs[:, 1]
        direction = geometry.displacements/geometry.distances[:, None]
        parallel = jnp.sum(normal[i]*direction, axis=1)*jnp.sum(normal[j]*direction, axis=1)
        transverse = jnp.sum(normal[i]*normal[j], axis=1)-parallel
        radial = jnp.exp(-params["decay"]*(geometry.distances-params["reference_distance"]))
        hopping = radial*(params["pp_sigma"]*parallel+params["pp_pi"]*transverse)
        return LocalCoefficients(onsite[:, None, None], hopping[:, None, None])
