"""Differentiable two-centre s,p coefficients for fixed orbital blocks.

The angular form follows Slater and Koster, Phys. Rev. 94, 1498 (1954),
doi:10.1103/PhysRev.94.1498. Radial laws and parameter values are supplied by
the caller; this module does not infer a material or a nuclear potential.
"""

from collections.abc import Mapping
from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from pyeph.core._configuration import boolean_scalar, integer_scalar
from pyeph.models.local import LocalCoefficients


def _spin_orbit_matrix(dtype):
    """L dot sigma in (s,px,py,pz) up, then (s,px,py,pz) down."""
    angular = jnp.asarray([
        [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, -1j], [0, 0, 1j, 0]],
        [[0, 0, 0, 0], [0, 0, 0, 1j], [0, 0, 0, 0], [0, -1j, 0, 0]],
        [[0, 0, 0, 0], [0, 0, -1j, 0], [0, 1j, 0, 0], [0, 0, 0, 0]],
    ], dtype=dtype)
    pauli = jnp.asarray([[[0, 1], [1, 0]], [[0, -1j], [1j, 0]],
                         [[1, 0], [0, -1]]], dtype=dtype)
    return jnp.einsum("xab,xij->aibj", pauli, angular).reshape(8, 8)


@dataclass(frozen=True)
class SlaterKosterSPCoefficients:
    """Fixed global Cartesian s,px,py,pz orbitals on every graph centre.

    Runtime params contain real arrays:
      onsite (N,4), hopping (E,5), decay (E,5), reference_distance (E,).
    The five directed-edge channels are ss, sp, ps, pp_sigma, pp_pi, with
    T[s,p] = +n*sp and T[p,s] = -n*ps for the displacement from i to j.
    Different sp and ps channels admit heteronuclear bonds. Reversing an edge
    swaps sp with ps; LocalBlockModel adds the Hermitian reverse automatically.

    Each channel is hopping*exp[-decay*(r-reference_distance)]. This explicit
    exponential extension is a modeling choice unless calibrated independently.
    LocalBlockModel applies the final smooth graph cutoff once, including its
    coordinate derivative. No hard neighbor changes occur inside this provider.

    With spinful=True, blocks have eight channels ordered spin-major, and
    params also contains soc (N,). The on-site term is soc*L dot sigma, so
    the isolated p manifold has eigenvalues -2*soc (doublet) and +soc
    (quartet). Thus soc is one third of the positive atomic splitting, not
    the coefficient multiplying L dot S when S=sigma/2.

    Optional phases are fixed +/-1 signs with shape (N,4), repeated for both
    spins. They transform onsite, hopping and all derivatives consistently;
    states/labels must use the same gauge. They never track coordinate-dependent
    eigensolver signs. Spinful output requires complex_valued=True on the model.
    Unequal onsite px/py/pz energies define a crystal field fixed to the declared
    global axes; rotating coordinates alone does not rotate that field.

    The model is a fixed orthonormal effective potential operator. Cartesian
    orbital angular dependence does not imply a moving AO overlap connection.
    Supply the neutral reference potential separately. Cs or other spectator
    atoms can remain in q with zero centre weights; absent descriptors then
    imply zero carrier force on them, not zero total nuclear force.
    """

    spinful: bool = False
    phases: tuple | None = None

    def __post_init__(self):
        object.__setattr__(self, "spinful", boolean_scalar(self.spinful, "spinful"))
        if self.phases is not None:
            try:
                phases = tuple(tuple(integer_scalar(x, "orbital phase") for x in row)
                               for row in self.phases)
            except TypeError as exc:
                raise ValueError("phases must contain four signs per site") from exc
            if not phases or any(len(row) != 4 or any(x not in (-1, 1) for x in row)
                                 for row in phases):
                raise ValueError("phases must contain four +/-1 signs per site")
            object.__setattr__(self, "phases", phases)

    def validate_params(self, params):
        keys = {"onsite", "hopping", "decay", "reference_distance"}
        if self.spinful:
            keys.add("soc")
        if not isinstance(params, Mapping) or set(params) != keys:
            raise ValueError(f"s,p params must have exactly these keys: {sorted(keys)}")
        values = {k: np.asarray(v) for k, v in params.items()}
        for name, value in values.items():
            if value.dtype.kind not in "iuf" or not np.isfinite(value).all():
                raise ValueError(f"{name} must contain finite real values")
        onsite, hopping = values["onsite"], values["hopping"]
        if onsite.ndim != 2 or onsite.shape[0] < 1 or onsite.shape[1] != 4:
            raise ValueError("onsite must have shape (nsites,4)")
        if hopping.ndim != 2 or hopping.shape[1] != 5:
            raise ValueError("hopping must have shape (nedges,5)")
        if (values["decay"].shape != hopping.shape
                or values["reference_distance"].shape != hopping.shape[:1]):
            raise ValueError("decay/reference_distance must have shape (nedges,5)/(nedges,)")
        if np.any(values["decay"] < 0) or np.any(values["reference_distance"] <= 0):
            raise ValueError("decays must be nonnegative and reference distances positive")
        if self.spinful and values["soc"].shape != onsite.shape[:1]:
            raise ValueError("soc must have shape (nsites,)")
        if self.phases is not None and len(self.phases) != len(onsite):
            raise ValueError("phases must contain one row per site")

    def __call__(self, params, q, geometry):
        n, e = geometry.centers.shape[0], geometry.pairs.shape[0]
        onsite = jnp.asarray(params["onsite"])
        hopping = jnp.asarray(params["hopping"])
        decay = jnp.asarray(params["decay"])
        distance = jnp.asarray(params["reference_distance"])
        if (onsite.shape != (n, 4) or hopping.shape != (e, 5)
                or decay.shape != (e, 5) or distance.shape != (e,)):
            raise ValueError("s,p parameter shapes disagree with the local graph")
        dtype = jnp.result_type(q, onsite, hopping, decay, distance)
        direction = geometry.displacements/geometry.distances[:, None]
        radial = hopping*jnp.exp(-decay*(geometry.distances-distance)[:, None])
        ss, sp, ps, sigma, pi = (radial[:, i] for i in range(5))
        transfer = jnp.zeros((e, 4, 4), dtype=dtype)
        transfer = transfer.at[:, 0, 0].set(ss)
        transfer = transfer.at[:, 0, 1:].set(sp[:, None]*direction)
        transfer = transfer.at[:, 1:, 0].set(-ps[:, None]*direction)
        pp = pi[:, None, None]*jnp.eye(3, dtype=dtype)
        pp = pp+(sigma-pi)[:, None, None]*direction[:, :, None]*direction[:, None, :]
        transfer = transfer.at[:, 1:, 1:].set(pp)
        diagonal = onsite[:, :, None]*jnp.eye(4, dtype=dtype)
        if self.spinful:
            soc = jnp.asarray(params["soc"])
            if soc.shape != (n,):
                raise ValueError("soc must have shape (nsites,)")
            complex_dtype = jnp.result_type(dtype, soc, 1j)
            spin_identity = jnp.eye(2, dtype=dtype)
            diagonal = jnp.einsum("ab,nij->naibj", spin_identity, diagonal).reshape(n, 8, 8)
            transfer = jnp.einsum("ab,eij->eaibj", spin_identity, transfer).reshape(e, 8, 8)
            diagonal = diagonal+soc[:, None, None]*_spin_orbit_matrix(complex_dtype)
        if self.phases is not None:
            if len(self.phases) != n:
                raise ValueError("phases must contain one row per site")
            phase = jnp.asarray(self.phases, dtype=dtype)
            if self.spinful:
                phase = jnp.tile(phase, (1, 2))
            i, j = geometry.pairs[:, 0], geometry.pairs[:, 1]
            diagonal = diagonal*phase[:, :, None]*phase[:, None, :]
            transfer = transfer*phase[i, :, None]*phase[j, None, :]
        return LocalCoefficients(diagonal, transfer)
