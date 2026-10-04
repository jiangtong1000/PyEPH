"""Structured native action for compiled periodic lattice EPC dictionaries.

The host compatibility builder supplies immutable directed edge indices and
sparse EPC terms. Array parameters stay dynamic under JIT; operator action
never forms a dense matrix or a matrix for every oscillator coordinate.
"""

from dataclasses import dataclass, field

import jax.numpy as jnp
import numpy as np

from pyeph.core._configuration import boolean_scalar, integer_scalar
from pyeph.core.contracts import ModelSpec
from pyeph.core.system import SystemSpec
from pyeph.core.units import UnitSystem
from pyeph.models.base import AutoDiffModel


@dataclass(frozen=True)
class LatticeEPCModel(AutoDiffModel):
    nstates_config: int
    nmodes: int
    ncells: int
    nhalf: int = 0
    nonlocal_phonons: bool = False
    unit_system: UnitSystem = field(default_factory=UnitSystem)
    spec: ModelSpec = field(init=False)

    def __post_init__(self):
        for name in ("nstates_config", "nmodes", "ncells", "nhalf"):
            object.__setattr__(self, name, integer_scalar(getattr(self, name), name))
        object.__setattr__(self, "nonlocal_phonons", boolean_scalar(self.nonlocal_phonons, "nonlocal_phonons"))
        if self.nstates_config < 1 or self.nmodes < 0 or self.ncells < 1 or self.nhalf < 0:
            raise ValueError("invalid lattice model dimensions")
        if self.nonlocal_phonons and (self.nhalf < 1 or 2*self.nhalf != self.ncells):
            raise ValueError("nonlocal compatibility modes require an even paired MP grid")
        ndof = self.nmodes * (2*self.nhalf if self.nonlocal_phonons else self.ncells)
        object.__setattr__(self, "spec", ModelSpec(
            SystemSpec(self.nstates_config, (max(ndof, 1),), coordinate_kind="canonical"),
            name="lattice_epc", complex_valued=True, probes=("current_x", "current_y"),
            unit_system=self.unit_system,
        ))

    def fields(self, params, q):
        """Canonical Q to historical dimensionless real-space oscillator X."""
        if self.nmodes == 0:
            return jnp.zeros((0, self.ncells), dtype=q.dtype)
        if not self.nonlocal_phonons:
            return q.reshape(self.nmodes, self.ncells) * jnp.sqrt(2*params["frequencies"][:, None])
        modes = q.reshape(self.nmodes, self.nhalf, 2)
        half = jnp.sqrt(params["frequencies"]) * (modes[..., 0] + 1j*modes[..., 1])
        return 2*jnp.real(jnp.einsum("mh,ch,mh->mc", half, params["phase"], params["gauge"]))

    def elements(self, params, q):
        fields = self.fields(params, q).reshape(-1)
        values = params["static"]
        if fields.size:
            increments = params["coefficients"] * fields[params["field_indices"]]
            values = values.at[params["term_edges"]].add(increments)
        return values

    def _action(self, params, values, vectors):
        scalar = vectors.ndim == 1
        v = vectors[:, None] if scalar else vectors
        output = jnp.zeros(v.shape, dtype=jnp.result_type(values, v))
        output = output.at[params["rows"]].add(values[:, None] * v[params["columns"]])
        return output[:, 0] if scalar else output

    def apply(self, params, q, vectors):
        return self._action(params, self.elements(params, q), vectors)

    def diagonal(self, params, q):
        values = self.elements(params, q)
        diagonal = jnp.where(params["rows"] == params["columns"], values, 0)
        return jnp.zeros(self.nstates, dtype=values.dtype).at[params["rows"]].add(diagonal)

    def probe_apply(self, params, context, probe, vectors):
        if probe not in self.spec.probes:
            raise ValueError(f"unknown physical probe {probe!r}")
        axis = self.spec.probes.index(probe)
        # Legacy displacement is r_column-r_row, with the omitted i restored.
        values = 1j * params["displacements"][:, axis] * self.elements(params, context.q)
        return self._action(params, values, vectors)

    def reference_energy(self, params, q):
        return .5*jnp.sum((params["canonical_frequencies"] * q)**2)

    def validate_params(self, params):
        edges = np.asarray(params["rows"]).size
        terms = np.asarray(params["term_edges"]).size
        for key, shape in {"rows": (edges,), "columns": (edges,), "static": (edges,),
                           "displacements": (edges, 2), "term_edges": (terms,),
                           "field_indices": (terms,), "coefficients": (terms,),
                           "canonical_frequencies": self.spec.system.q_shape}.items():
            array = np.asarray(params[key])
            if array.shape != shape or not np.isfinite(array).all():
                raise ValueError(f"{key} must be finite with shape {shape}")
        for key, limit in (("rows", self.nstates), ("columns", self.nstates),
                           ("term_edges", edges), ("field_indices", self.nmodes*self.ncells)):
            values = np.asarray(params[key])
            if not np.issubdtype(values.dtype, np.integer) or np.any(values < 0) or np.any(values >= limit):
                raise ValueError(f"{key} contains an invalid integer index")
        if np.any(np.asarray(params["canonical_frequencies"]) < 0):
            raise ValueError("canonical frequencies must be nonnegative")
