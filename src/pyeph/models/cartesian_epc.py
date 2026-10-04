"""Periodic linear EPC in Cartesian displacements, without a dense Jacobian.

Primitive-cell hopping, derivative and force-constant stencils are reused in
every simulation cell. Electronic image channels stay separate even when they
wrap onto the same finite edge; this is essential for the Peierls current.
"""

from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np

from pyeph.core._configuration import boolean_scalar, integer_scalar, real_scalar
from pyeph.core.contracts import LowRankWeight, ModelSpec
from pyeph.core.system import SystemSpec
from pyeph.core.units import UnitSystem
from pyeph.models.base import AutoDiffModel


@dataclass(frozen=True)
class CartesianEPCModel(AutoDiffModel):
    """A harmonic reference and a linear fixed-Wannier carrier Hamiltonian.

    ``q`` has shape ``(ncells*natoms,3)`` and denotes Cartesian displacement
    from the reference crystal, with ordinary canonical momenta and masses.
    Parameters contain primitive-cell directed stencils. The Hamiltonian is
    the Hermitian part of that directed operator; ingestion must explicitly
    validate or authorize this projection and report any change to source data.

    Currents use fixed reference Wannier centers, not instantaneous atom
    positions. They are the Peierls derivative of this effective Hamiltonian,
    not a claim of a complete ab initio optical velocity operator.
    """

    norbitals: int
    natoms: int
    ncells: int
    unit_system: UnitSystem = field(default_factory=UnitSystem)
    charge: float = -1.0
    term_batch_size: int = 256
    complex_valued: bool = True
    spec: ModelSpec = field(init=False)

    def __post_init__(self):
        for name in ("norbitals", "natoms", "ncells", "term_batch_size"):
            value = integer_scalar(getattr(self, name), name)
            if value < 1:
                raise ValueError(f"{name} must be positive")
            object.__setattr__(self, name, value)
        object.__setattr__(self, "charge", real_scalar(self.charge, "charge"))
        object.__setattr__(self, "complex_valued", boolean_scalar(self.complex_valued, "complex_valued"))
        if not isinstance(self.unit_system, UnitSystem):
            raise TypeError("unit_system must be a UnitSystem")
        object.__setattr__(self, "spec", ModelSpec(
            SystemSpec(self.ncells*self.norbitals, (self.ncells*self.natoms, 3)),
            name="cartesian_epc", complex_valued=self.complex_valued,
            probes=tuple(f"current_{axis}" for axis in "xyz"),
            unit_system=self.unit_system))

    def elements(self, params, q):
        """Evaluate image-resolved hopping with bounded EPC-term workspace."""
        u = jnp.asarray(q).reshape(self.ncells, self.natoms, 3)
        dtype = jnp.result_type(params["hopping"], params["epc_values"], q)
        hopping = jnp.asarray(params["hopping"], dtype=dtype)
        values = jnp.broadcast_to(hopping, (self.ncells, hopping.size))
        if params["epc_values"].shape[0] == 0:
            return values

        def add_batch(index, values):
            cells = params["neighbors"][params["epc_offsets"][index]].T
            displacement = u[cells, params["epc_atoms"][index]]
            increments = jnp.sum(displacement*params["epc_values"][index], axis=-1)
            return values.at[:, params["epc_channels"][index]].add(increments)

        return jax.lax.fori_loop(0, params["epc_values"].shape[0], add_batch, values)

    def _action(self, params, values, vectors):
        scalar = vectors.ndim == 1
        v = jnp.asarray(vectors).reshape(self.ncells, self.norbitals, -1)
        cells = params["neighbors"][params["hopping_offsets"]].T
        rows, columns = params["hopping_orbitals"].T
        output = jnp.zeros_like(v, dtype=jnp.result_type(v, values))
        output = output.at[:, rows].add(.5*values[..., None]*v[cells, columns])
        output = output.at[cells, columns].add(.5*values.conj()[..., None]*v[:, rows])
        output = output.reshape(self.nstates, -1)
        return output[:, 0] if scalar else output

    def apply(self, params, q, vectors):
        return self._action(params, self.elements(params, q), vectors)

    def prepare_action(self, params, q):
        values = self.elements(params, q)
        return lambda vectors: self._action(params, values, vectors)

    def apply_peierls(self, params, q, wavevector, vectors):
        """Apply H(kappa), using every channel's unwrapped physical displacement."""
        phase = jnp.exp(1j*(params["displacements"]@wavevector))
        return self._action(params, self.elements(params, q)*phase, vectors)

    def probe_apply(self, params, context, probe, vectors):
        if probe not in self.spec.probes:
            return super().probe_apply(params, context, probe, vectors)
        axis = "xyz".index(probe[-1])
        values = self.elements(params, context.q)
        current = 1j*self.charge*params["displacements"][:, axis]*values
        return self._action(params, current, vectors)

    def _electronic_weights(self, params, weight):
        cells = params["neighbors"][params["hopping_offsets"]].T
        rows, columns = params["hopping_orbitals"].T
        if isinstance(weight, LowRankWeight):
            if (weight.left.ndim != 2 or weight.left.shape[0] != self.nstates
                    or weight.right.shape != weight.left.shape):
                raise ValueError("weight factors must have equal (nstates, rank) shapes")
            left = jnp.asarray(weight.left).reshape(self.ncells, self.norbitals, -1)
            right = jnp.asarray(weight.right).reshape(self.ncells, self.norbitals, -1)
            forward = jnp.sum(left[:, rows].conj()*right[cells, columns], axis=-1)
            reverse = jnp.sum(left[cells, columns].conj()*right[:, rows], axis=-1)
        else:
            weight = jnp.asarray(weight)
            if weight.shape != (self.nstates, self.nstates):
                raise ValueError("dense weight must have shape (nstates,nstates)")
            row = jnp.arange(self.ncells)[:, None]*self.norbitals+rows
            column = cells*self.norbitals+columns
            forward, reverse = weight[row, column].conj(), weight[column, row].conj()
        return .5*(forward+reverse.conj())

    def contract_gradient(self, params, q, weight):
        """Contract linear derivatives directly, without a reverse-mode loop tape."""
        weights = self._electronic_weights(params, weight)
        gradient = jnp.zeros((self.ncells, self.natoms, 3), dtype=q.dtype)
        if params["epc_values"].shape[0] == 0:
            return gradient.reshape(q.shape)

        def add_batch(index, gradient):
            cells = params["neighbors"][params["epc_offsets"][index]].T
            terms = jnp.real(weights[:, params["epc_channels"][index], None]
                             * params["epc_values"][index])
            return gradient.at[cells, params["epc_atoms"][index]].add(terms)

        return jax.lax.fori_loop(0, params["epc_values"].shape[0], add_batch,
                                gradient).reshape(q.shape)

    def reference_energy(self, params, q):
        u = jnp.asarray(q).reshape(self.ncells, self.natoms, 3)
        rows, columns = params["ifc_atoms"].T
        cells = params["neighbors"][params["ifc_offsets"]].T
        return .5*jnp.einsum("cti,tij,ctj->", u[:, rows], params["ifc_values"],
                             u[cells, columns])

    def reference_gradient(self, params, q):
        u = jnp.asarray(q).reshape(self.ncells, self.natoms, 3)
        rows, columns = params["ifc_atoms"].T
        cells = params["neighbors"][params["ifc_offsets"]].T
        gradient = jnp.zeros_like(u)
        forward = .5*jnp.einsum("tij,ctj->cti", params["ifc_values"], u[cells, columns])
        reverse = .5*jnp.einsum("tij,cti->ctj", params["ifc_values"], u[:, rows])
        gradient = gradient.at[:, rows].add(forward)
        gradient = gradient.at[cells, columns].add(reverse)
        return gradient.reshape(q.shape)

    def dense_hessian(self, params, *, max_bytes=512*1024**2):
        """Explicit finite-system preparation, with a bound on matrix allocation.

        The bound counts the output matrix; construction can temporarily need
        several such arrays. Periodic Fourier baths avoid this dense operation.
        """
        ndof = self.ncells*self.natoms*3
        max_bytes = integer_scalar(max_bytes, "max_bytes")
        required = ndof*ndof*np.asarray(params["ifc_values"]).dtype.itemsize
        if max_bytes < 1 or required > max_bytes:
            raise ValueError(f"dense Hessian needs {required} bytes, above max_bytes={max_bytes}; "
                             "use a periodic Fourier bath or explicitly raise the bound")
        atoms = np.asarray(params["ifc_atoms"])
        cells = np.asarray(params["neighbors"])[np.asarray(params["ifc_offsets"])].T
        first = np.arange(self.ncells)[:, None]*self.natoms+atoms[:, 0]
        second = cells*self.natoms+atoms[:, 1]
        rows = 3*first[..., None, None]+np.arange(3)[None, None, :, None]
        columns = 3*second[..., None, None]+np.arange(3)[None, None, None, :]
        result = np.zeros((ndof, ndof), dtype=np.asarray(params["ifc_values"]).dtype)
        np.add.at(result, (rows, columns), np.asarray(params["ifc_values"])[None])
        return .5*(result+result.T)

    def validate_params(self, params):
        """Validate stencil dimensions and indices before tracing native kernels."""
        neighbors = np.asarray(params["neighbors"])
        if (neighbors.ndim != 2 or neighbors.shape[1] != self.ncells
                or not np.issubdtype(neighbors.dtype, np.integer)
                or np.any(neighbors < 0) or np.any(neighbors >= self.ncells)):
            raise ValueError("neighbors must contain valid cell indices")
        nhop = np.asarray(params["hopping"]).size
        nifc = len(params["ifc_values"])
        nchunks = len(params["epc_values"])
        shape = (nchunks, self.term_batch_size)
        shapes = {"hopping": (nhop,), "hopping_orbitals": (nhop, 2),
                  "hopping_offsets": (nhop,), "displacements": (nhop, 3),
                  "epc_channels": shape, "epc_atoms": shape, "epc_offsets": shape,
                  "epc_values": (*shape, 3), "ifc_atoms": (nifc, 2),
                  "ifc_offsets": (nifc,), "ifc_values": (nifc, 3, 3)}
        for name, expected in shapes.items():
            a = np.asarray(params[name])
            if a.shape != expected or not np.isfinite(a).all():
                raise ValueError(f"{name} must be finite with shape {expected}")
        if not self.complex_valued and (np.iscomplexobj(params["hopping"])
                                       or np.iscomplexobj(params["epc_values"])):
            raise ValueError("real CartesianEPCModel requires real hopping and derivative arrays")
        if np.iscomplexobj(params["ifc_values"]) or np.iscomplexobj(params["displacements"]):
            raise ValueError("force constants and displacements must be real")
        limits = {"hopping_orbitals": self.norbitals, "hopping_offsets": len(neighbors),
                  "epc_channels": nhop, "epc_atoms": self.natoms,
                  "epc_offsets": len(neighbors), "ifc_atoms": self.natoms,
                  "ifc_offsets": len(neighbors)}
        for name, limit in limits.items():
            a = np.asarray(params[name])
            if not np.issubdtype(a.dtype, np.integer) or np.any(a < 0) or np.any(a >= limit):
                raise ValueError(f"{name} contains invalid integer indices")
