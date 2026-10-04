"""Host data boundary for periodic Cartesian derivatives and force constants.

This representation is independent of the electronic-structure file format.
Every electronic record means <0,i|h|Re,j>, and every derivative record means
<0,i|d h / d u(Rp,a)|Re,j>. Atom and Wannier positions are Cartesian; cells
are integer primitive-lattice translations. Unit conversion happens here.
"""

from dataclasses import dataclass, field
from itertools import product

import jax.numpy as jnp
import numpy as np

from pyeph.core._configuration import integer_scalar
from pyeph.core.units import UnitSystem
from pyeph.models.cartesian_epc import CartesianEPCModel
from pyeph.models.fourier_epc import FourierCartesianEPCModel


def _array(value, name, shape, *, integer=False, real=False):
    raw = np.asarray(value)
    if raw.shape != shape or not np.isfinite(raw).all():
        raise ValueError(f"{name} must be finite with shape {shape}")
    if real and np.iscomplexobj(raw):
        raise ValueError(f"{name} must be real")
    if integer and (raw.dtype.kind == "b" or np.iscomplexobj(raw) or np.any(raw != np.round(raw))):
        raise ValueError(f"{name} must contain exact integers")
    array = np.array(raw, dtype=np.int64 if integer else None, copy=True)
    array.setflags(write=False)
    return array


def _pair_audit(records, reverse, adjoint):
    """Measure the full sparse union with its translated Hermitian counterpart."""
    keys = set(records) | {reverse(key) for key in records}
    norm = difference = maximum = max_value = 0.
    missing = 0
    for key in keys:
        value = records.get(key, 0.)
        other = records.get(reverse(key), 0.)
        defect = np.asarray(value)-adjoint(np.asarray(other))
        norm += float(np.sum(abs(np.asarray(value))**2))
        difference += float(np.sum(abs(defect)**2))
        maximum = max(maximum, float(np.max(abs(defect))))
        max_value = max(max_value, float(np.max(abs(np.asarray(value)))))
        missing += int(key in records and reverse(key) not in records)
    return dict(terms=len(records), missing_reverse_terms=missing,
                maximum_absolute_value=max_value, maximum_absolute_defect=maximum,
                relative_l2_defect=float(np.sqrt(difference/norm)) if norm else 0.,
                projection_relative_l2_change=float(.5*np.sqrt(difference/norm)) if norm else 0.)


@dataclass(frozen=True)
class CompiledCartesianEPC:
    """A native model, dynamic parameters, nuclear masses, and ingestion evidence."""

    model: CartesianEPCModel
    params: dict
    masses: object
    reference_positions: object
    report: dict


@dataclass(frozen=True)
class RealSpaceEPC:
    """Format-independent primitive-cell stencils, with owned read-only arrays.

    ``epc_channels`` selects a hopping record; ``epc_atoms`` and ``epc_cells``
    select the perturbed nucleus. ``epc_values`` is the three-component
    Cartesian derivative, with no oscillator normalization. IFCs are directed
    Cartesian Hessian blocks, before division by any atomic masses.
    """

    cell: object
    atom_positions: object
    masses: object
    wannier_centers: object
    hopping_orbitals: object
    hopping_cells: object
    hopping_values: object
    epc_channels: object
    epc_atoms: object
    epc_cells: object
    epc_values: object
    ifc_atoms: object
    ifc_cells: object
    ifc_values: object
    unit_system: UnitSystem = field(default_factory=UnitSystem)

    def __post_init__(self):
        if not isinstance(self.unit_system, UnitSystem):
            raise TypeError("unit_system must be a UnitSystem")
        nat, nwan = len(self.atom_positions), len(self.wannier_centers)
        nhop, nepc, nifc = len(self.hopping_values), len(self.epc_values), len(self.ifc_values)
        if nat < 1 or nwan < 1 or nhop < 1:
            raise ValueError("at least one atom, orbital and hopping record are required")
        shapes = {"cell": (3, 3), "atom_positions": (nat, 3), "masses": (nat,),
                  "wannier_centers": (nwan, 3), "hopping_orbitals": (nhop, 2),
                  "hopping_cells": (nhop, 3), "hopping_values": (nhop,),
                  "epc_channels": (nepc,), "epc_atoms": (nepc,),
                  "epc_cells": (nepc, 3), "epc_values": (nepc, 3),
                  "ifc_atoms": (nifc, 2), "ifc_cells": (nifc, 3), "ifc_values": (nifc, 3, 3)}
        integers = {"hopping_orbitals", "hopping_cells", "epc_channels", "epc_atoms",
                    "epc_cells", "ifc_atoms", "ifc_cells"}
        complex_allowed = {"hopping_values", "epc_values"}
        for name, shape in shapes.items():
            object.__setattr__(self, name, _array(getattr(self, name), name, shape,
                integer=name in integers, real=name not in complex_allowed))
        if abs(np.linalg.det(self.cell)) < 1e-12 or np.any(self.masses <= 0):
            raise ValueError("cell must be nonsingular and masses positive")
        for name, limit in (("hopping_orbitals", nwan), ("epc_channels", nhop),
                            ("epc_atoms", nat), ("ifc_atoms", nat)):
            values = getattr(self, name)
            if np.any(values < 0) or np.any(values >= limit):
                raise ValueError(f"{name} contains out-of-range indices")

    def converted(self, units):
        """Convert energies, lengths, masses, and all Cartesian derivatives together."""
        if not isinstance(units, UnitSystem):
            raise TypeError("units must be a UnitSystem")
        old = self.unit_system
        length = old.length_bohr/units.length_bohr
        energy = old.energy_hartree/units.energy_hartree
        mass = units.energy_hartree*units.length_bohr**2/(old.energy_hartree*old.length_bohr**2)
        return RealSpaceEPC(
            self.cell*length, self.atom_positions*length, self.masses*mass,
            self.wannier_centers*length, self.hopping_orbitals, self.hopping_cells,
            self.hopping_values*energy, self.epc_channels, self.epc_atoms, self.epc_cells,
            self.epc_values*(energy/length), self.ifc_atoms, self.ifc_cells,
            self.ifc_values*(energy/length**2), units)

    def hermiticity_audit(self):
        """Audit infinite-image stencils before any finite-cell aliasing."""
        hopping, epc, ifc = {}, {}, {}
        for orbitals, cell, value in zip(self.hopping_orbitals, self.hopping_cells,
                                         self.hopping_values, strict=True):
            key = (*orbitals, *cell)
            hopping[key] = hopping.get(key, 0.)+value
        for channel, atom, cell, value in zip(self.epc_channels, self.epc_atoms,
                self.epc_cells, self.epc_values, strict=True):
            key = (*self.hopping_orbitals[channel], *self.hopping_cells[channel], atom, *cell)
            epc[key] = epc.get(key, 0.)+value
        for atoms, cell, value in zip(self.ifc_atoms, self.ifc_cells, self.ifc_values, strict=True):
            key = (*atoms, *cell)
            ifc[key] = ifc.get(key, 0.)+value

        def pair(key):
            return (key[1], key[0], *(-np.asarray(key[2:5])))

        def derivative_pair(key):
            re, rp = np.asarray(key[2:5]), np.asarray(key[6:9])
            return (key[1], key[0], *(-re), key[5], *(rp-re))

        return dict(hopping=_pair_audit(hopping, pair, np.conj),
                    epc=_pair_audit(epc, derivative_pair, np.conj),
                    ifc=_pair_audit(ifc, pair, np.transpose))

    def compile_supercell(self, mesh, *, hermiticity="require", tolerance=1e-10,
                          carrier="electron", term_batch_size=256, real_tolerance=None,
                          epc_backend="direct", max_spectral_bytes=256*1024**2):
        """Compile periodic stencils without replicating derivative coefficients.

        ``require`` rejects non-Hermitian image stencils above the declared
        absolute tolerance. ``project`` explicitly authorizes their Hermitian
        part; the report quantifies that change in the unaliased source data.
        Both routes remove accepted floating-point anti-Hermitian noise.

        A hole has h_hole=-h_electron.T and charge +1. Its neutral reference
        potential is unchanged; the EPR IFCs must describe that reference.

        Complex coefficients are retained by default. ``real_tolerance`` is an
        explicit authorization to discard imaginary h/g coefficients only when
        each maximum is below this numerical absolute bound in its own units
        (energy for h, energy/length for g). The report retains separate absolute
        and relative removed norms. This permits real-Hamiltonian methods only
        when their additional spectral assumptions also hold.

        ``epc_backend='direct'`` retains the bounded term-chunk evaluator.
        ``'fft'`` transforms only the periodic nuclear-cell translation and
        keeps the same original dynamic parameters and electronic image channels.
        ``max_spectral_bytes`` bounds one complex Fourier coefficient grid;
        FFT construction and trajectory executable workspaces are additional.
        """
        if epc_backend == "fft":
            # Preserve original scalar types before _array can coerce a mixed
            # boolean/integer sequence; the FFT model owns an integer mesh.
            try:
                mesh = tuple(integer_scalar(value, "mesh dimension") for value in mesh)
            except TypeError as error:
                raise ValueError("mesh must contain three integer dimensions") from error
        mesh = _array(mesh, "mesh", (3,), integer=True, real=True)
        if np.any(mesh < 1):
            raise ValueError("mesh dimensions must be positive")
        if epc_backend not in ("direct", "fft"):
            raise ValueError("epc_backend must be 'direct' or 'fft'")
        max_spectral_bytes = integer_scalar(max_spectral_bytes, "max_spectral_bytes")
        if max_spectral_bytes < 1:
            raise ValueError("max_spectral_bytes must be positive")
        if hermiticity not in ("require", "project"):
            raise ValueError("hermiticity must be 'require' or 'project'")
        if carrier not in ("electron", "hole"):
            raise ValueError("carrier must be 'electron' or 'hole'")
        if not np.isfinite(tolerance) or tolerance < 0:
            raise ValueError("tolerance must be finite and nonnegative")
        if real_tolerance is not None and (not np.isfinite(real_tolerance) or real_tolerance < 0):
            raise ValueError("real_tolerance must be finite and nonnegative")
        real_audit = {}
        for name, coefficients, units in (("hopping", self.hopping_values, "energy"),
                                           ("epc", self.epc_values, "energy/length")):
            norm = np.linalg.norm(coefficients)
            imaginary_norm = float(np.linalg.norm(np.imag(coefficients)))
            real_audit[name] = dict(units=units,
                maximum_absolute_imaginary=float(np.max(abs(np.imag(coefficients)), initial=0.)),
                absolute_imaginary_l2_norm=imaginary_norm,
                relative_imaginary_l2_norm=float(imaginary_norm/norm) if norm else 0.)
        maximum_imaginary = max(value["maximum_absolute_imaginary"] for value in real_audit.values())
        if real_tolerance is not None and maximum_imaginary > real_tolerance:
            raise ValueError(f"imaginary coefficient {maximum_imaginary} exceeds real_tolerance={real_tolerance}")
        audit = self.hermiticity_audit()
        invalid = [name for name, values in audit.items()
                   if values["maximum_absolute_defect"] > tolerance]
        if invalid and hermiticity == "require":
            raise ValueError("non-Hermitian real-space stencils: "+", ".join(invalid)
                             +"; inspect hermiticity_audit() before explicitly choosing projection")
        cells = np.asarray(list(product(*(range(int(n)) for n in mesh))), dtype=np.int64)
        translations = np.unique(np.concatenate((self.hopping_cells, self.epc_cells,
                                                   self.ifc_cells, np.zeros((1, 3), dtype=int))), axis=0)
        offsets = {tuple(value): index for index, value in enumerate(translations)}
        neighbor_coordinates = (translations[:, None]+cells[None]) % mesh
        neighbors = np.ravel_multi_index(neighbor_coordinates.transpose(2, 0, 1), tuple(mesh))
        nwan, nat = len(self.wannier_centers), len(self.atom_positions)
        model_class = CartesianEPCModel if epc_backend == "direct" else FourierCartesianEPCModel
        options = {} if epc_backend == "direct" else {
            "mesh": tuple(int(value) for value in mesh), "max_spectral_bytes": max_spectral_bytes}
        model = model_class(nwan, nat, len(cells), self.unit_system,
                            1. if carrier == "hole" else -1., term_batch_size,
                            complex_valued=real_tolerance is None, **options)
        nchunks = (len(self.epc_values)+model.term_batch_size-1)//model.term_batch_size
        nterms = nchunks*model.term_batch_size

        def padded(array):
            shape = (nterms, *array.shape[1:])
            result = np.zeros(shape, dtype=array.dtype)
            result[:len(array)] = array
            return result.reshape(nchunks, model.term_batch_size, *array.shape[1:])

        hopping = self.hopping_values
        coupling = self.epc_values
        if carrier == "hole":
            hopping, coupling = -hopping.conj(), -coupling.conj()
        if real_tolerance is not None:
            hopping, coupling = hopping.real, coupling.real
        rows, columns = self.hopping_orbitals.T
        params = dict(
            neighbors=neighbors, hopping_orbitals=self.hopping_orbitals,
            hopping_offsets=np.array([offsets[tuple(r)] for r in self.hopping_cells]),
            hopping=hopping,
            displacements=self.hopping_cells@self.cell+self.wannier_centers[columns]-self.wannier_centers[rows],
            epc_channels=padded(self.epc_channels), epc_atoms=padded(self.epc_atoms),
            epc_offsets=padded(np.array([offsets[tuple(r)] for r in self.epc_cells], dtype=np.int64)),
            epc_values=padded(coupling), ifc_atoms=self.ifc_atoms,
            ifc_offsets=np.array([offsets[tuple(r)] for r in self.ifc_cells], dtype=np.int64),
            ifc_values=self.ifc_values)
        model.validate_params(params)
        params = {key: jnp.asarray(value) for key, value in params.items()}
        masses = jnp.asarray(np.tile(self.masses, len(cells))[:, None])
        positions = (cells[:, None]@self.cell+self.atom_positions).reshape(-1, 3)
        report = dict(mesh=mesh.tolist(), hermiticity_policy=hermiticity,
            hermiticity_tolerance=float(tolerance), hermiticity=audit, carrier=carrier,
            primitive_hopping_channels=len(hopping), primitive_epc_terms=len(coupling),
            primitive_ifc_terms=len(self.ifc_values), ncells=len(cells),
            nstates=model.nstates, cartesian_degrees_of_freedom=positions.size,
            padded_epc_terms=nterms, term_batch_size=model.term_batch_size,
            epc_backend=epc_backend,
            max_spectral_bytes=max_spectral_bytes if epc_backend == "fft" else None,
            parameter_bytes=sum(value.size*value.dtype.itemsize for value in params.values()),
            fixed_wannier_centers=True, unstable_modes_modified=False,
            real_tolerance=real_tolerance, imaginary_coefficients=real_audit,
            imaginary_coefficients_discarded=real_tolerance is not None)
        return CompiledCartesianEPC(model, params, masses, jnp.asarray(positions), report)
