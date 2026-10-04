"""Decode PERTURBO EPR files into format-independent Cartesian EPC stencils.

The file convention is <0,i|dV(Rp,a)|Re,j>. Ordered orbital pairs have their
own electronic Wigner-Seitz cells. Stored coefficients already include the
WS degeneracy factors; neither another division nor a lower-pair conjugation
belongs in this reader. Complex values and unstable IFCs are retained.
"""

from dataclasses import dataclass, replace
import hashlib
from pathlib import Path

import h5py
import numpy as np

from pyeph.adapters.real_space_epc import RealSpaceEPC
from pyeph.core.units import UnitSystem
from pyeph.post_qe2pert.post_qe2pert import PostQE2Pert


PERTURBO_CONVENTION_COMMIT = "0f993052cf57d5b57d8df83452a0cebef357c823"


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _complex_dataset(handle, real_name, imaginary_name, shape, error):
    real, imaginary = handle[real_name][:], handle[imaginary_name][:]
    if (real.shape != shape or imaginary.shape != shape
            or np.iscomplexobj(real) or np.iscomplexobj(imaginary)):
        raise ValueError(error)
    return real+1j*imaginary


@dataclass(frozen=True)
class EPRDataset:
    """Decoded source stencils and evidence for the EPR convention choices."""

    data: RealSpaceEPC
    metadata: dict

    def compile_supercell(self, mesh, **options):
        result = self.data.compile_supercell(mesh, **options)
        return replace(result, report={**result.report, "source": dict(self.metadata)})


def read_epr(path, *, polar="error"):
    """Read raw Cartesian EPR data in Rydberg/Bohr reduced units.

    ``polar='short_range'`` explicitly accepts omission of stored long-range
    polar terms. With the default ``'error'``, a polar source is rejected so a
    partial polar Hamiltonian cannot silently become a materials model.
    Long-range phonon and electron-phonon corrections are not added here.

    EPR masses are Rydberg masses (physical masses divided by 2 electron
    masses). With UnitSystem(energy_hartree=.5, length_bohr=1), these are exactly
    the canonical masses used by pyeph; no extra factor or oscillator scaling
    is applied. IFCs have units Ry/Bohr**2, derivatives Ry/Bohr, hoppings Ry.
    """
    path = Path(path)
    if polar not in ("error", "short_range"):
        raise ValueError("polar must be 'error' or 'short_range'")
    identity = _sha256(path)
    with h5py.File(path, "r") as handle:
        source_polar = bool(handle["basic_data/lpolar"][()])
        spinor = bool(handle["basic_data/spinor"][()])
        if source_polar and polar == "error":
            raise ValueError("polar EPR source requires explicit polar='short_range'; "
                             "long-range corrections are not implemented by this reader")

    geometry = PostQE2Pert(path)
    for name in ("nat", "num_wann"):
        value = getattr(geometry, name)
        if np.ndim(value) or int(value) != value or value < 1:
            raise ValueError(f"EPR {name} must be a positive integer")
    for name in ("kc_dim", "qc_dim"):
        value = np.asarray(getattr(geometry, name))
        if value.shape != (3,) or np.any(value < 1) or np.any(value != np.round(value)):
            raise ValueError(f"EPR {name} must contain three positive integers")
    if (geometry.at.shape != (3, 3) or not np.isfinite(geometry.at).all()
            or abs(np.linalg.det(geometry.at)) < 1e-12
            or not np.isfinite(geometry.alat) or geometry.alat <= 0):
        raise ValueError("EPR lattice must be finite and nonsingular")
    nat, nwan = int(geometry.nat), int(geometry.num_wann)
    atom_fractional = np.linalg.solve(geometry.at.T, geometry.tau.T).T
    electronic_images = geometry.init_rvec_images(kdim=geometry.kc_dim)
    phonon_images = geometry.init_rvec_images(kdim=geometry.qc_dim)

    def ws(images, mesh, first, second):
        indices, _ = geometry.set_wigner_seitz_cell(mesh, images, first, second)
        return images["vec_cryst"][indices]

    electronic_cells = {(i, j): ws(electronic_images, geometry.kc_dim,
        geometry.wannier_center_cryst[i], geometry.wannier_center_cryst[j])
        for i in range(nwan) for j in range(nwan)}
    phonon_cells = {(i, a): ws(phonon_images, geometry.qc_dim,
        geometry.wannier_center_cryst[i], atom_fractional[a])
        for i in range(nwan) for a in range(nat)}

    channel_lookup, hopping_orbitals, hopping_cells, hopping_values = {}, [], [], []

    def channel(i, j, cell):
        key = (i, j, *cell)
        if key not in channel_lookup:
            channel_lookup[key] = len(hopping_values)
            hopping_orbitals.append((i, j))
            hopping_cells.append(cell)
            hopping_values.append(0.j)
        return channel_lookup[key]

    epc_channels, epc_atoms, epc_cells, epc_values = [], [], [], []
    ifc_atoms, ifc_cells, ifc_values = [], [], []
    with h5py.File(path, "r") as handle:
        for j in range(nwan):
            for i in range(j+1):
                label = j*(j+1)//2+i+1
                cells = electronic_cells[i, j]
                values = _complex_dataset(handle, f"electron_wannier/hopping_r{label}",
                    f"electron_wannier/hopping_i{label}", (len(cells),),
                    f"EPR hopping {label} does not match its Wigner-Seitz cell")
                for cell, value in zip(cells, values, strict=True):
                    hopping_values[channel(i, j, cell)] += value
                    if i != j:
                        hopping_values[channel(j, i, -cell)] += value.conjugate()

        for j in range(nwan):
            for i in range(nwan):
                re = electronic_cells[i, j]
                channels = np.array([channel(i, j, cell) for cell in re], dtype=np.int64)
                for atom in range(nat):
                    rp = phonon_cells[i, atom]
                    label = f"{atom+1}_{j+1}_{i+1}"
                    values = _complex_dataset(handle, f"eph_matrix_wannier/ep_hop_r_{label}",
                        f"eph_matrix_wannier/ep_hop_i_{label}", (len(rp), len(re), 3),
                        f"EPR derivative {label} does not match its ordered Wigner-Seitz cells")
                    count = len(rp)*len(re)
                    epc_channels.append(np.tile(channels, len(rp)))
                    epc_atoms.append(np.full(count, atom, dtype=np.int64))
                    epc_cells.append(np.repeat(rp, len(re), axis=0))
                    epc_values.append(values.reshape(count, 3))

        for j in range(nat):
            for i in range(j+1):
                label = j*(j+1)//2+i+1
                cells = ws(phonon_images, geometry.qc_dim, atom_fractional[i], atom_fractional[j])
                # HDF5 exposes Fortran's two Cartesian axes in reverse order.
                values = handle[f"force_constant/ifc{label}"][:].swapaxes(-1, -2)
                if values.shape != (len(cells), 3, 3):
                    raise ValueError(f"EPR IFC {label} does not match its Wigner-Seitz cell")
                ifc_atoms.append(np.tile((i, j), (len(cells), 1)))
                ifc_cells.append(cells)
                ifc_values.append(values)
                if i != j:
                    ifc_atoms.append(np.tile((j, i), (len(cells), 1)))
                    ifc_cells.append(-cells)
                    ifc_values.append(values.swapaxes(-1, -2))

    if _sha256(path) != identity:
        raise RuntimeError("EPR input changed while it was being read")
    data = RealSpaceEPC(
        geometry.at*geometry.alat, geometry.tau*geometry.alat, geometry.mass,
        geometry.wannier_center_cryst@geometry.at*geometry.alat,
        hopping_orbitals, hopping_cells, hopping_values,
        np.concatenate(epc_channels), np.concatenate(epc_atoms),
        np.concatenate(epc_cells), np.concatenate(epc_values),
        np.concatenate(ifc_atoms), np.concatenate(ifc_cells), np.concatenate(ifc_values),
        UnitSystem(energy_hartree=.5, length_bohr=1.))
    metadata = dict(source_file=path.name, source_sha256=identity,
        source_format="PERTURBO EPR", convention_reference_commit=PERTURBO_CONVENTION_COMMIT,
        kc_dim=np.asarray(geometry.kc_dim).tolist(), qc_dim=np.asarray(geometry.qc_dim).tolist(),
        nat=nat, num_wann=nwan, spinor=spinor, source_polar=source_polar,
        polar_policy=polar, long_range_included=False,
        energies="Ry", lengths="Bohr", masses="2 electron masses",
        derivative="Ry/Bohr", force_constant="Ry/Bohr^2",
        complex_values_retained=True, oscillator_normalization_applied=False,
        unstable_modes_modified=False, ws_degeneracy_already_in_source=True)
    return EPRDataset(data, metadata)
