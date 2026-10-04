"""Periodic harmonic motion without a dense supercell mode matrix.

The real-space Hessian stencil is K[(0,a),(R,b)]. Its cell Fourier transform
uses exp(+2 pi i k.R), consistent with NumPy/JAX's negative-phase forward FFT.
Only primitive-cell dynamical matrices are diagonalized on the host. Nuclear
states remain real Cartesian displacements and canonical momenta.
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from pyeph.core._configuration import integer_scalar, real_scalar
from pyeph.execution.random import trajectory_keys


@dataclass(frozen=True, init=False)
class PeriodicHarmonicBath:
    """Translation-invariant harmonic bath with cell-major Cartesian states.

    ``mesh`` has three positive cell counts, with z fastest in flattened
    states. ``masses`` contains one positive mass per primitive-cell atom.
    Full directed ``ifc_atoms, ifc_cells, ifc_values`` specify real Cartesian
    Hessian blocks with shapes (terms,2), (terms,3), and (terms,3,3).

    Unstable modes are rejected by default. ``frozen_below=w`` explicitly
    constrains every mode with squared frequency <= w**2, including unstable
    modes. Their initial modal positions stay fixed and their momenta must be
    zero. The unmodified signed spectrum remains available for auditing.
    ``zero_tolerance`` is in squared-frequency units; active numerical zeros
    move freely. No stabilization or acoustic sum-rule correction is implicit.
    """

    mesh: tuple
    masses: object
    equilibrium: object
    squared_frequencies: object
    eigenvectors: object
    frozen_modes: object
    frozen_below: float | None
    zero_tolerance: float
    prescribed = True

    def __init__(self, mesh, masses, ifc_atoms, ifc_cells, ifc_values, *,
                 equilibrium=None, frozen_below=None, zero_tolerance=0.):
        mesh = tuple(integer_scalar(n, "mesh cell count") for n in mesh)
        if len(mesh) != 3 or min(mesh) < 1:
            raise ValueError("mesh must have three positive integer cell counts")
        mass = np.asarray(masses)
        if (mass.ndim != 1 or not mass.size or np.iscomplexobj(mass)
                or not np.isfinite(mass).all() or np.any(mass <= 0)):
            raise ValueError("masses must be a nonempty positive finite real atom array")
        q_shape = (int(np.prod(mesh))*len(mass), 3)
        equilibrium = np.zeros(q_shape) if equilibrium is None else np.asarray(equilibrium)
        if (equilibrium.shape != q_shape or np.iscomplexobj(equilibrium)
                or not np.isfinite(equilibrium).all()):
            raise ValueError(f"equilibrium must be a finite real array with shape {q_shape}")
        tolerance = real_scalar(zero_tolerance, "zero_tolerance")
        if tolerance < 0:
            raise ValueError("zero_tolerance must be nonnegative")
        if frozen_below is not None:
            frozen_below = real_scalar(frozen_below, "frozen_below")
            if frozen_below < 0:
                raise ValueError("frozen_below must be nonnegative")
        dynamical = _dynamical_matrices(mesh, mass, ifc_atoms, ifc_cells, ifc_values)
        values, vectors = np.linalg.eigh(dynamical)
        frozen = (np.zeros(values.shape, bool) if frozen_below is None
                  else values <= frozen_below**2)
        unstable = np.argwhere((values < -tolerance) & ~frozen)
        if len(unstable):
            raise ValueError(f"{len(unstable)} unstable periodic harmonic modes; "
                             "supply stable force constants or explicitly set frozen_below")
        # Thresholds near a roundoff-split degeneracy can otherwise give the
        # two conjugate wavevectors different constrained subspaces.
        _validate_real_projector(vectors, frozen, "frozen_below")
        _validate_real_projector(vectors, (values <= tolerance) & ~frozen, "zero_tolerance")
        object.__setattr__(self, "mesh", mesh)
        object.__setattr__(self, "frozen_below", frozen_below)
        object.__setattr__(self, "zero_tolerance", tolerance)
        arrays = {"masses": np.tile(mass, int(np.prod(mesh)))[:, None],
                  "equilibrium": equilibrium, "squared_frequencies": values,
                  "eigenvectors": vectors, "frozen_modes": frozen}
        for name, value in arrays.items():
            dtype = (bool if name == "frozen_modes" else
                     jnp.result_type(1j) if name == "eigenvectors" else jnp.result_type(1.))
            object.__setattr__(self, name, jnp.array(value, dtype=dtype, copy=True))

    @property
    def frequencies(self):
        return jnp.sqrt(jnp.where(self.squared_frequencies > self.zero_tolerance,
                                  self.squared_frequencies, 0.))

    def validate(self, q_shape):
        if tuple(q_shape) != self.equilibrium.shape:
            raise ValueError("periodic harmonic bath and model coordinate shapes differ")

    def _to_modes(self, value):
        field = value.reshape(self.mesh + (-1,))
        fourier = jnp.fft.fftn(field, axes=(0, 1, 2), norm="ortho")
        return jnp.einsum("...im,...i->...m", self.eigenvectors.conj(), fourier)

    def _from_modes(self, value):
        fourier = jnp.einsum("...im,...m->...i", self.eigenvectors, value)
        # Real source IFCs and conjugate-compatible spectral functions preserve
        # a real field; the imaginary remainder is only FFT/eigensolver roundoff.
        field = jnp.fft.ifftn(fourier, axes=(0, 1, 2), norm="ortho").real
        return field.reshape(self.equilibrium.shape)

    def point(self, state, elapsed):
        root = jnp.sqrt(self.masses)
        # Transform q and p together so the same mode matrices are reused by
        # a wider matrix product and the FFTs share one batched operation.
        q, p = jax.vmap(self._to_modes)(
            jnp.stack(((state.q-self.equilibrium)*root, state.p/root)))
        w = self.frequencies
        phase = w*elapsed
        advanced_q = q*jnp.cos(phase) + p*elapsed*jnp.sinc(phase/jnp.pi)
        advanced_p = p*jnp.cos(phase) - w*q*jnp.sin(phase)
        advanced_q = jnp.where(self.frozen_modes, q, advanced_q)
        advanced_p = jnp.where(self.frozen_modes, 0., advanced_p)
        physical = jax.vmap(self._from_modes)(jnp.stack((advanced_q, advanced_p)))
        return self.equilibrium+physical[0]/root, physical[1]*root

    def validate_initial_state(self, state, *, batch=False):
        if not np.any(self.frozen_modes):
            return
        momenta = state.p if batch else state.p[None, ...]
        modes = np.asarray(jax.vmap(self._to_modes)(momenta/jnp.sqrt(self.masses)))
        scale = np.maximum(1., np.max(abs(modes).reshape(len(modes), -1), axis=1))
        tolerance = 128*np.finfo(momenta.dtype).eps*scale
        residual = np.max(np.abs(modes[:, np.asarray(self.frozen_modes)]), axis=1)
        if np.any(residual > tolerance):
            raise ValueError("frozen periodic harmonic modes require zero initial modal momentum")


def _dynamical_matrices(mesh, masses, ifc_atoms, ifc_cells, ifc_values):
    atoms, cells, blocks = map(np.asarray, (ifc_atoms, ifc_cells, ifc_values))
    count, natoms = len(blocks), len(masses)
    if atoms.shape != (count, 2) or cells.shape != (count, 3) or blocks.shape != (count, 3, 3):
        raise ValueError("IFC atoms, cells, values must have shapes (terms,2), (terms,3), (terms,3,3)")
    for indices in (atoms, cells):
        if indices.dtype.kind not in "iu":
            raise ValueError("IFC atom and cell indices must contain exact integers")
    if np.any(atoms < 0) or np.any(atoms >= natoms):
        raise ValueError("IFC atom index outside primitive cell")
    if np.iscomplexobj(blocks) or not np.isfinite(blocks).all():
        raise ValueError("IFC blocks must be finite and real")
    records = {}
    for (a, b), cell, block in zip(atoms, cells, blocks):
        key = (int(a), int(b), *map(int, cell))
        records[key] = records.get(key, 0.)+block
    for (a, b, x, y, z), block in records.items():
        reverse = records.get((b, a, -x, -y, -z), np.zeros((3, 3)))
        if not np.allclose(block, reverse.T, atol=1e-12, rtol=1e-10):
            raise ValueError("IFC stencil must include transposed reverse-cell blocks")
    # After validating unwrapped image pairs, fold the stencil onto the finite
    # mesh once. The positive-phase transform is IFFT times the cell count;
    # this avoids recomputing a phase array for every atomic IFC block.
    real_space = np.zeros(mesh+(3*natoms, 3*natoms), dtype=np.float64)
    wrapped = cells % mesh
    rows = 3*atoms[:, 0, None, None]+np.arange(3)[None, :, None]
    columns = 3*atoms[:, 1, None, None]+np.arange(3)[None, None, :]
    inverse_root = 1/np.sqrt(np.asarray(masses, dtype=np.float64))
    weighted = (blocks*inverse_root[atoms[:, 0], None, None]
                * inverse_root[atoms[:, 1], None, None])
    indices = tuple(wrapped[:, axis, None, None] for axis in range(3))+(rows, columns)
    np.add.at(real_space, indices, weighted)
    dynamical = np.fft.ifftn(real_space, axes=(0, 1, 2))*np.prod(mesh)
    # Only accepted source roundoff is removed; no force-constant policy is
    # inferred from a finite mesh or an electronic EPC projection setting.
    return (dynamical+dynamical.conj().swapaxes(-1, -2))/2


def _validate_real_projector(vectors, mask, threshold_name):
    if not np.any(mask) or np.all(mask):
        return
    mesh = mask.shape[:3]
    for index in np.ndindex(mesh):
        partner = tuple((-i) % n for i, n in zip(index, mesh))
        if index > partner:
            continue
        modes = vectors[index][:, mask[index]]
        other = vectors[partner][:, mask[partner]]
        if not np.allclose(modes @ modes.conj().T, (other @ other.conj().T).conj(),
                           atol=1e-10, rtol=1e-10):
            raise ValueError(f"{threshold_name} splits a conjugate mode subspace; move the "
                             "threshold away from the degenerate frequency")


def sample_periodic_harmonic(bath, temperature, trajectory_ids, *, seed=0,
                             distribution="classical", free_positions=None):
    """Sample real Cartesian fields with thermal mode covariance and stable IDs.

    Real white-noise fields automatically supply the paired Fourier statistics.
    Positive modes follow classical or Wigner statistics. Frozen displacements
    and momenta are zero around equilibrium. Active zero modes have Maxwell
    momenta and require explicit physical ``free_positions`` displacements
    lying entirely in the free-mode subspace, shaped like one state or a batch.
    """
    if not isinstance(bath, PeriodicHarmonicBath):
        raise TypeError("bath must be a PeriodicHarmonicBath")
    temperature = real_scalar(temperature, "temperature")
    if temperature < 0 or distribution not in {"classical", "wigner"}:
        raise ValueError("temperature must be nonnegative and distribution classical or wigner")
    keys = trajectory_keys(seed, trajectory_ids, typed=True)
    w, frozen = np.asarray(bath.frequencies), np.asarray(bath.frozen_modes)
    free = (w == 0) & ~frozen
    positive = (w > 0) & ~frozen
    safe_w = np.where(positive, w, 1.)
    if distribution == "classical":
        qvar, pvar = temperature/safe_w**2, np.full(w.shape, temperature)
    else:
        occupation = np.ones(w.shape) if temperature == 0 else 1/np.tanh(safe_w/(2*temperature))
        qvar, pvar = occupation/(2*safe_w), occupation*safe_w/2
        pvar = np.where(free, temperature, pvar)
    qscale = jnp.asarray(np.sqrt(np.where(positive, qvar, 0.)))
    pscale = jnp.asarray(np.sqrt(np.where(frozen, 0., pvar)))
    free_fields = jnp.zeros((len(keys),)+bath.equilibrium.shape)
    if np.any(free):
        if free_positions is None:
            raise ValueError("active zero modes require explicit free_positions")
        positions = np.asarray(free_positions)
        if np.iscomplexobj(positions) or not np.isfinite(positions).all():
            raise ValueError("free_positions must be finite and real")
        positions = np.broadcast_to(positions, free_fields.shape)
        modal = jax.vmap(bath._to_modes)(jnp.asarray(positions)*jnp.sqrt(bath.masses))
        projected = jax.vmap(bath._from_modes)(modal*jnp.asarray(free))/jnp.sqrt(bath.masses)
        if not np.allclose(positions, np.asarray(projected), atol=1e-10, rtol=1e-10):
            raise ValueError("free_positions must lie in the active zero-mode subspace")
        free_fields = jnp.asarray(positions)
    elif free_positions is not None:
        raise ValueError("free_positions supplied but the bath has no active zero modes")

    def one(key, free_field):
        qkey, pkey = jax.random.split(key)
        shape, dtype = bath.equilibrium.shape, bath.equilibrium.dtype
        qnoise = jax.random.normal(qkey, shape, dtype=dtype)
        pnoise = jax.random.normal(pkey, shape, dtype=dtype)
        q = bath._from_modes(bath._to_modes(qnoise)*qscale)
        p = bath._from_modes(bath._to_modes(pnoise)*pscale)
        return (bath.equilibrium+free_field+q/jnp.sqrt(bath.masses),
                p*jnp.sqrt(bath.masses))

    return jax.vmap(one)(keys, free_fields)
