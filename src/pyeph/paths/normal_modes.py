"""Prescribed harmonic motion in physical coordinates with general force constants.

The eigensystem diagonalizes M**(-1/2) K M**(-1/2); momenta are canonical,
and all quantities use the model's declared hbar=1 unit system. Electronic
state never enters this bath. Diagonalization is host preparation, while motion
uses JAX matrix/vector operations. Dense normal modes are a finite-system route.
"""

from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np

from pyeph.core._configuration import real_scalar
from pyeph.execution.random import trajectory_keys
from pyeph.initialization import sample_harmonic


def _orthogonality_tolerance(size, dtype):
    """Preparation policy allowing accumulation across a complete mode basis."""
    return max(64., 8*np.sqrt(size))*np.finfo(dtype).eps


@dataclass(frozen=True)
class NormalModeBath:
    """A complete real mass-weighted mode basis in arbitrary coordinate shape.

    Negative squared frequencies are rejected unless their mode indices are
    explicitly frozen. Frozen coordinates retain their incoming modal position
    and require zero modal momentum. Zero modes move freely and have no thermal
    position distribution. ``zero_tolerance`` is an explicit squared-frequency
    tolerance, not an automatic stabilization of the force constants.
    Supplied mode vectors must be orthonormal at the selected JAX precision;
    use ``from_hessian`` to recompute modes from a lower-precision Hessian.
    """

    squared_frequencies: object
    eigenvectors: object
    masses: object
    equilibrium: object
    frozen_modes: tuple = ()
    zero_tolerance: float = 0.0
    orthogonality_error: float = field(init=False)
    prescribed = True

    def __post_init__(self):
        equilibrium = np.asarray(self.equilibrium)
        if equilibrium.ndim < 1 or not equilibrium.size:
            raise ValueError("equilibrium must be a nonempty coordinate array")
        mass = np.broadcast_to(np.asarray(self.masses), equilibrium.shape)
        values, vectors = map(np.asarray, (self.squared_frequencies, self.eigenvectors))
        n = equilibrium.size
        if values.shape != (n,) or vectors.shape != (n, n):
            raise ValueError("a complete mode eigensystem must match the coordinate dimension")
        for value in (equilibrium, mass, values, vectors):
            if np.iscomplexobj(value) or not np.isfinite(value).all():
                raise ValueError("normal-mode data must be finite and real")
        if np.any(mass <= 0):
            raise ValueError("masses must be positive")
        # The stored basis must be orthonormal at execution precision. Using
        # only elementwise or source-dtype tolerances can accept a basis whose
        # own sampled momenta violate its frozen-mode constraints. A row-norm
        # bound controls leakage against the modal momentum's Euclidean norm.
        working_dtype = np.dtype(jnp.result_type(1.))
        stored_vectors = np.asarray(vectors, dtype=working_dtype).astype(np.float64)
        gram_error = stored_vectors.T @ stored_vectors - np.eye(n)
        error = float(np.max(np.linalg.norm(gram_error, axis=1)))
        if not np.isfinite(error) or error > _orthogonality_tolerance(n, working_dtype):
            raise ValueError("mass-weighted eigenvector columns must be orthonormal at the "
                             "selected execution precision; use from_hessian for a "
                             "lower-precision source matrix")
        object.__setattr__(self, "orthogonality_error", error)
        tolerance = real_scalar(self.zero_tolerance, "zero_tolerance")
        if tolerance < 0:
            raise ValueError("zero_tolerance must be nonnegative")
        frozen = tuple(self.frozen_modes)
        if any(isinstance(i, (bool, np.bool_)) or not isinstance(i, (int, np.integer))
               or not 0 <= i < n for i in frozen) or len(set(frozen)) != len(frozen):
            raise ValueError("frozen_modes must be unique in-range integer mode indices")
        frozen = tuple(sorted(int(i) for i in frozen))
        active = np.ones(n, dtype=bool)
        active[list(frozen)] = False
        unstable = np.flatnonzero(active & (values < -tolerance))
        if unstable.size:
            raise ValueError(f"unstable harmonic modes {unstable.tolist()}; supply a stable model "
                             "or explicitly constrain these modes with frozen_modes")
        object.__setattr__(self, "zero_tolerance", tolerance)
        object.__setattr__(self, "frozen_modes", frozen)
        for name, value in (("equilibrium", equilibrium), ("masses", mass),
                            ("squared_frequencies", values), ("eigenvectors", vectors)):
            object.__setattr__(self, name, jnp.array(value, dtype=jnp.result_type(1.), copy=True))

    @classmethod
    def from_hessian(cls, hessian, masses, equilibrium, *, frozen_modes=(), zero_tolerance=0.):
        """Diagonalize a real symmetric Cartesian Hessian without discarding modes."""
        equilibrium = np.asarray(equilibrium)
        mass = np.broadcast_to(np.asarray(masses), equilibrium.shape).reshape(-1)
        hessian = np.asarray(hessian)
        if (hessian.shape != (mass.size, mass.size) or np.iscomplexobj(hessian)
                or not np.isfinite(hessian).all()
                or not np.allclose(hessian, hessian.T, atol=1e-12, rtol=1e-10)):
            raise ValueError("hessian must be a finite real symmetric coordinate matrix")
        if np.iscomplexobj(mass) or not np.isfinite(mass).all() or np.any(mass <= 0):
            raise ValueError("masses must be finite, real and positive")
        # Host preparation uses double precision even when the source arrays
        # are float32; execution still follows the caller's JAX precision.
        inverse_root = 1 / np.sqrt(mass.astype(np.float64))
        symmetric_hessian = (hessian.astype(np.float64)+hessian.T)/2
        dynamical = inverse_root[:, None] * symmetric_hessian * inverse_root[None, :]
        values, vectors = np.linalg.eigh(dynamical)
        return cls(values, vectors, masses, equilibrium, tuple(frozen_modes), zero_tolerance)

    @property
    def orthogonality_tolerance(self):
        """Allowed maximum row norm of the stored basis's Gram error."""
        return _orthogonality_tolerance(self.equilibrium.size, self.eigenvectors.dtype)

    @property
    def frequencies(self):
        return jnp.sqrt(jnp.where(self.squared_frequencies > self.zero_tolerance,
                                  self.squared_frequencies, 0.))

    def validate(self, q_shape):
        if tuple(q_shape) != self.equilibrium.shape:
            raise ValueError("normal-mode bath and model coordinate shapes differ")

    def to_modes(self, q, p):
        """Physical coordinates/canonical momenta to unit-mass normal coordinates."""
        root_mass = jnp.sqrt(self.masses).reshape(-1)
        displacement = (q-self.equilibrium).reshape(-1)
        return (self.eigenvectors.T @ (root_mass*displacement),
                self.eigenvectors.T @ (p.reshape(-1)/root_mass))

    def from_modes(self, q, p):
        root_mass = jnp.sqrt(self.masses).reshape(-1)
        return (self.equilibrium + ((self.eigenvectors @ q)/root_mass).reshape(self.equilibrium.shape),
                (root_mass*(self.eigenvectors @ p)).reshape(self.equilibrium.shape))

    def point(self, state, elapsed):
        q, p = self.to_modes(state.q, state.p)
        w = self.frequencies
        phase = w*elapsed
        advanced_q = q*jnp.cos(phase) + p*elapsed*jnp.sinc(phase/jnp.pi)
        advanced_p = p*jnp.cos(phase) - w*q*jnp.sin(phase)
        if self.frozen_modes:
            indices = jnp.asarray(self.frozen_modes)
            advanced_q = advanced_q.at[indices].set(q[indices])
            advanced_p = advanced_p.at[indices].set(0.)
        return self.from_modes(advanced_q, advanced_p)

    def validate_initial_state(self, state, *, batch=False):
        """Reject momentum incompatible with declared frozen-mode constraints."""
        if not self.frozen_modes:
            return
        momentum = np.asarray(state.p)
        momentum = momentum.reshape((-1, self.equilibrium.size)) if batch else momentum.reshape(1, -1)
        modes = (momentum/np.sqrt(np.asarray(self.masses)).reshape(-1)) @ np.asarray(self.eigenvectors)
        tolerance = (4*_orthogonality_tolerance(self.equilibrium.size, momentum.dtype)
                     * np.maximum(1., np.linalg.norm(modes, axis=1, keepdims=True)))
        if np.any(np.abs(modes[:, self.frozen_modes]) > tolerance):
            raise ValueError("frozen normal modes require zero initial modal momentum")


def sample_normal_modes(bath, temperature, trajectory_ids, *, seed=0,
                        distribution="classical", free_positions=None):
    """Sample positive-frequency modes; explicitly condition any free positions.

    Frozen modal coordinates/momenta are zero relative to ``equilibrium``.
    ``free_positions`` must supply one position per active zero mode, optionally
    with a leading trajectory axis. Free momenta follow Maxwell statistics;
    this does not assert a normalizable free-particle position ensemble.
    """
    if not isinstance(bath, NormalModeBath):
        raise TypeError("bath must be a NormalModeBath")
    keys = trajectory_keys(seed, trajectory_ids, typed=True)
    count, n = len(keys), len(bath.squared_frequencies)
    active = np.ones(n, dtype=bool)
    active[list(bath.frozen_modes)] = False
    frequencies = np.asarray(bath.frequencies)
    positive = np.flatnonzero(active & (frequencies > 0))
    free = np.flatnonzero(active & (frequencies == 0))
    q, p = jnp.zeros((count, n)), jnp.zeros((count, n))
    thermal_q, thermal_p = sample_harmonic(frequencies[positive], 1., temperature,
                                          trajectory_ids, seed=seed, distribution=distribution)
    q, p = q.at[:, positive].set(thermal_q), p.at[:, positive].set(thermal_p)
    if free.size:
        if free_positions is None:
            raise ValueError("active zero modes require explicit free_positions")
        positions = np.asarray(free_positions)
        if np.iscomplexobj(positions) or not np.isfinite(positions).all():
            raise ValueError("free_positions must be finite and real")
        positions = np.broadcast_to(positions, (count, len(free)))
        free_keys = jax.vmap(lambda key: jax.random.fold_in(key, 0x46524545))(keys)
        momentum = jax.vmap(lambda key: jax.random.normal(key, (len(free),), dtype=p.dtype))(free_keys)
        q = q.at[:, free].set(jnp.asarray(positions))
        p = p.at[:, free].set(momentum*jnp.sqrt(temperature))
    elif free_positions is not None:
        raise ValueError("free_positions supplied but the bath has no active zero modes")
    return jax.vmap(bath.from_modes)(q, p)
