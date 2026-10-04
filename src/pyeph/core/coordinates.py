"""Linear canonical coordinate maps with explicit gradient pullbacks."""

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np


@dataclass(frozen=True)
class MassWeightedModes:
    """R = R0 + M^(-1/2) L Q for orthonormal columns of L.

    Q and its conjugate P have unit masses. The columns may span a retained
    subspace; projection then drops motion orthogonal to that subspace explicitly.
    Cartesian atom masses have shape (natoms,), mode_vectors (3*natoms,nmodes).
    """

    reference: object
    masses: object
    mode_vectors: object

    def __post_init__(self):
        r, m, modes = map(np.asarray, (self.reference, self.masses, self.mode_vectors))
        if any(np.iscomplexobj(x) for x in (r, m, modes)):
            raise ValueError("canonical coordinate maps require real geometry, masses and modes")
        if r.ndim != 2 or r.shape[1] != 3 or m.shape != (r.shape[0],):
            raise ValueError("reference must be (atoms,3) and masses (atoms,)")
        if modes.ndim != 2 or modes.shape[0] != r.size or modes.shape[1] < 1:
            raise ValueError("mode vectors must have shape (3*atoms,nmodes)")
        if any(not np.isfinite(x).all() for x in (r, m, modes)) or np.any(m <= 0):
            raise ValueError("coordinate data must be finite and masses positive")
        if not np.allclose(modes.T @ modes, np.eye(modes.shape[1]), atol=1e-10, rtol=1e-10):
            raise ValueError("mass-weighted mode columns must be orthonormal")
        for name in ("reference", "masses", "mode_vectors"):
            object.__setattr__(self, name, jnp.array(getattr(self, name), copy=True))

    @property
    def nmodes(self):
        return self.mode_vectors.shape[1]

    def to_cartesian(self, q):
        displacement = self.mode_vectors @ q
        return self.reference + displacement.reshape(self.reference.shape) / jnp.sqrt(self.masses[:, None])

    def to_modes(self, positions):
        weighted = (positions - self.reference) * jnp.sqrt(self.masses[:, None])
        return self.mode_vectors.T @ weighted.reshape(-1)

    def velocity(self, p):
        return (self.mode_vectors @ p).reshape(self.reference.shape) / jnp.sqrt(self.masses[:, None])

    def momentum(self, cartesian_momentum):
        return self.mode_vectors.T @ (cartesian_momentum / jnp.sqrt(self.masses[:, None])).reshape(-1)

    def pullback_gradient(self, cartesian_gradient):
        return self.mode_vectors.T @ (cartesian_gradient / jnp.sqrt(self.masses[:, None])).reshape(-1)
