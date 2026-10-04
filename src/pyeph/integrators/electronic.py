"""Electronic integrators with vector/block actions and explicit numerical policy."""

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from pyeph.core._configuration import integer_scalar, real_scalar
from pyeph.integrators.krylov import LanczosOptions


@dataclass(frozen=True)
class Integrator:
    dt: float
    electronic: str | LanczosOptions = "rk4"
    electronic_substeps: int = 1

    def __post_init__(self):
        object.__setattr__(self, "dt", real_scalar(self.dt, "dt"))
        object.__setattr__(self, "electronic_substeps", integer_scalar(
            self.electronic_substeps, "electronic_substeps"))
        if self.dt <= 0:
            raise ValueError("dt must be finite and positive")
        if not isinstance(self.electronic, LanczosOptions) and (
                not isinstance(self.electronic, str) or
                self.electronic not in {"rk4", "exponential_midpoint"}):
            raise ValueError("electronic integrator must be rk4, exponential_midpoint, "
                             "or LanczosOptions")
        if not isinstance(self.electronic_substeps, int) or self.electronic_substeps < 1:
            raise ValueError("electronic_substeps must be a positive integer")

    @property
    def electronic_name(self):
        """Stable result label; the full numerical options remain in provenance."""
        return "lanczos_midpoint" if isinstance(self.electronic, LanczosOptions) else self.electronic


def rk4_step(apply_at, time, vectors, dt):
    """Fourth-order step; `apply_at(t, vectors)` applies the time-dependent H."""
    def rhs(t, x):
        return -1j * apply_at(t, x)
    k1 = rhs(time, vectors)
    k2 = rhs(time + dt / 2, vectors + dt * k1 / 2)
    k3 = rhs(time + dt / 2, vectors + dt * k2 / 2)
    k4 = rhs(time + dt, vectors + dt * k3)
    return vectors + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6


def exponential_action(hamiltonian, vectors, dt):
    """Exact frozen dense Hermitian evolution; intended for modest state counts."""
    energies, rotation = jnp.linalg.eigh(hamiltonian)
    transformed = rotation.conj().T @ vectors
    phase = jnp.exp(-1j * dt * energies)
    if transformed.ndim == 2:
        phase = phase[:, None]
    return rotation @ (phase * transformed)


def propagate(apply_at, time, vectors, dt, *, algorithm="rk4", substeps=1):
    """Apply a fixed number of electronic substeps without Python time loops."""
    h = dt / substeps

    def advance(i, x):
        t = time + i * h
        if algorithm == "rk4":
            return rk4_step(apply_at, t, x, h)
        if algorithm == "exponential_midpoint":
            matrix = apply_at(t + h / 2, jnp.eye(x.shape[0], dtype=x.dtype))
            return exponential_action(matrix, x, h)
        raise ValueError(f"unknown electronic algorithm {algorithm!r}")

    return jax.lax.fori_loop(0, substeps, advance, vectors)
