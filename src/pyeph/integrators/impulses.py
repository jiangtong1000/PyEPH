"""Canonical momentum impulses for a resolved step in an active potential."""

from typing import Any, NamedTuple

import jax.numpy as jnp


class MomentumImpulse(NamedTuple):
    momentum: Any
    accepted: Any
    valid: Any
    energy_error: Any
    parallel_before: Any
    parallel_after: Any


def energy_conserving_impulse(momentum, masses, direction, energy_change, *,
                              direction_tolerance=1e-14):
    """Rescale or reflect along ``direction/sqrt(masses)`` in weighted space.

    The dynamics method supplies the physical, mass-unweighted direction and
    final-minus-initial potential difference. Only the parallel component of
    canonical momentum changes. A zero incident component is unresolved;
    this kernel does not invent a downhill direction at a grazing event.
    The caller must separately validate its event and the outgoing direction.
    """
    p, m, d = jnp.asarray(momentum), jnp.asarray(masses), jnp.asarray(direction)
    if any(jnp.issubdtype(x.dtype, jnp.complexfloating) for x in (p, m, d)):
        raise ValueError("surface impulses require real momentum, masses and direction")
    m = jnp.broadcast_to(m, p.shape)
    mass_valid = jnp.all(jnp.isfinite(m) & (m > 0))
    sqrt_mass = jnp.sqrt(jnp.where(m > 0, m, 1.0))
    weighted_d = d / sqrt_mass
    length = jnp.linalg.norm(weighted_d)
    unit = weighted_d / jnp.where(length > 0, length, 1.0)
    parallel = jnp.sum(p / sqrt_mass * unit)
    valid = (mass_valid & jnp.all(jnp.isfinite(p)) & jnp.all(jnp.isfinite(d))
             & jnp.isfinite(energy_change) & (length > direction_tolerance)
             & (jnp.abs(parallel) > direction_tolerance))
    radicand = parallel**2 - 2 * energy_change
    accepted = radicand >= 0
    outgoing = jnp.where(accepted, jnp.sign(parallel) * jnp.sqrt(jnp.maximum(radicand, 0)),
                         -parallel)
    updated = p + sqrt_mass * (outgoing - parallel) * unit
    error = (jnp.sum((updated**2 - p**2) / (2 * jnp.where(m > 0, m, 1.0)))
             + jnp.where(accepted, energy_change, 0))
    valid = (valid & jnp.isfinite(length) & jnp.isfinite(parallel)
             & jnp.isfinite(outgoing) & jnp.all(jnp.isfinite(updated)) & jnp.isfinite(error))
    return MomentumImpulse(jnp.where(valid, updated, p), accepted & valid, valid,
                           jnp.where(valid, error, jnp.nan), parallel, outgoing)
