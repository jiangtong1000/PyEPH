"""Measurements supply JAX-compatible arrays; the runner handles sampling/output."""

from dataclasses import dataclass

import jax.numpy as jnp


@dataclass(frozen=True)
class ElectronicPopulation:
    """Diabatic populations for amplitudes (not a generic mapping-method estimator)."""

    def validate(self, problem):
        if getattr(problem.method, "uses_mapping_estimator", False):
            raise ValueError("mapping methods require their own population estimator")

    def evaluate(self, problem, state):
        populations = jnp.abs(state.electronic)**2
        return {"population": populations, "norm": jnp.sum(populations, axis=0)}


@dataclass(frozen=True)
class FunctionalMeasurement:
    """An extension point for a pure function f(problem, state) -> array PyTree."""

    function: object
    required_probes: tuple[str, ...] = ()

    def __post_init__(self):
        if isinstance(self.required_probes, str):
            raise ValueError("required_probes must be a collection of names, not one string")
        probes = tuple(self.required_probes)
        if any(not isinstance(p, str) or not p for p in probes):
            raise ValueError("required_probes must contain nonempty strings")
        object.__setattr__(self, "required_probes", probes)

    def validate(self, problem):
        missing = set(self.required_probes) - set(problem.model.spec.probes)
        if missing:
            raise ValueError(f"model does not define required physical probes: {sorted(missing)}")

    def evaluate(self, problem, state):
        return self.function(problem, state)
