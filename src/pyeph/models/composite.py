"""Explicit composition and physically compensated energy-reference shifts."""

from dataclasses import dataclass, field, replace
from typing import Callable

import jax
import jax.numpy as jnp

from pyeph.core.contracts import ModelSpec, contracted_value, prepared_action
from pyeph.core.validation import validate_model_at
from pyeph.models.base import AutoDiffModel


def _complete_force_operations(model):
    return model.spec.force_support and all(callable(getattr(model, name, None))
        for name in ("reference_gradient", "contract_gradient"))


@dataclass(frozen=True)
class SumModel(AutoDiffModel):
    """Sum Hamiltonians *and* reference potentials in a common fixed basis.

    Dynamic params are a tuple with one entry per child. A neural residual
    normally supplies zero reference energy. Probe additivity is explicit:
    position operators, for example, must not be blindly added with models.
    """

    models: tuple
    additive_probes: tuple = ()
    spec: ModelSpec = field(init=False)

    def __post_init__(self):
        models = tuple(self.models)
        if not models:
            raise ValueError("SumModel requires at least one model")
        reference = models[0].spec
        for model in models[1:]:
            if model.spec.system != reference.system:
                raise ValueError("summed models must have identical coordinates and basis identity")
            for key in ("basis_kind", "electronic_sector", "energy_convention", "unit_system"):
                if getattr(model.spec, key) != getattr(reference, key):
                    raise ValueError(f"summed models disagree on {key}")
        probes = tuple(self.additive_probes)
        if any(not all(p in m.spec.probes for m in models) for p in probes):
            raise ValueError("every component must explicitly implement each additive probe")
        object.__setattr__(self, "models", models)
        object.__setattr__(self, "additive_probes", probes)
        object.__setattr__(self, "spec", replace(
            reference, name="sum(" + ",".join(m.spec.name for m in models) + ")",
            complex_valued=any(m.spec.complex_valued for m in models), probes=probes,
            force_support=all(_complete_force_operations(m) for m in models),
            native_jax=all(m.spec.native_jax for m in models)))

    def _params(self, params):
        if params is None:
            return (None,) * len(self.models)
        if not isinstance(params, (tuple, list)) or len(params) != len(self.models):
            raise ValueError("provide exactly one parameter PyTree per component")
        return params

    @property
    def execution_mode(self):
        if all(m.spec.native_jax or getattr(m, "execution_mode", None) == "host_callback"
               for m in self.models):
            return "host_callback" if not self.spec.native_jax else "native_jax"
        return "unsupported"

    def apply(self, params, q, vectors):
        return sum(m.apply(p, q, vectors) for m, p in zip(self.models, self._params(params)))

    def prepare_action(self, params, q):
        actions = tuple(prepared_action(model, p, q)
                        for model, p in zip(self.models, self._params(params)))
        return lambda vectors: sum(action(vectors) for action in actions)

    def reference_energy(self, params, q):
        return sum(m.reference_energy(p, q) for m, p in zip(self.models, self._params(params)))

    def reference_gradient(self, params, q):
        return sum(m.reference_gradient(p, q) for m, p in zip(self.models, self._params(params)))

    def contract_gradient(self, params, q, weight):
        # Each provider owns its differentiation; a callback need not expose a
        # JAX autodiff graph merely to participate in additive composition.
        return sum(m.contract_gradient(p, q, weight)
                   for m, p in zip(self.models, self._params(params)))

    def probe_apply(self, params, context, probe, vectors):
        if probe not in self.spec.probes:
            return super().probe_apply(params, context, probe, vectors)
        return sum(m.probe_apply(p, context, probe, vectors)
                   for m, p in zip(self.models, self._params(params)))

    def validate_params(self, params):
        for model, p in zip(self.models, self._params(params)):
            if hasattr(model, "validate_params"):
                model.validate_params(p)

    def validate_geometry(self, q):
        for model in self.models:
            if hasattr(model, "validate_geometry"):
                model.validate_geometry(q)

    def validate_at(self, params, q, *, batch=False):
        for model, p in zip(self.models, self._params(params)):
            validate_model_at(model, p, q, batch=batch)

    def validate_complete_gradients(self, params, q):
        for model, p in zip(self.models, self._params(params)):
            check = getattr(model, "validate_complete_gradients", None)
            if callable(check):
                check(p, q)
            elif not model.spec.native_jax:
                raise ValueError("external composition component has no derivative preflight")


@dataclass(frozen=True)
class ReferenceShiftModel(AutoDiffModel):
    """h'=h-f(Q)I and V_ref'=V_ref+f(Q), for normalized one-carrier states.

    Params are (base_params, shift_params), and shift_fn(shift_params,q) returns
    a scalar. This controlled transformation preserves forces and total energy;
    dropping its reference compensation does not.
    """

    model: object
    shift_fn: Callable
    spec: ModelSpec = field(init=False)

    def __post_init__(self):
        if self.model.spec.energy_convention != "reference_plus_carrier":
            raise ValueError("reference shift requires a declared reference-plus-carrier energy")
        if not callable(self.shift_fn):
            raise ValueError("shift_fn must be a native JAX real scalar function")
        object.__setattr__(self, "spec", replace(self.model.spec,
            name=f"shift({self.model.spec.name})",
            force_support=_complete_force_operations(self.model)))

    def _shift_value(self, params, q):
        value = jnp.asarray(self.shift_fn(params, q))
        if value.ndim != 0 or jnp.issubdtype(value.dtype, jnp.complexfloating):
            raise ValueError("reference shift must return one real scalar")
        # Preserve an explicit invalid value through compiled evaluation. In
        # particular, an infinite shift must not cancel between h and V_ref.
        return jnp.where(jnp.isfinite(value), value, jnp.nan)

    def apply(self, params, q, vectors):
        base, shift = params
        return self.model.apply(base, q, vectors) - self._shift_value(shift, q) * vectors

    def prepare_action(self, params, q):
        base, shift = params
        action = prepared_action(self.model, base, q)
        value = self._shift_value(shift, q)
        return lambda vectors: action(vectors) - value * vectors

    def reference_energy(self, params, q):
        base, shift = params
        return self.model.reference_energy(base, q) + self._shift_value(shift, q)

    @property
    def execution_mode(self):
        return getattr(self.model, "execution_mode", "native_jax" if self.spec.native_jax else None)

    def reference_gradient(self, params, q):
        base, shift = params
        return self.model.reference_gradient(base, q) + jax.grad(self._shift_value, argnums=1)(shift, q)

    def contract_gradient(self, params, q, weight):
        base, shift = params
        # The shift's partial coordinate derivative holds weight fixed, but
        # an outer force derivative must retain the response of its trace.
        trace = contracted_value(lambda vectors: vectors, weight)
        return (self.model.contract_gradient(base, q, weight)
                - trace * jax.grad(self._shift_value, argnums=1)(shift, q))

    def probe_apply(self, params, context, probe, vectors):
        return self.model.probe_apply(params[0], context, probe, vectors)

    def validate_params(self, params):
        if not isinstance(params, (tuple, list)) or len(params) != 2:
            raise ValueError("reference-shift params are (base_params,shift_params)")
        if hasattr(self.model, "validate_params"):
            self.model.validate_params(params[0])

    def validate_geometry(self, q):
        if hasattr(self.model, "validate_geometry"):
            self.model.validate_geometry(q)

    def validate_at(self, params, q, *, batch=False):
        validate_model_at(self.model, params[0], q, batch=batch)

    def validate_complete_gradients(self, params, q):
        check = getattr(self.model, "validate_complete_gradients", None)
        if callable(check):
            check(params[0], q)
        elif not self.model.spec.native_jax:
            raise ValueError("external reference-shift base has no derivative preflight")
