"""Optional differentiation helpers; custom models need only match the protocols."""

import jax
import jax.numpy as jnp

from pyeph.core.contracts import contracted_value


class AutoDiffModel:
    """Convenience implementation for a fully differentiable fixed-basis model.

    Concrete models provide `spec`, `apply`, and `reference_energy`. This mixin
    neither stores mutable caches nor requires a dense Hamiltonian Jacobian.
    """

    @property
    def nstates(self):
        return self.spec.system.nstates

    def dense(self, params, q):
        return self.apply(params, q, jnp.eye(self.nstates, dtype=jnp.result_type(q, 1.0)))

    def reference_gradient(self, params, q):
        return jax.grad(lambda x: self.reference_energy(params, x))(q)

    def contract_gradient(self, params, q, weight):
        # The inner derivative varies x alone, so its closed-over weight is
        # fixed. Keep the outer graph: force sensitivities must still include
        # the response of a weight that depends on an electronic state/parameter.
        return jax.grad(lambda x: contracted_value(lambda v: self.apply(params, x, v), weight))(q)

    def probe_apply(self, params, context, probe, vectors):
        raise NotImplementedError(f"{self.spec.name} does not define probe {probe!r}")
