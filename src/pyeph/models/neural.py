"""A dependency-light native JAX residual Hamiltonian for interface validation.

This dense tanh MLP demonstrates a complete differentiable provider. It is
not rotation-equivariant, size-extensive, or a pretrained materials model.
Its zero reference potential makes energy bookkeeping explicit in SumModel.
"""

from dataclasses import dataclass, field
from math import prod

import jax
import jax.numpy as jnp
import numpy as np

from pyeph.core._configuration import boolean_scalar, integer_scalar
from pyeph.core.contracts import ModelSpec
from pyeph.core.system import SystemSpec
from pyeph.models.base import AutoDiffModel


@dataclass(frozen=True, init=False)
class NeuralResidualModel(AutoDiffModel):
    nstates_config: int
    q_shape: tuple
    hidden_sizes: tuple = (16, 16)
    complex_valued: bool = False
    coordinate_kind: str = "cartesian"
    basis_id: str = "fixed"
    spec: ModelSpec = field(init=False)

    def __init__(self, nstates, q_shape, hidden_sizes=(16, 16),
                 complex_valued=False, coordinate_kind="cartesian", basis_id="fixed"):
        for name, value in (("nstates_config", nstates), ("q_shape", q_shape),
                            ("hidden_sizes", hidden_sizes), ("complex_valued", complex_valued),
                            ("coordinate_kind", coordinate_kind), ("basis_id", basis_id)):
            object.__setattr__(self, name, value)
        self.__post_init__()

    def __post_init__(self):
        object.__setattr__(self, "nstates_config", integer_scalar(self.nstates_config, "nstates"))
        object.__setattr__(self, "complex_valued", boolean_scalar(self.complex_valued, "complex_valued"))
        q_shape = tuple(integer_scalar(n, "q_shape dimension") for n in self.q_shape)
        hidden = tuple(integer_scalar(n, "hidden layer size") for n in self.hidden_sizes)
        if any(not isinstance(n, int) or n < 1 for n in hidden):
            raise ValueError("hidden layer sizes must be positive integers")
        object.__setattr__(self, "q_shape", q_shape)
        object.__setattr__(self, "hidden_sizes", hidden)
        object.__setattr__(self, "spec", ModelSpec(
            system=SystemSpec(self.nstates_config, q_shape, basis_id=self.basis_id,
                              coordinate_kind=self.coordinate_kind),
            name="dense_neural_residual", complex_valued=self.complex_valued))

    def init_params(self, key, scale=0.01, zero_last=True):
        """Xavier hidden weights; zero residual initially unless requested."""
        output = self.nstates**2 * (2 if self.complex_valued else 1)
        sizes = (prod(self.q_shape),) + self.hidden_sizes + (output,)
        keys = jax.random.split(key, len(sizes) - 1)
        layers = []
        for index, (a, b, k) in enumerate(zip(sizes[:-1], sizes[1:], keys)):
            w = jax.random.normal(k, (a, b), dtype=jnp.float64) * jnp.sqrt(2.0 / (a + b))
            if index == len(keys) - 1:
                w = jnp.zeros_like(w) if zero_last else scale * w
            layers.append(dict(weight=w, bias=jnp.zeros(b)))
        return dict(layers=tuple(layers), q_center=jnp.zeros(self.q_shape),
                    q_scale=jnp.ones(self.q_shape))

    def dense(self, params, q):
        if params is None:
            raise ValueError("initialize the neural residual with init_params(key)")
        x = ((q - params["q_center"]) / params["q_scale"]).reshape(-1)
        for layer in params["layers"][:-1]:
            x = jnp.tanh(x @ layer["weight"] + layer["bias"])
        last = params["layers"][-1]
        x = x @ last["weight"] + last["bias"]
        n = self.nstates**2
        raw = x[:n].reshape(self.nstates, self.nstates)
        if self.complex_valued:
            raw = raw + 1j * x[n:].reshape(self.nstates, self.nstates)
        return 0.5 * (raw + raw.conj().T)

    def apply(self, params, q, vectors):
        return self.dense(params, q) @ vectors

    def prepare_action(self, params, q):
        # Keep the geometry/parameter graph while sharing network inference
        # across all vectors and columns in one frozen-geometry action.
        matrix = self.dense(params, q)
        return lambda vectors: matrix @ vectors

    def reference_energy(self, params, q):
        return jnp.zeros((), dtype=jnp.asarray(q).dtype)

    def validate_params(self, params):
        if params is None:
            raise ValueError("initialize neural parameters with init_params(key)")
        for key in ("q_center", "q_scale"):
            a = np.asarray(params[key])
            if a.shape != self.q_shape or not np.all(np.isfinite(a)) or np.iscomplexobj(a):
                raise ValueError(f"{key} must be finite and real with shape {self.q_shape}")
        if np.any(np.asarray(params["q_scale"]) <= 0):
            raise ValueError("neural coordinate scales must be positive")
        sizes = (prod(self.q_shape),) + self.hidden_sizes + (self.nstates**2 * (2 if self.complex_valued else 1),)
        if len(params["layers"]) != len(sizes) - 1:
            raise ValueError("neural layer count disagrees with the static architecture")
        for a, b, layer in zip(sizes[:-1], sizes[1:], params["layers"]):
            for key, shape in (("weight", (a, b)), ("bias", (b,))):
                value = np.asarray(layer[key])
                if value.shape != shape or not np.all(np.isfinite(value)) or np.iscomplexobj(value):
                    raise ValueError(f"neural {key} must be finite and real with shape {shape}")
