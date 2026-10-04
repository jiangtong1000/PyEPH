"""Public LF dressing preserves sparse actions and Hamiltonian override semantics."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pyeph.core.contracts import prepared_action
from pyeph.models.epc import EdgeEPCModel, LinearEPCModel
from pyeph.workflows.polaron_transport import PolaronDressedModel


def fixture(model_type=EdgeEPCModel):
    model = model_type(5, 3, ((0, 1), (0, 4), (1, 3), (2, 4)), complex_valued=True)
    rng = np.random.default_rng(3911)
    params = model.default_params()
    params.update(onsite=jnp.asarray(rng.normal(size=5)),
                  hopping=jnp.asarray(rng.normal(size=4) + 1j*rng.normal(size=4)),
                  onsite_coupling=jnp.asarray(rng.normal(size=(3, 5))),
                  hopping_coupling=jnp.asarray(rng.normal(size=(3, 4)) + 1j*rng.normal(size=(3, 4))))
    q = jnp.array([.2, -.31, .17])
    vectors = rng.normal(size=(5, 2)) + 1j*rng.normal(size=(5, 2))
    return model, params, q, jnp.asarray(vectors)


def matrix(model, params, q):
    h = np.diag(np.asarray(params["onsite"]) + np.asarray(q) @ np.asarray(params["onsite_coupling"])).astype(complex)
    values = np.asarray(params["hopping"]) + np.asarray(q) @ np.asarray(params["hopping_coupling"])
    for (i, j), value in zip(model.edges, values, strict=True):
        h[i, j], h[j, i] = value, value.conjugate()
    return h


def narrowed(h, factor):
    result = factor*h
    np.fill_diagonal(result, np.diag(h))
    return result


@pytest.mark.parametrize("factor", [0., .27, 1.])
@pytest.mark.parametrize("columns", [False, True])
def test_sparse_dressing_matches_complex_disordered_dense_definition(factor, columns):
    model, params, q, vectors = fixture()
    vectors = vectors if columns else vectors[:, 0]
    dressed = PolaronDressedModel(model, factor)
    expected = narrowed(matrix(model, params, q), factor) @ np.asarray(vectors)
    actual = jax.jit(dressed.apply)(params, q, vectors)
    prepared = jax.jit(lambda p, x, v: prepared_action(dressed, p, x)(v))(params, q, vectors)
    np.testing.assert_allclose(actual, expected, atol=2e-13, rtol=2e-13)
    np.testing.assert_allclose(prepared, expected, atol=2e-13, rtol=2e-13)
    np.testing.assert_allclose(model.diagonal(params, q), np.diag(matrix(model, params, q)), atol=2e-14)


def test_sparse_dressing_coordinate_and_parameter_gradients_match_finite_difference():
    model, params, q, vectors = fixture()
    dressed = PolaronDressedModel(model, .27)

    def objective(coordinate, scale):
        p = {**params, "hopping": scale*params["hopping"]}
        value = dressed.apply(p, coordinate, vectors)
        return jnp.sum(jnp.abs(value)**2)

    def reference(coordinate, scale):
        p = {**params, "hopping": scale*params["hopping"]}
        value = narrowed(matrix(model, p, coordinate), .27) @ np.asarray(vectors)
        return np.sum(abs(value)**2)

    actual_q, actual_scale = jax.jit(jax.grad(objective, argnums=(0, 1)))(q, 1.1)
    epsilon = 1e-5
    expected_q = [(reference(np.asarray(q) + epsilon*np.eye(3)[i], 1.1)
                   - reference(np.asarray(q) - epsilon*np.eye(3)[i], 1.1))/(2*epsilon) for i in range(3)]
    expected_scale = (reference(q, 1.1+epsilon) - reference(q, 1.1-epsilon))/(2*epsilon)
    np.testing.assert_allclose(actual_q, expected_q, atol=2e-8, rtol=2e-9)
    np.testing.assert_allclose(actual_scale, expected_scale, atol=2e-8, rtol=2e-9)


def test_sparse_diagonal_capability_does_not_materialize_an_identity_action():
    class RejectDenseAction(EdgeEPCModel):
        diagonal = EdgeEPCModel.diagonal

        def apply(self, params, q, vectors):
            if vectors.ndim == 2 and vectors.shape[1] == self.nstates:
                raise AssertionError("sparse LF dressing must not request a full matrix")
            return super().apply(params, q, vectors)

    model, params, q, vectors = fixture(RejectDenseAction)
    actual = jax.jit(PolaronDressedModel(model, .37).apply)(params, q, vectors)
    np.testing.assert_allclose(actual, narrowed(matrix(model, params, q), .37) @ vectors, atol=2e-13)


@pytest.mark.parametrize("override", ["subclass", "instance", "both_methods"])
def test_changed_apply_never_uses_an_inherited_stale_diagonal(override):
    shift = jnp.array([.8, -.2, .3, -.4, .5])

    class ChangedApply(EdgeEPCModel):
        def apply(self, params, q, vectors):
            local = shift[:, None] if vectors.ndim == 2 else shift
            return super().apply(params, q, vectors) + q[0]*local*vectors

    class ChangedBoth(ChangedApply):
        def diagonal(self, params, q):
            return super().diagonal(params, q) + q[0]*shift

    model_type = {"subclass": ChangedApply, "instance": EdgeEPCModel, "both_methods": ChangedBoth}[override]
    model, params, q, vectors = fixture(model_type)
    if override == "instance":
        original = model.apply
        object.__setattr__(model, "apply", lambda p, x, v: original(p, x, v)
                           + x[0]*(shift[:, None] if v.ndim == 2 else shift)*v)
    expected = matrix(model, params, q) + np.diag(np.asarray(q[0]*shift))
    actual = jax.jit(PolaronDressedModel(model, .31).apply)(params, q, vectors)
    np.testing.assert_allclose(actual, narrowed(expected, .31) @ vectors, atol=2e-13)


def test_model_without_diagonal_keeps_dense_reference_fallback():
    sparse, params, q, vectors = fixture()
    dense = LinearEPCModel(5, 3, complex_valued=True)
    h0 = matrix(sparse, params, np.zeros(3))
    couplings = np.stack([matrix(sparse, params, direction)-h0 for direction in np.eye(3)])
    dense_params = dense.create_params(h0, couplings)
    actual = jax.jit(PolaronDressedModel(dense, .29).apply)(dense_params, q, vectors)
    np.testing.assert_allclose(actual, narrowed(matrix(sparse, params, q), .29) @ vectors, atol=2e-13)


@pytest.mark.parametrize("instance_override", [False, True])
def test_inherited_diagonal_respects_the_models_elements_extension(instance_override):
    shift = jnp.array([.8, -.2, .3, -.4, .5])

    class ChangedElements(EdgeEPCModel):
        def elements(self, params, q):
            onsite, hopping = super().elements(params, q)
            return onsite + q[0]*shift, hopping

    model, params, q, vectors = fixture(EdgeEPCModel if instance_override else ChangedElements)
    if instance_override:
        original = model.elements

        def elements(p, x):
            onsite, hopping = original(p, x)
            return onsite + x[0]*shift, hopping

        object.__setattr__(model, "elements", elements)
    expected = matrix(model, params, q) + np.diag(np.asarray(q[0]*shift))
    actual = jax.jit(PolaronDressedModel(model, .31).apply)(params, q, vectors)
    np.testing.assert_allclose(actual, narrowed(expected, .31) @ vectors, atol=2e-13)


@pytest.mark.parametrize("value,exception", [(42, TypeError), (lambda p, q: jnp.zeros((5, 1)), ValueError)])
def test_invalid_explicit_diagonal_fails_with_a_clear_contract_error(value, exception):
    model, params, q, vectors = fixture()
    object.__setattr__(model, "diagonal", value)
    with pytest.raises(exception, match="model.diagonal"):
        PolaronDressedModel(model, .3).apply(params, q, vectors)
