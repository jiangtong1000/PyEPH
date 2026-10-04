"""External sparse coefficients, complete forces, lifecycle and native policy."""

from dataclasses import FrozenInstanceError, replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pyeph.adapters.torch_local import TorchLocalBlockModel
from pyeph import CoupledClassical, Ehrenfest, Execution, Integrator, Problem, Simulation, make_state
from pyeph.core.contracts import LowRankWeight, pure_state_weight, prepared_action
from pyeph.core.state import stack_states
from pyeph.integrators.krylov import LanczosOptions
from pyeph.models.local import AtomCenterMap, LocalBlockGraph, LocalBlockModel, LocalCoefficients

torch = pytest.importorskip("torch")


def provider(p, q, g):
    onsite = (p["bias"]+p["spring"]*q[:, 0]**2)[:, None, None]
    hopping = (p["hop"]*torch.exp(-.2*g.distances))[:, None, None]
    return LocalCoefficients(onsite, hopping)


def native_provider(p, q, g):
    return LocalCoefficients((p["bias"]+p["spring"]*q[:, 0]**2)[:, None, None],
                             (p["hop"]*jnp.exp(-.2*g.distances))[:, None, None])


def fixture():
    graph = LocalBlockGraph(2, 1, ((0, 1),), switch_on=.7, cutoff=2.)
    centers = AtomCenterMap((0, 1), (1., 1.), 2)
    q = jnp.array([[.1, .2, .3], [1.3, .3, .1]])
    p = dict(bias=jnp.array([-.1, .15]), spring=jnp.array(.03), hop=jnp.array(.05))
    return TorchLocalBlockModel(graph, centers, provider), LocalBlockModel(graph, centers, native_provider), p, q


def test_value_prepared_force_and_complete_audit():
    model, native, params, q = fixture()
    model.validate_at(params, q)
    model.validate_complete_gradients(params, q)
    assert not model.spec.native_jax and model.spec.force_support
    vector = jnp.array([.7+.2j, -.4j])
    vectors = jnp.stack((vector, vector.conj()), axis=1)
    for x in (vector, vectors):
        np.testing.assert_allclose(model.apply(params, q, x), native.apply(params, q, x), atol=2e-16)
        np.testing.assert_allclose(prepared_action(model, params, q)(x), native.apply(params, q, x), atol=2e-16)
    for w in (pure_state_weight(vector), jnp.outer(vector, vector.conj()),
              LowRankWeight(vectors, vectors[:, ::-1])):
        np.testing.assert_allclose(model.contract_gradient(params, q, w),
                                   native.contract_gradient(params, q, w), atol=2e-16)
    assert model.reference_energy(params, q) == 0
    np.testing.assert_array_equal(model.reference_gradient(params, q), np.zeros_like(q))


def test_jit_batch_dynamic_parameters_and_input_ownership():
    model, native, params, q = fixture()
    c = jnp.array([1., .3j])
    w = pure_state_weight(c)
    f = jax.jit(jax.vmap(lambda x: (model.apply(params, x, c), model.contract_gradient(params, x, w))))
    coords = jnp.stack((q, q+.04))
    actual = f(coords)
    expected = jax.vmap(lambda x: (native.apply(params, x, c), native.contract_gradient(params, x, w)))(coords)
    for a, b in zip(actual, expected):
        np.testing.assert_allclose(a, b, atol=2e-16)
    model.validate_at(params, coords, batch=True)
    dynamic = jax.jit(lambda p: model.apply(p, q, c))
    changed = {**params, "hop": .1}
    np.testing.assert_allclose(dynamic(changed), native.apply(changed, q, c), atol=2e-16)


@pytest.mark.parametrize("batch", [False, True])
def test_complete_ehrenfest_native_parity_cached_update_and_checkpoint(tmp_path, batch):
    model, native, params, q = fixture()
    nuclei = CoupledClassical(jnp.array([[2.], [3.]]))
    execution = Execution(allow_host_callbacks=True, chunk_size=2)
    runner = Simulation(Problem(model, params, nuclei, Ehrenfest()), Integrator(.01), execution)
    reference = Simulation(Problem(native, params, nuclei, Ehrenfest()), Integrator(.01))
    initial = make_state(q, jnp.ones_like(q)*.02, [1., 0.])
    if batch:
        initial = stack_states([initial, make_state(q+.03, jnp.ones_like(q)*.03, [0., 1.], trajectory_id=1)])
    actual, expected = runner.run(initial, 4), reference.run(initial, 4)
    for a, b in zip(jax.tree.leaves(actual.final_state), jax.tree.leaves(expected.final_state)):
        np.testing.assert_allclose(a, b, atol=3e-15)
    changed = {**params, "hop": jnp.array(.1)}
    runner.update_parameters(changed)
    reference.update_parameters(changed)
    for a, b in zip(jax.tree.leaves(runner.run(initial, 2).final_state),
                    jax.tree.leaves(reference.run(initial, 2).final_state)):
        np.testing.assert_allclose(a, b, atol=3e-15)
    path = tmp_path/"torch-local.h5"
    with pytest.raises(ValueError, match="coefficient_provider"):
        runner.save_checkpoint(path, initial)
    artifacts = {"model.coefficient_provider": "synthetic-torch-provider-v1"}
    runner.save_checkpoint(path, initial, artifact_ids=artifacts)
    restored = runner.load_checkpoint(path, artifact_ids=artifacts)
    np.testing.assert_array_equal(restored.q, initial.q)
    with pytest.raises(ValueError, match="provenance mismatch: model"):
        runner.load_checkpoint(path, artifact_ids={**artifacts, "model.coefficient_provider": "v2"})


def test_external_execution_permission_and_checked_gate():
    model, _, params, q = fixture()
    problem = Problem(model, params, CoupledClassical(1.), Ehrenfest())
    with pytest.raises(ValueError, match="allow_host_callbacks"):
        Simulation(problem, Integrator(.1))
    with pytest.raises(ValueError, match="native"):
        Simulation(problem, Integrator(.1, LanczosOptions()), Execution(allow_host_callbacks=True))
    with pytest.raises((ValueError, TypeError), match="JVP|jvp|derivative|pure_callback"):
        jax.grad(lambda x: jnp.real(jnp.sum(model.apply(params, x, jnp.ones(2)))))(q)


def constant_provider(p, q, g):
    return LocalCoefficients(torch.eye(2, dtype=q.dtype)[None], torch.empty((0, 2, 2), dtype=q.dtype))


def test_empty_graph_genuine_constants_and_no_hidden_attachment():
    graph, centers = LocalBlockGraph(1, 2, ()), AtomCenterMap((0,), (1.,), 1)
    q = jnp.zeros((1, 3))
    model = TorchLocalBlockModel(graph, centers, constant_provider)
    with pytest.raises(ValueError, match="disconnected"):
        model.validate_at(None, q)
    constant = replace(model, coordinate_independent_hamiltonian=True)
    constant.validate_complete_gradients(None, q)
    np.testing.assert_array_equal(constant.contract_gradient(None, q, pure_state_weight(jnp.array([1., 0.]))), np.zeros_like(q))
    np.testing.assert_array_equal(constant.apply(None, q, jnp.ones(2)), np.ones(2))


def detached_provider(p, q, g):
    original = provider(p, q, g)
    return LocalCoefficients(original.onsite.detach()+.001*q[:, 1, None, None]**2, original.hopping)


def test_partial_detach_audit_and_constant_flag_do_not_hide_missing_derivatives():
    model, _, params, q = fixture()
    detached = replace(model, coefficient_provider=detached_provider,
                       coordinate_independent_hamiltonian=True)
    detached.validate_at(params, q)
    with pytest.raises(ValueError, match="incomplete local derivatives"):
        detached.validate_complete_gradients(params, q)


@pytest.mark.parametrize("failure", ["shape", "numpy", "integer", "nan", "nonhermitian", "complex"])
def test_provider_preflight_rejects_malformed_coefficients(failure):
    model, _, params, q = fixture()
    def bad(p, x, g):
        result = provider(p, x, g)
        if failure == "shape":
            return result._replace(onsite=result.onsite[..., 0])
        if failure == "numpy":
            return result._replace(onsite=result.onsite.detach().numpy())
        if failure == "integer":
            return result._replace(onsite=result.onsite.to(torch.int64))
        if failure == "nan":
            return result._replace(hopping=result.hopping*float("nan"))
        if failure == "nonhermitian":
            return result._replace(onsite=result.onsite.to(torch.complex128)+.1j)
        return result._replace(hopping=result.hopping.to(torch.complex128))
    altered = replace(model, coefficient_provider=bad, complex_valued=failure == "nonhermitian")
    with pytest.raises((TypeError, ValueError)):
        altered.validate_at(params, q)


@pytest.mark.parametrize("field,value", [("q", np.zeros((2, 3), dtype=np.int32)),
                                         ("q", np.zeros((2, 3), dtype=np.complex128)),
                                         ("q", np.ones((2, 3))),
                                         ("q", np.full((2, 3), np.nan)),
                                         ("params", {"x": np.inf})])
def test_bad_inputs_are_rejected(field, value):
    model, _, params, q = fixture()
    with pytest.raises(ValueError):
        model.validate_at(value if field == "params" else params, value if field == "q" else q)


def test_static_flags_are_immutable_owned_values():
    model, _, _, _ = fixture()
    flag, tolerance = np.array(True), np.array(1e-4)
    configured = replace(model, coordinate_independent_hamiltonian=flag, audit_atol=tolerance)
    flag[...] = False
    tolerance[...] = 4.
    assert configured.coordinate_independent_hamiltonian is True
    assert configured.audit_atol == 1e-4
    with pytest.raises(FrozenInstanceError):
        configured.coefficient_provider = provider


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_declared_coordinate_precision_and_constant_flag_preserve_attached_gradients(dtype):
    model, native, params, q = fixture()
    q = q.astype(dtype)
    params = jax.tree.map(lambda value: value.astype(dtype), params)
    enabled = replace(model, coordinate_independent_hamiltonian=True)
    enabled.validate_complete_gradients(params, q)
    c = jnp.array([.7+.1j, -.3j], dtype=jnp.complex64 if dtype == jnp.float32 else jnp.complex128)
    actual = enabled.contract_gradient(params, q, pure_state_weight(c))
    assert actual.dtype == dtype
    np.testing.assert_allclose(actual, native.contract_gradient(params, q, pure_state_weight(c)),
                               atol=2e-8 if dtype == jnp.float32 else 2e-16)
    assert np.max(np.abs(actual)) > 1e-3


@pytest.mark.parametrize("weight", [jnp.ones(2), jnp.ones((2, 3)),
                                    LowRankWeight(jnp.ones(2), jnp.ones(2)),
                                    LowRankWeight(jnp.ones((2, 1)), jnp.ones((2, 2)))])
def test_invalid_contraction_shapes_fail_before_callback(weight):
    model, _, params, q = fixture()
    with pytest.raises(ValueError, match="weight"):
        model.contract_gradient(params, q, weight)


def test_changed_runtime_parameters_reject_checkpoint_identity(tmp_path):
    model, _, params, q = fixture()
    sim = Simulation(Problem(model, params, CoupledClassical(1.), Ehrenfest()),
                     Integrator(.01), Execution(allow_host_callbacks=True))
    path = tmp_path/"changed-params.h5"
    artifacts = {"model.coefficient_provider": "same-provider-v1"}
    sim.save_checkpoint(path, make_state(q, jnp.zeros_like(q), [1., 0.]), artifact_ids=artifacts)
    sim.update_parameters({**params, "hop": jnp.array(.07)})
    with pytest.raises(ValueError, match="params"):
        sim.load_checkpoint(path, artifact_ids=artifacts)


def test_bad_gradient_is_rejected_even_when_value_is_finite():
    model, _, params, q = fixture()
    def bad_gradient(p, x, g):
        original = provider(p, x, g)
        # At x00=.1 the value is finite, but sqrt has an infinite derivative.
        return original._replace(onsite=original.onsite+torch.sqrt(x[0, 0]-.1))
    changed = replace(model, coefficient_provider=bad_gradient)
    with pytest.raises(ValueError, match="gradient.*finite"):
        changed.validate_at(params, q)
