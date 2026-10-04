"""Provider-owned derivatives compose without a cross-framework autodiff graph."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pyeph import CoupledClassical, Ehrenfest, Execution, Integrator, Problem, Simulation
from pyeph.adapters.torch import TorchHamiltonianAdapter
from pyeph.core.contracts import LowRankWeight
from pyeph.core.state import make_state
from pyeph.core.units import UnitSystem
from pyeph.models.analytic import SpinBosonModel
from pyeph.models.composite import ReferenceShiftModel, SumModel

torch = pytest.importorskip("torch")


def _models():
    native = SpinBosonModel()
    def h(p, q):
        z = p["bias"] + torch.dot(p["coupling"], q)
        v = p["delta"] + q.sum()*0
        return torch.stack((torch.stack((z, v)), torch.stack((v, -z))))
    def ref(p, q):
        return .5*torch.sum((p["omega"]*(q-p["q_eq"]))**2)+p["reference_offset"]
    external = TorchHamiltonianAdapter(native.spec, h, ref)
    params = (native.default_params(), native.default_params() | {"delta": .07, "bias": .1})
    return SumModel((native, external)), SumModel((native, native)), params


def test_mixed_provider_sum_values_and_contracted_forces():
    mixed, native, params = _models()
    q = jnp.array([.23])
    left, right = jnp.array([[1+1j], [.3]]), jnp.array([[.2j], [1-.4j]])
    for weight in (LowRankWeight(left, right), left@right.conj().T):
        np.testing.assert_allclose(jax.jit(mixed.contract_gradient)(params, q, weight),
                                   native.contract_gradient(params, q, weight), atol=1e-14)
    np.testing.assert_allclose(mixed.reference_gradient(params, q),
                               native.reference_gradient(params, q), atol=1e-14)
    np.testing.assert_allclose(mixed.apply(params, q, left), native.apply(params, q, left), atol=1e-14)
    mixed.validate_complete_gradients(params, q)
    assert mixed.execution_mode == "host_callback"
    assert not mixed.spec.native_jax


def test_mixed_ehrenfest_and_reference_shift_preserve_complete_forces():
    mixed, native, params = _models()
    q, p = [.23], [.12]
    initial = make_state(q, p, [1, 0])
    problem = Problem(mixed, params, CoupledClassical(1.), Ehrenfest())
    policy = Execution(allow_host_callbacks=True, verify_external_gradients=True, chunk_size=5)
    with pytest.raises(ValueError, match="allow_host_callbacks"):
        Simulation(problem, Integrator(.01))
    actual = Simulation(problem, Integrator(.01), policy).run(initial, 15)
    expected = Simulation(replace(problem, model=native), Integrator(.01)).run(initial, 15)
    for field in ("q", "p", "electronic"):
        np.testing.assert_allclose(getattr(actual.final_state, field),
                                   getattr(expected.final_state, field), atol=2e-14)
    shifted = ReferenceShiftModel(mixed, lambda scale, x: scale*jnp.sum(x*x))
    shifted_problem = replace(problem, model=shifted, params=(params, .3))
    result = Simulation(shifted_problem, Integrator(.01), policy).run(initial, 15)
    for field in ("q", "p"):
        np.testing.assert_allclose(getattr(result.final_state, field),
                                   getattr(expected.final_state, field), atol=2e-14)
    np.testing.assert_allclose(np.abs(result.final_state.electronic)**2,
                               np.abs(expected.final_state.electronic)**2, atol=2e-12)


def test_compensated_shift_gradient_for_general_nonunit_trace_weight():
    mixed, _, params = _models()
    shifted = ReferenceShiftModel(mixed, lambda scale, q: scale*jnp.sum(q*q))
    q = jnp.array([.23])
    weight = jnp.array([[2.+1j, .2j], [.3, -.4]])
    expected = mixed.contract_gradient(params, q, weight) - 2*.3*q*1.6
    np.testing.assert_allclose(jax.jit(shifted.contract_gradient)((params, .3), q, weight),
                               expected, atol=1e-14)


def test_composition_rejects_mixed_units_and_ambiguous_parameter_container():
    model = SpinBosonModel()
    alternate = SpinBosonModel()
    object.__setattr__(alternate, "spec", replace(alternate.spec, unit_system=UnitSystem(.1, 1.)))
    with pytest.raises(ValueError, match="unit_system"):
        SumModel((model, alternate))
    model = SumModel((model, model))
    with pytest.raises(ValueError, match="one parameter PyTree"):
        model.validate_params({"a": {}, "b": {}})


def test_mixed_model_restart_requires_explicit_external_component_identity(tmp_path):
    model, _, params = _models()
    simulation = Simulation(Problem(model, params, CoupledClassical(1.), Ehrenfest()),
                            Integrator(.01), Execution(allow_host_callbacks=True))
    state = make_state([.23], [.12], [1, 0])
    file = tmp_path / "mixed.h5"
    with pytest.raises(ValueError, match=r"model.models\[1\]"):
        simulation.save_checkpoint(file, state)
    identity = {"model.models[1]": "torch-provider-code-weights-v1"}
    simulation.save_checkpoint(file, state, artifact_ids=identity)
    restored = simulation.load_checkpoint(file, artifact_ids=identity)
    np.testing.assert_array_equal(restored.q, state.q)
    np.testing.assert_array_equal(restored.electronic, state.electronic)
    with pytest.raises(ValueError, match="model"):
        simulation.load_checkpoint(file, artifact_ids={"model.models[1]": "torch-provider-code-weights-v2"})
