"""Actual optional Torch execution through the declared host boundary."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pyeph.adapters.torch import TorchHamiltonianAdapter
from pyeph.core.contracts import ProbeContext, pure_state_weight
from pyeph.core.problem import CoupledClassical, Problem
from pyeph.core.state import make_state, stack_states
from pyeph.dynamics.cpa import CPA
from pyeph.dynamics.ehrenfest import Ehrenfest
from pyeph.execution.runner import Execution, Runner
from pyeph.integrators.electronic import Integrator
from pyeph.models.analytic import SpinBosonModel
from pyeph.paths.harmonic import HarmonicBath

torch = pytest.importorskip("torch")


def torch_spin_boson(p, q):
    z = p["bias"] + torch.dot(p["coupling"], q)
    delta = p["delta"] + q.sum()*0
    return torch.stack((torch.stack((z, delta)), torch.stack((delta, -z))))


def torch_reference(p, q):
    return .5*torch.sum((p["omega"]*(q-p["q_eq"]))**2) + p["reference_offset"]


def fixture():
    native = SpinBosonModel(2)
    p = native.default_params() | dict(omega=jnp.array([.3, .7]), coupling=jnp.array([.02, -.03]))
    adapter = TorchHamiltonianAdapter(native.spec, torch_spin_boson, torch_reference)
    return native, adapter, p, jnp.array([.2, -.1])


def test_torch_values_complete_derivatives_and_jit_batch():
    native, adapter, p, q = fixture()
    adapter.validate_complete_gradients(p, q)
    c = jnp.array([1, 2j])/jnp.sqrt(5.)
    for v in (c, jnp.stack((c, c.conj()), axis=-1)):
        np.testing.assert_allclose(jax.jit(adapter.apply)(p, q, v), native.apply(p, q, v), atol=1e-13)
    w = pure_state_weight(c)
    for weight in (w, w.left@w.right.conj().T):
        np.testing.assert_allclose(jax.jit(adapter.contract_gradient)(p, q, weight),
                                   native.contract_gradient(p, q, weight), atol=1e-13)
    np.testing.assert_allclose(jax.jit(adapter.reference_gradient)(p, q), native.reference_gradient(p, q), atol=1e-13)
    qs = jnp.stack((q, q+.1, -q))
    np.testing.assert_allclose(jax.jit(jax.vmap(lambda x: adapter.apply(p, x, c)))(qs),
                               jax.vmap(lambda x: native.apply(p, x, c))(qs), atol=1e-13)
    np.testing.assert_allclose(jax.jit(jax.vmap(lambda x: adapter.contract_gradient(p, x, w)))(qs),
                               jax.vmap(lambda x: native.contract_gradient(p, x, w))(qs), atol=1e-13)


def test_torch_complex_offdiagonal_derivative_and_probe():
    native, _, p, q = fixture()
    def hamiltonian(p, q):
        v = p["delta"] + 1j*torch.sin(q[0])
        z = q[1]+0j
        return torch.stack((torch.stack((z, v)), torch.stack((v.conj(), -z))))
    def probe(p, context):
        return torch.diag(context.q.to(torch.complex128))
    model = TorchHamiltonianAdapter(replace(native.spec, complex_valued=True), hamiltonian,
                                     probes={"position_test": probe})
    model.validate_complete_gradients(p, q)
    weight = jnp.zeros((2, 2), dtype=complex).at[0, 1].set(1j)
    np.testing.assert_allclose(model.contract_gradient(p, q, weight), [np.cos(q[0]), 0.], atol=1e-13)
    np.testing.assert_allclose(model.probe_apply(p, ProbeContext(q), "position_test", jnp.eye(2)),
                               np.diag(q), atol=1e-13)


@pytest.mark.parametrize("method", (CPA(), Ehrenfest()))
def test_torch_callback_complete_trajectories_match_native(method):
    native, adapter, p, q = fixture()
    treatment = HarmonicBath(p["omega"]) if isinstance(method, CPA) else CoupledClassical(1.)
    problem = Problem(adapter, p, treatment, method)
    with pytest.raises(ValueError, match="allow_host_callbacks"):
        Runner(problem, Integrator(.02))
    initial = stack_states([make_state(q, jnp.array([.02, -.03]), jnp.array([1., 0.]), trajectory_id=0),
                            make_state(-q, jnp.array([-.01, .02]), jnp.array([0., 1.]), trajectory_id=1)])
    result = Runner(problem, Integrator(.02), Execution(allow_host_callbacks=True,
                    verify_external_gradients=True, chunk_size=7)).run(initial, 12)
    expected = Runner(replace(problem, model=native), Integrator(.02), Execution(chunk_size=7)).run(initial, 12)
    np.testing.assert_allclose(result.final_state.q, expected.final_state.q, atol=1e-12)
    np.testing.assert_allclose(result.final_state.p, expected.final_state.p, atol=1e-12)
    np.testing.assert_allclose(result.final_state.electronic, expected.final_state.electronic, atol=1e-12)
    jax.tree.map(lambda a, b: np.testing.assert_allclose(a, b, atol=1e-12),
                 result.observables, expected.observables)


def test_detached_baseline_fails_derivative_preflight():
    native, _, p, q = fixture()
    def bad(p, x):
        # The attached spin-boson term gives requires_grad=True, concealing a
        # detached baseline. Connectivity alone cannot detect this defect.
        return torch_spin_boson(p, x) + torch.diag(x.detach()**2)
    bad_model = TorchHamiltonianAdapter(native.spec, bad, torch_reference)
    with pytest.raises(ValueError, match="detached baseline"):
        bad_model.validate_complete_gradients(p, q)
    bad_problem = Problem(bad_model, p, CoupledClassical(1.), Ehrenfest())
    initial = make_state(q, jnp.zeros_like(q), jnp.array([1., 0.]))
    with pytest.raises(ValueError, match="detached baseline"):
        Runner(bad_problem, Integrator(.02), Execution(allow_host_callbacks=True,
                verify_external_gradients=True)).run(initial, 1)
    detached = TorchHamiltonianAdapter(native.spec, lambda p, x: torch_spin_boson(p, x).detach())
    with pytest.raises(ValueError, match="detached from coordinates"):
        detached.validate_complete_gradients(p, q)
    numpy_model = TorchHamiltonianAdapter(native.spec, lambda p, x: np.eye(2))
    with pytest.raises(TypeError, match="PyTorch tensor"):
        numpy_model.validate_complete_gradients(p, q)


def test_declared_constant_model_and_no_cross_framework_autograd_claim():
    native, _, p, q = fixture()
    constant = TorchHamiltonianAdapter(native.spec, lambda p, q: torch.diag(p["coupling"]),
                                       coordinate_independent_hamiltonian=True)
    constant.validate_complete_gradients(p, q)
    np.testing.assert_allclose(constant.contract_gradient(p, q, jnp.eye(2)), 0., atol=0)
    np.testing.assert_allclose(constant.reference_gradient(p, q), 0., atol=0)
    with pytest.raises(ValueError, match="JVP|JVPs"):
        jax.grad(lambda x: constant.reference_energy(p, x))(q)


def test_torch_dynamic_parameters_and_float32_preflight():
    native, adapter, p, q = fixture()
    c = jnp.array([1., 0.])
    compiled = jax.jit(adapter.apply)
    changed = p | dict(coupling=p["coupling"]*2)
    np.testing.assert_allclose(compiled(changed, q, c), native.apply(changed, q, c), atol=1e-13)
    single = jax.tree.map(lambda x: jnp.asarray(x, dtype=jnp.float32), p)
    adapter.validate_complete_gradients(single, q.astype(jnp.float32))
