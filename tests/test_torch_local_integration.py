"""Sparse Torch carrier + scalar reference through public workflow boundaries."""

from dataclasses import dataclass, replace
from importlib.metadata import version

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.linalg import expm

from pyeph import (CoupledClassical, Ehrenfest, Execution, Integrator, PrescribedPath,
                   Problem, Simulation, make_state)
from pyeph.adapters.torch_local import TorchLocalBlockModel
from pyeph.adapters.torch_reference import TorchReferenceModel
from pyeph.core.contracts import pure_state_weight
from pyeph.execution.runner import SimulationError
from pyeph.io.provenance import problem_manifest
from pyeph.models.composite import SumModel
from pyeph.models.local import AtomCenterMap, LocalBlockGraph, LocalCoefficients
from pyeph.paths.harmonic import ConstantPath
from pyeph.workflows.column_transport import (initialize_column_transport_state,
                                               make_column_transport_problem)

torch = pytest.importorskip("torch")


def local_provider(p, q, geometry):
    internal = torch.sum((q[0]-q[1])**2)
    onsite = torch.stack((p["bias"][0]+.025*internal, p["bias"][1]+.015*q[2, 1]**2))
    hopping = p["hop"]*torch.exp(-.3*geometry.distances)*torch.exp(.2j*(q[0, 1]-q[2, 2]))
    return LocalCoefficients(onsite[:, None, None], hopping[:, None, None])


def reference_provider(p, q):
    return .5*p["spring"]*torch.sum(q**2)+.003*torch.sum(q**4)


def fixture():
    carrier = TorchLocalBlockModel(
        LocalBlockGraph(2, 1, ((0, 1),)), AtomCenterMap((0, 0, 1), (.6, .4, 1.), 2),
        local_provider, complex_valued=True)
    reference = TorchReferenceModel(carrier.spec, reference_provider, zero_probes=carrier.spec.probes)
    model = SumModel((carrier, reference), additive_probes=carrier.spec.probes)
    params = (dict(bias=jnp.array([-.08, .12]), hop=jnp.array(.07)), dict(spring=jnp.array(.04)))
    q = jnp.array([[.1, .2, -.1], [.3, -.1, .2], [1.4, .3, .1]])
    identities = {"model.models[0].coefficient_provider": "sparse-torch-test-v1",
                  "model.models[1].reference_fn": "scalar-torch-test-v1"}
    return model, params, q, identities


def independent_hamiltonian(params, q):
    internal = np.sum((q[0]-q[1])**2)
    d = q[2]-(.6*q[0]+.4*q[1])
    hopping = float(params["hop"])*np.exp(-.3*np.linalg.norm(d))*np.exp(.2j*(q[0, 1]-q[2, 2]))
    return np.array([[float(params["bias"][0])+.025*internal, hopping],
                     [hopping.conjugate(), float(params["bias"][1])+.015*q[2, 1]**2]])


def test_complete_mixed_external_force_audit_and_runtime_checkpoint_identity(tmp_path):
    model, params, q, identities = fixture()
    c = np.array([np.sqrt(.6), np.sqrt(.4)*np.exp(.4j)])
    weight = pure_state_weight(jnp.asarray(c))
    force = -model.reference_gradient(params, q)-model.contract_gradient(params, q, weight)

    def energy(x):
        return (.5*float(params[1]["spring"])*np.sum(x**2)+.003*np.sum(x**4)
                + np.vdot(c, independent_hamiltonian(params[0], x)@c).real)

    gradient = np.empty(q.shape)
    for index in np.ndindex(q.shape):
        dx = np.zeros(q.shape)
        dx[index] = 1e-5
        gradient[index] = (energy(np.asarray(q)+dx)-energy(np.asarray(q)-dx))/2e-5
    np.testing.assert_allclose(force, -gradient, atol=8e-12)
    problem = Problem(model, params, CoupledClassical(jnp.array([[2.], [3.], [4.]])), Ehrenfest())
    integrator = Integrator(.005, "rk4")
    manifest = problem_manifest(problem, integrator, artifact_ids=identities)
    assert manifest["complete"]
    assert manifest["payload"]["runtime"]["versions"]["torch"] == version("torch")
    run = Simulation(problem, integrator, Execution(allow_host_callbacks=True, verify_external_gradients=True,
                                                   chunk_size=2))
    initial = make_state(q, jnp.ones_like(q)*.01, c)
    full, prefix = run.run(initial, 4), run.run(initial, 2)
    path = tmp_path/"mixed-sparse.h5"
    run.save_checkpoint(path, prefix.final_state, artifact_ids=identities)
    restored = run.load_checkpoint(path, artifact_ids=identities)
    continued = run.run(restored, 2)
    for actual, expected in zip(jax.tree.leaves(continued.final_state), jax.tree.leaves(full.final_state), strict=True):
        np.testing.assert_allclose(actual, expected, atol=3e-15, rtol=0)
    # Torch version must also be present for the sparse carrier by itself.
    carrier_problem = Problem(model.models[0], params[0], problem.nuclear_treatment, Ehrenfest())
    carrier_manifest = problem_manifest(carrier_problem, integrator,
                                        artifact_ids={"model.coefficient_provider": identities["model.models[0].coefficient_provider"]})
    assert carrier_manifest["payload"]["runtime"]["versions"]["torch"] == version("torch")


def test_sparse_torch_columns_match_independent_complex_correlation_and_keep_origin(tmp_path):
    model, params, q, identities = fixture()
    problem = make_column_transport_problem(model, params, PrescribedPath(ConstantPath(q)),
                                             probes=("current_x", "current_y"))
    factor = np.array([[np.sqrt(.6)], [np.sqrt(.4)*np.exp(.3j)]])
    initial = initialize_column_transport_state(problem, q, jnp.zeros_like(q), factor,
                                                trajectory_id=31, seed=17, time=.7)
    run = Simulation(problem, Integrator(.01, "rk4"), Execution(allow_host_callbacks=True, chunk_size=2))
    full, prefix = run.run(initial, 8), run.run(initial, 4)
    h = independent_hamiltonian(params[0], np.asarray(q))
    d = np.asarray(q[2]-(.6*q[0]+.4*q[1]))
    currents = []
    for axis in range(2):
        value = -1j*d[axis]*h[0, 1]  # charge=-1, hbar=1.
        currents.append(np.array([[0., value], [value.conjugate(), 0.]]))
    u = expm(-.08j*h)
    rho = factor@factor.conj().T
    expected = np.array([[np.trace(a@u@b@rho@u.conj().T) for b in currents] for a in currents])
    np.testing.assert_allclose(full.observables["current_correlation"][-1], expected, atol=2e-14)
    file = tmp_path/"sparse-columns.h5"
    run.save_checkpoint(file, prefix.final_state, artifact_ids=identities)
    restored = run.load_checkpoint(file, artifact_ids=identities)
    continued = run.run(restored, 4)
    np.testing.assert_allclose(continued.observables["current_correlation"][-1], expected, atol=2e-14)
    np.testing.assert_array_equal(restored.method_state["column_transport"]["time0"], .7)


def test_partial_detach_is_rejected_before_publication_by_complete_sparse_audit():
    model, params, q, _ = fixture()
    def detached(p, x, g):
        result = local_provider(p, x, g)
        return result._replace(onsite=result.onsite.detach()+.001*x[:2, 2, None, None]**2)
    bad = replace(model.models[0], coefficient_provider=detached)
    problem = Problem(bad, params[0], CoupledClassical(1.), Ehrenfest())
    run = Simulation(problem, Integrator(.01, "rk4"),
                     Execution(allow_host_callbacks=True, verify_external_gradients=True))
    published = []
    with pytest.raises(ValueError, match="incomplete local derivatives"):
        run.run(make_state(q, jnp.zeros_like(q), [1., 0.]), 2,
                observer=lambda time, values: published.append(time))
    assert published == []


@dataclass(frozen=True)
class TranslatingPath:
    origin: object
    speed: object

    def position(self, time):
        return self.origin+time*self.speed

    def velocity(self, time):
        return self.speed


def test_failed_sparse_provider_retains_last_accepted_chunk():
    model, params, q, _ = fixture()
    def bounded(p, x, g):
        coefficients = local_provider(p, x, g)
        if float(x[0, 0].detach()) > .116:
            return coefficients._replace(onsite=coefficients.onsite*float("nan"))
        return coefficients
    carrier = replace(model.models[0], coefficient_provider=bounded)
    from pyeph import CPA

    path = TranslatingPath(q, jnp.ones_like(q))
    problem = Problem(carrier, params[0], PrescribedPath(path), CPA())
    run = Simulation(problem, Integrator(.01, "rk4"), Execution(allow_host_callbacks=True, chunk_size=1))
    initial = make_state(q, jnp.zeros_like(q), [1., 0.])
    with pytest.raises(SimulationError) as caught:
        run.run(initial, 3, collect=False)
    assert int(caught.value.last_valid_state.step) == 1
    assert np.isfinite(np.asarray(caught.value.last_valid_state.electronic)).all()
