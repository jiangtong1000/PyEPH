"""Complete external scalar forces composed with a native local carrier."""

from dataclasses import FrozenInstanceError, replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pyeph.adapters.torch_reference import TorchReferenceModel
from pyeph.core.contracts import LowRankWeight, ModelSpec, ProbeContext, prepared_action, pure_state_weight
from pyeph.core.problem import CoupledClassical, PrescribedPath, Problem
from pyeph.core.state import make_state, stack_states
from pyeph.core.system import SystemSpec
from pyeph.core.units import UnitSystem
from pyeph.dynamics.cpa import CPA
from pyeph.dynamics.ehrenfest import Ehrenfest
from pyeph.dynamics.mashrm import MASHRM, MASHRMPopulation
from pyeph.execution.runner import Execution
from pyeph.integrators.electronic import Integrator
from pyeph.integrators.krylov import LanczosOptions
from pyeph.io.provenance import problem_manifest
from pyeph.models.base import AutoDiffModel
from pyeph.models.composite import SumModel
from pyeph.models.local import AtomCenterMap, LocalBlockGraph, LocalBlockModel, LocalCoefficients
from pyeph.paths.harmonic import ConstantPath
from pyeph.simulation import Simulation

torch = pytest.importorskip("torch")


def torch_energy(p, q):
    return (0.5*torch.sum(p["spring"]*(q-p["q0"])**2)
            + p["quartic"]*torch.sum(q**4) + p["collective"]*q.sum()**2 + p["offset"])


def native_energy(p, q):
    return (0.5*jnp.sum(p["spring"]*(q-p["q0"])**2)
            + p["quartic"]*jnp.sum(q**4) + p["collective"]*q.sum()**2 + p["offset"])


def carrier_coefficients(p, q, geometry):
    onsite = jnp.stack((p["bias"]+0.02*q[0, 0]**2, -0.05*q[1, 1]))[:, None, None]
    hopping = (p["hop"]*jnp.exp(-0.2*geometry.distances))[:, None, None]
    return LocalCoefficients(onsite, hopping)


class NativeReference(AutoDiffModel):
    def __init__(self, spec):
        self.spec = replace(spec, name="native_reference", native_jax=True)

    def apply(self, params, q, vectors):
        return jnp.zeros_like(vectors)

    def reference_energy(self, params, q):
        return native_energy(params, q)

    def probe_apply(self, params, context, probe, vectors):
        return jnp.zeros_like(vectors)


def fixture():
    carrier = LocalBlockModel(LocalBlockGraph(2, 1, ((0, 1),), switch_on=1.0, cutoff=3.0),
                             AtomCenterMap((0, 0, 1), (0.4, 0.6, 1.0), 2), carrier_coefficients)
    reference = TorchReferenceModel(carrier.spec, torch_energy, zero_probes=carrier.spec.probes)
    combined = SumModel((carrier, reference), additive_probes=carrier.spec.probes)
    native = SumModel((carrier, NativeReference(reference.spec)), additive_probes=carrier.spec.probes)
    q = jnp.array([[-0.1, 0.2, 0.1], [0.3, -0.1, 0.2], [1.7, 0.3, -0.2]])
    params = (dict(bias=jnp.array(0.07), hop=jnp.array(0.04)),
              dict(spring=jnp.full(q.shape, 0.08), q0=q*0.9, quartic=jnp.array(0.003),
                   collective=jnp.array(0.006), offset=jnp.array(0.17)))
    return combined, native, params, q


def finite_gradient(function, q, step=1e-5):
    result = np.empty_like(q)
    for index in np.ndindex(q.shape):
        direction = np.zeros_like(q)
        direction[index] = step
        result[index] = (function(np.asarray(q)+direction)-function(np.asarray(q)-direction))/(2*step)
    return result


def independent_total_energy(params, q, c):
    cp, rp = jax.tree.map(np.asarray, params)
    center0 = 0.4*q[0]+0.6*q[1]
    distance = np.linalg.norm(q[2]-center0)
    u = np.clip((distance-1)/2, 0, 1)
    support = 1-10*u**3+15*u**4-6*u**5
    transfer = cp["hop"]*np.exp(-0.2*distance)*support
    h = np.array([[cp["bias"]+0.02*q[0, 0]**2, transfer], [transfer, -0.05*q[1, 1]]])
    reference = (0.5*np.sum(rp["spring"]*(q-rp["q0"])**2)+rp["quartic"]*np.sum(q**4)
                 +rp["collective"]*np.sum(q)**2+rp["offset"])
    return reference+np.vdot(c, h@c).real


def test_complete_scalar_and_composed_forces_against_independent_finite_difference():
    model, native, params, q = fixture()
    reference = model.models[1]
    reference.validate_complete_gradients(params[1], q)
    reference.validate_at(params[1], jnp.stack((q, q+0.1)), batch=True)
    c = jnp.array([1, 2j])/jnp.sqrt(5)
    actual = model.reference_gradient(params, q)+model.contract_gradient(params, q, pure_state_weight(c))
    expected = finite_gradient(lambda x: independent_total_energy(params, x, np.asarray(c)), q)
    np.testing.assert_allclose(actual, expected, atol=2e-10, rtol=2e-8)
    np.testing.assert_allclose(actual, native.reference_gradient(params, q)
                               +native.contract_gradient(params, q, pure_state_weight(c)), atol=1e-14)
    np.testing.assert_allclose(model.probe_apply(params, ProbeContext(q), "current_x", c),
                               model.models[0].probe_apply(params[0], ProbeContext(q), "current_x", c), atol=0)


def test_scalar_callback_jit_batch_dynamic_params_and_no_cross_framework_ad():
    model, _, params, q = fixture()
    reference = model.models[1]
    functions = jax.jit(jax.vmap(lambda p, x: (reference.reference_energy(p, x),
                                               reference.reference_gradient(p, x)), in_axes=(None, 0)))
    qs = jnp.stack((q, q+0.07))
    for factor in (1.0, 1.3):
        p = params[1] | {"spring": params[1]["spring"]*factor}
        energy, gradient = functions(p, qs)
        expected = jax.vmap(jax.value_and_grad(native_energy, argnums=1), in_axes=(None, 0))(p, qs)
        np.testing.assert_allclose(energy, expected[0], atol=1e-14)
        np.testing.assert_allclose(gradient, expected[1], atol=1e-14)
    with pytest.raises(ValueError, match="JVP"):
        jax.grad(lambda x: reference.reference_energy(params[1], x))(q)


def test_zero_electronic_operations_never_call_torch_or_construct_dense_matrix():
    def forbidden(params, q):
        raise AssertionError("electronic zero operation called the energy provider")

    spec = ModelSpec(SystemSpec(10_000, (3, 3)))
    model = TorchReferenceModel(spec, forbidden, zero_probes=("current_x",))
    q, vectors = jnp.ones((3, 3)), jnp.ones((10_000, 2), dtype=complex)
    weight = LowRankWeight(vectors, vectors)

    def operations(q, v):
        return (prepared_action(model, None, q)(v), model.contract_gradient(None, q, weight),
                model.probe_apply(None, ProbeContext(q), "current_x", v))

    traced = str(jax.make_jaxpr(operations)(q, vectors))
    assert "pure_callback" not in traced and "10000,10000" not in traced
    for value in jax.jit(operations)(q, vectors):
        np.testing.assert_array_equal(value, np.zeros_like(value))
    with pytest.raises(NotImplementedError):
        model.probe_apply(None, ProbeContext(q), "position_x", vectors)


@pytest.mark.parametrize("method", [CPA(), Ehrenfest()])
def test_public_native_carrier_external_reference_trajectories_and_cached_updates(method):
    model, native, params, q = fixture()
    nuclei = PrescribedPath(ConstantPath(q)) if isinstance(method, CPA) else CoupledClassical(jnp.array([1., 2., 3.])[:, None])
    problem = Problem(model, params, nuclei, method)
    with pytest.raises(ValueError, match="allow_host_callbacks"):
        Simulation(problem, Integrator(0.01))
    execution = Execution(allow_host_callbacks=True, verify_external_gradients=True, chunk_size=4)
    external = Simulation(problem, Integrator(0.01), execution)
    all_native = Simulation(replace(problem, model=native), Integrator(0.01), Execution(chunk_size=4))
    initial = stack_states([make_state(q, jnp.full_like(q, p), c, trajectory_id=i)
                            for i, (p, c) in enumerate(((0.01, [1., 0.]), (-0.01, [0., 1.])))])
    for factor in (1.0, 1.2):
        changed = (params[0], params[1] | {"spring": factor*params[1]["spring"]})
        external.update_parameters(changed)
        all_native.update_parameters(changed)
        actual, expected = external.run(initial, 8), all_native.run(initial, 8)
        for field in ("q", "p", "electronic"):
            np.testing.assert_allclose(getattr(actual.final_state, field), getattr(expected.final_state, field), atol=2e-13)
        jax.tree.map(lambda a, b: np.testing.assert_allclose(a, b, atol=2e-13), actual.observables, expected.observables)


def test_existing_native_only_method_restrictions_remain():
    model, _, params, _ = fixture()
    problem = Problem(model, params, CoupledClassical(1.0), Ehrenfest())
    execution = Execution(allow_host_callbacks=True)
    with pytest.raises(ValueError, match="native JAX"):
        Simulation(problem, Integrator(0.01, electronic=LanczosOptions()), execution)
    rm = replace(problem, method=MASHRM(), measurement=MASHRMPopulation())
    with pytest.raises(ValueError, match="real native"):
        Simulation(rm, Integrator(0.01, "exponential_midpoint"), execution)


@pytest.mark.parametrize("kind", ["detached", "partial", "numpy", "shape", "complex", "energy_nan", "gradient_nan"])
def test_invalid_external_outputs_and_incomplete_derivatives_rejected(kind):
    model, _, params, q = fixture()
    if kind == "detached":
        def energy(p, x):
            return torch_energy(p, x).detach()
    elif kind == "partial":
        def energy(p, x):
            return x.sum() + torch.sum(x.detach()**2)
    elif kind == "numpy":
        def energy(p, x):
            return np.array(1.)
    elif kind == "shape":
        def energy(p, x):
            return torch_energy(p, x).reshape(1)
    elif kind == "complex":
        def energy(p, x):
            return torch_energy(p, x).to(torch.complex128)
    elif kind == "energy_nan":
        def energy(p, x):
            return x.sum()*torch.tensor(float("nan"))
    else:
        def energy(p, x):
            # Finite energy, infinite derivative at every supplied coordinate.
            return torch.sqrt((x-x.detach())**2).sum()
    reference = replace(model.models[1], reference_fn=energy)
    with pytest.raises((TypeError, ValueError)):
        reference.validate_complete_gradients(params[1], q)


def test_incomplete_baseline_is_detected_by_runner_opt_in_before_output():
    model, _, params, q = fixture()

    def bad(p, x):
        return x.sum()+torch.sum(x.detach()**2)

    reference = replace(model.models[1], reference_fn=bad)
    model = SumModel((model.models[0], reference))
    run = Simulation(Problem(model, params, CoupledClassical(1.0), Ehrenfest()), Integrator(0.01),
                     Execution(allow_host_callbacks=True, verify_external_gradients=True))
    initial = make_state(q, jnp.zeros_like(q), [1, 0])
    published = []
    with pytest.raises(ValueError, match="detached baseline"):
        run.run(initial, 1, observer=lambda t, values: published.append(t))
    assert published == []


def test_legitimate_constant_reference_explicit_declaration_and_dtype_conversion():
    spec = ModelSpec(SystemSpec(2, (1,)))

    def constant(p, q):
        return p["offset"]

    p, q = {"offset": jnp.array(0.7)}, jnp.array([0.2])
    undeclared = TorchReferenceModel(spec, constant)
    with pytest.raises(ValueError, match="detached"):
        undeclared.validate_at(p, q)
    declared = replace(undeclared, coordinate_independent_reference=True)
    declared.validate_complete_gradients(p, q)
    np.testing.assert_array_equal(declared.reference_gradient(p, q), [0.0])
    declared.validate_at(p, q.astype(jnp.float32))
    huge = p | {"offset": jnp.array(1e100)}
    with pytest.raises(ValueError, match="output dtype"):
        declared.validate_at(huge, q.astype(jnp.float32))


def test_dtype_mixed_provider_audit_uses_complete_float32_gradient():
    model, _, params, q = fixture()
    q = q.astype(jnp.float32)
    # Provider parameters remain float64, while derivative coordinates are float32.
    model.models[1].validate_complete_gradients(params[1], q)


def test_provider_unit_conversion_remains_in_complete_gradient():
    from pyeph.core.units import BOHR_ANGSTROM, HARTREE_EV

    def energy_ev(p, q_bohr):
        position_angstrom = q_bohr*BOHR_ANGSTROM
        value_ev = 0.5*p["spring_ev_angstrom2"]*torch.sum(position_angstrom**2)
        return value_ev/HARTREE_EV

    model = TorchReferenceModel(ModelSpec(SystemSpec(2, (2, 3))), energy_ev)
    p = {"spring_ev_angstrom2": jnp.array(0.4)}
    q = jnp.array([[0.2, 0.7, -0.3], [-0.1, 0.4, 1.2]])
    model.validate_complete_gradients(p, q)
    np.testing.assert_allclose(model.reference_gradient(p, q),
        0.4*BOHR_ANGSTROM**2/HARTREE_EV*q, atol=2e-17, rtol=2e-14)


@pytest.mark.parametrize("quantity", ["energy", "gradient"])
def test_compiled_callback_rejects_nonfinite_outputs_at_operation_boundary(quantity):
    def energy(p, q):
        if quantity == "energy":
            return q.sum()*torch.tensor(float("nan"))
        return torch.sqrt((q-q.detach())**2).sum()

    model = TorchReferenceModel(ModelSpec(SystemSpec(2, (1,))), energy)
    operation = model.reference_energy if quantity == "energy" else model.reference_gradient
    with pytest.raises(Exception, match="nonfinite|finite"):
        jax.jit(operation)(None, jnp.array([0.3])).block_until_ready()


def test_constructor_preserves_units_snapshots_options_and_freezes_configuration():
    unit = UnitSystem.from_ev_angstrom()
    spec = ModelSpec(SystemSpec(2, (1,), basis_id="aligned-dimer"), unit_system=unit,
                     probes=("position_x",))
    names, tolerance, flag = ["current_x"], np.array(1e-6), np.array(True)
    model = TorchReferenceModel(spec, lambda p, q: q.sum()*0, zero_probes=names,
                                audit_atol=tolerance, coordinate_independent_reference=flag)
    names.append("current_y")
    tolerance[...] = 2
    flag[...] = False
    assert model.spec.system == spec.system and model.spec.unit_system == unit
    assert model.spec.probes == ("current_x",) and model.audit_atol == 1e-6
    assert model.coordinate_independent_reference and not model.spec.native_jax
    with pytest.raises(FrozenInstanceError):
        model.reference_fn = lambda p, q: q.sum()


def test_strict_composed_checkpoint_identifies_reference_provider_and_parameter_changes(tmp_path):
    model, _, params, q = fixture()
    execution = Execution(allow_host_callbacks=True, chunk_size=3)
    run = Simulation(Problem(model, params, CoupledClassical(1.0), Ehrenfest()), Integrator(0.01), execution)
    manifest = problem_manifest(run.problem, run.integrator)
    assert set(manifest["unresolved"]) == {"model.models[0].coefficient_provider", "model.models[1].reference_fn"}
    artifacts = {"model.models[0].coefficient_provider": "test-local-provider-v1",
                 "model.models[1].reference_fn": "test-reference-provider-v1"}
    initial = make_state(q, jnp.full_like(q, 0.01), [1, 0])
    full = run.run(initial, 6).final_state
    partial = run.run(initial, 3).final_state
    path = tmp_path / "reference.h5"
    with pytest.raises(ValueError, match="incomplete"):
        run.save_checkpoint(path, partial)
    run.save_checkpoint(path, partial, artifact_ids=artifacts)
    loaded = run.load_checkpoint(path, artifact_ids=artifacts)
    resumed = run.run(loaded, 3).final_state
    jax.tree.map(lambda a, b: np.testing.assert_allclose(a, b, atol=2e-13), resumed, full)
    with pytest.raises(ValueError, match="model"):
        run.load_checkpoint(path, artifact_ids=artifacts | {"model.models[1].reference_fn": "v2"})
    run.update_parameters((params[0], params[1] | {"quartic": params[1]["quartic"]*2}))
    with pytest.raises(ValueError, match="params"):
        run.load_checkpoint(path, artifact_ids=artifacts)


@pytest.mark.parametrize("kwargs", [
    {"zero_probes": "current_x"}, {"zero_probes": ("current_x", "current_x")},
    {"audit_step": 0}, {"audit_atol": -1}, {"audit_rtol": np.nan},
    {"coordinate_independent_reference": 1},
])
def test_invalid_configuration_rejected(kwargs):
    with pytest.raises(ValueError):
        TorchReferenceModel(ModelSpec(SystemSpec(2, (1,))), lambda p, q: q.sum(), **kwargs)
