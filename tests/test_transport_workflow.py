"""Native transport recipes against independent SciPy propagation and traces."""

from dataclasses import replace
import itertools

import jax.numpy as jnp
import numpy as np
import pytest
from scipy.linalg import expm

from pyeph.core.contracts import ModelSpec
from pyeph.core.problem import CoupledClassical, PrescribedPath
from pyeph.core.state import stack_states
from pyeph.core.system import SystemSpec
from pyeph.dynamics.ehrenfest import Ehrenfest
from pyeph.execution.runner import Execution
from pyeph.integrators.electronic import Integrator
from pyeph.io.checkpoint import load_checkpoint, save_checkpoint
from pyeph.io.provenance import problem_manifest
from pyeph.models.base import AutoDiffModel
from pyeph.paths.harmonic import ConstantPath, HarmonicBath
from pyeph.simulation import Simulation
from pyeph.workflows.polaron_transport import (
    PolaronDressedModel,
    PolaronTransportMeasurement,
    initialize_polaron_transport_state,
    make_polaron_transport_problem,
)
from pyeph.workflows.transport import (
    initialize_transport_state,
    make_transport_problem,
)


class SmallTransportModel(AutoDiffModel):
    spec = ModelSpec(SystemSpec(2, (1,), coordinate_kind="normal_mode"),
                     name="transport_reference", complex_valued=True,
                     probes=("current_x", "current_y"))

    def dense(self, params, q):
        return params["h0"] + q[0] * params["coupling"]

    def apply(self, params, q, vectors):
        return self.dense(params, q) @ vectors

    def reference_energy(self, params, q):
        return 0.5 * params["omega"] ** 2 * jnp.sum(q * q)

    def probe_apply(self, params, context, probe, vectors):
        h = self.dense(params, context.q)
        current = 1j * jnp.array([[0.0, 1.0], [-1.0, 0.0]]) * h
        if probe == "current_y":
            # Deliberately depends on the complete physical probe context.
            current = current + jnp.eye(2) * context.velocity[0] * 0.1
        return current @ vectors


def _params():
    return dict(h0=jnp.array([[0.7, 0.31 + 0.17j], [0.31 - 0.17j, -0.4]]),
                coupling=jnp.array([[0.25, 0.07j], [-0.07j, -0.15]]), omega=0.9)


def test_direct_lf_configuration_snapshots_numpy_inputs_before_jit_capture():
    original = make_polaron_transport_problem(
        SmallTransportModel(), _params(), HarmonicBath([.9]), [1.7], [.4], 1.2,
        hopping_pairs=[[0, 1], [1, 0]], estimator="full")
    factor = np.array(original.model.factor)
    mutable = {name: np.array(getattr(original.measurement, name))
               for name in ("frequencies", "couplings", "beta", "quad_indices", "sector_indices")}
    model = PolaronDressedModel(original.model.base_model, factor)
    measurement = PolaronTransportMeasurement(**mutable)
    problem = replace(original, model=model, measurement=measurement).validate()
    state = initialize_polaron_transport_state(problem, [.3], [-.2])
    integrator = Integrator(.03)
    simulation = Simulation(problem, integrator, Execution(chunk_size=3))
    expected = simulation.run(state, 5)
    identities = {"model.base_model": "test-small-transport-model"}
    before = problem_manifest(problem, integrator, artifact_ids=identities)
    factor[...] = 0.
    for array in mutable.values():
        array[...] = 0
    after = problem_manifest(problem, integrator, artifact_ids=identities)
    for section in ("model", "measurement"):
        assert before["payload"][section] == after["payload"][section]
    assert type(model.factor) is float and type(measurement.beta) is float
    for actual in (simulation.run(state, 5), Simulation(problem, integrator).run(state, 5)):
        np.testing.assert_allclose(actual.final_state.electronic, expected.final_state.electronic, atol=1e-14)
        np.testing.assert_allclose(actual.observables["current_correlation"],
                                   expected.observables["current_correlation"], atol=1e-14)
    assert measurement.frequencies.dtype == original.measurement.frequencies.dtype
    assert measurement.couplings.dtype == original.measurement.couplings.dtype


def _numpy_h(params, q):
    return np.asarray(params["h0"]) + q * np.asarray(params["coupling"])


def _numpy_j(h, velocity, probe):
    current = 1j * np.array([[0, 1], [-1, 0]]) * h
    if probe == "current_y":
        current += np.eye(2) * velocity * 0.1
    return current


def _numpy_rho(h, beta):
    rho = expm(-beta * h)
    return rho / np.trace(rho)


def _numpy_rk4(u, time, dt, h_at):
    k1 = -1j * h_at(time) @ u
    k2 = -1j * h_at(time + dt / 2) @ (u + dt * k1 / 2)
    k3 = -1j * h_at(time + dt / 2) @ (u + dt * k2 / 2)
    k4 = -1j * h_at(time + dt) @ (u + dt * k3)
    return u + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6


def _trace(u, rho, jt, j0):
    return np.trace(jt @ u @ j0 @ rho @ u.conj().T)


def _lf_trace(u, rho, jt, j0, phi0, phit):
    total = 0j
    for i, j, k, ell in itertools.product(range(2), repeat=4):
        sector = int(i == k) - int(j == k) - int(i == ell) + int(j == ell)
        factor = np.exp((-2 + int(i == j) + int(k == ell)) * phi0 - sector * phit)
        g = sum(u[i, m].conjugate() * rho[ell, m] for m in range(2))
        total += jt[i, j] * j0[k, ell] * u[j, k] * g * factor
    return total


@pytest.mark.parametrize("algorithm", ["rk4", "exponential_midpoint"])
def test_static_thermal_transport_matches_scipy_exponential(algorithm):
    params, q, beta = _params(), 0.3, 1.2
    problem = make_transport_problem(SmallTransportModel(), params, PrescribedPath(ConstantPath([q])))
    state = initialize_transport_state(problem, [q], [0.0], beta)
    result = Simulation(problem, Integrator(0.01, algorithm), Execution(chunk_size=17)).run(state, 40)
    h = _numpy_h(params, q)
    rho, current = _numpy_rho(h, beta), _numpy_j(h, 0, "current_x")
    expected = [_trace(expm(-1j * t * h), rho, current, current) for t in result.times]
    np.testing.assert_allclose(result.observables["current_correlation"][:, 0], expected, atol=2e-11)
    np.testing.assert_allclose(result.final_state.method_state["transport"]["rho0"], rho, atol=2e-13)
    assert result.observables["unitary_error"].max() < 2e-11


def test_harmonic_transport_matches_independent_numpy_rk4_and_probe_context():
    params, q0, p0, beta, dt, steps = _params(), 0.3, -0.2, 1.2, 0.025, 25
    w = params["omega"]
    def q_at(t):
        return q0 * np.cos(w * t) + p0 / w * np.sin(w * t)

    def v_at(t):
        return p0 * np.cos(w * t) - q0 * w * np.sin(w * t)

    def h_at(t):
        return _numpy_h(params, q_at(t))
    probes = ("current_x", "current_y")
    problem = make_transport_problem(SmallTransportModel(), params, HarmonicBath([w]), probes=probes)
    state = initialize_transport_state(problem, [q0], [p0], beta)
    result = Simulation(problem, Integrator(dt), Execution(chunk_size=8)).run(state, steps)
    rho = _numpy_rho(h_at(0), beta)
    currents0 = [_numpy_j(h_at(0), v_at(0), probe) for probe in probes]
    u, expected = np.eye(2, dtype=complex), []
    for step in range(steps + 1):
        t = step * dt
        expected.append([_trace(u, rho, _numpy_j(h_at(t), v_at(t), probe), j0)
                         for probe, j0 in zip(probes, currents0)])
        u = _numpy_rk4(u, t, dt, h_at)
    np.testing.assert_allclose(result.observables["current_correlation"], expected, atol=2e-13)
    np.testing.assert_allclose(result.final_state.q, [q_at(steps * dt)], atol=2e-14)


@pytest.mark.parametrize("lf", [False, True])
def test_transport_batch_chunk_stream_and_restart_parity(lf, tmp_path):
    params = _params()
    if lf:
        problem = make_polaron_transport_problem(
            SmallTransportModel(), params, HarmonicBath([params["omega"]]), [1.7], [0.4], 1.2,
            hopping_pairs=[[0, 1], [1, 0]], thermal_policy="offdiagonal",
        )
        def initialize(q, p, i):
            return initialize_polaron_transport_state(problem, [q], [p], trajectory_id=i)
    else:
        problem = make_transport_problem(SmallTransportModel(), params, HarmonicBath([params["omega"]]))
        def initialize(q, p, i):
            return initialize_transport_state(problem, [q], [p], 1.2, trajectory_id=i)
    states = [initialize(q, p, i) for i, (q, p) in enumerate([(0.3, -0.2), (-0.7, 0.1), (0.9, 0.4)])]
    batch = stack_states(states)
    integrator = Integrator(0.015)
    sim = Simulation(problem, integrator, Execution(chunk_size=6))
    whole = sim.run(batch, 23)
    first = sim.run(batch, 9)
    checkpoint = tmp_path / "transport.h5"
    metadata = {"recipe": "lf_cpa" if lf else "thermal_cpa", "beta": 1.2}
    save_checkpoint(checkpoint, first.final_state, metadata=metadata)
    restored, restored_metadata = load_checkpoint(checkpoint, expected_metadata=metadata)
    assert restored_metadata == metadata
    second = Simulation(problem, integrator, Execution(chunk_size=11)).run(restored, 14)
    stitched = np.concatenate([first.observables["current_correlation"], second.observables["current_correlation"][1:]])
    np.testing.assert_allclose(stitched, whole.observables["current_correlation"], atol=2e-13)
    np.testing.assert_allclose(second.final_state.electronic, whole.final_state.electronic, atol=2e-13)
    np.testing.assert_allclose(second.final_state.method_state["transport"]["time0"], 0)
    assert not np.allclose(batch.method_state["transport"]["rho0"][0], batch.method_state["transport"]["rho0"][1])
    for i, initial in enumerate(states):
        single = Simulation(problem, integrator, Execution(chunk_size=10)).run(initial, 23)
        np.testing.assert_allclose(single.observables["current_correlation"], whole.observables["current_correlation"][:, i], atol=2e-13)
    chunks = []
    stream = sim.run(batch, 23, observer=lambda t, values: chunks.append(values["current_correlation"]), collect=False)
    assert stream.observables == {}
    np.testing.assert_allclose(np.concatenate(chunks), whole.observables["current_correlation"], atol=2e-13)


@pytest.mark.parametrize("policy", ["legacy_full", "offdiagonal"])
def test_lf_transport_policy_against_four_index_reference(policy):
    params, q, beta, frequency, coupling = _params(), 0.3, 1.2, 1.7, 0.8
    problem = make_polaron_transport_problem(
        SmallTransportModel(), params, PrescribedPath(ConstantPath([q])), [frequency], [coupling], beta,
        hopping_pairs=[[0, 1], [1, 0]], thermal_policy=policy,
    )
    state = initialize_polaron_transport_state(problem, [q], [0.0], time=1.7)
    result = Simulation(problem, Integrator(0.02, "exponential_midpoint"), Execution(chunk_size=7)).run(state, 20)
    h = _numpy_h(params, q)
    phi0 = (coupling / frequency) ** 2 / np.tanh(beta * frequency / 2)
    factor = np.exp(-phi0)
    h_dressed = factor * h
    np.fill_diagonal(h_dressed, np.diag(h))
    h_initial = factor * h if policy == "legacy_full" else h_dressed
    rho, current = _numpy_rho(h_initial, beta), _numpy_j(h, 0, "current_x")
    np.testing.assert_allclose(state.method_state["transport"]["rho0"], rho, atol=2e-13)
    expected = []
    for absolute_time in result.times:
        t = absolute_time - 1.7
        phit = (coupling / frequency) ** 2 * (
            np.cos(frequency * t) / np.tanh(beta * frequency / 2) - 1j * np.sin(frequency * t))
        expected.append(_lf_trace(expm(-1j * t * h_dressed), rho, current, current, phi0, phit))
    np.testing.assert_allclose(result.observables["current_correlation"][:, 0], expected, atol=2e-13)


def test_lf_initial_diagonal_policy_has_observable_effect():
    model, params, bath = SmallTransportModel(), _params(), HarmonicBath([0.9])
    states = []
    for policy in ["legacy_full", "offdiagonal"]:
        problem = make_polaron_transport_problem(model, params, bath, [1.7], [1.0], 2.0,
                                                 hopping_pairs=[[0, 1], [1, 0]], thermal_policy=policy)
        states.append(initialize_polaron_transport_state(problem, [0.3], [0.0]))
    assert not np.allclose(states[0].method_state["transport"]["rho0"], states[1].method_state["transport"]["rho0"])


def test_legacy_callback_conversion_and_empty_lf_reduce_to_bare():
    model, params, bath = SmallTransportModel(), _params(), HarmonicBath([0.9])
    def callback(p, ctx, probe):
        return model.probe_apply(p, ctx, probe, jnp.eye(2)) / 1j
    bare = make_transport_problem(model, params, bath)
    imported = make_transport_problem(model, params, bath, probe_callback=callback,
                                      current_convention="legacy_without_i")
    lf = make_polaron_transport_problem(model, params, bath, [], [], 1.2,
                                       hopping_pairs=[[0, 1], [1, 0]])
    outputs = []
    for problem in [bare, imported, lf]:
        state = initialize_transport_state(problem, [0.3], [-0.2], 1.2)
        outputs.append(Simulation(problem, Integrator(0.01)).run(state, 10).observables["current_correlation"])
    np.testing.assert_allclose(outputs[0], outputs[1], atol=2e-13)
    np.testing.assert_allclose(outputs[0], outputs[2], atol=2e-13)


def test_workflow_rejects_inconsistent_preparation_and_feedback():
    model, params = SmallTransportModel(), _params()
    bare = make_transport_problem(model, params, HarmonicBath([0.9]))
    with pytest.raises(ValueError, match="CPA"):
        replace(bare, method=Ehrenfest(), nuclear_treatment=CoupledClassical(1)).validate()
    with pytest.raises(ValueError, match="required physical probes"):
        make_transport_problem(model, params, HarmonicBath([0.9]), probes=("unknown",))
    path_problem = make_transport_problem(model, params, PrescribedPath(ConstantPath([0.0])))
    with pytest.raises(ValueError, match="agree with the prescribed path"):
        initialize_transport_state(path_problem, [1.0], [0.0], 1.2)
    def bad_callback(p, ctx, probe):
        return jnp.array([[0, 1], [-1, 0]])
    bad = make_transport_problem(model, params, HarmonicBath([0.9]), probe_callback=bad_callback)
    with pytest.raises(ValueError, match="Hermitian"):
        initialize_transport_state(bad, [0.0], [0.0], 1.2)
    lf = make_polaron_transport_problem(model, params, HarmonicBath([0.9]), [1.7], [0.4], 1.2,
                                       hopping_pairs=[[0, 1], [1, 0]])
    with pytest.raises(ValueError, match="quantum-bath inverse temperature"):
        initialize_transport_state(lf, [0.0], [0.0], 2.0)
    with pytest.raises(ValueError, match="thermal_policy"):
        make_polaron_transport_problem(model, params, HarmonicBath([0.9]), [1.7], [0.4], 1.2,
                                       hopping_pairs=[[0, 1], [1, 0]], thermal_policy="automatic")
    incomplete = make_polaron_transport_problem(model, params, HarmonicBath([0.9]), [1.7], [0.4], 1.2,
                                                hopping_pairs=[[0, 1]])
    with pytest.raises(ValueError, match="omit nonzero initial current edges"):
        initialize_polaron_transport_state(incomplete, [0.0], [0.0])
