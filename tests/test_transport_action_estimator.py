"""Exact current actions preserve complex traces, gradients, and extension hooks."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pyeph.core.contracts import ModelSpec
from pyeph.core.state import make_state
from pyeph.core.system import SystemSpec
from pyeph.models.base import AutoDiffModel
from pyeph.paths.harmonic import HarmonicBath
from pyeph.workflows.transport import TransportMeasurement, make_transport_problem


class CurrentActionModel(AutoDiffModel):
    spec = ModelSpec(SystemSpec(3, (2,), coordinate_kind="normal_mode"),
                     name="complex_current_action_reference", complex_valued=True,
                     probes=("current_x", "current_y"))

    def __init__(self, legacy=False):
        self.legacy = legacy

    def apply(self, params, q, vectors):
        return params["hamiltonian"] @ vectors

    def reference_energy(self, params, q):
        return jnp.sum(q*q)/2

    def probe_apply(self, params, context, probe, vectors):
        index = self.spec.probes.index(probe)
        current = (params["currents"][index] + context.q[0]*params["derivatives"][index]
                   + (context.time + context.velocity[1])*jnp.eye(3))
        current = params["scale"]*current
        return (current/1j if self.legacy else current) @ vectors


def fixture(*, legacy=False):
    rng = np.random.default_rng(60121)

    def hermitian(shape):
        raw = rng.normal(size=shape) + 1j*rng.normal(size=shape)
        return (raw + raw.conj().swapaxes(-1, -2))/2

    params = {"hamiltonian": jnp.asarray(hermitian((3, 3))),
              "currents": jnp.asarray(hermitian((2, 3, 3))),
              "derivatives": jnp.asarray(hermitian((2, 3, 3))), "scale": .7}
    factor = rng.normal(size=(3, 2)) + 1j*rng.normal(size=(3, 2))
    density = factor @ factor.conj().T
    density /= np.trace(density)
    # Deliberately nonunitary: the estimator must use U†, not assume U^-1.
    unitary = rng.normal(size=(3, 3)) + 1j*rng.normal(size=(3, 3))
    state = make_state([.23, -.31], [-.1, .28], unitary, time=.61,
                       method_state={"transport": {"rho0": jnp.asarray(density),
                           "currents0": jnp.asarray(hermitian((2, 3, 3)))}})
    model = CurrentActionModel(legacy)
    problem = make_transport_problem(model, params, HarmonicBath([.8, .9]),
                                     probes=model.spec.probes,
                                     current_convention="legacy_without_i" if legacy else "physical")
    return problem, state


def numpy_reference(problem, state, *, current_multiplier=1.):
    params = problem.params
    u, rho = np.asarray(state.electronic), np.asarray(state.method_state["transport"]["rho0"])
    values = []
    for index in range(2):
        current = current_multiplier*params["scale"]*(
            np.asarray(params["currents"][index]) + float(state.q[0])*np.asarray(params["derivatives"][index])
            + (float(state.time) + float(state.p[1]))*np.eye(3))
        initial = np.asarray(state.method_state["transport"]["currents0"][index])
        values.append(np.trace(current @ u @ initial @ rho @ u.conj().T))
    return np.asarray(values)


@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize("compiled", [False, True])
def test_action_estimator_matches_independent_complex_nonunitary_trace(legacy, compiled):
    problem, state = fixture(legacy=legacy)

    def evaluate(s):
        return problem.measurement.evaluate(problem, s)

    actual = (jax.jit(evaluate) if compiled else evaluate)(state)
    expected = numpy_reference(problem, state)
    assert np.max(np.abs(expected.imag)) > .1
    np.testing.assert_allclose(actual["current_correlation"], expected, atol=2e-13, rtol=2e-13)
    expected_unitarity = np.max(abs(np.asarray(state.electronic).conj().T
                                   @ np.asarray(state.electronic) - np.eye(3)))
    np.testing.assert_allclose(actual["unitary_error"], expected_unitarity, atol=2e-13)


def test_action_estimator_geometry_and_parameter_gradients_match_finite_differences():
    problem, state = fixture()

    def objective(q0, scale):
        varied = replace(problem, params={**problem.params, "scale": scale})
        changed = state._replace(q=state.q.at[0].set(q0))
        value = varied.measurement.evaluate(varied, changed)["current_correlation"]
        return jnp.sum(value.real + .37*value.imag)

    def reference(q0, scale):
        varied = replace(problem, params={**problem.params, "scale": scale})
        value = numpy_reference(varied, state._replace(q=state.q.at[0].set(q0)))
        return np.sum(value.real + .37*value.imag)

    q0, scale, epsilon = float(state.q[0]), problem.params["scale"], 1e-5
    actual = jax.jit(jax.grad(objective, argnums=(0, 1)))(q0, scale)
    expected = [(reference(q0 + epsilon, scale) - reference(q0 - epsilon, scale))/(2*epsilon),
                (reference(q0, scale + epsilon) - reference(q0, scale - epsilon))/(2*epsilon)]
    np.testing.assert_allclose(actual, expected, atol=2e-9, rtol=2e-9)


@pytest.mark.parametrize("extension", ["matrix_callback", "currents_override"])
def test_existing_current_extension_hooks_retain_their_physical_definition(extension):
    problem, state = fixture()

    if extension == "matrix_callback":
        def callback(params, context, probe):
            return 2*problem.model.probe_apply(params, context, probe, jnp.eye(3))
        measurement = TransportMeasurement(probes=problem.measurement.probes, probe_callback=callback)
    else:
        class ModifiedCurrents(TransportMeasurement):
            def currents(self, problem, state):
                return 2*super().currents(problem, state)
        measurement = ModifiedCurrents(probes=problem.measurement.probes)
    problem = replace(problem, measurement=measurement)
    actual = jax.jit(lambda s: measurement.evaluate(problem, s))(state)
    np.testing.assert_allclose(actual["current_correlation"],
                               numpy_reference(problem, state, current_multiplier=2.), atol=2e-13)
