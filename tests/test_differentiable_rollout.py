"""Smooth trajectory sensitivities against independent differential equations."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.integrate import solve_ivp

from pyeph import (
    CPA, CoupledClassical, Ehrenfest, Execution, Integrator, LanczosOptions, MASH2,
    Problem, Simulation, make_state, stack_states,
)
from pyeph.core.contracts import ModelSpec
from pyeph.core.problem import PrescribedPath
from pyeph.core.system import SystemSpec
from pyeph.execution._validation import validate_run_span
from pyeph.execution.differentiable import DifferentiableRollout
from pyeph.models.base import AutoDiffModel
from pyeph.models.composite import SumModel
from pyeph.models.neural import NeuralResidualModel
from pyeph.observables.population import FunctionalMeasurement
from pyeph.paths.harmonic import HarmonicBath
from pyeph.paths.nuclear import RecordedNuclearPath


class SmoothModel(AutoDiffModel):
    spec = ModelSpec(SystemSpec(2, (1,), coordinate_kind="normal_mode"), name="smooth-fixture")

    def matrix(self, params, q):
        diagonal = params["bias"] + params["slope"] * q[0] + .04 * q[0]**2
        coupling = params["coupling"] * jnp.exp(-.3 * q[0]**2)
        return jnp.array([[diagonal, coupling], [coupling, -diagonal]])

    def apply(self, params, q, vectors):
        return self.matrix(params, q) @ vectors

    def reference_energy(self, params, q):
        return .5 * params["spring"] * q[0]**2 + .025 * q[0]**4


def measure(problem, state):
    energy = (state.p[0]**2 / (2*1.4) + problem.model.reference_energy(problem.params, state.q)
              + jnp.vdot(state.electronic,
                         problem.model.apply(problem.params, state.q, state.electronic)).real)
    return dict(q=state.q, p=state.p, population=jnp.abs(state.electronic)**2, energy=energy)


def inputs(theta):
    params = dict(bias=jnp.asarray(.31), slope=theta[0], coupling=theta[1], spring=theta[2])
    return params, jnp.array([theta[3]]), jnp.array([theta[4]]), jnp.array([
        jnp.cos(theta[5]), jnp.sin(theta[5]) * jnp.exp(.3j)])


THETA = np.array([.45, .7, .6, .35, .21, .4])


def fixture(method):
    params, q, p, c = inputs(jnp.asarray(THETA))
    nuclei = HarmonicBath(.7, 1.4) if method == "cpa" else CoupledClassical(1.4)
    dynamics = CPA() if method == "cpa" else Ehrenfest()
    problem = Problem(SmoothModel(), params, nuclei, dynamics, FunctionalMeasurement(measure))
    initial = make_state(q, p, c, time=.7, step=3, trajectory_id=19)
    return problem, initial


def recorded_fixture(times):
    problem, initial = fixture("cpa")
    path = RecordedNuclearPath([0., 1.], [[0.], [1.]], [[1.], [1.]])
    problem = replace(problem, nuclear_treatment=PrescribedPath(path))
    states = [initial._replace(q=path.position(time), time=jnp.asarray(time),
                               trajectory_id=jnp.uint32(index))
              for index, time in enumerate(times)]
    return problem, states[0] if len(states) == 1 else stack_states(states)


@pytest.mark.parametrize("times", [(.9,), (.7, .9)])
def test_recorded_requested_span_rejected_before_tracing_or_output(monkeypatch, times):
    problem, initial = recorded_fixture(times)
    integrator = Integrator(.1)
    calls = []

    def unexpected_trace(*args):
        calls.append(args)
        pytest.fail("an invalid requested span must be rejected before tracing")

    monkeypatch.setattr("pyeph.execution.differentiable._trace_programs", unexpected_trace)
    with pytest.raises(ValueError, match="outside recorded domain"):
        DifferentiableRollout(problem, integrator, initial, steps=2)
    assert calls == []
    received = []
    simulation = Simulation(problem, integrator)
    with pytest.raises(ValueError, match="outside recorded domain"):
        simulation.run(initial, 2, observer=lambda times, values: received.append(values))
    assert received == [] and simulation._compiled == {}


@pytest.mark.parametrize("times", [(.8,), (.7, .8)])
def test_recorded_requested_span_accepts_closed_endpoint_and_matches_runner(times):
    problem, initial = recorded_fixture(times)
    integrator = Integrator(.1)
    rollout = DifferentiableRollout(problem, integrator, initial, steps=2)
    actual = rollout(problem.params, initial)
    expected = Simulation(problem, integrator).run(initial, 2)
    assert all(np.isfinite(value).all() for value in jax.tree.leaves(actual))
    np.testing.assert_allclose(actual.final_state.time, np.asarray(times).squeeze() + .2,
                               atol=1e-14, rtol=0)
    for left, right in zip(jax.tree.leaves(actual.final_state),
                           jax.tree.leaves(expected.final_state), strict=True):
        np.testing.assert_allclose(left, right, atol=1e-14, rtol=0)


@pytest.mark.parametrize("failure", ["counter", "time"])
def test_requested_counter_and_clock_overflow_rejected_before_tracing_or_output(monkeypatch, failure):
    problem, initial = fixture("ehrenfest")
    if failure == "counter":
        initial = initial._replace(step=jnp.asarray(np.iinfo(np.int32).max - 1, dtype=jnp.int32))
    integrator = Integrator(np.finfo(np.float64).max if failure == "time" else .1)

    def unexpected_trace(*args):
        pytest.fail("requested overflow must be rejected before tracing")

    monkeypatch.setattr("pyeph.execution.differentiable._trace_programs", unexpected_trace)
    with pytest.raises(ValueError, match="overflow"):
        DifferentiableRollout(problem, integrator, initial, steps=2)
    received = []
    simulation = Simulation(problem, integrator)
    with pytest.raises(ValueError, match="overflow"):
        simulation.run(initial, 2, observer=lambda times, values: received.append(values))
    assert received == [] and simulation._compiled == {}


@pytest.mark.parametrize("method,batched,origin", [
    ("cpa", False, 1e20), ("ehrenfest", False, -1e20),
    ("cpa", True, -1e20), ("ehrenfest", True, 1e20),
])
def test_unresolvable_macro_clock_rejected_before_tracing_or_output(
        monkeypatch, method, batched, origin):
    problem, initial = fixture(method)
    stagnant = initial._replace(time=jnp.asarray(origin), trajectory_id=jnp.uint32(20))
    initial = stack_states([initial, stagnant]) if batched else stagnant
    integrator = Integrator(.01, electronic_substeps=2)
    assert np.float64(origin) + integrator.dt == origin

    def unexpected_trace(*args):
        pytest.fail("an unresolvable clock must be rejected before tracing")

    monkeypatch.setattr("pyeph.execution.differentiable._trace_programs", unexpected_trace)
    with pytest.raises(ValueError, match="clock resolution.*shift the time origin"):
        DifferentiableRollout(problem, integrator, initial, steps=2)
    received = []
    simulation = Simulation(problem, integrator)
    with pytest.raises(ValueError, match="clock resolution.*shift the time origin"):
        simulation.run(initial, 2, observer=lambda times, values: received.append(values))
    assert received == [] and simulation._compiled == {}


@pytest.mark.parametrize("algorithm", ["rk4", "exponential_midpoint", LanczosOptions()])
def test_cpa_stage_alias_rejected_even_when_macro_clock_advances(monkeypatch, algorithm):
    problem, initial = fixture("cpa")
    origin = float(2**46)
    initial = initial._replace(time=jnp.asarray(origin))
    integrator = Integrator(.03125, algorithm, electronic_substeps=2)
    macro = origin + np.arange(3) * integrator.dt
    stages = origin + np.arange(5) * (integrator.dt / 4)
    assert np.all(np.diff(macro) > 0)
    assert np.any(np.diff(stages) == 0)

    def unexpected_trace(*args):
        pytest.fail("aliased CPA stages must be rejected before tracing")

    monkeypatch.setattr("pyeph.execution.differentiable._trace_programs", unexpected_trace)
    if algorithm == "rk4":
        with pytest.raises(ValueError, match="clock resolution"):
            DifferentiableRollout(problem, integrator, initial, steps=2)
    received = []
    simulation = Simulation(problem, integrator)
    with pytest.raises(ValueError, match="clock resolution"):
        simulation.run(initial, 2, observer=lambda times, values: received.append(values))
    assert received == [] and simulation._compiled == {}


@pytest.mark.parametrize("times", [(-.15,), (-.15, -0., 0., 10000.)])
def test_resolved_negative_zero_and_large_origins_match_operational_runner(times):
    problem, initial = fixture("cpa")
    states = [initial._replace(time=jnp.asarray(time), trajectory_id=jnp.uint32(index))
              for index, time in enumerate(times)]
    initial = states[0] if len(states) == 1 else stack_states(states)
    integrator = Integrator(.1, electronic_substeps=2)
    actual = DifferentiableRollout(problem, integrator, initial, steps=3)(problem.params, initial)
    expected = Simulation(problem, integrator).run(initial, 3)
    assert np.all(np.diff(actual.times, axis=0) > 0)
    for left, right in zip(jax.tree.leaves(actual),
                           jax.tree.leaves((expected.final_state, expected.times,
                                           expected.observables)), strict=True):
        np.testing.assert_allclose(left, right, atol=2e-14, rtol=0)


def test_zero_steps_preserve_large_origins_and_signed_zero():
    problem, initial = fixture("cpa")
    times = [-1e20, -0., 0., 1e20]
    initial = stack_states([initial._replace(time=jnp.asarray(time), trajectory_id=jnp.uint32(i))
                            for i, time in enumerate(times)])
    integrator = Integrator(.01, electronic_substeps=2)
    actual = DifferentiableRollout(problem, integrator, initial, steps=0)(problem.params, initial)
    expected = Simulation(problem, integrator).run(initial, 0)
    for result in (actual, expected):
        assert np.asarray(result.final_state.time).tobytes() == np.asarray(initial.time).tobytes()
        np.testing.assert_array_equal(result.final_state.electronic, initial.electronic)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_clock_resolution_uses_the_stored_supported_dtype(dtype):
    problem, initial = fixture("cpa")
    integrator = Integrator(.01, electronic_substeps=2)
    normal = initial._replace(time=np.asarray([-1., -0., 0., 1.], dtype=dtype))
    validate_run_span(problem.nuclear_treatment, integrator, normal, 4)
    unresolved = initial._replace(time=np.asarray(1e20, dtype=dtype))
    with pytest.raises(ValueError, match=f"{np.dtype(dtype)} clock resolution"):
        validate_run_span(problem.nuclear_treatment, integrator, unresolved, 2)


@pytest.mark.parametrize("dtype", [np.int64, np.float16, jnp.bfloat16])
def test_nonzero_clock_dtype_support_is_explicit(dtype):
    problem, initial = fixture("cpa")
    initial = initial._replace(time=np.asarray(0, dtype=dtype))
    # Zero-step host inspection does not need a representable propagation grid.
    validate_run_span(problem.nuclear_treatment, Integrator(.1), initial, 0)
    with pytest.raises(ValueError, match="float32 or float64 time"):
        validate_run_span(problem.nuclear_treatment, Integrator(.1), initial, 1)


def test_clock_resolution_checks_span_end_and_substep_underflow_without_step_enumeration():
    problem, initial = fixture("cpa")
    initial = initial._replace(time=np.asarray(0, dtype=np.float32))
    # The first macrostep advances, but a long span loses the required spacing.
    with pytest.raises(ValueError, match="clock resolution"):
        validate_run_span(problem.nuclear_treatment, Integrator(1.), initial, 2**24)
    with pytest.raises(ValueError, match="clock resolution"):
        validate_run_span(problem.nuclear_treatment,
                          Integrator(1e-40, electronic_substeps=10**7), initial, 1)
    # Large step counts and near-limit finite coordinates require no time grid allocation.
    initial = initial._replace(time=np.asarray([-1e6, 0., 1e6], dtype=np.float64))
    validate_run_span(problem.nuclear_treatment, Integrator(.01), initial, 10**8)
    initial = initial._replace(time=np.asarray([-1e300, 0., 1e300], dtype=np.float64))
    validate_run_span(problem.nuclear_treatment, Integrator(1e299), initial, 2)


def terminal_loss(result):
    values = result.observables
    return (.7*values["population"][-1, 1] + .2*values["q"][-1, 0]**2
            + .1*values["p"][-1, 0]**2 + .03*values["energy"][-1])


def objective(rollout, initial):
    def loss(theta):
        params, q, p, c = inputs(theta)
        return terminal_loss(rollout(params, initial._replace(q=q, p=p, electronic=c)))
    return loss


def independent_loss(theta, method, duration):
    slope, coupling, spring, q0, p0, angle = theta
    c0 = np.array([np.cos(angle), np.sin(angle)*np.exp(.3j)])
    def matrix(q):
        d = .31 + slope*q + .04*q*q
        v = coupling*np.exp(-.3*q*q)
        return np.array([[d, v], [v, -d]])
    def rhs(time, state):
        q, p = state[:2]
        c = state[2:4] + 1j*state[4:6]
        dc = -1j*matrix(q)@c
        if method == "cpa":
            force = -1.4*.7**2*q
        else:
            electronic = ((slope+.08*q)*(abs(c[0])**2-abs(c[1])**2)
                          - .6*q*coupling*np.exp(-.3*q*q)*2*np.real(np.conj(c[0])*c[1]))
            force = -spring*q-.1*q**3-electronic
        return np.r_[p/1.4, force, dc.real, dc.imag]
    solved = solve_ivp(rhs, (0, duration), np.r_[q0, p0, c0.real, c0.imag],
                       method="DOP853", rtol=2e-13, atol=2e-14)
    assert solved.success
    q, p = solved.y[:2, -1]
    c = solved.y[2:4, -1]+1j*solved.y[4:6, -1]
    energy = p*p/(2*1.4)+.5*spring*q*q+.025*q**4+np.vdot(c, matrix(q)@c).real
    return .7*abs(c[1])**2+.2*q*q+.1*p*p+.03*energy


def finite_gradient(function, theta, step=2e-5):
    result = []
    for index in range(len(theta)):
        direction = np.eye(len(theta))[index]*step
        result.append((function(theta+direction)-function(theta-direction))/(2*step))
    return np.asarray(result)


@pytest.mark.parametrize("method", ["cpa", "ehrenfest"])
@pytest.mark.parametrize("batched", [False, True])
def test_values_match_operational_runner_and_parameters_are_runtime(method, batched):
    problem, initial = fixture(method)
    if batched:
        second = make_state(initial.q+.1, initial.p-.03, initial.electronic,
                            time=1.1, step=3, trajectory_id=71)
        initial = stack_states([initial, second])
    integrator = Integrator(.06, "rk4", electronic_substeps=2)
    rollout = DifferentiableRollout(problem, integrator, initial, steps=5)
    evaluate = jax.jit(rollout)
    for coupling in (.7, .82):
        params = problem.params | dict(coupling=jnp.asarray(coupling))
        actual = evaluate(params, initial)
        expected = Simulation(replace(problem, params=params), integrator,
                              Execution(chunk_size=5, save_every=1)).run(initial, 5)
        np.testing.assert_allclose(actual.times, expected.times, atol=1e-15)
        for a, b in zip(jax.tree.leaves(actual.final_state), jax.tree.leaves(expected.final_state), strict=True):
            np.testing.assert_allclose(a, b, atol=2e-14)
        for key in expected.observables:
            np.testing.assert_allclose(actual.observables[key], expected.observables[key], atol=2e-14)


@pytest.mark.parametrize("method", ["cpa", "ehrenfest"])
def test_parameter_coordinate_electronic_and_observable_loss_gradients_match_fd(method):
    problem, initial = fixture(method)
    rollout = DifferentiableRollout(problem, Integrator(.1), initial, steps=9)
    loss = jax.jit(objective(rollout, initial))
    reverse = jax.jit(jax.grad(loss))(jnp.asarray(THETA))
    expected = finite_gradient(loss, THETA)
    np.testing.assert_allclose(reverse, expected, atol=3e-9, rtol=2e-7)
    tangent = jnp.array([.2, -.1, .3, .4, -.2, .1])
    _, forward = jax.jvp(loss, (jnp.asarray(THETA),), (tangent,))
    np.testing.assert_allclose(forward, reverse@tangent, atol=3e-13)
    # The initial electronic angle must influence subsequent nuclear force.
    if method == "ehrenfest":
        assert abs(float(reverse[-1])) > .05
    target = jnp.zeros((10, 2))
    def population_loss(coupling):
        result = rollout(problem.params | dict(coupling=coupling), initial)
        return jnp.mean((result.observables["population"]-target)**2)
    derivative = jax.grad(population_loss)(problem.params["coupling"])
    step = 1e-5
    expected = (population_loss(.7+step)-population_loss(.7-step))/(2*step)
    np.testing.assert_allclose(derivative, expected, atol=1e-9, rtol=1e-7)


@pytest.mark.parametrize("method,minimum_order", [("cpa", 3.5), ("ehrenfest", 1.8)])
def test_gradients_converge_to_independent_continuum_equations(method, minimum_order):
    problem, initial = fixture(method)
    exact = finite_gradient(lambda theta: independent_loss(theta, method, 1.2), THETA, step=1e-5)
    errors = []
    for steps in (4, 8, 16):
        rollout = DifferentiableRollout(problem, Integrator(1.2/steps), initial, steps=steps)
        actual = jax.jit(jax.grad(objective(rollout, initial)))(jnp.asarray(THETA))
        errors.append(np.linalg.norm(np.asarray(actual)-exact))
    orders = np.log2(np.asarray(errors[:-1])/np.asarray(errors[1:]))
    assert min(orders) > minimum_order, (errors, orders)
    assert errors[-1] < (2e-5 if method == "cpa" else 2e-3)


def test_rematerialization_preserves_values_gradients_and_batch_mean_sensitivities():
    problem, initial = fixture("ehrenfest")
    plain = DifferentiableRollout(problem, Integrator(.1), initial, steps=7)
    checkpointed = DifferentiableRollout(problem, Integrator(.1), initial, steps=7, rematerialize=True)
    for differentiated in (False, True):
        first, second = objective(plain, initial), objective(checkpointed, initial)
        if differentiated:
            first, second = jax.grad(first), jax.grad(second)
        np.testing.assert_allclose(jax.jit(first)(THETA), jax.jit(second)(THETA), atol=3e-14)
    second = make_state(initial.q+.1, initial.p, initial.electronic, step=3, trajectory_id=31)
    states = stack_states([initial, second])
    batch = DifferentiableRollout(problem, Integrator(.1), states, steps=7, rematerialize=True)
    def batched_loss(coupling):
        return jnp.mean(batch(problem.params | dict(coupling=coupling), states).observables["q"][-1]**2)
    def singles_loss(coupling):
        params = problem.params | dict(coupling=coupling)
        return sum(jnp.sum(plain(params, state).observables["q"][-1]**2) for state in (initial, second))/2
    np.testing.assert_allclose(jax.grad(batched_loss)(.7), jax.grad(singles_loss)(.7), atol=3e-14)


def test_zero_steps_and_preflight_validation_are_explicit():
    problem, initial = fixture("cpa")
    rollout = DifferentiableRollout(problem, Integrator(.1), initial, steps=0)
    actual = jax.jit(rollout)(problem.params, initial)
    np.testing.assert_array_equal(actual.times, [initial.time])
    np.testing.assert_array_equal(actual.final_state.electronic, initial.electronic)
    with pytest.raises(ValueError, match="nonfinite"):
        rollout.preflight(problem.params, initial._replace(q=jnp.array([jnp.nan])))
    with pytest.raises(ValueError, match="batching"):
        rollout.preflight(problem.params, stack_states([initial]))


@pytest.mark.parametrize("algorithm", ["exponential_midpoint", LanczosOptions(max_dimension=2)])
def test_unqualified_electronic_solvers_are_rejected(algorithm):
    problem, initial = fixture("cpa")
    with pytest.raises(ValueError, match="requires RK4"):
        DifferentiableRollout(problem, Integrator(.1, algorithm), initial, steps=2)


def test_event_methods_and_declared_external_models_are_rejected():
    problem, initial = fixture("ehrenfest")
    with pytest.raises(ValueError, match="CPA and Ehrenfest"):
        DifferentiableRollout(replace(problem, method=MASH2()), Integrator(.1), initial, steps=2)
    class External(SmoothModel):
        spec = replace(SmoothModel.spec, native_jax=False)
    with pytest.raises(ValueError, match="native JAX"):
        DifferentiableRollout(replace(problem, model=External()), Integrator(.1), initial, steps=2)


def test_hidden_host_callback_is_rejected_during_trace_without_execution():
    problem, initial = fixture("cpa")
    calls = []
    class Callback(SmoothModel):
        def apply(self, params, q, vectors):
            jax.debug.callback(lambda value: calls.append(value), q)
            return super().apply(params, q, vectors)
    with pytest.raises(ValueError, match="callbacks"):
        DifferentiableRollout(replace(problem, model=Callback()), Integrator(.1), initial, steps=2)
    assert not calls


def test_derivative_only_callback_is_rejected_without_executing_it():
    calls = []
    @jax.custom_jvp
    def guarded(value):
        return value
    @guarded.defjvp
    def derivative(primals, tangents):
        value, = primals
        direction, = tangents
        jax.debug.callback(lambda x: calls.append(x), value)
        return value, direction
    problem, initial = fixture("cpa")
    def measurement(problem, state):
        return {"population": guarded(jnp.abs(state.electronic)**2)}
    problem = replace(problem, measurement=FunctionalMeasurement(measurement))
    with pytest.raises(ValueError, match="callbacks"):
        DifferentiableRollout(problem, Integrator(.1), initial, steps=3)
    assert not calls


def test_preflight_rejects_static_mutation_before_reusing_a_compiled_rollout():
    class MutableModel(SmoothModel):
        def __init__(self):
            self.h = np.array([[.2, .3], [.3, -.2]])
        def matrix(self, params, q):
            return super().matrix(params, q) + jnp.asarray(self.h)
    problem, initial = fixture("cpa")
    model = MutableModel()
    problem = replace(problem, model=model)
    rollout = DifferentiableRollout(problem, Integrator(.1), initial, steps=3)
    compiled = jax.jit(rollout)
    before = compiled(problem.params, initial)
    rollout.preflight(problem.params, initial)
    rollout.preflight(problem.params | dict(coupling=jnp.asarray(.8)), initial)
    model.h[0, 1] = model.h[1, 0] = 2.
    with pytest.raises(ValueError, match="static rollout computation changed"):
        rollout.preflight(problem.params, initial)
    updated = DifferentiableRollout(problem, Integrator(.1), initial, steps=3)
    after = jax.jit(updated)(problem.params, initial)
    assert not np.allclose(before.observables["population"], after.observables["population"])


def test_neural_residual_weights_differentiate_through_complete_feedback_force():
    problem, initial = fixture("ehrenfest")
    neural = NeuralResidualModel(2, (1,), hidden_sizes=(3,), coordinate_kind="normal_mode")
    weights = neural.init_params(jax.random.key(23), zero_last=False, scale=.2)
    composed = replace(problem, model=SumModel((problem.model, neural)), params=(problem.params, weights))
    rollout = DifferentiableRollout(composed, Integrator(.08), initial, steps=8, rematerialize=True)
    def loss(params):
        result = rollout(params, initial)
        return terminal_loss(result)
    tangent = jax.tree.map(lambda value: jnp.ones_like(value)*.07, composed.params)
    derivative = jax.grad(loss)(composed.params)
    actual = sum(jnp.vdot(a, b).real for a, b in zip(jax.tree.leaves(derivative),
                 jax.tree.leaves(tangent), strict=True))
    delta = 2e-5
    plus = jax.tree.map(lambda a, b: a+delta*b, composed.params, tangent)
    minus = jax.tree.map(lambda a, b: a-delta*b, composed.params, tangent)
    expected = (loss(plus)-loss(minus))/(2*delta)
    np.testing.assert_allclose(actual, expected, atol=3e-9, rtol=2e-7)
    assert np.linalg.norm(np.asarray(derivative[1]["layers"][0]["weight"])) > 1e-5


def test_rk4_gradient_remains_regular_at_an_exact_electronic_degeneracy():
    problem, initial = fixture("cpa")
    params = problem.params | dict(bias=jnp.asarray(0.), coupling=jnp.asarray(0.))
    initial = initial._replace(q=jnp.zeros(1), p=jnp.zeros(1))
    rollout = DifferentiableRollout(replace(problem, params=params), Integrator(.1), initial, steps=9)
    def population(coupling):
        return rollout(params | dict(coupling=coupling), initial).observables["population"][-1, 1]
    actual = jax.grad(population)(0.)
    c = np.asarray(initial.electronic)
    expected = 2*np.real(np.conj(c[1])*(-1j*.9*c[0]))
    np.testing.assert_allclose(actual, expected, atol=2e-14)
