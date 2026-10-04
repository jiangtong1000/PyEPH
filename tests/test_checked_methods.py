"""Independent action references and transactional method-stage failure checks."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.integrate import solve_ivp
from scipy.linalg import expm

from pyeph.core.contracts import ModelSpec
from pyeph.core.problem import CoupledClassical, PrescribedPath, Problem
from pyeph.core.state import make_state, stack_states
from pyeph.core.system import SystemSpec
from pyeph.dynamics.checked import (
    PHASE_CPA,
    PHASE_EHRENFEST_FIRST,
    PHASE_EHRENFEST_SECOND,
    PHASE_ENDPOINT,
    PHASE_FORCE_FIRST,
    PHASE_FORCE_SECOND,
    empty_checked_info,
    finish_checked_step,
)
from pyeph.dynamics.cpa import CPA
from pyeph.dynamics.ehrenfest import Ehrenfest
from pyeph.integrators.electronic import Integrator
from pyeph.integrators.krylov import LanczosOptions, lanczos_action
from pyeph.models.base import AutoDiffModel
from pyeph.paths.harmonic import ConstantPath, HarmonicPath


class LinearModel(AutoDiffModel):
    spec = ModelSpec(SystemSpec(2, (1,), coordinate_kind="canonical"), name="checked-test")

    def apply(self, params, q, vectors):
        a, coupling, _ = params
        return jnp.array([[a * q[0], coupling], [coupling.conj(), -a * q[0]]]) @ vectors

    def reference_energy(self, params, q):
        return 0.5 * jnp.real(params[2]) * jnp.sum(q**2)


PARAMS = jnp.array([0.4, 0.23, 0.6])


def assert_same_state(actual, expected):
    for left, right in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
        np.testing.assert_array_equal(left, right)


def configured(dt=0.2, *, dimension=2, substeps=1):
    return Integrator(dt, LanczosOptions(max_dimension=dimension, atol=1e-11, rtol=1e-10),
                      electronic_substeps=substeps)


def state_at(q=0.7, p=0.3, *, identity=0, time=0.0, block=False):
    c = np.array([[1, 2 + 1j, 0], [0, -0.3j, 0]]) if block else [1, 0]
    return make_state([q], [p], c, time=time, step=7, trajectory_id=identity, seed=19,
                      method_state={"counter": np.array([3, 8], dtype=np.int32)})


@pytest.mark.parametrize("batch", [False, True])
@pytest.mark.parametrize("block", [False, True])
def test_empty_diagnostics_have_shapes_without_provider_calls(batch, block):
    state = state_at(block=block)
    if batch:
        state = stack_states([state, state_at(identity=1, block=block)])
    info = jax.jit(lambda s: empty_checked_info(s, batch=batch))(state)
    shape = ((2,) if batch else ()) + ((3,) if block else ())
    assert info.action.status.shape == shape
    assert info.macrostep_budget.shape == shape
    assert info.failed_trajectories.shape == ((2,) if batch else ())
    np.testing.assert_array_equal(info.action.value, state.electronic)
    assert int(info.substep) == -1


@pytest.mark.parametrize("batch", [False, True])
@pytest.mark.parametrize("block", [False, True])
def test_cpa_frozen_complex_actions_and_column_budgets(batch, block):
    states = [state_at(block=block, time=0.4), state_at(identity=1, block=block, time=1.7)]
    initial = stack_states(states) if batch else states[0]
    params = jnp.array([0.4, 0.23 + 0.12j, 0.6])
    problem = Problem(LinearModel(), params, PrescribedPath(ConstantPath([0.7])), CPA())
    integrator = configured(0.6, substeps=3)
    step = problem.method.build_checked_step(problem, integrator, batch=batch)
    actual, info = jax.jit(step)(initial)
    jax.block_until_ready((actual, info))
    h = np.array([[0.28, 0.23 + 0.12j], [0.23 - 0.12j, -0.28]])
    expected = np.stack([expm(-0.6j * h) @ np.asarray(s.electronic) for s in states])
    np.testing.assert_allclose(actual.electronic, expected if batch else expected[0], atol=2e-14)
    assert int(info.code) == 0
    np.testing.assert_array_equal(actual.step, initial.step + 1)
    np.testing.assert_allclose(actual.time, initial.time + 0.6)
    axis = 1 if batch else 0
    expected_budget = 1e-11 + 1e-10 * np.linalg.norm(initial.electronic, axis=axis)
    np.testing.assert_allclose(info.macrostep_budget, expected_budget, rtol=1e-14)
    assert np.all(info.accumulated_error_estimate <= info.macrostep_budget)
    if block:
        np.testing.assert_array_equal(actual.electronic[..., -1], 0)
        np.testing.assert_array_equal(info.accumulated_error_estimate[..., -1], 0)


def test_time_varying_cpa_converges_to_independent_scipy_ode():
    path = HarmonicPath([0.7], [0.3], [1.3])
    problem = Problem(LinearModel(), PARAMS, PrescribedPath(path), CPA())
    initial = state_at()

    def rhs(t, c):
        q = 0.7 * np.cos(1.3 * t) + 0.3 / 1.3 * np.sin(1.3 * t)
        h = np.array([[0.4 * q, 0.23], [0.23, -0.4 * q]])
        return -1j * h @ c

    reference = solve_ivp(rhs, (0, 1.6), [1 + 0j, 0j], rtol=2e-13, atol=2e-14).y[:, -1]
    errors = []
    for dt in (0.1, 0.05, 0.025):
        step = problem.method.build_checked_step(problem, configured(dt))

        @jax.jit
        def evolve(state):
            return jax.lax.scan(lambda s, _: step(s), state, None, length=round(1.6 / dt))

        actual, information = evolve(initial)
        assert np.all(information.code == 0)
        errors.append(np.linalg.norm(np.asarray(actual.electronic) - reference))
    assert errors[0] / errors[1] > 3.9
    assert errors[1] / errors[2] > 3.9
    assert errors[-1] < 5e-5


@pytest.mark.parametrize("batch", [False, True])
def test_ehrenfest_matches_independent_frozen_exponentials_and_verlet(batch):
    mass, dt = 1.7, 0.4
    states = [state_at(), state_at(-0.3, -0.2, identity=1, time=1.4)]
    initial = stack_states(states) if batch else states[0]
    problem = Problem(LinearModel(), PARAMS, CoupledClassical(mass), Ehrenfest())
    step = problem.method.build_checked_step(problem, configured(dt, substeps=2), batch=batch)
    actual, info = jax.jit(step)(initial)
    assert int(info.code) == 0
    expected = []
    for state in states:
        q, p = float(state.q[0]), float(state.p[0])
        def h(x):
            return np.array([[0.4 * x, 0.23], [0.23, -0.4 * x]])
        c_half = expm(-0.5j * dt * h(q)) @ np.asarray(state.electronic)
        population_difference = abs(c_half[0])**2 - abs(c_half[1])**2
        def f(x):
            return -0.6 * x - 0.4 * population_difference
        p_half = p + dt / 2 * f(q)
        q_new = q + dt * p_half / mass
        p_new = p_half + dt / 2 * f(q_new)
        c_new = expm(-0.5j * dt * h(q_new)) @ c_half
        expected.append((q_new, p_new, c_new))
    for i, reference in enumerate(expected if batch else expected[:1]):
        observed = jax.tree.map(lambda x: x[i], actual) if batch else actual
        np.testing.assert_allclose(observed.q, [reference[0]], atol=2e-14)
        np.testing.assert_allclose(observed.p, [reference[1]], atol=2e-14)
        np.testing.assert_allclose(observed.electronic, reference[2], atol=2e-14)
    assert np.all(info.accumulated_error_estimate <= info.macrostep_budget)


class CountedPath:
    prescribed = True

    def __init__(self, calls, *, fail_endpoint=False, fail_midpoint=False):
        self.calls, self.fail_endpoint, self.fail_midpoint = calls, fail_endpoint, fail_midpoint

    def point(self, state, elapsed):
        jax.debug.callback(lambda t: self.calls.append(float(t)), elapsed, ordered=True)
        bad = ((elapsed == 1.0) if self.fail_endpoint else
               ((elapsed < 1.0) if self.fail_midpoint else jnp.asarray(False)))
        return jnp.where(bad, jnp.full_like(state.q, jnp.nan), state.q), state.p


@pytest.mark.parametrize("batch", [False, True])
def test_cpa_failure_skips_remaining_midpoints_and_endpoint_and_rolls_back(batch):
    calls = []
    problem = Problem(LinearModel(), PARAMS, CountedPath(calls), CPA())
    initial = state_at()
    if batch:
        initial = stack_states([initial, state_at(identity=1)])
    step = problem.method.build_checked_step(problem, configured(1, dimension=1, substeps=3),
                                             batch=batch)
    actual, info = jax.jit(step)(initial)
    jax.block_until_ready((actual, info))
    assert_same_state(actual, initial)
    assert int(info.code) == 1 and int(info.phase) == PHASE_CPA and int(info.substep) == 0
    # elapsed is shared across the batch, so its callback runs once per stage.
    np.testing.assert_allclose(calls, [1 / 6])
    assert np.all(info.failed_trajectories)


@pytest.mark.parametrize("failure", ["midpoint", "endpoint"])
def test_cpa_nonfinite_path_is_a_physical_failure(failure):
    calls = []
    treatment = CountedPath(calls, fail_endpoint=failure == "endpoint",
                            fail_midpoint=failure == "midpoint")
    problem = Problem(LinearModel(), PARAMS, treatment, CPA())
    initial = state_at()
    actual, info = jax.jit(problem.method.build_checked_step(problem, configured(1)))(initial)
    jax.block_until_ready((actual, info))
    assert_same_state(actual, initial)
    assert int(info.code) == 2
    assert int(info.phase) == (PHASE_CPA if failure == "midpoint" else PHASE_ENDPOINT)
    assert len(calls) == (1 if failure == "midpoint" else 2)
    assert int(info.action.iterations) == (0 if failure == "midpoint" else 2)


class StagedModel(AutoDiffModel):
    """Deliberate failure fixture; its discontinuity is not a physical test model."""
    spec = LinearModel.spec

    def __init__(self, force_calls, *, first_action_failure=False, nonfinite_force=None):
        self.force_calls = force_calls
        self.first_action_failure = first_action_failure
        self.nonfinite_force = nonfinite_force

    def apply(self, params, q, vectors):
        coupling = jnp.where((q[0] > 0) | self.first_action_failure, 1.0, 0.0)
        return jnp.array([[0.2, coupling], [coupling, -0.2]]) @ vectors

    def reference_gradient(self, params, q):
        jax.debug.callback(lambda x: self.force_calls.append(float(x)), q[0], ordered=True)
        bad = ((q[0] < 0) if self.nonfinite_force == "first" else
               ((q[0] > 0) if self.nonfinite_force == "second" else jnp.asarray(False)))
        return jnp.where(bad, jnp.full_like(q, jnp.nan), jnp.zeros_like(q))

    def contract_gradient(self, params, q, weight):
        return jnp.zeros_like(q)

    def reference_energy(self, params, q):
        return jnp.asarray(0.0)


@pytest.mark.parametrize("batch", [False, True])
@pytest.mark.parametrize("failure", ["first_action", "second_action", "first_force", "second_force"])
def test_ehrenfest_failures_stop_later_stages_and_rollback_whole_batch(batch, failure):
    calls = []
    model = StagedModel(calls, first_action_failure=failure == "first_action",
                        nonfinite_force={"first_force": "first", "second_force": "second"}.get(failure))
    problem = Problem(model, None, CoupledClassical(1.0), Ehrenfest())
    initial = state_at(-0.2, 1)
    if batch:
        initial = stack_states([initial, state_at(-0.3, 0.1, identity=1)])
    actual, info = jax.jit(problem.method.build_checked_step(
        problem, configured(1, dimension=1), batch=batch))(initial)
    jax.block_until_ready((actual, info))
    assert_same_state(actual, initial)
    expected_phase = {"first_action": PHASE_EHRENFEST_FIRST,
                      "second_action": PHASE_EHRENFEST_SECOND,
                      "first_force": PHASE_FORCE_FIRST, "second_force": PHASE_FORCE_SECOND}
    assert int(info.phase) == expected_phase[failure]
    assert int(info.code) == (1 if failure.endswith("action") else 2)
    force_stages = {"first_action": 0, "second_action": 2, "first_force": 1, "second_force": 2}
    assert len(calls) == force_stages[failure] * (2 if batch else 1)
    if batch and failure.startswith("second"):
        np.testing.assert_array_equal(info.failed_trajectories, [True, False])


def test_final_accumulated_budget_is_an_independent_commit_gate():
    initial = state_at()
    candidate = initial._replace(q=initial.q + 1, time=initial.time + 1, step=initial.step + 1)
    info = empty_checked_info(initial)._replace(macrostep_budget=jnp.asarray(0.1),
                                               accumulated_error_estimate=jnp.asarray(0.11))
    actual, diagnostic = jax.jit(finish_checked_step, static_argnames=("batch",))(
        initial, candidate, info, batch=False)
    assert_same_state(actual, initial)
    assert int(diagnostic.code) == 3 and int(diagnostic.phase) == PHASE_ENDPOINT


@pytest.mark.parametrize("method,substeps,actions", [(CPA(), 3, 3), (Ehrenfest(), 2, 4)])
def test_subaction_receives_divided_macrostep_budget(method, substeps, actions):
    initial, dt = state_at(), 0.3
    model = LinearModel()
    generous = LanczosOptions(max_dimension=2, atol=1e-8, rtol=0)
    standalone = lanczos_action(lambda v: model.apply(PARAMS, initial.q, v),
                                initial.electronic, dt / actions, generous)
    # The full allowance accepts this action, but allocating it to every action
    # would incorrectly multiply the macrostep budget. The divided share rejects.
    budget = 2 * float(standalone.error_estimate)
    options = replace(generous, atol=budget)
    assert int(lanczos_action(lambda v: model.apply(PARAMS, initial.q, v),
                             initial.electronic, dt / actions, options).status) == 0
    nuclei = PrescribedPath(ConstantPath(initial.q)) if isinstance(method, CPA) else CoupledClassical(1)
    problem = Problem(model, PARAMS, nuclei, method)
    actual, info = jax.jit(method.build_checked_step(
        problem, Integrator(dt, options, substeps)))(initial)
    assert_same_state(actual, initial)
    assert int(info.code) == 1 and int(info.substep) == 0
    assert float(info.macrostep_budget) == budget
    assert float(info.action.error_estimate) > budget / actions


@pytest.mark.parametrize("method", [CPA(), Ehrenfest()])
def test_checked_hooks_reject_unvalidated_external_backend(method):
    model = LinearModel()
    model.spec = replace(model.spec, native_jax=False)
    nuclei = PrescribedPath(ConstantPath([0.7])) if isinstance(method, CPA) else CoupledClassical(1)
    problem = Problem(model, PARAMS, nuclei, method)
    with pytest.raises(ValueError, match="native JAX"):
        method.build_checked_step(problem, configured())


@pytest.mark.parametrize("field,dtype", [("q", jnp.float32), ("p", jnp.int32),
                                         ("time", jnp.float32), ("electronic", jnp.complex64)])
def test_direct_checked_hooks_report_unsupported_precision_before_branch_typing(field, dtype):
    initial = state_at()
    initial = initial._replace(**{field: getattr(initial, field).astype(dtype)})
    problem = Problem(LinearModel(), PARAMS, PrescribedPath(ConstantPath([0.7])), CPA())
    step = problem.method.build_checked_step(problem, configured())
    with pytest.raises(ValueError, match="checked propagation requires"):
        jax.jit(step)(initial)


def test_nonfinite_drift_does_not_call_endpoint_force():
    calls = []

    class LargeForce(StagedModel):
        def apply(self, params, q, vectors):
            return jnp.zeros_like(vectors)

        def reference_gradient(self, params, q):
            jax.debug.callback(lambda x: calls.append(float(x)), q[0], ordered=True)
            return jnp.full_like(q, 1e308)

    problem = Problem(LargeForce(calls), None, CoupledClassical(1), Ehrenfest())
    initial = state_at()
    actual, info = jax.jit(problem.method.build_checked_step(problem, configured(1e154)))(initial)
    jax.block_until_ready((actual, info))
    assert_same_state(actual, initial)
    assert int(info.code) == 2 and int(info.phase) == PHASE_FORCE_FIRST
    assert len(calls) == 1
