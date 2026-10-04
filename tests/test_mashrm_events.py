"""Bounded real RM event semantics, separate from continuous-oracle accuracy."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.linalg import expm
from scipy.optimize import brentq

from pyeph.core.contracts import ModelSpec
from pyeph.core.problem import CoupledClassical, Problem
from pyeph.core.system import SystemSpec
from pyeph.dynamics.mashrm import MASHRM, MASHRMError, MASHRMPopulation, mapping_state
from pyeph.execution.runner import Execution
from pyeph.integrators.electronic import Integrator
from pyeph.models.base import AutoDiffModel
from pyeph.simulation import Simulation


class RotatingThree(AutoDiffModel):
    """Two rotating isolated constant surfaces plus an uncoupled spectator."""

    spec = ModelSpec(SystemSpec(3, (1,), coordinate_kind="canonical"), name="rm-event-fixture")

    def apply(self, params, q, vectors):
        angle = 2*q[0]
        h = jnp.array([[-.02*jnp.cos(angle), -.02*jnp.sin(angle), 0.],
                       [-.02*jnp.sin(angle), .02*jnp.cos(angle), 0.],
                       [0., 0., .4]])
        return h@vectors

    def reference_energy(self, params, q):
        return jnp.zeros((), q.dtype)


def problem(method=None):
    return Problem(RotatingThree(), None, CoupledClassical(1.),
                   method or MASHRM(event_substeps=1),
                   MASHRMPopulation(include_density=True, include_nuclei=True))


def state(momentum=.4, *, populations=(.4, .4, .2), phases=(0., np.pi, .7),
          active=0, time=2.5, step=7):
    c = np.sqrt(populations)*np.exp(1j*np.asarray(phases))
    return mapping_state(RotatingThree(), None, [0.], [momentum], c, active=active,
                          time=time, step=step, trajectory_id=913, seed=48)


def integrate(initial, duration, method=None):
    selected = problem(method)
    integrator = Integrator(duration, "exponential_midpoint")
    return jax.jit(selected.method.build_step(selected, integrator))(initial)


def rotation(q):
    return np.array([[np.cos(q), -np.sin(q), 0.],
                     [np.sin(q), np.cos(q), 0.], [0., 0., 1.]])


def numpy_smooth(q, p, c, duration):
    """Independent one-segment VV/midpoint equations for constant surfaces.

    This checks the discrete event convention, not convergence to exact
    coupled dynamics. SciPy matrix exponentiation and analytic zero surface
    gradients do not call production propagation/force/event helpers.
    """
    end_q = q+p*duration
    u = rotation((q+end_q)/2)
    matrix = u@np.diag([-.02, .02, .4])@u.T
    return end_q, p, expm(-1j*duration*matrix)@c


def pair_margin(q, c):
    coefficients = rotation(q).T@c
    return abs(coefficients[0])**2-abs(coefficients[1])**2


def assert_unchanged_physical(actual, expected):
    for field in ("q", "p", "electronic", "time", "step", "key", "trajectory_id"):
        np.testing.assert_array_equal(getattr(actual, field), getattr(expected, field))


@pytest.mark.parametrize("momentum,accepted", [(.4, True), (.2, False)])
def test_exact_initial_incoming_event_is_applied_once_without_projecting_c(momentum, accepted):
    # A few ulps of positive pair residual must remain in c, not be projected
    # onto an exact population equator. The local time criterion resolves it.
    initial = state(momentum, populations=(.4+1e-14, .4-1e-14, .2))
    dt = .01
    result = integrate(initial, dt)
    outgoing = np.sqrt(momentum**2-.08) if accepted else -momentum
    q, p, c = numpy_smooth(0., outgoing, np.asarray(initial.electronic), dt)
    assert int(result.method_state["status"]) == 0
    assert int(result.method_state["events"]) == int(result.method_state["event_attempts"]) == 1
    assert int(result.method_state["accepted"]) == int(accepted)
    assert int(result.method_state["frustrated"]) == int(not accepted)
    assert int(result.method_state["active"]) == int(accepted)
    assert int(result.method_state["localization_iterations"]) == 0
    np.testing.assert_allclose(result.q, [q], atol=2e-16, rtol=0.)
    np.testing.assert_allclose(result.p, [p], atol=2e-16, rtol=0.)
    np.testing.assert_allclose(result.electronic, c, atol=9e-16, rtol=0.)
    assert float(result.method_state["max_event_residual"]) > 1e-14
    assert float(result.method_state["max_impulse_energy_error"]) < 2e-16
    assert int(result.step) == int(initial.step)+1
    assert float(result.time) == pytest.approx(float(initial.time)+dt)
    np.testing.assert_array_equal(result.key, initial.key)
    # Continuing the outgoing state must not repeat the already applied hop.
    restarted = integrate(result, dt)
    assert int(restarted.method_state["status"]) == 0
    assert int(restarted.method_state["events"]) == 1


@pytest.mark.parametrize("duration", [.01, 1e-11])
def test_initial_outgoing_boundary_and_nearby_outgoing_endpoint_do_not_hop(duration):
    initial = state(-.4)
    result = integrate(initial, duration)
    q, p, c = numpy_smooth(0., -.4, np.asarray(initial.electronic), duration)
    assert int(result.method_state["status"]) == int(result.method_state["events"]) == 0
    assert int(result.method_state["active"]) == 0
    np.testing.assert_allclose(result.q, [q], atol=2e-16, rtol=0.)
    np.testing.assert_allclose(result.p, [p], atol=2e-16, rtol=0.)
    np.testing.assert_allclose(result.electronic, c, atol=9e-16, rtol=0.)


@pytest.mark.parametrize("kind", ["zero_momentum", "zero_direction", "threshold", "triple"])
def test_unresolved_initial_events_fail_without_moving_the_physical_state(kind):
    if kind == "zero_momentum":
        initial, status = state(0.), 4
    elif kind == "zero_direction":
        initial, status = state(.4, phases=(0., np.pi/2, .7)), 4
    elif kind == "threshold":
        initial, status = state(np.sqrt(.08)), 4
    else:
        initial, status = state(.4, populations=(1/3, 1/3, 1/3)), 6
    result = integrate(initial, .01)
    assert int(result.method_state["status"]) == status
    assert int(result.method_state["events"]) == 0
    assert_unchanged_physical(result, initial)


def test_tied_lower_populations_are_allowed_when_the_largest_population_is_unique():
    initial = state(0., populations=(.6, .2, .2))
    result = integrate(initial, .1)
    assert int(result.method_state["status"]) == int(result.method_state["events"]) == 0
    assert int(result.method_state["active"]) == 0
    expected = np.exp(-.1j*np.array([-.02, .02, .4]))*np.asarray(initial.electronic)
    np.testing.assert_allclose(result.electronic, expected, atol=3e-16)


def crossing_time(initial):
    def margin(duration):
        q, _, c = numpy_smooth(0., float(initial.p[0]), np.asarray(initial.electronic), duration)
        return pair_margin(q, c)

    return brentq(margin, 0., 1., xtol=5e-15)


def test_terminal_incoming_event_with_positive_roundoff_residual_is_right_continuous():
    initial = state(.4, populations=(.6, .3, .1))
    exact = crossing_time(initial)
    # Stay a few dozen ulps on the positive side, robust to independent SciPy
    # versus JAX eigensolver rounding, far below the declared local time limit.
    duration = exact-64*np.spacing(exact)
    expected_q, _, expected_c = numpy_smooth(0., .4, np.asarray(initial.electronic), duration)
    assert 0 < pair_margin(expected_q, expected_c) < 1e-13
    result = integrate(initial, duration)
    assert int(result.method_state["status"]) == 0
    assert int(result.method_state["events"]) == int(result.method_state["accepted"]) == 1
    assert int(result.method_state["active"]) == 1
    assert int(result.method_state["localization_iterations"]) == 0
    assert float(result.method_state["max_event_bracket_width"]) == 0.
    assert 0 < float(result.method_state["max_event_residual"]) < 1e-13
    # The final output is post-impulse; positions/c remain the incoming endpoint.
    np.testing.assert_allclose(result.q, [expected_q], atol=2e-16, rtol=0.)
    np.testing.assert_allclose(result.electronic, expected_c, atol=1e-15, rtol=0.)
    np.testing.assert_allclose(result.p, [np.sqrt(.08)], atol=3e-16, rtol=0.)
    retained_margin = pair_margin(float(result.q[0]), np.asarray(result.electronic))
    assert retained_margin > 0  # No amplitude/equator projection was introduced.
    np.testing.assert_allclose(result.method_state["max_event_residual"], retained_margin,
                               atol=8*np.finfo(float).eps, rtol=0.)
    continuation = integrate(result, .01)
    assert int(continuation.method_state["status"]) == 0
    assert int(continuation.method_state["events"]) == 1


def test_population_tolerance_alone_does_not_snap_a_distant_positive_endpoint():
    initial = state(.4, populations=(.6, .3, .1))
    duration = crossing_time(initial)-1e-7
    method = MASHRM(event_substeps=1, event_tolerance=1e-4, event_time_tolerance=1e-10)
    result = integrate(initial, duration, method)
    assert int(result.method_state["status"]) == int(result.method_state["events"]) == 0
    assert int(result.method_state["active"]) == 0
    assert 0 < pair_margin(float(result.q[0]), np.asarray(result.electronic)) < 1e-4
    np.testing.assert_allclose(result.p, initial.p, atol=2e-16, rtol=0.)


@pytest.mark.parametrize("momentum", [2e-12, -2e-12])
def test_slow_transverse_rate_does_not_turn_initial_population_roundoff_into_an_instant_event(momentum):
    initial = state(momentum, populations=(.4+6e-15, .4-6e-15, .2))
    c = np.asarray(initial.electronic)
    margin = pair_margin(0., c)
    rate = 4*np.real(c[0].conj()*c[1])*momentum
    assert margin > 0 and abs(rate) > 1e-12
    assert margin/abs(rate) > 1e-4
    # The signed margin is below64eps, but its local crossing time is thousands
    # of times longer than this segment and millions of times the time bound.
    duration = 1e-5
    result = integrate(initial, duration)
    assert int(result.method_state["status"]) == int(result.method_state["events"]) == 0
    assert int(result.method_state["active"]) == 0
    # At q~1e-17 a dense eigensolver may round a tiny rotation to the identity,
    # leaving an absolute force-roundoff contribution far below the momentum.
    np.testing.assert_allclose(result.p, initial.p, atol=1e-22, rtol=0.)
    np.testing.assert_allclose(result.q, [momentum*duration], atol=1e-28, rtol=0.)


@pytest.mark.parametrize("momentum", [2e-12, -2e-12])
def test_wrong_initial_ownership_outside_local_time_bound_fails_for_either_rate_sign(momentum):
    initial = state(momentum, populations=(.4-6e-15, .4+6e-15, .2))
    c = np.asarray(initial.electronic)
    margin = pair_margin(0., c)
    rate = 4*np.real(c[0].conj()*c[1])*momentum
    assert margin < 0 and abs(rate) > 1e-12
    assert abs(margin/rate) > 1e-4
    result = integrate(initial, .01)
    assert int(result.method_state["status"]) == 4
    assert int(result.method_state["events"]) == int(result.method_state["event_attempts"]) == 0
    assert_unchanged_physical(result, initial)


def test_localization_budget_failure_retains_the_interval_entry_state():
    initial = state(.4, populations=(.6, .3, .1))
    method = MASHRM(event_substeps=1, bisection_iterations=1)
    result = integrate(initial, 1., method)
    assert int(result.method_state["status"]) == 3
    assert int(result.method_state["events"]) == 0
    assert int(result.method_state["event_attempts"]) == 1
    assert int(result.method_state["localization_iterations"]) == 1
    assert float(result.method_state["max_event_bracket_width"]) == .5
    assert_unchanged_physical(result, initial)


def test_event_capacity_failure_retains_completed_internal_segments_and_initial_macrostep_counter():
    initial = state(.4)
    method = MASHRM(event_substeps=16, max_events_per_step=1)
    result = integrate(initial, 8., method)
    assert int(result.method_state["status"]) == 5
    assert int(result.method_state["events"]) == int(result.method_state["accepted"]) == 1
    assert int(result.method_state["event_attempts"]) == 2
    assert int(result.method_state["active"]) == 1
    elapsed = float(result.time-initial.time)
    assert 0 < elapsed < 8.
    completed = round(elapsed/.5)
    assert elapsed == completed*.5
    q, p, c = 0., np.sqrt(.08), np.asarray(initial.electronic)
    for _ in range(completed):
        q, p, c = numpy_smooth(q, p, c, .5)
    np.testing.assert_allclose(result.q, [q], atol=3e-14, rtol=0.)
    np.testing.assert_allclose(result.p, [p], atol=3e-14, rtol=0.)
    np.testing.assert_allclose(result.electronic, c, atol=3e-14, rtol=0.)
    assert int(result.step) == int(initial.step)
    np.testing.assert_array_equal(result.key, initial.key)


def test_runner_failed_chunk_exposes_checkpoint_safe_entry_and_unpublished_partial_state():
    initial = state(.4, populations=(.6, .3, .1))
    selected = problem(MASHRM(event_substeps=1, bisection_iterations=1))
    runner = Simulation(selected, Integrator(.25, "exponential_midpoint"),
                        Execution(chunk_size=3, save_every=1))
    observed = []
    with pytest.raises(MASHRMError) as caught:
        runner.run(initial, 3, observer=lambda times, values: observed.append(np.asarray(times)))
    failure = caught.value
    assert int(failure.failed_state.method_state["status"]) == 3
    assert_unchanged_physical(failure.last_valid_state, initial)
    assert len(observed) == 1
    np.testing.assert_array_equal(observed[0], [float(initial.time)])
    # The first macrostep succeeded inside the rejected output chunk; the
    # second retains that state rather than its unconverged event candidate.
    q, p, c = numpy_smooth(0., .4, np.asarray(initial.electronic), .25)
    np.testing.assert_allclose(failure.failed_state.q, [q], atol=2e-15, rtol=0.)
    np.testing.assert_allclose(failure.failed_state.p, [p], atol=2e-15, rtol=0.)
    np.testing.assert_allclose(failure.failed_state.electronic, c, atol=2e-15, rtol=0.)
    assert float(failure.failed_state.time) == float(initial.time)+.25
    assert int(failure.failed_state.step) == int(initial.step)+1
    assert not runner._running
    # Retry from the returned checkpoint-safe state with a sufficient budget.
    retry = Simulation(replace(selected, method=MASHRM(event_substeps=1)),
                       runner.integrator, runner.execution).run(failure.last_valid_state, 3)
    assert int(retry.final_state.method_state["status"]) == 0
    assert int(retry.final_state.method_state["events"]) == 1


def test_accepted_attempt_counter_overflow_rejects_before_committing_an_impulse():
    initial = state(.4)
    limit = np.iinfo(np.int32).max
    counters = {**initial.method_state, "events": jnp.int32(limit),
                "accepted": jnp.int32(limit), "event_attempts": jnp.int32(limit)}
    initial = initial._replace(method_state=counters)
    result = integrate(initial, .01)
    assert int(result.method_state["status"]) == 7
    assert_unchanged_physical(result, initial)
    for field in ("events", "accepted", "frustrated", "event_attempts", "localization_iterations"):
        np.testing.assert_array_equal(result.method_state[field], initial.method_state[field])
    runner = Simulation(problem(), Integrator(.01, "exponential_midpoint"))
    with pytest.raises(MASHRMError, match="counter") as caught:
        runner.run(initial, 1)
    assert_unchanged_physical(caught.value.last_valid_state, initial)
    assert_unchanged_physical(caught.value.failed_state, initial)
    assert int(caught.value.failed_state.method_state["status"]) == 7


def test_localization_counter_overflow_retains_the_accepted_segment_entry():
    initial = state(.4, populations=(.6, .3, .1))
    initial = initial._replace(method_state={**initial.method_state,
        "localization_iterations": jnp.int32(np.iinfo(np.int32).max-1)})
    result = integrate(initial, 1.)
    assert int(result.method_state["status"]) == 7
    assert_unchanged_physical(result, initial)
    for field in ("events", "accepted", "frustrated", "event_attempts", "localization_iterations"):
        np.testing.assert_array_equal(result.method_state[field], initial.method_state[field])


def test_full_diagnostic_counters_still_allow_a_segment_without_events():
    initial = state(0., populations=(.6, .2, .2))
    limit = np.iinfo(np.int32).max
    initial = initial._replace(method_state={**initial.method_state,
        "events": jnp.int32(limit), "accepted": jnp.int32(limit),
        "event_attempts": jnp.int32(limit), "localization_iterations": jnp.int32(limit)})
    final = Simulation(problem(), Integrator(.01, "exponential_midpoint")).run(initial, 1).final_state
    assert int(final.method_state["status"]) == 0
    assert int(final.step) == int(initial.step)+1
    for field in ("events", "accepted", "frustrated", "event_attempts", "localization_iterations"):
        np.testing.assert_array_equal(final.method_state[field], initial.method_state[field])


def test_unrepresentable_downhill_impulse_does_not_commit_nonfinite_momentum():
    # Every physical input is finite, but squaring this incident parallel
    # momentum overflows. A valid input flag from the generic impulse is not
    # sufficient evidence that its proposed output can be committed.
    # Exact dyadic top amplitudes distinguish this genuine boundary from a
    # tiny positive residual, whose invalid prospective shortcut may defer.
    initial = mapping_state(RotatingThree(), None, [0.], [-1e200],
                            np.array([.625, -.625, np.sqrt(.21875)], dtype=complex),
                            active=1, time=0.)
    result = integrate(initial, 1e-200)
    assert int(result.method_state["status"]) == 4
    assert int(result.method_state["events"]) == 0
    assert_unchanged_physical(result, initial)
