"""Independent event-placement regressions; no surface or mapping projection."""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.linalg import expm
from scipy.optimize import brentq

from pyeph import CoupledClassical, Execution, Integrator, MASHRM, Problem, Simulation
from pyeph.core.contracts import ModelSpec
from pyeph.core.state import TrajectoryState
from pyeph.core.system import SystemSpec
from pyeph.dynamics.mashrm import MASHRMPopulation, mapping_state
from pyeph.models.base import AutoDiffModel
from pyeph.models.epc import LinearEPCModel


def load_id900():
    fixture = Path(__file__).parent/"fixtures/mashrm_id900_location.json"
    saved = json.loads(fixture.read_text())
    old = saved["last_valid_state"]
    methods = {key: jnp.asarray(value, dtype=jnp.int32 if isinstance(value, int) else jnp.float64)
               for key, value in old["method_state"].items()}
    state = TrajectoryState(
        jnp.asarray(old["q"]), jnp.asarray(old["p"]),
        jnp.asarray(old["electronic_real"])+1j*jnp.asarray(old["electronic_imag"]),
        jnp.float64(old["time"]), jnp.int64(old["step"]),
        jnp.uint32(old["trajectory_id"]), jnp.asarray(old["key"], dtype=jnp.uint32), methods)
    data = saved["model"]
    model = LinearEPCModel(3, 2)
    params = model.create_params(data["h0"], data["coupling"], data["omega"], data["q_eq"])
    problem = Problem(model, params, CoupledClassical(data["masses"]), MASHRM(**saved["method"]),
                      MASHRMPopulation(include_nuclei=True))
    return saved, problem, state


def test_saved_id900_replays_from_last_valid_state_without_rejecting_its_own_hop():
    saved, problem, initial = load_id900()
    assert saved["original_runtime_sha256"] == "0817f665642ecd5523715a01a7beab35c9c90116bc1b5b886c01adfb48791db8"
    assert int(initial.trajectory_id) == 900
    assert int(initial.method_state["accepted"]) == 0
    assert int(initial.method_state["frustrated"]) == 1
    # Replay the retained t=1.6 state, not the post-impulse failed state. The
    # original full 1024-ID campaign and its failure remain archived unchanged.
    result = Simulation(problem, Integrator(saved["dt"], "exponential_midpoint"),
                        Execution(chunk_size=10)).run(initial, 10)
    final = result.final_state
    assert int(final.method_state["status"]) == 0
    assert int(final.method_state["accepted"]) == 1
    assert int(final.method_state["frustrated"]) == 1
    assert int(final.method_state["events"]) == 2
    assert int(final.method_state["active"]) == 2
    assert int(final.trajectory_id) == 900
    np.testing.assert_array_equal(final.key, initial.key)
    np.testing.assert_allclose(final.time, 2., atol=2e-15, rtol=0.)
    assert float(final.method_state["max_event_residual"]) <= problem.method.event_tolerance
    assert float(final.method_state["max_event_bracket_width"]) <= problem.method.event_time_tolerance
    assert np.max(abs(np.asarray(result.observables["mapping_norm"])-1)) < 1e-12


class RotatingThree(AutoDiffModel):
    """Explicit constant surfaces; their analytic nuclear forces vanish."""

    spec = ModelSpec(SystemSpec(3, (1,), coordinate_kind="canonical"),
                     name="rm-outgoing-location-reference")

    def apply(self, params, q, vectors):
        angle = 2*q[0]
        return jnp.array([[-.02*jnp.cos(angle), -.02*jnp.sin(angle), 0.],
                          [-.02*jnp.sin(angle), .02*jnp.cos(angle), 0.],
                          [0., 0., .4]])@vectors

    def reference_energy(self, params, q):
        return jnp.zeros((), q.dtype)


def rotating_state(momentum, populations=(.6, .3, .1)):
    c = np.sqrt(populations)*np.exp(1j*np.array([0., np.pi, .7]))
    return mapping_state(RotatingThree(), None, [0.], [momentum], c, active=0,
                          trajectory_id=91, seed=8)


def rotating_step(initial, duration, method=None):
    problem = Problem(RotatingThree(), None, CoupledClassical(1.),
                      method or MASHRM(event_substeps=1), MASHRMPopulation())
    return jax.jit(problem.method.build_step(
        problem, Integrator(duration, "exponential_midpoint")))(initial)


def rotation(q):
    return np.array([[np.cos(q), -np.sin(q), 0.],
                     [np.sin(q), np.cos(q), 0.], [0., 0., 1.]])


def independent_smooth(q, p, c, duration):
    """NumPy/SciPy discrete VV/midpoint reference, not a continuous-ODE claim."""
    endpoint = q+p*duration
    u = rotation((q+endpoint)/2)
    h = u@np.diag([-.02, .02, .4])@u.T
    return endpoint, expm(-1j*duration*h)@c


def margin(q, c):
    coefficients = rotation(q).T@c
    return abs(coefficients[0])**2-abs(coefficients[1])**2


def signed_rate(q, c, p):
    coefficients = rotation(q).T@c
    return 4*np.real(coefficients[0].conj()*coefficients[1])*p


def independent_bracket(initial, duration, ptol, ttol):
    """Retain both signs and require both true endpoint residuals to converge."""
    p, c = float(initial.p[0]), np.asarray(initial.electronic)
    def value(t):
        q, new_c = independent_smooth(float(initial.q[0]), p, c, t)
        return margin(q, new_c)
    lower, upper = 0., duration
    fl, fu = value(lower), value(upper)
    assert fl > 0 > fu
    count = 0
    while upper-lower > ttol or max(abs(fl), abs(fu)) > ptol:
        midpoint = lower+(upper-lower)/2
        fm = value(midpoint)
        assert fm != 0.  # These deliberately non-dyadic test roots are isolated.
        if fm > 0:
            lower, fl = midpoint, fm
        else:
            upper, fu = midpoint, fm
        count += 1
        assert count <= 48
    return lower, upper, fl, fu, count


@pytest.mark.parametrize("momentum,accepted", [(.4, True), (.2, False)])
def test_bracket_endpoint_is_owned_by_outgoing_surface_without_mapping_projection(momentum, accepted):
    initial = rotating_state(momentum)
    # Explicitly enlarged localization tolerances make endpoint selection
    # distinguishable from midpoint/equator projection; comparison tolerances
    # remain tight. Production default-tolerance behavior is tested by ID900.
    method = MASHRM(event_substeps=1, event_tolerance=1e-8, event_time_tolerance=1e-7)
    duration = 1.
    lower, upper, fl, fu, count = independent_bracket(
        initial, duration, method.event_tolerance, method.event_time_tolerance)
    selected = upper if accepted else lower
    expected_residual = fu if accepted else fl
    q_event, c_event = independent_smooth(0., momentum, np.asarray(initial.electronic), selected)
    p_out = np.sqrt(momentum**2-.08) if accepted else -momentum
    expected_q, expected_c = independent_smooth(q_event, p_out, c_event, duration-selected)
    result = rotating_step(initial, duration, method)
    diagnostics = result.method_state
    assert int(diagnostics["status"]) == 0
    assert int(diagnostics["events"]) == int(diagnostics["event_attempts"]) == 1
    assert int(diagnostics["accepted"]) == int(accepted)
    assert int(diagnostics["frustrated"]) == int(not accepted)
    assert int(diagnostics["localization_iterations"]) == count
    np.testing.assert_allclose(result.q, [expected_q], atol=2e-14, rtol=0.)
    np.testing.assert_allclose(result.p, [p_out], atol=3e-15, rtol=0.)
    np.testing.assert_allclose(result.electronic, expected_c, atol=3e-14, rtol=0.)
    np.testing.assert_allclose(diagnostics["max_event_residual"], abs(expected_residual),
                               atol=3e-15, rtol=0.)
    np.testing.assert_allclose(diagnostics["max_event_bracket_width"], upper-lower,
                               atol=2e-16, rtol=0.)
    # Nuclear free flight determines the actual selected event time without
    # requiring a production event-history API. Inverting the outgoing unitary
    # similarly recovers the unchanged event mapping amplitudes.
    observed_time = (float(result.q[0])-p_out*duration)/(momentum-p_out)
    np.testing.assert_allclose(observed_time, selected, atol=2e-13, rtol=0.)
    u = rotation((q_event+float(result.q[0]))/2)
    outgoing_h = u@np.diag([-.02, .02, .4])@u.T
    observed_c_event = expm(1j*(duration-selected)*outgoing_h)@np.asarray(result.electronic)
    np.testing.assert_allclose(observed_c_event, c_event, atol=3e-14, rtol=0.)
    post_margin = (-1 if accepted else 1)*margin(q_event, observed_c_event)
    assert 1e-12 < post_margin <= method.event_tolerance
    np.testing.assert_allclose(result.time, duration, atol=2e-15, rtol=0.)


def test_positive_instant_uphill_shortcut_waits_for_actual_crossing_when_outgoing_flow_slows():
    momentum = np.sqrt(.08+.01**2)
    initial = rotating_state(momentum, populations=(.4+1e-12, .4-1e-12, .2))
    residual = margin(0., np.asarray(initial.electronic))
    incoming = signed_rate(0., np.asarray(initial.electronic), momentum)
    outgoing = -signed_rate(0., np.asarray(initial.electronic), .01)
    assert 0 < residual/(-incoming) < 1e-10 < residual/outgoing
    duration = 1e-13  # Neither the initial nor terminal shortcut is admissible.
    q, c = independent_smooth(0., momentum, np.asarray(initial.electronic), duration)
    assert margin(q, c)/(-signed_rate(q, c, .01)) > 1e-10
    early = rotating_step(initial, duration)
    assert int(early.method_state["status"]) == int(early.method_state["events"]) == 0
    np.testing.assert_allclose(early.q, [q], atol=2e-16, rtol=0.)
    np.testing.assert_allclose(early.electronic, c, atol=2e-15, rtol=0.)
    later = rotating_step(early, .001)
    assert int(later.method_state["status"]) == 0
    assert int(later.method_state["accepted"]) == int(later.method_state["events"]) == 1
    assert int(later.method_state["active"]) == 1
    assert int(later.method_state["localization_iterations"]) > 0
    np.testing.assert_allclose(later.p, [.01], atol=2e-14, rtol=0.)


def test_positive_terminal_uphill_shortcut_defers_then_continuation_locates_real_crossing():
    momentum = np.sqrt(.08+.01**2)
    initial = rotating_state(momentum)
    def function(t):
        q, c = independent_smooth(0., momentum, np.asarray(initial.electronic), t)
        return margin(q, c)
    exact = brentq(function, 0., 1., xtol=5e-15)
    duration = exact-3e-11
    q, c = independent_smooth(0., momentum, np.asarray(initial.electronic), duration)
    residual = margin(q, c)
    assert 0 < residual/(-signed_rate(q, c, momentum)) < 1e-10
    assert residual/(-signed_rate(q, c, .01)) > 1e-10
    early = rotating_step(initial, duration)
    assert int(early.method_state["status"]) == int(early.method_state["events"]) == 0
    np.testing.assert_allclose(early.q, [q], atol=2e-15, rtol=0.)
    np.testing.assert_allclose(early.electronic, c, atol=2e-15, rtol=0.)
    later = rotating_step(early, .001)
    assert int(later.method_state["status"]) == 0
    assert int(later.method_state["events"]) == int(later.method_state["accepted"]) == 1
    assert int(later.method_state["localization_iterations"]) > 0
    np.testing.assert_allclose(later.p, [.01], atol=2e-14, rtol=0.)


def test_invalid_positive_threshold_preview_defers_but_actual_bracket_still_fails():
    initial = rotating_state(np.sqrt(.08), populations=(.4+1e-12, .4-1e-12, .2))
    early = rotating_step(initial, 1e-12)
    assert int(early.method_state["status"]) == int(early.method_state["event_attempts"]) == 0
    assert margin(float(early.q[0]), np.asarray(early.electronic)) > 0
    failed = rotating_step(early, .001)
    assert int(failed.method_state["status"]) == 4
    assert int(failed.method_state["event_attempts"]) == 1
    assert int(failed.method_state["events"]) == 0
    for field in ("q", "p", "electronic", "time", "step", "key", "trajectory_id"):
        np.testing.assert_array_equal(getattr(failed, field), getattr(early, field))


@pytest.mark.parametrize("populations", [(.390625, .390625, .21875), (.6, .3, .1)])
def test_genuine_exact_or_bracketed_threshold_event_is_not_suppressed(populations):
    # The equal top amplitudes are exactly representable +/-0.625, rather than
    # an approximate equator whose one-ulp positive residual permits deferral.
    initial = rotating_state(np.sqrt(.08), populations=populations)
    failed = rotating_step(initial, 1.)
    assert int(failed.method_state["status"]) == 4
    assert int(failed.method_state["events"]) == 0
    assert int(failed.method_state["event_attempts"]) == 1
    for field in ("q", "p", "electronic", "time", "step", "key", "trajectory_id"):
        np.testing.assert_array_equal(getattr(failed, field), getattr(initial, field))
