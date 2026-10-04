"""Continuum MASH checks against separately written SciPy equations/events."""

import numpy as np
import pytest

from benchmarks.mash2_reference import (
    cases,
    compare_case,
    initial_vector,
    numpy_quantities,
    reference_for,
    rotating_params,
    scipy_mash,
)


@pytest.mark.parametrize("kind,params", [
    ("rotating", rotating_params()),
    ("tully1", dict(a=.01, b=1.6, c=.005, d=1.)),
])
def test_independent_reference_analytic_derivatives(kind, params):
    delta = 1e-6
    for q in (-2.1, -.5, .0, .7, 2.3):
        h, dh, _, dv = numpy_quantities(kind, params, q)
        plus = numpy_quantities(kind, params, q+delta)
        minus = numpy_quantities(kind, params, q-delta)
        np.testing.assert_allclose(dh, (plus[0]-minus[0])/(2*delta), atol=2e-8, rtol=2e-6)
        np.testing.assert_allclose(dv, (plus[2]-minus[2])/(2*delta), atol=1e-9)
        np.testing.assert_array_equal(h, h.T)


def test_reference_equatorial_initial_event_is_applied_once_without_time_nudge():
    params = rotating_params()
    c = initial_vector("rotating", params, -1., 0., 0.)
    incoming = scipy_mash("rotating", params, q=-1., p=1.6, c=c, active=0,
                          mass=1., stop=.1, max_step=.01)
    assert len(incoming["events"]) == 1
    event = incoming["events"][0]
    assert event["time"] == 0.0 and event["accepted"]
    assert event["outgoing_signed_rate"] > 0
    outgoing = scipy_mash("rotating", params, q=-1., p=event["momentum_after"], c=c,
                          active=1, mass=1., stop=.1, max_step=.01)
    assert not outgoing["events"]
    np.testing.assert_allclose(incoming["final_state"], outgoing["final_state"], atol=2e-13)
    # This initial point is on the same equator, but its momentum is already
    # directed back into the lower hemisphere: no impulse may be invented.
    reflected = scipy_mash("rotating", params, q=-1., p=-.45, c=c, active=0,
                           mass=1., stop=.1, max_step=.01)
    assert not reflected["events"]


def test_reference_event_exactly_at_stop_uses_outgoing_final_sample():
    case = cases()[0]
    first = reference_for(case)["events"][0]["time"]
    found_terminal_event = False
    # Root roundoff can place neighboring floating endpoints on either side.
    # At any *detected* terminal event, stored samples must use one convention.
    for offset in (0, 1, 2, 4, 8, 16):
        stop = first+offset*np.spacing(first)
        result = reference_for({**case, "stop": stop}, sample_times=[0., stop])
        np.testing.assert_array_equal(result["states"][-1], result["final_state"])
        assert result["active"][-1] == result["final_active"]
        if result["events"] and result["events"][-1]["time"] == stop:
            found_terminal_event = True
            assert result["active"][-1] == 1
            assert result["states"][-1, 1] == result["events"][-1]["momentum_after"]
    assert found_terminal_event


@pytest.fixture(scope="module")
def nonlinear_reports():
    return {case["name"]: compare_case(case, localization_checks=case["name"] == "rotating_mixed")
            for case in cases()}


@pytest.mark.parametrize("name", [case["name"] for case in cases()])
def test_complete_nonlinear_mash_trajectories_converge_to_scipy(nonlinear_reports, name):
    report = nonlinear_reports[name]
    case = report["configuration"]
    audit = report["reference_audit"]
    assert audit["final_state_difference"] < 1e-8
    assert audit["max_event_time_difference"] < 1e-8
    assert audit["max_energy_drift"] < 1e-9
    assert audit["max_mapping_norm_error"] < 1e-10
    assert min(report["observed_orders"]) > 1.8
    assert max(report["observed_orders"]) < 2.2
    for event in report["reference_events"]:
        assert event["equator_residual"] < 1e-11
        assert event["incoming_signed_rate"] < 0 < event["outgoing_signed_rate"]
        assert abs(event["impulse_energy_error"]) < 1e-12
        if not event["accepted"]:
            assert event["active_after"] == event["active_before"]
            assert event["momentum_after"] == -event["momentum_before"]
    for run in report["runs"]:
        assert run["status"] == 0
        assert run["accepted"] == case["expected_accepted"]
        assert run["frustrated"] == case["expected_frustrated"]
        assert run["events"] == len(report["reference_events"])
        assert run["final_active"] == report["reference_final_active"]
        assert run["active_sample_mismatches"] == 0
        assert run["max_reference_event_bracket_miss"] == 0
        assert run["max_event_residual"] <= run["event_tolerance"]
        assert run["max_impulse_energy_error"] < 1e-12
        assert run["max_mapping_norm_error"] < 1e-10
    first, _, last = report["runs"]
    assert last["trajectory_q_max_absolute_error"] < first["trajectory_q_max_absolute_error"]/10
    assert last["trajectory_p_away_from_events_max_absolute_error"] < (
        first["trajectory_p_away_from_events_max_absolute_error"]/10)
    assert last["max_energy_drift"] < first["max_energy_drift"]/10


def test_localization_and_subdivision_refine_independently(nonlinear_reports):
    report = nonlinear_reports["rotating_mixed"]
    loose, tight = report["event_tolerance_refinement"]
    assert loose["events"] == tight["events"] == 9
    assert tight["max_event_residual"] <= tight["event_tolerance"]
    assert loose["max_event_residual"] <= loose["event_tolerance"]
    key = "final_state_difference_from_tolerance_1e11"
    assert tight[key] < loose[key]/50
    assert tight[key] < 1e-6
    assert report["subdivision_check"]["final_state_difference_from_fine_dt"] < 1e-9
