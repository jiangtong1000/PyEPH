"""Falsification checks for platform qualification, without expensive timing runs."""

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from benchmarks.platform_qualification import (
    compare_arrays, compare_replicate, timed, validate_batched_result, validate_continuation,
    validate_final_state, validate_scalar_reference, validate_state_replicate,
)
from pyeph.core.state import TrajectoryState


def scalar_result():
    times = np.array([0., .8, 1.6])
    observables = {
        "q": np.array([[.1], [.2], [.3]]), "p": np.array([[.2], [.3], [.4]]),
        "electronic": np.tile(np.array([1.+0j, 0j]), (3, 1)),
        "current": np.zeros((3, 3)), "norm": np.ones(3), "energy": np.full(3, .5),
        "population": np.tile([1., 0.], (3, 1)),
    }
    state = TrajectoryState(observables["q"][-1].copy(), observables["p"][-1].copy(),
                            observables["electronic"][-1].copy(), np.asarray(times[-1]),
                            np.asarray(16), np.uint32(19), np.zeros(2, dtype=np.uint32), ())
    return SimpleNamespace(times=times, observables=observables, final_state=state)


def batched_result(scalar, batch=2):
    def leading(value):
        return np.broadcast_to(value, (batch, *np.shape(value))).copy()
    times = np.broadcast_to(scalar.times[:, None], (len(scalar.times), batch)).copy()
    observables = {key: np.broadcast_to(value[:, None], (len(value), batch, *value.shape[1:])).copy()
                   for key, value in scalar.observables.items()}
    state = TrajectoryState(*(leading(value) for value in scalar.final_state[:7]), ())
    state = state._replace(trajectory_id=np.arange(100, 100+batch, dtype=np.uint32))
    return SimpleNamespace(times=times, observables=observables, final_state=state)


def reference_arrays(scalar):
    return {key: value.copy() for key, value in scalar.observables.items()
            if key in ("q", "p", "electronic", "current")}


def test_finite_scalar_and_batch_evidence_passes_with_intentionally_different_ids():
    scalar = scalar_result()
    errors = validate_scalar_reference(scalar, deepcopy(scalar), reference_arrays(scalar))
    assert all(value == {"coarse": 0., "fine": 0.} for value in errors.values())
    for batch, result in ((1, scalar), (2, batched_result(scalar))):
        evidence = validate_batched_result(result, scalar, batch)
        assert evidence["sample_time_max_error"] == 0
        assert all(value == 0 for value in evidence["observable_max_errors"].values())


@pytest.mark.parametrize("which", ["coarse", "fine", "reference"])
@pytest.mark.parametrize("invalid", [np.nan, np.inf])
def test_nonfinite_current_cannot_pass_reference_error_comparisons(which, invalid):
    coarse, fine = scalar_result(), scalar_result()
    reference = reference_arrays(coarse)
    target = reference if which == "reference" else (coarse if which == "coarse" else fine).observables
    target["current"][1, 1] = invalid
    with pytest.raises(RuntimeError, match="nonfinite"):
        validate_scalar_reference(coarse, fine, reference)


def test_scalar_time_grid_shape_and_refinement_are_required():
    scalar = scalar_result()
    fine = deepcopy(scalar)
    fine.times[1] += .01
    with pytest.raises(RuntimeError, match="sample times"):
        validate_scalar_reference(scalar, fine, reference_arrays(scalar))
    wrong = reference_arrays(scalar)
    wrong["q"] = wrong["q"][:, None]
    with pytest.raises(RuntimeError, match="shapes disagree"):
        validate_scalar_reference(scalar, scalar, wrong)
    with pytest.raises(RuntimeError, match="must cover"):
        validate_scalar_reference(scalar, scalar, {})
    reference = reference_arrays(scalar)
    scalar.observables["q"][1] += 1e-6
    with pytest.raises(RuntimeError, match="refinement"):
        validate_scalar_reference(scalar, deepcopy(scalar), reference)


@pytest.mark.parametrize("name", ["q", "p", "electronic", "current", "population", "norm", "energy"])
def test_every_lane_and_sampled_observable_is_compared_not_only_final_electronics(name):
    scalar = scalar_result()
    batch = batched_result(scalar)
    # A middle output row in the second lane is wrong; the final state is exact.
    batch.observables[name][1, 1] += .01
    with pytest.raises(RuntimeError, match=f"batched observable {name}"):
        validate_batched_result(batch, scalar, 2)


def test_batched_nonfinite_output_times_and_missing_fields_are_rejected():
    scalar = scalar_result()
    batch = batched_result(scalar)
    batch.observables["current"][:] = np.nan
    with pytest.raises(RuntimeError, match="nonfinite.*batched observable current"):
        validate_batched_result(batch, scalar, 2)
    batch = batched_result(scalar)
    batch.times[1, 1] += .1
    with pytest.raises(RuntimeError, match="batched sample times"):
        validate_batched_result(batch, scalar, 2)
    batch = batched_result(scalar)
    del batch.observables["current"]
    with pytest.raises(RuntimeError, match="fields disagree"):
        validate_batched_result(batch, scalar, 2)
    batch = batched_result(scalar)
    batch.observables["q"] = batch.observables["q"][:, :1]
    with pytest.raises(RuntimeError, match="shape mismatch"):
        validate_batched_result(batch, scalar, 2)


@pytest.mark.parametrize("name", ["q", "p", "electronic", "time", "step"])
def test_all_physical_final_state_fields_are_checked_even_without_output(name):
    scalar = scalar_result()
    state = batched_result(scalar).final_state
    changed = getattr(state, name).copy()
    changed[1] += 1
    state = state._replace(**{name: changed})
    with pytest.raises(RuntimeError, match=f"batched final {name}"):
        validate_final_state(state, scalar.final_state, 2)


def test_nonfinite_final_state_and_large_integer_counter_mismatch_fail_closed():
    scalar = scalar_result()
    state = batched_result(scalar).final_state
    state.q[1] = np.nan
    with pytest.raises(RuntimeError, match="nonfinite"):
        validate_final_state(state, scalar.final_state, 2)
    with pytest.raises(RuntimeError, match="counter"):
        compare_arrays(np.array([2**60+1], dtype=np.int64), np.array([2**60], dtype=np.int64),
                        "counter", atol=0, rtol=0)


def test_each_timed_repetition_is_validated_instead_of_only_the_last():
    scalar = scalar_result()
    calls = []
    def run():
        calls.append(len(calls))
        result = batched_result(scalar)
        if len(calls) == 2:
            result.observables["current"][1, 1, 0] = np.nan
        return result
    with pytest.raises(RuntimeError, match="nonfinite"):
        timed(run, 3, validate=lambda result: validate_batched_result(result, scalar, 2))
    assert len(calls) == 2


def continuation_result(full):
    return SimpleNamespace(times=full.times[1:].copy(),
                           observables={name: value[1:].copy()
                                        for name, value in full.observables.items()},
                           final_state=deepcopy(full.final_state))


def test_roundoff_continuation_is_qualified_and_reported_as_not_bytewise():
    full = scalar_result()
    continued = continuation_result(full)
    continued.final_state.p[0] = np.nextafter(continued.final_state.p[0], 1.)
    continued.observables["p"][-1] = continued.final_state.p
    evidence = validate_continuation(continued, full, 1)
    assert evidence["numerically_qualified"] and not evidence["bytewise_equal"]
    assert not evidence["final_state"]["bytewise_equal"]
    assert evidence["final_state"]["leaves"][".p"]["max_absolute_error"] > 0
    assert not evidence["samples"]["p"]["bytewise_equal"]
    assert evidence["final_state"]["leaves"][".key"]["bytewise_equal"]
    with pytest.raises(RuntimeError, match="checkpoint roundtrip"):
        validate_state_replicate(continued.final_state, full.final_state,
                                  "checkpoint roundtrip", exact=True)


def test_bytewise_roundtrip_distinguishes_signed_zero_and_preserves_dtypes():
    evidence = compare_replicate(np.array([-0.]), np.array([0.]), "signed zero")
    assert evidence["max_absolute_error"] == 0 and not evidence["bytewise_equal"]
    with pytest.raises(RuntimeError, match="bytewise"):
        compare_replicate(np.array([-0.]), np.array([0.]), "signed zero", exact=True)
    for exact in (False, True):
        with pytest.raises(RuntimeError, match="dtype"):
            compare_replicate(np.array([1.], dtype=np.float32), np.array([1.]),
                              "precision changed", exact=exact)


@pytest.mark.parametrize("field", ["step", "trajectory_id", "key"])
def test_discrete_state_and_rng_differences_are_always_rejected(field):
    state = scalar_result().final_state
    value = np.asarray(getattr(state, field)).copy()
    value += np.asarray(1, dtype=value.dtype)
    changed = state._replace(**{field: value})
    with pytest.raises(RuntimeError, match=field):
        validate_state_replicate(changed, state, "replicate")


def test_state_tree_method_boolean_shape_and_nonfinite_changes_are_rejected():
    state = scalar_result().final_state
    with pytest.raises(RuntimeError, match="state-tree"):
        validate_state_replicate(state._replace(method_state=(np.array(True),)), state, "replicate")
    boolean_state = state._replace(method_state=(np.array(True),))
    with pytest.raises(RuntimeError, match="method_state"):
        validate_state_replicate(state._replace(method_state=(np.array(False),)),
                                  boolean_state, "replicate")
    with pytest.raises(RuntimeError, match="shape"):
        validate_state_replicate(state._replace(p=state.p[:, None]), state, "replicate")
    with pytest.raises(RuntimeError, match="nonfinite"):
        validate_state_replicate(state._replace(p=np.array([np.nan])), state, "replicate")


@pytest.mark.parametrize("field", ["q", "p", "electronic", "current", "norm", "energy", "population"])
def test_resumed_sampled_suffix_corruption_rejected_even_with_correct_final_state(field):
    full = scalar_result()
    continued = continuation_result(full)
    continued.observables[field][0] += .01
    with pytest.raises(RuntimeError, match=f"continuation observable {field}"):
        validate_continuation(continued, full, 1)


def test_continuation_grid_missing_fields_and_excess_floating_error_rejected():
    full = scalar_result()
    continued = continuation_result(full)
    continued.times[0] += .01
    with pytest.raises(RuntimeError, match="sample times"):
        validate_continuation(continued, full, 1)
    continued = continuation_result(full)
    del continued.observables["current"]
    with pytest.raises(RuntimeError, match="fields"):
        validate_continuation(continued, full, 1)
    changed = full.final_state._replace(p=full.final_state.p + 1e-9)
    with pytest.raises(RuntimeError, match="replicate.*p"):
        validate_state_replicate(changed, full.final_state, "replicate")


def test_exact_roundtrip_and_unchanged_continuation_record_true_bytewise_flags():
    full = batched_result(scalar_result())
    evidence = validate_state_replicate(deepcopy(full.final_state), full.final_state,
                                        "checkpoint roundtrip", exact=True)
    assert evidence["numerically_qualified"] and evidence["bytewise_equal"]
    continued = continuation_result(full)
    assert validate_continuation(continued, full, 1)["bytewise_equal"]
