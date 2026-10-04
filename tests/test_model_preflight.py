"""Concrete model checks run on the host before any initial publication."""

import jax.numpy as jnp
import numpy as np
import pytest

from pyeph import CoupledClassical, Ehrenfest, Execution, Integrator, Problem, Simulation, make_state
from pyeph.core.state import stack_states
from pyeph.core.validation import validate_model_at
from pyeph.io.checkpoint import load_checkpoint, save_checkpoint
from pyeph.models.analytic import SpinBosonModel
from pyeph.models.composite import ReferenceShiftModel, SumModel


def model_with_preflight(calls):
    class CheckedSpin(SpinBosonModel):
        def validate_geometry(self, q):
            raise AssertionError("validate_at owns geometry; do not also call the old hook")

        def validate_at(self, params, q, *, batch=False):
            values = np.asarray(q)
            calls.append((values.copy(), batch, float(params["bias"])))
            assert values.shape == ((2, 1) if batch else (1,))
            if np.any(values + params["bias"] < 0):
                raise ValueError("provider invalid at this coordinate and parameter")

    return CheckedSpin()


def initial_state(batch=False, *, bad=False):
    first = make_state([.2], [.1], [1., 0.], trajectory_id=3)
    second = make_state([-.3 if bad else .4], [.1], [1., 0.], trajectory_id=7)
    return stack_states([first, second]) if batch else (second if bad else first)


def make_runner(model, params=None, *, jit=True):
    params = model.default_params() if params is None else params
    return Simulation(Problem(model, params, CoupledClassical(1.), Ehrenfest()),
                      Integrator(.001), Execution(jit=jit, chunk_size=2))


@pytest.mark.parametrize("batch", [False, True])
@pytest.mark.parametrize("jit", [False, True])
def test_model_preflight_runs_once_outside_compiled_steps(batch, jit):
    calls = []
    model = model_with_preflight(calls)
    state = initial_state(batch)
    result = make_runner(model, jit=jit).run(state, 5, collect=False)
    assert len(calls) == 1
    np.testing.assert_array_equal(calls[0][0], state.q)
    assert calls[0][1] is batch
    np.testing.assert_array_equal(result.final_state.step, np.asarray(state.step)+5)


@pytest.mark.parametrize("batch", [False, True])
@pytest.mark.parametrize("steps", [0, 2])
@pytest.mark.parametrize("collect", [False, True])
def test_bad_initial_provider_never_publishes_or_advances(batch, steps, collect):
    calls, published = [], []
    runner = make_runner(model_with_preflight(calls))
    state = initial_state(batch, bad=True)
    with pytest.raises(ValueError, match="provider invalid"):
        runner.run(state, steps, collect=collect,
                   observer=lambda *args: published.append(args))
    assert not published
    assert len(calls) == 1  # one batch call includes the second, invalid lane
    np.testing.assert_array_equal(state.step, 0)
    # A preflight failure releases the existing Runner lifecycle guard.
    runner.run(initial_state(batch), 0, collect=False)


@pytest.mark.parametrize("batch", [False, True])
def test_no_output_and_updated_parameters_still_recheck_actual_model(batch):
    calls = []
    model = model_with_preflight(calls)
    runner = make_runner(model)
    state = initial_state(batch)
    runner.run(state, 2, collect=False)
    runner.update_parameters({**model.default_params(), "bias": -1.})
    with pytest.raises(ValueError, match="provider invalid"):
        runner.run(state, 0, collect=False)
    assert calls[-1][2] == -1.


@pytest.mark.parametrize("batch", [False, True])
def test_checkpoint_save_and_load_recheck_provider_at_saved_coordinates(batch, tmp_path):
    calls = []
    runner = make_runner(model_with_preflight(calls))
    identities = {"model": "test-checked-spin-v1"}
    good, bad = initial_state(batch), initial_state(batch, bad=True)
    path = tmp_path/"valid.h5"
    runner.save_checkpoint(path, good, artifact_ids=identities)
    runner.load_checkpoint(path, artifact_ids=identities)
    assert len(calls) == 2
    with pytest.raises(ValueError, match="provider invalid"):
        runner.save_checkpoint(tmp_path/"invalid.h5", bad, artifact_ids=identities)
    assert not (tmp_path/"invalid.h5").exists()
    _, metadata = load_checkpoint(path)
    save_checkpoint(tmp_path/"bad_state.h5", bad, metadata=metadata)
    with pytest.raises(ValueError, match="provider invalid"):
        runner.load_checkpoint(tmp_path/"bad_state.h5", artifact_ids=identities)


@pytest.mark.parametrize("batch", [False, True])
def test_nested_composition_forwards_params_batch_and_legacy_geometry(batch):
    calls, geometry_calls = [], []
    checked = model_with_preflight(calls)

    class GeometryOnly(SpinBosonModel):
        def validate_geometry(self, q):
            assert np.asarray(q).shape == (1,)
            geometry_calls.append(np.array(q))

    legacy = GeometryOnly()
    shift = ReferenceShiftModel(checked, lambda value, q: value*jnp.sum(q*q))
    model = SumModel((legacy, shift))
    params = (legacy.default_params(), (checked.default_params(), .3))
    state = initial_state(batch)
    make_runner(model, params).run(state, 0, collect=False)
    assert len(calls) == 1 and calls[0][1] is batch
    assert len(geometry_calls) == (2 if batch else 1)
    invalid = (params[0], ({**params[1][0], "bias": -1.}, params[1][1]))
    with pytest.raises(ValueError, match="provider invalid"):
        make_runner(model, invalid).run(state, 0, collect=False)


def test_no_optional_hooks_remains_a_valid_structural_model():
    validate_model_at(object(), None, jnp.array([.2]))
    validate_model_at(object(), None, jnp.array([[.2], [.3]]), batch=True)
