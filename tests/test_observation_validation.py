"""Measurement validity gates whole published blocks without changing dynamics."""

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
import pytest

from pyeph import CPA, Execution, Integrator, PrescribedPath, Problem, Simulation, make_state
from pyeph.core.state import stack_states
from pyeph.execution.runner import SimulationError
from pyeph.models.analytic import SpinBosonModel
from pyeph.observables.population import FunctionalMeasurement
from pyeph.paths.nuclear import RecordedNuclearPath


@dataclass(frozen=True)
class LimitedMeasurement:
    limit: float = .13
    bad_id: int = 1

    def validate(self, problem):
        pass

    def evaluate(self, problem, state):
        bad = (state.time > self.limit) & (state.trajectory_id == self.bad_id)
        return {"value": jnp.where(bad, jnp.nan, state.time), "bad": bad}

    def validate_observations(self, values):
        if np.any(values["bad"]):
            raise ValueError("probe outside calibrated domain")


def make_runner(measurement=None, *, batch=False, save_every=1, jit=True, checked=False):
    from pyeph.integrators.krylov import LanczosOptions

    model = SpinBosonModel()
    path = RecordedNuclearPath([0., 1.], [[0.], [1.]], [[1.], [1.]])
    problem = Problem(model, model.default_params(), PrescribedPath(path), CPA(),
                      measurement or LimitedMeasurement())
    electronic = LanczosOptions(max_dimension=2) if checked else "rk4"
    runner = Simulation(problem, Integrator(.05, electronic=electronic),
                        Execution(chunk_size=2, save_every=save_every, jit=jit))
    states = [make_state([0.], [0.], [1., 0.], trajectory_id=i) for i in (0, 1)]
    return runner, stack_states(states) if batch else states[1]


@pytest.mark.parametrize("batch", [False, True])
@pytest.mark.parametrize("jit", [False, True])
def test_invalid_observation_keeps_previous_published_chunk(batch, jit):
    runner, initial = make_runner(batch=batch, jit=jit)
    published = []
    with pytest.raises(SimulationError, match="chunk observation validation") as caught:
        runner.run(initial, 6, observer=lambda t, v: published.append((t, v)))
    error = caught.value
    assert isinstance(error.__cause__, ValueError)
    assert "calibrated domain" in str(error.__cause__)
    np.testing.assert_allclose(error.last_valid_state.time, .1)
    np.testing.assert_allclose(error.failed_state.time, .2)
    np.testing.assert_array_equal(error.last_valid_state.trajectory_id, initial.trajectory_id)
    assert len(published) == 2
    np.testing.assert_allclose(np.concatenate([t for t, _ in published], axis=0)[-1], .1)
    # Failure releases the runner lifecycle guard and does not contaminate state.
    np.testing.assert_allclose(runner.run(initial, 2).final_state.time, .1)


def test_invalid_initial_observation_never_publishes():
    runner, initial = make_runner(LimitedMeasurement(limit=-1.))
    published = []
    with pytest.raises(SimulationError, match="initial observation validation") as caught:
        runner.run(initial, 0, observer=lambda *args: published.append(args))
    assert caught.value.last_valid_state is initial
    assert caught.value.failed_state is None
    assert published == []


@pytest.mark.parametrize("checked", [False, True])
def test_sparse_observations_validate_only_requested_blocks(checked):
    runner, initial = make_runner(save_every=5, checked=checked)
    published = []
    with pytest.raises(SimulationError, match="observation validation") as caught:
        runner.run(initial, 6, observer=lambda *args: published.append(args))
    # Two chunks with no requested output succeed; the saved t=.25 invalidates
    # the last chunk, retaining its entry even though it was not published.
    np.testing.assert_allclose(caught.value.last_valid_state.time, .2)
    np.testing.assert_allclose(caught.value.failed_state.time, .3)
    assert len(published) == 1


def test_no_output_skips_measurement_and_its_validation():
    @dataclass(frozen=True)
    class NeverObserve(LimitedMeasurement):
        def evaluate(self, problem, state):
            raise AssertionError("unrequested measurement was traced")

        def validate_observations(self, values):
            raise AssertionError("unrequested measurement was validated")

    runner, initial = make_runner(NeverObserve())
    result = runner.run(initial, 5, collect=False)
    np.testing.assert_allclose(result.final_state.time, .25)
    assert result.times.size == 0 and result.observables == {}


def test_measurement_without_hook_can_report_undefined_diagnostics():
    measurement = FunctionalMeasurement(lambda problem, state: {"undefined": jnp.nan})
    runner, initial = make_runner(measurement)
    result = runner.run(initial, 2)
    assert np.isnan(result.observables["undefined"]).all()
