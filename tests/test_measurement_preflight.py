"""Workflow origins are checked before propagation, even without output."""

from dataclasses import dataclass, field

import jax.numpy as jnp
import numpy as np
import pytest

from pyeph import CoupledClassical, Ehrenfest, Execution, Integrator, Problem, Simulation, make_state
from pyeph.core.state import stack_states
from pyeph.io.checkpoint import load_checkpoint, save_checkpoint
from pyeph.models.analytic import SpinBosonModel


@dataclass(frozen=True)
class OriginMeasurement:
    calls: list = field(default_factory=list)

    def validate(self, problem):
        pass

    def validate_initial_state(self, problem, state, *, batch=False):
        self.calls.append(batch)
        expected = np.asarray(problem.params["bias"])
        if np.any(np.asarray(state.method_state["origin_bias"]) != expected):
            raise ValueError("origin belongs to different parameters")

    def evaluate(self, problem, state):
        return {"time": state.time}


def make_runner():
    model = SpinBosonModel()
    measurement = OriginMeasurement()
    problem = Problem(model, model.default_params(), CoupledClassical(1.), Ehrenfest(), measurement)
    return Simulation(problem, Integrator(.01), Execution(chunk_size=2)), measurement


def initial(runner, *, batch=False, bad=False):
    bias = runner.problem.params["bias"]
    states = []
    for i in (3, 7):
        state = make_state([.2], [.1], [1., 0.], trajectory_id=i)
        value = bias + (1. if bad and i == 7 else 0.)
        states.append(state._replace(method_state={"origin_bias": jnp.asarray(value)}))
    return stack_states(states) if batch else states[-1]


@pytest.mark.parametrize("batch", [False, True])
@pytest.mark.parametrize("collect", [False, True])
def test_origin_preflight_once_on_host_and_before_publication(batch, collect):
    runner, measurement = make_runner()
    published = []
    with pytest.raises(ValueError, match="origin belongs"):
        runner.run(initial(runner, batch=batch, bad=True), 0, collect=collect,
                   observer=lambda *args: published.append(args))
    assert published == [] and measurement.calls == [batch]
    good = initial(runner, batch=batch)
    result = runner.run(good, 5, collect=collect)
    assert measurement.calls == [batch, batch]
    np.testing.assert_array_equal(result.final_state.step, np.asarray(good.step)+5)
    runner.update_parameters({**runner.problem.params, "bias": .23})
    with pytest.raises(ValueError, match="origin belongs"):
        runner.run(result.final_state, 1, collect=False)


@pytest.mark.parametrize("batch", [False, True])
def test_origin_preflight_on_checkpoint_save_and_load(batch, tmp_path):
    runner, measurement = make_runner()
    identities = {"measurement": "origin-measurement-test-v1"}
    # The mutable call log belongs only to this test spy; keep the serialized
    # dataclass fields identical across save/load so the scientific manifest
    # does not intentionally reject our observation of the hook itself.
    good = initial(runner, batch=batch)
    saved = tmp_path/"valid.h5"
    runner.save_checkpoint(saved, good, artifact_ids=identities)
    assert measurement.calls == [batch]
    measurement.calls.clear()
    runner.load_checkpoint(saved, artifact_ids=identities)
    assert measurement.calls == [batch]
    bad = initial(runner, batch=batch, bad=True)
    measurement.calls.clear()
    with pytest.raises(ValueError, match="origin belongs"):
        runner.save_checkpoint(tmp_path/"invalid.h5", bad, artifact_ids=identities)
    assert not (tmp_path/"invalid.h5").exists()
    _, metadata = load_checkpoint(saved)
    save_checkpoint(tmp_path/"tampered.h5", bad, metadata=metadata)
    measurement.calls.clear()
    with pytest.raises(ValueError, match="origin belongs"):
        runner.load_checkpoint(tmp_path/"tampered.h5", artifact_ids=identities)
