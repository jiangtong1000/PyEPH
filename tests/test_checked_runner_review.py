"""Independent runtime-gating audit; callbacks here are test instrumentation."""

from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pyeph.core.contracts import ModelSpec
from pyeph.core.problem import CoupledClassical, PrescribedPath, Problem
from pyeph.core.state import make_state, stack_states
from pyeph.core.system import SystemSpec
from pyeph.dynamics.cpa import CPA
from pyeph.dynamics.ehrenfest import Ehrenfest
from pyeph.execution.runner import Execution, SimulationError
from pyeph.integrators.electronic import Integrator
from pyeph.integrators.krylov import LanczosOptions
from pyeph.observables.population import FunctionalMeasurement
from pyeph.simulation import Simulation


@dataclass(frozen=True)
class GuardModel:
    """A discontinuous test fixture used only to force a known action rejection."""

    actions: list = field(default_factory=list)
    forces: list = field(default_factory=list)
    spec: ModelSpec = field(default_factory=lambda: ModelSpec(SystemSpec(2, (1,))))

    def apply(self, params, q, vectors):
        jax.debug.callback(lambda x: self.actions.append(float(x)), q[0], ordered=True)
        coupling = jnp.where(q[0] > params["switch"], params["coupling"], 0.)
        return jnp.array([[0., coupling], [coupling, 1.]]) @ vectors

    def reference_gradient(self, params, q):
        jax.debug.callback(lambda x: self.forces.append(("reference", float(x))),
                           q[0], ordered=True)
        return jnp.zeros_like(q)

    def contract_gradient(self, params, q, weight):
        jax.debug.callback(lambda x: self.forces.append(("electronic", float(x))),
                           q[0], ordered=True)
        return jnp.zeros_like(q)


@dataclass(frozen=True)
class ClockPath:
    calls: list = field(default_factory=list)

    def position(self, time):
        jax.debug.callback(lambda x: self.calls.append(float(x)), time, ordered=True)
        return jnp.asarray([time])

    def velocity(self, time):
        return jnp.ones(1)


def assert_identical(left, right):
    assert jax.tree.structure(left) == jax.tree.structure(right)
    for a, b in zip(jax.tree.leaves(left), jax.tree.leaves(right), strict=True):
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("stride", [1, 2, 100])
def test_one_bad_column_freezes_entire_batch_and_discards_private_chunk(stride):
    model, path, measured, published = GuardModel(), ClockPath(), [], []

    def measure(problem, state):
        jax.debug.callback(lambda identity, step: measured.append((int(identity), int(step))),
                           state.trajectory_id, state.step, ordered=True)
        return {"step": state.step, "nested": (state.electronic,
                {"momentum": state.p, "positive": state.q > 0})}

    sim = Simulation(Problem(model, {"switch": 2.6, "coupling": .4},
                     PrescribedPath(path), CPA(), FunctionalMeasurement(measure)),
                     Integrator(.1, LanczosOptions(max_dimension=1)),
                     Execution(chunk_size=4, save_every=stride, check_finite=False))
    columns = np.array([[1., 0.], [0., 0.]])
    states = [make_state([time], [0.], columns, time=time, step=7, trajectory_id=identity,
                         method_state={"tracking": (np.array([3, 4], np.uint64),
                                                     {"flag": np.array(True)})})
              for time, identity in ((1.3, 17), (2.3, 29))]
    initial = stack_states(states)._replace(step=jnp.asarray([7, 7], dtype=jnp.int16))
    with pytest.raises(SimulationError, match="checked electronic") as caught:
        sim.run(initial, 8, observer=lambda t, v: published.append((t, v)))
    jax.effects_barrier()
    error = caught.value
    assert error.__cause__ is None
    assert_identical(error.last_valid_state, initial)
    np.testing.assert_array_equal(error.failed_state.step, [10, 10])
    assert error.failed_state.step.dtype == jnp.int16
    np.testing.assert_allclose(error.failed_state.time, [1.6, 2.6], atol=1e-15)
    np.testing.assert_array_equal(error.failed_state.trajectory_id, [17, 29])
    assert_identical(error.failed_state.method_state, initial.method_state)
    np.testing.assert_array_equal(error.failed_state.key, initial.key)
    info = error.diagnostics["step_info"]
    assert int(error.diagnostics["failed_macro_index"]) == 3
    np.testing.assert_array_equal(info.failed_trajectories, [False, True])
    np.testing.assert_array_equal(info.action.status, [[0, 0], [1, 0]])
    assert info.action.error_estimate.shape == (2, 2)
    # Earlier accepted rows were measured privately, but none of this failed
    # chunk is public. The initial row is a separate, already accepted boundary.
    assert len(published) == 1
    np.testing.assert_array_equal(published[0][0], np.array([[1.3, 2.3]]))
    expected_steps = [7] + [step for step in (8, 9, 10) if step % stride == 0]
    assert sorted(measured) == sorted((identity, step) for identity in (17, 29)
                                     for step in expected_steps)
    assert max(model.actions) <= 2.65 + 1e-14
    assert max(path.calls) <= 2.65 + 1e-14  # No rejected endpoint at 2.7.
    assert not model.forces


def test_batched_first_half_rejection_suppresses_all_force_stages_and_later_macros():
    model = GuardModel()
    sim = Simulation(Problem(model, {"switch": 0., "coupling": .4},
                              CoupledClassical(1.), Ehrenfest()),
                     Integrator(.2, LanczosOptions(max_dimension=1), electronic_substeps=3),
                     Execution(chunk_size=5, check_finite=False))
    initial = stack_states([make_state([q], [.2], [1., 0.], time=time, step=11,
                                      trajectory_id=identity, method_state={"tag": np.array(3j)})
                            for q, time, identity in ((-1., 1.4, 19), (1., 3.7, 41))])
    with pytest.raises(SimulationError) as caught:
        sim.run(initial, 7, collect=False)
    jax.effects_barrier()
    assert not model.forces
    assert_identical(caught.value.failed_state, initial)
    assert_identical(caught.value.last_valid_state, initial)
    info = caught.value.diagnostics["step_info"]
    np.testing.assert_array_equal(info.failed_trajectories, [False, True])
    np.testing.assert_array_equal(info.action.status, [0, 1])
    assert int(info.substep) == 0
    assert int(caught.value.diagnostics["failed_macro_index"]) == 0


def test_second_half_rejection_rolls_back_nuclei_and_no_rejected_measurement_runs():
    model, measured = GuardModel(), []

    def measure(problem, state):
        jax.debug.callback(lambda s: measured.append(int(s)), state.step, ordered=True)
        return {"coordinate": state.q, "step": state.step}

    sim = Simulation(Problem(model, {"switch": .4, "coupling": .4},
                              CoupledClassical(1.), Ehrenfest(), FunctionalMeasurement(measure)),
                     Integrator(.5, LanczosOptions(max_dimension=1)),
                     Execution(chunk_size=4, check_finite=False))
    initial = make_state([0.], [2.], [1., 0.], time=3.7, step=11, trajectory_id=57,
                         seed=12, method_state={"bookkeeping": [np.array([7, 8]), np.array(False)]})
    with pytest.raises(SimulationError) as caught:
        sim.run(initial, 4)
    jax.effects_barrier()
    assert_identical(caught.value.failed_state, initial)
    assert_identical(caught.value.last_valid_state, initial)
    assert measured == [11]
    assert sorted(model.forces) == [("electronic", 0.), ("electronic", 1.),
                                    ("reference", 0.), ("reference", 1.)]
    assert max(model.actions) == 1.
    assert int(caught.value.diagnostics["step_info"].code) == 1
