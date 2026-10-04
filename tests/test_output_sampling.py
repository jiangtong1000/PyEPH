"""Sparse output changes observations/storage, never propagation or failure checks."""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pyeph.core.problem import CoupledClassical, Problem
from pyeph.core.state import make_state, stack_states
from pyeph.dynamics.ehrenfest import Ehrenfest
from pyeph.dynamics.mash2 import MASH2, MASHError, MASHPopulation, mapping_state
from pyeph.execution.runner import Execution
from pyeph.integrators.electronic import Integrator
from pyeph.models.analytic import SpinBosonModel
from pyeph.observables.population import FunctionalMeasurement
from pyeph.simulation import Simulation


def nested_measurement(problem, state):
    return {"nuclei": (state.q, {"momentum": state.p}),
            "electronic": [state.electronic, jnp.outer(state.electronic, state.electronic.conj())],
            "step": state.step, "positive_q": state.q > 0,
            "scalar": jnp.sum(state.q**2)}


def simulation(*, stride=1, chunk=7, measurement=None, jit=True):
    model = SpinBosonModel()
    params = dict(omega=jnp.array([0.7]), coupling=jnp.array([0.1]), q_eq=jnp.array([0.]),
                  delta=0.2, bias=0.)
    problem = Problem(model, params, CoupledClassical(1.), Ehrenfest(),
                      measurement or FunctionalMeasurement(nested_measurement))
    return Simulation(problem, Integrator(0.05, "exponential_midpoint"),
                      Execution(jit=jit, chunk_size=chunk, save_every=stride))


def initial(*, batch=False):
    states = [make_state([0.4+i*.1], [0.2-i*.05], [np.sqrt(.8), 1j*np.sqrt(.2)],
                         time=1.3, step=7, trajectory_id=i)
              for i in range(3)]
    return stack_states(states) if batch else states[0]


def assert_same_tree(left, right, *, atol=2e-14):
    assert jax.tree.structure(left) == jax.tree.structure(right)
    jax.tree.map(lambda a, b: np.testing.assert_allclose(a, b, rtol=0, atol=atol), left, right)


@pytest.mark.parametrize("batch", [False, True])
@pytest.mark.parametrize("jit", [False, True])
def test_sparse_nested_observations_match_all_step_reference(batch, jit):
    state = initial(batch=batch)
    all_steps = simulation(chunk=5, jit=jit).run(state, 23)
    sparse = simulation(stride=4, chunk=6, jit=jit).run(state, 23)
    indices = [0] + [i for i in range(1, 24) if (7+i) % 4 == 0 or i == 23]
    assert_same_tree(sparse.final_state, all_steps.final_state)
    np.testing.assert_allclose(sparse.times, all_steps.times[indices], atol=5e-16)
    assert_same_tree(sparse.observables,
                     jax.tree.map(lambda x: x[indices], all_steps.observables))
    assert sparse.observables["step"].dtype == np.asarray(state.step).dtype
    assert sparse.observables["positive_q"].dtype == bool


def test_restart_schedule_is_absolute_with_initial_and_final_always_retained():
    state = initial()
    runner = simulation(stride=4, chunk=3)
    whole = runner.run(state, 23)
    first = runner.run(state, 9)
    second = runner.run(first.final_state, 14)
    # First ends at absolute16 (already scheduled); second starts there.
    np.testing.assert_array_equal(first.observables["step"], [7, 8, 12, 16])
    np.testing.assert_array_equal(second.observables["step"], [16, 20, 24, 28, 30])
    assert_same_tree(second.final_state, whole.final_state)
    np.testing.assert_allclose(np.concatenate((first.times, second.times[1:])), whole.times,
                               atol=5e-16)
    emitted = []
    streamed = runner.run(state, 23, collect=False,
                          observer=lambda t, x: emitted.append((t, x)))
    assert streamed.observables == {} and streamed.times.size == 0
    assert_same_tree(streamed.final_state, whole.final_state)
    np.testing.assert_allclose(np.concatenate([item[0] for item in emitted]), whole.times,
                               atol=5e-16)


def test_measurement_executes_only_for_selected_states_including_zero_output_chunks():
    observed = []

    def measure(problem, state):
        # An effectful callback is test instrumentation only. Scientific
        # measurements remain pure; this exposes runtime invocation counts.
        jax.debug.callback(lambda step: observed.append(int(step)), state.step, ordered=True)
        return {"step": state.step}

    runner = simulation(stride=10, chunk=2, measurement=FunctionalMeasurement(measure))
    result = runner.run(initial(), 16)
    jax.effects_barrier()
    assert observed == [7, 10, 20, 23]
    np.testing.assert_array_equal(result.observables["step"], observed)
    # Several two-step chunks have no selected state; no measurement is traced
    # in those blocks and no (zero-length) observer calls are emitted.
    assert any(len(key) == 3 and isinstance(key[0], bool) and not key[2]
               for key in runner._compiled)


class MustNotEvaluate:
    def __init__(self):
        self.validations = 0

    def validate(self, problem):
        self.validations += 1

    def evaluate(self, problem, state):
        raise AssertionError("unconsumed measurement must not be traced or evaluated")


def test_no_output_execution_skips_initial_and_step_measurements_but_validates():
    measurement = MustNotEvaluate()
    runner = simulation(stride=1, chunk=3, measurement=measurement)
    state = initial()
    result = runner.run(state, 11, collect=False)
    reference = simulation(chunk=3).run(state, 11)
    assert result.observables == {} and result.times.size == 0
    assert_same_tree(result.final_state, reference.final_state)
    assert measurement.validations > 0
    empty = runner.run(state, 0, collect=False)
    assert_same_tree(empty.final_state, state)
    with pytest.raises(ValueError, match="normalized"):
        runner.run(state._replace(electronic=2*state.electronic), 1, collect=False)


def test_private_block_default_and_sparse_buffers_and_zero_shape():
    runner, state = simulation(), initial(batch=True)
    dense = runner._block(True, 11)(runner.problem.params, state)
    selected = np.array([1, 7, 10])
    sparse = runner._block(True, 11, selected)(runner.problem.params, state)
    assert_same_tree(dense[0], sparse[0])
    assert_same_tree(jax.tree.map(lambda x: x[selected], dense[1]), sparse[1])
    assert all(x.shape[0] == 3 for x in jax.tree.leaves(sparse[1]))
    zero = runner._block(True, 11, [])(runner.problem.params, state)
    assert_same_tree(zero[0], dense[0])
    assert zero[1][0].shape == (0, 3) and zero[1][1] == {}
    # Equal schedules share the same function irrespective of input container.
    assert runner._block(True, 11, selected) is runner._block(True, 11, [1, 7, 10])
    assert runner._block(True, 11) is runner._block(True, 11, np.arange(11))


@pytest.mark.parametrize("indices", [[-1], [3], [1, 1], [2, 1], [1.], [[1]],
                                     np.array([2, 1], dtype=np.uint32)])
def test_private_block_rejects_invalid_sample_schedules(indices):
    with pytest.raises(ValueError, match="sample_indices"):
        simulation()._block(False, 3, indices)


def test_no_output_mash_failure_preserves_physical_time_and_chunk_checkpoint():
    class DegenerateInBand(SpinBosonModel):
        def apply(self, params, q, vectors):
            action = super().apply(params, q, vectors)
            return jnp.where((q[0] > .0003) & (q[0] < .5), 0., action)

    # Initial spectra pass model-aware preflight. The first lane enters an
    # unsupported degeneracy during propagation; the other lane stays valid.
    model = DegenerateInBand()
    good = model.default_params()
    states = [mapping_state(model, good, [q], [0.1], [0, 0, -1],
                            trajectory_id=i, time=5., step=7)
              for i, q in enumerate([0., 1.])]
    runner = Simulation(Problem(model, good, CoupledClassical(1.), MASH2(), MASHPopulation()),
                        Integrator(.01, "exponential_midpoint"),
                        Execution(chunk_size=4, save_every=100))
    state = stack_states(states)
    with pytest.raises(MASHError) as caught:
        runner.run(state, 8, collect=False)
    failed = caught.value.failed_state
    np.testing.assert_allclose(failed.time, [5.0025, 5.04], atol=1e-14)
    np.testing.assert_array_equal(failed.step, [7, 11])
    np.testing.assert_array_equal(failed.method_state["status"], [1, 0])
    assert_same_tree(caught.value.last_valid_state, state)


@dataclass(frozen=True)
class CallbackMeasurement:
    def validate(self, problem):
        pass

    def evaluate(self, problem, state):
        value = jax.pure_callback(lambda q: np.asarray(np.sum(q*q), dtype=q.dtype),
                                  jax.ShapeDtypeStruct((), state.q.dtype), state.q,
                                  vmap_method="sequential")
        return {"host_value": value, "complex": state.electronic[0]}


def test_sparse_abstract_shape_supports_host_callbacks_and_runtime_parameters():
    runner = simulation(stride=3, chunk=7, measurement=CallbackMeasurement())
    state = initial(batch=True)
    selected = np.array([2, 5, 6])
    block = runner._block(True, 7, selected)
    baseline = block(runner.problem.params, state)
    params = {**runner.problem.params, "delta": 0.7}
    changed = block(params, state)
    dense = runner._block(True, 7)(params, state)
    assert not np.allclose(changed[0].electronic, baseline[0].electronic)
    assert_same_tree(changed[0], dense[0])
    assert_same_tree(changed[1], jax.tree.map(lambda x: x[selected], dense[1]))
