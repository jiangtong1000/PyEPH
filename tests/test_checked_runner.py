"""Public checked propagation, sampling, parameter reuse and strict restart."""

from dataclasses import dataclass, replace
import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pyeph import (CPA, CoupledClassical, Ehrenfest, Execution, Integrator,
                   LanczosOptions, PrescribedPath, Problem, RecordedCPA,
                   Simulation, make_state, stack_states)
from pyeph.execution.runner import SimulationError
from pyeph.models.analytic import SpinBosonModel
from pyeph.paths.electronic import FixedBasisElectronicPath
from pyeph.paths.nuclear import RecordedNuclearPath


def _assert_state(a, b, *, exact=False):
    assert jax.tree.structure(a) == jax.tree.structure(b)
    for actual, expected in zip(jax.tree.leaves(a), jax.tree.leaves(b), strict=True):
        if exact:
            np.testing.assert_array_equal(actual, expected)
        else:
            np.testing.assert_allclose(actual, expected, atol=3e-13, rtol=3e-13)


def _make(method="cpa", *, batch=False, columns=False, chunk=3, stride=2, jit=True,
          electronic=None, measurement=None):
    model = SpinBosonModel()
    path = RecordedNuclearPath([2., 4.], [[.2], [.6]], [[.2], [.2]])
    treatment = PrescribedPath(path) if method == "cpa" else CoupledClassical(1.)
    problem = Problem(model, model.default_params(), treatment,
                      CPA() if method == "cpa" else Ehrenfest(), measurement)
    sim = Simulation(problem, Integrator(.03, electronic or LanczosOptions(), 2),
                     Execution(chunk_size=chunk, save_every=stride, jit=jit))
    c = np.array([[1., .2j, 0.], [0., .3, 0.]]) if columns else np.array([1., 0.])
    states = [make_state([.2], [.2], c, time=2., step=7, trajectory_id=10+i,
                         method_state={"user": np.array([11, 23], dtype=np.uint64)})
              for i in range(2 if batch else 1)]
    return sim, stack_states(states) if batch else states[0]


@pytest.mark.parametrize("method,batch,columns", [
    ("cpa", False, False), ("cpa", False, True), ("cpa", True, True),
    ("ehrenfest", False, False), ("ehrenfest", True, False),
])
@pytest.mark.parametrize("jit", [False, True])
def test_checked_output_matches_dense_frozen_action(method, batch, columns, jit):
    sim, initial = _make(method, batch=batch, columns=columns, jit=jit)
    dense, _ = _make(method, batch=batch, columns=columns, jit=jit,
                     electronic="exponential_midpoint")
    result, reference = sim.run(initial, 7), dense.run(initial, 7)
    _assert_state(result.final_state, reference.final_state)
    np.testing.assert_array_equal(result.times, reference.times)
    for name in result.observables:
        np.testing.assert_allclose(result.observables[name], reference.observables[name],
                                   atol=3e-13, rtol=3e-13)
    assert result.metadata["electronic_integrator"] == "lanczos_midpoint"
    json.dumps(result.metadata)
    assert int(np.asarray(result.final_state.step).flat[0]) == 14
    np.testing.assert_array_equal(result.final_state.method_state["user"],
                                  initial.method_state["user"])


@pytest.mark.parametrize("method", ["cpa", "ehrenfest"])
def test_checked_checkpoint_resume_and_option_identity(tmp_path, method):
    sim, initial = _make(method, batch=True)
    expected = sim.run(initial, 13).final_state
    partial = sim.run(initial, 5).final_state
    path = tmp_path / "checked.h5"
    sim.save_checkpoint(path, partial)
    resumed, _ = _make(method, batch=True, chunk=5, stride=3)
    actual = resumed.run(resumed.load_checkpoint(path), 8).final_state
    _assert_state(actual, expected)
    changed = Simulation(sim.problem, replace(sim.integrator, electronic=LanczosOptions(rtol=1e-8)))
    with pytest.raises(ValueError, match="integrator"):
        changed.load_checkpoint(path)


def test_checked_parameter_update_reuses_compiled_blocks():
    sim, initial = _make()
    original = sim.run(initial, 6)
    cache = dict(sim._compiled)
    sim.update_parameters({**sim.problem.params, "delta": .6})
    result = sim.run(initial, 6)
    fresh = Simulation(sim.problem, sim.integrator, sim.execution).run(initial, 6)
    assert sim._compiled == cache
    _assert_state(result.final_state, fresh.final_state)
    assert not np.allclose(result.final_state.electronic, original.final_state.electronic)


@dataclass(frozen=True)
class NoEvaluation:
    def validate(self, problem):
        pass

    def evaluate(self, problem, state):
        raise AssertionError("no-output run must not trace a measurement")


def test_checked_no_output_has_no_measurement_tracing_even_at_zero_steps():
    sim, initial = _make(measurement=NoEvaluation())
    for steps in (0, 5):
        result = sim.run(initial, steps, collect=False)
        assert result.times.size == 0 and result.observables == {}
        assert int(result.final_state.step) == 7 + steps


@dataclass(frozen=True)
class LaterCouplingModel(SpinBosonModel):
    def apply(self, params, q, vectors):
        coupling = jnp.where(q[0] > .14, .3, 0.)
        matrix = jnp.array([[q[0], coupling], [coupling, -q[0]]])
        return matrix @ vectors


@pytest.mark.parametrize("collect,check_finite", [(True, True), (False, False)])
def test_failed_chunk_is_private_and_checkpoint_retry_is_deterministic(tmp_path, collect,
                                                                      check_finite):
    model = LaterCouplingModel()
    path = RecordedNuclearPath([2., 3.], [[0.], [1.]], [[1.], [1.]])
    problem = Problem(model, model.default_params(), PrescribedPath(path), CPA())
    sim = Simulation(problem, Integrator(.05, LanczosOptions(max_dimension=1)),
                     Execution(chunk_size=2, check_finite=check_finite))
    initial = make_state([0.], [1.], [1., 0.], time=2., step=7, trajectory_id=97,
                         method_state={"counter": np.array(19, dtype=np.int64)})
    published = []
    observer = (lambda times, values: published.extend(times.tolist())) if collect else None
    with pytest.raises(SimulationError, match="checked electronic") as caught:
        sim.run(initial, 7, observer=observer, collect=collect)
    error = caught.value
    assert error.__cause__ is None
    assert int(error.last_valid_state.step) == 9
    assert int(error.failed_state.step) == 10
    assert int(error.diagnostics["failed_macro_index"]) == 1
    assert int(error.diagnostics["step_info"].code) == 1
    assert int(error.diagnostics["trajectory_ids"]) == 97
    np.testing.assert_allclose(error.diagnostics["attempted_time"], 2.15, atol=1e-15)
    np.testing.assert_allclose(published, [2., 2.05, 2.1] if collect else [])
    checkpoint = tmp_path / "retained.h5"
    sim.save_checkpoint(checkpoint, error.last_valid_state,
                        artifact_ids={"model": "later-coupling-test-v1"})
    restored = sim.load_checkpoint(checkpoint,
                                   artifact_ids={"model": "later-coupling-test-v1"})
    with pytest.raises(SimulationError) as retry:
        sim.run(restored, 5, collect=False)
    _assert_state(retry.value.last_valid_state, error.last_valid_state, exact=True)
    _assert_state(retry.value.failed_state, error.failed_state, exact=True)
    capable = Simulation(problem, Integrator(.05, LanczosOptions(max_dimension=2)), sim.execution)
    _assert_state(capable.run(restored, 5).final_state, capable.run(initial, 7).final_state)


@pytest.mark.parametrize("dtype", [jnp.float64, jnp.complex64])
def test_checked_precision_rejected_before_observer(dtype):
    sim, initial = _make()
    electronic = initial.electronic.real if dtype == jnp.float64 else initial.electronic.astype(dtype)
    initial = initial._replace(electronic=electronic)
    calls = []
    with pytest.raises(ValueError, match="complex128"):
        sim.run(initial, 0, observer=lambda *args: calls.append(args))
    assert not calls


def test_checked_time_overflow_rejected_before_output_even_when_optional_checks_disabled():
    sim, initial = _make("ehrenfest")
    sim = Simulation(sim.problem, replace(sim.integrator, dt=1e308),
                     Execution(check_finite=False))
    calls = []
    with pytest.raises(ValueError, match="overflow the time"):
        sim.run(initial, 2, observer=lambda *args: calls.append(args))
    assert not calls


@pytest.mark.parametrize("field", ["q", "p", "time"])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.int64])
def test_checked_mixed_or_integer_nuclear_precision_rejected_before_output(field, dtype):
    sim, initial = _make("ehrenfest")
    initial = initial._replace(**{field: getattr(initial, field).astype(dtype)})
    calls = []
    with pytest.raises(ValueError, match="float64 coordinates"):
        sim.run(initial, 1, observer=lambda *args: calls.append(args))
    assert not calls


def test_checked_unsupported_profile_rejected_and_old_names_unchanged():
    path = FixedBasisElectronicPath([0., 1.], np.zeros((2, 2, 2)))
    with pytest.raises(ValueError, match="RecordedCPA.*checked"):
        RecordedCPA(path, Integrator(.01, LanczosOptions()))
    for name in ("rk4", "exponential_midpoint"):
        assert Integrator(.01, name).electronic_name == name
    for invalid in ("lanczos", {}, [], True):
        with pytest.raises(ValueError, match="electronic integrator"):
            Integrator(.01, invalid)


def test_checked_method_and_external_provider_require_explicit_support():
    class OrdinaryOnly(CPA):
        build_checked_step = None

    class ExternalModel(SpinBosonModel):
        execution_mode = "host_callback"

        def __post_init__(self):
            super().__post_init__()
            object.__setattr__(self, "spec", replace(self.spec, native_jax=False))

    sim, _ = _make()
    with pytest.raises(ValueError, match="build_checked_step"):
        Simulation(replace(sim.problem, method=OrdinaryOnly()), sim.integrator)
    external = replace(sim.problem, model=ExternalModel())
    with pytest.raises(ValueError, match="native JAX"):
        Simulation(external, sim.integrator, Execution(allow_host_callbacks=True))


def test_checked_observer_keeps_original_exception():
    sim, initial = _make()
    failure = OSError("test output unavailable")

    def observer(times, values):
        if times[-1] > 2.:
            raise failure

    with pytest.raises(OSError) as caught:
        sim.run(initial, 7, observer=observer)
    assert caught.value is failure
    assert int(sim.run(initial, 1, collect=False).final_state.step) == 8
