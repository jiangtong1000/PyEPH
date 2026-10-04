"""Runner configuration cannot silently diverge from a cached propagator."""

from dataclasses import dataclass, replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pyeph import CoupledClassical, Ehrenfest, Execution, Integrator, Problem, Simulation
from pyeph.core.state import make_state, stack_states
from pyeph.dynamics.recorded import RecordedCPA
from pyeph.io.provenance import problem_manifest
from pyeph.models.epc import LinearEPCModel
from pyeph.observables.population import ElectronicPopulation
from pyeph.paths.electronic import AdiabaticElectronicPath, FixedBasisElectronicPath


@dataclass(frozen=True)
class CheckedEhrenfest(Ehrenfest):
    def validate(self, problem):
        super().validate(problem)
        if abs(np.asarray(problem.params["h0"])[0, 1]) > 0.8:
            raise ValueError("test method coupling limit")


@dataclass(frozen=True)
class ParameterMeasurement:
    def validate(self, problem):
        if problem.params["reference_offset"] > 0.5:
            raise ValueError("test measurement offset limit")

    def evaluate(self, problem, state):
        return {"coupling": problem.params["h0"][0, 1],
                "coordinate": state.q,
                "population": jnp.abs(state.electronic)**2}


def _simulation(*, measurement=None, method=None, jit=True):
    model = LinearEPCModel(2, 1)
    params = model.create_params([[0.1, 0.2], [0.2, -0.1]],
                                 [[[0.2, 0.0], [0.0, -0.2]]], omega=[0.7])
    problem = Problem(model, params, CoupledClassical(1.0), method or Ehrenfest(), measurement)
    return Simulation(problem, Integrator(0.03, "exponential_midpoint"),
                      Execution(jit=jit, chunk_size=8, save_every=2))


def _initial(batch=False):
    states = [make_state([0.2 + i*0.1], [0.3], [1, 0], trajectory_id=i)
              for i in range(2 if batch else 1)]
    return stack_states(states) if batch else states[0]


def _assert_state(actual, expected):
    for a, b in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
        np.testing.assert_allclose(a, b, atol=2e-14, rtol=2e-14)


@pytest.mark.parametrize("compiled", [False, True])
def test_static_configuration_is_read_only_before_and_after_compilation(compiled):
    sim = _simulation()
    initial = _initial()
    if compiled:
        expected = sim.run(initial, 6)
    replacements = {
        "problem": replace(sim.problem, nuclear_treatment=CoupledClassical(2.0)),
        "integrator": Integrator(0.06, "exponential_midpoint"),
        "execution": Execution(jit=False),
        "measurement": ParameterMeasurement(),
    }
    for name, value in replacements.items():
        original = getattr(sim, name)
        with pytest.raises(AttributeError):
            setattr(sim, name, value)
        assert getattr(sim, name) is original
    if compiled:
        _assert_state(sim.run(initial, 6).final_state, expected.final_state)


@pytest.mark.parametrize("batch", [False, True])
@pytest.mark.parametrize("steps", [0, 6])
def test_validated_parameter_update_reuses_cache_and_updates_all_measurements(batch, steps):
    sim = _simulation(measurement=ParameterMeasurement())
    initial = _initial(batch)
    old = sim.run(initial, steps)
    compiled = dict(sim._compiled)
    params = {**sim.problem.params, "h0": jnp.array([[0.1, 0.6], [0.6, -0.1]])}
    sim.update_parameters(params)
    actual = sim.run(initial, steps)
    fresh = Simulation(sim.problem, sim.integrator, sim.execution).run(initial, steps)
    assert sim._compiled == compiled
    assert sim.problem.params is params
    assert sim.measurement is sim.problem.measurement
    np.testing.assert_array_equal(actual.observables["coupling"], 0.6)
    if steps:
        assert not np.allclose(old.final_state.electronic, actual.final_state.electronic)
    _assert_state(actual.final_state, fresh.final_state)
    for name in actual.observables:
        np.testing.assert_allclose(actual.observables[name], fresh.observables[name],
                                   atol=2e-14, rtol=2e-14)


@pytest.mark.parametrize("jit", [False, True])
def test_repeated_zero_step_measurement_uses_current_scalar_and_batch_state(jit):
    sim = _simulation(measurement=ParameterMeasurement(), jit=jit)
    for batch in (False, True):
        initial = _initial(batch)
        sim.run(initial, 0)
        cached = dict(sim._compiled)
        changed = initial._replace(q=initial.q + 0.7, time=initial.time + 1.2,
                                   electronic=jnp.flip(initial.electronic, axis=-1))
        result = sim.run(changed, 0)
        assert sim._compiled == cached
        assert result.final_state is changed
        np.testing.assert_array_equal(result.times, np.asarray(changed.time)[None])
        np.testing.assert_array_equal(result.observables["coordinate"], np.asarray(changed.q)[None])
        np.testing.assert_array_equal(result.observables["population"],
                                      np.abs(np.asarray(changed.electronic))[None]**2)


@pytest.mark.parametrize("failure", ["model", "method", "measurement"])
def test_failed_parameter_update_is_atomic_and_runs_all_problem_validators(failure):
    sim = _simulation(measurement=ParameterMeasurement(), method=CheckedEhrenfest())
    expected = sim.run(_initial(), 6)
    original_problem = sim.problem
    compiled = dict(sim._compiled)
    params = dict(original_problem.params)
    if failure == "model":
        params["h0"] = jnp.array([[0.1, 0.6], [0.2, -0.1]])
        match = "Hermitian"
    elif failure == "method":
        params["h0"] = jnp.array([[0.1, 0.9], [0.9, -0.1]])
        match = "method coupling limit"
    else:
        params["reference_offset"] = 0.9
        match = "measurement offset limit"
    with pytest.raises(ValueError, match=match):
        sim.update_parameters(params)
    assert sim.problem is original_problem
    assert sim._compiled == compiled
    _assert_state(sim.run(_initial(), 6).final_state, expected.final_state)


def test_updated_parameters_define_strict_checkpoint_identity(tmp_path):
    sim = _simulation()
    initial = _initial()
    old_manifest = problem_manifest(sim.problem, sim.integrator)
    old_file, new_file = tmp_path / "old.h5", tmp_path / "new.h5"
    sim.save_checkpoint(old_file, initial)
    sim.run(initial, 6)
    params = {**sim.problem.params, "h0": jnp.array([[0.1, 0.6], [0.6, -0.1]])}
    sim.update_parameters(params)
    assert isinstance(sim.measurement, ElectronicPopulation)
    assert problem_manifest(sim.problem, sim.integrator) != old_manifest
    with pytest.raises(ValueError, match="params"):
        sim.load_checkpoint(old_file)
    current = sim.run(initial, 6).final_state
    sim.save_checkpoint(new_file, current)
    rebuilt = Simulation(sim.problem, sim.integrator, sim.execution)
    _assert_state(rebuilt.load_checkpoint(new_file), current)
    _assert_state(rebuilt.run(initial, 6).final_state, current)


def _recorded(profile, coupling=0.2):
    if profile == "fixed":
        h = np.array([[0.1, coupling], [coupling, -0.1]])
        path = FixedBasisElectronicPath([0.0, 1.0], [h, h])
        return RecordedCPA(path, Integrator(0.03, "exponential_midpoint"))
    rotation = np.array([[np.cos(coupling), -np.sin(coupling)],
                         [np.sin(coupling), np.cos(coupling)]])
    path = AdiabaticElectronicPath(np.arange(4)*0.03, np.zeros((4, 2)),
                                  np.repeat(rotation[None], 3, axis=0))
    return RecordedCPA(path)


@pytest.mark.parametrize("profile", ["fixed", "adiabatic"])
def test_recorded_configuration_is_frozen_and_new_instance_uses_changed_path(profile):
    sim = _recorded(profile)
    initial = sim.initialize([1, 0])
    expected = sim.run(initial, 2)
    changed = _recorded(profile, coupling=0.6)
    for name, value in {"path": changed.path, "integrator": Integrator(0.06),
                        "execution": Execution(jit=False), "max_subspace_loss": None}.items():
        original = getattr(sim, name)
        with pytest.raises(AttributeError):
            setattr(sim, name, value)
        assert getattr(sim, name) is original
    _assert_state(sim.run(initial, 2).final_state, expected.final_state)
    result = changed.run(changed.initialize([1, 0]), 2)
    assert not np.allclose(result.final_state.electronic, expected.final_state.electronic)


@pytest.mark.parametrize("profile", ["fixed", "adiabatic"])
def test_recorded_step_overflow_is_rejected_before_output(profile):
    sim = _recorded(profile)
    initial = sim.initialize([1, 0])._replace(step=jnp.array(127, dtype=jnp.int8))
    output = []
    with pytest.raises(ValueError, match="overflow recorded step"):
        sim.run(initial, 1, observer=lambda *args: output.append(args))
    assert not output
    sim.run(initial, 0)


def test_recorded_frame_index_overflow_is_rejected_despite_available_data():
    times = np.arange(129)*0.01
    path = AdiabaticElectronicPath(times, np.zeros((129, 2)),
                                  np.repeat(np.eye(2)[None], 128, axis=0))
    sim = RecordedCPA(path)
    initial = sim.initialize([1, 0], frame_index=127)
    narrow = initial._replace(frame_index=jnp.array(127, dtype=jnp.int8))
    with pytest.raises(ValueError, match="overflow recorded frame_index"):
        sim.run(narrow, 1)
    result = sim.run(initial, 1)
    assert result.final_state.frame_index == 128


def test_execution_and_recorded_scalar_options_do_not_alias_mutable_arrays():
    flag, chunk, stride = np.array(True), np.array(8), np.array(2)
    execution = Execution(jit=flag, check_finite=flag, allow_host_callbacks=flag,
                          verify_external_gradients=flag, chunk_size=chunk, save_every=stride)
    flag[...], chunk[...], stride[...] = False, 1, 1
    for name in ("jit", "check_finite", "allow_host_callbacks", "verify_external_gradients"):
        assert getattr(execution, name) is True
    assert execution.chunk_size == 8 and type(execution.chunk_size) is int
    assert execution.save_every == 2 and type(execution.save_every) is int
    limit = np.array(0.1)
    sim = RecordedCPA(_recorded("adiabatic").path, max_subspace_loss=limit)
    limit[...] = 0.5
    assert sim.max_subspace_loss == 0.1 and type(sim.max_subspace_loss) is float


@pytest.mark.parametrize("options", [
    {"jit": 1}, {"check_finite": [True]}, {"allow_host_callbacks": np.nan},
    {"verify_external_gradients": "yes"}, {"chunk_size": True}, {"save_every": 2.0},
])
def test_execution_rejects_ambiguous_static_scalars(options):
    with pytest.raises(ValueError):
        Execution(**options)


@pytest.mark.parametrize("action", ["parameters", "run"])
def test_observer_cannot_update_parameters_or_reenter_same_runner(action):
    sim = _simulation()
    initial = _initial()
    expected = sim.run(initial, 6)
    original_problem = sim.problem
    params = {**sim.problem.params, "h0": jnp.array([[0.1, 0.6], [0.6, -0.1]])}
    visits = []

    def observer(times, values):
        visits.append(times)
        with pytest.raises(RuntimeError, match="running"):
            if action == "parameters":
                sim.update_parameters(params)
            else:
                sim.run(initial, 0)

    actual = sim.run(initial, 6, observer=observer)
    assert len(visits) == 2
    assert sim.problem is original_problem
    _assert_state(actual.final_state, expected.final_state)
    sim.update_parameters(params)
    assert sim.problem.params is params


def test_observer_checkpoint_is_allowed_while_parameter_updates_are_blocked(tmp_path):
    sim = _simulation()
    initial = _initial()
    saved = []

    def observer(times, values):
        path = tmp_path / f"observer-{len(saved)}.h5"
        sim.save_checkpoint(path, initial)
        saved.append(path)

    sim.run(initial, 6, observer=observer)
    assert len(saved) == 2
    for path in saved:
        _assert_state(sim.load_checkpoint(path), initial)


@pytest.mark.parametrize("failure", ["validation", "observer"])
def test_failed_run_releases_lifecycle_guard(failure):
    sim = _simulation()
    initial = _initial()
    if failure == "validation":
        with pytest.raises(ValueError, match="steps"):
            sim.run(initial, -1)
    else:
        def reject_output(times, values):
            if times[-1] > 0:
                raise RuntimeError("deliberate observer failure")
        with pytest.raises(RuntimeError, match="deliberate observer failure"):
            sim.run(initial, 6, observer=reject_output)
    params = {**sim.problem.params, "h0": jnp.array([[0.1, 0.6], [0.6, -0.1]])}
    sim.update_parameters(params)
    expected = Simulation(sim.problem, sim.integrator, sim.execution).run(initial, 6)
    _assert_state(sim.run(initial, 6).final_state, expected.final_state)
