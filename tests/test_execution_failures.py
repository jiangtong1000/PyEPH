"""External evaluation failures retain a published, resumable chunk boundary."""

from dataclasses import dataclass, replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pyeph import (CPA, Execution, Integrator, PrescribedPath, Problem, Simulation,
                   make_state)
from pyeph.execution.runner import SimulationError
from pyeph.core.state import stack_states
from pyeph.models.analytic import SpinBosonModel
from pyeph.observables.population import FunctionalMeasurement
from pyeph.paths.nuclear import RecordedNuclearPath


@dataclass(frozen=True)
class DomainLimitedModel(SpinBosonModel):
    limit: float = .13

    def apply(self, params, q, vectors):
        def check_domain(coordinates):
            if float(coordinates[0]) > self.limit:
                raise RuntimeError("external provider left its supported geometry domain")
            return np.array(1., dtype=coordinates.dtype)

        factor = jax.pure_callback(check_domain, jax.ShapeDtypeStruct((), q.dtype), q,
                                   vmap_method="sequential")
        return factor * super().apply(params, q, vectors)


def _simulation(model=None, *, jit=True, check_finite=True, measurement=None,
                allow_host_callbacks=False):
    model = model or DomainLimitedModel()
    path = RecordedNuclearPath([0., 1.], [[0.], [1.]], [[1.], [1.]])
    problem = Problem(model, model.default_params(), PrescribedPath(path), CPA(), measurement)
    return Simulation(problem, Integrator(.05),
                      Execution(chunk_size=2, jit=jit, check_finite=check_finite,
                                allow_host_callbacks=allow_host_callbacks))


@dataclass(frozen=True)
class ExternalDomainLimitedModel(DomainLimitedModel):
    execution_mode = "host_callback"

    def __post_init__(self):
        super().__post_init__()
        object.__setattr__(self, "spec", replace(self.spec, native_jax=False))


@pytest.mark.parametrize("jit", [False, True])
def test_allowed_host_callback_initial_measurement_reuses_updated_parameters(jit):
    model = ExternalDomainLimitedModel()
    measurement = FunctionalMeasurement(
        lambda problem, state: {"energy": jnp.real(jnp.vdot(
            state.electronic, problem.model.apply(problem.params, state.q, state.electronic)))})
    with pytest.raises(ValueError, match="allow_host_callbacks"):
        _simulation(model, measurement=measurement, jit=jit)
    simulation = _simulation(model, measurement=measurement, jit=jit, allow_host_callbacks=True)
    initial = stack_states([make_state([0.], [0.], np.array([1., sign])/np.sqrt(2.),
                                       trajectory_id=i) for i, sign in enumerate((1., -1.))])
    old = simulation.run(initial, 0)
    np.testing.assert_allclose(old.observables["energy"], [[.1, -.1]], atol=1e-15)
    cached = dict(simulation._compiled)
    simulation.update_parameters({**simulation.problem.params, "delta": .4})
    changed = simulation.run(initial, 0)
    assert simulation._compiled == cached
    np.testing.assert_allclose(changed.observables["energy"], [[.4, -.4]], atol=1e-15)


@pytest.mark.parametrize("jit", [False, True])
@pytest.mark.parametrize("collect,check_finite", [(True, True), (False, False)])
def test_provider_failure_keeps_chunk_boundary_and_runner_can_be_reused(tmp_path, jit,
                                                                       collect, check_finite):
    simulation = _simulation(jit=jit, check_finite=check_finite)
    initial = make_state([0.], [0.], [1., 0.], trajectory_id=19,
                         method_state={"counter": np.array(7, dtype=np.int64)})
    published = []
    observer = (lambda times, values: published.extend(np.asarray(times).tolist())) if collect else None
    with pytest.raises(SimulationError, match="execution") as caught:
        simulation.run(initial, 8, observer=observer, collect=collect)
    error = caught.value
    assert error.__cause__ is not None
    assert error.failed_state is None
    accepted = error.last_valid_state
    assert int(accepted.step) == 2
    np.testing.assert_allclose(accepted.time, .1, atol=1e-15)
    np.testing.assert_allclose(accepted.q, [.1], atol=1e-15)
    assert int(accepted.trajectory_id) == 19
    np.testing.assert_array_equal(accepted.key, initial.key)
    assert int(accepted.method_state["counter"]) == 7
    np.testing.assert_allclose(published, [0., .05, .1] if collect else [])

    # The public strict checkpoint can persist the retained state. Continuing
    # with an independently reconstructed, valid provider matches a full run.
    checkpoint = tmp_path/"last-published.h5"
    simulation.save_checkpoint(checkpoint, accepted, artifact_ids={"model": "domain-fixture-v1"})
    restored = simulation.load_checkpoint(checkpoint, artifact_ids={"model": "domain-fixture-v1"})
    repaired = _simulation(SpinBosonModel(), jit=jit, check_finite=check_finite)
    resumed = repaired.run(restored, 6).final_state
    uninterrupted = repaired.run(initial, 8).final_state
    for actual, expected in zip(jax.tree.leaves(resumed), jax.tree.leaves(uninterrupted), strict=True):
        np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-14)
    # A failed call releases the lifecycle guard; a valid shorter run still works.
    np.testing.assert_allclose(simulation.run(initial, 2).final_state.q, [.1], atol=1e-15)


def test_initial_evaluation_failure_retains_initial_state_before_publication():
    model = DomainLimitedModel(limit=-.1)
    measurement = FunctionalMeasurement(
        lambda problem, state: {"energy": jnp.real(jnp.vdot(
            state.electronic, problem.model.apply(problem.params, state.q, state.electronic)))})
    simulation = _simulation(model, measurement=measurement)
    initial = make_state([0.], [0.], [1., 0.])
    published = []
    with pytest.raises(SimulationError, match="initial observation") as caught:
        simulation.run(initial, 1, observer=lambda *args: published.append(args))
    assert caught.value.last_valid_state is initial
    assert caught.value.failed_state is None and caught.value.__cause__ is not None
    assert published == []


def test_host_observer_failure_keeps_its_original_exception():
    simulation = _simulation(SpinBosonModel())
    failure = OSError("output device unavailable")

    def observer(times, values):
        if times[-1] > 0:
            raise failure

    with pytest.raises(OSError) as caught:
        simulation.run(make_state([0.], [0.], [1., 0.]), 4, observer=observer)
    assert caught.value is failure


def test_preflight_failure_is_not_wrapped_as_execution_failure():
    simulation = _simulation()
    with pytest.raises(ValueError, match="prescribed path"):
        simulation.run(make_state([.2], [0.], [1., 0.]), 1)
