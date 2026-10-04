"""Prescribed geometry and initial observations must refer to the same point."""

import jax.numpy as jnp
import numpy as np
import pytest

from pyeph import CPA, Execution, Integrator, Problem, Simulation, make_state, stack_states
from pyeph.core.problem import PrescribedPath
from pyeph.models.analytic import SpinBosonModel
from pyeph.observables.population import FunctionalMeasurement
from pyeph.paths.harmonic import ConstantPath, HarmonicBath, HarmonicPath


def simulation(treatment):
    model = SpinBosonModel()
    return Simulation(Problem(model, model.default_params(), treatment, CPA(),
                              FunctionalMeasurement(lambda problem, state: {"q": state.q})),
                      Integrator(.03), Execution(chunk_size=7))


def test_mismatched_initial_geometry_rejected_before_observations():
    run = simulation(PrescribedPath(ConstantPath([1.])))
    received = []
    with pytest.raises(ValueError, match="initial coordinates must agree"):
        run.run(make_state([0.], [0.], [1., 0.]), 0,
                observer=lambda time, values: received.append(values))
    assert received == [] and run._compiled == {}


def test_batch_checks_each_geometry_at_its_own_initial_time():
    path = HarmonicPath([.4], [.3], [.7])
    run = simulation(PrescribedPath(path))
    states = [make_state(path.position(time), [5.], [1., 0.], time=time, trajectory_id=i)
              for i, time in enumerate((.2, .7))]
    batch = stack_states(states)
    result = run.run(batch, 0)
    np.testing.assert_array_equal(result.final_state.q, batch.q)
    np.testing.assert_array_equal(result.final_state.p, batch.p)
    wrong = batch._replace(q=batch.q.at[1, 0].add(.1))
    with pytest.raises(ValueError, match="each trajectory's initial time"):
        run.run(wrong, 1)


def test_state_owned_harmonic_bath_keeps_independent_initial_geometries():
    run = simulation(HarmonicBath([.7]))
    initial = stack_states([make_state([q], [.1], [1., 0.], trajectory_id=i)
                            for i, q in enumerate((-.8, .3))])
    result = run.run(initial, 1)
    assert result.final_state.q[0, 0] != result.final_state.q[1, 0]


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_precision_tolerance_accepts_roundoff_without_overwriting_coordinates(dtype):
    run = simulation(PrescribedPath(ConstantPath([1.])))
    epsilon = np.finfo(dtype).eps
    initial = make_state([1.], [0.], [1., 0.])
    q = jnp.asarray([1.+4*epsilon], dtype=dtype)
    initial = initial._replace(q=q, p=initial.p.astype(dtype), time=initial.time.astype(dtype),
                               electronic=initial.electronic.astype(
                                   jnp.complex64 if dtype == np.float32 else jnp.complex128))
    result = run.run(initial, 0)
    np.testing.assert_array_equal(result.final_state.q, q)
    np.testing.assert_array_equal(result.observables["q"][0], q)
    with pytest.raises(ValueError, match="initial coordinates must agree"):
        run.run(initial._replace(q=jnp.asarray([1.01], dtype=dtype)), 0)


@pytest.mark.parametrize("bad_value", [np.array([np.nan]), np.array([1j]), np.array([0., 0.])])
def test_path_must_return_valid_geometry_at_the_requested_time(bad_value):
    class BadLaterPath:
        def position(self, time):
            return jnp.array([0.]) if float(time) == 0 else bad_value

        def velocity(self, time):
            return jnp.array([0.])

    run = simulation(PrescribedPath(BadLaterPath()))
    with pytest.raises(ValueError, match="finite real arrays"):
        run.run(make_state([0.], [0.], [1., 0.], time=1.), 0)


def test_large_absolute_time_path_can_resume_after_clock_roundoff():
    path = HarmonicPath([.4], [.3], [.7])
    run = simulation(PrescribedPath(path))
    initial = make_state(path.position(10000.), [0.], [1., 0.], time=10000.)
    first = run.run(initial, 19)
    second = run.run(first.final_state, 11)
    np.testing.assert_allclose(second.final_state.q, path.position(second.final_state.time),
                               atol=1e-11, rtol=1e-11)
