"""Coordinate-domain transaction and numerical-failure regression checks."""
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pyeph import CPA, CoupledClassical, Ehrenfest, Execution, Integrator, LanczosOptions
from pyeph import Problem, Simulation, make_state
from pyeph.core.geometry import CoordinateBox
from pyeph.dynamics.checked import PHASE_CPA, PHASE_MEASUREMENT
from pyeph.execution.runner import SimulationError
from pyeph.models.analytic import SpinBosonModel
from pyeph.observables.population import ElectronicPopulation


def setup():
    model = SpinBosonModel()
    problem = Problem(model, model.default_params(), CoupledClassical(1.), Ehrenfest(),
                      geometry_guard=CoordinateBox([-2.], [2.]))
    state = make_state([.2], [.1], [1., 0.], trajectory_id=19,
                       method_state={"tag": np.array([13, 17])})
    return problem, state, Integrator(.02, LanczosOptions(), 2)


def identical(a, b):
    for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b), strict=True):
        assert np.asarray(x).tobytes() == np.asarray(y).tobytes()


def test_guarded_late_measurement_fault_discards_complete_chunk(monkeypatch):
    # Instrument an exact built-in solely to inject a late numerical failure.
    # No custom-measurement eligibility is added to the runtime contract.
    original = ElectronicPopulation.evaluate

    def faulty(self, problem, state):
        values = original(self, problem, state)
        return {**values, "fault": jnp.where(state.step >= 2, jnp.nan, 0.)}

    monkeypatch.setattr(ElectronicPopulation, "evaluate", faulty)
    problem, state, integrator = setup()
    published = []
    simulation = Simulation(problem, integrator, Execution(chunk_size=4, check_finite=False))
    with pytest.raises(SimulationError) as caught:
        simulation.run(state, 4, observer=lambda *args: published.append(args))
    assert len(published) == 1
    identical(caught.value.last_valid_state, state)
    assert int(caught.value.failed_state.step) == 1
    info = caught.value.diagnostics["step_info"]
    assert int(info.code) == 2 and int(info.phase) == PHASE_MEASUREMENT
    assert int(caught.value.diagnostics["failed_macro_index"]) == 1
    assert float(info.attempted_time) == pytest.approx(.04)


def test_nonfinite_cpa_path_preserves_first_attempt_geometry():
    class BadPath:
        prescribed = True

        def point(self, state, elapsed):
            return jnp.array([jnp.nan]), jnp.zeros_like(state.p)

    problem, state, integrator = setup()
    problem = replace(problem, method=CPA(), nuclear_treatment=BadPath())
    result, info = jax.jit(problem.method.build_checked_step(problem, integrator))(state)
    identical(result, state)
    assert int(info.code) == 2 and int(info.phase) == PHASE_CPA
    assert np.isnan(np.asarray(info.attempted_q)).all()
    assert float(info.attempted_time) == pytest.approx(.005)


def test_guarded_kernel_normal_trace_has_no_callback_and_preserves_scalar_result():
    problem, state, integrator = setup()
    guarded = problem.method.build_checked_step(problem, integrator)
    ordinary = replace(problem, geometry_guard=None).method.build_checked_step(
        replace(problem, geometry_guard=None), integrator)
    final, info = jax.jit(guarded)(state)
    reference, _ = jax.jit(ordinary)(state)
    for actual, expected in zip(jax.tree.leaves(final), jax.tree.leaves(reference), strict=True):
        actual, expected = np.asarray(actual), np.asarray(expected)
        assert actual.shape == expected.shape and actual.dtype == expected.dtype
        if actual.dtype.kind in "fc":
            assert np.isfinite(actual).all() and np.isfinite(expected).all()
            np.testing.assert_allclose(actual, expected, atol=3e-13, rtol=3e-13)
        else:
            assert actual.tobytes() == expected.tobytes()
    assert int(info.code) == 0
    traced = str(jax.make_jaxpr(guarded)(state))
    assert not any(name in traced for name in ("debug_callback", "pure_callback", "io_callback"))


def test_guard_does_not_silently_enter_ordinary_or_differentiable_step():
    problem, _, _ = setup()
    with pytest.raises(ValueError, match="checked Lanczos"):
        problem.method.build_step(problem, Integrator(.02))
    with pytest.raises(ValueError, match="checked Lanczos"):
        Simulation(problem, Integrator(.02))


def test_nested_bounds_remain_immutable_and_reconstruct_shape():
    source = np.array([[-1., -2.], [-3., -4.]])
    guard = CoordinateBox(source, -source)
    source[:] = 100
    rebuilt = replace(guard)
    assert rebuilt == guard and guard.shape == (2, 2)
    np.testing.assert_array_equal(guard.lower, [[-1., -2.], [-3., -4.]])


@pytest.mark.parametrize("lower,upper", [(0., 1.), ([1.], [0.]), ([0.], [np.inf])])
def test_invalid_domain_bounds_rejected(lower, upper):
    with pytest.raises(ValueError):
        CoordinateBox(lower, upper)
