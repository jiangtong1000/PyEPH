"""Density-factor layout, validation, and pure measurement regression tests."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pyeph.core.contracts import ProbeContext
from pyeph.core.problem import CoupledClassical, Problem
from pyeph.dynamics.ehrenfest import Ehrenfest
from pyeph.models.epc import LinearEPCModel
from pyeph.paths.harmonic import HarmonicBath
from pyeph.workflows.column_transport import (
    ColumnTransportMeasurement,
    initialize_column_transport_state,
    initialize_infinite_temperature_columns,
    make_column_transport_problem,
)


def probe(params, context, name, vectors):
    del params
    x = jnp.array([[0., 1.+.3j, 0.], [1.-.3j, .2, -.1j], [0., .1j, -.4]])
    y = jnp.array([[.3, .2j, .1], [-.2j, -.5, .2], [.1, .2, .1]])
    matrix = x if name == "x" else y
    return (1+.2*context.q[0]) * (matrix @ vectors)


def physical_problem(callback=probe):
    model = LinearEPCModel(3, 1)
    params = model.create_params(h0=np.diag([-.2, .3, .7]),
                                  coupling=np.array([[[.1, .02, 0], [.02, -.1, 0], [0, 0, .03]]]),
                                  omega=[.8])
    return make_column_transport_problem(model, params, HarmonicBath([.8]),
                                         probes=("x", "y"), probe_callback=callback)


def density_factor():
    values = np.array([[.2+.1j, .4], [.3, -.1j], [.1-.2j, .2+.3j]])
    return values / np.linalg.norm(values)


def test_exact_factor_layout_and_all_current_origin_axes():
    problem = physical_problem()
    factor = density_factor()
    state = initialize_column_transport_state(problem, [.2], [.1], factor, time=.7, trajectory_id=17)
    context = ProbeContext(state.q)
    currents = [np.asarray(probe(problem.params, context, name, jnp.eye(3, dtype=complex))) for name in ("x", "y")]
    expected_columns = np.concatenate((factor, currents[0] @ factor, currents[1] @ factor), axis=1)
    np.testing.assert_allclose(state.electronic, expected_columns, atol=2e-16, rtol=0)
    measured = jax.jit(lambda s: problem.measurement.evaluate(problem, s))(state)
    rho = factor @ factor.conj().T
    expected = np.array([[np.trace(a @ b @ rho) for b in currents] for a in currents])
    np.testing.assert_allclose(measured["current_correlation"], expected, atol=2e-16, rtol=1e-15)
    assert measured["lag_time"] == 0
    np.testing.assert_allclose(measured["column_norm_squared_drift"], 0., atol=2e-16, rtol=0)
    assert state.method_state["column_transport"]["rank"] == 2
    assert state.method_state["column_transport"]["time0"] == .7


def test_factor_and_origin_inputs_are_owned_and_zero_columns_are_valid():
    problem = physical_problem()
    factor = np.zeros((3, 2), complex)
    factor[0, 0] = 1
    q = np.array([.2])
    p = np.array([.1])
    state = initialize_column_transport_state(problem, q, p, factor)
    expected = np.asarray(state.electronic).copy()
    factor[:] = 8
    q[:] = 9
    p[:] = 10
    np.testing.assert_array_equal(state.electronic, expected)
    np.testing.assert_array_equal(state.method_state["column_transport"]["q0"], [.2])
    np.testing.assert_array_equal(state.method_state["column_transport"]["p0"], [.1])
    result = problem.measurement.evaluate(problem, state)
    assert np.isfinite(result["current_correlation"]).all()
    np.testing.assert_array_equal(result["column_norm_squared_drift"][1::2], 0.)


@pytest.mark.parametrize("factor", [np.ones(3), np.ones((2, 1)), np.ones((3, 0)),
                                    np.zeros((3, 1)), np.ones((3, 2)),
                                    np.full((3, 1), np.nan), np.ones((3, 1), dtype=bool)])
def test_invalid_density_factor_is_never_renormalized(factor):
    with pytest.raises(ValueError, match="factor|trace"):
        initialize_column_transport_state(physical_problem(), [.2], [.1], factor)


@pytest.mark.parametrize("probes", ["x", (), ("",), ("x", "x"), (1,)])
def test_probe_names_explicit_unique_and_nonempty(probes):
    with pytest.raises(ValueError, match="probes"):
        ColumnTransportMeasurement(probes)


def test_callback_shape_and_initial_nonfinite_rejection():
    def wrong_shape(params, context, name, vectors):
        return vectors[:, 0]

    def nonfinite(params, context, name, vectors):
        return vectors * jnp.nan

    for callback in (wrong_shape, nonfinite):
        with pytest.raises(ValueError, match="shape|nonfinite"):
            initialize_column_transport_state(physical_problem(callback), [.2], [.1], density_factor())


def test_only_prescribed_cpa_is_accepted():
    problem = physical_problem()
    with pytest.raises(ValueError, match="CPA"):
        ColumnTransportMeasurement(("x",), probe).validate(
            Problem(problem.model, problem.params, CoupledClassical(1.), Ehrenfest()))
    with pytest.raises(ValueError, match="physical probes"):
        make_column_transport_problem(problem.model, problem.params, HarmonicBath([.8]))


@pytest.mark.parametrize("field,value", [
    ("schema", jnp.int32(2)), ("rank", jnp.int32(3)), ("nprobes", jnp.int32(1)),
    ("rank", jnp.int64(2)), ("preparation_code", jnp.int32(7)),
    ("trace_ids", jnp.array([4, 4], dtype=jnp.uint32)),
    ("trajectory_id0", jnp.uint32(99)), ("configuration_digest", jnp.zeros(32, dtype=jnp.uint8)),
    ("factor_digest", jnp.zeros(32, dtype=jnp.int32)), ("time0", jnp.array(4.)),
    ("norms_squared0", jnp.full(6, -1.)),
])
def test_origin_payload_validation(field, value):
    problem = physical_problem()
    state = initialize_column_transport_state(problem, [.2], [.1], density_factor())
    payload = {**state.method_state["column_transport"], field: value}
    with pytest.raises(ValueError):
        problem.measurement.validate_initial_state(problem, state._replace(method_state={"column_transport": payload}))


def test_observation_validation_rejects_nonfinite_and_axis_mismatch():
    measurement = ColumnTransportMeasurement(("x", "y"), probe)
    with pytest.raises(ValueError, match="nonfinite"):
        measurement.validate_observations({"current_correlation": np.full((2, 2), np.nan)})
    with pytest.raises(ValueError, match="axes"):
        measurement.validate_observations({"current_correlation": np.zeros((2, 1))})


@pytest.mark.parametrize("ids", [[], [1, 1], [-1, 2], [2**32], [1.0], [True]])
def test_invalid_trace_ids(ids):
    with pytest.raises(ValueError, match="trace_ids"):
        initialize_infinite_temperature_columns(physical_problem(), [.2], [.1], trace_ids=ids)


def test_phase_density_trace_is_one_and_preparation_is_explicit():
    problem = physical_problem()
    state = initialize_infinite_temperature_columns(problem, [.2], [.1], trace_ids=[7, 3, 99], seed=21, trajectory_id=12)
    factor = np.asarray(state.electronic[:, :3])
    np.testing.assert_allclose(abs(factor), np.full((3, 3), 1/3), atol=1e-16, rtol=0)
    assert np.isclose(np.vdot(factor, factor), 1.)
    payload = state.method_state["column_transport"]
    assert payload["preparation_code"] == 1
    assert payload["trace_seed"] == 21
    np.testing.assert_array_equal(payload["trace_ids"], [7, 3, 99])
    problem.measurement.validate_initial_state(problem, state)


def test_changed_numerical_parameters_invalidate_original_origin():
    problem = physical_problem()
    state = initialize_column_transport_state(problem, [.2], [.1], density_factor())
    changed = replace(problem, params={**problem.params, "h0": problem.params["h0"]+.1*jnp.eye(3)})
    with pytest.raises(ValueError, match="configuration changed"):
        problem.measurement.validate_initial_state(changed, state)


@pytest.mark.parametrize("seed", [-1, 2**32, 1.5, True])
def test_exact_and_stochastic_initializers_reject_invalid_seed(seed):
    problem = physical_problem()
    for initializer, kwargs in ((initialize_column_transport_state, {"factor": density_factor()}),
                                (initialize_infinite_temperature_columns, {"trace_ids": [0]})):
        with pytest.raises(ValueError, match="seed"):
            initializer(problem, [.2], [.1], seed=seed, **kwargs)


def test_fixed_threefry_trace_stream_and_two_word_state_key_under_rbg_default():
    from pyeph import Integrator, Simulation

    with jax.default_prng_impl("rbg"):
        problem = physical_problem()
        full = initialize_infinite_temperature_columns(problem, [.2], [.1], trace_ids=[2, 7, 20], seed=93, trajectory_id=5)
        single = initialize_infinite_temperature_columns(problem, [.2], [.1], trace_ids=[7], seed=93, trajectory_id=5)
        np.testing.assert_allclose(full.electronic[:, 1]*np.sqrt(3), single.electronic[:, 0], atol=3e-16, rtol=0)
        assert full.key.shape == (2,) and full.key.dtype == np.uint32
        result = Simulation(problem, Integrator(.01, "rk4")).run(full, 1)
        assert np.isfinite(result.observables["current_correlation"]).all()
        np.testing.assert_array_equal(result.final_state.key, full.key)
        deterministic = initialize_column_transport_state(problem, [.2], [.1], density_factor(), seed=93, trajectory_id=5)
        assert deterministic.key.shape == (2,)
        Simulation(problem, Integrator(.01, "rk4")).run(deterministic, 0, collect=False)
    # The explicit stream is partition-stable; a process configuration change
    # remains an intentional origin/checkpoint provenance mismatch.
    with pytest.raises(ValueError, match="configuration changed"):
        problem.measurement.validate_initial_state(problem, full)
