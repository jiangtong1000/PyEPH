"""Public independent numerical evidence for deterministic real three-state RM."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from benchmarks.mapping_reference import (
    MASS, RotatingThreeStateModel, diagnostics, direction, fixture, impulse,
    parameters, populations, projectors, propagate_native, reference,
    rotated_parameters, smooth_rhs, spectral_data, total_energy, unpack,
)
from pyeph.representations.adiabatic import model_surfaces, surface_force
from pyeph.representations.mapping_rm import rm_impulse_direction


def test_analytic_eigensystem_and_projector_derivatives_have_independent_checks():
    q = np.array([-.23, .31])
    params, model = parameters(), RotatingThreeStateModel()
    levels, u, _ = spectral_data(q, params)
    native = jax.tree.map(jnp.asarray, params)
    h = np.asarray(model.apply(native, jnp.asarray(q), jnp.eye(3)))
    np.testing.assert_allclose(h@u, u*levels, atol=3e-15)
    np.testing.assert_allclose(u.T@u, np.eye(3), atol=3e-15)
    project, derivative = projectors(q, params)
    np.testing.assert_allclose(project.sum(axis=0), np.eye(3), atol=3e-15)
    width = 1e-5
    for i in range(2):
        shift = np.eye(2)[i]*width
        finite = (projectors(q+shift, params)[0]-projectors(q-shift, params)[0])/(2*width)
        np.testing.assert_allclose(derivative[i], finite, atol=6e-11)


def test_analytic_forces_and_full_spectator_direction_match_native_contractions():
    params, model = parameters(), RotatingThreeStateModel()
    native = jax.tree.map(jnp.asarray, params)
    q, _, c = unpack(np.array(fixture("accepted")[1]["crossing_row"]))
    data = model_surfaces(model, native, jnp.asarray(q))
    for active in range(3):
        expected = -params["stiffness"]*q-params["slopes"][:, active]
        force, valid = surface_force(model, native, jnp.asarray(q), data, active)
        assert bool(valid)
        np.testing.assert_allclose(force, expected, atol=3e-15)
    actual = rm_impulse_direction(model, native, jnp.asarray(q), data, jnp.asarray(c), 0, 1)
    assert bool(actual.valid)
    expected = direction(q, c, 0, 1)
    np.testing.assert_allclose(actual.value, expected, atol=3e-15)
    u = spectral_data(q)[1]
    coefficients = u.T@c
    coefficients[2] = 0
    pair_only = direction(q, u@coefficients, 0, 1)
    assert np.linalg.norm(expected-pair_only) > .03
    # Partial coordinate differentiation holds fixed-basis c, not U(q)^T c.
    width = 1e-5
    finite = []
    for i in range(2):
        shift = np.eye(2)[i]*width
        plus, minus = populations(q+shift, c), populations(q-shift, c)
        finite.append(((plus[0]-plus[1])-(minus[0]-minus[1]))/(4*width))
    np.testing.assert_allclose(expected, finite, atol=6e-11)


@pytest.mark.parametrize("kind", ["accepted", "frustrated"])
def test_declared_impulse_conserves_energy_and_tangential_momentum_without_projection(kind):
    crossing = np.array(fixture(kind)[1]["crossing_row"])
    q, p, c = unpack(crossing)
    updated, active, event = impulse(crossing, 0, 1)
    new_q, new_p, new_c = unpack(updated)
    np.testing.assert_array_equal(new_q, q)
    np.testing.assert_array_equal(new_c, c)
    normal = np.asarray(event["normal"])
    tangent = np.eye(2)-np.outer(normal, normal)
    np.testing.assert_allclose(tangent@((new_p-p)/np.sqrt(MASS)), 0, atol=1e-15)
    assert event["accepted"] == (kind == "accepted")
    assert active == (1 if kind == "accepted" else 0)
    assert event["incoming_rate"] < -.2 and event["outgoing_rate"] > .2
    assert total_energy(crossing, 0) == pytest.approx(total_energy(updated, active), abs=1e-15)


@pytest.mark.parametrize("kind", ["accepted", "frustrated"])
def test_independent_event_time_norm_energy_and_reference_refinement(kind):
    times = np.linspace(0., .8, 41)
    result = reference(kind, times)
    assert len(result.events) == 1
    event = result.events[0]
    assert event["time"] == pytest.approx(.173, abs=2e-13)
    assert event["accepted"] == (kind == "accepted")
    check = diagnostics(result.rows, result.active)
    assert check["maximum_norm_defect"] < 4e-14
    assert check["maximum_energy_drift"] < 4e-14
    refined = reference(kind, times, max_step=.0025, rtol=3e-14, atol=3e-15)
    np.testing.assert_allclose(result.rows, refined.rows, atol=4e-13, rtol=0)
    assert refined.events[0]["time"] == pytest.approx(event["time"], abs=2e-13)


@pytest.mark.parametrize("kind", ["accepted", "frustrated"])
def test_reference_is_covariant_under_a_constant_real_electronic_basis_rotation(kind):
    params = rotated_parameters()
    transform = params["basis_rotation"]
    times = np.linspace(0, .8, 21)
    original, rotated = reference(kind, times), reference(kind, times, params)
    recovered = rotated.rows.copy()
    recovered[:, 4:7] = rotated.rows[:, 4:7]@transform
    recovered[:, 7:10] = rotated.rows[:, 7:10]@transform
    np.testing.assert_allclose(original.rows, recovered, atol=4e-14, rtol=0)
    np.testing.assert_array_equal(original.active, rotated.active)
    np.testing.assert_allclose(original.events[0]["direction"], rotated.events[0]["direction"],
                               atol=4e-14)


@pytest.fixture(scope="module")
def runtime_records():
    return {kind: [(dt, *propagate_native(kind, dt, .8)) for dt in (.08, .04, .02)]
            for kind in ("accepted", "frustrated")}


@pytest.mark.parametrize("kind", ["accepted", "frustrated"])
def test_complete_runtime_trajectory_has_second_order_convergence(runtime_records, kind):
    errors = []
    for _, times, rows, values in runtime_records[kind]:
        expected = reference(kind, times)
        record = diagnostics(rows, values["active"], expected=expected.rows)
        errors.append(max(record[name] for name in (
            "maximum_coordinate_error", "maximum_momentum_error", "maximum_electronic_component_error")))
        assert int(values["accepted"][-1]) == (1 if kind == "accepted" else 0)
        assert int(values["frustrated"][-1]) == (0 if kind == "accepted" else 1)
        assert record["maximum_norm_defect"] < 1e-13
        np.testing.assert_array_equal(values["active"], expected.active)
    assert 3 < errors[0]/errors[1] < 5
    assert 3 < errors[1]/errors[2] < 5
    assert errors[-1] < 5e-5
    assert record["maximum_energy_drift"] < 2e-6


@pytest.mark.parametrize("kind", ["accepted", "frustrated"])
def test_complete_runtime_is_covariant_including_impulse_outcome(runtime_records, kind):
    params = rotated_parameters()
    transform = params["basis_rotation"]
    _, times, rows, values = runtime_records[kind][-1]
    rotated_times, rotated, other_values = propagate_native(kind, .02, .8, params)
    recovered = rotated.copy()
    recovered[:, 4:7] = rotated[:, 4:7]@transform
    recovered[:, 7:10] = rotated[:, 7:10]@transform
    np.testing.assert_allclose(times, rotated_times, atol=0)
    np.testing.assert_allclose(rows, recovered, atol=3e-11, rtol=0)
    for name in ("active", "accepted", "frustrated"):
        np.testing.assert_array_equal(values[name], other_values[name])


def test_projector_margin_rate_follows_full_coupled_equations():
    row, _ = fixture("accepted")
    q, p, c = unpack(row)
    rate = 2*np.dot(direction(q, c, 0, 1), p/MASS)
    width = 1e-6
    shift = width*smooth_rhs(0, row, 0)
    qp, _, cp = unpack(row+shift)
    qm, _, cm = unpack(row-shift)
    plus, minus = populations(qp, cp), populations(qm, cm)
    finite = ((plus[0]-plus[1])-(minus[0]-minus[1]))/(2*width)
    assert rate == pytest.approx(finite, abs=2e-10)
