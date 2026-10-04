"""Independent fixed-Hilbert-space checks of the moving-frame research proof."""

import numpy as np
import pytest
from scipy.linalg import eigvalsh

from benchmarks.moving_frame_feedback import (
    MASS, covariant_gradient, diagnostics, electronic_derivative, frame, initial,
    metric_counterexample, physical_force, physical_quantities, propagate_moving,
    propagate_native, reference, transform,
)


@pytest.mark.parametrize("kind", ["moving", "regauged"])
def test_generalized_energies_norm_and_connection_recover_fixed_frame(kind):
    q, p, psi = initial()
    h, dh, _, gradient = physical_quantities(q)
    b, db = frame(q, kind)
    data = transform(h, dh, b, db)
    c = np.linalg.solve(b, psi)
    np.testing.assert_allclose(eigvalsh(data.hamiltonian, data.metric), np.linalg.eigvalsh(h),
                               atol=3e-15)
    assert np.vdot(c, data.metric@c).real == pytest.approx(1., abs=3e-15)
    np.testing.assert_allclose(covariant_gradient(data),
                               [b.conj().T @ derivative @ b for derivative in dh], atol=3e-15)
    dc = electronic_derivative(data, c, p/MASS)
    dpsi = b@dc + np.einsum("i,ijk,k->j", p/MASS, db, c)
    np.testing.assert_allclose(dpsi, -1j*h@psi, atol=3e-15)
    np.testing.assert_allclose(physical_force(data, c, gradient),
                               -gradient-np.einsum("a,iab,b->i", psi.conj(), dh, psi).real,
                               atol=3e-15)


@pytest.mark.parametrize("kind", ["moving", "regauged"])
def test_force_finite_difference_holds_physical_state_fixed_not_raw_coefficients(kind):
    q, _, psi = initial()
    h, dh, _, gradient = physical_quantities(q)
    b, db = frame(q, kind)
    data = transform(h, dh, b, db)
    c = np.linalg.solve(b, psi)
    width = 2e-5
    finite, raw = [], []
    for axis in range(2):
        actual_energy, raw_energy = [], []
        for sign in (-1, 1):
            displaced = q.copy()
            displaced[axis] += sign*width
            shifted_h, _, neutral, _ = physical_quantities(displaced)
            shifted_b, _ = frame(displaced, kind)
            shifted_c = np.linalg.solve(shifted_b, psi)
            shifted_matrix = shifted_b.conj().T @ shifted_h @ shifted_b
            actual_energy.append(neutral+np.vdot(shifted_c, shifted_matrix@shifted_c).real)
            raw_energy.append(neutral+np.vdot(c, shifted_matrix@c).real)
        finite.append(-(actual_energy[1]-actual_energy[0])/(2*width))
        raw.append(-(raw_energy[1]-raw_energy[0])/(2*width))
    force = physical_force(data, c, gradient)
    np.testing.assert_allclose(force, finite, atol=3e-11)
    assert np.max(abs(force-raw)) > .01


def test_H_S_and_metric_derivatives_do_not_determine_physical_force():
    result = metric_counterexample()
    for key in ("raw_h_error", "raw_metric_error", "raw_h_derivative", "metric_derivative"):
        assert result[key] < 3e-15
    assert result["connection_norm"] > .9
    assert result["stationary_embedding_force"] == 0
    assert result["rotating_embedding_force"] == pytest.approx(-.42, abs=3e-15)


def test_metric_only_connection_preserves_instantaneous_norm_but_changes_physical_motion():
    q, p, psi = initial()
    b, db = frame(q)
    h, dh, _, _ = physical_quantities(q)
    data = transform(h, dh, b, db)
    c = np.linalg.solve(b, psi)
    dc = electronic_derivative(data, c, p/MASS, connection="metric_only")
    ds_dt = np.einsum("i,ijk->jk", p/MASS,
                       data.connection+data.connection.conj().transpose(0, 2, 1))
    norm_derivative = 2*np.vdot(c, data.metric@dc).real+np.vdot(c, ds_dt@c).real
    assert abs(norm_derivative) < 3e-15
    physical_derivative = b@dc+np.einsum("i,ijk,k->j", p/MASS, db, c)
    assert np.max(abs(physical_derivative+1j*h@psi)) > .01


def test_complete_frame_connection_is_flat_with_noncommuting_coordinate_generators():
    q = np.array([.3, -.5])
    def gamma(position):
        b, db = frame(position, "regauged")
        return np.array([np.linalg.solve(b, derivative) for derivative in db])
    width = 1e-5
    derivative_x_y = (gamma(q+[width, 0])[1]-gamma(q-[width, 0])[1])/(2*width)
    derivative_y_x = (gamma(q+[0, width])[0]-gamma(q-[0, width])[0])/(2*width)
    gx, gy = gamma(q)
    commutator = gx@gy-gy@gx
    assert np.linalg.norm(commutator) > .01
    np.testing.assert_allclose(derivative_x_y-derivative_y_x+commutator, 0, atol=2e-11)


@pytest.mark.parametrize("kind", ["moving", "regauged"])
def test_complete_coupled_moving_trajectory_converges_to_independent_reference(kind):
    errors = []
    for dt in (.08, .04, .02):
        times, rows = propagate_moving(dt, 1.6, kind=kind)
        record = diagnostics(rows, kind, reference(times))
        errors.append(max(record["maximum_coordinate_error"], record["maximum_momentum_error"],
                          record["maximum_electronic_component_error"]))
    assert 12 < errors[0]/errors[1] < 20
    assert 12 < errors[1]/errors[2] < 20
    assert record["maximum_metric_norm_defect"] < 2e-9
    assert record["maximum_total_energy_drift"] < 2e-9


def test_native_fixed_frame_ehrenfest_converges_to_same_reference():
    expected = reference(np.array([0., 1.6]))[-1]
    errors = [np.max(abs(propagate_native(dt, 1.6)-expected)) for dt in (.04, .02)]
    assert 3.8 < errors[0]/errors[1] < 4.2


def test_degenerate_spectrum_requires_no_arbitrary_eigenvectors_for_ehrenfest_force():
    h = np.diag([0., 0., 1.])
    dh = np.array([np.diag([.3, -.3, .1]), [[0, .2j, 0], [-.2j, 0, .1], [0, .1, 0]]])
    q, _, psi = initial()
    physical = -np.einsum("a,iab,b->i", psi.conj(), dh, psi).real
    for kind in ("moving", "regauged"):
        b, db = frame(q, kind)
        data = transform(h, dh, b, db)
        c = np.linalg.solve(b, psi)
        np.testing.assert_allclose(physical_force(data, c, np.zeros(2)), physical, atol=3e-15)


def test_incomplete_or_singular_frame_is_not_silently_orthogonalized():
    with pytest.raises(ValueError, match="complete square"):
        transform(np.eye(3), np.zeros((2, 3, 3)), np.ones((3, 2)), np.zeros((2, 3, 2)))
    with pytest.raises(ValueError, match="singular"):
        transform(np.eye(3), np.zeros((2, 3, 3)), np.diag([1., 1., 1e-12]),
                  np.zeros((2, 3, 3)))
