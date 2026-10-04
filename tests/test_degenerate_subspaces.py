"""Independent subspace geometry and a concrete obstruction to vector labels."""

import numpy as np
import pytest

from benchmarks.degenerate_subspaces import (
    I2, SZ, TIME_REVERSAL, ambiguity, coordinate_gauge, curvature, diagnostics,
    finite_speed, hamiltonian, hamiltonian_derivatives, holonomy, kato_generator,
    kato_reference, negative_frame, parameters, path, physical_observables,
    polar_transport, polar_unitary, projector_data,
)


@pytest.mark.parametrize("q", [[.55, 0, .25], [-.2, .3, -.1], [0., 0., 0.]])
def test_actual_time_reversal_symmetry_and_kramers_pairs(q):
    h = hamiltonian(q)
    np.testing.assert_allclose(TIME_REVERSAL@TIME_REVERSAL.conj(), -np.eye(4), atol=0)
    np.testing.assert_allclose(TIME_REVERSAL@h.conj()@TIME_REVERSAL.conj().T, h, atol=0)
    projector, _, energy = projector_data(q)
    np.testing.assert_allclose(h@h, energy**2*np.eye(4), atol=3e-16)
    np.testing.assert_allclose(np.linalg.eigvalsh(h), [-energy, -energy, energy, energy], atol=1e-15)
    frame = negative_frame(q)
    state = frame@np.array([np.sqrt(.7), 1j*np.sqrt(.3)])
    partner = TIME_REVERSAL@state.conj()
    assert abs(np.vdot(state, partner)) < 2e-16
    np.testing.assert_allclose(h@partner, -energy*partner, atol=3e-16)
    np.testing.assert_allclose(projector, frame@frame.conj().T, atol=3e-16)


@pytest.mark.parametrize("s", [.0, .237, .681])
def test_projector_derivative_and_kato_generator_are_independently_verified(s):
    q, tangent = path(s)
    projector, derivatives, _ = projector_data(q)
    np.testing.assert_allclose(projector@projector, projector, atol=3e-16)
    assert np.trace(projector) == pytest.approx(2)
    width = 1e-5
    for axis in range(3):
        shift = width*np.eye(3)[axis]
        finite = (projector_data(q+shift)[0]-projector_data(q-shift)[0])/(2*width)
        np.testing.assert_allclose(derivatives[axis], finite, atol=2e-10)
        np.testing.assert_allclose(projector@derivatives[axis]@projector, 0, atol=2e-16)
    derivative = np.einsum("i,ijk->jk", tangent, derivatives)
    generator = kato_generator(s)
    np.testing.assert_allclose(generator.conj().T, -generator, atol=3e-16)
    np.testing.assert_allclose(generator@projector-projector@generator, derivative, atol=8e-16)


@pytest.fixture(scope="module")
def records():
    result = {}
    for loop in ("xy", "yz"):
        series = []
        for steps in (32, 64, 128):
            times = np.linspace(0, 1, steps+1)
            expected, actual = kato_reference(times, loop), polar_transport(times, loop)
            series.append((times, expected, actual))
        result[loop] = series
    return result


@pytest.mark.parametrize("loop", ["xy", "yz"])
def test_polar_transport_converges_to_full_space_projector_ode(records, loop):
    errors = []
    for times, expected, actual in records[loop]:
        result = diagnostics(times, actual.operators, loop, expected=expected)
        errors.append(result["maximum_transport_operator_error"])
        assert result["maximum_partial_isometry_defect"] < 3e-14
        assert result["maximum_subspace_leakage_amplitude"] < 2e-15
        assert result["maximum_instantaneous_eigenvalue_residual"] < 2e-15
        assert actual.minimum_link_singular_value > .99
    assert 3.8 < errors[0]/errors[1] < 4.2
    assert 3.8 < errors[1]/errors[2] < 4.2
    assert errors[-1] < 1.3e-4
    refined = kato_reference(times, loop, rtol=3e-14, atol=3e-15, max_step=.0025)
    np.testing.assert_allclose(refined, expected, atol=2e-12, rtol=0)
    reference_check = diagnostics(times, expected, loop)
    assert reference_check["maximum_partial_isometry_defect"] < 2e-12
    assert reference_check["maximum_subspace_leakage_amplitude"] < 2e-12


@pytest.mark.parametrize("loop", ["xy", "yz"])
def test_coordinate_dependent_u2_gauge_preserves_transport_spin_and_holonomy_invariants(records, loop):
    times, _, original = records[loop][-1]
    rotated = polar_transport(times, loop, gauge=coordinate_gauge)
    np.testing.assert_allclose(rotated.operators, original.operators, atol=3e-14, rtol=0)
    np.testing.assert_allclose(physical_observables(rotated.operators),
                               physical_observables(original.operators), atol=3e-14, rtol=0)
    initial_gauge = coordinate_gauge(parameters()["base"])
    w = holonomy(original.operators[-1])
    transformed = holonomy(rotated.operators[-1], initial_gauge=initial_gauge)
    np.testing.assert_allclose(transformed, initial_gauge.conj().T@w@initial_gauge, atol=3e-14)
    np.testing.assert_allclose(np.sort(np.angle(np.linalg.eigvals(transformed))),
                               np.sort(np.angle(np.linalg.eigvals(w))), atol=3e-14)


def test_arbitrary_discontinuous_sample_gauges_and_different_endpoint_frames_cancel(records):
    times, _, original = records["xy"][0]
    rng = np.random.default_rng(1037)
    gauges = []
    for _ in times:
        matrix = rng.normal(size=(2, 2))+1j*rng.normal(size=(2, 2))
        gauges.append(np.linalg.qr(matrix)[0])
    assert np.max(abs(gauges[0]-gauges[-1])) > .5
    rotated = polar_transport(times, "xy", gauge=np.array(gauges))
    np.testing.assert_allclose(rotated.operators, original.operators, atol=2e-14, rtol=0)


def test_two_based_loop_holonomies_do_not_commute(records):
    first = holonomy(records["xy"][-1][1][-1])
    second = holonomy(records["yz"][-1][1][-1])
    commutator = first@second-second@first
    assert np.linalg.norm(commutator) > .6
    gauge = coordinate_gauge(parameters()["base"])
    changed_first, changed_second = gauge.conj().T@first@gauge, gauge.conj().T@second@gauge
    assert np.linalg.norm(changed_first@changed_second-changed_second@changed_first) == pytest.approx(
        np.linalg.norm(commutator), abs=3e-15)
    initial = np.array([np.sqrt(.7), 1j*np.sqrt(.3)])
    after_ab, after_ba = second@first@initial, first@second@initial
    rho_ab, rho_ba = np.outer(after_ab, after_ab.conj()), np.outer(after_ba, after_ba.conj())
    assert np.max(abs(rho_ab-rho_ba)) > .1


def test_individual_eigenvector_populations_are_ambiguous_at_fixed_physical_state():
    result = ambiguity()
    np.testing.assert_allclose(result["individual_populations"], [[1, 0], [.5, .5], [0, 1]], atol=3e-16)
    assert result["cluster_population"] == pytest.approx(1)
    assert result["curvature_eigenvalues"][0] < -.6
    assert result["curvature_eigenvalues"][1] > .6
    assert abs(result["curvature_trace"]) < 3e-16


def test_cluster_population_and_scalar_force_do_not_fix_internal_state_observables():
    q = parameters()["base"]
    frame = negative_frame(q)
    projector, _, energy = projector_data(q)
    omega = curvature(q)
    eigenvalues, internal_states = np.linalg.eigh(omega)
    projected_gradient = np.array([frame.conj().T@dh@frame for dh in hamiltonian_derivatives()])
    np.testing.assert_allclose(projected_gradient, -(q/energy)[:, None, None]*I2, atol=4e-16)
    for index in range(2):
        state = frame@internal_states[:, index]
        assert np.vdot(state, projector@state).real == pytest.approx(1)
        assert np.vdot(state, hamiltonian(q)@state).real == pytest.approx(-energy)
        assert np.vdot(internal_states[:, index], omega@internal_states[:, index]).real == pytest.approx(
            eigenvalues[index])
    assert eigenvalues[1]-eigenvalues[0] > 1.2
    spin_z = np.kron(I2, SZ)
    # Replacing a pure doublet state by P/2 erases its polarization.
    assert np.vdot(frame[:, 0], spin_z@frame[:, 0]).real == pytest.approx(1)
    assert abs(np.trace(.5*projector@spin_z)) < 3e-16


def test_finite_speed_schrodinger_is_distinct_from_adiabatic_geometric_transport():
    times = np.linspace(0, 1, 129)
    params = parameters()
    initial = negative_frame(params["base"])@np.array([np.sqrt(.7), 1j*np.sqrt(.3)])
    transported = np.einsum("tij,j->ti", kato_reference(times), initial)
    density = np.einsum("ti,tj->tij", transported, transported.conj())
    differences, leakage = [], []
    for duration in (4., 20., 100.):
        actual = finite_speed(times, duration)
        actual_density = np.einsum("ti,tj->tij", actual, actual.conj())
        differences.append(np.max(abs(actual_density-density)))
        leakage.append(max(np.linalg.norm((np.eye(4)-projector_data(path(s)[0])[0])@state)**2
                           for s, state in zip(times, actual)))
        np.testing.assert_allclose(np.sum(abs(actual)**2, axis=1), 1., atol=2e-12, rtol=0)
    assert leakage[0] > .1 and leakage[-1] < .002
    assert differences[0] > .25 and .005 < differences[-1] < .03
    assert all(a > b for a, b in zip(differences, differences[1:]))


def test_singular_overlaps_and_nonunitary_gauges_are_not_repaired_silently():
    with pytest.raises(ValueError, match="rank-deficient"):
        polar_unitary(np.diag([1., 0.]))
    with pytest.raises(ValueError, match="unitary"):
        polar_transport(np.linspace(0, 1, 5), gauge=lambda q: np.diag([1., 2.]))
    invalid = parameters() | {"delta": 0.}
    with pytest.raises(ValueError, match="separated"):
        projector_data(np.zeros(3), invalid)
    with pytest.raises(ValueError, match="positive delta"):
        negative_frame(np.zeros(3), invalid)


def test_native_individual_surface_validation_still_rejects_doublets():
    from pyeph.representations.adiabatic import diagonalize, validate_surfaces

    data = diagonalize(hamiltonian(parameters()["base"]))
    with pytest.raises(ValueError, match="degenerate"):
        validate_surfaces(data)
