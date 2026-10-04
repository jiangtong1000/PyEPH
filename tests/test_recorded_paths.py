"""Recorded path interpolation, explicit domains and overlap endpoint conventions."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pyeph.paths.electronic import AdiabaticElectronicPath, FixedBasisElectronicPath
from pyeph.paths.nuclear import RecordedNuclearPath


def polynomial_path():
    times = np.array([2.0, 2.3, 4.1])
    coefficients = np.arange(1, 7).reshape(2, 3) / 7
    positions = np.array([coefficients*t**3 - 2*t for t in times])
    velocities = np.array([3*coefficients*t**2 - 2 for t in times])
    return RecordedNuclearPath(times, positions, velocities), coefficients


def test_cubic_hermite_recovers_positions_velocities_and_exact_endpoints():
    path, coefficients = polynomial_path()
    assert path.domain == (2.0, 4.1)
    for t in [2.0, 2.1, 2.3, 3.5, 4.1]:
        np.testing.assert_allclose(path.position(t), coefficients*t**3 - 2*t, atol=2e-14)
        np.testing.assert_allclose(path.velocity(t), 3*coefficients*t**2 - 2, atol=2e-13)
        np.testing.assert_allclose(jax.jacfwd(path.position)(t), path.velocity(t), atol=2e-13)
    np.testing.assert_allclose(path.position(path.end_time), path.positions[-1], atol=1e-14)
    np.testing.assert_allclose(path.velocity(path.end_time), path.velocities[-1], atol=1e-14)


def test_nuclear_path_is_jittable_and_does_not_extrapolate():
    path, coefficients = polynomial_path()
    times = jnp.array([2.0, 2.4, 3.0, 4.1])
    result = jax.jit(jax.vmap(path.position))(times)
    np.testing.assert_allclose(result, coefficients[None]*np.asarray(times)[:, None, None]**3
                               - 2*np.asarray(times)[:, None, None], atol=2e-14)
    path.validate_time(times)
    path.validate_span(2.0, 4.1)
    for t in [1.99, 4.11, np.nan]:
        with pytest.raises(ValueError):
            path.validate_time(t)
        assert not bool(path.contains(t))
        assert np.isnan(jax.jit(path.position)(t)).all()
        assert np.isnan(jax.jit(path.velocity)(t)).all()
    with pytest.raises(ValueError, match="stop"):
        path.validate_span(3.0, 2.0)


@pytest.mark.parametrize("times", [[0, 0], [1, 0], [0, np.inf], [0]])
def test_nuclear_path_rejects_invalid_time_grids(times):
    with pytest.raises(ValueError):
        RecordedNuclearPath(times, np.zeros((len(times), 1)), np.zeros((len(times), 1)))


def test_nuclear_path_rejects_shape_and_nonfinite_data():
    with pytest.raises(ValueError, match="equal"):
        RecordedNuclearPath([0, 1], [[0], [1]], [[0, 1], [1, 2]])
    with pytest.raises(ValueError, match="finite"):
        RecordedNuclearPath([0, 1], [[0], [np.nan]], [[0], [1]])


def electronic_matrix(t):
    return np.array([[t, 0.3 + 0.2j*t], [0.3 - 0.2j*t, -0.4*t]])


def test_fixed_basis_linear_interpolation_preserves_complex_hermiticity():
    times = np.array([2., 3., 5.])
    path = FixedBasisElectronicPath(times, np.array([electronic_matrix(t) for t in times]),
                                   basis_id="effective_sites")
    assert path.nstates == 2 and not path.force_support
    assert not hasattr(path, "contract_gradient")
    for t in [2., 2.25, 3., 4.9, 5.]:
        frame = jax.jit(path.sample_frame)(t)
        assert bool(frame.valid)
        np.testing.assert_allclose(frame.hamiltonian, electronic_matrix(t), atol=1e-14)
        np.testing.assert_allclose(frame.hamiltonian, frame.hamiltonian.conj().T, atol=1e-14)
        c = jnp.array([1., 1j])
        np.testing.assert_allclose(path.apply(t, c), electronic_matrix(t) @ c, atol=1e-14)
    assert not bool(path.sample_frame(5.1).valid)
    with pytest.raises(ValueError, match="not 'force'"):
        path.sample_frame(3., "force")


def test_fixed_basis_dataset_transforms_covariantly():
    times = np.array([0., 1.])
    h = np.array([electronic_matrix(t) for t in times])
    u = np.array([[1, 1j], [1j, 1]]) / np.sqrt(2)
    original = FixedBasisElectronicPath(times, h)
    rotated = FixedBasisElectronicPath(times, u.conj().T @ h @ u, basis_id="rotated")
    np.testing.assert_allclose(rotated.sample_frame(0.4).hamiltonian,
                               u.conj().T @ original.sample_frame(0.4).hamiltonian @ u,
                               atol=1e-14)


def test_frames_only_fixed_path_rejects_intermediate_values():
    path = FixedBasisElectronicPath([0., 1.], [electronic_matrix(0), electronic_matrix(1)],
                                   interpolation="frames_only")
    path.validate_sample_time([0., 1.])
    with pytest.raises(ValueError, match="recorded frame"):
        path.validate_sample_time(0.5)
    assert not bool(jax.jit(path.sample_frame)(0.5).valid)
    assert np.isnan(path.sample_frame(0.5).hamiltonian).all()


def test_fixed_basis_dataset_rejects_nonhermitian_data():
    with pytest.raises(ValueError, match="Hermitian"):
        FixedBasisElectronicPath([0, 1], [[[0, 1], [0, 0]], [[0, 1], [0, 0]]])


def test_adiabatic_frames_require_defined_times_and_adjacent_transport_endpoints():
    overlap = np.array([np.eye(2), np.diag([1., 0.8])])
    path = AdiabaticElectronicPath([2., 3., 5.], [[0, 1], [0.1, 0.9], [0.2, 0.8]], overlap)
    assert not path.force_support and path.domain == (2., 5.)
    path.validate_sample_time([2., 3., 5.])
    assert bool(jax.jit(path.sample_frame)(5.).valid)
    np.testing.assert_allclose(path.sample_frame(5.).energies, [0.2, 0.8])
    with pytest.raises(ValueError, match="intermediate"):
        path.validate_sample_time(4.)
    assert not bool(path.sample_frame(4.).valid)
    assert np.isnan(path.sample_frame(4.).energies).all()
    for endpoints in [(2., 5.), (3., 2.)]:
        with pytest.raises(ValueError, match="adjacent forward"):
            path.interval_transport(*endpoints)
    np.testing.assert_allclose(path.interval_transport(3., 5.).diagnostics.maximum_norm_loss, 0.36)
    np.testing.assert_allclose(jax.jit(path.transport_at)(1).matrix, np.diag([1., 0.8]))
    assert not bool(jax.jit(path.transport_at)(2).valid)


def test_adiabatic_overlap_polar_conversion_is_explicit_and_preserves_diagnostics():
    phase = np.exp(0.7j)
    overlaps = np.array([np.diag([phase, 0.8])])
    raw = AdiabaticElectronicPath([0, 1], [[0, 1], [0, 1]], overlaps)
    polar = AdiabaticElectronicPath([0, 1], [[0, 1], [0, 1]], overlaps,
                                   transport_mode="polar")
    r, p = raw.interval_transport(0, 1), polar.interval_transport(0, 1)
    np.testing.assert_allclose(r.matrix, np.diag([phase.conjugate(), 0.8]))
    np.testing.assert_allclose(p.matrix, np.diag([phase.conjugate(), 1.]))
    assert bool(p.projection_applied) and not bool(r.projection_applied)
    np.testing.assert_allclose(p.diagnostics.maximum_norm_loss, r.diagnostics.maximum_norm_loss)


def test_energies_alone_do_not_supply_transport_or_spatial_forces():
    path = AdiabaticElectronicPath([0, 1], [[0, 1], [0.1, 1.1]])
    with pytest.raises(ValueError, match="energies alone"):
        path.interval_transport(0, 1)
    with pytest.raises(ValueError, match="not 'coupling'"):
        path.sample_frame(0, "coupling")
    assert not path.force_support and not hasattr(path, "contract_gradient")
    with pytest.raises(ValueError, match="contractions"):
        AdiabaticElectronicPath([0, 1], [[0, 1], [0.1, 1.1]], [np.eye(2)*1.1])
