"""Selected spectral contractions and basis transport against independent algebra."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pyeph.core.contracts import ModelSpec
from pyeph.core.system import SystemSpec
from pyeph.models.base import AutoDiffModel
from pyeph.representations.adiabatic import (
    derivative_coupling,
    diagonalize,
    model_surfaces,
    surface_force,
    validate_surfaces,
)
from pyeph.representations.connection import analyze_overlap, transport_from_overlap
from pyeph.representations.gauge import align_phases, transform_operator, transform_state


class ComplexReference(AutoDiffModel):
    spec = ModelSpec(SystemSpec(2, (2,)), name="complex_reference", complex_valued=True)

    def apply(self, params, q, vectors):
        h = jnp.array([[q[0], 0.4 + 0.2j*q[1]], [0.4 - 0.2j*q[1], -q[0]]])
        return h @ vectors

    def reference_energy(self, params, q):
        return 0.5 * jnp.sum(q**2)


def test_selected_surface_force_and_complex_nac_match_explicit_derivatives():
    model, q = ComplexReference(), jnp.array([0.7, -0.3])
    data = model_surfaces(model, None, q)
    validate_surfaces(data)
    np.testing.assert_allclose(data.gaps[0, 1], data.energies[1] - data.energies[0])
    derivatives = np.array([[[1, 0], [0, -1]], [[0, 0.2j], [-0.2j, 0]]])
    u = np.asarray(data.vectors)
    expected_force = -q - np.einsum("i,kij,j->k", u[:, 0].conj(), derivatives, u[:, 0]).real
    expected_coupling = (np.einsum("i,kij,j->k", u[:, 0].conj(), derivatives, u[:, 1])
                         / float(data.gaps[0, 1]))
    force = jax.jit(lambda x: surface_force(model, None, x, model_surfaces(model, None, x), 0))(q)
    coupling = jax.jit(lambda x: derivative_coupling(
        model, None, x, model_surfaces(model, None, x), 0, 1))(q)
    assert bool(force.valid) and bool(coupling.valid)
    np.testing.assert_allclose(force.value, expected_force, atol=1e-13)
    np.testing.assert_allclose(coupling.value, expected_coupling, atol=1e-13)
    reverse = derivative_coupling(model, None, q, data, 1, 0)
    np.testing.assert_allclose(reverse.value, -coupling.value.conj(), atol=1e-13)

    eps = 1e-5
    def total_surface(position):
        h = np.asarray(model.apply(None, jnp.asarray(position), jnp.eye(2)))
        return 0.5 * np.sum(position**2) + np.linalg.eigvalsh(h)[0]
    finite_difference = np.array([
        -(total_surface(np.asarray(q) + eps*np.eye(2)[i])
          - total_surface(np.asarray(q) - eps*np.eye(2)[i])) / (2*eps)
        for i in range(2)
    ])
    np.testing.assert_allclose(force.value, finite_difference, atol=1e-10)


def test_surface_contractions_are_covariant_under_independent_complex_phases():
    model, q = ComplexReference(), jnp.array([0.7, -0.3])
    original = model_surfaces(model, None, q)
    phases = jnp.exp(1j*jnp.array([0.8, -0.4]))
    changed = original._replace(vectors=original.vectors * phases[None, :])
    np.testing.assert_allclose(surface_force(model, None, q, changed, 0).value,
                               surface_force(model, None, q, original, 0).value, atol=1e-13)
    np.testing.assert_allclose(
        derivative_coupling(model, None, q, changed, 0, 1).value,
        phases[0].conj()*phases[1]*derivative_coupling(model, None, q, original, 0, 1).value,
        atol=1e-13,
    )


def test_degeneracies_are_explicit_and_never_regularized_into_finite_couplings():
    model, q = ComplexReference(), jnp.array([0.0, 0.0])
    data = diagonalize(jnp.eye(2))
    assert bool(data.near_degenerate[0, 1])
    assert not bool(data.near_degenerate[0, 0])
    with pytest.raises(ValueError, match="degenerate"):
        validate_surfaces(data)
    force = surface_force(model, None, q, data, 0)
    coupling = derivative_coupling(model, None, q, data, 0, 1)
    assert not bool(force.valid) and np.isnan(force.value).all()
    assert not bool(coupling.valid) and np.isnan(coupling.value).all()
    close = diagonalize(jnp.diag(jnp.array([0.0, 1e-12])), gap_tolerance=1e-10)
    with pytest.raises(ValueError, match="degenerate"):
        validate_surfaces(close)
    isolated = diagonalize(jnp.diag(jnp.array([0.0, 0.0, 1.0])))
    validate_surfaces(isolated, states=[2])


def test_invalid_hamiltonians_and_surface_requests_are_rejected():
    with pytest.raises(ValueError, match="square"):
        diagonalize(jnp.ones((2, 3)))
    with pytest.raises(ValueError, match="nonfinite"):
        validate_surfaces(diagonalize(jnp.array([[0., 1.], [0., 1.]])))
    data = model_surfaces(ComplexReference(), None, jnp.array([0.7, 0.2]))
    with pytest.raises(ValueError, match="out of range"):
        validate_surfaces(data, states=[2])
    diagonal = derivative_coupling(ComplexReference(), None, jnp.array([0.7, 0.2]), data, 0, 0)
    assert not bool(diagonal.valid) and np.isnan(diagonal.value).all()
    invalid = jax.jit(lambda i: surface_force(
        ComplexReference(), None, jnp.array([0.7, 0.2]), data, i))(jnp.array(3))
    assert not bool(invalid.valid) and np.isnan(invalid.value).all()


class InvalidGradientProvider(ComplexReference):
    """Provider fault injection, independent of otherwise valid spectral data."""

    def __init__(self, failure):
        self.failure = failure

    def reference_gradient(self, params, q):
        if self.failure == "reference":
            return jnp.full_like(q, jnp.nan)
        return super().reference_gradient(params, q)

    def contract_gradient(self, params, q, weight):
        if self.failure == "electronic":
            return jnp.full_like(q, jnp.inf)
        if self.failure == "imaginary":
            # At q[1]=0 the eigenvectors are real; only the imaginary NAC
            # contraction uses a purely imaginary left vector.
            bad = jnp.all(jnp.abs(jnp.real(weight.left)) < 1e-14)
            value = super().contract_gradient(params, q, weight)
            return jnp.where(bad, jnp.nan, value)
        if self.failure == "ignore_weight":
            return jnp.zeros_like(q)
        return super().contract_gradient(params, q, weight)


@pytest.mark.parametrize("failure", ["reference", "electronic"])
def test_surface_force_rejects_nonfinite_provider_derivatives(failure):
    q = jnp.array([0.7, 0.])
    model = InvalidGradientProvider(failure)
    data = model_surfaces(model, None, q)
    validate_surfaces(data)
    result = jax.jit(lambda: surface_force(model, None, q, data, 0))()
    assert not bool(result.valid)
    assert np.isnan(result.value).all()


@pytest.mark.parametrize("failure", ["electronic", "imaginary"])
def test_selected_nac_rejects_nonfinite_real_or_imaginary_contractions(failure):
    q = jnp.array([0.7, 0.])
    model = InvalidGradientProvider(failure)
    data = model_surfaces(model, None, q)
    result = jax.jit(lambda: derivative_coupling(model, None, q, data, 0, 1))()
    assert not bool(result.valid)
    assert np.isnan(result.value).all()
    if failure == "imaginary":
        assert bool(surface_force(model, None, q, data, 0).valid)


def test_selected_vector_finiteness_is_checked_even_if_provider_ignores_weight():
    q = jnp.array([0.7, 0.])
    model = InvalidGradientProvider("ignore_weight")
    good = model_surfaces(model, None, q)
    bad = good._replace(vectors=good.vectors.at[:, 0].set(jnp.nan))
    force = surface_force(model, None, q, bad, 0)
    nac = derivative_coupling(model, None, q, bad, 0, 1)
    assert not bool(force.valid) and np.isnan(force.value).all()
    assert not bool(nac.valid) and np.isnan(nac.value).all()
    # The force on state1 uses only its finite vector and the valid energies.
    assert bool(surface_force(model, None, q, bad, 1).valid)


@pytest.mark.parametrize("gap", [jnp.inf, jnp.nan, 0.])
def test_selected_nac_rejects_invalid_gap_even_when_contracted_gradient_is_zero(gap):
    q = jnp.array([0.7, 0.])
    model = InvalidGradientProvider("ignore_weight")
    good = model_surfaces(model, None, q)
    bad = good._replace(gaps=good.gaps.at[0, 1].set(gap))
    result = jax.jit(lambda d: derivative_coupling(model, None, q, d, 0, 1))(bad)
    assert not bool(result.valid)
    assert np.isnan(result.value).all()


def test_host_spectral_validation_rejects_stale_gap_sign_and_unsorted_energies():
    data = diagonalize(jnp.diag(jnp.array([-0.1, 0.3])))
    with pytest.raises(ValueError, match="gaps must equal"):
        validate_surfaces(data._replace(gaps=-data.gaps))
    tiny = diagonalize(jnp.diag(jnp.array([-1e-15, 1e-15])), gap_tolerance=0.)
    with pytest.raises(ValueError, match="gaps must equal"):
        validate_surfaces(tiny._replace(gaps=-tiny.gaps))
    energies = data.energies[::-1]
    with pytest.raises(ValueError, match="sorted"):
        validate_surfaces(data._replace(energies=energies, vectors=data.vectors[:, ::-1],
                                       gaps=energies[None, :]-energies[:, None]))


def test_phase_alignment_handles_real_and_complex_columns_and_detects_lost_tracking():
    reference = jnp.eye(2, dtype=complex)
    candidate = reference * jnp.exp(1j*jnp.array([0.3, 2.9]))
    aligned = jax.jit(align_phases)(reference, candidate)
    assert np.all(aligned.valid)
    np.testing.assert_allclose(aligned.vectors, reference, atol=1e-14)
    real = align_phases(jnp.eye(2), jnp.diag(jnp.array([-1., 1.])))
    np.testing.assert_allclose(real.vectors, jnp.eye(2))
    lost = align_phases(reference, reference[:, ::-1])
    assert not np.any(lost.valid)
    np.testing.assert_allclose(lost.vectors, reference[:, ::-1])


def test_constant_basis_transform_preserves_complex_expectation():
    u = jnp.array([[1, 1j], [1j, 1]], dtype=complex) / jnp.sqrt(2.)
    h = jnp.array([[0.1, 0.2j], [-0.2j, -0.3]])
    c = jnp.array([1, 2j]) / jnp.sqrt(5.)
    rotated_h, rotated_c = transform_operator(h, u), transform_state(c, u)
    np.testing.assert_allclose(jnp.vdot(rotated_c, rotated_h @ rotated_c),
                               jnp.vdot(c, h @ c), atol=1e-14)


def test_raw_overlap_and_polar_transport_keep_subspace_loss_visible():
    overlap = jnp.diag(jnp.array([1., 0.8]))
    raw = jax.jit(transport_from_overlap)(overlap)
    polar = transport_from_overlap(overlap, mode="polar")
    np.testing.assert_allclose(raw.diagnostics.singular_values, [1., 0.8])
    np.testing.assert_allclose(raw.diagnostics.maximum_norm_loss, 0.36)
    assert not bool(raw.diagnostics.is_isometry)
    assert bool(raw.valid) and bool(polar.valid)
    assert not bool(raw.projection_applied) and bool(polar.projection_applied)
    c = jnp.array([0., 1.])
    np.testing.assert_allclose(jnp.vdot(raw.matrix @ c, raw.matrix @ c), 0.64)
    np.testing.assert_allclose(polar.matrix.conj().T @ polar.matrix, jnp.eye(2), atol=1e-14)
    np.testing.assert_allclose(polar.raw_overlap, overlap)
    np.testing.assert_allclose(polar.diagnostics.maximum_norm_loss, 0.36)


def test_overlap_orientation_is_new_coefficients_equal_overlap_adjoint_times_old():
    phases = jnp.exp(1j*jnp.array([0.3, -0.6]))
    old_basis = jnp.eye(2, dtype=complex)
    new_basis = old_basis * phases[None, :]
    overlap = old_basis.conj().T @ new_basis
    c0 = jnp.array([1, 2j]) / jnp.sqrt(5.)
    transport = transport_from_overlap(overlap)
    np.testing.assert_allclose(new_basis @ (transport.matrix @ c0), old_basis @ c0, atol=1e-14)
    assert bool(transport.diagnostics.is_isometry)


def test_singular_and_expanding_overlap_status():
    raw = transport_from_overlap(jnp.diag(jnp.array([1., 0.])))
    assert bool(raw.valid) and bool(raw.diagnostics.rank_deficient)
    polar = transport_from_overlap(jnp.diag(jnp.array([1., 0.])), mode="polar")
    assert not bool(polar.valid) and np.isnan(polar.matrix).all()
    expanding = transport_from_overlap(jnp.diag(jnp.array([1., 1.1])))
    assert not bool(expanding.valid) and np.isnan(expanding.matrix).all()
    with pytest.raises(ValueError, match="square"):
        analyze_overlap(jnp.ones((2, 3)))
