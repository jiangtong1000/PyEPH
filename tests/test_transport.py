"""Independent thermal-state and current-trace checks for JAX transport."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.linalg import expm

from pyeph.observables.transport.greenkubo import (
    current_correlation,
    legacy_current_correlation,
    legacy_current_to_physical,
    thermal_density_matrix,
)


def _hermitian(rng, n):
    a = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    return (a + a.conj().T) / 2


def _trace_reference(u, rho, jt, j0):
    # Explicit five-index trace, independent of the kernel's matrix products.
    n = len(u)
    result = 0j
    for i in range(n):
        for j in range(n):
            for k in range(n):
                for ell in range(n):
                    for m in range(n):
                        result += jt[i, j] * u[j, k] * j0[k, ell] * rho[ell, m] * u[i, m].conj()
    return result


@pytest.mark.parametrize("beta", [0.0, 0.01, 1.0, 7.0])
def test_thermal_state_matches_matrix_exponential(beta):
    h = _hermitian(np.random.default_rng(7), 4)
    expected = expm(-beta * h)
    expected /= np.trace(expected)
    actual = np.asarray(jax.jit(thermal_density_matrix)(h, beta))
    np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=2e-12)
    np.testing.assert_allclose(actual, actual.conj().T, atol=2e-14)
    np.testing.assert_allclose(np.trace(actual), 1, atol=2e-14)
    assert np.linalg.eigvalsh(actual).min() >= -2e-14


def test_thermal_state_low_temperature_degeneracy_and_energy_shift():
    h = np.diag([-1000.0, -1000.0, -999.0])
    expected = np.diag([0.5, 0.5, 0.0])
    for beta in [1e6, np.inf]:
        rho = jax.jit(thermal_density_matrix)(h, beta)
        assert np.isfinite(rho).all()
        np.testing.assert_allclose(rho, expected, atol=1e-14)
    h = _hermitian(np.random.default_rng(8), 3)
    np.testing.assert_allclose(
        thermal_density_matrix(h + 137 * np.eye(3), 2.7),
        thermal_density_matrix(h, 2.7),
        rtol=2e-12,
        atol=2e-12,
    )


def test_thermal_batch_and_shared_matrix_broadcast():
    rng = np.random.default_rng(9)
    h = np.stack([_hermitian(rng, 3) for _ in range(3)])
    beta = np.array([0.0, 0.4, 3.0])
    expected = np.stack([thermal_density_matrix(hi, bi) for hi, bi in zip(h, beta)])
    np.testing.assert_allclose(jax.jit(thermal_density_matrix)(h, beta), expected, atol=1e-13)
    expected_shared = np.stack([thermal_density_matrix(h[0], bi) for bi in beta])
    np.testing.assert_allclose(thermal_density_matrix(h[0], beta), expected_shared, atol=1e-13)
    np.testing.assert_allclose(jax.vmap(thermal_density_matrix)(h, beta), expected, atol=1e-13)


def test_complex_current_correlation_matches_explicit_trace_and_batches():
    rng = np.random.default_rng(10)
    u, _ = np.linalg.qr(_hermitian(rng, 3))
    rho = np.asarray(thermal_density_matrix(_hermitian(rng, 3), 0.7))
    jt, j0 = _hermitian(rng, 3), _hermitian(rng, 3)
    expected = _trace_reference(u, rho, jt, j0)
    actual = jax.jit(current_correlation)(u, rho, jt, j0)
    np.testing.assert_allclose(actual, expected, atol=2e-12)
    batched = jax.jit(current_correlation)(np.stack([u, u]), rho, np.stack([jt, 2 * jt]), j0)
    np.testing.assert_allclose(batched, [expected, 2 * expected], atol=2e-12)


def test_legacy_conversion_retains_minus_sign():
    rng = np.random.default_rng(11)
    raw = rng.normal(size=(3, 3))
    raw = raw - raw.T
    physical = legacy_current_to_physical(raw)
    np.testing.assert_allclose(physical, physical.conj().T)
    u = expm(-0.3j * _hermitian(rng, 3))
    rho = np.asarray(thermal_density_matrix(_hermitian(rng, 3), 0.4))
    expected = -_trace_reference(u, rho, raw, raw)
    np.testing.assert_allclose(legacy_current_correlation(u, rho, raw, raw), expected, atol=2e-12)
    at_zero = legacy_current_correlation(np.eye(3), rho, raw, raw)
    assert float(jnp.real(at_zero)) >= 0
    np.testing.assert_allclose(jnp.imag(at_zero), 0, atol=1e-14)


def test_current_correlation_basis_covariance_and_gradient():
    rng = np.random.default_rng(12)
    u = expm(-0.2j * _hermitian(rng, 3))
    rho = np.asarray(thermal_density_matrix(_hermitian(rng, 3), 0.7))
    jt, j0 = _hermitian(rng, 3), _hermitian(rng, 3)
    v, _ = np.linalg.qr(_hermitian(rng, 3))
    def transform(x):
        return v.conj().T @ x @ v
    expected = current_correlation(u, rho, jt, j0)
    actual = current_correlation(*(transform(x) for x in [u, rho, jt, j0]))
    np.testing.assert_allclose(actual, expected, atol=3e-12)
    derivative = jax.jit(jax.grad(lambda scale: jnp.real(current_correlation(u, rho, scale * jt, j0))))(1.0)
    np.testing.assert_allclose(derivative, np.real(expected), atol=2e-12)


@pytest.mark.parametrize("shape", [(3,), (2, 3), (0, 0)])
def test_thermal_rejects_invalid_electronic_shape(shape):
    with pytest.raises(ValueError):
        thermal_density_matrix(np.zeros(shape), 1.0)


def test_current_rejects_mismatched_electronic_dimensions():
    with pytest.raises(ValueError, match="dimensions"):
        current_correlation(np.eye(2), np.eye(2), np.eye(3), np.eye(2))
