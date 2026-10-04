"""LF kernels checked against independent NumPy mode and four-index sums."""

import itertools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pyeph.models.polaron import band_narrow_hamiltonian, lf_band_narrowing, lf_phi
from pyeph.observables.transport.greenkubo import current_correlation, thermal_density_matrix
from pyeph.observables.transport.polaron import (
    build_lf_sectors,
    lf_current_correlation,
    lf_current_correlation_dense,
)


def _phi_reference(w, g, beta, time):
    total = 0j
    for frequency, coupling in zip(w, g):
        total += abs(coupling / frequency) ** 2 * (
            np.cos(frequency * time) / np.tanh(beta * frequency / 2)
            - 1j * np.sin(frequency * time)
        )
    return total


def _lf_reference(u, rho, jt, j0, phi0, phit):
    n = len(u)
    total = 0j
    # Four independent state indices and a separate explicit density sum.
    for i, j, k, ell in itertools.product(range(n), repeat=4):
        sector = int(i == k) - int(j == k) - int(i == ell) + int(j == ell)
        factor = np.exp((-2 + int(i == j) + int(k == ell)) * phi0 - sector * phit)
        g_il = sum(u[i, m].conjugate() * rho[ell, m] for m in range(n))
        total += jt[i, j] * j0[k, ell] * u[j, k] * g_il * factor
    return total


def _complex_example():
    rng = np.random.default_rng(37)
    raw = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
    u, _ = np.linalg.qr(raw)
    h = raw + raw.conj().T
    rho = np.asarray(thermal_density_matrix(h, 0.3))
    jt = np.array([[0, 0.3 + 0.2j, 0], [0.3 - 0.2j, 0, -0.2j], [0, 0.2j, 0]])
    j0 = np.array([[0, -0.1j, 0], [0.1j, 0, 0.4], [0, 0.4, 0]])
    return u, rho, jt, j0


@pytest.mark.parametrize("beta", [0.01, 0.5, 1000.0, np.inf])
@pytest.mark.parametrize("time", [0.0, 0.7, -0.7])
def test_lf_phi_matches_independent_mode_sum(beta, time):
    w, g = np.array([0.7, 2.1, 3.0]), np.array([-0.2, 0.4, 0.13])
    expected = _phi_reference(w, g, beta, time)
    np.testing.assert_allclose(jax.jit(lf_phi)(w, g, beta, time), expected, atol=2e-13)
    expected_f = np.exp(-_phi_reference(w, g, beta, 0.0).real)
    np.testing.assert_allclose(jax.jit(lf_band_narrowing)(w, g, beta), expected_f, atol=2e-13)


def test_lf_empty_bath_time_batch_and_derivative():
    times = np.array([0, 0.3, 1.0])
    np.testing.assert_allclose(lf_phi(np.empty(0), np.empty(0), 1.0, times), np.zeros(3))
    np.testing.assert_allclose(lf_band_narrowing(np.empty(0), np.empty(0), 1.0), 1.0)
    w, g = np.array([0.7, 2.1]), np.array([-0.2, 0.4])
    expected = np.array([_phi_reference(w, g, 0.5, t) for t in times])
    np.testing.assert_allclose(jax.jit(lf_phi)(w, g, 0.5, times), expected, atol=2e-13)
    derivative = jax.jit(jax.grad(lambda t: jnp.real(lf_phi(w, g, 0.5, t))))(0.7)
    epsilon = 1e-6
    reference = (_phi_reference(w, g, 0.5, 0.7 + epsilon).real - _phi_reference(w, g, 0.5, 0.7 - epsilon).real) / (2 * epsilon)
    np.testing.assert_allclose(derivative, reference, rtol=2e-8, atol=1e-10)


def test_band_narrowing_preserves_disorder_and_hermiticity():
    h = np.array([[2.0, 0.5 + 0.7j], [0.5 - 0.7j, -3.0]])
    actual = np.asarray(jax.jit(band_narrow_hamiltonian)(h, 0.2))
    np.testing.assert_allclose(actual.diagonal(), h.diagonal())
    np.testing.assert_allclose(actual[0, 1], 0.2 * h[0, 1])
    np.testing.assert_allclose(actual, actual.conj().T)
    batch = band_narrow_hamiltonian(h, np.array([0.2, 1.0]))
    np.testing.assert_allclose(batch[0], actual)
    np.testing.assert_allclose(batch[1], h)
    assert not np.allclose(actual, 0.2 * h)  # Deliberate correction of legacy whole-H scaling.


def test_lf_sparse_sectors_match_four_index_numpy_reference():
    u, rho, jt, j0 = _complex_example()
    edges = np.argwhere((abs(jt) + abs(j0)) > 0)
    quads, sectors = build_lf_sectors(edges, nstates=3)
    assert quads.shape == (len(edges) ** 2, 4)
    phi0, phit = 0.43, 0.17 - 0.23j
    expected = _lf_reference(u, rho, jt, j0, phi0, phit)
    actual = jax.jit(lf_current_correlation)(u, rho, jt, j0, quads, sectors, phi0, phit)
    dense = jax.jit(lf_current_correlation_dense)(u, rho, jt, j0, phi0, phit)
    np.testing.assert_allclose(actual, expected, atol=2e-13)
    np.testing.assert_allclose(dense, expected, atol=2e-13)


def test_lf_zero_coupling_reduces_to_bare_correlation():
    u, rho, jt, j0 = _complex_example()
    # Including diagonal probes exercises the delta_ij and delta_kl terms too.
    jt += np.diag([0.1, -0.3, 0.2])
    j0 += np.diag([0.3, 0.1, 0.0])
    quads, sectors = build_lf_sectors(np.argwhere((abs(jt) + abs(j0)) > 0))
    expected = current_correlation(u, rho, jt, j0)
    np.testing.assert_allclose(lf_current_correlation(u, rho, jt, j0, quads, sectors, 0.0, 0.0), expected, atol=2e-13)
    np.testing.assert_allclose(lf_current_correlation_dense(u, rho, jt, j0, 0.0, 0.0), expected, atol=2e-13)


def test_lf_batch_gauge_invariance_and_zero_current_support():
    u, rho, jt, j0 = _complex_example()
    quads, sectors = build_lf_sectors(np.argwhere((abs(jt) + abs(j0)) > 0))
    phit = np.array([0.17 - 0.23j, -0.1 + 0.03j])
    actual = jax.jit(lf_current_correlation)(u, rho, jt, j0, quads, sectors, 0.43, phit)
    expected = [_lf_reference(u, rho, jt, j0, 0.43, p) for p in phit]
    np.testing.assert_allclose(actual, expected, atol=2e-13)
    gauge = np.diag(np.exp(1j * np.array([0.7, -0.3, 1.1])))
    def transform(x):
        return gauge.conj().T @ x @ gauge
    transformed = lf_current_correlation(*(transform(x) for x in (u, rho, jt, j0)), quads, sectors, 0.43, phit)
    np.testing.assert_allclose(transformed, expected, atol=2e-13)
    empty_quads, empty_sectors = build_lf_sectors([])
    empty = lf_current_correlation(u, rho, 0 * jt, 0 * j0, empty_quads, empty_sectors, 0.43, phit)
    np.testing.assert_allclose(empty, np.zeros(2), atol=1e-14)


@pytest.mark.parametrize("pairs", [[[0, 1], [0, 1]], [[-1, 0]], [[0.5, 1.0]], [0, 1, 2], [[0, 3]]])
def test_lf_sector_builder_rejects_invalid_edges(pairs):
    with pytest.raises(ValueError):
        build_lf_sectors(pairs, nstates=3)


def test_lf_rejects_mismatched_mode_shapes():
    with pytest.raises(ValueError, match="same modes"):
        lf_phi(np.ones(2), np.ones(3), 1.0, 0.0)
