"""Independent exact-arithmetic definitions for an isolated compact LF prototype."""

import importlib.util
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest


SPEC = importlib.util.spec_from_file_location("compact_lf_prototype",
    Path(__file__).resolve().parents[1]/"benchmarks/compact_lf_prototype.py")
prototype = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = prototype
SPEC.loader.exec_module(prototype)


def direct(unitary, density, jt, j0, pairs, phi0, phit):
    """Literal index contraction; no compact-builder or baseline identities."""
    factor = unitary.conj() @ density.T
    value = 0j
    for i, j in pairs:
        for k, ell in pairs:
            diagonal_count = int(i == j)+int(k == ell)
            sector = int(i == k)-int(j == k)-int(i == ell)+int(j == ell)
            bath = np.exp((-2+diagonal_count)*phi0-sector*phit)
            value += jt[i, j]*j0[k, ell]*unitary[j, k]*factor[i, ell]*bath
    return value


def fixture(n=7):
    rng = np.random.default_rng(539)
    return [rng.normal(size=(n, n))+1j*rng.normal(size=(n, n)) for _ in range(4)]


SUPPORTS = {
    "empty": [],
    "diagonal": [(0, 0), (3, 3), (6, 6)],
    "one_direction": [(0, 1)],
    "disconnected": [(0, 1), (1, 0), (3, 4), (4, 3)],
    "mixed": [(0, 1), (1, 0), (0, 3), (2, 6), (6, 2), (4, 4), (1, 1), (4, 0)],
    "dense": [(i, j) for i in range(7) for j in range(7)],
}


@pytest.mark.parametrize("name", SUPPORTS)
@pytest.mark.parametrize("phi0,phit", [(.0, .0), (.8, .0), (.8, .7-.31j), (.8, 1e-14-2e-14j),
                                      (1000., 1000.-.2j)])
def test_matches_literal_masked_four_index_sum_for_complex_nonunitary_inputs(name, phi0, phit):
    matrices = fixture()
    pairs = SUPPORTS[name]
    topology = prototype.build_compact_sectors(pairs, 7)
    actual = jax.jit(lambda *arrays: prototype.compact_lf_correlation(
        *arrays, topology, phi0, phit))(*matrices)
    expected = direct(*matrices, pairs, phi0, phit)
    np.testing.assert_allclose(actual, expected, atol=2e-10, rtol=3e-13)
    assert np.isfinite(actual)


@pytest.mark.parametrize("name", SUPPORTS)
def test_builder_contains_exactly_the_nonzero_cartesian_pair_sectors(name):
    pairs = SUPPORTS[name]
    topology = prototype.build_compact_sectors(pairs, 7)
    expected = {}
    for i, j in pairs:
        for k, ell in pairs:
            sector = int(i == k)-int(j == k)-int(i == ell)+int(j == ell)
            if sector:
                expected[i, j, k, ell] = sector
    actual = {tuple(quad): int(sector) for quad, sector in
              zip(topology.quad_indices, topology.sector_indices, strict=True)}
    assert actual == expected
    assert len(actual) == len(topology.quad_indices)
    assert all(not array.flags.writeable for array in
               (topology.support_pairs, topology.quad_indices, topology.sector_indices))


def test_values_outside_declared_support_never_enter_baseline_or_correction():
    matrices = fixture()
    pairs = SUPPORTS["mixed"]
    topology = prototype.build_compact_sectors(pairs, 7)
    current_t, current_0 = matrices[2:]
    masked_t, masked_0 = np.zeros_like(current_t), np.zeros_like(current_0)
    for i, j in pairs:
        masked_t[i, j], masked_0[i, j] = current_t[i, j], current_0[i, j]
    full = prototype.compact_lf_correlation(*matrices, topology, .6, .3+.17j)
    masked = prototype.compact_lf_correlation(*matrices[:2], masked_t, masked_0, topology, .6, .3+.17j)
    np.testing.assert_array_equal(full, masked)


@pytest.mark.parametrize("capture_constants", [False, True])
def test_broadcast_probe_and_trajectory_axes_match_independent_single_cases(capture_constants):
    u, rho, jt, j0 = fixture()
    topology = prototype.build_compact_sectors(SUPPORTS["mixed"], 7)
    us = np.stack([u, u*.7+.1j])[:, None]
    rhos = np.stack([rho, rho*.9])[:, None]
    currents = np.stack([jt, jt*.4+1j, jt.T])
    phi0 = np.array([.5, .8])[:, None]
    phit = np.array([.3+.1j, .2-.4j])[:, None]
    def evaluate(u, rho, jt, j0, phi0, phit):
        return prototype.compact_lf_correlation(u, rho, jt, j0, topology, phi0, phit)
    result = (jax.jit(lambda: evaluate(us, rhos, currents, j0, phi0, phit))()
              if capture_constants else jax.jit(evaluate)(us, rhos, currents, j0, phi0, phit))
    expected = [[direct(us[b, 0], rhos[b, 0], currents[p], j0, SUPPORTS["mixed"],
                        phi0[b, 0], phit[b, 0]) for p in range(3)] for b in range(2)]
    np.testing.assert_allclose(result, expected, atol=2e-11, rtol=3e-13)


def test_coordinate_bath_and_matrix_gradients_match_independent_finite_difference():
    u, rho, jt, j0 = fixture()
    pairs = SUPPORTS["mixed"]
    topology = prototype.build_compact_sectors(pairs, 7)
    def objective(x):
        value = prototype.compact_lf_correlation(u+x[0]*np.eye(7), rho, jt*x[1], j0,
                                                topology, x[2], x[3]+.23j)
        return value.real
    def reference(x):
        return direct(u+x[0]*np.eye(7), rho, jt*x[1], j0, pairs, x[2], x[3]+.23j).real
    x = np.array([.2, .8, .7, .3])
    step = 1e-5
    expected = [(reference(x+step*d)-reference(x-step*d))/(2*step) for d in np.eye(4)]
    np.testing.assert_allclose(jax.jit(jax.grad(objective))(x), expected, atol=2e-8, rtol=2e-9)


def test_strong_coupling_gradient_avoids_overflow_in_inactive_algebra():
    u, rho, jt, j0 = fixture()
    topology = prototype.build_compact_sectors(SUPPORTS["mixed"], 7)
    def objective(phi):
        return prototype.compact_lf_correlation(u, rho, jt, j0, topology, phi, phi+.2j).real
    actual = jax.jit(jax.value_and_grad(objective))(1000.)
    assert np.isfinite(np.asarray(actual)).all()
    expected = direct(u, rho, jt, j0, SUPPORTS["mixed"], 1000., 1000.+.2j).real
    np.testing.assert_allclose(actual[0], expected, atol=2e-11, rtol=3e-13)
    np.testing.assert_allclose(actual[1], 0., atol=2e-11)


@pytest.mark.parametrize("pairs,nstates", [([(0, 1), (0, 1)], 2), ([(0, 2)], 2),
    ([(-1, 0)], 2), ([(0., 1.)], 2), ([(0, 1, 2)], 3), ([], 0), ([], True)])
def test_invalid_support_is_rejected(pairs, nstates):
    with pytest.raises(ValueError):
        prototype.build_compact_sectors(pairs, nstates)


def test_bounded_degree_topology_avoids_cartesian_pair_storage():
    n = 1000
    pairs = [(i, (i+d) % n) for i in range(n) for d in (-2, -1, 1, 2)]
    topology = prototype.build_compact_sectors(pairs, n)
    assert len(topology.quad_indices) <= 16*len(pairs)
    assert topology.storage_bytes < len(pairs)**2


def test_matrix_shape_mismatch_is_rejected():
    topology = prototype.build_compact_sectors([], 7)
    with pytest.raises(ValueError, match="shape"):
        prototype.compact_lf_correlation(jnp.eye(6), *fixture()[1:], topology, .4, .2)
