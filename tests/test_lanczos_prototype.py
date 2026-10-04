"""Independent checks of an experimental action; no production integration."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.linalg import expm
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import expm_multiply

from benchmarks.lanczos_prototype import (
    LanczosError, LanczosOptions, lanczos_action, require_success,
)


def random_problem(size=12, seed=917):
    rng = np.random.default_rng(seed)
    raw = rng.normal(size=(size, size)) + 1j * rng.normal(size=(size, size))
    matrix = (raw + raw.conj().T) / (2 * np.sqrt(size))
    vector = rng.normal(size=size) + 1j * rng.normal(size=size)
    return matrix, vector / np.linalg.norm(vector)


def action(matrix, vector, time, **options):
    matrix = jnp.asarray(matrix)
    return lanczos_action(lambda x: matrix @ x, vector, time, LanczosOptions(**options))


@pytest.mark.parametrize("size,capacity,time", [(7, 7, .7), (31, 16, .5), (49, 24, -.8)])
def test_complex_hermitian_against_independent_dense_exponential(size, capacity, time):
    matrix, vector = random_problem(size)
    result = action(matrix, vector, time, max_dimension=capacity)
    calculated = require_success(result)
    expected = expm(-1j * time * matrix) @ vector
    error = np.linalg.norm(calculated - expected)
    assert error <= float(result.error_bound) + 2e-14
    assert float(result.error_bound) <= 1.01e-10
    assert error < 1e-11
    assert abs(np.linalg.norm(calculated) - 1) < 3e-14
    assert float(result.orthogonality_error) < 1e-13


@pytest.mark.parametrize("shift", [-7.0, 0.0, 9.0])
def test_scalar_shift_changes_only_global_phase(shift):
    matrix, vector = random_problem()
    time = .4
    ordinary = action(matrix, vector, time)
    shifted = action(matrix + shift * np.eye(len(vector)), vector, time)
    np.testing.assert_allclose(require_success(shifted),
                               np.exp(-1j * shift * time) * require_success(ordinary),
                               atol=3e-14, rtol=3e-14)


@pytest.mark.parametrize("scale", [0.0, 1e-250, 1e-100, 1e-15])
def test_near_zero_hermitian_operator_does_not_fail_relative_diagnostic(scale):
    matrix, vector = random_problem(8)
    result = action(scale * matrix, vector, .3)
    require_success(result)
    np.testing.assert_allclose(result.value, vector, atol=2e-14, rtol=2e-14)


def test_zero_vector_and_zero_duration_bypass_operator_at_runtime():
    # NaN-producing provider would fail if its result were used. JAX still
    # traces both cond branches; no Python call-count assertion is meaningful.
    def bad_apply(x):
        return jnp.full_like(x, jnp.nan)
    zero = jnp.zeros(4, jnp.complex128)
    vector = jnp.array([1, 2j, -3, 4], jnp.complex128)
    for result, expected in ((lanczos_action(bad_apply, zero, 2), zero),
                             (lanczos_action(bad_apply, vector, 0), vector)):
        np.testing.assert_array_equal(require_success(result), expected)
        assert int(result.iterations) == 0
        assert float(result.error_bound) == 0


def test_exact_invariant_subspace_breakdown_and_unnormalized_input():
    matrix = np.diag([2., 4., 9., 12.])
    vector = np.array([3. + 4j, 0, 0, 0])
    result = action(matrix, vector, .9)
    assert int(result.iterations) == 1
    assert float(result.truncation_bound) == 0
    np.testing.assert_allclose(require_success(result), np.exp(-1.8j) * vector,
                               atol=1e-14, rtol=1e-14)


def test_capacity_exhaustion_is_failure_even_when_norm_is_preserved():
    matrix, vector = random_problem(20)
    result = action(matrix, vector, 2, max_dimension=2)
    assert int(result.status) == 1
    assert int(result.iterations) == 2
    assert np.linalg.norm(result.value - expm(-2j * matrix) @ vector) > .01
    assert abs(np.linalg.norm(result.value) - 1) < 2e-14
    with pytest.raises(LanczosError, match="maximum Krylov dimension") as caught:
        require_success(result)
    assert caught.value.result is result


def test_accepted_finite_truncation_error_and_dimension_convergence():
    matrix, vector = random_problem(40, seed=587)
    reference = expm(-.5j * matrix) @ vector
    errors = []
    for capacity in (3, 6, 9):
        result = action(matrix, vector, .5, max_dimension=capacity, atol=1e-7, rtol=0.)
        error = np.linalg.norm(result.value - reference)
        errors.append(error)
        assert error <= float(result.error_bound) + 2e-14
    require_success(result)
    assert errors[0] > errors[1] * 100
    assert errors[1] > errors[2] * 100
    assert errors[2] > 1e-12
    assert float(result.truncation_bound) > 100 * float(result.roundoff_allowance)


def test_near_breakdown_retains_nonzero_residual_and_can_fail():
    vector = np.ones(2) / np.sqrt(2)
    result = action(np.diag([0., .01]), vector, 100., breakdown_rtol=1.)
    assert int(result.iterations) == 1
    assert int(result.status) == 6
    assert float(result.truncation_bound) > .4
    assert np.linalg.norm(result.value - expm(-100j * np.diag([0., .01])) @ vector) \
        <= float(result.error_bound)


def test_unattainable_tolerance_reports_roundoff_floor():
    result = action(np.diag([1., 2.]), np.array([1., 0.]), .1, atol=1e-30, rtol=0.)
    assert int(result.status) == 5
    assert float(result.roundoff_allowance) > 1e-30
    with pytest.raises(LanczosError, match="roundoff"):
        require_success(result)


def test_large_shift_and_unattainable_orthogonality_have_explicit_failure():
    matrix, vector = random_problem(8)
    shifted = action(matrix + 1e12 * np.eye(8), vector, .2)
    assert int(shifted.status) in (4, 5, 6)
    assert float(shifted.error_bound) > 1e-10
    strict = action(matrix, vector, .2, orthogonality_tolerance=1e-20)
    assert int(strict.status) == 4
    with pytest.raises(LanczosError, match="orthogonality"):
        require_success(strict)


def test_single_state_real_input():
    result = action(np.array([[3.]]), np.array([2.]), -.7)
    assert int(result.iterations) == 1
    np.testing.assert_allclose(require_success(result), np.array([2 * np.exp(2.1j)]),
                               atol=2e-15)


@pytest.mark.parametrize("kind", ["vector", "operator", "time"])
def test_nonfinite_values_have_explicit_status(kind):
    matrix = np.diag([1., 2.])
    vector = np.array([1. + 0j, 1.])
    time = .1
    if kind == "vector":
        vector[0] = np.nan
    elif kind == "operator":
        matrix[0, 0] = np.inf
    else:
        time = np.nan
    result = action(matrix, vector, time)
    assert int(result.status) == 2
    assert np.isinf(result.error_bound)
    with pytest.raises(LanczosError, match="nonfinite"):
        require_success(result)


def test_projected_hermiticity_is_diagnostic_not_global_certificate():
    detected = action(np.diag([1j, 2j]), np.array([1., 0.]), .3)
    assert int(detected.status) == 3
    # The unexplored second column is non-Hermitian. A Krylov action on e0
    # cannot discover it, although the result for this vector is exact.
    invisible = np.array([[1., 7.], [0., 2.]])
    result = action(invisible, np.array([1., 0.]), .3)
    assert int(result.status) == 0
    np.testing.assert_allclose(result.value, expm(-.3j * invisible)[:, 0], atol=1e-14)


def test_column_actions_gram_and_combination_error_are_controlled_by_accuracy():
    matrix, vector = random_problem(32)
    _, second = random_problem(32, seed=425)
    columns = np.column_stack((vector, second, 2 * vector + .7j * second, np.zeros(32)))
    result = action(matrix, columns, .6, max_dimension=16)
    value = np.asarray(require_success(result))
    expected = expm(-.6j * matrix) @ columns
    assert result.status.shape == (4,)
    assert int(result.iterations[-1]) == 0
    np.testing.assert_array_equal(value[:, -1], 0)
    frobenius_bound = np.linalg.norm(result.error_bound)
    assert np.linalg.norm(value - expected) <= frobenius_bound + 3e-14
    gram_error = np.linalg.norm(value.conj().T @ value - columns.conj().T @ columns)
    assert gram_error <= (2 * np.linalg.norm(columns) * frobenius_bound
                          + frobenius_bound**2 + 5e-14)
    assert np.linalg.norm(value[:, 2] - 2 * value[:, 0] - .7j * value[:, 1]) < 2e-12


def test_independent_column_spaces_are_not_exactly_linear_at_low_capacity():
    matrix, vector = random_problem(14)
    _, second = random_problem(14, 825)
    result = action(matrix, np.column_stack((vector, second, vector + second)),
                    1.2, max_dimension=2)
    assert np.all(np.asarray(result.status) == 1)
    assert np.linalg.norm(result.value[:, 2] - result.value[:, 0] - result.value[:, 1]) > 1e-3


def test_jit_vmap_and_runtime_matrix_arguments_match_independent_actions():
    matrix, vector = random_problem(9)
    matrices = jnp.asarray(np.stack((matrix, 1.3 * matrix, matrix + .4 * np.eye(9))))
    vectors = jnp.asarray(np.stack((vector, 2 * vector, np.zeros(9))))
    options = LanczosOptions(max_dimension=9)
    def one(h, v):
        return lanczos_action(lambda x: h @ x, v, .2, options)
    batched = jax.jit(jax.vmap(one))(matrices, vectors)
    for index in range(3):
        eager = one(matrices[index], vectors[index])
        np.testing.assert_allclose(batched.value[index], eager.value, atol=2e-14, rtol=2e-14)
        assert int(batched.status[index]) == 0
        np.testing.assert_allclose(batched.value[index],
                                   expm(-.2j * np.asarray(matrices[index])) @ vectors[index],
                                   atol=3e-14, rtol=3e-14)


def test_matrix_free_complex_ring_against_independent_sparse_taylor_action():
    size = 1024
    rng = np.random.default_rng(2658)
    diagonal = .2 * np.cos(np.arange(size))
    hopping = .3 + .17j
    vector = rng.normal(size=size) + 1j * rng.normal(size=size)
    vector /= np.linalg.norm(vector)
    indices = np.arange(size)
    matrix = coo_matrix((np.concatenate((diagonal, np.full(size, hopping),
                                         np.full(size, hopping.conjugate()))),
                         (np.tile(indices, 3), np.concatenate((indices,
                          (indices - 1) % size, (indices + 1) % size)))),
                        shape=(size, size)).tocsr()
    def apply(v):
        return jnp.asarray(diagonal) * v + hopping * jnp.roll(v, 1) \
            + hopping.conjugate() * jnp.roll(v, -1)
    result = jax.jit(lambda v: lanczos_action(
        apply, v, .4, LanczosOptions(max_dimension=16)))(jnp.asarray(vector))
    expected = expm_multiply(-.4j * matrix, vector)
    error = np.linalg.norm(require_success(result) - expected)
    assert error <= float(result.error_bound) + 1e-14
    assert error < 1e-12


def test_shape_precision_and_option_validation():
    with pytest.raises(ValueError, match="float64 or complex128"):
        lanczos_action(lambda x: x, jnp.ones(3, jnp.float32), .1)
    with pytest.raises(ValueError, match="same vector shape"):
        lanczos_action(lambda x: jnp.ones((3, 1)), jnp.ones(3), .1)
    with pytest.raises(ValueError, match="one real scalar"):
        lanczos_action(lambda x: x, jnp.ones(3), .1j)
    with pytest.raises(ValueError, match="max_dimension"):
        LanczosOptions(max_dimension=True)
    with pytest.raises(ValueError, match="positive"):
        LanczosOptions(atol=0., rtol=0.)
