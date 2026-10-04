"""Public numerical API and immutable options, complementing the action oracles."""

from dataclasses import FrozenInstanceError

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from benchmarks.lanczos_prototype import lanczos_action as benchmark_action
from pyeph.integrators.krylov import LanczosOptions, lanczos_action, require_success


def test_options_snapshot_mutable_numpy_scalars():
    dimension = np.array(12, dtype=np.int64)
    tolerance = np.array(1e-9)
    options = LanczosOptions(max_dimension=dimension, atol=tolerance, rtol=0.)
    dimension[...] = 1
    tolerance[...] = 10
    assert options.max_dimension == 12
    assert options.atol == 1e-9
    assert type(options.max_dimension) is int
    assert type(options.atol) is float
    with pytest.raises(FrozenInstanceError):
        options.max_dimension = 1


@pytest.mark.parametrize("kwargs", [dict(max_dimension=True), dict(max_dimension=2.),
                                    dict(max_dimension=0), dict(atol=np.inf),
                                    dict(rtol=np.nan), dict(breakdown_rtol=-1),
                                    dict(orthogonality_tolerance=1j)])
def test_invalid_options_fail_at_host_construction(kwargs):
    with pytest.raises(ValueError):
        LanczosOptions(**kwargs)


def test_public_result_names_total_as_estimate_and_legacy_wrapper_is_exact():
    matrix = jnp.array([[.4, .1 + .2j], [.1 - .2j, -.3]], jnp.complex128)
    vector = jnp.array([.3 + .2j, -.4j])
    def apply(v):
        return matrix @ v
    result = lanczos_action(apply, vector, .7)
    legacy = benchmark_action(apply, vector, .7)
    assert hasattr(result, "error_estimate")
    assert not hasattr(result, "error_bound")
    assert hasattr(legacy, "error_bound")
    for left, right in zip(result, legacy, strict=True):
        np.testing.assert_array_equal(left, right)
    assert float(result.error_estimate) == float(result.truncation_bound
        + result.recurrence_bound + result.roundoff_allowance)
    np.testing.assert_array_equal(require_success(result), legacy.value)


def test_checked_result_is_a_jax_pytree_with_runtime_operator_arguments():
    vectors = jnp.eye(3, dtype=jnp.complex128)[:, :2]
    function = jax.jit(lambda diagonal: lanczos_action(
        lambda v: diagonal * v, vectors, .3))
    first = function(jnp.array([1., 2., 3.]))
    second = function(jnp.array([2., 3., 4.]))
    assert np.all(np.asarray(first.status) == 0)
    assert np.all(np.asarray(second.status) == 0)
    np.testing.assert_allclose(second.value, np.exp(-.3j) * first.value, atol=2e-15)
    assert len(jax.tree.leaves(first)) == 9


def test_dynamic_absolute_budget_replaces_relative_rule_per_column():
    rng = np.random.default_rng(826)
    raw = rng.normal(size=(32, 32)) + 1j * rng.normal(size=(32, 32))
    matrix = jnp.asarray((raw + raw.conj().T) / (2 * np.sqrt(32)))
    vector = rng.normal(size=32) + 1j * rng.normal(size=32)
    vector /= np.linalg.norm(vector)
    columns = jnp.asarray(np.column_stack((vector, vector)))
    options = LanczosOptions(max_dimension=9, atol=1e-4, rtol=0.)
    function = jax.jit(lambda budget: lanczos_action(
        lambda v: matrix @ v, columns, .5, options, absolute_budget=budget))
    first = function(jnp.array([1e-6, 1e-12]))
    second = function(jnp.array([1e-12, 1e-6]))
    np.testing.assert_array_equal(first.status, [0, 1])
    np.testing.assert_array_equal(second.status, [1, 0])
    np.testing.assert_array_equal(first.value, second.value)
    scalar = lanczos_action(lambda v: matrix @ v, columns, .5, options, absolute_budget=1e-6)
    assert np.all(np.asarray(scalar.status) == 0)


@pytest.mark.parametrize("budget", [-1., np.inf, np.nan])
def test_invalid_runtime_budgets_fail_even_for_trivial_actions(budget):
    result = jax.jit(lambda b: lanczos_action(
        lambda v: v, jnp.zeros(2, jnp.complex128), 0., absolute_budget=b))(budget)
    assert int(result.status) == 2


def test_zero_budget_exact_bypass_and_budget_shape_validation():
    result = lanczos_action(lambda v: v, jnp.zeros(2), 1., absolute_budget=0.)
    assert int(result.status) == 0
    assert float(result.error_estimate) == 0
    for budget in (jnp.ones(2), 1j, True):
        with pytest.raises(ValueError, match="absolute_budget"):
            lanczos_action(lambda v: v, jnp.ones(3), .1, absolute_budget=budget)
