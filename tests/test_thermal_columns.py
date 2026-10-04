"""Runtime thermal preparation checked against independent dense equations."""

from itertools import product

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.linalg import expm
from scipy.special import ive

from pyeph import thermal


def make_plan(beta, lower, upper, *, tolerance=1e-12, max_degree=100000):
    return thermal.ThermalFilterPlan(beta, lower, upper, polynomial_atol=tolerance,
        max_degree=max_degree, action_id="test-operator-v1", bounds_id="fixture-exact-enclosure-v1")


def ring_action(onsite, hopping, vectors):
    return (onsite[:, None]*vectors + hopping[:, None]*jnp.roll(vectors, -1, axis=0)
            + jnp.conj(jnp.roll(hopping, 1))[:, None]*jnp.roll(vectors, 1, axis=0))


@pytest.mark.parametrize("z", [1e-10, .03, .7, 5., 100., 10000.])
def test_chernoff_bound_dominates_direct_bessel_tail_and_uniform_error(z):
    plan = make_plan(2*z, -1., 1., tolerance=1e-9)
    assert plan.truncation_bound <= 1e-9
    if plan.degree:
        assert thermal._log_tail_bound(z, plan.degree-1) > np.log(1e-9)
    orders = np.arange(plan.degree+1, plan.degree+150+int(20*np.sqrt(z)))
    direct_tail = 2*np.sum(ive(orders, z))
    assert direct_tail <= plan.truncation_bound*(1+1e-11)
    x = np.cos(np.linspace(0, np.pi, 1001))
    error = np.max(abs(np.polynomial.chebyshev.chebval(x, plan.coefficients)-np.exp(-z*(x+1))))
    # This additive allowance explicitly covers observed floating evaluation,
    # and is not included in the mathematical truncation certificate.
    assert error <= plan.truncation_bound+3e-12


@pytest.mark.parametrize("beta", [0., .3, 8., 100.])
@pytest.mark.parametrize("complex_hamiltonian", [False, True])
def test_filter_matches_dense_expm_for_disordered_hermitian_models(beta, complex_hamiltonian):
    rng = np.random.default_rng(951)
    raw = rng.normal(size=(7, 7))
    if complex_hamiltonian:
        raw = raw+1j*rng.normal(size=(7, 7))
    h = (raw+raw.conj().T)/2
    spectrum = np.linalg.eigvalsh(h)
    lower, upper = spectrum[0]-.1, spectrum[-1]+.1
    plan = make_plan(beta, lower, upper, tolerance=1e-11)
    omega = thermal.thermal_random_columns(7, [1, 9, 15], seed=43,
        kind="complex" if complex_hamiltonian else "real")
    result = jax.jit(lambda columns: thermal.prepare_thermal_columns(lambda v: h@v, columns, plan))(omega)
    expected = expm(-beta*(h-lower*np.eye(7))/2)@omega
    observed = np.linalg.norm(np.asarray((result.factor*jnp.exp(result.log_norm_squared/2)))-expected)
    assert observed <= plan.truncation_bound*np.linalg.norm(omega)+2e-12
    assert bool(result.status == 0)
    np.testing.assert_allclose(result.factor, expected/np.linalg.norm(expected), atol=2e-10, rtol=2e-10)
    np.testing.assert_allclose(np.linalg.norm(result.factor), 1., atol=2e-15)


@pytest.mark.parametrize("beta,bounds", [(0., (-9., 5.)), (11., (2., 2.))])
def test_identity_limits_do_not_probe_or_materialize_the_hamiltonian(beta, bounds):
    plan = make_plan(beta, *bounds)
    omega = thermal.thermal_random_columns(11, [0, 8], kind="real")
    def forbidden(_):
        raise AssertionError("identity limit must not probe action")
    result = thermal.prepare_thermal_columns(forbidden, omega, plan)
    np.testing.assert_array_equal((result.factor*jnp.exp(result.log_norm_squared/2)), omega)
    assert result.scaled_partition_estimate == pytest.approx(11)
    assert result.log_partition_estimate == pytest.approx(np.log(11)-beta*bounds[0])
    assert plan.degree == 0 and plan.truncation_bound == 0


@pytest.mark.parametrize("kind", ["real", "complex"])
def test_column_identity_streams_preserve_partitions_order_and_rank_prefix(kind):
    whole = thermal.thermal_random_columns(13, [2, 7, 9, 11], seed=15, trajectory_id=4, kind=kind)
    for ids, indices in (([2, 7], [0, 1]), ([11, 2], [3, 0]), ([9], [2])):
        actual = thermal.thermal_random_columns(13, ids, seed=15, trajectory_id=4, kind=kind)
        np.testing.assert_array_equal(actual, whole[:, indices])
    np.testing.assert_allclose(abs(whole), 1., atol=2e-16)
    other = thermal.thermal_random_columns(13, [2, 7, 9, 11], seed=15, trajectory_id=5, kind=kind)
    assert not np.array_equal(other, whole)


def test_exact_sign_enumeration_exposes_ratio_bias_and_correct_pooled_normalization():
    h = np.array([[.1, .6], [.6, 1.4]])
    observable = np.diag([1., -1.])
    plan = make_plan(2., -.7, 2.1, tolerance=1e-13)
    omega = np.array(list(product([-1., 1.], repeat=2))).T
    exact = expm(-plan.beta*(h-plan.lower*np.eye(2))/2)
    exact_density = exact@exact
    target = np.trace(exact_density@observable)/np.trace(exact_density)
    result = thermal.prepare_thermal_columns(lambda v: h@v, omega, plan)
    y, factor = map(np.asarray, ((result.factor*jnp.exp(result.log_norm_squared/2)), result.factor))
    pooled = np.vdot(factor, observable@factor).real
    individually_normalized = y/np.linalg.norm(y, axis=0)
    rank_one_mean = np.mean(np.sum(individually_normalized*(observable@individually_normalized), axis=0))
    assert abs(rank_one_mean-target) > .01
    np.testing.assert_allclose(pooled, target, atol=2e-13)
    np.testing.assert_allclose(result.scaled_partition_estimate, np.trace(exact_density), atol=2e-13)
    np.testing.assert_allclose(np.sum(abs(factor)**2), 1., atol=2e-15)
    assert np.ptp(np.sum(abs(factor)**2, axis=0)) > .05


def test_fixed_bound_action_parameter_gradient_matches_independent_expm_difference():
    rng = np.random.default_rng(92)
    raw = rng.normal(size=(5, 5))+1j*rng.normal(size=(5, 5))
    base = (raw+raw.conj().T)/8
    shift = np.diag(np.linspace(-.2, .3, 5))
    observable = np.diag(np.linspace(.3, -.9, 5))
    omega = thermal.thermal_random_columns(5, [2, 8, 19], seed=39)
    plan = make_plan(1.7, -2., 2., tolerance=1e-13)
    def objective(scale):
        result = thermal.prepare_thermal_columns(lambda v: (base+scale*shift)@v, omega, plan)
        factor = result.factor
        return jnp.vdot(factor, observable@factor).real
    def reference(scale):
        y = expm(-plan.beta*(base+scale*shift-plan.lower*np.eye(5))/2)@omega
        return np.vdot(y, observable@y).real/np.vdot(y, y).real
    step = 1e-5
    expected = (reference(.6+step)-reference(.6-step))/(2*step)
    np.testing.assert_allclose(jax.jit(jax.grad(objective))(.6), expected, rtol=2e-8, atol=2e-10)


def test_sparse_forward_hlo_contains_no_global_square_arrays_or_eigensolver():
    n, rank = 67, 3
    plan = make_plan(4., -3., 3., tolerance=1e-10)
    omega = thermal.thermal_random_columns(n, np.arange(rank), seed=5)
    def prepare(onsite, hopping, columns):
        return thermal.prepare_thermal_columns(lambda v: ring_action(onsite, hopping, v), columns, plan)
    kernel = jax.jit(prepare)
    lowered = kernel.lower(jnp.linspace(-1., 1., n), jnp.full((n,), .3+.1j), omega)
    hlo = str(lowered.compiler_ir(dialect="stablehlo"))
    assert f"{n}x{n}" not in hlo
    assert "eigh" not in hlo and "lapack" not in hlo
    assert f"{n}x{rank}" in hlo


def test_zero_or_nonfinite_normalization_is_explicitly_invalid():
    plan = make_plan(0., -1., 1.)
    for columns, expected_status in ((np.zeros((3, 2)), 2),
                                     (np.full((3, 2), np.inf), 1),
                                     (np.full((3, 2), np.nan), 1)):
        result = thermal.prepare_thermal_columns(lambda v: v, columns, plan)
        assert int(result.status) == expected_status
        assert np.isnan(np.asarray(result.factor)).all()


@pytest.mark.parametrize("args,kwargs", [((-1., 0., 1.), {}), ((1., 2., 1.), {}),
    ((1., 0., np.inf), {}), ((True, 0., 1.), {}), ((1., 0., 1.), {"tolerance": 0.}),
    ((100., 0., 1.), {"max_degree": 1}), ((1e-200, 0., 1e-200), {})])
def test_invalid_planning_or_insufficient_degree_budget_fails(args, kwargs):
    with pytest.raises(ValueError):
        make_plan(*args, **kwargs)


def test_invalid_action_shape_and_duplicate_column_ids_fail():
    with pytest.raises(ValueError, match="preserve"):
        thermal.prepare_thermal_columns(lambda v: v[:, 0], np.ones((3, 2)), make_plan(1., 0., 2.))
    with pytest.raises(ValueError, match="unique"):
        thermal.thermal_random_columns(5, [2, 2])


def test_a_positive_exact_arithmetic_tail_is_not_reported_as_zero_after_underflow():
    plan = make_plan(2e-200, -1., 1., tolerance=1e-300)
    assert np.isfinite(plan.log_truncation_bound)
    assert plan.log_truncation_bound < np.log(np.nextafter(0., 1.))
    assert plan.truncation_bound == np.nextafter(0., 1.)


@pytest.mark.parametrize("enable_x64", [True, False])
def test_execution_tail_underflow_uses_normal_tiny_but_identity_stays_zero(enable_x64):
    previous = jax.config.x64_enabled
    try:
        jax.config.update("jax_enable_x64", enable_x64)
        omega = jnp.ones((3, 2))
        result = thermal.prepare_thermal_columns(lambda v: jnp.zeros_like(v), omega,
            make_plan(2e-200, -1., 1., tolerance=1e-300))
        assert int(result.status) == 0
        assert result.relative_truncation_bound == np.finfo(result.factor.real.dtype).tiny
        assert result.normalized_factor_error_bound > 0
        identity = thermal.prepare_thermal_columns(lambda v: v, omega, make_plan(0., -1., 1.))
        assert identity.relative_truncation_bound == 0
        invalid = thermal.prepare_thermal_columns(lambda v: jnp.full_like(v, jnp.nan), omega,
                                                 make_plan(2e-200, -1., 1., tolerance=1e-300))
        assert np.isnan(invalid.relative_truncation_bound)
    finally:
        jax.config.update("jax_enable_x64", previous)


def test_loose_lower_bound_is_rejected_even_when_formal_tail_is_tiny():
    # exp[-beta*(0-lower)/2] is far below working roundoff. A tiny requested
    # polynomial tail must not bless a factor dominated by recurrence error.
    plan = make_plan(1., -1000., 1., tolerance=1e-100)
    result = thermal.prepare_thermal_columns(lambda v: jnp.zeros_like(v), jnp.ones((4, 3)), plan)
    assert int(result.status) in (1, 2, 3, 4)
    with pytest.raises(thermal.ThermalPreparationError) as error:
        thermal.require_success(result)
    assert error.value.result is result
    assert np.isnan(result.factor).all()


def test_relative_truncation_and_precision_screens_are_separate_failures():
    columns = jnp.ones((3, 2))
    coarse = make_plan(1., -1., 1., tolerance=.1)
    result = thermal.prepare_thermal_columns(lambda v: .4*v, columns, coarse, rtol=1e-8)
    assert int(result.status) == 3
    assert result.relative_truncation_bound > 1e-8
    exact = make_plan(0., -1., 1.)
    result = thermal.prepare_thermal_columns(lambda v: v, columns, exact, rtol=1e-20)
    assert result.relative_truncation_bound == 0.
    assert result.heuristic_precision_screen > 1e-20
    assert int(result.status) == 4


def test_scaled_normalization_survives_raw_partition_underflow():
    columns = jnp.asarray([[1., 2.], [-3., 4.]])*1e-200
    result = thermal.prepare_thermal_columns(lambda v: v, columns, make_plan(0., -1., 1.))
    factor = thermal.require_success(result)
    np.testing.assert_allclose(factor, np.array([[1., 2.], [-3., 4.]])/np.sqrt(30.), atol=3e-16)
    assert bool(result.scaled_partition_underflow)
    assert result.scaled_partition_estimate == 0.
    assert np.isfinite(result.log_partition_estimate)
    assert result.log_partition_estimate == pytest.approx(np.log(15.)+2*np.log(1e-200))


def test_precision_screen_uses_actual_single_precision_execution():
    previous = jax.config.x64_enabled
    try:
        jax.config.update("jax_enable_x64", False)
        result = thermal.prepare_thermal_columns(lambda v: v, jnp.ones((3, 2)),
                                                make_plan(0., -1., 1.), rtol=1e-9)
        assert result.factor.dtype == jnp.float32
        assert result.heuristic_precision_screen == pytest.approx(np.finfo(np.float32).eps)
        assert int(result.status) == 4
    finally:
        jax.config.update("jax_enable_x64", previous)


def test_scaled_normalization_survives_raw_partition_overflow_with_json_diagnostics():
    import json
    columns = jnp.asarray([[1., 2.], [-3., 4.]])*1e200
    result = thermal.prepare_thermal_columns(lambda v: v, columns, make_plan(0., -1., 1.))
    np.testing.assert_allclose(thermal.require_success(result),
                              np.array([[1., 2.], [-3., 4.]])/np.sqrt(30.), atol=3e-16)
    assert bool(result.scaled_partition_overflow)
    assert np.isfinite(result.log_partition_estimate)
    diagnostics = result.diagnostics()
    assert diagnostics["scaled_partition_estimate"] is None
    json.dumps(diagnostics, allow_nan=False)


def test_vmap_keeps_each_geometry_normalized_and_partition_weights_separate():
    columns = jnp.asarray([[1., 1.], [1., -1.]])
    plan = make_plan(2., -1., 3.)
    matrices = jnp.asarray([np.diag([0., .7]), np.diag([1.3, 2.])])
    results = jax.jit(jax.vmap(lambda h: thermal.prepare_thermal_columns(lambda v: h@v, columns, plan)))(matrices)
    factors = thermal.require_success(results)
    np.testing.assert_allclose(np.sum(abs(factors)**2, axis=(1, 2)), 1., atol=3e-16)
    np.testing.assert_allclose(factors[0], factors[1], atol=2e-13)
    assert results.scaled_partition_estimate[0] > 10*results.scaled_partition_estimate[1]


def test_partitioned_columns_pool_by_norm_weight_not_per_column_normalization():
    h = jnp.asarray([[.1, .6], [.6, 1.4]])
    omega = thermal.thermal_random_columns(2, [1, 8, 19, 30, 32], seed=53)
    plan = make_plan(2., -.7, 2.1)
    full = thermal.prepare_thermal_columns(lambda v: h@v, omega, plan)
    parts = [thermal.prepare_thermal_columns(lambda v: h@v, x, plan)
             for x in (omega[:, :2], omega[:, 2:])]
    logs = jnp.array([p.log_norm_squared for p in parts])
    weights = jax.nn.softmax(logs)
    pooled = np.concatenate([thermal.require_success(p)*np.sqrt(w)
                             for p, w in zip(parts, weights, strict=True)], axis=1)
    np.testing.assert_allclose(pooled, thermal.require_success(full), atol=3e-16)


def test_scalar_column_id_and_rbg_keep_the_declared_threefry_stream():
    reference = thermal.thermal_random_columns(7, [4, 29], seed=13, trajectory_id=17)
    with jax.default_prng_impl("rbg"):
        actual = thermal.thermal_random_columns(7, 29, seed=13, trajectory_id=17)
    np.testing.assert_array_equal(actual, reference[:, 1:])


def test_metadata_contains_caller_identities_and_owned_coefficients():
    plan = make_plan(2., -1., 1.)
    metadata = plan.metadata()
    assert metadata["action_id"] == "test-operator-v1"
    assert metadata["bounds_id"] == "fixture-exact-enclosure-v1"
    assert len(metadata["coefficients_sha256"]) == 64
    assert metadata["floating_error_certificate"] is None
    assert not plan.coefficients.flags.writeable
    for key in ("action_id", "bounds_id"):
        values = dict(action_id="operator", bounds_id="bounds")
        values[key] = ""
        with pytest.raises(ValueError, match=key):
            thermal.ThermalFilterPlan(1., -1., 1., **values)
