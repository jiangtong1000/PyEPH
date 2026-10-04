"""Independent equation checks for the isolated thermal-filter prototype."""

import importlib.util
from itertools import product
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.linalg import expm
from scipy.special import ive


SPEC = importlib.util.spec_from_file_location("thermal_filter_prototype",
    Path(__file__).resolve().parents[1]/"benchmarks/thermal_filter_prototype.py")
prototype = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = prototype
SPEC.loader.exec_module(prototype)


@pytest.mark.parametrize("z", [1e-10, .03, .7, 5., 100., 10000.])
def test_chernoff_bound_dominates_direct_bessel_tail_and_uniform_error(z):
    plan = prototype.make_plan(2*z, -1., 1., tolerance=1e-9)
    assert plan.truncation_bound <= 1e-9
    if plan.degree:
        assert prototype.log_tail_bound(z, plan.degree-1) > np.log(1e-9)
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
    plan = prototype.make_plan(beta, lower, upper, tolerance=1e-11)
    omega = prototype.random_columns(7, [1, 9, 15], seed=43,
        kind="complex" if complex_hamiltonian else "real")
    result = jax.jit(lambda columns: prototype.prepare_columns(lambda v: h@v, columns, plan))(omega)
    expected = expm(-beta*(h-lower*np.eye(7))/2)@omega
    observed = np.linalg.norm(np.asarray(result["filtered"])-expected)
    assert observed <= plan.truncation_bound*np.linalg.norm(omega)+2e-12
    assert bool(result["valid"])
    np.testing.assert_allclose(result["factor"], expected/np.linalg.norm(expected), atol=2e-10, rtol=2e-10)
    np.testing.assert_allclose(np.linalg.norm(result["factor"]), 1., atol=2e-15)


@pytest.mark.parametrize("beta,bounds", [(0., (-9., 5.)), (11., (2., 2.))])
def test_identity_limits_do_not_probe_or_materialize_the_hamiltonian(beta, bounds):
    plan = prototype.make_plan(beta, *bounds)
    omega = prototype.random_columns(11, [0, 8], kind="real")
    def forbidden(_):
        raise AssertionError("identity limit must not probe action")
    result = prototype.prepare_columns(forbidden, omega, plan)
    np.testing.assert_array_equal(result["filtered"], omega)
    assert result["scaled_partition_estimate"] == 11
    assert result["log_partition_estimate"] == pytest.approx(np.log(11)-beta*bounds[0])
    assert plan.degree == 0 and plan.truncation_bound == 0


@pytest.mark.parametrize("kind", ["real", "complex"])
def test_column_identity_streams_preserve_partitions_order_and_rank_prefix(kind):
    whole = prototype.random_columns(13, [2, 7, 9, 11], seed=15, trajectory_id=4, kind=kind)
    for ids, indices in (([2, 7], [0, 1]), ([11, 2], [3, 0]), ([9], [2])):
        actual = prototype.random_columns(13, ids, seed=15, trajectory_id=4, kind=kind)
        np.testing.assert_array_equal(actual, whole[:, indices])
    np.testing.assert_allclose(abs(whole), 1., atol=2e-16)
    other = prototype.random_columns(13, [2, 7, 9, 11], seed=15, trajectory_id=5, kind=kind)
    assert not np.array_equal(other, whole)


def test_exact_sign_enumeration_exposes_ratio_bias_and_correct_pooled_normalization():
    h = np.array([[.1, .6], [.6, 1.4]])
    observable = np.diag([1., -1.])
    plan = prototype.make_plan(2., -.7, 2.1, tolerance=1e-13)
    omega = np.array(list(product([-1., 1.], repeat=2))).T
    exact = expm(-plan.beta*(h-plan.lower*np.eye(2))/2)
    exact_density = exact@exact
    target = np.trace(exact_density@observable)/np.trace(exact_density)
    result = prototype.prepare_columns(lambda v: h@v, omega, plan)
    y, factor = map(np.asarray, (result["filtered"], result["factor"]))
    pooled = np.vdot(factor, observable@factor).real
    individually_normalized = y/np.linalg.norm(y, axis=0)
    rank_one_mean = np.mean(np.sum(individually_normalized*(observable@individually_normalized), axis=0))
    assert abs(rank_one_mean-target) > .01
    np.testing.assert_allclose(pooled, target, atol=2e-13)
    np.testing.assert_allclose(result["scaled_partition_estimate"], np.trace(exact_density), atol=2e-13)
    np.testing.assert_allclose(np.sum(abs(factor)**2), 1., atol=2e-15)
    assert np.ptp(np.sum(abs(factor)**2, axis=0)) > .05


def test_fixed_bound_action_parameter_gradient_matches_independent_expm_difference():
    rng = np.random.default_rng(92)
    raw = rng.normal(size=(5, 5))+1j*rng.normal(size=(5, 5))
    base = (raw+raw.conj().T)/8
    shift = np.diag(np.linspace(-.2, .3, 5))
    observable = np.diag(np.linspace(.3, -.9, 5))
    omega = prototype.random_columns(5, [2, 8, 19], seed=39)
    plan = prototype.make_plan(1.7, -2., 2., tolerance=1e-13)
    def objective(scale):
        result = prototype.prepare_columns(lambda v: (base+scale*shift)@v, omega, plan)
        factor = result["factor"]
        return jnp.vdot(factor, observable@factor).real
    def reference(scale):
        y = expm(-plan.beta*(base+scale*shift-plan.lower*np.eye(5))/2)@omega
        return np.vdot(y, observable@y).real/np.vdot(y, y).real
    step = 1e-5
    expected = (reference(.6+step)-reference(.6-step))/(2*step)
    np.testing.assert_allclose(jax.jit(jax.grad(objective))(.6), expected, rtol=2e-8, atol=2e-10)


def test_sparse_forward_hlo_contains_no_global_square_arrays_or_eigensolver():
    n, rank = 67, 3
    plan = prototype.make_plan(4., -3., 3., tolerance=1e-10)
    omega = prototype.random_columns(n, np.arange(rank), seed=5)
    def prepare(onsite, hopping, columns):
        return prototype.prepare_columns(lambda v: prototype.ring_action(onsite, hopping, v), columns, plan)
    kernel = jax.jit(prepare)
    lowered = kernel.lower(jnp.linspace(-1., 1., n), jnp.full((n,), .3+.1j), omega)
    hlo = str(lowered.compiler_ir(dialect="stablehlo"))
    assert f"{n}x{n}" not in hlo
    assert "eigh" not in hlo and "lapack" not in hlo
    assert f"{n}x{rank}" in hlo


def test_zero_or_nonfinite_normalization_is_explicitly_invalid():
    plan = prototype.make_plan(0., -1., 1.)
    for columns in (np.zeros((3, 2)), np.full((3, 2), np.inf)):
        result = prototype.prepare_columns(lambda v: v, columns, plan)
        assert not bool(result["valid"])
        assert np.isnan(np.asarray(result["factor"])).all()


@pytest.mark.parametrize("args,kwargs", [((-1., 0., 1.), {}), ((1., 2., 1.), {}),
    ((1., 0., np.inf), {}), ((True, 0., 1.), {}), ((1., 0., 1.), {"tolerance": 0.}),
    ((100., 0., 1.), {"max_degree": 1})])
def test_invalid_planning_or_insufficient_degree_budget_fails(args, kwargs):
    with pytest.raises(ValueError):
        prototype.make_plan(*args, **kwargs)


def test_invalid_action_shape_and_duplicate_column_ids_fail():
    with pytest.raises(ValueError, match="preserve"):
        prototype.filter_columns(lambda v: v[:, 0], np.ones((3, 2)), prototype.make_plan(1., 0., 2.))
    with pytest.raises(ValueError, match="unique"):
        prototype.random_columns(5, [2, 2])


def test_a_positive_exact_arithmetic_tail_is_not_reported_as_zero_after_underflow():
    plan = prototype.make_plan(2e-200, -1., 1., tolerance=1e-300)
    assert np.isfinite(plan.log_truncation_bound)
    assert plan.log_truncation_bound < np.log(np.nextafter(0., 1.))
    assert plan.truncation_bound == np.nextafter(0., 1.)
