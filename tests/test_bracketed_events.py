"""Residual, time-width and retained-bracket guarantees for event localization."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pyeph.integrators.events import bisect_bracketed_crossing


def assert_valid_bracket(root, function, duration, population_tolerance, time_tolerance):
    assert bool(root.converged)
    assert 0 <= float(root.lower) <= float(root.time) <= float(root.upper) <= duration
    assert float(root.upper-root.lower) <= time_tolerance
    assert abs(float(root.residual)) <= population_tolerance
    # An eager transcendental call and a fused compiled call can differ by a
    # few ulps even at the same retained endpoint.
    np.testing.assert_allclose(root.residual, function(root.time), rtol=0.,
                               atol=4*np.finfo(root.residual.dtype).eps
                               * (1+abs(float(root.residual))))
    assert float(function(root.lower)) >= 0 >= float(function(root.upper))


@pytest.mark.parametrize("shape", ["linear", "exponential", "cosine"])
def test_jitted_nonlinear_crossings_retain_a_bracket_and_meet_both_tolerances(shape):
    if shape == "linear":
        def function(time):
            return .314159-time
        exact, duration = .314159, 1.
    elif shape == "exponential":
        def function(time):
            return jnp.exp(.37)-jnp.exp(time)
        exact, duration = .37, 1.
    else:
        def function(time):
            return jnp.cos(time)-.4
        exact, duration = np.arccos(.4), 2.
    ptol, ttol = 1e-11, 2e-11
    root = jax.jit(lambda width: bisect_bracketed_crossing(
        function, width, population_tolerance=ptol, time_tolerance=ttol))(duration)
    assert_valid_bracket(root, function, duration, ptol, ttol)
    assert abs(float(root.time)-exact) <= ttol
    assert 0 < int(root.iterations) <= 48


def test_a_small_population_residual_does_not_replace_the_time_width_requirement():
    def function(time):
        return 1e-16*(.31-time)

    # Every point has an acceptable population residual from the start; only
    # retaining a sufficiently narrow sign bracket localizes this flat root.
    root = bisect_bracketed_crossing(function, 1., population_tolerance=1e-12,
                                     time_tolerance=1e-8)
    assert_valid_bracket(root, function, 1., 1e-12, 1e-8)
    assert int(root.iterations) >= 26
    assert abs(float(root.time)-.31) <= 1e-8


def test_a_small_time_width_does_not_replace_the_population_residual_requirement():
    def function(time):
        return 1e8*(.314159-time)

    root = bisect_bracketed_crossing(function, 1., population_tolerance=1e-5,
                                     time_tolerance=.1)
    assert_valid_bracket(root, function, 1., 1e-5, .1)
    assert int(root.iterations) > 30


@pytest.mark.parametrize("root_time", [.5, 1.])
def test_exact_midpoint_and_terminal_roots_collapse_the_bracket(root_time):
    def function(time):
        return root_time-time

    root = jax.jit(lambda: bisect_bracketed_crossing(function, 1.))()
    assert_valid_bracket(root, function, 1., 1e-10, 1e-10)
    np.testing.assert_array_equal(np.asarray([root.lower, root.time, root.upper]),
                                  np.full(3, root_time))
    assert float(root.residual) == 0.
    assert int(root.iterations) == (1 if root_time == .5 else 0)


def test_returned_endpoint_does_not_keep_a_better_residual_outside_the_final_bracket():
    def function(time):
        return (time+.001)*(.7-time)

    # The initial residual at t=0 is smaller than any of the next two sampled
    # values, but t=0 is excluded once the lower boundary moves to .5.
    root = bisect_bracketed_crossing(function, 1., iterations=2,
                                     population_tolerance=.04, time_tolerance=.3)
    assert_valid_bracket(root, function, 1., .04, .3)
    np.testing.assert_array_equal(np.asarray([root.lower, root.time, root.upper]), [.5, .75, .75])
    assert abs(float(function(0.))) < abs(float(root.residual))


def test_nonmonotone_multiple_root_bracket_does_not_claim_the_first_incoming_root():
    def function(time):
        return -(time-.2)*(time-.5)*(time-.8)

    root = bisect_bracketed_crossing(function, 1.)
    assert_valid_bracket(root, function, 1., 1e-10, 1e-10)
    assert float(root.time) == .5
    # This exact interior zero is outgoing. The localizer promises a root of
    # the sign bracket, not root uniqueness, first passage, or an incoming
    # derivative. Dynamics must check its event direction independently.
    assert float(jax.grad(function)(root.time)) > 0


@pytest.mark.parametrize("return_time", [.73, 1.])
def test_initial_outgoing_boundary_is_not_mistaken_for_later_return(return_time):
    def function(time):
        return time*(return_time-time)

    root = jax.jit(lambda: bisect_bracketed_crossing(function, 1.))()
    assert_valid_bracket(root, function, 1., 1e-10, 1e-10)
    assert abs(float(root.time)-return_time) <= 1e-10
    assert float(root.time) > .7


def test_initial_incoming_boundary_is_approached_without_inventing_a_later_root():
    def function(time):
        return -time

    root = bisect_bracketed_crossing(function, 1.)
    assert_valid_bracket(root, function, 1., 1e-10, 1e-10)
    assert float(root.time) == 0.
    assert int(root.iterations) > 0


def test_budget_exhaustion_reports_failure_with_a_retained_bracket():
    def function(time):
        return .314159-time

    root = bisect_bracketed_crossing(function, 1., iterations=2,
                                     population_tolerance=1e-12, time_tolerance=1e-12)
    assert not bool(root.converged)
    assert int(root.iterations) == 2
    assert 0 <= float(root.lower) <= float(root.time) <= float(root.upper) <= 1.
    np.testing.assert_equal(root.residual, function(root.time))
    assert float(root.upper-root.lower) > 1e-12


def test_sign_discontinuity_cannot_converge_only_by_narrowing_the_bracket():
    def function(time):
        return jnp.where(time < .314159, 1., -1.)

    root = bisect_bracketed_crossing(function, 1., iterations=48,
                                     population_tolerance=.01, time_tolerance=1e-8)
    assert not bool(root.converged)
    assert int(root.iterations) == 48
    assert float(root.upper-root.lower) <= 1e-8
    assert abs(float(root.residual)) == 1.


def test_float32_stagnation_returns_failure_before_excessive_iteration_budget():
    def function(time):
        return jnp.where(time < jnp.float32(.31), jnp.float32(1.), jnp.float32(-1.))

    root = jax.jit(lambda duration: bisect_bracketed_crossing(
        function, duration, iterations=200, population_tolerance=1e-12,
        time_tolerance=1e-12))(jnp.float32(1.))
    assert not bool(root.converged)
    assert int(root.iterations) < 100
    assert root.time.dtype == jnp.float32
    assert np.isfinite([root.lower, root.upper, root.time]).all()


@pytest.mark.parametrize("duration", [0., -1., np.inf, np.nan])
def test_invalid_numerical_durations_return_nonconvergence(duration):
    root = jax.jit(lambda width: bisect_bracketed_crossing(lambda time: .3-time, width))(duration)
    assert not bool(root.converged)
    assert int(root.iterations) == 0


@pytest.mark.parametrize("case", ["all_positive", "all_negative", "wrong_direction", "invalid_exact_end"])
def test_unbracketed_or_wrongly_directed_endpoints_do_not_converge(case):
    def function(time):
        if case == "all_positive":
            return 1.+time
        if case == "all_negative":
            return -1.-time
        if case == "wrong_direction":
            return time-.3
        return time-1.

    root = bisect_bracketed_crossing(function, 1.)
    assert not bool(root.converged)
    assert int(root.iterations) == 0


@pytest.mark.parametrize("location", [0., .5, 1.])
def test_nonfinite_sample_invalidates_localization(location):
    def function(time):
        return jnp.where(time == location, jnp.nan, .3-time)

    root = jax.jit(lambda: bisect_bracketed_crossing(function, 1.))()
    assert not bool(root.converged)
    assert int(root.iterations) == (1 if location == .5 else 0)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_real_duration_dtype_is_preserved_in_jit(dtype):
    duration = jnp.array(1., dtype=dtype)

    def function(time):
        return jnp.asarray(.37, dtype=dtype)-time

    tolerance = 1e-6 if dtype == jnp.float32 else 1e-11
    root = jax.jit(lambda width: bisect_bracketed_crossing(function, width,
        population_tolerance=tolerance, time_tolerance=tolerance))(duration)
    assert_valid_bracket(root, function, 1., tolerance, tolerance)
    assert all(value.dtype == dtype for value in (root.time, root.residual, root.lower, root.upper))
    assert root.iterations.dtype == jnp.int32
    assert root.converged.dtype == jnp.bool_


def test_integer_duration_is_promoted_to_floating_arithmetic():
    root = jax.jit(lambda duration: bisect_bracketed_crossing(
        lambda time: .3-time, duration))(jnp.int32(1))
    assert_valid_bracket(root, lambda time: .3-time, 1., 1e-10, 1e-10)
    assert jnp.issubdtype(root.time.dtype, jnp.floating)


def test_vmap_keeps_valid_and_unbracketed_trajectories_distinct():
    roots = jnp.array([.2, .37, 1.2])
    actual = jax.jit(jax.vmap(lambda location: bisect_bracketed_crossing(
        lambda time: location-time, 1.)))(roots)
    np.testing.assert_array_equal(actual.converged, [True, True, False])
    np.testing.assert_allclose(actual.time[:2], roots[:2], atol=1e-10, rtol=0.)
    assert int(actual.iterations[-1]) == 0
    assert np.all(np.asarray(actual.lower[:2]) <= np.asarray(actual.time[:2]))
    assert np.all(np.asarray(actual.time[:2]) <= np.asarray(actual.upper[:2]))


@pytest.mark.parametrize("iterations", [0, -1, 2.5, True, [5]])
def test_invalid_iteration_configuration_is_rejected(iterations):
    with pytest.raises(ValueError, match="iterations"):
        bisect_bracketed_crossing(lambda time: .3-time, 1., iterations=iterations)


@pytest.mark.parametrize("name", ["population_tolerance", "time_tolerance"])
@pytest.mark.parametrize("value", [0., -1., np.inf, np.nan, True, 1j, [1e-5]])
def test_invalid_tolerances_are_rejected_clearly(name, value):
    with pytest.raises(ValueError):
        bisect_bracketed_crossing(lambda time: .3-time, 1., **{name: value})


@pytest.mark.parametrize("duration", [jnp.ones(1), 1j, True])
def test_malformed_duration_is_rejected(duration):
    with pytest.raises(ValueError, match="real scalar"):
        bisect_bracketed_crossing(lambda time: .3-time, duration)


@pytest.mark.parametrize("kind", ["array", "complex", "boolean"])
def test_malformed_crossing_value_is_rejected(kind):
    def function(time):
        if kind == "array":
            return jnp.array([.3-time])
        if kind == "complex":
            return .3-time+0j
        return time < .3

    with pytest.raises(ValueError, match="real scalar"):
        bisect_bracketed_crossing(function, 1.)


def test_large_finite_positive_duration_does_not_overflow_midpoint():
    duration = 1e308

    def function(time):
        return 9e307-time

    root = jax.jit(lambda width: bisect_bracketed_crossing(function, width,
        population_tolerance=1e295, time_tolerance=1e297))(duration)
    assert_valid_bracket(root, function, duration, 1e295, 1e297)
    assert abs(float(root.time)/duration-.9) < 1e-12


def test_optional_both_endpoint_policy_refines_the_other_side_without_changing_default():
    def function(time):
        return .31-time

    options = dict(population_tolerance=.02, time_tolerance=.1)
    default = bisect_bracketed_crossing(function, 1., **options)
    strict = jax.jit(lambda: bisect_bracketed_crossing(
        function, 1., both_endpoints=True, **options))()
    assert_valid_bracket(default, function, 1., .02, .1)
    assert_valid_bracket(strict, function, 1., .02, .1)
    assert abs(float(function(default.lower))) > .02
    assert max(abs(float(function(strict.lower))), abs(float(function(strict.upper)))) <= .02
    assert int(strict.iterations) > int(default.iterations)
    np.testing.assert_array_equal(
        default, bisect_bracketed_crossing(function, 1., both_endpoints=False, **options))


def test_both_endpoint_policy_exhaustion_does_not_accept_only_the_better_endpoint():
    options = dict(iterations=4, population_tolerance=.02, time_tolerance=.1)
    default = bisect_bracketed_crossing(lambda t: .31-t, 1., **options)
    strict = bisect_bracketed_crossing(lambda t: .31-t, 1., both_endpoints=True, **options)
    assert bool(default.converged) and not bool(strict.converged)
    assert abs(float(strict.residual)) < .02
    assert int(strict.iterations) == 4
    assert float(strict.lower) <= float(strict.time) <= float(strict.upper)


@pytest.mark.parametrize("root_time", [.5, 1.])
def test_both_endpoint_policy_preserves_exact_collapsed_roots(root_time):
    root = bisect_bracketed_crossing(lambda t: root_time-t, 1., both_endpoints=True)
    assert bool(root.converged)
    np.testing.assert_array_equal([root.lower, root.time, root.upper], [root_time]*3)


def test_both_endpoint_policy_batches_valid_and_invalid_brackets():
    roots = jnp.array([.31, .37, 1.2])
    result = jax.jit(jax.vmap(lambda r: bisect_bracketed_crossing(
        lambda t: r-t, 1., population_tolerance=.02, time_tolerance=.1,
        both_endpoints=True)))(roots)
    np.testing.assert_array_equal(result.converged, [True, True, False])
    assert np.all(abs(np.asarray(roots[:2]-result.lower[:2])) <= .02)
    assert np.all(abs(np.asarray(roots[:2]-result.upper[:2])) <= .02)


@pytest.mark.parametrize("invalid", [1, 0, 1., None, [True], "yes"])
def test_both_endpoint_policy_requires_a_static_boolean(invalid):
    with pytest.raises(ValueError, match="both_endpoints"):
        bisect_bracketed_crossing(lambda t: .3-t, 1., both_endpoints=invalid)
