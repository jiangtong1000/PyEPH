"""Canonical preparation against independent quadrature, not a phonon marginal."""

from dataclasses import FrozenInstanceError

import jax
import numpy as np
import pytest
from numpy.polynomial.hermite import hermgauss
from scipy.integrate import quad_vec
from scipy.special import logsumexp

from pyeph import CoupledClassical, Integrator, MASHRM, MASHRMPopulation, Problem, Simulation
from pyeph.models.epc import LinearEPCModel
from pyeph.workflows.mashrm_equilibrium import (
    CanonicalSamplingError,
    LinearEPCCanonical,
)


def fixture(d=1):
    if d == 1:
        model = LinearEPCModel(2, 1)
        params = model.create_params([[.15, .23], [.23, -.25]],
                                     [[[.5, 0.], [0., -.3]]], [.8], [.35])
    else:
        model = LinearEPCModel(3, 2)
        params = model.create_params(
            [[-.35, .16, .02], [.16, .15, .08], [.02, .08, .65]],
            [[[.2, .12, 0], [.12, -.1, .04], [0, .04, .08]],
             [[.04, .02, .03], [.02, .12, -.09], [.03, -.09, -.18]]],
            [.9, 1.2], [.3, -.2])
    return model, {name: np.array(value) for name, value in params.items()}


def density(c):
    """Independent RM constants and cc† orientation."""
    n = c.shape[-1]
    harmonic = sum(1/k for k in range(1, n+1))
    alpha = (n-1)/(harmonic-1)
    return alpha*c[..., :, None]*c.conj()[..., None, :]+(1-alpha)/n*np.eye(n)


def assert_sample_mean(values, expected, *, sigma=6., absolute=1e-11):
    values = np.asarray(values)
    expected = np.asarray(expected)
    error = abs(values.mean(axis=0)-expected)
    bound = sigma*values.std(axis=0, ddof=1)/np.sqrt(len(values))+absolute
    assert np.all(error < bound), (error, bound)


def quadrature_1d(params, beta, *, eps):
    """Analytic Pauli exponential and adaptive real-space integration.

    No production model, eigensolver, sampler, or mapping helper is called.
    """
    h0, g = params["h0"], params["coupling"][0]
    omega, center = params["omega"][0], params["q_eq"][0]

    def integrand(q):
        h = h0+q*g
        scalar, z, x = .5*np.trace(h), .5*(h[0, 0]-h[1, 1]), h[0, 1]
        radius = np.hypot(z, x)
        energies = np.array([scalar-radius, scalar+radius])
        logz = logsumexp(-beta*energies)
        weight = np.exp(-.5*beta*(omega*(q-center))**2+logz)
        rho = .5*(np.eye(2)-np.tanh(beta*radius)/radius*np.array([[z, x], [x, -z]]))
        active = np.exp(-beta*energies-logz)
        return weight*np.r_[1., q, q*q, active, rho.ravel()]

    value, error = quad_vec(integrand, -np.inf, np.inf, epsabs=eps, epsrel=eps)
    return value/value[0], error


def quadrature_2d(params, beta, order):
    """Tensor Gaussian quadrature in the bare reference measure, reweighted."""
    nodes, weights = hermgauss(order)
    x, y = np.meshgrid(nodes, nodes, indexing="ij")
    q = params["q_eq"]+np.sqrt(2)*np.stack([x.ravel(), y.ravel()], axis=-1)/(
        np.sqrt(beta)*params["omega"])
    weight = (weights[:, None]*weights[None, :]/np.pi).ravel()
    h = params["h0"]+np.einsum("ba,aij->bij", q, params["coupling"])
    e, u = np.linalg.eigh(h)
    boltz = np.exp(-beta*e)
    normalized = weight*boltz.sum(axis=1)
    normalized /= normalized.sum()
    sector = boltz/boltz.sum(axis=1, keepdims=True)
    rho = np.einsum("bin,bn,bjn->bij", u, sector, u)
    return dict(q=normalized@q, qq=np.einsum("b,bi,bj->ij", normalized, q, q),
                active=normalized@sector, rho=np.einsum("b,bij->ij", normalized, rho))


def test_coupled_1d_distribution_mapping_sectors_density_and_momenta():
    model, params = fixture()
    beta, mass = 2.1, 1.7
    exact, _ = quadrature_1d(params, beta, eps=2e-11)
    tighter, _ = quadrature_1d(params, beta, eps=2e-13)
    np.testing.assert_allclose(exact, tighter, atol=3e-12, rtol=0.)
    result = LinearEPCCanonical(model, params, mass, beta).sample(np.arange(16384), seed=114)
    state = result.state
    q, p, c = map(np.asarray, (state.q, state.p, state.electronic))
    active = np.asarray(state.method_state["active"])
    assert_sample_mean(q[:, 0], exact[1])
    assert_sample_mean(q[:, 0]**2, exact[2])
    assert_sample_mean(np.eye(2)[active], exact[3:5])
    rho = density(c)
    assert_sample_mean(rho.real, exact[5:].reshape(2, 2))
    assert_sample_mean(rho.imag, np.zeros((2, 2)))
    assert_sample_mean(p[:, 0], 0.)
    assert_sample_mean(p[:, 0]**2, mass/beta)
    assert_sample_mean((q[:, 0]-exact[1])*p[:, 0], 0.)
    # This coupling shifts the actual coordinate mean substantially; a bare
    # reference Gaussian plus a conditional electronic draw would fail.
    assert abs(exact[1]-params["q_eq"][0]) > .1
    h = params["h0"]+q[:, :, None]*params["coupling"][0]
    _, u = np.linalg.eigh(h)
    cad = np.einsum("bin,bi->bn", u, c)
    np.testing.assert_array_equal(np.argmax(abs(cad)**2, axis=1), active)
    # For two states, the largest population of a conditional uniform complex
    # sphere has CDF 2*x-1 on [1/2,1]. A focused sampler fails this measure test.
    maximum = np.sort(np.max(abs(cad)**2, axis=1))
    indices = np.arange(1, len(maximum)+1)/len(maximum)
    cdf = 2*maximum-1
    distance = max(np.max(indices-cdf), np.max(cdf-(indices-1/len(maximum))))
    assert distance < np.sqrt(np.log(2/1e-7)/(2*len(maximum)))
    np.testing.assert_allclose(np.sum(abs(c)**2, axis=1), 1., atol=3e-15, rtol=0.)
    np.testing.assert_array_equal(result.status, 0)
    assert np.max(result.attempts) < 10000
    assert np.any(np.asarray(result.attempts) > 1)


def test_noncommuting_2d_distribution_matches_converged_quadrature():
    model, params = fixture(2)
    beta, masses = 1.7, np.array([1.2, 2.3])
    # Ordered individual eigenvalues are less smooth near small gaps than the
    # full matrix exponential, so sector probabilities need a finer grid.
    exact, refined = quadrature_2d(params, beta, 192), quadrature_2d(params, beta, 256)
    for key in exact:
        np.testing.assert_allclose(exact[key], refined[key],
                                   atol=3e-8 if key == "active" else 2e-12, rtol=0.)
    result = LinearEPCCanonical(model, params, masses, beta).sample(np.arange(12288), seed=872)
    state = result.state
    q, p, c = map(np.asarray, (state.q, state.p, state.electronic))
    active = np.asarray(state.method_state["active"])
    assert_sample_mean(q, refined["q"])
    assert_sample_mean(q[:, :, None]*q[:, None, :], refined["qq"])
    assert_sample_mean(np.eye(3)[active], refined["active"], absolute=3e-8)
    assert_sample_mean(density(c).real, refined["rho"])
    assert_sample_mean(density(c).imag, np.zeros((3, 3)))
    assert_sample_mean(p, [0., 0.])
    assert_sample_mean(p*p, masses/beta)
    assert_sample_mean(p[:, 0]*p[:, 1], 0.)
    # Direct quadrature of E_proposal[acceptance] independently checks the
    # otherwise easy-to-miss Gaussian normalization in the envelope.
    nodes, weights = hermgauss(64)
    x, y = np.meshgrid(nodes, nodes, indexing="ij")
    z = np.stack([x.ravel(), y.ravel()], axis=-1)*np.sqrt(2)
    kappa = result.metadata["kappa"]
    q_proposal = params["q_eq"]+z/(np.sqrt(beta*kappa)*params["omega"])
    e = np.linalg.eigvalsh(params["h0"]+np.einsum("ba,aij->bij", q_proposal,
                                                               params["coupling"]))
    estar = np.linalg.eigvalsh(params["h0"]+np.einsum("a,aij->ij", params["q_eq"],
                                                     params["coupling"]))
    loga = (logsumexp(-beta*e, axis=1)-logsumexp(-beta*estar)
            -(1-kappa)/(2*kappa)*np.sum(z*z, axis=1)-result.metadata["envelope_penalty"])
    expected_acceptance = np.sum(
        (weights[:, None]*weights[None, :]/np.pi).ravel()*np.exp(loga))
    # First attempts are independent Bernoulli acceptances, unlike 1/mean(T).
    assert_sample_mean(np.asarray(result.attempts) == 1, expected_acceptance)


def test_uncoupled_limit_first_proposal_exact_and_public_rm_compatible():
    model = LinearEPCModel(3, 2)
    params = model.create_params(np.diag([-.4, .1, .7]), np.zeros((2, 3, 3)),
                                 [.8, 1.2], [.3, -.1])
    sampler = LinearEPCCanonical(model, params, [1., 2.], 1.3)
    result = sampler.sample(np.arange(64), seed=123)
    assert sampler.kappa == 1.
    np.testing.assert_array_equal(result.attempts, 1)
    np.testing.assert_array_equal(result.log_acceptance, 0.)
    problem = Problem(model, sampler.params, CoupledClassical(sampler.masses), MASHRM(),
                      MASHRMPopulation())
    simulation = Simulation(problem, Integrator(.001, "exponential_midpoint"))
    initial = jax.tree.map(lambda a: a[:2], result.state)
    output = simulation.run(initial, 0)
    np.testing.assert_array_equal(output.final_state.q, initial.q)


def assert_same_state(left, right):
    for a, b in zip(jax.tree.leaves(left), jax.tree.leaves(right), strict=True):
        np.testing.assert_array_equal(a, b)


def test_ids_partition_order_seed_budget_and_reserved_keys():
    model, params = fixture(2)
    ids = np.array([81, 5, 900, 22, 6, 1, 4], dtype=np.uint32)
    sampler = LinearEPCCanonical(model, params, [1., 2.], 1.7, max_trials=300)
    whole = sampler.sample(ids, seed=12)
    pieces = [sampler.sample(ids[:3], seed=12), sampler.sample(ids[3:], seed=12)]
    merged = jax.tree.map(lambda *x: np.concatenate(x), *(x.state for x in pieces))
    assert_same_state(whole.state, merged)
    reverse = sampler.sample(ids[::-1], seed=12)
    assert_same_state(whole.state, jax.tree.map(lambda x: x[::-1], reverse.state))
    longer = LinearEPCCanonical(model, params, [1., 2.], 1.7, max_trials=500).sample(ids, seed=12)
    assert_same_state(whole.state, longer.state)
    np.testing.assert_array_equal(whole.attempts, longer.attempts)
    assert whole.metadata["preparation_id"] == pieces[0].metadata["preparation_id"]
    assert not np.array_equal(whole.state.q, sampler.sample(ids, seed=13).state.q)
    assert len(np.unique(np.asarray(whole.state.key), axis=0)) == len(ids)
    assert whole.state.key.shape == (len(ids), 2)


@pytest.mark.parametrize("flag", ["jax_threefry_partitionable", "jax_high_dynamic_range_gumbel"])
def test_rng_configuration_cannot_change_under_cached_sampler_or_preparation_identity(flag):
    model, params = fixture()
    sampler = LinearEPCCanonical(model, params, 1., 2.)
    before = sampler.sample([1, 5], seed=12)
    previous = getattr(jax.config, flag)
    try:
        jax.config.update(flag, not previous)
        with pytest.raises(ValueError, match="configuration changed"):
            sampler.sample([1, 5], seed=12)
        changed = LinearEPCCanonical(model, params, 1., 2.).sample([1, 5], seed=12)
        assert before.metadata["preparation_id"] != changed.metadata["preparation_id"]
        if flag == "jax_threefry_partitionable":
            assert not np.array_equal(before.state.q, changed.state.q)
    finally:
        jax.config.update(flag, previous)
    assert_same_state(before.state, sampler.sample([1, 5], seed=12).state)


def test_active_categorical_receives_common_energy_shift_removed(monkeypatch):
    original = jax.random.categorical
    recorded = []

    def sample(key, logits):
        jax.debug.callback(lambda value: recorded.append(np.asarray(value)), logits)
        return original(key, logits)

    monkeypatch.setattr(jax.random, "categorical", sample)
    model, params = fixture()
    LinearEPCCanonical(model, params, 1., 2.).sample([1, 5, 13], seed=12)
    assert len(recorded) == 3
    for logits in recorded:
        assert logits[0] == 0.
        assert np.all(logits <= 0.)


def test_owned_params_masses_and_scalar_configuration_prevent_stale_bound():
    model, params = fixture(2)
    masses, beta, kappa = np.array([1., 2.]), np.array(1.7), np.array(.7)
    sampler = LinearEPCCanonical(model, params, masses, beta, kappa=kappa)
    before = sampler.sample([2, 3], seed=99)
    for value in params.values():
        value[...] = 99.
    masses[:] = 100.
    beta[...] = 12.
    kappa[...] = .1
    external = sampler.params
    external["h0"] = np.ones((3, 3))
    external.clear()
    after = sampler.sample([2, 3], seed=99)
    assert_same_state(before.state, after.state)
    assert before.metadata == after.metadata
    assert sampler.beta == 1.7 and sampler.kappa == .7
    with pytest.raises(FrozenInstanceError):
        sampler.beta = 3.
    with pytest.raises(TypeError):
        sampler.params["h0"][0, 0] = 0.


def test_preparation_metadata_matches_samples_is_fresh_and_needs_no_randomness(monkeypatch):
    model, params = fixture()
    sampler = LinearEPCCanonical(model, params, 1., 2.)
    result = sampler.sample([1, 5], seed=18)

    def forbidden(*args, **kwargs):
        raise AssertionError("metadata inspection must not generate a random key")

    monkeypatch.setattr(jax.random, "key", forbidden)
    metadata = sampler.preparation_metadata(seed=18)
    assert metadata == result.metadata
    metadata["seed"] = 12
    metadata["jax_configuration"].clear()
    assert sampler.preparation_metadata(seed=18) == result.metadata
    assert sampler.preparation_metadata(seed=19)["preparation_id"] != result.metadata["preparation_id"]
    with pytest.raises(ValueError, match="seed"):
        sampler.preparation_metadata(seed=-1)


def test_reference_offset_and_common_electronic_shift_cancel_from_sampling():
    model, params = fixture()
    changed = dict(params, reference_offset=np.array(12345.))
    first = LinearEPCCanonical(model, params, 1., 2.).sample(np.arange(32), seed=9)
    second = LinearEPCCanonical(model, changed, 1., 2.).sample(np.arange(32), seed=9)
    assert_same_state(first.state, second.state)
    shifted = dict(params, h0=params["h0"]+16*np.eye(2), reference_offset=np.array(-16.))
    third = LinearEPCCanonical(model, shifted, 1., 2.).sample(np.arange(32), seed=9)
    np.testing.assert_array_equal(first.attempts, third.attempts)
    np.testing.assert_array_equal(first.state.q, third.state.q)
    np.testing.assert_allclose(first.state.electronic, third.state.electronic, atol=2e-14, rtol=0.)


def test_capacity_failure_keeps_all_ids_and_successful_rows_are_budget_independent():
    model, params = fixture()
    ids = np.arange(32)
    with pytest.raises(CanonicalSamplingError) as caught:
        LinearEPCCanonical(model, params, 1., 2., kappa=.04, max_trials=1).sample(ids, seed=72)
    result = caught.value.result
    np.testing.assert_array_equal(result.state.trajectory_id, ids)
    np.testing.assert_array_equal(result.attempts, 1)
    assert set(np.asarray(result.status)) == {0, 1}
    good = np.asarray(result.status) == 0
    assert np.isfinite(np.asarray(result.state.q)).all()
    assert np.isnan(np.asarray(result.state.electronic)[~good]).all()
    assert np.all(np.asarray(result.state.method_state["status"])[~good] == 1)
    completed = LinearEPCCanonical(model, params, 1., 2., kappa=.04, max_trials=300).sample(ids, seed=72)
    assert_same_state(jax.tree.map(lambda x: x[good], result.state),
                      jax.tree.map(lambda x: x[good], completed.state))


def test_accepted_nonisolated_spectrum_fails_without_redrawing_or_conditioning():
    model = LinearEPCModel(3, 1)
    params = model.create_params(np.diag([0., .4, .4+2e-8]), np.zeros((1, 3, 3)))
    with pytest.raises(CanonicalSamplingError) as caught:
        LinearEPCCanonical(model, params, 1., 1., gap_tolerance=1e-7).sample([10, 5])
    result = caught.value.result
    np.testing.assert_array_equal(result.status, 4)
    np.testing.assert_array_equal(result.attempts, 1)
    np.testing.assert_array_equal(result.log_acceptance, 0.)
    compatible = LinearEPCCanonical(model, params, 1., 1., gap_tolerance=1e-9).sample([10, 5])
    np.testing.assert_array_equal(result.state.q, compatible.state.q)


def test_nonfinite_finish_is_explicit_after_accepted_gaussian():
    model = LinearEPCModel(2, 1)
    params = model.create_params(np.diag([1e308, 1.00001e308]), np.zeros((1, 2, 2)))
    params["reference_offset"] = 1e308
    with pytest.raises(CanonicalSamplingError) as caught:
        LinearEPCCanonical(model, params, 1., 1e-305).sample([10, 5])
    np.testing.assert_array_equal(caught.value.result.status, 2)
    np.testing.assert_array_equal(caught.value.result.attempts, 1)


def test_envelope_violation_is_detected_not_silently_clamped(monkeypatch):
    import pyeph.workflows.mashrm_equilibrium as module

    original = module.logsumexp
    monkeypatch.setattr(module, "logsumexp", lambda x: original(x)-100.)
    model, params = fixture()
    with pytest.raises(CanonicalSamplingError) as caught:
        LinearEPCCanonical(model, params, 1., 2.).sample([0, 1, 3])
    np.testing.assert_array_equal(caught.value.result.status, 3)
    np.testing.assert_array_equal(caught.value.result.attempts, 1)
    assert np.all(np.asarray(caught.value.result.log_acceptance) > 0)


@pytest.mark.parametrize("name,value", [
    ("beta", 0), ("beta", np.inf), ("beta", True), ("beta", [2.]),
    ("kappa", 0), ("kappa", 1), ("kappa", 1.1), ("kappa", 1j),
    ("max_trials", 0), ("max_trials", 2**31), ("max_trials", 1.),
    ("gap_tolerance", 0), ("gap_tolerance", np.nan),
])
def test_invalid_scalar_configuration(name, value):
    model, params = fixture()
    options = dict(beta=2., kappa=None, max_trials=10, gap_tolerance=1e-10)
    options[name] = value
    with pytest.raises(ValueError):
        LinearEPCCanonical(model, params, 1., **options)


@pytest.mark.parametrize("name,value", [
    ("omega", [0.]), ("omega", [-1.]), ("omega", [np.inf]), ("omega", [1e200]),
    ("omega", [1e-200]), ("q_eq", [np.nan]), ("q_eq", [1j]),
    ("reference_offset", np.inf), ("reference_offset", 1j),
    ("h0", [[0., 1e-13], [0., .2]]),
    ("coupling", [[[np.nan, 0.], [0., 0.]]]),
])
def test_invalid_parameter_arrays(name, value):
    model, params = fixture()
    params[name] = value
    with pytest.raises(ValueError):
        LinearEPCCanonical(model, params, 1., 2.)


@pytest.mark.parametrize("mass", [0., -1., np.nan, 1j, [1., 2.], True])
def test_invalid_masses(mass):
    model, params = fixture()
    with pytest.raises(ValueError):
        LinearEPCCanonical(model, params, mass, 2.)


@pytest.mark.parametrize("ids", [[], [1, 1], [-1], [2**32], [1.], [True], [[1]]])
def test_invalid_ids(ids):
    model, params = fixture()
    with pytest.raises(ValueError):
        LinearEPCCanonical(model, params, 1., 2.).sample(ids)


@pytest.mark.parametrize("seed", [-1, 2**32, 1., True])
def test_invalid_seeds(seed):
    model, params = fixture()
    with pytest.raises(ValueError):
        LinearEPCCanonical(model, params, 1., 2.).sample([1], seed=seed)


def test_subclasses_complex_models_and_disabled_x64_rejected():
    class DifferentPhysics(LinearEPCModel):
        pass

    with pytest.raises(ValueError, match="exactly real"):
        LinearEPCCanonical(DifferentPhysics(2, 1), None, 1., 1.)
    with pytest.raises(ValueError, match="exactly real"):
        LinearEPCCanonical(LinearEPCModel(2, 1, complex_valued=True), None, 1., 1.)
    model, params = fixture()
    previous = jax.config.x64_enabled
    try:
        jax.config.update("jax_enable_x64", False)
        with pytest.raises(ValueError, match="x64"):
            LinearEPCCanonical(model, params, 1., 1.)
    finally:
        jax.config.update("jax_enable_x64", previous)


def test_tiny_nonzero_coupling_does_not_take_uncoupled_branch():
    model, params = fixture()
    params["coupling"] *= 1e-20
    sampler = LinearEPCCanonical(model, params, 1., 1.)
    assert sampler.kappa < 1.
    assert sampler.kappa == np.nextafter(1., 0.)
    assert not sampler.sample([1]).metadata["zero_coupling"]
    params["coupling"] *= 1e-200
    with pytest.raises(ValueError, match="envelope"):
        LinearEPCCanonical(model, params, 1., 1.)
