"""RM conditional measure, density convention, and prescribed-electronic limit."""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.linalg import expm

from pyeph.dynamics.mashrm_mapping import (
    mapping_density,
    mapping_observable,
    mapping_populations,
    rm_coefficients,
    sample_population_conditional,
)


def samples(nstates, population, count=32768):
    keys = jax.random.split(jax.random.PRNGKey(8100+nstates), count)
    draw = jax.jit(jax.vmap(lambda key: sample_population_conditional(
        key, nstates, population=population)))
    return np.asarray(draw(keys))


def max_simplex_cdf(x, nstates):
    """Inclusion-exclusion volume of a uniform simplex with all p_i <= x."""
    result = np.ones_like(x)
    for k in range(1, nstates+1):
        result += (-1)**k*math.comb(nstates, k)*np.maximum(1-k*x, 0.)**(nstates-1)
    return np.clip(result, 0., 1.)


def hoeffding_bound(width, count, *, comparisons, failure_probability=1e-7):
    """Union bound for sample means of bounded independent real variables."""
    return width*np.sqrt(np.log(2*comparisons/failure_probability)/(2*count))


@pytest.mark.parametrize("nstates,population", [(2, 0), (3, 1), (7, 6)])
def test_actual_conditional_sphere_measure_not_focused_magnitudes(nstates, population):
    c = samples(nstates, population)
    probabilities = np.abs(c)**2
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.,
                               atol=8*nstates*np.finfo(float).eps, rtol=0.)
    np.testing.assert_array_equal(probabilities.argmax(axis=1), population)
    # The target population equals the unrestricted simplex maximum in law.
    # A focused/moment-equivalent sampler fails this distribution test.
    ordered = np.sort(probabilities[:, population])
    exact = max_simplex_cdf(ordered, nstates)
    count = len(ordered)
    indices = np.arange(1, count+1)/count
    ks_distance = max(np.max(indices-exact), np.max(exact-(indices-1/count)))
    dkw = np.sqrt(np.log(2/1e-7)/(2*count))
    assert ks_distance < dkw
    harmonic = sum(1/k for k in range(1, nstates+1))
    expected = np.full(nstates, (nstates-harmonic)/(nstates*(nstates-1)))
    expected[population] = harmonic/nstates
    bound = hoeffding_bound(1., count, comparisons=nstates)
    np.testing.assert_allclose(probabilities.mean(axis=0), expected, rtol=0., atol=bound)
    alpha, b = rm_coefficients(nstates)
    density_mean = alpha*np.einsum("bi,bj->ij", c, c.conj())/count+b*np.eye(nstates)
    target_density = np.zeros((nstates, nstates))
    target_density[population, population] = 1.
    density_bound = hoeffding_bound(alpha, count, comparisons=2*nstates**2)
    assert np.max(np.abs((density_mean-target_density).real)) < density_bound
    assert np.max(np.abs((density_mean-target_density).imag)) < density_bound
    # For N=2, RM has a uniform hemisphere; original MASH2's |Sz|-weighted
    # hemisphere instead has E|Sz|=2/3. This difference is intentional.
    if nstates == 2:
        z = 2*probabilities[:, population]-1
        assert abs(z.mean()-.5) < hoeffding_bound(1., count, comparisons=1)


def test_coefficients_and_density_observable_conventions():
    assert rm_coefficients(2) == (2., -.5)
    np.testing.assert_allclose(rm_coefficients(3), (12/5, -7/15), rtol=0., atol=5e-16)
    c = jnp.array([1., 2j, -1.+.5j])
    c /= jnp.linalg.norm(c)
    rho = mapping_density(c)
    np.testing.assert_allclose(mapping_populations(c), jnp.diag(rho).real, atol=5e-16)
    np.testing.assert_allclose(jnp.trace(rho), 1., atol=5e-16)
    assert np.linalg.eigvalsh(np.asarray(rho)).min() < 0
    identity = jnp.eye(3)
    np.testing.assert_allclose(mapping_observable(c, identity@c, 3.), 1., atol=5e-16)
    for n in range(3):
        for m in range(3):
            operator = jnp.zeros((3, 3), dtype=complex).at[n, m].set(1.)
            value = mapping_observable(c, operator@c, jnp.trace(operator))
            np.testing.assert_allclose(value, rho[m, n], atol=5e-16)
    # Normalization is caller-owned: never conceal a bad mapping state.
    alpha, b = rm_coefficients(3)
    np.testing.assert_allclose(jnp.trace(mapping_density(2*c)), 4*alpha+3*b, atol=3e-15)


def test_operator_action_and_density_transform_covariantly_in_a_complex_basis():
    c = np.array([.2+.3j, -.5j, .7])
    c /= np.linalg.norm(c)
    generator = np.array([[.1, .2j, .3], [-.2j, -.5, .17j], [.3, -.17j, .7]])
    unitary = expm(-.8j*generator)
    operator = np.array([[1., 2j, .3], [-1j, -.2, 2.], [.1j, .5, .4]])
    rotated_c = unitary@c
    rotated_operator = unitary@operator@unitary.conj().T
    np.testing.assert_allclose(mapping_density(rotated_c), unitary@mapping_density(c)@unitary.conj().T,
                               atol=7e-16)
    np.testing.assert_allclose(mapping_observable(rotated_c, rotated_operator@rotated_c,
                                                np.trace(rotated_operator)),
                               mapping_observable(c, operator@c, np.trace(operator)), atol=7e-16)


def test_keys_jit_vmap_and_partition_identity():
    keys = jax.random.split(jax.random.PRNGKey(771), 11)

    def sample(key):
        return sample_population_conditional(key, 5, population=2)

    draw = jax.jit(jax.vmap(sample))
    whole = draw(keys)
    pieces = jnp.concatenate([draw(keys[:4]), draw(keys[4:])])
    np.testing.assert_array_equal(whole, pieces)
    np.testing.assert_array_equal(whole, draw(keys))
    np.testing.assert_allclose(whole[0], jax.jit(sample)(keys[0]), rtol=0., atol=5e-16)
    assert whole.shape == (11, 5) and whole.dtype == jnp.complex128
    typed = jax.random.key(25)
    assert sample_population_conditional(typed, 3).shape == (3,)
    np.testing.assert_allclose(jax.jit(mapping_density)(whole[0]), mapping_density(whole[0]), atol=5e-16)
    np.testing.assert_allclose(jax.vmap(mapping_populations)(whole).sum(axis=1), 1., atol=1e-15)


def test_population_preparation_density_moment_and_noncommuting_unitary_path():
    nstates, prepared = 3, 1
    c0 = samples(nstates, prepared, count=65536)
    alpha, b = rm_coefficients(nstates)
    physical_initial = np.eye(nstates)[:, prepared]
    # Independently generated real-symmetric Hamiltonians are noncommuting.
    h1 = np.array([[.2, .3, -.17], [.3, -.4, .12], [-.17, .12, .6]])
    h2 = np.array([[-.1, -.24, .41], [-.24, .7, .28], [.41, .28, -.3]])
    assert np.linalg.norm(h1@h2-h2@h1) > .2
    steps = [(h1, .7), (h2, 1.1), (h1, -.4), (h2, .9)]
    cumulative = np.eye(nstates, dtype=complex)
    mean_initial = alpha*np.einsum("bi,bj->ij", c0, c0.conj())/len(c0)+b*np.eye(nstates)
    # Each real/imag rho element has range width at most alpha for unit c.
    # One union bound covers all entries and all (correlated) output times.
    bound = hoeffding_bound(alpha, len(c0), comparisons=2*nstates**2*(len(steps)+1))
    for h, dt in [(h1, 0.), *steps]:
        cumulative = expm(-1j*h*dt)@cumulative
        c = c0@cumulative.T
        sampled_density = np.asarray(jax.jit(jax.vmap(mapping_density))(jnp.asarray(c))).mean(axis=0)
        physical = cumulative@physical_initial
        exact = np.outer(physical, physical.conj())
        assert np.max(np.abs((sampled_density-exact).real)) < bound
        assert np.max(np.abs((sampled_density-exact).imag)) < bound
        # Separate deterministic algebra from finite Monte Carlo uncertainty.
        np.testing.assert_allclose(sampled_density, cumulative@mean_initial@cumulative.conj().T,
                                   atol=2e-14)
        populations = np.asarray(jax.vmap(mapping_populations)(jnp.asarray(c))).mean(axis=0)
        np.testing.assert_allclose(populations, np.diag(exact).real, rtol=0., atol=bound)
        coherence = np.zeros((3, 3), dtype=complex)
        coherence[0, 2] = 1.
        observed = np.asarray(jax.vmap(mapping_observable)(jnp.asarray(c),
            jnp.asarray(c@coherence.T), jnp.zeros(len(c)))).mean()
        assert abs((observed-exact[2, 0]).real) < bound
        assert abs((observed-exact[2, 0]).imag) < bound
    assert abs(exact[0, 2].imag) > .01  # The complex-coherence convention is exercised.


@pytest.mark.parametrize("nstates", [1, 0, -3, 3., True, [3]])
def test_invalid_static_state_counts(nstates):
    with pytest.raises(ValueError):
        rm_coefficients(nstates)
    with pytest.raises(ValueError):
        sample_population_conditional(jax.random.PRNGKey(1), nstates)


@pytest.mark.parametrize("population", [-1, 3, 1., True, [1]])
def test_invalid_population_index(population):
    with pytest.raises(ValueError):
        sample_population_conditional(jax.random.PRNGKey(1), 3, population=population)


def test_vector_only_shape_contracts():
    for bad in (1., jnp.ones((3, 1)), jnp.ones((4, 3)), jnp.ones((1,))):
        with pytest.raises(ValueError, match="one vector"):
            mapping_density(bad)
        with pytest.raises(ValueError, match="one vector"):
            mapping_populations(bad)
    with pytest.raises(ValueError, match="operator action"):
        mapping_observable(jnp.ones(3), jnp.ones((3, 1)), 0.)
    with pytest.raises(ValueError, match="trace must be scalar"):
        mapping_observable(jnp.ones(3), jnp.ones(3), jnp.zeros(1))
