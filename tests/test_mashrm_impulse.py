"""Rank-two RM algebra against independent NAC sums and projector derivatives.

Complex-Hermitian fixtures test algebra, not a complex/SOC hopping method.
"""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pyeph.core.contracts import LowRankWeight, ModelSpec, SurfaceData
from pyeph.core.system import SystemSpec
from pyeph.models.base import AutoDiffModel
from pyeph.models.composite import ReferenceShiftModel
from pyeph.representations.adiabatic import model_surfaces
from pyeph.representations.mapping_rm import (
    RM_DEGENERATE,
    RM_INVALID_PAIR,
    RM_INVALID_SPECTRUM,
    RM_NONFINITE,
    RM_NONFINITE_GRADIENT,
    RM_SUCCESS,
    rm_impulse_direction,
    rm_impulse_weight,
)


class NonlinearHamiltonian(AutoDiffModel):
    def __init__(self, count, complex_valued=True):
        self.spec = ModelSpec(SystemSpec(count, (3,)), complex_valued=complex_valued)

    def apply(self, params, q, vectors):
        matrix = (params["constant"]
                  + (1+.2*params["gain"])*jnp.einsum("q,qij->ij", jnp.sin(q), params["linear"])
                  + jnp.einsum("q,qij->ij", q*q, params["curvature"]))
        return matrix @ vectors

    def reference_energy(self, params, q):
        return .1*jnp.sum(q**2)

    def reference_gradient(self, params, q):
        raise AssertionError("a population-projector gradient must not request reference forces")


def fixture(count=3, complex_valued=True):
    rng = np.random.default_rng(824+count)

    def random_matrix(shape):
        value = rng.normal(size=shape)
        return value+1j*rng.normal(size=shape) if complex_valued else value

    unitary, _ = np.linalg.qr(random_matrix((count, count)))
    energies = np.arange(count)*1.7-.5*count
    linear = .13*random_matrix((3, count, count))
    curvature = .06*random_matrix((3, count, count))
    params = dict(constant=(unitary*energies)@unitary.conj().T,
                  linear=(linear+linear.conj().transpose(0, 2, 1))/2,
                  curvature=(curvature+curvature.conj().transpose(0, 2, 1))/2,
                  gain=np.array(.37))
    q = np.array([-.2, .1, .3])
    c = rng.normal(size=count)+1j*rng.normal(size=count)
    c /= np.linalg.norm(c)
    return NonlinearHamiltonian(count, complex_valued), params, q, c


def numpy_model(params, q):
    """Explicit NumPy Hamiltonian and derivatives, independent of JAX AD."""
    params = {key: np.asarray(value) for key, value in params.items()}
    gain = 1+.2*params["gain"]
    h = params["constant"]+gain*np.einsum("q,qij->ij", np.sin(q), params["linear"])
    h += np.einsum("q,qij->ij", q*q, params["curvature"])
    dh = (gain*np.cos(q)[:, None, None]*params["linear"]
          + 2*q[:, None, None]*params["curvature"])
    return h, dh


def direct_nac_sum(params, q, c, active=0, competitor=1):
    h, dh = numpy_model(params, q)
    energies, vectors = np.linalg.eigh(h)
    cad = vectors.conj().T@c
    result = np.zeros(len(q))
    for j, gradient in enumerate(dh):
        for a, sign in ((active, 1), (competitor, -1)):
            for k in range(len(c)):
                if k != a:
                    nac = np.vdot(vectors[:, k], gradient@vectors[:, a])/(energies[a]-energies[k])
                    result[j] += sign*np.real(cad[k].conj()*nac*cad[a])
    return result


def population_margin(params, q, c, active=0, competitor=1):
    _, vectors = np.linalg.eigh(numpy_model(params, q)[0])
    population = np.abs(vectors.conj().T@c)**2
    return population[active]-population[competitor]


def projector_finite_difference(params, q, c, step=2e-5):
    # q varies while the fixed-diabatic c remains fixed. No eigenvector phase
    # alignment is needed because each evaluation is a projector population.
    return np.array([.5*(population_margin(params, q+dq, c)
                         - population_margin(params, q-dq, c))/(2*step)
                     for dq in step*np.eye(len(q))])


@pytest.mark.parametrize("count", [3, 5, 9])
@pytest.mark.parametrize("complex_valued", [False, True])
def test_rank_two_direction_matches_full_nac_sum_and_projector_finite_differences(count, complex_valued):
    model, params, q, c = fixture(count, complex_valued)
    data = model_surfaces(model, params, q)
    weights = rm_impulse_weight(data, c, 0, 1)
    assert bool(weights.valid)
    assert isinstance(weights.weight, LowRankWeight)
    assert weights.weight.left.shape == weights.weight.right.shape == (count, 2)
    np.testing.assert_allclose(jnp.vdot(weights.weight.left, weights.weight.right), 0., atol=2e-15)
    result = jax.jit(lambda p, x, z: rm_impulse_direction(
        model, p, x, model_surfaces(model, p, x), z, 0, 1))(params, q, c)
    expected = direct_nac_sum(params, q, c)
    assert bool(result.valid)
    np.testing.assert_allclose(result.value, expected, rtol=2e-13, atol=2e-15)
    np.testing.assert_allclose(result.value, projector_finite_difference(params, q, c),
                               rtol=2e-7, atol=2e-10)
    h, dh = numpy_model(params, q)
    _, u = np.linalg.eigh(h)
    np.testing.assert_allclose(result.populations, np.abs(u.conj().T@c)**2, atol=2e-15)
    # Directly contract the factors with each explicit derivative as another
    # check of the LowRankWeight conjugation/orientation convention.
    explicit = [np.vdot(weights.weight.left, gradient@weights.weight.right).real for gradient in dh]
    np.testing.assert_allclose(explicit, expected, atol=2e-15)


@pytest.mark.parametrize("count", [3, 5, 9])
def test_eigenphase_permutation_and_mapping_global_phase_covariance(count):
    model, params, q, c = fixture(count)
    data = model_surfaces(model, params, q)
    original = rm_impulse_direction(model, params, q, data, c, 0, 1)
    phases = jnp.exp(1j*jnp.linspace(-2.4, .7, count))
    permutation = np.random.default_rng(count).permutation(count)
    a, b = (int(np.flatnonzero(permutation == i)[0]) for i in (0, 1))
    changed = SurfaceData(data.energies[permutation], (data.vectors*phases)[:, permutation],
                          data.gaps[permutation][:, permutation],
                          data.near_degenerate[permutation][:, permutation])
    actual = jax.jit(lambda d, z, i, j: rm_impulse_direction(model, params, q, d, z, i, j))(
        changed, c*np.exp(.84j), a, b)
    assert bool(actual.valid)
    np.testing.assert_allclose(actual.value, original.value, atol=3e-15)
    np.testing.assert_allclose(actual.populations, original.populations[permutation], atol=3e-15)
    reverse = rm_impulse_direction(model, params, q, changed, c, b, a)
    np.testing.assert_allclose(reverse.value, -original.value, atol=3e-15)


@pytest.mark.parametrize("count", [3, 5, 9])
def test_scalar_reference_shift_cancels_without_norm_assumption(count):
    model, params, q, c = fixture(count)
    c = (2.1-.8j)*c  # Deliberately not normalized: no reference-force term enters delta.

    def shift(alpha, position):
        return alpha*(jnp.sin(position[0])+.3*jnp.sum(position**2))

    shifted = ReferenceShiftModel(model, shift)
    baseline = rm_impulse_direction(model, params, q, model_surfaces(model, params, q), c, 0, 1)
    shifted_params = (params, .72)
    actual = rm_impulse_direction(shifted, shifted_params, q,
        model_surfaces(shifted, shifted_params, q), c, 0, 1)
    assert bool(actual.valid)
    np.testing.assert_allclose(actual.value, baseline.value, atol=1e-14)
    np.testing.assert_allclose(actual.populations, baseline.populations, atol=1e-14)


def test_norm_homogeneity_zero_vector_and_population_boundary_rate():
    model, params, q, c = fixture(5)
    data = model_surfaces(model, params, q)
    original = rm_impulse_direction(model, params, q, data, c, 0, 1)
    for scale in (2-.4j, 0.):
        actual = rm_impulse_direction(model, params, q, data, scale*c, 0, 1)
        assert bool(actual.valid)
        np.testing.assert_allclose(actual.value, abs(scale)**2*original.value, atol=2e-15)
        np.testing.assert_allclose(actual.populations, abs(scale)**2*original.populations, atol=2e-15)
    velocity = np.array([.7, -.2, .3])
    # Electronic evolution contributes zero instantaneously because [h,P_a]=0.
    dc = -1j*(numpy_model(params, q)[0]+.19*np.eye(5))@c
    step = 1e-6
    finite_rate = (population_margin(params, q+step*velocity, c+step*dc)
                   - population_margin(params, q-step*velocity, c-step*dc))/(2*step)
    np.testing.assert_allclose(2*np.dot(original.value, velocity), finite_rate, atol=3e-10)


@pytest.mark.parametrize("field", ["q", "c", "gain"])
def test_outer_sensitivities_through_spectral_weight_are_not_detached(field):
    model, params, q, c = fixture()
    dq, dc = np.array([.17, -.31, .23]), np.array([.1+.2j, -.3j, .23-.1j])

    def arguments(t):
        return (dict(params, gain=params["gain"]+t if field == "gain" else params["gain"]),
                q+t*dq if field == "q" else q, c+t*dc if field == "c" else c)

    def calculated(t):
        p, x, z = arguments(t)
        return rm_impulse_direction(model, p, x, model_surfaces(model, p, x), z, 0, 1).value

    def expected(t):
        return direct_nac_sum(*arguments(t))

    step = 1e-5
    numerical = (expected(step)-expected(-step))/(2*step)
    np.testing.assert_allclose(jax.jacfwd(calculated)(0.), numerical, rtol=2e-6, atol=2e-10)
    np.testing.assert_allclose(jax.jacrev(calculated)(0.), numerical, rtol=2e-6, atol=2e-10)
    assert np.linalg.norm(numerical) > 1e-5


@pytest.mark.parametrize("fault", ["energies", "vectors", "gaps", "mapping", "coordinates"])
def test_nonfinite_numerical_inputs_return_explicit_compiled_status(fault):
    model, params, q, c = fixture()
    data = model_surfaces(model, params, q)
    if fault in ("energies", "vectors", "gaps"):
        value = getattr(data, fault)
        data = data._replace(**{fault: value.at[0].set(jnp.nan)})
    elif fault == "mapping":
        c[0] = np.nan
    else:
        q[0] = np.inf
    result = jax.jit(lambda d, x, z: rm_impulse_direction(model, params, x, d, z, 0, 1))(data, q, c)
    assert int(result.status) == RM_NONFINITE and not bool(result.valid)
    assert np.isnan(result.value).all()


@pytest.mark.parametrize("fault", ["nonorthogonal", "stale_gaps", "near_diagonal", "near_asymmetric"])
def test_inconsistent_complete_spectral_data_is_rejected(fault):
    model, params, q, c = fixture()
    data = model_surfaces(model, params, q)
    if fault == "nonorthogonal":
        data = data._replace(vectors=data.vectors.at[:, 0].multiply(1.1))
    elif fault == "stale_gaps":
        data = data._replace(gaps=-data.gaps)
    elif fault == "near_diagonal":
        data = data._replace(near_degenerate=data.near_degenerate.at[0, 0].set(True))
    else:
        data = data._replace(near_degenerate=data.near_degenerate.at[0, 1].set(True))
    result = jax.jit(rm_impulse_weight)(data, c, 0, 1)
    assert int(result.status) == RM_INVALID_SPECTRUM
    assert np.isnan(result.weight.left).all() and np.isnan(result.weight.right).all()


@pytest.mark.parametrize("fault", ["crossing_pair", "spectators", "near_gap", "declared_near"])
def test_all_spectrum_isolation_required_and_gaps_never_floored(fault):
    model, params, q, c = fixture(5)
    data = model_surfaces(model, params, q)
    if fault == "declared_near":
        data = data._replace(near_degenerate=data.near_degenerate.at[3, 4].set(True).at[4, 3].set(True))
    else:
        i, j = (0, 1) if fault != "spectators" else (3, 4)
        energies = data.energies.at[j].set(data.energies[i]+(1e-12 if fault == "near_gap" else 0.))
        data = data._replace(energies=energies, gaps=energies[None, :]-energies[:, None])
    result = jax.jit(lambda d: rm_impulse_direction(model, params, q, d, c, 0, 1))(data)
    assert int(result.status) == RM_DEGENERATE and not bool(result.valid)
    assert np.isnan(result.value).all()


@pytest.mark.parametrize("a,b", [(-1, 1), (0, 3), (2, 2), (4, -1)])
def test_invalid_runtime_state_indices_do_not_use_clipped_physics(a, b):
    model, params, q, c = fixture()
    data = model_surfaces(model, params, q)
    result = jax.jit(lambda i, j: rm_impulse_weight(data, c, i, j))(a, b)
    assert int(result.status) == RM_INVALID_PAIR
    assert np.isnan(result.weight.left).all()


def test_one_contraction_and_no_provider_execution_for_invalid_scalar_input():
    original, params, q, c = fixture()
    calls = []

    class CountedModel(NonlinearHamiltonian):
        def contract_gradient(self, params, q, weight):
            assert weight.left.shape == weight.right.shape == (3, 2)
            jax.debug.callback(lambda _: calls.append("gradient"), q, ordered=True)
            return super().contract_gradient(params, q, weight)

    model = CountedModel(3)
    data = model_surfaces(original, params, q)
    compiled = jax.jit(lambda d, z: rm_impulse_direction(model, params, q, d, z, 0, 1))
    assert bool(jax.block_until_ready(compiled(data, c)).valid)
    bad = data._replace(energies=data.energies.at[0].set(jnp.nan))
    assert not bool(jax.block_until_ready(compiled(bad, c)).valid)
    assert calls == ["gradient"]


def test_nonfinite_gradient_is_distinct_from_invalid_spectral_data():
    original, params, q, c = fixture()

    class InvalidGradient(NonlinearHamiltonian):
        def contract_gradient(self, params, q, weight):
            return jnp.full_like(q, jnp.inf)

    model = InvalidGradient(3)
    result = jax.jit(lambda: rm_impulse_direction(
        model, params, q, model_surfaces(original, params, q), c, 0, 1))()
    assert int(result.status) == RM_NONFINITE_GRADIENT
    assert np.isnan(result.value).all() and np.isfinite(result.populations).all()


def test_vmap_preserves_per_trajectory_status_and_one_vector_contract():
    model, params, q, c = fixture()
    data = model_surfaces(model, params, q)
    actual = jax.jit(jax.vmap(lambda z, i: rm_impulse_direction(
        model, params, q, data, z, i, 1)))(jnp.stack((c, 2*c, c)), jnp.array([0, 0, 3]))
    np.testing.assert_array_equal(actual.status, [RM_SUCCESS, RM_SUCCESS, RM_INVALID_PAIR])
    np.testing.assert_allclose(actual.value[1], 4*actual.value[0], atol=2e-15)
    assert np.isnan(actual.value[2]).all()


def test_truncated_spectrum_cannot_be_used_as_the_complete_model_space():
    model, params, q, c = fixture(5)
    data = model_surfaces(model, params, q)
    # An arbitrary low-energy window is not a complete-space RM contraction.
    truncated = SurfaceData(data.energies[:3], data.vectors[:, :3],
                            data.gaps[:3, :3], data.near_degenerate[:3, :3])
    with pytest.raises(ValueError, match="complete"):
        rm_impulse_direction(model, params, q, truncated, c, 0, 1)
    with pytest.raises(ValueError, match="complete"):
        rm_impulse_weight(truncated, c[:3], 0, 1)


@pytest.mark.parametrize("fault", ["block_c", "complex_energy", "mask_dtype", "float_index", "bool_index"])
def test_malformed_static_data_raises_before_numerical_use(fault):
    model, params, q, c = fixture()
    data = model_surfaces(model, params, q)
    a = 0
    if fault == "block_c":
        c = c[:, None]
    elif fault == "complex_energy":
        data = data._replace(energies=data.energies.astype(complex))
    elif fault == "mask_dtype":
        data = data._replace(near_degenerate=data.near_degenerate.astype(int))
    else:
        a = .0 if fault == "float_index" else False
    with pytest.raises(ValueError):
        rm_impulse_weight(data, c, a, 1)


@pytest.mark.parametrize("flag,value", [("native_jax", False), ("force_support", False),
                                         ("basis_kind", "moving_ao")])
def test_unsupported_model_capability_is_rejected(flag, value):
    model, params, q, c = fixture()
    data = model_surfaces(model, params, q)
    model.spec = replace(model.spec, **{flag: value})
    with pytest.raises(ValueError, match="native fixed-basis force-capable"):
        rm_impulse_direction(model, params, q, data, c, 0, 1)
