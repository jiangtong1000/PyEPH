"""Identity streams retain legacy bytes and do not inherit ambient RBG semantics."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pyeph import CoupledClassical, Execution, Integrator, Problem, Simulation, make_state
from pyeph.dynamics.mash2 import (
    MASH2,
    MASHPopulation,
    mapping_state as mash2_state,
    sample_adiabatic_population,
    sample_population_spin,
)
from pyeph.dynamics.mashrm import (
    MASHRM,
    MASHRMPopulation,
    mapping_state as mashrm_state,
    sample_population,
)
from pyeph.dynamics.mashrm_mapping import sample_population_conditional
from pyeph.execution.random import as_threefry_key, event_key, trajectory_keys
from pyeph.initialization import sample_harmonic
from pyeph.models.epc import LinearEPCModel


def legacy_keys(seed, ids):
    """The pre-change raw-key formula under its original default implementation."""
    with jax.default_prng_impl("threefry2x32"):
        return jnp.stack([jax.random.fold_in(jax.random.PRNGKey(seed), int(i)) for i in ids])


@pytest.mark.parametrize("ambient", ["threefry2x32", "rbg", "unsafe_rbg"])
def test_trajectory_state_keys_keep_legacy_bytes_and_two_word_schema(ambient):
    ids = np.array([7, 0, 2**32-1], dtype=np.uint32)
    expected = legacy_keys(129, ids)
    with jax.default_prng_impl(ambient):
        raw = trajectory_keys(129, ids)
        typed = trajectory_keys(129, ids, typed=True)
        np.testing.assert_array_equal(raw, expected)
        np.testing.assert_array_equal(jax.random.key_data(typed), expected)
        assert raw.shape == (3, 2) and raw.dtype == jnp.uint32
        for index, identity in enumerate(ids):
            state = make_state([.1], [.2], [1., 0.], seed=129, trajectory_id=int(identity))
            assert state.key.shape == (2,) and state.key.dtype == jnp.uint32
            np.testing.assert_array_equal(state.key, expected[index])
        np.testing.assert_array_equal(jax.random.key_data(as_threefry_key(raw)), expected)
        np.testing.assert_array_equal(jax.random.key_data(as_threefry_key(typed)), expected)


@pytest.mark.parametrize("distribution", ["classical", "wigner"])
@pytest.mark.parametrize("ambient", ["rbg", "unsafe_rbg"])
def test_harmonic_samples_keep_legacy_bytes_across_partition_and_permutation(ambient, distribution):
    ids = np.array([9, 71, 4, 25, 13])
    w, m, temperature = jnp.array([.8, 1.7]), jnp.array([1.2, 2.3]), .4
    with jax.default_prng_impl("threefry2x32"):
        keys = legacy_keys(127, ids)
        if distribution == "classical":
            qvar, pvar = temperature/(m*w**2), m*temperature
        else:
            occupation = 1/jnp.tanh(w/(2*temperature))
            qvar, pvar = occupation/(2*m*w), occupation*m*w/2

        def old_sample(key):
            qkey, pkey = jax.random.split(key)
            return (jax.random.normal(qkey, w.shape, dtype=w.dtype)*jnp.sqrt(qvar),
                    jax.random.normal(pkey, w.shape, dtype=w.dtype)*jnp.sqrt(pvar))

        expected = jax.vmap(old_sample)(keys)
    with jax.default_prng_impl(ambient):
        full = sample_harmonic(w, m, temperature, ids, seed=127, distribution=distribution)
        part = sample_harmonic(w, m, temperature, ids[2:], seed=127, distribution=distribution)
        order = np.array([2, 4, 0, 3, 1])
        permuted = sample_harmonic(w, m, temperature, ids[order], seed=127,
                                  distribution=distribution)
        for actual, original, subset, reordered in zip(full, expected, part, permuted, strict=True):
            np.testing.assert_array_equal(actual, original)
            np.testing.assert_array_equal(subset, actual[2:])
            np.testing.assert_array_equal(reordered, actual[order])


@pytest.mark.parametrize("ambient", ["rbg", "unsafe_rbg"])
def test_event_keys_keep_raw_bytes_and_typed_draws_across_uint32_step_boundary(ambient):
    key = legacy_keys(39, [81])[0]
    step = jnp.int64(2**32+5)
    with jax.default_prng_impl("threefry2x32"):
        expected = jax.random.fold_in(jax.random.fold_in(jax.random.fold_in(key, 1), 5), 7)
        expected_draw = jax.random.normal(expected, (6,))
    with jax.default_prng_impl(ambient):
        raw = event_key(key, step, 7)
        typed = jax.jit(lambda k, s: event_key(k, s, 7, typed=True))(key, step)
        np.testing.assert_array_equal(raw, expected)
        np.testing.assert_array_equal(jax.random.key_data(typed), expected)
        np.testing.assert_array_equal(jax.random.normal(typed, (6,)), expected_draw)
        np.testing.assert_array_equal(event_key(as_threefry_key(key), step, 7), expected)
        assert raw.shape == (2,) and raw.dtype == jnp.uint32
        assert not np.array_equal(raw, event_key(key, 5, 7))
        assert not np.array_equal(raw, event_key(key, step, 8))


@pytest.mark.parametrize("method_name", ["mash2", "mashrm"])
def test_mapping_samples_preserve_legacy_draws_and_restart_under_rbg(method_name, tmp_path):
    n = 2 if method_name == "mash2" else 3
    model = LinearEPCModel(n, 1)
    params = model.create_params(np.diag(np.linspace(-.4, .7, n)), np.zeros((1, n, n)), omega=[.3])
    kwargs = dict(seed=173, trajectory_id=23)
    with jax.default_prng_impl("threefry2x32"):
        next_key, sample_key = jax.random.split(legacy_keys(173, [23])[0])
        if method_name == "mash2":
            expected = mash2_state(model, params, [.2], [.3], sample_population_spin(sample_key),
                                   active=0, **kwargs)
        else:
            expected = mashrm_state(model, params, [.2], [.3],
                                    sample_population_conditional(sample_key, n, population=0),
                                    basis="adiabatic", **kwargs)
        expected = expected._replace(key=next_key)
    with jax.default_prng_impl("rbg"):
        if method_name == "mash2":
            initial = sample_adiabatic_population(model, params, [.2], [.3], **kwargs)
            method, measurement = MASH2(event_substeps=1), MASHPopulation()
        else:
            initial = sample_population(model, params, [.2], [.3], **kwargs)
            method, measurement = MASHRM(event_substeps=1), MASHRMPopulation()
        for actual, reference in zip(jax.tree.leaves(initial), jax.tree.leaves(expected), strict=True):
            np.testing.assert_array_equal(actual, reference)
        problem = Problem(model, params, CoupledClassical(1.3), method, measurement)
        simulation = Simulation(problem, Integrator(.02, "exponential_midpoint"),
                                Execution(chunk_size=2))
        full = simulation.run(initial, 5)
        first = simulation.run(initial, 2)
        checkpoint = tmp_path/f"{method_name}.h5"
        simulation.save_checkpoint(checkpoint, first.final_state)
        resumed = simulation.run(simulation.load_checkpoint(checkpoint), 3)
        for actual, reference in zip(jax.tree.leaves(resumed.final_state),
                                      jax.tree.leaves(full.final_state), strict=True):
            np.testing.assert_allclose(actual, reference, atol=2e-14, rtol=0.)
        np.testing.assert_array_equal(resumed.final_state.key, initial.key)
        assert resumed.final_state.key.shape == (2,)
