import json

import h5py
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pyeph.core.state import make_state, stack_states
from pyeph.execution.random import event_key, trajectory_keys
from pyeph.initialization import sample_harmonic
from pyeph.io.checkpoint import array_fingerprint, load_checkpoint, save_checkpoint
from pyeph.io.hdf5 import HDF5Observer


def test_checkpoint_preserves_batched_physical_method_and_random_state(tmp_path):
    states = [make_state([i + 0.1], [0.3], [1, 0], trajectory_id=i, seed=8,
                         method_state={"transport": {"rho0": jnp.eye(2) / 2},
                                       "events": [jnp.array(i), None]}) for i in range(3)]
    state = stack_states(states)
    path = tmp_path / "restart.h5"
    metadata = {"model_hash": "fixture", "dt": 0.01, "basis_id": "site_order_v1"}
    save_checkpoint(path, state, metadata=metadata)
    loaded, actual_metadata = load_checkpoint(path, expected_metadata=metadata)
    assert actual_metadata == metadata
    assert jax.tree.structure(loaded) == jax.tree.structure(state)
    for expected, actual in zip(jax.tree.leaves(state), jax.tree.leaves(loaded), strict=True):
        np.testing.assert_array_equal(actual, expected)
    with pytest.raises(ValueError, match="metadata mismatch"):
        load_checkpoint(path, expected_metadata={"dt": 0.02})


def test_failed_checkpoint_write_keeps_previous_file(tmp_path):
    path = tmp_path / "restart.h5"
    state = make_state([0], [1], [1, 0])
    save_checkpoint(path, state, metadata={"generation": 1})
    broken = state._replace(method_state={"unsafe": object()})
    with pytest.raises(TypeError):
        save_checkpoint(path, broken, metadata={"generation": 2})
    _, metadata = load_checkpoint(path)
    assert metadata["generation"] == 1
    assert len(list(tmp_path.iterdir())) == 1


def test_checkpoint_detects_corrupted_array(tmp_path):
    path = tmp_path / "restart.h5"
    save_checkpoint(path, make_state([0], [1], [1, 0]), metadata={})
    with h5py.File(path, "r+") as f:
        manifest = json.loads(f.attrs["manifest"])
        index = manifest["keys"].index("q")
        f[f"state/{index}/value"][0] = 3.0
    with pytest.raises(ValueError, match="checksum"):
        load_checkpoint(path)


def test_checkpoint_never_truncates_unsigned_64bit_method_state(tmp_path):
    path = tmp_path / "unsigned-counter.h5"
    previous = jax.config.x64_enabled
    try:
        jax.config.update("jax_enable_x64", False)
        state = make_state([0], [1], [1, 0],
                           method_state={"count": np.array(2**40 + 7, dtype=np.uint64)})
        save_checkpoint(path, state, metadata={})
        with pytest.raises(ValueError, match="requires JAX x64"):
            load_checkpoint(path)
        jax.config.update("jax_enable_x64", True)
        loaded, _ = load_checkpoint(path)
        assert loaded.method_state["count"].dtype == jnp.uint64
        assert int(loaded.method_state["count"]) == 2**40 + 7
    finally:
        jax.config.update("jax_enable_x64", previous)


def test_parameter_fingerprint_sensitive_to_dtype_shape_and_values():
    x = {"h": np.eye(2), "extra": (np.array([1, 2]),)}
    assert array_fingerprint(x) == array_fingerprint(x)
    assert array_fingerprint(x) != array_fingerprint({**x, "h": np.eye(2, dtype=np.float32)})
    assert array_fingerprint(x) != array_fingerprint({**x, "h": np.ones((2, 2))})


def test_hdf5_stream_matches_chunks_and_rejects_shape_changes(tmp_path):
    path = tmp_path / "observations.h5"
    with HDF5Observer(path, metadata={"unit": "atomic"}) as output:
        output(np.array([0., 1.]), {"population": np.array([[1., 0.], [0.8, 0.2]]),
                                    "nested": {"current": np.array([1j, 2j])}})
        output(np.array([2.]), {"population": np.array([[0.5, 0.5]]),
                                "nested": {"current": np.array([3j])}})
        with pytest.raises(ValueError, match="structure"):
            output(np.array([3.]), {"other": np.array([1.])})
        with pytest.raises(ValueError, match="shape"):
            output(np.array([3.]), {"population": np.zeros((1, 3)),
                                    "nested": {"current": np.array([3j])}})
    with h5py.File(path) as f:
        np.testing.assert_array_equal(f["time"][:], [0, 1, 2])
        np.testing.assert_array_equal(f["observables/nested/current"][:], [1j, 2j, 3j])
        assert f["observables/population"].shape == (3, 2)
    with pytest.raises(FileExistsError):
        HDF5Observer(path)


def test_invalid_stream_metadata_does_not_truncate_existing_output(tmp_path):
    path = tmp_path / "observations.h5"
    with HDF5Observer(path, metadata={"generation": 1}) as output:
        output(np.array([0.]), {"population": np.array([[1., 0.]])})
    original = path.read_bytes()
    with pytest.raises(ValueError):
        HDF5Observer(path, metadata={"invalid": float("nan")}, overwrite=True)
    assert path.read_bytes() == original


@pytest.mark.parametrize("distribution", ["classical", "wigner"])
def test_harmonic_sampling_partition_and_theoretical_variance(distribution):
    ids = np.arange(16000)
    w, m, temperature = np.array([0.8, 1.7]), np.array([1.2, 2.3]), 0.4
    q, p = sample_harmonic(w, m, temperature, ids, seed=127, distribution=distribution)
    q1, p1 = sample_harmonic(w, m, temperature, ids[:57], seed=127, distribution=distribution)
    q2, p2 = sample_harmonic(w, m, temperature, ids[57:], seed=127, distribution=distribution)
    np.testing.assert_array_equal(q, np.concatenate([q1, q2]))
    np.testing.assert_array_equal(p, np.concatenate([p1, p2]))
    if distribution == "classical":
        qvar, pvar = temperature / (m * w**2), m * temperature
    else:
        occupation = 1 / np.tanh(w / (2 * temperature))
        qvar, pvar = occupation / (2 * m * w), occupation * m * w / 2
    np.testing.assert_allclose(np.var(q, axis=0), qvar, rtol=0.035)
    np.testing.assert_allclose(np.var(p, axis=0), pvar, rtol=0.035)


def test_zero_temperature_sampling_and_invalid_inputs():
    q, p = sample_harmonic([1], 1, 0, [0, 1], distribution="classical")
    np.testing.assert_array_equal(q, 0)
    np.testing.assert_array_equal(p, 0)
    q, p = sample_harmonic([1], 1, 0, range(10), distribution="wigner")
    assert np.any(np.asarray(q) != 0) and np.any(np.asarray(p) != 0)
    with pytest.raises(ValueError, match="strictly positive"):
        sample_harmonic([0], 1, 1, [0])
    with pytest.raises(ValueError, match="unique"):
        trajectory_keys(1, [3, 3])


def test_event_keys_do_not_repeat_after_uint32_step_wrap():
    key = trajectory_keys(10, [123])[0]
    a = event_key(key, jnp.int64(5), 2)
    b = event_key(key, jnp.int64(5 + 2**32), 2)
    np.testing.assert_array_equal(a, event_key(key, jnp.int64(5), 2))
    assert not np.array_equal(a, b)
    assert not np.array_equal(a, event_key(key, jnp.int64(5), 3))
