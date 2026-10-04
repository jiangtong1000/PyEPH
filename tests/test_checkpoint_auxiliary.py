"""Workflow continuation data stays separate, explicit and atomically saved."""

import json

import h5py
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pyeph.core.state import ElectronicPathState, make_state, stack_states
from pyeph.io.checkpoint import load_checkpoint, save_checkpoint


def assert_tree_equal(actual, expected):
    assert jax.tree.structure(actual) == jax.tree.structure(expected)
    for a, b in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
        assert np.asarray(a).dtype == np.asarray(b).dtype
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("recorded", [False, True])
def test_auxiliary_nested_numerical_tree_roundtrip_preserves_state_and_dtypes(tmp_path, recorded):
    state = (ElectronicPathState(jnp.array([1., 0.], dtype=complex), jnp.float64(.7),
                                  jnp.int64(12), jnp.int32(3)) if recorded else
             stack_states([make_state([.1+i], [.2], [1., 0.], trajectory_id=10+i,
                                      method_state={"events": jnp.int32(i)}) for i in range(2)]))
    auxiliary = {
        "origin": {"v0": np.array([[.3, -.4], [.7, .2]], dtype=np.float64),
                   "time0": jnp.float64(.125), "ids": np.array([10, 11], dtype=np.uint32)},
        "nested": [None, (np.array([1+2j], dtype=np.complex128),
                           np.array([False, True], dtype=bool))],
        "small": np.array([-3, 2], dtype=np.int16),
    }
    metadata = {"model": "fixture-v1", "workflow": "origin-v1", "beta": 2.3}
    path = tmp_path/"workflow.h5"
    save_checkpoint(path, state, metadata=metadata, auxiliary=auxiliary)
    with h5py.File(path) as handle:
        assert handle.attrs["schema_version"] == 2  # schema-1 readers must reject this file
        assert "auxiliary" in handle and "auxiliary_manifest" in handle.attrs
    with pytest.raises(ValueError, match="with_auxiliary=True"):
        load_checkpoint(path)
    loaded, actual_metadata, actual_auxiliary = load_checkpoint(
        path, expected_metadata=metadata, with_auxiliary=True)
    assert type(loaded) is type(state)
    assert_tree_equal(loaded, state)
    assert actual_metadata == metadata
    assert_tree_equal(actual_auxiliary, auxiliary)
    with pytest.raises(ValueError, match="metadata mismatch"):
        load_checkpoint(path, expected_metadata={"workflow": "other"}, with_auxiliary=True)


def test_ordinary_schema_one_file_retains_two_tuple_and_supports_opt_in(tmp_path):
    path = tmp_path/"ordinary.h5"
    state = make_state([.2], [.3], [1., 0.])
    save_checkpoint(path, state, metadata={"legacy": True})
    with h5py.File(path, "r+") as handle:
        assert handle.attrs["schema_version"] == 1
        assert "auxiliary" not in handle and "auxiliary_manifest" not in handle.attrs
        # Original files that predate state_type still default to TrajectoryState.
        del handle.attrs["state_type"]
    loaded, metadata = load_checkpoint(path)
    assert_tree_equal(loaded, state)
    assert metadata == {"legacy": True}
    loaded, metadata, auxiliary = load_checkpoint(path, with_auxiliary=True)
    assert_tree_equal(loaded, state)
    assert auxiliary is None


@pytest.mark.parametrize("auxiliary", [{}, [], (), False, np.zeros(0, dtype=np.float32)])
def test_empty_or_false_auxiliary_is_still_explicit_continuation_context(tmp_path, auxiliary):
    path = tmp_path/"empty.h5"
    save_checkpoint(path, make_state([0.], [0.], [1.]), metadata={}, auxiliary=auxiliary)
    with pytest.raises(ValueError, match="auxiliary workflow data"):
        load_checkpoint(path)
    _, _, loaded = load_checkpoint(path, with_auxiliary=True)
    assert_tree_equal(loaded, auxiliary)


@pytest.mark.parametrize("flag", [0, 1, "true", None, [], np.array([True]), jnp.array([True])])
def test_with_auxiliary_flag_requires_a_boolean_scalar(tmp_path, flag):
    with pytest.raises(ValueError, match="boolean scalar"):
        load_checkpoint(tmp_path/"unused.h5", with_auxiliary=flag)


@pytest.mark.parametrize("flag", [np.bool_(True), jnp.array(True)])
def test_numpy_and_jax_boolean_scalars_are_supported(tmp_path, flag):
    path = tmp_path/"boolean.h5"
    save_checkpoint(path, make_state([0.], [0.], [1.]), metadata={}, auxiliary={"origin": 1})
    assert int(load_checkpoint(path, with_auxiliary=flag)[2]["origin"]) == 1


@pytest.mark.parametrize("damage", ["value", "shape", "dtype"])
def test_auxiliary_array_integrity_is_checked(tmp_path, damage):
    path = tmp_path/"corrupted.h5"
    save_checkpoint(path, make_state([0.], [0.], [1.]), metadata={},
                    auxiliary={"v0": np.array([.2, .3], dtype=np.float64)})
    with h5py.File(path, "r+") as handle:
        if damage == "value":
            handle["auxiliary/0/value"][0] = .9
        else:
            description = json.loads(handle.attrs["auxiliary_manifest"])
            description["children"][0][damage] = [3] if damage == "shape" else np.dtype("float32").str
            handle.attrs["auxiliary_manifest"] = json.dumps(description)
    with pytest.raises(ValueError, match="checksum" if damage == "value" else "shape or dtype"):
        load_checkpoint(path, with_auxiliary=True)


@pytest.mark.parametrize("dtype,value", [(np.float64, .123456789012345),
                                         (np.complex128, .2+.3j),
                                         (np.uint64, 2**40+17)])
def test_auxiliary_never_silently_narrows_when_x64_is_disabled(tmp_path, dtype, value):
    path = tmp_path/"wide.h5"
    previous = jax.config.x64_enabled
    try:
        jax.config.update("jax_enable_x64", False)
        # Every physical-state leaf is narrow; only auxiliary data needs x64.
        state = make_state([0.], [0.], [1.])
        auxiliary = {"wide": np.asarray(value, dtype=dtype)}
        save_checkpoint(path, state, metadata={}, auxiliary=auxiliary)
        with pytest.raises(ValueError, match="requires JAX x64"):
            load_checkpoint(path, with_auxiliary=True)
        jax.config.update("jax_enable_x64", True)
        _, _, actual = load_checkpoint(path, with_auxiliary=True)
        assert_tree_equal(actual, auxiliary)
    finally:
        jax.config.update("jax_enable_x64", previous)


@pytest.mark.parametrize("previous_auxiliary", [None, {"v0": np.array([.2])}])
@pytest.mark.parametrize("broken", [{"first": np.array([1.]), "bad": object()},
                                    {2: np.array([1.])}])
def test_failed_auxiliary_write_preserves_existing_checkpoint_atomically(tmp_path, previous_auxiliary, broken):
    path = tmp_path/"atomic.h5"
    state = make_state([.2], [.3], [1.])
    save_checkpoint(path, state, metadata={"generation": 1}, auxiliary=previous_auxiliary)
    original = path.read_bytes()
    with pytest.raises(TypeError):
        save_checkpoint(path, state, metadata={"generation": 2}, auxiliary=broken)
    assert path.read_bytes() == original
    assert list(tmp_path.iterdir()) == [path]
    loaded, metadata, auxiliary = load_checkpoint(path, with_auxiliary=True)
    assert_tree_equal(loaded, state)
    assert metadata["generation"] == 1
    assert_tree_equal(auxiliary, previous_auxiliary)


@pytest.mark.parametrize("damage", ["tree", "manifest", "both", "schema"])
def test_incomplete_or_mislabeled_auxiliary_cannot_be_ignored(tmp_path, damage):
    path = tmp_path/"incomplete.h5"
    save_checkpoint(path, make_state([0.], [0.], [1.]), metadata={}, auxiliary={"v0": .2})
    with h5py.File(path, "r+") as handle:
        if damage in ("tree", "both"):
            del handle["auxiliary"]
        if damage in ("manifest", "both"):
            del handle.attrs["auxiliary_manifest"]
        if damage == "schema":
            handle.attrs["schema_version"] = 1
    with pytest.raises(ValueError, match="auxiliary"):
        load_checkpoint(path, with_auxiliary=True)
    with pytest.raises(ValueError, match="auxiliary"):
        load_checkpoint(path)
