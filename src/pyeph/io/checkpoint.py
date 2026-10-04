"""Atomic, versioned HDF5 trajectory checkpoints without executable serialization."""

import hashlib
import json
import os
from pathlib import Path
import tempfile

import h5py
import jax
import jax.numpy as jnp
import numpy as np

from pyeph.core.state import ElectronicPathState, TrajectoryState
from pyeph.core._configuration import boolean_scalar

SCHEMA_VERSION = 1
AUXILIARY_SCHEMA_VERSION = 2


def _write_tree(group, value):
    if isinstance(value, dict):
        if not all(isinstance(k, str) for k in value):
            raise TypeError("checkpoint dictionary keys must be strings")
        keys = list(value)
        return {"kind": "dict", "keys": keys,
                "children": [_write_tree(group.create_group(str(i)), value[k]) for i, k in enumerate(keys)]}
    if isinstance(value, (tuple, list)):
        if hasattr(value, "_fields"):
            raise TypeError("custom named tuples in checkpoint trees need conversion to a plain dictionary")
        return {"kind": "tuple" if isinstance(value, tuple) else "list",
                "children": [_write_tree(group.create_group(str(i)), v) for i, v in enumerate(value)]}
    if value is None:
        return {"kind": "none"}
    array = np.asarray(jax.device_get(value))
    if array.dtype.kind not in "biufc":
        raise TypeError("checkpoint leaves must be numerical arrays or None")
    group.create_dataset("value", data=array)
    digest = hashlib.sha256(array.tobytes(order="C")).hexdigest()
    return {"kind": "array", "shape": list(array.shape), "dtype": array.dtype.str, "sha256": digest}


def _read_tree(group, description):
    kind = description["kind"]
    if kind == "none":
        return None
    if kind == "array":
        array = group["value"][()]
        array = np.asarray(array)
        if list(array.shape) != description["shape"] or array.dtype.str != description["dtype"]:
            raise ValueError("checkpoint array shape or dtype disagrees with its manifest")
        if hashlib.sha256(array.tobytes(order="C")).hexdigest() != description["sha256"]:
            raise ValueError("checkpoint array checksum mismatch")
        needs_x64 = ((array.dtype.kind in "iuf" and array.dtype.itemsize == 8)
                     or (array.dtype.kind == "c" and array.dtype.itemsize == 16))
        if not jax.config.x64_enabled and needs_x64:
            raise ValueError("checkpoint requires JAX x64; enable it before loading to avoid precision loss")
        return jnp.asarray(array)
    children = [_read_tree(group[str(i)], child) for i, child in enumerate(description["children"])]
    if kind == "dict":
        return dict(zip(description["keys"], children, strict=True))
    if kind == "tuple":
        return tuple(children)
    if kind == "list":
        return children
    raise ValueError(f"unknown checkpoint node {kind!r}")


def save_checkpoint(path, state, *, metadata, auxiliary=None):
    """Persist all state, including random key and method data, by atomic replace.

    Metadata must include caller-owned model/config provenance. Executable model
    code and Python callbacks are never stored. Existing checkpoints survive a
    failed write. Reconstruct the Problem explicitly before resuming.

    An optional auxiliary numerical tree preserves workflow-owned continuation
    context separately from the physical state. Such files use schema 2 so old
    readers reject them instead of discarding that context. Ordinary files
    retain schema 1. Loading auxiliary data requires explicit opt-in.
    """
    if type(state) not in (TrajectoryState, ElectronicPathState):
        raise TypeError("checkpoint state must be a declared PyEPH trajectory state")
    metadata_json = json.dumps(metadata, sort_keys=True, allow_nan=False)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    try:
        with h5py.File(temporary, "w") as handle:
            handle.attrs["schema_version"] = (SCHEMA_VERSION if auxiliary is None
                                               else AUXILIARY_SCHEMA_VERSION)
            handle.attrs["state_type"] = type(state).__name__
            handle.attrs["metadata"] = metadata_json
            manifest = _write_tree(handle.create_group("state"), state._asdict())
            handle.attrs["manifest"] = json.dumps(manifest)
            if auxiliary is not None:
                auxiliary_manifest = _write_tree(handle.create_group("auxiliary"), auxiliary)
                handle.attrs["auxiliary_manifest"] = json.dumps(auxiliary_manifest)
            handle.flush()
        with open(temporary, "rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_checkpoint(path, *, expected_metadata=None, with_auxiliary=False):
    """Load state plus metadata, checking requested provenance keys exactly.

    The default two-tuple is available only for files without auxiliary data.
    ``with_auxiliary=True`` returns ``(state, metadata, auxiliary)``; the last
    item is None for ordinary schema-1 files. This opt-in prevents accidentally
    dropping a workflow's saved continuation context.
    """
    with_auxiliary = boolean_scalar(with_auxiliary, "with_auxiliary")
    with h5py.File(path, "r") as handle:
        schema = handle.attrs.get("schema_version")
        if schema not in (SCHEMA_VERSION, AUXILIARY_SCHEMA_VERSION):
            raise ValueError("unsupported checkpoint schema version")
        has_tree = "auxiliary" in handle
        has_manifest = "auxiliary_manifest" in handle.attrs
        if has_tree != has_manifest or (schema == AUXILIARY_SCHEMA_VERSION and not has_tree):
            raise ValueError("checkpoint auxiliary tree or manifest is missing")
        if has_tree and schema != AUXILIARY_SCHEMA_VERSION:
            raise ValueError("auxiliary data requires checkpoint schema 2")
        if has_tree and not with_auxiliary:
            raise ValueError("checkpoint contains auxiliary workflow data; load with "
                             "with_auxiliary=True to preserve continuation context")
        metadata = json.loads(handle.attrs["metadata"])
        state_type = handle.attrs.get("state_type", "TrajectoryState")
        types = {"TrajectoryState": TrajectoryState, "ElectronicPathState": ElectronicPathState}
        if state_type not in types:
            raise ValueError("unknown checkpoint state type")
        for key, value in (expected_metadata or {}).items():
            if key not in metadata or metadata[key] != value:
                raise ValueError(f"checkpoint metadata mismatch for {key!r}")
        data = _read_tree(handle["state"], json.loads(handle.attrs["manifest"]))
        auxiliary = (_read_tree(handle["auxiliary"], json.loads(handle.attrs["auxiliary_manifest"]))
                     if has_tree else None)
    state = types[state_type](**data)
    return (state, metadata, auxiliary) if with_auxiliary else (state, metadata)


def array_fingerprint(tree):
    """Content hash for numerical parameter PyTrees, including shapes and dtypes."""
    leaves, structure = jax.tree.flatten(tree)
    digest = hashlib.sha256(str(structure).encode())
    for leaf in leaves:
        array = np.asarray(jax.device_get(leaf))
        if array.dtype.kind not in "biufc":
            raise TypeError("fingerprint accepts numerical parameter trees only")
        digest.update(str(array.shape).encode())
        digest.update(array.dtype.str.encode())
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()
