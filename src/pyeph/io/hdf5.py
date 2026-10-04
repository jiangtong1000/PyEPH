"""Chunked observation output compatible with Runner's streaming callback."""

import json
from pathlib import Path

import h5py
import numpy as np


def _flatten(values, prefix=""):
    if not isinstance(values, dict):
        yield prefix or "value", np.asarray(values)
        return
    for key, value in values.items():
        if not isinstance(key, str) or not key or "/" in key or key in {".", ".."}:
            raise ValueError("observable keys must be simple nonempty names")
        yield from _flatten(value, f"{prefix}/{key}" if prefix else key)


class HDF5Observer:
    """Append time-major samples. A context manager closes and flushes the file.

    Existing files are preserved unless overwrite=True is explicitly selected.
    Restart files are separate atomic checkpoints; this stream is not a checkpoint.
    """

    def __init__(self, path, *, metadata=None, overwrite=False):
        metadata_json = json.dumps(metadata or {}, sort_keys=True, allow_nan=False)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = h5py.File(path, "w" if overwrite else "x")
        self.handle.attrs["schema_version"] = 1
        self.handle.attrs["metadata"] = metadata_json
        self._names = None

    def __call__(self, times, values):
        arrays = {"time": np.asarray(times), **{f"observables/{k}": v for k, v in _flatten(values)}}
        count = len(arrays["time"])
        if self._names is not None and set(arrays) != self._names:
            raise ValueError("observation structure changed during a stream")
        # Check the full append before changing any dataset.
        for name, array in arrays.items():
            if array.ndim == 0 or array.shape[0] != count or array.dtype.kind not in "biufc":
                raise ValueError("observations must be numerical arrays with a shared leading time axis")
            if name in self.handle:
                dataset = self.handle[name]
                if dataset.shape[1:] != array.shape[1:] or dataset.dtype != array.dtype:
                    raise ValueError("observation shape or dtype changed during a stream")
        for name, array in arrays.items():
            if name not in self.handle:
                self.handle.create_dataset(name, data=array, maxshape=(None, *array.shape[1:]), chunks=True)
            else:
                dataset = self.handle[name]
                old_size = dataset.shape[0]
                dataset.resize(old_size + count, axis=0)
                dataset[old_size:] = array
        self._names = set(arrays)
        self.handle.flush()

    def close(self):
        self.handle.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
