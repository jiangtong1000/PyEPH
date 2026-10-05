"""Explicit, non-executable identities for reconstructing a checkpointed run.

Native dataclasses describe configuration; numerical arrays are hashed in full.
An opaque provider or custom callable additionally needs a caller-supplied
artifact identity. Its Python source alone cannot identify captured weights,
closures, external files, or mutable global state.
"""

from dataclasses import fields, is_dataclass
import hashlib
from importlib import metadata
import json
import math
from pathlib import Path
import platform
import sys

import jax
import numpy as np

SCHEMA_VERSION = 1


def _json_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("utf-8")


def _hash(value):
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def _type_name(value):
    cls = value if isinstance(value, type) else type(value)
    return f"{cls.__module__}.{cls.__qualname__}"


def _array(value):
    array = np.asarray(jax.device_get(value))
    if array.dtype.kind not in "biufc":
        raise TypeError("provenance arrays must have numerical or boolean dtype")
    return {"kind": "array", "shape": list(array.shape), "dtype": array.dtype.str,
            "sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest()}


def _numeric_tree(value):
    """Plain numerical containers have stable structure without custom reprs."""
    if value is None:
        return None
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("parameter dictionary keys must be strings")
        return {"kind": "dict", "items": {k: _numeric_tree(value[k]) for k in sorted(value)}}
    if type(value) in (tuple, list):
        return {"kind": type(value).__name__, "items": [_numeric_tree(x) for x in value]}
    if isinstance(value, (jax.Array, np.ndarray, np.generic, bool, int, float, complex)):
        return _array(value)
    raise TypeError("params must use numerical leaves and plain dict/list/tuple containers; "
                    "convert custom PyTree nodes explicitly before making a manifest")


class _Encoder:
    def __init__(self, artifact_ids):
        self.artifact_ids = {} if artifact_ids is None else dict(artifact_ids)
        if any(not isinstance(k, str) or not k or not isinstance(v, str) or not v.strip()
               for k, v in self.artifact_ids.items()):
            raise ValueError("artifact_ids must map exact object paths to nonempty identity strings")
        self.used = set()
        self.unresolved = []
        self.modules = set()
        self.active = set()

    def identity(self, value, path, reason):
        self.modules.add(getattr(value, "__module__", type(value).__module__))
        identity = self.artifact_ids.get(path)
        if identity is None:
            self.unresolved.append(path)
        else:
            self.used.add(path)
        result = {"reason": reason, "artifact_id": identity}
        if callable(value):
            result["callable"] = (f"{getattr(value, '__module__', type(value).__module__)}."
                                  f"{getattr(value, '__qualname__', type(value).__qualname__)}")
        return result

    def encode(self, value, path):
        if value is None or type(value) in (str, bool, int):
            return value
        if type(value) is float:
            if math.isnan(value):
                raise ValueError(f"NaN configuration scalar at {path}")
            if math.isinf(value):
                # Positive inverse-temperature infinity is a valid zero-T
                # declaration. Encode it explicitly; JSON Infinity is invalid.
                return {"kind": "float", "value": "+infinity" if value > 0 else "-infinity"}
            return value
        if isinstance(value, (jax.Array, np.ndarray, np.generic, complex)):
            return _array(value)
        if id(value) in self.active:
            raise ValueError(f"cyclic configuration at {path}; use an opaque artifact provider")
        self.active.add(id(value))
        try:
            return self._container(value, path)
        finally:
            self.active.remove(id(value))

    def _container(self, value, path):
        if isinstance(value, dict):
            if not all(isinstance(k, str) for k in value):
                raise TypeError(f"configuration dictionary keys at {path} must be strings")
            return {"kind": "dict", "items": {
                k: self.encode(value[k], f"{path}[{json.dumps(k)}]") for k in sorted(value)}}
        if type(value) in (tuple, list):
            return {"kind": type(value).__name__, "items": [
                self.encode(x, f"{path}[{i}]") for i, x in enumerate(value)]}
        self.modules.add(type(value).__module__)
        if is_dataclass(value) and not isinstance(value, type):
            result = {"kind": "dataclass", "type": _type_name(value), "fields": {
                f.name: self.encode(getattr(value, f.name), f"{path}.{f.name}")
                for f in fields(value)}}
            # Package classes are written to keep configuration in their fields.
            # A custom dataclass can still consult arbitrary captured/global data.
            if not type(value).__module__.startswith("pyeph."):
                result["external_identity"] = self.identity(
                    value, path, "custom class behavior may depend on data outside its fields")
            return result
        return {"kind": "opaque", "type": _type_name(value), **self.identity(
            value, path, "object/callable may capture code, weights, files, or mutable state")}

    def finish(self):
        unused = sorted(set(self.artifact_ids) - self.used)
        if unused:
            raise ValueError(f"artifact_ids do not name opaque objects: {', '.join(unused)}")


def _source_evidence(modules):
    """Read source files as bytes, without importing or executing their contents."""
    root = Path(__file__).resolve().parents[1]
    files = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
             for path in sorted(root.rglob("*.py"))}
    external = {}
    for name in sorted(modules):
        if name.startswith("pyeph."):
            continue
        module = sys.modules.get(name)
        source = getattr(module, "__file__", None)
        path = None if source is None else Path(source)
        external[name] = (hashlib.sha256(path.read_bytes()).hexdigest()
                          if path is not None and path.suffix == ".py" and path.is_file()
                          else None)
    return {"pyeph_python_tree_sha256": _hash(files), "pyeph_python_file_count": len(files),
            "external_module_source_sha256": external}


def _runtime_evidence(modules):
    versions = {}
    packages = ["jax", "jaxlib", "numpy", "scipy", "h5py"]
    if {"pyeph.adapters.torch", "pyeph.adapters.torch_reference",
        "pyeph.adapters.torch_local"} & set(modules):
        packages.append("torch")
    for name in packages:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    versions["pyeph"] = getattr(sys.modules.get("pyeph"), "__version__", None)
    return {"versions": versions, "python": platform.python_version(),
            "jax_enable_x64": bool(jax.config.x64_enabled),
            "jax_default_matmul_precision": jax.config.jax_default_matmul_precision,
            "jax_random_configuration": {
                name: getattr(jax.config, name, None) for name in (
                    "jax_default_prng_impl", "jax_threefry_partitionable",
                    "jax_random_seed_offset")}}


def _manifest(kind, payload, encoder):
    encoder.finish()
    payload["source"] = _source_evidence(encoder.modules)
    payload["runtime"] = _runtime_evidence(encoder.modules)
    result = {"schema": SCHEMA_VERSION, "kind": kind, "payload": payload,
              "complete": not encoder.unresolved, "unresolved": sorted(encoder.unresolved)}
    return {**result, "fingerprint": _hash(result)}


def problem_manifest(problem, integrator, *, artifact_ids=None):
    """Identify the model, numerical params, nuclei, method and integrator.

    ``artifact_ids`` maps exact opaque-object paths (reported in ``unresolved``)
    to caller-controlled identities covering *all* code and captured artifacts.
    For example, ``{"model": "sha256:<provider bundle digest>"}`` identifies a
    Torch adapter, and ``model.shift_fn`` identifies a reference-shift callback.
    No provider is called and no model files are stored by this function.
    """
    encoder = _Encoder(artifact_ids)
    parameters = _numeric_tree(problem.params)
    payload = {"model_spec": encoder.encode(problem.model.spec, "model_spec"),
               "model": encoder.encode(problem.model, "model"),
               "params": {"tree": parameters, "sha256": _hash(parameters)},
               "nuclear_treatment": encoder.encode(problem.nuclear_treatment, "nuclear_treatment"),
               "method": encoder.encode(problem.method, "method"),
               "integrator": encoder.encode(integrator, "integrator"),
               "measurement": encoder.encode(problem.measurement, "measurement")}
    if problem.geometry_guard is not None:
        payload["geometry_guard"] = encoder.encode(problem.geometry_guard, "geometry_guard")
    return _manifest("problem", payload, encoder)


def recorded_path_manifest(path, integrator=None, *, method=None, artifact_ids=None):
    """Identify recorded arrays, interpolation/transport and propagation policy.

    Pass the runner's numerical method policy explicitly, for example
    ``method={"name": "recorded_cpa", "max_subspace_loss": 1e-8}``. Omitting
    it makes this a dataset identity, which is incomplete for strict run resume.
    ``integrator=None`` is valid for propagation between adiabatic frames.
    """
    encoder = _Encoder(artifact_ids)
    payload = {"path": encoder.encode(path, "path"),
               "integrator": encoder.encode(integrator, "integrator"),
               "method": encoder.encode(method, "method")}
    if method is None:
        encoder.unresolved.append("method")
    # These path declarations are class constants in the native implementation.
    payload["path_contract"] = {name: encoder.encode(getattr(path, name, None), f"path.{name}")
                                for name in ("basis_id", "basis_kind", "force_support",
                                             "interpolation", "unit_system")}
    if getattr(path, "unit_system", None) is None:
        encoder.unresolved.append("path.unit_system")
    if getattr(path, "basis_kind", None) == "fixed_orthonormal" and integrator is None:
        encoder.unresolved.append("integrator")
    return _manifest("recorded_path", payload, encoder)


def validate_manifest(manifest, *, require_complete=True):
    """Check JSON integrity and completeness; return the unchanged manifest.

    A hash detects mismatches, not authenticity or correctness of caller-supplied
    artifact identities. No data is loaded or executable model reconstructed.
    """
    required = {"schema", "kind", "payload", "fingerprint", "complete", "unresolved"}
    if not isinstance(manifest, dict) or set(manifest) != required:
        raise ValueError("invalid provenance manifest fields")
    if manifest["schema"] != SCHEMA_VERSION or manifest["kind"] not in {"problem", "recorded_path"}:
        raise ValueError("unsupported provenance manifest schema or kind")
    if (not isinstance(manifest["payload"], dict) or type(manifest["complete"]) is not bool
            or not isinstance(manifest["unresolved"], list)
            or not all(isinstance(x, str) for x in manifest["unresolved"])
            or manifest["complete"] != (not manifest["unresolved"])):
        raise ValueError("invalid provenance completeness declaration")
    expected = _hash({k: v for k, v in manifest.items() if k != "fingerprint"})
    if manifest["fingerprint"] != expected:
        raise ValueError("provenance manifest checksum mismatch")
    if require_complete and not manifest["complete"]:
        raise ValueError("incomplete provenance; supply explicit identities/configuration for: "
                         + ", ".join(manifest["unresolved"]))
    return manifest


def assert_matching_manifest(saved, expected, *, strict=True):
    """Reject different identities; strict mode also rejects incomplete ones.

    ``strict=False`` only permits incomplete manifests. It never ignores changed
    hashes, units, numerical data, source evidence, or runtime versions.
    """
    validate_manifest(saved, require_complete=strict)
    validate_manifest(expected, require_complete=strict)
    if saved["fingerprint"] != expected["fingerprint"]:
        sections = sorted(k for k in set(saved["payload"]) | set(expected["payload"])
                          if saved["payload"].get(k) != expected["payload"].get(k))
        details = ", ".join(sections) or "kind/completeness"
        raise ValueError(f"checkpoint provenance mismatch: {details}")
    return expected
