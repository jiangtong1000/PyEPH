"""Strict data-only model bundles; reconstruction belongs to trusted provider code."""

import hashlib
import io
import json
from pathlib import Path

import numpy as np


SCHEMA = "pyeph.provider_bundle.v1"


def bundle_identity(value):
    """SHA256 of canonical JSON data, never a Python object's representation."""
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def _json_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def _sha256(value):
    return (isinstance(value, str) and len(value) == 64
            and all(c in "0123456789abcdef" for c in value))


def _contract(value):
    if not isinstance(value, dict):
        raise ValueError("provider contract must be a JSON object")
    for key in ("provider", "provider_version", "basis_id", "neutral_reference", "scope"):
        if not isinstance(value.get(key), str) or not value[key].strip():
            raise ValueError(f"provider contract requires {key}")
    if value.get("units") != {"energy": "hartree", "length": "bohr"}:
        raise ValueError("provider contract requires hartree/bohr units")
    if value.get("basis_kind") != "fixed_effective_orthonormal":
        raise ValueError("bundle v1 requires a fixed effective orthonormal basis")
    if value.get("carrier") not in {"hole", "electron"}:
        raise ValueError("provider contract requires a carrier convention")
    for key in ("baseline_sha256", "dataset_sha256"):
        if not _sha256(value.get(key)):
            raise ValueError(f"provider contract requires a lowercase {key}")
    code = value.get("code_hashes")
    if (not isinstance(code, dict) or not code
            or any(not isinstance(k, str) or not k.strip() or not _sha256(v)
                   for k, v in code.items())):
        raise ValueError("code_hashes must identify the provider implementation and dependencies")
    if not isinstance(value.get("configuration"), dict):
        raise ValueError("provider contract requires a static configuration")
    return json.loads(_json_bytes(value))


def _validation(value):
    if (not isinstance(value, dict) or not isinstance(value.get("scope"), str)
            or not value["scope"].strip()):
        raise ValueError("validation requires an explicit scope")
    checks = value.get("checks")
    if (not isinstance(checks, list) or not checks
            or any(not isinstance(check, dict) or not isinstance(check.get("name"), str)
                   or not check["name"].strip() or check.get("passed") is not True
                   for check in checks)):
        raise ValueError("validation requires named, explicitly passing checks")
    return json.loads(_json_bytes(value))


def _arrays(value):
    if not isinstance(value, dict) or not value:
        raise ValueError("bundle requires named numerical arrays")
    result = {}
    for name, item in value.items():
        if (not isinstance(name, str) or not name or name.startswith("/")
                or any(part in {"", ".", ".."} for part in name.split("/"))
                or "\\" in name):
            raise ValueError("array names must be nonempty relative keys without traversal")
        array = np.asarray(item)
        if array.dtype.kind not in "biufc" or not np.isfinite(array).all():
            raise ValueError("bundle arrays must be finite numerical values; no object serialization")
        result[name] = np.array(array, copy=True)
    return result


def _array_schema(arrays):
    return {name: dict(shape=list(value.shape), dtype=value.dtype.str)
            for name, value in arrays.items()}


def save_bundle(directory, arrays, *, contract, validation, migration=None):
    """Write a fresh immutable-by-convention directory and return its manifest.

    The contract identifies code, baseline, dataset, physical conventions and
    static provider configuration. Validation records caller-executed checks;
    checksums prove identity, not scientific correctness or authenticity.
    Existing directories are never overwritten. A failed write leaves its
    incomplete directory for inspection, without a valid manifest.
    """
    contract, validation, arrays = _contract(contract), _validation(validation), _arrays(arrays)
    if migration is not None:
        migration = json.loads(_json_bytes(migration))
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    payload = io.BytesIO()
    np.savez_compressed(payload, **arrays)
    data = payload.getvalue()
    (directory / "arrays.npz").write_bytes(data)
    record = dict(schema=SCHEMA, contract=contract, validation=validation,
                  arrays_file="arrays.npz", arrays_sha256=hashlib.sha256(data).hexdigest(),
                  array_schema=_array_schema(arrays), migration=migration)
    record["identity"] = bundle_identity(record)
    (directory / "bundle.json").write_bytes(_json_bytes(record) + b"\n")
    return record


def _read_bundle(manifest):
    path = Path(manifest)
    record = json.loads(path.read_text())
    keys = {"schema", "contract", "validation", "arrays_file", "arrays_sha256",
            "array_schema", "migration", "identity"}
    if not isinstance(record, dict) or set(record) != keys or record["schema"] != SCHEMA:
        raise ValueError("unsupported or malformed provider bundle schema")
    if record["identity"] != bundle_identity({k: v for k, v in record.items() if k != "identity"}):
        raise ValueError("bundle manifest checksum mismatch")
    _contract(record["contract"])
    _validation(record["validation"])
    filename = record["arrays_file"]
    if (not isinstance(filename, str) or Path(filename).name != filename
            or Path(filename).suffix != ".npz" or "\\" in filename):
        raise ValueError("arrays_file must name a sibling NPZ file")
    payload = path.parent / filename
    if payload.is_symlink():
        raise ValueError("bundle arrays must be a local sibling file, not a symlink")
    data = payload.read_bytes()
    if hashlib.sha256(data).hexdigest() != record["arrays_sha256"]:
        raise ValueError("bundle array checksum mismatch")
    with np.load(io.BytesIO(data), allow_pickle=False) as archive:
        if len(set(archive.files)) != len(archive.files):
            raise ValueError("bundle contains duplicate array names")
        arrays = _arrays({name: archive[name] for name in archive.files})
    if _array_schema(arrays) != record["array_schema"]:
        raise ValueError("bundle array schema mismatch")
    return arrays, record


def load_bundle(manifest, *, expected_contract):
    """Verify an exact caller-owned contract and return plain NumPy arrays.

    No module is imported from a manifest, and no precision conversion occurs.
    Trusted provider code must reconstruct the model, validate parameters and
    explicitly choose the device/precision before dynamics.
    """
    expected_contract = _contract(expected_contract)
    arrays, record = _read_bundle(manifest)
    if record["contract"] != expected_contract:
        changed = sorted(k for k in set(record["contract"]) | set(expected_contract)
                         if record["contract"].get(k) != expected_contract.get(k))
        raise ValueError(f"provider contract mismatch: {', '.join(changed)}; revalidate explicitly")
    return arrays, record


def revalidate_bundle(source, destination, *, expected_contract, validator, reason):
    """Requalify unchanged weights after a code/version change into a fresh bundle.

    ``validator(arrays, old_contract, new_contract)`` is trusted caller code
    that reconstructs the new provider and returns scoped passing checks. It
    must execute the required numerical/physical validation, not merely copy
    an earlier report. Conventions, baseline and configuration cannot change
    here: an actual model conversion requires a separate explicit export.
    Strict trajectory checkpoints are unaffected.
    """
    if not isinstance(reason, str) or not reason.strip() or not callable(validator):
        raise ValueError("revalidation requires a reason and an explicit validator")
    expected_contract = _contract(expected_contract)
    arrays, old = _read_bundle(source)
    changed = sorted(k for k in set(old["contract"]) | set(expected_contract)
                     if old["contract"].get(k) != expected_contract.get(k))
    if not changed or not set(changed) <= {"code_hashes", "provider_version"}:
        raise ValueError("revalidation permits only code_hashes/provider_version changes")
    copies = {name: np.array(value, copy=True) for name, value in arrays.items()}
    for value in copies.values():
        value.flags.writeable = False
    validation = validator(copies, json.loads(_json_bytes(old["contract"])),
                           json.loads(_json_bytes(expected_contract)))
    migration = dict(source_bundle_identity=old["identity"], reason=reason, changed=changed,
                     source_contract=old["contract"], operation="unchanged-weight revalidation")
    return save_bundle(destination, arrays, contract=expected_contract,
                       validation=validation, migration=migration)
