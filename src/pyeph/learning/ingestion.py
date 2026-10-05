"""Explicit conversion evidence beside the fixed-basis label importer.

Provider-owned records identify supplied inputs and outputs; they do not infer
units, validate a physical teacher, or reconstruct executable providers.
"""

import hashlib
import io
import json
from importlib import metadata as packages
from pathlib import Path
import platform

import numpy as np

from pyeph.core import units
from .bundles import bundle_identity


def _json(value):
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def _array_identity(arrays):
    result = {}
    for name, value in arrays.items():
        if not isinstance(name, str) or not name:
            raise ValueError("conversion arrays require nonempty string names")
        array = np.asarray(value)
        if array.dtype.kind not in "biufcUS" or (array.dtype.kind not in "US"
                                                 and not np.isfinite(array).all()):
            raise ValueError("conversion arrays must contain finite numbers or strings")
        result[name] = dict(shape=list(array.shape), dtype=array.dtype.str,
                           sha256=hashlib.sha256(array.tobytes(order="C")).hexdigest())
    if not result:
        raise ValueError("conversion evidence requires arrays")
    return result


def _factors(values):
    result = {}
    for name, value in values.items():
        if not isinstance(name, str) or not name or isinstance(value, (bool, complex)):
            raise ValueError("conversion factors require named positive real scalars")
        array = np.asarray(value)
        if array.shape or array.dtype.kind not in "iuf" or not np.isfinite(array) or array <= 0:
            raise ValueError("conversion factors require named positive real scalars")
        value = float(array)
        result[name] = dict(value=value, hex=value.hex())
    if not result:
        raise ValueError("conversion factors must be explicit")
    return result


def record_conversion(raw_arrays, converted_arrays, *, raw_units, factors,
                      raw_configuration, converted_configuration, source_identity,
                      importer_identity, dependency_versions):
    """Identify one caller-performed conversion, including static configuration.

    Units/factors and their dimensional application remain importer-owned.
    Supply all geometry-dependent configuration, not only parameter arrays.
    """
    for value in (raw_units, source_identity, importer_identity, dependency_versions):
        if not isinstance(value, dict) or not value:
            raise ValueError("conversion provenance requires nonempty declarations")
    if not isinstance(raw_configuration, dict) or not isinstance(converted_configuration, dict):
        raise ValueError("conversion configuration must be explicit JSON objects")
    record = _json(dict(raw_units=raw_units, factors=_factors(factors),
                        raw_arrays=_array_identity(raw_arrays),
                        converted_arrays=_array_identity(converted_arrays),
                        raw_configuration=raw_configuration,
                        converted_configuration=converted_configuration,
                        source_identity=source_identity, importer_identity=importer_identity,
                        dependency_versions=dependency_versions))
    record["identity"] = bundle_identity(record)
    return record


def validate_conversion(record, arrays, *, configuration):
    """Reject changed converted arrays/configuration or malformed factor evidence.

    This checks declared identity, not the physical truth of a conversion.
    Providers must also test their derivative chain rule and source conventions.
    """
    fields = {"raw_units", "factors", "raw_arrays", "converted_arrays", "raw_configuration",
              "converted_configuration", "source_identity", "importer_identity",
              "dependency_versions", "identity"}
    if not isinstance(record, dict) or set(record) != fields:
        raise ValueError("invalid conversion provenance fields")
    if record["identity"] != bundle_identity({k: v for k, v in record.items() if k != "identity"}):
        raise ValueError("conversion provenance checksum mismatch")
    try:
        normalized = _factors({k: v["value"] for k, v in record["factors"].items()})
    except (KeyError, TypeError, AttributeError) as error:
        raise ValueError("invalid conversion factors") from error
    if normalized != record["factors"]:
        raise ValueError("conversion factor value/hex mismatch")
    if _array_identity(arrays) != record["converted_arrays"]:
        raise ValueError("converted array identity mismatch")
    if _json(configuration) != record["converted_configuration"]:
        raise ValueError("converted static configuration mismatch")
    return record


def _static_configuration(value, factor, arrays, metadata):
    value = _json({} if value is None else value)
    if not isinstance(value, dict) or set(value) - {"cell", "cutoff", "switch_on"}:
        raise ValueError("label importer static lengths support only cell, cutoff and switch_on")
    result = dict(q_shape=list(arrays["q"].shape[1:]),
                  nstates=arrays[f"h_{metadata['carrier']}"].shape[1], basis_id=metadata["basis_id"])
    if "cell" in value:
        cell = np.asarray(value["cell"])
        if (cell.shape != (3, 3) or cell.dtype.kind not in "iuf"
                or not np.isfinite(cell).all() or np.linalg.det(cell) == 0):
            raise ValueError("static cell must be finite nonsingular real (3, 3)")
        converted = cell*factor
        if not np.isfinite(converted).all() or np.linalg.det(converted) == 0:
            raise ValueError("converted cell must remain finite and nonsingular")
        result["cell"] = converted.tolist()
    lengths = {}
    if "cutoff" in value:
        lengths["cutoff"] = _factors({"cutoff": value["cutoff"]})["cutoff"]["value"]
    if "switch_on" in value:
        switch = np.asarray(value["switch_on"])
        if (switch.shape or switch.dtype.kind not in "iuf" or not np.isfinite(switch)
                or switch < 0 or "cutoff" not in lengths or switch >= lengths["cutoff"]):
            raise ValueError("switch_on must be a finite length in [0, cutoff)")
        lengths["switch_on"] = float(switch)
    for key, length in lengths.items():
        result[key] = length*factor
        if not np.isfinite(result[key]):
            raise ValueError("converted static length overflow")
    if "cutoff" in result and result["cutoff"] <= 0:
        raise ValueError("converted cutoff must remain positive")
    if "switch_on" in result and not 0 <= result["switch_on"] < result["cutoff"]:
        raise ValueError("converted switch_on must remain below cutoff")
    return value, result


def import_labels(source_manifest, destination, *, static_configuration=None,
                  conversion_factors=None):
    """Import the existing dense label profile from eV/angstrom or hartree/bohr.

    The source manifest uses the existing label fields and checksum-bound NPZ,
    with its actual source units. Static cell/cutoff/switch lengths are supplied
    independently in those length units. Optional recorded factors contain
    exactly ``energy_to_hartree`` and ``length_to_bohr``; this permits explicit
    replay of an earlier conversion without consulting ambient constants.
    Existing atomic-unit artifacts remain readable without this metadata.
    """
    from .labels import _read_arrays, validate_labels

    source = Path(source_manifest)
    manifest_bytes = source.read_bytes()
    metadata = json.loads(manifest_bytes)
    if "ingestion" in metadata or "static_configuration" in metadata:
        raise ValueError("already imported labels must be replayed, not silently re-converted")
    raw, payload = _read_arrays(metadata, source)
    raw_units = metadata.get("units")
    if raw_units == {"energy": "eV", "length": "angstrom"}:
        default = dict(energy_to_hartree=1/units.HARTREE_EV,
                       length_to_bohr=1/units.BOHR_ANGSTROM)
    elif raw_units == {"energy": "hartree", "length": "bohr"}:
        default = dict(energy_to_hartree=1., length_to_bohr=1.)
    else:
        raise ValueError("label importer requires eV/angstrom or hartree/bohr source units")
    factors = default if conversion_factors is None else conversion_factors
    if not isinstance(factors, dict) or set(factors) != set(default):
        raise ValueError("supply exactly energy_to_hartree and length_to_bohr")
    factors = {key: item["value"] for key, item in _factors(factors).items()}
    if raw_units == {"energy": "hartree", "length": "bohr"} and factors != default:
        raise ValueError("atomic-unit labels require identity conversion factors")
    energy, length = factors["energy_to_hartree"], factors["length_to_bohr"]
    derivative = energy/length
    _factors(dict(gradient_to_hartree_per_bohr=derivative))
    arrays = {name: np.array(value, copy=True) for name, value in raw.items()}
    converted_metadata = {**metadata, "units": {"energy": "hartree", "length": "bohr"}}
    # Validate source shapes/types first without requiring atomic numerical magnitudes.
    validate_labels(arrays, converted_metadata)
    for name in ("h_electron", "h_hole", "neutral_energy"):
        if name in arrays:
            arrays[name] = arrays[name]*energy
    arrays["q"] = arrays["q"]*length
    for name in ("electronic_gradient", "neutral_force"):
        if name in arrays:
            arrays[name] = arrays[name]*derivative
    validate_labels(arrays, converted_metadata)
    raw_config, config = _static_configuration(static_configuration, length, arrays, metadata)
    record = record_conversion(raw, arrays, raw_units=raw_units,
        factors={**factors, "gradient_to_hartree_per_bohr": derivative},
        raw_configuration=raw_config, converted_configuration=config,
        source_identity=dict(manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
                             arrays_sha256=hashlib.sha256(payload).hexdigest()),
        importer_identity={Path(__file__).name: hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                           "units.py": hashlib.sha256(Path(units.__file__).read_bytes()).hexdigest(),
                           "labels.py": hashlib.sha256(Path(__file__).with_name("labels.py").read_bytes()).hexdigest(),
                           "bundles.py": hashlib.sha256(Path(__file__).with_name("bundles.py").read_bytes()).hexdigest()},
        dependency_versions={**{k: packages.version(k) for k in ("numpy", "scipy")},
                             "python": platform.python_version()})
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    converted_metadata.update(arrays_file="labels.npz",
        arrays_sha256=hashlib.sha256(buffer.getvalue()).hexdigest(), ingestion=record,
        static_configuration=config)
    validate_labels(arrays, converted_metadata)
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    raw_directory = destination/"raw"
    raw_directory.mkdir()
    (raw_directory/metadata["arrays_file"]).write_bytes(payload)
    (raw_directory/"source.json").write_bytes(manifest_bytes)
    (destination/"labels.npz").write_bytes(buffer.getvalue())
    (destination/"labels.json").write_text(json.dumps(converted_metadata, indent=2,
                                                    allow_nan=False)+"\n")
    return converted_metadata
