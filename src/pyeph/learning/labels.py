"""Small fixed-basis training labels with explicit physical provenance.

This host helper loads dense reference labels, not a universal Hamiltonian
representation. Production providers still supply the existing model operations.
Metadata declares the teacher's assumptions; checking it does not establish
that an AO-derived target is a physically exact global diabatic Hamiltonian.
"""

import hashlib
import io
import json
from pathlib import Path

import numpy as np


SCHEMA = "pyeph.fixed_basis_labels.v1"


def validate_labels(arrays, metadata):
    """Check an atomic-unit, fixed-effective-basis dataset without filtering it.

    q: (samples, atoms, 3), h_hole/h_electron: (samples, states, states),
    electronic_gradient: (samples, states, states, atoms, 3),
    neutral_energy: (samples,), neutral_force: (samples, atoms, 3).
    Gradients are positive energy derivatives; forces carry the minus sign.
    Geometry/group IDs are strings. No missing electronic labels are imputed.
    """
    if metadata.get("schema") != SCHEMA:
        raise ValueError(f"schema must be {SCHEMA}")
    if metadata.get("units") != {"energy": "hartree", "length": "bohr"}:
        raise ValueError("convert labels explicitly to hartree and bohr before loading")
    if metadata.get("basis_kind") != "fixed_effective_orthonormal":
        raise ValueError("this loader requires a declared fixed effective orthonormal basis")
    for key in ("basis_id", "electronic_energy_definition", "phase_convention",
                "neutral_reference", "label_scope"):
        if not isinstance(metadata.get(key), str) or not metadata[key].strip():
            raise ValueError(f"metadata requires a nonempty {key}")
    if (not isinstance(metadata.get("sources"), list) or not metadata["sources"]
            or any(not isinstance(source, str) or not source.strip()
                   for source in metadata["sources"])):
        raise ValueError("metadata requires source provenance")
    carrier = metadata.get("carrier")
    if carrier not in {"hole", "electron"}:
        raise ValueError("carrier must explicitly be hole or electron")
    key = f"h_{carrier}"
    required = {"q", key, "species", "fragment", "geometry_ids", "groups"}
    if not required <= arrays.keys():
        raise ValueError(f"missing label arrays: {sorted(required-arrays.keys())}")
    allowed = required | {"electronic_gradient", "neutral_energy", "neutral_force"}
    if arrays.keys() - allowed:
        raise ValueError(f"unknown label arrays: {sorted(arrays.keys()-allowed)}")

    q, h = np.asarray(arrays["q"]), np.asarray(arrays[key])
    if q.ndim != 3 or q.shape[0] == 0 or q.shape[1] == 0 or q.shape[2] != 3:
        raise ValueError("q must have shape (samples, atoms, 3) with nonempty samples/atoms")
    samples, atoms, _ = q.shape
    if h.ndim != 3 or h.shape[0] != samples or h.shape[1] == 0 or h.shape[1] != h.shape[2]:
        raise ValueError("electronic labels must be square matrices with one per geometry")
    states = h.shape[1]
    shapes = {"q": q.shape, key: h.shape, "species": (atoms,), "fragment": (atoms,),
              "electronic_gradient": (samples, states, states, atoms, 3),
              "neutral_energy": (samples,), "neutral_force": q.shape}
    for name, shape in shapes.items():
        if name not in arrays:
            continue
        value = np.asarray(arrays[name])
        kinds = "iu" if name in {"species", "fragment"} else "fc" if name in {
            key, "electronic_gradient"} else "f"
        if value.shape != shape or value.dtype.kind not in kinds or not np.all(np.isfinite(value)):
            raise ValueError(f"{name} must have finite numeric values of shape {shape}")
    if np.any(np.asarray(arrays["species"]) < 1) or np.any(np.asarray(arrays["species"]) > 118):
        raise ValueError("species must be atomic numbers from 1 through 118")
    if np.any(np.asarray(arrays["fragment"]) < 0):
        raise ValueError("fragment assignments must be nonnegative")
    for name in ("geometry_ids", "groups"):
        values = np.asarray(arrays[name])
        if (values.shape != (samples,) or values.dtype.kind not in "US"
                or any(not str(v).strip() for v in values.astype(str))):
            raise ValueError(f"{name} must contain a nonempty string for every geometry")
    if len(set(np.asarray(arrays["geometry_ids"]).astype(str))) != samples:
        raise ValueError("geometry_ids must be unique")
    if not np.allclose(h, h.conj().swapaxes(1, 2), rtol=1e-11, atol=1e-12):
        raise ValueError("electronic labels must already be Hermitian")
    if "electronic_gradient" in arrays:
        gradient = np.asarray(arrays["electronic_gradient"])
        if not np.allclose(gradient, gradient.conj().swapaxes(1, 2), rtol=1e-10, atol=1e-11):
            raise ValueError("electronic derivative labels must already be Hermitian")
    if "neutral_force" in arrays and "neutral_energy" not in arrays:
        raise ValueError("neutral forces require the corresponding reference energies")
    return {name: np.array(value, copy=True) for name, value in arrays.items()}


def load_labels(manifest_path):
    """Load a JSON manifest and its checksum-bound NPZ; never enable pickle."""
    path = Path(manifest_path)
    metadata = json.loads(path.read_text())
    filename = metadata.get("arrays_file")
    if (not isinstance(filename, str) or not filename or Path(filename).name != filename
            or Path(filename).suffix != ".npz" or "\\" in filename):
        raise ValueError("arrays_file must name a sibling NPZ file")
    payload = path.parent / filename
    if payload.is_symlink():
        raise ValueError("arrays_file must be a local sibling NPZ, not a symlink")
    data = payload.read_bytes()
    if hashlib.sha256(data).hexdigest() != metadata.get("arrays_sha256"):
        raise ValueError("label array checksum mismatch")
    with np.load(io.BytesIO(data), allow_pickle=False) as archive:
        if len(set(archive.files)) != len(archive.files):
            raise ValueError("label archive contains duplicate array names")
        arrays = {name: archive[name] for name in archive.files}
    return validate_labels(arrays, metadata), metadata


def grouped_split(groups, *, validation_groups, test_groups):
    """Hold out complete structural families/trajectories, never adjacent rows."""
    groups = np.asarray(groups)
    if groups.ndim != 1 or groups.size == 0 or groups.dtype.kind not in "US":
        raise ValueError("groups must be a nonempty one-dimensional string array")
    groups = groups.astype(str)
    if any(not value.strip() for value in groups):
        raise ValueError("groups must contain nonempty labels")
    if isinstance(validation_groups, (str, bytes)) or isinstance(test_groups, (str, bytes)):
        raise ValueError("holdout groups must be collections, not a single string")
    available = set(groups)
    validation, test = set(validation_groups), set(test_groups)
    if not validation or not test or validation & test or not (validation | test) <= available:
        raise ValueError("validation/test groups must be nonempty, disjoint and present")
    masks = {"train": ~np.isin(groups, list(validation | test)),
             "validation": np.isin(groups, list(validation)), "test": np.isin(groups, list(test))}
    if any(not np.any(mask) for mask in masks.values()):
        raise ValueError("each split must contain at least one geometry")
    return {name: np.flatnonzero(mask) for name, mask in masks.items()}
