"""Reference-label ingestion must preserve identity, units and family holdouts."""

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


_PATH = Path(__file__).resolve().parents[1] / "examples/materials_data.py"
_SPEC = importlib.util.spec_from_file_location("materials_data", _PATH)
data = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(data)


def labels():
    q = np.arange(36, dtype=float).reshape(4, 3, 3) / 10
    h = np.array([[[.1, .02j], [-.02j, -.1]]] * 4)
    arrays = dict(q=q, h_hole=h, species=np.array([6, 1, 1]), fragment=np.array([0, 0, 1]),
                  geometry_ids=np.array(["a", "b", "c", "d"]),
                  groups=np.array(["packing0", "packing0", "packing1", "packing2"]),
                  neutral_energy=np.ones(4), neutral_force=np.zeros_like(q),
                  electronic_gradient=np.zeros((4, 2, 2, 3, 3), dtype=complex))
    metadata = dict(schema=data.SCHEMA, units=dict(energy="hartree", length="bohr"),
                    basis_kind="fixed_effective_orthonormal", basis_id="fixture:two_states",
                    electronic_energy_definition="explicit Hermitian test matrix",
                    phase_convention="fixed labels", neutral_reference="constant test energy",
                    label_scope="numerical fixture", sources=["test_materials_data.py"], carrier="hole")
    return arrays, metadata


def test_complex_labels_and_derivatives_round_trip_are_not_projected(tmp_path):
    arrays, metadata = labels()
    payload = tmp_path / "labels.npz"
    np.savez(payload, **arrays)
    metadata.update(arrays_file=payload.name,
                    arrays_sha256=hashlib.sha256(payload.read_bytes()).hexdigest())
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(metadata))
    restored, description = data.load_labels(manifest)
    assert description == metadata
    for name, value in arrays.items():
        np.testing.assert_array_equal(restored[name], value)
    assert np.iscomplexobj(restored["h_hole"])
    changed = data.validate_labels(arrays, metadata)
    changed["q"][0, 0, 0] += 1
    np.testing.assert_array_equal(arrays["q"], restored["q"])
    payload.write_bytes(payload.read_bytes() + b"unexpected mutation")
    with pytest.raises(ValueError, match="checksum"):
        data.load_labels(manifest)


def test_labels_do_not_silently_repair_missing_or_incompatible_physics():
    arrays, metadata = labels()
    for key, value in (("units", dict(energy="eV", length="angstrom")),
                       ("basis_kind", "moving_nonorthogonal_AO"),
                       ("phase_convention", ""), ("sources", []), ("carrier", "unspecified")):
        with pytest.raises(ValueError):
            data.validate_labels(arrays, metadata | {key: value})
    for key, value in (("h_hole", np.ones((4, 2, 3))),
                       ("q", arrays["q"].astype(complex)),
                       ("species", np.array([6, 0, 1])),
                       ("fragment", np.array([0, -1, 1])),
                       ("geometry_ids", np.array(["same"] * 4)),
                       ("groups", np.array(["", "a", "b", "c"]))):
        with pytest.raises(ValueError):
            data.validate_labels(arrays | {key: value}, metadata)
    h = arrays["h_hole"].copy()
    h[0, 0, 1] = .4
    with pytest.raises(ValueError, match="Hermitian"):
        data.validate_labels(arrays | {"h_hole": h}, metadata)
    gradient = arrays["electronic_gradient"].copy()
    gradient[0, 0, 1, 0, 0] = 1
    with pytest.raises(ValueError, match="derivative labels"):
        data.validate_labels(arrays | {"electronic_gradient": gradient}, metadata)
    with pytest.raises(ValueError, match="reference energies"):
        data.validate_labels({k: v for k, v in arrays.items() if k != "neutral_energy"}, metadata)


def test_holdouts_remain_disjoint_when_group_has_multiple_frames():
    arrays, _ = labels()
    split = data.grouped_split(arrays["groups"], validation_groups=["packing1"], test_groups=["packing2"])
    np.testing.assert_array_equal(split["train"], [0, 1])
    np.testing.assert_array_equal(split["validation"], [2])
    np.testing.assert_array_equal(split["test"], [3])
    for validation, test in ((["packing1"], ["packing1"]), (["absent"], ["packing2"]),
                             ([], ["packing2"]), (["packing0", "packing1"], ["packing2"]),
                             ("packing1", ["packing2"])):
        with pytest.raises(ValueError):
            data.grouped_split(arrays["groups"], validation_groups=validation, test_groups=test)


@pytest.mark.parametrize("filename", ["../labels.npz", "/tmp/labels.npz", "", ".", "labels.npy"])
def test_manifest_cannot_redirect_the_payload(tmp_path, filename):
    _, metadata = labels()
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(metadata | {"arrays_file": filename}))
    with pytest.raises(ValueError, match="sibling NPZ"):
        data.load_labels(manifest)
