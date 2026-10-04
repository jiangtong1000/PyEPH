"""Data/model lifecycle failures must remain explicit and auditable."""

import json

import numpy as np
import pytest

from pyeph.learning import (
    bundle_identity, error_metrics, grouped_split, load_bundle,
    revalidate_bundle, save_bundle, validation_report,
)


def contract():
    return dict(provider="test.affine", provider_version="1", code_hashes={"provider.py": "a"*64},
                baseline_sha256="b"*64, dataset_sha256="d"*64,
                basis_kind="fixed_effective_orthonormal", basis_id="fixture:two_states",
                units=dict(energy="hartree", length="bohr"), carrier="electron",
                neutral_reference="harmonic test potential", scope="numerical fixture",
                configuration=dict(states=2))


def validation():
    return dict(scope="finite affine reference", checks=[dict(name="reference equality", passed=True)])


def arrays():
    return {"baseline/onsite": np.array([.1, .2]), "residual/weight": np.array([.3+.4j])}


def test_bundle_roundtrip_has_no_runtime_conversion_or_executable_payload(tmp_path):
    record = save_bundle(tmp_path / "bundle", arrays(), contract=contract(), validation=validation())
    restored, manifest = load_bundle(tmp_path / "bundle/bundle.json", expected_contract=contract())
    assert manifest == record
    for key, value in arrays().items():
        np.testing.assert_array_equal(restored[key], value)
        assert restored[key].dtype == value.dtype
    with pytest.raises(FileExistsError):
        save_bundle(tmp_path / "bundle", arrays(), contract=contract(), validation=validation())
    path = tmp_path / "bundle/arrays.npz"
    path.write_bytes(path.read_bytes() + b"mutation")
    with pytest.raises(ValueError, match="checksum"):
        load_bundle(tmp_path / "bundle/bundle.json", expected_contract=contract())


@pytest.mark.parametrize("field,value", [
    ("carrier", "hole"), ("basis_id", "different"), ("neutral_reference", "new potential"),
    ("baseline_sha256", "c"*64), ("dataset_sha256", "e"*64),
    ("configuration", dict(states=3)), ("code_hashes", {"provider.py": "f"*64}),
])
def test_bundle_rejects_changed_physical_or_code_contract(tmp_path, field, value):
    save_bundle(tmp_path / "bundle", arrays(), contract=contract(), validation=validation())
    with pytest.raises(ValueError, match="contract mismatch"):
        load_bundle(tmp_path / "bundle/bundle.json", expected_contract=contract() | {field: value})


def test_manifest_mutation_does_not_silently_change_reconstruction(tmp_path):
    record = save_bundle(tmp_path / "bundle", arrays(), contract=contract(), validation=validation())
    record["contract"]["configuration"]["states"] = 3
    path = tmp_path / "bundle/bundle.json"
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="manifest checksum"):
        load_bundle(path, expected_contract=contract())


@pytest.mark.parametrize("values", [{"weights": np.array([object()])},
                                    {"weights": np.array([np.nan])},
                                    {"../weights": np.ones(1)}])
def test_unsafe_arrays_rejected_before_creating_directory(tmp_path, values):
    with pytest.raises(ValueError):
        save_bundle(tmp_path / "invalid", values, contract=contract(), validation=validation())
    assert not (tmp_path / "invalid").exists()


def test_revalidation_executes_caller_checks_and_preserves_source(tmp_path):
    before = save_bundle(tmp_path / "old", arrays(), contract=contract(), validation=validation())
    source = tmp_path / "old/bundle.json"
    original = source.read_bytes()
    changed = contract() | dict(provider_version="2", code_hashes={"provider.py": "f"*64})
    calls = []
    def oracle(values, old, new):
        calls.append((old, new))
        # Independent declared affine observable against its reference.
        actual = values["baseline/onsite"].sum() + values["residual/weight"].real.sum()
        np.testing.assert_allclose(actual, .6, atol=1e-15)
        return validation() | dict(observable=float(actual))
    after = revalidate_bundle(source, tmp_path / "new", expected_contract=changed,
                              validator=oracle, reason="equivalent implementation refactor")
    assert len(calls) == 1
    assert after["migration"]["source_bundle_identity"] == before["identity"]
    assert source.read_bytes() == original
    assert before["identity"] != after["identity"]
    load_bundle(tmp_path / "new/bundle.json", expected_contract=changed)
    with pytest.raises(ValueError, match="explicitly passing"):
        revalidate_bundle(source, tmp_path / "failed", expected_contract=changed,
                          validator=lambda *args: dict(scope="bad", checks=[dict(name="reference", passed=False)]),
                          reason="failed numerical gate")
    assert not (tmp_path / "failed").exists()
    with pytest.raises(ValueError, match="only code_hashes"):
        revalidate_bundle(source, tmp_path / "units", expected_contract=contract() | dict(carrier="hole"),
                          validator=oracle, reason="convention change requires actual conversion")
    assert len(calls) == 1


def test_complex_metric_has_no_cancellation_and_no_broadcasting():
    actual = error_metrics(np.array([1j, 1.]), np.zeros(2))
    assert actual == dict(rmse=1., max_abs=1.)
    assert error_metrics(np.array([0], dtype=np.uint8), np.array([1], dtype=np.uint8))["rmse"] == 1.
    with pytest.raises(ValueError, match="identical"):
        error_metrics(np.ones((3, 2)), np.ones((3, 1)))
    with pytest.raises(ValueError, match="finite"):
        error_metrics(np.array([np.nan]), np.ones(1))


def test_family_report_names_all_geometries_and_rejects_leakage():
    groups = np.array(["train", "train", "validation", "test"])
    ids = np.array(["a", "b", "c", "d"])
    split = grouped_split(groups, validation_groups=["validation"], test_groups=["test"])
    target = {"energy": np.arange(4.)}
    predicted = {"energy": np.arange(4.) + np.array([0., 1., 2., 3.])}
    kwargs = dict(geometry_ids=ids, groups=groups, units={"energy": "hartree"}, scope="affine fixture")
    report = validation_report(predicted, target, split, **kwargs)
    assert report["splits"]["test"]["errors"]["energy"]["rmse"] == 3.
    assert report["splits"]["train"]["geometry_ids"] == ["a", "b"]
    assert len(bundle_identity(report)) == 64
    byte_report = validation_report(predicted, target, split, **(kwargs | dict(
        geometry_ids=ids.astype("S"), groups=groups.astype("S"))))
    assert json.loads(json.dumps(byte_report)) == report
    with pytest.raises(ValueError, match="leak"):
        validation_report(predicted, target, split, **(kwargs | dict(groups=np.array(["a", "b", "b", "c"]))))
    with pytest.raises(ValueError, match="exactly once"):
        validation_report(predicted, target, split | dict(test=np.array([2])), **kwargs)
