"""Host domain diagnostics retain the offending geometry and stable identity."""

import hashlib
import json

import numpy as np
import pytest

from pyeph.learning import DomainViolation, GeometryDomainMonitor


def descriptor(q):
    return np.array([np.linalg.norm(q[1] - q[0])])


def monitor(path, **kwargs):
    return GeometryDomainMonitor(path, descriptor, [1.], [2.], descriptor_id="bond-distance:v1",
                                 metadata=dict(model="fixture", units="bohr"), **kwargs)


def test_out_of_domain_batch_preserves_only_failed_geometries_and_ids(tmp_path):
    q = np.zeros((2, 2, 2, 3))
    q[:, :, 1, 0] = [[1.5, 1.6], [1.4, 2.1]]
    times = np.array([[0., 0.], [1., 1.]])
    watch = monitor(tmp_path, trajectory_ids=[8, 91])
    watch(times[:1], dict(q=q[:1]))
    assert list(tmp_path.iterdir()) == []
    with pytest.raises(DomainViolation) as caught:
        watch(times, dict(q=q))
    report_path = caught.value.record_path
    record = json.loads(report_path.read_text())
    payload = report_path.parent / record["arrays_file"]
    assert hashlib.sha256(payload.read_bytes()).hexdigest() == record["arrays_sha256"]
    with np.load(payload, allow_pickle=False) as data:
        np.testing.assert_array_equal(data["q"], q[1:, 1])
        np.testing.assert_array_equal(data["trajectory_ids"], [91])
        np.testing.assert_array_equal(data["times"], [1.])
    with pytest.raises(DomainViolation) as again:
        watch(times, dict(q=q))
    assert again.value.record_path != report_path
    assert report_path.exists()


def test_missing_coordinates_or_trajectory_ids_never_disable_monitor(tmp_path):
    watch = monitor(tmp_path)
    with pytest.raises(ValueError, match="coordinates"):
        watch(np.zeros(1), {})
    with pytest.raises(ValueError, match="trajectory IDs"):
        watch(np.zeros((1, 2)), dict(q=np.zeros((1, 2, 2, 3))))
    with pytest.raises(ValueError, match="ordered"):
        GeometryDomainMonitor(tmp_path, descriptor, [2.], [1.], descriptor_id="x", metadata={"x": 1})


def test_nonfinite_descriptor_is_saved_as_failure_not_repaired(tmp_path):
    watch = GeometryDomainMonitor(tmp_path, lambda q: np.array([np.nan]), [1.], [2.],
                                  descriptor_id="nonfinite-fixture", metadata={"model": "fixture"})
    with pytest.raises(DomainViolation) as caught:
        watch(np.zeros(1), dict(q=np.zeros((1, 2, 3))))
    with np.load(caught.value.record_path.parent / "geometry.npz", allow_pickle=False) as data:
        assert np.isnan(data["descriptors"][0, 0])


@pytest.mark.parametrize("invalid", ["q", "time", "descriptor_exception"])
def test_invalid_geometry_or_time_is_retained_even_with_constant_descriptor(tmp_path, invalid):
    q = np.zeros((1, 2, 3))
    times = np.zeros(1)
    def feature(geometry):
        if invalid == "descriptor_exception":
            raise ValueError("degenerate local frame")
        return np.array([1.5])
    if invalid == "q":
        q[0, 0, 0] = np.nan
    if invalid == "time":
        times[0] = np.nan
    watch = GeometryDomainMonitor(tmp_path, feature, [1.], [2.], descriptor_id="constant-fixture",
                                  metadata={"model": "fixture"})
    with pytest.raises(DomainViolation) as caught:
        watch(times, dict(q=q))
    report = json.loads(caught.value.record_path.read_text())
    assert report["failures"][0]["reasons"]
    with np.load(caught.value.record_path.parent / "geometry.npz", allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved["q"], q)
        np.testing.assert_array_equal(saved["times"], times)
