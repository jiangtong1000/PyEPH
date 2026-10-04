"""Opt-in host observation of a declared descriptor envelope during dynamics."""

import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np

from .bundles import bundle_identity


class DomainViolation(RuntimeError):
    """A sampled geometry exceeded declared bounds; its diagnostic was saved."""

    def __init__(self, record_path):
        self.record_path = Path(record_path)
        super().__init__(f"sampled geometry outside declared descriptor domain: {record_path}")


class GeometryDomainMonitor:
    """Runner observer that saves violating geometries before raising.

    ``descriptor(q)`` is trusted caller code returning a real vector for one
    geometry. Bounds are a declared coverage diagnostic, not a calibrated
    uncertainty estimate. ``values['q']`` must be saved by the measurement.
    Batched observations require stable trajectory IDs, in runner batch order.

    Only output samples are inspected, after their chunk executes. For checks
    after every accepted step use save_every=1 and chunk_size=1; intermediate
    integrator stages remain unchecked. A domain report is not a restart state.
    """

    def __init__(self, directory, descriptor, lower, upper, *, descriptor_id,
                 metadata, trajectory_ids=None):
        lower, upper = np.asarray(lower), np.asarray(upper)
        if (not callable(descriptor) or lower.ndim != 1 or not lower.size
                or upper.shape != lower.shape
                or any(a.dtype.kind not in "iuf" or not np.isfinite(a).all()
                       for a in (lower, upper)) or np.any(lower > upper)):
            raise ValueError("domain requires a descriptor and finite ordered vector bounds")
        if not isinstance(descriptor_id, str) or not descriptor_id.strip():
            raise ValueError("domain requires an explicit descriptor identity")
        if not isinstance(metadata, dict) or not metadata:
            raise ValueError("domain requires caller-owned model/data context")
        self.metadata = json.loads(json.dumps(metadata, allow_nan=False))
        self.directory, self.descriptor = Path(directory), descriptor
        self.lower, self.upper = np.array(lower, copy=True), np.array(upper, copy=True)
        self.descriptor_id = descriptor_id
        self.trajectory_ids = None if trajectory_ids is None else np.asarray(trajectory_ids)
        if self.trajectory_ids is not None:
            ids = self.trajectory_ids
            if (ids.ndim != 1 or not ids.size or ids.dtype.kind not in "iu"
                    or np.any(ids < 0) or len(set(ids.tolist())) != len(ids)):
                raise ValueError("trajectory IDs must be unique nonnegative integers")
            self.trajectory_ids = ids.copy()

    def __call__(self, times, values):
        if "q" not in values:
            raise ValueError("domain observation requires saved atomic coordinates named q")
        q, times = np.asarray(values["q"]), np.asarray(times)
        batched = q.ndim == 4
        if (q.ndim not in (3, 4) or q.shape[-1] != 3 or q.shape[-2] == 0
                or q.dtype.kind not in "f" or times.dtype.kind not in "iuf"
                or times.shape != q.shape[:-2]):
            raise ValueError("domain expects q=(frames,[batch,]atoms,3) and matching times")
        if not q.size:
            return
        if (batched and (self.trajectory_ids is None
                         or len(self.trajectory_ids) != q.shape[1])):
            raise ValueError("batched domain observation requires matching stable trajectory IDs")
        if not batched and self.trajectory_ids is not None and len(self.trajectory_ids) != 1:
            raise ValueError("single-trajectory domain observation accepts at most one trajectory ID")
        flat = q.reshape((-1,) + q.shape[-2:])
        descriptors, failed, reasons = [], [], []
        flat_times = times.reshape(-1)
        for index, geometry in enumerate(flat):
            reason = []
            if not np.isfinite(geometry).all():
                reason.append("nonfinite geometry")
                feature = np.full(self.lower.shape, np.nan)
            else:
                try:
                    feature = np.asarray(self.descriptor(geometry))
                    if feature.shape != self.lower.shape or feature.dtype.kind not in "iuf":
                        raise ValueError("descriptor must return the declared real vector shape")
                except Exception as exc:
                    reason.append(f"descriptor failed: {type(exc).__name__}: {exc}")
                    feature = np.full(self.lower.shape, np.nan)
            descriptors.append(feature)
            if not np.isfinite(flat_times[index]):
                reason.append("nonfinite observation time")
            if not np.isfinite(feature).all():
                reason.append("nonfinite descriptor")
            elif np.any(feature < self.lower) or np.any(feature > self.upper):
                reason.append("descriptor outside declared bounds")
            if reason:
                failed.append(index)
                reasons.append(dict(sample_index=index, reasons=reason))
        if not failed:
            return
        indices = np.asarray(failed, dtype=np.int64)
        payload = dict(q=flat[indices], descriptors=np.asarray(descriptors)[indices],
                       times=times.reshape(-1)[indices], sample_indices=indices,
                       lower=self.lower, upper=self.upper)
        if self.trajectory_ids is not None:
            batch_indices = indices % q.shape[1] if batched else np.zeros(len(indices), dtype=int)
            payload["trajectory_ids"] = self.trajectory_ids[batch_indices]
        self.directory.mkdir(parents=True, exist_ok=True)
        output = Path(tempfile.mkdtemp(prefix="domain-", dir=self.directory))
        np.savez_compressed(output / "geometry.npz", **payload)
        record = dict(schema="pyeph.domain_violation.v1", descriptor_id=self.descriptor_id,
                      context=self.metadata, violating_samples=len(indices),
                      failures=reasons,
                      observation_shape=list(q.shape), arrays_file="geometry.npz",
                      arrays_sha256=hashlib.sha256((output / "geometry.npz").read_bytes()).hexdigest(),
                      scope="descriptor-envelope violation at sampled output; not calibrated uncertainty")
        record["identity"] = bundle_identity(record)
        path = output / "report.json"
        path.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
        raise DomainViolation(path)
