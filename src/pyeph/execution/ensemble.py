"""Partition independent trajectories and merge their unweighted statistics.

This host layer bounds the number of simultaneous trajectories. It retains one
batch's sampled output and the ensemble mean/M2 on the output time grid. It does
not distribute an individual Hamiltonian or automatically start MPI processes.
"""

from dataclasses import dataclass

import jax
import numpy as np

from pyeph.io.provenance import assert_matching_manifest, problem_manifest, validate_manifest
from pyeph.observables.statistics import EnsembleMoments


def _ids(values):
    ids = np.asarray(values)
    if (ids.ndim != 1 or not np.issubdtype(ids.dtype, np.integer) or
            not ids.size or np.any(ids < 0) or np.any(ids >= 2**32) or
            np.unique(ids).size != ids.size):
        raise ValueError("trajectory IDs must be a nonempty vector of unique uint32 integers")
    return ids.astype(np.uint32)


def partition_ids(trajectory_ids, rank, size):
    """Round-robin host partition; identities do not depend on process count.

    This can be called using an MPI communicator's rank/size or used for manual
    process/device partitioning. A rank with no assigned IDs returns an empty
    uint32 array and should skip its local run.
    """
    if (not isinstance(size, int) or isinstance(size, bool) or size < 1 or
            not isinstance(rank, int) or isinstance(rank, bool) or not 0 <= rank < size):
        raise ValueError("rank and size must be integers with 0 <= rank < size")
    return _ids(trajectory_ids)[rank::size]


def _same_grid(left, right):
    if left.shape != right.shape:
        return False
    scale = max(1., float(np.max(np.abs(left))), float(np.max(np.abs(right))))
    dtype = np.result_type(left.dtype, right.dtype)
    eps = np.finfo(dtype if dtype.kind == "f" else np.float64).eps
    return np.allclose(left, right, rtol=0., atol=16*eps*scale)


@dataclass(frozen=True)
class EnsembleResult:
    """Unweighted independent-trajectory moments on a shared physical time grid.

    Complex M2 is sum(abs(x-mean)**2). `standard_error` is a trajectory sampling
    error, not a time-correlation-adjusted error or an importance-weighted error.
    `preparation_id` identifies the distribution, initialization code and seed;
    each stable trajectory ID identifies a distinct draw from that preparation.
    """

    times: object
    trajectory_ids: object
    mean: object
    m2: object
    simulation_manifest: dict
    preparation_id: str

    def validate(self):
        """Check reconstructed/distributed result data before statistical merging."""
        _ids(self.trajectory_ids)
        times = np.asarray(self.times)
        if (times.ndim != 1 or not times.size or times.dtype.kind not in "iuf" or
                not np.isfinite(times).all() or np.any(np.diff(times) <= 0)):
            raise ValueError("ensemble times must be finite, real and strictly increasing")
        if not isinstance(self.preparation_id, str) or not self.preparation_id.strip():
            raise ValueError("ensemble preparation_id must be nonempty")
        validate_manifest(self.simulation_manifest)
        means, structure = jax.tree.flatten(self.mean)
        squares, other_structure = jax.tree.flatten(self.m2)
        if not means or structure != other_structure:
            raise ValueError("ensemble mean and M2 must have matching nonempty structures")
        for mean, m2 in zip(means, squares, strict=True):
            mean, m2 = np.asarray(mean), np.asarray(m2)
            if (mean.ndim < 1 or mean.shape[0] != len(times) or mean.dtype.kind not in "biufc"
                    or not np.isfinite(mean).all()):
                raise ValueError("observable means must be finite and match the output time axis")
            if (m2.shape != mean.shape or m2.dtype.kind not in "iuf" or
                    not np.isfinite(m2).all() or np.any(m2 < 0)):
                raise ValueError("M2 must be real, nonnegative and match the mean shape")
        return self

    @property
    def count(self):
        return len(self.trajectory_ids)

    @property
    def variance(self):
        if self.count < 2:
            return jax.tree.map(lambda x: np.full(np.shape(x), np.nan), self.m2)
        return jax.tree.map(lambda x: x / (self.count - 1), self.m2)

    @property
    def standard_error(self):
        return jax.tree.map(lambda x: np.sqrt(x / self.count), self.variance)


def _from_batch(result, ids, manifest, preparation_id):
    times = np.asarray(result.times)
    if (times.ndim != 2 or times.shape[1] != len(ids) or times.shape[0] == 0 or
            not np.isfinite(times).all()):
        raise ValueError("ensemble batches must provide finite time-by-trajectory output")
    grid = times[:, 0]
    for index in range(1, len(ids)):
        if not _same_grid(grid, times[:, index]):
            raise ValueError("ensemble trajectories must share a physical output time grid")
    leaves, structure = jax.tree.flatten(result.observables)
    if not leaves:
        raise ValueError("ensemble measurements must contain at least one numerical observable")
    means, squares = [], []
    for value in leaves:
        value = np.asarray(value)
        if value.shape[:2] != times.shape or value.dtype.kind not in "biufc":
            raise ValueError("ensemble observations must have leading time and trajectory axes")
        moments = EnsembleMoments().update(value, axis=1)
        means.append(moments.mean)
        squares.append(moments.m2)
    return EnsembleResult(grid.copy(), ids.copy(), structure.unflatten(means),
                          structure.unflatten(squares), manifest, preparation_id)


def merge_ensembles(left, right):
    """Merge disjoint partitions after checking physics, preparation and time grid."""
    left.validate()
    right.validate()
    assert_matching_manifest(left.simulation_manifest, right.simulation_manifest)
    if left.preparation_id != right.preparation_id:
        raise ValueError("cannot merge different ensemble preparations")
    left_ids, right_ids = _ids(left.trajectory_ids), _ids(right.trajectory_ids)
    if np.intersect1d(left_ids, right_ids).size:
        raise ValueError("cannot merge duplicate trajectory IDs")
    if not _same_grid(np.asarray(left.times), np.asarray(right.times)):
        raise ValueError("cannot merge different physical output time grids")
    a, tree = jax.tree.flatten(left.mean)
    b, other_tree = jax.tree.flatten(right.mean)
    a2, tree2 = jax.tree.flatten(left.m2)
    b2, other_tree2 = jax.tree.flatten(right.m2)
    if not tree == other_tree == tree2 == other_tree2:
        raise ValueError("cannot merge different observable structures")
    means, squares = [], []
    for x, y, x2, y2 in zip(a, b, a2, b2, strict=True):
        if any(not np.isfinite(z).all() for z in (x, y, x2, y2)):
            raise ValueError("cannot merge nonfinite moments")
        if np.shape(x) != np.shape(x2) or np.shape(y) != np.shape(y2):
            raise ValueError("moment and mean shapes differ")
        merged = EnsembleMoments(len(left_ids), np.asarray(x), np.asarray(x2)).merge(
            EnsembleMoments(len(right_ids), np.asarray(y), np.asarray(y2)))
        means.append(merged.mean)
        squares.append(merged.m2)
    return EnsembleResult(np.asarray(left.times).copy(), np.concatenate((left_ids, right_ids)),
                          tree.unflatten(means), tree.unflatten(squares),
                          left.simulation_manifest, left.preparation_id)


def run_ensemble(simulation, initialize, trajectory_ids, steps, *, batch_size=32,
                 preparation_id, artifact_ids=None, batch_observer=None):
    """Initialize/run batches and retain moments rather than all trajectories.

    `initialize(ids)` returns a stacked TrajectoryState with exactly those IDs
    in order, including a leading axis for a one-trajectory batch. Initializers
    must derive random draws from these identities, not mutable global RNGs.
    `preparation_id` is a caller-owned revision covering initialization code,
    distribution parameters and seed. Strict model provenance also applies.

    Optional `batch_observer(ids, run_result)` can save final states or raw
    measurements. Output schedules come from the simulation's Execution policy.
    Batch size may change across reruns without changing the ensemble.
    """
    ids = _ids(trajectory_ids)
    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    if not isinstance(preparation_id, str) or not preparation_id.strip():
        raise ValueError("preparation_id must identify initialization code, distribution and seed")
    manifest = problem_manifest(simulation.problem, simulation.integrator, artifact_ids=artifact_ids)
    validate_manifest(manifest)
    ensemble = None
    for start in range(0, len(ids), batch_size):
        batch_ids = ids[start:start + batch_size]
        initial = initialize(batch_ids)
        if (np.ndim(initial.time) != 1 or
                not np.array_equal(np.asarray(initial.trajectory_id), batch_ids)):
            raise ValueError("initializer must return a batch with the requested trajectory IDs in order")
        result = simulation.run(initial, steps)
        batch = _from_batch(result, batch_ids, manifest, preparation_id)
        ensemble = batch if ensemble is None else merge_ensembles(ensemble, batch)
        if batch_observer is not None:
            batch_observer(batch_ids.copy(), result)
    assert_matching_manifest(manifest, problem_manifest(
        simulation.problem, simulation.integrator, artifact_ids=artifact_ids))
    return ensemble
