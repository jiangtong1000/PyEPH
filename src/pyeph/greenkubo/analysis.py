"""Read historical rank-current HDF5 files without coupling I/O to dynamics."""

from pathlib import Path

import h5py
import numpy as np


def merge_outputs(dump_dir, axes, nranks, safe_mode=True):
    """Merge equal-size rank ensembles using the legacy dataset layout.

    ``current_*_std`` retains its historical meaning: standard deviation of
    rank means (zero for serial). It is not an uncertainty across trajectories.
    Safe mode truncates snapshots to the shortest completed rank prefix.
    """
    directory = Path(dump_dir)
    if nranks < 1 or not axes or any(axis not in ("x", "y") for axis in axes):
        raise ValueError("merge requires positive nranks and x/y axes")
    records, steps, intervals, grids = [], [], [], []
    for rank in range(nranks):
        with h5py.File(directory/f"currents_{rank}.h5", "r") as handle:
            completed = int(handle.attrs["current_step"])
            if completed < 0 or any(completed > len(handle[f"current_{axis}"]) for axis in axes):
                raise ValueError("rank file reports an invalid completed prefix")
            steps.append(completed)
            intervals.append(float(handle.attrs["time_step"]))
            initial_time = float(handle.attrs.get("initial_time", 0.))
            grid = (handle["time"][:completed] if "time" in handle else
                    initial_time+np.arange(completed)*intervals[-1])
            if grid.shape != (completed,) or not np.isfinite(grid).all():
                raise ValueError("rank file has an invalid time grid")
            grids.append(grid)
            records.append({axis: handle[f"current_{axis}"][:completed] for axis in axes})
    if len(set(intervals)) != 1:
        raise ValueError("rank files have different time steps")
    if not safe_mode and len(set(steps)) != 1:
        raise ValueError("all ranks must reach the same step for a final merge")
    count = min(steps)
    if any(not np.allclose(grid[:count], grids[0][:count], rtol=0, atol=1e-12) for grid in grids[1:]):
        raise ValueError("rank files have different observation time grids")
    initial_time = float(grids[0][0]) if count else 0.
    destination = directory/("collected_current_autocorr_tmp.h5" if safe_mode else "collected_current_autocorr.h5")
    with h5py.File(destination, "w") as handle:
        handle.attrs["time_step"] = intervals[0]
        # Retain old sample-count * dt metadata, even though final sample time
        # is (sample-count-1)*dt for an initial-time-inclusive trajectory.
        handle.attrs["total_time"] = count*intervals[0]
        handle.attrs["current_step"] = count
        handle.attrs["initial_time"] = initial_time
        handle.attrs["final_time"] = float(grids[0][count-1]) if count else initial_time
        # Preserve ordinary legacy dataset keys exactly. Resumed segments need
        # an explicit time array so their C(t) cannot be mistaken for a new t=0.
        if initial_time != 0.:
            handle["time"] = grids[0][:count]
        handle.attrs["std_definition"] = "population standard deviation of equal-size rank means"
        for axis in axes:
            values = np.asarray([record[axis][:count] for record in records], dtype=complex)
            handle[f"current_{axis}"] = values.mean(axis=0)
            handle[f"current_{axis}_std"] = values.std(axis=0)
    return destination
