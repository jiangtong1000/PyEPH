import numpy
import h5py
from pathlib import Path
import shutil

def merge_outputs(dump_dir: Path, axes: list[str], nranks: int, safe_mode: bool = True):
    """
    Merge HDF5 files from independent ranks (trajectories) into a single file.
    
    This function consolidates current autocorrelation data from multiple parallel
    trajectories without interrupting ongoing simulations or causing IO conflicts.
    
    Args:
        dump_dir: Directory containing the HDF5 files (currents_{rank}.h5)
        axes: List of axes to merge (e.g., ['x', 'y'])
        nranks: Number of ranks (parallel trajectories/files)
        safe_mode: If True, creates temporary copies to avoid IO conflicts with
                   running simulations. If False, directly reads from original files
                   (requires all ranks to be finished).
    
    Returns:
        Path: Path to the merged output file
    
    Output Structure:
        The merged HDF5 file contains:
        - Attributes:
            - time_step: Time step size
            - total_time: Total simulation time
        - Datasets (for each axis in axes):
            - current_{axis}: Mean current autocorrelation across all ranks
            - current_{axis}_std: Standard deviation across ranks
    
    Notes:
        - In safe_mode, uses data up to the minimum step reached by all ranks
        - In non-safe_mode, requires all ranks to have reached the same step
    """
    assert all(a in ["x", "y"] for a in axes)
    
    at_steps = []
    file_paths = []
    if safe_mode:
        output_file = dump_dir / f"collected_current_autocorr_tmp.h5"
        for rank in range(nranks):
            fpath = dump_dir / f"currents_{rank}.h5"
            tmp_copy = dump_dir / f"tmp_copy_{rank}.h5"
            file_paths.append(tmp_copy)
            shutil.copy(fpath, tmp_copy)
            at_steps.append(h5py.File(tmp_copy).attrs["current_step"])
        print(f'the fastest rank reached at step {numpy.max(at_steps)}')
        print(f'the slowest rank reached at step {numpy.min(at_steps)}')
        at_step = numpy.min(at_steps)
        print(f'use data up to step {at_step} for on-the-fly analysis')
        time_step = h5py.File(tmp_copy).attrs["time_step"]
    else:
        output_file = dump_dir / f"collected_current_autocorr.h5"
        print("make sure all ranks finished, to avoid IO conflicts")
        for rank in range(nranks):
            fpath = dump_dir / f"currents_{rank}.h5"
            file_paths.append(fpath)
            at_steps.append(h5py.File(fpath).attrs["current_step"])
        assert len(numpy.unique(at_steps)) == 1, "all ranks must reach the same step"
        at_step = at_steps[0]
        time_step = h5py.File(fpath).attrs["time_step"]
        
    fa = h5py.File(output_file, "w")
    fa.attrs["time_step"] = time_step
    fa.attrs["total_time"] = at_step * time_step

    for axis_name in axes:
        data = []
        for fpath in file_paths:
            with h5py.File(fpath, "r") as f:
                data.append(f[f"current_{axis_name}"][:at_step])
        data = numpy.concatenate(data).astype(numpy.complex128)
        data = data.reshape(nranks, -1)
        data_mean = data.mean(axis=0)
        data_std = data.std(axis=0)
        fa.create_dataset(f"current_{axis_name}", data=data_mean)
        fa.create_dataset(f"current_{axis_name}_std", data=data_std)
    fa.close()
                        
    if safe_mode:
        for tmp_copy in file_paths:
            tmp_copy.unlink()
    return output_file