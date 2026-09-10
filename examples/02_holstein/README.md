# Holstein charge-dynamics demo

This self-contained example constructs a one-dimensional model and computes
its current autocorrelation. It does not reproduce a complete molecular-crystal
production calculation. Inputs are generated in `run.py`; no data download,
electronic-structure program, QCPBC installation or GPU is required.

## Run on one CPU process

After installing the baseline requirements, run from the repository root:

```bash
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export USE_MPI=false
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
cd examples/02_holstein
python run.py 0
```

The argument selects a temperature; `0` means approximately 183.85 K.
Defaults are 42 sites, 50 trajectories per rank, a time step of 0.01 and a
total time of 50 in reduced model units (5,000 steps). To retain several
runs, use separate working directories: the script writes to
`currents_output/` in the current working directory.

## Verify the result

```bash
python - <<'PY'
import h5py
import numpy as np

with h5py.File('currents_output/collected_current_autocorr.h5', 'r') as f:
    assert f['current_x'].shape == (5000,)
    assert f['current_x_std'].shape == (5000,)
    assert np.isfinite(f['current_x'][:]).all()
    assert np.isfinite(f['current_x_std'][:]).all()
    assert np.isclose(f.attrs['time_step'], 0.01)
    assert np.isclose(f.attrs['total_time'], 50.0)
    print('Output checks passed; C_x(0) =', f['current_x'][0])
PY
```

`current_x_std` is the variation across rank-averaged currents. It is zero
for one rank and is not a trajectory-level sampling error estimate. The
simulation uses base seed 1120, with separate streams for each rank.
Changing the rank count changes both the random streams and sample count.

## Observed timing

An audit on macOS 26.6.2 (arm64), Python 3.12.8 took approximately 28 seconds
to install the original baseline requirements and 48 seconds to run the
complete serial demo with single-thread BLAS. This is one observation:
network, package caching and first-use compilation affect timings. The timed
installation excluded external electronic-structure programs, MPI and QCPBC,
and preceded addition of the small `psutil` dependency. CPU model and RAM
were not recorded; production resource requirements still need documentation.

The audit used NumPy 2.5.3, SciPy 1.18.1, h5py 3.16.0 and Numba 0.67.0.
The observed `current_x[0]` was `0.6567366510024867+0j`; it has not been
established as a cross-platform numerical reference.

## MPI

Install a working MPI implementation and `requirements-mpi.txt`, then run:

```bash
USE_MPI=true mpirun -np 2 python run.py 0
```

This uses 50 trajectories per rank (100 in total). The Slurm script is a
template: replace its partition, environment and resource requests.
