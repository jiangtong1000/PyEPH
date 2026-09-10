# Step 7: PyEPH post-processing

This script reads a PERTURBO `epr.h5` file and extracts real-space EPC data and
unweighted phonon eigenvectors on a time-reversal-paired q-grid. It prepares
input for subsequent localization; it does not run QCPBC or compute mobility.

## Small DNTT demo

After installing the baseline requirements, run from the repository root:

```bash
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export USE_MPI=false
python examples/01_abinitio_realspace_EPC/7_PolarEPH/run_pyeph.py \
  --epr-file pyeph/post_qe2pert/test/DNTT_epr.h5 \
  --nx 2 --ny 2 --output /tmp/pyeph_dntt_demo.h5
```

The included input is approximately 4.17 MB and has source grids q222/k222. No new DFT/DFPT calculation, MPI,
QCPBC or GPU is needed for this serial extraction. Choose an output path that
does not already exist; the script refuses to overwrite files. The small
2-by-2 grid demonstrates the interface and is not a production-converged grid.

## Expected output

For the bundled input and grid above, `freq_full` has shape `(4, 216)`,
`mode_full` has shape `(4, 216, 216)` and `mass` has shape `(72,)`.
The file also contains `gmat_raw`, electronic and phononic R-vectors, half/full
q-grids, time-reversal partner indices and the real-space grid. Frequencies
are in Ry. Eigenvectors are not mass weighted; negative interpolated
frequencies are retained at this extraction stage.

```bash
python - <<'PY'
import h5py
import numpy as np

with h5py.File('/tmp/pyeph_dntt_demo.h5', 'r') as f:
    assert f['freq_full'].shape == (4, 216)
    assert f['mode_full'].shape == (4, 216, 216)
    assert f['mass'].shape == (72,)
    for key in ('gmat_raw', 'freq_full', 'mode_full', 'mass'):
        assert np.isfinite(f[key][:]).all(), key
    nq_half = len(f['q_hbz'])
    partners = f['partner_hbz_for_minus'][:]
    modes = f['mode_full'][:]
    assert np.allclose(modes[nq_half:], modes[partners].conj())
    assert np.allclose(f['q_minus'][:], -f['q_hbz'][:][partners])
    print('DNTT extraction checks passed')
PY
```

## DNTT q332/k664 input

The companion data repository contains a separate
[DNTT EPR input (43.97 MiB)](https://github.com/JoonhoLee-Group/first-principles-transport-data/tree/master/data/epr/DNTT/q332_k664)
with q332/k664 source grids and a usage example.
It has a different cell from the bundled test fixture. The linked dataset README
also provides an extraction command that uses public PyEPH APIs.

If the two repositories are sibling directories, run from the PyEPH root:

```bash
USE_MPI=false python examples/01_abinitio_realspace_EPC/7_PolarEPH/run_pyeph.py \
  --epr-file ../first-principles-transport-data/data/epr/DNTT/q332_k664/DNTT_epr.h5 \
  --nx 2 --ny 2 --output /tmp/pyeph_dntt_q332_k664_demo.h5
```

The 2×2 interpolation grid is for a small demonstration and does not alter the
source EPR grids. This script uses `polar=False` and does not add the stored
long-range polar correction.

## Your own material

Complete stages 0-6 with your relaxed structure, pseudopotentials, k/q grids
and Wannier settings. Pass the resulting `PREFIX_epr.h5` using `--epr-file`.
The number of Wannier functions and atoms is read from that file. Set
`--nx` and `--ny` to at least 2. Validate the grid, phonon stability and
convergence for the chosen material.

## MPI

After installing MPI and `requirements-mpi.txt`, use:

```bash
USE_MPI=true mpirun -np 2 python \
  examples/01_abinitio_realspace_EPC/7_PolarEPH/run_pyeph.py \
  --epr-file pyeph/post_qe2pert/test/DNTT_epr.h5 \
  --nx 2 --ny 2 --output /tmp/pyeph_dntt_mpi_demo.h5
```

All ranks participate in the phonon calculation; only rank 0 assembles and
writes the output. The Slurm script requires local environment and scheduler
settings. Representative hardware, memory and production timing still need
to be documented.
