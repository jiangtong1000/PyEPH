"""Integration tests for eph mat calculations."""

import pytest
import numpy
import h5py
from pathlib import Path

from pyeph.post_qe2pert import parse_qpoint_path, CalcEphMatReciprocal
from pyeph.utils.constants import ryd_to_mev
from pyeph.utils.logger import get_mpi_rank, get_mpi_info
import pytest

def test_ephmat(size_mpi=4, size_serial=2):
    repo_root = Path(__file__).resolve().parents[0]
    epr_fname = repo_root / "DNTT_epr.h5"
    qpoint_path_string = """11
    0.0000  0.0000  0.0000   50
    0.0000  0.5000  0.0000   50
    0.0000  0.5000  0.5000   50
    0.0000  0.0000  0.5000   50
    0.0000  0.0000  0.0000   50
    -0.5000  0.0000  0.5000  50
    -0.5000  0.5000  0.5000  50
    0.0000  0.5000  0.0000   50
    -0.5000  0.5000  0.0000  50
    -0.5000  0.0000  0.0000  50
    0.0000  0.0000  0.0000    1"""

    kpoints = parse_qpoint_path(qpoint_path_string)
    qpoints = kpoints

    numpy.random.seed(0)

    mpi_info = get_mpi_info()
    if mpi_info['has_mpi'] and mpi_info['size'] > 1:
        comm = mpi_info['comm']
        if mpi_info['rank'] == 0:
            random_qpoint_idx = numpy.random.choice(len(qpoints), size=10, replace=False)
            random_kpoint_idx = numpy.random.choice(len(kpoints), size=10, replace=False)
        else:
            random_qpoint_idx = None
            random_kpoint_idx = None
        random_qpoint_idx = comm.bcast(random_qpoint_idx, root=0)
        random_kpoint_idx = comm.bcast(random_kpoint_idx, root=0)
    else:
        random_qpoint_idx = numpy.random.choice(len(qpoints), size=10, replace=False)
        random_kpoint_idx = numpy.random.choice(len(kpoints), size=10, replace=False)

    # DNTT_ephmat.h5 corresponds to these random kpoints and qpoints
    kpoints = kpoints[random_kpoint_idx]
    qpoints = qpoints[random_qpoint_idx]
    
    if mpi_info['has_mpi'] and mpi_info['size'] > 1:
        size = size_mpi
        comm = mpi_info['comm']
        if mpi_info['rank'] == 0:
            random_qpoint_idx = numpy.random.choice(len(qpoints), size=size, replace=False)
            random_kpoint_idx = numpy.random.choice(len(kpoints), size=size, replace=False)
        else:
            random_qpoint_idx = None
            random_kpoint_idx = None
        random_qpoint_idx = comm.bcast(random_qpoint_idx, root=0)
        random_kpoint_idx = comm.bcast(random_kpoint_idx, root=0)
    else:
        size = size_serial
        random_qpoint_idx = numpy.random.choice(len(qpoints), size=size, replace=False)
        random_kpoint_idx = numpy.random.choice(len(kpoints), size=size, replace=False)


    random_kpoint_idx = numpy.sort(random_kpoint_idx)
    random_qpoint_idx = numpy.sort(random_qpoint_idx)

    kpoints = kpoints[random_kpoint_idx]
    qpoints = qpoints[random_qpoint_idx]
    with h5py.File(repo_root / "DNTT_ephmat.h5", "r") as f:
        dpot_ref = f["deformation_potential"][:, random_qpoint_idx][:, :, random_kpoint_idx]
        gmod_ref = f["eph_matrix_elements"][:, random_qpoint_idx][:, :, random_kpoint_idx]
    ephmat = CalcEphMatReciprocal(epr_fname)
    results = ephmat.calc_ephmat(kpoints, qpoints)

    rank = get_mpi_rank()
    if rank == 0:
        assert numpy.allclose(results["deformation_potential"], dpot_ref)
        assert numpy.allclose(results["eph_matrix_elements"], gmod_ref)

# if __name__ == "__main__":
    # test_ephmat(size=10)