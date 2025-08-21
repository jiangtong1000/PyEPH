"""Integration tests for eph mat calculations."""

import pytest
import numpy
import h5py
from pathlib import Path

from pyeph.post_qe2pert import parse_qpoint_path, CalcEphMatReciprocal
from pyeph.utils.constants import ryd_to_mev
from pyeph.utils.logger import get_mpi_rank, get_mpi_info
import pytest

@pytest.mark.skip(reason="none")
def test_ephmat(size=5):
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
            random_qpoint_idx = numpy.random.choice(len(qpoints), size=size, replace=False)
            random_kpoint_idx = numpy.random.choice(len(kpoints), size=size, replace=False)
        else:
            random_qpoint_idx = None
            random_kpoint_idx = None
        random_qpoint_idx = comm.bcast(random_qpoint_idx, root=0)
        random_kpoint_idx = comm.bcast(random_kpoint_idx, root=0)
    else:
        random_qpoint_idx = numpy.random.choice(len(qpoints), size=size, replace=False)
        random_kpoint_idx = numpy.random.choice(len(kpoints), size=size, replace=False)

    kpoints = kpoints[random_kpoint_idx]
    qpoints = qpoints[random_qpoint_idx]
    print("kpoints:", kpoints)
    print("qpoints:", qpoints)

    ephmat = CalcEphMatReciprocal(epr_fname)
    results = ephmat.calc_ephmat(kpoints, qpoints)
    dpot = results["deformation_potential"]
    gmod = results["eph_matrix_elements"]

    with h5py.File("DNTT_ephmat.h5", "w") as f:
        f.create_dataset("deformation_potential", data=dpot)
        f.create_dataset("eph_matrix_elements", data=gmod)

if __name__ == "__main__":
    test_ephmat(size=3)