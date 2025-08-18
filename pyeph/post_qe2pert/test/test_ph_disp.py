"""Integration tests for phonon dispersion calculations."""

import pytest
import numpy
import h5py
from pathlib import Path

from pyeph.post_qe2pert import parse_qpoint_path, PhononDispersion
from pyeph.utils.constants import ryd_to_mev
from pyeph.utils.logger import get_mpi_rank

def test_phonon_dispersion(size=10):
    repo_root = Path(__file__).resolve().parents[0]
    epr_fname = repo_root / "DNTT_epr.h5"
    phdisp_ref_fname = repo_root / "DNTT_phdisp.h5"
    with h5py.File(phdisp_ref_fname) as fref:
        frequencies = fref["frequencies"][:]
    freq_ref_mev = frequencies * ryd_to_mev

    print("Loading data from DNTT_epr.h5 and DNTT_phdisp.yml...")
    qe2pert = PhononDispersion(epr_fname)

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

    qpoints = parse_qpoint_path(qpoint_path_string)
    random_qpoint_idx = numpy.random.choice(len(qpoints), size=size, replace=False)
    random_qpoints_selection = [qpoints[i] for i in random_qpoint_idx]
    force_constants = qe2pert.extract_force_constants()
    frequencies, _ = qe2pert.compute_phonon_dispersion(random_qpoints_selection, force_constants)
    frequencies = frequencies * ryd_to_mev
    freq_ref_mev = freq_ref_mev[random_qpoint_idx]
    
    # Only check results on rank 0 (where MPI results are gathered)
    rank = get_mpi_rank()
    if rank == 0:
        assert numpy.allclose(frequencies, freq_ref_mev)
        print("phonon dispersion suit test passed.")
    
if __name__ == "__main__":
    test_phonon_dispersion(size=10)