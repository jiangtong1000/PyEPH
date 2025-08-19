"""Integration tests for phonon dispersion calculations."""

import pytest
import numpy
import h5py
from pathlib import Path

from pyeph.post_qe2pert import parse_qpoint_path, ElectronBands
from pyeph.utils.constants import ryd_to_mev
from pyeph.utils.logger import get_mpi_rank, get_mpi_info

def test_electron_bands():
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

    qpoints = parse_qpoint_path(qpoint_path_string)
    electron_bands = ElectronBands(epr_fname)
    band_energies = electron_bands.calc_band_structure(qpoints)
    with h5py.File(repo_root / "DNTT_band_energies.h5", "r") as f:
        band_energies_ref = f["band_energies"][()]
    assert numpy.allclose(band_energies, band_energies_ref)


if __name__ == "__main__":
    test_electron_bands()