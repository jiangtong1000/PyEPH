import numpy as np
import matplotlib.pyplot as plt
import sys
import h5py
sys.path.append('..')
from pyeph.post_qe2pert import parse_qpoint_path, PhononDispersion
from pyeph.utils.constants import ryd_to_mev

from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
verbose = True if rank == 0 else False

# Initialize the PostQE2Pert object
qe2pert = PhononDispersion("DNTT_epr.h5")

# Parse q-point path
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
force_constants = qe2pert.extract_force_constants()
frequencies, modes = qe2pert.compute_phonon_dispersion(qpoints, force_constants)

if rank == 0:
    with h5py.File("DNTT_phdisp.h5") as fa:
        freq_ref = fa["frequencies"][()]
        assert np.allclose(freq_ref, frequencies)