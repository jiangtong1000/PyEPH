import sys
sys.path.append("/n/holylabs/LABS/joonholee_lab/Users/tjiang/packages/PyEPH")
from pyeph.post_qe2pert.eph_mat_mixed import CalcEphMatMixed
import mpi4py.MPI as MPI
import h5py

comm = MPI.COMM_WORLD

rank = comm.Get_rank()
size = comm.Get_size()

    
import numpy as np
from pyeph.utils.grid import rgrid_2d, generate_half_qgrids, rgrid_2d_full


Nx = 20
Ny = Nx
half_qgrids, minus_qgrids, full_qgrids, partner_qgrids_idx = generate_half_qgrids(Nx, Ny)
rph = rgrid_2d_full(Nx, Ny)

epr_file = "/n/holylabs/LABS/joonholee_lab/Users/tjiang/packages/PyEPH/pyeph/post_qe2pert/test/DNTT_epr.h5"
ep = CalcEphMatMixed(epr_file, verbose=True)
gmat_half, phonon_freqs_half, re_vecs, rph_vecs = ep.calc_ephmat_mixed(half_qgrids)
    
if rank == 0:
    gmat_full = np.empty((*gmat_half.shape[:-1], full_qgrids.shape[0]), dtype=np.complex128)
    phonon_freqs_full = np.empty((phonon_freqs_half.shape[0], full_qgrids.shape[0]), dtype=np.float64)
    gmat_full[..., :len(half_qgrids)] = gmat_half
    phonon_freqs_full[..., :len(half_qgrids)] = phonon_freqs_half
    for iq_minus, iq_half_partner in enumerate(partner_qgrids_idx):
        gmat_full[..., len(half_qgrids) + iq_minus] = gmat_half[..., iq_half_partner].conj()
        phonon_freqs_full[..., len(half_qgrids) + iq_minus] = phonon_freqs_half[..., iq_half_partner]

    with h5py.File(f"gmat_{Nx}_{Ny}.h5", "w") as f:
        f.create_dataset("gmat_mixed", data=gmat_full)
        f.create_dataset("phonon_freqs", data=phonon_freqs_full)
        f.create_dataset("re_vecs", data=re_vecs)
        f.create_dataset("rph_vecs", data=rph_vecs)
        f.create_dataset("qpoints", data=full_qgrids)
        f.create_dataset("rph", data=rph)

comm.Barrier()
if rank == 0:
    gmat_real = ep.ifft_to_real_space(gmat_full, full_qgrids, rph)
    gmat_real = gmat_real / len(full_qgrids)
    with h5py.File(f"gmat_{Nx}_{Ny}.h5", "a") as f:
        f.create_dataset("gmat_real", data=gmat_real)