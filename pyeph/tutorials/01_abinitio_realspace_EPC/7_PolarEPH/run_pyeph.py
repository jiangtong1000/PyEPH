"""
Compute electron-phonon coupling matrices using PyEPH.

Reads the Perturbo epr.h5 file and computes:
  - Wannier Hamiltonian hopping elements
  - e-ph coupling matrices (gmat_raw)
  - Phonon dispersion on a full q-grid (with TRS extension)

Output: eph_data_{Nx}.h5 that will be used to further localization
"""
from pyeph.post_qe2pert.phonon_disp import PhononDispersion
from pyeph.post_qe2pert.eph_mat_mixed import CalcEphMatMixed
from pyeph.post_qe2pert.wannier_phonon import trs_grid
from pyeph.utils.constants import ryd_to_mev

import numpy
from mpi4py import MPI
import h5py
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# ---- TODO: adjust these ----
Nx = 10                                     # q-grid dimension
Ny = Nx
num_wann = 2                                # TODO: must match qe2pert
epr_file = "../6_QE2PERT/PREFIX_epr.h5"     # TODO: replace PREFIX
# -----------------------------

ep = CalcEphMatMixed(epr_file, verbose=True)
ham_r_info = ep.ham_r_info
print('rvec_set_el:')
print(ep.rvec_set_el)

if rank == 0:
    ep = CalcEphMatMixed(epr_file, verbose=True)
    for jb in range(num_wann):
        for ib in range(jb + 1):
            print(f"H_{ib+1}{jb+1}:")
            print(ham_r_info[f'H_{ib+1}{jb+1}']['hopping_element']*ryd_to_mev)
            revec_idx = ham_r_info[f'H_{ib+1}{jb+1}']['rvec_indices']
            print([ep.rvec_set_el[i] for i in revec_idx])
    
    nre_vec_tot = len(ep.rvec_set_el)
    nrph_vec_tot = len(ep.rvec_set_ph_eph)
    gmat_raw = ep.extract_gmat_raw(nre_vec_tot, nrph_vec_tot)
    with h5py.File(f"eph_data_{Nx}.h5", "w") as f:
        f.create_dataset("gmat_raw", data=gmat_raw)
        f.create_dataset('rvec_set_ph_eph', data=ep.rvec_set_ph_eph)
        f.create_dataset('rvec_set_el', data=ep.rvec_set_el)
        f.create_dataset('mass', data=ep.mass)

    if rank == 0:
        q_hbz, q_minus, q_full, partner_hbz_for_minus, rph = trs_grid(Nx, Ny)
    else:
        q_hbz = None
        q_minus = None
        q_full = None
        partner_hbz_for_minus = None
        rph = None

    q_hbz = comm.bcast(q_hbz, root=0)
    q_minus = comm.bcast(q_minus, root=0)
    q_full = comm.bcast(q_full, root=0)
    partner_hbz_for_minus = comm.bcast(partner_hbz_for_minus, root=0)
    rph = comm.bcast(rph, root=0)

    phdisp = PhononDispersion(epr_file)
    force_constants = phdisp.extract_force_constants()
    freq_half, mode_half = phdisp.compute_phonon_dispersion(q_hbz, force_constants, mass_weight=False)

    if rank == 0:
        freq_full = numpy.empty((q_full.shape[0], 3*len(phdisp.mass)))
        mode_full = numpy.empty((q_full.shape[0], 3*len(phdisp.mass), 3*len(phdisp.mass)), dtype=numpy.complex128)
        freq_full[:len(q_hbz)] = freq_half
        mode_full[:len(q_hbz)] = mode_half
        for iq_minus, iq_half_partner in enumerate(partner_hbz_for_minus):
            freq_full[len(q_hbz) + iq_minus] = freq_half[iq_half_partner]
            mode_full[len(q_hbz) + iq_minus] = mode_half[iq_half_partner].conj()
        
        with h5py.File(f"eph_data_{Nx}.h5", "a") as f:
            f.create_dataset("freq_full", data=freq_full)
            f.create_dataset("mode_full", data=mode_full)
            f.create_dataset("mass", data=phdisp.mass)
