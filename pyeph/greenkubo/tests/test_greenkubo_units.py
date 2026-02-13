"""
Test the Hamiltonian construction.
"""
import os
os.environ['USE_MPI'] = 'false'

import numpy
import pytest

from pyeph.greenkubo.utils import wannier_center_and_cell_vecs_for_simple_2D
from pyeph.greenkubo.lattice import BravaisLattice2D
from pyeph.greenkubo.hamiltonian import ElectronPhononHamiltonian
from pyeph.greenkubo.estimator import current_from_density_no_polaron
from pyeph.greenkubo.propagator import DensityMatrixUnitaryPropagator
from pyeph.greenkubo.phonon import build_phonon_baths

import scipy.linalg
from itertools import product

@pytest.mark.parametrize("nx", [4, 6])
@pytest.mark.parametrize("ny", [6, 10])
def test_ham_static_and_current_operator(nx, ny):
    ncenter = 2
    wannier_center_pos, cell_vecs = wannier_center_and_cell_vecs_for_simple_2D(1.0, 1.0)
    diag_tmat = numpy.random.random((ncenter, ncenter))
    diag_tmat = diag_tmat + diag_tmat.T
    tmat = {
        (0, 0): diag_tmat,
        (1, 0): numpy.random.random((ncenter, ncenter)),
        (0, 1): numpy.random.random((ncenter, ncenter)),
        (1, 1): numpy.random.random((ncenter, ncenter)),
        (-1, 1): numpy.random.random((ncenter, ncenter))
    }
    
    h_ref = numpy.zeros((nx * ny * ncenter, nx * ny * ncenter))
    jx_ref = numpy.zeros((nx * ny * ncenter, nx * ny * ncenter))
    jy_ref = numpy.zeros((nx * ny * ncenter, nx * ny * ncenter))
    nsites_total = nx * ny * ncenter
    
    for j in range(nsites_total):
        j_cell_x = (j//ncenter) % nx
        j_cell_y = (j//ncenter) // nx
        j_site_idx = j % ncenter
        j_center_pos = wannier_center_pos[j_site_idx]
        j_pos = numpy.array([j_cell_x + j_center_pos[0], j_cell_y + j_center_pos[1]])
        for i in range(j+1):
            i_cell_x = (i//ncenter) % nx
            i_cell_y = (i//ncenter) // nx
            i_site_idx = i % ncenter
            i_center_pos = wannier_center_pos[i_site_idx]
            i_pos = numpy.array([i_cell_x + i_center_pos[0], i_cell_y + i_center_pos[1]])
            ij_dist = numpy.linalg.norm(j_pos - i_pos)
            dx = j_cell_x - i_cell_x
            dy = j_cell_y - i_cell_y
            for xshift, yshift in [(-nx,-ny),(-nx,0),(-nx,ny),(0,-ny),(0,ny),(nx,-ny),(nx,0),(nx,ny)]:
                xshift_pos = numpy.array([i_pos[0] + xshift, i_pos[1] + yshift])
                ij_dist_new = numpy.linalg.norm(xshift_pos - j_pos)
                if ij_dist_new < ij_dist:
                    ij_dist = ij_dist_new
                    dx = j_cell_x - (i_cell_x + xshift)
                    dy = j_cell_y - (i_cell_y + yshift)
            
            if (dx, dy) in tmat:
                h_ref[i, j] = tmat[(dx, dy)][i_site_idx, j_site_idx]
                h_ref[j, i] = h_ref[i, j]
                jx_ref[i, j] = (dx + j_center_pos[0] - i_center_pos[0]) * h_ref[i, j]
                jx_ref[j, i] = -jx_ref[i, j]
                jy_ref[i, j] = (dy + j_center_pos[1] - i_center_pos[1]) * h_ref[i, j]
                jy_ref[j, i] = -jy_ref[i, j]
                continue
            elif ((-dx, -dy) in tmat):
                h_ref[j, i] = tmat[(-dx, -dy)][j_site_idx, i_site_idx]
                h_ref[i, j] = h_ref[j, i]
                jx_ref[j, i] = (-dx + i_center_pos[0] - j_center_pos[0]) * h_ref[j, i]
                jx_ref[i, j] = -jx_ref[j, i]
                jy_ref[j, i] = (-dy + i_center_pos[1] - j_center_pos[1]) * h_ref[j, i]
                jy_ref[i, j] = -jy_ref[j, i]
                continue
    
    lattice = BravaisLattice2D(nx, ny, ncenter, wannier_center_pos, cell_vecs)
    ham = ElectronPhononHamiltonian(tmat, {}, lattice)
    ham.build_static_hopping_matrix()
    hstatic = ham.h_static.toarray()
    assert numpy.allclose(hstatic, hstatic.T)
    assert numpy.allclose(hstatic, h_ref)

    jx, jy = ham.build_jx_jy([ham.h_static])
    jx = jx[0].toarray()
    jy = jy[0].toarray()
    assert numpy.allclose(jx, jx_ref)
    assert numpy.allclose(jy, jy_ref)

@pytest.mark.parametrize("temperature", [0.1, 1.0, 10.0])
@pytest.mark.parametrize("nx", [4, 6])
@pytest.mark.parametrize("ny", [6, 10])
def test_estimator_classical(nx, ny, temperature):
    ncenter = 2
    beta = 1.0 / temperature
    wannier_center_pos, cell_vecs = wannier_center_and_cell_vecs_for_simple_2D(1.0, 1.0)
        
    diag_tmat = numpy.random.random((ncenter, ncenter))
    diag_tmat = diag_tmat + diag_tmat.T
    tmat = {
        (0, 0): diag_tmat,
        (1, 0): numpy.random.random((ncenter, ncenter)),
        (0, 1): numpy.random.random((ncenter, ncenter)),
        (1, 1): numpy.random.random((ncenter, ncenter)),
        (-1, 1): numpy.random.random((ncenter, ncenter))
    }
    
    lattice = BravaisLattice2D(nx, ny, ncenter, wannier_center_pos, cell_vecs)
    ham = ElectronPhononHamiltonian(tmat, {}, lattice)
    ham.build_static_hopping_matrix()
    hstatic = ham.h_static.toarray()
    jx, jy = ham.build_jx_jy([ham.h_static])
    
    eigvals, eigvecs = scipy.linalg.eigh(hstatic)
    rho = eigvecs @ numpy.diag(numpy.exp(-beta * eigvals)) @ eigvecs.T
    rho_0 = rho / rho.trace()
    jx_dense = jx[0].toarray()
    jy_dense = jy[0].toarray()
    j_rho0_x_T = (jx_dense @ rho_0).T
    j_rho0_y_T = (jy_dense @ rho_0).T

    random_matrix = numpy.random.random((nx*ny*ncenter, nx*ny*ncenter)) + 1.0j * numpy.random.random((nx*ny*ncenter, nx*ny*ncenter))
    u_t, _ = numpy.linalg.qr(random_matrix)
    ct_x = current_from_density_no_polaron(j_rho0_x_T, u_t, jx[0])
    ct_y = current_from_density_no_polaron(j_rho0_y_T, u_t, jy[0])
    
    ct_x_ref = -numpy.einsum('ji, jk, kl, li->', u_t.conj(), jx_dense, u_t, j_rho0_x_T.T)
    ct_y_ref = -numpy.einsum('ji, jk, kl, li->', u_t.conj(), jy_dense, u_t, j_rho0_y_T.T)
    assert numpy.allclose(ct_x, ct_x_ref)
    assert numpy.allclose(ct_y, ct_y_ref)

@pytest.mark.parametrize("temperature", [0.1])
@pytest.mark.parametrize("nx", [6])
@pytest.mark.parametrize("ny", [4])
@pytest.mark.parametrize("nmodes", [5])
def test_estimator_with_quantum_ph(nx, ny, temperature, nmodes):
    # the energy unit can be viewed as 100 meV
    beta = 1.0 / temperature
    ncenter = 2
    
    # build a random phonon bath with nmodes quantum modes
    quantum_ph_freq = numpy.random.uniform(0.3, 2.0, nmodes)
    classical_ph_freq = numpy.random.uniform(0.01, 0.2, nmodes)
    ph_cutoff = 0.25
    ph_freq = numpy.concatenate([classical_ph_freq, quantum_ph_freq])
    ph_freq = ph_freq.reshape(-1, 1)
    
    def generate_random_epc(nmodes, re_key):
        gmat_re = {}
        rph_keys = [(x, y) for x in range(-2, 3) for y in range(-2, 3)]
        for rph_key in rph_keys:
            onsite = (re_key == (0, 0)) and (rph_key == (0, 0))
            gmat_rp = numpy.random.random((ncenter, ncenter, nmodes)) * numpy.random.choice([-1, 1], size=(ncenter, ncenter, nmodes))
            if not onsite:
                gmat_rp[:, :, nmodes:] = 0.0
            else:
                gmat_rp[..., nmodes:] = numpy.abs(gmat_rp[..., nmodes:])
            gmat_re[rph_key] = gmat_rp
        return gmat_re
    
    re_keys = [(0, 0), (1, 0), (0, 1), (1, 1), (-1, 1)]
    gmat = {
        re_key: generate_random_epc(nmodes*2, re_key)
        for re_key in re_keys
    }
    
    ncenter = 2
    wannier_center_pos, cell_vecs = wannier_center_and_cell_vecs_for_simple_2D(1.0, 1.0)
        
    diag_tmat = numpy.random.uniform(-1.5, 1.5, (ncenter, ncenter))
    diag_tmat = diag_tmat + diag_tmat.T
    tmat = {
        (0, 0): diag_tmat,
        (1, 0): numpy.random.uniform(-1.5, 1.5, (ncenter, ncenter)),
        (0, 1): numpy.random.uniform(-1.5, 1.5, (ncenter, ncenter)),
        (1, 1): numpy.random.uniform(-1.5, 1.5, (ncenter, ncenter)),
        (-1, 1): numpy.random.uniform(-1.5, 1.5, (ncenter, ncenter))
    }
    
    classical_ph, quantum_ph = build_phonon_baths(ph_freq, gmat, ph_cutoff, temperature, 'Boltzmann')
    lattice = BravaisLattice2D(nx, ny, ncenter, wannier_center_pos, cell_vecs)
    ham = ElectronPhononHamiltonian(tmat, classical_ph.gmat, lattice)
    
    # compute the reference current
    time_now = numpy.random.uniform(1, 10)
    quantum_g_dimless = gmat[(0, 0)][(0, 0)][0, 0, nmodes:] / ph_freq[nmodes:, 0]
    phit = quantum_g_dimless **2 * (numpy.cos(ph_freq[nmodes:, 0] * time_now) / numpy.tanh(beta * ph_freq[nmodes:, 0] / 2) - 1.0j * numpy.sin(ph_freq[nmodes:, 0] * time_now))
    phit = phit.sum()
    phi_0 = quantum_g_dimless **2 / numpy.tanh(beta * ph_freq[nmodes:, 0] / 2)
    phi_0 = phi_0.sum()
    hstatic = ham.h_static.toarray() * numpy.exp(-phi_0)
    eigvals, eigvecs = scipy.linalg.eigh(hstatic)
    rho = eigvecs @ numpy.diag(numpy.exp(-beta * eigvals)) @ eigvecs.T
    rho = rho / rho.trace()
    
    ham.heps = [ham.h_static]
    jx_0, jy_0 = ham.build_jx_jy(ham.heps)
    
    total_sites = nx * ny * 2
    delta = numpy.identity(total_sites, dtype=numpy.int8)
    row, col = ham.h_static.nonzero()
    hopping_pairs = numpy.array(list(zip(row, col)))
        
    f_ijkl_array = numpy.zeros((total_sites, total_sites, total_sites, total_sites), dtype=numpy.complex128)
    for (i, j), (k, l) in product(hopping_pairs, repeat=2):
        phi_ijij0 = 0.0 if delta[i, j] else 2 * phi_0
        phi_klkl0 = 0.0 if delta[k, l] else 2 * phi_0
        phi_ijkl = phit * (delta[i, k] - delta[j, k] - delta[i, l] + delta[j, l])
        f_ijkl_array[i, j, k, l] = numpy.exp(-phi_ijij0/2.0 - phi_klkl0/2.0 - phi_ijkl)
    
    random_matrix = numpy.random.random((nx*ny*ncenter, nx*ny*ncenter)) + 1.0j * numpy.random.random((nx*ny*ncenter, nx*ny*ncenter))
    u_t, _ = numpy.linalg.qr(random_matrix)

    propagator = DensityMatrixUnitaryPropagator(nx*ny*ncenter, 1, 0.01, 10.0, temperature)
    propagator.build(ham, quantum_ph)
    assert propagator.polaron_prefactor == numpy.exp(-phi_0)
    

    hep_polaron_transform = [hep * propagator.polaron_prefactor for hep in ham.heps]
    propagator.initialize_density_matrix(hep_polaron_transform, jx_0, jy_0)
    
    assert numpy.allclose(propagator.rho0[0], rho)
    assert numpy.allclose(quantum_ph.phi0, phi_0)
    
    # for i,j,k,l in product(range(total_sites), repeat=4):
    for (i, j), (k, l) in product(hopping_pairs, repeat=2):
        phi_ijij0 = 0.0 if delta[i, j] else 2 * phi_0
        phi_klkl0 = 0.0 if delta[k, l] else 2 * phi_0
        if (i,j,k,l) not in propagator.F0:
            assert (jx_0[0][i, j] == 0.0 or jx_0[0][k, l] == 0.0) and (jy_0[0][i, j] == 0.0 or jy_0[0][k, l] == 0.0), f'{i}, {j}, {k}, {l}, {jx_0[0][i, j]}, {jx_0[0][k, l]}, {jy_0[0][i, j]}, {jy_0[0][k, l]}'
            continue
        assert propagator.F0[(i, j, k, l)] == numpy.exp(-phi_ijij0/2.0 - phi_klkl0/2.0), f'{i}, {j}, {k}, {l}, {propagator.F0[(i, j, k, l)]}, {numpy.exp(-phi_ijij0/2.0 - phi_klkl0/2.0)}, {numpy.exp(-phi_0)}'
    
    jx_t = jx_0[0].copy()
    jx_t.data = numpy.random.random(jx_0[0].data.shape)
    jx_t = (jx_t - jx_t.T) / 2
    jx_t.sort_indices()
    jx_t.eliminate_zeros()
    jx_t = [jx_t]
    
    jy_t = jy_0[0].copy()
    jy_t.data = numpy.random.random(jy_0[0].data.shape)
    jy_t = (jy_t - jy_t.T) / 2
    jy_t.sort_indices()
    jy_t.eliminate_zeros()
    jy_t = [jy_t]
    
    quantum_ph.update_phit(time_now)
    assert quantum_ph.phit == phit
    propagator.sec_weights = quantum_ph.sector_weights
    propagator.u_t = numpy.array([u_t])
            
    ct_x_naive, ct_y_naive = propagator.calculate_current_polaron(jx_t, jy_t, use_python=True)
    ct_x, ct_y = propagator.calculate_current_polaron(jx_t, jy_t, use_python=False)
    
    jx_t_dense = jx_t[0].toarray()
    jy_t_dense = jy_t[0].toarray()
    jx_0_dense = jx_0[0].toarray()
    jy_0_dense = jy_0[0].toarray()
    ct_x_ref = -numpy.einsum('im, ij, jk, kl, lm, ijkl->', u_t.conj(), jx_t_dense, u_t, jx_0_dense, rho, f_ijkl_array)
    ct_y_ref = -numpy.einsum('im, ij, jk, kl, lm, ijkl->', u_t.conj(), jy_t_dense, u_t, jy_0_dense, rho, f_ijkl_array)
    
    assert numpy.allclose(ct_x, ct_x_ref) and numpy.allclose(ct_x_naive, ct_x_ref)
    assert numpy.allclose(ct_y, ct_y_ref) and numpy.allclose(ct_y_naive, ct_y_ref)