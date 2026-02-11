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