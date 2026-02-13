import pytest
import numpy
from pyeph.greenkubo.mp_qmesh import half_q_points, get_mp_qmesh_info

@pytest.mark.parametrize("Nx", [4, 6, 8])
@pytest.mark.parametrize("Ny", [4, 6])
def test_mp_grid(Nx, Ny):
    q_half, q_partner, half_ij, partner_ij, Q = half_q_points(Nx, Ny)
    nmodes = 10
    nhalf_qpts = len(q_half)
    freq_half = numpy.random.random((nmodes, nhalf_qpts))
    freq_full, q_half, q_partner, half_ij, partner_ij, Q = get_mp_qmesh_info(Nx, Ny, freq_half)
    
    for iq in range(nhalf_qpts):
        assert numpy.allclose(Q[half_ij[iq, 0], half_ij[iq, 1]], q_half[iq])
        assert numpy.allclose(Q[partner_ij[iq, 0], partner_ij[iq, 1]], q_partner[iq])
        assert numpy.allclose(q_half[iq], -q_partner[iq])
        assert numpy.allclose(freq_full[:, half_ij[iq, 0], half_ij[iq, 1]], freq_half[:, iq])
        assert numpy.allclose(freq_full[:, partner_ij[iq, 0], partner_ij[iq, 1]], freq_half[:, iq])