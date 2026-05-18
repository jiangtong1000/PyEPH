from scipy.constants import physical_constants as c
import numpy


def wannier_center_and_cell_vecs_for_simple_2D(a: float, b: float):
    """
    Get the wannier center and cell vectors for a simple 2D lattice.
    a: lattice constant along x
    b: lattice constant along y
    Returns:
        wannier_center: the wannier center
        cell_vecs: the cell vectors
    """
    cell_vecs = numpy.array([[a, 0], [0, b]])
    wannier_center_pos = numpy.array([
        [0.25 * a, 0.25 * b],
        [0.75 * a, 0.75 * b]
    ])
    return wannier_center_pos, cell_vecs


au2ev = c["Hartree energy in eV"][0]
K2au = c["kelvin-hartree relationship"][0]
cm2au = 1.0e2 * c["inverse meter-hertz relationship"][0] / c["hartree-hertz relationship"][0]
fs2au = 1.0e-15 / c["atomic unit of time"][0]
ryd_to_ev = 13.605698066
ryd_to_mev = ryd_to_ev * 1000.0

mobility2au = au2ev * c["atomic unit of time"][0] / (c["atomic unit of length"][0] * 100) ** 2


def get_deltaV_from_g(gP, wP, temperature):
    # all in unit of V
    deltaV = gP * wP / numpy.sqrt(numpy.tanh(wP / (2 * temperature)))
    return deltaV


def get_g_from_deltaV(deltaV, wP, temperature):
    # all in unit of V
    gP = deltaV * numpy.sqrt(numpy.tanh(wP / (2 * temperature))) / wP
    return gP

# for debugging (my old code uses different ordering of sites)
def get_map(nx, ny):
    map = []
    total_sites = nx * ny * 2
    for i in range(0, total_sites, nx * 2):
        original_idx = numpy.arange(i, i + nx * 2)
        original_idx = original_idx.reshape(nx, 2).T
        original_idx = original_idx.flatten()
        map.extend(original_idx)
    return map