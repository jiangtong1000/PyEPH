"""
Typical model helpers for contructing EPC and Hopping matrices.
"""

import numpy

from pyeph.greenkubo.lattice import BravaisLattice2D
from pyeph.greenkubo.phonon import build_phonon_baths
from pyeph.greenkubo.hamiltonian import ElectronPhononHamiltonian
from pyeph.greenkubo.utils import (
    wannier_center_and_cell_vecs_for_simple_2D,
    get_g_from_deltaV,
)

def build_1d_Holstein_Peierls_model(
    J, dJ, wH, gH, wP, temperature,
    nsites,
    cpa_cutoff,
    energy_unit=1.0,
    model_type='bond'
    ):
    
    J /= energy_unit
    if len(wH) > 0:
        wH = numpy.array(wH) / energy_unit
        gH = numpy.array(gH) / energy_unit
    wP /= energy_unit
    temperature /= energy_unit
    gP = get_g_from_deltaV(dJ * numpy.abs(J), wP, temperature)
    gP = gP * wP
    cpa_cutoff = cpa_cutoff / energy_unit
    
    # 1 mol in a unit cell, 1 Peierls mode + n Holstein modes per cell
    nHolstein = len(wH)
    nmodes = 1 + nHolstein
    ncenter = 1
    wannier_center_pos = numpy.array([0.5, 0.0])
    cell_vecs = numpy.array([[1.0, 0.0], [0.0, 1.0]])
    ph_freq = numpy.concatenate([wH, [wP]])
    ph_freq = ph_freq.reshape(nmodes, 1)

    gnonlocal = numpy.zeros((ncenter, ncenter, nmodes))
    gnonlocal[0,0,-1] = gP

    if len(wH) > 0:
        glocal = numpy.zeros((ncenter, ncenter, nmodes))
        glocal[0,0,:nHolstein] = gH
        gmat = {
            (1,0): {(0, 0): gnonlocal},
            (0,0): {(0, 0): glocal}
        }
    else:
        gmat = {
            (1,0): {(0, 0): gnonlocal}
        }
    tmat = {(1, 0): numpy.array([-J])}
    
    nx = nsites
    ny = 1
    lattice = BravaisLattice2D(nx, ny, ncenter, wannier_center_pos, cell_vecs)
    
    distribution = 'Boltzmann'
    classical_ph, quantum_ph = build_phonon_baths(ph_freq, gmat, cpa_cutoff, temperature, distribution)

    ham = ElectronPhononHamiltonian(tmat, classical_ph.gmat, lattice)

    return ham, classical_ph, quantum_ph, lattice, temperature


def build_1d_ssh_optical_model(
    t,
    lambda_ep,
    ph_freq,
    nsites,
    direction='y'
):
    """
    lambda_ep: e-ph coupling strength
    there are multiple conventions, use it consistently, here: lambda_ep = 2g^2 / (t * omega)
    """
    ncenter = 1
    wannier_center_pos = numpy.array([0.5, 0.0])
    cell_vecs = numpy.array([[1.0, 0.0], [0.0, 1.0]])
    ph_freq = numpy.array([ph_freq]).reshape(1, 1)
    w_p = ph_freq[0, 0]
    nmodes = 1
    g = numpy.sqrt((w_p * numpy.abs(t) * lambda_ep) / 2.0)

    if direction == "x":
        disp = (1, 0)
        nx, ny = nsites, 1
    elif direction == "y":
        disp = (0, 1)
        nx, ny = 1, nsites
    gmat_dir = {
        disp: {
            (0, 0): -g * numpy.ones((ncenter, ncenter, nmodes)),
            disp: g * numpy.ones((ncenter, ncenter, nmodes)),
        }
    }
    tmat_dir = {disp: numpy.array([-t])}

    lattice = BravaisLattice2D(nx, ny, ncenter, wannier_center_pos, cell_vecs)

    return lattice, tmat_dir, gmat_dir, ph_freq


def get_garray_for_2d_ssh(
    g1, g2, g3, wP, 
    model_type,
    zigzag,
    ncenter,
    nmodes,
    offset=0
):
    garray_1 = numpy.zeros((ncenter, ncenter, nmodes))
    garray_1[:, :, 1+offset] = numpy.array([[0, g2 * wP], [g2 * wP, 0]])
    garray_2 = numpy.zeros((ncenter, ncenter, nmodes))
    garray_2[:, :, 0+offset] = numpy.array([[g1 * wP, 0], [0, 0]])
    garray_2[:, :, 3+offset] = numpy.array([[0, 0], [0, g1 * wP]])
    garray_3 = numpy.zeros((ncenter, ncenter, nmodes))
    garray_3[:, :, 2+offset] = numpy.array([[0, 0], [g3 * wP, 0]])
    
    # additional coupling when peierls modes are optical
    if model_type == 'optical':
        garray_1[:, :, 4+offset] = numpy.array([[0, -g2 * wP], [-g2 * wP, 0]])
        garray_2[:, :, 5+offset] = numpy.array([[0, 0], [0, -g3 * wP]])
        garray_3[:, :, 0+offset] = numpy.array([[-g1 * wP, 0], [0, 0]])
        garray_3[:, :, 3+offset] = numpy.array([[0, 0], [0, -g1 * wP]])
    
    g2w5 = numpy.zeros((ncenter, ncenter, nmodes))
    g2w5[:, :, 4+offset] = numpy.array([[0, 0], [g2 * wP, 0]])
    g3w6 = numpy.zeros((ncenter, ncenter, nmodes))
    g3w6[:, :, 5+offset] = numpy.array([[0, 0], [g3 * wP, 0]])
    minus_g2w2 = numpy.zeros((ncenter, ncenter, nmodes))
    minus_g2w2[:, :, 1+offset] = numpy.array([[0, 0], [-g2 * wP, 0]])
    minus_g3w3 = numpy.zeros((ncenter, ncenter, nmodes))
    minus_g3w3[:, :, 2+offset] = numpy.array([[0, 0], [-g3 * wP, 0]])
    
    if zigzag:
        garray_4 = g2w5
        garray_5 = g3w6
    else:
        garray_4 = g3w6
        garray_5 = g2w5
    
    if zigzag:
        garray_6 = minus_g2w2
        garray_7 = minus_g3w3
    else:
        garray_6 = minus_g3w3
        garray_7 = minus_g2w2
    
    return garray_1, garray_2, garray_3, garray_4, garray_5, garray_6, garray_7

def get_garray_for_2d_ssh_2(
    g1, g2, g3, wP, 
    ncenter,
    nmodes,
    offset=0
):
    garray_1 = numpy.zeros((ncenter, ncenter, nmodes))
    garray_1[:, :, 1+offset] = numpy.array([[0, g2 * wP], [g2 * wP, 0]])
    garray_2 = numpy.zeros((ncenter, ncenter, nmodes))
    garray_2[:, :, 0+offset] = numpy.array([[0, 0], [0, g1 * wP]])
    garray_2[:, :, 3+offset] = numpy.array([[g1 * wP, 0], [0, 0]])
    garray_3 = numpy.zeros((ncenter, ncenter, nmodes))
    garray_3[:, :, 2+offset] = numpy.array([[0, g3 * wP], [0, 0]])
    
    g2w5 = numpy.zeros((ncenter, ncenter, nmodes))
    g2w5[:, :, 4+offset] = numpy.array([[0, g2 * wP], [0, 0]])
    g3w6 = numpy.zeros((ncenter, ncenter, nmodes))
    g3w6[:, :, 5+offset] = numpy.array([[0, g3 * wP], [0, 0]])
    
    garray_4 = g3w6
    garray_5 = g2w5
    
    return garray_1, garray_2, garray_3, garray_4, garray_5

def build_2d_ssh_model(
    nx, ny,
    a, b,
    J1,J2,J3,dJ1,dJ2,dJ3,
    wP,
    temperature,
    model_type='bond',
    length_unit=1.0,
    energy_unit=1.0,
    zigzag=True
):  
    """
    Build a generic 2D SSH model with bond phonon modes
    Fratini, Nat. Mat. 2017 | Troisi, Phys. Rev. Applied 2024
    
    Input energies are in meV
    Input lengths are in Angstrom
    """
    assert nx > 1 and ny > 1, "nx and ny must be greater than 1"
    assert model_type in ['bond', 'optical'], "Invalid model type"
    ncenter = 2
    a = a / length_unit
    b = b / length_unit
    wannier_center_pos, cell_vecs = wannier_center_and_cell_vecs_for_simple_2D(a, b)
    lattice = BravaisLattice2D(nx, ny, ncenter, wannier_center_pos, cell_vecs)

    J1 /= energy_unit
    J2 /= energy_unit
    J3 /= energy_unit
    wP = wP / energy_unit
    temperature = temperature / energy_unit

    nmodes = 6 # two sites in a unit cell, each carry 3 modes coupled to hoppings
    ph_freq = numpy.array([wP] * nmodes).reshape(nmodes, 1)

    g1 = get_g_from_deltaV(dJ1 * numpy.abs(J1), wP, temperature)
    g2 = get_g_from_deltaV(dJ2 * numpy.abs(J2), wP, temperature)
    g3 = get_g_from_deltaV(dJ3 * numpy.abs(J3), wP, temperature)
    
    if model_type == 'optical':
        g1 = g1 / numpy.sqrt(2)
        g2 = g2 / numpy.sqrt(2)
        g3 = g3 / numpy.sqrt(2)
    
    garray_1, garray_2, garray_3, garray_4, garray_5, garray_6, garray_7 = get_garray_for_2d_ssh(
        g1, g2, g3, wP, model_type, zigzag, ncenter, nmodes, offset=0
    )
        
    gmat = {
        (0, 0): {(0, 0): garray_1},
        (1, 0): {(0, 0): garray_2, (1, 0): garray_3},
        (0, 1): {(0, 0): garray_4},
        (1, 1): {(0, 0): garray_5},
    }
    
    if model_type == 'optical':
        gmat[(0, 1)][(0, 1)] = garray_6
        gmat[(1, 1)][(1, 1)] = garray_7

    tmat = {
        (0, 0): numpy.array([[0, J2], [J2, 0]]),
        (1, 0): numpy.array([[J1, 0], [J3, J1]]),
        (0, 1): numpy.array([[0, 0], [J2, 0]]),
        (1, 1): numpy.array([[0, 0], [J3, 0]]),
    }
    if not zigzag:
        tmat[(0, 1)] = numpy.array([[0, 0], [J3, 0]])
        tmat[(1, 1)] = numpy.array([[0, 0], [J2, 0]])

    return lattice, tmat, gmat, ph_freq, temperature


def build_2d_ssh_model_2(
    nx, ny,
    a, b,
    J1,J2,J3,dJ1,dJ2,dJ3,
    wP,
    temperature,
    length_unit=1.0,
    energy_unit=1.0,
):  
    """
    Build a generic 2D SSH model with bond phonon modes
    Fratini, Nat. Mat. 2017 | Troisi, Phys. Rev. Applied 2024
    
    Input energies are in meV
    Input lengths are in Angstrom
    """
    assert nx > 1 and ny > 1, "nx and ny must be greater than 1"
    ncenter = 2
    a = a / length_unit
    b = b / length_unit
    wannier_center_pos, cell_vecs = wannier_center_and_cell_vecs_for_simple_2D(a, b)
    wannier_center_pos = numpy.array([
        [0.75 * a, 0.75 * b],
        [0.25 * a, 0.25 * b]
    ])
    lattice = BravaisLattice2D(nx, ny, ncenter, wannier_center_pos, cell_vecs)

    J1 /= energy_unit
    J2 /= energy_unit
    J3 /= energy_unit
    wP = wP / energy_unit
    temperature = temperature / energy_unit

    nmodes = 6 # two sites in a unit cell, each carry 3 modes coupled to hoppings
    ph_freq = numpy.array([wP] * nmodes).reshape(nmodes, 1)

    g1 = get_g_from_deltaV(dJ1 * numpy.abs(J1), wP, temperature)
    g2 = get_g_from_deltaV(dJ2 * numpy.abs(J2), wP, temperature)
    g3 = get_g_from_deltaV(dJ3 * numpy.abs(J3), wP, temperature)
    
    garray_1, garray_2, garray_3, garray_4, garray_5 = get_garray_for_2d_ssh_2(
        g1, g2, g3, wP, ncenter, nmodes, offset=0
    )
        
    gmat = {
        (0, 0): {(0, 0): garray_1},
        (1, 0): {(0, 0): garray_2, (1, 0): garray_3},
        (0, 1): {(0, 0): garray_4},
        (1, 1): {(0, 0): garray_5},
    }
    
    tmat = {
        (0, 0): numpy.array([[0, J2], [J2, 0]]),
        (1, 0): numpy.array([[J1, J3], [0, J1]]),
        (0, 1): numpy.array([[0, J3], [0, 0]]),
        (1, 1): numpy.array([[0, J2], [0, 0]]),
    }

    return lattice, tmat, gmat, ph_freq, temperature


def build_2d_multiple_Holstein_and_Peierls_model(
    nx, ny,
    a, b,
    J1, J2, J3,
    dJ1, dJ2, dJ3,
    wH, gH,
    wP,
    cpa_cutoff,
    temperature,
    model_type='bond',
    length_unit=1.0,
    energy_unit=1.0,
    zigzag=True
):
    """
    See Shuai, Z. et al. Nat. Comm. 2020 for the 1D version.
    This function is the 2D version combined with the 2D lattice in build_ssh_2d_bond_phonon_model.
    I also try to make gH consistently in energy unit within this package.
    """
    assert model_type in ['bond', 'optical'], "Invalid model type"
    ncenter = 2
    a = a / length_unit
    b = b / length_unit
    wannier_center_pos, cell_vecs = wannier_center_and_cell_vecs_for_simple_2D(a, b)
    lattice = BravaisLattice2D(nx, ny, ncenter, wannier_center_pos, cell_vecs)
    
    J1 /= energy_unit
    J2 /= energy_unit
    J3 /= energy_unit
    wP = wP / energy_unit
    cpa_cutoff = cpa_cutoff / energy_unit
    temperature = temperature / energy_unit
    wH = numpy.array(wH) / energy_unit
    gH = numpy.array(gH) / energy_unit
    
    # duplicate those classical Holstein modes for two sites in a unit cell
    # quantum modes are integrated out, so we only keep one (code-specific setup)
    wH_quantum = wH[wH > cpa_cutoff]
    wH_classical = wH[wH <= cpa_cutoff]
    nmodes_Holstein_classical = len(wH_classical)
    nmodes_Holstein_quantum = len(wH_quantum)
    gH_quantum = gH[wH > cpa_cutoff]
    gH_classical = gH[wH <= cpa_cutoff]
    wH = numpy.concatenate([wH_classical, wH_classical, wH_quantum])
    gH = numpy.concatenate([gH_classical, gH_classical, gH_quantum])
    nmodes_H = len(wH)
    
    g1 = get_g_from_deltaV(dJ1 * numpy.abs(J1), wP, temperature)
    g2 = get_g_from_deltaV(dJ2 * numpy.abs(J2), wP, temperature)
    g3 = get_g_from_deltaV(dJ3 * numpy.abs(J3), wP, temperature)
    
    if model_type == 'optical':
        g1 = g1 / numpy.sqrt(2)
        g2 = g2 / numpy.sqrt(2)
        g3 = g3 / numpy.sqrt(2)
        
    # build phonon
    nmodes_P = 6
    ph_freq = numpy.array([wP] * nmodes_P)
    ph_freq = numpy.concatenate([wH, ph_freq])
    nmodes = nmodes_H + nmodes_P
    ph_freq = ph_freq.reshape(nmodes, 1)
    
    offset = nmodes_H
    
    garray_1, garray_2, garray_3, garray_4, garray_5, garray_6, garray_7 = get_garray_for_2d_ssh(
        g1, g2, g3, wP, model_type, zigzag, ncenter, nmodes, offset=offset
    )
    garray_1[0, 0, :nmodes_H] = gH
    garray_1[1, 1, :nmodes_H] = gH

    gmat = {
        (0,0): {
            (0,0): garray_1
        },
        (1,0): {
            (0,0): garray_2,
            (1,0): garray_3
        },
        (0,1): {
            (0,0): garray_4
        },
        (1,1): {
            (0,0): garray_5
        }
    }
    
    if model_type == 'optical':
        gmat[(0,1)][(0,1)] = garray_6
        gmat[(1,1)][(1,1)] = garray_7
    
    distribution = 'Boltzmann'
    classical_ph, quantum_ph = build_phonon_baths(ph_freq, gmat, cpa_cutoff, temperature, distribution)

    # build Hamiltonian, initialize static t.b.
    tmat = {
        (0,0): numpy.array([[0, J2], [J2, 0]]),
        (1,0): numpy.array([[J1, 0], [J3, J1]]),
        (0,1): numpy.array([[0,0], [J2, 0]]),
        (1,1): numpy.array([[0,0], [J3, 0]])
    }
    
    if not zigzag:
        tmat[(0, 1)] = numpy.array([[0, 0], [J3, 0]])
        tmat[(1, 1)] = numpy.array([[0, 0], [J2, 0]])

    ham = ElectronPhononHamiltonian(tmat, classical_ph.gmat, lattice)
    
    return ham, classical_ph, quantum_ph, lattice, temperature