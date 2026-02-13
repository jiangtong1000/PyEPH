import os
os.environ['USE_MPI'] = 'false'

import h5py
import numpy
import pytest
import pathlib
import shutil

from pyeph.greenkubo.hamiltonian import ElectronPhononHamiltonian
from pyeph.greenkubo.lattice import BravaisLattice2D
from pyeph.greenkubo.phonon import ClassicPhononBath, build_phonon_baths
from pyeph.greenkubo.propagator import DensityMatrixUnitaryPropagator
from pyeph.greenkubo.simulation import GreenKuboSimulation

import pytest


def test_classic_phonon_initialization_boltzmann_reproducible():
    ph_freq = numpy.array([[2.0]])
    bath = ClassicPhononBath(ph_freq, temperature=2.0, gmat={}, distribution="Boltzmann")

    rng = numpy.random.default_rng(123)
    bath.initialize_position_and_momentum(nx=1, ny=1, ntraj=2, rng=rng)

    rng_expected = numpy.random.default_rng(123)
    beta = bath.beta
    w = bath.w[0]
    ncells = 1
    p0 = rng_expected.normal(0, numpy.sqrt(1 / beta), (2, ncells))
    q0 = rng_expected.normal(0, numpy.sqrt(1 / (beta * w ** 2)), (2, ncells))
    q0 = q0 * numpy.sqrt(w * 2)
    p0 = p0 * numpy.sqrt(2 / w)

    assert numpy.allclose(bath.q0[0], q0)
    assert numpy.allclose(bath.p0[0], p0)
    assert numpy.array_equal(bath.qfield, bath.q0)

def test_greenkubo_simulation_1D_CPA(tmp_path, test_mode=True):
    from pyeph.greenkubo.typical_model_helper import build_1d_ssh_optical_model

    nx = 1; ny = 31; ncenter = 1;
    wannier_center_pos = numpy.array([0.5, 0])
    cell_vecs = numpy.array([[1.0, 0.0], [0.0, 1.0]])
    lattice = BravaisLattice2D(nx, ny, ncenter, wannier_center_pos, cell_vecs)

    t = 1.0
    w_p = 0.044
    lmbda = 0.5
    lattice, tmat, gmat, ph_freq = build_1d_ssh_optical_model(t, lmbda, w_p, ny, direction="y")
    
    # build hamiltonian
    temperature = 1.0 # in unit of t
    cpa_cutoff = temperature
    distribution = 'Boltzmann'
    classical_ph, quantum_ph = build_phonon_baths(ph_freq, gmat, cpa_cutoff, temperature, distribution)
    ham = ElectronPhononHamiltonian(tmat, classical_ph.gmat, lattice)

    # build propagator
    time_step = 0.01
    total_time = 0.1
    ntraj_per_rank = 10
    propagator = DensityMatrixUnitaryPropagator(lattice.nsites, ntraj_per_rank, time_step, total_time, temperature)

    # assemble simulation and run
    simulation = GreenKuboSimulation(lattice, ham, classical_ph, quantum_ph, propagator)
    dump_interval = 500
    
    dump_dir = tmp_path / "currents_output"
    simulation.run(
        dump_dir=dump_dir,
        dump_interval=dump_interval
    )

    if not test_mode:
        return
    with h5py.File(dump_dir / "collected_current_autocorr.h5") as fa:
        with h5py.File(pathlib.Path(__file__).parent / "expected_1D_CPA.h5") as fb:
            for key in fa.keys():
                assert numpy.allclose(fa[key][:], fb[key][:])
    
    shutil.rmtree(dump_dir)
    

@pytest.mark.parametrize("model_type", ['bond', 'optical'])
@pytest.mark.parametrize("zigzag", [True, False])
def test_greenkubo_simulation_2D_CPA(tmp_path, model_type, zigzag, test_mode=True):
    from pyeph.greenkubo.typical_model_helper import build_2d_ssh_model
    
    # set up model parameters
    nx = 18; ny = 9;
    a = 7.2; b = 14.3;
    length_unit = 7.2
    wP = 6.0
    temperature = 25
    J1, J2, J3 = -96.1, 35.0, -14.7
    dJ1, dJ2, dJ3 = 0.246, 0.421, 0.321
    energy_unit = numpy.sqrt(J1**2 + J2**2 + J3**2)
    
    lattice, tmat, gmat, ph_freq, temperature = build_2d_ssh_model(nx, ny, a, b, J1, J2, J3, dJ1, dJ2, dJ3, wP, temperature, model_type, length_unit, energy_unit, zigzag=zigzag)
     
    # build hamiltonian
    cpa_cutoff = temperature
    distribution = 'Boltzmann'
    classical_ph, quantum_ph = build_phonon_baths(ph_freq, gmat, cpa_cutoff, temperature, distribution)
    ham = ElectronPhononHamiltonian(tmat, classical_ph.gmat, lattice)

    # build propagator
    time_step = 0.01
    total_time = 0.1
    ntraj_per_rank = 10
    propagator = DensityMatrixUnitaryPropagator(lattice.nsites, ntraj_per_rank, time_step, total_time, temperature)

    # assemble simulation and run
    simulation = GreenKuboSimulation(lattice, ham, classical_ph, quantum_ph, propagator)
    dump_dir = tmp_path / "currents_output"
    dump_interval = 20
    simulation.run(
        dump_dir=dump_dir,
        dump_interval=dump_interval
    )

    if not test_mode:
        return
    with h5py.File(dump_dir / "collected_current_autocorr.h5") as fa:
        with h5py.File(pathlib.Path(__file__).parent / f"expected_2D_CPA_{model_type}Peierls_zz{zigzag}.h5") as fb:
            for key in fa.keys():
                assert numpy.allclose(fa[key][:], fb[key][:])
    
    shutil.rmtree(dump_dir)
    
@pytest.mark.parametrize("model_type", ['bond', 'optical'])
@pytest.mark.parametrize("zigzag", [True, False])
def test_greenkubo_simulation_2D_polaron_transform_CPA(tmp_path, model_type, zigzag, test_mode=False):
    from pyeph.greenkubo.typical_model_helper import build_2d_multiple_Holstein_and_Peierls_model
    nx = 4; ny = 2;
    a = 7.2; b = 14.3;
    length_unit = 7.2
    J1, J2, J3 = -96.1, -35.0, -14.7
    dJ1, dJ2, dJ3 = 0.246, 0.421, 0.321
    energy_unit = numpy.sqrt(J1**2 + J2**2 + J3**2)
    
    wH = numpy.array([5.0, 10.0, 20.0, 30.0, 40.0])
    gH = numpy.array([1.0, 2.0, 1.0, 1.0, 0.8]) * wH
    wP = 6.0
    cpa_cutoff = 35
    temperature = 25
    ham, classical_ph, quantum_ph, lattice, temperature = build_2d_multiple_Holstein_and_Peierls_model(
        nx, ny, a, b,
        J1, J2, J3, dJ1, dJ2, dJ3,
        wH, gH, wP,
        cpa_cutoff, temperature,
        model_type,
        length_unit, energy_unit,
        zigzag=zigzag
    )

    # build propagator
    time_step = 0.01
    total_time = 0.1
    ntraj_per_rank = 2
    propagator = DensityMatrixUnitaryPropagator(lattice.nsites, ntraj_per_rank, time_step, total_time, temperature)

    # assemble simulation and run
    simulation = GreenKuboSimulation(lattice, ham, classical_ph, quantum_ph, propagator)
    dump_dir = tmp_path / "currents_output"
    dump_interval = 20
    simulation.run(
        dump_dir=dump_dir,
        dump_interval=dump_interval
    )

    if not test_mode:
        return
    with h5py.File(dump_dir / "collected_current_autocorr.h5") as fa:
        with h5py.File(pathlib.Path(__file__).parent / f"expected_2D_PT_CPA_{model_type}Peierls_zz{zigzag}.h5") as fb:
            for key in fa.keys():
                assert numpy.allclose(fa[key][:], fb[key][:])
    
    shutil.rmtree(dump_dir)


if __name__ == "__main__":
    test_mode = False
    # test_mode = True
    # test_greenkubo_simulation_1D_CPA(pathlib.Path("."), test_mode=test_mode)
    # test_greenkubo_simulation_2D_CPA(pathlib.Path("."), model_type='optical', zigzag=True, test_mode=test_mode)
    test_greenkubo_simulation_2D_polaron_transform_CPA(pathlib.Path("."), model_type='optical', zigzag=False, test_mode=test_mode)