import numpy

from pyeph.greenkubo.propagator import DensityMatrixUnitaryPropagator
from pyeph.greenkubo.simulation import GreenKuboSimulation
from pyeph.greenkubo.typical_model_helper import build_1d_Holstein_Peierls_model
from pyeph.greenkubo.utils import au2ev, K2au

def lf_cpa_transport(omega_H, gH, temperature, cutoff):
    wH = [omega_H] # meV
    gH_dimless = [gH] # dim-less
    gH = numpy.array(gH_dimless) * wH
    nsites = 42
    
    J = 100 # meV
    energy_unit = J
    dJ = 0.0 # disable Peierls term
    wP = 6.2
    
    ham, classical_ph, quantum_ph, lattice, temperature = build_1d_Holstein_Peierls_model(
        J, dJ, wH, gH, wP, temperature,
        nsites,
        cutoff,
        energy_unit = energy_unit,
        model_type = 'bond'
    )
    
    # build propagator
    time_step = 0.01
    total_time = 50
    ntraj_per_rank = 50  
    propagator = DensityMatrixUnitaryPropagator(lattice.nsites, ntraj_per_rank, time_step, total_time, temperature)

    # assemble simulation and run
    simulation = GreenKuboSimulation(lattice, ham, classical_ph, quantum_ph, propagator)
    dump_dir = "currents_output"
    dump_interval = 200
    simulation.run(
        dump_dir=dump_dir,
        dump_interval=dump_interval
    )

import sys
if __name__ == "__main__":
    
    # Kelvin
    temperatures = [183.84761032548917, 291.3788260551113, 461.8043178420568, 731.9105195970242, 1160.0, 1838.4761032548918, 2913.788260551113, 4618.0431784205675, 7319.105195970243, 11600.0]
    temp_idx = int(sys.argv[1])
    temperature = temperatures[temp_idx] * K2au * au2ev * 1e3 # meV
    omega_H = 50
    reorg_e = 100
    gH = numpy.sqrt(reorg_e / omega_H)
    job_type = 'CPA'
    
    if job_type == 'CPA':
        cutoff = omega_H * 2
    elif job_type == 'LF':
        cutoff = omega_H * 0.5
    lf_cpa_transport(omega_H, gH, temperature, cutoff)