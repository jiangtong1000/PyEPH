"""Optional real-MPI takeover acceptance; launched with python -m mpi4py.

FI_TCP_IFACE=lo0 USE_MPI=true JAX_ENABLE_X64=1 mpiexec -n 2 python -m mpi4py \
    tests/mpi/greenkubo_partition.py --output /fresh/output/directory

No pytest collection and no performance claim: this checks identical-sample
partition/merge, globally unique IDs, and resumed-segment time metadata.
"""

import argparse
import importlib.metadata
import json
import os
from pathlib import Path

import h5py
from mpi4py import MPI
import numpy as np

from pyeph.greenkubo.propagator import DensityMatrixUnitaryPropagator
from pyeph.greenkubo.simulation import GreenKuboSimulation
from pyeph.greenkubo.typical_model_helper import build_1d_Holstein_Peierls_model


def make_simulation(ntraj=2, initial_samples=None, seed=1120):
    ham, classical, quantum, lattice, temperature = build_1d_Holstein_Peierls_model(
        1., .15, [1.5], [.25], .3, .7, 4, .5,
    )
    propagator = DensityMatrixUnitaryPropagator(lattice.nsites, ntraj, .01, .05, temperature)
    return GreenKuboSimulation(lattice, ham, classical, quantum, propagator,
                              initial_samples=initial_samples, base_seed=seed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    comm = MPI.COMM_WORLD
    rank, size = comm.Get_rank(), comm.Get_size()
    if size != 2:
        raise ValueError("this acceptance case requires exactly two MPI ranks")
    if rank == 0:
        args.output.mkdir(parents=True, exist_ok=False)
    comm.Barrier()
    simulation = make_simulation()
    local_samples = simulation.classic_ph.initial_samples()
    all_samples = comm.gather(local_samples, root=0)
    all_ids = comm.gather(np.asarray(simulation.initial_state.trajectory_id), root=0)
    first = simulation.run(args.output/"first", steps=2, dump_interval=1, collect=True)
    simulation.save_checkpoint(args.output/f"checkpoint_{rank}.h5")
    restarted = make_simulation(seed=887)
    restarted.load_checkpoint(args.output/f"checkpoint_{rank}.h5")
    second = restarted.run(args.output/"resumed", dump_interval=1, collect=True)
    for actual, expected in zip(restarted.classic_ph.initial_samples(), local_samples):
        np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(restarted.initial_state.trajectory_id, simulation.initial_state.trajectory_id)
    local_c = np.concatenate((first.observables["current_correlation"],
                              second.observables["current_correlation"][1:]), axis=0)
    all_c = comm.gather(local_c, root=0)
    comm.Barrier()
    if rank == 0:
        np.testing.assert_array_equal(np.concatenate(all_ids), np.arange(4, dtype=np.uint32))
        samples = tuple(np.concatenate([item[index] for item in all_samples], axis=1) for index in (0, 1))
        # Rank zero performs an explicitly serial, identical-sample reference;
        # do not accidentally enter another MPI collective in its constructor.
        os.environ["USE_MPI"] = "false"
        serial = make_simulation(4, samples)
        expected = serial.run()
        os.environ["USE_MPI"] = "true"
        actual = np.concatenate(all_c, axis=1)
        np.testing.assert_allclose(actual, expected.observables["current_correlation"], rtol=1e-12, atol=1e-12)
        with h5py.File(args.output/"resumed/collected_current_autocorr.h5") as handle:
            np.testing.assert_allclose(handle["time"], [.02, .03, .04], atol=1e-14)
            assert handle.attrs["initial_time"] == .02
            np.testing.assert_allclose(handle["current_x"], actual[2:, :, 0].mean(axis=1), atol=1e-12)
            rank_means = np.array([value[2:, :, 0].mean(axis=1) for value in all_c])
            np.testing.assert_allclose(handle["current_x_std"], rank_means.std(axis=0), atol=1e-12)
        for other_rank in range(2):
            with h5py.File(args.output/f"resumed/initial_samples_{other_rank}.h5") as handle:
                np.testing.assert_array_equal(handle["trajectory_id"], all_ids[other_rank])
                np.testing.assert_array_equal(handle["q0"], all_samples[other_rank][0])
                np.testing.assert_array_equal(handle["p0"], all_samples[other_rank][1])
        report = {"ranks": size, "trajectories_per_rank": 2, "method": "full local LF-CPA",
                  "thermal_policy": "legacy_full", "steps": 4, "dt": .01,
                  "max_abs_partition_error": float(np.max(np.abs(actual-expected.observables["current_correlation"]))),
                  "precision": "float64/complex128", "mpi_library": MPI.Get_library_version(),
                  "dependencies": {name: importlib.metadata.version(name) for name in ("jax", "numpy", "mpi4py")},
                  "checks": ["identical saved samples", "rank-offset global IDs", "native dynamics",
                             "historical HDF5 reduction", "exact checkpoint sample restoration",
                             "resumed explicit time grid", "rank-mean standard deviation"]}
        (args.output/"acceptance.json").write_text(json.dumps(report, indent=2)+"\n")
        print(json.dumps(report, indent=2), flush=True)
    comm.Barrier()


if __name__ == "__main__":
    main()
