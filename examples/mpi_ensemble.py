#!/usr/bin/env python3
"""Actual MPI orchestration of independent native Ehrenfest trajectories.

From the source root, with PyEPH's mpi extra and a compatible MPI runtime::

    JAX_ENABLE_X64=1 mpiexec -n 2 python -m mpi4py examples/mpi_ensemble.py

The default seven- and one-trajectory cases verify unequal and empty rank
partitions against serial results. This is a correctness example, not a scaling
benchmark. It does not distribute one Hamiltonian across ranks.
"""

import argparse
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import time

import jax
import jax.numpy as jnp
import numpy as np
from mpi4py import MPI

import pyeph
from pyeph import CoupledClassical, Ehrenfest, Execution, Integrator, Problem, Simulation
from pyeph.core.state import make_state, stack_states
from pyeph.execution.ensemble import EnsembleResult, merge_ensembles, partition_ids, run_ensemble
from pyeph.initialization import sample_harmonic
from pyeph.models.analytic import SpinBosonModel
from pyeph.observables.population import FunctionalMeasurement

SEED = 724
TEMPERATURE = .15
FREQUENCIES = (.4, .7)
ELECTRONIC_INITIAL = (np.sqrt(.7), 1j*np.sqrt(.3))


def measurement(problem, state):
    """Only explicit state data; no hidden provider weights or mutable closure."""
    c = state.electronic
    return {"population": jnp.abs(c)**2,
            "coherence_01": c[0]*jnp.conj(c[1]), "coordinates": state.q}


def initialize(ids):
    """The global trajectory ID determines each draw, independently of rank."""
    q, p = sample_harmonic(FREQUENCIES, 1., TEMPERATURE, ids, seed=SEED,
                           distribution="wigner")
    return stack_states([make_state(q[i], p[i], ELECTRONIC_INITIAL,
                                    trajectory_id=int(identity), seed=SEED)
                         for i, identity in enumerate(ids)])


def build_simulation():
    model = SpinBosonModel(2)
    params = model.default_params() | {
        "omega": jnp.asarray(FREQUENCIES), "coupling": jnp.array([.03, -.02]),
        "delta": .07, "bias": .01}
    return Simulation(Problem(model, params, CoupledClassical(1.), Ehrenfest(),
                              FunctionalMeasurement(measurement)),
                      Integrator(.02, electronic="exponential_midpoint"),
                      Execution(chunk_size=13, save_every=7))


def identities():
    script = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    preparation = {"script_sha256": script, "seed": SEED, "distribution": "wigner",
                   "temperature": TEMPERATURE, "frequencies": FREQUENCIES, "mass": 1.,
                   "electronic_initial_real": np.real(ELECTRONIC_INITIAL).tolist(),
                   "electronic_initial_imag": np.imag(ELECTRONIC_INITIAL).tolist()}
    prep_digest = hashlib.sha256(json.dumps(preparation, sort_keys=True).encode()).hexdigest()
    # This explicit assertion covers this top-level, state-only function's
    # implementation. Relevant library versions are also in the run manifest.
    artifacts = {"measurement.function": "sha256:"+script}
    return "sha256:"+prep_digest, artifacts, preparation


def data_payload(result):
    """Transfer numerical/JSON fields, not executable models or compiled code."""
    if result is None:
        return None
    return {"times": np.asarray(result.times), "trajectory_ids": np.asarray(result.trajectory_ids),
            "mean": jax.tree.map(np.asarray, result.mean), "m2": jax.tree.map(np.asarray, result.m2),
            "simulation_manifest": result.simulation_manifest,
            "preparation_id": result.preparation_id}


def remember_final(destination):
    def record(ids, result):
        host = jax.device_get(result.final_state)._asdict()
        for i, identity in enumerate(ids):
            destination[int(identity)] = jax.tree.map(lambda x: np.asarray(x)[i], host)
    return record


def tree_error(actual, expected, *, undefined=False):
    assert jax.tree.structure(actual) == jax.tree.structure(expected)
    errors = []
    for a, b in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
        a, b = np.asarray(a), np.asarray(b)
        assert a.shape == b.shape
        if undefined:
            assert np.isnan(a).all() and np.isnan(b).all()
        else:
            np.testing.assert_allclose(a, b, atol=2e-12, rtol=2e-12)
            errors.append(float(np.max(abs(a-b))))
    return None if undefined else max(errors, default=0.)


def state_error(actual, expected):
    assert set(actual) == set(expected)
    largest = 0.
    for identity in actual:
        assert jax.tree.structure(actual[identity]) == jax.tree.structure(expected[identity])
        for a, b in zip(jax.tree.leaves(actual[identity]), jax.tree.leaves(expected[identity]), strict=True):
            a, b = np.asarray(a), np.asarray(b)
            if a.dtype.kind in "biu":
                np.testing.assert_array_equal(a, b)
            else:
                np.testing.assert_allclose(a, b, rtol=2e-12, atol=2e-12)
                largest = max(largest, float(np.max(abs(a-b))))
    return largest


def verify_case(comm, count, steps, batch_size, simulation, preparation_id, artifact_ids):
    rank, size = comm.Get_rank(), comm.Get_size()
    ids = np.arange(count, dtype=np.uint32)*37+17
    local_ids = partition_ids(ids, rank, size)
    local_states = {}
    # Empty ranks must still participate in every collective, but run_ensemble
    # deliberately requires a nonempty ID list: skip only the local simulation.
    partial = (run_ensemble(simulation, initialize, local_ids, steps, batch_size=batch_size,
                            preparation_id=preparation_id, artifact_ids=artifact_ids,
                            batch_observer=remember_final(local_states))
               if len(local_ids) else None)
    packages = comm.gather({"rank": rank, "ids": local_ids.tolist(),
                            "ensemble": data_payload(partial), "final_states": local_states}, root=0)
    report = None
    if rank == 0:
        merged, distributed_states = None, {}
        for package in packages:
            payload = package["ensemble"]
            if payload is not None:
                item = EnsembleResult(**payload).validate()
                merged = item if merged is None else merge_ensembles(merged, item)
            assert not (set(distributed_states) & set(package["final_states"]))
            distributed_states.update(package["final_states"])
        assert merged is not None and merged.count == count
        np.testing.assert_array_equal(np.sort(merged.trajectory_ids), ids)
        serial_states = {}
        # Different batching is intentional: neither rank nor local batch size
        # is allowed to determine random draws or trajectory identity.
        serial = run_ensemble(simulation, initialize, ids, steps, batch_size=3,
                              preparation_id=preparation_id, artifact_ids=artifact_ids,
                              batch_observer=remember_final(serial_states))
        errors = {"mean": tree_error(merged.mean, serial.mean),
                  "m2": tree_error(merged.m2, serial.m2),
                  "standard_error": tree_error(merged.standard_error, serial.standard_error,
                                                undefined=count == 1),
                  "final_state_by_global_id": state_error(distributed_states, serial_states)}
        np.testing.assert_allclose(merged.times, serial.times, atol=1e-15, rtol=0)
        raw = simulation.run(initialize(ids), steps)
        direct_mean = jax.tree.map(lambda x: np.mean(x, axis=1), raw.observables)
        direct_m2 = jax.tree.map(lambda x: np.sum(abs(x-x.mean(axis=1, keepdims=True))**2, axis=1),
                                 raw.observables)
        errors["direct_numpy_mean"] = tree_error(merged.mean, direct_mean)
        errors["direct_numpy_m2"] = tree_error(merged.m2, direct_m2)
        report = {"trajectories": count, "global_ids": ids.tolist(),
                  "rank_partitions": [{"rank": x["rank"], "ids": x["ids"], "count": len(x["ids"])}
                                      for x in packages],
                  "empty_ranks": [x["rank"] for x in packages if not x["ids"]],
                  "rank_batch_size": batch_size, "serial_batch_size": 3,
                  "steps": steps, "dt": simulation.integrator.dt,
                  "output_times": merged.times.tolist(), "count_after_merge": merged.count,
                  "maximum_absolute_errors": errors, "integer_state_fields_equal": True,
                  "single_sample_standard_error_undefined": count == 1,
                  "coherence_mean_final_real": float(merged.mean["coherence_01"][-1].real),
                  "coherence_mean_final_imag": float(merged.mean["coherence_01"][-1].imag),
                  "coherence_m2_final": float(merged.m2["coherence_01"][-1]),
                  "simulation_fingerprint": merged.simulation_manifest["fingerprint"],
                  "source_evidence": merged.simulation_manifest["payload"]["source"],
                  "passed": True}
    # Every rank reaches this collective, including ranks whose partition was
    # empty and ranks waiting while rank zero performs the serial reference.
    passed = comm.bcast(report["passed"] if rank == 0 else None, root=0)
    assert passed
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--counts", nargs="+", type=int, default=[7, 1])
    parser.add_argument("--steps", type=int, default=56)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--output", type=Path, default=Path("outputs/mpi_ensemble.json"),
                        help="JSON report path (default: %(default)s)")
    options = parser.parse_args()
    if min(options.counts+[options.steps, options.batch_size]) < 1 or max(options.counts)*37+17 >= 2**32:
        parser.error("counts, steps and batch size must be positive; generated IDs must fit uint32")
    jax.config.update("jax_enable_x64", True)
    comm = MPI.COMM_WORLD
    rank, size = comm.Get_rank(), comm.Get_size()
    if size < 2:
        raise ValueError("this validation requires at least two real MPI ranks; use mpiexec -n 2")
    preparation_id, artifact_ids, preparation = identities()
    simulation = build_simulation()
    source_start = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    hosts = comm.allgather(MPI.Get_processor_name())
    rank_runtime = comm.gather({"rank": rank, "backend": jax.default_backend(),
                                "devices": [str(x) for x in jax.devices()],
                                "python": platform.python_version(), "jax": jax.__version__,
                                "numpy": np.__version__, "script_sha256": source_start}, root=0)
    try:
        mpich_distribution = metadata.version("mpich")
    except metadata.PackageNotFoundError:
        mpich_distribution = None  # Other compatible MPI installations are allowed.
    report = {"timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "scope": "actual MPI orchestration and correctness; no timing/scaling claims",
              "mpi_ranks": size, "distinct_hosts": len(set(hosts)),
              "mpi_standard_version": list(MPI.Get_version()), "mpi_library_version": MPI.Get_library_version(),
              "mpi4py": metadata.version("mpi4py"), "mpich_distribution": mpich_distribution,
              "pyeph": pyeph.__version__, "jaxlib": metadata.version("jaxlib"),
              "scipy": metadata.version("scipy"), "platform": platform.platform(),
              "precision": "float64/complex128", "loopback_interface": os.environ.get("FI_TCP_IFACE"),
              "rank_runtime": rank_runtime, "preparation": preparation,
              "preparation_id": preparation_id, "artifact_ids": artifact_ids, "cases": []}
    for count in options.counts:
        case = verify_case(comm, count, options.steps, options.batch_size, simulation,
                           preparation_id, artifact_ids)
        if rank == 0:
            report["cases"].append(case)
            print(json.dumps(case, sort_keys=True), flush=True)
    source_end = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    assert source_end == source_start, "example source changed during MPI validation"
    if rank == 0:
        assert len({x["script_sha256"] for x in rank_runtime}) == 1
        report["passed"] = True
        options.output.parent.mkdir(parents=True, exist_ok=True)
        options.output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")


if __name__ == "__main__":
    main()
