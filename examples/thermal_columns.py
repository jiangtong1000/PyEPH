"""Finite-temperature column CPA on a native complex disordered ring.

Atomic model units (hbar=1, charge=1, spacing=1 bohr) are explicit. This short
toy calculation demonstrates preparation and continuation, not material mobility.
Run with JAX_ENABLE_X64=1 PYTHONPATH=src python examples/thermal_columns.py --output DIR.
"""

import argparse
import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from pyeph import Execution, Integrator, Simulation
from pyeph.io.checkpoint import array_fingerprint
from pyeph.io.hdf5 import HDF5Observer
from pyeph.io.provenance import problem_manifest
from pyeph.models.lattice_epc import LatticeEPCModel
from pyeph.paths.harmonic import HarmonicBath
from pyeph.thermal import (
    ThermalFilterPlan, prepare_thermal_columns, require_success, thermal_random_columns,
)
from pyeph.workflows.column_transport import (
    initialize_column_transport_state, make_column_transport_problem,
)


def ring_problem(nstates=12):
    """Native sparse ring with complex bonds and local harmonic displacements."""
    if isinstance(nstates, bool) or not isinstance(nstates, (int, np.integer)) or nstates < 3:
        raise ValueError("nstates must be an integer >=3")
    n = int(nstates)
    index = np.arange(n, dtype=np.int32)
    onsite = .35*np.sin(.71*index)+.13*np.cos(1.37*index)
    hopping = (-.45+.07*np.cos(.23*index))*np.exp(.17j*np.sin(.53*index))
    frequency, coupling = .7, .16
    model = LatticeEPCModel(n, 1, n)
    params = dict(rows=jnp.asarray(np.r_[index, index, (index+1) % n]),
                  columns=jnp.asarray(np.r_[index, (index+1) % n, index]),
                  static=jnp.asarray(np.r_[onsite, hopping, hopping.conj()]),
                  displacements=jnp.asarray(np.c_[np.r_[np.zeros(n), np.ones(n), -np.ones(n)],
                                                  np.zeros(3*n)]),
                  term_edges=jnp.asarray(index), field_indices=jnp.asarray(index),
                  coefficients=jnp.full(n, coupling), frequencies=jnp.array([frequency]),
                  canonical_frequencies=jnp.full(n, frequency))
    problem = make_column_transport_problem(model, params, HarmonicBath([frequency]))
    q, p = jnp.asarray(.17*np.sin(.41*index)), jnp.asarray(.11*np.cos(.67*index))
    # These host arrays also define the independent small-system oracle.
    inputs = dict(onsite=onsite, hopping=hopping, frequency=frequency, coupling=coupling,
                  q0=np.asarray(q), p0=np.asarray(p))
    return problem, q, p, inputs


def gershgorin_bounds(inputs):
    diagonal = inputs["onsite"]+inputs["coupling"]*np.sqrt(2*inputs["frequency"])*inputs["q0"]
    radius = abs(inputs["hopping"])+abs(np.roll(inputs["hopping"], 1))
    # Explicit rounding cushion; not an interval-arithmetic certificate.
    cushion = 1e-10*max(1., float(np.max(abs(diagonal)+radius)))
    return float(np.min(diagonal-radius)-cushion), float(np.max(diagonal+radius)+cushion)


def prepare(problem, q, p, inputs, *, beta=3., column_ids=(0, 1, 2, 3), seed=71,
            trajectory_id=5, polynomial_atol=1e-12, rtol=1e-6):
    """Prepare and accept a factor, then bind its recipe to the stored origin."""
    identity = array_fingerprint({"params": problem.params, "q0": q})
    plan = ThermalFilterPlan(beta, *gershgorin_bounds(inputs),
        action_id="native-ring-params-q0-sha256:"+identity,
        bounds_id="analytic-ring-gershgorin-with-rounding-cushion-v1:"+identity,
        polynomial_atol=polynomial_atol)
    columns = thermal_random_columns(problem.model.nstates, column_ids,
                                      seed=seed, trajectory_id=trajectory_id)
    kernel = jax.jit(lambda parameters, coordinate, omega: prepare_thermal_columns(
        lambda vectors: problem.model.apply(parameters, coordinate, vectors), omega, plan, rtol=rtol))
    prepared = kernel(problem.params, q, columns)
    initial = initialize_column_transport_state(problem, q, p, require_success(prepared),
                                                seed=seed, trajectory_id=trajectory_id)
    rank = columns.shape[1]
    stored_factor = np.asarray(initial.electronic[:, :rank])
    digest = hashlib.sha256(stored_factor.tobytes(order="C")).hexdigest()
    origin_digest = np.asarray(initial.method_state["column_transport"]["factor_digest"]).tobytes().hex()
    if digest != origin_digest:
        raise AssertionError("stored factor and column-origin digest disagree")
    record = dict(plan=plan.metadata(), diagnostics=prepared.diagnostics(),
                  seed=int(seed), trajectory_id=int(trajectory_id),
                  column_ids=np.atleast_1d(column_ids).tolist(), random_kind="complex",
                  random_namespace="threefry trajectory/0x54484D46/column_id",
                  factor_digest=origin_digest, factor_dtype=str(stored_factor.dtype),
                  factor_shape=list(stored_factor.shape),
                  estimator="one pooled finite-column ratio at one nuclear geometry; generally biased",
                  uncertainty="polynomial, trace, nuclear and time-discretization errors are separate")
    return initial, record, columns


def run(output, *, nstates=12, ncolumns=4, beta=3., dt=.02, steps=40, seed=71,
        trajectory_id=5):
    if not jax.config.x64_enabled:
        raise ValueError("this scientific recipe requires JAX_ENABLE_X64=1")
    if isinstance(ncolumns, bool) or not isinstance(ncolumns, int) or ncolumns < 1:
        raise ValueError("ncolumns must be a positive integer")
    problem, q, p, inputs = ring_problem(nstates)
    initial, preparation, omega = prepare(problem, q, p, inputs, beta=beta,
        column_ids=np.arange(ncolumns), seed=seed, trajectory_id=trajectory_id)
    simulation = Simulation(problem, Integrator(dt, "rk4"), Execution(chunk_size=16))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    manifest = problem_manifest(problem, simulation.integrator)
    record = dict(preparation=preparation, simulation_manifest=manifest,
                  requested_steps=steps, example_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    np.savez_compressed(output/"initial_preparation.npz", omega=omega,
                        factor=np.asarray(initial.electronic[:, :ncolumns]), **inputs)
    (output/"preparation.json").write_text(json.dumps(record, indent=2, allow_nan=False)+"\n")
    with HDF5Observer(output/"correlation.h5", metadata=record) as observer:
        result = simulation.run(initial, steps, observer=observer)
    simulation.save_checkpoint(output/"checkpoint.h5", result.final_state)
    # Resume from this checkpoint with the same problem/configuration; do not
    # regenerate a factor or replace the original current insertions.
    return result, record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--nstates", type=int, default=12)
    parser.add_argument("--ncolumns", type=int, default=4)
    parser.add_argument("--beta", type=float, default=3.)
    parser.add_argument("--dt", type=float, default=.02)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--seed", type=int, default=71)
    parser.add_argument("--trajectory-id", type=int, default=5)
    result, record = run(**vars(parser.parse_args()))
    print(json.dumps(dict(final_time=float(result.final_state.time),
                          preparation=record["preparation"]), indent=2))
