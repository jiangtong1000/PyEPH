"""Large finite-T column qualification with independent sparse references.

No global dense Hamiltonian, density or unitary is used. Exact-filter checks
condition on the same random columns, not on an exact thermal trace. Timings
are opt-in and require an otherwise idle CPU; correctness can run separately.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import re
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np
import scipy
from scipy.sparse import coo_matrix, eye
from scipy.sparse.linalg import expm_multiply

import pyeph
from pyeph import Execution, Integrator, Simulation
from pyeph.models.lattice_epc import LatticeEPCModel
from pyeph.paths.harmonic import HarmonicBath
from pyeph.thermal import ThermalFilterPlan, prepare_thermal_columns, require_success, thermal_random_columns
from pyeph.workflows.column_transport import initialize_column_transport_state, make_column_transport_problem


def fixture(family, n):
    """Explicit complex HP/nonlocal EPC coefficients in canonical coordinates.

    Atomic model units: hbar=|charge|=spacing=mass=1. Two oscillator modes per
    site; no physical-material or old-PyEPH parameter-equivalence claim.
    Every hopping term has its explicitly conjugated reverse. The nonlocal
    case couples long hops to displaced oscillator sites, without dense G.
    """
    if family not in ("hp", "nonlocal") or n < 8:
        raise ValueError("require family hp/nonlocal and at least8 states")
    i = np.arange(n, dtype=np.int32)
    frequencies = np.array([.7, 1.1])
    omega = np.repeat(frequencies, n)
    onsite = .35*np.sin(.71*i)+.13*np.cos(1.37*i)
    jumps = (1,) if family == "hp" else (1, 3, 7)
    rows, columns, static, displacement = [i], [i], [onsite.astype(complex)], [np.zeros(n)]
    term_edges, fields, coefficients = [i], [i], [np.full(n, .16+0j)]
    if family == "nonlocal":
        term_edges.append(i)
        fields.append((i+13) % n)
        coefficients.append(np.full(n, .035+0j))
    offset = n
    for jump in jumps:
        j = (i+jump) % n
        phase = np.exp(.17j*np.sin(.53*i+.13*jump))
        hopping = (-.45+.07*np.cos(.23*i))*phase/jump**1.3
        rows.extend((i, j))
        columns.extend((j, i))
        static.extend((hopping, hopping.conj()))
        displacement.extend((np.full(n, jump), np.full(n, -jump)))
        if family == "hp":
            indexes = (n+j, n+i)
            strengths = (.05*phase, -.05*phase)
        else:
            indexes = (n+(i+11) % n, n+(i+23) % n)
            strengths = (.025*phase/np.sqrt(jump), -.015*phase/np.sqrt(jump))
        for coordinate, strength in zip(indexes, strengths, strict=True):
            term_edges.extend((offset+i, offset+n+i))
            fields.extend((coordinate, coordinate))
            coefficients.extend((strength, strength.conj()))
        offset += 2*n
    host = dict(rows=np.concatenate(rows), columns=np.concatenate(columns),
                static=np.concatenate(static), displacement=np.concatenate(displacement),
                term_edges=np.concatenate(term_edges), field_indices=np.concatenate(fields),
                coefficients=np.concatenate(coefficients), frequencies=frequencies,
                canonical_frequencies=omega,
                q0=.17*np.sin(.41*np.arange(2*n)), p0=.11*np.cos(.67*np.arange(2*n)))
    params = {name: jnp.asarray(value) for name, value in host.items()
              if name not in ("displacement", "q0", "p0")}
    params["displacements"] = jnp.asarray(np.c_[host["displacement"], np.zeros(offset)])
    problem = make_column_transport_problem(LatticeEPCModel(n, 2, n), params, HarmonicBath(omega))
    return problem, jnp.asarray(host["q0"]), jnp.asarray(host["p0"]), host


def sparse_matrices(host, time_value, phase=0.):
    """Independent NumPy coefficient updates and SciPy CSR contractions."""
    omega = host["canonical_frequencies"]
    q = host["q0"]*np.cos(omega*time_value)+host["p0"]/omega*np.sin(omega*time_value)
    values = host["static"].copy()
    coordinate = host["field_indices"]
    np.add.at(values, host["term_edges"],
              host["coefficients"]*np.sqrt(2*omega[coordinate])*q[coordinate])
    values *= np.exp(1j*phase*host["displacement"])
    n = len(q)//2
    indexes = (host["rows"], host["columns"])
    h = coo_matrix((values, indexes), shape=(n, n)).tocsr()
    current = coo_matrix((1j*host["displacement"]*values, indexes), shape=(n, n)).tocsr()
    return h, current


def sparse_filter(matrix, omega, plan):
    shifted = (-plan.beta/2)*(matrix-plan.lower*eye(matrix.shape[0], format="csr"))
    y = expm_multiply(shifted, omega, traceA=shifted.diagonal().sum())
    norm = np.linalg.norm(y)
    return y/norm, float(2*np.log(norm)-np.log(omega.shape[1])-plan.beta*plan.lower)


def sparse_rk4(host, initial, dt, steps, stride):
    """Independent stage assembly at absolute times, including J(t) updates."""
    block, rank = np.array(initial, copy=True), initial.shape[1]//2
    times, correlation = [0.], []

    def observe(time_value):
        current = sparse_matrices(host, time_value)[1]
        return np.vdot(block[:, :rank], current@block[:, rank:])

    correlation.append(observe(0.))
    for step in range(steps):
        t = step*dt
        h0 = sparse_matrices(host, t)[0]
        hhalf = sparse_matrices(host, t+dt/2)[0]
        hend = sparse_matrices(host, t+dt)[0]
        k1 = -1j*(h0@block)
        k2 = -1j*(hhalf@(block+dt*k1/2))
        k3 = -1j*(hhalf@(block+dt*k2/2))
        k4 = -1j*(hend@(block+dt*k3))
        block += dt*(k1+2*k2+2*k3+k4)/6
        if (step+1) % stride == 0 or step+1 == steps:
            times.append((step+1)*dt)
            correlation.append(observe((step+1)*dt))
    return block, np.asarray(times), np.asarray(correlation)


def compiled_evidence(executable, n, path):
    hlo = executable.as_text()
    path.write_text(hlo)
    shapes = sorted(set(re.findall(r"\b(?:[a-z]+\d+)\[([0-9,]+)\]", hlo)))
    square = [shape for shape in shapes
              if shape.split(",").count(str(n)) >= 2 or str(n*n) in shape.split(",")]
    if square:
        raise AssertionError(f"global electronic square buffers found: {square}")
    memory = executable.memory_analysis()
    return dict(hlo_sha256=hashlib.sha256(hlo.encode()).hexdigest(),
                optimized_array_shapes=shapes, detected_global_square_shapes=square,
                compiler_memory={name: getattr(memory, name) for name in (
                    "argument_size_in_bytes", "output_size_in_bytes", "temp_size_in_bytes", "alias_size_in_bytes")})


def run_case(output, family, n, rank, beta, *, dt=.02, steps=32, stride=4, repeats=7, timing=False):
    started = perf_counter()
    problem, q, p, host = fixture(family, n)
    h, current = sparse_matrices(host, 0.)
    setup_seconds = perf_counter()-started
    diagonal = h.diagonal().real
    radius = np.asarray(abs(h).sum(axis=1)).ravel()-abs(h.diagonal())
    cushion = 1e-10*max(1., float(np.max(abs(diagonal)+radius)))
    lower, upper = float(np.min(diagonal-radius)-cushion), float(np.max(diagonal+radius)+cushion)
    identity = hashlib.sha256(b"".join(host[name].tobytes() for name in sorted(host))).hexdigest()
    started = perf_counter()
    plan = ThermalFilterPlan(beta, lower, upper, action_id="sparse-fixture-sha256:"+identity,
                             bounds_id="csr-gershgorin-with-rounding-cushion:"+identity)
    plan_seconds = perf_counter()-started
    ids = np.arange(rank)*5+2
    started = perf_counter()
    omega = thermal_random_columns(n, ids, seed=93, trajectory_id=17)
    omega.block_until_ready()
    rng_seconds = perf_counter()-started
    omega_host = np.asarray(omega)
    kernel = jax.jit(lambda params, q, omega: prepare_thermal_columns(
        lambda block: problem.model.apply(params, q, block), omega, plan))
    started = perf_counter()
    prepared_executable = kernel.lower(problem.params, q, omega).compile()
    filter_compile_seconds = perf_counter()-started
    filter_hlo = compiled_evidence(prepared_executable, n, output/"filter.hlo.txt")
    started = perf_counter()
    prepared = prepared_executable(problem.params, q, omega)
    jax.block_until_ready(prepared)
    factor = require_success(prepared)
    filter_first_seconds = perf_counter()-started
    started = perf_counter()
    exact, exact_log_partition = sparse_filter(h, omega_host, plan)
    scipy_first_seconds = perf_counter()-started
    factor_error = float(np.linalg.norm(np.asarray(factor)-exact))
    log_partition_error = abs(float(prepared.log_partition_estimate)-exact_log_partition)
    started = perf_counter()
    initial = initialize_column_transport_state(problem, q, p, factor, seed=93, trajectory_id=17)
    jax.block_until_ready(initial)
    initialization_seconds = perf_counter()-started
    stored_factor = np.asarray(initial.electronic[:, :rank])
    factor_digest = np.asarray(initial.method_state["column_transport"]["factor_digest"]).tobytes().hex()
    assert hashlib.sha256(stored_factor.tobytes()).hexdigest() == factor_digest
    independent_initial = np.concatenate((exact, current@exact), axis=1)
    factor_initial = np.concatenate((stored_factor, current@stored_factor), axis=1)
    np.testing.assert_allclose(initial.electronic, factor_initial, rtol=2e-14, atol=2e-15)
    simulation = Simulation(problem, Integrator(dt, "rk4"), Execution(chunk_size=steps, save_every=stride))
    selected = np.unique(np.r_[np.arange(stride, steps+1, stride), steps])-1
    started = perf_counter()
    executable = simulation._block(False, steps, selected).lower(problem.params, initial).compile()
    transport_compile_seconds = perf_counter()-started
    transport_hlo = compiled_evidence(executable, n, output/"transport.hlo.txt")
    started = perf_counter()
    result = simulation.run(initial, steps)
    jax.block_until_ready((result.final_state, result.observables, result.times))
    public_first_seconds = perf_counter()-started
    actual = np.asarray(result.observables["current_correlation"][:, 0, 0])
    reference, times, correlation = sparse_rk4(host, factor_initial, dt, steps, stride)
    exact_reference, _, exact_correlation = sparse_rk4(host, independent_initial, dt, steps, stride)
    fine, fine_times, fine_correlation = sparse_rk4(host, independent_initial, dt/2, 2*steps, 2*stride)
    np.testing.assert_allclose(result.times, times, atol=2e-15)
    np.testing.assert_allclose(fine_times, times, atol=2e-15)
    accuracy = dict(same_omega_normalized_factor_frobenius_error=factor_error,
        log_partition_absolute_error=log_partition_error,
        public_vs_sparse_rk4_column_max_error=float(np.max(abs(np.asarray(result.final_state.electronic)-reference))),
        public_vs_sparse_rk4_correlation_max_error=float(np.max(abs(actual-correlation))),
        preparation_only_same_rk4_correlation_max_error=float(np.max(abs(correlation-exact_correlation))),
        combined_vs_fine_sparse_rk4_correlation_max_error=float(np.max(abs(actual-fine_correlation))),
        independent_dt_half_column_max_delta=float(np.max(abs(exact_reference-fine))),
        independent_dt_half_correlation_max_delta=float(np.max(abs(exact_correlation-fine_correlation))))
    assert factor_error < 2e-10 and log_partition_error < 2e-10
    assert accuracy["public_vs_sparse_rk4_column_max_error"] < 2e-11
    assert accuracy["public_vs_sparse_rk4_correlation_max_error"] < 2e-10
    assert accuracy["independent_dt_half_correlation_max_delta"] < 1e-7

    records = {"filter_action": [], "filter_scipy_sparse": [], "public_transport": [], "compiled_transport": []}
    if timing:
        def action_filter():
            value = prepared_executable(problem.params, q, omega)
            jax.block_until_ready(value)
            require_success(value)
            return value

        functions = dict(filter_action=action_filter,
                         filter_scipy_sparse=lambda: sparse_filter(h, omega_host, plan),
                         public_transport=lambda: simulation.run(initial, steps),
                         compiled_transport=lambda: executable(problem.params, initial))
        for function in functions.values():
            value = function()
            jax.block_until_ready((value.final_state, value.observables, value.times)
                                  if hasattr(value, "final_state") else value)
        for repeat in range(repeats):
            names = tuple(functions) if repeat % 2 == 0 else tuple(reversed(functions))
            for name in names:
                started = perf_counter()
                value = functions[name]()
                jax.block_until_ready((value.final_state, value.observables, value.times)
                                      if hasattr(value, "final_state") else value)
                records[name].append(perf_counter()-started)
    arrays = dict(omega=omega_host, polynomial_factor=stored_factor, exact_filter_factor=exact,
                  initial_electronic=np.asarray(initial.electronic), final_electronic=np.asarray(result.final_state.electronic),
                  sparse_rk4_final=reference, exact_filter_sparse_rk4_final=exact_reference,
                  fine_sparse_rk4_final=fine, times=times, correlation=actual,
                  sparse_rk4_correlation=correlation, exact_filter_sparse_rk4_correlation=exact_correlation,
                  fine_sparse_rk4_correlation=fine_correlation,
                  **{"input_"+name: value for name, value in host.items()})
    np.savez_compressed(output/"arrays.npz", **arrays)
    report = dict(family=family, nstates=n, rank=rank, beta=beta, dt=dt, steps=steps, stride=stride,
        preparation_plan=plan.metadata(), diagnostics=prepared.diagnostics(),
        factor_digest=factor_digest, trace_seed=93, trajectory_id=17, column_ids=ids.tolist(),
        same_omega_reference_log_partition=exact_log_partition, accuracy=accuracy,
        setup_seconds=setup_seconds, plan_seconds=plan_seconds, random_columns_seconds=rng_seconds,
        filter_compile_seconds=filter_compile_seconds, filter_first_seconds=filter_first_seconds,
        scipy_sparse_filter_first_seconds=scipy_first_seconds, public_initialization_seconds=initialization_seconds,
        transport_compile_seconds=transport_compile_seconds, public_transport_first_seconds=public_first_seconds,
        fair_timing_requested=timing, warmed_seconds=records,
        warmed_median_seconds={name: (float(np.median(values)) if values else None) for name, values in records.items()},
        filter_compiled=filter_hlo, transport_compiled=transport_hlo,
        state_bytes=sum(np.asarray(value).nbytes for value in jax.tree.leaves(initial)),
        electronic_state_bytes=np.asarray(initial.electronic).nbytes,
        parameter_bytes=sum(np.asarray(value).nbytes for value in jax.tree.leaves(problem.params)),
        csr_storage_bytes=sum(value.nbytes for matrix in (h, current)
                              for value in (matrix.data, matrix.indices, matrix.indptr)),
        max_column_norm_squared_drift=float(np.max(result.observables["max_column_norm_squared_drift"])),
        arrays_sha256=hashlib.sha256((output/"arrays.npz").read_bytes()).hexdigest())
    (output/"report.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sizes", type=int, nargs="+", default=[1024, 4096])
    parser.add_argument("--ranks", type=int, nargs="+", default=[4, 16])
    parser.add_argument("--betas", type=float, nargs="+", default=[1., 8.])
    parser.add_argument("--families", choices=["hp", "nonlocal"], nargs="+", default=["hp", "nonlocal"])
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--timing", action="store_true")
    args = parser.parse_args()
    if not jax.config.x64_enabled or jax.default_backend() != "cpu":
        parser.error("this qualification requires scientific x64 on CPU")
    if min(args.sizes) < 8 or min(args.ranks) < 1 or args.steps < 1 or args.repeats < 1:
        parser.error("invalid sizes, ranks, steps or repetitions")
    args.output.mkdir(parents=True, exist_ok=False)
    root = Path(pyeph.__file__).resolve().parent

    def sources():
        return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(root.rglob("*.py"))}

    source_hashes = sources()
    driver_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    report = dict(started_utc=datetime.now(timezone.utc).isoformat(),
        versions=dict(python=platform.python_version(), jax=jax.__version__, numpy=np.__version__, scipy=scipy.__version__),
        source_sha256=source_hashes, driver_sha256=driver_hash,
        devices=list(map(str, jax.devices())), platform=platform.platform(),
        thread_environment={name: os.environ.get(name) for name in (
            "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")},
        execution_configuration={name: getattr(jax.config, name, None) for name in (
            "jax_default_prng_impl", "jax_threefry_partitionable", "jax_random_seed_offset", "jax_default_matmul_precision")},
        assumptions=["Caller-known Hermitian sparse model and initial-geometry Gershgorin spectral bounds.",
            "Expm_multiply checks the same random columns; no claim of exact finite-temperature trace.",
            "Independent RK4 checks discretization parity; dt/2 gives a separate sensitivity check.",
            "Trace columns are not nuclear samples; rank sampling evidence is the separate96-seed small-system study.",
            "Compiler buffers are not process peak RSS; HLO shape checks cover compiled filter and measured transport block.",
            "Warm timings exclude compilation, RNG, initial-factor/current insertion, CSR setup and output archival.",
            "Single-call timings in correctness-only runs are not fair performance measurements."], cases=[])
    for family in args.families:
        for n in args.sizes:
            for rank in args.ranks:
                for beta in args.betas:
                    name = f"{family}_N{n}_K{rank}_beta{beta:g}"
                    output = args.output/name
                    output.mkdir()
                    print(name, flush=True)
                    case = run_case(output, family, n, rank, beta, steps=args.steps,
                                    repeats=args.repeats, timing=args.timing)
                    report["cases"].append({"name": name, **case})
                    (args.output/"report.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    assert sources() == source_hashes, "runtime source changed"
    assert hashlib.sha256(Path(__file__).read_bytes()).hexdigest() == driver_hash
    report.update(complete=True, finished_utc=datetime.now(timezone.utc).isoformat())
    (args.output/"report.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")


if __name__ == "__main__":
    main()
