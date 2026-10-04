"""Independent finite-T column CPA accuracy and matched preparation timings.

Correctness uses NumPy/SciPy equations, including a dense time-dependent ODE.
Timing is opt-in and must be scheduled without other CPU workloads.
"""

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import time

import jax
import jax.numpy as jnp
import numpy as np
import scipy
from scipy.integrate import solve_ivp
from scipy.linalg import eigh, expm

import pyeph
from pyeph import Execution, Integrator, Simulation
from pyeph.thermal import ThermalFilterPlan, prepare_thermal_columns, require_success, thermal_random_columns
from pyeph.workflows.column_transport import initialize_column_transport_state


RECIPE_PATH = Path(__file__).resolve().parents[1]/"examples/thermal_columns.py"
SPEC = importlib.util.spec_from_file_location("thermal_column_recipe", RECIPE_PATH)
recipe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(recipe)


def independent_matrices(inputs, time_value):
    """Literal cyclic bonds and harmonic q(t); no model action/dense helper."""
    frequency = inputs["frequency"]
    q = inputs["q0"]*np.cos(frequency*time_value)+inputs["p0"]/frequency*np.sin(frequency*time_value)
    h = np.diag(inputs["onsite"]+inputs["coupling"]*np.sqrt(2*frequency)*q).astype(complex)
    current = np.zeros_like(h)
    for i, hopping in enumerate(inputs["hopping"]):
        j = (i+1) % len(h)
        h[i, j] += hopping
        h[j, i] += hopping.conjugate()
        current[i, j] += 1j*hopping
        current[j, i] -= 1j*hopping.conjugate()
    return h, current


def exact_factor(h, lower, beta, columns):
    value = expm(-beta*(h-lower*np.eye(len(h)))/2)@columns
    return value/np.linalg.norm(value)


def correctness(output, *, seeds=96):
    problem, q, p, inputs = recipe.ring_problem(7)
    initial, preparation, columns = recipe.prepare(problem, q, p, inputs, beta=3., column_ids=[2, 7, 12])
    lower = preparation["plan"]["spectral_interval"][0]
    h0, current0 = independent_matrices(inputs, 0.)
    exact = exact_factor(h0, lower, 3., np.asarray(columns))
    factor = np.asarray(initial.electronic[:, :3])
    exact_initial = initialize_column_transport_state(problem, q, p, exact, trajectory_id=5, seed=71)
    factor_error = float(np.linalg.norm(factor-exact))
    density_error = float(np.linalg.norm(factor@factor.conj().T-exact@exact.conj().T, ord="nuc"))
    plan = ThermalFilterPlan(3., *recipe.gershgorin_bounds(inputs),
                             action_id="oracle-native-ring", bounds_id="analytic-row-sums")
    complete = require_success(prepare_thermal_columns(
        lambda vectors: problem.model.apply(problem.params, q, vectors), jnp.eye(7), plan))
    exact_density = expm(-3.*h0)
    exact_density /= np.trace(exact_density)
    complete_density_error = float(np.linalg.norm(complete@complete.conj().T-exact_density))
    times = np.linspace(0., .8, 41)

    def rhs(t, flat):
        return (-1j*independent_matrices(inputs, t)[0]@flat.reshape(7, 7)).ravel()

    ode = solve_ivp(rhs, (0., .8), np.eye(7, dtype=complex).ravel(), method="DOP853",
                    t_eval=times, rtol=2e-12, atol=2e-14)
    if not ode.success:
        raise AssertionError(ode.message)
    unitaries = ode.y.T.reshape(-1, 7, 7)
    rho_exact = exact@exact.conj().T
    oracle = np.array([np.trace(independent_matrices(inputs, t)[1]@u@current0@rho_exact@u.conj().T)
                       for t, u in zip(times, unitaries, strict=True)])
    trajectory = {}
    arrays = dict(time=times, oracle_correlation=oracle, omega=columns,
                  polynomial_factor=factor, exact_filter_factor=exact, h0=h0,
                  current0=current0, oracle_unitaries=unitaries,
                  complete_basis_factor=complete, exact_thermal_density=exact_density,
                  **{"input_"+name: value for name, value in inputs.items()})
    for dt, stride in ((.04, 2), (.02, 1)):
        simulation = Simulation(problem, Integrator(dt, "rk4"), Execution(chunk_size=7))
        filtered_run = simulation.run(initial, round(.8/dt))
        exact_run = simulation.run(exact_initial, round(.8/dt))
        observed = np.asarray(filtered_run.observables["current_correlation"][:, 0, 0])
        exact_observed = np.asarray(exact_run.observables["current_correlation"][:, 0, 0])
        trajectory[str(dt)] = dict(
            combined_correlation_max_error=float(np.max(abs(observed-oracle[::stride]))),
            dynamics_only_correlation_max_error=float(np.max(abs(exact_observed-oracle[::stride]))),
            preparation_only_same_integrator_max_error=float(np.max(abs(observed-exact_observed))),
            full_column_max_error=float(np.max(abs(np.asarray(filtered_run.final_state.electronic)
                                                   -unitaries[-1]@np.concatenate((factor, current0@factor), axis=1)))))
        arrays[f"correlation_dt_{dt}"] = observed
        arrays[f"exact_factor_correlation_dt_{dt}"] = exact_observed
        arrays[f"final_electronic_dt_{dt}"] = filtered_run.final_state.electronic
    simulation = Simulation(problem, Integrator(.02, "rk4"), Execution(chunk_size=7))
    first = simulation.run(initial, 13)
    simulation.save_checkpoint(output/"checkpoint.h5", first.final_state)
    resumed = simulation.run(simulation.load_checkpoint(output/"checkpoint.h5"), 27)
    full = simulation.run(initial, 40)
    restart_error = max(float(np.max(abs(np.asarray(a)-np.asarray(b)), initial=0))
                        for a, b in zip(jax.tree.leaves(resumed.final_state),
                                        jax.tree.leaves(full.final_state), strict=True))
    for index, (a, b) in enumerate(zip(jax.tree.leaves(resumed.final_state),
                                      jax.tree.leaves(full.final_state), strict=True)):
        arrays[f"resumed_leaf_{index}"] = a
        arrays[f"uninterrupted_leaf_{index}"] = b

    # Independent trace replicas at one fixed nuclear geometry; each rank is
    # a count of random columns and can exceed the small reference dimension.
    observable = np.diag(np.linspace(-1., 1., 7))
    target = float(np.trace(exact_density@observable).real)
    ranks = {}
    for rank in (1, 4, 16, 64):
        kernel = jax.jit(lambda omega: prepare_thermal_columns(
            lambda v: problem.model.apply(problem.params, q, v), omega, plan))
        estimates = []
        for seed in range(seeds):
            prepared = kernel(thermal_random_columns(7, np.arange(rank), seed=seed, trajectory_id=5))
            value = np.asarray(require_success(prepared))
            estimates.append(float(np.vdot(value, observable@value).real))
        estimates = np.asarray(estimates)
        arrays[f"rank_{rank}_estimates"] = estimates
        ranks[str(rank)] = dict(mean=float(estimates.mean()), empirical_bias=float(estimates.mean()-target),
            empirical_rmse=float(np.sqrt(np.mean((estimates-target)**2))),
            across_seed_mean_sem=float(estimates.std(ddof=1)/np.sqrt(seeds)))
    np.savez_compressed(output/"accuracy_arrays.npz", **arrays)
    report = dict(preparation=preparation, normalized_factor_error=factor_error,
                  density_trace_norm_error=density_error, complete_basis_thermal_density_error=complete_density_error,
                  trajectory_errors=trajectory, restart_max_error=restart_error,
                  rank_reference=target, independent_trace_seeds=list(range(seeds)), ranks=ranks,
                  uncertainty="trace-seed variation at one fixed nuclear geometry; no nuclear convergence claim")
    assert factor_error < 2e-11 and density_error < 4e-11
    assert complete_density_error < 2e-11 and restart_error < 3e-13
    assert trajectory["0.02"]["combined_correlation_max_error"] < 1e-7
    return report


def timings(*, sizes=(32, 128, 512), ranks=(4, 16), repeats=7):
    """Matched finite-column estimator; dense eigensolve applies F to identical Omega."""
    rows = []
    for n in sizes:
        problem, q, _, inputs = recipe.ring_problem(n)
        h, _ = independent_matrices(inputs, 0.)
        for rank in ranks:
            for beta in (1., 8.):
                started = time.perf_counter()
                plan = ThermalFilterPlan(beta, *recipe.gershgorin_bounds(inputs),
                    action_id=f"benchmark-ring-N{n}", bounds_id="analytic-row-sums")
                plan_seconds = time.perf_counter()-started
                started = time.perf_counter()
                omega = thermal_random_columns(n, np.arange(rank), seed=93, trajectory_id=7)
                omega.block_until_ready()
                rng_seconds = time.perf_counter()-started
                kernel = jax.jit(lambda parameters, coordinate, columns: prepare_thermal_columns(
                    lambda v: problem.model.apply(parameters, coordinate, v), columns, plan))
                started = time.perf_counter()
                compiled = kernel.lower(problem.params, q, omega).compile()
                compile_seconds = time.perf_counter()-started
                started = time.perf_counter()
                candidate = compiled(problem.params, q, omega)
                jax.block_until_ready(candidate)
                first_seconds = time.perf_counter()-started
                require_success(candidate)
                native_times, dense_times, errors = [], [], []
                host_omega = np.asarray(omega)

                def dense_prepare():
                    energies, vectors = eigh(h, check_finite=False)
                    y = vectors@(np.exp(-beta*(energies-plan.lower)/2)[:, None]*(vectors.conj().T@host_omega))
                    return y/np.linalg.norm(y)

                dense_prepare()  # warm the dense numerical libraries too
                for repeat in range(repeats):
                    for method in (("native", "dense") if repeat % 2 == 0 else ("dense", "native")):
                        started = time.perf_counter()
                        if method == "native":
                            candidate = compiled(problem.params, q, omega)
                            jax.block_until_ready(candidate)
                            require_success(candidate)
                            native_times.append(time.perf_counter()-started)
                        else:
                            exact = dense_prepare()
                            dense_times.append(time.perf_counter()-started)
                    errors.append(float(np.linalg.norm(np.asarray(require_success(candidate))-exact)))
                assert max(errors) < 2e-9
                rows.append(dict(nstates=n, columns=rank, beta=beta, degree=plan.degree,
                    plan_seconds=plan_seconds, random_columns_seconds=rng_seconds,
                    compilation_seconds=compile_seconds, first_call_seconds=first_seconds,
                    native_seconds=native_times, dense_eigh_filter_seconds=dense_times,
                    native_median_seconds=float(np.median(native_times)),
                    dense_median_seconds=float(np.median(dense_times)),
                    maximum_matched_factor_error=max(errors), diagnostics=candidate.diagnostics()))
    return dict(cases=rows, scope="CPU preparation only; same H,beta,Omega,rank and pooled ratio",
        excluded="dense H assembly, input transfers, current insertion/origin metadata; reported separately from propagation",
        host_acceptance="warm native timings include require_success; Hamiltonian parameters and q are dynamic arguments",
        order="both methods warmed; paired execution order alternates between repetitions",
        caveat="finite-column thermal ratio remains biased; speed is not a convergence claim")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timing", action="store_true")
    parser.add_argument("--seeds", type=int, default=96)
    args = parser.parse_args()
    if not jax.config.x64_enabled or args.seeds < 2:
        parser.error("requires JAX_ENABLE_X64=1 and at least 2 independent trace seeds")
    root = Path(pyeph.__file__).resolve().parent
    def files():
        return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(root.rglob("*.py"))}
    hashes = files()
    driver_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    recipe_hash = hashlib.sha256(RECIPE_PATH.read_bytes()).hexdigest()
    args.output.mkdir(parents=True, exist_ok=False)
    report = dict(source_sha256=hashes, driver_sha256=driver_hash, recipe_sha256=recipe_hash,
        versions=dict(python=platform.python_version(), jax=jax.__version__, numpy=np.__version__, scipy=scipy.__version__),
        backend=jax.default_backend(), jax_enable_x64=jax.config.x64_enabled,
        hardware=dict(machine=platform.machine(), processor=platform.processor(), devices=list(map(str, jax.devices()))),
        execution_configuration={name: getattr(jax.config, name, None) for name in (
            "jax_default_prng_impl", "jax_threefry_partitionable", "jax_random_seed_offset",
            "jax_default_matmul_precision")},
        thread_environment={name: os.environ.get(name) for name in (
            "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")},
        correctness=correctness(args.output, seeds=args.seeds))
    if args.timing:
        if jax.default_backend() != "cpu":
            raise ValueError("this matched timing campaign is qualified for CPU only")
        report["timings"] = timings()
    assert hashes == files(), "runtime source changed during campaign"
    assert driver_hash == hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    assert recipe_hash == hashlib.sha256(RECIPE_PATH.read_bytes()).hexdigest()
    report["data_sha256"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in sorted(args.output.iterdir()) if p.is_file()}
    (args.output/"report.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps(report["correctness"], indent=2), flush=True)


if __name__ == "__main__":
    main()
