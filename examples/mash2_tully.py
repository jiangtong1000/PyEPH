"""Small, reproducible MASH2 Tully-1 convergence/energy benchmark.

Run from the repository: python examples/mash2_tully.py
This monokinetic classical ensemble is not a published scattering benchmark.
"""

import argparse
import json
from pathlib import Path
from time import perf_counter

import jax
import numpy as np

from pyeph.core.problem import CoupledClassical, Problem
from pyeph.core.state import stack_states
from pyeph.dynamics.mash2 import MASH2, MASHPopulation, sample_adiabatic_population
from pyeph.execution.runner import Execution
from pyeph.integrators.electronic import Integrator
from pyeph.models.analytic import TullyModel
from pyeph.simulation import Simulation


def benchmark(*, trajectories=64, duration=1000., dts=(1., 0.5), seed=2026):
    """Use identical initial spins at each timestep; return data and a summary."""
    jax.config.update("jax_enable_x64", True)
    model = TullyModel(1)
    initial = stack_states([
        sample_adiabatic_population(model, None, [-4.], [20.], active=0,
                                    seed=seed, trajectory_id=i)
        for i in range(trajectories)
    ])
    method = MASH2(event_substeps=2)
    problem = Problem(model, None, CoupledClassical(2000.), method,
                      MASHPopulation(include_nuclei=True))
    summary = dict(model="Tully1", preparation="monokinetic classical nuclei, lower population",
                   trajectories=trajectories, q_initial=-4., p_initial=20., mass=2000.,
                   duration=duration, seed=seed, event_substeps=method.event_substeps,
                   event_tolerance=method.event_tolerance,
                   scheme=method.numerical_scheme, jax_version=jax.__version__,
                   backend=jax.default_backend(), devices=[str(x) for x in jax.devices()], runs=[])
    data, previous = {}, None
    for index, dt in enumerate(dts):
        steps = round(duration/dt)
        if not np.isclose(steps*dt, duration, rtol=0, atol=1e-10):
            raise ValueError("duration must be an integer multiple of each dt")
        runner = Simulation(problem, Integrator(dt, "exponential_midpoint"),
                            Execution(chunk_size=128, save_every=max(1, round(10/dt))))
        start = perf_counter()
        result = runner.run(initial, steps)
        compile_and_run = perf_counter()-start
        start = perf_counter()
        repeated = runner.run(initial, steps, collect=False)
        jax.block_until_ready(repeated.final_state.electronic)
        cached_run = perf_counter()-start
        population = np.asarray(result.observables["population"])
        energy = np.asarray(result.observables["energy"])
        final_q = np.asarray(result.final_state.q)[:, 0]
        diagnostics = jax.device_get(result.final_state.method_state)
        final_population = population[-1].mean(axis=0)
        run = dict(dt=dt, steps=steps, compile_and_run_seconds=compile_and_run,
                   cached_run_seconds=cached_run,
                   final_adiabatic_population=final_population.tolist(),
                   population_standard_error=np.sqrt(final_population*(1-final_population)
                                                      /trajectories).tolist(),
                   fraction_right_of_origin=float(np.mean(final_q > 0)),
                   minimum_final_q=float(final_q.min()), maximum_final_q=float(final_q.max()),
                   max_absolute_energy_drift=float(np.max(np.abs(energy-energy[0]))),
                   max_mapping_norm_error=float(np.max(np.abs(
                       result.observables["mapping_norm"]-1))),
                   total_accepted=int(np.sum(diagnostics["accepted"])),
                   total_frustrated=int(np.sum(diagnostics["frustrated"])),
                   total_localization_iterations=int(np.sum(diagnostics["localization_iterations"])),
                   max_event_residual=float(np.max(diagnostics["max_event_residual"])),
                   max_impulse_energy_error=float(np.max(diagnostics["max_impulse_energy_error"])),
                   status_codes=np.unique(diagnostics["status"]).tolist())
        if previous is not None:
            run["max_final_q_change_from_previous_dt"] = float(np.max(np.abs(final_q-previous)))
        previous = final_q
        summary["runs"].append(run)
        for name in ("population", "energy", "mapping_norm", "q", "p", "events"):
            data[f"run{index}_{name}"] = np.asarray(result.observables[name])
        data[f"run{index}_time"] = result.times
    return summary, data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectories", type=int, default=64)
    parser.add_argument("--duration", type=float, default=1000.)
    parser.add_argument("--dts", type=float, nargs="+", default=[1., 0.5])
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", type=Path, default=Path("outputs/mash2_tully"),
                        help="JSON/NPZ output prefix (default: %(default)s)")
    args = parser.parse_args()
    if args.trajectories < 1 or args.duration <= 0 or any(dt <= 0 for dt in args.dts):
        parser.error("trajectory count, duration, and timesteps must be positive")
    summary, data = benchmark(trajectories=args.trajectories, duration=args.duration,
                              dts=tuple(args.dts), seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix(".json").write_text(json.dumps(summary, indent=2)+"\n")
    np.savez_compressed(args.output.with_suffix(".npz"), **data)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
