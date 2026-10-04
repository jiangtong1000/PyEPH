"""Real three-state RM ensemble on an analytic two-coordinate EPC model.

This checks an executable preparation/measurement workflow, not a material,
thermal-equilibrium, transport, or quantum benchmark. Run from the repository:
python examples/mashrm_epc.py
"""

import argparse
import json
from pathlib import Path

import jax
import numpy as np

from pyeph import CoupledClassical, Execution, Integrator, Problem, Simulation, stack_states
from pyeph.dynamics.mashrm import MASHRM, MASHRMPopulation, sample_population
from pyeph.models.epc import LinearEPCModel


def run_example(*, trajectories=32, duration=4., dts=(.04, .02), seed=2026):
    jax.config.update("jax_enable_x64", True)
    model = LinearEPCModel(3, 2)
    params = model.create_params(np.diag([-.4, 0., .6]), np.array([
        [[0., .6, .2], [.6, 0., .3], [.2, .3, 0.]],
        [[0., .1, .4], [.1, 0., -.5], [.4, -.5, 0.]],
    ]), omega=[.1, .15])
    initial = stack_states([sample_population(model, params, [0., 0.], [1.4, .6],
        population=0, basis="adiabatic", trajectory_id=i, seed=seed) for i in range(trajectories)])
    method = MASHRM(event_substeps=2)
    problem = Problem(model, params, CoupledClassical([1., 2.]), method,
                      MASHRMPopulation(basis="adiabatic", include_nuclei=True))
    summary = dict(model="analytic real three-state two-coordinate linear EPC",
        preparation="lower adiabatic conditional sphere; fixed supplied nuclear q,p",
        measurement="one-time adiabatic RM population", trajectories=trajectories,
        duration=duration, seed=seed, event_substeps=method.event_substeps,
        scheme=method.numerical_scheme, jax_version=jax.__version__,
        backend=jax.default_backend(), runs=[])
    arrays = {"initial_q": np.asarray(initial.q), "initial_p": np.asarray(initial.p),
              "initial_c": np.asarray(initial.electronic), "trajectory_id": np.asarray(initial.trajectory_id)}
    arrays.update({f"params_{name}": np.asarray(value) for name, value in params.items()})
    previous = None
    for index, dt in enumerate(dts):
        steps = round(duration/dt)
        if not np.isclose(steps*dt, duration, atol=1e-12, rtol=0.):
            raise ValueError("duration must be an integer multiple of every dt")
        simulation = Simulation(problem, Integrator(dt, "exponential_midpoint"),
                                Execution(chunk_size=50, save_every=max(1, round(.2/dt))))
        result = simulation.run(initial, steps)
        population = np.asarray(result.observables["population"])
        energy = np.asarray(result.observables["energy"])
        diagnostics = jax.device_get(result.final_state.method_state)
        final = population[-1]
        entry = dict(dt=dt, steps=steps, final_population=final.mean(axis=0).tolist(),
            # RM samples are not Bernoulli indicators; use their sample variance.
            population_standard_error=(final.std(axis=0, ddof=1)/np.sqrt(trajectories)).tolist(),
            max_energy_drift=float(abs(energy-energy[0]).max()),
            max_mapping_norm_error=float(abs(result.observables["mapping_norm"]-1.).max()),
            accepted=int(np.sum(diagnostics["accepted"])),
            frustrated=int(np.sum(diagnostics["frustrated"])),
            max_event_residual=float(np.max(diagnostics["max_event_residual"])),
            max_event_bracket_width=float(np.max(diagnostics["max_event_bracket_width"])),
            max_impulse_energy_error=float(np.max(diagnostics["max_impulse_energy_error"])),
            status_codes=np.unique(diagnostics["status"]).tolist())
        if previous is not None:
            differences = final-previous
            entry.update(paired_population_change=differences.mean(axis=0).tolist(),
                paired_change_standard_error=(differences.std(axis=0, ddof=1)/np.sqrt(trajectories)).tolist())
        previous = final
        summary["runs"].append(entry)
        for name in ("population", "energy", "mapping_norm", "active", "q", "p", "events"):
            arrays[f"run{index}_{name}"] = np.asarray(result.observables[name])
        arrays[f"run{index}_time"] = result.times
        arrays[f"run{index}_final_c"] = np.asarray(result.final_state.electronic)
    return summary, arrays


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectories", type=int, default=32)
    parser.add_argument("--duration", type=float, default=4.)
    parser.add_argument("--dts", type=float, nargs="+", default=[.04, .02])
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", type=Path, default=Path("outputs/mashrm_epc"),
                        help="JSON/NPZ output prefix (default: %(default)s)")
    args = parser.parse_args()
    if (args.trajectories < 2 or not np.isfinite(args.duration) or args.duration <= 0
            or any(not np.isfinite(dt) or dt <= 0 for dt in args.dts)):
        parser.error("at least two trajectories and finite positive times are required")
    summary, arrays = run_example(trajectories=args.trajectories, duration=args.duration,
                                 dts=tuple(args.dts), seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix(".json").write_text(json.dumps(summary, indent=2)+"\n")
    np.savez_compressed(args.output.with_suffix(".npz"), **arrays)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
