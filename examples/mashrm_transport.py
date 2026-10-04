"""Canonical RM velocity correlations with a preserved checkpoint origin.

Run: python examples/mashrm_transport.py
The small analytic model has nonzero electron--phonon coupling. This example
checks preparation, streamed products/statistics, and restart composition. It
does not establish equilibrium stationarity, material mobility, or timestep
convergence. Existing JSON/NPZ evidence is never overwritten.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import tempfile
import time

import jax
import numpy as np
import scipy

import pyeph
from pyeph import Execution, Integrator, MASHRM
from pyeph.models.epc import LinearEPCModel
from pyeph.observables.transport.mashrm import FixedPositionVelocity, RMVelocity
from pyeph.workflows.mashrm_equilibrium import LinearEPCCanonical
from pyeph.workflows.mashrm_transport import RMTransport


def source_fingerprint():
    root = Path(__file__).resolve().parents[1]
    if Path(pyeph.__file__).resolve().parent != root/"src/pyeph":
        raise RuntimeError("run this source-audited example against the repository's src/pyeph")
    files = sorted((root/"src/pyeph").rglob("*.py"))+[Path(__file__).resolve()]
    hashes = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
              for path in files}
    digest = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    return {"sha256": digest, "files": hashes}


def state_difference(actual, expected, *, exact=False):
    maximum = 0.
    for a, b in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
        a, b = np.asarray(a), np.asarray(b)
        if exact or a.dtype.kind in "biu":
            np.testing.assert_array_equal(a, b)
        else:
            np.testing.assert_allclose(a, b, atol=2e-12, rtol=2e-12)
            maximum = max(maximum, float(np.max(abs(a-b))))
    return maximum


def run_example(*, trajectories=32, seed=12):
    # Explicit numerical/RNG settings make the stored seed and IDs meaningful.
    for name, value in {"jax_enable_x64": True, "jax_default_prng_impl": "threefry2x32",
                        "jax_threefry_partitionable": True, "jax_random_seed_offset": 0,
                        "jax_high_dynamic_range_gumbel": False}.items():
        jax.config.update(name, value)
    started = datetime.now(timezone.utc).isoformat()
    source_start = source_fingerprint()
    clock = time.perf_counter()

    model = LinearEPCModel(3, 2)
    params = model.create_params(
        [[-.35, .16, .02], [.16, .15, .08], [.02, .08, .65]],
        [[[.2, .12, 0.], [.12, -.1, .04], [0., .04, .08]],
         [[.04, .02, .03], [.02, .12, -.09], [.03, -.09, -.18]]],
        omega=[.9, 1.2], q_eq=[.3, -.2])
    masses, beta = np.array([1., 2.]), 1.7
    sampler = LinearEPCCanonical(model, params, masses, beta)
    positions = {"x": np.array([0., 1., 2.]), "y": np.array([.4, -.2, .8])}
    measurement = RMVelocity(tuple(positions), FixedPositionVelocity(positions))
    # Finite fixed-position commutators v=i[h,X] contain no carrier charge,
    # periodic-image convention, or moving-center convective contribution.
    dt, steps, split_steps, save_every = .04, 50, 25, 5
    method = MASHRM(event_substeps=2)
    workflow = RMTransport(sampler, Integrator(dt, "exponential_midpoint"), measurement,
                           method=method, execution=Execution(chunk_size=25, save_every=save_every))
    ids = np.arange(trajectories, dtype=np.uint32)
    initial = workflow.prepare(ids, seed=seed)
    print(f"Prepared {trajectories} canonical trajectories; original velocities retained.", flush=True)

    # A production observer can write each bounded block directly to disk. This
    # small example archives the selected blocks to make its reductions auditable.
    streamed = {name: [] for name in (
        "times", "velocity", "velocity_correlation", "energy", "mapping_norm",
        "events", "accepted", "frustrated")}

    def observe(lag_times, values):
        streamed["times"].append(lag_times.copy())
        for name in streamed:
            if name != "times":
                streamed[name].append(np.asarray(values[name]).copy())

    full = workflow.run(initial, steps, observer=observe)
    saved = {name: np.concatenate(blocks, axis=0) for name, blocks in streamed.items()}
    products = saved["velocity_correlation"]  # (time, trajectory, initial_probe, current_probe)
    v0 = np.asarray(initial.origin.velocity0)
    expected = v0[None, :, :, None]*saved["velocity"][:, :, None, :]
    np.testing.assert_array_equal(products, expected)
    mean = products.mean(axis=1)
    m2 = np.sum((products-mean[:, None])**2, axis=1)
    sem = np.sqrt(m2/(trajectories-1)/trajectories)
    for name, expected in (("times", saved["times"]), ("mean", mean), ("m2", m2),
                           ("standard_error", sem)):
        np.testing.assert_allclose(getattr(full.statistics, name), expected, atol=2e-13, rtol=2e-12)

    prefix = workflow.run(initial, split_steps)
    with tempfile.TemporaryDirectory(prefix="pyeph-rm-transport-") as directory:
        checkpoint = Path(directory)/"origin_and_state.h5"
        workflow.save_checkpoint(checkpoint, prefix.final_state)  # Atomic workflow checkpoint.
        restored = workflow.load_checkpoint(checkpoint)
        state_difference(restored.state, prefix.final_state.state, exact=True)
        np.testing.assert_array_equal(restored.origin.velocity0, initial.origin.velocity0)
        np.testing.assert_array_equal(restored.origin.trajectory_ids, ids)
        assert restored.origin.time0 == initial.origin.time0 == 0.
        assert restored.origin.preparation_metadata == initial.origin.preparation_metadata
        checkpoint_bytes = checkpoint.stat().st_size
        suffix = workflow.run(restored, steps-split_steps)

    # Step25 is on the save schedule, so the only duplicate is the shared t=1
    # boundary. Keep the prefix boundary and drop exactly the first suffix row.
    assert prefix.statistics.times[-1] == suffix.statistics.times[0] == dt*split_steps
    restart_deltas, joined = {}, {}
    for name in ("times", "mean", "m2", "standard_error"):
        combined = np.concatenate((getattr(prefix.statistics, name),
                                   getattr(suffix.statistics, name)[1:]), axis=0)
        target = np.asarray(getattr(full.statistics, name))
        np.testing.assert_allclose(combined, target, atol=2e-13, rtol=2e-12)
        restart_deltas[name] = float(np.max(abs(combined-target)))
        joined[name] = combined
    restart_deltas["final_state"] = state_difference(suffix.final_state.state, full.final_state.state)
    assert suffix.final_state.origin.preparation_id == initial.origin.preparation_id
    np.testing.assert_array_equal(suffix.final_state.origin.velocity0, v0)
    diagnostics = {name: np.asarray(value) for name, value in full.final_state.state.method_state.items()}
    np.testing.assert_array_equal(diagnostics["status"], 0)
    np.testing.assert_array_equal(full.final_state.state.trajectory_id, ids)

    arrays = dict(trajectory_ids=ids, masses=masses, beta=np.array(beta),
                  initial_q=np.asarray(initial.state.q), initial_p=np.asarray(initial.state.p),
                  initial_mapping=np.asarray(initial.state.electronic),
                  initial_active=np.asarray(initial.state.method_state["active"]),
                  initial_key=np.asarray(initial.state.key), velocity0=v0,
                  times=saved["times"], correlation_mean=mean, correlation_m2=m2,
                  correlation_sem=sem, products=products, velocity=saved["velocity"],
                  energy=saved["energy"], mapping_norm=saved["mapping_norm"],
                  events=saved["events"], accepted=saved["accepted"], frustrated=saved["frustrated"],
                  restarted_times=joined["times"], restarted_mean=joined["mean"],
                  restarted_m2=joined["m2"], restarted_sem=joined["standard_error"])
    arrays.update({f"parameter_{name}": np.asarray(value) for name, value in sampler.params.items()})
    arrays.update({f"position_{name}": value for name, value in positions.items()})
    final = full.final_state.state
    arrays.update({f"final_{name}": np.asarray(getattr(final, name))
                   for name in ("q", "p", "electronic", "time", "step", "trajectory_id", "key")})
    arrays.update({f"final_method_{name}": value for name, value in diagnostics.items()})
    source_end = source_fingerprint()
    if source_end != source_start:
        raise RuntimeError("runtime/example source changed during the demonstration; evidence not saved")
    summary = dict(
        scope="analytic coupled canonical RM transport composition; no stationarity or mobility claim",
        started_utc=started, elapsed_seconds_including_compilation=time.perf_counter()-clock,
        timing_scope="single illustrative execution on a shared host, not a performance benchmark",
        trajectories=trajectories, seed=seed, beta=beta, dt=dt, steps=steps,
        duration=dt*steps, save_every=save_every, event_substeps=method.event_substeps,
        checkpoint_step=split_steps, checkpoint_time=dt*split_steps,
        checkpoint_atomic=True, checkpoint_restore_exact=True, checkpoint_bytes=checkpoint_bytes,
        checkpoint_retention="temporary HDF5 removed after verified restart",
        duplicate_policy="retain prefix endpoint; drop suffix row0 at shared aligned boundary",
        probes=list(positions), correlation_order="initial_i_times_current_j",
        standard_error="sample standard deviation of per-trajectory products divided by sqrt(B)",
        units=dict(time="atomic time", energy="hartree", position="bohr", velocity="bohr/atomic time"),
        preparation=initial.origin.preparation_metadata, workflow_fingerprint=workflow.fingerprint,
        accepted_events=int(diagnostics["accepted"].sum()),
        frustrated_events=int(diagnostics["frustrated"].sum()),
        trajectories_with_events=int(np.count_nonzero(diagnostics["events"])),
        max_event_residual=float(diagnostics["max_event_residual"].max()),
        max_event_bracket_width=float(diagnostics["max_event_bracket_width"].max()),
        max_impulse_energy_error=float(diagnostics["max_impulse_energy_error"].max()),
        max_saved_energy_drift=float(abs(saved["energy"]-saved["energy"][0]).max()),
        max_saved_mapping_norm_error=float(abs(saved["mapping_norm"]-1).max()),
        status_codes=np.unique(diagnostics["status"]).tolist(), restart_max_abs_deltas=restart_deltas,
        initial_correlation=mean[0].tolist(), final_correlation=mean[-1].tolist(),
        final_standard_error=sem[-1].tolist(),
        software=dict(python=platform.python_version(), jax=jax.__version__, numpy=np.__version__,
                      scipy=scipy.__version__, backend=jax.default_backend(),
                      devices=[str(device) for device in jax.devices()], platform=platform.platform()),
        source_unchanged=True, source_start=source_start, source_end=source_end,
    )
    return summary, arrays


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectories", type=int, default=32)
    parser.add_argument("--seed", type=int, default=12)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/mashrm_transport"),
                        help="JSON/NPZ output directory (default: %(default)s)")
    args = parser.parse_args()
    if not 2 <= args.trajectories < 2**32 or not 0 <= args.seed < 2**32:
        parser.error("trajectories must be at least2 and seed a uint32 integer")
    paths = [args.output_dir/f"mashrm_transport.{suffix}" for suffix in ("json", "npz")]
    if any(path.exists() for path in paths):
        parser.error("output evidence already exists; select another --output-dir")
    summary, arrays = run_example(trajectories=args.trajectories, seed=args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with paths[1].open("xb") as output:
        np.savez_compressed(output, **arrays)
    summary["npz_sha256"] = hashlib.sha256(paths[1].read_bytes()).hexdigest()
    with paths[0].open("x") as output:
        output.write(json.dumps(summary, indent=2, allow_nan=False)+"\n")
    print(f"Accepted/frustrated events: {summary['accepted_events']}/{summary['frustrated_events']}")
    print(f"Largest restart difference: {max(summary['restart_max_abs_deltas'].values()):.3g}")
    print(f"Saved {paths[0]} and {paths[1]}")


if __name__ == "__main__":
    main()
