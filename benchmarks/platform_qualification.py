"""Matched complete native workflows on explicitly selected CPU/GPU devices.

Both material families use illustrative effective models. Timings qualify the
software workload and numerical accuracy, not material mobility or a fitted
potential. Run CPU/GPU cases in separate otherwise idle processes on the same
host when interpreting hardware speed ratios.
"""

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import resource
import statistics
import sys
import time

import jax
import numpy as np

import pyeph
from pyeph import Execution, Integrator, Simulation, make_state, stack_states
from pyeph.io.checkpoint import array_fingerprint

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"examples"))
import oriented_fragments  # noqa: E402
import perovskite  # noqa: E402


QUALIFICATION_TOLERANCES = {
    "batch_atol": 3e-12, "batch_rtol": 3e-12, "time_atol": 1e-12,
    "reference_atol": 1e-5, "reference_momentum_atol": 1e-4,
    "refinement_ratio": .45, "refinement_floor": 3e-9,
}


def finite_array(value, label):
    """Qualification evidence must be numerical and finite, including observables."""
    array = np.asarray(value)
    if (array.dtype.kind not in "biufc" or not np.isfinite(array).all()):
        raise RuntimeError(f"nonfinite or nonnumerical qualification evidence: {label}")
    return array


def compare_arrays(actual, expected, label, *, atol, rtol=0.):
    """Reject shape broadcasting and NaNs before applying declared tolerances."""
    actual, expected = finite_array(actual, label), finite_array(expected, f"expected {label}")
    if actual.shape != expected.shape:
        raise RuntimeError(f"qualification shape mismatch: {label}: {actual.shape} != {expected.shape}")
    difference = float(np.max(np.abs(actual.astype(np.result_type(actual, expected, 1.))
                                      - expected))) if actual.size else 0.
    matches = (np.array_equal(actual, expected) if atol == 0 and rtol == 0 else
               np.allclose(actual, expected, atol=atol, rtol=rtol, equal_nan=False))
    if not np.isfinite(difference) or not matches:
        raise RuntimeError(f"qualification mismatch: {label}; maximum absolute error {difference}")
    return difference


def compare_replicate(actual, expected, label, *, exact=False, time=False):
    """Separate numerical agreement from bytes; discrete state is always exact."""
    actual, expected = finite_array(actual, label), finite_array(expected, f"expected {label}")
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        raise RuntimeError(f"qualification shape/dtype mismatch: {label}")
    discrete = actual.dtype.kind in "biu"
    tol = QUALIFICATION_TOLERANCES
    atol = 0. if exact or discrete else tol["time_atol"] if time else tol["batch_atol"]
    rtol = 0. if exact or discrete or time else tol["batch_rtol"]
    error = compare_arrays(actual, expected, label, atol=atol, rtol=rtol)
    bytewise = actual.tobytes() == expected.tobytes()
    if exact and not bytewise:
        raise RuntimeError(f"qualification bytewise mismatch: {label}")
    return {"max_absolute_error": error, "bytewise_equal": bytewise,
            "discrete": discrete, "atol": atol, "rtol": rtol}


def validate_state_replicate(actual, expected, label, *, exact=False):
    """Check every state leaf, including IDs, keys and method-specific state."""
    actual_leaves, actual_tree = jax.tree_util.tree_flatten_with_path(actual)
    expected_leaves, expected_tree = jax.tree_util.tree_flatten_with_path(expected)
    if actual_tree != expected_tree:
        raise RuntimeError(f"qualification state-tree mismatch: {label}")
    leaves = {}
    for (path, value), (_, reference) in zip(actual_leaves, expected_leaves, strict=True):
        name = jax.tree_util.keystr(path)
        leaves[name] = compare_replicate(value, reference, f"{label} {name}", exact=exact,
                                         time=name == ".time")
    return {"numerically_qualified": True,
            "bytewise_equal": all(value["bytewise_equal"] for value in leaves.values()),
            "leaves": leaves}


def validate_continuation(continued, uninterrupted, suffix_start):
    """Compare all resumed samples against the corresponding uninterrupted suffix."""
    if set(continued.observables) != set(uninterrupted.observables):
        raise RuntimeError("continuation observable fields disagree")
    samples = {"times": compare_replicate(continued.times, uninterrupted.times[suffix_start:],
                                           "continuation sample times", time=True)}
    for name, reference in uninterrupted.observables.items():
        samples[name] = compare_replicate(continued.observables[name], reference[suffix_start:],
                                           f"continuation observable {name}")
    final = validate_state_replicate(continued.final_state, uninterrupted.final_state,
                                      "continuation final state")
    return {"numerically_qualified": True, "final_state": final, "samples": samples,
            "bytewise_equal": final["bytewise_equal"] and
            all(value["bytewise_equal"] for value in samples.values())}


def validate_scalar_reference(coarse, fine, reference):
    """Qualify one complete scalar trace before using it as the batching oracle."""
    tol = QUALIFICATION_TOLERANCES
    times = finite_array(coarse.times, "scalar coarse times")
    if times.ndim != 1 or len(times) < 2 or np.any(np.diff(times) <= 0):
        raise RuntimeError("scalar qualification times must be a strictly increasing vector")
    compare_arrays(fine.times, times, "fine/coarse sample times", atol=tol["time_atol"])
    required = {"q", "p", "electronic", "current"}
    if not required <= set(reference):
        raise RuntimeError("independent reference must cover q, p, electronic and current")
    if set(coarse.observables) != set(fine.observables) or not set(reference) <= set(coarse.observables):
        raise RuntimeError("scalar/reference observable fields disagree")
    for name in coarse.observables:
        for label, result in (("coarse", coarse), ("fine", fine)):
            value = finite_array(result.observables[name], f"scalar {label} {name}")
            if not value.ndim or value.shape[0] != len(times):
                raise RuntimeError(f"scalar {label} {name} does not match the output time grid")
    errors = {}
    for name, value in reference.items():
        expected = finite_array(value, f"independent reference {name}")
        values = []
        for label, result in (("coarse", coarse), ("fine", fine)):
            actual = np.asarray(result.observables[name])
            if actual.shape != expected.shape:
                raise RuntimeError(f"scalar {label}/reference {name} shapes disagree")
            values.append(float(np.max(abs(actual-expected))))
        coarse_error, fine_error = values
        bound = tol["reference_momentum_atol"] if name == "p" else tol["reference_atol"]
        if (not np.isfinite(values).all() or coarse_error > bound
                or fine_error > tol["refinement_ratio"]*coarse_error+tol["refinement_floor"]):
            raise RuntimeError(f"workflow fails independent accuracy/refinement gate: {name} {values}")
        errors[name] = {"coarse": coarse_error, "fine": fine_error}
    return errors


def validate_final_state(actual, scalar, batch):
    """Compare physical final states; trajectory IDs/random keys intentionally differ."""
    tol, errors = QUALIFICATION_TOLERANCES, {}
    for name in ("q", "p", "electronic", "time", "step"):
        expected = finite_array(getattr(scalar, name), f"scalar final {name}")
        if batch != 1:
            expected = np.broadcast_to(expected, (batch, *expected.shape))
        atol = 0. if name == "step" else tol["time_atol"] if name == "time" else tol["batch_atol"]
        rtol = 0. if name in ("step", "time") else tol["batch_rtol"]
        errors[name] = compare_arrays(getattr(actual, name), expected, f"batched final {name}",
                                      atol=atol, rtol=rtol)
    if jax.tree.structure(actual.method_state) != jax.tree.structure(scalar.method_state):
        raise RuntimeError("batched final method-state structure disagrees")
    for index, (actual_leaf, scalar_leaf) in enumerate(zip(jax.tree.leaves(actual.method_state),
                                                        jax.tree.leaves(scalar.method_state), strict=True)):
        expected = finite_array(scalar_leaf, f"scalar final method state {index}")
        if batch != 1:
            expected = np.broadcast_to(expected, (batch, *expected.shape))
        errors[f"method_state_{index}"] = compare_arrays(
            actual_leaf, expected, f"batched final method state {index}", atol=tol["batch_atol"],
            rtol=tol["batch_rtol"])
    # Validate the remaining state leaves too. Exact IDs/keys are compared
    # across output policies/restart, not against a different trajectory ID.
    for leaf in jax.tree.leaves(actual):
        finite_array(leaf, "batched final state")
    return errors


def validate_batched_result(result, scalar, batch):
    """Every sampled observable and every lane must agree with the scalar oracle."""
    tol = QUALIFICATION_TOLERANCES
    expected_times = finite_array(scalar.times, "scalar sample times")
    if batch != 1:
        expected_times = np.broadcast_to(expected_times[:, None], (len(expected_times), batch))
    time_error = compare_arrays(result.times, expected_times, "batched sample times",
                                 atol=tol["time_atol"])
    if set(result.observables) != set(scalar.observables):
        raise RuntimeError("batched and scalar observable fields disagree")
    errors = {}
    for name, scalar_value in scalar.observables.items():
        expected = finite_array(scalar_value, f"scalar {name}")
        if batch != 1:
            expected = np.broadcast_to(expected[:, None], (len(expected), batch, *expected.shape[1:]))
        errors[name] = compare_arrays(result.observables[name], expected, f"batched observable {name}",
                                      atol=tol["batch_atol"], rtol=tol["batch_rtol"])
    return {"sample_time_max_error": time_error, "observable_max_errors": errors,
            "final_state_max_errors": validate_final_state(result.final_state, scalar.final_state, batch)}


def hashes():
    package = Path(pyeph.__file__).resolve().parent
    result = {"pyeph/"+p.relative_to(package).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in package.rglob("*.py")}
    for path in (Path(__file__), Path(oriented_fragments.__file__), Path(perovskite.__file__)):
        result[path.relative_to(ROOT).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def timed(function, repeats, *, validate=None):
    durations = []
    for _ in range(repeats):
        start = time.perf_counter()
        result = function()
        jax.block_until_ready(result.final_state)
        jax.block_until_ready(result.observables)
        durations.append(time.perf_counter()-start)
        # Falsification checks are outside the measured public-run interval.
        # Validate every repetition, not only the final repeated result.
        if validate is not None:
            validate(result)
    return {"median_seconds": statistics.median(durations), "seconds": durations}, result


def run_case(args, batch, output):
    module = oriented_fragments if args.family == "molecular" else perovskite
    start = time.perf_counter()
    problem, initial = module.fixture(method=args.method)
    # A fixed-basis declaration is shared by both devices. Selecting an
    # arbitrary vector from a degenerate eigensolver block would otherwise
    # change the physical initial state under last-bit platform differences.
    index = np.arange(problem.model.spec.system.nstates, dtype=float)
    amplitudes = np.cos(.37*index)+1j*np.sin(.23*index+.1)
    amplitudes /= np.linalg.norm(amplitudes)
    initial = make_state(initial.q, initial.p, amplitudes, trajectory_id=19, seed=20261004)
    inputs = {"q": np.asarray(initial.q), "p": np.asarray(initial.p),
              "electronic": np.asarray(initial.electronic),
              "masses": np.asarray(problem.nuclear_treatment.masses)}
    inputs.update({f"parameter_{i}": np.asarray(value)
                   for i, value in enumerate(jax.tree.leaves(problem.params))})
    np.savez(output/f"batch{batch}_inputs.npz", **inputs)
    preparation = time.perf_counter()-start
    integrator = Integrator(args.dt, "rk4", electronic_substeps=4)
    execution = Execution(chunk_size=16, save_every=8)
    scalar = Simulation(problem, integrator, execution)
    coarse = scalar.run(initial, args.steps)
    fine = Simulation(problem, replace(integrator, dt=args.dt/2),
                      replace(execution, chunk_size=32, save_every=16)).run(initial, args.steps*2)
    reference, nfev = module.scipy_reference(problem, initial, np.asarray(coarse.times))
    errors = validate_scalar_reference(coarse, fine, reference)
    states = [make_state(initial.q, initial.p, initial.electronic,
                          trajectory_id=100+i, seed=20261004) for i in range(batch)]
    batched = states[0] if batch == 1 else stack_states(states)
    simulation = Simulation(problem, integrator, execution)
    def validate(result):
        return validate_batched_result(result, coarse, batch)

    def validate_empty(result):
        return validate_final_state(result.final_state, coarse.final_state, batch)
    first, _ = timed(lambda: simulation.run(batched, args.steps), 1, validate=validate)
    warm, result = timed(lambda: simulation.run(batched, args.steps), args.repeats, validate=validate)
    empty_first, _ = timed(lambda: simulation.run(batched, args.steps, collect=False), 1,
                           validate=validate_empty)
    empty, empty_result = timed(lambda: simulation.run(batched, args.steps, collect=False), args.repeats,
                                validate=validate_empty)
    batch_evidence = validate_batched_result(result, coarse, batch)
    output_evidence = validate_state_replicate(empty_result.final_state, result.final_state,
                                               "output policy final state")
    source_id = hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
    artifacts = {"model.models[1]": source_id, "measurement.function": source_id}
    prefix = simulation.run(batched, args.steps//2)
    checkpoint = output/f"batch{batch}_checkpoint.h5"
    simulation.save_checkpoint(checkpoint, prefix.final_state, artifact_ids=artifacts)
    restored = simulation.load_checkpoint(checkpoint, artifact_ids=artifacts)
    roundtrip_evidence = validate_state_replicate(restored, prefix.final_state,
                                                  "checkpoint roundtrip", exact=True)
    continued = simulation.run(restored, args.steps//2)
    continuation_evidence = validate_continuation(continued, result,
                                                   args.steps//(2*execution.save_every))
    arrays = {name: np.asarray(value) for name, value in result.observables.items()}
    arrays["times"] = np.asarray(result.times)
    arrays.update({"reference_"+name: value for name, value in reference.items()})
    arrays.update({"continued_"+name: np.asarray(value)
                   for name, value in continued.observables.items()})
    arrays["continued_times"] = np.asarray(continued.times)
    for name, state in (("uninterrupted", result.final_state),
                        ("without_output", empty_result.final_state),
                        ("continued", continued.final_state)):
        for path, value in jax.tree_util.tree_flatten_with_path(state)[0]:
            arrays[f"{name}_final{jax.tree_util.keystr(path)}"] = np.asarray(value)
    archive = output/f"batch{batch}.npz"
    np.savez(archive, **arrays)
    return dict(batch=batch, nstates=problem.model.spec.system.nstates,
                coordinate_shape=list(initial.q.shape), description=simulation.describe(),
                preparation_seconds=preparation, independent_reference_nfev=nfev,
                input_arrays_sha256=array_fingerprint(inputs),
                independent_reference_errors=errors, first_public_run=first, warm_public_runs=warm,
                scalar_batch_agreement=batch_evidence,
                first_without_output=empty_first, warm_without_output=empty,
                trajectory_steps_per_second=batch*args.steps/warm["median_seconds"],
                checkpoint_roundtrip=roundtrip_evidence, continuation=continuation_evidence,
                output_policy_final_state=output_evidence,
                arrays_file=archive.name, arrays_sha256=hashlib.sha256(archive.read_bytes()).hexdigest())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=("molecular", "perovskite"), required=True)
    parser.add_argument("--method", choices=("cpa", "ehrenfest"), required=True)
    parser.add_argument("--device", choices=("cpu", "gpu"), required=True)
    parser.add_argument("--batches", nargs="+", type=int, default=[1, 16, 64])
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--dt", type=float, default=.1)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (args.steps < 32 or args.steps % 32 or args.repeats < 1 or
            min(args.batches) < 1 or len(set(args.batches)) != len(args.batches)
            or not np.isfinite(args.dt) or args.dt <= 0):
        parser.error("positive dt/batches/repeats and steps divisible by 32 are required")
    pyeph.configure_precision(True)
    devices = jax.devices(args.device)
    if not devices or devices[0].platform != args.device:
        raise RuntimeError("requested hardware is unavailable; no fallback is allowed")
    args.output.mkdir(parents=True, exist_ok=False)
    identity = hashes()
    report = dict(schema="pyeph.platform-qualification.v2",
                  created_utc=datetime.now(timezone.utc).isoformat(), family=args.family,
                  method=args.method, device=str(devices[0]), device_kind=devices[0].device_kind,
                  platform=platform.platform(), python=sys.version, source_hashes=identity,
                  versions={name: importlib.metadata.version(name)
                            for name in ("pyeph", "jax", "jaxlib", "numpy", "scipy", "h5py")},
                  environment={key: os.environ.get(key) for key in
                               ("JAX_PLATFORMS", "XLA_FLAGS", "OMP_NUM_THREADS",
                                "OPENBLAS_NUM_THREADS", "CUDA_VISIBLE_DEVICES")},
                  timing_scope="complete public run including validation, output and synchronization",
                  first_run_scope="cold runner after independent scalar reference checks; not process cold start",
                  qualification_validation_scope="reference/replicate comparisons outside measured run intervals",
                  batch_scope="identical initial physical states with distinct IDs; not a sampled ensemble",
                  restart_scope="exact checkpoint byte roundtrip; numerically qualified continuation; "
                                "bitwise trajectory reproducibility is measured, not guaranteed",
                  qualification_tolerances=dict(QUALIFICATION_TOLERANCES),
                  physical_scope="illustrative effective models, no material transferability claim",
                  steps=args.steps, dt=args.dt, results=[])
    with jax.default_device(devices[0]):
        for batch in args.batches:
            print(f"Qualifying {args.family}/{args.method}/{args.device}/batch{batch}", flush=True)
            report["results"].append(run_case(args, batch, args.output))
    if hashes() != identity:
        raise RuntimeError("runtime or benchmark sources changed during qualification")
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    report["process_peak_rss_bytes"] = int(peak if sys.platform == "darwin" else peak*1024)
    report["memory_scope"] = "host process peak including compilation and independent reference; not GPU peak"
    (args.output/"report.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps({"output": str(args.output), "cases": len(report["results"])}), flush=True)


if __name__ == "__main__":
    main()
