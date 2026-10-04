"""Repeat identical inputs through warmed native CPU/GPU computations.

This diagnostic measures same-callable variation, not accuracy or performance.
Finite repeats without variation do not prove universal determinism. Different
compiled output modes are compared separately and cannot establish runtime
nondeterminism merely by disagreeing with each other.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys

import jax
import jax.numpy as jnp
import numpy as np

import pyeph
from pyeph import Execution, Integrator, Simulation, make_state, stack_states
from pyeph.core.contracts import pure_state_weight
from pyeph.io.checkpoint import array_fingerprint
from pyeph.models._block import block_action

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples"))
import oriented_fragments  # noqa: E402
import perovskite  # noqa: E402


def array_tree(tree):
    """Copy numerical leaves to the host, preserving their exact dtypes and bytes."""
    leaves, structure = jax.tree_util.tree_flatten_with_path(tree)
    arrays = {}
    for path, value in leaves:
        array = np.asarray(value).copy()
        name = jax.tree_util.keystr(path) or "value"
        if array.dtype.kind not in "biufc" or not np.isfinite(array).all():
            raise RuntimeError(f"nonfinite or nonnumerical diagnostic value: {name}")
        arrays[name] = array
    return arrays, structure


def compare_arrays(actual, baseline):
    """Report bytes and numerical differences independently, without an accuracy gate."""
    if set(actual) != set(baseline):
        raise RuntimeError("diagnostic output fields changed")
    result = {}
    for name, reference in baseline.items():
        value = actual[name]
        if value.shape != reference.shape or value.dtype != reference.dtype:
            raise RuntimeError(f"diagnostic output shape/dtype changed: {name}")
        # Integer/Boolean outputs are discrete; their exact comparison must not
        # depend on loss of integer precision in floating-point subtraction.
        discrete = value.dtype.kind in "biu"
        if discrete:
            difference = int(np.max(np.abs(value.astype(object) - reference.astype(object)))) \
                if value.size else 0
        else:
            difference = float(np.max(np.abs(value - reference))) if value.size else 0.
            if not np.isfinite(difference):
                raise RuntimeError(f"nonfinite diagnostic difference: {name}")
        result[name] = {
            "bytewise_equal": value.tobytes() == reference.tobytes(),
            "max_absolute_difference": difference, "discrete": discrete,
        }
    return result


def source_hashes():
    package = Path(pyeph.__file__).resolve().parent
    result = {"pyeph/" + path.relative_to(package).as_posix():
              hashlib.sha256(path.read_bytes()).hexdigest() for path in package.rglob("*.py")}
    for path in (Path(__file__), Path(oriented_fragments.__file__), Path(perovskite.__file__)):
        result[path.relative_to(ROOT).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def record_probe(name, function, arguments, repeats, output, *, cache_snapshot=None):
    """Warm once, then repeatedly invoke one unchanged callable on the same arguments."""
    inputs, _ = array_tree(arguments)
    input_hash = array_fingerprint(inputs)
    input_file = output / f"{name}_inputs.npz"
    np.savez(input_file, **inputs)
    # Lower-level probes are already explicitly compiled. Public Simulation.run
    # warms its own fixed cache here; host validation remains part of that probe.
    baseline, structure = array_tree(function(*arguments))
    warmed_cache = cache_snapshot() if cache_snapshot is not None else None
    samples = [baseline]
    comparisons = []
    for _ in range(repeats):
        actual, current_structure = array_tree(function(*arguments))
        if current_structure != structure:
            raise RuntimeError(f"diagnostic output tree changed: {name}")
        if cache_snapshot is not None and cache_snapshot() != warmed_cache:
            raise RuntimeError(f"compiled public-run cache changed after warmup: {name}")
        if array_fingerprint(array_tree(arguments)[0]) != input_hash:
            raise RuntimeError(f"diagnostic inputs changed during repetition: {name}")
        comparisons.append(compare_arrays(actual, baseline))
        samples.append(actual)
    raw = {key: np.stack([sample[key] for sample in samples]) for key in baseline}
    archive = output / f"{name}_outputs.npz"
    np.savez(archive, **raw)
    fields = {}
    for key, value in baseline.items():
        errors = [comparison[key]["max_absolute_difference"] for comparison in comparisons]
        equal = [comparison[key]["bytewise_equal"] for comparison in comparisons]
        fields[key] = {
            "shape": list(value.shape), "dtype": value.dtype.str,
            "discrete": value.dtype.kind in "biu", "bytewise_equal_to_warmup": equal,
            "max_absolute_difference_to_warmup": errors,
            "maximum_absolute_difference": max(errors),
            "byte_sha256": [hashlib.sha256(sample[key].tobytes()).hexdigest()
                            for sample in samples],
        }
    result = {
        "repetitions_after_warmup": repeats, "raw_output_axis_zero": "warmup, then repetitions",
        "same_callable_byte_variation_observed": any(
            not field["bytewise_equal"] for comparison in comparisons for field in comparison.values()),
        "inputs_file": input_file.name, "input_arrays_sha256": input_hash,
        "inputs_file_sha256": hashlib.sha256(input_file.read_bytes()).hexdigest(),
        "outputs_file": archive.name, "outputs_file_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "output_arrays_sha256": [array_fingerprint(sample) for sample in samples], "fields": fields,
    }
    if warmed_cache is not None:
        result["runner_cache_keys"] = [entry[0] for entry in warmed_cache]
        result["runner_cached_callables_unchanged"] = True
        result["executable_scope"] = (
            "same warmed public runner and fixed argument signatures; one full-length trajectory "
            "block, plus its cached initial-observation callable when collection is enabled")
    else:
        result["executable_scope"] = "one explicit JAX lower(...).compile() executable"
    return result, baseline


def run(args):
    module = oriented_fragments if args.family == "molecular" else perovskite
    problem, initial = module.fixture(method=args.method)
    index = np.arange(problem.model.spec.system.nstates, dtype=float)
    amplitudes = np.cos(.37 * index) + 1j * np.sin(.23 * index + .1)
    amplitudes /= np.linalg.norm(amplitudes)
    states = [make_state(initial.q, initial.p, amplitudes, trajectory_id=100 + i, seed=20261004)
              for i in range(args.batch)]
    initial = states[0] if args.batch == 1 else stack_states(states)
    carrier = problem.model.models[0]
    integrator = Integrator(.1, "rk4", electronic_substeps=4)
    execution = Execution(chunk_size=args.steps, save_every=max(1, args.steps // 4))
    description = Simulation(problem, integrator, execution).describe()
    context, _ = array_tree({"state": initial, "params": problem.params,
                             "masses": problem.nuclear_treatment.masses})
    np.savez(args.output / "context_inputs.npz", **context)
    probes, modes = {}, {}
    for collect in (True, False):
        simulation = Simulation(problem, integrator, execution)

        def workflow(state):
            result = simulation.run(state, args.steps, collect=collect)
            return {"final_state": result.final_state, "times": result.times,
                    "observables": result.observables}

        def cache_snapshot():
            return tuple((repr(key), id(value)) for key, value in simulation._compiled.items())

        name = "simulation_collect" if collect else "simulation_no_output"
        probes[name], modes[name] = record_probe(name, workflow, (initial,), args.repeats,
                                                  args.output, cache_snapshot=cache_snapshot)

    # All lower-level arguments remain dynamic inputs to an explicitly compiled
    # executable. Nothing here captures fixed coordinates as foldable constants.
    q = jnp.stack([states[0].q] * args.batch)
    c = jnp.stack([states[0].electronic] * args.batch)
    onsite, hopping = carrier.coefficients(problem.params[0], states[0].q)
    kernels = {
        "atom_centers": (jax.vmap(carrier.centers.apply), (q,)),
        "fixed_coefficients_block_action": (
            lambda onsite, hopping, c: jax.vmap(
                lambda vector: block_action(onsite, hopping, carrier.graph.edges, vector))(c),
            (onsite, hopping, c)),
        "fixed_q_model_action": (
            lambda params, q, c: jax.vmap(lambda x, vector: problem.model.apply(params, x, vector))(q, c),
            (problem.params, q, c)),
    }
    if args.method == "ehrenfest":
        def force(params, q, c):
            def single(x, vector):
                return -problem.model.reference_gradient(params, x) - problem.model.contract_gradient(
                    params, x, pure_state_weight(vector))
            return jax.vmap(single)(q, c)
        kernels["complete_ehrenfest_force"] = (force, (problem.params, q, c))
    for name, (function, arguments) in kernels.items():
        executable = jax.jit(function).lower(*arguments).compile()
        probes[name], _ = record_probe(name, executable, arguments, args.repeats, args.output)
    if array_fingerprint(array_tree({"state": initial, "params": problem.params,
                                     "masses": problem.nuclear_treatment.masses})[0]) != array_fingerprint(context):
        raise RuntimeError("shared physical context changed during diagnostic")
    prefix = "['final_state']"
    collect_final = {key: value for key, value in modes["simulation_collect"].items()
                     if key.startswith(prefix)}
    no_output_final = {key: value for key, value in modes["simulation_no_output"].items()
                       if key.startswith(prefix)}
    return {
        "description": description, "context_input_arrays_sha256": array_fingerprint(context),
        "context_inputs_file_sha256": hashlib.sha256((args.output / "context_inputs.npz").read_bytes()).hexdigest(),
        "probes": probes, "cross_mode_warmup_final_state": compare_arrays(no_output_final, collect_final),
        "cross_mode_interpretation": "different compiled modes; differences alone do not establish "
                                     "same-executable runtime nondeterminism",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=("molecular", "perovskite"), required=True)
    parser.add_argument("--method", choices=("cpa", "ehrenfest"), required=True)
    parser.add_argument("--device", choices=("cpu", "gpu"), required=True)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.batch, args.steps, args.repeats) < 1:
        parser.error("batch, steps and repeats must be positive")
    pyeph.configure_precision(True)
    devices = jax.devices(args.device)
    if not devices or devices[0].platform != args.device:
        raise RuntimeError("requested device unavailable; no fallback is allowed")
    args.output.mkdir(parents=True, exist_ok=False)
    sources = source_hashes()
    report = {
        "schema": "pyeph.platform-reproducibility.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "family": args.family, "method": args.method, "batch": args.batch,
        "steps": args.steps, "dt": .1, "device": str(devices[0]), "device_kind": devices[0].device_kind,
        "platform": platform.platform(), "python": sys.version, "source_hashes": sources,
        "versions": {name: importlib.metadata.version(name)
                     for name in ("pyeph", "jax", "jaxlib", "numpy", "scipy")},
        "environment": {key: os.environ.get(key) for key in
                        ("JAX_PLATFORMS", "XLA_FLAGS", "CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS")},
        "interpretation": "Finite identical-input repetitions test these computations and this "
                          "environment only. Observed same-callable variation is evidence of "
                          "nondeterministic execution, not proof of its kernel-level cause; no "
                          "variation does not establish universal determinism. No timing, accuracy "
                          "qualification, material validation or performance claim is made.",
    }
    with jax.default_device(devices[0]):
        report.update(run(args))
    if source_hashes() != sources:
        raise RuntimeError("diagnostic or runtime source changed during execution")
    (args.output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(args.output), "same_callable_variation": {
        name: value["same_callable_byte_variation_observed"] for name, value in report["probes"].items()}}),
        flush=True)


if __name__ == "__main__":
    main()
