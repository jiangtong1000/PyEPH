"""Measure numerical takeover error against archived original-source runs.

Run with native Python/JAX, never with the isolated historical environment:
JAX_ENABLE_X64=true USE_MPI=false python compare_reference.py
This writes an acceptance report, not a replacement golden reference.
"""

import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys

import h5py
import jax.numpy as jnp
import numpy as np

from pyeph.greenkubo.simulation import GreenKuboSimulation

from generate_reference import make_case


def errors(actual, reference):
    actual, reference = np.asarray(actual), np.asarray(reference)
    absolute = np.abs(actual-reference)
    selected = np.abs(reference) > 1e-12
    return {"max_abs_error": float(np.max(absolute)),
            "relative_max_norm_error": float(np.max(absolute)/max(float(np.max(np.abs(reference))), 1e-300)),
            "max_elementwise_relative_for_reference_gt_1e-12":
            float(np.max(absolute[selected]/np.abs(reference[selected]))) if selected.any() else None}


def main():
    data = Path(__file__).resolve().parent
    report = {"python": sys.version,
              "dependencies": {name: importlib.metadata.version(name) for name in ("jax", "jaxlib", "numpy", "scipy", "h5py")},
              "precision": "float64/complex128", "integrator": "native classical-path RK4, one electronic substep",
              "thermal_policy": "legacy_full", "sampling": "exact saved original q0,p0 arrays; no seed-equivalence assumption",
              "reference_sha256": hashlib.sha256((data/"reference.h5").read_bytes()).hexdigest(),
              "tolerances": {"H0": {"rtol": 1e-10, "atol": 1e-11},
                             "rho0": {"rtol": 1e-9, "atol": 1e-11},
                             "correlation": {"rtol": 2e-8, "atol": 1e-9},
                             "final_U": {"rtol": 1e-9, "atol": 2e-11}}, "cases": {}}
    with h5py.File(data/"reference.h5") as archive:
        for name in archive:
            reference = archive[name]
            lattice, ham, classical, quantum, propagator = make_case(name)
            simulation = GreenKuboSimulation(lattice, ham, classical, quantum, propagator,
                initial_samples=(reference["q0"][...], reference["p0"][...]))
            model = getattr(simulation.problem.model, "base_model", simulation.problem.model)
            h0 = model.apply(simulation.problem.params, simulation.initial_state.q[0], jnp.eye(lattice.nsites))
            result = simulation.run()
            actuals = {"H0": (h0, reference["h0_first"]),
                       "rho0": (simulation.initial_state.method_state["transport"]["rho0"][0], reference["rho0_first"]),
                       "correlation": (result.observables["current_correlation"], reference["correlation"]),
                       "final_U": (result.final_state.electronic[0], reference["unitary_final_first"])}
            report["cases"][name] = {metric: errors(actual, original[...]) for metric, (actual, original) in actuals.items()}
            for metric, (actual, original) in actuals.items():
                np.testing.assert_allclose(actual, original[...], **report["tolerances"][metric])
            print(name, report["cases"][name]["correlation"], flush=True)
    (data/"native_comparison.json").write_text(json.dumps(report, indent=2)+"\n")


if __name__ == "__main__":
    main()
