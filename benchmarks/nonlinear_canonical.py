#!/usr/bin/env python3
"""Source-bound nonlinear canonical preparation against independent quadrature.

Run with JAX_ENABLE_X64=1. This confined one-coordinate two-state model uses
atomic units, a fixed orthonormal electronic basis, classical mass 1.7 and a
quartic neutral reference. It is a parameterized numerical fixture, not a
material model or a quantum-nuclear equilibrium reference.
"""

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time

import jax.numpy as jnp
import numpy as np
from scipy.integrate import quad
from scipy.special import expit

from pyeph.core.contracts import ModelSpec
from pyeph.core.system import SystemSpec
from pyeph.models.base import AutoDiffModel
from pyeph.workflows.canonical_metropolis import NativeCanonicalMetropolis


@dataclass(frozen=True)
class ConfinedModel(AutoDiffModel):
    spec = ModelSpec(SystemSpec(2, (1,), coordinate_kind="canonical"), name="nonlinear-canonical-fixture")

    def apply(self, params, q, vectors):
        x = q[0]
        bias = .2+.5*jnp.tanh(x)+.05*x*x
        hopping = .3+.1*jnp.cos(x)
        return jnp.array([[bias, hopping], [hopping, -bias]]) @ vectors

    def reference_energy(self, params, q):
        x = q[0]
        return .3*x*x+.12*x**4


def reference():
    def values(x):
        radius = np.hypot(.2+.5*np.tanh(x)+.05*x*x, .3+.1*np.cos(x))
        weight = np.exp(-1.4*(.3*x*x+.12*x**4)+np.logaddexp(1.4*radius, -1.4*radius))
        return weight, expit(2*1.4*radius)
    norm = quad(lambda x: values(x)[0], -10, 10, epsabs=1e-11)[0]
    moments = {str(k): quad(lambda x: x**k*values(x)[0], -10, 10, epsabs=1e-11)[0]/norm
               for k in (1, 2, 4)}
    population = quad(lambda x: values(x)[0]*values(x)[1], -10, 10, epsabs=1e-11)[0]/norm
    enlarged = quad(lambda x: values(x)[0], -12, 12, epsabs=1e-11)[0]
    return {"q_moments": moments, "active_ground": population,
            "relative_normalization_domain_change": abs(enlarged/norm-1)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chains", type=int, default=4096)
    args = parser.parse_args()
    if args.chains < 2:
        parser.error("--chains must be at least 2")
    if args.output.exists():
        parser.error("output directory must be fresh")
    args.output.mkdir(parents=True)
    script_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    sampler = NativeCanonicalMetropolis(ConfinedModel(), None, 1.7, 1.4, .7, [0.],
                                        artifact_ids={"model": "sha256:"+script_hash})
    expected = reference()
    reports = []
    for warmup in (0, 100, 500):
        initial = np.where(np.arange(args.chains)[:, None] % 2, -2., 2.)
        start = time.perf_counter()
        prepared = sampler.sample(np.arange(args.chains), initial,
                                   initialization_id="even-ID-plus2-odd-ID-minus2-v1",
                                   burn_in=warmup, production_steps=200, thin=10, seed=8129)
        elapsed = time.perf_counter()-start
        q = np.asarray(prepared.state.q)[:, 0]
        active = np.asarray(prepared.state.method_state["active"])
        reference_q, reference_q2, reference_q4 = (expected["q_moments"][str(k)] for k in (1, 2, 4))
        ground = expected["active_ground"]
        standard_errors = {"q": np.sqrt((reference_q2-reference_q**2)/args.chains),
                           "q_squared": np.sqrt((reference_q4-reference_q2**2)/args.chains),
                           "active_ground": np.sqrt(ground*(1-ground)/args.chains)}
        errors = {"q": float(q.mean()-reference_q), "q_squared": float(np.mean(q*q)-reference_q2),
                  "active_ground": float(np.mean(active == 0)-ground)}
        z_scores = {key: abs(errors[key])/standard_errors[key] for key in errors}
        diagnostics = prepared.diagnostics
        reports.append({"burn_in": warmup, "production_steps": 200, "thin": 10,
                        "chains": args.chains, "seed": 8129,
                        "elapsed_seconds_including_compilation": elapsed,
                        "moment_errors": errors, "endpoint_reference_standard_errors": standard_errors,
                        "absolute_error_in_standard_errors": z_scores,
                        "mean_acceptance": float(np.mean(diagnostics["acceptance_rate"])),
                        "max_split_rhat": float(np.max(diagnostics["split_rhat"])),
                        "lag_one": diagnostics["lag_one_correlation"].tolist(),
                        "warnings": diagnostics["warnings"], "preparation": prepared.metadata})
        sampler.save_chain(args.output/f"warmup_{warmup}.npz", prepared.chain)
    assert reports[-1]["max_split_rhat"] < 1.05
    assert max(reports[-1]["absolute_error_in_standard_errors"].values()) < 6
    assert script_hash == hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    report = {"script_sha256": script_hash, "reference": expected, "runs": reports,
              "scope": "coordinate/active equilibrium sampling on one confined analytic fixture",
              "limitations": ["Finite warmup is not an exact equilibrium draw",
                              "Diagnostic positions within each chain remain correlated",
                              "Independent-ID standard errors do not bound warmup or model bias",
                              "No nonlinear transport, SOC, degeneracy dynamics or quantum nuclei claim"],
              "passed": True}
    (args.output/"report.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps({"passed": True, "output": str(args.output), "runs": [
        {key: run[key] for key in ("burn_in", "max_split_rhat", "absolute_error_in_standard_errors")}
        for run in reports]}, indent=2))


if __name__ == "__main__":
    main()
