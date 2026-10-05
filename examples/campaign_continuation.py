"""Durable scalar segments for two illustrative fixed-provider calculations.

    JAX_ENABLE_X64=1 python -m examples.campaign_continuation NEW_DIR --create
    JAX_ENABLE_X64=1 python -m examples.campaign_continuation NEW_DIR

Use identical --family/--method options when creating and reopening. Select
--family periodic for the second numerical profile. These parameterized models
are not fitted material models; the box does not certify neighbor coverage.
Hard-crash recovery requires explicitly confirming that the old worker stopped
and calling Campaign.recover with its retained claim token. No timer revokes it.
"""

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import jax
import numpy as np

from examples import oriented_fragments, perovskite
from pyeph import Execution, Integrator, LanczosOptions, Simulation, make_state
from pyeph.core.geometry import CoordinateBox
from pyeph.execution.campaign import Campaign


def calculation(family, method):
    """Reconstruct known code; campaign artifacts never instantiate providers."""
    provider = oriented_fragments if family == "molecular" else perovskite
    problem, initial = provider.fixture(method=method)
    q = np.asarray(initial.q)
    problem = replace(problem, measurement=None,
                      geometry_guard=CoordinateBox(q - .4, q + .4))
    simulation = Simulation(problem, Integrator(.02, electronic=LanczosOptions()),
                            Execution(chunk_size=2, save_every=4))
    provider_digest = hashlib.sha256(Path(provider.__file__).read_bytes()).hexdigest()
    reference = "SpringReference" if family == "molecular" else "HarmonicReference"
    artifact_ids = {"model.models[1]": f"sha256:{provider_digest}:{reference}"}
    own_digest = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    preparation = f"sha256:{own_digest}:{provider_digest}:{family}:{method}:seed441"

    def initialize_one(identity):
        return make_state(initial.q, initial.p, initial.electronic,
                          trajectory_id=identity, seed=441)

    return simulation, initialize_one, preparation, artifact_ids, int(initial.trajectory_id)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--family", choices=("molecular", "periodic"), default="molecular")
    parser.add_argument("--method", choices=("cpa", "ehrenfest"), default="ehrenfest")
    parser.add_argument("--create", action="store_true", help="create the campaign and exit")
    args = parser.parse_args()
    if not jax.config.x64_enabled:
        parser.error("set JAX_ENABLE_X64=1 before constructing the calculation")
    simulation, initialize, preparation, artifacts, identity = calculation(args.family, args.method)
    if args.create:
        campaign = Campaign.create(
            args.directory, simulation, [identity], 9, preparation_id=preparation,
            shard_size=1, continuation_steps=3, artifact_ids=artifacts)
    else:
        campaign = Campaign(args.directory)
        campaign.run_next_scalar(simulation, initialize, preparation_id=preparation,
                                 artifact_ids=artifacts)
    ledger = campaign.ledger()
    report = {"family": args.family, "method": args.method,
              "statuses": [row["status"] for row in ledger],
              "scope": "fixed-provider scalar continuation; illustrative effective model"}
    if all(row["status"] == "completed" for row in ledger):
        result = campaign.merge()
        report.update(trajectory_count=result.count, times=result.times.tolist(),
                      final_population=result.mean["population"][-1].tolist())
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
