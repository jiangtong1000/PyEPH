#!/usr/bin/env python3
"""Run or resume a local durable ensemble without a scheduler dependency.

    JAX_ENABLE_X64=1 python examples/campaign.py outputs/first_campaign --create
    JAX_ENABLE_X64=1 python examples/campaign.py outputs/first_campaign --max-units 1
    JAX_ENABLE_X64=1 python examples/campaign.py outputs/first_campaign

Concurrent invocations may use the same directory on a qualified local
filesystem. This parameterized spin-boson example is a numerical demonstration,
not a calibrated material model.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from pyeph import CoupledClassical, Ehrenfest, Execution, Integrator, Problem, Simulation
from pyeph.core.state import make_state, stack_states
from pyeph.execution.campaign import Campaign
from pyeph.initialization import sample_harmonic
from pyeph.models.analytic import SpinBosonModel


def initialize(ids):
    q, p = sample_harmonic([1.], 1., .1, ids, seed=28)
    return stack_states([make_state(q[i], p[i], [1, 0], trajectory_id=int(identity), seed=28)
                         for i, identity in enumerate(ids)])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--create", action="store_true", help="create a new campaign, then exit")
    parser.add_argument("--max-units", type=int, help="stop cleanly after this many work units")
    args = parser.parse_args()
    if args.max_units is not None and args.max_units < 1:
        parser.error("--max-units must be positive")
    # Caller-owned identity includes the initializer code, seed and distribution.
    preparation = "sha256:" + hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    model = SpinBosonModel()
    simulation = Simulation(
        Problem(model, model.default_params(), CoupledClassical(1.), Ehrenfest()),
        Integrator(.01), Execution(chunk_size=16, save_every=5))
    if args.create:
        store = Campaign.create(args.directory, simulation, np.arange(17), 30,
                                 preparation_id=preparation, shard_size=4)
    else:
        store = Campaign(args.directory)
        completed = 0
        while args.max_units is None or completed < args.max_units:
            claim = store.run_next(simulation, initialize, preparation_id=preparation, batch_size=2)
            if claim is None:
                break
            completed += 1
            print(f"completed unit {claim.shard_id}: IDs {claim.trajectory_ids}", flush=True)
    rows = store.ledger()
    print(json.dumps({state: sum(row["status"] == state for row in rows)
                      for state in ("pending", "running", "failed", "completed")}, sort_keys=True))
    if all(row["status"] == "completed" for row in rows):
        ensemble = store.merge()
        print(json.dumps({"count": ensemble.count,
                          "final_population": ensemble.mean["population"][-1].tolist(),
                          "final_standard_error": ensemble.standard_error["population"][-1].tolist()}))


if __name__ == "__main__":
    main()
