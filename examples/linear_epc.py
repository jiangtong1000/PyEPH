"""Canonical EPC + independently sampled phonons + exact thermal CPA transport.

Run: JAX_ENABLE_X64=1 python examples/linear_epc.py
This small reference propagates the full U for each trajectory.
"""

import json

import jax.numpy as jnp
import numpy as np

from pyeph import Execution, Integrator, Simulation, stack_states
from pyeph.initialization import sample_harmonic
from pyeph.models.epc import LinearEPCModel
from pyeph.paths.harmonic import HarmonicBath
from pyeph.workflows.transport import initialize_transport_state, make_transport_problem


def run():
    model = LinearEPCModel(nstates=2, nmodes=1)
    params = model.create_params([[0, 0.2], [0.2, 0]], [[[0.1, 0], [0, -0.1]]], omega=[0.5])
    positions = jnp.array([0., 2.])

    def current(parameters, context, name):
        # Fixed diagonal-position site model: J=q_charge*i[H,X], hbar=1.
        h = model.dense(parameters, context.q)
        return -1j * h * (positions[None, :] - positions[:, None])

    problem = make_transport_problem(model, params, HarmonicBath([0.5]), probe_callback=current)
    q, p = sample_harmonic([0.5], 1.0, 0.3, np.arange(8), seed=2026)
    states = [initialize_transport_state(problem, q[i], p[i], 1 / 0.3, trajectory_id=i)
              for i in range(8)]
    result = Simulation(problem, Integrator(0.02), Execution(chunk_size=64, save_every=5)).run(
        stack_states(states), 200)
    return {"trajectories": 8, "final_time": float(result.times[-1, 0]),
            "max_unitarity_error": float(np.max(result.observables["unitary_error"])),
            "final_mean_current_correlation": float(np.mean(
                result.observables["current_correlation"][-1]).real)}


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
