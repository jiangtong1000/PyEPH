"""Periodic complex orbital blocks using the same Ehrenfest runner.

Run: JAX_ENABLE_X64=1 python examples/periodic.py
This tests a 3D periodic representation; it is not a fitted perovskite model.
"""

import json

import jax.numpy as jnp
import numpy as np

from pyeph import CoupledClassical, Ehrenfest, Execution, Integrator, Problem, Simulation, make_state
from pyeph.dynamics.ehrenfest import total_energy
from pyeph.models.periodic import PeriodicBlockModel
from pyeph.observables.population import FunctionalMeasurement


def run():
    cell = ((5., 0., 0.), (0., 5., 0.), (0., 0., 5.))
    edges = ((0, 1, 0, 0, 0), (0, 1, -1, 0, 0), (0, 1, 0, -1, 0), (0, 1, 0, 0, -1))
    model = PeriodicBlockModel(2, 2, edges, cell, cutoff=7, switch_on=6)
    q = jnp.array([[0.2, 0.3, 0.4], [2., 2.1, 2.2]])
    params = model.default_params()
    block = jnp.array([[0.04, 0.005j], [0.012 + 0.003j, -0.02]])
    params.update(hopping=jnp.stack([block] * len(edges)), decay=jnp.full(len(edges), 0.3),
                  reference_distance=jnp.linalg.norm(model.displacements(q), axis=1),
                  reference_positions=q, spring=jnp.full(2, 0.1))
    masses = jnp.array([[20.], [30.]])
    measurement = FunctionalMeasurement(lambda problem, state: {
        "energy": total_energy(problem.model, problem.params, state, masses),
        "population": jnp.abs(state.electronic)**2,
    })
    problem = Problem(model, params, CoupledClassical(masses), Ehrenfest(), measurement)
    initial = make_state(q, jnp.zeros_like(q), [1, 0, 0, 0])
    result = Simulation(problem, Integrator(0.02, "exponential_midpoint"),
                        Execution(chunk_size=64, save_every=5)).run(initial, 200)
    energy = result.observables["energy"]
    return {"final_time": float(result.times[-1]),
            "max_energy_drift": float(np.max(abs(energy - energy[0]))),
            "final_populations": result.observables["population"][-1].tolist()}


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
