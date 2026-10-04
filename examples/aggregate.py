"""Nonlinear 3D aggregate with coupled Ehrenfest motion and explicit energy.

Run: JAX_ENABLE_X64=1 python examples/aggregate.py
The parameters are a numerical fixture, not a calibrated material model.
"""

import json

import jax.numpy as jnp
import numpy as np

from pyeph import CoupledClassical, Ehrenfest, Execution, Integrator, Problem, Simulation, make_state
from pyeph.dynamics.ehrenfest import total_energy
from pyeph.models.aggregate import AggregateModel
from pyeph.observables.population import FunctionalMeasurement


def run():
    q = jnp.array([[0., 0., 0.], [2., 0.2, 0.1], [0.8, 2., 0.3]])
    model = AggregateModel(3, ((0, 1), (0, 2), (1, 2)), cutoff=5, switch_on=4)
    params = model.default_params()
    params.update(reference_positions=q, spring=jnp.full(3, 0.2),
                  onsite=jnp.array([0., 0.03, -0.02]))
    masses = jnp.array([[10.], [12.], [11.]])
    measurement = FunctionalMeasurement(lambda problem, state: {
        "energy": total_energy(problem.model, problem.params, state, masses),
        "population": jnp.abs(state.electronic)**2,
    })
    problem = Problem(model, params, CoupledClassical(masses), Ehrenfest(), measurement)
    initial = make_state(q, jnp.zeros_like(q).at[1, 0].set(0.05), [1, 0, 0])
    result = Simulation(problem, Integrator(0.02, "exponential_midpoint"),
                        Execution(chunk_size=64, save_every=5)).run(initial, 200)
    energy = result.observables["energy"]
    return {"final_time": float(result.times[-1]),
            "max_energy_drift": float(np.max(abs(energy - energy[0]))),
            "final_populations": result.observables["population"][-1].tolist()}


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
