"""Differentiate a smooth two-state, one-mode Ehrenfest observable loss.

The Hamiltonian and harmonic nuclear reference are parameterized numerical
fixtures in a fixed orthonormal carrier basis and canonical atomic units.
This demonstrates sensitivity, not a fitted or validated material model.
"""

import argparse
from dataclasses import replace
import hashlib
import inspect
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from pyeph import CoupledClassical, Ehrenfest, Execution, Integrator, Problem, Simulation, make_state
from pyeph.execution.differentiable import DifferentiableRollout
from pyeph.models.epc import LinearEPCModel


def calculate():
    if not jax.config.x64_enabled:
        raise ValueError("run this qualification example with explicit JAX_ENABLE_X64=1")
    model = LinearEPCModel(2, 1)
    params = model.create_params([[.2, .1], [.1, -.2]],
                                [[[.15, .03], [.03, -.15]]], omega=[.5])
    initial = make_state([.3], [.1], [1., 0.], trajectory_id=71)
    problem = Problem(model, params, CoupledClassical(1.), Ehrenfest())
    integrator = Integrator(.05, "rk4")
    rollout = DifferentiableRollout(problem, integrator, initial, steps=40, rematerialize=True)

    def runtime(theta):
        return params | dict(coupling=params["coupling"]*theta[0]), initial._replace(q=theta[1:])

    def loss(theta):
        parameters, state = runtime(theta)
        result = rollout(parameters, state)
        return (result.observables["population"][-1, 1] - .35)**2

    theta = jnp.array([1., .3])
    value, gradient = jax.jit(jax.value_and_grad(loss))(theta)
    h = 2e-5
    numerical = np.array([(loss(theta+h*direction)-loss(theta-h*direction))/(2*h)
                          for direction in np.eye(2)])
    np.testing.assert_allclose(gradient, numerical, atol=2e-9, rtol=2e-7)
    parameters, state = runtime(theta)
    result = jax.jit(rollout)(parameters, state)
    reference = Simulation(replace(problem, params=parameters), integrator,
                           Execution(chunk_size=40)).run(state, 40)
    np.testing.assert_allclose(result.observables["population"], reference.observables["population"],
                               atol=2e-14)
    sources = {"example": Path(__file__), "rollout": Path(inspect.getfile(DifferentiableRollout))}
    record = dict(scope="parameterized smooth Ehrenfest sensitivity; no material accuracy claim",
                  conventions=dict(basis="fixed orthonormal two-state carrier basis",
                                   coordinates="unit-mass canonical mode", units="atomic",
                                   reference="one harmonic mode, omega=0.5"),
                  parameters=["electronic coupling scale", "initial canonical coordinate"],
                  theta=np.asarray(theta).tolist(), loss=float(value),
                  gradient=np.asarray(gradient).tolist(), finite_difference=numerical.tolist(),
                  finite_difference_step=h,
                  gradient_max_error=float(np.max(abs(np.asarray(gradient)-numerical))),
                  operational_runner_population_max_error=float(np.max(abs(
                      np.asarray(result.observables["population"])-reference.observables["population"]))),
                  steps=40, dt=integrator.dt, electronic="rk4", rematerialize=True,
                  versions=dict(jax=jax.__version__, numpy=np.__version__),
                  source_sha256={name: hashlib.sha256(path.read_bytes()).hexdigest()
                                 for name, path in sources.items()})
    return result, record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result, record = calculate()
    if args.output is not None:
        args.output.mkdir(parents=True, exist_ok=False)
        np.savez_compressed(args.output/"trajectory.npz", times=np.asarray(result.times),
                            population=np.asarray(result.observables["population"]))
        record["trajectory_sha256"] = hashlib.sha256((args.output/"trajectory.npz").read_bytes()).hexdigest()
        (args.output/"report.json").write_text(json.dumps(record, indent=2)+"\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
