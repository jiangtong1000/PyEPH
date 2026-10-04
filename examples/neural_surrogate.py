#!/usr/bin/env python3
"""Fit a native JAX Hamiltonian residual and reuse unchanged Ehrenfest dynamics.

This is a deterministic 1D analytic demonstration. It fits the last layer of
a tanh network to residual matrix elements and their coordinate derivatives;
it is not a pretrained, atomistic, or equivariant material potential.
"""

import argparse
import json
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np

from pyeph import CoupledClassical, Ehrenfest, Execution, Integrator, Problem, configure_precision, make_state
from pyeph.core.contracts import ModelSpec
from pyeph.core.system import SystemSpec
from pyeph.dynamics.ehrenfest import total_energy
from pyeph.execution.runner import Runner
from pyeph.models.analytic import SpinBosonModel
from pyeph.models.base import AutoDiffModel
from pyeph.models.composite import SumModel
from pyeph.models.neural import NeuralResidualModel


class AnalyticResidual(AutoDiffModel):
    spec = ModelSpec(SystemSpec(2, (1,), coordinate_kind="normal_mode"), name="analytic_nonlinear_residual")

    def dense(self, params, q):
        x = q[0]
        diagonal = .03*jnp.sin(1.2*x)
        coupling = .02*jnp.exp(-.7*x*x)
        return jnp.array([[diagonal, coupling], [coupling, -diagonal]])

    def apply(self, params, q, vectors):
        return self.dense(params, q) @ vectors

    def reference_energy(self, params, q):
        return q.sum()*0


def fit_residual(seed=19):
    """Derivative-supervised linear least squares for the network output layer."""
    network = NeuralResidualModel(2, (1,), hidden_sizes=(32, 32), coordinate_kind="normal_mode")
    params = network.init_params(jax.random.key(seed))
    layers = list(params["layers"])
    keys = jax.random.split(jax.random.key(seed+1), len(layers)-1)
    for i, key in enumerate(keys):
        # Nonzero biases provide both even and odd functions of the coordinate.
        layers[i] = layers[i] | dict(bias=.5*jax.random.normal(key, layers[i]["bias"].shape, dtype=jnp.float64))
    params = params | dict(layers=tuple(layers), q_scale=jnp.array([.8]))

    def features(q):
        value = ((q-params["q_center"])/params["q_scale"]).reshape(-1)
        for layer in params["layers"][:-1]:
            value = jnp.tanh(value@layer["weight"]+layer["bias"])
        return jnp.concatenate((value, jnp.ones(1)))

    analytic = AnalyticResidual()
    train = jnp.linspace(-1.5, 1.5, 61)[:, None]
    phi = jax.vmap(features)(train)
    dphi = jax.vmap(jax.jacfwd(features))(train)[..., 0]
    target = jax.vmap(lambda x: analytic.dense(None, x).reshape(-1))(train)
    dtarget = jax.vmap(jax.jacfwd(lambda x: analytic.dense(None, x).reshape(-1)))(train)[..., 0]
    derivative_weight = .2
    design = jnp.concatenate((phi, derivative_weight*dphi), axis=0)
    targets = jnp.concatenate((target, derivative_weight*dtarget), axis=0)
    weights, _, rank, singular_values = jnp.linalg.lstsq(design, targets, rcond=1e-11)
    layers[-1] = dict(weight=weights[:-1], bias=weights[-1])
    params = params | dict(layers=tuple(layers))
    network.validate_params(params)
    return network, params, dict(seed=seed, training_geometries=61, training_interval=[-1.5, 1.5],
                                 trained_parameters="last affine layer only; fixed random tanh hidden layers",
                                 derivative_label_weight=derivative_weight, least_squares_rank=int(rank),
                                 output_layer_parameters=int(weights.size),
                                 discarded_singular_values=int(jnp.sum(singular_values <= singular_values[0]*1e-11)))


def demonstrate(output, steps=2000, dt=.01):
    configure_precision(True)
    start = time.perf_counter()
    network, nn_params, training = fit_residual()
    jax.block_until_ready(nn_params)
    training["wall_seconds_including_jax_setup"] = time.perf_counter()-start
    analytic = AnalyticResidual()
    # Midpoints of the 60 training intervals are distinct holdout geometries.
    holdout = jnp.linspace(-1.475, 1.475, 60)[:, None]
    prediction = jax.vmap(lambda q: network.dense(nn_params, q))(holdout)
    expected = jax.vmap(lambda q: analytic.dense(None, q))(holdout)
    dpred = jax.vmap(jax.jacfwd(lambda q: network.dense(nn_params, q)))(holdout)
    dtrue = jax.vmap(jax.jacfwd(lambda q: analytic.dense(None, q)))(holdout)
    errors = dict(holdout_geometries=60,
                  matrix_max_absolute_error=float(jnp.max(abs(prediction-expected))),
                  matrix_derivative_max_absolute_error=float(jnp.max(abs(dpred-dtrue))))

    baseline = SpinBosonModel(1)
    base_params = baseline.default_params() | dict(omega=jnp.array([.7]), coupling=jnp.array([.04]),
                                                  bias=.01, delta=.06)
    exact_model = SumModel((baseline, analytic))
    learned_model = SumModel((baseline, network))
    initial = make_state(jnp.array([.9]), jnp.array([.12]), jnp.array([1., 0.]))
    integration = Integrator(dt, electronic="exponential_midpoint")
    execution = Execution(chunk_size=128)
    results = []
    for model, params in ((exact_model, (base_params, None)), (learned_model, (base_params, nn_params))):
        problem = Problem(model, params, CoupledClassical(1.), Ehrenfest())
        result = Runner(problem, integration, execution).run(initial, steps)
        results.append(result)
    reference, learned = results
    errors |= dict(trajectory_steps=steps, timestep=dt,
                   population_max_absolute_error=float(np.max(abs(reference.observables["population"]-learned.observables["population"]))),
                   final_coordinate_absolute_error=float(jnp.max(abs(reference.final_state.q-learned.final_state.q))),
                   final_momentum_absolute_error=float(jnp.max(abs(reference.final_state.p-learned.final_state.p))),
                   learned_final_norm_error=float(abs(jnp.vdot(learned.final_state.electronic, learned.final_state.electronic).real-1)),
                   learned_total_energy_drift=float(total_energy(learned_model, (base_params, nn_params), learned.final_state, 1.)-
                                                    total_energy(learned_model, (base_params, nn_params), initial, 1.)))
    if errors["matrix_max_absolute_error"] > 2e-5 or errors["matrix_derivative_max_absolute_error"] > 2e-4:
        raise RuntimeError(f"neural holdout accuracy gate failed: {errors}")
    if errors["population_max_absolute_error"] > 2e-3 or errors["final_coordinate_absolute_error"] > 2e-3:
        raise RuntimeError(f"neural trajectory parity gate failed: {errors}")
    output.parent.mkdir(parents=True, exist_ok=True)
    weights_path = output.with_suffix(".npz")
    arrays = dict(q_center=np.asarray(nn_params["q_center"]), q_scale=np.asarray(nn_params["q_scale"]))
    for i, layer in enumerate(nn_params["layers"]):
        arrays[f"layer_{i}_weight"] = np.asarray(layer["weight"])
        arrays[f"layer_{i}_bias"] = np.asarray(layer["bias"])
    np.savez(weights_path, **arrays)
    record = dict(scope="analytic 1D residual demonstration, not a trained materials potential",
                  jax=jax.__version__, numpy=np.__version__, backend=jax.default_backend(), precision="float64/complex128",
                  training=training, validation=errors, saved_parameters=weights_path.name,
                  inference="same SumModel and Ehrenfest runner; only analytic residual provider replaced")
    output.write_text(json.dumps(record, indent=2)+"\n")
    print(json.dumps(record, indent=2))
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/neural_surrogate.json"),
                        help="JSON report path; NPZ uses the same stem (default: %(default)s)")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--dt", type=float, default=.01)
    arguments = parser.parse_args()
    demonstrate(arguments.output, arguments.steps, arguments.dt)
