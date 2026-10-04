"""Loaded NumPy parameters must work in direct and non-jitted model calls."""

import jax
import jax.numpy as jnp
import numpy as np

from pyeph import CoupledClassical, Ehrenfest, Execution, Integrator, Problem, Simulation, make_state
from pyeph.core.contracts import pure_state_weight
from pyeph.models.aggregate import AggregateModel


def test_numpy_aggregate_parameters_match_jax_in_eager_and_compiled_dynamics():
    model = AggregateModel(2, ((0, 1),))
    parameters = model.default_params() | {"environment": jnp.array([[.03, -.02]])}
    numpy_parameters = jax.tree.map(np.asarray, parameters)
    q = jnp.array([[0., 0., 0.], [2.2, .1, 0.]])
    electronic = jnp.array([2**-.5, 1j*2**-.5])
    for method, argument in ((model.apply, electronic),
                              (model.contract_gradient, pure_state_weight(electronic))):
        np.testing.assert_allclose(method(numpy_parameters, q, argument),
                                   method(parameters, q, argument), atol=2e-15)
    initial = make_state(q, jnp.zeros_like(q), electronic)
    results = []
    for params, jit in ((parameters, True), (numpy_parameters, True), (numpy_parameters, False)):
        problem = Problem(model, params, CoupledClassical(1.), Ehrenfest())
        simulation = Simulation(problem, Integrator(.01), Execution(jit=jit, chunk_size=3))
        results.append(simulation.run(initial, 7))
    for result in results[1:]:
        for expected, actual in zip(jax.tree.leaves(results[0].final_state),
                                    jax.tree.leaves(result.final_state), strict=True):
            np.testing.assert_allclose(actual, expected, atol=2e-15)
