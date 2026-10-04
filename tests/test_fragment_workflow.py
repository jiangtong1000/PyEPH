"""Public execution gates newly exercised by the oriented-fragment provider."""

from dataclasses import replace
import importlib.util
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pyeph import Execution, Integrator, Simulation
from pyeph.core.state import make_state, stack_states
from pyeph.execution.runner import SimulationError
from pyeph.integrators.krylov import LanczosOptions


_PATH = Path(__file__).resolve().parents[1]/"examples/oriented_fragments.py"
_SPEC = importlib.util.spec_from_file_location("oriented_fragments_example", _PATH)
example = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = example
_SPEC.loader.exec_module(example)


@pytest.mark.parametrize("method", ["cpa", "ehrenfest"])
def test_public_parameter_updates_and_awkward_batch_partition(method):
    problem, first = example.fixture(2, method)
    states = [make_state(first.q+.03*i, first.p, first.electronic,
                         seed=430, trajectory_id=17+4*i) for i in range(3)]
    runner = Simulation(problem, Integrator(.2, "rk4"), Execution(chunk_size=2))
    results = [runner.run(s, 3) for s in states]
    for group in (states[:2], states[2:]):
        result = runner.run(stack_states(group), 3)
        for index, initial in enumerate(group):
            reference = results[(int(initial.trajectory_id)-17)//4]
            for name in result.observables:
                np.testing.assert_allclose(result.observables[name][:, index], reference.observables[name], atol=2e-14)
    p, neutral = problem.params
    changed = ({**p, "pp_sigma": p["pp_sigma"]+.02}, neutral)
    runner.update_parameters(changed)
    actual = runner.run(first, 3)
    fresh = Simulation(replace(problem, params=changed), runner.integrator, runner.execution).run(first, 3)
    assert np.max(abs(actual.final_state.electronic-results[0].final_state.electronic)) > 1e-5
    for a, b in zip(jax.tree.leaves(actual.final_state), jax.tree.leaves(fresh.final_state), strict=True):
        np.testing.assert_array_equal(a, b)


def test_checked_cpa_retains_input_when_valid_frame_collapses_at_midpoint():
    problem, initial = example.fixture(2, "cpa")
    # Prescribed free motion hits collinearity inside the first electronic
    # step. Initial-state-only validation must not make that step acceptable.
    target = initial.q[0]+2*(initial.q[1]-initial.q[0])
    momentum = jnp.zeros_like(initial.p).at[2].set(
        2*(target-initial.q[2])*problem.nuclear_treatment.masses[2])
    initial = initial._replace(p=momentum)
    problem.model.validate_at(problem.params, initial.q)
    runner = Simulation(problem, Integrator(1., LanczosOptions(max_dimension=2)), Execution(chunk_size=1))
    with pytest.raises(SimulationError) as caught:
        runner.run(initial, 1)
    assert int(caught.value.last_valid_state.step) == 0
    for a, b in zip(jax.tree.leaves(caught.value.last_valid_state), jax.tree.leaves(initial), strict=True):
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("method", ["cpa", "ehrenfest"])
def test_runnable_reference_refinement_and_checkpoint(tmp_path, method):
    report = example.run_case(tmp_path, 2, method, .5, 4)
    assert report["restart_bitwise"]
    assert report["scipy_max_abs_errors"]["electronic"][1] < 1e-6
    if method == "ehrenfest":
        assert report["energy_change"][1] < .3*report["energy_change"][0]
