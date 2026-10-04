"""Strict public restart workflow: reproduce a run and reject changed physics."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pyeph import CoupledClassical, Ehrenfest, Execution, Integrator, Problem, Simulation
from pyeph.core.state import make_state, stack_states
from pyeph.dynamics.recorded import RecordedCPA
from pyeph.io.checkpoint import save_checkpoint
from pyeph.models.analytic import SpinBosonModel
from pyeph.paths.electronic import AdiabaticElectronicPath, FixedBasisElectronicPath


def _simulation(*, chunk_size=7, dt=.01):
    model = SpinBosonModel()
    problem = Problem(model, model.default_params(), CoupledClassical(1.), Ehrenfest())
    return Simulation(problem, Integrator(dt), Execution(chunk_size=chunk_size))


def _assert_state_close(actual, expected):
    assert jax.tree.structure(actual) == jax.tree.structure(expected)
    for a, b in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
        np.testing.assert_allclose(a, b, rtol=2e-13, atol=2e-13)


def test_public_restart_preserves_batched_dynamics_across_chunk_size(tmp_path):
    sim = _simulation()
    initial = stack_states([make_state([.2+i*.01], [.3], [1, 0], trajectory_id=i)
                            for i in range(3)])
    expected = sim.run(initial, 40).final_state
    partial = sim.run(initial, 17).final_state
    file = tmp_path / "native.h5"
    sim.save_checkpoint(file, partial)
    resumed_sim = _simulation(chunk_size=11)
    actual = resumed_sim.run(resumed_sim.load_checkpoint(file), 23).final_state
    _assert_state_close(actual, expected)


@pytest.mark.parametrize("changed", ["params", "integrator", "nuclear_treatment"])
def test_public_restart_rejects_changed_scientific_identity(tmp_path, changed):
    sim = _simulation()
    file = tmp_path / "native.h5"
    sim.save_checkpoint(file, make_state([.2], [.3], [1, 0]))
    problem, integrator = sim.problem, sim.integrator
    if changed == "params":
        problem = replace(problem, params={**problem.params, "delta": .2})
    elif changed == "integrator":
        integrator = replace(integrator, dt=.02)
    else:
        problem = replace(problem, nuclear_treatment=CoupledClassical(2.))
    with pytest.raises(ValueError, match=changed):
        Simulation(problem, integrator).load_checkpoint(file)


def test_public_restart_requires_identity_metadata(tmp_path):
    sim = _simulation()
    file = tmp_path / "manual.h5"
    save_checkpoint(file, make_state([0.], [0.], [1, 0]), metadata={})
    with pytest.raises(ValueError, match="no simulation manifest"):
        sim.load_checkpoint(file)


@pytest.mark.parametrize("representation", ["fixed", "adiabatic"])
def test_recorded_public_restart_and_policy_changes(tmp_path, representation):
    times = np.arange(7)*.1
    if representation == "fixed":
        h = np.array([[.1, .2j], [-.2j, -.1]])
        path = FixedBasisElectronicPath(times, np.repeat(h[None], len(times), axis=0))
        sim = RecordedCPA(path, Integrator(.1), Execution(chunk_size=2))
    else:
        angles = np.arange(len(times))*.1
        frames = np.array([[[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]] for a in angles])
        overlaps = np.einsum("nai,naj->nij", frames[:-1], frames[1:])
        path = AdiabaticElectronicPath(times, np.tile([-.1, .1], (len(times), 1)), overlaps)
        sim = RecordedCPA(path, execution=Execution(chunk_size=2))
    initial = sim.initialize([1, 0])
    expected = sim.run(initial, 6).final_state
    file = tmp_path / "recorded.h5"
    sim.save_checkpoint(file, sim.run(initial, 3).final_state)
    actual = sim.run(sim.load_checkpoint(file), 3).final_state
    _assert_state_close(actual, expected)
    changed = RecordedCPA(path, sim.integrator, max_subspace_loss=.1)
    with pytest.raises(ValueError, match="method"):
        changed.load_checkpoint(file)


@pytest.mark.parametrize("field,value,match", [
    ("step", jnp.array(.5), "step"),
    ("step", jnp.array(-1), "step"),
    ("trajectory_id", jnp.array(2**32), "uint32"),
    ("key", jnp.zeros(2, dtype=jnp.int32), "random keys"),
    ("electronic", jnp.ones(()), "electronic dimension"),
    ("q", jnp.array([1j]), "must be real"),
    ("time", jnp.array(1j), "must be real"),
])
def test_manually_constructed_invalid_state_is_rejected(field, value, match):
    initial = make_state([0.], [0.], [1, 0])._replace(**{field: value})
    with pytest.raises(ValueError, match=match):
        _simulation().run(initial, 0)


def test_malformed_batch_and_counter_overflow_are_rejected():
    initial = stack_states([make_state([0.], [0.], [1, 0], trajectory_id=i) for i in range(2)])
    with pytest.raises(ValueError, match="leading trajectory axis"):
        _simulation().run(initial._replace(electronic=jnp.ones((3, 2))), 0)
    initial = make_state([0.], [0.], [1, 0], step=np.iinfo(np.int64).max)
    with pytest.raises(ValueError, match="overflow"):
        _simulation().run(initial, 1)
