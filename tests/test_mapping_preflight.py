"""Current-model checks before mapping output, including native batch lanes."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pyeph import CoupledClassical, Integrator, Problem, Simulation
from pyeph.core.state import stack_states
from pyeph.dynamics.mash2 import MASH2, MASHError, MASHPopulation, mapping_state as spin_state
from pyeph.dynamics.mashrm import MASHRM, MASHRMPopulation, mapping_state as rm_state
from pyeph.models.base import AutoDiffModel
from pyeph.models.epc import LinearEPCModel


def setup(kind):
    n = 2 if kind == "mash2" else 3
    model = LinearEPCModel(n, 2)
    params = model.create_params(np.diag(np.linspace(-.4, .8, n)), np.zeros((2, n, n)),
                                 omega=[.2, .3])
    method = MASH2() if kind == "mash2" else MASHRM()
    measurement = MASHPopulation(include_nuclei=True) if kind == "mash2" else MASHRMPopulation(
        include_nuclei=True)
    problem = Problem(model, params, CoupledClassical([1., 2.]), method, measurement)
    return problem


def prepared(problem, *, q=(-.2, .1), trajectory_id=0, boundary=False):
    kwargs = dict(trajectory_id=trajectory_id, active=0)
    if isinstance(problem.method, MASH2):
        return spin_state(problem.model, problem.params, q, [.1, .2],
                          [1., 0., 0.] if boundary else [0., 0., -1.], **kwargs)
    return rm_state(problem.model, problem.params, q, [.1, .2],
                    np.sqrt([.4, .4, .2]) if boundary else [1., 0., 0.],
                    basis="adiabatic", **kwargs)


def simulation(problem):
    return Simulation(problem, Integrator(.01, "exponential_midpoint"))


@pytest.mark.parametrize("kind", ["mash2", "mashrm"])
@pytest.mark.parametrize("steps", [0, 1])
def test_updated_parameters_reject_stale_active_before_output_and_checkpoint(kind, steps, tmp_path):
    problem = setup(kind)
    state = prepared(problem)
    runner = simulation(problem)
    runner.run(state, 0)
    original = tmp_path/"original.h5"
    runner.save_checkpoint(original, state)
    updated = {**problem.params, "h0": jnp.flip(problem.params["h0"], axis=(0, 1))}
    runner.update_parameters(updated)
    observations = []
    with pytest.raises(ValueError, match="initial active surface"):
        runner.run(state, steps, observer=lambda *row: observations.append(row))
    assert not observations
    with pytest.raises(ValueError, match="initial active surface"):
        runner.run(state, steps, collect=False)
    invalid = tmp_path/"invalid.h5"
    with pytest.raises(ValueError, match="initial active surface"):
        runner.save_checkpoint(invalid, state)
    assert not invalid.exists()
    with pytest.raises(ValueError, match="params"):
        runner.load_checkpoint(original)
    assert int(state.method_state["active"]) == 0  # validation never reselects it
    replacement = prepared(runner.problem)
    runner.run(replacement, 0)
    changed = tmp_path/"updated.h5"
    runner.save_checkpoint(changed, replacement)
    loaded = runner.load_checkpoint(changed)
    np.testing.assert_array_equal(loaded.electronic, replacement.electronic)


@pytest.mark.parametrize("kind", ["mash2", "mashrm"])
def test_native_batch_and_single_preflights_agree_including_explicit_boundaries(kind):
    problem = setup(kind)
    states = [prepared(problem, q=(-.1*i, .02*i), trajectory_id=i, boundary=i == 3)
              for i in range(8)]
    runner = simulation(problem)
    actual = runner.run(stack_states(states), 0)
    expected = [runner.run(state, 0) for state in states]
    for key, value in actual.observables.items():
        combined = np.stack([result.observables[key] for result in expected], axis=1)
        np.testing.assert_allclose(value, combined, atol=3e-15, rtol=0.)


class FailureAtPositiveCoordinate(AutoDiffModel):
    """One specified lane can violate one physical condition in a pure model."""

    def __init__(self, model, failure):
        self.spec = model.spec
        self.failure = failure

    def apply(self, params, q, vectors):
        h = params["h0"]
        bad = q[0] > 0
        if self.failure == "complex":
            value = jnp.where(bad, .1j, 0j)
            h = h.astype(complex).at[0, 1].set(value).at[1, 0].set(-value)
        elif self.failure == "nonhermitian":
            h = h.at[0, 1].set(jnp.where(bad, .1, 0.))
        elif self.failure == "degenerate":
            h = h.at[1, 1].set(jnp.where(bad, h[0, 0], h[1, 1]))
        elif self.failure == "ownership":
            h = jnp.where(bad, jnp.flip(h, axis=(0, 1)), h)
        return h@vectors

    def reference_energy(self, params, q):
        return jnp.where(q[0] > 0, jnp.nan, 0.) if self.failure == "reference" else jnp.array(0.)

    def reference_gradient(self, params, q):
        return (jnp.full_like(q, jnp.where(q[0] > 0, jnp.nan, 0.))
                if self.failure == "force" else jnp.zeros_like(q))


@pytest.mark.parametrize("kind", ["mash2", "mashrm"])
@pytest.mark.parametrize("failure", ["complex", "nonhermitian", "degenerate", "reference",
                                     "force", "ownership"])
def test_invalid_late_batch_lane_rejects_before_initial_observation(kind, failure):
    problem = setup(kind)
    states = [prepared(problem, trajectory_id=i) for i in range(9)]
    batch = stack_states(states)
    batch = batch._replace(q=batch.q.at[7, 0].set(.2))
    broken = replace(problem, model=FailureAtPositiveCoordinate(problem.model, failure))
    runner = simulation(broken)
    observations = []
    with pytest.raises(ValueError, match=r"invalid trajectory lanes \[7\]"):
        runner.run(batch, 0, observer=lambda *row: observations.append(row))
    assert not observations
    # The same defect must be rejected by the single-trajectory path as well.
    lane = jax.tree.map(lambda value: value[7], batch)
    with pytest.raises(ValueError, match=r"invalid trajectory lanes \[0\]"):
        runner.run(lane, 0, collect=False)


@pytest.mark.parametrize("field,dtype", [("events", jnp.int64), ("active", jnp.int16),
                                        ("status", jnp.float64),
                                        ("max_event_residual", jnp.float32),
                                        ("max_impulse_energy_error", jnp.complex128)])
def test_mash2_canonical_metadata_dtypes_reject_before_compilation(field, dtype):
    problem = setup("mash2")
    initial = prepared(problem)
    state = initial._replace(method_state={**initial.method_state,
                                         field: initial.method_state[field].astype(dtype)})
    observations = []
    with pytest.raises(ValueError, match="constructor dtype"):
        simulation(problem).run(state, 1, observer=lambda *row: observations.append(row))
    assert not observations


def test_mash2_opaque_provider_preflight_keeps_host_lane_fallback():
    problem = setup("mash2")

    class HostModel:
        spec = replace(problem.model.spec, native_jax=False)

        def apply(self, params, q, vectors):
            # Converting a traced q to NumPy would fail, exposing an accidental vmap.
            assert np.asarray(q).shape == (2,)
            return jnp.asarray(np.asarray(params["h0"])@np.asarray(vectors))

        def reference_energy(self, params, q):
            return np.asarray(q)@np.asarray(q)/2

        def reference_gradient(self, params, q):
            return jnp.asarray(np.asarray(q))

        def contract_gradient(self, params, q, weight):
            return jnp.zeros_like(q)

    states = stack_states([prepared(problem, trajectory_id=i) for i in range(3)])
    problem.method.validate_initial_state(replace(problem, model=HostModel()), states, batch=True)


def test_direct_mash2_kernel_still_reports_inconsistent_hemisphere_status():
    problem = setup("mash2")
    initial = prepared(problem)
    invalid = initial._replace(method_state={**initial.method_state, "active": jnp.int32(1)})
    step = problem.method.build_step(problem, Integrator(.01, "exponential_midpoint"))
    failed = jax.jit(step)(invalid)
    assert int(failed.method_state["status"]) == 2
    np.testing.assert_array_equal(failed.q, initial.q)
    np.testing.assert_array_equal(failed.p, initial.p)
    assert float(failed.time) == float(initial.time)
    with pytest.raises(MASHError, match="hemisphere"):
        problem.method.validate_result(failed)
