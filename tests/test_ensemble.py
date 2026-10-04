from dataclasses import replace

import jax.numpy as jnp
import numpy as np
import pytest

from pyeph import CoupledClassical, Ehrenfest, Execution, Integrator, Problem, Simulation
from pyeph.core.state import make_state, stack_states
from pyeph.execution.ensemble import merge_ensembles, partition_ids, run_ensemble
from pyeph.initialization import sample_harmonic
from pyeph.models.analytic import SpinBosonModel


def _simulation():
    model = SpinBosonModel()
    return Simulation(Problem(model, model.default_params(), CoupledClassical(1.), Ehrenfest()),
                      Integrator(.01), Execution(chunk_size=5, save_every=3))


def _initialize(ids):
    q, p = sample_harmonic([1.], 1., .1, ids, seed=28)
    return stack_states([make_state(q[i], p[i], [1, 0], trajectory_id=int(identity), seed=28)
                         for i, identity in enumerate(ids)])


def _run(sim, ids, batch_size=3):
    return run_ensemble(sim, _initialize, ids, 12, batch_size=batch_size,
                        preparation_id="classical-w1-T.1-spin0-seed28-v1")


def test_ensemble_means_and_error_match_explicit_independent_trajectories():
    sim = _simulation()
    ids = np.arange(11)
    all_results = sim.run(_initialize(ids), 12)
    reduced = _run(sim, ids)
    np.testing.assert_allclose(reduced.times, all_results.times[:, 0], atol=1e-16)
    assert reduced.count == len(ids)
    np.testing.assert_array_equal(reduced.trajectory_ids, ids)
    for key, value in all_results.observables.items():
        np.testing.assert_allclose(reduced.mean[key], value.mean(axis=1), atol=5e-15)
        np.testing.assert_allclose(reduced.variance[key], value.var(axis=1, ddof=1), atol=5e-18)
        np.testing.assert_allclose(reduced.standard_error[key],
                                   value.std(axis=1, ddof=1)/np.sqrt(len(ids)), atol=2e-16)


def test_partition_merge_matches_whole_ensemble_and_changes_batch_size():
    sim = _simulation()
    ids = np.array([19, 4, 12, 20, 7, 1, 38, 10])
    expected = _run(sim, ids, batch_size=4)
    partial = [_run(sim, partition_ids(ids, rank, 3), batch_size=2) for rank in range(3)]
    actual = merge_ensembles(merge_ensembles(partial[2], partial[0]), partial[1])
    np.testing.assert_array_equal(np.sort(actual.trajectory_ids), np.sort(ids))
    for key in expected.mean:
        np.testing.assert_allclose(actual.mean[key], expected.mean[key], atol=5e-15)
        np.testing.assert_allclose(actual.variance[key], expected.variance[key], atol=5e-18)
    assert partition_ids(ids, 9, 10).shape == (0,)


def test_merge_rejects_different_physics_preparation_grid_or_duplicate_ids():
    sim = _simulation()
    a, b = _run(sim, [0, 1]), _run(sim, [2, 3])
    with pytest.raises(ValueError, match="duplicate"):
        merge_ensembles(a, a)
    with pytest.raises(ValueError, match="preparations"):
        merge_ensembles(a, replace(b, preparation_id="different-temperature"))
    with pytest.raises(ValueError, match="time grids"):
        merge_ensembles(a, replace(b, times=b.times+.1))
    changed = _simulation()
    changed.problem.params["delta"] = .7
    with pytest.raises(ValueError, match="params"):
        merge_ensembles(a, _run(changed, [2, 3]))


def test_one_sample_uncertainty_is_undefined_and_batch_callback_receives_states():
    received = []
    result = run_ensemble(_simulation(), _initialize, [17], 3, preparation_id="one-sample-v1",
                          batch_observer=lambda ids, out: received.append((ids, out.final_state)))
    assert result.count == 1
    assert np.isnan(result.standard_error["population"]).all()
    np.testing.assert_array_equal(received[0][0], [17])
    np.testing.assert_array_equal(received[0][1].trajectory_id, [17])


def test_wrong_initializer_identity_or_time_grid_is_rejected():
    sim = _simulation()
    with pytest.raises(ValueError, match="requested trajectory IDs"):
        run_ensemble(sim, lambda ids: _initialize(ids[::-1]), [1, 2], 0,
                     preparation_id="test")
    def staggered(ids):
        state = _initialize(ids)
        return state._replace(time=jnp.arange(len(ids))*.1)
    with pytest.raises(ValueError, match="physical output time grid"):
        run_ensemble(sim, staggered, [1, 2], 0, preparation_id="test")


@pytest.mark.parametrize("ids", [[1, 1], [-1, 2], [0, 2**32], [1., 2.], []])
def test_invalid_ids_rejected(ids):
    with pytest.raises(ValueError, match="trajectory IDs"):
        partition_ids(ids, 0, 1)


@pytest.mark.parametrize("change,match", [
    ("times", "strictly increasing"), ("empty_times", "strictly increasing"),
    ("mean", "output time axis"), ("negative_m2", "nonnegative"),
    ("complex_m2", "real"), ("preparation_id", "nonempty"),
])
def test_manually_reconstructed_ensemble_is_validated_before_merge(change, match):
    sim = _simulation()
    a, b = _run(sim, [0, 1]), _run(sim, [2, 3])
    changes = {
        "times": {"times": b.times[::-1]},
        "empty_times": {"times": np.array([])},
        "mean": {"mean": {**b.mean, "population": b.mean["population"][:-1]}},
        "negative_m2": {"m2": {**b.m2, "population": -np.ones_like(b.m2["population"])}},
        "complex_m2": {"m2": {**b.m2, "population": b.m2["population"].astype(complex)}},
        "preparation_id": {"preparation_id": ""},
    }
    with pytest.raises(ValueError, match=match):
        merge_ensembles(a, replace(b, **changes[change]))
