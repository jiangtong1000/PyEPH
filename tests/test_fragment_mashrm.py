"""Nonlinear all-atom fragment composition with real nonequilibrium RM MASH."""

from dataclasses import replace

import jax
import numpy as np
import pytest

from benchmarks.fragment_mashrm import (
    NumpyOracle, deterministic_event, estimator_audit, fixture, run_native,
)
from pyeph.core.state import stack_states
from pyeph.dynamics.mashrm import sample_population


@pytest.fixture(scope="module")
def periodic_events():
    problem, accepted, oracle, _ = deterministic_event(3, True, True)
    _, frustrated, _, _ = deterministic_event(3, True, False)
    runner, result = run_native(problem, stack_states((accepted, frustrated)), .1, 22)
    return problem, (accepted, frustrated), oracle, runner, result


def test_periodic_fragment_has_accepted_and_frustrated_events_with_rm_estimators(periodic_events):
    problem, _, _, _, result = periodic_events
    data = result.final_state.method_state
    np.testing.assert_array_equal(data["events"], [1, 1])
    np.testing.assert_array_equal(data["accepted"], [1, 0])
    np.testing.assert_array_equal(data["frustrated"], [0, 1])
    np.testing.assert_array_equal(data["active"], [1, 0])
    np.testing.assert_array_equal(data["status"], 0)
    assert max(data["max_event_bracket_width"]) <= 1.001e-10
    assert max(data["max_impulse_energy_error"]) < 1e-13
    assert np.max(abs(result.observables["mapping_norm"]-1)) < 2e-12
    assert np.max(abs(result.observables["energy"]-result.observables["energy"][0])) < 2e-8
    audit = estimator_audit(problem, result)
    assert audit["minimum_sampled_gap"] > .01
    # An individual conditional/deterministic RM estimate is not |c|^2.
    assert np.max(abs(result.observables["population"]-abs(result.observables["electronic"])**2)) > .1


def test_projector_impulse_reference_matches_fixed_c_population_finite_difference():
    problem, initial, oracle, _ = deterministic_event(3, True, False)
    row = oracle.pack(initial.q, initial.p, initial.electronic)
    direction = oracle.direction(row, 0, 1)
    shift = np.sin(np.arange(oracle.nq))
    step = 2e-5
    plus, minus = row.copy(), row.copy()
    plus[:oracle.nq] += step*shift
    minus[:oracle.nq] -= step*shift
    pop_plus, pop_minus = oracle.populations(plus), oracle.populations(minus)
    finite = ((pop_plus[0]-pop_plus[1])-(pop_minus[0]-pop_minus[1]))/(4*step)
    np.testing.assert_allclose(direction@shift, finite, atol=3e-9)


def test_conditional_sphere_preparation_is_id_stable_and_uses_native_real_fragment_basis():
    problem, base = fixture(3, True)
    ids = (105, 7, 31)
    def draw(index):
        return sample_population(problem.model, problem.params, base.q, base.p,
                                 population=0, basis="fixed", seed=430, trajectory_id=index)
    originals = {index: draw(index) for index in ids}
    for index in reversed(ids):
        actual = draw(index)
        for a, b in zip(jax.tree.leaves(actual), jax.tree.leaves(originals[index]), strict=True):
            np.testing.assert_array_equal(a, b)
        populations = abs(np.asarray(actual.electronic))**2
        assert populations[0] == max(populations)
        assert problem.model.spec.complex_valued is False


def test_periodic_reference_keeps_coherent_fragment_image_relabeling():
    problem, initial, oracle, _ = deterministic_event(3, True, True)
    carrier, reference = problem.model.models
    shifts = np.array([[1, -1, 0], [-1, 0, 1], [0, 1, -1]])
    moved = initial.q+shifts[np.asarray(carrier.centers.atom_site)]@np.asarray(carrier.graph.cell)
    changed_model = replace(problem.model, models=(carrier.rewrapped(shifts), reference))
    changed = replace(problem, model=changed_model)
    transformed = NumpyOracle(changed)
    h, energies, _, v, gradient, _ = oracle.quantities(np.asarray(initial.q))
    h2, e2, _, v2, g2, _ = transformed.quantities(np.asarray(moved))
    np.testing.assert_allclose(h2, h, atol=8e-16)
    np.testing.assert_allclose(e2, energies, atol=8e-16)
    np.testing.assert_allclose(v2, v, atol=5e-17)
    np.testing.assert_allclose(g2, gradient, atol=8e-16)
