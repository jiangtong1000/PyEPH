"""Coupled canonical transport composition across accepted/frustrated events.

This is a finite deterministic batch integration test, not a stationarity or
statistical-equilibrium claim. Every requested trajectory is retained.
"""

import jax
import numpy as np
import pytest

from pyeph import Execution, Integrator, MASHRM
from pyeph.execution.runner import Runner
from pyeph.models.epc import LinearEPCModel
from pyeph.observables.transport.mashrm import FixedPositionVelocity, RMVelocity
from pyeph.workflows.mashrm_equilibrium import LinearEPCCanonical
from pyeph.workflows.mashrm_transport import RMTransport


@pytest.fixture
def fixed_rng_configuration():
    """The chosen event-bearing IDs refer to this explicit PRNG configuration."""
    configuration = {"jax_threefry_partitionable": True,
                     "jax_random_seed_offset": 0,
                     "jax_high_dynamic_range_gumbel": False}
    previous = {name: getattr(jax.config, name) for name in configuration}
    try:
        for name, value in configuration.items():
            jax.config.update(name, value)
        yield
    finally:
        for name, value in previous.items():
            jax.config.update(name, value)


def coupled_workflow():
    # Same noncommuting three-state model used for independent canonical
    # quadrature in test_mashrm_equilibrium.py; no test-module import is needed.
    model = LinearEPCModel(3, 2)
    params = model.create_params(
        [[-.35, .16, .02], [.16, .15, .08], [.02, .08, .65]],
        [[[.2, .12, 0.], [.12, -.1, .04], [0., .04, .08]],
         [[.04, .02, .03], [.02, .12, -.09], [.03, -.09, -.18]]],
        omega=[.9, 1.2], q_eq=[.3, -.2])
    sampler = LinearEPCCanonical(model, params, [1., 2.], 1.7)
    positions = {"x": np.array([0., 1., 2.]), "y": np.array([.4, -.2, .8])}
    measurement = RMVelocity(tuple(positions), FixedPositionVelocity(positions),
                             include_nuclei=True)
    return RMTransport(sampler, Integrator(.04, electronic="exponential_midpoint"),
                       measurement, method=MASHRM(event_substeps=2),
                       execution=Execution(chunk_size=10))


def assert_same_state(actual, expected):
    for left, right in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
        if np.asarray(left).dtype.kind in "biu":
            np.testing.assert_array_equal(left, right)
        else:
            np.testing.assert_allclose(left, right, atol=2e-12, rtol=2e-12)


def numpy_endpoint_velocity(workflow, state):
    """Independent dense commutator and RM estimator at supplied phase points."""
    params = {name: np.asarray(value) for name, value in workflow.problem.params.items()}
    operators = [np.diag(np.asarray(position))
                 for _, position in workflow.measurement.probe_callback.operators]
    result = []
    matrices = []
    for q, c, active in zip(np.asarray(state.q), np.asarray(state.electronic),
                            np.asarray(state.method_state["active"]), strict=True):
        h = params["h0"] + np.einsum("a,aij->ij", q, params["coupling"])
        _, vectors = np.linalg.eigh(h)
        u = vectors[:, active]
        velocities = [1j*(h@position-position@h) for position in operators]
        # Gamma_3 = 47/432 is the independently integrated conditional fourth
        # moment; neither the runtime probe nor rm_velocity is called here.
        result.append([np.sqrt(2/(47/432))*np.real(np.vdot(c, u)*np.vdot(u, v@c))
                       for v in velocities])
        matrices.append(velocities)
    return np.array(result), np.array(matrices)


def test_canonical_coupled_products_and_checkpoint_across_both_event_types(
        tmp_path, fixed_rng_configuration):
    work = coupled_workflow()
    ids = np.arange(8, dtype=np.uint32)
    initial = work.prepare(ids, seed=12)
    np.testing.assert_array_equal(initial.state.trajectory_id, ids)
    np.testing.assert_array_equal(initial.state.method_state["status"], 0)
    np.testing.assert_array_equal(initial.state.method_state["events"], 0)

    # A separate public Runner supplies per-trajectory measurements without the
    # correlation workflow. This checks composition, not independent dynamics.
    direct = Runner(work.problem, work.integrator, work.execution).run(initial.state, 50)
    velocity = np.asarray(direct.observables["velocity"])
    expected_products = velocity[0][None, :, :, None]*velocity[:, :, None, :]
    blocks = []
    full = work.run(initial, 50, observer=lambda t, v: blocks.append(
        (t.copy(), v["velocity_correlation"].copy())))
    np.testing.assert_allclose(np.concatenate([t for t, _ in blocks]),
                               direct.times[:, 0], atol=2e-15, rtol=0)
    np.testing.assert_allclose(np.concatenate([v for _, v in blocks]),
                               expected_products, atol=2e-13, rtol=2e-12)
    np.testing.assert_allclose(full.statistics.mean, expected_products.mean(axis=1),
                               atol=2e-13, rtol=2e-12)
    expected_m2 = np.sum((expected_products-expected_products.mean(axis=1)[:, None])**2,
                         axis=1)
    np.testing.assert_allclose(full.statistics.m2, expected_m2, atol=2e-13, rtol=2e-12)
    assert_same_state(full.final_state.state, direct.final_state)

    # These are the complete predeclared IDs, including zero-event trajectories;
    # no failed or uninteresting rows are removed to obtain the event counts.
    final = full.final_state.state
    np.testing.assert_array_equal(final.trajectory_id, ids)
    np.testing.assert_array_equal(final.method_state["status"], 0)
    np.testing.assert_array_equal(final.method_state["accepted"], [0, 0, 1, 0, 0, 0, 1, 0])
    np.testing.assert_array_equal(final.method_state["frustrated"], [0, 0, 0, 0, 1, 1, 0, 0])
    expected0, operator0 = numpy_endpoint_velocity(work, initial.state)
    expected_final, operator_final = numpy_endpoint_velocity(work, final)
    np.testing.assert_allclose(velocity[0], expected0, atol=2e-13, rtol=2e-12)
    np.testing.assert_allclose(velocity[-1], expected_final, atol=2e-13, rtol=2e-12)
    assert np.max(abs(operator_final-operator0)) > 1e-3

    # At t=1.2 one reflection has occurred. Both accepted hops and the second
    # frustrated event occur after loading, so continuation crosses both types.
    prefix = work.run(initial, 30)
    np.testing.assert_array_equal(prefix.final_state.state.method_state["accepted"], 0)
    np.testing.assert_array_equal(prefix.final_state.state.method_state["frustrated"],
                                  [0, 0, 0, 0, 1, 0, 0, 0])
    checkpoint = tmp_path/"coupled-origin.h5"
    work.save_checkpoint(checkpoint, prefix.final_state)
    restored = work.load_checkpoint(checkpoint)
    np.testing.assert_array_equal(restored.origin.velocity0, initial.origin.velocity0)
    np.testing.assert_array_equal(restored.origin.trajectory_ids, ids)
    assert restored.origin.preparation_id == initial.origin.preparation_id
    assert restored.origin.time0 == 0.
    suffix = work.run(restored, 20)
    assert_same_state(suffix.final_state.state, final)
    np.testing.assert_allclose(suffix.statistics.times[0], 1.2, atol=2e-15, rtol=0)
    for name in ("times", "mean", "m2"):
        joined = np.concatenate((getattr(prefix.statistics, name),
                                 getattr(suffix.statistics, name)[1:]))
        np.testing.assert_allclose(joined, getattr(full.statistics, name),
                                   atol=2e-13, rtol=2e-12)
