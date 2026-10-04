"""Correlation products, independent-trajectory reduction and exact origins."""

from dataclasses import replace

import jax
import numpy as np
import pytest

from pyeph import Execution, Integrator, MASHRM
from pyeph.execution.ensemble import merge_ensembles
from pyeph.io.checkpoint import load_checkpoint
from pyeph.models.epc import LinearEPCModel
from pyeph.observables.transport.mashrm import FixedPositionVelocity, RMVelocity
from pyeph.workflows.mashrm_equilibrium import LinearEPCCanonical
from pyeph.workflows.mashrm_transport import RMCorrelationOrigin, RMTransport, RMTransportState


def workflow(*, chunk_size=3, save_every=1, beta=1.6, dt=.04, positions=None):
    model = LinearEPCModel(3, 1)
    params = model.create_params(np.diag([-.4, .1, .8]), np.zeros((1, 3, 3)), omega=[.7])
    sampler = LinearEPCCanonical(model, params, [2.], beta)
    positions = positions or {
        "x": np.array([[0., .8, .2], [.8, 1., .4], [.2, .4, 2.]]),
        "y": np.array([[0., -.2, .6], [-.2, .3, .9], [.6, .9, .5]])}
    measurement = RMVelocity(tuple(positions), FixedPositionVelocity(positions))
    return RMTransport(sampler, Integrator(dt, electronic="exponential_midpoint"), measurement,
                       method=MASHRM(event_substeps=1),
                       execution=Execution(chunk_size=chunk_size, save_every=save_every))


def assert_same_state(left, right):
    for a, b in zip(jax.tree.leaves(left), jax.tree.leaves(right), strict=True):
        np.testing.assert_allclose(a, b, atol=4e-14, rtol=4e-14)


def test_per_trajectory_outer_products_precede_moments_and_probe_order():
    work = workflow()
    initial = work.prepare([19, 4, 9, 27, 33, 61, 8], seed=41)
    blocks = []
    result = work.run(initial, 7, observer=lambda t, v: blocks.append((t, v)))
    velocities = np.concatenate([block[1]["velocity"] for block in blocks])
    products = velocities[0][None, :, :, None]*velocities[:, :, None, :]
    actual_products = np.concatenate([block[1]["velocity_correlation"] for block in blocks])
    np.testing.assert_array_equal(actual_products, products)
    np.testing.assert_allclose(result.statistics.mean, products.mean(axis=1), atol=2e-15)
    np.testing.assert_allclose(result.statistics.variance, products.var(axis=1, ddof=1), atol=2e-15)
    np.testing.assert_allclose(result.statistics.standard_error,
                               products.std(axis=1, ddof=1)/np.sqrt(7), atol=2e-15)
    assert np.max(abs(products.mean(axis=1)-np.einsum(
        "i,tj->tij", velocities[0].mean(axis=0), velocities.mean(axis=1)))) > .01
    assert result.metadata["correlation_order"] == "initial_i_times_current_j"
    assert result.final_state.origin is initial.origin
    assert set(result.final_state.state.method_state) == set(initial.state.method_state)


def test_disjoint_partition_merge_preserves_draws_and_correlation_statistics():
    work = workflow()
    ids = np.array([41, 3, 9, 12, 31, 7, 19], dtype=np.uint32)
    together = work.run(work.prepare(ids, seed=83), 4).statistics
    first = work.run(work.prepare(ids[:3], seed=83), 4).statistics
    second = work.run(work.prepare(ids[3:], seed=83), 4).statistics
    merged = merge_ensembles(first, second)
    np.testing.assert_array_equal(merged.trajectory_ids, ids)
    np.testing.assert_allclose(merged.mean, together.mean, atol=2e-15)
    np.testing.assert_allclose(merged.m2, together.m2, atol=3e-15)
    with pytest.raises(ValueError, match="duplicate"):
        merge_ensembles(first, first)


def test_static_h_canonical_vacf_matches_independent_thermal_trace():
    work = workflow(dt=.1)
    initial = work.prepare(np.arange(4096), seed=731)
    result = work.run(initial, 3).statistics
    h = np.asarray(work.problem.params["h0"])
    energies = np.diag(h)
    weights = np.exp(-work.sampler.beta*(energies-energies.min()))
    weights /= weights.sum()
    operators = [1j*(h@np.asarray(x)-np.asarray(x)@h)
                 for _, x in work.measurement.probe_callback.operators]
    exact = np.empty_like(result.mean)
    for it, lag in enumerate(result.times):
        phase = np.cos((energies[None, :]-energies[:, None])*lag)
        for i, vi in enumerate(operators):
            for j, vj in enumerate(operators):
                exact[it, i, j] = np.sum(weights[:, None]*(vi*vj.T).real*phase)
    # The independent finite ensemble has sampling error, not exact trace parity.
    assert np.all(abs(result.mean-exact) <= 5*result.standard_error+1e-12)
    assert np.all(result.standard_error > 0)


def test_checkpoint_continuation_retains_origin_and_shared_boundary(tmp_path):
    work = workflow(save_every=2)
    initial = work.prepare([6, 2, 31, 8], seed=73)
    full = work.run(initial, 8)
    prefix = work.run(initial, 4)
    path = tmp_path/"transport.h5"
    work.save_checkpoint(path, prefix.final_state)
    with pytest.raises(ValueError, match="auxiliary"):
        load_checkpoint(path)
    restored = work.load_checkpoint(path)
    suffix = work.run(restored, 4)
    assert_same_state(suffix.final_state.state, full.final_state.state)
    np.testing.assert_array_equal(restored.origin.velocity0, initial.origin.velocity0)
    np.testing.assert_array_equal(restored.origin.trajectory_ids, initial.origin.trajectory_ids)
    assert restored.origin.time0 == initial.origin.time0
    # An explicit one-sample deduplication joins adjacent time segments.
    assert prefix.statistics.times[-1] == suffix.statistics.times[0]
    np.testing.assert_allclose(np.concatenate((prefix.statistics.times, suffix.statistics.times[1:])),
                               full.statistics.times, atol=2e-16)
    for field in ("mean", "m2"):
        actual = np.concatenate((getattr(prefix.statistics, field), getattr(suffix.statistics, field)[1:]))
        np.testing.assert_allclose(actual, getattr(full.statistics, field), atol=3e-14)
    # A newly reconstructed workflow with the same scientific configuration works.
    reloaded = workflow(save_every=2).load_checkpoint(path)
    assert_same_state(reloaded.state, restored.state)


def test_off_schedule_segment_boundary_is_an_extra_observation():
    work = workflow(save_every=2)
    initial = work.prepare([6, 2], seed=73)
    full = work.run(initial, 6)
    prefix = work.run(initial, 3)
    suffix = work.run(prefix.final_state, 3)
    joined_times = np.concatenate((prefix.statistics.times, suffix.statistics.times[1:]))
    joined_mean = np.concatenate((prefix.statistics.mean, suffix.statistics.mean[1:]))
    np.testing.assert_allclose(joined_times, [.0, .08, .12, .16, .24], atol=2e-16)
    # Forced terminal output at step 3 survives; the unsplit run has no such sample.
    common = np.array([0, 1, 3, 4])
    np.testing.assert_allclose(joined_times[common], full.statistics.times, atol=2e-16)
    np.testing.assert_allclose(joined_mean[common], full.statistics.mean, atol=3e-14)
    assert_same_state(suffix.final_state.state, full.final_state.state)


def test_streaming_does_not_retain_output_history():
    work = workflow(chunk_size=2)
    initial = work.prepare([2, 31, 11], seed=9)
    blocks = []
    streamed = work.run(initial, 7, collect=False,
                        observer=lambda t, v: blocks.append((t.copy(), v["velocity_correlation"].copy())))
    collected = work.run(initial, 7)
    assert streamed.statistics is None
    assert max(len(t) for t, _ in blocks) <= 2
    np.testing.assert_allclose(np.concatenate([v for _, v in blocks]).mean(axis=1),
                               collected.statistics.mean, atol=2e-15)
    assert_same_state(streamed.final_state.state, collected.final_state.state)
    skipped = work.run(initial, 7, collect=False)
    assert skipped.statistics is None
    assert_same_state(skipped.final_state.state, collected.final_state.state)


def test_origin_snapshots_and_returned_metadata_cannot_change_continuation():
    work = workflow()
    initial = work.prepare([2, 31], seed=9)
    metadata = initial.origin.preparation_metadata
    metadata["seed"] = 99
    assert initial.origin.preparation_metadata["seed"] == 9
    v0 = np.array(initial.origin.velocity0)
    origin = RMCorrelationOrigin(initial.origin.trajectory_ids, v0, 0., work.fingerprint,
                                 initial.origin.preparation_metadata)
    v0[:] = 1e9
    np.testing.assert_array_equal(origin.velocity0, initial.origin.velocity0)
    assert work.identity is not work.identity


def test_observer_array_reuse_does_not_modify_collected_moments():
    work = workflow()
    initial = work.prepare([2, 31, 9], seed=9)
    expected = work.run(initial, 4).statistics

    def reuse_arrays(times, values):
        times[:] = 1e6
        values["velocity_correlation"][:] = -1e6
        values["velocity"] = np.zeros_like(values["velocity"])

    actual = work.run(initial, 4, observer=reuse_arrays).statistics
    np.testing.assert_array_equal(actual.times, expected.times)
    np.testing.assert_array_equal(actual.mean, expected.mean)
    np.testing.assert_array_equal(actual.m2, expected.m2)


@pytest.mark.parametrize("scale,reason", [(1e160, "products"), (1e80, "moments")])
def test_nonrepresentable_correlation_data_never_reaches_workflow_observer(scale, reason):
    position = scale*np.array([[0., .8, .2], [.8, 1., .4], [.2, .4, 2.]])
    work = workflow(positions={"x": position})
    initial = work.prepare([2, 31, 9], seed=9)
    published = []
    with pytest.raises(ValueError, match=reason):
        work.run(initial, 0, observer=lambda *args: published.append(args))
    assert published == []


def test_origin_metadata_is_revalidated_against_canonical_sampler():
    work = workflow()
    initial = work.prepare([2, 31], seed=9)
    metadata = initial.origin.preparation_metadata
    metadata["seed"] = 19
    altered = RMCorrelationOrigin(initial.origin.trajectory_ids, initial.origin.velocity0,
                                  initial.origin.time0, work.fingerprint, metadata)
    with pytest.raises(ValueError, match="preparation metadata mismatch"):
        work.run(RMTransportState(initial.state, altered), 0)


@pytest.mark.parametrize("change", ["id_order", "time", "origin_time", "probe_shape"])
def test_invalid_origin_or_batch_alignment_rejected_before_output(change):
    work = workflow()
    initial = work.prepare([2, 31], seed=9)
    if change == "id_order":
        initial = replace(initial, state=initial.state._replace(
            trajectory_id=initial.state.trajectory_id[::-1]))
    elif change == "time":
        initial = replace(initial, state=initial.state._replace(time=initial.state.time.at[1].set(.1)))
    else:
        origin = initial.origin
        altered = RMCorrelationOrigin(
            origin.trajectory_ids, origin.velocity0[:, :1] if change == "probe_shape" else origin.velocity0,
            .1 if change == "origin_time" else origin.time0, origin.workflow_fingerprint,
            origin.preparation_metadata)
        initial = RMTransportState(initial.state, altered)
    published = []
    with pytest.raises(ValueError):
        work.run(initial, 0, observer=lambda *args: published.append(args))
    assert published == []


@pytest.mark.parametrize("change", ["beta", "dt", "probe"])
def test_changed_scientific_configuration_rejects_origin_and_checkpoint(tmp_path, change):
    original = workflow()
    initial = original.prepare([2, 31], seed=9)
    path = tmp_path/"original.h5"
    original.save_checkpoint(path, initial)
    changed = workflow(**({"beta": 1.8} if change == "beta" else {"dt": .05}
                          if change == "dt" else {"positions": {"x": np.ones((3, 3))}}))
    with pytest.raises(ValueError, match="different workflow"):
        changed.run(initial, 0)
    with pytest.raises(ValueError, match="identity mismatch"):
        changed.load_checkpoint(path)


def test_gap_scope_and_opaque_probe_identity_are_explicit():
    work = workflow()
    with pytest.raises(ValueError, match="gap tolerances"):
        RMTransport(work.sampler, work.integrator, replace(work.measurement, gap_tolerance=1e-8))
    def callback(model, params, context, probe, vectors):
        return 0j*vectors
    with pytest.raises(ValueError, match="incomplete"):
        RMTransport(work.sampler, work.integrator, RMVelocity(probe_callback=callback))
    explicit = RMTransport(work.sampler, work.integrator, RMVelocity(probe_callback=callback),
                           artifact_ids={"measurement.probe_callback": "zero-velocity-test-v1"})
    assert explicit.identity["simulation"]["complete"]


def test_one_trajectory_has_mean_and_undefined_sem():
    work = workflow()
    result = work.run(work.prepare([8], seed=3), 0).statistics
    assert result.count == 1 and result.mean.shape == (1, 2, 2)
    assert np.isnan(result.standard_error).all()
