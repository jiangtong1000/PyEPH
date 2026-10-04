"""Independent physical-embedding checks for the AO-to-recorded-path bridge.

Columns of A are AO functions embedded in a known orthonormal physical space.
Thus S=A†A and S01=A0†A1 independently specify the meaning of the input arrays.
Tests reconstruct the physical wavefunction, rather than comparing two copies
of the row-coefficient projection formula. They do not certify arbitrary input
energies as eigenvalues or infer missing AO spatial force information.
"""

from dataclasses import replace

import jax
import numpy as np
import pytest
from scipy.linalg import expm

from pyeph.adapters.ao_frames import project_ao_path
from pyeph.core.units import ATOMIC_TIME_FS, HARTREE_EV
from pyeph.dynamics.recorded import RecordedCPA
from pyeph.execution.runner import Execution


def _unitary(rng, n):
    q, r = np.linalg.qr(rng.normal(size=(n, n))+1j*rng.normal(size=(n, n)))
    return q * np.exp(-1j*np.angle(np.diag(r)))[None, :]


def _embedding(*, gauge="none"):
    rng = np.random.default_rng(819231)
    times = np.array([1.1, 1.17, 1.26, 1.41])
    energies = np.array([-.4, .1, .55, 1.2])
    if gauge == "degenerate":
        energies[2] = energies[1]
    eigenvectors = _unitary(rng, 4)
    physical_h = (eigenvectors*energies) @ eigenvectors.conj().T
    frames, physical_eigenvectors, rows = [], [], []
    for index in range(len(times)):
        a = (np.diag([.8, 1.2, 1.6, 2.])
             + .12*(rng.normal(size=(4, 4))+1j*rng.normal(size=(4, 4))))
        change = np.eye(4, dtype=complex)
        if index and gauge == "phases":
            change = np.diag(np.exp(1j*rng.uniform(-np.pi, np.pi, 4)))
        elif index and gauge == "degenerate":
            # Only an exactly degenerate block permits this gauge rotation
            # while the frame Hamiltonian remains diagonal with the same E.
            change[1:3, 1:3] = _unitary(rng, 2)
        physical = eigenvectors @ change
        frames.append(a)
        physical_eigenvectors.append(physical)
        rows.append(np.linalg.solve(a, physical).T)
    frames = np.asarray(frames)
    metrics = np.asarray([a.conj().T @ a for a in frames])
    cross = np.asarray([a.conj().T @ b for a, b in zip(frames[:-1], frames[1:], strict=True)])
    return dict(times=times, energies=np.tile(energies, (len(times), 1)),
                coefficients=np.asarray(rows), metrics=metrics, cross_metrics=cross,
                embedding=frames, physical_eigenvectors=np.asarray(physical_eigenvectors),
                physical_h=physical_h)


def _project(data, *, retained_bands=None, **options):
    config = dict(retained_bands=np.arange(data["energies"].shape[1]) if retained_bands is None else retained_bands,
                  energy_unit="hartree", time_unit="atomic", ao_basis_id="synthetic-row-AO-order-v1",
                  basis_id="synthetic-adiabatic-order-v1", source_identity="independent-AO-embedding-v1")
    config.update(options)
    return project_ao_path(*(data[name] for name in ("times", "energies", "coefficients", "metrics", "cross_metrics")),
                           **config)


def _initial(n):
    c = np.arange(1, n+1)+.3j*np.arange(n, 0, -1)
    return c/np.linalg.norm(c)


@pytest.mark.parametrize("gauge", ["none", "phases", "degenerate"])
def test_changing_complex_nonorthogonal_aos_preserve_exact_physical_static_hamiltonian(gauge):
    data = _embedding(gauge=gauge)
    projection = _project(data)
    path = projection.path
    assert path.force_support is False
    assert path.basis_kind == "instantaneous_orthonormal"
    physical_frames = data["physical_eigenvectors"]
    overlaps = np.asarray([a.conj().T @ b for a, b in zip(physical_frames[:-1], physical_frames[1:], strict=True)])
    np.testing.assert_allclose(path.overlaps, overlaps, atol=2e-15, rtol=3e-15)
    for interval in range(len(data["times"])-1):
        np.testing.assert_allclose(path.transport_at(interval).matrix, overlaps[interval].conj().T, atol=2e-15)
    initial = _initial(4)
    physical0 = physical_frames[0] @ initial
    simulation = RecordedCPA(path, execution=Execution(chunk_size=1))
    state = simulation.initialize(initial)
    # Every endpoint is checked in the independently known physical Hilbert space.
    for frame in range(1, len(data["times"])):
        state = simulation.run(state, 1).final_state
        physical = data["embedding"][frame] @ data["coefficients"][frame].T @ np.asarray(state.electronic)
        expected = expm(-1j*(data["times"][frame]-data["times"][0])*data["physical_h"]) @ physical0
        np.testing.assert_allclose(physical, expected, atol=5e-15, rtol=5e-15)
        np.testing.assert_allclose(np.linalg.norm(state.electronic), 1., atol=4e-15)


def test_explicit_per_frame_retained_indices_keep_unsorted_state_order_and_crossing_labels():
    data = _embedding(gauge="phases")
    permutations = np.array([[2, 0, 3, 1], [1, 3, 0, 2], [3, 2, 1, 0], [0, 3, 2, 1]])
    logical_order = np.array([[0, 2], [2, 0], [0, 2], [2, 0]])
    selection = np.array([np.argsort(perm)[order] for perm, order in zip(permutations, logical_order, strict=True)])
    changed = {**data,
               "energies": np.asarray([e[perm] for e, perm in zip(data["energies"], permutations, strict=True)]),
               "coefficients": np.asarray([c[perm] for c, perm in zip(data["coefficients"], permutations, strict=True)])}
    projection = _project(changed, retained_bands=selection)
    selected_rows = np.asarray([c[index] for c, index in zip(changed["coefficients"], selection, strict=True)])
    expected_energies = np.asarray([e[index] for e, index in zip(changed["energies"], selection, strict=True)])
    np.testing.assert_array_equal(projection.path.energies, expected_energies)
    assert expected_energies[1, 0] > expected_energies[1, 1]  # No automatic sort.
    initial = _initial(2)
    simulation = RecordedCPA(projection.path)
    result = simulation.run(simulation.initialize(initial), 3)
    physical0 = data["embedding"][0] @ selected_rows[0].T @ initial
    final = data["embedding"][-1] @ selected_rows[-1].T @ np.asarray(result.final_state.electronic)
    exact = expm(-1j*(data["times"][-1]-data["times"][0])*data["physical_h"]) @ physical0
    np.testing.assert_allclose(final, exact, atol=6e-15, rtol=5e-15)


def test_explicit_ev_fs_conversion_matches_atomic_phase_evolution():
    data = _embedding(gauge="phases")
    atomic = _project(data)
    converted = _project({**data, "energies": data["energies"]*HARTREE_EV,
                          "times": data["times"]*ATOMIC_TIME_FS}, energy_unit="eV", time_unit="fs")
    np.testing.assert_allclose(converted.path.energies, atomic.path.energies, atol=2e-16)
    np.testing.assert_allclose(converted.path.times, atomic.path.times, atol=2e-16)
    assert converted.evidence.energy_to_hartree == 1/HARTREE_EV
    assert converted.evidence.time_to_atomic == 1/ATOMIC_TIME_FS
    assert converted.evidence.coefficient_convention == "row_kets"
    assert converted.evidence.cross_metric_convention == "bra_previous_ket_next"
    initial = _initial(4)
    direct, scaled = RecordedCPA(atomic.path), RecordedCPA(converted.path)
    a = direct.run(direct.initialize(initial), 3)
    b = scaled.run(scaled.initialize(initial), 3)
    np.testing.assert_allclose(a.final_state.electronic, b.final_state.electronic, atol=4e-15)


def test_full_cross_metric_cannot_hide_expansion_in_discarded_ao_direction():
    data = dict(times=np.array([0., .1]), energies=np.zeros((2, 3)),
                coefficients=np.repeat(np.eye(3, dtype=complex)[None], 2, axis=0),
                metrics=np.repeat(np.eye(3)[None], 2, axis=0),
                cross_metrics=np.diag([1., 1., 1.4])[None])
    # The retained O would be I_2. The supplied full AO union is nevertheless
    # impossible: its Gram matrix has eigenvalue -0.4 in a discarded direction.
    with pytest.raises(ValueError, match="cross|contraction|Gram"):
        _project(data, retained_bands=[0, 1])


def _moving_subspace(angle, *, discarded_band=False):
    embedding = []
    for index in range(3):
        a = np.zeros((3, 2))
        a[:, 0] = [np.cos(index*angle), 0., np.sin(index*angle)]
        a[:, 1] = [0., 1., 0.]
        embedding.append(a)
    if discarded_band:
        # A fixed, complete, orthonormal AO basis with one excluded band gives
        # the same retained projection loss. This separates band truncation
        # from the changing incomplete AO span in the other fixture.
        frames = []
        for index, retained in enumerate(embedding):
            complement = np.array([-np.sin(index*angle), 0., np.cos(index*angle)])
            frames.append(np.column_stack((retained, complement)))
        return dict(times=np.array([0., .1, .2]), energies=np.zeros((3, 3)),
                    coefficients=np.asarray(frames).transpose(0, 2, 1),
                    metrics=np.repeat(np.eye(3)[None], 3, axis=0),
                    cross_metrics=np.repeat(np.eye(3)[None], 2, axis=0))
    return dict(times=np.array([0., .1, .2]), energies=np.zeros((3, 2)),
                coefficients=np.repeat(np.eye(2, dtype=complex)[None], 3, axis=0),
                metrics=np.array([a.T @ a for a in embedding]),
                cross_metrics=np.array([a.T @ b for a, b in zip(embedding[:-1], embedding[1:], strict=True)]))


@pytest.mark.parametrize("discarded_band", [False, True])
def test_true_truncation_preserves_raw_norm_loss_and_polar_keeps_raw_diagnostics(discarded_band):
    angle = .31
    data = _moving_subspace(angle, discarded_band=discarded_band)
    raw = _project(data, retained_bands=[0, 1])
    polar = _project(data, retained_bands=[0, 1], transport_mode="polar")
    loss = np.sin(angle)**2
    for projection in (raw, polar):
        np.testing.assert_allclose(projection.diagnostics.maximum_subspace_loss, loss, atol=8e-16)
        np.testing.assert_allclose(projection.diagnostics.cross_metric_max_singular_values, 1., atol=5e-16)
        np.testing.assert_allclose(projection.diagnostics.overlap_singular_values,
                                   np.tile([1., np.cos(angle)], (2, 1)), atol=5e-16)
        assert projection.diagnostics.rank_deficient == (False, False)
        for interval in range(2):
            transport = projection.path.transport_at(interval)
            np.testing.assert_allclose(transport.diagnostics.maximum_norm_loss, loss, atol=8e-16)
            np.testing.assert_allclose(transport.raw_overlap, np.diag([np.cos(angle), 1.]), atol=5e-16)
        strict = RecordedCPA(projection.path)
        with pytest.raises(ValueError, match="subspace loss"):
            strict.run(strict.initialize([1., 0.]), 1)
    simulation = RecordedCPA(raw.path, max_subspace_loss=loss+1e-12)
    result = simulation.run(simulation.initialize([1., 0.]), 2)
    np.testing.assert_allclose(result.final_state.electronic, [np.cos(angle)**2, 0.], atol=6e-16)
    np.testing.assert_allclose(result.observables["norm"], np.cos(angle)**(2*np.arange(3)), atol=1e-15)
    projected = RecordedCPA(polar.path, max_subspace_loss=loss+1e-12)
    projected_result = projected.run(projected.initialize([1., 0.]), 2)
    np.testing.assert_allclose(projected_result.observables["norm"], 1., atol=1e-15)
    assert bool(polar.path.transport_at(0).projection_applied)


def test_singular_cross_subspace_has_no_accepted_polar_transport_even_if_loss_is_permitted():
    data = _moving_subspace(np.pi/2)
    # An implementation may reject at ingestion or at RecordedCPA preflight;
    # neither is allowed to make an arbitrary polar continuation look valid.
    with pytest.raises(ValueError, match="singular|rank|transport|polar"):
        projection = _project(data, transport_mode="polar")
        run = RecordedCPA(projection.path, max_subspace_loss=None)
        run.run(run.initialize([1., 0.]), 1)


def test_projection_snapshots_raw_arrays_and_evidence_is_bound_to_the_final_path(tmp_path):
    data = _embedding(gauge="phases")
    retained = np.arange(4)
    projection = _project(data, retained_bands=retained)
    energies, overlaps, times = [np.array(getattr(projection.path, name)) for name in ("energies", "overlaps", "times")]
    evidence = projection.evidence
    for name in ("times", "energies", "coefficients", "metrics", "cross_metrics"):
        data[name][...] = 7.
    retained[:] = 0
    for name, expected in (("energies", energies), ("overlaps", overlaps), ("times", times)):
        np.testing.assert_array_equal(getattr(projection.path, name), expected)
    assert projection.evidence == evidence
    with pytest.raises(ValueError, match="digest|evidence|projection|identity"):
        replace(projection.path, energies=energies+.01)

    simulation = RecordedCPA(projection.path, execution=Execution(chunk_size=1))
    initial = simulation.initialize(_initial(4))
    full = simulation.run(initial, 3)
    prefix = simulation.run(initial, 1)
    checkpoint = tmp_path/"ao_path.h5"
    simulation.save_checkpoint(checkpoint, prefix.final_state)
    loaded = simulation.load_checkpoint(checkpoint)
    for got, expected in zip(jax.tree.leaves(loaded), jax.tree.leaves(prefix.final_state), strict=True):
        np.testing.assert_array_equal(got, expected)
    resumed = simulation.run(loaded, 2)
    for got, expected in zip(jax.tree.leaves(resumed.final_state), jax.tree.leaves(full.final_state), strict=True):
        np.testing.assert_array_equal(got, expected)
    np.testing.assert_array_equal(np.r_[prefix.times, resumed.times[1:]], full.times)
    for key in full.observables:
        np.testing.assert_array_equal(np.concatenate((prefix.observables[key], resumed.observables[key][1:])), full.observables[key])
    # Numerically identical projected path, different caller-owned raw source identity.
    different = _project(_embedding(gauge="phases"), source_identity="different-raw-AO-input-v2")
    np.testing.assert_array_equal(different.path.overlaps, projection.path.overlaps)
    with pytest.raises(ValueError, match="path|manifest|identity"):
        RecordedCPA(different.path, execution=Execution(chunk_size=1)).load_checkpoint(checkpoint)
