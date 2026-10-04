"""Preprocessing partition ownership, bounded mode storage and exact q reuse."""

from itertools import islice
import logging
from types import SimpleNamespace

import numpy as np
import pytest

from pyeph.post_qe2pert import phonon_disp
from pyeph.post_qe2pert.eph_mat_reciprocal import CalcEphMatReciprocal, _local_pair_indices
from pyeph.post_qe2pert._support import bohr_to_ang, ryd_to_ev, ryd_to_mev


@pytest.mark.parametrize("nk,nq", [(0, 0), (0, 4), (3, 0), (3, 4), (7, 2), (2, 7), (4, 1)])
@pytest.mark.parametrize("ranks", [1, 2, 5, 19])
def test_grouping_preserves_every_flat_partition_including_empty_ranks(nk, nq, ranks):
    original = [(k, q) for k in range(nk) for q in range(nq)]
    all_pairs = []
    for rank, expected in enumerate(np.array_split(np.arange(len(original)), ranks)):
        start = len(original)//ranks*rank+min(rank, len(original) % ranks)
        actual = list(_local_pair_indices(nk, nq, start, start+len(expected)))
        assert sorted(actual) == [original[index] for index in expected]
        assert [q for _, q in actual] == sorted(q for _, q in actual)
        all_pairs.extend(actual)
    assert sorted(all_pairs) == original


def test_pair_enumeration_does_not_materialize_the_global_grid():
    pairs = _local_pair_indices(10**12, 3, 0, 3*10**12)
    assert list(islice(pairs, 4)) == [(0, 0), (1, 0), (2, 0), (3, 0)]


def synthetic_phonons():
    phonons = object.__new__(phonon_disp.PhononDispersion)
    phonons.nat = 1
    phonons.logger = logging.getLogger("synthetic-phonons")
    calls = []

    def solve(_, q, mass_weight=True):
        calls.append((tuple(q), mass_weight))
        phase = np.exp(1j*q[0])
        return np.array([-.1, .2, .2])+q[0], np.eye(3, dtype=complex)*phase

    phonons.solve_phonon_modes = solve
    return phonons, calls


@pytest.mark.parametrize("nq", [0, 1, 5])
def test_frequency_only_serial_result_preserves_values_and_avoids_mode_stack(monkeypatch, nq):
    monkeypatch.setenv("USE_MPI", "false")
    phonons, calls = synthetic_phonons()
    points = np.arange(nq*3, dtype=float).reshape(nq, 3)/10
    expected, modes = phonons.compute_phonon_dispersion(points, None, mass_weight=False)
    assert modes.shape == (nq, 3, 3)
    calls.clear()
    allocated = []
    zeros = np.zeros

    def tracked_zeros(shape, *args, **kwargs):
        allocated.append(shape)
        return zeros(shape, *args, **kwargs)

    monkeypatch.setattr(phonon_disp.np, "zeros", tracked_zeros)
    actual, omitted = phonons.compute_phonon_dispersion(
        points, None, mass_weight=False, return_modes=False)
    np.testing.assert_array_equal(actual, expected)
    assert omitted is None and len(calls) == nq
    assert all(not mass_weight for _, mass_weight in calls)
    assert (nq, 3, 3) not in allocated


@pytest.mark.parametrize("rank", [0, 3])
@pytest.mark.parametrize("return_modes", [False, True])
def test_mode_gather_is_skipped_on_every_rank_including_empty_rank(monkeypatch, rank, return_modes):
    phonons, calls = synthetic_phonons()
    gathered = []

    def gather(data, *args, **kwargs):
        gathered.append((data.shape, kwargs.get("is_complex", False)))
        return data

    phonons._gather_mpi_results = gather
    monkeypatch.setattr(phonon_disp, "get_mpi_info", lambda: dict(
        has_mpi=True, rank=rank, size=4, comm=object()))
    frequencies, modes = phonons.compute_phonon_dispersion(
        np.zeros((1, 3)), None, return_modes=return_modes)
    count = int(rank == 0)
    assert frequencies.shape == (count, 3)
    assert len(calls) == count
    assert gathered == ([( (count, 3), False), ((count, 3, 3), True)]
                         if return_modes else [((count, 3), False)])
    assert (modes is None) == (not return_modes)


def test_epc_reuses_one_mode_solve_per_local_q_and_keeps_physical_output(monkeypatch):
    monkeypatch.setenv("USE_MPI", "false")
    calculator = object.__new__(CalcEphMatReciprocal)
    calculator.num_wann, calculator.nat = 1, 1
    calculator.mass = np.array([2.])
    calculator.verbose = False
    calculator.force_constants = None
    calculator.logger = logging.getLogger("synthetic-epc")
    phonons, calls = synthetic_phonons()
    calculator.phonon_calc = phonons
    calculator.electron_calc = SimpleNamespace(
        solve_eigenvalue_vector=lambda k: (np.array([k[0]]), np.ones((1, 1))))
    calculator.eph_fourier_el_para = lambda k: k.copy()
    calculator.eph_fourier_elph = lambda q, k: np.array(
        [1+.3j+k[0], 2-.2j+q[0], 4+.1j+k[0]-q[0]]).reshape(1, 1, 3)
    calculator.eph_transform = lambda q, modes, uk, ukq, epc: epc @ modes
    kpoints = np.array([[0., 0., 0.], [.11, .12, 0.], [-.12, .03, 0.], [.24, 0., 0.]])
    qpoints = np.array([[0., 0., 0.], [.07, .08, 0.]])
    actual = calculator.calc_ephmat(kpoints, qpoints, phfreq_cutoff=.05)
    # One distributed frequency prepass and one grouped local solve per q.
    assert len(calls) == 2*len(qpoints)
    for iq, q in enumerate(qpoints):
        frequencies = np.array([-.1, .2, .2])+q[0]
        np.testing.assert_array_equal(actual["phonon_frequencies"][iq], frequencies)
        for ik, k in enumerate(kpoints):
            magnitudes = abs(np.array([1+.3j+k[0], 2-.2j+q[0], 4+.1j+k[0]-q[0]]))**2
            # The last two modes are degenerate and share their squared average.
            magnitudes[1:] = magnitudes[1:].mean()
            expected_dp = np.sqrt(2*magnitudes)*ryd_to_ev/bohr_to_ang
            expected_g = np.sqrt(np.where(frequencies > .05,
                .5*magnitudes/np.where(frequencies > .05, frequencies, 1.), 0.))*ryd_to_mev
            np.testing.assert_allclose(actual["deformation_potential"][:, iq, ik],
                                       expected_dp, atol=2e-13, rtol=2e-14)
            np.testing.assert_allclose(actual["eph_matrix_elements"][:, iq, ik],
                                       expected_g, atol=2e-11, rtol=2e-14)
