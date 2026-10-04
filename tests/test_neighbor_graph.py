"""Independent image, motion, and force checks for candidate graph lifecycle."""

from dataclasses import FrozenInstanceError
from itertools import product
import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pyeph.models.local import AtomCenterMap, LocalBlockModel, LocalCoefficients
from pyeph.models.neighbors import (
    NeighborCapacityError, NeighborCoverageError, NeighborGraph,
    unwrap_atoms, unwrap_positions, wrap_atoms, wrap_positions,
)


def snapshot(q, **kwargs):
    centers = AtomCenterMap(tuple(range(len(q))), (1.0,) * len(q), len(q))
    arguments = dict(norbitals=1, switch_on=0.8, cutoff=1.5, skin=0.4)
    return NeighborGraph(centers, q, **(arguments | kwargs))


def brute_images(q, cell, radius, extent):
    """Fixed oversized cube oracle, independent of reciprocal search bounds."""
    images = np.asarray(list(product(range(-extent, extent + 1), repeat=3)))
    translations = images @ cell
    answer = set()
    for a in range(len(q)):
        for b in range(a, len(q)):
            norms = np.sqrt(np.sum((q[b] + translations - q[a])**2, axis=1))
            for image in images[norms <= radius]:
                values = tuple(map(int, image))
                if a != b or values > (0, 0, 0):
                    answer.add((a, b, *values))
    return answer


@pytest.mark.parametrize("cell,extent", [
    (np.diag([1.2, 1.6, 1.8]), 4),
    (np.array([[2., 0, 0], [.9, 1.7, 0], [.3, .4, 2.1]]), 4),
    (np.array([[1., 0, 0], [5.3, .8, 0], [.2, .3, 1.2]]), 20),
])
def test_periodic_candidates_match_independent_full_image_cube(cell, extent):
    q = np.array([[.13, .21, .35], [.62, .57, .74]]) @ cell
    candidates = snapshot(q, cell=cell)
    expected = brute_images(q, cell, 1.9, extent)
    assert set(candidates.graph.edges) == expected
    if cell[0, 0] < 1.9:
        assert any(a == b for a, b, *_ in expected)
    assert len(candidates.graph.edges) == len(expected)
    assert candidates.require_coverage(q, exhaustive=True).covered


def test_finite_candidates_include_buffer_and_boundary_without_padding():
    q = np.array([[0., 0, 0], [1.5, 0, 0], [3.4, 0, 0], [20., 0, 0]])
    candidates = snapshot(q, capacity=20)
    assert candidates.graph.edges == ((0, 1, 0, 0, 0), (1, 2, 0, 0, 0))
    assert candidates.capacity == 20
    assert candidates.graph.nstates == 4
    assert snapshot([[0., 0, 0]], capacity=0).graph.edges == ()


def test_weighted_atom_centers_define_support_and_keep_zero_weight_atoms():
    mapping = AtomCenterMap((0, 0, 0, 1), (.25, .75, 0., 1.), 2)
    q = [[-1., 0, 0], [1., 0, 0], [10., 8, 9], [2., 0, 0]]
    candidates = NeighborGraph(mapping, q, 1, .5, 1.6, .2)
    assert candidates.reference_centers == ((.5, 0., 0.), (2., 0., 0.))
    assert len(candidates.graph.edges) == 1
    moved = np.array(q)
    moved[2] += 100
    assert candidates.check(moved).maximum_displacement == 0


def test_skin_certificate_and_omitted_edge_diagnostic_are_distinct():
    q = np.array([[0., 0, 0], [2., 0, 0]])
    candidates = snapshot(q)
    assert candidates.graph.edges == ()
    moved = q.copy()
    moved[1, 0] = 1.81
    report = candidates.require_coverage(moved)
    assert report.within_skin and report.covered
    moved[1, 0] = 1.7
    assert candidates.check(moved).covered is None
    report = candidates.check(moved, exhaustive=True)
    assert report.covered and report.rebuild_required and report.missing_edges == ()
    moved[1, 0] = 1.4
    with pytest.raises(NeighborCoverageError) as caught:
        candidates.require_coverage(moved, exhaustive=True)
    assert caught.value.report.missing_edges == ((0, 1, 0, 0, 0),)
    assert caught.value.report.covered is False
    rebuilt = candidates.rebuild(moved)
    assert rebuilt.graph.edges == ((0, 1, 0, 0, 0),)
    assert rebuilt.generation == 1 and rebuilt.parent_identity == candidates.identity
    assert rebuilt.require_coverage(moved).covered
    assert candidates.graph.edges == ()


def test_skin_bound_includes_motion_of_both_endpoints():
    q = np.array([[0., 0, 0], [1.95, 0, 0]])
    candidates = snapshot(q)
    moved = q + [[.19, 0, 0], [-.19, 0, 0]]
    assert candidates.require_coverage(moved, exhaustive=True).covered
    moved = q + [[.25, 0, 0], [-.25, 0, 0]]
    assert candidates.check(moved, exhaustive=True).missing_edges == ((0, 1, 0, 0, 0),)


def test_random_motion_within_skin_keeps_every_physical_periodic_image():
    rng = np.random.default_rng(347)
    cell = np.array([[2., 0, 0], [.8, 1.6, 0], [.3, .1, 2.3]])
    q = rng.uniform(size=(3, 3)) @ cell
    candidates = snapshot(q, cell=cell)
    for _ in range(8):
        shift = rng.normal(size=q.shape)
        shift *= .199 / np.linalg.norm(shift, axis=1)[:, None]
        moved = q + shift
        assert candidates.require_coverage(moved).covered
        assert brute_images(moved, cell, 1.5, 4) <= set(candidates.graph.edges)


def test_capacity_failure_and_search_budget_leave_existing_generation_unchanged():
    q = [[0., 0, 0], [2., 0, 0]]
    candidates = snapshot(q, capacity=0)
    identity = candidates.identity
    with pytest.raises(NeighborCapacityError) as caught:
        candidates.rebuild([[0., 0, 0], [1., 0, 0]])
    assert caught.value.required == 1 and caught.value.capacity == 0
    assert candidates.identity == identity
    assert len(candidates.rebuild([[0., 0, 0], [1., 0, 0]], capacity=2).graph.edges) == 1
    with pytest.raises(ValueError, match="max_image_checks"):
        snapshot(q, cell=np.eye(3), max_image_checks=5)


def test_wrapping_uses_explicit_images_and_keeps_fragments_whole():
    mapping = AtomCenterMap((0, 0, 1, 1), (.5, .5, .5, .5), 2)
    cell = np.array([[3., 0, 0], [.4, 3., 0], [.2, .1, 3.]])
    q = np.array([[-.2, .5, .3], [.4, .5, .3], [3.1, .4, .2], [3.7, .6, .4]])
    q += np.array([[0, 0, 0], [0, 0, 0], [2, -3, 1], [2, -3, 1]]) @ cell
    candidates = NeighborGraph(mapping, q, 1, .3, 1.5, .5, cell)
    wrapped, images = wrap_atoms(q, mapping, cell)
    np.testing.assert_allclose(unwrap_atoms(wrapped, images, mapping, cell), q, atol=2e-15)
    np.testing.assert_allclose(wrapped[1] - wrapped[0], q[1] - q[0], atol=2e-15)
    np.testing.assert_allclose(wrapped[3] - wrapped[2], q[3] - q[2], atol=2e-15)
    assert not candidates.check(wrapped).within_skin
    rewrapped = candidates.rewrapped(-images)
    assert rewrapped.require_coverage(wrapped, exhaustive=True).covered
    assert rewrapped.graph == candidates.graph.rewrapped(-images)
    assert rewrapped.parent_identity == candidates.identity


def test_unwrapping_does_not_guess_large_trajectory_winding():
    cell = np.diag([2., 3., 4.])
    q = np.array([[-8.2, 16.3, 25.4], [2.1, -3.2, .1]])
    wrapped, images = wrap_positions(q, cell)
    np.testing.assert_array_equal(images, [[-5, 5, 6], [1, -2, 0]])
    np.testing.assert_allclose(unwrap_positions(wrapped, images, cell), q, atol=2e-15)
    assert np.all((wrapped @ np.linalg.inv(cell) >= 0)
                  & (wrapped @ np.linalg.inv(cell) < 1))
    with pytest.raises(ValueError, match="integer shape"):
        unwrap_positions(wrapped, images.astype(float), cell)
    with pytest.raises(ValueError, match="integer image resolution"):
        wrap_positions([[2.**54, 0, 0]], cell)


@pytest.mark.parametrize("cutoff", [1., 1.8])
def test_rewrapping_rejects_large_shift_that_erases_subcell_geometry(cutoff):
    candidates = snapshot([[0., 0, 0], [1.5, 0, 0]], cell=np.eye(3)*4,
                          switch_on=.5, cutoff=cutoff, skin=.1)
    # Includes a zero-edge graph: absence of an edge is not evidence that a
    # coordinate transformation may silently round away a physical displacement.
    with pytest.raises(ValueError, match="precision"):
        candidates.rewrapped([[2**50, 0, 0], [2**50, 0, 0]])
    with pytest.raises(ValueError, match="precision"):
        unwrap_positions([[0., 0, 0], [1.5, 0, 0]],
                         [[2**50, 0, 0], [2**50, 0, 0]], np.eye(3)*4)


def test_translation_rejects_lost_small_shift_even_when_roundtrip_looks_exact():
    position = np.array([[1e16, 0, 0]])
    assert np.array_equal((position + [1., 0, 0]) - [1., 0, 0], position)
    with pytest.raises(ValueError, match="precision"):
        unwrap_positions(position, [[1, 0, 0]], np.eye(3))


def test_large_finite_distance_cannot_disappear_through_squared_norm_overflow():
    q = [[0., 0, 0], [1e200, 0, 0]]
    candidates = snapshot(q, switch_on=.5e200, cutoff=1.5e200, skin=.1e200)
    assert candidates.graph.edges == ((0, 1, 0, 0, 0),)
    assert candidates.require_coverage(q, exhaustive=True).covered


def coefficients(params, q, geometry):
    return LocalCoefficients(jnp.zeros((2, 1, 1)),
                             jnp.exp(-geometry.distances)[:, None, None])


def energy_and_force(candidates, q):
    model = LocalBlockModel(candidates.graph, candidates.centers, coefficients)
    vector = jnp.ones(2) / jnp.sqrt(2.)
    def energy(coordinates):
        return jnp.vdot(vector, model.apply(None, coordinates, vector)).real
    return float(energy(jnp.asarray(q))), -np.asarray(jax.grad(energy)(jnp.asarray(q)))


def independent_energy(r):
    if r <= 1.:
        support = 1.
    elif r >= 2.:
        support = 0.
    else:
        x = r - 1.
        support = 1 - 10*x**3 + 15*x**4 - 6*x**5
    return np.exp(-r) * support


def test_full_cutoff_force_matches_independent_finite_difference_and_is_continuous():
    candidates = snapshot([[0., 0, 0], [1.8, 0, 0]], switch_on=1., cutoff=2., skin=.8)
    for radius in (1.2, 1.8, 1.99, 2., 2.01):
        q = [[0., 0, 0], [radius, 0, 0]]
        energy, force = energy_and_force(candidates, q)
        expected_force = -(independent_energy(radius + 1e-5)
                           - independent_energy(radius - 1e-5)) / 2e-5
        assert energy == pytest.approx(independent_energy(radius), abs=2e-16)
        assert force[1, 0] == pytest.approx(expected_force, abs=2e-10)
        np.testing.assert_allclose(force.sum(axis=0), 0, atol=1e-15)
    below = energy_and_force(candidates, [[0., 0, 0], [2 - 1e-4, 0, 0]])
    assert abs(below[0]) < 2e-12 and np.max(np.abs(below[1])) < 5e-8
    _, at = energy_and_force(candidates, [[0., 0, 0], [2., 0, 0]])
    np.testing.assert_array_equal(at, 0.)


def test_safe_candidate_insertion_and_removal_do_not_change_energy_or_force():
    candidates = snapshot([[0., 0, 0], [2.5, 0, 0]], switch_on=1., cutoff=2., skin=.2)
    q = [[0., 0, 0], [2.1, 0, 0]]
    inserted = candidates.rebuild(q)
    assert candidates.graph.edges == () and len(inserted.graph.edges) == 1
    for old, new in zip(energy_and_force(candidates, q), energy_and_force(inserted, q)):
        np.testing.assert_array_equal(old, new)
    removed = inserted.rebuild([[0., 0, 0], [2.5, 0, 0]])
    assert removed.graph.edges == ()
    for old, new in zip(energy_and_force(inserted, removed.reference_coordinates),
                        energy_and_force(removed, removed.reference_coordinates)):
        np.testing.assert_array_equal(old, new)


def test_snapshot_and_metadata_are_immutable_and_identify_input_changes():
    q = np.array([[0., 0, 0], [1., 0, 0]])
    candidates = snapshot(q)
    identity = candidates.identity
    q[1] += 10
    metadata = candidates.metadata()
    metadata["reference_coordinates"][0][0] = 17
    assert candidates.identity == identity
    assert json.loads(json.dumps(candidates.metadata()))["edges"] == [[0, 1, 0, 0, 0]]
    assert snapshot(candidates.reference_coordinates, skin=.5).identity != identity
    with pytest.raises(FrozenInstanceError):
        candidates.skin = 3


@pytest.mark.parametrize("kwargs", [
    {"skin": -1}, {"skin": np.inf}, {"skin": True}, {"capacity": -1},
    {"capacity": 2.0}, {"capacity": True}, {"max_image_checks": 0},
    {"max_image_checks": True}, {"generation": -1}, {"parent_identity": ""},
    {"reference_coordinates": [[np.nan, 0, 0], [0, 0, 0]]},
    {"reference_coordinates": [[1j, 0, 0], [0, 0, 0]]},
])
def test_invalid_lifecycle_configuration_fails_before_search(kwargs):
    mapping = AtomCenterMap((0, 1), (1., 1.), 2)
    arguments = dict(centers=mapping, reference_coordinates=[[0., 0, 0], [1., 0, 0]],
                     norbitals=1, switch_on=.8, cutoff=1.5, skin=.4)
    with pytest.raises((ValueError, TypeError)):
        NeighborGraph(**(arguments | kwargs))
