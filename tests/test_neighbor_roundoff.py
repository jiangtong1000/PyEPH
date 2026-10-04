"""Exact float-input geometry oracles for host candidate boundaries."""
from fractions import Fraction
from itertools import product

import numpy as np
import pytest

from pyeph.models.local import AtomCenterMap
from pyeph.models.neighbors import (
    NeighborGraph, _HostGeometry, _fraction, _integer_geometry,
)


def exact(value):
    # Independent standard-library oracle under the test process's normal mode.
    return Fraction(float(value))


def squared(a, b):
    return sum((exact(y)-exact(x))**2 for x, y in zip(a, b))


def points(rows):
    return np.asarray([[float.fromhex(x) for x in row] for row in rows])


@pytest.mark.parametrize('reference,moved', [
    ([['0x1.02aec5ada8022p-1', '0x1.889938aa781e2p-1', '0x1.7a6acf2701cd2p-1'],
      ['-0x1.4945e8cebea1ap-1', '-0x1.7c4b848960cd8p-1', '0x1.b3b7f816b8ac9p-1']],
     [['0x1.8995d2698b402p-2', '0x1.373dc66f1dea6p-1', '0x1.8072ee68a942fp-1'],
      ['-0x1.0b620c55dc3f9p-1', '-0x1.2af0124e0699cp-1', '0x1.adafd8d51136cp-1']]),
    ([['0x1.f44c17ac0f1b8p-2', '0x1.6d1d326f709f2p-1', '-0x1.9050eecb51e1cp-2'],
      ['0x1.1bf2bdd9f508ap+1', '0x1.3936e72e6c618p-1', '0x1.8de220308a778p-2']],
     [['0x1.575fd3ca6dee4p-1', '0x1.67a6a3c6e9703p-1', '-0x1.3c4b8ee6b3f35p-2'],
      ['0x1.04a44bdcdb708p+1', '0x1.3ead75d6f3907p-1', '0x1.39dcc04bec891p-2']]),
])
def test_roundoff_cannot_omit_required_candidate(reference, moved):
    q, q_next = points(reference), points(moved)
    cutoff, skin = 1.5, .4
    assert squared(q[0], q[1]) < (exact(cutoff)+exact(skin))**2
    assert squared(q_next[0], q_next[1]) < exact(cutoff)**2
    assert all(squared(a, b) <= (exact(skin)/2)**2 for a, b in zip(q, q_next))
    candidates = NeighborGraph(AtomCenterMap((0, 1), (1., 1.), 2), q, 1, .8, cutoff, skin)
    assert candidates.graph.edges == ((0, 1, 0, 0, 0),)
    assert candidates.require_coverage(q_next).covered
    assert candidates.require_coverage(q_next, exhaustive=True).covered


def test_rounded_displacement_at_skin_boundary_cannot_accept_exact_excess():
    q = points([['0x1.e583042b9ec82p-1', '-0x1.4cfe194cbb720p-1', '0x1.6d418a87c8c9cp-2']])
    moved = points([['0x1.ad0d019bf3021p-1', '-0x1.3465dff69dc7cp-1', '0x1.9344b9c957e44p-3']])
    candidates = NeighborGraph(AtomCenterMap((0,), (1.,), 1), q, 1, .8, 1.5, .4)
    # The original platform rounded this norm onto the skin boundary. The
    # regression depends on exact geometry, not a particular libm rounding.
    assert squared(q[0], moved[0]) > (exact(.4)/2)**2
    assert not candidates.check(moved).within_skin


def test_zero_skin_exact_center_preserving_motion_and_equal_boundary():
    centers = AtomCenterMap((0, 0, 1), (.5, .5, 1.), 2)
    q = np.array([[0., 0, 0], [2., 0, 0], [3., 0, 0]])
    g = NeighborGraph(centers, q, 1, 1., 2., 0.)
    assert g.graph.edges == ((0, 1, 0, 0, 0),)
    assert g.require_coverage(q, exhaustive=True).covered
    moved = q + [[.5, 0, 0], [-.5, 0, 0], [0, 0, 0]]
    assert g.require_coverage(moved, exhaustive=True).covered
    moved[2, 0] = np.nextafter(moved[2, 0], np.inf)
    assert not g.check(moved).within_skin
    half = NeighborGraph(centers, q, 1, 1., 2., .5)
    assert half.require_coverage(q + [.25, 0, 0]).covered


def test_rewrap_rejects_new_required_candidate_instead_of_changing_membership():
    q = np.array([[.3, 0, 0], [2.2, 0, 0]])
    centers = AtomCenterMap((0, 1), (1., 1.), 2)
    g = NeighborGraph(centers, q, 1, .8, 1.5, .4, np.diag([10., 10., 10.]))
    assert not g.graph.edges
    shifted = q + [10., 0, 0]
    radius = exact(1.5) + exact(.4)
    assert squared(q[0], q[1]) > radius**2
    assert squared(shifted[0], shifted[1]) < radius**2
    before = g.identity
    with pytest.raises(ValueError, match='rewrapping loses candidate coverage'):
        g.rewrapped(np.array([[1, 0, 0], [1, 0, 0]]))
    assert g.identity == before and not g.graph.edges
    assert (0, 1, 0, 0, 0) in g.rebuild(shifted).graph.edges


@pytest.mark.parametrize('value', [0., -0., np.nextafter(0., 1.), -np.nextafter(0., 1.),
                                  np.finfo(float).tiny, 1e-200, 1e200, np.finfo(float).max])
def test_exact_decoder_matches_float_bits(value):
    assert _fraction(value) == exact(value)


@pytest.mark.parametrize('scale', [np.nextafter(0., 1.), 1e-200, 1e-160, 1e160, 1e200])
def test_norm_underflow_overflow_uses_exact_integer_geometry(scale):
    q = np.array([[0., 0., 0.], [scale, scale, 0.]])
    centers = AtomCenterMap((0, 1), (1., 1.), 2)
    graph = NeighborGraph(centers, q, 1, 0., scale, 0.)
    assert squared(q[0], q[1]) > exact(scale)**2
    assert graph.graph.edges == ()
    assert graph.require_coverage(q, exhaustive=True).covered
    one = NeighborGraph(AtomCenterMap((0,), (1.,), 1), [[0., 0., 0.]], 1, 0., 1., scale)
    assert not one.check(np.array([[scale, scale, 0.]])).within_skin


def test_integer_scaling_preserves_extreme_dyadics_and_distance_comparisons():
    rng = np.random.default_rng(7193)
    values = [0., -0., np.nextafter(0., 1.), np.finfo(float).tiny,
              1e-200, -1e-200, 1e200, np.finfo(float).max]
    values.extend(rng.uniform(-3., 3., 120))
    for index in range(len(values)-5):
        points = (tuple(map(exact, values[index:index+3])),
                  tuple(map(exact, values[index+3:index+6])))
        radius = abs(exact(values[index])) + abs(exact(.4))
        scaled, limit, _, denominator = _integer_geometry(points, radius)
        assert Fraction(limit, denominator) == radius
        assert all(Fraction(integer, denominator) == value
                   for row, integers in zip(points, scaled)
                   for value, integer in zip(row, integers))
        actual = sum((b-a)**2 for a, b in zip(*scaled)) <= limit**2
        expected = sum((b-a)**2 for a, b in zip(*points)) <= radius**2
        assert actual == expected


def test_weighted_centers_and_boundary_decisions_use_raw_atoms():
    centers = AtomCenterMap((0, 0, 0), (.25, .5, .25), 1)
    q = np.array([[1e16, .1, 1e-300], [.3, -.1, 0.], [-1e16, .3, -1e-300]])
    g = _HostGeometry(q, centers)
    expected = tuple(sum(exact(w)*exact(q[i, j]) for i, w in enumerate(centers.weights)) for j in range(3))
    assert g.exact(0) == expected
    snapshot = NeighborGraph(centers, q, 1, .8, 1.5, 0.)
    assert snapshot.require_coverage(q).covered


def test_underflowed_weighted_motion_cannot_pass_zero_skin():
    centers = AtomCenterMap((0, 0), (1., 1e-200), 1)
    q = np.zeros((2, 3))
    graph = NeighborGraph(centers, q, 1, .8, 1.5, 0.)
    moved = q.copy()
    moved[1, 0] = 1e-200
    assert exact(centers.weights[1]) * exact(moved[1, 0]) > 0
    assert not graph.check(moved).within_skin


def test_periodic_search_matches_exact_oversized_image_cube_at_boundaries():
    cell = np.array([[1.9, 0, 0], [.7, 1.4, 0], [.2, -.1, 1.6]])
    q = np.array([[.3, .1, .2], [2.2, -.2, .4]])
    centers = AtomCenterMap((0, 1), (1., 1.), 2)
    g = NeighborGraph(centers, q, 1, .8, 1.5, .4, cell)
    radius = exact(1.5) + exact(.4)
    # Independently bound each image coordinate by backward substitution in
    # this triangular cell, using |Cartesian component| <= radius. This proves
    # that the fixed oracle cube includes every possible required image.
    bounds = [Fraction(0)] * 3
    for j in (2, 1, 0):
        displacement = abs(exact(q[1, j]) - exact(q[0, j]))
        other_rows = sum(abs(exact(cell[k, j])) * bounds[k] for k in range(j+1, 3))
        bounds[j] = (radius + displacement + other_rows) / abs(exact(cell[j, j]))
    assert max(bounds) < 4
    expected = set()
    for a in range(2):
        for b in range(a, 2):
            for image in product(range(-4, 5), repeat=3):
                if a == b and image <= (0, 0, 0):
                    continue
                v = [exact(q[b, j])-exact(q[a, j])+sum(image[k]*exact(cell[k, j]) for k in range(3)) for j in range(3)]
                if sum(x*x for x in v) <= radius**2:
                    expected.add((a, b, *image))
    assert set(g.graph.edges) == expected


def test_periodic_image_budget_precedes_eager_product_pool(monkeypatch):
    import pyeph.models.neighbors as neighbors

    def forbidden_product(*args):
        raise AssertionError("oversized image ranges must never be pooled")

    monkeypatch.setattr(neighbors, "product", forbidden_product)
    # These bounds fit int32, but pooling one axis would allocate hundreds of
    # millions of entries. Interception makes the regression safe to execute.
    with pytest.raises(ValueError, match="max_image_checks"):
        NeighborGraph(AtomCenterMap((0,), (1.,), 1), [[0., 0., 0.]], 1,
                      .8, 1.5, .4, np.diag([1e-8, 1e-2, 1.]), max_image_checks=10)
