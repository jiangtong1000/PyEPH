"""Scientific invariants of fixed scalar channels on an explicit local graph.

The test provider is deliberately elementary and external: these checks concern
geometry/action composition, not a fitted NN or p/d-orbital equivariance. Its
neighbor messages are windowed by the provider; raw hoppings are windowed once
by LocalBlockModel. No neighbor search is implied by a finite support radius.
"""

from dataclasses import dataclass, replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pyeph.core.contracts import ProbeContext
from pyeph.core.problem import CoupledClassical, Problem
from pyeph.core.state import make_state
from pyeph.dynamics.ehrenfest import Ehrenfest
from pyeph.execution.runner import Runner
from pyeph.integrators.electronic import Integrator
from pyeph.io.provenance import assert_matching_manifest, problem_manifest, validate_manifest
from pyeph.models.local import AtomCenterMap, LocalBlockGraph, LocalBlockModel, LocalCoefficients


@dataclass(frozen=True)
class ScalarCoefficients:
    """Shared invariant scalar features; channel labels never rotate as orbitals."""

    internal_scale: float = 0.7

    def __call__(self, params, q, geometry):
        g = geometry
        site = g.atom_site
        radius2 = jnp.zeros(g.centers.shape[0], q.dtype).at[site].add(
            g.atom_weights * jnp.sum((q - g.centers[site]) ** 2, axis=-1))
        i, j = g.pairs[:, 0], g.pairs[:, 1]
        message = g.support * (0.2 + jnp.exp(-g.distances)
                               + 0.1 * (radius2[i] + radius2[j]))
        environment = jnp.zeros_like(radius2).at[i].add(message).at[j].add(message)
        node_factor = 1.0 + self.internal_scale * radius2 + params["message"] * environment
        edge_factor = (jnp.exp(-params["decay"] * g.distances)
                       * (1.0 + 0.3 * (radius2[i] + radius2[j])
                          + 0.2 * (environment[i] + environment[j])))
        return LocalCoefficients(node_factor[:, None, None] * params["onsite"],
                                 edge_factor[:, None, None] * params["hopping"])


def _params(block=2):
    onsite = np.array([[0.2, 0.07], [0.07, -0.1]])[:block, :block]
    hopping = np.array([[0.3, 0.02], [0.02, -0.15]])[:block, :block]
    return dict(onsite=jnp.asarray(onsite), hopping=jnp.asarray(hopping),
                decay=jnp.asarray(0.4), message=jnp.asarray(0.31))


def _fixture(periodic=False, block=2):
    q = jnp.array([[-0.2, 0.1, 0.05], [0.3, -0.1, -0.04],
                   [1.4, 0.6, -0.2], [1.7, 0.8, 0.15],
                   [2.5, -0.4, 0.3], [2.2, -0.1, 0.5]])
    centers = AtomCenterMap((0, 0, 1, 1, 2, 2), (0.4, 0.6, 0.7, 0.3, 0.5, 0.5), 3)
    if periodic:
        cell = np.array([[4.3, 0.2, 0.1], [0.4, 4.6, 0.3], [0.1, 0.2, 4.9]])
        edges = ((0, 1, 0, 0, 0), (0, 2, -1, 0, 0), (1, 2, 0, 0, 0),
                 (0, 0, 1, 0, 0))
    else:
        cell, edges = None, ((0, 1), (0, 2), (1, 2))
    graph = LocalBlockGraph(3, block, edges, cell=cell, switch_on=3.5, cutoff=5.5)
    return LocalBlockModel(graph, centers, ScalarCoefficients(), charge=-1.3), _params(block), q


def _rotation():
    # Rodrigues rotation about a generic axis; the electronic channel basis is fixed.
    axis = np.array([1., 2., -3.]) / np.sqrt(14.)
    x, y, z = axis
    cross = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
    angle = 0.73
    return np.eye(3) + np.sin(angle) * cross + (1 - np.cos(angle)) * (cross @ cross)


def _currents(model, params, q):
    eye = jnp.eye(model.nstates)
    context = ProbeContext(q=q, time=jnp.asarray(0.0))
    return jnp.stack([model.probe_apply(params, context, f"current_{a}", eye) for a in "xyz"])


def _energy_gradient(model, params, q):
    c = jnp.arange(1, model.nstates + 1) + 0.2j * jnp.arange(model.nstates)
    c = c / jnp.linalg.norm(c)
    return jax.grad(lambda x: jnp.vdot(c, model.apply(params, x, c)).real)(q)


@pytest.mark.parametrize("periodic", [False, True])
def test_scalar_geometry_and_vector_current_transform_together(periodic):
    model, params, q = _fixture(periodic)
    rotation = _rotation()
    transformed_q = q @ rotation.T + jnp.array([0.71, -0.37, 1.2])
    graph = model.graph if not periodic else replace(model.graph, cell=np.asarray(model.graph.cell) @ rotation.T)
    transformed = replace(model, graph=graph)
    np.testing.assert_allclose(transformed.dense(params, transformed_q), model.dense(params, q), atol=3e-15)
    np.testing.assert_allclose(_currents(transformed, params, transformed_q),
                               np.einsum("ab,bij->aij", rotation, _currents(model, params, q)),
                               atol=3e-15)
    gradient = _energy_gradient(model, params, q)
    np.testing.assert_allclose(_energy_gradient(transformed, params, transformed_q),
                               gradient @ rotation.T, atol=3e-15)
    np.testing.assert_allclose(np.sum(gradient, axis=0), 0.0, atol=3e-15)


def test_finite_current_is_charge_times_center_commutator_not_convective_motion():
    model, params, q = _fixture()
    h, centers = model.dense(params, q), model.geometry(q).centers
    current = _currents(model, params, q)
    for axis in range(3):
        position = jnp.diag(jnp.repeat(centers[:, axis], model.graph.norbitals))
        np.testing.assert_allclose(current[axis], 1j * model.charge * (h @ position - position @ h),
                                   atol=2e-15)
    changed_charge = replace(model, charge=0.7)
    np.testing.assert_allclose(_currents(changed_charge, params, q), current * (0.7 / model.charge), atol=2e-15)
    context = ProbeContext(q=q, velocity=jnp.full_like(q, 17.), time=jnp.asarray(0.0))
    np.testing.assert_allclose(model.probe_apply(params, context, "current_x", jnp.eye(model.nstates)),
                               current[0], atol=2e-15)
    assert not any(name.startswith("lab_current") for name in model.spec.probes)


@pytest.mark.parametrize("periodic", [False, True])
def test_atom_permutation_preserves_geometry_and_permutes_all_atomic_forces(periodic):
    model, params, q = _fixture(periodic)
    order = np.array([4, 1, 3, 0, 5, 2])
    center_map = AtomCenterMap(np.asarray(model.centers.atom_site)[order],
                              np.asarray(model.centers.weights)[order], model.graph.nsites)
    permuted = replace(model, centers=center_map)
    np.testing.assert_allclose(permuted.dense(params, q[order]), model.dense(params, q), atol=2e-15)
    np.testing.assert_allclose(_currents(permuted, params, q[order]), _currents(model, params, q), atol=2e-15)
    np.testing.assert_allclose(_energy_gradient(permuted, params, q[order]),
                               _energy_gradient(model, params, q)[order], atol=2e-15)


def test_site_relabeling_reverses_unique_edges_without_changing_scalar_channel_physics():
    model, params, q = _fixture(block=2)
    new_to_old = np.array([2, 0, 1])
    old_to_new = np.argsort(new_to_old)
    edges = tuple(sorted(tuple(sorted((old_to_new[a], old_to_new[b])))
                         for a, b, *_ in model.graph.edges))
    centers = replace(model.centers, atom_site=old_to_new[np.asarray(model.centers.atom_site)])
    relabeled = replace(model, graph=replace(model.graph, edges=edges), centers=centers)
    electronic_order = np.concatenate([2 * site + np.arange(2) for site in new_to_old])
    expected_h = np.asarray(model.dense(params, q))[np.ix_(electronic_order, electronic_order)]
    np.testing.assert_allclose(relabeled.dense(params, q), expected_h, atol=2e-15)
    expected_current = np.asarray(_currents(model, params, q))[:, electronic_order][:, :, electronic_order]
    np.testing.assert_allclose(_currents(relabeled, params, q), expected_current, atol=2e-15)


def test_internal_distortion_changes_coefficients_even_when_centers_do_not_move():
    model, params, q = _fixture()
    direction = jnp.zeros_like(q).at[0].set(jnp.array([0.6, -0.2, 0.1]) * 0.6)
    direction = direction.at[1].set(-jnp.array([0.6, -0.2, 0.1]) * 0.4)
    changed = q + 0.13 * direction
    np.testing.assert_allclose(model.geometry(changed).centers, model.geometry(q).centers, atol=3e-16)
    assert np.max(np.abs(model.dense(params, changed) - model.dense(params, q))) > 1e-4
    derivative = jax.jvp(lambda x: model.dense(params, x), (q,), (direction,))[1]
    eps = 1e-5
    finite_difference = (model.dense(params, q + eps * direction)
                         - model.dense(params, q - eps * direction)) / (2 * eps)
    assert np.max(np.abs(derivative)) > 1e-3
    np.testing.assert_allclose(derivative, finite_difference, atol=2e-11, rtol=2e-8)


def test_disconnected_replication_preserves_blocks_and_has_no_cross_component_response():
    model, params, q = _fixture()
    edges = tuple(model.graph.edges) + tuple((a + 3, b + 3, *image) for a, b, *image in model.graph.edges)
    centers = AtomCenterMap(tuple(model.centers.atom_site) + tuple(a + 3 for a in model.centers.atom_site),
                            tuple(model.centers.weights) * 2, 6)
    replicated = replace(model, graph=replace(model.graph, nsites=6, edges=edges), centers=centers)
    q2 = jnp.concatenate((q, q + jnp.array([0.3, 0.4, 0.2])))
    # Components may be geometrically close: explicit graph edges determine locality.
    h = np.asarray(model.dense(params, q))
    expected = np.kron(np.eye(2), h)
    np.testing.assert_allclose(replicated.dense(params, q2), expected, atol=2e-15)
    direction = jnp.zeros_like(q2).at[6:].set(jnp.arange(18).reshape(6, 3) * 0.01)
    response = jax.jvp(lambda x: replicated.dense(params, x), (q2,), (direction,))[1]
    np.testing.assert_allclose(response[:model.nstates], 0., atol=2e-15)
    assert np.max(np.abs(response[model.nstates:, model.nstates:])) > 1e-4
    for got, original in zip(_currents(replicated, params, q2), _currents(model, params, q)):
        np.testing.assert_allclose(got, np.kron(np.eye(2), original), atol=2e-15)


def _pair_fixture(edges=((0, 1),)):
    graph = LocalBlockGraph(2, 1, edges, switch_on=1., cutoff=2.)
    return LocalBlockModel(graph, AtomCenterMap((0, 1), (1., 1.), 2), ScalarCoefficients()), _params(1)


def test_cutoff_removes_neighbor_onsite_messages_and_hopping_with_two_smooth_derivatives():
    model, params = _pair_fixture()
    isolated = replace(model, graph=replace(model.graph, edges=()))

    def interaction(distance):
        q = jnp.array([[0., 0., 0.], [distance, 0., 0.]])
        return (model.dense(params, q) - isolated.dense(params, q)).reshape(-1)

    inside = np.asarray(interaction(1.7)).reshape(2, 2)
    assert np.max(np.abs(np.diag(inside))) > 1e-3
    assert abs(inside[0, 1]) > 1e-3
    for distance in (2., 2.1):
        np.testing.assert_allclose(interaction(distance), 0., atol=2e-15)
        np.testing.assert_allclose(jax.jacfwd(interaction)(distance), 0., atol=2e-14)
        np.testing.assert_allclose(jax.jacfwd(jax.jacfwd(interaction))(distance), 0., atol=2e-13)


def test_fixed_candidate_graph_does_not_infer_missing_near_neighbors():
    model, params = _pair_fixture(edges=())
    q = jnp.array([[0., 0., 0.], [0.7, 0., 0.]])
    explicitly_connected = replace(model, graph=replace(model.graph, edges=((0, 1),)))
    assert model.geometry(q).pairs.shape == (0, 2)
    np.testing.assert_array_equal(model.dense(params, q)[0, 1], 0.)
    np.testing.assert_array_equal(_currents(model, params, q), 0.)
    assert abs(explicitly_connected.dense(params, q)[0, 1]) > 0.1
    assert not np.allclose(np.diag(model.dense(params, q)), np.diag(explicitly_connected.dense(params, q)))


def test_coherent_fragment_periodic_rewrap_preserves_provider_and_full_image_current():
    model, params, q = _fixture(periodic=True)
    shifts = np.array([[1, -1, 0], [-2, 0, 1], [0, 1, -1]])
    rewrapped = model.rewrapped(shifts)
    shifted_q = q + shifts[np.asarray(model.centers.atom_site)] @ np.asarray(model.graph.cell)
    assert rewrapped.coefficient_provider is model.coefficient_provider
    assert rewrapped.centers is model.centers
    np.testing.assert_allclose(rewrapped.geometry(shifted_q).displacements, model.geometry(q).displacements,
                               atol=3e-15)
    np.testing.assert_allclose(rewrapped.dense(params, shifted_q), model.dense(params, q), atol=3e-15)
    np.testing.assert_allclose(_currents(rewrapped, params, shifted_q), _currents(model, params, q), atol=4e-15)
    np.testing.assert_allclose(_energy_gradient(rewrapped, params, shifted_q),
                               _energy_gradient(model, params, q), atol=3e-15)
    # Arbitrary individual atom wrapping is outside the coherent-fragment domain.
    incoherent_q = q.at[0].add(jnp.asarray(model.graph.cell)[0])
    assert not np.allclose(model.dense(params, incoherent_q), model.dense(params, q))


_PROVIDER_ID = {"model.coefficient_provider": "test-scalar-provider-code-and-config-v1"}


def _problem(model, params):
    return Problem(model, params, CoupledClassical(1.), Ehrenfest())


@pytest.mark.parametrize("change", ["centers", "cell", "edges", "cutoff", "charge", "provider", "params"])
def test_graph_map_provider_and_parameter_changes_are_scientific_identity(change):
    model, params, _ = _fixture(periodic=True)
    problem = _problem(model, params)
    before = problem_manifest(problem, Integrator(0.01), artifact_ids=_PROVIDER_ID)
    validate_manifest(before)
    if change == "centers":
        centers = replace(model.centers, weights=(0.5, 0.5, 0.7, 0.3, 0.5, 0.5))
        model = replace(model, centers=centers)
    elif change == "cell":
        model = replace(model, graph=replace(model.graph, cell=np.asarray(model.graph.cell) * 1.01))
    elif change == "edges":
        model = replace(model, graph=replace(model.graph, edges=model.graph.edges[:-1]))
    elif change == "cutoff":
        model = replace(model, graph=replace(model.graph, cutoff=5.6))
    elif change == "charge":
        model = replace(model, charge=1.)
    elif change == "provider":
        model = replace(model, coefficient_provider=ScalarCoefficients(internal_scale=0.8))
    else:
        params = {**params, "decay": params["decay"] + 0.01}
    after = problem_manifest(_problem(model, params), Integrator(0.01), artifact_ids=_PROVIDER_ID)
    with pytest.raises(ValueError, match="params" if change == "params" else "model"):
        assert_matching_manifest(before, after)


def test_opaque_mutable_provider_requires_owned_artifact_identity_for_strict_checkpoint(tmp_path):
    model, params, q = _fixture()
    captured = {"scale": 1.0}
    provider = model.coefficient_provider

    def opaque(params, q, geometry):
        coefficients = provider(params, q, geometry)
        return LocalCoefficients(*(captured["scale"] * x for x in coefficients))

    model = replace(model, coefficient_provider=opaque)
    runner = Runner(_problem(model, params), Integrator(0.01))
    state = make_state(q, jnp.zeros_like(q), jnp.eye(model.nstates, dtype=complex)[:, 0])
    manifest = problem_manifest(runner.problem, runner.integrator)
    assert manifest["unresolved"] == ["model.coefficient_provider"]
    path = tmp_path / "local_checkpoint.h5"
    with pytest.raises(ValueError, match="model.coefficient_provider|incomplete"):
        runner.save_checkpoint(path, state)
    runner.save_checkpoint(path, state, artifact_ids=_PROVIDER_ID)
    loaded = runner.load_checkpoint(path, artifact_ids=_PROVIDER_ID)
    for expected, actual in zip(jax.tree.leaves(state), jax.tree.leaves(loaded)):
        np.testing.assert_array_equal(actual, expected)
    before = np.asarray(model.dense(params, q))
    captured["scale"] = 1.1
    assert not np.allclose(model.dense(params, q), before)
    # Captured Python state cannot be fingerprinted automatically. The caller
    # must change the artifact ID when its code/weights/configuration changes.
    updated_id = {"model.coefficient_provider": "test-scalar-provider-code-and-config-v2"}
    with pytest.raises(ValueError, match="model"):
        runner.load_checkpoint(path, artifact_ids=updated_id)
