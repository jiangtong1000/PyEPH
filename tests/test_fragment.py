"""Independent physics checks for the effective oriented-fragment baseline."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pyeph.core.contracts import ProbeContext, pure_state_weight
from pyeph.models.fragment import OrientedFragmentCoefficients
from pyeph.models.local import AtomCenterMap, LocalBlockGraph, LocalBlockModel


def fixture(n=3, periodic=False):
    atoms = np.array([[-.4, -.3, 0.], [.7, -.2, .1], [-.2, .8, .05]])
    rotation = np.array([[1., 0., 0.], [0., .8, -.6], [0., .6, .8]])
    origins = np.array([[0., 0., 0.], [2.8, .2, .6], [1.3, 3.1, -.4]])[:n]
    q = np.concatenate([atoms@np.linalg.matrix_power(rotation, i).T+center
                        for i, center in enumerate(origins)])
    anchors = tuple(tuple(range(3*i, 3*i+3)) for i in range(n))
    mapping = AtomCenterMap(tuple(np.repeat(np.arange(n), 3)), (1/3,)*len(q), n)
    edges = tuple((a, b) for a in range(n) for b in range(a+1, n))
    cell = None
    if periodic:
        cell = ((5.1, .2, .1), (0., 5.4, .2), (.3, 0., 6.))
        edges += ((0, 1, -1, 0, 0), (0, 0, 0, 1, 0))
    graph = LocalBlockGraph(n, 1, edges, cell=cell, switch_on=2.6, cutoff=5.6)
    model = LocalBlockModel(graph, mapping, OrientedFragmentCoefficients(anchors),
                            charge=-1., basis_id="test-effective-axial-fragments")
    params = dict(onsite=jnp.linspace(-.05, .06, n),
                  deformation=jnp.asarray(np.tile([.03, -.02], (n, 1))),
                  bond_lengths=jnp.asarray(np.tile([1.05, 1.08], (n, 1))),
                  pp_sigma=jnp.asarray(.09), pp_pi=jnp.asarray(-.025),
                  decay=jnp.asarray(.35), reference_distance=jnp.asarray(3.))
    return model, params, jnp.asarray(q)


def numpy_hamiltonian(model, params, q, kappa=None):
    """Cartesian p-p tensor contraction, independently looped over images."""
    p, q = {k: np.asarray(v) for k, v in params.items()}, np.asarray(q)
    centers = np.zeros((model.nstates, 3))
    for atom, site, weight in zip(q, model.centers.atom_site, model.centers.weights):
        centers[site] += weight*atom
    h = np.zeros((model.nstates, model.nstates), complex)
    normals = []
    for site, (o, a, b) in enumerate(model.coefficient_provider.anchors):
        u, v = q[a]-q[o], q[b]-q[o]
        normal = np.cross(u, v)
        normal *= model.coefficient_provider.phases[site]/np.linalg.norm(normal)
        normals.append(normal)
        h[site, site] = p["onsite"][site]+p["deformation"][site] @ (
            np.array([np.linalg.norm(u), np.linalg.norm(v)])-p["bond_lengths"][site])
    for i, j, *image in model.graph.edges:
        d = centers[j]-centers[i]
        if model.graph.cell is not None:
            d += np.asarray(image)@np.asarray(model.graph.cell)
        r = np.linalg.norm(d)
        tensor = p["pp_pi"]*np.eye(3)+(p["pp_sigma"]-p["pp_pi"])*np.outer(d, d)/r**2
        value = normals[i] @ tensor @ normals[j]
        value *= np.exp(-p["decay"]*(r-p["reference_distance"]))
        if model.graph.cutoff is not None:
            s = np.clip((r-model.graph.switch_on)/(model.graph.cutoff-model.graph.switch_on), 0., 1.)
            value *= 1-10*s**3+15*s**4-6*s**5
        if kappa is not None:
            value *= np.exp(1j*np.dot(kappa, d))
        h[i, j] += value
        h[j, i] += np.conj(value)
    return h


def state(n):
    c = np.arange(1, n+1)+.3j*np.arange(n)
    return jnp.asarray(c/np.linalg.norm(c))


def gradient(model, params, q, c=None):
    return model.contract_gradient(params, q, pure_state_weight(state(model.nstates) if c is None else c))


def currents(model, params, q):
    return jnp.stack([model.probe_apply(params, ProbeContext(q), f"current_{a}",
                                       jnp.eye(model.nstates)) for a in "xyz"])


@pytest.mark.parametrize("n,periodic", [(2, False), (3, False), (3, True)])
def test_dense_action_force_and_current_against_independent_tensor(n, periodic):
    model, params, q = fixture(n, periodic)
    model.validate_at(params, q)
    expected = numpy_hamiltonian(model, params, q)
    np.testing.assert_allclose(model.dense(params, q), expected, atol=3e-16)
    c = state(n)
    np.testing.assert_allclose(jax.jit(model.apply)(params, q, c), expected@c, atol=3e-16)
    np.testing.assert_allclose(model.prepare_action(params, q)(c), expected@c, atol=3e-16)
    step = 2e-5
    finite = np.zeros(q.shape)
    for index in np.ndindex(q.shape):
        delta = np.zeros(q.shape)
        delta[index] = step
        finite[index] = np.vdot(c, (numpy_hamiltonian(model, params, q+delta)
                                   -numpy_hamiltonian(model, params, q-delta))@c).real/(2*step)
    np.testing.assert_allclose(gradient(model, params, q), finite, atol=2e-11, rtol=2e-7)
    for axis in range(3):
        delta = np.eye(3)[axis]*step
        current = model.charge*(numpy_hamiltonian(model, params, q, delta)
                                -numpy_hamiltonian(model, params, q, -delta))/(2*step)
        np.testing.assert_allclose(currents(model, params, q)[axis], current, atol=3e-10)


def test_rotation_translation_force_torque_and_current_covariance():
    model, params, q = fixture()
    axis = np.array([1., 2., -3.])/np.sqrt(14.)
    x, y, z = axis
    cross = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
    rotation = np.eye(3)+np.sin(.81)*cross+(1-np.cos(.81))*cross@cross
    transformed = q@rotation.T+np.array([13., -7., 1.])
    np.testing.assert_allclose(model.dense(params, transformed), model.dense(params, q), atol=3e-16)
    g = gradient(model, params, q)
    np.testing.assert_allclose(gradient(model, params, transformed), g@rotation.T, atol=3e-16)
    np.testing.assert_allclose(g.sum(axis=0), 0., atol=3e-16)
    np.testing.assert_allclose(jnp.cross(q, g).sum(axis=0), 0., atol=3e-16)
    np.testing.assert_allclose(currents(model, params, transformed),
                               np.einsum("ab,bij->aij", rotation, currents(model, params, q)), atol=3e-16)


def test_constant_fragment_phase_changes_basis_not_physics():
    model, params, q = fixture()
    signs = jnp.array([1., -1., 1.])
    flipped = replace(model, coefficient_provider=replace(model.coefficient_provider, phases=(1, -1, 1)))
    h = model.dense(params, q)
    np.testing.assert_allclose(flipped.dense(params, q), signs[:, None]*h*signs[None, :], atol=0.)
    c = state(model.nstates)
    np.testing.assert_allclose(gradient(flipped, params, q, signs*c), gradient(model, params, q, c), atol=2e-16)
    j = currents(model, params, q)
    jp = currents(flipped, params, q)
    np.testing.assert_allclose(jp, signs[None, :, None]*j*signs[None, None, :], atol=0.)
    np.testing.assert_allclose(jnp.einsum("i,aij,j->a", jnp.conj(signs*c), jp, signs*c),
                               jnp.einsum("i,aij,j->a", jnp.conj(c), j, c), atol=2e-16)


def test_orientation_changes_hopping_at_fixed_centers_and_bond_lengths():
    model, params, q = fixture(2)
    center = np.asarray(model.geometry(q).centers[1])
    rotation = np.array([[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]])
    rotated = q.at[3:].set((q[3:]-center)@rotation.T+center)
    np.testing.assert_allclose(model.geometry(rotated).distances, model.geometry(q).distances, atol=1e-15)
    np.testing.assert_allclose(jnp.diag(model.dense(params, rotated)), jnp.diag(model.dense(params, q)), atol=1e-16)
    assert abs(float(model.dense(params, rotated)[0, 1]-model.dense(params, q)[0, 1])) > .005


def test_atom_and_fragment_relabeling_preserve_all_derivatives():
    model, params, q = fixture()
    order = np.array([7, 1, 3, 8, 0, 6, 4, 2, 5])
    inverse = np.argsort(order)
    mapping = AtomCenterMap(np.asarray(model.centers.atom_site)[order],
                            np.asarray(model.centers.weights)[order], model.nstates)
    provider = replace(model.coefficient_provider, anchors=inverse[np.asarray(model.coefficient_provider.anchors)])
    permuted = replace(model, centers=mapping, coefficient_provider=provider)
    np.testing.assert_allclose(permuted.dense(params, q[order]), model.dense(params, q), atol=2e-16)
    np.testing.assert_allclose(gradient(permuted, params, q[order]), gradient(model, params, q)[order], atol=2e-16)
    site_order = np.array([2, 0, 1])
    site_inverse = np.argsort(site_order)
    remapped = replace(model, centers=replace(model.centers, atom_site=site_inverse[np.asarray(model.centers.atom_site)]),
                       coefficient_provider=replace(model.coefficient_provider,
                                                    anchors=np.asarray(model.coefficient_provider.anchors)[site_order]))
    remapped_params = {key: value[site_order] if value.ndim else value for key, value in params.items()}
    np.testing.assert_allclose(remapped.dense(remapped_params, q),
                               model.dense(params, q)[site_order][:, site_order], atol=2e-16)
    np.testing.assert_allclose(gradient(remapped, remapped_params, q, state(3)[site_order]),
                               gradient(model, params, q), atol=2e-16)


def test_periodic_fragment_rewrapping_preserves_image_resolved_current():
    model, params, q = fixture(periodic=True)
    shifts = np.array([[1, -1, 0], [-1, 0, 1], [0, 1, -1]])
    transformed = q+shifts[np.asarray(model.centers.atom_site)]@np.asarray(model.graph.cell)
    wrapped = model.rewrapped(shifts)
    np.testing.assert_allclose(wrapped.dense(params, transformed), model.dense(params, q), atol=4e-16)
    np.testing.assert_allclose(currents(wrapped, params, transformed), currents(model, params, q), atol=4e-16)
    np.testing.assert_allclose(gradient(wrapped, params, transformed), gradient(model, params, q), atol=4e-16)


def test_cutoff_has_two_continuous_zero_derivatives():
    model, params, q = fixture(2)
    displacement = model.geometry(q).displacements[0]
    direction = displacement/jnp.linalg.norm(displacement)

    def hopping(r):
        moved = q.at[3:].add((r-jnp.linalg.norm(displacement))*direction)
        return model.dense(params, moved)[0, 1]

    for derivative in (hopping, jax.grad(hopping), jax.grad(jax.grad(hopping))):
        np.testing.assert_allclose(derivative(jnp.asarray(model.graph.cutoff+.01)), 0., atol=0.)
        assert abs(float(derivative(jnp.asarray(model.graph.cutoff-1e-7)))) < 2e-8


@pytest.mark.parametrize("failure", ["collinear", "collapsed", "near_collinear", "wrong_fragment"])
def test_degenerate_or_misassigned_frames_fail_actual_preflight(failure):
    model, params, q = fixture(2)
    if failure == "wrong_fragment":
        model = replace(model, coefficient_provider=replace(model.coefficient_provider,
                                                             anchors=((0, 1, 3), (3, 4, 5))))
    else:
        q = q.at[2].set(q[0]+2*(q[1]-q[0]))
        if failure == "collapsed":
            q = q.at[1].set(q[0])
        elif failure == "near_collinear":
            q = q.at[2, 2].add(1e-9)
    with pytest.raises(ValueError, match="finite"):
        model.validate_at(params, q)
    assert not np.isfinite(jax.jit(model.dense)(params, q)).all()


def test_parameter_derivatives_remain_dynamic_and_batch_preflight_checks_every_frame():
    model, params, q = fixture(2)
    c = state(2)
    function = jax.jit(lambda p: jnp.vdot(c, model.apply(p, q, c)).real)
    tangent = {key: jnp.ones_like(value)*.13 for key, value in params.items()}
    _, derivative = jax.jvp(function, (params,), (tangent,))
    step = 1e-5
    plus = {key: value+step*tangent[key] for key, value in params.items()}
    minus = {key: value-step*tangent[key] for key, value in params.items()}
    finite = np.vdot(c, (numpy_hamiltonian(model, plus, q)-numpy_hamiltonian(model, minus, q))@c).real/(2*step)
    np.testing.assert_allclose(derivative, finite, atol=3e-11)
    invalid = q.at[2].set(q[0]+2*(q[1]-q[0]))
    with pytest.raises(ValueError, match="finite"):
        model.validate_at(params, jnp.stack((q, invalid)), batch=True)


def test_twelve_atom_dimer_includes_nonanchor_center_derivatives():
    model, params, q = fixture(2)
    extra = np.array([[.3, .2, .7], [-.6, .1, -.4], [.1, -.8, .2]])
    all_atoms = jnp.concatenate((q[:3], extra, q[3:], extra+np.array([2.8, .2, .6])))
    mapping = AtomCenterMap((0,)*6+(1,)*6, (1/6,)*12, 2)
    provider = OrientedFragmentCoefficients(((0, 1, 2), (6, 7, 8)))
    model = replace(model, centers=mapping, coefficient_provider=provider)
    model.validate_at(params, all_atoms)
    g = gradient(model, params, all_atoms)
    assert np.linalg.norm(np.asarray(g)[[3, 4, 5, 9, 10, 11]]) > 1e-4
    c = state(2)
    direction = np.cos(np.arange(all_atoms.size)).reshape(all_atoms.shape)
    width = 1e-5
    finite = np.vdot(c, (numpy_hamiltonian(model, params, all_atoms+width*direction)
                         -numpy_hamiltonian(model, params, all_atoms-width*direction))@c).real/(2*width)
    np.testing.assert_allclose(jnp.sum(g*direction), finite, atol=3e-11)


@pytest.mark.parametrize("kwargs", [dict(anchors=((0, 1, 1),)), dict(anchors=((False, 1, 2),)),
                                    dict(anchors=()), dict(anchors=((0, 1, 2),), phases=(0,)),
                                    dict(anchors=((0, 1, 2),), min_sine=1.),
                                    dict(anchors=((0, 1, 2),), min_bond_length=0.)])
def test_invalid_static_contracts(kwargs):
    with pytest.raises(ValueError):
        OrientedFragmentCoefficients(**kwargs)


@pytest.mark.parametrize("key,value", [("onsite", [0.]), ("decay", -1.),
                                        ("pp_pi", 1j), ("reference_distance", np.inf)])
def test_invalid_dynamic_contracts(key, value):
    model, params, _ = fixture(2)
    with pytest.raises(ValueError):
        model.validate_params({**params, key: value})
