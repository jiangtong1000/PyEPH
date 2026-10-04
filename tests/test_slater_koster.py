"""Independent orbital, spin, image and force invariants for s,p blocks."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pyeph.core.contracts import LowRankWeight, ProbeContext
from pyeph.models.local import AtomCenterMap, LocalBlockGraph, LocalBlockModel
from pyeph.models.slater_koster import SlaterKosterSPCoefficients


def fixture(spinful=True, periodic=True):
    edges = ((0, 1, 0, 0, 0), (0, 1, -1, 0, 0), (0, 0, 0, 1, 0)) if periodic else ((0, 1),)
    graph = LocalBlockGraph(2, 8 if spinful else 4, edges,
                            cell=((4.2, .2, 0.), (0., 4.5, .1), (.1, 0., 4.7)) if periodic else None,
                            switch_on=2.1, cutoff=5.2)
    model = LocalBlockModel(graph, AtomCenterMap((0, 1, 0), (1., 1., 0.), 2),
                            SlaterKosterSPCoefficients(spinful), complex_valued=spinful,
                            basis_id="test-fixed-global-sp-orbitals")
    params = dict(onsite=jnp.array([[-.8, .3, .3, .3], [-1.1, -.2, -.2, -.2]]),
                  hopping=jnp.tile(jnp.array([.07, .31, .23, -.24, .09]), (len(edges), 1)),
                  decay=jnp.tile(jnp.array([.3, .2, .4, .25, .35]), (len(edges), 1)),
                  reference_distance=jnp.ones(len(edges))*2.4)
    if spinful:
        params["soc"] = jnp.array([.18, .09])
    return model, params, jnp.array([[.1, -.1, .2], [2.6, .3, -.1], [.7, .9, -.2]])


def numpy_hamiltonian(model, params, q, phase=None):
    """Explicit component loops, including independent atomic SOC entries."""
    p = {k: np.asarray(v) for k, v in params.items()}
    q = np.asarray(q)
    b = model.graph.norbitals
    h = np.zeros((model.nstates, model.nstates), complex)
    spinful = model.coefficient_provider.spinful
    signs = model.coefficient_provider.phases
    signs = np.ones((2, 4)) if signs is None else np.asarray(signs)
    if spinful:
        signs = np.tile(signs, (1, 2))
    for atom in range(2):
        onsite = np.diag(p["onsite"][atom])
        if spinful:
            onsite = np.kron(np.eye(2), onsite).astype(complex)
            strength = p["soc"][atom]
            # Lz sigma_z, then (Lx-i Ly) in the up/down block.
            onsite[1, 2] = -1j*strength
            onsite[2, 1] = 1j*strength
            onsite[5, 6] = 1j*strength
            onsite[6, 5] = -1j*strength
            onsite[1, 7] = strength
            onsite[2, 7] = -1j*strength
            onsite[3, 5] = -strength
            onsite[3, 6] = 1j*strength
            onsite[7, 1] = strength
            onsite[7, 2] = 1j*strength
            onsite[5, 3] = -strength
            onsite[6, 3] = -1j*strength
        h[atom*b:(atom+1)*b, atom*b:(atom+1)*b] = onsite*signs[atom, :, None]*signs[atom, None, :]
    for index, (i, j, *image) in enumerate(model.graph.edges):
        d = q[j]-q[i]
        if model.graph.cell is not None:
            d = d+np.asarray(image)@np.asarray(model.graph.cell)
        r = np.linalg.norm(d)
        n = d/r
        ss, sp, ps, sigma, pi = p["hopping"][index]*np.exp(-p["decay"][index]*(r-p["reference_distance"][index]))
        t = np.zeros((4, 4), complex)
        t[0, 0] = ss
        for a in range(3):
            t[0, a+1] = sp*n[a]
            t[a+1, 0] = -ps*n[a]
            for c in range(3):
                t[a+1, c+1] = (sigma-pi)*n[a]*n[c]+(pi if a == c else 0.)
        if spinful:
            t = np.kron(np.eye(2), t)
        t = t*signs[i, :, None]*signs[j, None, :]
        x = np.clip((r-model.graph.switch_on)/(model.graph.cutoff-model.graph.switch_on), 0., 1.)
        t *= 1-10*x**3+15*x**4-6*x**5
        if phase is not None:
            t *= np.exp(1j*np.dot(phase, d))
        h[i*b:(i+1)*b, j*b:(j+1)*b] += t
        h[j*b:(j+1)*b, i*b:(i+1)*b] += t.conj().T
    return h


@pytest.mark.parametrize("spinful", [False, True])
@pytest.mark.parametrize("periodic", [False, True])
def test_operator_current_and_complete_force_against_literal_oracle(spinful, periodic):
    model, params, q = fixture(spinful, periodic)
    model.validate_at(params, q)
    h = numpy_hamiltonian(model, params, q)
    np.testing.assert_allclose(model.dense(params, q), h, atol=5e-16)
    np.testing.assert_allclose(h, h.conj().T, atol=0.)
    rng = np.random.default_rng(714)
    left = rng.normal(size=(model.nstates, 2))+1j*rng.normal(size=(model.nstates, 2))
    right = rng.normal(size=(model.nstates, 2))+1j*rng.normal(size=(model.nstates, 2))
    np.testing.assert_allclose(jax.jit(model.apply)(params, q, right), h@right, atol=2e-15)
    np.testing.assert_allclose(model.prepare_action(params, q)(right[:, 0]), h@right[:, 0], atol=2e-15)
    weight = LowRankWeight(jnp.asarray(left), jnp.asarray(right))
    actual = jax.jit(model.contract_gradient)(params, q, weight)
    step = 1e-5
    finite = np.zeros(q.shape)
    for index in np.ndindex(q.shape):
        d = np.zeros(q.shape)
        d[index] = step
        finite[index] = np.vdot(left, (numpy_hamiltonian(model, params, q+d)
                                      -numpy_hamiltonian(model, params, q-d))@right).real/(2*step)
    np.testing.assert_allclose(actual, finite, atol=2e-9, rtol=1e-7)
    np.testing.assert_array_equal(actual[2], 0.)  # spectator lacks carrier descriptors
    np.testing.assert_allclose(actual.sum(axis=0), 0., atol=2e-15)
    for axis in range(3):
        d = np.eye(3)[axis]*step
        current = model.charge*(numpy_hamiltonian(model, params, q, d)
                                -numpy_hamiltonian(model, params, q, -d))/(2*step)
        native = model.probe_apply(params, ProbeContext(q), f"current_{'xyz'[axis]}", right)
        np.testing.assert_allclose(native, current@right, atol=2e-9)


def test_atomic_soc_splitting_and_time_reversal():
    model, params, q = fixture()
    params = {**params, "hopping": jnp.zeros_like(params["hopping"])}
    h = np.asarray(model.dense(params, q))
    for i in range(2):
        eps, lam = float(params["onsite"][i, 1]), float(params["soc"][i])
        expected = np.sort([float(params["onsite"][i, 0])]*2+[eps-2*lam]*2+[eps+lam]*4)
        np.testing.assert_allclose(np.linalg.eigvalsh(h[i*8:(i+1)*8, i*8:(i+1)*8]), expected, atol=3e-16)
    model, params, q = fixture()
    h = np.asarray(model.dense(params, q))
    spin_flip = np.kron(np.eye(2), np.kron([[0., 1.], [-1., 0.]], np.eye(4)))
    np.testing.assert_allclose(spin_flip@h.conj()@spin_flip.T, h, atol=0.)
    np.testing.assert_allclose(np.diff(np.linalg.eigvalsh(h))[::2], 0., atol=2e-15)


@pytest.mark.parametrize("spinful", [False, True])
def test_rotation_and_fixed_gauge_covariance(spinful):
    model, params, q = fixture(spinful, False)
    theta = .71
    rotation = np.array([[np.cos(theta), -np.sin(theta), 0.],
                         [np.sin(theta), np.cos(theta), 0.], [0., 0., 1.]])
    orbital = np.eye(4)
    orbital[1:, 1:] = rotation
    if spinful:
        orbital = np.kron(np.diag(np.exp(np.array([-1j, 1j])*theta/2)), orbital)
    transform = np.kron(np.eye(2), orbital)
    h = np.asarray(model.dense(params, q))
    rotated = model.dense(params, q@rotation.T+4.)
    np.testing.assert_allclose(rotated, transform@h@transform.conj().T, atol=4e-16)
    signs = ((1, -1, 1, -1), (-1, 1, 1, -1))
    gauged = replace(model, coefficient_provider=replace(model.coefficient_provider, phases=signs))
    phase = np.tile(np.asarray(signs), (1, 2)) if spinful else np.asarray(signs)
    phase = phase.ravel()
    np.testing.assert_allclose(gauged.dense(params, q), phase[:, None]*h*phase[None, :], atol=0.)
    np.testing.assert_allclose(gauged.dense(params, q), numpy_hamiltonian(gauged, params, q), atol=5e-16)


def test_rewrapping_and_parameter_force_derivatives():
    model, params, q = fixture()
    shift = np.array([[1, 0, -1], [-1, 1, 0]])
    moved = q+shift[np.asarray(model.centers.atom_site)]@np.asarray(model.graph.cell)
    wrapped = model.rewrapped(shift)
    np.testing.assert_allclose(wrapped.dense(params, moved), model.dense(params, q), atol=5e-16)
    rng = np.random.default_rng(418)
    vector = jnp.asarray(rng.normal(size=model.nstates)+1j*rng.normal(size=model.nstates))
    weight = LowRankWeight(vector[:, None], vector[:, None])
    tangent = {k: jnp.ones_like(v)*.1 for k, v in params.items()}
    def fn(p):
        return model.contract_gradient(p, q, weight)
    _, derivative = jax.jvp(fn, (params,), (tangent,))
    step = 1e-5
    plus = {k: v+step*tangent[k] for k, v in params.items()}
    minus = {k: v-step*tangent[k] for k, v in params.items()}
    np.testing.assert_allclose(derivative, (fn(plus)-fn(minus))/(2*step), atol=3e-10)
    for axis in "xyz":
        np.testing.assert_allclose(wrapped.probe_apply(params, ProbeContext(moved), f"current_{axis}", vector),
                                   model.probe_apply(params, ProbeContext(q), f"current_{axis}", vector), atol=3e-15)


@pytest.mark.parametrize("spinful", [False, True])
def test_empty_support_and_smooth_cutoff(spinful):
    model, params, q = fixture(spinful, False)
    empty = replace(model, graph=replace(model.graph, edges=()))
    p = {**params, "hopping": params["hopping"][:0], "decay": params["decay"][:0],
         "reference_distance": params["reference_distance"][:0]}
    empty.validate_at(p, q)
    np.testing.assert_array_equal(empty.contract_gradient(p, q, jnp.eye(model.nstates)), 0.)
    direction = (q[1]-q[0])/jnp.linalg.norm(q[1]-q[0])

    def value(r):
        moved = q.at[1].set(q[0]+r*direction)
        return model.coefficients(params, moved).hopping[0, 0, 1].real

    for fn in (value, jax.grad(value), jax.grad(jax.grad(value))):
        np.testing.assert_allclose(fn(jnp.asarray(5.3)), 0., atol=0.)
        assert abs(float(fn(jnp.asarray(5.2-1e-7)))) < 1e-7


@pytest.mark.parametrize("key,bad", [("onsite", np.ones((2, 3))), ("hopping", np.ones((3, 4))),
                                     ("decay", np.ones((3, 5))*-.1),
                                     ("reference_distance", np.zeros(3)),
                                     ("soc", np.ones((2, 1))), ("soc", np.array([np.nan, 1.])),
                                     ("onsite", np.ones((2, 4), complex))])
def test_bad_parameters_rejected(key, bad):
    model, params, q = fixture()
    with pytest.raises(ValueError):
        model.validate_at({**params, key: bad}, q)


@pytest.mark.parametrize("kwargs", [dict(spinful=1), dict(phases=((1, 1, 0, 1),)),
                                     dict(phases=((1, 1, False, 1),)), dict(phases=(1, 1))])
def test_configuration_rejected(kwargs):
    with pytest.raises((TypeError, ValueError)):
        SlaterKosterSPCoefficients(**kwargs)
