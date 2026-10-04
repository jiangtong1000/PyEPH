"""Independent sparse Torch physics checks, not trained-model accuracy claims.

The NumPy oracle constructs a small physical Hamiltonian from explicit atomic
geometry; no native local model or adapter helper supplies reference values.
Large-system checks use sparse actions and scalar directional differences.
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pyeph.adapters.torch_local import TorchLocalBlockModel
from pyeph.core.contracts import LowRankWeight, ProbeContext, prepared_action, pure_state_weight
from pyeph.models.local import AtomCenterMap, LocalBlockGraph, LocalCoefficients

torch = pytest.importorskip("torch")


@dataclass(frozen=True)
class AtomicScalarHead:
    """Scalar descriptors with complete internal and neighbor derivatives.

    Internal radii intentionally include zero-center-weight atoms. Channel
    matrices are fixed complex labels, with no orbital-equivariance claim.
    """

    def __call__(self, p, q, g):
        radius = torch.zeros(len(g.centers), dtype=q.dtype).index_add(
            0, g.atom_site, torch.sum((q-g.centers[g.atom_site])**2, dim=-1))
        i, j = g.pairs[:, 0], g.pairs[:, 1]
        messages = g.support*(0.2+torch.exp(-g.distances)+0.1*(radius[i]+radius[j]))
        environment = torch.zeros_like(radius).index_add(0, i, messages).index_add(0, j, messages)
        onsite = (1+0.2*radius+0.1*environment)[:, None, None]*p["onsite"]
        scale = (torch.exp(-p["decay"]*g.distances)
                 * (1+0.15*(radius[i]+radius[j])+0.08*(environment[i]+environment[j])))
        phase = torch.exp(0.17j*(radius[i]-radius[j]))
        return LocalCoefficients(onsite, (scale*phase)[:, None, None]*p["hopping"])


def fixture(*, periodic=True):
    q = jnp.array([[-0.2, 0.1, 0.05], [0.3, -0.1, -0.04],
                   [1.4, 0.6, -0.2], [1.7, 0.8, 0.15],
                   [2.5, -0.4, 0.3], [2.2, -0.1, 0.5]])
    centers = AtomCenterMap((0, 0, 1, 1, 2, 2), (1., 0., .7, .3, .4, .6), 3)
    if periodic:
        cell = ((4.1, .2, .1), (.3, 4.3, .2), (.1, .3, 4.6))
        edges = ((0, 1, 0, 0, 0), (0, 2, -1, 0, 0), (1, 2, 0, 0, 0), (0, 0, 1, 0, 0))
    else:
        cell, edges = None, ((0, 1), (0, 2), (1, 2))
    graph = LocalBlockGraph(3, 2, edges, cell=cell, switch_on=1.0, cutoff=5.0)
    p = dict(onsite=jnp.array([[.23, .07+.04j], [.07-.04j, -.11]]),
             hopping=jnp.array([[.19+.08j, .03-.12j], [-.05+.09j, -.14-.06j]]),
             decay=jnp.array(.37))
    model = TorchLocalBlockModel(graph, centers, AtomicScalarHead(), charge=-1.7, complex_valued=True)
    return model, p, q


def numpy_blocks(model, params, q):
    p = jax.tree.map(np.asarray, params)
    q = np.asarray(q)
    n = model.graph.nsites
    site = np.asarray(model.centers.atom_site)
    centers = np.zeros((n, 3))
    for atom in range(len(q)):
        centers[site[atom]] += model.centers.weights[atom]*q[atom]
    displacements = []
    for i, j, *image in model.graph.edges:
        d = centers[j]-centers[i]
        if model.graph.cell is not None:
            d = d+np.asarray(image)@np.asarray(model.graph.cell)
        displacements.append(d)
    displacements = np.asarray(displacements)
    distance = np.linalg.norm(displacements, axis=1)
    u = np.clip((distance-model.graph.switch_on)/(model.graph.cutoff-model.graph.switch_on), 0, 1)
    support = 1-10*u**3+15*u**4-6*u**5
    radius = np.zeros(n)
    for atom in range(len(q)):
        radius[site[atom]] += np.sum((q[atom]-centers[site[atom]])**2)
    environment = np.zeros(n)
    for edge, (i, j, *_) in enumerate(model.graph.edges):
        value = support[edge]*(.2+np.exp(-distance[edge])+.1*(radius[i]+radius[j]))
        environment[i] += value
        environment[j] += value
    onsite = (1+.2*radius+.1*environment)[:, None, None]*p["onsite"]
    hopping = []
    for edge, (i, j, *_) in enumerate(model.graph.edges):
        scale = (np.exp(-p["decay"]*distance[edge])
                 * (1+.15*(radius[i]+radius[j])+.08*(environment[i]+environment[j])))
        hopping.append(support[edge]*scale*np.exp(.17j*(radius[i]-radius[j]))*p["hopping"])
    return onsite, np.asarray(hopping), displacements


def numpy_hamiltonian(model, params, q, wavevector=None):
    onsite, hopping, displacement = numpy_blocks(model, params, q)
    if wavevector is not None:
        hopping = hopping*np.exp(1j*(displacement@wavevector))[:, None, None]
    b, n = model.graph.norbitals, model.nstates
    h = np.zeros((n, n), dtype=complex)
    for i, block in enumerate(onsite):
        h[i*b:(i+1)*b, i*b:(i+1)*b] += block
    for block, (i, j, *_) in zip(hopping, model.graph.edges, strict=True):
        h[i*b:(i+1)*b, j*b:(j+1)*b] += block
        h[j*b:(j+1)*b, i*b:(i+1)*b] += block.conj().T
    return h


def finite_gradient(function, q, step=2e-5):
    q = np.asarray(q)
    result = np.empty_like(q)
    for index in np.ndindex(q.shape):
        direction = np.zeros_like(q)
        direction[index] = step
        result[index] = (function(q+direction)-function(q-direction))/(2*step)
    return result


@pytest.mark.parametrize("dense_weight", [False, True])
def test_complex_nonhermitian_weight_all_atomic_forces_against_independent_dense_fd(dense_weight):
    model, params, q = fixture()
    rng = np.random.default_rng(90012)
    left = (rng.normal(size=(6, 3))+1j*rng.normal(size=(6, 3)))/np.sqrt(6)
    right = (rng.normal(size=(6, 3))+1j*rng.normal(size=(6, 3)))/np.sqrt(6)
    weight = left@right.conj().T
    assert np.linalg.norm(weight-weight.conj().T) > 1
    value = weight if dense_weight else LowRankWeight(jnp.asarray(left), jnp.asarray(right))
    actual = jax.jit(model.contract_gradient)(params, q, value)
    expected = finite_gradient(lambda x: np.vdot(weight, numpy_hamiltonian(model, params, x)).real, q)
    np.testing.assert_allclose(actual, expected, atol=4e-10, rtol=2e-8)
    # Atom 1 has zero center weight but influences the local internal descriptor.
    assert np.linalg.norm(expected[1]) > 1e-3
    np.testing.assert_allclose(np.sum(actual, axis=0), 0., atol=3e-14)
    block = jnp.asarray(right)
    np.testing.assert_allclose(jax.jit(model.apply)(params, q, block),
                               numpy_hamiltonian(model, params, q)@right, atol=3e-15)


def test_complete_gradient_audit_covers_external_geometry_and_provider_terms():
    model, params, q = fixture()
    model.validate_complete_gradients(params, q)
    c = jnp.array([1., 2j, -.5, .3j, 1.2, -.7j])
    c = c/jnp.linalg.norm(c)
    expected = finite_gradient(lambda x: np.vdot(c, numpy_hamiltonian(model, params, x)@c).real, q)
    np.testing.assert_allclose(model.contract_gradient(params, q, pure_state_weight(c)),
                               expected, atol=3e-10, rtol=2e-8)


def simple_raw_head(p, q, g):
    onsite = torch.zeros((2, 1, 1), dtype=q.dtype)+q.sum()*0
    raw = p["hopping"]*(1+.2*g.distances)
    return LocalCoefficients(onsite, raw[:, None, None])


@pytest.mark.parametrize("distance", [.8, 1., 1.7, 3., 3.4])
def test_cutoff_is_applied_exactly_once_with_its_complete_force(distance):
    model = TorchLocalBlockModel(LocalBlockGraph(2, 1, ((0, 1),), switch_on=1., cutoff=3.),
                                 AtomCenterMap((0, 1), (1., 1.), 2), simple_raw_head)
    params = {"hopping": jnp.array(.27)}
    q = jnp.array([[0., 0., 0.], [distance, 0., 0.]])
    u = np.clip((distance-1)/2, 0., 1.)
    support = 1-10*u**3+15*u**4-6*u**5
    derivative = (-30*u**2+60*u**3-30*u**4)/2
    transfer = .27*(1+.2*distance)*support
    dtransfer = .27*(.2*support+(1+.2*distance)*derivative)
    np.testing.assert_allclose(model.apply(params, q, jnp.array([0., 1.])), [transfer, 0.], atol=5e-16)
    c = jnp.ones(2)/jnp.sqrt(2.)
    expected = np.array([[-dtransfer, 0., 0.], [dtransfer, 0., 0.]])
    np.testing.assert_allclose(model.contract_gradient(params, q, pure_state_weight(c)), expected, atol=1e-15)
    if 1 < distance < 3:
        assert abs(transfer-.27*(1+.2*distance)*support**2) > .01


@pytest.mark.parametrize("periodic", [False, True])
def test_complex_current_matches_independent_full_displacement_peierls_derivative(periodic):
    model, params, q = fixture(periodic=periodic)
    vectors = jnp.eye(model.nstates, dtype=complex)
    for axis, label in enumerate("xyz"):
        direction = np.eye(3)[axis]*2e-5
        expected = model.charge*(numpy_hamiltonian(model, params, q, direction)
                                 -numpy_hamiltonian(model, params, q, -direction))/(4e-5)
        actual = model.probe_apply(params, ProbeContext(q, jnp.ones_like(q)*13.), f"current_{label}", vectors)
        np.testing.assert_allclose(actual, expected, atol=2e-10, rtol=2e-8)
        np.testing.assert_allclose(actual, actual.conj().T, atol=2e-15)
    wavevector = jnp.array([.13, -.07, .03])
    np.testing.assert_allclose(model.apply_peierls(params, q, wavevector, vectors),
                               numpy_hamiltonian(model, params, q, wavevector), atol=3e-15)
    if periodic:
        # Only the unique self-image edge contributes to this onsite current
        # block; it is nonzero because the raw complex block is not Hermitian.
        current = model.probe_apply(params, ProbeContext(q), "current_x", vectors)
        assert np.linalg.norm(current[:2, :2]) > 1e-3
    assert not any(name.startswith("lab_current") for name in model.spec.probes)


def test_coherent_fragment_rewrap_preserves_complex_actions_currents_and_all_atomic_forces():
    model, params, q = fixture()
    shifts = np.array([[1, -1, 0], [-1, 0, 1], [0, 1, -1]])
    wrapped = model.rewrapped(shifts)
    assert isinstance(wrapped, TorchLocalBlockModel)
    assert wrapped.coefficient_provider is model.coefficient_provider
    shifted = q+jnp.asarray(shifts)[jnp.asarray(model.centers.atom_site)]@jnp.asarray(model.graph.cell)
    c = jnp.array([1., .2j, -.5, .7j, .3, -.1j])
    for axis in "xyz":
        np.testing.assert_allclose(wrapped.probe_apply(params, ProbeContext(shifted), f"current_{axis}", c),
                                   model.probe_apply(params, ProbeContext(q), f"current_{axis}", c), atol=3e-15)
    np.testing.assert_allclose(wrapped.apply(params, shifted, c), model.apply(params, q, c), atol=3e-15)
    np.testing.assert_allclose(wrapped.contract_gradient(params, shifted, pure_state_weight(c)),
                               model.contract_gradient(params, q, pure_state_weight(c)), atol=4e-15)


def chain_head(p, q, g):
    onsite = (.2+.03*q[:, 0])[:, None, None]*p["onsite"]
    hopping = (.1*torch.exp(-.2*g.distances))[:, None, None]*p["hopping"]
    return LocalCoefficients(onsite, hopping)


class SparseGuard(TorchLocalBlockModel):
    def dense(self, *args, **kwargs):
        raise AssertionError("sparse test must not request a dense Hamiltonian")

    def _action(self, coefficients, vectors):
        assert vectors.ndim == 1 or vectors.shape[1] <= 3
        assert coefficients.onsite.shape == (self.graph.nsites, 2, 2)
        assert coefficients.hopping.shape == (len(self.graph.edges), 2, 2)
        return super()._action(coefficients, vectors)


def test_large_sparse_actions_preparation_current_and_lowrank_force_need_no_full_matrix(monkeypatch):
    n = 256
    model = SparseGuard(LocalBlockGraph(n, 2, tuple((i, i+1) for i in range(n-1))),
                        AtomCenterMap(tuple(range(n)), (1.,)*n, n), chain_head, complex_valued=True)
    params = dict(onsite=jnp.array([[1., .2j], [-.2j, -.7]]),
                  hopping=jnp.array([[1.+.2j, .3-.1j], [-.4+.3j, -.8-.2j]]))
    q = np.column_stack((.7*np.arange(n), .1*np.sin(np.arange(n)), np.zeros(n)))
    rng = np.random.default_rng(10299)
    left = (rng.normal(size=(2*n, 2))+1j*rng.normal(size=(2*n, 2)))/np.sqrt(2*n)
    right = (rng.normal(size=(2*n, 2))+1j*rng.normal(size=(2*n, 2)))/np.sqrt(2*n)

    def oracle_action(x, v):
        v = v.reshape(n, 2, -1)
        d = np.diff(x, axis=0)
        onsite = (.2+.03*x[:, 0])[:, None, None]*np.asarray(params["onsite"])
        hop = (.1*np.exp(-.2*np.linalg.norm(d, axis=1)))[:, None, None]*np.asarray(params["hopping"])
        out = np.einsum("iab,ibk->iak", onsite, v)
        out[:-1] += np.einsum("iab,ibk->iak", hop, v[1:])
        out[1:] += np.einsum("iba,ibk->iak", hop.conj(), v[:-1])
        return out.reshape(2*n, -1)

    # Guard common dense-construction routes independently of model.dense.
    original_eye = torch.eye
    original_zeros = torch.zeros

    def guarded_eye(rows, *args, **kwargs):
        assert rows < model.nstates
        return original_eye(rows, *args, **kwargs)

    def guarded_zeros(*args, **kwargs):
        shape = args[0] if args and isinstance(args[0], (tuple, list)) else args
        assert tuple(shape) != (model.nstates, model.nstates)
        return original_zeros(*args, **kwargs)

    monkeypatch.setattr(torch, "eye", guarded_eye)
    monkeypatch.setattr(torch, "zeros", guarded_zeros)
    qj, lj, rj = map(jnp.asarray, (q, left, right))
    np.testing.assert_allclose(jax.jit(model.apply)(params, qj, rj), oracle_action(q, right), atol=3e-16)
    prepared = jax.jit(lambda p, x, v: prepared_action(model, p, x)(v))
    np.testing.assert_allclose(prepared(params, qj, rj), oracle_action(q, right), atol=3e-16)
    current = jax.jit(lambda p, x, v: model.probe_apply(p, ProbeContext(x), "current_x", v))(params, qj, rj)
    assert current.shape == (2*n, 2) and np.isfinite(current).all()
    gradient = np.asarray(jax.jit(model.contract_gradient)(params, qj, LowRankWeight(lj, rj)))
    direction = rng.normal(size=q.shape)
    step = 2e-5
    expected = (np.vdot(left, oracle_action(q+step*direction, right)).real
                -np.vdot(left, oracle_action(q-step*direction, right)).real)/(2*step)
    np.testing.assert_allclose(np.sum(gradient*direction), expected, atol=3e-10, rtol=2e-7)
