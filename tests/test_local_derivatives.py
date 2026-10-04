"""Independent all-atom derivative and physical-current checks for local blocks.

The synthetic provider uses scalar fixed-channel invariants, including an atom
with zero center weight. Complex coefficients test Hermitian operator algebra,
not a complex/SOC surface-hopping prescription or orbital equivariance.
"""

from dataclasses import dataclass, replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pyeph.core.contracts import LowRankWeight, ProbeContext, contracted_value, prepared_action
from pyeph.models.base import AutoDiffModel
from pyeph.models.composite import SumModel
from pyeph.models.local import AtomCenterMap, LocalBlockGraph, LocalBlockModel, LocalCoefficients


@dataclass(frozen=True)
class AtomicInvariantProvider:
    """Raw hops; the adapter owns their final switching envelope."""

    complex_valued: bool

    def __call__(self, params, q, geometry):
        g = geometry
        relative = q-g.centers[g.atom_site]
        squared = jnp.sum(relative*relative, axis=1)
        # The unweighted quartic term deliberately includes the zero-weight
        # ligand: a center map does not decide descriptor atom participation.
        internal = jnp.zeros(g.centers.shape[0], q.dtype).at[g.atom_site].add(
            g.atom_weights*squared+.13*squared*squared)
        a = jnp.array([[.7, .11], [.11, -.4]])
        base = jnp.array([[[.2, .04], [.04, -.1]], [[-.3, .05], [.05, .4]]])
        onsite = base+params[0]*internal[:, None, None]*a
        real = jnp.array([[.2, .07], [-.11, .15]])
        imaginary = jnp.array([[.03, .12], [.09, -.04]])
        i, j = g.pairs[:, 0], g.pairs[:, 1]
        hopping = (1+params[1]*(internal[i]+internal[j]))[:, None, None]*real
        if self.complex_valued:
            antisymmetric = jnp.array([[0., 1.], [-1., 0.]])
            onsite = onsite+1j*params[2]*internal[:, None, None]*antisymmetric
            hopping = hopping+1j*params[2]*(1+.2*(internal[i]-internal[j]))[:, None, None]*imaginary
        hopping = hopping*jnp.exp(-params[3]*g.distances**2)[:, None, None]
        return LocalCoefficients(onsite, hopping)


class NoDenseLocal(LocalBlockModel):
    def dense(self, *args, **kwargs):
        raise AssertionError("local action/derivative/current called dense()")


def fixture(periodic, complex_valued):
    sites, weights = (0, 0, 0, 1, 1), (.25, .75, 0., .4, .6)
    mapping = AtomCenterMap(sites, weights, 2)
    if periodic:
        edges = ((0, 1, 0, 0, 0), (0, 1, -1, 0, 0), (0, 0, 1, 0, 0))
        cell = ((3.7, 0., 0.), (.2, 4.1, 0.), (.1, .3, 4.3))
    else:
        edges, cell = ((0, 1),), None
    graph = LocalBlockGraph(2, 2, edges, cell, switch_on=1.4, cutoff=5.2)
    model = NoDenseLocal(graph, mapping, AtomicInvariantProvider(complex_valued),
                         charge=-.7, complex_valued=complex_valued)
    q = jnp.array([[-.25, .1, .02], [.08, -.07, .12], [.17, .31, -.16],
                   [1.7, .12, .2], [2.05, -.2, .08]])
    params = jnp.array([.7, .3, .18, .12])
    return model, params, q


def numpy_operator(model, params, q, wavevector=None):
    """Independent loops over atoms/blocks/images, without production helpers."""
    q, p = np.asarray(q), np.asarray(params)
    mapping, graph = model.centers, model.graph
    centers = np.zeros((graph.nsites, 3))
    for site, weight, atom in zip(mapping.atom_site, mapping.weights, q, strict=True):
        centers[site] += weight*atom
    internal = np.zeros(graph.nsites)
    for site, weight, atom in zip(mapping.atom_site, mapping.weights, q, strict=True):
        r2 = np.dot(atom-centers[site], atom-centers[site])
        internal[site] += weight*r2+.13*r2*r2
    onsite = np.array([[[.2, .04], [.04, -.1]], [[-.3, .05], [.05, .4]]], dtype=complex)
    for i in range(graph.nsites):
        onsite[i] += p[0]*internal[i]*np.array([[.7, .11], [.11, -.4]])
        if model.spec.complex_valued:
            onsite[i] += 1j*p[2]*internal[i]*np.array([[0., 1.], [-1., 0.]])
    matrix = np.zeros((model.nstates, model.nstates), dtype=complex)
    for i, block in enumerate(onsite):
        matrix[2*i:2*i+2, 2*i:2*i+2] += block
    for edge in graph.edges:
        i, j = edge[:2]
        displacement = centers[j]-centers[i]
        if graph.cell is not None:
            displacement += np.asarray(edge[2:]) @ np.asarray(graph.cell)
        r = np.linalg.norm(displacement)
        if graph.cutoff is None or r <= graph.switch_on:
            support = 1.
        elif r >= graph.cutoff:
            support = 0.
        else:
            x = (r-graph.switch_on)/(graph.cutoff-graph.switch_on)
            support = 1-10*x**3+15*x**4-6*x**5
        transfer = (1+p[1]*(internal[i]+internal[j]))*np.array([[.2, .07], [-.11, .15]], dtype=complex)
        if model.spec.complex_valued:
            transfer += 1j*p[2]*(1+.2*(internal[i]-internal[j]))*np.array([[.03, .12], [.09, -.04]])
        transfer *= np.exp(-p[3]*r*r)*support
        if wavevector is not None:
            transfer *= np.exp(1j*np.dot(displacement, wavevector))
        matrix[2*i:2*i+2, 2*j:2*j+2] += transfer
        matrix[2*j:2*j+2, 2*i:2*i+2] += transfer.conj().T
    return matrix, centers


def random_vectors(nstates):
    rng = np.random.default_rng(184)
    values = [rng.normal(size=(nstates, k))+1j*rng.normal(size=(nstates, k)) for k in (3, 2, 2)]
    return tuple(jnp.asarray(value) for value in values)


def finite_gradient(function, q, width=2e-5):
    q = np.asarray(q)
    output = np.zeros_like(q)
    for index in np.ndindex(q.shape):
        delta = np.zeros_like(q)
        delta[index] = width
        output[index] = (function(q+delta)-function(q-delta))/(2*width)
    return output


def traced_shapes(node):
    """Inspect shaped JAX IR values recursively, without backend-private APIs."""
    if hasattr(node, "eqns") and hasattr(node, "constvars"):
        variables = [*node.constvars, *node.invars, *node.outvars]
        for equation in node.eqns:
            variables.extend(equation.invars)
            variables.extend(equation.outvars)
            yield from traced_shapes(equation.params)
        for variable in variables:
            shape = getattr(getattr(variable, "aval", None), "shape", None)
            if shape is not None:
                yield tuple(shape)
    elif hasattr(node, "jaxpr"):
        yield from traced_shapes(node.jaxpr)
    elif isinstance(node, dict):
        for value in node.values():
            yield from traced_shapes(value)
    elif isinstance(node, (tuple, list)):
        for value in node:
            yield from traced_shapes(value)


@pytest.mark.parametrize("complex_valued", [False, True])
def test_sparse_action_and_lowrank_force_trace_has_no_global_square_operator(complex_valued):
    model, params, q = fixture(True, complex_valued)
    vectors, left, right = random_vectors(model.nstates)
    action = jax.make_jaxpr(lambda p, x, v: prepared_action(model, p, x)(v))(params, q, vectors)
    force = jax.make_jaxpr(model.contract_gradient)(params, q, LowRankWeight(left, right))
    for trace in (action, force):
        assert (model.nstates, model.nstates) not in set(traced_shapes(trace))


@pytest.mark.parametrize("periodic", [False, True])
@pytest.mark.parametrize("complex_valued", [False, True])
def test_complete_atom_force_jvp_and_prepared_action_against_independent_numpy(periodic, complex_valued):
    model, params, q = fixture(periodic, complex_valued)
    model.validate_at(params, q)
    vectors, left, right = random_vectors(model.nstates)
    weight = LowRankWeight(left, right)
    h, _ = numpy_operator(model, params, q)
    np.testing.assert_allclose(h, h.conj().T, atol=2e-16)
    if complex_valued:
        assert np.max(abs(h.imag)) > .001
    direct = jax.jit(model.apply)(params, q, vectors)
    prepared = jax.jit(lambda p, x, v: prepared_action(model, p, x)(v))
    np.testing.assert_allclose(direct, h@vectors, atol=3e-15, rtol=2e-14)
    np.testing.assert_allclose(prepared(params, q, vectors), direct, atol=3e-15, rtol=2e-14)
    np.testing.assert_allclose(prepared(params, q, vectors[:, 0]), direct[:, 0], atol=3e-15)
    changed = params.at[0].add(.17).at[2].add(.08)
    np.testing.assert_allclose(prepared(changed, q, vectors), numpy_operator(model, changed, q)[0]@vectors,
                               atol=3e-15, rtol=2e-14)
    assert np.max(abs(prepared(changed, q, vectors)-direct)) > 1e-4

    def energy(x):
        return np.real(np.vdot(left, numpy_operator(model, params, x)[0]@right))

    numerical = finite_gradient(energy, q)
    refined = finite_gradient(energy, q, 1e-5)
    np.testing.assert_allclose(numerical, refined, atol=8e-9, rtol=2e-7)
    gradient = jax.jit(model.contract_gradient)(params, q, weight)
    np.testing.assert_allclose(gradient, refined, atol=8e-9, rtol=3e-7)
    via_prepared = jax.grad(lambda x: contracted_value(prepared_action(model, params, x), weight))(q)
    np.testing.assert_allclose(via_prepared, gradient, atol=3e-14, rtol=3e-14)
    dense_weight = left@right.conj().T
    np.testing.assert_allclose(model.contract_gradient(params, q, dense_weight), gradient,
                               atol=3e-14, rtol=3e-14)
    # Translation invariance applies to this scalar-feature carrier only.
    np.testing.assert_allclose(np.sum(gradient, axis=0), 0., atol=4e-14)
    assert np.linalg.norm(gradient[2]) > 1e-4  # zero center weight, nonzero carrier force

    direction = jnp.array([[.3, -.1, .2], [-.1, 1/30, -1/15], [0., .2, -.1],
                            [0., 0., 0.], [0., 0., 0.]])
    # The first two displacements preserve their weighted center exactly;
    # atom2 is invisible to the center but visible to the internal descriptor.
    centers0 = numpy_operator(model, params, q)[1]
    centers1 = numpy_operator(model, params, q+direction)[1]
    np.testing.assert_allclose(centers0, centers1, atol=2e-16)
    assert abs(float(jnp.sum(gradient*direction))) > 1e-5
    actual_jvp = jax.jit(lambda x, dx: jax.jvp(
        lambda y: prepared_action(model, params, y)(vectors), (x,), (dx,))[1])(q, direction)
    epsilon = 2e-5
    expected_jvp = (numpy_operator(model, params, q+epsilon*direction)[0]
                    - numpy_operator(model, params, q-epsilon*direction)[0])@vectors/(2*epsilon)
    np.testing.assert_allclose(actual_jvp, expected_jvp, atol=3e-9, rtol=3e-7)


@pytest.mark.parametrize("complex_valued", [False, True])
def test_mixed_parameter_force_jvp_retains_provider_and_weight_response(complex_valued):
    model, params, q = fixture(True, complex_valued)
    _, left, right = random_vectors(model.nstates)
    direction = jnp.array([.3, -.2, .1, .15])

    def weight(p):
        return LowRankWeight(left*(1+.2*p[0]), right*(1-.1*p[1]))

    actual = jax.jit(lambda p: jax.jvp(
        lambda theta: model.contract_gradient(theta, q, weight(theta)),
        (p,), (direction,))[1])(params)

    def independent_force(p):
        fixed_left, fixed_right = weight(np.asarray(p))
        return finite_gradient(lambda x: np.real(np.vdot(
            fixed_left, numpy_operator(model, p, x)[0]@fixed_right)), q)

    epsilon = 2e-4
    numerical = (independent_force(params+epsilon*direction)
                 - independent_force(params-epsilon*direction))/(2*epsilon)
    np.testing.assert_allclose(actual, numerical, atol=2e-7, rtol=3e-5)
    assert np.linalg.norm(actual[2]) > 1e-4


@pytest.mark.parametrize("periodic", [False, True])
@pytest.mark.parametrize("complex_valued", [False, True])
def test_physical_current_uses_full_windowed_hopping_and_image_displacement(periodic, complex_valued):
    model, params, q = fixture(periodic, complex_valued)
    vectors, _, _ = random_vectors(model.nstates)
    h, centers = numpy_operator(model, params, q)
    for axis, name in enumerate("xyz"):
        epsilon = 2e-5*np.eye(3)[axis]
        hp = numpy_operator(model, params, q, epsilon)[0]
        hm = numpy_operator(model, params, q, -epsilon)[0]
        expected_matrix = model.charge*(hp-hm)/(4e-5)
        current = jax.jit(lambda x, v: model.probe_apply(
            params, ProbeContext(x), f"current_{name}", v))(q, vectors)
        np.testing.assert_allclose(current, expected_matrix@vectors, atol=3e-10, rtol=3e-8)
        phased_derivative = jax.jvp(lambda wavevector: model.apply_peierls(params, q, wavevector, vectors),
                                    (jnp.zeros(3),), (jnp.eye(3)[axis],))[1]
        np.testing.assert_allclose(current, model.charge*phased_derivative, atol=2e-14, rtol=2e-14)
        if not periodic:
            position = np.diag(np.repeat(centers[:, axis], 2))
            commutator = 1j*model.charge*(h@position-position@h)
            np.testing.assert_allclose(current, commutator@vectors, atol=3e-14, rtol=3e-14)


@dataclass(frozen=True)
class ConstantProvider:
    def __call__(self, params, q, geometry):
        return LocalCoefficients(params["onsite"], params["hopping"])


def test_nonzero_complex_self_image_current_is_not_a_finite_position_commutator():
    graph = LocalBlockGraph(1, 1, ((0, 0, 1, 0, 0),), ((3., 0., 0.), (0., 4., 0.), (0., 0., 5.)))
    model = NoDenseLocal(graph, AtomCenterMap((0,), (1.,), 1), ConstantProvider(),
                         charge=-.8, complex_valued=True)
    params = dict(onsite=jnp.array([[[.5+0j]]]), hopping=jnp.array([[[.2+.3j]]]))
    q, v = jnp.array([[.7, -.2, .1]]), jnp.array([.4+.2j])
    model.validate_at(params, q)
    np.testing.assert_allclose(model.apply(params, q, v), .9*v, atol=2e-16)
    expected = -2*model.charge*3*.3
    current = model.probe_apply(params, ProbeContext(q), "current_x", v)
    analytic = np.asarray(expected*v)
    current = np.asarray(current)
    assert current.shape == analytic.shape and current.dtype == analytic.dtype
    assert np.isfinite(current).all() and np.isfinite(analytic).all()
    precision = np.finfo(analytic.real.dtype).eps
    np.testing.assert_allclose(current, analytic, atol=0., rtol=16*precision)
    assert expected != 0  # [scalar H, scalar center position] would give zero.
    for axis in "yz":
        np.testing.assert_array_equal(model.probe_apply(params, ProbeContext(q), f"current_{axis}", v), 0.)
    np.testing.assert_allclose(model.contract_gradient(params, q, LowRankWeight(v[:, None], v[:, None])),
                               0., atol=2e-16)
    # Independently, H(k)=a+2*Re(t)*cos(L*k)-2*Im(t)*sin(L*k), so the central
    # difference is exactly J*sinc(L*h). Resolve its second-order truncation
    # with larger h before Richardson cancellation, avoiding subtraction of
    # nearly equal values at the old h=1e-6. No production derivative is used.
    length, onsite, hopping = 3., .5, .2+.3j
    scale = abs(model.charge)*(abs(onsite)+2*abs(hopping))*np.max(np.abs(v))
    steps = np.array([2e-3, 1e-3, 5e-4])
    differences, roundoff = [], []
    for step in steps:
        epsilon = jnp.array([step, 0., 0.])
        numerical = np.asarray(model.charge*(model.apply_peierls(params, q, epsilon, v)
                                  - model.apply_peierls(params, q, -epsilon, v))/(2*step))
        assert numerical.shape == analytic.shape and numerical.dtype == analytic.dtype
        assert np.isfinite(numerical).all()
        # Declared operation-scale allowance, not a formal guarantee for an
        # unspecified device's transcendental kernels; test it explicitly.
        allowance = 32*precision*scale/step
        discrete_exact = analytic*np.sinc(length*step/np.pi)
        np.testing.assert_allclose(numerical, discrete_exact, atol=allowance, rtol=0.)
        differences.append(numerical)
        roundoff.append(allowance)
    errors = np.max(np.abs(np.asarray(differences)-analytic), axis=1)
    orders = np.log2(errors[:-1]/errors[1:])
    assert np.all((orders > 1.95) & (orders < 2.05)), orders
    extrapolated = (4*differences[2]-differences[1])/3
    # Taylor remainder |sinc(x)-1+x^2/6| <= x^4/120 gives this conservative
    # bound after combining the two central differences. Keep the original
    # final 2e-11 accuracy gate, now against a qualified finite-difference oracle.
    truncation = np.max(np.abs(analytic))*(length*steps[1])**4/288
    bound = truncation+(4*roundoff[2]+roundoff[1])/3
    assert bound < 2e-11
    np.testing.assert_allclose(current, extrapolated, atol=bound, rtol=0.)
    np.testing.assert_allclose(current, extrapolated, atol=2e-11, rtol=0.)


@dataclass(frozen=True)
class AtomicReference(AutoDiffModel):
    spec: object

    def apply(self, params, q, vectors):
        return jnp.zeros_like(vectors)

    def reference_energy(self, params, q):
        return params["offset"]+.5*jnp.sum(params["spring"]*(q-params["q0"])**2)

    def probe_apply(self, params, context, probe, vectors):
        if probe not in self.spec.probes:
            return super().probe_apply(params, context, probe, vectors)
        return jnp.zeros_like(vectors)


@pytest.mark.parametrize("complex_valued", [False, True])
def test_nonzero_full_atom_reference_composes_once_with_carrier_force_and_current(complex_valued):
    carrier, params, q = fixture(True, complex_valued)
    reference = AtomicReference(replace(carrier.spec, name="independent_atomic_reference"))
    nuclear = dict(q0=q-jnp.array([.12, -.09, .04]),
                   spring=jnp.arange(q.size).reshape(q.shape)*.03+.7, offset=.37)
    total = SumModel((reference, carrier), additive_probes=carrier.spec.probes)
    combined = (nuclear, params)
    vectors, left, right = random_vectors(carrier.nstates)
    weight = LowRankWeight(left, right)

    def energy(x):
        dx = np.asarray(x)-np.asarray(nuclear["q0"])
        vref = nuclear["offset"]+.5*np.sum(np.asarray(nuclear["spring"])*dx*dx)
        electronic = np.real(np.vdot(left, numpy_operator(carrier, params, x)[0]@right))
        return float(vref+electronic)

    # Reference energy enters once even for a general nonunit-trace weight.
    actual = total.reference_gradient(combined, q)+total.contract_gradient(combined, q, weight)
    np.testing.assert_allclose(actual, finite_gradient(energy, q), atol=8e-9, rtol=3e-7)
    np.testing.assert_allclose(total.reference_gradient(combined, q),
                               nuclear["spring"]*(q-nuclear["q0"]), atol=3e-16)
    assert np.linalg.norm(total.reference_gradient(combined, q)) > .2
    np.testing.assert_allclose(total.apply(combined, q, vectors), carrier.apply(params, q, vectors), atol=2e-15)
    for axis in "xyz":
        np.testing.assert_allclose(total.probe_apply(combined, ProbeContext(q), f"current_{axis}", vectors),
                                   carrier.probe_apply(params, ProbeContext(q), f"current_{axis}", vectors), atol=2e-15)
