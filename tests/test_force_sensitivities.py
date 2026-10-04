"""Partial-coordinate forces retain the outer electronic-state response graph."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pyeph.core.contracts import LowRankWeight, ModelSpec, pure_state_weight
from pyeph.core.problem import CoupledClassical, Problem
from pyeph.core.state import make_state
from pyeph.core.system import SystemSpec
from pyeph.dynamics.ehrenfest import Ehrenfest
from pyeph.integrators.electronic import Integrator
from pyeph.models.base import AutoDiffModel
from pyeph.models.composite import ReferenceShiftModel


class NonlinearModel(AutoDiffModel):
    spec = ModelSpec(SystemSpec(2, (2,), coordinate_kind="canonical"),
                     name="force-sensitivity", complex_valued=True)

    def apply(self, params, q, vectors):
        a, b, _ = params
        x, y = q
        d = a*jnp.sin(x) + .15*y*y
        v = b*jnp.exp(-.2*x*x)*(1+.1*y)*jnp.exp(.4j*y)
        e = .11*x*y + .2*a*jnp.cos(y)
        return jnp.array([[d, v], [v.conj(), e]]) @ vectors

    def reference_energy(self, params, q):
        x, y = q
        return .5*params[2]*(x*x+1.3*y*y)+.04*x*y*y


def independent_gradients(params, q):
    """Explicit analytic NumPy derivatives, independent of production autodiff."""
    a, b, k = np.asarray(params)
    x, y = np.asarray(q)
    v = b*np.exp(-.2*x*x)*(1+.1*y)*np.exp(.4j*y)
    vx, vy = -.4*x*v, v*(.1/(1+.1*y)+.4j)
    derivatives = np.array([[[a*np.cos(x), vx], [vx.conjugate(), .11*y]],
                            [[.3*y, vy], [vy.conjugate(), .11*x-.2*a*np.sin(y)]]])
    reference_gradient = np.array([k*x+.04*y*y, 1.3*k*y+.08*x*y])
    return reference_gradient, derivatives


def independent_force(params, q, c, alpha=None):
    reference, derivatives = independent_gradients(params, q)
    c = np.asarray(c)
    result = -reference-np.real(np.einsum("i,aij,j->a", c.conjugate(), derivatives, c))
    if alpha is not None:
        result += (np.vdot(c, c).real-1)*alpha*np.array([np.cos(q[0]), .4*q[1]])
    return result


def force(model, params, q, c, *, dense_weight=False):
    weight = jnp.outer(c, c.conj()) if dense_weight else pure_state_weight(c)
    return -model.reference_gradient(params, q)-model.contract_gradient(params, q, weight)


def central_jacobian(function, point, step=2e-5):
    point = np.asarray(point, dtype=float)
    columns = []
    for index in range(len(point)):
        delta = np.zeros_like(point)
        delta[index] = step
        columns.append((np.asarray(function(point+delta))-np.asarray(function(point-delta)))/(2*step))
    return np.stack(columns, axis=-1)


def c_from_real(value):
    return value[:2]+1j*value[2:]


def shift_fn(alpha, q):
    return alpha*(jnp.sin(q[0])+.2*q[1]**2)


PARAMS = jnp.array([.4, .23, .6])
Q = jnp.array([.7, -.35])
C = jnp.array([.8, .3+.52j]) / jnp.sqrt(.8**2+.3**2+.52**2)


@pytest.mark.parametrize("dense_weight", [False, True])
@pytest.mark.parametrize("shifted", [False, True])
@pytest.mark.parametrize("field", ["q", "c_real_imag", "params"])
def test_force_jacobians_match_independent_finite_differences(dense_weight, shifted, field):
    model = ReferenceShiftModel(NonlinearModel(), shift_fn) if shifted else NonlinearModel()
    parameters = jnp.concatenate((PARAMS, jnp.array([.17]))) if shifted else PARAMS
    z = jnp.concatenate((C.real, C.imag))
    point = {"q": Q, "c_real_imag": z, "params": parameters}[field]

    def arguments(value):
        q = value if field == "q" else Q
        c = c_from_real(value) if field == "c_real_imag" else C
        p = value if field == "params" else parameters
        return q, c, p

    def calculated(value):
        q, c, p = arguments(value)
        dynamic = (p[:3], p[3]) if shifted else p
        return force(model, dynamic, q, c, dense_weight=dense_weight)

    def independent(value):
        q, c, p = arguments(value)
        return independent_force(p[:3], q, c, p[3] if shifted else None)

    np.testing.assert_allclose(calculated(point), independent(point), atol=2e-15)
    expected = central_jacobian(independent, point)
    np.testing.assert_allclose(jax.jacfwd(calculated)(point), expected, rtol=1e-7, atol=1e-9)
    np.testing.assert_allclose(jax.jacrev(calculated)(point), expected, rtol=1e-7, atol=1e-9)


def normalized_c(value, array_module=jnp):
    angle = value[0]+.2*value[1]
    return array_module.array([array_module.cos(angle), array_module.exp(.3j)*array_module.sin(angle)])


@pytest.mark.parametrize("dense_weight", [False, True])
def test_outer_dependent_weight_does_not_change_the_inner_partial_derivative(dense_weight):
    model = NonlinearModel()
    point = jnp.array([.4, .7])

    def calculated(value):
        params, q = PARAMS.at[0].set(value[0]), Q.at[0].set(value[1])
        return force(model, params, q, normalized_c(value), dense_weight=dense_weight)

    def independent(value):
        params, q = np.array(PARAMS), np.array(Q)
        params[0], q[0] = value
        # The instantaneous derivative is partial in q; c's response belongs
        # only in an outer derivative of the resulting force operation.
        return independent_force(params, q, normalized_c(value, np))

    np.testing.assert_allclose(calculated(point), independent(point), atol=2e-15)
    expected = central_jacobian(independent, point)
    np.testing.assert_allclose(jax.jacfwd(calculated)(point), expected, rtol=1e-7, atol=1e-9)
    np.testing.assert_allclose(jax.jacrev(calculated)(point), expected, rtol=1e-7, atol=1e-9)


@pytest.mark.parametrize("dense_weight", [False, True])
def test_complex_offdiagonal_weight_response(dense_weight):
    model = NonlinearModel()
    left = jnp.array([[.3+.4j, -.2j], [.7, .1-.6j]])
    right = jnp.array([[.2-.5j, .9], [-.4, .3+.2j]])
    variation = jnp.array([[.1j, .4], [-.2+.3j, .2]])

    def calculated(value):
        left_value = left+value[0]*variation
        right_value = right*jnp.exp(1j*value[1])
        weight = (left_value@right_value.conj().T if dense_weight
                  else LowRankWeight(left_value, right_value))
        return model.contract_gradient(PARAMS, Q, weight)

    def independent(value):
        left_value = np.asarray(left)+value[0]*np.asarray(variation)
        right_value = np.asarray(right)*np.exp(1j*value[1])
        w = left_value@right_value.conj().T
        _, derivatives = independent_gradients(PARAMS, Q)
        return np.real(np.einsum("ij,aij->a", w.conj(), derivatives))

    point = jnp.array([.3, -.2])
    np.testing.assert_allclose(calculated(point), independent(point), atol=2e-15)
    expected = central_jacobian(independent, point)
    np.testing.assert_allclose(jax.jacfwd(calculated)(point), expected, rtol=1e-7, atol=1e-9)
    np.testing.assert_allclose(jax.jacrev(calculated)(point), expected, rtol=1e-7, atol=1e-9)


def test_reference_shift_sensitivity_respects_normalization_sector():
    base = NonlinearModel()
    shifted = ReferenceShiftModel(base, shift_fn)
    alpha = .17
    point = jnp.array([.4, .7])

    def original(value):
        return force(base, PARAMS, Q, normalized_c(value))

    def compensated(value):
        return force(shifted, (PARAMS, alpha), Q, normalized_c(value))

    np.testing.assert_allclose(original(point), compensated(point), atol=2e-15)
    np.testing.assert_allclose(jax.jacfwd(original)(point), jax.jacfwd(compensated)(point), atol=2e-14)
    np.testing.assert_allclose(jax.jacrev(original)(point), jax.jacrev(compensated)(point), atol=2e-14)
    # Reference compensation is invariant on the unit sphere. Its derivative
    # normal to that sphere legitimately retains the varying carrier norm.
    z = jnp.concatenate((C.real, C.imag))
    delta = jax.jacfwd(lambda x: force(shifted, (PARAMS, alpha), Q, c_from_real(x))
                      - force(base, PARAMS, Q, c_from_real(x)))(z)
    expected = np.outer(alpha*np.array([np.cos(Q[0]), .4*Q[1]]), 2*np.asarray(z))
    np.testing.assert_allclose(delta, expected, atol=2e-14)


def test_smooth_pure_ehrenfest_sensitivity_matches_refined_finite_differences():
    model = NonlinearModel()
    initial = make_state(Q, [.25, -.1], C)
    integrator = Integrator(.025, "rk4", electronic_substeps=2)

    def trajectory(value):
        params = PARAMS.at[1].set(value[0])
        c = jnp.array([jnp.cos(value[1]), jnp.exp(.3j)*jnp.sin(value[1])])
        state = initial._replace(q=initial.q.at[0].set(value[2]), electronic=c)
        problem = Problem(model, params, CoupledClassical(1.3), Ehrenfest())
        step = problem.method.build_step(problem, integrator)
        final, _ = jax.lax.scan(lambda s, _: (step(s), None), state, None, length=12)
        return jnp.concatenate((final.q, final.p, jnp.abs(final.electronic)**2))

    point = jnp.array([.23, .45, .7])
    function = jax.jit(trajectory)
    forward = jax.jit(jax.jacfwd(trajectory))(point)
    reverse = jax.jit(jax.jacrev(trajectory))(point)
    np.testing.assert_allclose(forward, reverse, atol=2e-13, rtol=2e-13)
    errors = [np.max(abs(central_jacobian(function, point, h)-np.asarray(forward)))
              for h in (2e-3, 1e-3, 5e-4)]
    assert errors[0]/errors[1] > 3.8
    assert errors[1]/errors[2] > 3.8
    assert errors[-1] < 1e-6
