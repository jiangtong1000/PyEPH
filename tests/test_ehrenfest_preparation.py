"""Ordinary Ehrenfest preparation preserves the existing fixed-q splitting.

These tests make no provider invocation-count or performance claims. The
independent NumPy path explicitly implements the original force and electronic
equations; it never calls production propagation or preparation helpers.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.linalg import expm

from pyeph.core.contracts import ModelSpec, pure_state_weight
from pyeph.core.problem import CoupledClassical, Problem
from pyeph.core.state import make_state, stack_states
from pyeph.core.system import SystemSpec
from pyeph.dynamics.ehrenfest import Ehrenfest
from pyeph.execution.runner import Execution, SimulationError
from pyeph.integrators.electronic import Integrator
from pyeph.models.base import AutoDiffModel
from pyeph.simulation import Simulation


class PreparedNonlinearModel(AutoDiffModel):
    spec = ModelSpec(SystemSpec(2, (2,), coordinate_kind="canonical"),
                     name="ordinary-ehrenfest-preparation", complex_valued=True)

    @staticmethod
    def matrix(params, q):
        a, b, _ = params
        x, y = q
        z = a*jnp.sin(x)+.1*y*y
        v = b*(1+.2*y)*jnp.exp(.3j*x)
        return jnp.array([[z, v], [v.conj(), -.7*z+.05*x*y]])

    def apply(self, params, q, vectors):
        return self.matrix(params, q)@vectors

    def prepare_action(self, params, q):
        matrix = self.matrix(params, q)
        return lambda vectors: matrix@vectors

    def reference_energy(self, params, q):
        x, y = q
        return .5*params[2]*(x*x+1.2*y*y)+.03*x*y*y


def numpy_matrix(params, q, *, changed=False):
    a, b, _ = np.asarray(params)
    x, y = np.asarray(q)
    z = a*np.sin(x)+.1*y*y
    v = b*(1+.2*y)*np.exp(.3j*x)
    matrix = np.array([[z, v], [v.conjugate(), -.7*z+.05*x*y]])
    if changed:
        matrix += .08*(x-y)*np.diag([1., -1.])
    return matrix


def numpy_force(params, q, c, *, changed=False):
    a, b, spring = np.asarray(params)
    x, y = np.asarray(q)
    v = b*(1+.2*y)*np.exp(.3j*x)
    vx, vy = .3j*v, .2*b*np.exp(.3j*x)
    derivative = np.array([[[a*np.cos(x), vx], [vx.conjugate(), -.7*a*np.cos(x)+.05*y]],
                           [[.2*y, vy], [vy.conjugate(), -.14*y+.05*x]]])
    if changed:
        derivative += np.array([.08, -.08])[:, None, None]*np.diag([1., -1.])
    reference = np.array([spring*x+.03*y*y, 1.2*spring*y+.06*x*y])
    return -reference-np.einsum("i,aij,j->a", np.asarray(c).conj(), derivative, c).real


def numpy_evolve(params, q, p, c, mass, dt, steps, algorithm, substeps, *, changed=False):
    q, p, c = np.array(q), np.array(p), np.array(c)

    def electronic(x, amplitude, duration):
        matrix = numpy_matrix(params, x, changed=changed)
        h = duration/substeps
        for _ in range(substeps):
            if algorithm == "exponential_midpoint":
                amplitude = expm(-1j*h*matrix)@amplitude
            else:
                k1 = -1j*matrix@amplitude
                k2 = -1j*matrix@(amplitude+.5*h*k1)
                k3 = -1j*matrix@(amplitude+.5*h*k2)
                k4 = -1j*matrix@(amplitude+h*k3)
                amplitude = amplitude+h*(k1+2*k2+2*k3+k4)/6
        return amplitude

    for _ in range(steps):
        c = electronic(q, c, dt/2)
        p_half = p+dt/2*numpy_force(params, q, c, changed=changed)
        q = q+dt*p_half/np.asarray(mass)
        p = p_half+dt/2*numpy_force(params, q, c, changed=changed)
        c = electronic(q, c, dt/2)
    return q, p, c


PARAMS = jnp.array([.4, .23, .6])
MASS = jnp.array([1.3, 2.1])


def initial_states():
    c = np.array([.8, .3+.52j])
    c /= np.linalg.norm(c)
    return [make_state([.7, -.35], [.25, -.1], c, time=.4, trajectory_id=13),
            make_state([-.2, .6], [-.17, .21], c.conj(), time=1.1, trajectory_id=102)]


@pytest.mark.parametrize("algorithm,substeps", [("rk4", 1), ("rk4", 3),
                                                ("exponential_midpoint", 1), ("exponential_midpoint", 4)])
def test_public_batched_splitting_and_cached_updates_match_independent_original_equations(algorithm, substeps):
    model = PreparedNonlinearModel()
    integrator = Integrator(.08, algorithm, substeps)
    runner = Simulation(Problem(model, PARAMS, CoupledClassical(MASS), Ehrenfest()),
                        integrator, Execution(chunk_size=2))
    initial = stack_states(initial_states())
    first = runner.run(initial, 3).final_state
    cache = dict(runner._compiled)
    changed_params = PARAMS*jnp.array([1.2, .8, 1.1])
    runner.update_parameters(changed_params)
    moved = initial._replace(q=initial.q+.07, p=initial.p*jnp.array([1.2, .7]))
    second = runner.run(moved, 3).final_state
    assert runner._compiled == cache
    for result, start, params in ((first, initial, PARAMS), (second, moved, changed_params)):
        for lane in range(2):
            expected = numpy_evolve(params, start.q[lane], start.p[lane], start.electronic[lane],
                                    MASS, integrator.dt, 3, algorithm, substeps)
            for field, value in zip(("q", "p", "electronic"), expected, strict=True):
                np.testing.assert_allclose(getattr(result, field)[lane], value, atol=6e-15, rtol=8e-15)
        np.testing.assert_array_equal(result.trajectory_id, start.trajectory_id)
        np.testing.assert_array_equal(result.key, start.key)
        np.testing.assert_allclose(result.time, start.time+.24, atol=4e-16)
    assert np.max(abs(first.electronic-second.electronic)) > 1e-3


def finite_jacobian(function, point, step=2e-5):
    point = np.asarray(point)
    columns = []
    for i in range(len(point)):
        delta = np.zeros_like(point)
        delta[i] = step
        columns.append((np.asarray(function(point+delta))-np.asarray(function(point-delta)))/(2*step))
    return np.stack(columns, axis=-1)


def test_force_sensitivity_keeps_outer_parameter_geometry_and_electronic_response():
    model = PreparedNonlinearModel()

    def arguments(value, xp):
        params = xp.array([value[0], .23, .6])
        q = xp.array([value[1], -.35])
        c = xp.array([xp.cos(value[2]), xp.exp(.3j)*xp.sin(value[2])])
        return params, q, c

    def native(value):
        params, q, c = arguments(value, jnp)
        return -model.reference_gradient(params, q)-model.contract_gradient(params, q, pure_state_weight(c))

    point = jnp.array([.4, .7, .45])
    def reference(value):
        return numpy_force(*arguments(value, np))
    expected = finite_jacobian(reference, point)
    np.testing.assert_allclose(jax.jacfwd(native)(point), expected, atol=4e-10, rtol=2e-8)
    np.testing.assert_allclose(jax.jacrev(native)(point), expected, atol=4e-10, rtol=2e-8)


@pytest.mark.parametrize("algorithm", ["rk4", "exponential_midpoint"])
def test_prepared_trajectory_native_ad_matches_independent_split_finite_differences(algorithm):
    model = PreparedNonlinearModel()
    integrator = Integrator(.045, algorithm, 3)
    initial = initial_states()[0]

    def trajectory(value):
        params = jnp.array([value[0], value[1], .6])
        q, p = value[2:4], jnp.array([value[4], -.1])
        c = jnp.array([jnp.cos(value[5]), jnp.exp(.3j)*jnp.sin(value[5])])
        state = initial._replace(q=q, p=p, electronic=c)
        problem = Problem(model, params, CoupledClassical(MASS), Ehrenfest())
        step = problem.method.build_step(problem, integrator)
        final = jax.lax.fori_loop(0, 5, lambda _i, s: step(s), state)
        return jnp.concatenate((final.q, final.p, final.electronic.real, final.electronic.imag))

    def reference(value):
        params = np.array([value[0], value[1], .6])
        c = np.array([np.cos(value[5]), np.exp(.3j)*np.sin(value[5])])
        q, p, c = numpy_evolve(params, value[2:4], [value[4], -.1], c, MASS,
                              integrator.dt, 5, algorithm, 3)
        return np.r_[q, p, c.real, c.imag]

    point = jnp.array([.4, .23, .7, -.35, .25, .45])
    expected = finite_jacobian(reference, point)
    np.testing.assert_allclose(trajectory(point), reference(point), atol=6e-15)
    np.testing.assert_allclose(jax.jit(jax.jacfwd(trajectory))(point), expected, atol=5e-10, rtol=3e-8)
    np.testing.assert_allclose(jax.jit(jax.jacrev(trajectory))(point), expected, atol=5e-10, rtol=3e-8)


def changed_apply(model, params, q, vectors):
    extra = .08*(q[0]-q[1])*jnp.array([1., -1.])
    if vectors.ndim == 2:
        extra = extra[:, None]
    return PreparedNonlinearModel.apply(model, params, q, vectors)+extra*vectors


class ApplyOnlySubclass(PreparedNonlinearModel):
    def apply(self, params, q, vectors):
        return changed_apply(self, params, q, vectors)


class WithoutPreparation(PreparedNonlinearModel):
    prepare_action = None


@pytest.mark.parametrize("bad_hook", ["not_callable", "bad_result"])
def test_ordinary_ehrenfest_uses_the_existing_preparation_contract(bad_hook):
    model = PreparedNonlinearModel()
    model.prepare_action = (7 if bad_hook == "not_callable" else lambda _p, _q: 7)
    problem = Problem(model, PARAMS, CoupledClassical(MASS), Ehrenfest())
    step = problem.method.build_step(problem, Integrator(.03, "rk4", 2))
    with pytest.raises(TypeError, match="prepare_action"):
        jax.jit(step)(initial_states()[0])


@pytest.mark.parametrize("kind", ["subclass", "instance", "no_hook"])
def test_existing_apply_override_guard_is_respected_by_complete_ehrenfest_step(kind):
    if kind == "subclass":
        model = ApplyOnlySubclass()
    elif kind == "instance":
        model = PreparedNonlinearModel()
        model.apply = lambda params, q, vectors: changed_apply(model, params, q, vectors)
    else:
        model = WithoutPreparation()
    initial = initial_states()[0]
    integrator = Integrator(.08, "rk4", 2)
    result = Simulation(Problem(model, PARAMS, CoupledClassical(MASS), Ehrenfest()), integrator).run(initial, 4)
    expected = numpy_evolve(PARAMS, initial.q, initial.p, initial.electronic,
                            MASS, .08, 4, "rk4", 2, changed=kind != "no_hook")
    for field, value in zip(("q", "p", "electronic"), expected, strict=True):
        np.testing.assert_allclose(getattr(result.final_state, field), value, atol=5e-15)


def test_ordinary_prepared_step_retains_float32_complex64_support():
    model = PreparedNonlinearModel()
    initial = initial_states()[0]._replace(q=jnp.array([.7, -.35], jnp.float32),
        p=jnp.array([.25, -.1], jnp.float32), electronic=jnp.array([1., 0.], jnp.complex64),
        time=jnp.array(.4, jnp.float32))
    params, mass = PARAMS.astype(jnp.float32), MASS.astype(jnp.float32)
    problem = Problem(model, params, CoupledClassical(mass), Ehrenfest())
    step = jax.jit(problem.method.build_step(problem, Integrator(.08, "rk4", 3)))
    actual = step(initial)
    expected = numpy_evolve(params, initial.q, initial.p, initial.electronic, mass, .08, 1, "rk4", 3)
    assert actual.q.dtype == jnp.float32 and actual.electronic.dtype == jnp.complex64
    for field, value in zip(("q", "p", "electronic"), expected, strict=True):
        np.testing.assert_allclose(getattr(actual, field), value, atol=2e-7, rtol=2e-6)


class DomainLimitedCallback(AutoDiffModel):
    spec = ModelSpec(SystemSpec(2, (1,), coordinate_kind="canonical"),
                     name="ordinary-ehrenfest-callback", native_jax=False)
    execution_mode = "host_callback"

    def matrix(self, params, q):
        def callback(x):
            if float(x[0]) > .35:
                raise RuntimeError("prepared provider left its supported geometry domain")
            return np.array([[0., .2], [.2, 0.]], dtype=x.dtype)
        return jax.pure_callback(callback, jax.ShapeDtypeStruct((2, 2), q.dtype), q,
                                 vmap_method="sequential")

    def apply(self, params, q, vectors):
        return self.matrix(params, q)@vectors

    def prepare_action(self, params, q):
        matrix = self.matrix(params, q)
        return lambda vectors: matrix@vectors

    def reference_energy(self, params, q):
        return jnp.zeros((), q.dtype)

    def contract_gradient(self, params, q, weight):
        return jnp.zeros_like(q)


def test_genuine_callback_failure_retains_last_published_chunk_and_all_state_identity():
    model = DomainLimitedCallback()
    runner = Simulation(Problem(model, None, CoupledClassical(1.), Ehrenfest()), Integrator(.1, "rk4", 3),
                        Execution(chunk_size=2, allow_host_callbacks=True))
    initial = make_state([0.], [1.], [1., 0.], trajectory_id=19,
                          method_state={"marker": jnp.array(73)})
    published = []
    with pytest.raises(SimulationError, match="execution") as caught:
        runner.run(initial, 6, observer=lambda time, _values: published.extend(np.asarray(time).tolist()))
    error = caught.value
    assert error.failed_state is None and error.__cause__ is not None
    state = error.last_valid_state
    assert int(state.step) == 2 and int(state.trajectory_id) == 19
    assert int(state.method_state["marker"]) == 73
    np.testing.assert_array_equal(state.key, initial.key)
    np.testing.assert_allclose(state.q, [.2], atol=1e-15)
    np.testing.assert_allclose(state.p, [1.], atol=1e-15)
    np.testing.assert_allclose(published, [0., .1, .2], atol=1e-15)
    # Failure releases the Runner lifecycle guard. No assertion about how many
    # pure callbacks tracing or optimization executed is required.
    recovered = runner.run(initial, 2).final_state
    for a, b in zip(jax.tree.leaves(recovered), jax.tree.leaves(state), strict=True):
        np.testing.assert_array_equal(a, b)


def test_optional_torch_local_prepared_halves_match_unprepared_complete_trajectory():
    torch = pytest.importorskip("torch")
    from pyeph.adapters.torch_local import TorchLocalBlockModel
    from pyeph.models.local import AtomCenterMap, LocalBlockGraph, LocalCoefficients

    def head(params, q, geometry):
        return LocalCoefficients((params["bias"]*q[:, 0]**2)[:, None, None],
                                  (params["hop"]*torch.exp(-.3*geometry.distances))[:, None, None])

    class UnpreparedTorch(TorchLocalBlockModel):
        prepare_action = None

    graph, centers = LocalBlockGraph(2, 1, ((0, 1),)), AtomCenterMap((0, 1), (1., 1.), 2)
    params = dict(bias=jnp.array(.08), hop=jnp.array(.15))
    initial = make_state([[0., .1, 0.], [1.3, .2, .1]], [[.1, -.02, .03], [-.1, .03, -.04]],
                          [1/np.sqrt(2), 1j/np.sqrt(2)])
    results = []
    for model in (TorchLocalBlockModel(graph, centers, head), UnpreparedTorch(graph, centers, head)):
        problem = Problem(model, params, CoupledClassical(jnp.array([[1.3], [2.1]])), Ehrenfest())
        result = Simulation(problem, Integrator(.04, "rk4", 3),
                            Execution(chunk_size=2, allow_host_callbacks=True)).run(initial, 5)
        results.append(result)
    for a, b in zip(jax.tree.leaves(results[0].final_state), jax.tree.leaves(results[1].final_state), strict=True):
        np.testing.assert_allclose(a, b, atol=3e-15)
    jax.tree.map(lambda a, b: np.testing.assert_allclose(a, b, atol=3e-15),
                 results[0].observables, results[1].observables)
