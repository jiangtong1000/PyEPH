"""Prepared operators retain physical formulas, runtime inputs and AD graphs."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pyeph import (CPA, CoupledClassical, Ehrenfest, Execution, Integrator, LanczosOptions,
                   Problem, Simulation, make_state, stack_states)
from pyeph.core.contracts import ModelSpec, prepared_action, pure_state_weight
from pyeph.core.system import SystemSpec
from pyeph.execution.runner import SimulationError
from pyeph.models.aggregate import AggregateModel
from pyeph.models.base import AutoDiffModel
from pyeph.models.composite import ReferenceShiftModel, SumModel
from pyeph.models.epc import EdgeEPCModel, LinearEPCModel
from pyeph.models.neural import NeuralResidualModel
from pyeph.models.periodic import PeriodicBlockModel
from pyeph.paths.harmonic import HarmonicBath


class SparseWithoutDense(EdgeEPCModel):
    def dense(self, params, q):
        raise AssertionError("composition must not densify a sparse child")


def fixture(kind):
    if kind == "dense_epc":
        model = LinearEPCModel(3, 2, complex_valued=True)
        h = np.array([[.2, .1j, 0], [-.1j, -.3, .07], [0, .07, .4]])
        params = model.create_params(h, np.stack((h*.1, np.eye(3)*.2)))
        return model, params, jnp.array([.3, -.4])
    if kind in ("edge", "sum", "shift"):
        base = SparseWithoutDense(3, 2, ((0, 1), (1, 2)), complex_valued=True)
        bp = base.default_params() | dict(onsite=jnp.array([.2, -.3, .4]),
            hopping=jnp.array([.1j, .2]),
            onsite_coupling=jnp.array([[.1, .2, -.1], [-.2, .1, .3]]),
            hopping_coupling=jnp.array([[.1j, .2], [.05, -.1j]]))
        q = jnp.array([.3, -.4])
        if kind == "edge":
            return base, bp, q
        nn = NeuralResidualModel(3, (2,), hidden_sizes=(4,), complex_valued=True,
                                 coordinate_kind="normal_mode")
        np_ = nn.init_params(jax.random.key(22), zero_last=False, scale=.2)
        model, params = SumModel((base, nn)), (bp, np_)
        if kind == "sum":
            return model, params, q
        return (ReferenceShiftModel(model, lambda alpha, x: alpha*jnp.sum(x**2)),
                (params, .13), q)
    if kind == "aggregate":
        model = AggregateModel(3, ((0, 1), (1, 2)), complex_valued=True)
        params = model.default_params() | dict(hopping=jnp.array([.1j, .2-.04j]),
            environment=jnp.array([[.03, -.04], [.07, .02]]))
        return model, params, jnp.array([[0., 0., 0.], [2.1, .2, .1], [4., -.1, .2]])
    if kind == "periodic":
        model = PeriodicBlockModel(2, 2, ((0, 1, 0, 0, 0), (0, 1, -1, 0, 0)),
                                   ((4., 0., 0.), (0., 8., 0.), (0., 0., 8.)))
        params = model.default_params() | dict(hopping=jnp.array(
            [[[.03, .01j], [.02, -.01]], [[.05, .02], [-.01j, .03]]]))
        return model, params, jnp.array([[0., 0., 0.], [2.1, .2, .1]])
    if kind == "neural":
        model = NeuralResidualModel(3, (2,), hidden_sizes=(4,), complex_valued=True)
        return model, model.init_params(jax.random.key(23), zero_last=False, scale=.2), jnp.array([.3, -.4])
    raise ValueError(kind)


KINDS = ("dense_epc", "edge", "aggregate", "periodic", "neural", "sum", "shift")


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("block", [False, True])
def test_prepared_vector_block_values_and_linearity(kind, block):
    model, params, q = fixture(kind)
    rng = np.random.default_rng(31)
    shape = (model.nstates, 3) if block else (model.nstates,)
    x = jnp.asarray(rng.normal(size=shape)+1j*rng.normal(size=shape))
    y = jnp.asarray(rng.normal(size=shape)+1j*rng.normal(size=shape))
    alpha, beta = .3+.2j, -.1+.7j
    prepared = prepared_action(model, params, q)
    expected = model.apply(params, q, x)
    np.testing.assert_allclose(prepared(x), expected, atol=2e-14)
    compiled = jax.jit(lambda p, r, v: prepared_action(model, p, r)(v))
    np.testing.assert_allclose(compiled(params, q, x), expected, atol=2e-14)
    np.testing.assert_allclose(prepared(alpha*x+beta*y), alpha*prepared(x)+beta*prepared(y), atol=2e-14)
    np.testing.assert_array_equal(prepared(jnp.zeros_like(x)), jnp.zeros_like(x))


@pytest.mark.parametrize("kind", ["neural", "sum", "shift"])
def test_preparation_preserves_q_parameter_and_vector_derivatives(kind):
    model, params, q = fixture(kind)
    right = jnp.array([.2+.4j, -.7j, .5])
    left = jnp.array([.3j, .4, -.1+.2j])

    def evaluate(p, r, z, *, prepared):
        vectors = z[:3]+1j*z[3:]
        applied = (prepared_action(model, p, r)(vectors) if prepared
                   else model.apply(p, r, vectors))
        return jnp.vdot(left, applied).real

    z = jnp.concatenate((right.real, right.imag))

    def direct(p, r, z):
        return evaluate(p, r, z, prepared=False)

    def staged(p, r, z):
        return evaluate(p, r, z, prepared=True)

    # Parameter tangents exercise every NN/baseline leaf and reference shift.
    tangents = jax.tree.map(lambda x: jnp.ones_like(x)*(
        .013+.019j if jnp.iscomplexobj(x) else .013), params)
    dq, dz = jnp.array([.17, -.23]), jnp.array([.1, -.2, .3, -.4, .2, .1])
    _, forward = jax.jvp(staged, (params, q, z), (tangents, dq, dz))
    reverse = jax.grad(staged, argnums=(0, 1, 2))(params, q, z)
    # JAX's real-output complex cotangents use the real bilinear pairing.
    reverse_projection = sum(jnp.sum(a*b).real for a, b in zip(
        jax.tree.leaves(reverse), jax.tree.leaves((tangents, dq, dz)), strict=True))
    step = 2e-5
    plus = jax.tree.map(lambda p, dp: p+step*dp, params, tangents)
    minus = jax.tree.map(lambda p, dp: p-step*dp, params, tangents)
    finite = (direct(plus, q+step*dq, z+step*dz)-direct(minus, q-step*dq, z-step*dz))/(2*step)
    np.testing.assert_allclose(forward, finite, rtol=1e-7, atol=2e-10)
    np.testing.assert_allclose(reverse_projection, finite, rtol=1e-7, atol=2e-10)
    for actual, expected in zip(jax.tree.leaves(reverse), jax.tree.leaves(
            jax.grad(direct, argnums=(0, 1, 2))(params, q, z)), strict=True):
        np.testing.assert_allclose(actual, expected, atol=2e-14)


def test_prepared_reference_shift_and_force_compensation():
    shifted, params, q = fixture("shift")
    base, alpha = params
    c = jnp.array([.3+.2j, .6, -.5j])
    c = c/jnp.linalg.norm(c)
    actual = prepared_action(shifted, params, q)(c)
    expected = prepared_action(shifted.model, base, q)(c)-alpha*jnp.sum(q**2)*c
    np.testing.assert_allclose(actual, expected, atol=2e-14)
    weight = pure_state_weight(c)
    original_force = (shifted.model.reference_gradient(base, q)
                      + shifted.model.contract_gradient(base, q, weight))
    shifted_force = shifted.reference_gradient(params, q)+shifted.contract_gradient(params, q, weight)
    np.testing.assert_allclose(shifted_force, original_force, atol=2e-14)


class CountedModel(AutoDiffModel):
    spec = ModelSpec(SystemSpec(3, (2,), coordinate_kind="normal_mode"),
                     name="prepared-counted", complex_valued=True)

    def __init__(self, calls):
        self.calls = calls

    def matrix(self, params, q):
        return params["matrix"]+jnp.diag(q[0]*jnp.array([.2, -.1, .4]))+q[1]*params["slope"]

    def apply(self, params, q, vectors):
        return self.matrix(params, q) @ vectors

    def prepare_action(self, params, q):
        # Instrument only tests; production providers remain pure.
        jax.debug.callback(lambda x: self.calls.append(np.array(x)), q, ordered=True)
        matrix = self.matrix(params, q)
        return lambda vectors: matrix @ vectors

    def reference_energy(self, params, q):
        return .1*jnp.sum(q**2)


class UnpreparedModel(AutoDiffModel):
    """An equivalent provider without the optional hook."""
    spec = CountedModel.spec
    prepare_action = None

    def __init__(self, original):
        self.original = original

    def apply(self, params, q, vectors):
        return self.original.apply(params, q, vectors)

    def reference_energy(self, params, q):
        return self.original.reference_energy(params, q)


def counted_parameters():
    return {"matrix": jnp.array([[.2, .1j, .07], [-.1j, -.3, .2], [.07, .2, .4]]),
            "slope": jnp.array([[.1, .04, .03j], [.04, -.1, .06], [-.03j, .06, .2]])}


def simulation(model, params, *, ehrenfest=False, capacity=3, substeps=1):
    method = Ehrenfest() if ehrenfest else CPA()
    nuclei = CoupledClassical(1.3) if ehrenfest else HarmonicBath([.5, .7])
    return Simulation(Problem(model, params, nuclei, method),
        Integrator(.03, LanczosOptions(max_dimension=capacity, atol=1e-11, rtol=1e-10), substeps),
        Execution(chunk_size=2))


def initial_state(*, batch=False, block=False, zeros=False):
    c = np.array([[1., .2j, 0.], [0., .7, 0.], [0., .3j, 0.]]) if block else np.array([1., 0., 0.])
    c = np.zeros_like(c) if zeros else c
    first = make_state([.3, -.4], [.1, .2], c, time=.4, trajectory_id=7)
    if not batch:
        return first
    return stack_states([first, make_state([-.2, .1], [.3, -.1], c, time=1.2, trajectory_id=19)])


@pytest.mark.parametrize("batch,block", [(False, False), (False, True), (True, False), (True, True)])
def test_prepare_once_per_cpa_geometry_not_krylov_vector_or_column(batch, block):
    calls = []
    model, params = CountedModel(calls), counted_parameters()
    initial = initial_state(batch=batch, block=block)
    runner = simulation(model, params, substeps=2)
    result = runner.run(initial, 3, collect=False)
    jax.block_until_ready(result.final_state)
    assert len(calls) == 3*2*(2 if batch else 1)
    reference = simulation(UnpreparedModel(model), params, substeps=2).run(initial, 3, collect=False)
    np.testing.assert_allclose(result.final_state.electronic, reference.final_state.electronic, atol=2e-14)
    np.testing.assert_allclose(result.final_state.q, reference.final_state.q, atol=2e-14)


@pytest.mark.parametrize("batch", [False, True])
def test_prepared_ehrenfest_matches_fallback_with_regular_force_api(batch):
    calls = []
    model, params = CountedModel(calls), counted_parameters()
    initial = initial_state(batch=batch)
    result = simulation(model, params, ehrenfest=True, substeps=2).run(initial, 2, collect=False)
    jax.block_until_ready(result.final_state)
    assert len(calls) == 2*2*2*(2 if batch else 1)
    reference = simulation(UnpreparedModel(model), params, ehrenfest=True, substeps=2).run(
        initial, 2, collect=False)
    for actual, expected in zip(jax.tree.leaves(result.final_state),
                                jax.tree.leaves(reference.final_state), strict=True):
        np.testing.assert_allclose(actual, expected, atol=2e-14)


def test_cached_parameter_update_does_not_reuse_stale_prepared_arrays():
    model, params = CountedModel([]), counted_parameters()
    runner = simulation(model, params)
    initial = initial_state()
    original = runner.run(initial, 4, collect=False)
    cache = dict(runner._compiled)
    changed = params | {"matrix": params["matrix"]*2.7}
    runner.update_parameters(changed)
    updated = runner.run(initial, 4, collect=False)
    fresh = simulation(UnpreparedModel(model), changed).run(initial, 4, collect=False)
    assert runner._compiled == cache
    np.testing.assert_allclose(updated.final_state.electronic, fresh.final_state.electronic, atol=2e-14)
    assert not np.allclose(original.final_state.electronic, updated.final_state.electronic)


@pytest.mark.parametrize("batch,block", [(False, False), (False, True), (True, True)])
def test_zero_states_retain_zero_action_despite_possible_preparation_work(batch, block):
    model = CountedModel([])
    initial = initial_state(batch=batch, block=block, zeros=True)
    result = simulation(model, counted_parameters()).run(initial, 2, collect=False)
    np.testing.assert_array_equal(result.final_state.electronic, initial.electronic)
    np.testing.assert_array_equal(result.final_state.step, initial.step+2)
    # No assertion about preparation call count: tracing/compiler batching may
    # execute preparation even though the numerical action is exactly zero.


def test_rejection_skips_later_preparation_stages():
    calls = []
    model, params = CountedModel(calls), counted_parameters()
    with pytest.raises(SimulationError, match="electronic action rejected") as caught:
        simulation(model, params, capacity=1, substeps=3).run(
            initial_state(batch=True, block=True), 3, collect=False)
    assert len(calls) == 2  # first geometry action only, once for each trajectory
    assert int(caught.value.diagnostics["step_info"].substep) == 0
    assert np.all(np.asarray(caught.value.last_valid_state.step) == 0)


@pytest.mark.parametrize("bad_hook", [3, "dense"])
def test_invalid_preparation_hook_rejected(bad_hook):
    model = UnpreparedModel(CountedModel([]))
    model.prepare_action = bad_hook
    with pytest.raises(TypeError, match="must be callable or None"):
        prepared_action(model, counted_parameters(), jnp.zeros(2))


def test_invalid_preparation_result_rejected():
    model = UnpreparedModel(CountedModel([]))
    model.prepare_action = lambda params, q: jnp.eye(3)
    with pytest.raises(TypeError, match="must return a callable"):
        prepared_action(model, counted_parameters(), jnp.zeros(2))


def test_missing_hook_retains_apply_fallback():
    class Plain:
        def apply(self, params, q, vectors):
            return (params+q[0])*vectors
    np.testing.assert_allclose(prepared_action(Plain(), 2., jnp.array([.3]))(jnp.array([1j, 2.])),
                               jnp.array([2.3j, 4.6]))


def test_more_derived_apply_overrides_inherited_preparation():
    calls = []

    class Base:
        def apply(self, params, q, vectors):
            return 2*vectors

        def prepare_action(self, params, q):
            calls.append("base")
            return lambda vectors: 2*vectors

    class ApplyOnly(Base):
        def apply(self, params, q, vectors):
            return 3*vectors

    class Both(ApplyOnly):
        def apply(self, params, q, vectors):
            return 4*vectors

        def prepare_action(self, params, q):
            calls.append("both")
            return lambda vectors: 4*vectors

    class HookOnly(Base):
        def prepare_action(self, params, q):
            calls.append("hook")
            return lambda vectors: 2*vectors

    vectors, q = jnp.array([1j, 2.]), jnp.zeros(1)
    np.testing.assert_array_equal(prepared_action(ApplyOnly(), None, q)(vectors), 3*vectors)
    assert calls == []
    np.testing.assert_array_equal(prepared_action(Both(), None, q)(vectors), 4*vectors)
    np.testing.assert_array_equal(prepared_action(HookOnly(), None, q)(vectors), 2*vectors)
    assert calls == ["both", "hook"]
    instance = Base()
    instance.apply = lambda params, q, vectors: 5*vectors
    np.testing.assert_array_equal(prepared_action(instance, None, q)(vectors), 5*vectors)
    assert calls == ["both", "hook"]
    instance.prepare_action = lambda params, q: lambda vectors: 5*vectors
    np.testing.assert_array_equal(prepared_action(instance, None, q)(vectors), 5*vectors)


def test_inherited_neural_methods_dispatch_to_overridden_dense():
    class ChangedDense(NeuralResidualModel):
        def dense(self, params, q):
            return 2.3*super().dense(params, q)

    class ChangedApply(NeuralResidualModel):
        def apply(self, params, q, vectors):
            return 1.7*super().apply(params, q, vectors)

    base, params, q = fixture("neural")
    vectors = jnp.array([1., .2j, -.3])
    expected = base.apply(params, q, vectors)
    for cls, scale in ((ChangedDense, 2.3), (ChangedApply, 1.7)):
        model = cls(3, (2,), hidden_sizes=(4,), complex_valued=True)
        prepared = jax.jit(lambda p, r, v: prepared_action(model, p, r)(v))
        np.testing.assert_allclose(prepared(params, q, vectors), scale*expected, atol=2e-14)
        np.testing.assert_allclose(model.apply(params, q, vectors), scale*expected, atol=2e-14)


def test_inherited_sparse_methods_dispatch_to_overridden_elements():
    class ChangedElements(EdgeEPCModel):
        def elements(self, params, q):
            onsite, hopping = super().elements(params, q)
            return onsite*1.8, hopping*1.8

    base, params, q = fixture("edge")
    model = ChangedElements(3, 2, ((0, 1), (1, 2)), complex_valued=True)
    vectors = jnp.array([1., .2j, -.3])
    expected = 1.8*base.apply(params, q, vectors)
    np.testing.assert_allclose(prepared_action(model, params, q)(vectors), expected, atol=2e-14)
    np.testing.assert_allclose(model.apply(params, q, vectors), expected, atol=2e-14)


def test_ordinary_apply_does_not_start_using_a_subclass_prepare_hook():
    class OtherPreparation(NeuralResidualModel):
        def prepare_action(self, params, q):
            raise AssertionError("ordinary apply must retain its existing formula")

    base, params, q = fixture("neural")
    model = OtherPreparation(3, (2,), hidden_sizes=(4,), complex_valued=True)
    vectors = jnp.array([1., .2j, -.3])
    np.testing.assert_array_equal(model.apply(params, q, vectors), base.apply(params, q, vectors))
