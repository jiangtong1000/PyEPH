"""Independent scalar-domain failure tests; callbacks are audit instrumentation.

These tests certify stage placement and stored-float box membership only. They
do not turn a coordinate box into a neighbor/support or provider certificate.
"""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pyeph.core.contracts import ModelSpec
from pyeph.core.geometry import CoordinateBox
from pyeph.core.problem import CoupledClassical, Problem
from pyeph.core.state import make_state, stack_states
from pyeph.core.system import SystemSpec
from pyeph.dynamics.checked import PHASE_CPA, PHASE_EHRENFEST_FIRST, PHASE_FORCE_FIRST, PHASE_FORCE_SECOND
from pyeph.dynamics.cpa import CPA
from pyeph.dynamics.ehrenfest import Ehrenfest
from pyeph.execution.runner import Execution, SimulationError
from pyeph.integrators.electronic import Integrator
from pyeph.integrators.krylov import LanczosOptions
from pyeph.io.checkpoint import save_checkpoint
from pyeph.io.provenance import problem_manifest
from pyeph.models.analytic import SpinBosonModel
from pyeph.observables.population import ElectronicPopulation, FunctionalMeasurement
from pyeph.simulation import Simulation


class AuditModel:
    """Constant Hermitian action, with separately injected force failure."""

    spec = ModelSpec(SystemSpec(2, (1,)))

    def __init__(self, *, force_nan=False):
        self.actions, self.forces, self.preflight = [], [], []
        self.force_nan = force_nan

    def validate_at(self, params, q, *, batch=False):
        self.preflight.append(np.array(q, copy=True))

    def apply(self, params, q, vectors):
        jax.debug.callback(lambda value: self.actions.append(float(value)), q[0], ordered=True)
        coupling = params["coupling"]
        return jnp.array([[0., coupling], [coupling, 1.]]) @ vectors

    def reference_gradient(self, params, q):
        jax.debug.callback(lambda value: self.forces.append(float(value)), q[0], ordered=True)
        if self.force_nan == "second":
            return jnp.where(q > .1, jnp.full_like(q, jnp.nan), jnp.zeros_like(q))
        return jnp.full_like(q, jnp.nan) if self.force_nan else jnp.zeros_like(q)

    def contract_gradient(self, params, q, weight):
        return jnp.zeros_like(q)


class ReturningPath:
    """A smooth prescribed excursion with identical initial/final coordinates."""

    prescribed = True

    def validate(self, shape):
        assert shape == (1,)

    def point(self, state, elapsed):
        return jnp.asarray([4 * elapsed * (1 - elapsed)]), jnp.zeros_like(state.p)


def initial(*, q=0., p=0.):
    return make_state([q], [p], [1., 0.], time=2., step=7, trajectory_id=39,
                      seed=19, method_state={"tag": np.array([7, 11], np.int64)})


def integrator(*, dimension=2):
    return Integrator(1., LanczosOptions(max_dimension=dimension, atol=1e-11, rtol=1e-10))


def identical(actual, expected):
    assert jax.tree.structure(actual) == jax.tree.structure(expected)
    for left, right in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
        a, b = np.asarray(left), np.asarray(right)
        assert a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()


def checked(problem, state, *, jit=True, dimension=2):
    step = problem.method.build_checked_step(problem, integrator(dimension=dimension))
    result = (jax.jit(step) if jit else step)(state)
    jax.block_until_ready(result)
    jax.effects_barrier()
    return result


@pytest.mark.parametrize("jit", [False, True])
def test_returning_path_rejects_hidden_excursion_without_action(jit):
    model = AuditModel()
    problem = Problem(model, {"coupling": 0.}, ReturningPath(), CPA(),
                      geometry_guard=CoordinateBox([-.5], [.5]))
    start = initial()
    final, info = checked(problem, start, jit=jit)
    identical(final, start)
    assert int(info.code) == 4 and int(info.phase) == PHASE_CPA and int(info.substep) == 0
    np.testing.assert_array_equal(info.attempted_q, [1.])
    assert not model.actions and not model.forces


def test_direct_cpa_checks_initial_domain_even_if_path_ignores_initial_q():
    model = AuditModel()
    problem = Problem(model, {"coupling": 0.}, ReturningPath(), CPA(),
                      geometry_guard=CoordinateBox([-.5], [1.5]))
    start = initial(q=2.)
    final, info = checked(problem, start)
    identical(final, start)
    assert int(info.code) == 4 and int(info.phase) == 0
    np.testing.assert_array_equal(info.attempted_q, [2.])
    assert not model.actions and not model.forces


@pytest.mark.parametrize("jit", [False, True])
def test_ehrenfest_rejects_drift_before_second_force_or_half(jit):
    model = AuditModel()
    problem = Problem(model, {"coupling": 0.}, CoupledClassical(1.), Ehrenfest(),
                      geometry_guard=CoordinateBox([-.5], [.5]))
    start = initial(p=2.)
    final, info = checked(problem, start, jit=jit)
    identical(final, start)
    assert int(info.code) == 4 and int(info.phase) == PHASE_FORCE_SECOND
    np.testing.assert_array_equal(info.attempted_q, [2.])
    assert model.actions and set(model.actions) == {0.}
    assert model.forces == [0.]


@pytest.mark.parametrize("fault", ["action", "force"])
def test_earlier_numerical_failure_survives_later_outside_geometry(fault):
    model = AuditModel(force_nan=fault == "force")
    problem = Problem(model, {"coupling": 1. if fault == "action" else 0.},
                      CoupledClassical(1.), Ehrenfest(),
                      geometry_guard=CoordinateBox([-.5], [.5]))
    start = initial(p=2.)
    final, info = checked(problem, start, dimension=1 if fault == "action" else 2)
    identical(final, start)
    assert int(info.code) == (1 if fault == "action" else 2)
    assert int(info.phase) == (PHASE_EHRENFEST_FIRST if fault == "action" else PHASE_FORCE_FIRST)
    assert set(model.actions) == {0.}
    assert model.forces == ([] if fault == "action" else [0.])


def test_nonfinite_second_force_stops_second_electronic_half_inside_domain():
    model = AuditModel(force_nan="second")
    problem = Problem(model, {"coupling": 0.}, CoupledClassical(1.), Ehrenfest(),
                      geometry_guard=CoordinateBox([-2.], [2.]))
    start = initial(p=1.)
    final, info = checked(problem, start)
    identical(final, start)
    assert int(info.code) == 2 and int(info.phase) == PHASE_FORCE_SECOND
    np.testing.assert_array_equal(info.attempted_q, [1.])
    assert set(model.actions) == {0.}
    assert model.forces == [0., 1.]


def test_nonfinite_cpa_geometry_is_retained_without_entering_model():
    class NonfinitePath(ReturningPath):
        def point(self, state, elapsed):
            return jnp.full_like(state.q, jnp.nan), state.p

    model = AuditModel()
    problem = Problem(model, {"coupling": 0.}, NonfinitePath(), CPA(),
                      geometry_guard=CoordinateBox([-.5], [.5]))
    start = initial()
    final, info = checked(problem, start)
    identical(final, start)
    assert int(info.code) == 2 and int(info.phase) == PHASE_CPA
    assert np.isnan(np.asarray(info.attempted_q)).all()
    assert float(info.attempted_time) == 2.5
    assert not model.actions and not model.forces


def test_host_domain_rejects_before_preflight_run_save_and_load(tmp_path):
    model = AuditModel()
    problem = Problem(model, {"coupling": 0.}, CoupledClassical(1.), Ehrenfest(),
                      geometry_guard=CoordinateBox([-.5], [.5]))
    sim = Simulation(problem, integrator())
    outside = initial(q=2.)
    identities = {"model": "test-only-audit-provider"}
    with pytest.raises(ValueError):
        sim.run(outside, 0)
    with pytest.raises(ValueError):
        sim.save_checkpoint(tmp_path / "save.npz", outside, artifact_ids=identities)
    # Deliberately use low-level storage to forge a same-manifest state outside
    # the declared domain. Strict high-level load must reject before provider.
    manifest = problem_manifest(problem, integrator(), artifact_ids=identities)
    save_checkpoint(tmp_path / "load.npz", outside, metadata={"simulation_manifest": manifest})
    with pytest.raises(ValueError):
        sim.load_checkpoint(tmp_path / "load.npz", artifact_ids=identities)
    assert not model.preflight and not model.actions and not model.forces


def test_copied_bounds_and_manifest_prevent_mutation_and_wider_restart(tmp_path):
    lower, upper = np.array([-.5]), np.array([.5])
    guard = CoordinateBox(lower, upper)
    model = SpinBosonModel(nmodes=1)
    problem = Problem(model, model.default_params(), CoupledClassical(1.), Ehrenfest(),
                      geometry_guard=guard)
    before = problem_manifest(problem, integrator())
    lower[:] = -10
    upper[:] = 10
    assert problem_manifest(problem, integrator()) == before
    assert not bool(guard.contains(jnp.array([2.], dtype=jnp.float64)))
    sim = Simulation(problem, integrator())
    state = initial()
    path = tmp_path / "checkpoint.npz"
    sim.save_checkpoint(path, state)
    identical(sim.load_checkpoint(path), state)
    wider = Simulation(replace(problem, geometry_guard=CoordinateBox([-2.], [2.])), integrator())
    with pytest.raises(ValueError, match="provenance"):
        wider.load_checkpoint(path)


@pytest.mark.parametrize("mode", ["functional", "subclass"])
@pytest.mark.parametrize("collect", [False, True])
def test_unsupported_measurement_rejects_before_custom_hooks(mode, collect):
    calls = []

    class CustomPopulation(ElectronicPopulation):
        def validate(self, problem):
            calls.append("validate")
            raise AssertionError("unsupported measurement hook entered")

    measurement = (FunctionalMeasurement(lambda problem, state: {"q": state.q + 100.})
                   if mode == "functional" else CustomPopulation())
    model = AuditModel()
    problem = Problem(model, {"coupling": 0.}, CoupledClassical(1.), Ehrenfest(), measurement,
                      geometry_guard=CoordinateBox([-.5], [.5]))
    with pytest.raises((ValueError, TypeError)):
        Simulation(problem, integrator()).run(initial(), 0, collect=collect)
    assert not calls and not model.preflight and not model.actions


def test_guarded_initial_population_overflow_is_not_published():
    model = AuditModel()
    problem = Problem(model, {"coupling": 0.}, ReturningPath(), CPA(),
                      geometry_guard=CoordinateBox([-.5], [.5]))
    start = initial()._replace(electronic=jnp.array([1e200 + 0j, 0j], dtype=jnp.complex128))
    published = []
    sim = Simulation(problem, integrator(), Execution(check_finite=False))
    with pytest.raises(SimulationError):
        sim.run(start, 0, observer=lambda t, values: published.append((t, values)))
    assert not published


def test_direct_guarded_batch_is_rejected():
    model = AuditModel()
    problem = Problem(model, {"coupling": 0.}, CoupledClassical(1.), Ehrenfest(),
                      geometry_guard=CoordinateBox([-.5], [.5]))
    with pytest.raises(ValueError):
        problem.method.build_checked_step(problem, integrator(), batch=True)
    batch = stack_states([initial(), initial()._replace(trajectory_id=jnp.uint32(40))])
    with pytest.raises(ValueError):
        Simulation(problem, integrator()).run(batch, 0, collect=False)
    assert not model.preflight and not model.actions


def test_direct_checked_builder_rejects_unqualified_guard_callable():
    class AcceptAnything:
        def contains(self, q):
            return jnp.asarray(True)

    model = AuditModel()
    problem = Problem(model, {"coupling": 0.}, CoupledClassical(1.), Ehrenfest(),
                      geometry_guard=AcceptAnything())
    with pytest.raises((TypeError, ValueError)):
        problem.method.build_checked_step(problem, integrator())
    assert not model.actions and not model.forces


def test_integer_membership_preserves_subnormal_and_huge_origin_boundaries():
    tiny = np.nextafter(np.float64(0.), np.float64(1.))
    zero_box = CoordinateBox([-0.], [0.])
    tiny_box = CoordinateBox([tiny], [tiny])
    enormous = np.float64(1e300)
    high = np.nextafter(enormous, np.inf)
    large_box = CoordinateBox([enormous], [high])
    for guard, q, expected in (
        (zero_box, [-0.], True), (zero_box, [0.], True),
        (zero_box, [tiny], False), (zero_box, [-tiny], False),
        (tiny_box, [tiny], True), (tiny_box, [0.], False),
        (large_box, [enormous], True), (large_box, [high], True),
        (large_box, [np.nextafter(enormous, -np.inf)], False),
        (large_box, [np.nextafter(high, np.inf)], False),
        (large_box, [np.inf], False), (large_box, [np.nan], False),
    ):
        result = jax.jit(guard.contains)(jnp.asarray(q, dtype=jnp.float64))
        assert bool(result) is expected
