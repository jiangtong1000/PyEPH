"""Independent complex trace, path, factor and origin checks for CPA columns.

The SciPy reference evolves a full small U only in the oracle. The larger
native fixture explicitly forbids full-matrix actions. Random-phase checks
concern normalized infinite-temperature trace sampling, not finite-temperature
filtering or the statistical error of nuclear/trace mixtures.
"""

from dataclasses import dataclass, field, replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.integrate import solve_ivp
from scipy.linalg import expm

from pyeph.core.contracts import ModelSpec
from pyeph.core.problem import PrescribedPath
from pyeph.core.state import stack_states
from pyeph.core.system import SystemSpec
from pyeph.execution.runner import Execution, SimulationError
from pyeph.integrators.electronic import Integrator
from pyeph.integrators.krylov import LanczosOptions
from pyeph.paths.harmonic import ConstantPath, HarmonicBath, HarmonicPath
from pyeph.simulation import Simulation
from pyeph.workflows.column_transport import (
    initialize_column_transport_state,
    initialize_infinite_temperature_columns,
    make_column_transport_problem,
)


PROBES = ("left", "right")


@dataclass(frozen=True)
class SmallActionModel:
    states: int = 3
    spec: object = field(init=False)

    def __post_init__(self):
        object.__setattr__(self, "spec", ModelSpec(SystemSpec(self.states, (1,)),
            name="column-complex-oracle", complex_valued=True, force_support=False))

    def apply(self, params, q, vectors):
        return (params["h0"] + q[0]*params["h1"]) @ vectors

    def probe_apply(self, *args):
        raise AssertionError("the declared physical action callback must be used")


def physical_current(params, context, probe, vectors):
    index = PROBES.index(probe)
    current = (params["j0"][index] + context.q[0]*params["jq"][index]
               + context.velocity[0]*params["jv"][index] + context.time*params["jt"][index])
    return current @ vectors


def _params(states=3):
    rng = np.random.default_rng(24831)

    def hermitian(shape, scale):
        raw = rng.normal(size=shape) + 1j*rng.normal(size=shape)
        return jnp.asarray(scale*(raw+raw.conj().swapaxes(-1, -2))/2)

    return dict(h0=hermitian((states, states), .5), h1=hermitian((states, states), .3),
                j0=hermitian((2, states, states), .7), jq=hermitian((2, states, states), .2),
                jv=hermitian((2, states, states), .4), jt=hermitian((2, states, states), .1))


def _factor(states=3, rank=2):
    rng = np.random.default_rng(1834)
    factor = rng.normal(size=(states, rank))+1j*rng.normal(size=(states, rank))
    return factor/np.linalg.norm(factor)


def _problem(params=None, treatment=None, states=3):
    return make_column_transport_problem(SmallActionModel(states), _params(states) if params is None else params,
        PrescribedPath(ConstantPath([.2])) if treatment is None else treatment,
        probes=PROBES, probe_callback=physical_current)


def _numpy_currents(params, q, velocity, time):
    return (np.asarray(params["j0"])+q*np.asarray(params["jq"])
            + velocity*np.asarray(params["jv"])+time*np.asarray(params["jt"]))


def _trace_oracle(params, factor, times, position, velocity):
    n = factor.shape[0]
    h0, h1 = np.asarray(params["h0"]), np.asarray(params["h1"])
    t0 = times[0]
    solution = solve_ivp(lambda t, u: (-1j*(h0+position(t)*h1) @ u.reshape(n, n)).reshape(-1),
                         (t0, times[-1]), np.eye(n, dtype=complex).reshape(-1),
                         method="DOP853", t_eval=times, rtol=2e-13, atol=2e-15)
    assert solution.success
    rho = factor @ factor.conj().T
    currents0 = _numpy_currents(params, position(t0), velocity(t0), t0)
    answer = []
    for time, flat in zip(times, solution.y.T, strict=True):
        u = flat.reshape(n, n)
        current = _numpy_currents(params, position(time), velocity(time), time)
        answer.append([[np.trace(ja @ u @ jb @ rho @ u.conj().T) for jb in currents0]
                       for ja in current])
    return np.asarray(answer)


def test_complex_cross_components_match_independent_time_varying_scipy_path_and_moving_current():
    time0 = 2.3
    q0, p0, mass, omega = .37, .28, 1.7, .8
    path = HarmonicPath([q0], [p0], [omega], [mass], origin=time0)
    problem = _problem(treatment=PrescribedPath(path))
    factor = _factor()
    state = initialize_column_transport_state(problem, [q0], [p0], factor, time=time0, trajectory_id=41)
    result = Simulation(problem, Integrator(.01, electronic_substeps=2),
                        Execution(chunk_size=20, save_every=5)).run(state, 60)
    def position(t):
        return q0*np.cos(omega*(t-time0))+p0/(mass*omega)*np.sin(omega*(t-time0))

    def velocity(t):
        return -q0*omega*np.sin(omega*(t-time0))+p0/mass*np.cos(omega*(t-time0))
    exact = _trace_oracle(problem.params, factor, result.times, position, velocity)
    actual = result.observables["current_correlation"]
    assert actual.shape == (13, 2, 2)
    assert np.max(abs(exact.imag)) > .1
    assert np.max(abs(exact[:, 0, 1]-exact[:, 1, 0])) > .1
    np.testing.assert_allclose(actual, exact, atol=3e-10, rtol=2e-9)
    np.testing.assert_allclose(result.observables["lag_time"], result.times-time0, atol=4e-15)
    # A callback that ignored velocity would produce a different physical observable.
    wrong = _trace_oracle(problem.params, factor, result.times, position, lambda t: 0.)
    assert np.max(abs(actual-wrong)) > .03


def test_unsymmetrized_imaginary_pauli_cross_correlation_is_not_discarded():
    params = _params(2)
    params = {key: jnp.zeros_like(value) for key, value in params.items()}
    params["j0"] = jnp.array([[[0., 1.], [1., 0.]], [[0., -1j], [1j, 0.]]])
    problem = _problem(params, states=2)
    factor = np.diag(np.sqrt([.8, .2])).astype(complex)
    state = initialize_column_transport_state(problem, [.2], [0.], factor)
    result = Simulation(problem, Integrator(.03)).run(state, 2)
    expected = np.array([[1., .6j], [-.6j, 1.]])
    np.testing.assert_allclose(result.observables["current_correlation"], np.broadcast_to(expected, (3, 2, 2)), atol=2e-15)


@pytest.mark.parametrize("transformation", ["phase", "factor_columns", "electronic_basis"])
def test_correlation_depends_on_density_and_physical_basis_not_factor_gauge(transformation):
    params, factor = _params(), _factor()
    problem = _problem(params)
    original = initialize_column_transport_state(problem, [.2], [0.], factor)
    integrator = Integrator(.03, electronic_substeps=2)
    expected = Simulation(problem, integrator).run(original, 8)
    rng = np.random.default_rng(319)
    if transformation == "phase":
        factor = factor*np.exp(.83j)
    elif transformation == "factor_columns":
        u, _ = np.linalg.qr(rng.normal(size=(2, 2))+1j*rng.normal(size=(2, 2)))
        factor = factor @ u
    else:
        u, _ = np.linalg.qr(rng.normal(size=(3, 3))+1j*rng.normal(size=(3, 3)))
        params = {name: jnp.asarray(u.conj().T @ np.asarray(value) @ u) for name, value in params.items()}
        factor = u.conj().T @ factor
    transformed = _problem(params)
    initial = initialize_column_transport_state(transformed, [.2], [0.], factor)
    result = Simulation(transformed, integrator).run(initial, 8)
    np.testing.assert_allclose(result.observables["current_correlation"],
                               expected.observables["current_correlation"], atol=3e-14, rtol=2e-13)


@dataclass(frozen=True)
class MatrixFreeRing:
    states: int = 128
    max_columns: int = 6
    spec: object = field(init=False)

    def __post_init__(self):
        object.__setattr__(self, "spec", ModelSpec(SystemSpec(self.states, (1,)),
            name="column-action-only-ring", complex_valued=True, force_support=False, probes=PROBES))

    def _check(self, vectors):
        if vectors.ndim == 2 and vectors.shape[1] > self.max_columns:
            raise AssertionError("unexpected dense/identity/full-matrix action")

    def apply(self, params, q, vectors):
        self._check(vectors)
        diagonal = params["diagonal"]+q[0]*params["coupling"]
        if vectors.ndim == 2:
            diagonal = diagonal[:, None]
        return diagonal*vectors + .12*(jnp.roll(vectors, 1, axis=0)+jnp.roll(vectors, -1, axis=0))

    def probe_apply(self, params, context, probe, vectors):
        self._check(vectors)
        result = .12j*(jnp.roll(vectors, -1, axis=0)-jnp.roll(vectors, 1, axis=0))
        if probe == "right":
            diagonal = params["coupling"]*context.q[0]
            result = result + (diagonal[:, None] if vectors.ndim == 2 else diagonal)*vectors
        return result

    def dense(self, *args):
        raise AssertionError("no dense Hamiltonian is available")


@pytest.mark.parametrize("checked", [False, True])
def test_larger_operator_only_problem_keeps_rank_scaled_state_and_never_requests_identity(checked):
    model = MatrixFreeRing()
    params = dict(diagonal=jnp.linspace(-.8, .7, model.states), coupling=jnp.cos(jnp.arange(model.states)*.2))
    problem = make_column_transport_problem(model, params, PrescribedPath(ConstantPath([.2])), probes=PROBES)
    factor = np.zeros((model.states, 2), complex)
    factor[0, 0], factor[7, 1] = np.sqrt(.3), np.sqrt(.7)
    state = initialize_column_transport_state(problem, [.2], [0.], factor, trajectory_id=71)
    assert state.electronic.shape == (model.states, 6)
    assert all(np.shape(leaf) != (model.states, model.states) for leaf in jax.tree.leaves(state))
    electronic = LanczosOptions(max_dimension=12) if checked else "rk4"
    result = Simulation(problem, Integrator(.02, electronic, electronic_substeps=2),
                        Execution(chunk_size=3)).run(state, 6)
    assert result.final_state.electronic.shape == (model.states, 6)
    assert result.observables["current_correlation"].shape == (7, 2, 2)
    assert np.isfinite(result.observables["current_correlation"]).all()


def test_checked_columns_keep_zero_insertions_and_report_per_column_failure_without_advancing():
    params = _params()
    for name in ("j0", "jq", "jv", "jt"):
        params[name] = params[name].at[1].set(0.)
    problem = _problem(params)
    factor = np.zeros((3, 2), complex)
    factor[0, 0] = 1.
    state = initialize_column_transport_state(problem, [.2], [0.], factor)
    zero_columns = [1, 3, 4, 5]
    np.testing.assert_array_equal(state.electronic[:, zero_columns], 0.)
    runner = Simulation(problem, Integrator(.1, LanczosOptions(max_dimension=3)), Execution(chunk_size=2))
    result = runner.run(state, 4)
    h = np.asarray(params["h0"]+.2*params["h1"])
    rho, j0 = factor @ factor.conj().T, _numpy_currents(params, .2, 0., 0.)
    expected = []
    for time in result.times:
        u = expm(-1j*time*h)
        jt = _numpy_currents(params, .2, 0., time)
        expected.append([[np.trace(ja @ u @ jb @ rho @ u.conj().T) for jb in j0] for ja in jt])
    np.testing.assert_allclose(result.observables["current_correlation"], expected, atol=3e-14)
    np.testing.assert_array_equal(result.final_state.electronic[:, zero_columns], 0.)
    np.testing.assert_array_equal(result.observables["column_norm_squared_drift"][:, zero_columns], 0.)
    assert np.max(abs(result.observables["column_norm_squared_drift"])) < 2e-14

    options = LanczosOptions(max_dimension=1, atol=1e-13, rtol=1e-12)
    rejected = Simulation(problem, Integrator(.1, options), Execution(chunk_size=2))
    with pytest.raises(SimulationError) as caught:
        rejected.run(state, 4)
    error = caught.value
    info = error.diagnostics["step_info"]
    assert info.action.status.shape == (6,)
    np.testing.assert_array_equal(info.action.status[zero_columns], 0)
    assert info.action.status[0] != 0
    np.testing.assert_allclose(info.macrostep_budget,
                               options.atol+options.rtol*np.linalg.norm(state.electronic, axis=0),
                               atol=1e-27, rtol=2e-15)
    np.testing.assert_array_equal(error.last_valid_state.electronic, state.electronic)
    np.testing.assert_array_equal(error.failed_state.electronic, state.electronic)
    assert error.failed_state.time == error.last_valid_state.time == state.time


def test_batch_permutation_keeps_ids_origins_and_rank_columns_and_exact_checkpoint(tmp_path):
    problem = _problem(treatment=HarmonicBath([.7], masses=[1.3]))
    ids = [91, 3, 400]
    states = [initialize_infinite_temperature_columns(problem, [.1+.1*i], [.04-.02*i],
              trace_ids=[17, 8], seed=41, trajectory_id=identity, time=.4)
              for i, identity in enumerate(ids)]
    integrator, execution = Integrator(.03, electronic_substeps=2), Execution(chunk_size=4, save_every=2)
    runner = Simulation(problem, integrator, execution)
    batch = stack_states(states)
    full = runner.run(batch, 8)
    order = np.array([2, 0, 1])
    reordered = runner.run(stack_states([states[i] for i in order]), 8)
    np.testing.assert_array_equal(reordered.final_state.trajectory_id, np.array(ids)[order])
    np.testing.assert_allclose(reordered.observables["current_correlation"],
                               full.observables["current_correlation"][:, order], atol=2e-15)
    for i, state in enumerate(states):
        single = runner.run(state, 8)
        np.testing.assert_allclose(single.observables["current_correlation"],
                                   full.observables["current_correlation"][:, i], atol=3e-15)
    prefix = runner.run(batch, 4)
    path = tmp_path/"rank_columns.h5"
    artifacts = {"model": "column-test-model-v1", "measurement.probe_callback": "physical-current-v1"}
    runner.save_checkpoint(path, prefix.final_state, artifact_ids=artifacts)
    restored = runner.load_checkpoint(path, artifact_ids=artifacts)
    for got, expected in zip(jax.tree.leaves(restored), jax.tree.leaves(prefix.final_state), strict=True):
        np.testing.assert_array_equal(got, expected)
    suffix = runner.run(restored, 4)
    for key in full.observables:
        np.testing.assert_array_equal(np.concatenate((prefix.observables[key], suffix.observables[key][1:])),
                                      full.observables[key])
    np.testing.assert_array_equal(np.r_[prefix.times, suffix.times[1:]], full.times)
    for got, expected in zip(jax.tree.leaves(suffix.final_state), jax.tree.leaves(full.final_state), strict=True):
        np.testing.assert_array_equal(got, expected)
    np.testing.assert_allclose(full.observables["lag_time"][-1], .24, atol=2e-15)


def test_infinite_temperature_phase_columns_have_correct_scaling_stable_trace_ids_and_mean_density():
    problem = _problem()

    def draw(trace_ids, trajectory_id=27):
        state = initialize_infinite_temperature_columns(problem, [.2], [0.], trace_ids=trace_ids,
                                                        trajectory_id=trajectory_id, seed=83)
        return np.asarray(state.electronic[:, :len(trace_ids)])

    ids = np.array([19, 2, 301, 7], dtype=np.uint32)
    all_columns = draw(ids)
    np.testing.assert_allclose(abs(all_columns), 1/np.sqrt(3*len(ids)), atol=2e-16)
    np.testing.assert_array_equal(draw(ids[[2, 0, 3, 1]]), all_columns[:, [2, 0, 3, 1]])
    np.testing.assert_allclose(draw(ids[:2])*np.sqrt(2), all_columns[:, :2]*2, atol=2e-16)
    assert not np.allclose(draw(ids, trajectory_id=28), all_columns)
    samples = draw(np.arange(4096, dtype=np.uint32))
    empirical_density = samples @ samples.conj().T
    np.testing.assert_allclose(np.diag(empirical_density), np.full(3, 1/3), atol=3e-15)
    # Uniform independent phases give an off-diagonal standard deviation
    # 1/(N sqrt(K)); this deterministic regression uses a conservative 6σ gate.
    off_diagonal = empirical_density-np.eye(3)/3
    assert np.max(abs(off_diagonal)) < 6/(3*np.sqrt(4096))
    with pytest.raises(TypeError, match="beta"):
        initialize_infinite_temperature_columns(problem, [.2], [0.], trace_ids=[1], beta=1.)


def test_origin_identity_rejects_parameter_or_probe_order_changes_before_publication():
    problem = _problem()
    state = initialize_column_transport_state(problem, [.2], [0.], _factor())
    runner = Simulation(problem, Integrator(.03))
    runner.update_parameters({**problem.params, "j0": problem.params["j0"]*1.1})
    published = []
    with pytest.raises(ValueError):
        runner.run(state, 0, observer=lambda times, values: published.append(values))
    assert not published
    swapped = replace(problem, measurement=replace(problem.measurement, probes=PROBES[::-1]))
    with pytest.raises(ValueError):
        Simulation(swapped, Integrator(.03)).run(state, 0, collect=False)
