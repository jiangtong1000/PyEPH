"""Public real multistate RM method contracts, apart from event-oracle tests.

The independent event trajectories live in test_mapping_reference.py and the
all-state force contraction in test_mashrm_impulse.py. These tests exercise
preparation/measurement bases, smooth physical limits, and Runner integration.
"""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.linalg import expm

from pyeph import CoupledClassical, Ehrenfest, Execution, Integrator, Problem, Simulation
from pyeph.core.contracts import ModelSpec
from pyeph.core.problem import PrescribedPath
from pyeph.core.state import make_state, stack_states
from pyeph.core.system import SystemSpec
from pyeph.dynamics.mashrm import (
    MASHRM,
    MASHRMPopulation,
    mapping_state,
    sample_population,
    total_energy,
)
from pyeph.models.base import AutoDiffModel
from pyeph.models.epc import LinearEPCModel
from pyeph.observables.population import ElectronicPopulation
from pyeph.paths.harmonic import ConstantPath


def native_model(*, rotated=False, omega=(0., 0.), reference_offset=0.):
    model = LinearEPCModel(3, 2)
    diagonal = np.diag([-.4, .1, .8])
    angle = 1.2 if rotated else 0.
    rotation = np.array([[np.cos(angle), -np.sin(angle), 0.],
                         [np.sin(angle), np.cos(angle), 0.], [0., 0., 1.]])
    params = model.create_params(rotation@diagonal@rotation.T, np.zeros((2, 3, 3)), omega=omega)
    return model, {**params, "reference_offset": reference_offset}


def problem_for(model, params, *, measurement=None, method=None):
    return Problem(model, params, CoupledClassical(np.array([1., 2.])),
                   method or MASHRM(event_substeps=1),
                   measurement or MASHRMPopulation(include_density=True, include_nuclei=True))


def assert_state_equal(actual, expected, tolerance=2e-12):
    assert jax.tree.structure(actual) == jax.tree.structure(expected)
    for a, b in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
        np.testing.assert_allclose(a, b, atol=tolerance, rtol=0.)


def test_preparation_basis_is_declared_and_active_is_adiabatic():
    model, params = native_model(rotated=True)
    fixed = mapping_state(model, params, [.1, -.2], [.2, .3], [1., 0., 0.], basis="fixed")
    adiabatic = mapping_state(model, params, [.1, -.2], [.2, .3], [1., 0., 0.], basis="adiabatic")
    _, vectors = jnp.linalg.eigh(model.dense(params, fixed.q))
    np.testing.assert_array_equal(fixed.electronic, [1., 0., 0.])
    np.testing.assert_allclose(vectors.conj().T@adiabatic.electronic, [1., 0., 0.], atol=5e-16)
    assert int(fixed.method_state["active"]) == 1
    assert int(adiabatic.method_state["active"]) == 0
    # Complex mapping coefficients are valid for a real Hamiltonian.
    c = np.sqrt([.65, .25, .1])*np.exp(1j*np.array([.2, -.7, 1.3]))
    phased = mapping_state(model, params, [.1, -.2], [.2, .3], c, basis="adiabatic", active=0)
    np.testing.assert_allclose(vectors.conj().T@phased.electronic, c, atol=7e-16)
    with pytest.raises(ValueError):
        mapping_state(model, params, [.1, -.2], [.2, .3], c, basis="adiabatic", active=1)


def test_sampler_stable_ids_and_same_conditional_draw_in_declared_basis():
    model, params = native_model(rotated=True)
    kwargs = dict(population=0, seed=92, trajectory_id=7)
    fixed = sample_population(model, params, [.1, -.2], [.2, .3], basis="fixed", **kwargs)
    adiabatic = sample_population(model, params, [.1, -.2], [.2, .3], basis="adiabatic", **kwargs)
    repeated = sample_population(model, params, [.1, -.2], [.2, .3], basis="fixed", **kwargs)
    other = sample_population(model, params, [.1, -.2], [.2, .3], basis="fixed",
                              population=0, seed=92, trajectory_id=8)
    np.testing.assert_array_equal(fixed.electronic, repeated.electronic)
    np.testing.assert_array_equal(fixed.key, repeated.key)
    assert not np.allclose(fixed.electronic, other.electronic)
    _, vectors = jnp.linalg.eigh(model.dense(params, fixed.q))
    np.testing.assert_allclose(vectors.conj().T@adiabatic.electronic, fixed.electronic, atol=7e-16)
    np.testing.assert_array_equal(fixed.key, adiabatic.key)
    assert np.argmax(abs(fixed.electronic)**2) == 0
    assert np.argmax(abs(vectors.conj().T@adiabatic.electronic)**2) == 0
    assert int(fixed.method_state["active"]) == int(jnp.argmax(abs(vectors.conj().T@fixed.electronic)**2))
    assert int(adiabatic.method_state["active"]) == 0


@pytest.mark.parametrize("basis", ["fixed", "adiabatic"])
def test_measurement_uses_rm_population_and_density_not_amplitudes_or_active_indicator(basis):
    model, params = native_model(rotated=True, reference_offset=1.7)
    state = mapping_state(model, params, [.1, -.2], [.2, .3], [1., 0., 0.], basis="fixed")
    measurement = MASHRMPopulation(basis=basis, include_density=True, include_nuclei=True)
    problem = problem_for(model, params, measurement=measurement)
    result = measurement.evaluate(problem, state)
    c = np.asarray(state.electronic)
    if basis == "adiabatic":
        c = np.asarray(jnp.linalg.eigh(model.dense(params, state.q))[1]).conj().T@c
    expected = 2.4*np.outer(c, c.conj())-(7/15)*np.eye(3)
    np.testing.assert_allclose(result["population"], np.diag(expected).real, atol=7e-16)
    np.testing.assert_allclose(result["density"], expected, atol=7e-16)
    assert not np.allclose(result["population"], abs(c)**2)
    assert not np.allclose(result["population"], np.eye(3)[int(state.method_state["active"])])
    assert np.min(result["population"]) < 0
    np.testing.assert_array_equal(result["q"], state.q)
    np.testing.assert_array_equal(result["p"], state.p)
    assert float(result["mapping_norm"]) == pytest.approx(1.)
    expected_energy = .5*(.2**2+.3**2/2)+1.7+.1
    assert float(result["energy"]) == pytest.approx(expected_energy)
    assert float(total_energy(model, params, state, problem.nuclear_treatment.masses)) == pytest.approx(expected_energy)
    compact = MASHRMPopulation(basis=basis).evaluate(problem, state)
    assert not {"density", "q", "p"} & compact.keys()


def test_constant_hamiltonian_exact_electronic_limit_and_reference_phase_convention():
    model, params = native_model(rotated=True, reference_offset=3.5)
    c = np.sqrt([.7, .2, .1])*np.exp(1j*np.array([.2, .7, -.4]))
    initial = mapping_state(model, params, [.1, -.2], [.3, .6], c,
                            basis="adiabatic", time=1.75, step=23)
    problem = problem_for(model, params)
    result = Simulation(problem, Integrator(.04, "exponential_midpoint"),
                        Execution(chunk_size=10)).run(initial, 10)
    final = result.final_state
    expected_c = expm(-.4j*np.asarray(params["h0"]))@np.asarray(initial.electronic)
    np.testing.assert_allclose(final.electronic, expected_c, atol=8e-14, rtol=0.)
    # The native convention propagates h only, even when Vref is nonzero.
    assert np.linalg.norm(expected_c*np.exp(-1j*3.5*.4)-final.electronic) > .5
    np.testing.assert_allclose(final.q, [.1+.4*.3, -.2+.4*.6/2], atol=2e-15)
    np.testing.assert_array_equal(final.p, initial.p)
    assert int(final.method_state["active"]) == 0
    assert int(final.step) == 33 and float(final.time) == pytest.approx(2.15)
    for field in ("status", "events", "accepted", "frustrated", "event_attempts", "localization_iterations"):
        assert int(final.method_state[field]) == 0
    assert float(final.method_state["max_event_bracket_width"]) == 0.
    np.testing.assert_allclose(result.observables["energy"], float(total_energy(
        model, params, initial, problem.nuclear_treatment.masses)), atol=2e-14)
    expected_density = 2.4*np.outer(expected_c, expected_c.conj())-(7/15)*np.eye(3)
    np.testing.assert_allclose(result.observables["density"][-1], expected_density, atol=2e-13)


def test_uncoupled_born_oppenheimer_harmonic_limit_with_unequal_masses():
    omega = np.array([.7, 1.1])
    masses = np.array([1., 2.])
    model, params = native_model(omega=omega)
    initial = sample_population(model, params, [.7, -.2], [.3, .4], population=1, seed=14)
    problem = problem_for(model, params, measurement=MASHRMPopulation(basis="adiabatic", include_nuclei=True))
    result = Simulation(problem, Integrator(.01, "exponential_midpoint"),
                        Execution(chunk_size=100, save_every=10)).run(initial, 100)
    frequency = omega/np.sqrt(masses)
    exact_q = np.asarray(initial.q)*np.cos(frequency)+np.asarray(initial.p)*np.sin(frequency)/(masses*frequency)
    exact_p = np.asarray(initial.p)*np.cos(frequency)-masses*frequency*np.asarray(initial.q)*np.sin(frequency)
    np.testing.assert_allclose(result.final_state.q, exact_q, atol=6e-6, rtol=0.)
    np.testing.assert_allclose(result.final_state.p, exact_p, atol=6e-6, rtol=0.)
    assert int(result.final_state.method_state["active"]) == 1
    assert int(result.final_state.method_state["events"]) == 0
    np.testing.assert_allclose(result.observables["mapping_norm"], 1., atol=2e-13)
    np.testing.assert_allclose(result.observables["population"],
                               np.broadcast_to(result.observables["population"][0],
                                               result.observables["population"].shape), atol=5e-13)
    np.testing.assert_allclose(result.final_state.electronic,
                               expm(-1j*np.asarray(params["h0"]))@initial.electronic, atol=1e-13)


def test_batch_single_jit_and_strict_restart_agree(tmp_path):
    model, params = native_model(omega=(.4, .8))
    problem = problem_for(model, params)
    initial = [sample_population(model, params, [.2+.1*i, -.1], [.3, .2],
                                population=i, trajectory_id=17+i, seed=4) for i in range(2)]
    batch = stack_states(initial)
    integrator = Integrator(.02, "exponential_midpoint")
    simulation = Simulation(problem, integrator, Execution(chunk_size=4, save_every=2))
    result = simulation.run(batch, 8)
    single = Simulation(problem, integrator, Execution(chunk_size=4)).run(initial[0], 8)
    np.testing.assert_allclose(result.final_state.electronic[0], single.final_state.electronic, atol=2e-13)
    np.testing.assert_allclose(result.final_state.q[0], single.final_state.q, atol=2e-13)
    step = problem.method.build_step(problem, integrator)
    assert_state_equal(step(initial[0]), jax.jit(step)(initial[0]))
    partial = simulation.run(batch, 3)
    path = tmp_path/"mashrm.h5"
    simulation.save_checkpoint(path, partial.final_state)
    resumed = Simulation(problem, integrator, Execution(chunk_size=3, save_every=2))
    loaded = resumed.load_checkpoint(path)
    assert_state_equal(loaded, partial.final_state, tolerance=0.)
    continuation = resumed.run(loaded, 5)
    assert_state_equal(continuation.final_state, result.final_state)
    changed = replace(problem, measurement=MASHRMPopulation(basis="adiabatic"))
    with pytest.raises(ValueError, match="measurement"):
        Simulation(changed, integrator).load_checkpoint(path)


def test_updated_runtime_parameters_reach_cached_step_and_initial_measurement(tmp_path):
    model, params = native_model()
    problem = problem_for(model, params)
    initial = sample_population(model, params, [.1, -.1], [.2, .3], population=0, seed=19)
    simulation = Simulation(problem, Integrator(.04, "exponential_midpoint"), Execution(chunk_size=3))
    old = simulation.run(initial, 3)
    cache = dict(simulation._compiled)
    checkpoint = tmp_path/"old_params.h5"
    simulation.save_checkpoint(checkpoint, old.final_state)
    updated = {**params, "h0": jnp.diag(jnp.array([-.3, .2, .95])), "reference_offset": .7}
    simulation.update_parameters(updated)
    current = simulation.run(initial, 3)
    fresh = Simulation(simulation.problem, simulation.integrator, simulation.execution).run(initial, 3)
    assert simulation._compiled == cache
    assert_state_equal(current.final_state, fresh.final_state)
    assert not np.allclose(current.final_state.electronic, old.final_state.electronic)
    expected_initial_energy = float(total_energy(model, updated, initial, problem.nuclear_treatment.masses))
    assert float(current.observables["energy"][0]) == pytest.approx(expected_initial_energy)
    with pytest.raises(ValueError, match="params"):
        simulation.load_checkpoint(checkpoint)


@pytest.mark.parametrize("mapping", [[2., 0., 0.], [np.nan, 0., 1.], [1., 0.], np.eye(3)])
def test_preparation_rejects_invalid_norm_finiteness_and_shape(mapping):
    model, params = native_model()
    with pytest.raises(ValueError):
        mapping_state(model, params, [0., 0.], [0., 0.], mapping)


def test_basis_and_population_index_rejections_are_explicit():
    model, params = native_model()
    with pytest.raises(ValueError):
        mapping_state(model, params, [0., 0.], [0., 0.], [1., 0., 0.], basis="moving")
    with pytest.raises(ValueError):
        sample_population(model, params, [0., 0.], [0., 0.], population=3)
    with pytest.raises(ValueError):
        MASHRMPopulation(basis="moving")
    # Mutable string scalars must not become static configuration captured by JIT.
    with pytest.raises(ValueError):
        MASHRMPopulation(basis=np.array("fixed"))
    with pytest.raises(ValueError):
        mapping_state(model, params, [0., 0.], [0., 0.], [1., 0., 0.], basis=np.array("fixed"))
    tied = np.sqrt([.4, .4, .2])
    with pytest.raises(ValueError):
        mapping_state(model, params, [0., 0.], [0., 0.], tied)


class MisdeclaredComplexModel(AutoDiffModel):
    spec = ModelSpec(SystemSpec(3, (2,), coordinate_kind="canonical"), name="misdeclared_complex")

    def apply(self, params, q, vectors):
        h = jnp.array([[-.4, .1j, 0.], [-.1j, .1, 0.], [0., 0., .8]])
        return h@vectors

    def reference_energy(self, params, q):
        return jnp.zeros((), dtype=q.dtype)


def test_complex_degenerate_and_single_state_model_preparation_rejections():
    model, params = native_model()
    degenerate = {**params, "h0": jnp.diag(jnp.array([0., 0., .8]))}
    with pytest.raises(ValueError):
        mapping_state(model, degenerate, [0., 0.], [0., 0.], [1., 0., 0.])
    with pytest.raises(ValueError):
        mapping_state(MisdeclaredComplexModel(), None, [0., 0.], [0., 0.], [1., 0., 0.])
    single = LinearEPCModel(1, 2)
    single_params = single.create_params([[0.]], np.zeros((2, 1, 1)), omega=[0., 0.])
    with pytest.raises(ValueError):
        mapping_state(single, single_params, [0., 0.], [0., 0.], [1.])
    complex_model = LinearEPCModel(3, 2, complex_valued=True)
    with pytest.raises(ValueError):
        problem_for(complex_model, params).validate()


@pytest.mark.parametrize("integrator", [Integrator(.1), Integrator(.1, "exponential_midpoint", 2)])
def test_invalid_integrator_is_rejected_before_any_run(integrator):
    model, params = native_model()
    with pytest.raises(ValueError):
        Simulation(problem_for(model, params), integrator)


def test_measurement_and_physical_capability_preflight():
    model, params = native_model()
    problem = problem_for(model, params)
    with pytest.raises(ValueError):
        replace(problem, measurement=None).validate()
    with pytest.raises(ValueError, match="mapping"):
        ElectronicPopulation().validate(problem)
    with pytest.raises(ValueError):
        replace(problem, method=Ehrenfest()).validate()
    with pytest.raises(ValueError):
        replace(problem, nuclear_treatment=PrescribedPath(ConstantPath([0., 0.]))).validate()
    # Opaque provider capability flags must fail before native compilation.
    for spec in (replace(model.spec, native_jax=False), replace(model.spec, force_support=False)):
        class UnsupportedModel(MisdeclaredComplexModel):
            pass

        provider = UnsupportedModel()
        provider.spec = spec
        with pytest.raises((ValueError, TypeError)):
            problem_for(provider, None).validate()


def test_invalid_method_state_is_rejected_before_initial_observation():
    model, params = native_model()
    problem = problem_for(model, params)
    initial = mapping_state(model, params, [0., 0.], [.2, .3], [1., 0., 0.])
    malformed = [make_state(initial.q, initial.p, initial.electronic),
                 initial._replace(electronic=2*initial.electronic),
                 initial._replace(electronic=jnp.eye(3, dtype=complex))]
    simulation = Simulation(problem, Integrator(.01, "exponential_midpoint"))
    for state in malformed:
        observations = []
        with pytest.raises(ValueError):
            simulation.run(state, 0, observer=lambda *data: observations.append(data))
        assert not observations


@pytest.mark.parametrize("steps", [0, 1])
@pytest.mark.parametrize("output", [True, False])
def test_current_model_active_ownership_is_checked_before_output(steps, output):
    model, params = native_model()
    initial = mapping_state(model, params, [0., 0.], [.2, .3], [1., 0., 0.])
    changed = {**params, "h0": jnp.diag(jnp.array([.8, .1, -.4]))}
    simulation = Simulation(problem_for(model, changed), Integrator(.01, "exponential_midpoint"))
    observations = []
    observer = (lambda *data: observations.append(data)) if output else None
    with pytest.raises(ValueError, match="initial active surface"):
        simulation.run(initial, steps, collect=output, observer=observer)
    assert not observations


def test_parameter_update_and_checkpoint_revalidate_active_ownership(tmp_path):
    model, params = native_model()
    initial = mapping_state(model, params, [0., 0.], [.2, .3], [1., 0., 0.])
    simulation = Simulation(problem_for(model, params), Integrator(.01, "exponential_midpoint"))
    simulation.run(initial, 0)
    old_path = tmp_path/"valid_old_params.h5"
    simulation.save_checkpoint(old_path, initial)
    changed = {**params, "h0": jnp.diag(jnp.array([.8, .1, -.4]))}
    simulation.update_parameters(changed)
    with pytest.raises(ValueError, match="initial active surface"):
        simulation.run(initial, 0)
    rejected_path = tmp_path/"inconsistent.h5"
    with pytest.raises(ValueError, match="initial active surface"):
        simulation.save_checkpoint(rejected_path, initial)
    assert not rejected_path.exists()
    with pytest.raises(ValueError, match="params"):
        simulation.load_checkpoint(old_path)
    # Re-preparation is explicit; the runner never silently changes ownership.
    prepared = mapping_state(model, changed, initial.q, initial.p, initial.electronic)
    assert int(prepared.method_state["active"]) == 2
    assert int(initial.method_state["active"]) == 0
    simulation.run(prepared, 0)
    new_path = tmp_path/"valid_new_params.h5"
    simulation.save_checkpoint(new_path, prepared)
    assert_state_equal(simulation.load_checkpoint(new_path), prepared, tolerance=0.)


def test_initial_model_preflight_checks_every_batch_lane_and_allows_pair_boundaries():
    model, params = native_model()
    first = mapping_state(model, params, [0., 0.], [.2, .3], [1., 0., 0.], trajectory_id=1)
    second = mapping_state(model, params, [0., 0.], [.2, .3], [0., 1., 0.], trajectory_id=2)
    invalid = second._replace(method_state={**second.method_state, "active": jnp.int32(0)})
    simulation = Simulation(problem_for(model, params), Integrator(.01, "exponential_midpoint"))
    observations = []
    with pytest.raises(ValueError, match="initial active surface"):
        simulation.run(stack_states([first, invalid]), 0,
                       observer=lambda *data: observations.append(data))
    assert not observations
    simulation.run(stack_states([first, second]), 0)
    pair = mapping_state(model, params, [0., 0.], [.2, .3], np.sqrt([.4, .4, .2]), active=1)
    simulation.run(pair, 0)
    # The host does not decide incoming/outgoing/grazing event direction.
    assert int(pair.method_state["active"]) == 1


class InitialInvalidModel(AutoDiffModel):
    spec = ModelSpec(SystemSpec(3, (2,), coordinate_kind="canonical"), name="initial_invalid")

    def __init__(self, failure):
        self.failure = failure

    def apply(self, params, q, vectors):
        h = jnp.diag(jnp.array([-.4, .1, .8]))
        if self.failure == "complex":
            h = h.astype(complex).at[0, 1].set(.1j).at[1, 0].set(-.1j)
        elif self.failure == "nonhermitian":
            h = h.at[0, 1].set(.1)
        elif self.failure == "degenerate":
            h = h.at[1, 1].set(-.4)
        return h@vectors

    def reference_energy(self, params, q):
        return jnp.asarray(jnp.nan if self.failure == "reference" else 0., dtype=q.dtype)

    def reference_gradient(self, params, q):
        return jnp.full_like(q, jnp.nan if self.failure == "force" else 0.)


@pytest.mark.parametrize("failure", ["complex", "nonhermitian", "degenerate", "reference", "force"])
def test_initial_physical_model_failures_precede_observation(failure):
    model, params = native_model()
    initial = mapping_state(model, params, [0., 0.], [.2, .3], [1., 0., 0.])
    simulation = Simulation(problem_for(InitialInvalidModel(failure), None),
                            Integrator(.01, "exponential_midpoint"))
    observations = []
    with pytest.raises(ValueError):
        simulation.run(initial, 0, observer=lambda *data: observations.append(data))
    assert not observations


@pytest.mark.parametrize("field,dtype", [("events", jnp.int64), ("active", jnp.int16),
                                        ("max_event_residual", jnp.float32),
                                        ("max_event_residual", jnp.complex128)])
def test_noncanonical_method_state_dtypes_fail_before_output(field, dtype):
    model, params = native_model()
    initial = mapping_state(model, params, [0., 0.], [.2, .3], [1., 0., 0.])
    state = initial._replace(method_state={**initial.method_state,
                                         field: initial.method_state[field].astype(dtype)})
    simulation = Simulation(problem_for(model, params), Integrator(.01, "exponential_midpoint"))
    observations = []
    with pytest.raises(ValueError):
        simulation.run(state, 1, observer=lambda *data: observations.append(data))
    assert not observations


@pytest.mark.parametrize("options", [dict(event_substeps=0), dict(max_events_per_step=0),
                                     dict(bisection_iterations=0), dict(event_time_tolerance=0.),
                                     dict(rate_tolerance=-1.), dict(gap_tolerance=np.nan)])
def test_invalid_bounded_method_configuration(options):
    with pytest.raises(ValueError):
        MASHRM(**options)
