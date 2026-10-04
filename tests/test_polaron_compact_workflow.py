"""Public compact LF contracts, explicit-quad compatibility, and strict resume."""

from dataclasses import FrozenInstanceError, replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pyeph.execution.runner import Execution
from pyeph.integrators.electronic import Integrator
from pyeph.io.provenance import problem_manifest
from pyeph.models.epc import LinearEPCModel
from pyeph.models.polaron import lf_phi
from pyeph.observables.transport.polaron_compact import (
    CompactLFSectors, build_compact_lf_sectors, lf_current_correlation_compact,
)
from pyeph.paths.harmonic import HarmonicBath
from pyeph.simulation import Simulation
from pyeph.workflows.polaron_transport import (
    initialize_polaron_transport_state, make_polaron_transport_problem,
)


PAIRS = [(i, j) for i in range(3) for j in range(3)]
IDENTITIES = {"measurement.probe_callback": "test-diagonal-context-current-v1"}


def physical_current(params, context, probe):
    """Toy physical Hermitian current, with units fixed to the model test units."""
    base = jnp.array([[.3, .2+.4j, -.1j], [.2-.4j, -.2, .4j], [.1j, -.4j, .1]])
    current = base*(1+.2*context.q[0]) + jnp.diag(jnp.array([.1, -.3, .2]))*context.velocity[0]
    return current if probe == "x" else current.T*.7+jnp.eye(3)*.13


def legacy_current(params, context, probe):
    return -1j*physical_current(params, context, probe)


def problem_for(**options):
    model = LinearEPCModel(3, 1, complex_valued=True)
    h0 = np.array([[.6, .3+.1j, -.2j], [.3-.1j, -.4, .2], [.2j, .2, .1]])
    coupling = np.array([[[.1, .02j, .04], [-.02j, -.2, -.03j], [.04, .03j, .3]]])
    params = model.create_params(h0, coupling, omega=[.7])
    kwargs = dict(hopping_pairs=PAIRS, probes=("x", "y"), probe_callback=physical_current)
    kwargs.update(options)
    return make_polaron_transport_problem(model, params, HarmonicBath([.7]), [1.7], [.6],
                                          1.2, **kwargs)


def state_for(problem):
    return initialize_polaron_transport_state(problem, [.31], [-.22], time=.4, seed=53)


def test_direct_topology_and_builder_copy_inputs_and_cannot_accept_incomplete_corrections():
    source = np.array(PAIRS)
    direct, built = CompactLFSectors(source, 3), build_compact_lf_sectors(source, 3)
    expected = np.array(source)
    source[:] = 0
    for topology in (direct, built):
        np.testing.assert_array_equal(topology.support_pairs, expected)
        assert topology.nstates == 3
        with pytest.raises(FrozenInstanceError):
            topology.nstates = 4
        with pytest.raises(TypeError):
            topology.support_pairs[0, 0] = 2
    with pytest.raises(TypeError):
        CompactLFSectors(PAIRS, 3, quad_indices=np.empty((0, 4), dtype=int))


@pytest.mark.parametrize("constructor", [CompactLFSectors, build_compact_lf_sectors])
@pytest.mark.parametrize("pairs,nstates", [
    (np.empty((0, 4), dtype=int), 3), (np.empty((0, 2)), 3), ([(0, 1)], 2.0),
    ([(0, 1)], False), ([(0, 1)], 2**31), ([(0, 2)], 2), ([(0, 1)], 0),
    ([(0, 1), (0, 1)], 2), ([(0, 1.1)], 2), ([(False, True)], 2),
])
def test_all_topology_constructors_validate_shapes_types_and_ranges(constructor, pairs, nstates):
    with pytest.raises(ValueError):
        constructor(pairs, nstates)


@pytest.mark.parametrize("field,value", [("quad_indices", np.empty((0, 4), dtype=int)),
                                          ("sector_indices", np.empty(0, dtype=int))])
def test_compact_and_explicit_quad_configuration_is_rejected(field, value):
    with pytest.raises(ValueError, match="cannot be combined"):
        replace(problem_for().measurement, **{field: value})


def test_compact_topology_type_and_model_dimension_are_validated():
    problem = problem_for()
    with pytest.raises(TypeError, match="CompactLFSectors"):
        replace(problem.measurement, compact_topology={"support_pairs": PAIRS})
    wrong_size = replace(problem.measurement, compact_topology=CompactLFSectors(PAIRS, 4))
    with pytest.raises(ValueError, match="dimensions differ"):
        replace(problem, measurement=wrong_size).validate()
    with pytest.raises(TypeError, match="CompactLFSectors"):
        lf_current_correlation_compact(*([jnp.eye(3)]*4), None, .1, .2)
    with pytest.raises(ValueError, match="shape"):
        lf_current_correlation_compact(*([jnp.eye(4)]*4), problem.measurement.compact_topology, .1, .2)


@pytest.mark.parametrize("estimator", ["wrong", "", None, [], 3])
def test_unknown_estimator_is_rejected(estimator):
    with pytest.raises(ValueError, match="estimator must"):
        problem_for(estimator=estimator)


@pytest.mark.parametrize("policy", ["offdiagonal", "legacy_full"])
def test_public_default_matches_full_with_diagonal_context_currents_and_legacy_conversion(policy):
    problems = [problem_for(thermal_policy=policy),
                problem_for(thermal_policy=policy, estimator="full"),
                problem_for(thermal_policy=policy, probe_callback=legacy_current,
                            current_convention="legacy_without_i")]
    assert problems[0].measurement.compact_topology is not None
    assert problems[0].measurement.quad_indices is None
    assert problems[1].measurement.compact_topology is None
    results = [Simulation(problem, Integrator(.025), Execution(chunk_size=4)).run(state_for(problem), 9)
               for problem in problems]
    for result in results[1:]:
        np.testing.assert_allclose(result.observables["current_correlation"],
                                   results[0].observables["current_correlation"], atol=2e-14)
        for left, right in zip(jax.tree.leaves(result.final_state),
                               jax.tree.leaves(results[0].final_state), strict=True):
            np.testing.assert_array_equal(left, right)
    assert set(results[0].observables) == {"current_correlation", "unitary_error"}


def test_initial_support_validation_includes_diagonal_edges_without_correction_quads():
    omitted = [(i, j) for i, j in PAIRS if (i, j) != (0, 0)]
    with pytest.raises(ValueError, match="omit nonzero initial current edges"):
        state_for(problem_for(hopping_pairs=omitted))


def test_compact_configuration_is_immutable_through_jit_and_manifest():
    problem = problem_for()
    source = np.array(PAIRS)
    topology = CompactLFSectors(source, 3)
    problem = replace(problem, measurement=replace(problem.measurement, compact_topology=topology))
    sim = Simulation(problem, Integrator(.025))
    state = state_for(problem)
    before = problem_manifest(problem, sim.integrator, artifact_ids=IDENTITIES)
    result = sim.run(state, 3)
    source[:] = 0
    assert problem_manifest(problem, sim.integrator, artifact_ids=IDENTITIES) == before
    np.testing.assert_array_equal(sim.run(state, 3).observables["current_correlation"],
                                  result.observables["current_correlation"])


def test_direct_custom_quad_subset_keeps_its_declared_sum():
    problem = problem_for(estimator="full")
    quads = np.array([(i, j, i, j) for i, j in PAIRS])
    sectors = np.array([2 if i != j else 0 for i, j in PAIRS])
    measurement = replace(problem.measurement, quad_indices=quads, sector_indices=sectors)
    custom = replace(problem, measurement=measurement).validate()
    assert custom.measurement.compact_topology is None
    state = Simulation(custom, Integrator(.03)).run(state_for(custom), 7).final_state
    u, rho = np.asarray(state.electronic), np.asarray(state.method_state["transport"]["rho0"])
    jt, j0 = np.asarray(measurement.currents(custom, state)), np.asarray(state.method_state["transport"]["currents0"])
    phi0 = float(lf_phi(measurement.frequencies, measurement.couplings, measurement.beta, 0).real)
    phit = complex(lf_phi(measurement.frequencies, measurement.couplings, measurement.beta, .21))
    factor, expected = u.conj() @ rho.T, np.zeros(2, dtype=complex)
    for (i, j, k, ell), sector in zip(quads, sectors, strict=True):
        weight = np.exp((-2+int(i == j)+int(k == ell))*phi0-sector*phit)
        expected += jt[:, i, j]*j0[:, k, ell]*u[j, k]*factor[i, ell]*weight
    actual = measurement.evaluate(custom, state)["current_correlation"]
    np.testing.assert_allclose(actual, expected, atol=2e-14)
    full = problem.measurement.evaluate(problem, state)["current_correlation"]
    assert np.max(np.abs(full-actual)) > .01


def test_strict_restart_retains_compact_topology_and_rejects_estimator_or_callback_change(tmp_path):
    problem, integrator = problem_for(), Integrator(.025)
    sim = Simulation(problem, integrator, Execution(chunk_size=3))
    initial = state_for(problem)
    whole, first = sim.run(initial, 11), sim.run(initial, 4)
    checkpoint = tmp_path/"compact.h5"
    sim.save_checkpoint(checkpoint, first.final_state, artifact_ids=IDENTITIES)
    resume = Simulation(problem_for(), integrator, Execution(chunk_size=5))
    restored = resume.load_checkpoint(checkpoint, artifact_ids=IDENTITIES)
    second = resume.run(restored, 7)
    joined = np.concatenate([first.observables["current_correlation"],
                             second.observables["current_correlation"][1:]])
    np.testing.assert_allclose(joined, whole.observables["current_correlation"], atol=2e-14)
    np.testing.assert_allclose(second.final_state.electronic, whole.final_state.electronic, atol=2e-14)
    with pytest.raises(ValueError, match="checkpoint provenance mismatch"):
        Simulation(problem_for(estimator="full"), integrator).load_checkpoint(checkpoint,
                                                                              artifact_ids=IDENTITIES)
    with pytest.raises(ValueError, match="checkpoint provenance mismatch"):
        resume.load_checkpoint(checkpoint, artifact_ids={"measurement.probe_callback": "different"})
