"""Construction, numerical contracts, and sparse public execution of local blocks."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.linalg import expm

from pyeph.core.contracts import ForceModel, GeometryModel, ProbeContext, prepared_action
from pyeph.core.problem import PrescribedPath, Problem
from pyeph.core.state import make_state
from pyeph.dynamics.cpa import CPA
from pyeph.execution.runner import Execution
from pyeph.integrators.electronic import Integrator
from pyeph.models.local import AtomCenterMap, LocalBlockGraph, LocalBlockModel, LocalCoefficients
from pyeph.models.periodic import PeriodicBlockModel
from pyeph.paths.harmonic import ConstantPath
from pyeph.simulation import Simulation


def constant_coefficients(params, q, geometry):
    return LocalCoefficients(params["onsite"], params["hopping"])


def fixture(*, block=2, edges=((0, 1),), periodic=False):
    cell = np.diag([4.0, 5.0, 6.0]) if periodic else None
    graph = LocalBlockGraph(2, block, edges, cell, switch_on=1.0, cutoff=3.0)
    mapping = AtomCenterMap((0, 0, 1), (0.25, 0.75, 1.0), 2)
    model = LocalBlockModel(graph, mapping, constant_coefficients, complex_valued=True)
    rng = np.random.default_rng(14)
    onsite = rng.normal(size=(2, block, block)) + 1j*rng.normal(size=(2, block, block))
    onsite = (onsite + onsite.conj().transpose(0, 2, 1))*0.025
    hopping = (rng.normal(size=(len(edges), block, block))
               + 1j*rng.normal(size=(len(edges), block, block)))*0.05
    params = dict(onsite=jnp.asarray(onsite), hopping=jnp.asarray(hopping))
    q = jnp.asarray([[-0.3, 0.0, 0.0], [0.1, 0.0, 0.0], [2.0, 0.0, 0.0]])
    return model, params, q


def test_finite_normalization_static_snapshots_and_declared_coordinates():
    edges = [[0, 1]]
    weights = np.array([0.25, 0.75, 1.0])
    sites = np.array([0, 0, 1])
    charge, cutoff = np.array(-1.0), np.array(3.0)
    graph = LocalBlockGraph(np.array(2), 2, edges, cutoff=cutoff, switch_on=1.0)
    centers = AtomCenterMap(sites, weights, 2)
    model = LocalBlockModel(graph, centers, constant_coefficients, charge=charge)
    edges[0][0] = 1
    sites[:] = 1
    weights[:] = 3
    charge[...] = 4
    cutoff[...] = 100
    assert graph.edges == ((0, 1, 0, 0, 0),)
    assert graph.cutoff == 3 and model.charge == -1
    assert centers.atom_site == (0, 0, 1) and centers.weights == (0.25, 0.75, 1.0)
    assert model.spec.system.q_shape == (3, 3) and model.nstates == 4
    assert isinstance(model, GeometryModel) and isinstance(model, ForceModel)


@pytest.mark.parametrize("kwargs", [
    {"nsites": 0}, {"nsites": True}, {"nsites": 2.0}, {"norbitals": -1},
    {"edges": ((1, 0),)}, {"edges": ((0, 0),)}, {"edges": ((0, 2),)},
    {"edges": ((0.0, 1),)}, {"edges": ((False, 1),)},
    {"edges": ((0, 1), (0, 1, 0, 0, 0))},
    {"edges": ((0, 1, 1, 0, 0),)},
    {"edges": ((0, 0, -1, 0, 0),), "cell": np.eye(3)},
    {"edges": ((0, 1, 2**31, 0, 0),), "cell": np.eye(3)},
    {"cell": np.zeros((3, 3))}, {"cell": np.eye(3)*1j},
    {"cell": np.ones((2, 3))}, {"cell": np.eye(3)*np.nan},
    {"switch_on": 1.0}, {"cutoff": 2.0}, {"cutoff": 1, "switch_on": 2},
    {"cutoff": 2, "switch_on": -1}, {"cutoff": np.inf, "switch_on": 1},
])
def test_invalid_graphs_rejected(kwargs):
    arguments = dict(nsites=2, norbitals=1, edges=((0, 1),)) | kwargs
    with pytest.raises((ValueError, TypeError)):
        LocalBlockGraph(**arguments)


@pytest.mark.parametrize("sites,weights,nsites", [
    ((0.0, 1.0), (1, 1), 2), ((False, True), (1, 1), 2),
    ((0, 2), (1, 1), 2), ((0, -1), (1, 1), 2),
    ((0, 1), (1, -1), 2), ((0, 1), (1, np.nan), 2),
    ((0, 1), (1+0j, 1), 2), ((0, 1), (1, 1, 1), 2),
    ((0, 0), (0.5, 0.5), 2), ((0, 1), (2, 1), 2),
    ((), (), 1), ((0,), (1,), 1.0),
])
def test_invalid_center_maps_rejected(sites, weights, nsites):
    with pytest.raises(ValueError):
        AtomCenterMap(sites, weights, nsites)


def test_zero_weight_atom_remains_in_provider_coordinate_space():
    mapping = AtomCenterMap((0, 0, 1), (1.0, 0.0, 1.0), 2)
    q = jnp.asarray([[0.0, 0, 0], [0.3, 0.2, 0.1], [2.0, 0, 0]])

    def provider(params, coordinates, geometry):
        onsite = jnp.array([coordinates[1, 0], -coordinates[1, 0]])[:, None, None]
        return LocalCoefficients(onsite, jnp.zeros((1, 1, 1)))

    model = LocalBlockModel(LocalBlockGraph(2, 1, ((0, 1),)), mapping, provider)
    np.testing.assert_array_equal(mapping.apply(q), q[jnp.array([0, 2])])
    derivative = jax.grad(lambda x: model.apply(None, x, jnp.array([1.0, 0.0]))[0])(q)
    np.testing.assert_array_equal(derivative[1], [1.0, 0, 0])
    model.validate_at(None, q)


@pytest.mark.parametrize("periodic", [False, True])
@pytest.mark.parametrize("block", [1, 2])
def test_sparse_blocks_and_preparation_match_independent_matrix(periodic, block):
    edges = ((0, 1, 0, 0, 0), (0, 0, 1, 0, 0)) if periodic else ((0, 1),)
    model, params, q = fixture(block=block, edges=edges, periodic=periodic)
    # Disable cutoff here so the non-Hermitian self-image block contributes.
    model = replace(model, graph=replace(model.graph, cutoff=None, switch_on=None))
    model.validate_at(params, q)
    matrix = np.zeros((model.nstates, model.nstates), dtype=complex)
    for i, onsite in enumerate(np.asarray(params["onsite"])):
        matrix[i*block:(i+1)*block, i*block:(i+1)*block] += onsite
    for edge, hopping in zip(edges, np.asarray(params["hopping"])):
        i, j = edge[:2]
        matrix[i*block:(i+1)*block, j*block:(j+1)*block] += hopping
        matrix[j*block:(j+1)*block, i*block:(i+1)*block] += hopping.conj().T
    vectors = jnp.arange(model.nstates*3).reshape(model.nstates, 3)*0.2 + 0.1j
    for v in (vectors, vectors[:, 0]):
        actual = jax.jit(lambda p, x, y: prepared_action(model, p, x)(y))(params, q, v)
        np.testing.assert_allclose(actual, matrix @ v, atol=2e-14)
        np.testing.assert_allclose(model.apply(params, q, v), actual, atol=2e-14)
    np.testing.assert_array_equal(model.reference_gradient(params, q), np.zeros_like(q))


def test_final_hopping_support_applied_once_after_provider_composition():
    model, params, q = fixture(block=1)
    assert float(model.geometry(q).support[0]) == 0.5
    coefficients = model.coefficients(params, q)
    np.testing.assert_allclose(coefficients.hopping, params["hopping"]*0.5, atol=0)
    np.testing.assert_array_equal(coefficients.onsite, params["onsite"])
    finite = replace(model, graph=replace(model.graph, cutoff=None, switch_on=None))
    np.testing.assert_array_equal(finite.geometry(q).support, [1.0])


def test_empty_graph_and_zero_column_block_are_valid():
    model, params, q = fixture(block=2, edges=())
    model.validate_at(params, q)
    assert model.geometry(q).pairs.shape == (0, 2)
    assert model.geometry(q).displacements.shape == (0, 3)
    assert model.apply(params, q, jnp.empty((4, 0))).shape == (4, 0)
    np.testing.assert_array_equal(model.probe_apply(params, ProbeContext(q), "current_x", jnp.ones(4)), 0)


@pytest.mark.parametrize("defect", ["shape", "type", "bool", "complex", "nan", "hermitian"])
def test_invalid_provider_outputs_fail_explicit_preflight(defect):
    model, params, q = fixture(block=2)
    if defect == "type":
        model = replace(model, coefficient_provider=lambda p, x, g: (p["onsite"], p["hopping"]))
    elif defect == "shape":
        params = params | {"hopping": jnp.zeros((1, 2))}
    elif defect == "bool":
        params = params | {"onsite": jnp.zeros((2, 2, 2), dtype=bool)}
    elif defect == "complex":
        model = replace(model, complex_valued=False)
    elif defect == "nan":
        params = params | {"hopping": params["hopping"].at[0, 0, 0].set(jnp.nan)}
    elif defect == "hermitian":
        params = params | {"onsite": params["onsite"].at[0, 0, 1].add(0.3)}
    with pytest.raises((TypeError, ValueError)):
        model.validate_at(params, q)


def test_shape_and_declared_complex_checks_also_reject_during_tracing():
    model, params, q = fixture()
    wrong_shape = params | {"hopping": params["hopping"][..., 0]}
    with pytest.raises(ValueError, match="shape"):
        jax.jit(model.apply)(wrong_shape, q, jnp.ones(4))
    real = replace(model, complex_valued=False)
    with pytest.raises(ValueError, match="complex"):
        jax.jit(real.apply)(params, q, jnp.ones(4))


@pytest.mark.parametrize("qkind", ["shape", "nan", "complex", "integer", "coincident", "overflow"])
def test_mapped_geometry_preflight_rejections(qkind):
    model, params, q = fixture()
    if qkind == "shape":
        q = q[:-1]
    elif qkind == "nan":
        q = q.at[0, 0].set(jnp.nan)
    elif qkind == "complex":
        q = q.astype(complex)
    elif qkind == "integer":
        q = q.astype(int)
    elif qkind == "coincident":
        q = jnp.zeros_like(q)
    else:
        q = q.at[2, 0].set(1e300)
    with pytest.raises(ValueError):
        model.validate_at(params, q)


def test_batch_preflight_checks_every_lane_and_each_block_scale():
    model, params, q = fixture()
    batch = jnp.stack((q, q.at[2, 0].set(2.4)))
    model.validate_at(params, batch, batch=True)
    with pytest.raises(ValueError, match="nonzero"):
        model.validate_at(params, batch.at[1].set(0), batch=True)

    def provider(p, x, geometry):
        onsite = jnp.where(x[2, 0] < 2.1, jnp.ones((2, 2, 2))*1e15,
                           jnp.array([[[0.0, 0.1], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]]]))
        return LocalCoefficients(onsite, jnp.zeros((1, 2, 2)))

    invalid = replace(model, coefficient_provider=provider)
    with pytest.raises(ValueError, match="Hermitian"):
        invalid.validate_at(None, batch, batch=True)


def test_provider_parameter_validator_and_bad_optional_hook():
    class Provider:
        def __call__(self, params, q, geometry):
            return constant_coefficients(params, q, geometry)

        def validate_params(self, params):
            if params["label"] != "accepted":
                raise ValueError("provider label")

    model, params, q = fixture()
    model = replace(model, coefficient_provider=Provider())
    with pytest.raises(ValueError, match="label"):
        model.validate_at(params | {"label": "bad"}, q)
    model.validate_at(params | {"label": "accepted"}, q)
    Provider.validate_params = 3
    with pytest.raises(TypeError, match="validate_params"):
        model.validate_params(params)


def test_rewrap_keeps_provider_and_map_and_guards_integer_overflow():
    model, params, q = fixture(edges=((0, 1, -1, 0, 0),), periodic=True)
    shifts = np.array([[1, 0, 0], [-1, 0, 0]])
    wrapped = model.rewrapped(shifts)
    assert wrapped.coefficient_provider is model.coefficient_provider
    assert wrapped.centers is model.centers
    shifted_q = q + shifts[np.asarray(model.centers.atom_site)] @ np.asarray(model.graph.cell)
    np.testing.assert_allclose(wrapped.geometry(shifted_q).displacements, model.geometry(q).displacements, atol=1e-14)
    with pytest.raises(ValueError, match="integer"):
        model.rewrapped(shifts.astype(float))
    with pytest.raises(ValueError, match="int32"):
        model.rewrapped(np.array([[2**31+1, 0, 0], [0, 0, 0]], dtype=np.int64))


def test_float32_cell_representability_matches_native_geometry():
    model, params, q = fixture(edges=((0, 1, 1, 0, 0),), periodic=True)
    model = replace(model, graph=replace(model.graph, cell=np.diag([1e40, 1.0, 1.0])))
    with pytest.raises(ValueError, match="representable"):
        model.validate_at(params, q.astype(jnp.float32))


@pytest.mark.parametrize("valid", [False, True])
def test_finite_extreme_complex_components_do_not_overflow_hermiticity_check(valid):
    model, params, q = fixture()
    onsite = np.zeros((2, 2, 2), dtype=complex)
    onsite[0, 0, 1] = complex(1.5e308, 1.5e308)
    onsite[0, 1, 0] = complex(1.5e308, -1.5e308 if valid else 1.5e308)
    params = params | {"onsite": jnp.asarray(onsite)}
    with np.errstate(over="raise", invalid="raise"):
        if valid:
            model.validate_at(params, q)
        else:
            with pytest.raises(ValueError, match="Hermitian"):
                model.validate_at(params, q)


def test_existing_periodic_action_preserved_by_shared_helper():
    local, params, q = fixture(edges=((0, 1, 0, 0, 0), (0, 0, 1, 0, 0)), periodic=True)
    old = PeriodicBlockModel(2, 2, local.graph.edges, local.graph.cell)
    onsite, hopping = local.coefficients(params, q)
    vectors = jnp.array([[1, 1j], [-1j, 0.3], [0.2, 0.1j], [-0.8, 1]])
    np.testing.assert_array_equal(old._action(onsite, hopping, vectors), local.apply(params, q, vectors))


def test_public_cpa_cached_dynamic_params_match_independent_exponential():
    model, params, q = fixture(block=1)
    problem = Problem(model, params, PrescribedPath(ConstantPath(q)), CPA())
    simulation = Simulation(problem, Integrator(0.005), Execution(chunk_size=7))
    initial = make_state(q, jnp.zeros_like(q), jnp.array([1.0, 0.0]))
    for factor in (1.0, 1.7):
        actual_params = jax.tree.map(lambda x: x*factor, params)
        simulation.update_parameters(actual_params)
        result = simulation.run(initial, 20)
        onsite = np.asarray(actual_params["onsite"][:, 0, 0])
        hopping = np.asarray(actual_params["hopping"][0, 0, 0])*0.5
        matrix = np.array([[onsite[0], hopping], [hopping.conjugate(), onsite[1]]])
        np.testing.assert_allclose(result.final_state.electronic, expm(-0.1j*matrix)[:, 0], atol=2e-13)


@pytest.mark.parametrize("shape", [(2, 2), (4, 1, 1), (3,), ()])
def test_invalid_vector_shape_is_not_reinterpreted(shape):
    model, params, q = fixture()
    with pytest.raises(ValueError, match="vectors"):
        model.apply(params, q, jnp.zeros(shape))
