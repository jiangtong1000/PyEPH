"""Independent physics and public-boundary checks for opt-in Fourier EPC."""

from dataclasses import fields, replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.linalg import expm

from pyeph import CPA, CoupledClassical, Ehrenfest, Execution, Integrator, Problem, Simulation
from pyeph.core.contracts import LowRankWeight, ProbeContext, prepared_action
from pyeph.core.problem import PrescribedPath
from pyeph.core.state import make_state
from pyeph.dynamics.mash2 import mapping_state as mash2_state
from pyeph.dynamics.mashrm import mapping_state as mashrm_state
from pyeph.models.fourier_epc import FourierCartesianEPCModel
from pyeph.models.cartesian_epc import CartesianEPCModel
from pyeph.observables.population import ElectronicPopulation
from pyeph.paths.harmonic import ConstantPath, HarmonicBath

from test_cartesian_epc import dense_equations, stencil


def data_for(real=False):
    data = stencil()
    return (replace(data, hopping_values=data.hopping_values.real,
                    epc_values=data.epc_values.real) if real else data)


def compile_pair(data, mesh, *, carrier="electron", real=False):
    options = dict(carrier=carrier, hermiticity="project", term_batch_size=3,
                   real_tolerance=0. if real else None)
    return tuple(data.compile_supercell(mesh, epc_backend=backend, **options)
                 for backend in ("direct", "fft"))


def reference_data(data, carrier):
    return (replace(data, hopping_values=-data.hopping_values.conj(),
                    epc_values=-data.epc_values.conj()) if carrier == "hole" else data)


def assert_tree_close(actual, expected, tolerance=3e-12):
    assert jax.tree.structure(actual) == jax.tree.structure(expected)
    for left, right in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
        assert np.shape(left) == np.shape(right) and left.dtype == right.dtype
        np.testing.assert_allclose(left, right, atol=tolerance, rtol=tolerance)


def test_direct_default_is_unchanged_and_backend_choice_is_explicit():
    data = stencil()
    default = data.compile_supercell((1, 1, 1))
    assert type(default.model) is CartesianEPCModel
    with pytest.raises(ValueError, match="backend"):
        data.compile_supercell((1, 1, 1), epc_backend="automatic")
    with pytest.raises((ValueError, TypeError), match="mesh|integer"):
        data.compile_supercell((True, 3, 2), epc_backend="fft")


@pytest.mark.parametrize("mesh", [(1, 1, 1), (2, 3, 2)])
@pytest.mark.parametrize("carrier", ["electron", "hole"])
@pytest.mark.parametrize("real", [False, True])
def test_raw_images_actions_peierls_currents_and_original_params(mesh, carrier, real):
    data = data_for(real)
    direct, fourier = compile_pair(data, mesh, carrier=carrier, real=real)
    assert set(direct.params) == set(fourier.params)
    for name in direct.params:
        np.testing.assert_array_equal(direct.params[name], fourier.params[name])
    rng = np.random.default_rng(61)
    q = rng.normal(size=fourier.model.spec.system.q_shape)*.1
    columns = rng.normal(size=(fourier.model.nstates, 3))+1j*rng.normal(size=(fourier.model.nstates, 3))
    original = reference_data(data, carrier)
    expected, _ = dense_equations(original, mesh, q)
    model, params = fourier.model, fourier.params
    np.testing.assert_allclose(jax.jit(model.apply)(params, q, columns), expected@columns,
                               atol=2e-14, rtol=2e-13)
    np.testing.assert_allclose(jax.jit(lambda p, q, c: prepared_action(model, p, q)(c))(
        params, q, columns[:, 0]), expected@columns[:, 0], atol=2e-14, rtol=2e-13)
    wavevector = np.array([.31, -.17, .21])
    peierls = dense_equations(original, mesh, q, wavevector)[0]
    np.testing.assert_allclose(jax.jit(model.apply_peierls)(params, q, wavevector, columns),
                               peierls@columns, atol=2e-14, rtol=2e-13)
    for axis in range(3):
        delta = np.eye(3)[axis]*1e-6
        plus = dense_equations(original, mesh, q, delta)[0]
        minus = dense_equations(original, mesh, q, -delta)[0]
        current = model.charge*(plus-minus)/(2e-6)
        actual = model.probe_apply(params, ProbeContext(q), "current_"+"xyz"[axis], columns)
        np.testing.assert_allclose(actual, current@columns, atol=2e-10, rtol=2e-9)


def test_declared_real_dtype_preserves_complex_states_and_initializes_both_mash_methods():
    direct, compiled = compile_pair(data_for(True), (1, 1, 1), real=True)
    model, params = compiled.model, compiled.params
    q, p = jnp.arange(6.).reshape(2, 3)*.03, jnp.ones((2, 3))*.02
    h = jax.jit(model.dense)(params, q)
    assert h.dtype == jnp.float64 and not model.spec.complex_valued
    np.testing.assert_allclose(h, direct.model.dense(direct.params, q), atol=2e-15)
    actual = jax.jit(model.apply)(params, q, 1j*jnp.eye(2))
    assert actual.dtype == jnp.complex128
    np.testing.assert_allclose(actual, 1j*np.asarray(h), atol=2e-15)
    current = model.probe_apply(params, ProbeContext(q), "current_x", jnp.eye(2))
    assert jnp.iscomplexobj(current) and np.max(abs(np.asarray(current).imag)) > .01
    for state in (mash2_state(model, params, q, p, [0., 0., -1.]),
                  mashrm_state(model, params, q, p, [1., 0.], basis="adiabatic")):
        np.testing.assert_allclose(np.linalg.norm(state.electronic), 1., atol=2e-15)
        assert int(state.method_state["active"]) == 0


@pytest.mark.parametrize("real", [False, True])
def test_empty_epc_bypasses_fourier_allocation_even_with_one_byte_limit(real):
    data = data_for(real)
    data = replace(data, epc_channels=np.empty(0, dtype=int), epc_atoms=np.empty(0, dtype=int),
        epc_cells=np.empty((0, 3), dtype=int), epc_values=np.empty((0, 3), dtype=float if real else complex))
    compiled = data.compile_supercell((2, 3, 1), epc_backend="fft", max_spectral_bytes=1,
                                      real_tolerance=0. if real else None)
    q = jnp.arange(36.).reshape(12, 3)/10
    columns = jnp.ones((compiled.model.nstates, 2), dtype=complex)*(1+.2j)
    expected = dense_equations(data, (2, 3, 1), q)[0]@np.asarray(columns)
    np.testing.assert_allclose(jax.jit(compiled.model.apply)(compiled.params, q, columns), expected, atol=2e-15)
    force = jax.jit(compiled.model.contract_gradient)(compiled.params, q, LowRankWeight(columns, columns))
    np.testing.assert_array_equal(force, jnp.zeros_like(q))
    jaxpr = str(jax.make_jaxpr(compiled.model.apply)(compiled.params, q, columns)).lower()
    assert "fft[" not in jaxpr and "fft_type" not in jaxpr


@pytest.mark.parametrize("real", [False, True])
def test_coordinate_and_source_parameter_ad_match_independent_image_differences(real):
    data, mesh = data_for(real), (2, 3, 2)
    _, compiled = compile_pair(data, mesh, real=real)
    model, params = compiled.model, compiled.params
    rng = np.random.default_rng(103)
    q = rng.normal(size=model.spec.system.q_shape)*.2
    direction = rng.normal(size=q.shape)
    columns = rng.normal(size=(model.nstates, 2))+1j*rng.normal(size=(model.nstates, 2))
    left = rng.normal(size=columns.shape)+1j*rng.normal(size=columns.shape)
    epsilon = 1e-5
    derivative = (dense_equations(data, mesh, q+epsilon*direction)[0]
                  -dense_equations(data, mesh, q-epsilon*direction)[0])/(2*epsilon)
    actual = jax.jit(lambda q, dq: jax.jvp(lambda x: model.apply(params, x, columns), (q,), (dq,))[1])(q, direction)
    np.testing.assert_allclose(actual, derivative@columns, atol=3e-11, rtol=2e-10)
    weight = LowRankWeight(left, columns)
    force = jax.jit(model.contract_gradient)(params, q, weight)
    expected = np.vdot(left@columns.conj().T, derivative).real
    np.testing.assert_allclose(np.sum(force*direction), expected, atol=3e-10, rtol=2e-10)
    # An arbitrary complex low-rank weight tests the nonconjugated -k transpose.
    ad_force = jax.grad(lambda x: jnp.vdot(left, model.apply(params, x, columns)).real)(q)
    np.testing.assert_allclose(force, ad_force, atol=3e-14, rtol=3e-13)
    scaled_force = jax.grad(lambda scale: jnp.sum(model.contract_gradient(
        params, q, LowRankWeight(scale*left, columns))))(1.)
    np.testing.assert_allclose(scaled_force, np.sum(force), atol=3e-14, rtol=3e-13)
    tangent = rng.normal(size=params["epc_values"].shape)
    if not real:
        tangent = tangent+1j*rng.normal(size=tangent.shape)
    tangent = tangent.reshape(-1, 3)
    tangent[len(data.epc_values):] = 0.
    raw = replace(data, hopping_values=np.zeros_like(data.hopping_values),
                  epc_values=tangent[:len(data.epc_values)])
    expected = dense_equations(raw, mesh, q)[0]@columns
    tangent = tangent.reshape(params["epc_values"].shape)
    actual = jax.jit(lambda values, delta: jax.jvp(
        lambda g: model.apply({**params, "epc_values": g}, q, columns),
        (values,), (delta,))[1])(params["epc_values"], tangent)
    np.testing.assert_allclose(actual, expected, atol=3e-14, rtol=3e-13)


def test_public_parameter_update_reuses_compilation_but_not_old_fourier_coefficients():
    data, mesh = stencil(), (2, 1, 1)
    compiled = data.compile_supercell(mesh, epc_backend="fft", term_batch_size=3)
    q = jnp.arange(12.).reshape(4, 3)*.05
    c = np.array([1., .3j, .1, -.2j])
    c /= np.linalg.norm(c)
    initial = make_state(q, jnp.zeros_like(q), c)
    problem = Problem(compiled.model, compiled.params, PrescribedPath(ConstantPath(q)), CPA(), ElectronicPopulation())
    simulation = Simulation(problem, Integrator(.01), Execution(chunk_size=3))
    old = simulation.run(initial, 6)
    cache = dict(simulation._compiled)
    changed = {**compiled.params, "epc_values": 1.7*compiled.params["epc_values"]}
    simulation.update_parameters(changed)
    actual = simulation.run(initial, 6)
    assert simulation._compiled == cache
    h = dense_equations(replace(data, epc_values=1.7*data.epc_values), mesh, q)[0]
    expected = expm(-.06j*h)@c
    np.testing.assert_allclose(actual.final_state.electronic, expected, atol=3e-10, rtol=3e-10)
    assert np.max(abs(np.asarray(actual.final_state.electronic-old.final_state.electronic))) > 1e-4
    broken = {**changed, "neighbors": changed["neighbors"].at[0, 1].set(changed["neighbors"][0, 0])}
    with pytest.raises(ValueError, match="translation|periodic"):
        simulation.update_parameters(broken)
    assert simulation.problem.params is changed


def test_prepare_action_preserves_subclass_apply_and_elements_dispatch():
    class AddedIdentity(FourierCartesianEPCModel):
        def apply(self, params, q, vectors):
            return super().apply(params, q, vectors)+.17*vectors

    class ScaledElements(FourierCartesianEPCModel):
        def elements(self, params, q):
            return 1.3*super().elements(params, q)

    compiled = stencil().compile_supercell((2, 1, 1), epc_backend="fft", term_batch_size=3)
    config = {field.name:getattr(compiled.model, field.name)
              for field in fields(compiled.model) if field.init}
    q, columns = jnp.ones((4, 3))*.03, jnp.eye(4, dtype=complex)
    original = compiled.model.apply(compiled.params, q, columns)
    for cls, expected in ((AddedIdentity, original+.17*columns), (ScaledElements, 1.3*original)):
        model = cls(**config)
        actual = jax.jit(lambda p, q, c: prepared_action(model, p, q)(c))(compiled.params, q, columns)
        np.testing.assert_allclose(actual, expected, atol=3e-15)


def test_spectral_admission_uses_actual_coordinate_and_weight_precision():
    data = stencil()
    data = replace(data, hopping_values=data.hopping_values.astype(np.complex64),
                   epc_values=data.epc_values.astype(np.complex64))
    mesh = (2, 3, 1)
    single_kernel = int(np.prod(mesh))*len(data.hopping_values)*len(data.masses)*3*8
    compiled = data.compile_supercell(mesh, epc_backend="fft", max_spectral_bytes=single_kernel)
    model, params = compiled.model, compiled.params
    q = jnp.ones(model.spec.system.q_shape, dtype=jnp.float32)*.1
    assert model.elements(params, q).dtype == jnp.complex64
    with pytest.raises(ValueError, match="spectral|Fourier|bytes"):
        jax.jit(model.elements)(params, q.astype(jnp.float64))
    narrow = jnp.ones((model.nstates, 1), dtype=jnp.complex64)
    assert model.contract_gradient(params, q, LowRankWeight(narrow, narrow)).dtype == jnp.float32
    with pytest.raises(ValueError, match="spectral|Fourier|bytes"):
        jax.jit(model.contract_gradient)(params, q,
            LowRankWeight(narrow.astype(jnp.complex128), narrow.astype(jnp.complex128)))


@pytest.mark.parametrize("updates", [{"mesh": (2, 1, 1)}, {"mesh": (True, 3, 2)},
    {"mesh": (1.5, 2, 2)}, {"max_spectral_bytes": True}, {"max_spectral_bytes": 0}])
def test_invalid_static_fft_contract_rejected(updates):
    compiled = stencil().compile_supercell((2, 3, 1), epc_backend="fft")
    with pytest.raises((ValueError, TypeError)):
        replace(compiled.model, **updates)


def test_mesh_is_owned_and_numpy_integer_inputs_are_supported():
    compiled = stencil().compile_supercell((2, 3, 1), epc_backend="fft")
    mesh = [np.int64(2), np.int64(3), np.int64(1)]
    model = replace(compiled.model, mesh=mesh, max_spectral_bytes=np.int64(1000000))
    mesh[0] = 9
    assert model.mesh == (2, 3, 1)
    model.validate_params(compiled.params)


@pytest.mark.parametrize("method", ["cpa", "ehrenfest"])
def test_full_public_trajectory_and_restart_match_direct_backend(method, tmp_path):
    direct, fourier = compile_pair(stencil(), (2, 1, 1))
    q, p = jnp.arange(12.).reshape(4, 3)*.01, jnp.ones((4, 3))*.03
    c = jnp.asarray([1., .1j, .3, -.2j])
    c /= jnp.linalg.norm(c)
    initial = make_state(q, p, c, trajectory_id=29, seed=91)

    def simulation(compiled, chunk=4):
        treatment = (HarmonicBath(jnp.ones(q.shape)*.7, compiled.masses) if method == "cpa"
                     else CoupledClassical(compiled.masses))
        problem = Problem(compiled.model, compiled.params, treatment,
                          CPA() if method == "cpa" else Ehrenfest(), ElectronicPopulation())
        return Simulation(problem, Integrator(.003), Execution(chunk_size=chunk, save_every=1))

    sim = simulation(fourier)
    expected, actual = simulation(direct).run(initial, 8), sim.run(initial, 8)
    assert_tree_close(actual.final_state, expected.final_state)
    assert_tree_close(actual.observables, expected.observables)
    filename = tmp_path/f"{method}.h5"
    sim.save_checkpoint(filename, sim.run(initial, 3).final_state)
    resumed_sim = simulation(fourier, chunk=5)
    resumed = resumed_sim.run(resumed_sim.load_checkpoint(filename), 5)
    assert_tree_close(resumed.final_state, actual.final_state)
    with pytest.raises(ValueError, match="model"):
        simulation(direct).load_checkpoint(filename)
