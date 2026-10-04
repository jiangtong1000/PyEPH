"""Independent finite-cell equations for general Cartesian EPC stencils."""

from dataclasses import replace
from itertools import product

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pyeph.adapters.real_space_epc import RealSpaceEPC
from pyeph.core.contracts import LowRankWeight, ProbeContext
from pyeph.core.units import UnitSystem


def stencil():
    orbitals = np.array([[0, 0], [1, 1], [0, 1], [1, 0],
                         [0, 1], [1, 0], [0, 0], [0, 0]])
    cells = np.array([[0, 0, 0]]*4+[[1, 0, 0], [-1, 0, 0], [1, 0, 0], [-1, 0, 0]])
    hopping = np.array([.2, .7, .1+.03j, .1-.03j, .03+.04j, .03-.04j,
                        .02+.01j, .02-.01j])
    coupling = np.array([[.1, .2, 0.], [.2+.03j, -.1j, .04], [.2-.03j, .1j, .04],
                         [.03j, .1, -.05j], [-.03j, .1, .05j],
                         [.02+.01j, .03j, .04], [.02-.01j, -.03j, .04]])
    return RealSpaceEPC(
        [[2., .1, 0.], [0., 3., .2], [.1, 0., 4.]], [[0., .1, .2], [.5, .3, .4]],
        [1.5, 3.], [[.1, .2, .3], [.7, .4, .1]], orbitals, cells, hopping,
        [0, 2, 3, 4, 5, 6, 7], [0, 1, 1, 0, 0, 1, 1],
        [[0, 0, 0], [0, 1, 0], [0, 1, 0], [1, 0, 0], [0, 0, 0],
         [0, 0, 0], [-1, 0, 0]], coupling,
        [[0, 0], [1, 1], [0, 1], [1, 0]],
        [[0, 0, 0], [0, 0, 0], [1, 0, 0], [-1, 0, 0]],
        [4*np.eye(3), 5*np.eye(3), -np.eye(3), -np.eye(3)],
        UnitSystem(.5, 1.))


def dense_equations(data, mesh, displacement, kappa=None):
    """Literal translated matrix/block assembly, without production action helpers."""
    cells = list(product(*(range(n) for n in mesh)))
    lookup = {c: i for i, c in enumerate(cells)}
    nat, nwan = len(data.masses), len(data.wannier_centers)
    u = np.asarray(displacement).reshape(len(cells), nat, 3)
    raw = np.zeros((len(cells)*nwan,)*2, dtype=complex)
    hessian = np.zeros((len(cells)*nat*3,)*2)
    for origin, c in enumerate(cells):
        for channel, ((i, j), re, hopping) in enumerate(zip(
                data.hopping_orbitals, data.hopping_cells, data.hopping_values, strict=True)):
            destination = lookup[tuple((np.array(c)+re) % mesh)]
            value = hopping
            for term in np.flatnonzero(data.epc_channels == channel):
                perturbed = lookup[tuple((np.array(c)+data.epc_cells[term]) % mesh)]
                value += data.epc_values[term]@u[perturbed, data.epc_atoms[term]]
            if kappa is not None:
                d = re@data.cell+data.wannier_centers[j]-data.wannier_centers[i]
                value *= np.exp(1j*np.dot(kappa, d))
            raw[origin*nwan+i, destination*nwan+j] += value
        for (a, b), r, block in zip(data.ifc_atoms, data.ifc_cells, data.ifc_values, strict=True):
            destination = lookup[tuple((np.array(c)+r) % mesh)]
            first, second = (origin*nat+a)*3, (destination*nat+b)*3
            hessian[first:first+3, second:second+3] += block
    return .5*(raw+raw.conj().T), .5*(hessian+hessian.T)


def test_uncoupled_and_free_nuclei_have_constant_electronics_and_zero_forces():
    data = replace(stencil(), epc_channels=np.empty(0, dtype=int),
        epc_atoms=np.empty(0, dtype=int), epc_cells=np.empty((0, 3), dtype=int),
        epc_values=np.empty((0, 3), dtype=complex), ifc_atoms=np.empty((0, 2), dtype=int),
        ifc_cells=np.empty((0, 3), dtype=int), ifc_values=np.empty((0, 3, 3)))
    compiled = data.compile_supercell((2, 1, 1))
    model, params = compiled.model, compiled.params
    q = jnp.arange(12.).reshape(4, 3)/10
    c = jnp.array([1., 2j, 3., 4j])
    expected, k = dense_equations(data, (2, 1, 1), q)
    np.testing.assert_allclose(jax.jit(model.apply)(params, q, c), expected@c, atol=1e-15)
    np.testing.assert_array_equal(jax.jit(model.contract_gradient)(params, q, jnp.eye(4)),
                                  jnp.zeros_like(q))
    np.testing.assert_array_equal(jax.jit(model.reference_gradient)(params, q), jnp.zeros_like(q))
    assert jax.jit(model.reference_energy)(params, q) == 0.
    np.testing.assert_array_equal(model.dense_hessian(params), k)


@pytest.mark.parametrize("integer_hopping", [False, True])
def test_cartesian_derivative_promotes_static_hopping_dtype(integer_hopping):
    data = stencil()
    data = replace(data,
        hopping_values=np.round(data.hopping_values.real).astype(int) if integer_hopping
            else data.hopping_values.real,
        epc_values=data.epc_values.real if integer_hopping else data.epc_values)
    compiled = data.compile_supercell((2, 1, 1))
    model, params = compiled.model, compiled.params
    q = jnp.arange(12.).reshape(4, 3)/10
    c = jnp.array([1., 2j, 3., 4j])
    expected, _ = dense_equations(data, (2, 1, 1), q)
    np.testing.assert_allclose(jax.jit(model.apply)(params, q, c), expected@c, atol=2e-15)
    weight = jnp.outer(c, c.conj())
    direction = np.arange(12.).reshape(4, 3)/17
    epsilon = 1e-5
    plus, _ = dense_equations(data, (2, 1, 1), q+epsilon*direction)
    minus, _ = dense_equations(data, (2, 1, 1), q-epsilon*direction)
    derivative = np.vdot(c, (plus-minus)@c).real/(2*epsilon)
    native = jax.jit(model.contract_gradient)(params, q, weight)
    np.testing.assert_allclose(np.sum(native*direction), derivative, rtol=2e-10, atol=1e-10)


@pytest.mark.parametrize("batch_size", [1, 3, 256])
@pytest.mark.parametrize("mesh", [(1, 1, 1), (2, 2, 1)])
def test_sparse_action_and_harmonic_force_match_literal_stencils(batch_size, mesh):
    data = stencil()
    compiled = data.compile_supercell(mesh, term_batch_size=batch_size)
    model, params = compiled.model, compiled.params
    rng = np.random.default_rng(381)
    q = rng.normal(size=model.spec.system.q_shape)
    expected, hessian = dense_equations(data, mesh, q)
    columns = rng.normal(size=(model.nstates, 3))+1j*rng.normal(size=(model.nstates, 3))
    np.testing.assert_allclose(jax.jit(model.apply)(params, q, columns), expected@columns, atol=8e-16)
    np.testing.assert_allclose(model.prepare_action(params, q)(columns[:, 0]),
                               expected@columns[:, 0], atol=8e-16)
    np.testing.assert_allclose(model.dense_hessian(params), hessian, atol=1e-15)
    np.testing.assert_allclose(jax.jit(model.reference_gradient)(params, q).reshape(-1),
                               hessian@q.reshape(-1), atol=3e-15)
    np.testing.assert_allclose(model.reference_energy(params, q),
                               .5*q.reshape(-1)@hessian@q.reshape(-1), atol=2e-14)


@pytest.mark.parametrize("factored", [False, True])
def test_contracted_electronic_force_matches_independent_central_differences(factored):
    data, mesh = stencil(), (2, 1, 1)
    compiled = data.compile_supercell(mesh, term_batch_size=3)
    model, params = compiled.model, compiled.params
    rng = np.random.default_rng(922)
    q = rng.normal(size=model.spec.system.q_shape)*.2
    left = rng.normal(size=(model.nstates, 2))+1j*rng.normal(size=(model.nstates, 2))
    right = rng.normal(size=(model.nstates, 2))+1j*rng.normal(size=(model.nstates, 2))
    dense_weight = left@right.conj().T
    weight = LowRankWeight(left, right) if factored else dense_weight
    force = np.asarray(jax.jit(model.contract_gradient)(params, q, weight)).reshape(-1)
    expected = []
    for index in range(q.size):
        shift = np.zeros_like(q).reshape(-1)
        shift[index] = 1e-5
        plus = dense_equations(data, mesh, q+shift.reshape(q.shape))[0]
        minus = dense_equations(data, mesh, q-shift.reshape(q.shape))[0]
        expected.append(np.real(np.vdot(dense_weight, plus-minus))/(2e-5))
    np.testing.assert_allclose(force, expected, atol=8e-11, rtol=1e-9)
    # Electronic-state sensitivities must survive the explicit force contraction.
    def objective(scale):
        return jnp.sum(model.contract_gradient(params, q, LowRankWeight(scale*left, right)))
    np.testing.assert_allclose(jax.grad(objective)(1.), np.sum(force), atol=1e-15)


@pytest.mark.parametrize("axis", range(3))
def test_peierls_current_keeps_distinct_images_aliasing_to_one_finite_edge(axis):
    data, mesh = stencil(), (1, 1, 1)
    compiled = data.compile_supercell(mesh)
    model, params = compiled.model, compiled.params
    q = np.arange(6).reshape(2, 3)*.03
    shift = np.eye(3)[axis]*1e-6
    plus = dense_equations(data, mesh, q, shift)[0]
    minus = dense_equations(data, mesh, q, -shift)[0]
    expected = model.charge*(plus-minus)/(2e-6)
    current = np.asarray(model.probe_apply(params, ProbeContext(q), f"current_{'xyz'[axis]}", np.eye(2)))
    np.testing.assert_allclose(current, expected, atol=5e-11, rtol=1e-9)
    np.testing.assert_allclose(current, current.conj().T, atol=1e-15)
    # A self-image channel contributes even though row and column both wrap to0.
    if axis == 0:
        assert abs(current[0, 0]) > .01


def test_unit_conversion_preserves_hamiltonian_force_and_phonon_spectrum():
    original = stencil()
    target = UnitSystem.from_ev_angstrom()
    converted = original.converted(target)
    before, after = original.compile_supercell((2, 1, 1)), converted.compile_supercell((2, 1, 1))
    length = original.unit_system.length_bohr/target.length_bohr
    energy = original.unit_system.energy_hartree/target.energy_hartree
    q = np.arange(12).reshape(4, 3)*.01
    np.testing.assert_allclose(after.model.dense(after.params, q*length),
                               before.model.dense(before.params, q)*energy, atol=1e-14)
    np.testing.assert_allclose(after.model.reference_gradient(after.params, q*length),
                               before.model.reference_gradient(before.params, q)*energy/length, atol=1e-14)
    spectra = []
    for item in (before, after):
        masses = np.repeat(np.asarray(item.masses).reshape(-1), 3)
        spectra.append(np.linalg.eigvalsh(item.model.dense_hessian(item.params)
                                         /np.sqrt(masses[:, None]*masses)))
    np.testing.assert_allclose(spectra[1], spectra[0]*energy**2, atol=3e-13)


def test_hole_transformation_and_dynamic_parameter_reuse():
    data = stencil()
    electron = data.compile_supercell((2, 1, 1))
    hole = data.compile_supercell((2, 1, 1), carrier="hole")
    q = np.arange(12).reshape(4, 3)*.01
    np.testing.assert_allclose(hole.model.dense(hole.params, q),
                               -electron.model.dense(electron.params, q).T, atol=1e-15)
    assert hole.model.charge == 1 and electron.model.charge == -1
    action = jax.jit(electron.model.dense)
    action(electron.params, q)
    changed = {**electron.params, "epc_values": 1.7*electron.params["epc_values"]}
    expected = dense_equations(replace(data, epc_values=1.7*data.epc_values), (2, 1, 1), q)[0]
    np.testing.assert_allclose(action(changed, q), expected, atol=1e-15)


def test_projection_is_explicit_quantified_and_preserves_original_stencils():
    data = stencil()
    bad = np.array(data.epc_values, copy=True)
    bad[1] *= 1.3
    changed = replace(data, epc_values=bad)
    with pytest.raises(ValueError, match="non-Hermitian.*epc"):
        changed.compile_supercell((2, 1, 1))
    result = changed.compile_supercell((2, 1, 1), hermiticity="project")
    assert result.report["hermiticity"]["epc"]["projection_relative_l2_change"] > 0
    np.testing.assert_array_equal(changed.epc_values, bad)
    bad[:] = 0
    assert np.max(abs(changed.epc_values)) > 0
    with pytest.raises(ValueError, match="read-only"):
        changed.epc_values[0] = 0
    q = np.arange(12).reshape(4, 3)*.01
    np.testing.assert_allclose(result.model.dense(result.params, q),
                               dense_equations(changed, (2, 1, 1), q)[0], atol=1e-15)


def test_primitive_derivative_and_ifc_arrays_are_not_replicated_with_supercell_size():
    data = stencil()
    small, large = data.compile_supercell((1, 1, 1)), data.compile_supercell((101, 1, 1))
    for name in ("epc_values", "ifc_values", "hopping", "epc_channels", "epc_atoms"):
        np.testing.assert_array_equal(small.params[name], large.params[name])
    assert large.params["neighbors"].shape[1] == 101
    with pytest.raises(ValueError, match="dense Hessian needs"):
        large.model.dense_hessian(large.params, max_bytes=1024)


def test_real_conversion_is_explicit_bounded_and_separately_audited():
    data = stencil()
    with pytest.raises(ValueError, match="imaginary coefficient"):
        data.compile_supercell((1, 1, 1), real_tolerance=1e-9)
    almost = replace(data, hopping_values=data.hopping_values.real+1e-12j*data.hopping_values.imag,
                     epc_values=data.epc_values.real+1e-12j*data.epc_values.imag)
    result = almost.compile_supercell((1, 1, 1), real_tolerance=1e-9)
    assert not result.model.spec.complex_valued
    assert not np.iscomplexobj(result.params["hopping"])
    assert not np.iscomplexobj(result.params["epc_values"])
    assert np.iscomplexobj(almost.epc_values)
    assert result.report["imaginary_coefficients_discarded"]
    for name in ("hopping", "epc"):
        evidence = result.report["imaginary_coefficients"][name]
        assert 0 < evidence["maximum_absolute_imaginary"] < 1e-9
        assert 0 < evidence["relative_imaginary_l2_norm"] < 1e-9


@pytest.mark.parametrize("kwargs", [{"mesh": (0, 1, 1)}, {"mesh": (1.5, 1, 1)},
    {"hermiticity": "repair"}, {"tolerance": -1.}, {"carrier": "guess"}, {"term_batch_size": 0}])
def test_invalid_compilation_choices_fail(kwargs):
    options = {"mesh": (1, 1, 1), **kwargs}
    with pytest.raises(ValueError):
        stencil().compile_supercell(**options)
