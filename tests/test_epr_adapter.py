"""EPR storage, Cartesian units, and real-input audits independent of old PyEPH outputs."""

from pathlib import Path

import h5py
import jax
import numpy as np
import pytest

from pyeph.adapters.epr import read_epr
from pyeph.core.contracts import ProbeContext, pure_state_weight
from pyeph.core.units import UnitSystem
from pyeph.paths.normal_modes import NormalModeBath


DATA = Path(__file__).parent/"compatibility/data/preprocessing/DNTT_epr.h5"


def synthetic_epr(path, *, polar=False):
    """An analytically labeled2×1×1 WS fixture with complex ordered derivatives.

    Orbital fractional centers are .1 and .4, and the atom is at .2. For each
    residue the nearest image gives the literal R lists below; no production
    WS helper is used to construct or label this input.
    """
    electron_cells = {(0, 0): [(0, 0, 0), (-1, 0, 0), (1, 0, 0)],
                      (1, 1): [(0, 0, 0), (-1, 0, 0), (1, 0, 0)],
                      (0, 1): [(0, 0, 0), (-1, 0, 0)],
                      (1, 0): [(0, 0, 0), (1, 0, 0)]}
    phonon_cells = {0: [(0, 0, 0), (-1, 0, 0)],
                   1: [(0, 0, 0), (1, 0, 0)]}
    source_derivatives = {}
    with h5py.File(path, "w") as f:
        for name, value in dict(at=np.eye(3), bg=np.eye(3), alat=2.,
                kc_dim=[2, 1, 1], qc_dim=[2, 1, 1], num_wann=2, nat=1,
                wannier_center_cryst=[[.1, 0., 0.], [.4, 0., 0.]],
                tau=[[.2, 0., 0.]], mass=[100.], volume=8., lpolar=int(polar), spinor=1).items():
            f[f"basic_data/{name}"] = value
        for number, values in enumerate(([.2, .03-.01j, .03+.01j],
                                          [.1+.04j, .05-.02j],
                                          [.7, -.02+.03j, -.02-.03j]), 1):
            f[f"electron_wannier/hopping_r{number}"] = np.real(values)
            f[f"electron_wannier/hopping_i{number}"] = np.imag(values)
        for i, j in electron_cells:
            re, rp = electron_cells[i, j], phonon_cells[i]
            real = (np.arange(len(re)*len(rp)*3).reshape(len(rp), len(re), 3)+10*i+j)/1000
            values = real+1j*(real+.2)
            source_derivatives[i, j] = values
            f[f"eph_matrix_wannier/ep_hop_r_1_{j+1}_{i+1}"] = values.real
            f[f"eph_matrix_wannier/ep_hop_i_1_{j+1}_{i+1}"] = values.imag
        block = np.array([[-.3, .01, .02], [-.01, -.4, .03], [.04, .05, -.2]])
        ifc = np.array([np.diag([2., 3., 4.]), block, block.T])
        f["force_constant/ifc1"] = ifc.swapaxes(-1, -2)
    return electron_cells, phonon_cells, source_derivatives, ifc


def test_ordered_cells_complex_derivatives_and_ifc_axis_convention(tmp_path):
    path = tmp_path/"complex_epr.h5"
    electronic_cells, phonon_cells, expected, ifc = synthetic_epr(path)
    source = read_epr(path)
    data = source.data
    for i, j in electronic_cells:
        for pi, rp in enumerate(phonon_cells[i]):
            for ei, re in enumerate(electronic_cells[i, j]):
                channel = np.flatnonzero(np.all(data.hopping_orbitals == (i, j), axis=1)
                                         & np.all(data.hopping_cells == re, axis=1))[0]
                term = np.flatnonzero((data.epc_channels == channel)
                                      & np.all(data.epc_cells == rp, axis=1))[0]
                np.testing.assert_array_equal(data.epc_values[term], expected[i, j][pi, ei])
    np.testing.assert_array_equal(data.ifc_values, ifc)
    np.testing.assert_array_equal(data.cell, 2*np.eye(3))
    np.testing.assert_array_equal(data.atom_positions, [[.4, 0., 0.]])
    np.testing.assert_array_equal(data.wannier_centers, [[.2, 0., 0.], [.8, 0., 0.]])
    assert source.metadata["spinor"] and source.metadata["complex_values_retained"]
    assert source.metadata["ws_degeneracy_already_in_source"]
    # Source static coefficients already include degeneracy: they stay unchanged.
    index = np.flatnonzero(np.all(data.hopping_orbitals == (0, 0), axis=1)
                           & np.all(data.hopping_cells == (1, 0, 0), axis=1))[0]
    assert data.hopping_values[index] == .03+.01j


def test_rydberg_mass_and_derivative_conversion_are_consistent(tmp_path):
    path = tmp_path/"units_epr.h5"
    synthetic_epr(path)
    data = read_epr(path).data
    assert data.unit_system == UnitSystem(.5, 1.)
    converted = data.converted(UnitSystem())
    np.testing.assert_array_equal(converted.masses, [200.])
    np.testing.assert_array_equal(converted.hopping_values, .5*data.hopping_values)
    np.testing.assert_array_equal(converted.epc_values, .5*data.epc_values)
    np.testing.assert_array_equal(converted.ifc_values, .5*data.ifc_values)


def test_polar_omission_requires_explicit_choice_and_enters_report(tmp_path):
    path = tmp_path/"polar_epr.h5"
    synthetic_epr(path, polar=True)
    with pytest.raises(ValueError, match="requires explicit"):
        read_epr(path)
    source = read_epr(path, polar="short_range")
    result = source.compile_supercell((1, 1, 1), hermiticity="project")
    assert result.report["source"]["source_polar"]
    assert result.report["source"]["polar_policy"] == "short_range"
    assert not result.report["source"]["long_range_included"]
    assert not result.report["source"]["oscillator_normalization_applied"]


def test_malformed_ordered_epc_dimensions_fail(tmp_path):
    path = tmp_path/"bad_epr.h5"
    synthetic_epr(path)
    with h5py.File(path, "a") as f:
        for part in ("r", "i"):
            name = f"eph_matrix_wannier/ep_hop_{part}_1_1_2"
            del f[name]
            f[name] = np.zeros((2, 3, 3))
    with pytest.raises(ValueError, match="ordered Wigner-Seitz"):
        read_epr(path)


@pytest.mark.parametrize("dataset,shape", [
    ("electron_wannier/hopping_i1", (1,)),
    ("eph_matrix_wannier/ep_hop_i_1_1_2", (1, 1, 3)),
])
def test_broadcastable_malformed_imaginary_dataset_fails(tmp_path, dataset, shape):
    path = tmp_path/"broadcast_epr.h5"
    synthetic_epr(path)
    with h5py.File(path, "a") as handle:
        del handle[dataset]
        handle[dataset] = np.zeros(shape)
    with pytest.raises(ValueError, match="Wigner-Seitz"):
        read_epr(path)


@pytest.fixture(scope="module")
def dntt():
    return read_epr(DATA, polar="short_range")


def test_dntt_raw_stencils_expose_image_hermiticity_defect_without_modifying_values(dntt):
    audit = dntt.data.hermiticity_audit()
    assert audit["hopping"]["maximum_absolute_defect"] < 1e-15
    assert audit["ifc"]["maximum_absolute_defect"] < 1e-15
    assert audit["epc"]["missing_reverse_terms"] > 0
    assert audit["epc"]["relative_l2_defect"] > .01
    assert np.max(abs(dntt.data.epc_values.imag)) > 0
    with pytest.raises(ValueError, match="non-Hermitian.*epc"):
        dntt.compile_supercell((2, 2, 2))


@pytest.mark.parametrize("mesh,on_source_grid", [((2, 2, 2), True), ((3, 1, 1), False)])
def test_dntt_projection_preserves_source_grid_and_exposes_off_grid_change(dntt, mesh, on_source_grid):
    data = dntt.data
    result = dntt.compile_supercell(mesh, hermiticity="project")
    model, params = result.model, result.params
    q = np.random.default_rng(190).normal(size=model.spec.system.q_shape)*.015
    raw = np.zeros((model.nstates, model.nstates), dtype=complex)
    u = q.reshape(model.ncells, 72, 3)
    cells = np.array(list(np.ndindex(*mesh)))
    for origin, cell in enumerate(cells):
        for channel, ((i, j), re, hopping) in enumerate(zip(
                data.hopping_orbitals, data.hopping_cells, data.hopping_values, strict=True)):
            terms = np.flatnonzero(data.epc_channels == channel)
            destination = np.ravel_multi_index((cell+re) % mesh, mesh)
            perturbed = np.ravel_multi_index(((cell+data.epc_cells[terms]) % mesh).T, mesh)
            value = hopping+np.sum(data.epc_values[terms]*u[perturbed, data.epc_atoms[terms]])
            raw[origin*2+i, destination*2+j] += value
    native = np.asarray(jax.jit(model.dense)(params, q))
    if on_source_grid:
        # Coarse-mesh aliasing hides the much larger unaliased image defect.
        assert np.max(abs(raw-raw.conj().T)) < 3e-11
        np.testing.assert_allclose(native, raw, atol=2e-11, rtol=0)
    else:
        # Projection is a changed interpolation away from the source mesh.
        assert np.max(abs(native-raw)) > 1e-8
    np.testing.assert_allclose(native, native.conj().T, atol=1e-16)
    # Different summation orders over 28800 terms retain double-precision accuracy.
    np.testing.assert_allclose(native, .5*(raw+raw.conj().T), atol=2e-15, rtol=0)


def test_dntt_forces_and_image_current_match_finite_differences(dntt):
    result = dntt.compile_supercell((1, 1, 1), hermiticity="project")
    model, params = result.model, result.params
    q = np.random.default_rng(661).normal(size=model.spec.system.q_shape)*.01
    state = np.array([np.sqrt(.3), 1j*np.sqrt(.7)])
    gradient = np.asarray(model.contract_gradient(params, q, pure_state_weight(state)))
    for atom, axis in ((0, 0), (16, 2), (48, 1), (71, 2)):
        shift = np.zeros_like(q)
        shift[atom, axis] = 1e-5
        plus = np.vdot(state, model.apply(params, q+shift, state)).real
        minus = np.vdot(state, model.apply(params, q-shift, state)).real
        np.testing.assert_allclose(gradient[atom, axis], (plus-minus)/2e-5, atol=3e-12)
    for axis in range(3):
        shift = np.eye(3)[axis]*1e-6
        plus = model.apply_peierls(params, q, shift, np.eye(2))
        minus = model.apply_peierls(params, q, -shift, np.eye(2))
        current = model.probe_apply(params, ProbeContext(q), f"current_{'xyz'[axis]}", np.eye(2))
        np.testing.assert_allclose(current, model.charge*(plus-minus)/2e-6, atol=2e-10)


def test_dntt_signed_acoustic_roundoff_is_retained_with_explicit_zero_tolerance(dntt):
    result = dntt.compile_supercell((1, 1, 1), hermiticity="project")
    hessian = result.model.dense_hessian(result.params)
    mass = np.repeat(np.asarray(result.masses).reshape(-1), 3)
    eigenvalues = np.linalg.eigvalsh(hessian/np.sqrt(mass[:, None]*mass))
    # These are numerical translation zeros, not demonstrated physical instabilities.
    assert np.max(abs(eigenvalues[:3])) < 1e-16
    assert eigenvalues[3] > 1e-9
    bath = NormalModeBath.from_hessian(hessian, result.masses, np.zeros((72, 3)),
                                      zero_tolerance=1e-16)
    np.testing.assert_allclose(bath.squared_frequencies, eigenvalues, atol=2e-18)
    assert not result.report["unstable_modes_modified"]


def test_genuinely_unstable_source_ifc_is_not_replaced_by_a_frequency_floor(tmp_path):
    path = tmp_path/"unstable_epr.h5"
    synthetic_epr(path)
    with h5py.File(path, "a") as f:
        f["force_constant/ifc1"][...] = np.array([-np.eye(3), np.zeros((3, 3)), np.zeros((3, 3))])
    result = read_epr(path).compile_supercell((1, 1, 1), hermiticity="project")
    hessian = result.model.dense_hessian(result.params)
    np.testing.assert_array_equal(hessian, -np.eye(3))
    with pytest.raises(ValueError, match="unstable"):
        NormalModeBath.from_hessian(hessian, result.masses, np.zeros((1, 3)))
