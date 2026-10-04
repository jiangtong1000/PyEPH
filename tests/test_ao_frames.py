"""AO ingestion checks explicit metrics, ordering, precision and source evidence."""

from dataclasses import FrozenInstanceError, asdict, replace
import hashlib
import json

import jax
import numpy as np
import pytest

from pyeph.adapters.ao_frames import AOArrayEvidence, ProjectedAOElectronicPath, project_ao_path
from pyeph.core.units import ATOMIC_TIME_FS, HARTREE_EV
from pyeph.dynamics.recorded import RecordedCPA
from pyeph.paths.electronic import AdiabaticElectronicPath


def inputs():
    return dict(times=np.array([0., .2, .5]),
                energies=np.tile([-.3, .7], (3, 1)),
                coefficients=np.tile(np.eye(2), (3, 1, 1)),
                metrics=np.tile(np.eye(2), (3, 1, 1)),
                cross_metrics=np.tile(np.eye(2), (2, 1, 1)),
                retained_bands=np.array([0, 1]), energy_unit="hartree", time_unit="atomic",
                ao_basis_id="ordered-test-AOs", basis_id="selected-eigenstates",
                source_identity="synthetic-numeric-fixture-v1")


def test_identity_projection_diagnostics_bound_evidence_and_column_dynamics():
    data = inputs()
    result = project_ao_path(**data)
    assert isinstance(result.path, (ProjectedAOElectronicPath, AdiabaticElectronicPath))
    assert result.evidence is result.path.projection_evidence
    assert result.path.force_support is False
    np.testing.assert_array_equal(result.path.energies, data["energies"])
    np.testing.assert_array_equal(result.path.overlaps, data["cross_metrics"])
    assert result.diagnostics.metric_min_eigenvalues == (1., 1., 1.)
    assert result.diagnostics.metric_condition_numbers == (1., 1., 1.)
    assert result.diagnostics.metric_hermiticity_errors == (0., 0., 0.)
    assert result.diagnostics.retained_gram_errors == (0., 0., 0.)
    assert result.diagnostics.cross_metric_max_singular_values == (1., 1.)
    assert result.diagnostics.overlap_singular_values == ((1., 1.), (1., 1.))
    assert result.diagnostics.maximum_subspace_loss == (0., 0.)
    assert result.diagnostics.rank_deficient == (False, False)
    assert result.evidence.retained_bands == ((0, 1), (0, 1), (0, 1))
    # The complete evidence is JSON-ready through dataclasses.asdict.
    json.dumps(asdict(result.evidence), allow_nan=False)
    evidence = {x.name: x for x in result.evidence.input_arrays}
    for name in ("times", "energies", "coefficients", "metrics", "cross_metrics", "retained_bands"):
        assert evidence[name].shape == data[name].shape
        assert evidence[name].dtype == data[name].dtype.str
        assert evidence[name].sha256 == hashlib.sha256(data[name].tobytes(order="C")).hexdigest()
    runner = RecordedCPA(result.path)
    c = np.array([[1., .3j], [0., np.sqrt(.91)]])
    final = runner.run(runner.initialize(c), 2).final_state
    np.testing.assert_allclose(final.electronic, np.exp(-.5j*data["energies"][0])[:, None]*c,
                               atol=2e-15, rtol=2e-15)


def test_nonorthogonal_complex_metric_direct_projection_without_repair():
    data = inputs()
    metric = np.array([[1.3, .2+.15j], [.2-.15j, .9]])
    factor = np.linalg.cholesky(metric)
    coeff = np.linalg.inv(factor.conj().T).T
    data.update(coefficients=np.tile(coeff, (3, 1, 1)),
                metrics=np.tile(metric, (3, 1, 1)), cross_metrics=np.tile(metric, (2, 1, 1)))
    result = project_ao_path(**data)
    np.testing.assert_allclose(result.path.overlaps, np.tile(np.eye(2), (2, 1, 1)), atol=5e-16)
    eigen = np.linalg.eigvalsh(metric)
    np.testing.assert_allclose(result.diagnostics.metric_min_eigenvalues, eigen[0], atol=3e-16)
    np.testing.assert_allclose(result.diagnostics.metric_condition_numbers, eigen[1]/eigen[0])


def test_unit_conversion_and_selected_row_order_are_explicit():
    data = inputs()
    data["times"] *= ATOMIC_TIME_FS
    data["energies"] *= HARTREE_EV
    data.update(energy_unit="eV", time_unit="fs", retained_bands=[1, 0])
    result = project_ao_path(**data)
    np.testing.assert_allclose(result.path.times, [0., .2, .5], atol=2e-16)
    np.testing.assert_allclose(result.path.energies, np.tile([.7, -.3], (3, 1)), atol=2e-16)
    assert result.evidence.energy_to_hartree == 1/HARTREE_EV
    assert result.evidence.time_to_atomic == 1/ATOMIC_TIME_FS
    assert result.evidence.retained_bands == ((1, 0),)*3
    assert result.path.unit_system.energy_hartree == 1.


@pytest.mark.parametrize("name,value,match", [
    ("times", [0.], "at least two"),
    ("times", [0., 0., 1.], "strictly increasing"),
    ("times", [1., .5, 0.], "strictly increasing"),
    ("times", [0j, .2, .5], "real numerical"),
    ("times", [False, True, True], "numerical"),
    ("times", [0., np.nan, 1.], "finite"),
    ("energies", [[0., 1.]], "frames, bands"),
    ("energies", np.ones((3, 2), complex), "real numerical"),
    ("energies", np.full((3, 2), np.inf), "finite"),
    ("coefficients", np.ones((3, 2)), "row-ket"),
    ("coefficients", np.ones((3, 3, 2)), "row-ket"),
    ("coefficients", np.full((3, 2, 2), "1"), "numerical"),
    ("metrics", None, "numerical"),
    ("metrics", np.eye(2), "instantaneous metrics"),
    ("cross_metrics", np.eye(2), "cross_metrics"),
    ("cross_metrics", np.full((2, 2, 2), np.nan), "finite"),
    ("retained_bands", [0., 1.], "integer indices"),
    ("retained_bands", [False, True], "integer indices"),
    ("retained_bands", np.empty(0, dtype=int), "nonempty"),
    ("retained_bands", [0, 0], "unique"),
    ("retained_bands", [-1, 1], "in range"),
    ("retained_bands", [0, 2], "in range"),
    ("retained_bands", np.array([2**64-1], dtype=np.uint64), "in range"),
    ("retained_bands", [[0, 1]], "frames, retained"),
    ("retained_bands", 0, "frames, retained"),
    ("energy_unit", "eV/atom", "energy_unit"),
    ("time_unit", "ps", "time_unit"),
    ("energy_unit", None, "energy_unit"),
    ("ao_basis_id", "", "ao_basis_id"),
    ("basis_id", None, "basis_id"),
    ("source_identity", " ", "source_identity"),
    ("transport_mode", "corrected", "transport_mode"),
])
def test_invalid_inputs_fail_without_inference(name, value, match):
    data = inputs()
    data[name] = value
    with pytest.raises(ValueError, match=match):
        project_ao_path(**data)


@pytest.mark.parametrize("metric,match", [
    (np.zeros((2, 2)), "positive definite"),
    (np.diag([1., 0.]), "positive definite"),
    (np.diag([1., -.1]), "positive definite"),
    (np.array([[1., .1], [0., 1.]]), "Hermitian"),
    (np.array([[1.+.01j, 0.], [0., 1.]]), "Hermitian"),
])
def test_metrics_are_spd_without_floors_or_symmetrization(metric, match):
    data = inputs()
    data["metrics"] = np.tile(metric, (3, 1, 1))
    with pytest.raises(ValueError, match=match):
        project_ao_path(**data)


def test_row_normalization_alone_does_not_meet_full_gram_check():
    data = inputs()
    c = np.array([[1., 0.], [.3, np.sqrt(.91)]])
    np.testing.assert_allclose(np.sum(c*c, axis=1), 1.)
    data["coefficients"] = np.tile(c, (3, 1, 1))
    with pytest.raises(ValueError, match="Gram defect 0.3"):
        project_ao_path(**data)


def test_discarded_ao_cross_direction_is_still_checked():
    data = inputs()
    data["retained_bands"] = [0]
    data["cross_metrics"][1, 1, 1] = 1.01
    with pytest.raises(ValueError, match="full AO cross metric.*interval 1"):
        project_ao_path(**data)


def test_raw_rank_deficiency_is_explicit_and_polar_rejects_it():
    data = inputs()
    data["cross_metrics"][0, 1, 1] = 0.
    raw = project_ao_path(**data)
    assert raw.diagnostics.maximum_subspace_loss == (1., 0.)
    assert raw.diagnostics.rank_deficient == (True, False)
    assert bool(raw.path.transport_at(0).valid)
    assert float(raw.path.transport_at(0).diagnostics.maximum_norm_loss) == 1.
    with pytest.raises(ValueError, match="polar.*full-rank.*interval 0"):
        project_ao_path(**data, transport_mode="polar")


def test_float32_export_defects_are_reported_not_repaired():
    data = inputs()
    data["metrics"] *= 2.
    data["cross_metrics"] *= 2.
    data["coefficients"] = (data["coefficients"]*np.sqrt(.5)).astype(np.float32)
    with pytest.raises(ValueError, match="orthonormality.*input dtype float32.*no normalization"):
        project_ao_path(**data)
    # Exact float32 data can still meet the same checks; dtype alone is not rejected.
    data = inputs()
    data["coefficients"] = data["coefficients"].astype(np.float32)
    result = project_ao_path(**data)
    assert next(x for x in result.evidence.input_arrays if x.name == "coefficients").dtype == "<f4"


def test_x64_is_required_without_changing_global_precision():
    previous = bool(jax.config.jax_enable_x64)
    try:
        jax.config.update("jax_enable_x64", False)
        with pytest.raises(ValueError, match="jax_enable_x64"):
            project_ao_path(**inputs())
        assert not jax.config.jax_enable_x64
    finally:
        jax.config.update("jax_enable_x64", previous)


@pytest.mark.parametrize("case", ["time_conversion", "time_duration", "time_collapse", "gram", "condition"])
def test_finite_inputs_with_unrepresentable_derived_quantities_are_rejected(case):
    data = inputs()
    if case == "time_conversion":
        data.update(times=[0., 1e307, 1e308], time_unit="fs")
    elif case == "time_duration":
        # Adjacent intervals must themselves be representable.
        data["times"] = [-1e308, 1e308, 1.1e308]
    elif case == "time_collapse":
        data["times"] = np.array([2**53, 2**53+1, 2**53+2], dtype=np.int64)
    elif case == "gram":
        data["coefficients"] = np.full((3, 2, 2), 1e308)
    else:
        data["metrics"] = np.tile(np.diag([1e-320, 1.]), (3, 1, 1))
    with pytest.raises(ValueError, match="nonfinite|strictly increasing|ill-conditioned"):
        project_ao_path(**data)


def test_output_has_owned_arrays_and_recursively_immutable_evidence():
    data = inputs()
    result = project_ao_path(**data)
    original = asdict(result.evidence)
    for name in ("times", "energies", "coefficients", "metrics", "cross_metrics", "retained_bands"):
        data[name][...] = 7
    assert asdict(result.evidence) == original
    np.testing.assert_array_equal(result.path.times, [0., .2, .5])
    with pytest.raises(FrozenInstanceError):
        result.evidence.source_identity = "changed"
    with pytest.raises(FrozenInstanceError):
        result.path.projection_evidence = None
    with pytest.raises(TypeError, match="tuple"):
        replace(result.evidence, retained_bands=[[0, 1]])
    with pytest.raises(ValueError, match="convention"):
        replace(result.evidence, validation_tolerance=np.array(1e-10))
    with pytest.raises(TypeError, match="shape"):
        AOArrayEvidence("q", [2], "<f8", "a"*64)


def test_retained_metric_check_is_not_a_claim_about_discarded_eigenvectors():
    data = inputs()
    data["coefficients"][:, 1] *= .5
    data["retained_bands"] = [0]
    result = project_ao_path(**data)
    assert result.path.nstates == 1
    assert result.diagnostics.retained_gram_errors == (0., 0., 0.)
