"""Independent exact cases for a research-only fixed-subspace error experiment."""

from dataclasses import replace
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.linalg import expm


_PATH = Path(__file__).resolve().parents[1]/"benchmarks"/"reduced_space_error.py"
_SPEC = importlib.util.spec_from_file_location("reduced_space_error_reference", _PATH)
experiment = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = experiment
_SPEC.loader.exec_module(experiment)


def test_constant_rabi_loss_matches_exact_state_and_is_covered_by_enclosure():
    coupling = .3
    h = np.array([[0., coupling], [coupling, 0.]], complex)
    case = experiment.Case("rabi", (h, np.zeros_like(h)), (1., 1.), (0,), 2.)
    report, result = experiment.run_case(case, samples=101)
    t = result["times"]
    exact_full = np.stack([np.cos(coupling*t), -1j*np.sin(coupling*t)], axis=1)
    np.testing.assert_allclose(result["full_state"], exact_full, atol=3e-13)
    np.testing.assert_array_equal(result["lifted_state"], np.tile([1., 0.], (len(t), 1)))
    np.testing.assert_allclose(result["wavefunction_error"], 2*np.sin(coupling*t/2), atol=3e-13)
    assert np.all(result["wavefunction_error"] <= result["coefficient_enclosed_bound"]+2e-13)
    assert report["max_population_error"] > .3
    assert report["reduced_norm_defect"] == 0.


def test_coarsely_sampled_residual_is_not_an_error_bound():
    case = experiment.cases()[-1]
    report, result = experiment.run_case(case, samples=2)
    exact = expm(-2j*np.array([[0., 1.], [1., 0.]]))@np.array([1., 0.])
    np.testing.assert_allclose(result["full_state"][-1], exact, atol=3e-12)
    assert report["final_wavefunction_error"] == pytest.approx(2*np.sin(1.), abs=3e-12)
    assert report["final_sampled_residual_indicator"] < 1e-12
    assert report["final_coefficient_enclosed_bound"] >= np.pi
    assert report["reduced_norm_defect"] == 0.
    assert report["max_population_error"] > .8


def test_initial_omitted_amplitude_is_included_even_when_residual_is_zero():
    h = np.diag([0., .5, 2.]).astype(complex)
    case = experiment.Case("initial-mismatch", (h, np.zeros_like(h)), (1., 1.), (0, 1), 2.)
    initial = np.array([np.sqrt(.5), 0., np.sqrt(.5)], complex)
    report, result = experiment.run_case(case, samples=51, initial=initial)
    np.testing.assert_allclose(result["wavefunction_error"], np.sqrt(.5), atol=3e-13)
    np.testing.assert_array_equal(result["residual_norm"], 0.)
    assert np.all(result["coefficient_enclosed_bound"] >= np.linalg.norm(initial[2:]))
    assert report["bound_inputs"]["initial_mismatch_upper"] > .7


def test_weak_and_strong_gap_cases_preserve_norm_but_differ_in_physical_observables():
    records = [experiment.run_case(case, samples=301) for case in experiment.cases()[:3]]
    for report, values in records:
        assert report["coefficient_bound_covers_numerical_errors"]
        assert report["reduced_norm_defect"] < 1e-10
        assert np.all(abs(values["current_full"]-values["current_reduced"])
                      <= values["current_error_bound"]+1e-10)
    weak, far, near = [record for record, _ in records]
    assert near["max_population_error"] > 10*weak["max_population_error"]
    assert near["max_omitted_population"] > 5*far["max_omitted_population"]
    assert near["max_current_error"] > .05


def test_supplied_matrices_are_exactly_hermitian_and_embedding_has_distinct_axes():
    case = experiment.cases()[0]
    with pytest.raises(ValueError, match="distinct"):
        experiment.validate_case(replace(case, retained=(0, 0)))
    bad = case.matrices[0].copy()
    bad[0, 1] += 1e-8j
    with pytest.raises(ValueError, match="Hermitian"):
        experiment.validate_case(replace(case, matrices=(bad, case.matrices[1])))
    with pytest.raises(ValueError, match="coefficient bounds"):
        experiment.validate_case(replace(case, bounds=(1., .9)))


def test_frobenius_enclosure_handles_complex_entries_zero_and_subnormals():
    assert experiment.frobenius_upper(np.zeros((2, 3))) == 0.
    assert experiment.frobenius_upper(np.array([3., 4.])) >= 5.
    assert experiment.frobenius_upper(np.array([3+4j])) >= 5.
    tiny = np.nextafter(0., 1.)
    assert experiment.frobenius_upper(np.array([tiny])) >= tiny
    with pytest.raises(ValueError, match="overflowed"):
        experiment.frobenius_upper(np.array([1e308]))
