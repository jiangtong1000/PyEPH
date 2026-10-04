"""Public thermal-factor recipe, independent dynamics oracle and strict restart."""

import hashlib
import importlib.util
import json
from pathlib import Path

import h5py
import numpy as np

from pyeph.core.contracts import ProbeContext


_PATH = Path(__file__).resolve().parents[1]/"benchmarks/thermal_columns.py"
_SPEC = importlib.util.spec_from_file_location("thermal_columns_driver", _PATH)
driver = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(driver)


def test_native_ring_and_current_match_independent_complex_bond_equations():
    problem, q, _, inputs = driver.recipe.ring_problem(7)
    h, current = driver.independent_matrices(inputs, 0.)
    np.testing.assert_allclose(problem.model.apply(problem.params, q, np.eye(7)), h, atol=2e-16)
    np.testing.assert_allclose(problem.model.probe_apply(
        problem.params, ProbeContext(q), "current_x", np.eye(7)), current, atol=2e-16)
    lower, upper = driver.recipe.gershgorin_bounds(inputs)
    eigenvalues = np.linalg.eigvalsh(h)
    assert lower < eigenvalues[0] and upper > eigenvalues[-1]


def test_filtered_public_cpa_matches_dense_expm_ode_and_strict_restart(tmp_path):
    report = driver.correctness(tmp_path, seeds=4)
    errors = report["trajectory_errors"]
    assert errors["0.04"]["combined_correlation_max_error"] > 10*errors["0.02"]["combined_correlation_max_error"]
    assert errors["0.02"]["preparation_only_same_integrator_max_error"] < 2e-11
    assert report["restart_max_error"] < 3e-13
    assert report["complete_basis_thermal_density_error"] < 2e-11


def test_recipe_stream_and_saved_digest_bind_actual_initializer_factor(tmp_path):
    output = tmp_path/"recipe"
    result, record = driver.recipe.run(output, nstates=5, ncolumns=3, steps=3)
    with np.load(output/"initial_preparation.npz") as arrays:
        digest = hashlib.sha256(arrays["factor"].tobytes()).hexdigest()
    assert record["preparation"]["factor_digest"] == digest
    saved = json.loads((output/"preparation.json").read_text())
    assert saved == record
    with h5py.File(output/"correlation.h5") as handle:
        np.testing.assert_array_equal(handle["observables/current_correlation"],
                                      result.observables["current_correlation"])
        assert json.loads(handle.attrs["metadata"])["preparation"]["factor_digest"] == digest
    assert (output/"checkpoint.h5").is_file()
