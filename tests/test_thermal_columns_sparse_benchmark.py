"""Sparse large-system qualification equations checked cheaply on small systems."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
from scipy.linalg import expm

from pyeph.core.contracts import ProbeContext
from pyeph.thermal import ThermalFilterPlan


_PATH = Path(__file__).resolve().parents[1]/"benchmarks/thermal_columns_sparse.py"
_SPEC = importlib.util.spec_from_file_location("thermal_columns_sparse_driver", _PATH)
driver = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(driver)


@pytest.mark.parametrize("family", ["hp", "nonlocal"])
def test_sparse_hp_and_nonlocal_equations_are_hermitian_and_current_is_phase_derivative(family):
    problem, _, _, host = driver.fixture(family, 17)
    t = .31
    frequency = host["canonical_frequencies"]
    q = host["q0"]*np.cos(frequency*t)+host["p0"]/frequency*np.sin(frequency*t)
    h, current = driver.sparse_matrices(host, t)
    np.testing.assert_allclose(h.toarray(), h.conj().T.toarray(), atol=2e-16)
    vectors = np.random.default_rng(52).normal(size=(17, 3))+1j*np.random.default_rng(15).normal(size=(17, 3))
    np.testing.assert_allclose(problem.model.apply(problem.params, q, vectors), h@vectors, atol=8e-16)
    np.testing.assert_allclose(problem.model.probe_apply(
        problem.params, ProbeContext(q), "current_x", vectors), current@vectors, atol=8e-16)
    phase = 1e-6
    derivative = (driver.sparse_matrices(host, t, phase)[0]-driver.sparse_matrices(host, t, -phase)[0])/(2*phase)
    np.testing.assert_allclose(derivative.toarray(), current.toarray(), atol=8e-11)


def test_sparse_exponential_reference_agrees_with_independent_dense_expm():
    _, _, _, host = driver.fixture("nonlocal", 13)
    h, _ = driver.sparse_matrices(host, 0.)
    plan = ThermalFilterPlan(8., -3., 3., action_id="test", bounds_id="analytic-test")
    omega = np.random.default_rng(42).normal(size=(13, 4))
    factor, log_partition = driver.sparse_filter(h, omega, plan)
    y = expm(-plan.beta*(h.toarray()-plan.lower*np.eye(13))/2)@omega
    np.testing.assert_allclose(factor, y/np.linalg.norm(y), atol=2e-14)
    np.testing.assert_allclose(log_partition, np.log(np.sum(abs(y)**2)/4)-plan.beta*plan.lower, atol=2e-14)


@pytest.mark.parametrize("family", ["hp", "nonlocal"])
def test_public_cpa_sparse_reference_and_hlo_guard(family, tmp_path):
    report = driver.run_case(tmp_path, family, 17, 4, 3., steps=8, stride=4)
    assert report["accuracy"]["same_omega_normalized_factor_frobenius_error"] < 2e-11
    assert report["accuracy"]["public_vs_sparse_rk4_correlation_max_error"] < 2e-13
    assert not report["filter_compiled"]["detected_global_square_shapes"]
    assert not report["transport_compiled"]["detected_global_square_shapes"]
