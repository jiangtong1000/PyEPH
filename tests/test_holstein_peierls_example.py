"""Independent ring physics and application-level restart/statistics checks."""

from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import sys

import h5py
import jax
import numpy as np
import pytest
from scipy.integrate import solve_ivp
from scipy.linalg import expm

from pyeph import Execution, Integrator, Simulation
from pyeph.core.contracts import ProbeContext
from pyeph.io.hdf5 import HDF5Observer


_PATH = Path(__file__).resolve().parents[1] / "examples/holstein_peierls.py"
_SPEC = importlib.util.spec_from_file_location("holstein_peierls_example", _PATH)
example = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = example
_SPEC.loader.exec_module(example)


def independent_matrices(inputs, temperature, q, *, phase=0):
    """Direct real-space equation, deliberately independent of edge coefficients."""
    n, scale = inputs.nsites, inputs.hopping_mev
    wh, wp = inputs.holstein_frequency_mev / scale, inputs.peierls_frequency_mev / scale
    gh = np.sqrt(inputs.reorganization_mev * inputs.holstein_frequency_mev) / scale
    gp = inputs.peierls_fraction * np.sqrt(np.tanh(wp / (2 * temperature)))
    h = np.zeros((n, n), dtype=complex)
    v = np.zeros_like(h)
    if inputs.mode == "cpa":
        h[np.diag_indices(n)] = gh * np.sqrt(2 * wh) * q[:n]
    for i in range(n):
        j = (i + 1) % n
        hopping = -1 + gp * np.sqrt(2 * wp) * q[-n + i]
        h[i, j] += hopping * np.exp(1j * phase)
        h[j, i] += hopping * np.exp(-1j * phase)
        v[i, j] += 1j * hopping
        v[j, i] -= 1j * hopping
    return h, v


@pytest.mark.parametrize("nsites", [3, 4, 7])
def test_ring_hamiltonian_and_current_match_equation_and_uniform_phase_derivative(nsites):
    inputs = example.Inputs(nsites=nsites, peierls_fraction=.23)
    problem, metadata = example.build_problem(inputs)
    q = np.linspace(-.8, .7, 2 * nsites)
    h, velocity = independent_matrices(inputs, metadata["temperature_reduced"], q)
    actual = problem.model.apply(problem.params, q, np.eye(nsites))
    current = problem.model.probe_apply(problem.params, ProbeContext(q), "velocity_x", np.eye(nsites))
    np.testing.assert_allclose(actual, h, atol=2e-16, rtol=2e-16)
    np.testing.assert_allclose(current, velocity, atol=2e-16, rtol=2e-16)
    phase = 1e-5
    plus, _ = independent_matrices(inputs, metadata["temperature_reduced"], q, phase=phase)
    minus, _ = independent_matrices(inputs, metadata["temperature_reduced"], q, phase=-phase)
    np.testing.assert_allclose(current, (plus - minus) / (2 * phase), atol=3e-11, rtol=3e-11)


@pytest.mark.parametrize("mode", ["cpa", "lf", "band"])
def test_clean_ring_has_analytical_ballistic_correlation_and_no_diffusive_plateau(mode, tmp_path):
    inputs = example.Inputs(nsites=5, trajectories=2, mode=mode, reorganization_mev=0,
                            steps=5, dt=.01)
    report = example.run_workflow(inputs, tmp_path / mode)
    k = 2 * np.pi * np.arange(inputs.nsites) / inputs.nsites
    energy = -2 * np.cos(k)
    beta = report["units_and_conventions"]["beta_reduced"]
    weights = np.exp(-beta * (energy - energy.min()))
    expected = np.sum(weights * (2 * np.sin(k))**2) / np.sum(weights)
    with np.load(tmp_path / mode / "analysis.npz") as data:
        np.testing.assert_allclose(data["mean_correlation"], expected, atol=2e-11)
        np.testing.assert_allclose(data["mean_running_integral"], expected * data["time_reduced"],
                                   atol=2e-12)
        assert "finite_window_mobility_proxy_cm2_per_Vs" not in data
        np.testing.assert_allclose(data["sem_running_integral"], 0, atol=1e-15)


def test_dynamic_ring_converges_to_independent_scipy_propagation():
    inputs = example.Inputs(nsites=4, trajectories=1, peierls_fraction=.3, reorganization_mev=25)
    problem, metadata = example.build_problem(inputs)
    initial, q0, p0, _ = example.prepare(inputs, problem, metadata)
    omega = np.asarray(problem.params["omega"])
    q0, p0 = q0[0], p0[0]
    temperature = metadata["temperature_reduced"]

    def matrices(time):
        q = q0 * np.cos(omega * time) + p0 / omega * np.sin(omega * time)
        return independent_matrices(inputs, temperature, q)

    end = .8
    def rhs(time, flat):
        return (-1j * matrices(time)[0] @ flat.reshape(inputs.nsites, inputs.nsites)).ravel()

    oracle = solve_ivp(rhs, (0, end), np.eye(inputs.nsites, dtype=complex).ravel(),
                       method="DOP853", rtol=2e-13, atol=2e-14)
    assert oracle.success
    u = oracle.y[:, -1].reshape(inputs.nsites, inputs.nsites)
    h0, v0 = matrices(0)
    rho = expm(-metadata["beta_reduced"] * h0)
    rho /= np.trace(rho)
    correlation = np.trace(matrices(end)[1] @ u @ v0 @ rho @ u.conj().T)
    errors = []
    for dt in (.04, .02):
        result = Simulation(problem, Integrator(dt), Execution(chunk_size=20)).run(
            initial, round(end / dt))
        errors.append(np.max(np.abs(np.asarray(result.final_state.electronic[0]) - u)))
        np.testing.assert_allclose(result.observables["current_correlation"][-1, 0, 0],
                                   correlation, atol=3e-6 if dt == .04 else 2e-7)
    assert errors[1] < 2e-7
    assert 12 < errors[0] / errors[1] < 20


def test_stable_ids_preserve_samples_under_partition():
    inputs = example.Inputs(nsites=3, trajectories=5, first_id=29)
    problem, metadata = example.build_problem(inputs)
    _, q, p, ids = example.prepare(inputs, problem, metadata)
    subset = replace(inputs, trajectories=2, first_id=31)
    _, qsub, psub, subids = example.prepare(subset, problem, metadata)
    np.testing.assert_array_equal(subids, ids[2:4])
    np.testing.assert_array_equal(qsub, q[2:4])
    np.testing.assert_array_equal(psub, p[2:4])


def test_streamed_restart_and_replayed_samples_preserve_complete_transport_origin(tmp_path):
    inputs = example.Inputs(nsites=4, trajectories=3, steps=6, chunk_size=2,
                            peierls_fraction=.15, spacing_angstrom=7.2)
    whole = example.run_workflow(inputs, tmp_path / "whole")
    first = example.run_workflow(replace(inputs, steps=2), tmp_path / "first",
                                 samples=tmp_path / "whole/initial_samples.npz")
    resumed = example.run_workflow(replace(inputs, steps=4, chunk_size=3),
                                   tmp_path / "resumed", resume=tmp_path / "first")
    assert first["final_time_reduced"] == inputs.dt * 2
    assert resumed["final_time_reduced"] == pytest.approx(whole["final_time_reduced"], abs=2e-16)
    with np.load(tmp_path / "whole/analysis.npz") as a, np.load(tmp_path / "resumed/analysis.npz") as b:
        for key in a:
            np.testing.assert_allclose(a[key], b[key], atol=3e-12, rtol=3e-12)
    with h5py.File(tmp_path / "resumed/trajectory.h5") as stream:
        assert stream["time"].shape == (5, 3)
    with pytest.raises(ValueError, match="preserve all inputs"):
        example.run_workflow(replace(inputs, seed=4), tmp_path / "bad", resume=tmp_path / "first")
    with pytest.raises(FileExistsError):
        example.run_workflow(inputs, tmp_path / "whole")


def test_integral_sampling_error_includes_time_covariance_and_single_sample_is_undefined(tmp_path):
    times = np.array([0., .2, .7, 1.])
    amplitudes = np.array([1., 3., 5.])
    for count in (1, 3):
        path = tmp_path / f"samples{count}.h5"
        correlation = np.broadcast_to(amplitudes[:count], (4, count)).astype(complex)
        with HDF5Observer(path, metadata={"fingerprint": "synthetic-constant-curves",
                                         "trajectory_ids": list(range(count))}) as stream:
            stream(np.broadcast_to(times[:, None], (4, count)),
                   {"current_correlation": correlation[..., None],
                    "unitary_error": np.zeros((4, count))})
        arrays, _ = example.analyze_streams([path])
        np.testing.assert_allclose(arrays["mean_running_integral"], times * np.mean(amplitudes[:count]))
        if count == 1:
            assert np.isnan(arrays["sem_running_integral"]).all()
        else:
            np.testing.assert_allclose(arrays["sem_running_integral"], times * 2 / np.sqrt(3))


def test_resume_rejects_modified_data_and_sampler_rejects_wrong_temperature(tmp_path):
    inputs = example.Inputs(nsites=3, trajectories=1, steps=1)
    example.run_workflow(inputs, tmp_path / "run")
    problem, metadata = example.build_problem(replace(inputs, temperature_kelvin=300))
    with pytest.raises(ValueError, match="nuclear temperature"):
        example.prepare(replace(inputs, temperature_kelvin=300), problem, metadata,
                        tmp_path / "run/initial_samples.npz")
    with h5py.File(tmp_path / "run/trajectory.h5", "a") as handle:
        handle["observables/current_correlation"][0, 0, 0] += 1
    with pytest.raises(ValueError, match="saved workflow data changed"):
        example.run_workflow(inputs, tmp_path / "resumed", resume=tmp_path / "run")
    report = json.loads((tmp_path / "run/run.json").read_text())
    assert report["final_integral_nuclear_sampling_sem"] is None


def test_example_requires_explicit_x64(tmp_path):
    # No global precision changes are made by importing or assembling the example.
    before = jax.config.x64_enabled
    assert before
    jax.config.update("jax_enable_x64", False)
    try:
        with pytest.raises(ValueError, match="JAX_ENABLE_X64"):
            example.run_workflow(example.Inputs(), tmp_path / "bad")
    finally:
        jax.config.update("jax_enable_x64", before)
