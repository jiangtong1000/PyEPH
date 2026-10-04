#!/usr/bin/env python3
"""Fixed-subspace CPA error experiment, with coefficient-enclosed residual bounds.

This research benchmark changes no dynamics or mapping implementation. Atomic
units set hbar=1. Full and projected wavefunctions share a prescribed electronic
Hamiltonian; nuclei do not respond to either trajectory.
"""

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import platform

import numpy as np
import scipy
from scipy.integrate import solve_ivp


def _add_upper(a, b):
    if a == 0:
        return b
    if b == 0:
        return a
    return math.nextafter(a+b, math.inf)


def _multiply_upper(a, b):
    if a == 0 or b == 0:
        return 0.
    return math.nextafter(a*b, math.inf)


def frobenius_upper(matrix):
    """Outward-rounded Frobenius upper bound on a supplied binary64 matrix.

    Positive products, sums and square root each receive one outward ULP. This
    bounds these exact input entries under ordinary correctly rounded binary64
    arithmetic. It is not an enclosure of upstream matrix construction or an
    ODE solver. Zero entries remain exactly zero; subnormal products round up.
    """
    array = np.asarray(matrix)
    if array.dtype.kind not in "biufc" or not np.isfinite(array).all():
        raise ValueError("bound input must be finite numerical entries")
    total = 0.
    for part in (array.real, array.imag):
        for value in np.ravel(part):
            total = _add_upper(total, _multiply_upper(abs(float(value)), abs(float(value))))
    if not np.isfinite(total):
        raise ValueError("bound arithmetic overflowed")
    return 0. if total == 0 else math.nextafter(math.sqrt(total), math.inf)


@dataclass(frozen=True)
class Case:
    name: str
    matrices: tuple
    bounds: tuple
    retained: tuple
    duration: float
    omega: float = 1.

    def coefficients(self, time):
        return np.array([1., np.sin(self.omega*time)])

    def hamiltonian(self, time):
        return sum(value*matrix for value, matrix in zip(self.coefficients(time), self.matrices, strict=True))


def cases():
    answer = []
    for name, coupling, gap in (("weak_large_gap", .02, 5.),
                                ("strong_large_gap", .35, 5.),
                                ("strong_near_resonance", .35, .1)):
        base = np.array([[0., .2, coupling], [.2, .4, -.3*coupling],
                         [coupling, -.3*coupling, gap]], dtype=complex)
        modulation = np.array([[0., 0., .3*coupling], [0., 0., -.09*coupling],
                               [.3*coupling, -.09*coupling, 0.]], dtype=complex)
        answer.append(Case(name, (base, modulation), (1., 1.), (0, 1), 12., .8))
    # A deliberately aliased residual diagnostic: sin(t) vanishes at both
    # sampled endpoints 0,pi, although its integral and population transfer do not.
    answer.append(Case("aliased_half_sine", (np.zeros((2, 2), complex),
                                              np.array([[0., 1.], [1., 0.]], complex)),
                        (1., 1.), (0,), np.pi))
    return answer


def validate_case(case):
    matrices = [np.asarray(matrix) for matrix in case.matrices]
    if len(matrices) != 2 or len(case.bounds) != 2:
        raise ValueError("this experiment uses constant plus sinusoidal matrix coefficients")
    n = matrices[0].shape[0]
    if any(matrix.shape != (n, n) or not np.isfinite(matrix).all()
           or not np.array_equal(matrix, matrix.conj().T) for matrix in matrices):
        raise ValueError("coefficient matrices must be finite and exactly Hermitian")
    retained = np.asarray(case.retained)
    if (retained.ndim != 1 or retained.dtype.kind not in "iu" or not len(retained)
            or np.any(retained < 0) or np.any(retained >= n)
            or len(set(retained)) != len(retained)):
        raise ValueError("retained indices must be distinct valid electronic states")
    if (not np.isfinite(case.duration) or case.duration <= 0
            or not np.isfinite(case.omega) or case.omega <= 0
            or not np.isfinite(case.bounds).all() or min(case.bounds) < 1.):
        raise ValueError("positive duration/frequency and valid analytic coefficient bounds are required")
    omitted = np.array([index for index in range(n) if index not in set(retained)], dtype=int)
    return n, retained, omitted


def coefficient_bound(case, initial, times):
    """Conservative Duhamel bound from global analytic coefficient enclosures.

    The embedding selects fixed coordinate axes exactly. Thus Q H_k P is the
    omitted/retained submatrix, without rounded dense projector multiplication.
    |1|<=1 and |sin(omega*t)|<=1 are analytical continuum enclosures, not maxima
    inferred from sampled times. Larger explicit bounds remain conservative.
    """
    n, retained, omitted = validate_case(case)
    initial = np.asarray(initial, dtype=complex)
    times = np.asarray(times)
    if initial.shape != (n,) or not np.isfinite(initial).all():
        raise ValueError("initial state must be a finite full-space vector")
    if times.ndim != 1 or np.any(times < 0) or not np.isfinite(times).all():
        raise ValueError("bound times must be finite and nonnegative")
    coupling = 0.
    for limit, matrix in zip(case.bounds, case.matrices, strict=True):
        block = np.asarray(matrix)[np.ix_(omitted, retained)]
        coupling = _add_upper(coupling, _multiply_upper(float(limit), frobenius_upper(block)))
    initial_mismatch = frobenius_upper(initial[omitted])
    reduced_norm = frobenius_upper(initial[retained])
    rate = _multiply_upper(coupling, reduced_norm)
    result = np.array([_add_upper(initial_mismatch, _multiply_upper(float(time), rate)) for time in times])
    return result, {"initial_mismatch_upper": initial_mismatch,
                    "reduced_norm_upper": reduced_norm, "coupling_norm_upper": coupling}


def run_case(case, *, samples=601, initial=None):
    n, retained, omitted = validate_case(case)
    initial = np.eye(n, dtype=complex)[0] if initial is None else np.asarray(initial, dtype=complex)
    if initial.shape != (n,) or not np.isfinite(initial).all():
        raise ValueError("initial state must be a finite full-space vector")
    if not np.isclose(np.vdot(initial, initial).real, 1., atol=1e-12, rtol=0.):
        raise ValueError("physical benchmark initial state must have unit norm")
    if not isinstance(samples, int) or isinstance(samples, bool) or samples < 2:
        raise ValueError("samples must be an integer >= 2")
    grid = np.linspace(0., case.duration, samples)
    embedding = np.eye(n, dtype=complex)[:, retained]
    projected_initial = initial[retained]
    options = dict(method="DOP853", rtol=2e-12, atol=2e-13,
                   max_step=min(.03, case.duration/200), t_eval=grid)
    full = solve_ivp(lambda time, state: -1j*case.hamiltonian(time)@state,
                      (0., case.duration), initial, **options)
    reduced = solve_ivp(lambda time, state: -1j*(
        case.hamiltonian(time)[np.ix_(retained, retained)]@state),
        (0., case.duration), projected_initial, **options)
    if not full.success or not reduced.success:
        raise RuntimeError("independent full/projected ODE solve failed")
    psi = full.y.T
    lifted = reduced.y.T@embedding.T
    residual = np.empty(samples)
    current_full, current_reduced = np.empty(samples), np.empty(samples)
    current_operator_norm = np.empty(samples)
    position = np.arange(n, dtype=float)
    for index, time in enumerate(grid):
        h = case.hamiltonian(time)
        residual[index] = np.linalg.norm(h[np.ix_(omitted, retained)]@reduced.y[:, index])
        current = 1j*h*(position[None, :]-position[:, None])
        current_full[index] = np.vdot(psi[index], current@psi[index]).real
        current_reduced[index] = np.vdot(lifted[index], current@lifted[index]).real
        current_operator_norm[index] = frobenius_upper(current)
    indicator = np.r_[0., np.cumsum(np.diff(grid)*(residual[:-1]+residual[1:])/2)]
    indicator += np.linalg.norm(initial-embedding@projected_initial)
    bound, bound_inputs = coefficient_bound(case, initial, grid)
    error = np.linalg.norm(psi-lifted, axis=1)
    populations = abs(psi)**2
    reduced_populations = abs(lifted)**2
    norm_defect = np.max(abs(np.sum(abs(reduced.y.T)**2, axis=1)-np.vdot(projected_initial, projected_initial).real))
    # This observable inequality uses exact trajectories; these comparisons use
    # high-accuracy numerical solutions and do not certify the ODE solver error.
    norm_sum = _add_upper(frobenius_upper(initial), frobenius_upper(projected_initial))
    current_error = abs(current_full-current_reduced)
    population_error = np.max(abs(populations-reduced_populations), axis=1)
    report = {"name": case.name, "samples": samples, "duration": case.duration,
              "retained": retained.tolist(), "omitted": omitted.tolist(),
              "omega": case.omega, "coefficient_bounds": list(case.bounds),
              "matrices_real": [np.asarray(value).real.tolist() for value in case.matrices],
              "matrices_imag": [np.asarray(value).imag.tolist() for value in case.matrices],
              "initial_real": initial.real.tolist(), "initial_imag": initial.imag.tolist(),
              "final_wavefunction_error": float(error[-1]), "max_wavefunction_error": float(error.max()),
              "max_population_error": float(population_error.max()),
              "max_current_error": float(current_error.max()),
              "max_omitted_population": float(np.sum(populations[:, omitted], axis=1).max()),
              "reduced_norm_defect": float(norm_defect),
              "final_coefficient_enclosed_bound": float(bound[-1]),
              "final_sampled_residual_indicator": float(indicator[-1]), "bound_inputs": bound_inputs,
              "coefficient_bound_covers_numerical_errors": bool(np.all(error <= bound+2e-10)),
              "ode_full_evaluations": full.nfev, "ode_reduced_evaluations": reduced.nfev,
              "ode_settings": {key: value for key, value in options.items() if key != "t_eval"},
              "bound_scope": "continuum projection error for exactly specified coefficient matrices; numerical ODE error excluded",
              "indicator_scope": "sampled trapezoid diagnostic; not an upper bound"}
    arrays = dict(times=grid, full_state=psi, lifted_state=lifted, residual_norm=residual,
                  sampled_residual_indicator=indicator, coefficient_enclosed_bound=bound,
                  wavefunction_error=error, population_error=population_error,
                  current_full=current_full, current_reduced=current_reduced,
                  current_error_bound=np.array([_multiply_upper(_multiply_upper(norm_sum, norm), error)
                      for norm, error in zip(current_operator_norm, bound, strict=True)]))
    return report, arrays


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    source = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    records = []
    for case in cases():
        count = 2 if case.name == "aliased_half_sine" else 601
        report, arrays = run_case(case, samples=count)
        np.savez(args.output/f"{case.name}.npz", **arrays)
        records.append(report)
    assert all(record["coefficient_bound_covers_numerical_errors"] for record in records)
    assert records[2]["max_wavefunction_error"] > .2
    assert records[2]["reduced_norm_defect"] < 1e-10
    assert records[-1]["final_wavefunction_error"] > 1
    assert records[-1]["final_sampled_residual_indicator"] < 1e-12
    assert source == hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    report = {"script_sha256": source, "cases": records, "passed": True,
              "scope": "fixed-space prescribed-path projection experiment; no runtime state truncation",
              "units": "atomic units; hbar=1", "nuclear_feedback": False,
              "runtime": {"python": platform.python_version(), "numpy": np.__version__,
                          "scipy": scipy.__version__, "platform": platform.platform()}}
    (args.output/"report.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps({"passed": True, "cases": [{key: record[key] for key in (
        "name", "max_population_error", "max_current_error", "reduced_norm_defect",
        "final_wavefunction_error", "final_coefficient_enclosed_bound", "final_sampled_residual_indicator")}
        for record in records]}, indent=2))


if __name__ == "__main__":
    main()
