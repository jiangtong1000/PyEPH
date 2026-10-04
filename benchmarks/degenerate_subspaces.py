"""Gauge-covariant transport of isolated Kramers doublets on prescribed paths.

An independent research proof, not a hopping method or nuclear-force provider.
Atomic units; fixed four-state site/spin basis; one carrier; scalar reference
zero. Geometric adiabatic transport and finite-speed Schrodinger propagation
are different calculations and are reported separately.
"""

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import platform

import numpy as np
import scipy
from scipy.integrate import solve_ivp


I2 = np.eye(2, dtype=complex)
SX = np.array([[0., 1.], [1., 0.]], dtype=complex)
SY = np.array([[0., -1j], [1j, 0.]], dtype=complex)
SZ = np.diag([1., -1.]).astype(complex)
TIME_REVERSAL = np.kron(I2, 1j*SY)  # Theta psi = TIME_REVERSAL @ psi.conj().


def parameters():
    return {"delta": .4, "base": np.array([.55, 0., .25]),
            "radius_xy": .42, "radius_yz": .36}


def coupling(q):
    x, y, z = np.asarray(q)
    return x*I2+1j*y*SY+1j*z*SZ


def hamiltonian(q, params=None):
    params = parameters() if params is None else params
    delta = params["delta"]
    t = coupling(q)
    return np.block([[delta*I2, t], [t.conj().T, -delta*I2]])


def hamiltonian_derivatives():
    zero = np.zeros((2, 2), dtype=complex)
    return np.array([np.block([[zero, dt], [dt.conj().T, zero]])
                     for dt in (I2, 1j*SY, 1j*SZ)])


def projector_data(q, params=None):
    """Analytic lower-doublet projector and its spatial derivatives."""
    params = parameters() if params is None else params
    energy = np.sqrt(params["delta"]**2+np.dot(q, q))
    if energy <= 1e-12:
        raise ValueError("the two doublets must remain separated")
    h = hamiltonian(q, params)
    p = .5*(np.eye(4)-h/energy)
    dp = -.5*(hamiltonian_derivatives()/energy-np.asarray(q)[:, None, None]*h/energy**3)
    return p, dp, energy


def negative_frame(q, params=None):
    """One analytic 4x2 orthonormal frame; no individual eigensolver vectors."""
    params = parameters() if params is None else params
    delta = params["delta"]
    if delta <= 0:
        raise ValueError("this analytic frame chart requires positive delta")
    energy = np.sqrt(delta**2+np.dot(q, q))
    return np.vstack([-coupling(q), (energy+delta)*I2])/np.sqrt(2*energy*(energy+delta))


def coordinate_gauge(q):
    """A smooth, coordinate-dependent U(2) gauge with noncommuting factors."""
    x, y, z = q
    phase, first, second = .21*x+.17*y*y, .6*x+.31*np.sin(z), .5*y+.22*z*z
    return (np.exp(1j*phase)*(np.cos(first)*I2-1j*np.sin(first)*SX)
            @ (np.cos(second)*I2-1j*np.sin(second)*SZ))


def path(s, loop="xy", params=None):
    """Two closed paths based at the same q; s is a dimensionless path parameter."""
    params = parameters() if params is None else params
    angle = 2*np.pi*s
    co, si = np.cos(angle), np.sin(angle)
    if loop == "xy":
        radius = params["radius_xy"]
        shift, tangent = np.array([co-1, si, 0]), np.array([-si, co, 0])
    elif loop == "yz":
        radius = params["radius_yz"]
        shift, tangent = np.array([0, si, co-1]), np.array([0, co, -si])
    else:
        raise ValueError("loop must be xy or yz")
    return params["base"]+radius*shift, 2*np.pi*radius*tangent


def kato_generator(s, loop="xy", params=None):
    q, tangent = path(s, loop, params)
    projector, derivatives, _ = projector_data(q, params)
    derivative = np.einsum("i,ijk->jk", tangent, derivatives)
    return derivative@projector-projector@derivative


def _times(value):
    times = np.asarray(value, dtype=float)
    if (times.ndim != 1 or len(times) < 2 or not np.isfinite(times).all()
            or times[0] != 0 or times[-1] != 1 or np.any(np.diff(times) <= 0)):
        raise ValueError("a strictly increasing path grid must run from 0 to 1")
    return times


def kato_reference(times, loop="xy", params=None, *, rtol=2e-12, atol=2e-14, max_step=.01):
    """Full-space DOP853 integration Y'=[P',P]Y, Y(0)=P(0).

    Y is a partial isometry from the initial doublet into the current one. This
    removes the common dynamical phase and enforces the adiabatic subspace
    approximation; it is not finite-speed propagation by h alone.
    """
    times = _times(times)
    initial = projector_data(path(0, loop, params)[0], params)[0]
    def rhs(s, values):
        return (kato_generator(s, loop, params)@values.reshape(4, 4)).ravel()
    result = solve_ivp(rhs, (0., 1.), initial.ravel(), method="DOP853", t_eval=times,
                       rtol=rtol, atol=atol, max_step=max_step)
    if not result.success:
        raise RuntimeError(result.message)
    return result.y.T.reshape(-1, 4, 4)


def polar_unitary(matrix):
    """Unique full-rank unitary polar factor; no singular-value floor."""
    matrix = np.asarray(matrix)
    if matrix.shape != (2, 2) or not np.isfinite(matrix).all():
        raise ValueError("a finite 2x2 overlap is required")
    left, singular, right = np.linalg.svd(matrix)
    if singular[-1] <= 1e-10:
        raise ValueError("rank-deficient overlap: refine the path or change the frame chart")
    return left@right, float(singular[-1])


@dataclass
class PolarResult:
    times: np.ndarray
    operators: np.ndarray
    minimum_link_singular_value: float


def polar_transport(times, loop="xy", params=None, *, gauge=None):
    """Discrete full-doublet transport; each sample may have an arbitrary U(2) gauge.

    ``gauge`` is None, a function of q, or an array (nsteps+1,2,2). Returning
    physical 4x4 partial isometries includes the initial frame transformation,
    so even nonperiodic endpoint gauges are handled without a hidden reset.
    """
    times = _times(times)
    frames = []
    for index, s in enumerate(times):
        q = path(s, loop, params)[0]
        frame = negative_frame(q, params)
        if gauge is not None:
            rotation = np.asarray(gauge(q) if callable(gauge) else gauge[index])
            if (rotation.shape != (2, 2) or not np.isfinite(rotation).all()
                    or not np.allclose(rotation.conj().T@rotation, I2, atol=1e-12, rtol=0)):
                raise ValueError("frame gauge must be unitary")
            frame = frame@rotation
        frames.append(frame)
    initial = frames[0]
    coefficients, operators, minimum = I2.copy(), [initial@initial.conj().T], 1.
    for previous, current in zip(frames, frames[1:]):
        link, singular = polar_unitary(current.conj().T@previous)
        coefficients = link@coefficients
        operators.append(current@coefficients@initial.conj().T)
        minimum = min(minimum, singular)
    return PolarResult(times, np.array(operators), minimum)


def holonomy(operator, params=None, *, initial_gauge=None):
    params = parameters() if params is None else params
    frame = negative_frame(params["base"], params)
    if initial_gauge is not None:
        frame = frame@initial_gauge
    return frame.conj().T@operator@frame


def curvature(q, first=0, second=1, params=None):
    """Hermitian matrix i F†[d_i P,d_j P]F in the chosen analytic lower frame."""
    _, derivatives, _ = projector_data(q, params)
    frame = negative_frame(q, params)
    commutator = derivatives[first]@derivatives[second]-derivatives[second]@derivatives[first]
    return 1j*frame.conj().T@commutator@frame


def ambiguity(params=None):
    """The same physical state has incompatible individual eigenvector populations."""
    params = parameters() if params is None else params
    frame = negative_frame(params["base"], params)
    state = frame[:, 0]
    mix = (I2-1j*SY)/np.sqrt(2)
    swap = -1j*SY
    populations = [np.abs((frame@g).conj().T@state)**2 for g in (I2, mix, swap)]
    omega = curvature(params["base"], params=params)
    eigenvalues = np.linalg.eigvalsh(omega)
    spin = np.kron(I2, SZ)
    spin_matrix = frame.conj().T@spin@frame
    return {
        "individual_populations": [value.tolist() for value in populations],
        "cluster_population": float(np.vdot(state, frame@frame.conj().T@state).real),
        "curvature_eigenvalues": eigenvalues.tolist(),
        "curvature_trace": float(np.trace(omega).real),
        "spin_z_eigenvalues": np.linalg.eigvalsh(spin_matrix).tolist(),
    }


def physical_observables(operators, params=None):
    params = parameters() if params is None else params
    # One pure spinor is specified physically at the initial geometry.
    initial = negative_frame(params["base"], params)@np.array([np.sqrt(.7), 1j*np.sqrt(.3)])
    states = np.einsum("tij,j->ti", operators, initial)
    spin = np.array([np.kron(I2, pauli) for pauli in (SX, SY, SZ)])
    return np.einsum("ti,kij,tj->tk", states.conj(), spin, states).real


def finite_speed(times, duration, loop="xy", params=None):
    """Exact finite-speed electronic ODE i dpsi/ds = duration*h(q(s))*psi.

    The path is externally prescribed. Leakage is measured, not discarded,
    and comparison with geometric transport uses densities to remove phase.
    """
    times = _times(times)
    if not np.isfinite(duration) or duration <= 0:
        raise ValueError("duration must be finite and positive")
    params = parameters() if params is None else params
    initial = negative_frame(params["base"], params)@np.array([np.sqrt(.7), 1j*np.sqrt(.3)])
    def rhs(s, state):
        return -1j*duration*hamiltonian(path(s, loop, params)[0], params)@state
    result = solve_ivp(rhs, (0, 1), initial, method="DOP853", t_eval=times,
                       rtol=2e-12, atol=2e-14, max_step=min(.01, .15/duration))
    if not result.success:
        raise RuntimeError(result.message)
    return result.y.T


def diagnostics(times, operators, loop="xy", params=None, *, expected=None):
    p0 = projector_data(path(0, loop, params)[0], params)[0]
    norm, support, energy = [], [], []
    for s, operator in zip(times, operators):
        q = path(s, loop, params)[0]
        projector, _, gap_half = projector_data(q, params)
        norm.append(np.max(abs(operator.conj().T@operator-p0)))
        support.append(np.max(abs(projector@operator-operator)))
        energy.append(np.max(abs(hamiltonian(q, params)@operator+gap_half*operator)))
    result = {"maximum_partial_isometry_defect": float(max(norm)),
              "maximum_subspace_leakage_amplitude": float(max(support)),
              "maximum_instantaneous_eigenvalue_residual": float(max(energy))}
    if expected is not None:
        result["maximum_transport_operator_error"] = float(np.max(abs(operators-expected)))
    w = holonomy(operators[-1], params)
    result.update(holonomy_eigenphases=np.sort(np.angle(np.linalg.eigvals(w))).tolist(),
                  holonomy_trace_real=float(np.trace(w).real),
                  holonomy_trace_imag=float(np.trace(w).imag))
    return result


def run(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    source = Path(__file__).read_bytes()
    params, report, arrays, final = parameters(), {}, {}, {}
    for loop in ("xy", "yz"):
        records = []
        for steps in (32, 64, 128):
            times = np.linspace(0, 1, steps+1)
            expected = kato_reference(times, loop, params)
            result = polar_transport(times, loop, params)
            records.append({"steps": steps, "minimum_link_singular_value":
                            result.minimum_link_singular_value,
                            **diagnostics(times, result.operators, loop, params, expected=expected)})
            arrays[f"{loop}_{steps}_times"] = times
            arrays[f"{loop}_{steps}_polar"] = result.operators
            arrays[f"{loop}_{steps}_reference"] = expected
        gauged = polar_transport(times, loop, params, gauge=coordinate_gauge)
        refined = kato_reference(times, loop, params, rtol=3e-14, atol=3e-15, max_step=.0025)
        report[loop] = {"convergence": records,
                        "gauge_transport_max_error": float(np.max(abs(gauged.operators-result.operators))),
                        "gauge_spin_max_error": float(np.max(abs(physical_observables(gauged.operators)
                                                                - physical_observables(result.operators)))),
                        "reference_refinement_max_error": float(np.max(abs(refined-expected))),
                        "reference": diagnostics(times, expected, loop, params)}
        final[loop] = holonomy(expected[-1], params)
        arrays[f"{loop}_gauged_polar"] = gauged.operators
    commutator = final["xy"]@final["yz"]-final["yz"]@final["xy"]
    report["holonomy_commutator_frobenius_norm"] = float(np.linalg.norm(commutator))
    report["eigenvector_ambiguity"] = ambiguity(params)
    times = np.linspace(0, 1, 257)
    geometric = kato_reference(times, params=params)
    initial = negative_frame(params["base"], params)@np.array([np.sqrt(.7), 1j*np.sqrt(.3)])
    adiabatic = np.einsum("tij,j->ti", geometric, initial)
    adiabatic_density = np.einsum("ti,tj->tij", adiabatic, adiabatic.conj())
    finite_records = []
    for duration in (4., 20., 100.):
        states = finite_speed(times, duration, params=params)
        density = np.einsum("ti,tj->tij", states, states.conj())
        leakage = [np.linalg.norm((np.eye(4)-projector_data(path(s, params=params)[0], params)[0])
                                  @state)**2 for s, state in zip(times, states)]
        finite_records.append({"duration": duration, "maximum_upper_doublet_population": float(max(leakage)),
                               "maximum_density_error_vs_adiabatic": float(np.max(abs(density-adiabatic_density))),
                               "maximum_norm_defect": float(np.max(abs(np.sum(abs(states)**2, axis=1)-1)))})
        arrays[f"finite_speed_{int(duration)}_states"] = states
    report["finite_speed_distinct_equations"] = finite_records
    arrays["finite_speed_times"] = times
    arrays["finite_speed_geometric_reference"] = geometric
    (output/"benchmark_source.py").write_bytes(source)
    np.savez(output/"inputs.npz", **params, initial_spinor=np.array([np.sqrt(.7), 1j*np.sqrt(.3)]),
             time_reversal=TIME_REVERSAL)
    np.savez(output/"trajectories.npz", **arrays)
    if Path(__file__).read_bytes() != source:
        raise RuntimeError("benchmark source changed during the run")
    report["evidence"] = {
        "schema": 1, "scope": "prescribed-path adiabatic doublet geometry; no hopping or nuclear feedback",
        "source_sha256": hashlib.sha256(source).hexdigest(),
        "inputs_sha256": hashlib.sha256((output/"inputs.npz").read_bytes()).hexdigest(),
        "trajectories_sha256": hashlib.sha256((output/"trajectories.npz").read_bytes()).hexdigest(),
        "python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
        "primary_references": ["https://doi.org/10.1143/JPSJ.5.435",
                               "https://doi.org/10.1103/PhysRevLett.52.2111",
                               "https://doi.org/10.1103/PhysRevX.6.041031",
                               "https://doi.org/10.1021/acs.jpclett.2c01802"],
    }
    (output/"report.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="fresh output directory; refuses overwrite")
    args = parser.parse_args()
    print(json.dumps(run(args.output), indent=2))


if __name__ == "__main__":
    main()
