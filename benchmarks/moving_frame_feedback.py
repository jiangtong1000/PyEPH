"""Complete moving-frame Ehrenfest proof, not a general AO dynamics adapter.

Atomic units; one normalized carrier in a complete three-dimensional electronic
Hilbert space, two classical coordinates, and an explicit scalar reference.
The invertible frame and its full spatial connection are known analytically.
No electronic-structure total-energy interpretation is inferred from H and S.
"""

import argparse
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import platform

import jax
import jax.numpy as jnp
import numpy as np
import scipy
from scipy.integrate import solve_ivp
from scipy.linalg import expm

from pyeph import CoupledClassical, Ehrenfest, Execution, Integrator, Problem, Simulation
from pyeph.core.contracts import ModelSpec
from pyeph.core.state import make_state
from pyeph.core.system import SystemSpec
from pyeph.models.base import AutoDiffModel


MASS = np.array([1.2, .8])


def parameters():
    """Explicit illustrative fixed-frame parameters; no fitted material model."""
    return {
        "h0": np.array([[-.25, .14, .04j], [.14, .15, .08], [-.04j, .08, .45]]),
        "hx": np.array([[.28, .04, 0], [.04, -.2, .02], [0, .02, .12]]),
        "hy": np.array([[0, .1j, .03], [-.1j, .08, .04j], [.03, -.04j, -.1]]),
        "hz": np.diag([.2, -.3, .1]),
        "stiffness": np.array([.7, .9]),
        "quartic": .02,
        "nonlinear": .07,
    }


def physical_quantities(q, params=None):
    """Independent NumPy Hamiltonian and analytic physical derivatives."""
    p = parameters() if params is None else params
    x, y = q
    h = p["h0"] + x*p["hx"] + y*p["hy"] + p["nonlinear"]*np.sin(x*y)*p["hz"]
    dh = np.stack([p["hx"] + p["nonlinear"]*y*np.cos(x*y)*p["hz"],
                   p["hy"] + p["nonlinear"]*x*np.cos(x*y)*p["hz"]])
    reference = .5*np.dot(p["stiffness"], np.asarray(q)**2) + p["quartic"]*x*x*y*y
    gradient = p["stiffness"]*q + 2*p["quartic"]*np.array([x*y*y, y*x*x])
    return h, dh, reference, gradient


def _basic_frame(q):
    """Unitary rotation times positive scales times an invertible shear."""
    x, y = q
    generator = np.array([[.2, .3j, .12], [-.3j, -.1, .17j], [.12, -.17j, .05]])
    unitary = expm(-1j*x*generator)
    scale = np.diag(np.exp(y*np.array([.17, -.11, .08])))
    shear = np.eye(3, dtype=complex)
    shear[0, 1], shear[1, 2] = .13*x + .07j*y, .09*y - .06j*x
    dx, dy = np.zeros((3, 3), complex), np.zeros((3, 3), complex)
    dx[0, 1], dx[1, 2] = .13, -.06j
    dy[0, 1], dy[1, 2] = .07j, .09
    b = unitary @ scale @ shear
    db = np.stack([-1j*generator @ b + unitary @ scale @ dx,
                   unitary @ np.diag([.17, -.11, .08]) @ scale @ shear
                   + unitary @ scale @ dy])
    return b, db


def frame(q, kind="moving"):
    """Two coordinate-dependent nonunitary gauges of the same complete space."""
    if kind == "fixed":
        return np.eye(3, dtype=complex), np.zeros((2, 3, 3), dtype=complex)
    b, db = _basic_frame(q)
    if kind == "moving":
        return b, db
    if kind != "regauged":
        raise ValueError("kind must be fixed, moving or regauged")
    # A second invertible gauge G(q) changes both the metric and connection.
    g, dg_base = _basic_frame(np.array([-.6*q[1], .5*q[0]]))
    dg = np.stack([.5*dg_base[1], -.6*dg_base[0]])
    return b @ g, np.stack([db[i] @ g + b @ dg[i] for i in range(2)])


@dataclass(frozen=True)
class FrameData:
    metric: np.ndarray
    hamiltonian: np.ndarray
    connection: np.ndarray
    hamiltonian_derivative: np.ndarray


def transform(h, dh, b, db):
    """Construct S,H,A_i,dH_i from a known square complete embedding B.

    This research helper explicitly rejects rectangular or rank-deficient B.
    It is not an interface for inferring a connection from a learned metric.
    """
    h, dh, b, db = map(np.asarray, (h, dh, b, db))
    if (b.ndim != 2 or not b.shape[0] or b.shape[0] != b.shape[1] or h.shape != b.shape
            or dh.ndim != 3 or dh.shape[1:] != b.shape or db.shape != dh.shape):
        raise ValueError("complete square frame and matching spatial derivatives are required")
    if not all(np.isfinite(value).all() for value in (h, dh, b, db)):
        raise ValueError("frame data must be finite")
    singular = np.linalg.svd(b, compute_uv=False)
    if singular[-1] <= 1e-10 * singular[0]:
        raise ValueError("frame is singular or exceeds the declared conditioning limit")
    if not np.allclose(h, h.conj().T, atol=1e-13, rtol=1e-13):
        raise ValueError("physical Hamiltonian must be Hermitian")
    if not np.allclose(dh, dh.conj().transpose(0, 2, 1), atol=1e-13, rtol=1e-13):
        raise ValueError("physical Hamiltonian derivatives must be Hermitian")
    metric = b.conj().T @ b
    ham = b.conj().T @ h @ b
    connection = np.stack([b.conj().T @ derivative for derivative in db])
    gradient = np.stack([derivative.conj().T @ h @ b + b.conj().T @ dh[i] @ b
                         + b.conj().T @ h @ derivative for i, derivative in enumerate(db)])
    return FrameData(metric, ham, connection, gradient)


def covariant_gradient(data):
    """D_i H = d_i H - Gamma_i^dagger H - H Gamma_i, Gamma_i=S^-1 A_i."""
    gamma = np.stack([np.linalg.solve(data.metric, ai) for ai in data.connection])
    h = data.hamiltonian
    return np.stack([data.hamiltonian_derivative[i] - gi.conj().T @ h - h @ gi
                     for i, gi in enumerate(gamma)])


def electronic_derivative(data, c, velocity, *, connection="full"):
    if connection == "full":
        ai = data.connection
    elif connection == "metric_only":
        # Intentionally incomplete negative control: keeps dS=A+A† but removes
        # all anti-Hermitian information. It can preserve norm and still be wrong.
        ai = .5*(data.connection + data.connection.conj().transpose(0, 2, 1))
    else:
        raise ValueError("connection must be full or metric_only")
    return np.linalg.solve(data.metric,
                           -1j*data.hamiltonian @ c - np.einsum("i,ijk,k->j", velocity, ai, c))


def physical_force(data, c, reference_gradient):
    return -reference_gradient - np.einsum("a,iab,b->i", c.conj(), covariant_gradient(data), c).real


def initial():
    psi = np.array([.6, .3+.25j, -.4j])
    psi /= np.linalg.norm(psi)
    return np.array([-.7, .4]), np.array([.45, -.28]), psi


def pack(q, p, c):
    return np.r_[q, p, c.real, c.imag]


def unpack(row):
    return row[:2], row[2:4], row[4:7]+1j*row[7:10]


def fixed_rhs(time, row):
    q, p, psi = unpack(row)
    h, dh, _, gradient = physical_quantities(q)
    force = -gradient - np.einsum("a,iab,b->i", psi.conj(), dh, psi).real
    return pack(p/MASS, force, -1j*h@psi)


def moving_rhs(time, row, kind="moving", connection="full"):
    q, p, c = unpack(row)
    h, dh, _, gradient = physical_quantities(q)
    b, db = frame(q, kind)
    data = transform(h, dh, b, db)
    return pack(p/MASS, physical_force(data, c, gradient),
                electronic_derivative(data, c, p/MASS, connection=connection))


def reference(times, *, rtol=2e-12, atol=2e-14, max_step=.04):
    q, p, psi = initial()
    result = solve_ivp(fixed_rhs, (0., float(times[-1])), pack(q, p, psi),
                       method="DOP853", t_eval=times, rtol=rtol, atol=atol, max_step=max_step)
    if not result.success:
        raise RuntimeError(result.message)
    return result.y.T


def propagate_moving(dt, duration, *, kind="moving", connection="full"):
    """Unrenormalized fourth-order integration of the full coupled moving equations."""
    steps = int(round(duration/dt))
    if dt <= 0 or duration <= 0 or not np.isclose(steps*dt, duration, atol=1e-13, rtol=0):
        raise ValueError("positive dt must exactly divide the positive duration")
    times = np.arange(steps+1)*dt
    q, p, psi = initial()
    rows = [pack(q, p, np.linalg.solve(frame(q, kind)[0], psi))]
    for time in times[:-1]:
        row = rows[-1]
        k1 = moving_rhs(time, row, kind, connection)
        k2 = moving_rhs(time+dt/2, row+dt*k1/2, kind, connection)
        k3 = moving_rhs(time+dt/2, row+dt*k2/2, kind, connection)
        k4 = moving_rhs(time+dt, row+dt*k3, kind, connection)
        rows.append(row+dt*(k1+2*k2+2*k3+k4)/6)
    return times, np.array(rows)


def to_physical(rows, kind):
    return np.array([pack(q, p, frame(q, kind)[0] @ c) for q, p, c in map(unpack, rows)])


def diagnostics(rows, kind, expected):
    physical = to_physical(rows, kind)
    energies, norms, conditions = [], [], []
    for row in rows:
        q, p, c = unpack(row)
        h, _, reference_energy, _ = physical_quantities(q)
        b, _ = frame(q, kind)
        psi = b @ c
        energies.append(.5*np.sum(p*p/MASS)+reference_energy+np.vdot(psi, h@psi).real)
        norms.append(np.vdot(c, b.conj().T @ b @ c).real)
        conditions.append(np.linalg.cond(b))
    return {
        "maximum_coordinate_error": float(np.max(abs(physical[:, :2]-expected[:, :2]))),
        "maximum_momentum_error": float(np.max(abs(physical[:, 2:4]-expected[:, 2:4]))),
        "maximum_electronic_component_error": float(np.max(abs(physical[:, 4:]-expected[:, 4:]))),
        "maximum_metric_norm_defect": float(np.max(abs(np.array(norms)-1))),
        "maximum_total_energy_drift": float(np.max(abs(np.array(energies)-energies[0]))),
        "maximum_frame_condition": float(max(conditions)),
    }


@dataclass(frozen=True)
class FixedReferenceModel(AutoDiffModel):
    """The same physical equations through the existing fixed-basis contract."""

    spec: ModelSpec = field(default_factory=lambda: ModelSpec(
        SystemSpec(3, (2,), "complete-fixed-reference"), name="moving_frame_reference",
        complex_valued=True))

    def apply(self, params, q, vectors):
        h = (params["h0"] + q[0]*params["hx"] + q[1]*params["hy"]
             + params["nonlinear"]*jnp.sin(q[0]*q[1])*params["hz"])
        return h @ vectors

    def reference_energy(self, params, q):
        return .5*jnp.dot(params["stiffness"], q*q)+params["quartic"]*q[0]**2*q[1]**2


def propagate_native(dt, duration):
    model = FixedReferenceModel()
    params = jax.tree.map(jnp.asarray, parameters())
    simulation = Simulation(Problem(model, params, CoupledClassical(MASS), Ehrenfest()),
                            Integrator(dt, electronic="exponential_midpoint"),
                            Execution(chunk_size=64))
    result = simulation.run(make_state(*initial()), int(round(duration/dt)))
    final = result.final_state
    return pack(np.asarray(final.q), np.asarray(final.p), np.asarray(final.electronic))


def metric_counterexample():
    """Identical raw H,S and derivatives can describe different physical forces."""
    h0 = np.diag([.3, -.3]).astype(complex)
    generator = np.array([[0., -.7], [.7, 0.]])
    b = expm(.2*generator)
    physical_h = b @ h0 @ b.conj().T
    physical_dh = generator @ physical_h - physical_h @ generator
    data = transform(physical_h, physical_dh[None], b, (generator@b)[None])
    c = np.ones(2)/np.sqrt(2.)
    return {"raw_h_error": float(np.max(abs(data.hamiltonian-h0))),
            "raw_metric_error": float(np.max(abs(data.metric-np.eye(2)))),
            "raw_h_derivative": float(np.max(abs(data.hamiltonian_derivative))),
            "metric_derivative": float(np.max(abs(data.connection[0]+data.connection[0].conj().T))),
            "connection_norm": float(np.linalg.norm(data.connection)),
            "stationary_embedding_force": 0.,
            "rotating_embedding_force": float(physical_force(data, c, np.zeros(1))[0])}


def run(output):
    """Save raw convergence trajectories and exact source/parameter evidence."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    if not jax.config.x64_enabled:
        raise ValueError("this qualification run requires explicit JAX x64")
    duration, dts = 4., (.08, .04, .02)
    report, arrays = {}, {}
    for kind in ("moving", "regauged"):
        records = []
        for index, dt in enumerate(dts):
            times, rows = propagate_moving(dt, duration, kind=kind)
            expected = reference(times)
            records.append({"dt": dt, **diagnostics(rows, kind, expected)})
            arrays[f"{kind}_{index}_times"] = times
            arrays[f"{kind}_{index}_moving"] = rows
            arrays[f"{kind}_{index}_reference"] = expected
        report[kind] = records
    native = []
    reference_final = reference(np.array([0., duration]))[-1]
    for dt in dts:
        final = propagate_native(dt, duration)
        native.append({"dt": dt, "maximum_final_error": float(np.max(abs(final-reference_final)))})
    times, bad_rows = propagate_moving(.02, duration, connection="metric_only")
    report["metric_only_negative_control"] = diagnostics(bad_rows, "moving", reference(times))
    arrays["metric_only_rows"] = bad_rows
    report["native_fixed_frame"] = native
    report["same_H_S_counterexample"] = metric_counterexample()
    tight = reference(np.array([0., duration]), rtol=3e-14, atol=3e-15, max_step=.01)[-1]
    report["reference_refinement_max_final_change"] = float(np.max(abs(tight-reference_final)))
    source = Path(__file__).read_bytes()
    (output/"benchmark_source.py").write_bytes(source)
    inputs = parameters()
    inputs.update(mass=MASS, initial_q=initial()[0], initial_p=initial()[1], initial_psi=initial()[2])
    np.savez(output/"inputs.npz", **inputs)
    np.savez(output/"trajectories.npz", **arrays)
    report["evidence"] = {
        "schema": 1, "scope": "complete square moving-frame numerical proof; atomic units",
        "duration": duration, "source_sha256": hashlib.sha256(source).hexdigest(),
        "inputs_sha256": hashlib.sha256((output/"inputs.npz").read_bytes()).hexdigest(),
        "trajectories_sha256": hashlib.sha256((output/"trajectories.npz").read_bytes()).hexdigest(),
        "python": platform.python_version(), "numpy": np.__version__,
        "scipy": scipy.__version__, "jax": jax.__version__, "jax_x64": jax.config.x64_enabled,
        "primary_references": ["https://doi.org/10.1103/PhysRevB.95.115155",
                               "https://doi.org/10.1063/1.3700800"],
    }
    (output/"report.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="new result directory; refuses overwrite")
    args = parser.parse_args()
    jax.config.update("jax_enable_x64", True)
    print(json.dumps(run(args.output), indent=2))


if __name__ == "__main__":
    main()
