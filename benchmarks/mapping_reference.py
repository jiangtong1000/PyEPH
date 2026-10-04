"""Independent analytic-projector / DOP853 reference for real three-state RM.

Atomic units, one normalized mapping vector in a fixed orthonormal electronic
basis, two classical canonical coordinates and an explicit scalar reference.
These newly declared fixtures are numerical qualifications, not material models
or tests of an ensemble preparation/estimator. No reference software is needed.
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

import pyeph
from pyeph import CoupledClassical, Execution, Integrator, MASHRM, Problem, Simulation
from pyeph.core.contracts import ModelSpec
from pyeph.core.system import SystemSpec
from pyeph.dynamics.mashrm import MASHRMPopulation, mapping_state
from pyeph.models.base import AutoDiffModel


MASS = np.array([1.3, .8])


def parameters():
    """All changing model parameters are explicit; nothing is fitted to data."""
    return {
        "levels": np.array([-.3, .1, .65]),
        "slopes": np.array([[.04, -.02, 0.], [0., 0., .03]]),
        "angles": np.array([.7, .17, .3, .11]),
        "stiffness": np.array([.04, .048]),
        "basis_rotation": np.eye(3),
    }


def rotation(angle, pair):
    result, derivative = np.eye(3), np.zeros((3, 3))
    a, b = pair
    co, si = np.cos(angle), np.sin(angle)
    result[a, a] = result[b, b] = co
    result[a, b], result[b, a] = -si, si
    derivative[a, a] = derivative[b, b] = -si
    derivative[a, b], derivative[b, a] = -co, co
    return result, derivative


def rotated_parameters():
    params = parameters()
    params["basis_rotation"] = rotation(.43, (0, 1))[0] @ rotation(-.37, (1, 2))[0]
    return params


def spectral_data(q, params=None):
    """Known eigenvectors and their analytic derivatives; no diagonalization."""
    p = parameters() if params is None else params
    a, b, c, d = p["angles"]
    first, dfirst = rotation(a*q[0]+b*np.sin(q[1]), (0, 1))
    second, dsecond = rotation(c*q[1]+d*q[0], (1, 2))
    u = first @ second
    du = np.stack([a*dfirst@second+d*first@dsecond,
                   b*np.cos(q[1])*dfirst@second+c*first@dsecond])
    transform = p["basis_rotation"]
    return (p["levels"]+np.asarray(q)@p["slopes"], transform@u,
            np.einsum("ab,ibc->iac", transform, du))


def projectors(q, params=None):
    _, u, du = spectral_data(q, params)
    projectors = np.einsum("ia,ja->aij", u, u)
    derivatives = (np.einsum("kia,ja->kaij", du, u)
                   + np.einsum("ia,kja->kaij", u, du))
    return projectors, derivatives


def populations(q, c, params=None):
    return np.abs(spectral_data(q, params)[1].T@c)**2


def direction(q, c, active, competitor, params=None):
    """Half the analytic projector-margin gradient, holding fixed-basis c fixed."""
    _, derivatives = projectors(q, params)
    return .5*np.einsum("a,iab,b->i", c.conj(),
                        derivatives[:, active]-derivatives[:, competitor], c).real


def pack(q, p, c):
    return np.r_[q, p, c.real, c.imag]


def unpack(row):
    return row[:2], row[2:4], row[4:7]+1j*row[7:10]


def smooth_rhs(time, row, active, params=None):
    params = parameters() if params is None else params
    q, p, c = unpack(row)
    levels, u, _ = spectral_data(q, params)
    h = (u*levels)@u.T
    force = -params["stiffness"]*q-params["slopes"][:, active]
    return pack(p/MASS, force, -1j*h@c)


def total_energy(row, active, params=None):
    params = parameters() if params is None else params
    q, p, _ = unpack(row)
    return float(.5*np.dot(p/MASS, p)+.5*np.dot(params["stiffness"], q*q)
                 + spectral_data(q, params)[0][active])


def impulse(row, active, competitor, params=None):
    """Independent mass-weighted energy step or reflection at a pair boundary."""
    q, p, c = unpack(row)
    delta = direction(q, c, active, competitor, params)
    normal = delta/np.sqrt(MASS)
    magnitude = np.linalg.norm(normal)
    if magnitude < 1e-12:
        raise ValueError("oracle profile excludes zero impulse directions")
    normal /= magnitude
    mass_momentum = p/np.sqrt(MASS)
    parallel = np.dot(mass_momentum, normal)
    if parallel >= -1e-10:
        raise ValueError("oracle requires an isolated incoming crossing")
    levels = spectral_data(q, params)[0]
    gap = levels[competitor]-levels[active]
    discriminant = parallel**2-2*gap
    if abs(discriminant) < 1e-10:
        raise ValueError("oracle profile excludes threshold impulses")
    accepted = bool(discriminant > 0)
    outgoing = np.copysign(np.sqrt(discriminant), parallel) if accepted else -parallel
    new_p = np.sqrt(MASS)*(mass_momentum+(outgoing-parallel)*normal)
    new_active = competitor if accepted else active
    updated = pack(q, new_p, c)
    outgoing_delta = direction(q, c, new_active, active if accepted else competitor, params)
    return updated, new_active, {
        "active_before": int(active), "competitor": int(competitor),
        "active_after": int(new_active), "accepted": accepted,
        "energy_error": abs(total_energy(updated, new_active, params)
                            - total_energy(row, active, params)),
        "incoming_rate": float(2*np.dot(delta, p/MASS)),
        "outgoing_rate": float(2*np.dot(outgoing_delta, new_p/MASS)),
        "direction": delta.tolist(), "normal": normal.tolist(),
        "parallel_kinetic": float(.5*parallel**2), "energy_gap": float(gap),
    }


def fixture(kind, params=None):
    """Declare a crossing, then integrate backward to a smooth incoming state.

    This gives one deterministic state, not an RM ensemble preparation. The
    spectator has nonzero population and a distinct phase in both fixtures.
    """
    if kind not in ("accepted", "frustrated"):
        raise ValueError("kind must be accepted or frustrated")
    params = parameters() if params is None else params
    q = np.array([-.23, .31])
    amplitudes = np.sqrt([.47, .47, .06])*np.exp(1j*np.array([0., .27, -.55]))
    levels, u, _ = spectral_data(q, params)
    c = u@amplitudes
    normal = direction(q, c, 0, 1, params)/np.sqrt(MASS)
    normal /= np.linalg.norm(normal)
    factor = 1.8 if kind == "accepted" else .25
    parallel = -np.sqrt(2*factor*(levels[1]-levels[0]))
    tangent = np.array([-normal[1], normal[0]])
    p = np.sqrt(MASS)*(parallel*normal+.2*tangent)
    crossing = pack(q, p, c)
    lead = .173
    solution = solve_ivp(lambda t, row: smooth_rhs(t, row, 0, params), (0., -lead), crossing,
                         method="DOP853", rtol=3e-14, atol=3e-15, max_step=.005)
    if not solution.success:
        raise RuntimeError(solution.message)
    initial = solution.y[:, -1]
    initial_q, _, initial_c = unpack(initial)
    if populations(initial_q, initial_c, params).argmax() != 0:
        raise RuntimeError("declared backward fixture is not in the incoming region")
    return initial, {"kind": kind, "declared_crossing_time": lead,
                     "crossing_row": crossing.tolist(), "initial_row": initial.tolist(),
                     "adiabatic_population": [.47, .47, .06],
                     "adiabatic_phase": [0., .27, -.55],
                     "parallel_kinetic_to_gap_ratio": factor, "tangent_momentum": .2}


@dataclass
class ReferenceResult:
    times: np.ndarray
    rows: np.ndarray
    active: np.ndarray
    events: list


def reference(kind, times, params=None, *, rtol=2e-12, atol=2e-14, max_step=.01):
    """Piecewise DOP853 with analytic margin roots and independent impulses.

    No production diagonalizer, force, event locator or impulse helper is used.
    Sampling is right-continuous at an impulse, with unmodified q and c.
    """
    times = np.asarray(times, dtype=float)
    if (times.ndim != 1 or len(times) < 2 or times[0] != 0 or not np.isfinite(times).all()
            or np.any(np.diff(times) <= 0)):
        raise ValueError("times must start at zero and be finite and strictly increasing")
    params = parameters() if params is None else params
    row, _ = fixture(kind, params)
    start, active, segments, events = 0., 0, [], []
    while start < times[-1]:
        competitors = [other for other in range(3) if other != active]
        event_functions = []
        for competitor in competitors:
            def margin(time, state, other=competitor, incoming=active):
                q, _, c = unpack(state)
                pop = populations(q, c, params)
                return pop[incoming]-pop[other]
            margin.direction, margin.terminal = -1, True
            event_functions.append(margin)
        result = solve_ivp(lambda t, state: smooth_rhs(t, state, active, params),
                           (start, times[-1]), row, method="DOP853", dense_output=True,
                           events=event_functions, rtol=rtol, atol=atol, max_step=max_step)
        if not result.success:
            raise RuntimeError(result.message)
        end = float(result.t[-1])
        segments.append((start, end, active, result.sol))
        if result.status == 0:
            break
        hits = [i for i, roots in enumerate(result.t_events) if len(roots)]
        if len(hits) != 1 or end <= start+1e-10 or len(events) >= 16:
            raise ValueError("oracle profile excludes simultaneous, grazing or excess events")
        competitor = competitors[hits[0]]
        before = result.y[:, -1]
        q, _, c = unpack(before)
        pop = populations(q, c, params)
        spectator = next(i for i in range(3) if i not in (active, competitor))
        if abs(pop[active]-pop[competitor]) > 1e-9 or pop[active]-pop[spectator] < 1e-5:
            raise ValueError("oracle requires an isolated two-state largest-population tie")
        row, active, record = impulse(before, active, competitor, params)
        record.update(time=end, before=before.tolist(), after=row.tolist(),
                      residual=float(abs(pop[record["active_before"]]-pop[competitor])))
        events.append(record)
        start = end
    rows, surfaces = [], []
    for time in times:
        # The last segment containing a boundary gives its outgoing state.
        index = max(i for i, segment in enumerate(segments) if segment[0] <= time)
        _, _, surface, solution = segments[index]
        sampled = solution(time)
        if events and time == events[-1]["time"] and time == times[-1]:
            sampled, surface = np.asarray(events[-1]["after"]), events[-1]["active_after"]
        rows.append(sampled)
        surfaces.append(surface)
    return ReferenceResult(times, np.array(rows), np.array(surfaces), events)


@dataclass(frozen=True)
class RotatingThreeStateModel(AutoDiffModel):
    """Native system under test; the oracle above does not call this model."""

    spec: ModelSpec = field(default_factory=lambda: ModelSpec(
        SystemSpec(3, (2,), "analytic-three-state-fixed"), name="mapping_reference"))

    def apply(self, params, q, vectors):
        a, b, c, d = params["angles"]
        theta, phi = a*q[0]+b*jnp.sin(q[1]), c*q[1]+d*q[0]
        ct, st, cp, sp = jnp.cos(theta), jnp.sin(theta), jnp.cos(phi), jnp.sin(phi)
        u = params["basis_rotation"] @ jnp.array(
            [[ct, -st*cp, st*sp], [st, ct*cp, -ct*sp], [0., sp, cp]])
        levels = params["levels"]+q@params["slopes"]
        return (u*levels) @ (u.T@vectors)

    def reference_energy(self, params, q):
        return .5*jnp.dot(params["stiffness"], q*q)


@dataclass(frozen=True)
class TraceMeasurement:
    supports_mashrm = True

    def validate(self, problem):
        MASHRMPopulation().validate(problem)

    def evaluate(self, problem, state):
        result = MASHRMPopulation(include_nuclei=True).evaluate(problem, state)
        return dict(result, electronic=state.electronic)


def propagate_native(kind, dt, duration, params=None):
    if dt <= 0 or duration <= 0 or not np.isclose(round(duration/dt)*dt, duration,
                                                 atol=1e-13, rtol=0):
        raise ValueError("positive dt must divide positive duration")
    params = parameters() if params is None else params
    initial, _ = fixture(kind, params)
    model = RotatingThreeStateModel()
    native_params = jax.tree.map(jnp.asarray, params)
    problem = Problem(model, native_params, CoupledClassical(MASS), MASHRM(
        event_substeps=2, event_tolerance=1e-11, event_time_tolerance=1e-11), TraceMeasurement())
    simulation = Simulation(problem, Integrator(dt, electronic="exponential_midpoint"),
                            Execution(chunk_size=32))
    prepared = mapping_state(model, native_params, *unpack(initial), active=0)
    result = simulation.run(prepared, int(round(duration/dt)))
    values = result.observables
    rows = np.column_stack([values["q"], values["p"], values["electronic"].real,
                            values["electronic"].imag])
    return np.asarray(result.times), rows, {name: np.asarray(value) for name, value in values.items()}


def diagnostics(rows, active, params=None, expected=None):
    energy = np.array([total_energy(row, surface, params) for row, surface in zip(rows, active)])
    norms = np.array([np.vdot(unpack(row)[2], unpack(row)[2]).real for row in rows])
    result = {"maximum_energy_drift": float(np.max(abs(energy-energy[0]))),
              "maximum_norm_defect": float(np.max(abs(norms-1)))}
    if expected is not None:
        for name, selection in (("coordinate", slice(0, 2)), ("momentum", slice(2, 4)),
                                 ("electronic_component", slice(4, 10))):
            result[f"maximum_{name}_error"] = float(np.max(abs(rows[:, selection]
                                                               - expected[:, selection])))
    return result


def run(output):
    """Save immutable input/source snapshots and complete native/reference traces."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    if not jax.config.x64_enabled:
        raise ValueError("this qualification requires explicit JAX x64")
    package = Path(pyeph.__file__).parent
    runtime_sources = {str(path.relative_to(package)): path.read_bytes()
                       for path in sorted(package.rglob("*.py"))}
    for name, content in runtime_sources.items():
        destination = output/"runtime_sources"/name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)
    duration, dts, report, arrays, inputs = .8, (.08, .04, .02), {}, {}, parameters()
    inputs["mass"] = MASS
    for kind in ("accepted", "frustrated"):
        initial, declaration = fixture(kind)
        inputs[f"{kind}_initial"] = initial
        records = []
        for index, dt in enumerate(dts):
            times, rows, values = propagate_native(kind, dt, duration)
            expected = reference(kind, times)
            records.append({"dt": dt, **diagnostics(rows, values["active"], expected=expected.rows),
                            "accepted": int(values["accepted"][-1]),
                            "frustrated": int(values["frustrated"][-1])})
            arrays[f"{kind}_{index}_times"] = times
            arrays[f"{kind}_{index}_native"] = rows
            arrays[f"{kind}_{index}_reference"] = expected.rows
            arrays[f"{kind}_{index}_active"] = values["active"]
        tight = reference(kind, times, rtol=3e-14, atol=3e-15, max_step=.0025)
        report[kind] = {
            "declaration": declaration, "native": records, "events": expected.events,
            "reference": diagnostics(expected.rows, expected.active),
            "reference_refinement_max_change": float(np.max(abs(tight.rows-expected.rows))),
            "reference_refinement_event_time_change": float(max(
                abs(a["time"]-b["time"]) for a, b in zip(tight.events, expected.events))),
        }
        transformed = rotated_parameters()
        rotated_times, rotated_rows, rotated_values = propagate_native(kind, dts[-1], duration,
                                                                        transformed)
        rotation_matrix = transformed["basis_rotation"]
        recovered = rotated_rows.copy()
        recovered[:, 4:7] = rotated_rows[:, 4:7]@rotation_matrix
        recovered[:, 7:10] = rotated_rows[:, 7:10]@rotation_matrix
        report[kind]["native_basis_covariance_max_error"] = float(np.max(abs(rows-recovered)))
        report[kind]["native_basis_covariance_active_equal"] = bool(np.array_equal(
            values["active"], rotated_values["active"]))
        arrays[f"{kind}_rotated_times"] = rotated_times
        arrays[f"{kind}_rotated_native"] = rotated_rows
    source = Path(__file__).read_bytes()
    (output/"benchmark_source.py").write_bytes(source)
    np.savez(output/"inputs.npz", **inputs)
    np.savez(output/"trajectories.npz", **arrays)
    if any((package/name).read_bytes() != content for name, content in runtime_sources.items()):
        raise RuntimeError("runtime source changed during qualification; use a new output directory")
    report["evidence"] = {
        "schema": 1, "scope": "real isolated three-state deterministic RM numerical reference",
        "duration": duration, "source_sha256": hashlib.sha256(source).hexdigest(),
        "inputs_sha256": hashlib.sha256((output/"inputs.npz").read_bytes()).hexdigest(),
        "trajectories_sha256": hashlib.sha256((output/"trajectories.npz").read_bytes()).hexdigest(),
        "runtime_source_sha256": {name: hashlib.sha256(content).hexdigest()
                                   for name, content in runtime_sources.items()},
        "python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
        "jax": jax.__version__, "jax_x64": jax.config.x64_enabled,
        "primary_reference": "https://doi.org/10.1063/5.0226001",
    }
    (output/"report.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="new output directory; refuses overwrite")
    args = parser.parse_args()
    jax.config.update("jax_enable_x64", True)
    print(json.dumps(run(args.output), indent=2))


if __name__ == "__main__":
    main()
