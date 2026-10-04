#!/usr/bin/env python3
"""Independent small SciPy event-driven references for real two-state MASH.

The oracle uses explicit NumPy Hamiltonians, analytic surface forces, a
gauge-invariant equator, and one-dimensional momentum impulses. It never calls
production dynamics, force, surface, event-localization or impulse helpers.
"""

import argparse
from dataclasses import dataclass, field
import json
from pathlib import Path
import platform
import time

import jax
import jax.numpy as jnp
import numpy as np
import scipy
from scipy.integrate import solve_ivp

from pyeph import CoupledClassical, Execution, Integrator, Problem, Simulation
from pyeph.core.contracts import ModelSpec
from pyeph.core.system import SystemSpec
from pyeph.dynamics.mash2 import MASH2, MASHPopulation, mapping_state
from pyeph.models.analytic import TullyModel
from pyeph.models.base import AutoDiffModel


@dataclass(frozen=True)
class NonlinearRotatingModel(AutoDiffModel):
    """A confined rotating-angle/gap fixture; not a material parametrization."""

    spec: ModelSpec = field(default_factory=lambda: ModelSpec(
        SystemSpec(2, (1,), coordinate_kind="canonical"), name="nonlinear_rotating_reference"))

    def apply(self, params, q, vectors):
        x = q[0]
        angle = params["rotation"]*x + params["angle_curve"]*jnp.sin(x)
        rho = params["gap_half"] + params["gap_curve"]*jnp.cos(x)
        z, v = rho*jnp.cos(angle), rho*jnp.sin(angle)
        return jnp.array([[z, v], [v, -z]]) @ vectors

    def reference_energy(self, params, q):
        return .5*params["spring"]*q[0]**2


def rotating_params():
    return dict(rotation=1.3, angle_curve=.35, gap_half=.2, gap_curve=.035, spring=.25)


def numpy_quantities(kind, params, q):
    """Return h, dh/dq, Vref, dVref/dq using independent scalar formulas."""
    if kind == "rotating":
        theta = params["rotation"]*q + params["angle_curve"]*np.sin(q)
        theta_prime = params["rotation"] + params["angle_curve"]*np.cos(q)
        rho = params["gap_half"] + params["gap_curve"]*np.cos(q)
        rho_prime = -params["gap_curve"]*np.sin(q)
        z, v = rho*np.cos(theta), rho*np.sin(theta)
        dz = rho_prime*np.cos(theta)-rho*np.sin(theta)*theta_prime
        dv = rho_prime*np.sin(theta)+rho*np.cos(theta)*theta_prime
        ref, ref_prime = .5*params["spring"]*q*q, params["spring"]*q
    elif kind == "tully1":
        a, b, c, d = (params[name] for name in ("a", "b", "c", "d"))
        z = a*(1 if q >= 0 else -1)*(1-np.exp(-b*abs(q)))
        dz = a*b*np.exp(-b*abs(q))
        v = c*np.exp(-d*q*q)
        dv = -2*d*q*v
        ref, ref_prime = 0., 0.
    else:
        raise ValueError(f"unknown independent fixture {kind}")
    return np.array([[z, v], [v, -z]]), np.array([[dz, dv], [dv, -dz]]), ref, ref_prime


def _geometry(kind, params, q):
    h, dh, ref, ref_prime = numpy_quantities(kind, params, q)
    rho = np.sqrt(h[0, 0]**2+h[0, 1]**2)
    if rho <= 1e-8:
        raise ValueError("reference fixture must retain a nonzero gap")
    rho_prime = (h[0, 0]*dh[0, 0]+h[0, 1]*dh[0, 1])/rho
    axis = h/rho
    axis_prime = dh/rho-h*rho_prime/rho**2
    return h, rho, rho_prime, ref, ref_prime, axis, axis_prime


def _vector(y):
    return y[2:4]+1j*y[4:6]


def _equator(y, kind, params):
    c = _vector(y)
    axis = _geometry(kind, params, y[0])[5]
    return float(np.vdot(c, axis @ c).real)


def _equator_derivative(y, kind, params, mass):
    # [h,h/rho]=0: the electronic commutator vanishes in d(c† axis c)/dt.
    c = _vector(y)
    return float(y[1]/mass*np.vdot(c, _geometry(kind, params, y[0])[6] @ c).real)


def initial_vector(kind, params, q, z, phi):
    """Independent spinor with an explicitly fixed analytic real eigenvector gauge."""
    if not -1 <= z <= 1:
        raise ValueError("z must be in [-1,1]")
    h = numpy_quantities(kind, params, q)[0]
    angle = np.arctan2(h[0, 1], h[0, 0])
    rotation = np.array([[-np.sin(angle/2), np.cos(angle/2)],
                         [np.cos(angle/2), np.sin(angle/2)]])
    adiabatic = np.array([np.sqrt((1-z)/2), np.sqrt((1+z)/2)*np.exp(-1j*phi)])
    return rotation @ adiabatic


def scipy_mash(kind, params, *, q, p, c, active, mass, stop, sample_times=None,
               rtol=2e-11, atol=2e-13, max_step=.05, event_limit=100):
    """Integrate the continuum MASH equations and each one-dimensional jump.

    At a root only the incoming-to-outgoing boundary is terminal. After an
    impulse the outgoing derivative is checked and the zero at exactly the
    segment's initial time is suppressed; no finite time displacement is used.
    Equatorial initial data can be incoming or outgoing and are handled once.
    """
    if active not in (0, 1) or mass <= 0 or stop <= 0:
        raise ValueError("invalid active surface, mass or duration")
    y = np.r_[q, p, np.asarray(c).real, np.asarray(c).imag].astype(float)
    if abs(np.vdot(c, c).real-1) > 1e-9:
        raise ValueError("reference mapping vector must be normalized")
    clock, segments, events = 0., [], []

    def jump(y, t, surface):
        _, rho, _, ref, _, _, _ = _geometry(kind, params, y[0])
        before = y[1]
        energy_change = (1-2*surface)*2*rho
        radicand = before**2-2*mass*energy_change
        accepted = bool(radicand >= 0)
        if abs(before) < 1e-10:
            raise ValueError("reference event is a grazing incident trajectory")
        outgoing = np.sign(before)*np.sqrt(radicand) if accepted else -before
        next_surface = 1-surface if accepted else surface
        updated = y.copy()
        updated[1] = outgoing
        incoming_rate = (2*surface-1)*_equator_derivative(y, kind, params, mass)
        outgoing_rate = (2*next_surface-1)*_equator_derivative(updated, kind, params, mass)
        if incoming_rate >= -1e-11 or outgoing_rate <= 1e-11:
            raise ValueError("reference event lacks a resolved incoming/outgoing crossing")
        events.append(dict(time=float(t), q=float(y[0]), momentum_before=float(before),
                           momentum_after=float(outgoing), active_before=surface,
                           active_after=next_surface, accepted=accepted, gap=float(2*rho),
                           equator_residual=abs(_equator(y, kind, params)),
                           incoming_signed_rate=incoming_rate, outgoing_signed_rate=outgoing_rate,
                           impulse_energy_error=float((outgoing**2-before**2)/(2*mass)
                                                      + (energy_change if accepted else 0))))
        return updated, next_surface

    signed = (2*active-1)*_equator(y, kind, params)
    if signed < -1e-10:
        raise ValueError("reference initial surface disagrees with hemisphere")
    if abs(signed) < 1e-12:
        rate = (2*active-1)*_equator_derivative(y, kind, params, mass)
        if rate < -1e-11:
            y, active = jump(y, 0., active)
        elif rate <= 1e-11:
            raise ValueError("grazing equatorial initial condition")

    while clock < stop:
        side, start = 2*active-1, clock

        def rhs(t, state):
            h, _, rho_prime, _, ref_prime, _, _ = _geometry(kind, params, state[0])
            dc = -1j*h @ _vector(state)
            return np.r_[state[1]/mass, -ref_prime-side*rho_prime, dc.real, dc.imag]

        def boundary(t, state):
            value = side*_equator(state, kind, params)
            if t == start and abs(value) < 1e-11:
                return 1e-12
            return value

        boundary.direction = -1
        boundary.terminal = True
        result = solve_ivp(rhs, (start, stop), y, events=boundary, dense_output=True,
                           rtol=rtol, atol=atol, max_step=max_step, method="DOP853")
        if not result.success:
            raise RuntimeError(result.message)
        clock, y = float(result.t[-1]), result.y[:, -1]
        segments.append((start, clock, active, result.sol))
        if not result.t_events[0].size:
            break
        if len(events) >= event_limit or clock <= start:
            raise RuntimeError("reference event limit or repeated zero-time event")
        y, active = jump(y, clock, active)

    times = np.linspace(0., stop, 101) if sample_times is None else np.asarray(sample_times)
    if np.any(times < 0) or np.any(times > stop):
        raise ValueError("reference sample times outside integration interval")
    samples, surfaces = [], []
    for t in times:
        # A terminal impulse can occur exactly at stop without creating a new
        # dense segment. The final sample must still use its outgoing state.
        if t == stop:
            samples.append(y.copy())
            surfaces.append(active)
            continue
        # Select the outgoing segment at other event endpoints.
        index = min(np.searchsorted([s[1] for s in segments], t, side="right"), len(segments)-1)
        segment = segments[index]
        samples.append(segment[3](float(t)))
        surfaces.append(segment[2])
    states = np.asarray(samples)
    surfaces = np.asarray(surfaces)
    energies = []
    for row, surface in zip(states, surfaces, strict=True):
        _, rho, _, ref, _, _, _ = _geometry(kind, params, row[0])
        energies.append(row[1]**2/(2*mass)+ref+(2*surface-1)*rho)
    return dict(final_state=y, final_active=active, events=events, times=times,
                states=states, active=surfaces, energies=np.asarray(energies),
                rtol=rtol, atol=atol, max_step=max_step)


def _production(case, dt, subdivisions, tolerance):
    model = NonlinearRotatingModel() if case["kind"] == "rotating" else TullyModel(1)
    c = initial_vector(case["kind"], case["params"], case["q"], case["z"], case["phi"])
    # The production constructor only supplies validated state/diagnostic fields.
    # Replace c with the independently constructed fixed-diabatic initial vector.
    spin = [np.sqrt(1-case["z"]**2)*np.cos(case["phi"]),
            np.sqrt(1-case["z"]**2)*np.sin(case["phi"]), case["z"]]
    initial = mapping_state(model, case["params"], [case["q"]], [case["p"]], spin,
                            active=case["active"])._replace(electronic=jnp.asarray(c))
    steps = round(case["stop"]/dt)
    if not np.isclose(steps*dt, case["stop"]):
        raise ValueError("reference duration must be divisible by production dt")
    method = MASH2(event_substeps=subdivisions, event_tolerance=tolerance)
    simulation = Simulation(Problem(model, case["params"], CoupledClassical(case["mass"]),
                                    method, MASHPopulation(include_nuclei=True)),
                            Integrator(dt, electronic="exponential_midpoint"),
                            Execution(chunk_size=min(steps, 128), save_every=1))
    return simulation.run(initial, steps)


def cases():
    return [
        dict(name="rotating_mixed", kind="rotating", params=rotating_params(),
             q=-1., p=1.6, z=-.4, phi=0., active=0, mass=1., stop=25.,
             dts=[.1, .05, .025], oracle_max_step=.05, expected_accepted=8, expected_frustrated=1),
        dict(name="rotating_frustrated", kind="rotating", params=rotating_params(),
             q=-1., p=.45, z=-.4, phi=0., active=0, mass=1., stop=25.,
             dts=[.1, .05, .025], oracle_max_step=.05, expected_accepted=0, expected_frustrated=5),
        dict(name="tully1_accepted", kind="tully1", params=dict(a=.01, b=1.6, c=.005, d=1.),
             q=-4., p=20., z=-.4, phi=0., active=0, mass=2000., stop=1000.,
             # Resolve the change in curvature at q=0 before the crossing event.
             # These oracle settings were checked by independent step/tolerance refinement.
             dts=[2., 1., .5], oracle_max_step=1., oracle_rtol=2e-12, oracle_atol=2e-14,
             expected_accepted=1, expected_frustrated=0),
    ]


def reference_for(case, *, tight=False, sample_times=None):
    c = initial_vector(case["kind"], case["params"], case["q"], case["z"], case["phi"])
    refinement = 10 if tight else 1
    return scipy_mash(case["kind"], case["params"], q=case["q"], p=case["p"], c=c,
                      active=case["active"], mass=case["mass"], stop=case["stop"],
                      sample_times=sample_times, max_step=case["oracle_max_step"]/(2 if tight else 1),
                      rtol=case.get("oracle_rtol", 2e-11)/refinement,
                      atol=case.get("oracle_atol", 2e-13)/refinement)


def _packed(state):
    c = np.asarray(state.electronic)
    return np.r_[np.asarray(state.q), np.asarray(state.p), c.real, c.imag]


def compare_case(case, *, localization_checks=False):
    fine_dt = min(case["dts"])
    times = np.linspace(0., case["stop"], round(case["stop"]/fine_dt)+1)
    reference = reference_for(case, sample_times=times)
    tight = reference_for(case, tight=True, sample_times=times)
    event_times = np.array([e["time"] for e in reference["events"]])
    accepted = sum(e["accepted"] for e in reference["events"])
    frustrated = len(reference["events"])-accepted
    if (accepted, frustrated) != (case["expected_accepted"], case["expected_frustrated"]):
        raise RuntimeError("independent reference event sequence changed")
    if len(reference["events"]) != len(tight["events"]):
        raise RuntimeError("reference step refinement changed the detected event count")
    audit = {"final_state_difference": float(np.max(abs(reference["final_state"]-tight["final_state"]))),
             "max_event_time_difference": float(max(abs(a["time"]-b["time"]) for a, b in zip(
                 reference["events"], tight["events"], strict=True))),
             "max_energy_drift": float(np.max(abs(reference["energies"]-reference["energies"][0]))),
             "max_mapping_norm_error": float(np.max(abs(np.sum(
                 reference["states"][:, 2:4]**2+reference["states"][:, 4:6]**2, axis=1)-1)))}
    runs, outputs = [], []
    for dt in case["dts"]:
        result = _production(case, dt, 1, 1e-11)
        indices = np.rint(result.times/fine_dt).astype(int)
        expected = reference["states"][indices]
        q = result.observables["q"][:, 0]
        p = result.observables["p"][:, 0]
        away = np.min(abs(result.times[:, None]-event_times), axis=1) > 2*dt
        diag = result.final_state.method_state
        observed_events = np.asarray(result.observables["events"])
        # Production stores counters, not localized event times. Verify that
        # each reference event lies within its production outer-step bracket.
        bracket_misses = []
        for count, t in enumerate(event_times, 1):
            completed = np.flatnonzero(observed_events >= count)
            if not len(completed):
                bracket_misses.append(float("inf"))
                continue
            index = completed[0]
            left, right = result.times[max(index-1, 0)], result.times[index]
            bracket_misses.append(max(float(left-t), float(t-right), 0.))
        packed = _packed(result.final_state)
        runs.append({"dt": dt, "event_substeps": 1, "event_tolerance": 1e-11,
                     "final_state_max_absolute_error": float(np.max(abs(packed-reference["final_state"]))),
                     "trajectory_q_max_absolute_error": float(np.max(abs(q-expected[:, 0]))),
                     "trajectory_p_max_absolute_error": float(np.max(abs(p-expected[:, 1]))),
                     "trajectory_p_away_from_events_max_absolute_error": float(np.max(abs(p[away]-expected[away, 1]))),
                     "active_sample_mismatches": int(np.sum(np.argmax(result.observables["population"], axis=1)
                                                            != reference["active"][indices])),
                     "max_reference_event_bracket_miss": max(bracket_misses),
                     "max_energy_drift": float(np.max(abs(result.observables["energy"]-result.observables["energy"][0]))),
                     "max_mapping_norm_error": float(np.max(abs(result.observables["mapping_norm"]-1))),
                     "events": int(diag["events"]), "accepted": int(diag["accepted"]),
                     "frustrated": int(diag["frustrated"]), "status": int(diag["status"]),
                     "max_event_residual": float(diag["max_event_residual"]),
                     "max_impulse_energy_error": float(diag["max_impulse_energy_error"]),
                     "final_active": int(diag["active"]), "final_state": packed.tolist()})
        outputs.append(packed)
    errors = [row["final_state_max_absolute_error"] for row in runs]
    report = {"configuration": case, "reference_events": reference["events"], "reference_audit": audit,
              "reference_final_state": reference["final_state"].tolist(),
              "reference_final_active": reference["final_active"], "runs": runs,
              "observed_orders": [float(np.log2(errors[i]/errors[i+1])) for i in range(len(errors)-1)]}
    if localization_checks:
        comparisons = []
        for tolerance in (1e-5, 1e-8):
            result = _production(case, case["dts"][1], 1, tolerance)
            comparisons.append({"dt": case["dts"][1], "event_tolerance": tolerance,
                                "final_state_difference_from_tolerance_1e11": float(np.max(
                                    abs(_packed(result.final_state)-outputs[1]))),
                                "max_event_residual": float(result.final_state.method_state["max_event_residual"]),
                                "events": int(result.final_state.method_state["events"])})
        refined = _production(case, case["dts"][0], 4, 1e-11)
        report["event_tolerance_refinement"] = comparisons
        report["subdivision_check"] = {"outer_dt": case["dts"][0], "event_substeps": 4,
                                       "same_effective_dt": fine_dt,
                                       "final_state_difference_from_fine_dt": float(np.max(abs(
                                           _packed(refined.final_state)-outputs[-1])))}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("benchmarks/results/mash2_reference.json"))
    options = parser.parse_args()
    jax.config.update("jax_enable_x64", True)
    report = {"timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "platform": platform.platform(), "jax": jax.__version__, "scipy": scipy.__version__,
              "scope": "classical MASH numerical reference; no quantum or literature population parity",
              "cases": []}
    for case in cases():
        result = compare_case(case, localization_checks=case["name"] == "rotating_mixed")
        report["cases"].append(result)
        print(json.dumps(result), flush=True)
    options.output.parent.mkdir(parents=True, exist_ok=True)
    options.output.write_text(json.dumps(report, indent=2)+"\n")


if __name__ == "__main__":
    main()
