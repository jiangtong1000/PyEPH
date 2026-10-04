"""Nonequilibrium RM event qualification of the oriented-fragment provider.

Illustrative effective parameters, not a material-fitted Hamiltonian. The
NumPy/SciPy oracle uses independently assembled H, coordinate finite differences,
and spectral-projector equations; no production force or event helper enters it.
"""

import argparse
from dataclasses import dataclass, replace
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np
from scipy.integrate import solve_ivp

from pyeph import Execution, Integrator, Simulation
from pyeph.core.contracts import ProbeContext
from pyeph.core.state import stack_states
from pyeph.dynamics.mashrm import MASHRM, MASHRMPopulation, mapping_state, sample_population
from pyeph.dynamics.mashrm_mapping import mapping_observable
from pyeph.models.composite import SumModel


ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location("fragment_rm_example", ROOT/"examples/oriented_fragments.py")
example = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = example
_SPEC.loader.exec_module(example)


@dataclass(frozen=True)
class RMMeasurement:
    """Correct one-time RM density estimator plus electronic-vector diagnostics."""

    supports_mashrm = True

    def validate(self, problem):
        MASHRMPopulation().validate(problem)

    def evaluate(self, problem, state):
        current = []
        for probe in problem.model.spec.probes:
            def apply(x):
                return problem.model.probe_apply(problem.params, ProbeContext(state.q), probe, x)
            current.append(mapping_observable(state.electronic, apply(state.electronic),
                                               jnp.trace(apply(jnp.eye(problem.model.nstates)))).real)
        return MASHRMPopulation(include_density=True, include_nuclei=True).evaluate(problem, state) | {
            "electronic": state.electronic, "current": jnp.stack(current)}


def fixture(nsites=3, periodic=False):
    problem, initial = example.fixture(nsites, "ehrenfest")
    if periodic:
        carrier, reference = problem.model.models
        graph = replace(carrier.graph,
                        edges=carrier.graph.edges+((0, 1, -1, 0, 0),),
                        cell=((5.2, .1, .2), (0., 6., .1), (.1, .2, 6.5)))
        carrier = replace(carrier, graph=graph)
        # Only intrafragment neutral springs: periodic image relabeling then
        # preserves the reference without inventing an intermolecular force field.
        reference = replace(reference, pairs=reference.pairs[:3*nsites])
        neutral = {key: value[:3*nsites] for key, value in problem.params[1].items()}
        problem = replace(problem, model=SumModel((carrier, reference), additive_probes=carrier.spec.probes),
                          params=(problem.params[0], neutral))
    return replace(problem, method=MASHRM(event_substeps=2), measurement=RMMeasurement()), initial


class NumpyOracle:
    """Small complete-spectrum reference, with an explicit finite-difference width."""

    def __init__(self, problem, width=1e-5):
        self.problem = problem
        self.shape = problem.model.spec.system.q_shape
        self.nq = int(np.prod(self.shape))
        self.n = problem.model.nstates
        self.mass = np.broadcast_to(np.asarray(problem.nuclear_treatment.masses), self.shape).ravel()
        self.width = width

    def unpack(self, row):
        q, p = row[:self.nq], row[self.nq:2*self.nq]
        c = row[2*self.nq:2*self.nq+self.n]+1j*row[2*self.nq+self.n:]
        return q.reshape(self.shape), p, c

    def pack(self, q, p, c):
        return np.r_[np.asarray(q).ravel(), np.asarray(p).ravel(), np.asarray(c).real, np.asarray(c).imag]

    def quantities(self, q, derivatives=False):
        h, reference, gradient, _ = example.numpy_quantities(self.problem, q)
        energies, vectors = np.linalg.eigh(h)
        if np.min(np.diff(energies)) < 1e-7:
            raise ValueError("oracle requires every surface to remain isolated")
        dh = None
        if derivatives:
            dh = np.empty((self.nq, self.n, self.n))
            for flat in range(self.nq):
                delta = np.zeros(self.shape)
                delta.flat[flat] = self.width
                plus = example.numpy_quantities(self.problem, q+delta)[0]
                minus = example.numpy_quantities(self.problem, q-delta)[0]
                dh[flat] = (plus-minus)/(2*self.width)
        return h, energies, vectors, reference, gradient.ravel(), dh

    def populations(self, row):
        q, _, c = self.unpack(row)
        return abs(self.quantities(q)[2].T@c)**2

    def direction(self, row, active, other):
        q, _, c = self.unpack(row)
        _, energies, vectors, _, _, dh = self.quantities(q, True)
        # Independent complete spectral-projector response; eigenvector signs
        # cancel between the two factors in each derivative projector.
        derivative = np.zeros((self.nq, self.n, self.n), complex)
        for a, sign in ((active, 1.), (other, -1.)):
            ua = vectors[:, a]
            for k in range(self.n):
                if k == a:
                    continue
                uk = vectors[:, k]
                response = np.einsum("i,xij,j->x", uk, dh, ua)/(energies[a]-energies[k])
                derivative += sign*response[:, None, None]*(np.outer(uk, ua)+np.outer(ua, uk))
        return .5*np.einsum("i,xij,j->x", c.conj(), derivative, c).real

    def rhs(self, time, row, active):
        q, p, c = self.unpack(row)
        h, _, vectors, _, gradient, dh = self.quantities(q, True)
        ua = vectors[:, active]
        force = -gradient-np.einsum("i,xij,j->x", ua, dh, ua)
        dc = -1j*h@c  # Scalar reference phase is consistently omitted.
        return np.r_[p/self.mass, force, dc.real, dc.imag]

    def energy(self, row, active):
        q, p, _ = self.unpack(row)
        _, energies, _, reference, _, _ = self.quantities(q)
        return float(.5*np.sum(p*p/self.mass)+reference+energies[active])

    def impulse(self, row, active, other, time):
        q, p, _ = self.unpack(row)
        direction = self.direction(row, active, other)
        normal = direction/np.sqrt(self.mass)
        length = np.linalg.norm(normal)
        if length < 1e-10:
            raise ValueError("unresolved impulse direction")
        normal /= length
        incoming = float((p/np.sqrt(self.mass))@normal)
        energies = self.quantities(q)[1]
        gap = energies[other]-energies[active]
        remaining = incoming*incoming-2*gap
        if incoming >= -1e-10 or abs(remaining) < 1e-9:
            raise ValueError("grazing or threshold event is outside the reference scope")
        accepted = bool(remaining > 0)
        outgoing = -np.sqrt(remaining) if accepted else -incoming
        after = row.copy()
        after[self.nq:2*self.nq] = p+np.sqrt(self.mass)*(outgoing-incoming)*normal
        next_active = other if accepted else active
        pop = self.populations(row)
        if abs(pop[active]-pop[other]) > 1e-9 or np.max(pop)-pop[active] > 1e-9:
            raise ValueError("root is not a largest-population pair boundary")
        if self.n > 2 and min(pop[active]-pop[k] for k in range(self.n) if k not in (active, other)) < 1e-5:
            raise ValueError("ambiguous competing event")
        rate_after = 2*self.direction(after, next_active, other if next_active == active else active) @ (
            after[self.nq:2*self.nq]/self.mass)
        if rate_after <= 0:
            raise ValueError("impulse does not leave an outgoing boundary")
        return after, next_active, dict(time=float(time), active_before=active, active_after=next_active,
            accepted=accepted, gap=float(gap), incoming_projection=incoming, outgoing_projection=float(outgoing),
            energy_error=self.energy(after, next_active)-self.energy(row, active),
            population_residual=float(abs(pop[active]-pop[other])), before=row.tolist(), after=after.tolist())

    def solve(self, initial, times, *, max_step=.05, rtol=1e-10, atol=2e-12):
        row = self.pack(initial.q, initial.p, initial.electronic)
        active = int(initial.method_state["active"])
        start, stop = float(times[0]), float(times[-1])
        events, segments = [], []
        outgoing_other = None
        while start < stop:
            boundaries = []
            competitors = [i for i in range(self.n) if i != active]
            for other in competitors:
                def boundary(time, value, other=other):
                    if time == start and other == outgoing_other:
                        return 1e-12
                    pop = self.populations(value)
                    return pop[active]-pop[other]
                boundary.direction, boundary.terminal = -1, True
                boundaries.append(boundary)
            solution = solve_ivp(lambda t, x: self.rhs(t, x, active), (start, stop), row,
                                 method="DOP853", rtol=rtol, atol=atol, max_step=max_step,
                                 events=boundaries, dense_output=True)
            if not solution.success:
                raise RuntimeError(solution.message)
            end = float(solution.t[-1])
            segments.append((start, end, active, solution.sol))
            row = solution.y[:, -1]
            hits = [i for i, hits in enumerate(solution.t_events) if len(hits)]
            if not hits:
                break
            if len(hits) != 1 or len(events) >= 6 or end <= start+1e-9:
                raise RuntimeError("unresolved competing/repeated event or capacity exhaustion")
            other = competitors[hits[0]]
            before_active = active
            row, active, event = self.impulse(row, active, other, end)
            events.append(event)
            outgoing_other = other if active == before_active else before_active
            start = end
        samples, surfaces = [], []
        for time in times:
            if time == stop:
                samples.append(row)
                surfaces.append(active)
            else:
                for begin, end, surface, interpolant in reversed(segments):
                    if begin <= time <= end:
                        samples.append(interpolant(time))
                        surfaces.append(surface)
                        break
                else:
                    raise RuntimeError("missing reference output segment")
        return np.asarray(samples), np.asarray(surfaces), events


def deterministic_event(nsites=3, periodic=False, accepted=True):
    problem, base = fixture(nsites, periodic)
    oracle = NumpyOracle(problem)
    q = np.asarray(base.q)
    _, energies, vectors, *_ = oracle.quantities(q)
    populations = np.array([.51, .49]) if nsites == 2 else np.array([.46, .44, .1])
    phases = np.array([0., np.pi]) if nsites == 2 else np.array([0., np.pi, .7])
    c = vectors@(np.sqrt(populations)*np.exp(1j*phases))
    row = oracle.pack(q, np.zeros_like(q), c)
    direction = oracle.direction(row, 0, 1)/np.sqrt(oracle.mass)
    length = np.linalg.norm(direction)
    direction /= length
    speed = (1.6 if accepted else .4)*np.sqrt(2*(energies[1]-energies[0]))
    p = (-np.sqrt(oracle.mass)*speed*direction).reshape(q.shape)
    initial = mapping_state(problem.model, problem.params, q, p, c,
                             active=0, trajectory_id=21 if accepted else 34, seed=430)
    estimate = .02/(2*length*speed)
    return problem, initial, oracle, estimate


def pilot(output):
    output.mkdir(parents=True, exist_ok=False)
    result = []
    for nsites, periodic in ((2, False), (3, False), (3, True)):
        for accepted in (True, False):
            problem, initial, oracle, estimate = deterministic_event(nsites, periodic, accepted)
            stop = 2.5*estimate
            times = np.linspace(0., stop, 17)
            samples, active, events = oracle.solve(initial, times, max_step=stop/32)
            label = f"n{nsites}_{'periodic' if periodic else 'finite'}_{'accepted' if accepted else 'frustrated'}"
            np.savez_compressed(output/f"{label}.npz", times=times, state=samples, active=active)
            result.append(dict(label=label, estimate=estimate, stop=stop, events=events))
            print(label, "estimate", estimate, "events", [(e["time"], e["accepted"]) for e in events], flush=True)
    (output/"pilot.json").write_text(json.dumps(result, indent=2)+"\n")


def estimator_audit(problem, result):
    """Independent one-time RM estimators and sampled spectral-domain checks."""
    values = result.observables
    c = np.asarray(values["electronic"])
    n = problem.model.nstates
    alpha = (n-1)/sum(1/k for k in range(2, n+1))
    offset = (1-alpha)/n
    density = alpha*np.einsum("...i,...j->...ij", c, c.conj())+offset*np.eye(n)
    np.testing.assert_allclose(values["density"], density, atol=5e-15, rtol=0.)
    np.testing.assert_allclose(values["population"], density.diagonal(axis1=-2, axis2=-1).real,
                               atol=5e-15, rtol=0.)
    np.testing.assert_allclose(np.trace(density, axis1=-2, axis2=-1), 1., atol=3e-12, rtol=0.)
    oracle = NumpyOracle(problem)
    gaps, margins, current_error = [], [], []
    for index in np.ndindex(c.shape[:-1]):
        q = np.asarray(values["q"])[index]
        h, _, _, currents = example.numpy_quantities(problem, q)
        energies, vectors = np.linalg.eigh(h)
        gaps.append(np.min(np.diff(energies)))
        pops = abs(vectors.T@c[index])**2
        margins.append(pops[int(values["active"][index])]-np.max(pops))
        current = np.einsum("ij,aji->a", density[index], currents).real
        current_error.append(np.max(abs(current-values["current"][index])))
        # The neutral reference is included in active-surface total energy.
        row = oracle.pack(q, np.asarray(values["p"])[index], c[index])
        np.testing.assert_allclose(values["energy"][index],
                                   oracle.energy(row, int(values["active"][index])), atol=2e-15)
    carrier = problem.model.models[0]
    carrier.validate_at(problem.params[0], np.asarray(values["q"]).reshape(-1, *oracle.shape), batch=True)
    if min(gaps) < 1e-7 or min(margins) < -2e-10:
        raise AssertionError("sampled spectrum or active population ownership failed")
    if max(current_error) > 2e-14:
        raise AssertionError("one-time RM current differs from the independent density trace")
    return dict(minimum_sampled_gap=float(min(gaps)), minimum_active_margin=float(min(margins)),
                maximum_current_estimator_error=float(max(current_error)))


def run_native(problem, states, dt, steps, *, tolerance=1e-10, substeps=2):
    problem = replace(problem, method=replace(problem.method, event_tolerance=tolerance,
                                               event_time_tolerance=tolerance, event_substeps=substeps))
    runner = Simulation(problem, Integrator(dt, "exponential_midpoint"),
                        Execution(chunk_size=steps, save_every=max(1, round(.1/dt))))
    return runner, runner.run(states, steps)


def event_intervals(result):
    times = np.asarray(result.times)
    counts = np.asarray(result.observables["events"])
    changes = np.flatnonzero(np.diff(counts))
    if any(np.diff(counts)[changes] != 1):
        raise AssertionError("output interval contains multiple events; tighten observation spacing")
    return [[float(times[i]), float(times[i+1])] for i in changes]


def run_family(output, nsites=3, periodic=False, *, refinements=(.1, .05, .025),
               refine_oracle=True, extra_checks=True):
    label = f"n{nsites}_{'periodic' if periodic else 'finite'}"
    problem, accepted, oracle, _ = deterministic_event(nsites, periodic, True)
    _, frustrated, _, slow_time = deterministic_event(nsites, periodic, False)
    states = stack_states((accepted, frustrated))
    # A common final time supports accepted/frustrated trajectories in one batch.
    stop = np.ceil(1.6*slow_time/.2)*.2
    count = round(stop/.1)
    times = np.arange(count+1)*.1
    arrays = {"initial_q": np.asarray(states.q), "initial_p": np.asarray(states.p),
              "initial_c": np.asarray(states.electronic), "times": times}
    references, reference_active, event_reports, reference_audits = [], [], [], []
    for name, initial, should_accept in (("accepted", accepted, True), ("frustrated", frustrated, False)):
        reference, active, events = oracle.solve(initial, times, max_step=.04)
        if len(events) != 1 or events[0]["accepted"] != should_accept:
            raise AssertionError("controlled event topology changed")
        audit = dict(max_state_refinement=None, event_time_refinement=None)
        if refine_oracle:
            refined = NumpyOracle(problem, width=5e-6)
            second, second_active, second_events = refined.solve(initial, times, max_step=.02,
                                                                   rtol=2e-11, atol=4e-13)
            np.testing.assert_array_equal(active, second_active)
            audit = dict(max_state_refinement=float(np.max(abs(reference-second))),
                         event_time_refinement=abs(events[0]["time"]-second_events[0]["time"]))
            if max(audit.values()) > 3e-8:
                raise AssertionError("independent finite-difference/ODE reference did not stabilize")
            reference, active, events = second, second_active, second_events
        references.append(reference)
        reference_active.append(active)
        event_reports.append(events)
        reference_audits.append(audit)
        arrays[f"reference_{name}"] = reference
        arrays[f"reference_{name}_active"] = active
    runs, results, runners = [], [], []
    for dt in refinements:
        steps = round(stop/dt)
        runner, result = run_native(problem, states, dt, steps)
        np.testing.assert_allclose(result.times[:, 0], times, atol=3e-14, rtol=0.)
        runs.append(dict(dt=dt, event_substeps=2, tolerance=1e-10, audits=estimator_audit(problem, result), cases=[]))
        for index, name in enumerate(("accepted", "frustrated")):
            # Scalar views permit use of the same observable-interval audit.
            scalar = replace(result, times=np.asarray(result.times)[:, index],
                             observables={key: value[:, index] for key, value in result.observables.items()})
            final = jax.tree.map(lambda x: x[index], result.final_state)
            if int(final.method_state["status"]) or int(final.method_state["events"]) != 1:
                raise AssertionError("native event qualification failed")
            assert int(final.method_state["accepted"]) == (index == 0)
            assert int(final.method_state["frustrated"]) == (index == 1)
            observed = np.stack([oracle.pack(q, p, c) for q, p, c in zip(
                scalar.observables["q"], scalar.observables["p"], scalar.observables["electronic"])])
            np.testing.assert_array_equal(scalar.observables["active"], reference_active[index])
            intervals = event_intervals(scalar)
            event_time = event_reports[index][0]["time"]
            if not intervals[0][0]-1e-12 <= event_time <= intervals[0][1]+1e-12:
                raise AssertionError("independent root is outside the observed native event interval")
            delta = observed-references[index]
            case = dict(name=name, events=1, accepted=int(final.method_state["accepted"]),
                        frustrated=int(final.method_state["frustrated"]), native_event_intervals=intervals,
                        q_error=float(np.max(abs(delta[:, :oracle.nq]))),
                        p_error=float(np.max(abs(delta[:, oracle.nq:2*oracle.nq]))),
                        c_error=float(np.max(abs(delta[:, 2*oracle.nq:]))),
                        energy_drift=float(np.max(abs(scalar.observables["energy"]-scalar.observables["energy"][0]))),
                        norm_error=float(np.max(abs(scalar.observables["mapping_norm"]-1))),
                        max_event_residual=float(final.method_state["max_event_residual"]),
                        max_event_bracket_width=float(final.method_state["max_event_bracket_width"]),
                        max_impulse_energy_error=float(final.method_state["max_impulse_energy_error"]))
            if max(case["max_event_residual"], case["max_event_bracket_width"]) > 1.001e-10:
                raise AssertionError("native localized event exceeded its declared tolerance")
            runs[-1]["cases"].append(case)
        for key, value in result.observables.items():
            arrays[f"dt_{dt}_{key}"] = value
        results.append(result)
        runners.append(runner)
    checks = {}
    if extra_checks:
        # Separate event-tolerance sensitivity from propagation-step refinement.
        _, loose = run_native(problem, states, refinements[0], round(stop/refinements[0]), tolerance=1e-6)
        checks["loose_event_tolerance"] = 1e-6
        checks["loose_vs_tight_final_qp_max"] = max(float(np.max(abs(getattr(loose.final_state, key)
            -getattr(results[0].final_state, key)))) for key in ("q", "p"))
        np.testing.assert_array_equal(loose.observables["events"], results[0].observables["events"])
        arrays.update({f"loose_{key}": value for key, value in loose.observables.items()})
        # Restart at an early interior point, before either controlled event.
        runner = runners[0]
        first = runner.run(states, 1)
        digest = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        example_digest = hashlib.sha256(Path(example.__file__).read_bytes()).hexdigest()
        ids = {"model.models[1]": f"sha256:{example_digest}:SpringReference",
               "measurement": f"sha256:{digest}:RMMeasurement"}
        checkpoint = output/f"{label}.h5"
        runner.save_checkpoint(checkpoint, first.final_state, artifact_ids=ids)
        restored = runner.load_checkpoint(checkpoint, artifact_ids=ids)
        resumed = runner.run(restored, round(stop/refinements[0])-1)
        for a, b in zip(jax.tree.leaves(resumed.final_state), jax.tree.leaves(results[0].final_state), strict=True):
            np.testing.assert_array_equal(a, b)
        for key in results[0].observables:
            np.testing.assert_array_equal(np.concatenate((first.observables[key], resumed.observables[key][1:])),
                                           results[0].observables[key])
        checks["restart_bitwise"] = True
        partition_error = 0.
        for index, initial in enumerate((accepted, frustrated)):
            single = runner.run(initial, round(stop/refinements[0]))
            for name, value in single.observables.items():
                expected = results[0].observables[name][:, index]
                np.testing.assert_allclose(value, expected, atol=3e-12, rtol=0.)
                partition_error = max(partition_error, float(np.max(abs(value-expected))))
        checks["single_vs_batch_max_abs"] = partition_error
    if len(runs) >= 3:
        for index in (0, 1):
            for name in ("q_error", "p_error", "c_error", "energy_drift"):
                values = [run["cases"][index][name] for run in runs]
                # The reference has a finite finite-difference/ODE floor.
                if any(b > .45*a+5e-9 for a, b in zip(values, values[1:])):
                    raise AssertionError(f"unresolved propagation convergence: {label} {index} {name} {values}")
    np.savez_compressed(output/f"{label}.npz", **arrays)
    return dict(label=label, nsites=nsites, periodic=periodic, duration=stop,
                preparation="deterministic near-boundary mapping vectors; not a sampled ensemble",
                reference_events=event_reports, reference_refinement=reference_audits, runs=runs, checks=checks)


def sampled_ensemble(output):
    problem, base = fixture(3, True)
    ids = (7, 31, 105)
    def draw(index):
        return sample_population(problem.model, problem.params, base.q, base.p,
                                 population=0, basis="fixed", seed=430, trajectory_id=index)
    states = [draw(i) for i in ids]
    for index in (105, 7, 31):
        repeated = draw(index)
        for a, b in zip(jax.tree.leaves(repeated), jax.tree.leaves(states[ids.index(index)]), strict=True):
            np.testing.assert_array_equal(a, b)
    runner, result = run_native(problem, stack_states(states), .05, 4)
    audit = estimator_audit(problem, result)
    error = 0.
    for subset, indices in ((states[:2], (0, 1)), (states[2:], (2,))):
        partition = runner.run(stack_states(subset), 4)
        for name, value in partition.observables.items():
            expected = result.observables[name][:, indices]
            np.testing.assert_allclose(value, expected, atol=3e-12, rtol=0.)
            error = max(error, float(np.max(abs(value-expected))))
    np.savez_compressed(output/"conditional_ensemble.npz", initial_c=np.asarray(stack_states(states).electronic),
                        initial_keys=np.asarray(stack_states(states).key), trajectory_ids=ids,
                        times=result.times, **result.observables)
    return dict(preparation="RM conditional-sphere population0 in fixed basis; nuclear q,p supplied independently",
                seed=430, trajectory_ids=ids, basis="fixed", audits=audit,
                partition_max_abs=error, sampling_reorder_bitwise=True,
                interpretation="3 deterministic seed draws test preparation/execution; not a converged statistical estimate")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--family", choices=("all", "dimer", "trimer", "periodic"), default="all")
    args = parser.parse_args()
    if not jax.config.x64_enabled:
        parser.error("set JAX_ENABLE_X64=1")
    if args.pilot:
        pilot(args.output)
        return
    args.output.mkdir(parents=True, exist_ok=False)
    files = sorted((Path(example.pyeph.__file__).resolve().parent).rglob("*.py"))+[
        Path(__file__).resolve(), Path(example.__file__).resolve()]
    source = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    families = {"dimer": (2, False), "trimer": (3, False), "periodic": (3, True)}
    selected = families if args.family == "all" else {args.family: families[args.family]}
    reports = [run_family(args.output, n, periodic) for n, periodic in selected.values()]
    ensemble = sampled_ensemble(args.output)
    assert source == {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    report = dict(scope="Nonequilibrium real isolated-spectrum RM numerical provider qualification",
                   excluded="material calibration, conditional canonical RM transport, quantum exactness, complex/SOC MASH",
                   source_hashes=source, families=reports, sampled_ensemble=ensemble)
    (args.output/"report.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({"output": str(args.output), "families": [row["label"] for row in reports]}, indent=2))


if __name__ == "__main__":
    main()
