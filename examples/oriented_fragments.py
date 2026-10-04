"""Illustrative atomistic dimer/trimer through the native local-block contract.

JAX_ENABLE_X64=1 PYTHONPATH=src .venv/bin/python examples/oriented_fragments.py \
    --output outputs/materials_20261004/molecular/example

All parameters are illustrative atomic-unit values, not fitted DNTT data.
Each three-atom fragment defines one effective axial carrier state. The
reference is an explicit spring network, not an electronic-structure teacher.
CPA and Ehrenfest are compared separately with independent NumPy/SciPy ODEs.
"""

import argparse
from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
import platform

import jax
import jax.numpy as jnp
import numpy as np
from scipy.integrate import solve_ivp

import pyeph
from pyeph import CPA, Ehrenfest, Execution, Integrator, Problem, Simulation
from pyeph.core.contracts import ProbeContext
from pyeph.core.problem import CoupledClassical
from pyeph.core.state import make_state
from pyeph.models.base import AutoDiffModel
from pyeph.models.composite import SumModel
from pyeph.models.fragment import OrientedFragmentCoefficients
from pyeph.models.local import AtomCenterMap, LocalBlockGraph, LocalBlockModel
from pyeph.observables.population import FunctionalMeasurement
from pyeph.paths.harmonic import HarmonicBath


@dataclass(frozen=True)
class SpringReference(AutoDiffModel):
    """Illustrative rotation/translation invariant neutral spring network."""

    spec: object
    pairs: tuple

    def apply(self, params, q, vectors):
        return jnp.zeros_like(vectors)

    def reference_energy(self, params, q):
        pairs = jnp.asarray(self.pairs)
        lengths = jnp.linalg.norm(q[pairs[:, 1]]-q[pairs[:, 0]], axis=1)
        return .5*jnp.sum(params["spring"]*(lengths-params["lengths"])**2)

    def probe_apply(self, params, context, probe, vectors):
        if probe not in self.spec.probes:
            return super().probe_apply(params, context, probe, vectors)
        return jnp.zeros_like(vectors)


def measure(problem, state):
    model, params, c = problem.model, problem.params, state.electronic
    current = jnp.stack([jnp.vdot(c, model.probe_apply(params, ProbeContext(state.q), name, c)).real
                         for name in model.spec.probes])
    energy = (jnp.sum(state.p**2/(2*problem.nuclear_treatment.masses))
              +model.reference_energy(params, state.q)+jnp.vdot(c, model.apply(params, state.q, c)).real)
    return dict(q=state.q, p=state.p, electronic=c, population=jnp.abs(c)**2,
                norm=jnp.vdot(c, c).real, energy=energy, current=current)


def fixture(nsites=3, method="ehrenfest"):
    """Small nondegenerate effective model; no molecular chemistry is inferred."""
    if nsites not in (2, 3) or method not in ("cpa", "ehrenfest"):
        raise ValueError("choose 2/3 fragments and cpa/ehrenfest")
    triangle = np.array([[-.4, -.3, 0.], [.7, -.2, .1], [-.2, .8, .05]])
    rotation = np.array([[1., 0., 0.], [0., .8, -.6], [0., .6, .8]])
    origins = np.array([[0., 0., 0.], [2.8, .2, .6], [1.3, 3.1, -.4]])[:nsites]
    equilibrium = np.concatenate([triangle@np.linalg.matrix_power(rotation, i).T+origin
                                  for i, origin in enumerate(origins)])
    anchors = tuple(tuple(range(3*i, 3*i+3)) for i in range(nsites))
    centers = AtomCenterMap(tuple(np.repeat(np.arange(nsites), 3)), (1/3,)*len(equilibrium), nsites)
    graph = LocalBlockGraph(nsites, 1, tuple((i, j) for i in range(nsites) for j in range(i+1, nsites)),
                            switch_on=2.6, cutoff=5.6)
    carrier = LocalBlockModel(graph, centers, OrientedFragmentCoefficients(anchors),
                              charge=1., basis_id="illustrative-fragment-hole-axial-v1")
    p_carrier = dict(onsite=jnp.linspace(-.05, .06, nsites),
                     deformation=jnp.asarray(np.tile([.03, -.02], (nsites, 1))),
                     bond_lengths=jnp.asarray([[np.linalg.norm(equilibrium[a]-equilibrium[o]),
                                                np.linalg.norm(equilibrium[b]-equilibrium[o])]
                                               for o, a, b in anchors]),
                     pp_sigma=jnp.asarray(.09), pp_pi=jnp.asarray(-.025),
                     decay=jnp.asarray(.35), reference_distance=jnp.asarray(3.))
    pairs = tuple(edge for o, a, b in anchors for edge in ((o, a), (o, b), (a, b)))
    pairs += tuple((3*i+k, 3*j+k) for i, j, *_ in graph.edges for k in range(3))
    reference = SpringReference(replace(carrier.spec, name="illustrative_spring_network"), pairs)
    model = SumModel((carrier, reference), additive_probes=carrier.spec.probes)
    p_reference = dict(spring=jnp.full((len(pairs),), .025),
                       lengths=jnp.asarray([np.linalg.norm(equilibrium[j]-equilibrium[i]) for i, j in pairs]))
    q = jnp.asarray(equilibrium+.015*np.sin(np.arange(equilibrium.size).reshape(equilibrium.shape)))
    p = .35*jnp.cos(jnp.arange(q.size).reshape(q.shape))
    masses = jnp.linspace(150., 250., len(q))[:, None]
    c = jnp.arange(1, nsites+1)+.3j*jnp.arange(nsites)
    c /= jnp.linalg.norm(c)
    treatment = CoupledClassical(masses) if method == "ehrenfest" else HarmonicBath(0., masses)
    problem = Problem(model, (p_carrier, p_reference), treatment,
                      Ehrenfest() if method == "ehrenfest" else CPA(),
                      FunctionalMeasurement(measure, required_probes=model.spec.probes))
    return problem, make_state(q, p, c, seed=430, trajectory_id=17)


def numpy_quantities(problem, q):
    """Independent Cartesian-tensor H, neutral energy/gradient, and currents."""
    carrier, reference = problem.model.models
    params, neutral = ({k: np.asarray(v) for k, v in p.items()} for p in problem.params)
    n = carrier.nstates
    center = np.zeros((n, 3))
    for atom, site, weight in zip(q, carrier.centers.atom_site, carrier.centers.weights):
        center[site] += weight*atom
    normal = []
    h = np.zeros((n, n))
    for site, (o, a, b) in enumerate(carrier.coefficient_provider.anchors):
        u, v = q[a]-q[o], q[b]-q[o]
        cross = np.cross(u, v)
        normal.append(cross/np.linalg.norm(cross)*carrier.coefficient_provider.phases[site])
        h[site, site] = params["onsite"][site]+np.dot(params["deformation"][site],
            np.array([np.linalg.norm(u), np.linalg.norm(v)])-params["bond_lengths"][site])
    currents = np.zeros((3, n, n), complex)
    for i, j, *image in carrier.graph.edges:
        displacement = center[j]-center[i]
        if carrier.graph.cell is not None:
            displacement += np.asarray(image)@np.asarray(carrier.graph.cell)
        r = np.linalg.norm(displacement)
        tensor = params["pp_pi"]*np.eye(3)+(params["pp_sigma"]-params["pp_pi"])*np.outer(displacement, displacement)/r**2
        switch = np.clip((r-carrier.graph.switch_on)/(carrier.graph.cutoff-carrier.graph.switch_on), 0., 1.)
        transfer = (normal[i]@tensor@normal[j])*np.exp(-params["decay"]*(r-params["reference_distance"]))
        transfer *= 1-10*switch**3+15*switch**4-6*switch**5
        h[i, j] += transfer
        h[j, i] += transfer
        currents[:, i, j] += 1j*carrier.charge*displacement*transfer
        currents[:, j, i] -= 1j*carrier.charge*displacement*transfer
    energy, gradient = 0., np.zeros_like(q)
    for edge, spring, length in zip(reference.pairs, neutral["spring"], neutral["lengths"]):
        i, j = edge
        d = q[j]-q[i]
        r = np.linalg.norm(d)
        energy += .5*spring*(r-length)**2
        value = spring*(r-length)*d/r
        gradient[j] += value
        gradient[i] -= value
    return h, energy, gradient, currents


def scipy_reference(problem, initial, times, *, rtol=2e-10, atol=2e-12):
    """DOP853 with NumPy H and central-difference electronic force, no JAX AD."""
    shape, nq, n = initial.q.shape, initial.q.size, problem.model.nstates
    mass = np.asarray(problem.nuclear_treatment.masses)
    y0 = np.r_[np.asarray(initial.q).ravel(), np.asarray(initial.p).ravel(),
               np.asarray(initial.electronic).real, np.asarray(initial.electronic).imag]

    def rhs(time, y):
        q, p = y[:nq].reshape(shape), y[nq:2*nq].reshape(shape)
        c = y[2*nq:2*nq+n]+1j*y[2*nq+n:]
        h, _, force_gradient, _ = numpy_quantities(problem, q)
        if isinstance(problem.method, CPA):
            force = np.zeros_like(q)
        else:
            delta = 1e-5
            for index in np.ndindex(shape):
                shift = np.zeros(shape)
                shift[index] = delta
                derivative = (numpy_quantities(problem, q+shift)[0]-numpy_quantities(problem, q-shift)[0])/(2*delta)
                force_gradient[index] += np.vdot(c, derivative@c).real
            force = -force_gradient
        dc = -1j*h@c
        return np.r_[(p/mass).ravel(), force.ravel(), dc.real, dc.imag]

    result = solve_ivp(rhs, (float(times[0]), float(times[-1])), y0, method="DOP853",
                       t_eval=times, rtol=rtol, atol=atol)
    if not result.success:
        raise RuntimeError(result.message)
    q = result.y[:nq].T.reshape(-1, *shape)
    p = result.y[nq:2*nq].T.reshape(-1, *shape)
    c = result.y[2*nq:2*nq+n].T+1j*result.y[2*nq+n:].T
    current = np.array([np.einsum("i,aij,j->a", x.conj(), numpy_quantities(problem, y)[3], x).real
                        for x, y in zip(c, q)])
    return dict(q=q, p=p, electronic=c, current=current), result.nfev


def run_case(output, nsites, method, dt, steps):
    problem, initial = fixture(nsites, method)
    carrier = problem.model.models[0]
    arrays = dict(initial_q=np.asarray(initial.q), initial_p=np.asarray(initial.p),
                  initial_electronic=np.asarray(initial.electronic),
                  anchors=np.asarray(carrier.coefficient_provider.anchors),
                  phases=np.asarray(carrier.coefficient_provider.phases),
                  atom_site=np.asarray(carrier.centers.atom_site),
                  center_weights=np.asarray(carrier.centers.weights),
                  edges=np.asarray(carrier.graph.edges))
    for component, params in zip(("carrier", "reference"), problem.params):
        arrays.update({f"input_{component}_{name}": np.asarray(value) for name, value in params.items()})
    results = []
    label = f"{method}_{nsites}"
    for suffix, width, count in (("coarse", dt, steps), ("fine", dt/2, 2*steps)):
        runner = Simulation(problem, Integrator(width, "rk4", electronic_substeps=2),
                            Execution(chunk_size=count//2, save_every=1 if suffix == "coarse" else 2))
        result = runner.run(initial, count)
        # Recheck every recorded actual geometry, including all frame guards.
        carrier.validate_at(problem.params[0], result.observables["q"], batch=True)
        results.append(result)
        for name, value in result.observables.items():
            arrays[f"{suffix}_{name}"] = value
        if suffix == "coarse":
            first = runner.run(initial, count//2)
            script_digest = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
            ids = {"model.models[1]": f"sha256:{script_digest}:SpringReference",
                   "measurement.function": f"sha256:{script_digest}:measure"}
            checkpoint = output/f"{label}.h5"
            runner.save_checkpoint(checkpoint, first.final_state, artifact_ids=ids)
            restored = runner.load_checkpoint(checkpoint, artifact_ids=ids)
            resumed = runner.run(restored, count-count//2)
            for a, b in zip(jax.tree.leaves(resumed.final_state), jax.tree.leaves(result.final_state), strict=True):
                np.testing.assert_array_equal(a, b)
            for name in result.observables:
                np.testing.assert_array_equal(np.concatenate((first.observables[name], resumed.observables[name][1:])),
                                               result.observables[name])
    coarse, fine = results
    np.testing.assert_allclose(coarse.times, fine.times, atol=2e-14, rtol=0.)
    reference, evaluations = scipy_reference(problem, initial, coarse.times)
    errors = {name: [float(np.max(abs(r.observables[name]-value))) for r in results]
              for name, value in reference.items()}
    for name, (coarse_error, fine_error) in errors.items():
        if fine_error > .4*coarse_error+2e-9:
            raise AssertionError(f"{label} {name}: dt/2 did not reduce the resolved reference error")
    if errors["electronic"][1] > 1e-4 or errors["p"][1] > 1e-3:
        raise AssertionError("short workflow accuracy target unmet; reduce --dt")
    for name, value in reference.items():
        arrays[f"reference_{name}"] = value
    arrays["times"] = coarse.times
    np.savez_compressed(output/f"{label}.npz", **arrays)
    return dict(nsites=nsites, method=method, dt=[dt, dt/2], steps=[steps, 2*steps],
                scipy_nfev=evaluations, scipy_max_abs_errors=errors, restart_bitwise=True,
                norm_drift=[float(np.max(abs(r.observables["norm"]-1))) for r in results],
                energy_change=[float(np.max(abs(r.observables["energy"]-r.observables["energy"][0]))) for r in results],
                energy_conservation_expected=method == "ehrenfest")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--dt", type=float, default=.5)
    parser.add_argument("--sites", type=int, nargs="+", choices=(2, 3), default=(2, 3))
    parser.add_argument("--methods", nargs="+", choices=("cpa", "ehrenfest"), default=("cpa", "ehrenfest"))
    args = parser.parse_args()
    if not jax.config.x64_enabled:
        parser.error("set JAX_ENABLE_X64=1 for this numerical qualification example")
    if args.steps < 2 or args.steps % 2 or not np.isfinite(args.dt) or args.dt <= 0:
        parser.error("steps must be positive/even and dt finite/positive")
    args.output.mkdir(parents=True, exist_ok=False)
    runtime = Path(pyeph.__file__).resolve().parent
    files = list(runtime.rglob("*.py"))+[Path(__file__).resolve()]
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    reports = [run_case(args.output, n, method, args.dt, args.steps) for n in args.sites for method in args.methods]
    if hashes != {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}:
        raise RuntimeError("runtime/example source changed during qualification; rerun a fixed snapshot")
    summary = dict(scope="Illustrative effective axial fragment model, not fitted molecular material accuracy",
                   units="Hartree, bohr, electron mass, atomic time, charge in +e",
                   carrier="one hole; energies already effective hole energies, not negated electron levels",
                   basis="fixed orthonormal fragment labels; ordered anchor normal and fixed phase +1",
                   reference="explicit illustrative neutral spring network",
                   excluded="AO overlap/connection, ionic/convective current, DNTT calibration, converged mobility",
                   angular_source="https://doi.org/10.1103/PhysRev.94.1498",
                   python=platform.python_version(), jax=jax.__version__, numpy=np.__version__,
                   sources=hashes, cases=reports)
    (args.output/"report.json").write_text(json.dumps(summary, indent=2)+"\n")
    print(json.dumps(dict(output=str(args.output), cases=reports), indent=2))


if __name__ == "__main__":
    main()
