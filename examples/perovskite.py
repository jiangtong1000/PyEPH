"""Explicit Pb/I orbital blocks for a small disordered CsPbI3 model.

JAX_ENABLE_X64=1 PYTHONPATH=src .venv/bin/python examples/perovskite.py \
    --output outputs/materials_20261004/perovskite/example

Equilibrium electronic parameters: Nestoklon, Comput. Mater. Sci. 196,
110535 (2021), doi:10.1016/j.commatsci.2021.110535, Table I sp3 column;
arXiv:2012.14705v2. Only numerical parameter facts are transcribed, not the
article. The exponential radial extension and harmonic neutral reference are
illustrative choices, not published force fits. No experimental mobility,
charged-state force accuracy, or complex/SOC MASH support is claimed.
"""

import argparse
from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from scipy.constants import physical_constants
from scipy.integrate import solve_ivp

import pyeph
from pyeph import CPA, Ehrenfest, Execution, Integrator, Problem, Simulation
from pyeph.core._configuration import integer_scalar
from pyeph.core.contracts import ProbeContext
from pyeph.core.problem import CoupledClassical
from pyeph.core.state import make_state
from pyeph.core.units import ATOMIC_TIME_FS, BOHR_ANGSTROM, HARTREE_EV
from pyeph.models.base import AutoDiffModel
from pyeph.models.composite import SumModel
from pyeph.models.local import AtomCenterMap, LocalBlockGraph, LocalBlockModel
from pyeph.models.slater_koster import SlaterKosterSPCoefficients
from pyeph.observables.population import FunctionalMeasurement
from pyeph.paths.normal_modes import NormalModeBath


# a=anion I, c=cation Pb. Energies and hopping are eV, distances Angstrom.
SOURCE_PARAMETERS = dict(lattice=6.289, E_sa=-13.387, E_sc=-5.885,
                         E_pa=-2.310, E_pc=1.774, ss=.003,
                         s_a_p_c=.627, s_c_p_a=1.048, pp_sigma=-1.376,
                         pp_pi=.635, soc_a=.300, soc_c=.433)


@dataclass(frozen=True)
class HarmonicReference(AutoDiffModel):
    """Explicit independent tethers to the supplied equilibrium crystal.

    This is a controlled disorder reference, not a fitted CsPbI3 potential.
    It has no acoustic zero modes and supplies forces on all atoms, including
    Cs, whose electronic orbitals are omitted by the source carrier model.
    """

    spec: object

    def apply(self, params, q, vectors):
        return jnp.zeros_like(vectors)

    def reference_energy(self, params, q):
        return .5*jnp.sum(params["spring"]*(q-params["equilibrium"])**2)

    def probe_apply(self, params, context, probe, vectors):
        if probe not in self.spec.probes:
            return super().probe_apply(params, context, probe, vectors)
        return jnp.zeros_like(vectors)


def build_cspbi3(mesh=(1, 1, 1), *, spinful=True, decay_per_angstrom=.8):
    """Construct source sp3 blocks; all returned numerical values are atomic units.

    The graph contains exactly the six source-model nearest Pb--I bonds per
    Pb, retaining distinct cell images in small cells. Cs coordinates are
    retained with zero electronic-centre weight. Atomic coordinates are
    coherently unwrapped. This fixed connectivity is not a bond-reaction model.
    """
    mesh = tuple(integer_scalar(v, "mesh extent") for v in mesh)
    if len(mesh) != 3 or min(mesh) < 1:
        raise ValueError("mesh must contain three positive integers")
    decay_per_angstrom = float(decay_per_angstrom)
    if not np.isfinite(decay_per_angstrom) or decay_per_angstrom < 0:
        raise ValueError("decay_per_angstrom must be finite and nonnegative")
    provider = SlaterKosterSPCoefficients(spinful=spinful)
    a = SOURCE_PARAMETERS["lattice"]/BOHR_ANGSTROM
    cell = np.diag(np.asarray(mesh)*a)
    basis = np.array([[0., 0., 0.], [.5, 0., 0.], [0., .5, 0.],
                      [0., 0., .5], [.5, .5, .5]])
    cells = list(np.ndindex(mesh))
    q = np.concatenate([(np.asarray(c)+basis)*a for c in cells])
    mapping = tuple(s for i in range(len(cells)) for s in (4*i, 4*i+1, 4*i+2, 4*i+3, 4*i))
    centers = AtomCenterMap(mapping, (1., 1., 1., 1., 0.)*len(cells), 4*len(cells))
    edges = []
    for index, cell_index in enumerate(cells):
        for axis in range(3):
            for offset in (0, -1):
                neighbor = np.asarray(cell_index).copy()
                neighbor[axis] += offset
                image = np.floor_divide(neighbor, mesh)
                wrapped = np.mod(neighbor, mesh)
                other = int(np.ravel_multi_index(tuple(wrapped), mesh))
                i, j = 4*index, 4*other+axis+1
                if i > j:
                    i, j, image = j, i, -image
                edges.append((i, j, *map(int, image)))
    graph = LocalBlockGraph(4*len(cells), 8 if provider.spinful else 4, tuple(edges),
                            cell=cell, switch_on=.625*a, cutoff=.75*a)
    model = LocalBlockModel(graph, centers, provider, charge=-1., complex_valued=provider.spinful,
                            basis_id="CsPbI3-Nestoklon-sp3-global-Cartesian-spin-major-v1")
    src = SOURCE_PARAMETERS
    onsite = np.array([[src["E_sc"], *([src["E_pc"]]*3)] if i % 4 == 0 else
                        [src["E_sa"], *([src["E_pa"]]*3)] for i in range(graph.nsites)])
    hops = []
    for i, _, *_ in graph.edges:
        sp, ps = ((src["s_c_p_a"], src["s_a_p_c"]) if i % 4 == 0 else
                  (src["s_a_p_c"], src["s_c_p_a"]))
        hops.append([src["ss"], sp, ps, src["pp_sigma"], src["pp_pi"]])
    params = dict(onsite=jnp.asarray(onsite/HARTREE_EV), hopping=jnp.asarray(hops)/HARTREE_EV,
                  decay=jnp.full((len(hops), 5), decay_per_angstrom*BOHR_ANGSTROM),
                  reference_distance=jnp.full((len(hops),), a/2))
    if provider.spinful:
        params["soc"] = jnp.asarray([src["soc_c"] if i % 4 == 0 else src["soc_a"]
                                      for i in range(graph.nsites)])/HARTREE_EV
    amu = physical_constants["atomic mass constant"][0]/physical_constants["electron mass"][0]
    masses = jnp.asarray(np.tile([207.2, 126.90447, 126.90447, 126.90447, 132.90545196],
                                 len(cells)))[:, None]*amu
    model.validate_at(params, q)
    return model, params, jnp.asarray(q), masses


def measure(problem, state):
    model, params, c = problem.model, problem.params, state.electronic
    currents = jnp.stack([jnp.vdot(c, model.probe_apply(params, ProbeContext(state.q), p, c)).real
                          for p in model.spec.probes])
    energy = (jnp.sum(state.p**2/(2*problem.nuclear_treatment.masses))
              +model.reference_energy(params, state.q)+jnp.vdot(c, model.apply(params, state.q, c)).real)
    return dict(q=state.q, p=state.p, electronic=c, norm=jnp.vdot(c, c).real,
                current=currents, energy=energy)


def fixture(mesh=(1, 1, 1), *, method="ehrenfest", seed=441):
    if method not in ("cpa", "ehrenfest"):
        raise ValueError("SOC example supports cpa or ehrenfest")
    carrier, params, equilibrium, masses = build_cspbi3(mesh)
    reference = HarmonicReference(replace(carrier.spec, name="illustrative_crystal_tethers"))
    model = SumModel((carrier, reference), additive_probes=carrier.spec.probes)
    # 0.15 eV/Angstrom^2 is a declared numerical reference, not a material fit.
    spring = jnp.full(equilibrium.shape, .15*BOHR_ANGSTROM**2/HARTREE_EV)
    neutral = dict(equilibrium=equilibrium, spring=spring)
    rng = np.random.default_rng(seed)
    q = equilibrium+jnp.asarray(rng.normal(size=equilibrium.shape)*.035/BOHR_ANGSTROM)
    p = jnp.asarray(rng.normal(size=q.shape)*3.)
    _, vectors = np.linalg.eigh(np.asarray(carrier.dense(params, q)))
    ncell = carrier.graph.nsites//4
    # Source sp3 filling is 26 occupied spin orbitals per formula unit.
    c = jnp.asarray((vectors[:, 26*ncell]+1j*vectors[:, 26*ncell+1])/np.sqrt(2.))
    if method == "cpa":
        frequencies2 = np.asarray(spring/masses).ravel()
        treatment = NormalModeBath(frequencies2, np.eye(q.size), masses, equilibrium)
    else:
        treatment = CoupledClassical(masses)
    problem = Problem(model, (params, neutral), treatment, CPA() if method == "cpa" else Ehrenfest(),
                      FunctionalMeasurement(measure, required_probes=model.spec.probes))
    return problem, make_state(q, p, c, seed=seed, trajectory_id=19)


def numpy_quantities(carrier, params, q, c=None):
    """Independent Cartesian tensor assembly and analytic bond derivatives.

    The force contracts each independently differentiated two-centre tensor;
    it neither differentiates JAX nor calls the production coefficient helper.
    """
    p = {k: np.asarray(v) for k, v in params.items()}
    q = np.asarray(q)
    n, b = carrier.graph.nsites, carrier.graph.norbitals
    centers = np.zeros((n, 3))
    for atom, site, weight in zip(q, carrier.centers.atom_site, carrier.centers.weights):
        centers[site] += weight*atom
    h = np.zeros((n*b, n*b), complex)
    current = np.zeros((3, n*b, n*b), complex)
    gradient = np.zeros_like(q)
    # Explicit spin-orbit matrix entries independent of the runtime L generators.
    soc = np.zeros((8, 8), complex)
    for i, j, value in [(1, 2, -1j), (5, 6, 1j), (1, 7, 1.),
                         (2, 7, -1j), (3, 5, -1.), (3, 6, 1j)]:
        soc[i, j], soc[j, i] = value, np.conj(value)
    for i in range(n):
        block = np.diag(p["onsite"][i])
        if b == 8:
            block = np.kron(np.eye(2), block)+p["soc"][i]*soc
        h[i*b:(i+1)*b, i*b:(i+1)*b] += block
    for e, (i, j, *image) in enumerate(carrier.graph.edges):
        d = centers[j]-centers[i]+np.asarray(image)@np.asarray(carrier.graph.cell)
        r = np.linalg.norm(d)
        u = d/r
        x = np.clip((r-carrier.graph.switch_on)/(carrier.graph.cutoff-carrier.graph.switch_on), 0., 1.)
        support = 1-10*x**3+15*x**4-6*x**5
        dsupport = (-30*x**2+60*x**3-30*x**4)/(carrier.graph.cutoff-carrier.graph.switch_on)
        raw = p["hopping"][e]*np.exp(-p["decay"][e]*(r-p["reference_distance"][e]))
        radial, dradial = raw*support, raw*(dsupport-p["decay"][e]*support)
        ss, sp, ps, sigma, pi = radial
        t = np.zeros((4, 4))
        t[0, 0], t[0, 1:], t[1:, 0] = ss, sp*u, -ps*u
        t[1:, 1:] = pi*np.eye(3)+(sigma-pi)*np.outer(u, u)
        dt = np.zeros((3, 4, 4))
        for axis in range(3):
            du = (np.eye(3)[axis]-u*u[axis])/r
            ss_d, sp_d, ps_d, sigma_d, pi_d = dradial*u[axis]
            dt[axis, 0, 0] = ss_d
            dt[axis, 0, 1:] = sp_d*u+sp*du
            dt[axis, 1:, 0] = -ps_d*u-ps*du
            dt[axis, 1:, 1:] = (pi_d*np.eye(3)+(sigma_d-pi_d)*np.outer(u, u)
                                +(sigma-pi)*(np.outer(du, u)+np.outer(u, du)))
        if b == 8:
            t = np.kron(np.eye(2), t)
            dt = np.stack([np.kron(np.eye(2), v) for v in dt])
        si, sj = slice(i*b, (i+1)*b), slice(j*b, (j+1)*b)
        h[si, sj] += t
        h[sj, si] += t.T
        for axis in range(3):
            block = 1j*carrier.charge*d[axis]*t
            current[axis, si, sj] += block
            current[axis, sj, si] += block.conj().T
        if c is not None:
            bond = 2*np.einsum("i,aij,j->a", np.conj(c[si]), dt, c[sj]).real
            for atom, site, weight in zip(range(len(q)), carrier.centers.atom_site, carrier.centers.weights):
                gradient[atom] += weight*((site == j)-(site == i))*bond
    return h, gradient, current


def scipy_reference(problem, initial, times):
    """DOP853 with independent NumPy electronic action and analytic forces."""
    carrier = problem.model.models[0]
    p_carrier, neutral = problem.params
    q_shape, size, n = initial.q.shape, initial.q.size, carrier.nstates
    masses = np.asarray(problem.nuclear_treatment.masses)
    qeq, spring = np.asarray(neutral["equilibrium"]), np.asarray(neutral["spring"])
    y0 = np.r_[np.asarray(initial.q).ravel(), np.asarray(initial.p).ravel(),
               np.asarray(initial.electronic).real, np.asarray(initial.electronic).imag]

    def rhs(time, y):
        q, p = y[:size].reshape(q_shape), y[size:2*size].reshape(q_shape)
        c = y[2*size:2*size+n]+1j*y[2*size+n:]
        h, gradient, _ = numpy_quantities(carrier, p_carrier, q,
                                          None if isinstance(problem.method, CPA) else c)
        force = -spring*(q-qeq)-gradient
        dc = -1j*h@c
        return np.r_[(p/masses).ravel(), force.ravel(), dc.real, dc.imag]

    result = solve_ivp(rhs, (float(times[0]), float(times[-1])), y0, t_eval=np.asarray(times),
                       method="DOP853", rtol=2e-11, atol=2e-13)
    if not result.success:
        raise RuntimeError(result.message)
    qs = result.y[:size].T.reshape(-1, *q_shape)
    ps = result.y[size:2*size].T.reshape(-1, *q_shape)
    cs = result.y[2*size:2*size+n].T+1j*result.y[2*size+n:].T
    currents = np.array([np.einsum("i,aij,j->a", c.conj(), numpy_quantities(carrier, p_carrier, q)[2], c).real
                         for q, c in zip(qs, cs)])
    return dict(q=qs, p=ps, electronic=cs, current=currents), result.nfev


def run_case(output, method, mesh, dt, steps):
    problem, initial = fixture(mesh, method=method)
    results = []
    arrays = {}
    for label, width, count, stride in (("coarse", dt, steps, 1), ("fine", dt/2, 2*steps, 2)):
        runner = Simulation(problem, Integrator(width, "rk4", electronic_substeps=4),
                            Execution(chunk_size=count//2, save_every=stride))
        result = runner.run(initial, count)
        results.append(result)
        arrays.update({f"{label}_{k}": np.asarray(v) for k, v in result.observables.items()})
        if label == "coarse":
            prefix = runner.run(initial, count//2)
            digest = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
            ids = {"model.models[1]": f"sha256:{digest}:HarmonicReference",
                   "measurement.function": f"sha256:{digest}:measure"}
            path = output/f"{method}_checkpoint.h5"
            runner.save_checkpoint(path, prefix.final_state, artifact_ids=ids)
            restored = runner.load_checkpoint(path, artifact_ids=ids)
            resumed = runner.run(restored, count-count//2)
            for a, b in zip(jax.tree.leaves(resumed.final_state), jax.tree.leaves(result.final_state), strict=True):
                np.testing.assert_array_equal(a, b)
    reference, nfev = scipy_reference(problem, initial, results[0].times)
    errors = {k: [float(np.max(abs(r.observables[k]-v))) for r in results] for k, v in reference.items()}
    for name, (coarse, fine) in errors.items():
        if fine > .4*coarse+2e-9:
            raise AssertionError(f"{method}/{name} refinement did not reduce resolved error")
    if errors["electronic"][1] > 1e-5 or errors["p"][1] > 1e-4:
        raise AssertionError("short workflow accuracy target unmet; reduce dt")
    arrays.update({f"reference_{k}": v for k, v in reference.items()})
    arrays["times"] = results[0].times
    arrays["initial_q"], arrays["initial_p"], arrays["initial_electronic"] = map(
        np.asarray, (initial.q, initial.p, initial.electronic))
    for prefix, params in zip(("carrier", "neutral"), problem.params):
        arrays.update({f"{prefix}_{k}": np.asarray(v) for k, v in params.items()})
    arrays["cell"] = problem.model.models[0].graph.cell
    arrays["edges"] = problem.model.models[0].graph.edges
    np.savez_compressed(output/f"{method}.npz", **arrays)
    return dict(method=method, mesh=list(mesh), natoms=initial.q.shape[0], nstates=problem.model.nstates,
                dt_atomic=[dt, dt/2], duration_fs=dt*steps*ATOMIC_TIME_FS,
                scipy_nfev=nfev, max_abs_errors=errors, restart_bitwise=True,
                norm_drift=[float(np.max(abs(r.observables["norm"]-1))) for r in results],
                energy_change_hartree=[float(np.max(abs(r.observables["energy"]-r.observables["energy"][0])))
                                      for r in results],
                energy_conservation_expected=method == "ehrenfest")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mesh", nargs=3, type=int, default=(1, 1, 1))
    parser.add_argument("--dt", type=float, default=2.)
    parser.add_argument("--steps", type=int, default=16)
    args = parser.parse_args()
    if args.steps < 4 or args.steps % 2 or not np.isfinite(args.dt) or args.dt <= 0:
        raise ValueError("steps must be even and >=4; dt must be finite and positive")
    pyeph.configure_precision(True)
    args.output.mkdir(parents=True, exist_ok=False)
    source_root = Path(pyeph.__file__).resolve().parent
    sources = {str(p.relative_to(source_root)): hashlib.sha256(p.read_bytes()).hexdigest()
               for p in sorted(source_root.rglob("*.py"))}
    cases = [run_case(args.output, method, tuple(args.mesh), args.dt, args.steps)
             for method in ("cpa", "ehrenfest")]
    current = {str(p.relative_to(source_root)): hashlib.sha256(p.read_bytes()).hexdigest()
               for p in sorted(source_root.rglob("*.py"))}
    if current != sources:
        raise RuntimeError("runtime sources changed during qualification; use an immutable source snapshot")
    summary = dict(complete=True, source_parameters=SOURCE_PARAMETERS, source_doi="10.1016/j.commatsci.2021.110535",
                   script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), sources=sources,
                   physical_scope="published equilibrium sp3 parameters; illustrative radial and neutral extensions",
                   basis="fixed orthonormal global Cartesian s,px,py,pz; spin-major; Pb/I sites; Cs omitted electronically",
                   cases=cases, devices=[str(d) for d in jax.devices()])
    (args.output/"report.json").write_text(json.dumps(summary, indent=2)+"\n")
    print(json.dumps(dict(complete=True, cases=cases), indent=2))


if __name__ == "__main__":
    main()
