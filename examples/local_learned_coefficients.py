"""Finite/periodic local coefficients with complete atomic forces and currents.

Run: PYTHONPATH=src .venv/bin/python examples/local_learned_coefficients.py

Random toy NN weights illustrate the interface; no fitting or material accuracy
is claimed. Scalar distances/internal radii describe fixed scalar channels, not
molecular orientation or rotating p/d orbitals. Graphs explicitly contain the
candidate pairs; there is no neighbor search. Periodic fragments must remain
coherently unwrapped. Hopping charge currents omit moving-center convection,
intra-center dipoles, ionic currents, and AO connection terms. All units are
atomic. Existing JSON/NPZ evidence is never overwritten.
"""

import argparse
from dataclasses import dataclass, replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import tempfile
import time

import jax
import jax.numpy as jnp
import numpy as np

import pyeph
from pyeph import Ehrenfest, Execution, Integrator, Simulation
from pyeph.core.contracts import ProbeContext, pure_state_weight
from pyeph.core.problem import CoupledClassical, Problem
from pyeph.core.state import make_state
from pyeph.models.base import AutoDiffModel
from pyeph.models.composite import SumModel
from pyeph.models.local import AtomCenterMap, LocalBlockGraph, LocalBlockModel, LocalCoefficients
from pyeph.observables.population import FunctionalMeasurement


def mlp(params, features):
    return jnp.tanh(features @ params["w0"] + params["b0"]) @ params["w1"] + params["b1"]


@dataclass(frozen=True)
class ScalarCoefficientHead:
    """Example-only shared scalar head, plus a simple analytic carrier baseline."""

    block: int
    hidden: int = 5

    def init_nn(self, seed):
        rng = np.random.default_rng(seed)

        def layer(nin, nout):
            return dict(w0=jnp.asarray(rng.normal(size=(nin, self.hidden)) * .2),
                        b0=jnp.asarray(rng.normal(size=self.hidden) * .04),
                        w1=jnp.asarray(rng.normal(size=(self.hidden, nout)) * .08),
                        b1=jnp.asarray(rng.normal(size=nout) * .02))

        return dict(message=layer(3, self.hidden), node=layer(1+self.hidden, self.block**2),
                    edge=layer(3+self.hidden, self.block**2))

    def components(self, params, q, g):
        radius2 = jnp.zeros(g.centers.shape[0], q.dtype).at[g.atom_site].add(
            g.atom_weights * jnp.sum((q-g.centers[g.atom_site])**2, axis=-1))
        i, j = g.pairs[:, 0], g.pairs[:, 1]
        pair = jnp.stack((g.distances, radius2[i]+radius2[j], radius2[i]*radius2[j]), axis=-1)
        nn, baseline = params["nn"], params["baseline"]
        messages = mlp(nn["message"], pair) * g.support[:, None]
        env = jnp.zeros((g.centers.shape[0], self.hidden), q.dtype).at[i].add(messages).at[j].add(messages)
        nodes = mlp(nn["node"], jnp.concatenate((radius2[:, None], env), axis=-1))
        edges = mlp(nn["edge"], jnp.concatenate((pair, env[i]+env[j]), axis=-1))

        def symmetric(values):
            blocks = values.reshape(-1, self.block, self.block)
            return (blocks + blocks.swapaxes(-1, -2)) * .5

        neural = LocalCoefficients(symmetric(nodes), symmetric(edges))
        analytic = LocalCoefficients(
            baseline["onsite"] + radius2[:, None, None] * baseline["radius_response"],
            jnp.exp(-baseline["decay"] * (g.distances-baseline["r0"]))[:, None, None]
            * baseline["transfer"])
        return analytic, neural

    def __call__(self, params, q, geometry):
        baseline, neural = self.components(params, q, geometry)
        # Both terms are RAW; the adapter applies final hopping support once.
        return LocalCoefficients(*(a+b for a, b in zip(baseline, neural, strict=True)))


@dataclass(frozen=True)
class AtomicReference(AutoDiffModel):
    """Reference-only harmonic energy in the same all-atom coordinate space."""

    spec: object

    def apply(self, params, q, vectors):
        return jnp.zeros_like(vectors)

    def reference_energy(self, params, q):
        return .5*jnp.sum(params["spring"][:, None]*(q-params["q_eq"])**2)

    def probe_apply(self, params, context, probe, vectors):
        if probe not in self.spec.probes:
            return super().probe_apply(params, context, probe, vectors)
        return jnp.zeros_like(vectors)  # Nuclear reference owns no hopping current.


def measure(problem, state):
    model, params, c = problem.model, problem.params, state.electronic
    energy = (jnp.sum(state.p**2/(2*problem.nuclear_treatment.masses))
              + model.reference_energy(params, state.q) + jnp.vdot(c, model.apply(params, state.q, c)).real)
    context = ProbeContext(state.q, state.p/problem.nuclear_treatment.masses, state.time)
    current = jnp.stack([jnp.vdot(c, model.probe_apply(params, context, probe, c)).real
                         for probe in model.spec.probes])
    return dict(q=state.q, p=state.p, electronic=c, energy=energy,
                norm=jnp.vdot(c, c).real, current=current)


def fixture(periodic, seed):
    if periodic:
        q = jnp.array([[-.2, .1, .05], [.3, -.1, -.04], [1.4, .6, -.2], [1.7, .8, .15]])
        centers = AtomCenterMap((0, 0, 1, 1), (.4, .6, .7, .3), 2)
        graph = LocalBlockGraph(2, 2, ((0, 1, 0, 0, 0), (0, 1, -1, 0, 0), (0, 0, 1, 0, 0)),
                                cell=((4.3, .2, .1), (.4, 4.6, .3), (.1, .2, 4.9)),
                                switch_on=3.5, cutoff=5.5)
    else:
        q = jnp.array([[-.2, .1, .05], [.3, -.1, -.04], [1.4, .6, -.2], [1.7, .8, .15],
                       [2.5, -.4, .3], [2.2, -.1, .5]])
        centers = AtomCenterMap((0, 0, 1, 1, 2, 2), (.4, .6, .7, .3, .5, .5), 3)
        graph = LocalBlockGraph(3, 1, ((0, 1), (0, 2), (1, 2)), switch_on=3.5, cutoff=5.5)
    head = ScalarCoefficientHead(graph.norbitals)
    block = graph.norbitals
    baseline = dict(onsite=jnp.stack([jnp.eye(block)*(-.13+.15*i) for i in range(graph.nsites)]),
                    radius_response=jnp.eye(block)*.09,
                    transfer=jnp.asarray([[.08, .025], [.025, -.04]])[:block, :block],
                    decay=jnp.asarray(.35), r0=jnp.asarray(1.6))
    carrier = LocalBlockModel(graph, centers, head, charge=-1.)
    reference = AtomicReference(replace(carrier.spec, name="all_atom_harmonic_reference"))
    model = SumModel((carrier, reference), additive_probes=carrier.spec.probes)
    displacement = jnp.arange(q.size).reshape(q.shape)*.001 + .03
    params = (dict(baseline=baseline, nn=head.init_nn(seed)),
              dict(spring=jnp.linspace(.15, .25, len(q)), q_eq=q+displacement))
    masses = jnp.linspace(1., 2., len(q))[:, None]
    c = jnp.arange(1, model.nstates+1) + .2j*jnp.arange(model.nstates)
    c = c/jnp.linalg.norm(c)
    initial = make_state(q, .025*jnp.cos(jnp.arange(q.size).reshape(q.shape)), c, seed=seed)
    measurement = FunctionalMeasurement(measure, required_probes=carrier.spec.probes)
    return Problem(model, params, CoupledClassical(masses), Ehrenfest(), measurement), initial


def validate_numerics(problem, state):
    model, params, q, c = problem.model, problem.params, state.q, state.electronic
    carrier = model.models[0]
    energy = jax.jit(lambda x: model.reference_energy(params, x)+jnp.vdot(c, model.apply(params, x, c)).real)
    force = -(model.reference_gradient(params, q)+model.contract_gradient(params, q, pure_state_weight(c)))
    eps = 2e-5
    displacements = np.eye(q.size).reshape(-1, *q.shape)*eps
    force_fd = np.array([-(energy(q+d)-energy(q-d))/(2*eps) for d in displacements]).reshape(q.shape)
    np.testing.assert_allclose(force, force_fd, atol=2e-9, rtol=2e-7)
    current = jnp.stack([model.probe_apply(params, ProbeContext(q), name, c) for name in model.spec.probes])

    def full_peierls(kappa):
        # The reference has zero electronic action; all carrier coefficients are phased together.
        return carrier.apply_peierls(params[0], q, kappa, c)+model.models[1].apply(params[1], q, c)

    current_fd = jnp.stack([carrier.charge*(full_peierls(d)-full_peierls(-d))/(2*eps)
                            for d in jnp.eye(3)*eps])
    np.testing.assert_allclose(current, current_fd, atol=2e-9, rtol=2e-7)
    nn_zero = {**params[0]["nn"]}
    for name in ("node", "edge"):
        nn_zero[name] = {**nn_zero[name], "w1": jnp.zeros_like(nn_zero[name]["w1"]),
                         "b1": jnp.zeros_like(nn_zero[name]["b1"])}
    baseline_params = {**params[0], "nn": nn_zero}
    raw_baseline, _ = carrier.coefficient_provider.components(params[0], q, carrier.geometry(q))
    recovered = carrier.coefficients(baseline_params, q)
    baseline_error = max(float(jnp.max(abs(recovered.onsite-raw_baseline.onsite))),
                         float(jnp.max(abs(recovered.hopping-raw_baseline.hopping*carrier.geometry(q).support[:, None, None]))))
    assert baseline_error == 0.
    baseline_current = jnp.stack([carrier.probe_apply(baseline_params, ProbeContext(q), name, c)
                                  for name in model.spec.probes])
    learned_current_norm = float(jnp.linalg.norm(current-baseline_current))
    assert learned_current_norm > 1e-6
    neural_gradient = jax.grad(lambda nn: jnp.vdot(c, carrier.apply({**params[0], "nn": nn}, q, c)).real)(params[0]["nn"])
    gradient_norm = float(jnp.sqrt(sum(jnp.vdot(x, x).real for x in jax.tree.leaves(neural_gradient))))
    assert gradient_norm > 1e-6
    summary = dict(force_fd_max_abs=float(np.max(abs(force-force_fd))),
                   current_fd_max_abs=float(jnp.max(abs(current-current_fd))), zero_nn_recovery_max_abs=baseline_error,
                   learned_current_action_norm=learned_current_norm,
                   trainable_nn_energy_gradient_norm=gradient_norm,
                   reference_energy=float(model.reference_energy(params, q)),
                   reference_force_norm=float(jnp.linalg.norm(model.reference_gradient(params, q))))
    return summary, dict(force=force, force_fd=force_fd, current_action=current, current_action_fd=current_fd)


def archive_tree(prefix, value, arrays):
    if isinstance(value, dict):
        for key, child in value.items():
            archive_tree(f"{prefix}_{key}", child, arrays)
    elif isinstance(value, (tuple, list)):
        for index, child in enumerate(value):
            archive_tree(f"{prefix}_{index}", child, arrays)
    else:
        arrays[prefix] = np.asarray(value)


def source_fingerprint():
    root = Path(__file__).resolve().parents[1]
    if Path(pyeph.__file__).resolve().parent != root/"src/pyeph":
        raise RuntimeError("run this source-audited example against this repository's src/pyeph")
    files = sorted((root/"src/pyeph").rglob("*.py"))+[Path(__file__).resolve()]
    hashes = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in files}
    return dict(sha256=hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest(), files=hashes)


def run_case(periodic, seed, source, arrays):
    name = "periodic" if periodic else "finite"
    problem, initial = fixture(periodic, seed)
    summary, checks = validate_numerics(problem, initial)
    identity = source["files"]["examples/local_learned_coefficients.py"]
    ids = {path: f"example-source-sha256:{identity}:{label}" for path, label in (
        ("model.models[0].coefficient_provider", "ScalarCoefficientHead"),
        ("model.models[1]", "AtomicReference"), ("measurement.function", "measure"))}
    results, runners = [], []
    for dt, steps in ((.04, 20), (.02, 40)):
        runner = Simulation(problem, Integrator(dt, "rk4", electronic_substeps=2),
                            Execution(chunk_size=steps//2, save_every=steps//10))
        results.append(runner.run(initial, steps))
        runners.append(runner)
    coarse, fine = results
    np.testing.assert_allclose(coarse.times, fine.times, atol=3e-16, rtol=0.)
    runner = runners[0]
    prefix = runner.run(initial, 10)
    with tempfile.TemporaryDirectory(prefix="pyeph-local-example-") as directory:
        checkpoint = Path(directory)/"local.h5"
        runner.save_checkpoint(checkpoint, prefix.final_state, artifact_ids=ids)
        restored = runner.load_checkpoint(checkpoint, artifact_ids=ids)
        for actual, expected in zip(jax.tree.leaves(restored), jax.tree.leaves(prefix.final_state), strict=True):
            np.testing.assert_array_equal(actual, expected)
        suffix = runner.run(restored, 10)
        checkpoint_bytes = checkpoint.stat().st_size
    restart_error = 0.
    for key, target in coarse.observables.items():
        joined = np.concatenate((prefix.observables[key], suffix.observables[key][1:]))
        np.testing.assert_array_equal(joined, target)
        restart_error = max(restart_error, float(np.max(abs(joined-target))))
    np.testing.assert_array_equal(np.r_[prefix.times, suffix.times[1:]], coarse.times)
    for actual, expected in zip(jax.tree.leaves(suffix.final_state), jax.tree.leaves(coarse.final_state), strict=True):
        np.testing.assert_array_equal(actual, expected)
    # A numeric parameter update reuses this runner's compiled kernel, not a mutated closure.
    nn = problem.params[0]["nn"]
    changed_nn = {**nn, "edge": {**nn["edge"], "b1": nn["edge"]["b1"]+.01}}
    runner.update_parameters(({**problem.params[0], "nn": changed_nn}, problem.params[1]))
    updated = runner.run(initial, 20)
    update_delta = float(np.linalg.norm(updated.final_state.electronic-coarse.final_state.electronic))
    assert update_delta > 1e-5
    update_q_delta = float(np.linalg.norm(updated.final_state.q-coarse.final_state.q))
    assert update_q_delta > 1e-8
    summary.update(dict(
        atoms=len(initial.q), sites=problem.model.models[0].graph.nsites,
        channels_per_site=problem.model.models[0].graph.norbitals, seed=seed,
        dt=[.04, .02], steps=[20, 40], duration=.8, electronic="rk4", electronic_substeps=2,
        energy_drift=[float(np.max(abs(r.observables["energy"]-r.observables["energy"][0]))) for r in results],
        norm_error=[float(np.max(abs(r.observables["norm"]-1))) for r in results],
        coarse_fine_max_abs={key: float(np.max(abs(coarse.observables[key]-fine.observables[key])))
                             for key in coarse.observables},
        checkpoint_step=10, checkpoint_time=.4, checkpoint_restore_exact=True, checkpoint_bytes=checkpoint_bytes,
        restart_max_abs=restart_error, duplicate_policy="keep prefix endpoint, drop shared suffix row0",
        updated_parameter_electronic_l2=update_delta, updated_parameter_nuclear_q_l2=update_q_delta, artifact_ids=ids,
        provider_config=vars(problem.model.models[0].coefficient_provider), charge=problem.model.models[0].charge,
        graph=vars(problem.model.models[0].graph), atom_map=vars(problem.model.models[0].centers)))
    assert max(summary["norm_error"]) < 1e-8
    assert summary["energy_drift"][1] < summary["energy_drift"][0]
    archive_tree(f"{name}_params", problem.params, arrays)
    archive_tree(f"{name}_updated_params", runner.problem.params, arrays)
    archive_tree(f"{name}_initial", initial._asdict(), arrays)
    archive_tree(f"{name}_masses", problem.nuclear_treatment.masses, arrays)
    archive_tree(f"{name}_checks", checks, arrays)
    for label, result in (("coarse", coarse), ("fine", fine), ("updated", updated), ("restart_suffix", suffix)):
        archive_tree(f"{name}_{label}", dict(times=result.times, **result.observables), arrays)
        archive_tree(f"{name}_{label}_final", result.final_state._asdict(), arrays)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/local_learned_coefficients"),
                        help="JSON/NPZ output directory (default: %(default)s)")
    args = parser.parse_args()
    paths = [args.output_dir/f"local_learned_coefficients.{suffix}" for suffix in ("json", "npz")]
    if any(path.exists() for path in paths):
        parser.error("output evidence already exists; select another --output-dir")
    jax.config.update("jax_enable_x64", True)
    started, clock = datetime.now(timezone.utc).isoformat(), time.perf_counter()
    source = source_fingerprint()
    arrays = {}
    cases = {name: run_case(periodic, seed, source, arrays)
             for name, periodic, seed in (("finite", False, 37), ("periodic", True, 41))}
    source_end = source_fingerprint()
    if source != source_end:
        raise RuntimeError("runtime/example source changed; evidence not saved")
    summary = dict(scope="random scalar-channel interface demonstration; no fit, orbital equivariance, or material accuracy",
                   units="atomic", started_utc=started, cases=cases, source_start=source, source_end=source_end,
                   source_unchanged=True, elapsed_seconds_including_compilation=time.perf_counter()-clock,
                   timing_scope="illustrative shared-host execution; not a performance benchmark",
                   software=dict(python=platform.python_version(), numpy=np.__version__, jax=jax.__version__,
                                 backend=jax.default_backend(), devices=[str(d) for d in jax.devices()]))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with paths[1].open("xb") as handle:
        np.savez_compressed(handle, **arrays)
    summary["npz_sha256"] = hashlib.sha256(paths[1].read_bytes()).hexdigest()
    with paths[0].open("x") as handle:
        handle.write(json.dumps(summary, indent=2, allow_nan=False)+"\n")
    print(json.dumps({name: {k: case[k] for k in ("force_fd_max_abs", "current_fd_max_abs", "energy_drift", "norm_error", "restart_max_abs")}
                      for name, case in cases.items()}, indent=2))
    print(f"Saved {paths[0]} and {paths[1]}")


if __name__ == "__main__":
    main()
