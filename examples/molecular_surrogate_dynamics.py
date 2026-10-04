"""Run a saved local molecular surrogate through unchanged CPA/Ehrenfest.

The independent NumPy reference evaluates the exported weights and finite-
differences the complete energy. These checks quantify integrator error for
the learned model. Teacher error along the saved path is a separate check.
"""

import argparse
import hashlib
import inspect
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from scipy.constants import physical_constants
from scipy.integrate import solve_ivp

from pyeph.learning import load_labels
from molecular_residual import load_artifact
from oriented_fragments import measure
from pyeph import CPA, Ehrenfest, Execution, Integrator, Problem, Simulation
from pyeph.core.problem import CoupledClassical
from pyeph.core.state import make_state, stack_states
from pyeph.observables.population import FunctionalMeasurement
from pyeph.paths.harmonic import HarmonicBath


def fixture(model, params, q, *, method="ehrenfest", trajectory_id=71):
    if method not in ("cpa", "ehrenfest"):
        raise ValueError("choose cpa or ehrenfest")
    amu = physical_constants["atomic mass constant"][0] / physical_constants["electron mass"][0]
    masses = jnp.asarray([12.011, 12.011, 1.008, 1.008, 1.008, 1.008] * 2)[:, None] * amu
    treatment = CoupledClassical(masses) if method == "ehrenfest" else HarmonicBath(0., masses)
    problem = Problem(model, params, treatment, Ehrenfest() if method == "ehrenfest" else CPA(),
                      FunctionalMeasurement(measure, required_probes=model.spec.probes))
    p = .12*jnp.cos(jnp.arange(36).reshape(12, 3))
    return problem, make_state(jnp.asarray(q), p, jnp.asarray([.8, .36+.48j]),
                               seed=261004, trajectory_id=trajectory_id)


def numpy_values(problem, q):
    """Independent matrix, relative neutral energy and current from plain arrays.

    Keep the neutral offset out of finite differences to avoid subtracting
    approximately -155 Hartree when only the local energy variation matters.
    """
    carrier = problem.model.models[0]
    params, reference = jax.tree.map(np.asarray, problem.params)
    base, network = params["baseline"], params["network"]
    q = np.asarray(q)
    swapped = np.concatenate((q[6:], q[:6]))
    pairs = np.triu_indices(12, 1)
    def distances(x):
        return np.linalg.norm(x[pairs[0]]-x[pairs[1]], axis=1)
    def neural(p, x, states):
        feature = (distances(x)-p["q_center"])/p["q_scale"]
        for layer in p["layers"][:-1]:
            feature = np.tanh(feature@layer["weight"]+layer["bias"])
        output = feature@p["layers"][-1]["weight"]+p["layers"][-1]["bias"]
        raw = output.reshape(states, states)
        return (raw+raw.T)/2
    residual = (neural(network, q, 2)+neural(network, swapped, 2)[::-1, ::-1])/2
    h = np.zeros((2, 2))
    normals = []
    provider = carrier.coefficient_provider.baseline
    for i, (o, a, b) in enumerate(provider.anchors):
        u, v = q[a]-q[o], q[b]-q[o]
        normal = np.cross(u, v)
        normals.append(provider.phases[i]*normal/np.linalg.norm(normal))
        h[i, i] = base["onsite"][i]+base["deformation"][i]@(
            np.array([np.linalg.norm(u), np.linalg.norm(v)])-base["bond_lengths"][i])
    phase = np.diag(provider.phases)
    residual = phase@residual@phase
    h[np.diag_indices(2)] += np.diag(residual)
    centers = q.reshape(2, 6, 3).mean(axis=1)
    delta = centers[1]-centers[0]
    distance = np.linalg.norm(delta)
    tensor = base["pp_pi"]*np.eye(3)+(base["pp_sigma"]-base["pp_pi"])*np.outer(delta, delta)/distance**2
    raw = normals[0]@tensor@normals[1]*np.exp(-base["decay"]*(distance-base["reference_distance"]))
    x = np.clip((distance-carrier.graph.switch_on)/(carrier.graph.cutoff-carrier.graph.switch_on), 0., 1.)
    support = 1-10*x**3+15*x**4-6*x**5
    h[0, 1] = h[1, 0] = (raw+residual[0, 1])*support
    def scalar(x):
        feature = (distances(x)-reference["center"])/reference["scale"]
        polynomial = np.r_[1., feature, feature**2]@reference["polynomial"]
        return polynomial+neural(reference["network"], x, 1)[0, 0]
    energy = (scalar(q)+scalar(swapped))/2
    current = np.zeros((3, 2, 2), complex)
    current[:, 0, 1] = 1j*carrier.charge*delta*h[0, 1]
    current[:, 1, 0] = current[:, 0, 1].conj()
    return h, energy, current


def numpy_force(problem, q, c, *, step=2e-5):
    def energy(x):
        h, e, _ = numpy_values(problem, x)
        return e + np.vdot(c, h@c).real
    force = np.empty((12, 3))
    for atom, axis in np.ndindex(12, 3):
        delta = np.zeros((12, 3))
        delta[atom, axis] = step
        force[atom, axis] = -(energy(q+delta)-energy(q-delta))/(2*step)
    return force


def scipy_reference(problem, initial, times, *, force_step=2e-5):
    masses = np.asarray(problem.nuclear_treatment.masses)
    y0 = np.r_[np.asarray(initial.q).ravel(), np.asarray(initial.p).ravel(),
               np.asarray(initial.electronic).real, np.asarray(initial.electronic).imag]
    def rhs(time, y):
        q, p = y[:36].reshape(12, 3), y[36:72].reshape(12, 3)
        c = y[72:74]+1j*y[74:76]
        h, _, _ = numpy_values(problem, q)
        force = np.zeros_like(q) if isinstance(problem.method, CPA) else numpy_force(problem, q, c, step=force_step)
        dc = -1j*h@c
        return np.r_[(p/masses).ravel(), force.ravel(), dc.real, dc.imag]
    result = solve_ivp(rhs, (float(times[0]), float(times[-1])), y0, t_eval=times,
                       method="DOP853", rtol=2e-11, atol=2e-13)
    if not result.success:
        raise RuntimeError(result.message)
    q = result.y[:36].T.reshape(-1, 12, 3)
    p = result.y[36:72].T.reshape(-1, 12, 3)
    c = result.y[72:74].T+1j*result.y[74:76].T
    current = np.array([np.einsum("i,aij,j->a", v.conj(), numpy_values(problem, x)[2], v).real
                        for x, v in zip(q, c, strict=True)])
    return dict(q=q, p=p, electronic=c, current=current), result.nfev


def compare_restart_states(actual, expected, *, exact=False):
    """Qualify this example's state storage separately from its continuation.

    The 2e-12 allowance is a declared numerical comparison policy, scaled in
    each field's own units. It is not a floating-point error theorem for all
    devices. Shapes, dtypes, tree structure and discrete state remain exact.
    """
    actual_leaves, actual_tree = jax.tree_util.tree_flatten_with_path(actual)
    expected_leaves, expected_tree = jax.tree_util.tree_flatten_with_path(expected)
    if actual_tree != expected_tree:
        raise AssertionError("restart state tree differs")
    fields = {}
    for (path, left), (_, right) in zip(actual_leaves, expected_leaves, strict=True):
        name = jax.tree_util.keystr(path)
        left, right = np.asarray(left), np.asarray(right)
        if left.shape != right.shape or left.dtype != right.dtype:
            raise AssertionError(f"restart shape/dtype differs: {name}")
        if (left.dtype.kind not in "biufc" or not np.isfinite(left).all()
                or not np.isfinite(right).all()):
            raise AssertionError(f"nonfinite or nonnumerical restart field: {name}")
        bytewise = left.tobytes() == right.tobytes()
        discrete = left.dtype.kind in "biu"
        if exact or discrete:
            if not bytewise:
                raise AssertionError(f"restart bytes differ: {name}")
            error, scale, tolerance = 0., 0., 0.
        else:
            scale = float(np.max(np.abs(right))) if right.size else 0.
            tolerance = 2e-12
            np.testing.assert_allclose(left, right, atol=tolerance*scale,
                                       rtol=tolerance, equal_nan=False, err_msg=name)
            error = float(np.max(np.abs(left-right))) if right.size else 0.
        fields[name] = dict(bytewise_equal=bytewise, max_absolute_error=error,
                            scale=scale, atol=tolerance*scale, rtol=tolerance)
    return dict(numerically_qualified=True,
                bytewise_equal=all(value["bytewise_equal"] for value in fields.values()),
                fields=fields)


def run_case(problem, initial, output, *, steps=24, dt=.5, artifact_metadata=None):
    name = "cpa" if isinstance(problem.method, CPA) else "ehrenfest"
    if steps < 1 or not np.isfinite(dt) or dt <= 0:
        raise ValueError("steps and timestep must be positive")
    identities = {}
    for path, obj in (("model.models[0].coefficient_provider", problem.model.models[0].coefficient_provider),
                       ("model.models[1]", problem.model.models[1]),
                       ("measurement.function", measure)):
        source = Path(inspect.getfile(obj if inspect.isfunction(obj) else type(obj)))
        bundle = dict(source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                      artifacts=artifact_metadata)
        identities[path] = "sha256:"+hashlib.sha256(json.dumps(bundle, sort_keys=True).encode()).hexdigest()
    results, arrays = [], {}
    for factor in (1, 2):
        runner = Simulation(problem, Integrator(dt/factor, "exponential_midpoint"),
                             Execution(chunk_size=7, save_every=factor))
        result = runner.run(initial, steps*factor)
        results.append(result)
        arrays.update({f"{key}_{factor}": value for key, value in result.observables.items()})
        for q in result.observables["q"]:
            problem.model.validate_at(problem.params, q)
        if factor == 2:
            # An odd partition checks save phase and checkpoint reconstruction.
            partition = min(7, steps*factor-1)
            part = runner.run(initial, partition)
            runner.save_checkpoint(output/f"{name}.h5", part.final_state, artifact_ids=identities)
            state = runner.load_checkpoint(output/f"{name}.h5", artifact_ids=identities)
            checkpoint_check = compare_restart_states(state, part.final_state, exact=True)
            resumed = runner.run(state, steps*factor-partition)
            continuation_check = compare_restart_states(resumed.final_state, result.final_state)
            # Compare distinct stable IDs in one batch against individual runs.
            second = make_state(initial.q, initial.p*1.03, initial.electronic,
                                 seed=261004, trajectory_id=73)
            singles = [runner.run(s, 4) for s in (initial, second)]
            batched = runner.run(stack_states((initial, second)), 4)
            for index, single in enumerate(singles):
                for key in single.observables:
                    np.testing.assert_allclose(batched.observables[key][:, index], single.observables[key], atol=2e-12, rtol=2e-12)
    reference, nfev = scipy_reference(problem, initial, results[0].times)
    errors = {key: [float(np.max(abs(r.observables[key]-value))) for r in results]
              for key, value in reference.items()}
    for key, (coarse, fine) in errors.items():
        if fine > .4*coarse+2e-9:
            raise AssertionError(f"{name} {key} refinement failed: {coarse}, {fine}")
    if errors["electronic"][1] > 1e-5 or errors["p"][1] > 1e-4:
        raise AssertionError("surrogate integration accuracy gate failed")
    arrays.update({f"reference_{key}": value for key, value in reference.items()})
    arrays["times"] = results[0].times
    np.savez_compressed(output/f"{name}.npz", **arrays)
    energy = [float(np.max(abs(r.observables["energy"]-r.observables["energy"][0]))) for r in results]
    if name == "ehrenfest" and energy[1] > .4*energy[0]+2e-11:
        raise AssertionError("Ehrenfest energy drift did not refine")
    return dict(method=name, dt=[dt, dt/2], steps=[steps, 2*steps], reference_nfev=nfev,
                errors=errors, energy_change=energy, energy_conservation_expected=name == "ehrenfest",
                norm_drift=[float(np.max(abs(r.observables["norm"]-1))) for r in results],
                checkpoint_roundtrip_bitwise=checkpoint_check["bytewise_equal"],
                restart_bitwise=continuation_check["bytewise_equal"],
                restart_numerically_qualified=continuation_check["numerically_qualified"],
                restart_comparison=continuation_check,
                restart_scope="exact checkpoint bytes; numerical final-state continuation",
                batch_partition_checked=True,
                max_coordinate_excursion_bohr=float(np.max(abs(results[1].observables["q"]-initial.q))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    parser.add_argument("labels", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--dt", type=float, default=.5)
    args = parser.parse_args()
    models, params, report = load_artifact(args.artifact)
    arrays, metadata = load_labels(args.labels)
    if metadata["arrays_sha256"] != report["dataset"]["arrays_sha256"]:
        raise ValueError("dynamics dataset differs from the fitted artifact")
    initial_id = report["splits"]["test"][0]
    row = list(arrays["geometry_ids"]).index(initial_id)
    args.output.mkdir(parents=True, exist_ok=False)
    artifact_ids = dict(model=report["parameters_sha256"], dataset=metadata["arrays_sha256"],
                        dynamics=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    cases = []
    for method in ("cpa", "ehrenfest"):
        problem, initial = fixture(models[-1], params[1:], arrays["q"][row], method=method)
        cases.append(run_case(problem, initial, args.output, steps=args.steps, dt=args.dt,
                              artifact_metadata=artifact_ids))
    record = dict(scope="integrator validation against the learned model's own NumPy/SciPy reference",
                  initial_geometry_id=initial_id, artifacts=artifact_ids, cases=cases,
                  limitation="local fitted domain; coordinate excursion is recorded, not a calibrated uncertainty bound")
    (args.output/"report.json").write_text(json.dumps(record, indent=2)+"\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
