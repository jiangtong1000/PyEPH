"""Full-gradient windows on a generated fixed-basis two-state neural Hamiltonian.

JAX_ENABLE_X64=1 python benchmarks/windowed_sensitivities.py --output new_record

Atomic units, one canonical coordinate and explicit mass 1.4. This is numerical
qualification, not a fitted material model or a new dynamics method. Positive
step observations enter one scalar loss; window boundaries retain the full
state tangent. Clocks are numerically equivalent within tested tolerances,
not bitwise identical. Compiler memory estimates are not process peak memory.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import platform
import traceback
import zipfile
import sys

import jax
import jax.numpy as jnp
import numpy as np

from pyeph import CPA, CoupledClassical, Ehrenfest, Integrator, Problem, make_state
from pyeph.execution.differentiable import DifferentiableRollout
from pyeph.models.analytic import SpinBosonModel
from pyeph.models.composite import SumModel
from pyeph.models.neural import NeuralResidualModel
from pyeph.observables.population import FunctionalMeasurement
from pyeph.paths.harmonic import HarmonicBath


def observation(problem, state):
    return dict(q=state.q, p=state.p, population=jnp.abs(state.electronic)**2)


def fixture(method):
    base = SpinBosonModel(1)
    bp = base.default_params() | dict(omega=jnp.array([.6]), coupling=jnp.array([.12]),
                                     bias=.08, delta=.09)
    neural = NeuralResidualModel(2, (1,), hidden_sizes=(8, 8), coordinate_kind="normal_mode")
    np_ = neural.init_params(jax.random.key(711), zero_last=False, scale=.1)
    model = SumModel((base, neural))
    nuclei = HarmonicBath(.6, 1.4) if method == "cpa" else CoupledClassical(1.4)
    dynamics = CPA() if method == "cpa" else Ehrenfest()
    initial = make_state(jnp.array([.35]), jnp.array([.2]),
                         jnp.array([jnp.cos(.4), jnp.sin(.4)*jnp.exp(.3j)]),
                         time=.7, step=3, trajectory_id=19)
    return Problem(model, (bp, np_), nuclei, dynamics,
                   FunctionalMeasurement(observation)), initial


def sample_sum(values):
    # Exclude segment-initial samples; each global positive step occurs once.
    return jnp.sum((values["population"][1:, 1] - .3)**2
                   + .07*jnp.sum(values["q"][1:]**2, axis=-1)
                   + .02*jnp.sum(values["p"][1:]**2, axis=-1))


def objectives(problem, initial, steps, window, dt=.02):
    if steps % window:
        raise ValueError("this experiment requires complete equal windows")
    integrator = Integrator(dt)
    # This construction also performs the shared whole-request host preflight.
    full = DifferentiableRollout(problem, integrator, initial, steps=steps)
    remat = DifferentiableRollout(problem, integrator, initial, steps=steps,
                                  rematerialize=True)
    piece = DifferentiableRollout(problem, integrator, initial, steps=window,
                                  rematerialize=True)

    def single(rollout):
        def loss(params):
            result = rollout(params, initial)
            return sample_sum(result.observables)/steps + .03*result.final_state.q[0]**2
        return loss

    def windowed(params, *, detach=False):
        def advance(carry, index):
            state, loss = carry
            if detach:
                state = jax.tree.map(jax.lax.stop_gradient, state)
            # Anchor each window to the original clock rather than repeatedly
            # accumulating segment durations. Internal arithmetic can still
            # differ by last-bit association from one monolithic scan.
            state = state._replace(time=initial.time + index*window*dt)
            result = piece(params, state)
            return (result.final_state, loss + sample_sum(result.observables)), None

        (final, value), _ = jax.lax.scan(jax.checkpoint(advance),
                                       (initial, jnp.asarray(0., jnp.float64)),
                                       jnp.arange(steps//window))
        return value/steps + .03*final.q[0]**2

    return dict(full=single(full), step_remat=single(remat), window_remat=windowed,
                detached=lambda params: windowed(params, detach=True))


def leaves(tree):
    return [np.asarray(x) for x in jax.tree.leaves(tree)]


def flat(tree):
    return np.concatenate([x.reshape(-1) for x in leaves(tree)])


def direction(params, seed):
    rng = np.random.default_rng(seed)
    vectors = jax.tree.map(lambda x: jnp.asarray(rng.normal(size=np.shape(x)), dtype=jnp.asarray(x).dtype), params)
    scale = np.linalg.norm(flat(vectors))
    return jax.tree.map(lambda x: x/scale, vectors)


def shifted(params, tangent, step):
    return jax.tree.map(lambda x, dx: x+step*dx, params, tangent)


def compile_measure(function, params):
    start = time.perf_counter()
    executable = jax.jit(jax.value_and_grad(function)).lower(params).compile()
    compile_seconds = time.perf_counter()-start
    value, gradient = jax.block_until_ready(executable(params))
    stats = executable.memory_analysis()
    fields = ("argument_size_in_bytes", "output_size_in_bytes", "alias_size_in_bytes",
              "temp_size_in_bytes", "generated_code_size_in_bytes")
    memory = None if stats is None else {field: int(getattr(stats, field)) for field in fields}
    samples = []
    for _ in range(5):
        start = time.perf_counter()
        jax.block_until_ready(executable(params))
        samples.append(time.perf_counter()-start)
    return executable, dict(value=float(value), gradient=flat(gradient),
                            compile_seconds=compile_seconds,
                            execution_seconds=samples, compiler_memory_bytes=memory)


def numpy_hamiltonian(params, coordinate):
    """Independent NumPy matrix and analytic coordinate derivative, no JAX calls."""
    base, neural = params
    q = np.asarray(coordinate)
    x = ((q-neural["q_center"])/neural["q_scale"]).reshape(-1)
    derivative = (np.ones_like(q)/neural["q_scale"]).reshape(-1)
    for layer in neural["layers"][:-1]:
        x = np.tanh(x @ layer["weight"] + layer["bias"])
        derivative = (derivative @ layer["weight"])*(1-x*x)
    last = neural["layers"][-1]
    raw = (x @ last["weight"] + last["bias"]).reshape(2, 2)
    draw = (derivative @ last["weight"]).reshape(2, 2)
    z = float(base["bias"] + base["coupling"] @ q)
    coupling = float(base["coupling"][0])
    h = np.array([[z, base["delta"]], [base["delta"], -z]]) + .5*(raw+raw.T)
    dh = np.diag([coupling, -coupling]) + .5*(draw+draw.T)
    return h, dh


def numpy_trajectory(params, initial, method, steps, dt=.02):
    """Explicit NumPy RK4 and electronic-half/Verlet splitting, same discrete scheme.

    The mass is 1.4. CPA has the independently prescribed frequency .6;
    Ehrenfest uses V_ref=.5*omega**2*(Q-Q_eq)**2 with the declared mass.
    No production Hamiltonian, derivative, propagator or force is called.
    """
    params = jax.tree.map(np.asarray, params)
    q, p, c = (np.array(value, copy=True) for value in
               (initial.q, initial.p, initial.electronic))
    q_rows, p_rows, c_rows = [q.copy()], [p.copy()], [c.copy()]

    def rk4(h_at, start, vector, duration):
        k1 = -1j*h_at(start) @ vector
        k2 = -1j*h_at(start+duration/2) @ (vector+duration*k1/2)
        k3 = -1j*h_at(start+duration/2) @ (vector+duration*k2/2)
        k4 = -1j*h_at(start+duration) @ (vector+duration*k3)
        return vector+duration*(k1+2*k2+2*k3+k4)/6

    def force(coordinate, electronic):
        derivative = numpy_hamiltonian(params, coordinate)[1]
        neutral = params[0]["omega"]**2*(coordinate-params[0]["q_eq"])
        return -neutral-np.vdot(electronic, derivative @ electronic).real

    for index in range(steps):
        if method == "cpa":
            start = float(initial.time) + index*dt

            def oscillator(elapsed):
                phase = .6*elapsed
                return (q*np.cos(phase)+p/1.4*np.sin(phase)/.6,
                        p*np.cos(phase)-1.4*.6*q*np.sin(phase))

            c = rk4(lambda t: numpy_hamiltonian(params, oscillator(t-start)[0])[0], start, c, dt)
            q, p = oscillator(dt)
        elif method == "ehrenfest":
            h = numpy_hamiltonian(params, q)[0]
            c = rk4(lambda t: h, 0., c, dt/2)
            half = p+dt/2*force(q, c)
            q = q+dt*half/1.4
            p = half+dt/2*force(q, c)
            h = numpy_hamiltonian(params, q)[0]
            c = rk4(lambda t: h, 0., c, dt/2)
        else:
            raise ValueError("choose cpa or ehrenfest")
        q_rows.append(q.copy())
        p_rows.append(p.copy())
        c_rows.append(c.copy())
    return dict(q=np.asarray(q_rows), p=np.asarray(p_rows), electronic=np.asarray(c_rows),
                population=np.abs(c_rows)**2)


def numpy_loss(values):
    count = len(values["q"])-1
    terms = ((values["population"][1:, 1]-.3)**2
             + .07*np.sum(values["q"][1:]**2, axis=-1)
             + .02*np.sum(values["p"][1:]**2, axis=-1))
    return float(np.sum(terms)/count+.03*values["q"][-1, 0]**2)


def plain(value):
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(v) for v in value]
    if isinstance(value, (np.ndarray, jax.Array)):
        return plain(np.asarray(value).tolist())
    if isinstance(value, np.generic):
        return plain(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return {"nonfinite": str(value)}
    return value


def save_json(path, value):
    with Path(path).open("x") as stream:
        json.dump(plain(value), stream, indent=2, allow_nan=False)
        stream.write("\n")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def retain_arrays(directory, stage, arrays):
    """Write raw evidence before rejecting nonfinite values; never overwrite it."""
    arrays = {name: np.asarray(value) for name, value in arrays.items()}
    path = directory/(stage+".npz")
    with path.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    diagnostics = {name: dict(shape=list(value.shape), dtype=value.dtype.str,
                  finite=bool(np.isfinite(value).all()),
                  nonfinite_count=int(np.count_nonzero(~np.isfinite(value))))
                   for name, value in arrays.items()}
    save_json(directory/(stage+"-finiteness.json"), diagnostics)
    if not all(record["finite"] for record in diagnostics.values()):
        raise FloatingPointError(f"nonfinite {stage} evidence retained")
    return dict(file=path.name, sha256=sha(path), finiteness=diagnostics)


def short_checks(problem, initial, funcs, steps, directory):
    """Qualified short-horizon AD/FD/HVP and independent discrete-primal checks."""
    params = problem.params
    tangent, second = direction(params, 73), direction(params, 91)
    gradient = jax.jit(jax.grad(funcs["window_remat"]))
    scalar = jax.jit(funcs["window_remat"])

    def hvp(function, vector):
        return jax.block_until_ready(jax.jit(lambda p, v:
            jax.jvp(jax.grad(function), (p,), (v,))[1])(params, vector))

    g = flat(jax.block_until_ready(gradient(params)))
    hv, hw = flat(hvp(funcs["window_remat"], tangent)), flat(hvp(funcs["window_remat"], second))
    full_hv = flat(hvp(funcs["full"], tangent))
    direct = float(g @ flat(tangent))
    detached_value, detached_gradient = jax.block_until_ready(jax.jit(
        jax.value_and_grad(funcs["detached"]))(params))
    method = type(problem.method).__name__.lower()
    reference = numpy_trajectory(params, initial, method, steps)
    rollout = DifferentiableRollout(problem, Integrator(.02), initial, steps=steps)
    native = jax.block_until_ready(rollout(params, initial))
    primal_errors = {name: float(np.max(np.abs(np.asarray(native.observables[name])-reference[name])))
                     for name in ("q", "p", "population")}
    primal_errors["final_electronic"] = float(np.max(np.abs(
        np.asarray(native.final_state.electronic)-reference["electronic"][-1])))
    value = float(scalar(params))
    primal_errors["loss"] = abs(value-numpy_loss(reference))
    values, native_values, derivatives, native_derivatives = [], [], [], []
    for coordinate in (-.4, .35, .9):
        q = np.array([coordinate])
        matrix, derivative = numpy_hamiltonian(jax.tree.map(np.asarray, params), q)
        values.append(matrix)
        derivatives.append(derivative)
        native_values.append(np.asarray(problem.model.dense(params, jnp.asarray(q))))
        native_derivatives.append(np.asarray(jax.jacfwd(problem.model.dense, argnums=1)(params, jnp.asarray(q)))[..., 0])
    matrix_error = float(np.max(np.abs(np.asarray(values)-native_values)))
    coordinate_derivative_error = float(np.max(np.abs(np.asarray(derivatives)-native_derivatives)))
    fd, first_values, gradient_values, independent_fd, independent_values = [], [], [], [], []
    widths = (1e-3, 5e-4)
    for width in widths:
        plus, minus = shifted(params, tangent, width), shifted(params, tangent, -width)
        # Native positive scales remain physical: every normalized direction
        # component has magnitude <=1 and both widths are below the unit scale.
        problem.model.validate_params(plus)
        problem.model.validate_params(minus)
        pair = np.array([scalar(plus), scalar(minus)], dtype=float)
        gradients = np.stack((flat(gradient(plus)), flat(gradient(minus))))
        independent = np.array([numpy_loss(numpy_trajectory(p, initial, method, steps))
                                for p in (plus, minus)])
        first_values.append(pair)
        gradient_values.append(gradients)
        independent_values.append(independent)
        independent_fd.append(float((independent[0]-independent[1])/(2*width)))
        fd.append(dict(width=width, first=float((pair[0]-pair[1])/(2*width)),
                       hv=(gradients[0]-gradients[1])/(2*width)))
    arrays = dict(direction=flat(tangent), second_direction=flat(second), gradient=g,
                  hvp=hv, second_hvp=hw, monolithic_hvp=full_hv,
                  detached_gradient=flat(detached_gradient), detached_value=detached_value,
                  window_value=value, fd_widths=widths, fd_loss_pairs=first_values,
                  fd_gradient_pairs=gradient_values, independent_fd_loss_pairs=independent_values,
                  independent_first_fd=independent_fd,
                  matrix=values, native_matrix=native_values, coordinate_derivative=derivatives,
                  native_coordinate_derivative=native_derivatives,
                  native_final_electronic=native.final_state.electronic,
                  **{f"independent_{name}": value for name, value in reference.items()},
                  **{f"native_{name}": value for name, value in native.observables.items()})
    raw = retain_arrays(directory, "short-derivatives", arrays)
    first_error = np.abs(np.asarray([record["first"] for record in fd])-direct)
    independent_error = np.abs(np.asarray(independent_fd)-direct)
    hv_error = np.asarray([np.max(np.abs(record["hv"]-hv)) for record in fd])
    hv_relative = np.asarray([np.linalg.norm(record["hv"]-hv)/np.linalg.norm(hv) for record in fd])
    # Declared floating-point allowance for subtractive FD checks, not a
    # theorem bounding all upstream evaluation errors.
    eps = np.finfo(np.float64).eps
    first_floor = 64*eps*max(1., float(np.max(np.abs(first_values))))/widths[-1]
    gradient_floor = 64*eps*max(1., float(np.max(np.abs(gradient_values))))/widths[-1]
    detached_error = float(np.linalg.norm(flat(detached_gradient)-g))
    detached_relative = detached_error/float(np.linalg.norm(g))
    reduced = dict(first_errors=first_error, independent_first_errors=independent_error,
                   hv_max_errors=hv_error, hv_relative_errors=hv_relative,
                   monolithic_hvp_max_error=np.max(np.abs(full_hv-hv)),
                   symmetry_error=abs(flat(second) @ hv-flat(tangent) @ hw),
                   detached_gradient_error=detached_error, detached_gradient_relative_error=detached_relative,
                   detached_value_error=abs(float(detached_value)-value),
                   matrix_error=matrix_error, coordinate_derivative_error=coordinate_derivative_error,
                   **{f"primal_{name}": value for name, value in primal_errors.items()})
    reduced_raw = retain_arrays(directory, "short-reduced", reduced)
    gates = dict(
        first_gradient_fd=bool(np.max(first_error) <= 2e-9),
        first_gradient_refines=bool(first_error[1] <= .4*first_error[0]+first_floor),
        independent_first_gradient_fd=bool(np.max(independent_error) <= 2e-9),
        independent_first_gradient_refines=bool(independent_error[1] <= .4*independent_error[0]+first_floor),
        hvp_fd=bool(hv_relative[-1] <= 2e-6),
        hvp_refines=bool(hv_error[1] <= .4*hv_error[0]+gradient_floor),
        monolithic_hvp=bool(reduced["monolithic_hvp_max_error"] <= 2e-9),
        hessian_symmetry=bool(reduced["symmetry_error"] <= 2e-9),
        independent_primal=all(value <= 2e-11 for value in primal_errors.values()),
        independent_matrix=matrix_error <= 2e-12,
        independent_coordinate_derivative=coordinate_derivative_error <= 2e-12,
        detached_same_primal=bool(reduced["detached_value_error"] <= 1e-10),
        detached_wrong_gradient=detached_error > 1e-6 and detached_relative > 1e-3,
    )
    return dict(gates=gates, errors=reduced, raw=raw, reduced_raw=reduced_raw,
                directional_derivative=direct, fd_widths=widths,
                fd_roundoff_allowances=dict(first=first_floor, hvp=gradient_floor),
                scope="first/HVP/independent-primal checks at shortest requested horizon only")


def case(problem, initial, steps, window, directory, *, short):
    directory.mkdir(exist_ok=False)
    funcs = objectives(problem, initial, steps, window)
    measured = {name: compile_measure(funcs[name], problem.params)[1]
                for name in ("full", "step_remat", "window_remat")}
    raw = retain_arrays(directory, "values-gradients", {
        f"{name}_{field}": value[field] for name, value in measured.items()
        for field in ("value", "gradient", "compile_seconds", "execution_seconds")})
    reference = measured["full"]
    differences = {name: dict(value=abs(value["value"]-reference["value"]),
                              gradient_max=float(np.max(abs(value["gradient"]-reference["gradient"]))))
                   for name, value in measured.items()}
    reduced = retain_arrays(directory, "differences", {
        f"{name}_{key}": value for name, record in differences.items() for key, value in record.items()})
    gates = dict(same_objective=all(record["value"] <= 1e-10 for record in differences.values()),
                 complete_gradient=all(record["gradient_max"] <= 2e-9 for record in differences.values()))
    higher = short_checks(problem, initial, funcs, steps, directory) if short else None
    if higher:
        gates.update(higher["gates"])
    report = dict(steps=steps, dt=.02, window=window, gates=gates, passed=all(gates.values()),
                  differences=differences, measurements=measured, raw=raw, reduced_raw=reduced,
                  short_checks=higher)
    save_json(directory/"report.json", report)
    if not report["passed"]:
        raise AssertionError(f"numerical gates failed: {[name for name, ok in gates.items() if not ok]}")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, nargs="+", default=[64, 512, 2048])
    parser.add_argument("--window", type=int, default=16)
    args = parser.parse_args()
    if not jax.config.x64_enabled:
        raise ValueError("explicit float64 mode is required")
    if (args.window <= 0 or len(set(args.steps)) != len(args.steps)
            or any(count <= args.window or count % args.window for count in args.steps)):
        raise ValueError("distinct step counts must be multiples of a positive window and span at least two windows")
    args.output.mkdir(exist_ok=False, parents=True)
    import pyeph
    import jaxlib
    package = Path(pyeph.__file__).resolve().parent
    runtime = {p.relative_to(package).as_posix(): sha(p) for p in package.rglob("*.py")}
    source_hash = sha(__file__)
    with zipfile.ZipFile(args.output/"sources.zip", "x", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(__file__, "benchmarks/windowed_sensitivities.py")
        for name in runtime:
            archive.write(package/name, "pyeph/"+name)
    report = dict(scope="generated real fixed-orthonormal two-state Hamiltonian; one canonical coordinate, explicit mass1.4; atomic units",
                  limitations="numerical clock reassociation, no bitwise trajectory equivalence; compiler memory estimates are not peak RSS; timing samples under local contention are not performance claims; no material or long-time conditioning claim",
                  python=sys.version, platform=platform.platform(), jax=jax.__version__, jaxlib=jaxlib.__version__,
                  numpy=np.__version__, devices=list(map(str, jax.devices())), package=str(package),
                  jax_enable_x64=bool(jax.config.x64_enabled),
                  jax_default_matmul_precision=jax.config.jax_default_matmul_precision,
                  source_sha256=source_hash, runtime=runtime, source_archive_sha256=sha(args.output/"sources.zip"),
                  window=args.window, step_counts=args.steps, cases=[], passed=False)
    for method in ("cpa", "ehrenfest"):
        problem, initial = fixture(method)
        pairs, _ = jax.tree.flatten_with_path(problem.params)
        inputs = {f"parameter_{i}": value for i, (_, value) in enumerate(pairs)}
        inputs.update(q=initial.q, p=initial.p, electronic=initial.electronic, time=initial.time,
                      step=initial.step, direction=flat(direction(problem.params, 73)),
                      second_direction=flat(direction(problem.params, 91)))
        retain_arrays(args.output, method+"-inputs", inputs)
        save_json(args.output/(method+"-parameter-schema.json"), [dict(index=i, path=str(path),
                  shape=list(np.shape(value)), dtype=np.asarray(value).dtype.str)
                  for i, (path, value) in enumerate(pairs)])
        for count in args.steps:
            directory = args.output/f"{method}-{count}"
            try:
                result = case(problem, initial, count, args.window, directory, short=count == min(args.steps))
                report["cases"].append(dict(method=method, steps=count, passed=True,
                                            gates=result["gates"], report_sha256=sha(directory/"report.json")))
            except Exception as exc:
                failure = dict(method=method, steps=count, passed=False,
                               exception=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
                save_json(directory/"failure.json", failure)
                report["cases"].append(failure)
            save_json(args.output/f"progress-{len(report['cases'])}.json", report)
            print(json.dumps(plain(report["cases"][-1])), flush=True)
            jax.clear_caches()
    report["sources_unchanged"] = source_hash == sha(__file__) and runtime == {
        p.relative_to(package).as_posix(): sha(p) for p in package.rglob("*.py")}
    report["passed"] = report["sources_unchanged"] and all(item["passed"] for item in report["cases"])
    save_json(args.output/"report.json", report)
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
