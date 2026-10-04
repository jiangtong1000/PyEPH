"""Fit a generated smooth two-state model to independent SciPy trajectories.

Run with JAX_ENABLE_X64=1 and --output NEW_DIRECTORY. CPA and Ehrenfest are
separate fixed protocols. This checks differentiable numerical calibration,
not material accuracy, general identifiability or a new dynamics method.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import traceback
import zipfile

import jax
import jax.numpy as jnp
import numpy as np
import scipy
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

import pyeph
from pyeph import CPA, CoupledClassical, Ehrenfest, Integrator, Problem, make_state
from pyeph.core.state import stack_states
from pyeph.execution.differentiable import DifferentiableRollout
from pyeph.models.analytic import SpinBosonModel
from pyeph.observables.population import FunctionalMeasurement
from pyeph.paths.harmonic import HarmonicBath


BIAS, OMEGA = .04, .35
TRUTH = np.array([.12, .09])  # g per canonical length, delta in hartree
INITIAL = np.array([.06, .15])
DT, STEPS, STRIDE = .05, 160, 8
BOUNDS = np.log(np.array([[.02, .3], [.02, .3]]))
OPTIMIZER = dict(maxiter=100, maxls=30, gtol=1e-9, ftol=1e-15)
TRAIN, HOLDOUT = np.array([0, 1, 2]), np.array([3, 4])
Q0 = np.array([-.8, .4, 1.1, -.3, .7])
P0 = np.array([.2, -.35, .05, -.1, .3])
ANGLES = np.array([0., .7, 1.2, .4, 1.])
PHASES = np.array([0., .4, -.6, 1., -.3])
C0 = np.column_stack((np.cos(ANGLES/2), np.exp(1j*PHASES)*np.sin(ANGLES/2)))
GATES = dict(teacher_refinement=2e-10, discrete_gradient=2e-8,
             independent_gradient_floor=1e-8, parameter_relative_error=.002,
             holdout_scaled_rmse=.001, sensitivity_rank_tolerance=1e-8,
             integrator_refinement_floor=1e-9)


def plain(value):
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
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


def sha(payload):
    return hashlib.sha256(payload).hexdigest()


def require_finite(directory, stage, values):
    """Save explicit diagnostics; retain failed arrays before rejecting a stage."""
    diagnostics = {}
    for name, value in values.items():
        array = np.asarray(value)
        mask = np.isfinite(array)
        diagnostics[name] = dict(finite=bool(np.all(mask)), shape=list(array.shape),
                                 nonfinite_count=int(np.count_nonzero(~mask)))
    save_json(directory/(stage+"-finiteness.json"), diagnostics)
    if not all(item["finite"] for item in diagnostics.values()):
        np.savez_compressed(directory/(stage+"-nonfinite.npz"), **values)
        failed = [name for name, item in diagnostics.items() if not item["finite"]]
        raise FloatingPointError(f"nonfinite {stage} evidence retained for {failed}")
    return diagnostics


def teacher(parameters, indices, times, method, *, refined=False):
    """Independent NumPy ODE; no production Hamiltonian/force/step calls."""
    g, delta = np.asarray(parameters, dtype=float)
    outputs, states, diagnostics = [], [], []
    for index in indices:
        initial = np.r_[Q0[index], P0[index], C0[index].real, C0[index].imag]

        def rhs(_time, y):
            q, p = y[:2]
            c = y[2:4]+1j*y[4:6]
            z = BIAS+g*q
            dc = -1j*np.array([z*c[0]+delta*c[1], delta*c[0]-z*c[1]])
            force = -OMEGA**2*q
            if method == "ehrenfest":
                force -= g*(abs(c[0])**2-abs(c[1])**2)
            return np.r_[p, force, dc.real, dc.imag]

        factor = .1 if refined else 1.
        result = solve_ivp(rhs, (float(times[0]), float(times[-1])), initial,
                           t_eval=times, method="DOP853", rtol=2e-12*factor,
                           atol=2e-14*factor, max_step=.1*(.5 if refined else 1.))
        if not result.success or result.y.shape != (6, len(times)):
            raise RuntimeError(f"independent teacher failed for initial condition {index}: "
                               f"{result.message}")
        q, p = result.y[:2]
        c = result.y[2:4]+1j*result.y[4:6]
        product = c[0].conj()*c[1]
        values = [2*product.real, 2*product.imag, abs(c[0])**2-abs(c[1])**2]
        if method == "ehrenfest":
            values += [q, p]
        outputs.append(np.stack(values, axis=-1))
        states.append(result.y.T)
        diagnostics.append(dict(initial_index=int(index), success=bool(result.success),
                                message=result.message, nfev=result.nfev,
                                max_norm_error=float(np.max(abs(np.sum(abs(c)**2, axis=0)-1)))))
    return np.stack(outputs, axis=1), np.stack(states, axis=1), diagnostics


def measurement(problem, state):
    c = state.electronic
    product = c[0].conj()*c[1]
    values = [2*product.real, 2*product.imag, jnp.abs(c[0])**2-jnp.abs(c[1])**2]
    if isinstance(problem.method, Ehrenfest):
        values += [state.q[0], state.p[0]]
    return {"features": jnp.stack(values)}


def native(method, indices, *, refinement=1):
    model = SpinBosonModel(1)
    base = model.default_params() | dict(omega=jnp.array([OMEGA]), bias=jnp.array(BIAS),
                                        coupling=jnp.array([INITIAL[0]]), delta=jnp.array(INITIAL[1]))
    initial = stack_states([make_state([Q0[i]], [P0[i]], C0[i], trajectory_id=int(i))
                            for i in indices])
    nuclear = HarmonicBath([OMEGA], 1.) if method == "cpa" else CoupledClassical(1.)
    problem = Problem(model, base, nuclear, CPA() if method == "cpa" else Ehrenfest(),
                      FunctionalMeasurement(measurement))
    rollout = DifferentiableRollout(problem, Integrator(DT/refinement, "rk4"), initial,
                                    steps=STEPS*refinement, rematerialize=True)

    def parameters(theta):
        g, delta = jnp.exp(theta)
        return base | dict(coupling=jnp.reshape(g, (1,)), delta=delta)

    def predict(theta):
        return rollout(parameters(theta), initial).observables["features"][::STRIDE*refinement]

    return jax.jit(predict), rollout, parameters, initial


def finite_difference(function, theta, width):
    return np.array([(function(theta+width*d)-function(theta-width*d))/(2*width)
                     for d in np.eye(len(theta))])


def run_case(method, directory):
    directory.mkdir()
    times = np.arange(0, STEPS+1, STRIDE)*DT
    target, raw_target, teacher_info = teacher(TRUTH, np.arange(5), times, method)
    refined_target, refined_raw, refined_info = teacher(TRUTH, np.arange(5), times, method, refined=True)
    np.savez_compressed(directory/"teacher.npz", times=times, observations=target,
                        raw_states=raw_target, refined_observations=refined_target,
                        refined_raw_states=refined_raw)
    save_json(directory/"teacher.json", dict(parameters=TRUTH, diagnostics=teacher_info,
                                           refined_diagnostics=refined_info,
                                           arrays_sha256=sha((directory/"teacher.npz").read_bytes())))
    finiteness = {"teacher": require_finite(directory, "teacher", dict(
        times=times, target=target, raw_target=raw_target, refined_target=refined_target,
        refined_raw=refined_raw,
        norm_errors=[item["max_norm_error"] for item in teacher_info+refined_info]))}
    scales = np.array([1., 1., 1.] + ([1., .5] if method == "ehrenfest" else []))
    training_target = jnp.asarray(target[:, TRAIN])
    predict, rollout, params, initial = native(method, TRAIN)
    predict_fine, _, _, _ = native(method, TRAIN, refinement=2)

    def loss(theta):
        return jnp.mean(((predict(theta)-training_target)/scales)**2)

    def loss_fine(theta):
        return jnp.mean(((predict_fine(theta)-training_target)/scales)**2)

    value_gradient = jax.jit(jax.value_and_grad(loss))
    theta0 = np.log(INITIAL)
    initial_loss, gradient = value_gradient(theta0)
    gradient = np.asarray(gradient)
    gradient_fine = np.asarray(jax.grad(loss_fine)(theta0))
    finite = [finite_difference(lambda x: float(loss(x)), theta0, width)
              for width in (1e-4, 5e-5)]

    def independent_loss(theta):
        values, _, _ = teacher(np.exp(theta), TRAIN, times, method, refined=True)
        return float(np.mean(((values-target[:, TRAIN])/scales)**2))

    independent_gradient = finite_difference(independent_loss, theta0, 2e-5)
    finiteness["initial_gradient"] = require_finite(directory, "initial-gradient", dict(
        initial_loss=initial_loss, gradient=gradient, gradient_fine=gradient_fine,
        finite_differences=finite, independent_gradient=independent_gradient))
    history = []

    def objective(theta):
        value, derivative = value_gradient(theta)
        history.append(dict(log_parameters=theta.copy(), loss=float(value),
                            gradient=np.asarray(derivative).copy()))
        return float(value), np.asarray(derivative)

    # Only the declared training trajectories enter this single fixed fit.
    fit = minimize(objective, theta0, method="L-BFGS-B", jac=True,
                   bounds=BOUNDS, options=OPTIMIZER)
    fitted = np.exp(fit.x)
    optimizer_record = dict(success=bool(fit.success), status=int(fit.status), message=str(fit.message),
                            nit=int(fit.nit), nfev=int(fit.nfev), initial_loss=float(initial_loss),
                            final_loss=float(fit.fun), final_gradient=fit.jac,
                            history=history, fitted_parameters=fitted)
    save_json(directory/"optimizer.json", optimizer_record)
    finiteness["optimizer"] = require_finite(directory, "optimizer", dict(
        final_loss=fit.fun, final_log_parameters=fit.x, final_gradient=fit.jac,
        fitted_parameters=fitted, evaluated_loss=[item["loss"] for item in history],
        evaluated_gradient=[item["gradient"] for item in history],
        evaluated_log_parameters=[item["log_parameters"] for item in history]))
    rollout.preflight(params(jnp.asarray(fit.x)), initial)
    train_prediction = np.asarray(predict(fit.x))
    held_predict, _, _, _ = native(method, HOLDOUT)
    held_fine_predict, _, _, _ = native(method, HOLDOUT, refinement=2)
    held_coarse, held_fine = np.asarray(held_predict(fit.x)), np.asarray(held_fine_predict(fit.x))
    held_reference, held_reference_raw, fitted_teacher_info = teacher(
        fitted, HOLDOUT, times, method, refined=True)
    sensitivity = np.asarray(jax.jacfwd(lambda x: (predict(x)/scales).ravel())(fit.x))
    finiteness["prediction"] = require_finite(directory, "prediction", dict(
        train_prediction=train_prediction, held_coarse=held_coarse, held_fine=held_fine,
        held_reference=held_reference, held_reference_raw=held_reference_raw,
        sensitivity=sensitivity, scales=scales,
        fitted_teacher_norm_errors=[item["max_norm_error"] for item in fitted_teacher_info]))
    singular = np.linalg.svd(sensitivity, compute_uv=False)
    coarse_error = float(np.max(abs(held_coarse-held_reference)))
    fine_error = float(np.max(abs(held_fine-held_reference)))
    coarse_gradient_error = float(np.max(abs(gradient-independent_gradient)))
    fine_gradient_error = float(np.max(abs(gradient_fine-independent_gradient)))
    holdout_rmse = float(np.sqrt(np.mean(((held_coarse-target[:, HOLDOUT])/scales)**2)))
    teacher_refinement_error = float(np.max(abs(refined_target-target)))
    discrete_gradient_error = float(max(np.max(abs(x-gradient)) for x in finite))
    train_rmse = float(np.sqrt(np.mean(((train_prediction-target[:, TRAIN])/scales)**2)))
    holdout_fine_rmse = float(np.sqrt(np.mean(((held_fine-target[:, HOLDOUT])/scales)**2)))
    relative_parameter_error = fitted/TRUTH-1
    condition_number = float(singular[0]/singular[-1])
    finiteness["reduced_errors"] = require_finite(directory, "reduced-errors", dict(
        coarse_error=coarse_error, fine_error=fine_error,
        coarse_gradient_error=coarse_gradient_error, fine_gradient_error=fine_gradient_error,
        teacher_refinement_error=teacher_refinement_error, discrete_gradient_error=discrete_gradient_error,
        train_rmse=train_rmse, holdout_rmse=holdout_rmse, holdout_fine_rmse=holdout_fine_rmse,
        singular_values=singular, condition_number=condition_number,
        relative_parameter_error=relative_parameter_error))
    checks = dict(
        finite_numerical_evidence=all(item["finite"] for stage in finiteness.values()
                                      for item in stage.values()),
        optimizer_success=bool(fit.success),
        teacher_self_refinement=bool(teacher_refinement_error < GATES["teacher_refinement"]),
        discrete_gradient=bool(discrete_gradient_error < GATES["discrete_gradient"]),
        independent_gradient_refinement=bool(fine_gradient_error <= .4*coarse_gradient_error+GATES["independent_gradient_floor"]),
        known_parameter_recovery=bool(np.max(abs(fitted/TRUTH-1)) < GATES["parameter_relative_error"]),
        local_sensitivity_rank=bool(singular[-1] > GATES["sensitivity_rank_tolerance"]*singular[0]),
        heldout_prediction=bool(holdout_rmse < GATES["holdout_scaled_rmse"]),
        heldout_integrator_refinement=bool(fine_error <= .4*coarse_error+GATES["integrator_refinement_floor"]),
    )
    np.savez_compressed(directory/"arrays.npz", times=times, teacher=target,
                        teacher_raw=raw_target, teacher_refined=refined_target,
                        train_prediction=train_prediction, heldout_coarse=held_coarse,
                        heldout_fine=held_fine, fitted_parameter_teacher=held_reference,
                        fitted_parameter_teacher_raw=held_reference_raw, sensitivity=sensitivity,
                        fitted=fitted, gradient=gradient, gradient_fine=gradient_fine,
                        finite_differences=finite, independent_gradient=independent_gradient,
                        scales=scales)
    report = dict(method=method, checks=checks, passed=all(checks.values()),
                  finiteness=finiteness,
                  optimizer=optimizer_record,
                  fitted_parameters=fitted, relative_parameter_error=relative_parameter_error,
                  teacher=teacher_info, teacher_refined=refined_info,
                  teacher_self_refinement_max_error=teacher_refinement_error,
                  fitted_parameter_teacher=fitted_teacher_info,
                  gradient_checks=dict(log_parameter_steps=[1e-4, 5e-5],
                                       independent_log_parameter_step=2e-5,
                                       discrete_max_error=discrete_gradient_error,
                                       coarse_independent_max_error=coarse_gradient_error,
                                       fine_independent_max_error=fine_gradient_error),
                  sensitivity_singular_values=singular,
                  sensitivity_condition_number=condition_number,
                  train_scaled_rmse=train_rmse,
                  heldout_scaled_rmse=holdout_rmse,
                  heldout_fine_scaled_rmse=holdout_fine_rmse,
                  integrator_error_at_same_fitted_parameters=dict(coarse=coarse_error, fine=fine_error),
                  arrays_sha256=sha((directory/"arrays.npz").read_bytes()))
    save_json(directory/"report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not jax.config.x64_enabled:
        parser.error("set JAX_ENABLE_X64=1 before starting this benchmark")
    args.output.mkdir(parents=True, exist_ok=False)
    package = Path(pyeph.__file__).resolve().parent
    sources = {"pyeph/"+p.relative_to(package).as_posix(): p for p in package.rglob("*.py")}
    sources["benchmarks/dynamics_calibration.py"] = Path(__file__).resolve()
    content = {name: path.read_bytes() for name, path in sources.items()}
    with zipfile.ZipFile(args.output/"sources.zip", "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, payload in content.items():
            archive.writestr(name, payload)
    np.savez(args.output/"inputs.npz", q0=Q0, p0=P0, electronic0=C0,
             train_indices=TRAIN, holdout_indices=HOLDOUT, truth=TRUTH, initial=INITIAL,
             parameter_bounds=np.exp(BOUNDS))
    report = dict(scope="Generated smooth-model dynamics calibration; no material or novel-method claim",
                  conventions=dict(basis="fixed orthonormal two-state effective carrier basis",
                                   units="hartree and atomic time; hbar=1; q is a canonical normal coordinate with chosen unit mass, g is energy per canonical length",
                                   coordinate_scope="Generated canonical mode, not an atomic-geometry label contract",
                                   carrier="one normalized effective electronic amplitude; no charge-sign conversion",
                                   hamiltonian="(bias+g*q)*sigma_z+delta*sigma_x; identity shift fixed to zero",
                                   reference="V0=omega^2*q^2/2; unit canonical nuclear mass",
                                   bias=BIAS, omega=OMEGA, fitted=["positive g", "positive delta"]),
                  identification_limits="Known basis, positive coupling convention, bias, mass and reference potential. Singular values check only local identification for this noiseless finite dataset; no global uniqueness claim.",
                  protocol=dict(dt=DT, steps=STEPS, observation_stride=STRIDE, gates=GATES,
                                optimizer=OPTIMIZER, teacher="Independent DOP853 explicit equations",
                                split="Three fixed training initial conditions; two held out; no fitting-setting selection on holdouts",
                                finer_audit="Same fitted parameters at half timestep; no refitting. Numerical refinement uses a SciPy oracle at those fitted parameters, separately from truth-target prediction error."),
                  created_utc=datetime.now(timezone.utc).isoformat(),
                  versions=dict(python=platform.python_version(), jax=jax.__version__,
                                numpy=np.__version__, scipy=scipy.__version__),
                  platform=platform.platform(), devices=[str(x) for x in jax.devices()],
                  source_sha256={name:sha(payload) for name, payload in content.items()}, cases={})
    save_json(args.output/"protocol.json", report)
    for method in ("cpa", "ehrenfest"):
        try:
            report["cases"][method] = run_case(method, args.output/method)
        except Exception:
            failure = dict(passed=False, traceback=traceback.format_exc())
            report["cases"][method] = failure
            save_json(args.output/(method+"-failure.json"), failure)
    report["sources_unchanged"] = all(path.read_bytes() == content[name] for name, path in sources.items())
    report["passed"] = report["sources_unchanged"] and all(case["passed"] for case in report["cases"].values())
    report["inputs_sha256"] = sha((args.output/"inputs.npz").read_bytes())
    report["source_archive_sha256"] = sha((args.output/"sources.zip").read_bytes())
    save_json(args.output/"report.json", report)
    print(json.dumps(plain(dict(passed=report["passed"], cases={name: {
        key:case.get(key) for key in ("passed", "checks", "fitted_parameters", "heldout_scaled_rmse", "traceback")}
        for name, case in report["cases"].items()})), indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
