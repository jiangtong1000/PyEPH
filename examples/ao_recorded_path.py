"""Recorded AO frames against independent fixed-basis electronic propagation.

This synthetic example supplies the same-time and cross-time AO metrics
explicitly. No quantum-chemistry program, fitted material or external weights
are needed. Atomic-unit data are exported to eV/fs before ingestion. Full and
truncated electronic spaces illustrate the difference between numerical
transport loss and physical band completeness. Output evidence is never
overwritten. Run after installing PyEPH: python examples/ao_recorded_path.py
"""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import tempfile
import time
import zipfile

import jax
import numpy as np
import scipy
from scipy.integrate import solve_ivp

import pyeph
from pyeph.core.units import ATOMIC_TIME_FS, HARTREE_EV
from pyeph.dynamics.recorded import RecordedCPA


def physical_hamiltonian(time):
    """Explicit Hermitian three-state Hamiltonian in one fixed physical basis."""
    a = .14 * np.exp(-.04*(time-3.)**2) * np.exp(.23j*time)
    b = .10 * np.sin(.63*time) + .045j*np.cos(.41*time)
    c = .065 * np.cos(.37*time) * np.exp(-.31j*time)
    return np.array([
        [.16*np.tanh(.6*(time-3.)), a, b],
        [a.conjugate(), -.11*np.tanh(.5*(time-4.)), c],
        [b.conjugate(), c.conjugate(), .34+.055*np.cos(.47*time)],
    ], dtype=np.complex128)


def ao_embedding(time):
    """Nonsingular nonunitary columns representing a changing AO basis."""
    return np.array([
        [1.1+.07*np.sin(time), .12*np.exp(.17j*time), .035j],
        [.04j*np.cos(.3*time), .95+.04*np.cos(.4*time), .09*np.exp(-.21j*time)],
        [.06*np.sin(.2*time), -.025j, 1.2+.06*np.sin(.5*time)],
    ], dtype=np.complex128)


def exported_frames(times):
    energy, physical_vectors = np.linalg.eigh(np.stack([physical_hamiltonian(t) for t in times]))
    ao = np.stack([ao_embedding(t) for t in times])
    # Coefficients are rows of kets: |phi_a> = sum_mu |chi_mu> C[a,mu].
    coefficients = np.swapaxes(np.linalg.solve(ao, physical_vectors), -1, -2)
    adjoint = ao.conj().swapaxes(-1, -2)
    return dict(times=times*ATOMIC_TIME_FS, energies=energy*HARTREE_EV,
                coefficients=coefficients, metrics=adjoint@ao,
                cross_metrics=adjoint[:-1]@ao[1:]), physical_vectors


def input_digest(data):
    digest = hashlib.sha256()
    for key, value in sorted(data.items()):
        array = np.ascontiguousarray(value)
        digest.update(key.encode()+b"\0"+str(array.dtype).encode()+b"\0")
        digest.update(str(array.shape).encode()+b"\0"+array.tobytes())
    return digest.hexdigest()


def source_hashes():
    root = Path(pyeph.__file__).resolve().parent
    files = {f"src/pyeph/{p.relative_to(root)}": p for p in root.rglob("*.py")}
    files["examples/ao_recorded_path.py"] = Path(__file__).resolve()
    return {name: hashlib.sha256(path.read_bytes()).hexdigest()
            for name, path in sorted(files.items())}, files


def oracle(initial, times, *, tighter=False):
    solution = solve_ivp(lambda t, c: -1j*physical_hamiltonian(t)@c,
                         (float(times[0]), float(times[-1])), initial,
                         method="DOP853", t_eval=times,
                         rtol=3e-14 if tighter else 2e-12,
                         atol=2e-15 if tighter else 2e-14,
                         max_step=.025 if tighter else .05)
    if not solution.success:
        raise RuntimeError(solution.message)
    return solution.y.T


def run_case(dt, retained, arrays):
    from pyeph.adapters.ao_frames import project_ao_path

    times = np.linspace(0., 8., round(8./dt)+1)
    inputs, vectors = exported_frames(times)
    projected = project_ao_path(
        **inputs, retained_bands=tuple(range(retained)),
        energy_unit="eV", time_unit="fs", ao_basis_id="synthetic-three-AO-order",
        basis_id=f"synthetic-lowest-{retained}-states",
        source_identity="sha256:"+input_digest(inputs), transport_mode="raw")
    coefficients0 = np.array([np.sqrt(.6), np.sqrt(.4)*np.exp(.4j), 0.], dtype=np.complex128)
    initial_physical = vectors[0]@coefficients0
    reference = oracle(initial_physical, times)
    reference_tight = oracle(initial_physical, times, tighter=True)
    # No loss cap for the truncation demonstration; its loss is reported below.
    runner = RecordedCPA(projected.path, execution=pyeph.Execution(chunk_size=32),
                         max_subspace_loss=None if retained < 3 else 1e-10)
    initial = runner.initialize(coefficients0[:retained])
    result = runner.run(initial, len(times)-1)
    physical_final = vectors[-1, :, :retained]@np.asarray(result.final_state.electronic)
    reference_final_in_frame = vectors[-1].conj().T@reference[-1]
    overlaps = np.asarray(projected.path.overlaps)
    singular = np.linalg.svd(overlaps, compute_uv=False)
    loss = np.maximum(0., 1.-singular[:, -1]**2)
    checkpoint_index = (len(times)-1)//2
    prefix = runner.run(initial, checkpoint_index, collect=False)
    with tempfile.TemporaryDirectory(prefix="pyeph-ao-example-") as folder:
        file = Path(folder)/"recorded.h5"
        runner.save_checkpoint(file, prefix.final_state)
        restored = runner.load_checkpoint(file)
        suffix = runner.run(restored, len(times)-1-checkpoint_index)
    restart_error = float(np.max(np.abs(suffix.final_state.electronic-result.final_state.electronic)))
    if restart_error != 0.:
        raise AssertionError("aligned recorded checkpoint continuation changed amplitudes")
    label = f"states{retained}_steps{len(times)-1}"
    for key, value in inputs.items():
        arrays[f"{label}_input_{key}"] = value
    arrays.update({f"{label}_physical_vectors": vectors,
                   f"{label}_oracle": reference, f"{label}_oracle_tight": reference_tight,
                   f"{label}_final_coefficients": np.asarray(result.final_state.electronic),
                   f"{label}_physical_final": physical_final,
                   f"{label}_overlaps": overlaps, f"{label}_singular_values": singular,
                   f"{label}_times": result.times})
    arrays.update({f"{label}_observed_{key}": value for key, value in result.observables.items()})
    return dict(label=label, retained_states=retained, dt=dt, intervals=len(times)-1,
                maximum_ao_condition=float(max(np.linalg.cond(ao_embedding(t)) for t in times)),
                final_physical_amplitude_l2_error=float(np.linalg.norm(physical_final-reference[-1])),
                oracle_tightening_max_l2=float(np.max(np.linalg.norm(reference-reference_tight, axis=1))),
                final_norm=float(np.vdot(physical_final, physical_final).real),
                maximum_interval_raw_norm_loss=float(np.max(loss)),
                summed_interval_loss_bounds=float(np.sum(loss)),
                reference_final_discarded_population=float(np.sum(abs(reference_final_in_frame[retained:])**2)),
                restart_max_abs=restart_error, checkpoint_index=checkpoint_index,
                projection_diagnostics=asdict(projected.diagnostics),
                projection_evidence=asdict(projected.evidence))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/ao_recorded_path"),
                        help="JSON/NPZ/source ZIP output directory (default: %(default)s)")
    args = parser.parse_args()
    outputs = {kind: args.output_dir/f"ao_recorded_path.{kind}" for kind in ("json", "npz", "zip")}
    if any(p.exists() for p in outputs.values()):
        parser.error("evidence already exists; choose a fresh --output-dir")
    pyeph.configure_precision(True)
    started, clock = datetime.now(timezone.utc).isoformat(), time.perf_counter()
    source, source_files = source_hashes()
    arrays = {}
    cases = [run_case(dt, retained, arrays) for retained in (3, 2) for dt in (.08, .04, .02)]
    if source != source_hashes()[0]:
        raise RuntimeError("runtime/example changed during validation; results not archived")
    if any(c["oracle_tightening_max_l2"] > 1e-11 for c in cases):
        raise AssertionError("independent oracle did not converge tightly")
    full, truncated = cases[:3], cases[3:]
    ratios = [full[k]["final_physical_amplitude_l2_error"]
              /full[k+1]["final_physical_amplitude_l2_error"] for k in range(2)]
    if not all(3.8 < ratio < 4.2 for ratio in ratios):
        raise AssertionError("full-space interval propagation did not converge at second order")
    if any(abs(c["final_norm"]-1.) > 1e-12 for c in full):
        raise AssertionError("complete-space transport lost norm")
    if not (truncated[-1]["final_physical_amplitude_l2_error"] > .5
            and truncated[-1]["reference_final_discarded_population"] > .3
            and truncated[-1]["maximum_interval_raw_norm_loss"]
            < truncated[0]["maximum_interval_raw_norm_loss"] / 10):
        raise AssertionError("truncation example no longer exhibits its stated limitation")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with outputs["npz"].open("xb") as f:
        np.savez_compressed(f, **arrays)
    with zipfile.ZipFile(outputs["zip"], "x", zipfile.ZIP_DEFLATED) as z:
        for name, path in source_files.items():
            z.write(path, name)
    record = dict(started_utc=started, finished_utc=datetime.now(timezone.utc).isoformat(),
                  scope="Synthetic recorded AO basis transport and band-truncation demonstration; electronic CPA only",
                  physical_units="atomic", input_units={"energy": "eV", "time": "fs"},
                  software={"python": platform.python_version(), "numpy": np.__version__,
                            "scipy": scipy.__version__, "jax": jax.__version__,
                            "backend": jax.default_backend()},
                  cases=cases, full_space_error_refinement_ratios=ratios,
                  source_start=source, source_unchanged=True,
                  elapsed_seconds_including_compilation=time.perf_counter()-clock,
                  timing_scope="shared-host example run; not a performance benchmark",
                  artifacts={str(outputs[k]): hashlib.sha256(outputs[k].read_bytes()).hexdigest()
                             for k in ("npz", "zip")},
                  limits=["No real electronic-structure dataset or trained material model was used.",
                          "Overlap contraction and orthonormality cannot certify discarded-band physics.",
                          "A small adjacent-frame loss does not bound full-space propagation error.",
                          "No nuclear forces, spatial derivative couplings or MASH rescaling are inferred."])
    with outputs["json"].open("x") as f:
        f.write(json.dumps(record, indent=2, allow_nan=False)+"\n")
    print(json.dumps([{k: c[k] for k in ("retained_states", "dt", "final_physical_amplitude_l2_error",
                                        "final_norm", "maximum_interval_raw_norm_loss",
                                        "reference_final_discarded_population")} for c in cases], indent=2))
    print(f"Saved {outputs['json']}")


if __name__ == "__main__":
    main()
