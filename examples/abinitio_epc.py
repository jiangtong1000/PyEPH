"""Cartesian ab initio EPC to periodic CPA current correlations.

This composes the general EPR/data/model/FFT-bath boundaries. Explicit source
and phonon policies are saved alongside streamed correlations and a restart.
See docs/AB_INITIO_EPC.md. No finite-window correlation is labeled a mobility.
"""

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time

import jax
import numpy as np

from pyeph import Execution, Integrator, Simulation, stack_states
from pyeph.adapters.epr import read_epr
from pyeph.core.units import HARTREE_EV
from pyeph.io.hdf5 import HDF5Observer
from pyeph.io.provenance import problem_manifest
from pyeph.observables.statistics import EnsembleMoments
from pyeph.paths.periodic_harmonic import PeriodicHarmonicBath, sample_periodic_harmonic
from pyeph.workflows.transport import initialize_transport_state, make_transport_problem


@dataclass(frozen=True)
class Inputs:
    mesh: tuple = (1, 1, 1)
    carrier: str = "hole"
    polar: str = "error"
    hermiticity: str = "require"
    freeze_below_mev: float | None = None
    zero_tolerance: float = 0.
    free_positions_zero: bool = False
    temperature_kelvin: float = 300.
    distribution: str = "classical"
    trajectories: int = 4
    seed: int = 2026
    first_id: int = 0
    dt_fs: float = .01
    steps: int = 20
    chunk_size: int = 32
    save_every: int = 1
    term_batch_size: int = 256
    epc_backend: str = "direct"
    max_spectral_bytes: int = 256*1024**2

    def validate(self):
        for name, minimum in (("trajectories", 1), ("seed", 0), ("first_id", 0),
                              ("steps", 0), ("chunk_size", 1), ("save_every", 1),
                              ("term_batch_size", 1), ("max_spectral_bytes", 1)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if self.epc_backend not in ("direct", "fft"):
            raise ValueError("epc_backend must be 'direct' or 'fft'")
        if self.seed >= 2**32 or self.first_id+self.trajectories > 2**32:
            raise ValueError("seed and trajectory identities must fit uint32")
        for name in ("temperature_kelvin", "dt_fs"):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        return self


def build_problem(epr, inputs):
    """Read a source, select explicit interpolation/phonon policies, build CPA."""
    inputs.validate()
    source = read_epr(epr, polar=inputs.polar)
    compiled = source.compile_supercell(inputs.mesh, carrier=inputs.carrier,
        hermiticity=inputs.hermiticity, term_batch_size=inputs.term_batch_size,
        epc_backend=inputs.epc_backend, max_spectral_bytes=inputs.max_spectral_bytes)
    data, units = source.data, source.data.unit_system
    energy_mev = units.energy_hartree*HARTREE_EV*1000
    threshold = None if inputs.freeze_below_mev is None else inputs.freeze_below_mev/energy_mev
    bath = PeriodicHarmonicBath(inputs.mesh, data.masses, data.ifc_atoms,
        data.ifc_cells, data.ifc_values, frozen_below=threshold,
        zero_tolerance=inputs.zero_tolerance)
    problem = make_transport_problem(compiled.model, compiled.params, bath,
                                     probes=("current_x", "current_y", "current_z"))
    spectrum = np.asarray(bath.squared_frequencies)
    temperature = units.temperature_from_kelvin(inputs.temperature_kelvin)
    metadata = {"ingestion": compiled.report,
        "energy_unit_mev": energy_mev, "length_unit_bohr": units.length_bohr,
        "time_unit_fs": units.time_fs, "temperature_reduced": temperature,
        "beta_reduced": 1/temperature,
        "current_convention": "charge times fixed-Wannier-center Peierls velocity",
        "correlation": "single-carrier unsymmetrized physical-current autocorrelation",
        "nuclear_ensemble": "neutral harmonic ensemble; no carrier-dependent reweighting",
        "coordinate_order": "cell-major (z fastest), then atom and Cartesian xyz",
        "phonons": {"squared_frequency_zero_tolerance": inputs.zero_tolerance,
            "freeze_below_mev": inputs.freeze_below_mev,
            "minimum_signed_squared_frequency": float(spectrum.min()),
            "physical_unstable_count": int(np.sum(spectrum < -inputs.zero_tolerance)),
            "numerical_zero_count": int(np.sum(abs(spectrum) <= inputs.zero_tolerance)),
            "frozen_count": int(np.sum(bath.frozen_modes)),
            "active_free_count": int(np.sum((np.asarray(bath.frequencies) == 0)
                                            & ~np.asarray(bath.frozen_modes))),
            "source_ifcs_modified": False},
        "electronic_propagator": "full U, quadratic storage; suitable for modest state counts"}
    return problem, metadata


def prepare(problem, metadata, inputs):
    """Stable trajectory IDs and physical Cartesian thermal samples."""
    ids = np.arange(inputs.first_id, inputs.first_id+inputs.trajectories, dtype=np.uint32)
    free = np.zeros(problem.model.spec.system.q_shape) if inputs.free_positions_zero else None
    q, p = sample_periodic_harmonic(problem.nuclear_treatment, metadata["temperature_reduced"],
        ids, seed=inputs.seed, distribution=inputs.distribution, free_positions=free)
    states = [initialize_transport_state(problem, q[i], p[i], metadata["beta_reduced"],
                                         trajectory_id=int(identity), seed=inputs.seed)
              for i, identity in enumerate(ids)]
    return stack_states(states)


def run_workflow(epr, inputs, output, *, resume=None):
    if not jax.config.x64_enabled:
        raise ValueError("set JAX_ENABLE_X64=1 for this ab initio reference workflow")
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    start = time.perf_counter()
    problem, metadata = build_problem(epr, inputs)
    integrator = Integrator(problem.model.spec.unit_system.time_from_fs(inputs.dt_fs))
    simulation = Simulation(problem, integrator, Execution(chunk_size=inputs.chunk_size,
                                                           save_every=inputs.save_every))
    manifest = problem_manifest(problem, integrator)
    if resume is None:
        initial = prepare(problem, metadata, inputs)
    else:
        resume = Path(resume)
        previous = json.loads((resume.parent/"run.json").read_text())
        current_inputs = json.loads(json.dumps(asdict(inputs)))
        mutable = {"steps", "chunk_size", "save_every"}
        if any(current_inputs[name] != value for name, value in previous["inputs"].items()
               if name not in mutable):
            raise ValueError("resume must preserve all inputs except steps, chunk_size and save_every")
        old_source = previous["conventions"]["ingestion"]["source"]["source_sha256"]
        if old_source != metadata["ingestion"]["source"]["source_sha256"]:
            raise ValueError("resume EPR source has changed")
        initial = simulation.load_checkpoint(resume)
    expected_ids = np.arange(inputs.first_id, inputs.first_id+inputs.trajectories, dtype=np.uint32)
    if not np.array_equal(initial.trajectory_id, expected_ids):
        raise ValueError("restart trajectory IDs differ from the requested ensemble")
    metadata = {**metadata, "fingerprint": manifest["fingerprint"], "inputs": asdict(inputs),
                "trajectory_ids": expected_ids.tolist()}
    preparation_seconds = time.perf_counter()-start
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output/"segment_start_state.npz", q=np.asarray(initial.q), p=np.asarray(initial.p),
                        electronic=np.asarray(initial.electronic), time=np.asarray(initial.time),
                        trajectory_ids=np.asarray(initial.trajectory_id))
    maximum_unitarity = 0.
    start = time.perf_counter()
    with HDF5Observer(output/"trajectory.h5", metadata=metadata) as raw, \
            HDF5Observer(output/"ensemble.h5", metadata=metadata) as ensemble:
        def observe(times, values):
            nonlocal maximum_unitarity
            raw(times, values)
            correlation = values["current_correlation"]
            real = EnsembleMoments().update(correlation.real, axis=1)
            imag = EnsembleMoments().update(correlation.imag, axis=1)
            ensemble(times[:, 0], {"mean_current_correlation": real.mean+1j*imag.mean,
                "sem_real": real.standard_error, "sem_imag": imag.standard_error})
            maximum_unitarity = max(maximum_unitarity, float(np.max(values["unitary_error"])))
        result = simulation.run(initial, inputs.steps, observer=observe, collect=False)
    run_seconds = time.perf_counter()-start
    if problem_manifest(problem, integrator) != manifest:
        raise RuntimeError("runtime source identity changed during this calculation")
    simulation.save_checkpoint(output/"checkpoint.h5", result.final_state)
    report = {"inputs": asdict(inputs), "conventions": metadata, "manifest": manifest,
        "preparation_seconds": preparation_seconds, "cold_run_seconds": run_seconds,
        "final_time_fs": float(np.max(result.final_state.time))*metadata["time_unit_fs"],
        "maximum_unitarity_error": maximum_unitarity,
        "scope": "finite periodic linear Cartesian EPC, prescribed neutral harmonic CPA; "
                 "no long-range polar contribution or converged materials mobility established"}
    (output/"run.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epr", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--mesh", type=int, nargs=3, default=(1, 1, 1))
    parser.add_argument("--carrier", choices=("electron", "hole"), default="hole")
    parser.add_argument("--polar", choices=("error", "short_range"), default="error")
    parser.add_argument("--hermiticity", choices=("require", "project"), default="require")
    parser.add_argument("--epc-backend", choices=("direct", "fft"), default="direct")
    parser.add_argument("--freeze-below-mev", type=float)
    parser.add_argument("--zero-tolerance", type=float, default=0.)
    parser.add_argument("--free-positions-zero", action="store_true")
    parser.add_argument("--distribution", choices=("classical", "wigner"), default="classical")
    for name, default in (("temperature_kelvin", 300.), ("dt_fs", .01)):
        parser.add_argument("--"+name.replace("_", "-"), type=float, default=default)
    for name, default in (("trajectories", 4), ("seed", 2026), ("first_id", 0),
                          ("steps", 20), ("chunk_size", 32), ("save_every", 1),
                          ("term_batch_size", 256), ("max_spectral_bytes", 256*1024**2)):
        parser.add_argument("--"+name.replace("_", "-"), type=int, default=default)
    args = vars(parser.parse_args())
    epr, output, resume = (args.pop(name) for name in ("epr", "output", "resume"))
    report = run_workflow(epr, Inputs(**args), output, resume=resume)
    print(json.dumps({key: report[key] for key in ("final_time_fs", "maximum_unitarity_error",
                                                "preparation_seconds", "cold_run_seconds", "scope")},
                     indent=2))


if __name__ == "__main__":
    main()
