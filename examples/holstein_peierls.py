"""Native Holstein--Peierls CPA transport, including local LF approximations.

Run from the checkout with PYTHONPATH=src and JAX_ENABLE_X64=true. See
docs/HOLSTEIN_PEIERLS_WORKFLOW.md for equations, units, restart, and convergence.
This example composes native models/workflows; it does not use the legacy facade.
"""

import argparse
from dataclasses import asdict, dataclass, replace
import hashlib
import json
from pathlib import Path

import h5py
import jax
import jax.numpy as jnp
import numpy as np
from scipy.constants import elementary_charge, hbar
from scipy.integrate import cumulative_trapezoid

from pyeph import Execution, Integrator, Simulation, stack_states
from pyeph.core.problem import Problem
from pyeph.core.units import UnitSystem
from pyeph.initialization import sample_harmonic
from pyeph.io.hdf5 import HDF5Observer
from pyeph.io.provenance import problem_manifest
from pyeph.models.base import AutoDiffModel
from pyeph.models.epc import EdgeEPCModel
from pyeph.models.polaron import lf_band_narrowing
from pyeph.observables.statistics import EnsembleMoments
from pyeph.paths.harmonic import HarmonicBath
from pyeph.workflows.polaron_transport import make_polaron_transport_problem
from pyeph.workflows.transport import (
    TransportMeasurement, initialize_transport_state, make_transport_problem,
)


@dataclass(frozen=True)
class Inputs:
    """Physical inputs and execution settings for a one-orbital periodic ring."""

    nsites: int = 8
    trajectories: int = 8
    seed: int = 1120
    first_id: int = 0
    temperature_kelvin: float = 183.84761032548917
    hopping_mev: float = 100.0
    holstein_frequency_mev: float = 50.0
    reorganization_mev: float = 100.0
    peierls_frequency_mev: float = 6.2
    peierls_fraction: float = 0.0
    mode: str = "cpa"
    thermal_policy: str = "offdiagonal"
    spacing_angstrom: float | None = None
    dt: float = 0.02
    steps: int = 200
    chunk_size: int = 64

    def validate(self):
        for name, minimum in (("nsites", 3), ("trajectories", 1), ("seed", 0),
                              ("first_id", 0), ("steps", 0), ("chunk_size", 1)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if self.seed >= 2**32 or self.first_id + self.trajectories > 2**32:
            raise ValueError("seed and trajectory IDs must fit uint32")
        for name in ("temperature_kelvin", "hopping_mev", "holstein_frequency_mev",
                     "peierls_frequency_mev", "dt"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("reorganization_mev", "peierls_fraction"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.spacing_angstrom is not None and (
                not np.isfinite(self.spacing_angstrom) or self.spacing_angstrom <= 0):
            raise ValueError("spacing_angstrom must be finite and positive when supplied")
        if self.mode not in {"cpa", "lf", "band"}:
            raise ValueError("mode must be cpa, lf, or band")
        if self.thermal_policy not in {"offdiagonal", "legacy_full"}:
            raise ValueError("thermal_policy must be offdiagonal or legacy_full")
        return self


@dataclass(frozen=True)
class RingEPCModel(AutoDiffModel):
    """An EdgeEPCModel with a periodic bond velocity and declared reduced units.

    The wrap edge (0,N-1) has displacement -1. Its velocity is obtained from
    the bond Peierls phase, not a discontinuous position operator on the ring.
    velocity_scale is the narrowed-current approximation used only in 'band'.
    """

    base_model: EdgeEPCModel
    unit_system: UnitSystem
    velocity_scale: float = 1.0

    @property
    def spec(self):
        return replace(self.base_model.spec, name="holstein_peierls_ring",
                       probes=("velocity_x",), unit_system=self.unit_system)

    def validate_params(self, params):
        self.base_model.validate_params(params)

    def apply(self, params, q, vectors):
        return self.base_model.apply(params, q, vectors)

    def prepare_action(self, params, q):
        return self.base_model.prepare_action(params, q)

    def diagonal(self, params, q):
        return self.base_model.diagonal(params, q)

    def reference_energy(self, params, q):
        return self.base_model.reference_energy(params, q)

    def probe_apply(self, params, context, probe, vectors):
        if probe != "velocity_x":
            raise ValueError("this ring defines only velocity_x")
        _, hopping = self.base_model.elements(params, context.q)
        edges = jnp.asarray(self.base_model.edges)
        i, j = edges[:, 0], edges[:, 1]
        displacement = jnp.where((i == 0) & (j == self.nstates - 1), -1.0, 1.0)
        velocity = self.velocity_scale * 1j * displacement * hopping
        columns = vectors[:, None] if vectors.ndim == 1 else vectors
        out = jnp.zeros(columns.shape, dtype=jnp.result_type(columns, velocity))
        out = out.at[i].add(velocity[:, None] * columns[j])
        out = out.at[j].add(velocity.conj()[:, None] * columns[i])
        return out[:, 0] if vectors.ndim == 1 else out


@dataclass(frozen=True)
class BandMeasurement(TransportMeasurement):
    """Narrowed-band correlation without the fluctuating quantum-bath factor."""

    thermal_policy: str = "offdiagonal"

    def initial_hamiltonian(self, problem, state):
        if self.thermal_policy == "legacy_full":
            return problem.model.factor * problem.model.bare_hamiltonian(problem.params, state.q)
        return super().initial_hamiltonian(problem, state)


def build_problem(inputs):
    """Construct H(Q), its ring velocity, canonical bath, and CPA measurement."""
    inputs.validate()
    n, energy = inputs.nsites, inputs.hopping_mev
    # Length is one lattice spacing. When unknown, the nominal Angstrom unit
    # is only bookkeeping; the output forbids a physical length conversion.
    units = UnitSystem.from_ev_angstrom(energy / 1000, inputs.spacing_angstrom or 1.0)
    temperature = units.temperature_from_kelvin(inputs.temperature_kelvin)
    wh, wp = inputs.holstein_frequency_mev / energy, inputs.peierls_frequency_mev / energy
    gh = np.sqrt(inputs.reorganization_mev * inputs.holstein_frequency_mev) / energy
    # Preserve the historical Wigner-calibrated dJ convention explicitly.
    gp = inputs.peierls_fraction * np.sqrt(np.tanh(wp / (2 * temperature)))
    classical_h = inputs.mode == "cpa"
    frequencies = np.repeat([wh, wp] if classical_h else [wp], n)
    edges = tuple((i, i + 1) for i in range(n - 1)) + ((0, n - 1),)
    base = EdgeEPCModel(n, len(frequencies), edges)
    params = base.default_params()
    onsite = np.zeros((len(frequencies), n))
    hopping = np.zeros((len(frequencies), n))
    if classical_h:
        onsite[:n] = np.eye(n) * gh * np.sqrt(2 * wh)
    # Each directed forward bond i -> i+1 has its independent mode at i.
    hopping[-n:] = np.eye(n) * gp * np.sqrt(2 * wp)
    params.update(hopping=-jnp.ones(n), onsite_coupling=jnp.asarray(onsite),
                  hopping_coupling=jnp.asarray(hopping), omega=jnp.asarray(frequencies))
    beta = 1 / temperature
    factor = 1.0 if classical_h else float(lf_band_narrowing([wh], [gh], beta))
    model = RingEPCModel(base, units, factor if inputs.mode == "band" else 1.0)
    bath = HarmonicBath(frequencies)
    if classical_h:
        problem = make_transport_problem(model, params, bath, probes=("velocity_x",))
    else:
        directed = tuple(edges) + tuple((j, i) for i, j in edges)
        problem = make_polaron_transport_problem(
            model, params, bath, [wh], [gh], beta, hopping_pairs=directed,
            thermal_policy=inputs.thermal_policy, probes=("velocity_x",))
        if inputs.mode == "band":
            problem = Problem(problem.model, params, bath, problem.method,
                              BandMeasurement(probes=("velocity_x",),
                                              thermal_policy=inputs.thermal_policy)).validate()
    metadata = {
        "energy_unit_mev": energy, "time_unit_fs": units.time_fs,
        "length_unit": "one lattice spacing", "spacing_angstrom": inputs.spacing_angstrom,
        "temperature_reduced": temperature, "beta_reduced": beta,
        "classical_mode_order": ["Holstein", "bond Peierls"] if classical_h else ["bond Peierls"],
        "coordinate_order": "mode-major then site; canonical unit-mass Q,P",
        "holstein_coupling_reduced": gh, "peierls_coupling_reduced": gp,
        "classical_hopping_rms_reduced": gp * np.sqrt(2 * temperature / wp),
        "lf_narrowing_factor": factor,
        "correlation": "single-carrier unsymmetrized charge-free velocity autocorrelation",
        "initial_nuclear_ensemble": "independent classical harmonic oscillators, no carrier reweighting",
        "physical_length_known": inputs.spacing_angstrom is not None,
    }
    return problem, metadata


def artifact_ids(inputs):
    identity = "sha256:" + hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    result = {"model" if inputs.mode == "cpa" else "model.base_model": identity}
    if inputs.mode == "band":
        result["measurement"] = identity
    return result


def prepare(inputs, problem, metadata, samples=None):
    """Use stable trajectory identities, or replay actual saved Q/P exactly."""
    ids = np.arange(inputs.first_id, inputs.first_id + inputs.trajectories, dtype=np.uint32)
    if samples is None:
        q, p = sample_harmonic(problem.params["omega"], 1.0,
                               metadata["temperature_reduced"], ids, seed=inputs.seed)
    else:
        with np.load(samples, allow_pickle=False) as archive:
            q, p, saved_ids = archive["q0"], archive["p0"], archive["trajectory_ids"]
            if not np.array_equal(ids, saved_ids):
                raise ValueError("saved sample trajectory IDs differ from requested identities")
            if not np.array_equal(np.asarray(problem.params["omega"]), archive["frequencies"]):
                raise ValueError("saved sample frequencies differ from the current canonical bath")
            if float(archive["temperature_reduced"]) != metadata["temperature_reduced"]:
                raise ValueError("saved samples have a different nuclear temperature")
    expected = (inputs.trajectories, *problem.model.spec.system.q_shape)
    if np.shape(q) != expected or np.shape(p) != expected:
        raise ValueError(f"saved Q/P require shape {expected}")
    states = [initialize_transport_state(problem, q[i], p[i], metadata["beta_reduced"],
                                         trajectory_id=int(identity), seed=inputs.seed)
              for i, identity in enumerate(ids)]
    return stack_states(states), np.asarray(q), np.asarray(p), ids


def analyze_streams(paths):
    """Join contiguous restart segments and integrate each independent trajectory.

    Integrating first retains the time covariance in the trajectory SEM. This
    uncertainty covers nuclear Monte Carlo sampling, not finite-time/model error.
    """
    times, correlations, errors, identity = [], [], [], None
    for path in paths:
        with h5py.File(path) as handle:
            metadata = json.loads(handle.attrs["metadata"])
            current_identity = (metadata["fingerprint"], metadata["trajectory_ids"])
            if identity is not None and current_identity != identity:
                raise ValueError("restart segments have different physical identities or trajectories")
            identity = current_identity
            t = handle["time"][...]
            c = handle["observables/current_correlation"][..., 0]
            e = handle["observables/unitary_error"][...]
        if t.ndim != 2 or not np.allclose(t, t[:, :1], rtol=0, atol=1e-12):
            raise ValueError("analysis requires synchronized trajectory times")
        t = t[:, 0]
        if times:
            if not np.isclose(t[0], times[-1][-1], rtol=0, atol=1e-12):
                raise ValueError("restart segments must share exactly one common boundary")
            np.testing.assert_allclose(c[0], correlations[-1][-1], atol=1e-12, rtol=1e-12)
            t, c, e = t[1:], c[1:], e[1:]
        if len(t):
            times.append(t)
            correlations.append(c)
            errors.append(e)
    time = np.concatenate(times)
    corr = np.concatenate(correlations)
    if time[0] != 0 or (len(time) > 1 and not np.all(np.diff(time) > 0)):
        raise ValueError("complete transport analysis requires a strictly increasing grid from time zero")
    integrals = cumulative_trapezoid(corr.real, time, axis=0, initial=0)
    corr_real = EnsembleMoments().update(corr.real, axis=1)
    corr_imag = EnsembleMoments().update(corr.imag, axis=1)
    integrated = EnsembleMoments().update(integrals, axis=1)
    arrays = {"time_reduced": time, "mean_correlation": corr_real.mean + 1j * corr_imag.mean,
              "sem_real_correlation": corr_real.standard_error,
              "sem_imag_correlation": corr_imag.standard_error,
              "mean_running_integral": integrated.mean,
              "sem_running_integral": integrated.standard_error}
    return arrays, float(np.max(np.concatenate(errors)))


def run_workflow(inputs, output, *, samples=None, resume=None):
    """Stream a fresh segment, save a checkpoint, and analyze its complete history."""
    if not jax.config.x64_enabled:
        raise ValueError("set JAX_ENABLE_X64=true before running this reference workflow")
    inputs.validate()
    output = Path(output).resolve()
    problem, metadata = build_problem(inputs)
    integrator = Integrator(inputs.dt)
    simulation = Simulation(problem, integrator, Execution(chunk_size=inputs.chunk_size))
    identities = artifact_ids(inputs)
    manifest = problem_manifest(problem, integrator, artifact_ids=identities)
    segments = []
    if resume is None:
        initial, q0, p0, ids = prepare(inputs, problem, metadata, samples)
    else:
        if samples is not None:
            raise ValueError("resume uses its saved state; do not also supply initial samples")
        previous = json.loads((Path(resume) / "run.json").read_text())
        if any(asdict(inputs)[name] != value for name, value in previous["inputs"].items()
               if name not in {"steps", "chunk_size"}):
            raise ValueError("resume must preserve all inputs except steps and chunk_size")
        initial = simulation.load_checkpoint(Path(resume) / "checkpoint.h5", artifact_ids=identities)
        segments = previous["segments"]
        for path, expected in previous["data_sha256"].items():
            if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
                raise ValueError(f"saved workflow data changed: {path}")
        with np.load(previous["initial_samples"], allow_pickle=False) as archive:
            q0, p0, ids = archive["q0"], archive["p0"], archive["trajectory_ids"]
        if len(ids) != inputs.trajectories or not np.array_equal(ids, np.asarray(initial.trajectory_id)):
            raise ValueError("resume sample archive does not match checkpoint trajectories")
    output.mkdir(parents=True, exist_ok=False)
    sample_path = output / "initial_samples.npz"
    np.savez_compressed(sample_path, q0=q0, p0=p0, trajectory_ids=ids,
                        frequencies=np.asarray(problem.params["omega"]),
                        temperature_reduced=metadata["temperature_reduced"])
    stream_path = output / "trajectory.h5"
    stream_metadata = {**metadata, "fingerprint": manifest["fingerprint"],
                       "trajectory_ids": [int(x) for x in ids], "inputs": asdict(inputs)}
    with HDF5Observer(stream_path, metadata=stream_metadata) as observer:
        result = simulation.run(initial, inputs.steps, observer=observer, collect=False)
    if problem_manifest(problem, integrator, artifact_ids=artifact_ids(inputs)) != manifest:
        raise RuntimeError("source or runtime identity changed during the calculation")
    simulation.save_checkpoint(output / "checkpoint.h5", result.final_state, artifact_ids=identities)
    segments.append(str(stream_path))
    arrays, unitarity = analyze_streams(segments)
    arrays["time_fs"] = arrays["time_reduced"] * metadata["time_unit_fs"]
    # beta times the integral is reported as a reduced finite-window response.
    # Its DC interpretation still needs stationarity and a converged long-time limit.
    beta = metadata["beta_reduced"]
    arrays["mean_beta_integral"] = beta * arrays["mean_running_integral"]
    arrays["sem_beta_integral"] = beta * arrays["sem_running_integral"]
    if inputs.spacing_angstrom is not None:
        prefactor = elementary_charge * (inputs.spacing_angstrom * 1e-10)**2 / hbar * 1e4
        arrays["finite_window_mobility_proxy_cm2_per_Vs"] = arrays["mean_beta_integral"] * prefactor
        arrays["sem_mobility_proxy_cm2_per_Vs"] = arrays["sem_beta_integral"] * prefactor
    np.savez_compressed(output / "analysis.npz", **arrays)
    def scalar(x):
        return float(x) if np.isfinite(x) else None

    report = {"inputs": asdict(inputs), "units_and_conventions": metadata,
              "segments": segments, "initial_samples": str(sample_path),
              "data_sha256": {str(path): hashlib.sha256(Path(path).read_bytes()).hexdigest()
                              for path in [*segments, sample_path]},
              "manifest": manifest, "final_time_reduced": float(arrays["time_reduced"][-1]),
              "max_unitarity_error": unitarity,
              "final_integral": scalar(arrays["mean_running_integral"][-1]),
              "final_integral_nuclear_sampling_sem": scalar(arrays["sem_running_integral"][-1]),
              "scope": "finite periodic CPA/LF model; no diffusive plateau or material mobility established"}
    (output / "run.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", type=Path, help="prior output directory; preserve its inputs")
    parser.add_argument("--samples", type=Path, help="replay an initial_samples.npz archive")
    parser.add_argument("--original", action="store_true", help="42 sites, 50 trajectories, 4999 steps, dt=.01")
    for name, value in asdict(Inputs()).items():
        kind = type(value) if value is not None else float
        parser.add_argument("--" + name.replace("_", "-"), type=kind, default=None)
    args = parser.parse_args()
    values = asdict(Inputs())
    if args.original:
        values.update(nsites=42, trajectories=50, dt=.01, steps=4999)
    if args.resume:
        if args.original:
            parser.error("--original and --resume are separate entry points")
        values = json.loads((args.resume / "run.json").read_text())["inputs"]
        forbidden = [name for name in values if name not in {"steps", "chunk_size"}
                     and getattr(args, name) is not None]
        if forbidden:
            parser.error("resume may only override steps and chunk_size")
    values.update({name: getattr(args, name) for name in values if getattr(args, name) is not None})
    report = run_workflow(Inputs(**values), args.output, samples=args.samples, resume=args.resume)
    print(json.dumps({key: report[key] for key in ("final_time_reduced", "max_unitarity_error",
          "final_integral", "final_integral_nuclear_sampling_sem", "scope")}, indent=2))


if __name__ == "__main__":
    main()
