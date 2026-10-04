"""Historical recipe facade executing the single native CPA dynamics engine."""

from dataclasses import dataclass, replace
from pathlib import Path

import h5py
import jax.numpy as jnp
import numpy as np

from pyeph.core.contracts import ProbeContext
from pyeph.core.problem import Problem
from pyeph.core.state import stack_states
from pyeph.dynamics.cpa import CPA
from pyeph.execution.runner import Execution
from pyeph.paths.harmonic import HarmonicBath
from pyeph.simulation import Simulation
from pyeph.workflows.polaron_transport import (
    PolaronDressedModel, make_polaron_transport_problem,
)
from pyeph.workflows.transport import (
    TransportMeasurement, initialize_transport_state, make_transport_problem,
)

from .analysis import merge_outputs
from ._precision import require_legacy_precision
from .mpi_random import MPIRandomContext

SOURCE_REVISION = "6c4693acbb69a06a5bc8b0593abde2170ff38843"


class _StructuredLFModel(PolaronDressedModel):
    """Historical private name; structured dressing lives in the public wrapper."""


class _NarrowedCurrentModel(_StructuredLFModel):
    def probe_apply(self, params, context, probe, vectors):
        return self.factor*self.base_model.probe_apply(params, context, probe, vectors)


@dataclass(frozen=True)
class _BandNarrowMeasurement(TransportMeasurement):
    thermal_policy: str = "legacy_full"

    def initial_hamiltonian(self, problem, state):
        if self.thermal_policy == "legacy_full":
            return problem.model.factor*problem.model.bare_hamiltonian(problem.params, state.q)
        return super().initial_hamiltonian(problem, state)


def _assemble(ham, classical, quantum, propagator, *, thermal_policy, trajectory_id_start=0, seed=0):
    if thermal_policy not in {"legacy_full", "offdiagonal"}:
        raise ValueError("thermal_policy must be legacy_full or offdiagonal")
    if (not isinstance(trajectory_id_start, (int, np.integer)) or trajectory_id_start < 0
            or trajectory_id_start+propagator.ntraj > 2**32):
        raise ValueError("trajectory IDs must fit the global uint32 identity range")
    if ham.lattice.nsites != propagator.nsites:
        raise ValueError("Hamiltonian and propagator state counts differ")
    if quantum is not None and not np.isclose(quantum.beta, propagator.beta, rtol=1e-13):
        raise ValueError("quantum bath and electronic thermal preparation must share a temperature")
    model, params = ham.native(classical)
    treatment = HarmonicBath(params["canonical_frequencies"])
    probes = ("current_x", "current_y")
    if quantum is None:
        problem = make_transport_problem(model, params, treatment, probes=probes)
    elif quantum.band_narrow_only:
        wrapped = _NarrowedCurrentModel(model, quantum.polaron_prefactor)
        measurement = _BandNarrowMeasurement(probes=probes, thermal_policy=thermal_policy)
        problem = Problem(wrapped, params, treatment, CPA(), measurement).validate()
    else:
        problem = make_polaron_transport_problem(
            model, params, treatment, quantum.w, quantum.couplings, propagator.beta,
            hopping_pairs=ham.hopping_pairs, thermal_policy=thermal_policy, probes=probes,
        )
        problem = replace(problem, model=_StructuredLFModel(model, problem.model.factor))
    q, p = classical.canonical_initial()
    legacy_q, legacy_p = classical.initial_samples()
    states = []
    for trajectory in range(propagator.ntraj):
        state = initialize_transport_state(problem, q[trajectory], p[trajectory], propagator.beta,
                                           trajectory_id=trajectory_id_start+trajectory, seed=seed)
        state.method_state["transport"]["legacy_q0"] = jnp.asarray(legacy_q[:, trajectory])
        state.method_state["transport"]["legacy_p0"] = jnp.asarray(legacy_p[:, trajectory])
        # Preserve bare initial probes separately for compatibility attribute
        # views, including exactly zero narrowing factors at strong coupling.
        if quantum is not None and quantum.band_narrow_only:
            eye = jnp.eye(propagator.nsites, dtype=state.electronic.dtype)
            context = ProbeContext(state.q, state.p, state.time)
            bare = jnp.stack([model.probe_apply(params, context, probe, eye) for probe in probes])
            state.method_state["transport"]["bare_currents0"] = bare
        states.append(state)
    return problem, stack_states(states)


class GreenKuboSimulation:
    """Compatibility constructor backed by native Problem/CPA/Simulation.

    Compatibility defaults to the historical LF whole-H thermal preparation;
    the modern LF workflow independently retains its offdiagonal default.
    Initial samples can be injected as (X,Y) to compare identical trajectories.
    """

    def __init__(self, lattice, ham, classic_ph, quantum_ph, propagator, base_seed=1120,
                 *, thermal_policy="legacy_full", initial_samples=None, execution=None):
        require_legacy_precision()
        self.mpi_handler = MPIRandomContext(base_seed)
        self.lattice, self.ham = lattice, ham
        self.classic_ph, self.quantum_ph, self.propagator = classic_ph, quantum_ph, propagator
        self.base_seed, self.thermal_policy = base_seed, thermal_policy
        self.execution = execution or Execution()
        if self.execution.save_every != 1:
            raise ValueError("legacy current output requires Execution(save_every=1); "
                             "use native Simulation/HDF5Observer for sparse observation grids")
        if initial_samples is not None:
            classic_ph.set_initial_samples(*initial_samples)
        self.is_1d_along_x, self.is_1d_along_y = lattice.ny == 1, lattice.nx == 1
        self.axes = [axis for axis, enabled in (("x", lattice.nx > 1), ("y", lattice.ny > 1)) if enabled]
        if not self.axes:
            self.axes = ["x", "y"]
        self.time_step, self.total_time = propagator.time_step, propagator.total_time
        self.n_time_steps = len(propagator.time_range)
        self.current_x, self.current_y, self.current_times = [], [], []
        self.build()

    def build(self):
        self.classic_ph.initialize_position_and_momentum(
            self.lattice.nx, self.lattice.ny, self.propagator.ntraj, self.mpi_handler.rng)
        if self.quantum_ph is not None:
            self.propagator.build(self.ham, self.quantum_ph)
        self.polaron_prefactor = self.propagator.polaron_prefactor
        self.problem, self.initial_state = _assemble(
            self.ham, self.classic_ph, self.quantum_ph, self.propagator,
            thermal_policy=self.thermal_policy,
            trajectory_id_start=self.mpi_handler.rank*self.propagator.ntraj,
            seed=self.base_seed,
        )
        self.state = self.initial_state
        self.native_simulation = Simulation(self.problem, self.propagator.integrator, self.execution)
        self.propagator._sync_native(self.problem, self.state)
        self.ham.heps = self.ham.build_ep_variation_matrix(self.classic_ph.qfield)

    def save_initial_samples(self, filename):
        """Archive actual samples and coordinate conventions, independent of RNG."""
        q, p = self.classic_ph.initial_samples()
        with h5py.File(filename, "x") as handle:
            handle["q0"], handle["p0"] = q, p
            handle["canonical_q0"] = np.asarray(self.initial_state.q)
            handle["canonical_p0"] = np.asarray(self.initial_state.p)
            handle["trajectory_id"] = np.asarray(self.initial_state.trajectory_id)
            handle.attrs["source_revision"] = SOURCE_REVISION
            handle.attrs["distribution"] = self.classic_ph.distribution
            handle.attrs["coordinate_convention"] = "legacy_half_grid_XY" if self.classic_ph.nonlocal_phonons else "legacy_local_XY"
            handle.attrs["nonlocal_field_order"] = "x_major" if self.classic_ph.nonlocal_phonons else "y_major"
            handle.attrs["nx"], handle.attrs["ny"] = self.lattice.nx, self.lattice.ny
            handle.attrs["thermal_policy"] = self.thermal_policy
            handle.attrs["sampling_identity"] = "historical NumPy rank stream; samples archived explicitly"
            handle.attrs["energy_hartree"] = self.lattice.unit_system.energy_hartree
            handle.attrs["length_bohr"] = self.lattice.unit_system.length_bohr
            handle.attrs["physical_unit_scale_known"] = self.lattice.unit_scale_known

    @staticmethod
    def load_initial_samples(filename):
        with h5py.File(filename) as handle:
            return handle["q0"][...], handle["p0"][...]

    def initialize_dump(self, dump_dir, dump_interval):
        self.dump_dir = Path(dump_dir)
        self.dump_dir.mkdir(parents=True, exist_ok=True)
        self.dump_interval = int(dump_interval) if dump_interval is not None else self.n_time_steps
        if self.dump_interval < 1:
            raise ValueError("dump_interval must be positive")
        self.dump_fname = self.dump_dir/f"currents_{self.mpi_handler.rank}.h5"
        if self.dump_fname.exists() or (self.dump_dir/"collected_current_autocorr.h5").exists():
            raise FileExistsError("transport output already exists; choose a new dump directory")
        with h5py.File(self.dump_fname, "x") as handle:
            handle.attrs["time_step"], handle.attrs["total_time"] = self.time_step, self.total_time
            handle.attrs["current_step"] = 0
            handle.attrs["initial_time"] = float(np.asarray(self.state.time)[0])
            handle.attrs["thermal_policy"] = self.thermal_policy
            handle.attrs["engine"] = "native_pyeph_cpa"
            for axis in self.axes:
                handle.create_dataset(f"current_{axis}", (self._run_samples,), dtype=complex)
            handle.create_dataset("time", (self._run_samples,), dtype=float)
        self._written = 0

    def _write_observation(self, times, values):
        times = np.asarray(times)
        if times.ndim != 2 or not np.allclose(times, times[:, :1], rtol=0, atol=1e-12):
            raise ValueError("legacy output requires synchronized trajectory times")
        means = np.asarray(values["current_correlation"]).mean(axis=1)
        count = len(means)
        with h5py.File(self.dump_fname, "a") as handle:
            handle["time"][self._written:self._written+count] = times[:, 0]
            for axis in self.axes:
                column = ("x", "y").index(axis)
                handle[f"current_{axis}"][self._written:self._written+count] = means[:, column]
            self._written += count
            handle.attrs["current_step"] = self._written

    def run(self, dump_dir=None, dump_interval=None, *, collect=False, steps=None, keep_rank_files=False):
        remaining = self.n_time_steps-1-int(np.asarray(self.state.step)[0])
        count = remaining if steps is None else steps
        if not isinstance(count, int) or count < 0:
            raise ValueError("steps must be nonnegative; the configured trajectory has already ended")
        self._run_samples = count+1
        execution = replace(self.execution, chunk_size=int(dump_interval) if dump_interval else self.execution.chunk_size)
        self.native_simulation = Simulation(self.problem, self.propagator.integrator, execution)
        observer = None
        if dump_dir is not None:
            self.initialize_dump(dump_dir, dump_interval)
            self.save_initial_samples(self.dump_dir/f"initial_samples_{self.mpi_handler.rank}.h5")
            observer = self._write_observation
        self.result = self.native_simulation.run(self.state, count, observer=observer,
                                                collect=collect or dump_dir is None)
        self.state = self.result.final_state
        self.propagator._sync_native(self.problem, self.state)
        self.classic_ph.update_position(self.propagator.time)
        self.ham.heps = self.ham.build_ep_variation_matrix(self.classic_ph.qfield)
        if self.quantum_ph is not None:
            self.quantum_ph.update_phit(self.propagator.time)
            self.propagator.sec_weights = self.quantum_ph.sector_weights
        if dump_dir is not None:
            self.mpi_handler.barrier()
            if self.mpi_handler.rank == 0:
                self.output_file = merge_outputs(self.dump_dir, self.axes, self.mpi_handler.size, safe_mode=False)
            self.mpi_handler.barrier()
            if not keep_rank_files:
                self.dump_fname.unlink()
        return self.result

    def save_checkpoint(self, filename):
        self.native_simulation.save_checkpoint(filename, self.state)

    def load_checkpoint(self, filename):
        self.state = self.native_simulation.load_checkpoint(filename)
        # Retain exact sampled arrays in the native checkpoint payload. Inverse
        # harmonic evolution would recover them only up to floating-point error.
        payload = self.state.method_state["transport"]
        if "legacy_q0" not in payload or "legacy_p0" not in payload:
            raise ValueError("compatibility checkpoint lacks exact original initial samples")
        x0 = np.asarray(payload["legacy_q0"]).swapaxes(0, 1)
        y0 = np.asarray(payload["legacy_p0"]).swapaxes(0, 1)
        bath = self.classic_ph
        bath.set_initial_samples(x0, y0)
        bath.initialize_position_and_momentum(self.lattice.nx, self.lattice.ny,
                                             self.propagator.ntraj, self.mpi_handler.rng)
        q0, p0 = map(jnp.asarray, bath.canonical_initial())
        eye = jnp.broadcast_to(jnp.eye(self.lattice.nsites, dtype=self.state.electronic.dtype), self.state.electronic.shape)
        self.initial_state = self.state._replace(q=q0, p=p0, electronic=eye, time=payload["time0"],
                                                  step=jnp.zeros_like(self.state.step))
        self.propagator._sync_native(self.problem, self.state)
        bath.update_position(self.propagator.time)
        return self.state
