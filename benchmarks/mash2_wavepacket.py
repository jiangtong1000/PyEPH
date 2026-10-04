"""Published modified-Tully wavepacket preparation and an independent FFT oracle.

The quantum solver is a benchmark reference, not a production PyEPH method.
Eq. 38 and Fig. 5 of https://arxiv.org/abs/2212.11773 specify the potential,
mass, Gaussian Wigner preparation, and final time. No published curve is
digitized here: both quantum and trajectory results are calculated afresh.
"""

import argparse
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import platform
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np
from scipy.constants import physical_constants

from pyeph import CoupledClassical, Execution, Integrator, Problem, Simulation
from pyeph.core.contracts import ModelSpec
from pyeph.core.state import stack_states
from pyeph.core.system import SystemSpec
from pyeph.dynamics.mash2 import MASH2, MASHPopulation, sample_adiabatic_population
from pyeph.models.base import AutoDiffModel


MASS = 2000.0
FINAL_TIME = 150e-15 / physical_constants["atomic unit of time"][0]
CASES = {"low": (np.sqrt(2*MASS*0.03), 0.5), "high": (20.0, 0.1)}


@dataclass(frozen=True)
class ModifiedTully(AutoDiffModel):
    """Standalone public-protocol extension; deliberately separate from TullyModel."""

    spec: ModelSpec = field(default_factory=lambda: ModelSpec(
        SystemSpec(nstates=2, q_shape=(1,), coordinate_kind="canonical"),
        name="mash_paper_modified_tully1"))

    def apply(self, params, q, vectors):
        x = q[0]
        z, coupling = 0.01*jnp.tanh(1.6*x), 0.005*jnp.exp(-x*x)
        matrix = jnp.array([[z, coupling], [coupling, -z]])
        return matrix @ vectors

    def reference_energy(self, params, q):
        return jnp.zeros((), dtype=q.dtype)


def quantum_potential(q):
    """Independent NumPy matrix entries; no calls into production model code."""
    return 0.01*np.tanh(1.6*q), 0.005*np.exp(-q*q)


def lower_vectors(q):
    z, coupling = quantum_potential(np.asarray(q))
    angle = np.arctan2(coupling, z)
    return np.stack((np.sin(angle/2), -np.cos(angle/2)), axis=-1)


def split_operator(psi, q, *, mass, duration, dt, potential=quantum_potential):
    """Strang FFT propagation for real traceless two-state potentials.

    psi has shape (grid,2) and sum(abs(psi)**2)=1 (discrete normalization).
    Periodic boundaries are explicit; callers must check edge probabilities.
    The effective dt exactly divides the requested duration.
    """
    q = np.asarray(q, dtype=float)
    psi = np.asarray(psi, dtype=complex).copy()
    if (q.ndim != 1 or len(q) < 4 or psi.shape != (len(q), 2)
            or not np.all(np.isfinite(q)) or not np.all(np.isfinite(psi))):
        raise ValueError("finite uniform grid and (grid,2) wavefunction required")
    spacing = np.diff(q)
    if spacing[0] <= 0 or not np.allclose(spacing, spacing[0], rtol=1e-10, atol=1e-12):
        raise ValueError("positive uniform grid spacing required")
    if not all(np.isfinite(x) and x > 0 for x in (mass, duration, dt)):
        raise ValueError("positive finite mass, duration and dt required")
    steps = int(np.ceil(duration/dt))
    h = duration/steps
    momentum = 2*np.pi*np.fft.fftfreq(len(q), spacing[0])
    kinetic_half = np.exp(-0.25j*h*momentum**2/mass)[:, None]
    z, coupling = (np.broadcast_to(x, q.shape) for x in potential(q))
    radius = np.hypot(z, coupling)
    diagonal = np.cos(h*radius)
    off = -1j*h*np.sinc(h*radius/np.pi)
    propagator00, propagator11 = diagonal+off*z, diagonal-off*z
    propagator01 = off*coupling
    start = perf_counter()
    edge_count = max(1, len(q)//20)
    maximum_edge_probability = float(np.sum(np.abs(psi[:edge_count])**2)
                                     + np.sum(np.abs(psi[-edge_count:])**2))
    # Adjacent half kinetic steps are combined. The returned state is in q-space.
    psi_k = np.fft.fft(psi, axis=0, norm="ortho") * kinetic_half
    for index in range(steps):
        psi = np.fft.ifft(psi_k, axis=0, norm="ortho")
        psi = np.column_stack((propagator00*psi[:, 0]+propagator01*psi[:, 1],
                               propagator01*psi[:, 0]+propagator11*psi[:, 1]))
        if index % max(1, steps//64) == 0:
            maximum_edge_probability = max(maximum_edge_probability, float(
                np.sum(np.abs(psi[:edge_count])**2)+np.sum(np.abs(psi[-edge_count:])**2)))
        psi_k = np.fft.fft(psi, axis=0, norm="ortho") * kinetic_half
        if index != steps-1:
            psi_k *= kinetic_half
    psi = np.fft.ifft(psi_k, axis=0, norm="ortho")
    maximum_edge_probability = max(maximum_edge_probability, float(
        np.sum(np.abs(psi[:edge_count])**2)+np.sum(np.abs(psi[-edge_count:])**2)))
    return psi, dict(dt=h, steps=steps, seconds=perf_counter()-start,
                     sampled_max_outer_five_percent_probability=maximum_edge_probability,
                     edge_monitor_interval=max(1, steps//64))


def quantum_run(case, *, points=8192, dt=0.5, limits=(-50., 90.), duration=FINAL_TIME):
    """Return discrete probabilities and continuum densities in atomic units.

    ``probability_q`` and ``probability_p`` each sum to the propagated norm.
    Their corresponding densities integrate to that norm using dx and dp.
    """
    if (not isinstance(points, (int, np.integer)) or points < 4
            or np.shape(limits) != (2,) or not np.isfinite(limits).all()
            or limits[1] <= limits[0]):
        raise ValueError("at least four grid points and finite increasing limits required")
    p0, gamma = CASES[case]
    q = np.linspace(*limits, points, endpoint=False)
    dx = q[1]-q[0]
    scalar = (gamma/np.pi)**0.25 * np.exp(-0.5*gamma*(q+15.)**2+1j*p0*(q+15.))
    psi0 = scalar[:, None] * lower_vectors(q) * np.sqrt(dx)
    initial_norm = np.sum(np.abs(psi0)**2)
    psi, stats = split_operator(psi0, q, mass=MASS, duration=duration, dt=dt)
    psi_k = np.fft.fftshift(np.fft.fft(psi, axis=0, norm="ortho"), axes=0)
    momentum = np.fft.fftshift(2*np.pi*np.fft.fftfreq(points, dx))
    dp = 2*np.pi/(points*dx)
    probability_q, probability_p = np.sum(np.abs(psi)**2, axis=1), np.sum(np.abs(psi_k)**2, axis=1)
    lower = lower_vectors(q)
    p_lower = np.sum(np.abs(np.sum(lower*psi, axis=1))**2)
    stats.update(points=points, limits=list(limits), dx=float(dx), dp=float(dp),
                 units=dict(position="bohr", momentum="atomic", time="atomic", energy="hartree"),
                 initial_norm=float(initial_norm),
                 final_norm=float(probability_q.sum()),
                 final_population=[float(p_lower), float(probability_q.sum()-p_lower)],
                 final_mean_q=float(q@probability_q), final_mean_p=float(momentum@probability_p),
                 edge_probability=float(probability_q[(q < limits[0]+10) | (q > limits[1]-10)].sum()),
                 momentum_edge_probability=float(probability_p[np.abs(momentum) > .8*np.max(momentum)].sum()))
    return stats, dict(q=q, probability_q=probability_q, momentum=momentum,
                       probability_p=probability_p, density_q=probability_q/dx,
                       density_p=probability_p/dp, psi=psi)


def mash_run(case, *, trajectories=1024, seed=4729, dt=1., batch_size=128,
             duration=FINAL_TIME, event_substeps=2):
    """Independent Gaussian nuclear samples plus weighted lower-hemisphere spins."""
    p0, gamma = CASES[case]
    rng = np.random.default_rng(seed)
    # Eq. 37: Var(q)=1/(2 gamma), Var(p)=gamma/2, hbar=1.
    q = rng.normal(-15., np.sqrt(1/(2*gamma)), trajectories)
    p = rng.normal(p0, np.sqrt(gamma/2), trajectories)
    model = ModifiedTully()
    problem = Problem(model, None, CoupledClassical(MASS), MASH2(event_substeps=event_substeps),
                      MASHPopulation())
    steps = int(np.ceil(duration/dt))
    h = duration/steps
    simulation = Simulation(problem, Integrator(h, "exponential_midpoint"),
                            Execution(chunk_size=256, save_every=steps))
    final_q, final_p, active, energy_errors = [], [], [], []
    counters = {name: 0 for name in ("accepted", "frustrated", "events")}
    max_norm_error = max_event_residual = max_impulse_error = 0.
    start = perf_counter()
    for offset in range(0, trajectories, batch_size):
        stop = min(offset+batch_size, trajectories)
        initial = stack_states([
            sample_adiabatic_population(model, None, [q[i]], [p[i]], active=0,
                                       seed=seed+1, trajectory_id=i)
            for i in range(offset, stop)])
        result = simulation.run(initial, steps, collect=False)
        state = jax.device_get(result.final_state)
        final_q.extend(state.q[:, 0])
        final_p.extend(state.p[:, 0])
        active.extend(state.method_state["active"])
        for name in counters:
            counters[name] += int(np.sum(state.method_state[name]))
        max_norm_error = max(max_norm_error, float(np.max(np.abs(np.sum(np.abs(state.electronic)**2, axis=1)-1))))
        max_event_residual = max(max_event_residual, float(np.max(state.method_state["max_event_residual"])))
        max_impulse_error = max(max_impulse_error, float(np.max(state.method_state["max_impulse_energy_error"])))
        z0, v0 = quantum_potential(q[offset:stop])
        zf, vf = quantum_potential(state.q[:, 0])
        e0 = p[offset:stop]**2/(2*MASS)-np.hypot(z0, v0)
        ef = state.p[:, 0]**2/(2*MASS)+(2*state.method_state["active"]-1)*np.hypot(zf, vf)
        energy_errors.extend(np.abs(ef-e0))
    final_q, final_p, active = map(np.asarray, (final_q, final_p, active))
    populations = np.bincount(active, minlength=2)/trajectories
    stats = dict(trajectories=trajectories, seed=seed, dt=h, steps=steps, batch_size=batch_size,
                 event_substeps=event_substeps, seconds_including_compile=perf_counter()-start,
                 final_population=populations.tolist(),
                 population_standard_error=np.sqrt(populations*(1-populations)/trajectories).tolist(),
                 final_mean_q=float(final_q.mean()), final_mean_p=float(final_p.mean()),
                 max_final_energy_error=float(np.max(energy_errors)), max_norm_error=max_norm_error,
                 max_event_residual=max_event_residual, max_impulse_energy_error=max_impulse_error,
                 **counters)
    return stats, dict(initial_q=q, initial_p=p, final_q=final_q, final_p=final_p, active=active)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=CASES, default="high")
    parser.add_argument("--mode", choices=("quantum", "mash", "both"), default="both")
    parser.add_argument("--points", type=int, default=8192)
    parser.add_argument("--quantum-dt", type=float, default=.5)
    parser.add_argument("--mash-dt", type=float, default=1.)
    parser.add_argument("--trajectories", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--duration", type=float, default=FINAL_TIME)
    parser.add_argument("--output", type=Path, default=Path("benchmarks/results/mash2_wavepacket"))
    args = parser.parse_args()
    if min(args.points, args.trajectories, args.batch_size, args.quantum_dt, args.mash_dt, args.duration) <= 0:
        parser.error("counts, timesteps and duration must be positive")
    jax.config.update("jax_enable_x64", True)
    root = Path(__file__).resolve().parents[1]
    sources = list((root/"src/pyeph").rglob("*.py")) + [Path(__file__).resolve()]
    hashes = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}
    stats = dict(case=args.case, mass=MASS, duration=args.duration, p0=CASES[args.case][0], gamma=CASES[args.case][1],
                 q0=-15., paper="https://arxiv.org/abs/2212.11773", paper_case="Eq. 38 / Fig. 5",
                 platform=platform.platform(), python=platform.python_version(), jax=jax.__version__,
                 numpy=np.__version__, source_sha256=hashes, runs={})
    data = {}
    if args.mode in ("quantum", "both"):
        stats["runs"]["quantum"], values = quantum_run(args.case, points=args.points,
                                                      dt=args.quantum_dt, duration=args.duration)
        data.update({"quantum_"+key: value for key, value in values.items()})
    if args.mode in ("mash", "both"):
        stats["runs"]["mash"], values = mash_run(args.case, trajectories=args.trajectories,
                                                dt=args.mash_dt, batch_size=args.batch_size,
                                                duration=args.duration)
        data.update({"mash_"+key: value for key, value in values.items()})
    stats["source_unchanged"] = all(hashlib.sha256((root/name).read_bytes()).hexdigest() == digest
                                    for name, digest in hashes.items())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix(".json").write_text(json.dumps(stats, indent=2)+"\n")
    np.savez_compressed(args.output.with_suffix(".npz"), **data)
    print(json.dumps({key: value for key, value in stats.items() if key != "source_sha256"}, indent=2))


if __name__ == "__main__":
    main()
