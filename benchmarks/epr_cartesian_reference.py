"""Audit EPR ingestion and compare native Gamma-cell dynamics to a dense ODE.

This is an equation/infrastructure check, not a transport or materials accuracy
claim. The independent NumPy equations assemble raw Cartesian source stencils;
they do not invoke native model actions/forces or legacy PyEPH dynamics.
Run with JAX_ENABLE_X64=1 and --epr FILE --output NEW_DIRECTORY.
"""

import argparse
import hashlib
import json
from pathlib import Path
import platform
import time

import jax
import jax.numpy as jnp
import numpy as np
from scipy.integrate import solve_ivp

from pyeph import CPA, CoupledClassical, Ehrenfest, Execution, Integrator, Problem
from pyeph import Simulation, make_state
from pyeph.adapters.epr import read_epr
from pyeph.core.contracts import ProbeContext, pure_state_weight
from pyeph.paths.normal_modes import NormalModeBath


def gamma_equations(data):
    """Literal full directed sums followed by the explicitly selected projection."""
    n, d = len(data.wannier_centers), 3*len(data.masses)
    h = np.zeros((n, n), complex)
    g = np.zeros((d, n, n), complex)
    k = np.zeros((d, d))
    np.add.at(h, tuple(data.hopping_orbitals.T), data.hopping_values)
    rows, cols = data.hopping_orbitals[data.epc_channels].T
    for axis in range(3):
        np.add.at(g, (3*data.epc_atoms+axis, rows, cols), data.epc_values[:, axis])
    for a in range(3):
        for b in range(3):
            np.add.at(k, (3*data.ifc_atoms[:, 0]+a, 3*data.ifc_atoms[:, 1]+b),
                      data.ifc_values[:, a, b])
    return (h+h.conj().T)/2, (g+g.conj().swapaxes(-1, -2))/2, (k+k.T)/2


def run(path, output, *, stop=8., timesteps=(.5, .25, .125), zero_tolerance=1e-16):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    if not jax.config.x64_enabled:
        raise RuntimeError("this numerical reference requires JAX_ENABLE_X64=1")
    start = time.perf_counter()
    source = read_epr(path, polar="short_range")
    read_seconds = time.perf_counter()-start
    start = time.perf_counter()
    compiled = source.compile_supercell((1, 1, 1), hermiticity="project")
    compile_seconds = time.perf_counter()-start
    model, params, data = compiled.model, compiled.params, source.data
    q_shape = model.spec.system.q_shape
    h, g, k = gamma_equations(data)
    mass = np.repeat(data.masses, 3)
    values, vectors = np.linalg.eigh(k/np.sqrt(mass[:, None]*mass[None, :]))
    unstable = np.flatnonzero(values < -zero_tolerance)
    if unstable.size:
        raise ValueError(f"Gamma reference has physical unstable modes: {unstable.tolist()}")
    zero = np.flatnonzero(abs(values) <= zero_tolerance)
    bath = NormalModeBath(values, vectors, compiled.masses, np.zeros(q_shape),
                          frozen_modes=tuple(zero.tolist()), zero_tolerance=zero_tolerance)
    rng = np.random.default_rng(3917)
    q0, p0 = .02*rng.normal(size=len(mass)), .1*rng.normal(size=len(mass))
    qm, pm = vectors.T@(q0*np.sqrt(mass)), vectors.T@(p0/np.sqrt(mass))
    qm[zero], pm[zero] = 0., 0.
    q0, p0 = vectors@qm/np.sqrt(mass), np.sqrt(mass)*(vectors@pm)
    c0 = rng.normal(size=model.nstates)+1j*rng.normal(size=model.nstates)
    c0 /= np.linalg.norm(c0)
    initial = make_state(q0.reshape(q_shape), p0.reshape(q_shape), c0)
    omega = np.sqrt(np.maximum(values, 0.))
    omega[zero] = 0.

    def motion(t):
        phase = omega*t
        q = qm*np.cos(phase)+pm*t*np.sinc(phase/np.pi)
        p = pm*np.cos(phase)-omega*qm*np.sin(phase)
        return vectors@q/np.sqrt(mass), np.sqrt(mass)*(vectors@p)

    def hamiltonian(q):
        return h+np.einsum("d,dij->ij", q, g)

    def ehrenfest_rhs(t, state):
        q, p, real, imag = np.split(state, (len(mass), 2*len(mass), 2*len(mass)+len(c0)))
        c = real+1j*imag
        dc = -1j*hamiltonian(q)@c
        force = -k@q-np.einsum("i,dij,j->d", c.conj(), g, c).real
        return np.concatenate((p/mass, force, dc.real, dc.imag))

    cpa = solve_ivp(lambda t, c: -1j*hamiltonian(motion(t)[0])@c,
                    (0., stop), c0, method="DOP853", rtol=2e-13, atol=2e-14)
    eh = solve_ivp(ehrenfest_rhs, (0., stop), np.concatenate((q0, p0, c0.real, c0.imag)),
                   method="DOP853", rtol=2e-13, atol=2e-14)
    if not cpa.success or not eh.success:
        raise RuntimeError("independent dense ODE failed")
    ehq, ehp, ehr, ehi = np.split(eh.y[:, -1], (len(mass), 2*len(mass), 2*len(mass)+len(c0)))
    oracle = {"cpa": (*motion(stop), cpa.y[:, -1]), "ehrenfest": (ehq, ehp, ehr+1j*ehi)}
    action_error = float(np.max(abs(np.asarray(model.apply(params, initial.q, initial.electronic))
                                   -hamiltonian(q0)@c0)))
    force = np.asarray(model.reference_gradient(params, initial.q)
                       +model.contract_gradient(params, initial.q, pure_state_weight(initial.electronic)))
    force_error = float(np.max(abs(force.reshape(-1)-(k@q0+
                           np.einsum("i,dij,j->d", c0.conj(), g, c0).real))))
    current_error = {}
    identity = jnp.eye(model.nstates, dtype=complex)
    delta = 2e-6
    for axis, label in enumerate("xyz"):
        direction = jnp.eye(3)[axis]*delta
        finite_difference = model.charge*(model.apply_peierls(params, initial.q, direction, identity)
            -model.apply_peierls(params, initial.q, -direction, identity))/(2*delta)
        actual = model.probe_apply(params, ProbeContext(initial.q), f"current_{label}", identity)
        current_error[label] = float(np.max(abs(np.asarray(actual-finite_difference))))
    results, arrays = {}, dict(h=h, g=g, k=k, q0=q0, p0=p0, c0=c0,
                              squared_frequencies=values, zero_modes=zero)
    for name, method, treatment in (("cpa", CPA(), bath),
                                    ("ehrenfest", Ehrenfest(), CoupledClassical(compiled.masses))):
        problem = Problem(model, params, treatment, method)
        rows = []
        oq, op, oc = oracle[name]
        arrays[f"{name}_oracle_q"], arrays[f"{name}_oracle_p"] = oq, op
        arrays[f"{name}_oracle_c"] = oc
        for index, dt in enumerate(timesteps):
            steps = round(stop/dt)
            if not np.isclose(steps*dt, stop):
                raise ValueError("stop must be an integer number of all timesteps")
            simulation = Simulation(problem, Integrator(dt), Execution(chunk_size=steps))
            start = time.perf_counter()
            result = simulation.run(initial, steps)
            seconds = time.perf_counter()-start
            q, p, c = map(np.asarray, (result.final_state.q, result.final_state.p,
                                       result.final_state.electronic))
            energy = .5*np.sum(p.reshape(-1)**2/mass)+.5*q.reshape(-1)@k@q.reshape(-1)
            energy += np.vdot(c, hamiltonian(q.reshape(-1))@c).real
            energy0 = .5*np.sum(p0**2/mass)+.5*q0@k@q0+np.vdot(c0, hamiltonian(q0)@c0).real
            row = dict(dt=dt, steps=steps, cold_seconds=seconds,
                       q_error=float(np.max(abs(q.reshape(-1)-oq))),
                       p_error=float(np.max(abs(p.reshape(-1)-op))),
                       c_error=float(np.max(abs(c-oc))),
                       norm_defect=float(abs(np.vdot(c, c)-1)), energy_change=float(energy-energy0))
            rows.append(row)
            for field, value in (("q", q), ("p", p), ("c", c)):
                arrays[f"{name}_{index}_{field}"] = value
            if index == len(timesteps)-1:
                half = simulation.run(initial, steps//2)
                checkpoint = output/f"{name}_checkpoint.h5"
                simulation.save_checkpoint(checkpoint, half.final_state)
                restored = simulation.load_checkpoint(checkpoint)
                restarted = simulation.run(restored, steps-steps//2)
                restart_equal = all(np.array_equal(a, b) for a, b in zip(
                    jax.tree.leaves(result.final_state), jax.tree.leaves(restarted.final_state)))
                row["checkpoint_restart_bitwise_equal"] = restart_equal
                if not restart_equal:
                    raise AssertionError("checkpoint restart changed native final state")
        results[name] = rows
        print(name, json.dumps(rows), flush=True)
    report = dict(source=source.metadata, compilation=compiled.report,
        read_seconds=read_seconds, preparation_seconds=compile_seconds,
        scope="Gamma-cell equation/infrastructure validation; no materials or mobility claim",
        stop_reduced=stop, stop_fs=stop*data.unit_system.time_fs,
        source_modes_modified=False, constrained_initial_and_cpa_modes=zero.tolist(),
        ehrenfest_constraint="none after initialization; full original model force",
        squared_frequency_zero_tolerance=zero_tolerance,
        signed_frequency_squared_min=float(values.min()), numeric_zero_modes=len(zero),
        physical_unstable_modes=len(unstable),
        electronic_acoustic_sum_max=float(abs(g.reshape(len(data.masses), 3, model.nstates,
                                                       model.nstates).sum(axis=0)).max()),
        action_max_error=action_error, force_max_error=force_error,
        peierls_current_finite_difference_max_errors=current_error,
        dynamics=results, environment=dict(python=platform.python_version(), jax=jax.__version__,
            numpy=np.__version__, platform=platform.platform(), devices=[str(x) for x in jax.devices()]),
        driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    np.savez_compressed(output/"arrays.npz", **arrays)
    (output/"report.json").write_text(json.dumps(report, indent=2)+"\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epr", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.epr, args.output), indent=2))
