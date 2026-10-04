"""Real isolated Gamma-cell MASH infrastructure check against a smooth ODE.

Imaginary coefficients are discarded only under an explicit numeric bound.
This benchmark verifies a no-event interval of one prepared mapping trajectory;
it establishes neither material surface-hopping validity nor event statistics.
"""

import argparse
import json
from pathlib import Path

import jax
import numpy as np
from scipy.integrate import solve_ivp

from pyeph import CoupledClassical, Execution, Integrator, Problem, Simulation
from pyeph.adapters.epr import read_epr
from pyeph.dynamics.mash2 import MASH2, MASHPopulation, mapping_state
from pyeph.dynamics.mashrm import MASHRM, MASHRMPopulation
from pyeph.dynamics.mashrm import mapping_state as rm_state

from epr_cartesian_reference import gamma_equations


def run(path, output, *, real_tolerance=1e-9, stop=8., dts=(1., .5, .25)):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    if not jax.config.x64_enabled:
        raise RuntimeError("this reference requires JAX_ENABLE_X64=1")
    source = read_epr(path, polar="short_range")
    complex_model = source.compile_supercell((1, 1, 1), hermiticity="project")
    shape = complex_model.model.spec.system.q_shape
    rng = np.random.default_rng(285)
    q, p = .02*rng.normal(size=shape), .1*rng.normal(size=shape)
    spin = [np.sqrt(.75)*np.cos(.3), np.sqrt(.75)*np.sin(.3), -.5]
    try:
        mapping_state(complex_model.model, complex_model.params, q, p, spin)
    except ValueError as error:
        complex_rejection = str(error)
    else:
        raise AssertionError("default complex source was not rejected by MASH2")
    compiled = source.compile_supercell((1, 1, 1), hermiticity="project",
                                       real_tolerance=real_tolerance)
    model, params = compiled.model, compiled.params
    initial = mapping_state(model, params, q, p, spin)
    rm_initial = rm_state(model, params, q, p, initial.electronic, basis="fixed", active=0)
    h, g, k = gamma_equations(source.data)
    h, g = h.real, g.real
    masses = np.repeat(source.data.masses, 3)
    n, d = model.nstates, q.size
    if n != 2:
        raise ValueError("the isolated two-state Gamma benchmark requires two orbitals")

    def rhs(t, y):
        position, momentum, real, imaginary = np.split(y, (d, 2*d, 2*d+n))
        c = real+1j*imaginary
        matrix = h+np.einsum("d,dij->ij", position, g)
        energies, rotation = np.linalg.eigh(matrix)
        if energies[1]-energies[0] <= 1e-10:
            raise ValueError("independent reference encountered a degenerate spectrum")
        active = rotation[:, 0]
        force = -k@position-np.einsum("i,dij,j->d", active, g, active)
        dc = -1j*matrix@c
        return np.concatenate((momentum/masses, force, dc.real, dc.imag))

    c = np.asarray(initial.electronic)
    reference = solve_ivp(rhs, (0., stop), np.concatenate((q.ravel(), p.ravel(), c.real, c.imag)),
                          method="DOP853", rtol=2e-13, atol=2e-14)
    if not reference.success:
        raise RuntimeError("independent surface ODE failed")
    rq, rp, rr, ri = np.split(reference.y[:, -1], (d, 2*d, 2*d+n))
    rc = rr+1j*ri
    arrays = dict(q0=q, p0=p, c0=c, reference_q=rq, reference_p=rp, reference_c=rc)
    rows = []
    for name, method, measurement, start in (
        ("mash2", MASH2(event_substeps=1), MASHPopulation(include_nuclei=True), initial),
        ("mashrm", MASHRM(event_substeps=1), MASHRMPopulation(include_nuclei=True), rm_initial),
    ):
        problem = Problem(model, params, CoupledClassical(compiled.masses), method, measurement)
        for index, dt in enumerate(dts):
            steps = round(stop/dt)
            if not np.isclose(steps*dt, stop):
                raise ValueError("stop must be an integer number of steps")
            simulation = Simulation(problem, Integrator(dt, "exponential_midpoint"),
                                    Execution(chunk_size=steps))
            result = simulation.run(start, steps)
            final, diagnostics = result.final_state, result.final_state.method_state
            if int(diagnostics["events"]) != 0 or int(diagnostics["status"]) != 0:
                raise AssertionError("this smooth reference is qualified only for no-event intervals")
            energy = np.asarray(result.observables["energy"])
            rows.append(dict(method=name, dt=dt, steps=steps, status=int(diagnostics["status"]),
                events=int(diagnostics["events"]),
                q_error=float(abs(np.asarray(final.q).ravel()-rq).max()),
                p_error=float(abs(np.asarray(final.p).ravel()-rp).max()),
                c_error=float(abs(np.asarray(final.electronic)-rc).max()),
                energy_drift=float(abs(energy-energy[0]).max()),
                mapping_norm_defect=float(abs(np.vdot(final.electronic, final.electronic)-1))))
            arrays.update({f"{name}_{index}_{key}": np.asarray(value) for key, value in
                (("q", final.q), ("p", final.p), ("c", final.electronic))})
    report = dict(source=source.metadata, compilation=compiled.report,
        default_complex_rejection=complex_rejection, stop_reduced=stop,
        stop_fs=stop*source.data.unit_system.time_fs, runs=rows,
        scope="one isolated real two-state Gamma trajectory with zero events; "
              "no DNTT hopping statistics or generic real-material applicability established")
    np.savez_compressed(output/"arrays.npz", **arrays)
    (output/"report.json").write_text(json.dumps(report, indent=2)+"\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epr", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.epr, args.output), indent=2))
