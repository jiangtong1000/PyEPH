"""Run the pinned original engine in its own environment; never a pytest step.

Usage: legacy-python generate_reference.py --source /path/to/original/PyEPH
This writes actual initial samples and legacy trajectories to reference.h5.
The nine historical expected files are copied unchanged, never regenerated.
"""

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

REVISION = "6c4693acbb69a06a5bc8b0593abde2170ff38843"


def make_case(name):
    import numpy as np
    from pyeph.greenkubo.typical_model_helper import (
        build_1d_ssh_optical_model, build_2d_ssh_model,
        build_2d_multiple_Holstein_and_Peierls_model,
    )
    from pyeph.greenkubo.hamiltonian import ElectronPhononHamiltonian
    from pyeph.greenkubo.lattice import BravaisLattice2D
    from pyeph.greenkubo.phonon import build_phonon_baths, ClassicalPhononNonlocal
    from pyeph.greenkubo.propagator import DensityMatrixUnitaryPropagator

    if name == "1D_CPA":
        lattice, t, g, w = build_1d_ssh_optical_model(1., .5, .044, 31, direction="y")
        temperature, ntraj, dt, total = 1., 10, .01, .1
        classical, quantum = build_phonon_baths(w, g, temperature, temperature, "Boltzmann")
        ham = ElectronPhononHamiltonian(t, classical.gmat, lattice)
    elif name.startswith("nonlocal_"):
        _, distribution, gauge = name.split("_")
        lattice = BravaisLattice2D(4, 2, 1, np.zeros((1, 2)))
        w = np.array([[.4, .5, .6, .7], [.8, 1., 1.2, 1.4]])
        t = {(1, 0): np.array([[-.7]]), (0, 1): np.array([[.2]])}
        g = {(1, 0): {(0, 0): np.array([[[.03, -.02]]]),
                     (0, 1): np.array([[[.01, .04]]])}}
        temperature, ntraj, dt, total = .6, 2, .02, .1
        classical = ClassicalPhononNonlocal(w, temperature, g, distribution, use_gauge_phase=gauge == "True")
        quantum = None
        ham = ElectronPhononHamiltonian(t, g, lattice)
    else:
        band_only = name == "band_narrow_only"
        parts = name.split("_")
        model_type = "optical" if band_only else parts[-2].replace("Peierls", "")
        zigzag = False if band_only else parts[-1] == "zzTrue"
        if name.startswith("2D_CPA"):
            j1, j2, j3 = -96.1, 35., -14.7
            energy = np.sqrt(j1*j1+j2*j2+j3*j3)
            lattice, t, g, w, temperature = build_2d_ssh_model(
                18, 9, 7.2, 14.3, j1, j2, j3, .246, .421, .321, 6., 25.,
                model_type, 7.2, energy, zigzag=zigzag,
            )
            classical, quantum = build_phonon_baths(w, g, temperature, temperature, "Boltzmann")
            ham = ElectronPhononHamiltonian(t, classical.gmat, lattice)
            ntraj = 10
        else:
            j1, j2, j3 = -96.1, -35., -14.7
            energy = np.sqrt(j1*j1+j2*j2+j3*j3)
            w = np.array([5., 10., 20., 30., 40.])
            g = np.array([1., 2., 1., 1., .8])*w
            ham, classical, quantum, lattice, temperature = build_2d_multiple_Holstein_and_Peierls_model(
                4, 2, 7.2, 14.3, j1, j2, j3, .246, .421, .321, w, g, 6., 35., 25.,
                model_type, 7.2, energy, zigzag=zigzag,
            )
            ntraj = 2
            if band_only:
                quantum.band_narrow_only = True
        dt, total = .01, .1
    propagator = DensityMatrixUnitaryPropagator(lattice.nsites, ntraj, dt, total, temperature)
    return lattice, ham, classical, quantum, propagator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    args = parser.parse_args()
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=args.source, text=True).strip()
    if revision != REVISION:
        raise RuntimeError(f"source revision {revision} differs from pinned {REVISION}")
    os.environ["USE_MPI"] = "false"
    sys.path.insert(0, str(args.source.resolve()))
    import h5py
    import numpy as np
    from pyeph.greenkubo.simulation import GreenKuboSimulation

    destination = Path(__file__).resolve().parent
    output = destination/"reference.h5"
    if output.exists():
        raise FileExistsError("references already exist; never silently regenerate them")
    expected = args.source/"pyeph/greenkubo/tests"
    names = ["1D_CPA"] + [f"2D_{method}_{model}Peierls_zz{zigzag}"
        for method in ("CPA", "PT_CPA") for model in ("bond", "optical") for zigzag in (True, False)]
    manifest = {"source_repository": "https://github.com/jiangtong1000/PyEPH", "source_revision": revision,
                "python": sys.version, "dependencies": {name: importlib.metadata.version(name)
                for name in ("numpy", "scipy", "h5py", "numba", "llvmlite")}, "sources": {},
                "historical_expected": {}, "comparison_to_historical": {}}
    for file in (args.source/"pyeph/greenkubo").glob("*.py"):
        manifest["sources"][str(file.relative_to(args.source))] = hashlib.sha256(file.read_bytes()).hexdigest()
    for file in (args.source/"pyeph/utils").glob("*.py"):
        manifest["sources"][str(file.relative_to(args.source))] = hashlib.sha256(file.read_bytes()).hexdigest()
    for name in names:
        file = expected/f"expected_{name}.h5"
        shutil.copyfile(file, destination/file.name)
        manifest["historical_expected"][file.name] = hashlib.sha256(file.read_bytes()).hexdigest()
    names += ["band_narrow_only"] + [f"nonlocal_{distribution}_{gauge}"
               for distribution in ("Boltzmann", "Wigner") for gauge in (False, True)]
    with h5py.File(output, "x") as archive:
        archive.attrs["source_revision"] = revision
        for name in names:
            print(f"Generating pinned old-engine case {name}", flush=True)
            lattice, ham, classical, quantum, propagator = make_case(name)
            GreenKuboSimulation(lattice, ham, classical, quantum, propagator)
            group = archive.create_group(name)
            nonlocal_phonons = name.startswith("nonlocal_")
            group["q0"] = classical.q0_half if nonlocal_phonons else classical.q0
            group["p0"] = classical.p0_half if nonlocal_phonons else classical.p0
            group.create_dataset("h0_first", data=ham.heps[0].toarray(), compression="gzip")
            group.create_dataset("rho0_first", data=propagator.rho0[0], compression="gzip")
            currents, fields = [], []
            for step, _time in enumerate(propagator.time_range):
                if step:
                    propagator.evolve(ham, classical, quantum)
                jx, jy = ham.build_jx_jy(ham.heps)
                currents.append(np.stack(propagator.calculate_current(jx, jy), axis=-1))
                if nonlocal_phonons:
                    fields.append(classical.qfield.copy())
            currents = np.asarray(currents)
            group["correlation"] = currents
            group["times"] = propagator.time_range
            group.create_dataset("unitary_final_first", data=propagator.u_t[0], compression="gzip")
            if fields:
                group["real_space_fields"] = fields
            fixture = destination/f"expected_{name}.h5"
            if fixture.exists():
                comparisons = {}
                with h5py.File(fixture) as golden:
                    for axis in ("x", "y"):
                        key = f"current_{axis}"
                        if key in golden:
                            actual = currents[..., ("x", "y").index(axis)].mean(axis=1)
                            reference = golden[key][...]
                            comparisons[key] = {"max_abs_error": float(np.max(np.abs(actual-reference))),
                                                "allclose": bool(np.allclose(actual, reference))}
                manifest["comparison_to_historical"][name] = comparisons
                print(comparisons, flush=True)
            archive.flush()
    manifest["reference_sha256"] = hashlib.sha256(output.read_bytes()).hexdigest()
    (destination/"provenance.json").write_text(json.dumps(manifest, indent=2)+"\n")


if __name__ == "__main__":
    main()
