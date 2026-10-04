# Copyright (c) 2026, the PyEPH contributors.
# Migrated from https://github.com/jiangtong1000/PyEPH, revision
# 6c4693acbb69a06a5bc8b0593abde2170ff38843, under BSD-3-Clause (see LICENSE).
"""Extract real-space EPC data and phonon modes from a PERTURBO epr.h5 file.

This stage prepares data for subsequent localization; it does not run QCPBC.
Use --help for the input path, q-grid and output options.
"""

import argparse
from pathlib import Path

import h5py
import numpy as np

from pyeph.post_qe2pert.eph_mat_mixed import CalcEphMatMixed
from pyeph.post_qe2pert.phonon_disp import PhononDispersion
from ._grids import generate_half_qgrids, rgrid_2d_full
from ._support import get_mpi_info


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--epr-file", type=Path,
        required=True,
        help="PERTURBO epr.h5 input",
    )
    parser.add_argument("--nx", type=int, default=10, help="q-grid size along x (>=2)")
    parser.add_argument("--ny", type=int, help="q-grid size along y (>=2; default: nx)")
    parser.add_argument("--output", type=Path, help="output HDF5 file")
    args = parser.parse_args(argv)
    ny = args.ny if args.ny is not None else args.nx
    if args.nx < 2 or ny < 2:
        parser.error("--nx and --ny must be at least 2")
    if not args.epr_file.is_file():
        parser.error(f"input not found: {args.epr_file}; specify --epr-file")
    output = args.output or Path(
        f"eph_data_{args.nx}.h5" if ny == args.nx else f"eph_data_{args.nx}_{ny}.h5"
    )
    if output.exists():
        parser.error(f"output already exists: {output}; choose a new --output path")

    mpi_info = get_mpi_info()
    # Every rank constructs the same grid and enters the distributed phonon
    # calculation. Only the root receives the complete gathered mode arrays.
    q_hbz, q_minus, q_full, partners = generate_half_qgrids(args.nx, ny)
    phdisp = PhononDispersion(args.epr_file)
    force_constants = phdisp.extract_force_constants()
    freq_half, mode_half = phdisp.compute_phonon_dispersion(
        q_hbz, force_constants, mass_weight=False,
    )
    if mpi_info["rank"] != 0:
        return

    ep = CalcEphMatMixed(args.epr_file)
    gmat_raw = ep.extract_gmat_raw(len(ep.rvec_set_el), len(ep.rvec_set_ph_eph))
    partner_indices = np.asarray(partners, dtype=int)
    freq_full = np.concatenate((freq_half, freq_half[partner_indices]), axis=0)
    mode_full = np.concatenate((mode_half, mode_half[partner_indices].conj()), axis=0)
    output.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output, "x") as f:
        f.create_dataset("gmat_raw", data=gmat_raw)
        f.create_dataset("rvec_set_ph_eph", data=ep.rvec_set_ph_eph)
        f.create_dataset("rvec_set_el", data=ep.rvec_set_el)
        f.create_dataset("mass", data=phdisp.mass)
        f.create_dataset("freq_full", data=freq_full)
        f.create_dataset("mode_full", data=mode_full)
        f.create_dataset("q_hbz", data=q_hbz)
        f.create_dataset("q_minus", data=q_minus)
        f.create_dataset("q_full", data=q_full)
        f.create_dataset("partner_hbz_for_minus", data=partner_indices)
        f.create_dataset("rph", data=rgrid_2d_full(args.nx, ny))
        f.attrs["source_epr_file"] = args.epr_file.name
        f.attrs["nx"] = args.nx
        f.attrs["ny"] = ny
        f.attrs["mass_weighted_modes"] = False
        f["freq_full"].attrs["units"] = "Ry"
        f["q_full"].attrs["units"] = "fractional reciprocal coordinates"
    print(f"Wrote {output}: {len(q_full)} q-points, {3 * len(phdisp.mass)} modes")


if __name__ == "__main__":
    main()
