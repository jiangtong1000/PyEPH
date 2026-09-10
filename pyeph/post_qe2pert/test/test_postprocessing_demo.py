"""Exercise the documented extraction command with the bundled DNTT input."""

import os
from pathlib import Path
import subprocess
import sys

import h5py
import numpy as np


def test_dntt_postprocessing_demo(tmp_path):
    root = Path(__file__).resolve().parents[3]
    source = Path(__file__).with_name("DNTT_epr.h5")
    output = tmp_path / "dntt_demo.h5"
    command = [
        sys.executable,
        str(root / "examples/01_abinitio_realspace_EPC/7_PolarEPH/run_pyeph.py"),
        "--epr-file", str(source), "--nx", "2", "--ny", "2",
        "--output", str(output),
    ]
    env = os.environ.copy()
    env.update(PYTHONPATH=str(root), USE_MPI="false")
    run = subprocess.run(
        command, cwd=tmp_path, env=env, capture_output=True, text=True, timeout=120,
    )
    assert run.returncode == 0, run.stdout + run.stderr
    with h5py.File(output) as f, h5py.File(source) as original:
        assert f["freq_full"].shape == (4, 216)
        assert f["mode_full"].shape == (4, 216, 216)
        np.testing.assert_array_equal(f["mass"][:], original["basic_data/mass"][:])
        for key in ("gmat_raw", "freq_full", "mode_full", "mass"):
            assert np.isfinite(f[key][:]).all(), key
        modes = f["mode_full"][:]
        partners = f["partner_hbz_for_minus"][:]
        nq_half = len(f["q_hbz"])
        np.testing.assert_allclose(modes[nq_half:], modes[partners].conj())
        np.testing.assert_allclose(f["q_minus"][:], -f["q_hbz"][:][partners])
        for mode in modes:
            np.testing.assert_allclose(mode.conj().T @ mode, np.eye(216), atol=1e-10)

    # A repeated command must leave the existing scientific output intact.
    before = output.read_bytes()
    repeated = subprocess.run(
        command, cwd=tmp_path, env=env, capture_output=True, text=True, timeout=30,
    )
    assert repeated.returncode != 0
    assert "output already exists" in repeated.stderr
    assert output.read_bytes() == before
