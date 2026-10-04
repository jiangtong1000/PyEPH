"""Public ingestion-to-current workflow, restart continuity and ensemble provenance."""

from dataclasses import replace
import importlib.util
from pathlib import Path
import sys

import h5py
import numpy as np
import pytest

from pyeph.io.checkpoint import load_checkpoint
from test_epr_adapter import synthetic_epr


_PATH = Path(__file__).resolve().parents[1]/"examples/abinitio_epc.py"
_SPEC = importlib.util.spec_from_file_location("abinitio_epc_example", _PATH)
example = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = example
_SPEC.loader.exec_module(example)


def test_streamed_full_thermal_response_restart_and_initialization_identity(tmp_path):
    source = tmp_path/"source.h5"
    synthetic_epr(source)
    inputs = example.Inputs(hermiticity="project", trajectories=2, dt_fs=.001, steps=4)
    whole = example.run_workflow(source, inputs, tmp_path/"whole")
    first = example.run_workflow(source, replace(inputs, steps=2), tmp_path/"first")
    second = example.run_workflow(source, replace(inputs, steps=2), tmp_path/"second",
                                  resume=tmp_path/"first/checkpoint.h5")
    assert second["final_time_fs"] == whole["final_time_fs"]
    assert max(run["maximum_unitarity_error"] for run in (whole, first, second)) < 1e-9
    assert second["conventions"]["ingestion"]["source"]["complex_values_retained"]
    checkpoint, _ = load_checkpoint(tmp_path/"first/checkpoint.h5")
    with np.load(tmp_path/"second/segment_start_state.npz", allow_pickle=False) as starting:
        for name in starting.files:
            expected = np.asarray(getattr(checkpoint, "trajectory_id" if name == "trajectory_ids"
                                           else name))
            actual = starting[name]
            assert actual.shape == expected.shape and actual.dtype == expected.dtype
            assert np.isfinite(actual).all() and np.isfinite(expected).all()
            assert actual.tobytes() == expected.tobytes()
    with h5py.File(tmp_path/"whole/trajectory.h5") as full:
        correlation_scale = float(np.max(np.abs(full["observables/current_correlation"][:])))
    scales = {"current_correlation": correlation_scale, "mean_current_correlation": correlation_scale,
              "sem_real": correlation_scale, "sem_imag": correlation_scale, "unitary_error": 1.}
    for file in ("trajectory.h5", "ensemble.h5"):
        with h5py.File(tmp_path/"whole"/file) as full, \
                h5py.File(tmp_path/"first"/file) as part1, \
                h5py.File(tmp_path/"second"/file) as part2:
            names = ["time", *[f"observables/{key}" for key in full["observables"]]]
            for name in names:
                joined = np.concatenate((part1[name][:], part2[name][1:]), axis=0)
                expected = full[name][:]
                assert expected.shape == joined.shape and expected.dtype == joined.dtype
                assert np.isfinite(expected).all() and np.isfinite(joined).all()
                if name == "time" or expected.dtype.kind in "biu":
                    assert expected.tobytes() == joined.tobytes()
                else:
                    # A declared numerical allowance for recomputed reductions,
                    # in correlation units (also SEM) or the unitary identity's
                    # dimensionless scale. Serialization above remains exact.
                    allowance = 128*np.finfo(expected.real.dtype).eps
                    np.testing.assert_allclose(joined, expected, rtol=allowance,
                        atol=allowance*scales[name.split("/")[-1]], equal_nan=False)
    # Nuclear temperature, sampling distribution and RNG identity cannot be
    # reconstructed from a Hamiltonian/checkpoint fingerprint alone.
    for changed in ({"temperature_kelvin": 100.}, {"distribution": "wigner"}, {"seed": 9}):
        with pytest.raises(ValueError, match="preserve all inputs"):
            example.run_workflow(source, replace(inputs, **changed), tmp_path/"invalid",
                                 resume=tmp_path/"first/checkpoint.h5")
    assert not (tmp_path/"invalid").exists()
