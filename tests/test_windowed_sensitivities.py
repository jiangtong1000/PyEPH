"""Window boundary sensitivities need full state tangents and independent equations."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


_PATH = Path(__file__).resolve().parents[1]/"benchmarks/windowed_sensitivities.py"
_SPEC = importlib.util.spec_from_file_location("windowed_sensitivities_benchmark", _PATH)
benchmark = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(benchmark)


@pytest.mark.parametrize("method", ["cpa", "ehrenfest"])
def test_short_window_oracle_and_detached_state_counterexample(tmp_path, method):
    problem, initial = benchmark.fixture(method)
    functions = benchmark.objectives(problem, initial, steps=16, window=4)
    report = benchmark.short_checks(problem, initial, functions, 16, tmp_path)
    assert all(report["gates"].values()), report
    assert report["errors"]["detached_value_error"] <= 1e-10
    assert report["errors"]["detached_gradient_relative_error"] > 1e-3
    with np.load(tmp_path/"short-derivatives.npz", allow_pickle=False) as arrays:
        assert all(np.isfinite(arrays[name]).all() for name in arrays.files)
        assert arrays["fd_widths"].tolist() == [1e-3, 5e-4]
        assert arrays["hvp"].shape == arrays["gradient"].shape
        assert arrays["fd_gradient_pairs"].shape == (2, 2, arrays["gradient"].size)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_nonfinite_raw_evidence_is_saved_and_rejected(tmp_path, bad):
    expected = np.array([1., bad])
    with pytest.raises(FloatingPointError, match="nonfinite deliberate evidence retained"):
        benchmark.retain_arrays(tmp_path, "deliberate", {"values": expected})
    with np.load(tmp_path/"deliberate.npz", allow_pickle=False) as arrays:
        np.testing.assert_array_equal(arrays["values"], expected)
    record = json.loads((tmp_path/"deliberate-finiteness.json").read_text())
    assert record["values"]["finite"] is False
    assert record["values"]["nonfinite_count"] == 1
    with pytest.raises(FileExistsError):
        benchmark.retain_arrays(tmp_path, "deliberate", {"values": np.ones(2)})
