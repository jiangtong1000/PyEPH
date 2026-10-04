"""Check distribution metrics against exactly integrable probability laws."""

import hashlib
import json

import numpy as np
import pytest

from benchmarks.mash2_distribution_analysis import empirical_comparison, load_completed_results
from benchmarks.mash2_quantum_validation import distribution_difference
from benchmarks.mash2_wavepacket import CASES, FINAL_TIME, MASS


def test_uniform_density_refinement_does_not_change_bin_cdf():
    result = distribution_difference(np.array([.5, 1.5]), np.array([.5, .5]),
                                     np.array([.25, .75, 1.25, 1.75]), np.full(4, .25))
    assert result["cdf_supremum"] == 0.
    assert result["wasserstein_1_piecewise_uniform"] == 0.


def test_empirical_midpoint_quantiles_against_uniform_reference():
    # For n uniformly spaced midpoint atoms on [0,1], D=1/(2n) and W1=1/(4n).
    n = 20
    result = empirical_comparison((np.arange(n)+.5)/n, np.array([.25, .75]), np.array([.5, .5]))
    np.testing.assert_allclose(result["cdf_supremum"], 1/(2*n), atol=1e-15)
    np.testing.assert_allclose(result["wasserstein_1"], 1/(4*n), atol=1e-15)
    assert result["dkw_95_bonferroni_all_marginals"] > result["dkw_95_single_marginal"]


def test_empirical_tail_atoms_are_counted_in_distance_and_histogram():
    result = empirical_comparison([-1., 2.], np.array([.25, .75]), np.array([.5, .5]))
    np.testing.assert_allclose(result["cdf_supremum"], .5, atol=1e-15)
    np.testing.assert_allclose(result["wasserstein_1"], 1.25, atol=1e-15)
    assert result["histogram_lower_tail_mass"] == .5
    assert result["histogram_upper_tail_mass"] == .5


def _write_validation_inputs(directory):
    """Synthetic arrays test ingestion contracts, not physical agreement."""
    quantum = dict(source_unchanged=True, duration_atomic=FINAL_TIME, duration_fs=150.,
                   mass_atomic=MASS, cases={})
    reference = {}
    for case, (p0, gamma) in CASES.items():
        quantum["cases"][case] = dict(q0=-15., p0=float(p0), gamma=gamma,
                                      levels={"time_half": dict(final_population=[.5, .5], final_norm=1.)})
        for coordinate, grid in (("q", "q"), ("p", "momentum")):
            reference[case+"_"+grid] = np.arange(4.)
            reference[case+"_probability_"+coordinate] = np.full(4, .25)
            reference[case+"_density_"+coordinate] = np.full(4, .25)
        for step, requested in (("1", 1.), ("05", .5)):
            count = 12
            steps = int(np.ceil(FINAL_TIME/requested))
            metadata = dict(case=case, duration=FINAL_TIME, mass=MASS, q0=-15.,
                            p0=float(p0), gamma=gamma, source_unchanged=True,
                            source_sha256={"example.py": "a"*64}, python="test", jax="test", numpy="test",
                            runs={"mash": dict(trajectories=count, seed=7, event_substeps=2,
                                               steps=steps, dt=FINAL_TIME/steps, final_population=[.5, .5])})
            path = directory/f"mash2_wavepacket_{case}_dt{step}"
            path.with_suffix(".json").write_text(json.dumps(metadata))
            arrays = dict(mash_initial_q=np.linspace(-16., -14., count),
                          mash_initial_p=np.linspace(p0-.1, p0+.1, count),
                          mash_final_q=np.linspace(0., 3., count),
                          mash_final_p=np.linspace(0., 3., count), mash_active=np.tile([0, 1], count//2))
            np.savez(path.with_suffix(".npz"), **arrays)
    data_path = directory/"mash2_quantum_validation.npz"
    np.savez(data_path, **reference)
    quantum["plotting_data"] = dict(level="time_half", sha256=hashlib.sha256(data_path.read_bytes()).hexdigest())
    (directory/"mash2_quantum_validation.json").write_text(json.dumps(quantum))


def test_saved_experiment_validation_accepts_matched_records(tmp_path):
    _write_validation_inputs(tmp_path)
    _, _, data, metadata, paths = load_completed_results(tmp_path)
    assert len(paths) == 10 and len(data) == len(metadata) == 4


@pytest.mark.parametrize("fault,match", [
    ("duration", "preparation"), ("gamma", "preparation"),
    ("source", "source identity"), ("seed", "mapping seeds"),
    ("dt", "dt/steps"), ("count", "trajectory count"),
    ("pairing", "initial nuclear"), ("data_hash", "SHA256"),
])
def test_saved_experiment_validation_rejects_mislabeled_or_unpaired_data(tmp_path, fault, match):
    _write_validation_inputs(tmp_path)
    path = tmp_path/"mash2_wavepacket_high_dt05.json"
    metadata = json.loads(path.read_text())
    if fault in ("duration", "gamma"):
        metadata[fault] += 1
    elif fault == "source":
        metadata["source_sha256"]["example.py"] = "b"*64
    elif fault == "seed":
        metadata["runs"]["mash"]["seed"] += 1
    elif fault == "dt":
        metadata["runs"]["mash"]["dt"] = 1.
    elif fault == "count":
        metadata["runs"]["mash"]["trajectories"] += 1
    elif fault == "pairing":
        with np.load(path.with_suffix(".npz")) as archive:
            arrays = {key: value.copy() for key, value in archive.items()}
        arrays["mash_initial_q"][0] += .1
        np.savez(path.with_suffix(".npz"), **arrays)
    elif fault == "data_hash":
        with open(tmp_path/"mash2_quantum_validation.npz", "ab") as handle:
            handle.write(b"unexpected extra data")
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match=match):
        load_completed_results(tmp_path)
