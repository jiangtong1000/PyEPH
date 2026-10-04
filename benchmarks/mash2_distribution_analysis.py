"""Read saved MASH ensembles and compare distributions to the FFT reference.

Inputs are never modified. This reports errors and sampling uncertainty; it
does not turn agreement with an internally computed reference into paper-curve
reproduction. Run after all four low/high, dt1/dt05 ensembles have completed.

The complete generation recipe and input filenames are documented in
docs/MASH2.md under "Wavepacket distribution comparison".

"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import ks_2samp

from benchmarks.mash2_quantum_validation import QUANTILE_LEVELS, distribution_cdf
from benchmarks.mash2_wavepacket import CASES, FINAL_TIME, MASS


def load_completed_results(directory):
    """Load and validate the common physical experiment before any comparison.

    Nuclear samples are compared byte-for-byte. Mapping pairing is supported
    by the shared seed+1, row IDs, code and JAX version, since the benchmark
    does not separately persist initial mapping vectors or PRNG configuration.
    """
    directory = Path(directory)
    quantum_path = directory/"mash2_quantum_validation.json"
    quantum_data_path = directory/"mash2_quantum_validation.npz"
    inputs = [quantum_path, quantum_data_path]
    for case in CASES:
        for step in ("1", "05"):
            inputs.extend((directory/f"mash2_wavepacket_{case}_dt{step}.json",
                           directory/f"mash2_wavepacket_{case}_dt{step}.npz"))
    missing = [str(path) for path in inputs if not path.is_file()]
    if missing:
        raise ValueError("required completed benchmark files are missing: "+", ".join(missing))
    quantum = json.loads(quantum_path.read_text())
    if (quantum.get("source_unchanged") is not True
            or not np.isclose(quantum["duration_atomic"], FINAL_TIME, rtol=0, atol=1e-9)
            or quantum["duration_fs"] != 150. or quantum["mass_atomic"] != MASS):
        raise ValueError("quantum provenance must describe the unchanged 150fs, mass2000 reference")
    if hashlib.sha256(quantum_data_path.read_bytes()).hexdigest() != quantum["plotting_data"]["sha256"]:
        raise ValueError("quantum NPZ does not match its recorded SHA256")
    if quantum["plotting_data"]["level"] != "time_half":
        raise ValueError("the comparison requires the finer-time quantum reference")
    with np.load(quantum_data_path) as archive:
        reference = {name: value.copy() for name, value in archive.items()}
    data, metadata = {}, {}
    source_identity = versions = None
    for case, (p0, gamma) in CASES.items():
        qcase = quantum["cases"][case]
        for name, expected in (("q0", -15.), ("p0", p0), ("gamma", gamma)):
            if not np.isclose(qcase[name], expected, rtol=0, atol=1e-12):
                raise ValueError(f"quantum preparation differs in {case}.{name}")
        for coordinate, grid_name in (("q", "q"), ("p", "momentum")):
            grid = reference[case+"_"+grid_name]
            probability = reference[case+"_probability_"+coordinate]
            density = reference[case+"_density_"+coordinate]
            if (grid.ndim != 1 or len(grid) < 2 or probability.shape != grid.shape
                    or density.shape != grid.shape or not np.isfinite(grid).all()
                    or not np.isfinite(probability).all() or not np.isfinite(density).all()
                    or np.any(probability < 0) or np.any(density < 0)
                    or not np.all(np.diff(grid) > 0)
                    or not np.allclose(np.diff(grid), grid[1]-grid[0], rtol=1e-10, atol=1e-12)
                    or not np.isclose(probability.sum(), 1., rtol=0, atol=1e-8)
                    or not np.allclose(density*(grid[1]-grid[0]), probability, rtol=1e-10, atol=1e-14)):
                raise ValueError("quantum grid, probability masses and density normalization disagree")
        for step, requested_dt in (("1", 1.), ("05", .5)):
            path = directory/f"mash2_wavepacket_{case}_dt{step}"
            entry = json.loads(path.with_suffix(".json").read_text())
            if entry["case"] != case or entry.get("source_unchanged") is not True:
                raise ValueError("MASH case or unchanged-source declaration does not match")
            for name, expected in (("mass", MASS), ("duration", FINAL_TIME),
                                   ("q0", -15.), ("p0", p0), ("gamma", gamma)):
                if not np.isclose(entry[name], expected, rtol=0, atol=1e-9 if name == "duration" else 1e-12):
                    raise ValueError(f"MASH preparation differs in {case}.{name}")
            identity = entry["source_sha256"]
            if (not isinstance(identity, dict) or not identity
                    or any(not isinstance(value, str) or len(value) != 64
                           or any(c not in "0123456789abcdef" for c in value) for value in identity.values())):
                raise ValueError("MASH source identity must contain SHA256 entries")
            runtime = tuple(entry[name] for name in ("python", "jax", "numpy"))
            if source_identity is None:
                source_identity, versions = identity, runtime
            if source_identity != identity or versions != runtime:
                raise ValueError("all MASH runs must share source identity and Python/JAX/NumPy versions")
            stats = entry["runs"]["mash"]
            n = stats["trajectories"]
            if type(n) is not int or n < 1 or type(stats["seed"]) is not int:
                raise ValueError("MASH trajectory count and mapping seed must be exact integers")
            expected_steps = int(np.ceil(FINAL_TIME/requested_dt))
            if stats["steps"] != expected_steps or not np.isclose(stats["dt"], FINAL_TIME/expected_steps, rtol=0, atol=1e-12):
                raise ValueError("MASH dt/steps do not match the labeled refinement")
            with np.load(path.with_suffix(".npz")) as archive:
                arrays = {name: archive["mash_"+name].copy()
                          for name in ("initial_q", "initial_p", "final_q", "final_p", "active")}
            if any(value.shape != (n,) or np.iscomplexobj(value) or not np.isfinite(value).all()
                   for value in arrays.values()):
                raise ValueError("MASH sample arrays must be finite real vectors matching the trajectory count")
            active = arrays["active"]
            if active.dtype.kind not in "iu" or np.any((active != 0) & (active != 1)):
                raise ValueError("MASH active surfaces must be integer 0 or 1")
            population = np.bincount(active, minlength=2)/n
            if not np.allclose(population, stats["final_population"], rtol=0, atol=1e-14):
                raise ValueError("MASH saved surfaces and reported populations disagree")
            data[case, step], metadata[case, step] = arrays, entry
        coarse, fine = data[case, "1"], data[case, "05"]
        coarse_stats, fine_stats = (metadata[case, step]["runs"]["mash"] for step in ("1", "05"))
        for name in ("trajectories", "seed", "event_substeps"):
            if coarse_stats[name] != fine_stats[name]:
                raise ValueError("paired refinements must use identical counts, mapping seeds and event subdivision")
        if any(coarse[name].dtype != fine[name].dtype
               or coarse[name].tobytes(order="C") != fine[name].tobytes(order="C")
               for name in ("initial_q", "initial_p")):
            raise ValueError("paired refinements must contain identical initial nuclear samples")
    return quantum, reference, data, metadata, inputs


def empirical_comparison(samples, grid, probability, *, simultaneous_comparisons=8, bins=64):
    samples = np.sort(np.asarray(samples, dtype=float))
    if not len(samples) or not np.isfinite(samples).all():
        raise ValueError("samples must be nonempty and finite")
    edges, cumulative = distribution_cdf(grid, probability)
    reference_at_samples = np.interp(samples, edges, cumulative)
    n = len(samples)
    ks = np.max(np.maximum(np.arange(1, n+1)/n-reference_at_samples,
                           reference_at_samples-np.arange(n)/n))
    # The empirical CDF is constant between samples, whereas the reference CDF
    # is piecewise linear. Integrate their absolute difference exactly on the
    # joint intervals, including intervals containing a sign change.
    union = np.union1d(samples, edges)
    empirical = np.searchsorted(samples, union[:-1], side="right")/n
    reference = np.interp(union, edges, cumulative)
    left, right = empirical-reference[:-1], empirical-reference[1:]
    area = .5*(np.abs(left)+np.abs(right))
    crossing = left*right < 0
    area[crossing] = .5*(left[crossing]**2+right[crossing]**2)/np.abs(right[crossing]-left[crossing])
    wasserstein = np.sum(area*np.diff(union))
    reference_quantiles = np.interp(QUANTILE_LEVELS, cumulative, edges)
    sample_quantiles = np.quantile(samples, QUANTILE_LEVELS)
    # Include explicit infinite tail bins so no finite plotting range silently
    # discards probability. L1 depends on this chosen binning and is reported
    # alongside the histogram-independent CDF distance.
    limits = np.interp([.001, .999], cumulative, edges)
    histogram_edges = np.concatenate(([-np.inf], np.linspace(*limits, bins+1), [np.inf]))
    observed = np.histogram(samples, histogram_edges)[0]/n
    expected = np.diff(np.interp(histogram_edges, edges, cumulative))
    return dict(n=n, cdf_supremum=float(ks), wasserstein_1=float(wasserstein),
                dkw_95_single_marginal=float(np.sqrt(np.log(2/.05)/(2*n))),
                dkw_95_bonferroni_all_marginals=float(np.sqrt(np.log(2*simultaneous_comparisons/.05)/(2*n))),
                histogram_l1_mass_difference=float(np.sum(np.abs(observed-expected))),
                histogram_finite_bins=bins, histogram_finite_limits=limits.tolist(),
                histogram_lower_tail_mass=float(observed[0]), histogram_upper_tail_mass=float(observed[-1]),
                quantile_levels=QUANTILE_LEVELS, empirical_quantiles=sample_quantiles.tolist(),
                quantum_quantiles=reference_quantiles.tolist(),
                quantile_errors=(sample_quantiles-reference_quantiles).tolist(),
                empirical_mean=float(np.mean(samples)),
                mean_standard_error=float(np.std(samples, ddof=1)/np.sqrt(n)) if n > 1 else None,
                quantum_mean=float(grid@probability/probability.sum()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("benchmarks/results"))
    parser.add_argument("--output", type=Path, default=Path("benchmarks/results/mash2_distribution_analysis.json"))
    args = parser.parse_args()
    try:
        quantum, reference, data, metadata, inputs = load_completed_results(args.results)
    except ValueError as error:
        parser.error(str(error))
    report = dict(input_sha256={str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in inputs},
                  analysis_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  quantum_level="time_half", duration_fs=150.,
                  interpretation="Distribution discrepancies include method error and Monte Carlo sampling; DKW bands bound sampling alone under IID preparation.",
                  mapping_pairing_evidence="same seed+1, trajectory row IDs, code and JAX version; initial mapping vectors and PRNG configuration were not separately persisted",
                  cases={})
    for case in ("low", "high"):
        entry, trajectories = dict(runs={}), {}
        qstats = quantum["cases"][case]["levels"]["time_half"]
        qpopulation = np.asarray(qstats["final_population"])/qstats["final_norm"]
        for step in ("1", "05"):
            stats = metadata[case, step]["runs"]["mash"]
            arrays = data[case, step]
            trajectories[step] = arrays
            population = np.bincount(arrays["active"], minlength=2)/len(arrays["active"])
            np.testing.assert_allclose(population, stats["final_population"], atol=1e-14)
            standard_error = np.sqrt(population*(1-population)/len(arrays["active"]))
            entry["runs"]["dt"+step] = dict(
                dt=stats["dt"], trajectories=stats["trajectories"], seed=stats["seed"],
                population=population.tolist(), quantum_population=qpopulation.tolist(),
                population_error=(population-qpopulation).tolist(),
                population_standard_error=standard_error.tolist(),
                distributions={coordinate: empirical_comparison(
                    arrays["final_"+coordinate], reference[case+"_"+grid],
                    reference[case+"_probability_"+coordinate])
                    for coordinate, grid in (("q", "q"), ("p", "momentum"))})
        coarse, fine = trajectories["1"], trajectories["05"]
        for key in ("initial_q", "initial_p"):
            np.testing.assert_array_equal(coarse[key], fine[key], err_msg="time-step comparison requires matched initial samples")
        entry["matched_timestep_refinement"] = dict(
            identical_initial_nuclear_samples=True,
            changed_final_active_count=int(np.count_nonzero(coarse["active"] != fine["active"])),
            nuclear={coordinate: dict(
                mean_absolute_shift=float(np.mean(np.abs(coarse["final_"+coordinate]-fine["final_"+coordinate]))),
                maximum_absolute_shift=float(np.max(np.abs(coarse["final_"+coordinate]-fine["final_"+coordinate]))),
                empirical_cdf_distance=float(ks_2samp(coarse["final_"+coordinate], fine["final_"+coordinate]).statistic))
                for coordinate in ("q", "p")})
        report["cases"][case] = entry
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report["cases"], indent=2))


if __name__ == "__main__":
    main()
