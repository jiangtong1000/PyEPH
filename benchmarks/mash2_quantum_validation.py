"""Convergence evidence for the independent modified-Tully FFT reference.

Run from the repository root with ``PYTHONPATH=src python -m
benchmarks.mash2_quantum_validation``. Both published preparations are evolved
for 150 fs. This computes new reference data; it does not digitize paper curves.
"""

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform

import numpy as np
from scipy.signal import find_peaks

from benchmarks.mash2_wavepacket import CASES, FINAL_TIME, MASS, quantum_run


LEVELS = {
    "baseline": dict(points=8192, dt=.5, limits=(-50., 90.)),
    "time_half": dict(points=8192, dt=.25, limits=(-50., 90.)),
    "grid_double": dict(points=16384, dt=.5, limits=(-50., 90.)),
    "box_wider": dict(points=12288, dt=.5, limits=(-85., 125.)),
}
QUANTILE_LEVELS = [.01, .05, .25, .5, .75, .95, .99]


def distribution_cdf(grid, probability):
    """CDF of masses uniformly spread over their centered grid cells.

    This interpolation is explicit because comparing unequal grid meshes can
    otherwise confuse a bin-width effect with quantum propagation error.
    """
    grid, probability = np.asarray(grid), np.asarray(probability)
    spacing = grid[1]-grid[0]
    edges = np.concatenate(([grid[0]-spacing/2], grid+spacing/2))
    cumulative = np.concatenate(([0.], np.cumsum(probability)/np.sum(probability)))
    return edges, cumulative


def distribution_summary(grid, probability):
    edges, cumulative = distribution_cdf(grid, probability)
    spacing = grid[1]-grid[0]
    density = probability/spacing
    peaks, _ = find_peaks(density, prominence=.05*density.max(),
                         distance=max(1, int(np.ceil(.25/spacing))))
    return dict(norm=float(probability.sum()),
                mean=float(grid@probability/probability.sum()),
                quantile_levels=QUANTILE_LEVELS,
                quantiles=np.interp(QUANTILE_LEVELS, cumulative, edges).tolist(),
                modes_above_five_percent_prominence=grid[peaks].tolist())


def distribution_difference(grid, probability, refined_grid, refined_probability):
    edges, cumulative = distribution_cdf(grid, probability)
    fine_edges, fine_cumulative = distribution_cdf(refined_grid, refined_probability)
    union = np.union1d(edges, fine_edges)
    differences = np.abs(np.interp(union, edges, cumulative)
                         - np.interp(union, fine_edges, fine_cumulative))
    return dict(mass_difference=float(abs(probability.sum()-refined_probability.sum())),
                cdf_supremum=float(differences.max()),
                wasserstein_1_piecewise_uniform=float(np.trapezoid(differences, union)),
                max_quantile_difference=float(np.max(np.abs(
                    np.interp(QUANTILE_LEVELS, cumulative, edges)
                    - np.interp(QUANTILE_LEVELS, fine_cumulative, fine_edges)))))


def compare(base_stats, base, refined_stats, refined):
    result = {coordinate: distribution_difference(base[grid], base["probability_"+coordinate],
                                                  refined[grid], refined["probability_"+coordinate])
              for coordinate, grid in (("q", "q"), ("p", "momentum"))}
    result["population_max_difference"] = float(np.max(np.abs(
        np.asarray(base_stats["final_population"])-refined_stats["final_population"])))
    # These refinement grids have common position samples. Compare the full
    # two-component complex wavefunction, correcting discrete sqrt(dx) weights.
    indices = np.rint((base["q"]-refined["q"][0])/refined_stats["dx"]).astype(int)
    np.testing.assert_allclose(refined["q"][indices], base["q"], atol=1e-12)
    sampled = refined["psi"][indices]*np.sqrt(base_stats["dx"]/refined_stats["dx"])
    result["wavefunction_l2_common_position_mesh"] = float(np.linalg.norm(base["psi"]-sampled))
    sampled_probability = np.sum(np.abs(sampled)**2, axis=1)
    result["common_position_mesh"] = distribution_difference(
        base["q"], base["probability_q"], base["q"], sampled_probability)
    result["common_position_mesh"]["l1_mass_difference"] = float(np.sum(np.abs(
        base["probability_q"]-sampled_probability)))
    if (len(refined["q"]) > len(base["q"])
            and np.isclose(base_stats["dx"], refined_stats["dx"], rtol=0, atol=1e-12)):
        # Box enlargement changes dp. Zero-pad the baseline state into the
        # wider same-dx position mesh before Fourier comparison, so a different
        # probability-bin width is not mistaken for a propagation error.
        embedded = np.zeros_like(refined["psi"])
        embedded[indices] = base["psi"]
        common_probability = np.sum(np.abs(np.fft.fftshift(
            np.fft.fft(embedded, axis=0, norm="ortho"), axes=0))**2, axis=1)
        result["box_common_momentum_mesh"] = distribution_difference(
            refined["momentum"], common_probability,
            refined["momentum"], refined["probability_p"])
        result["box_common_momentum_mesh"]["l1_mass_difference"] = float(np.sum(np.abs(
            common_probability-refined["probability_p"])))
        result["box_common_momentum_mesh"]["definition"] = (
            "baseline state zero-padded in the wider same-dx position mesh; both transformed on its momentum mesh")
        result["wavefunction_l2_zero_padded_wider_box"] = float(np.linalg.norm(embedded-refined["psi"]))
    if np.array_equal(base["q"], refined["q"]):
        for coordinate in ("q", "p"):
            result[coordinate]["same_mesh_l1_mass_difference"] = float(np.sum(np.abs(
                base["probability_"+coordinate]-refined["probability_"+coordinate])))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=Path("benchmarks/results/mash2_quantum_validation"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    source_paths = [Path(__file__).resolve(), root/"benchmarks/mash2_wavepacket.py",
                    root/"tests/test_mash2_wavepacket.py"]
    hashes = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
              for path in source_paths}
    report = dict(paper="https://arxiv.org/html/2212.11773#S4.SS1.SSS1",
                  reference_kind="independently computed FFT reference; no digitized paper data",
                  mass_atomic=MASS, duration_atomic=FINAL_TIME, duration_fs=150.,
                  source_sha256=hashes, platform=platform.platform(),
                  python=platform.python_version(),
                  dependencies={name: importlib.metadata.version(name) for name in ("numpy", "scipy", "jax")},
                  cdf_definition="normalized probability masses spread uniformly over centered cells",
                  cases={})
    arrays = {}
    for case, (p0, gamma) in CASES.items():
        entry = dict(q0=-15., p0=float(p0), gamma=gamma, levels={}, comparisons={})
        values = {}
        for name, configuration in LEVELS.items():
            stats, data = quantum_run(case, **configuration)
            stats["distributions"] = {coordinate: distribution_summary(data[grid], data["probability_"+coordinate])
                                      for coordinate, grid in (("q", "q"), ("p", "momentum"))}
            entry["levels"][name], values[name] = stats, data
            print(f"{case}: {name}, population={stats['final_population']}, norm={stats['final_norm']:.12g}", flush=True)
        for name in LEVELS:
            if name != "baseline":
                entry["comparisons"][name] = compare(entry["levels"]["baseline"], values["baseline"],
                                                       entry["levels"][name], values[name])
        report["cases"][case] = entry
        # Persist plotting arrays for the finest time step at the baseline box;
        # complete convergence diagnostics remain in JSON, avoiding duplicate psi.
        arrays.update({case+"_"+key: value for key, value in values["time_half"].items()})
    report["source_unchanged"] = all(hashlib.sha256((root/name).read_bytes()).hexdigest() == digest
                                     for name, digest in hashes.items())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    data_path = args.output.with_suffix(".npz")
    np.savez_compressed(data_path, **arrays)
    report["plotting_data"] = dict(path=str(data_path), level="time_half",
                                  sha256=hashlib.sha256(data_path.read_bytes()).hexdigest())
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({"source_unchanged": report["source_unchanged"],
                      "comparisons": {case: entry["comparisons"] for case, entry in report["cases"].items()}}, indent=2))


if __name__ == "__main__":
    main()
