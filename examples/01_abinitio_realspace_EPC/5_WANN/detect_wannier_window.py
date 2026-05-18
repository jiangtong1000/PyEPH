"""Detect Wannier windows from a Quantum ESPRESSO NSCF output file.

This script targets disentangling the HOMO-1/HOMO pair from HOMO-2 and LUMO,
which is used for modeling the hole transport in organic crystals.
"""

import argparse
import re
import sys
from pathlib import Path


FLOAT_RE = re.compile(r"[-+]?(?:\d+\.\d*|\.\d+)(?:[Ee][-+]?\d+)?")
HOMO_LUMO_RE = re.compile(
    r"highest occupied,\s*lowest unoccupied level\s*\(ev\):\s*([-\d.]+)\s+([-\d.]+)",
    re.IGNORECASE,
)
NELEC_RE = re.compile(r"number of electrons\s*=\s*([-\d.]+)", re.IGNORECASE)
NBND_RE = re.compile(r"number of Kohn-Sham states\s*=\s*(\d+)", re.IGNORECASE)


def parse_nscf(path: Path) -> dict:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()

    homo_e = None
    lumo_e = None
    nelec = None
    nbnd_from_header = None
    k_blocks = []

    i = 0
    while i < len(lines):
        line = lines[i]

        nele_match = NELEC_RE.search(line)
        if nele_match:
            nelec = float(nele_match.group(1))

        nbnd_match = NBND_RE.search(line)
        if nbnd_match:
            nbnd_from_header = int(nbnd_match.group(1))

        hl_match = HOMO_LUMO_RE.search(line)
        if hl_match:
            homo_e = float(hl_match.group(1))
            lumo_e = float(hl_match.group(2))

        if "k =" in line and "bands (ev):" in line:
            coords = [float(v) for v in FLOAT_RE.findall(line)[:3]]
            i += 1
            energies = []

            while i < len(lines):
                s = lines[i].strip()
                if not s:
                    if energies:
                        break
                    i += 1
                    continue

                vals = FLOAT_RE.findall(lines[i])
                if vals:
                    energies.extend(float(v) for v in vals)
                    i += 1
                    continue

                if energies:
                    break
                i += 1

            if energies:
                k_blocks.append({"k_index": len(k_blocks) + 1, "coords": coords, "energies": energies})
            continue

        i += 1

    if homo_e is None or lumo_e is None:
        raise ValueError("Could not find HOMO/LUMO summary line in nscf.out")
    if not k_blocks:
        raise ValueError("Could not find any 'k = ... bands (ev):' blocks in nscf.out")

    nbnd_values = {len(kb["energies"]) for kb in k_blocks}
    if len(nbnd_values) != 1:
        raise ValueError(
            f"Inconsistent band counts across k-points: {sorted(nbnd_values)}"
        )
    nbnd_from_bands = nbnd_values.pop()

    return {
        "homo_e": homo_e,
        "lumo_e": lumo_e,
        "nelec": nelec,
        "nbnd_from_header": nbnd_from_header,
        "nbnd": nbnd_from_bands,
        "k_blocks": k_blocks,
    }


def find_band_indices(k_blocks: list, homo_e: float, lumo_e: float) -> tuple[int, int, float, float]:
    nbnd = len(k_blocks[0]["energies"])
    nk = len(k_blocks)

    homo_band_max = []
    lumo_band_min = []
    for ib in range(nbnd):
        band_vals = [k_blocks[ik]["energies"][ib] for ik in range(nk)]
        homo_band_max.append(max(band_vals))
        lumo_band_min.append(min(band_vals))

    homo_idx = min(range(nbnd), key=lambda i: abs(homo_band_max[i] - homo_e))
    lumo_idx = min(range(nbnd), key=lambda i: abs(lumo_band_min[i] - lumo_e))

    homo_err = abs(homo_band_max[homo_idx] - homo_e)
    lumo_err = abs(lumo_band_min[lumo_idx] - lumo_e)

    return homo_idx + 1, lumo_idx + 1, homo_err, lumo_err


def closest_match_for_band(k_blocks: list, band_index_1based: int, target_e: float) -> tuple[int, list[float], float, float]:
    ib = band_index_1based - 1
    best = None
    for kb in k_blocks:
        e = kb["energies"][ib]
        diff = abs(e - target_e)
        if best is None or diff < best[3]:
            best = (kb["k_index"], kb["coords"], e, diff)
    return best


def fmt_list(vals: list[float], ndigits: int = 4) -> str:
    return "[" + ", ".join(f"{v:.{ndigits}f}" for v in vals) + "]"


def classify_margin(margin: float) -> str:
    if margin <= 0.0:
        return "OVERLAP (not safe)"
    if margin < 0.05:
        return "SMALL POSITIVE MARGIN (borderline)"
    return "POSITIVE MARGIN (safe)"


def main() -> int:
    default_nscf = Path(__file__).resolve().parents[1] / "4_NSCF" / "nscf.out"

    parser = argparse.ArgumentParser(description="Detect HOMO/LUMO Wannier windows from QE nscf.out.")
    parser.add_argument(
        "nscf_out",
        nargs="?",
        default=str(default_nscf),
        help="Path to nscf.out (default: ../4_NSCF/nscf.out)",
    )
    args = parser.parse_args()

    nscf_path = Path(args.nscf_out).expanduser().resolve()
    if not nscf_path.exists():
        print(f"ERROR: File not found: {nscf_path}", file=sys.stderr)
        return 2

    data = parse_nscf(nscf_path)
    k_blocks = data["k_blocks"]
    homo_e = data["homo_e"]
    lumo_e = data["lumo_e"]

    homo_idx, lumo_idx, homo_err, lumo_err = find_band_indices(k_blocks, homo_e, lumo_e)

    if homo_idx - 2 < 1:
        print("ERROR: HOMO-2 band is out of range.", file=sys.stderr)
        return 3
    if lumo_idx > data["nbnd"]:
        print("ERROR: LUMO band is out of range.", file=sys.stderr)
        return 3

    homo_minus_2_e_list = [kb["energies"][homo_idx - 3] for kb in k_blocks]
    homo_minus_1_e_list = [kb["energies"][homo_idx - 2] for kb in k_blocks]
    homo_e_list = [kb["energies"][homo_idx - 1] for kb in k_blocks]
    lumo_e_list = [kb["energies"][lumo_idx - 1] for kb in k_blocks]

    emin = min(homo_minus_1_e_list)
    emax = max(homo_e_list)

    max_homo_minus_2 = max(homo_minus_2_e_list)
    min_lumo = min(lumo_e_list)

    lower_margin = emin - max_homo_minus_2
    upper_margin = min_lumo - emax
    homo_minus_1_homo_gap = min(homo_e_list) - max(homo_minus_1_e_list)

    homo_match = closest_match_for_band(k_blocks, homo_idx, homo_e)
    lumo_match = closest_match_for_band(k_blocks, lumo_idx, lumo_e)

    print(f"Input: {nscf_path}")
    print(f"nk = {len(k_blocks)}, nbnd = {data['nbnd']}")
    if data["nbnd_from_header"] is not None:
        print(f"nbnd (header) = {data['nbnd_from_header']}")
    if data["nelec"] is not None:
        print(f"number of electrons = {data['nelec']:.4f}")
    print()
    print(f"HOMO/LUMO from summary line: HOMO = {homo_e:.4f} eV, LUMO = {lumo_e:.4f} eV")
    print(f"Detected band indices (1-based): HOMO band = {homo_idx}, LUMO band = {lumo_idx}")
    print(f"Band-index match error: HOMO {homo_err:.6f} eV, LUMO {lumo_err:.6f} eV")
    print(
        "Closest HOMO point: "
        f"k={homo_match[0]} ({homo_match[1][0]:.4f}, {homo_match[1][1]:.4f}, {homo_match[1][2]:.4f}), "
        f"E={homo_match[2]:.4f} eV, |dE|={homo_match[3]:.6f} eV"
    )
    print(
        "Closest LUMO point: "
        f"k={lumo_match[0]} ({lumo_match[1][0]:.4f}, {lumo_match[1][1]:.4f}, {lumo_match[1][2]:.4f}), "
        f"E={lumo_match[2]:.4f} eV, |dE|={lumo_match[3]:.6f} eV"
    )
    print()
    print(f"homo_minus_2_e_list = {fmt_list(homo_minus_2_e_list)}")
    print(f"homo_minus_1_e_list = {fmt_list(homo_minus_1_e_list)}")
    print(f"homo_e_list         = {fmt_list(homo_e_list)}")
    print(f"lumo_e_list         = {fmt_list(lumo_e_list)}")
    print()
    print(f"emin (min HOMO-1 over k) = {emin:.4f} eV")
    print(f"emax (max HOMO over k) = {emax:.4f} eV")
    print(
        "Recommended frozen window for HOMO-1/HOMO pair: "
        f"dis_froz_min = {emin:.4f}, dis_froz_max = {emax:.4f}"
    )
    print()
    print(f"max(HOMO-2) = {max_homo_minus_2:.4f} eV")
    print(f"min(LUMO) = {min_lumo:.4f} eV")
    print(f"Lower-side margin: min(HOMO-1) - max(HOMO-2) = {lower_margin:.4f} eV -> {classify_margin(lower_margin)}")
    print(f"Upper-side margin: min(LUMO) - max(HOMO) = {upper_margin:.4f} eV -> {classify_margin(upper_margin)}")
    print(f"Band separation within target pair: min(HOMO) - max(HOMO-1) = {homo_minus_1_homo_gap:.4f} eV")
    if lower_margin > 0.0 and upper_margin > 0.0:
        print("Diagnosis: HOMO-1/HOMO pair is disentangled from HOMO-2/LUMO across k.")
    else:
        print("Diagnosis: HOMO-1/HOMO pair is NOT cleanly disentangled from HOMO-2/LUMO across k.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
