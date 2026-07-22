#!/usr/bin/env python3
"""Audit a per-q Quantum ESPRESSO irrep-chunk calculation.

This tool is read-only. It checks irrep/mode coverage, dynmat files, pattern
identity, direct-access dvscf sizes, and records in a merged dvscf.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path


IRREP_RANGE_RE = re.compile(
    r"\b(start_irr|last_irr)\s*=\s*(\d+)", re.IGNORECASE
)
PREFIX_RE = re.compile(r"\bprefix\s*=\s*['\"]([^'\"]+)['\"]", re.IGNORECASE)
TOTAL_IRREPS_RE = re.compile(r"There are\s+(\d+)\s+irreducible representations")
MODE_RE = re.compile(
    r"Representation\s+#\s*(\d+)\s+modes?\s+#\s*((?:\d+\s*)+)",
    re.IGNORECASE,
)


class AuditError(RuntimeError):
    pass


@dataclass(frozen=True)
class Chunk:
    path: str
    start_irr: int
    last_irr: int
    first_mode: int
    last_mode: int
    dvscf: str
    dvscf_size: int
    dynmat_count: int


def read_chunk_input(path: Path) -> tuple[str, int, int]:
    text = path.read_text()
    prefix_match = PREFIX_RE.search(text)
    values = {name.lower(): int(value) for name, value in IRREP_RANGE_RE.findall(text)}
    if prefix_match is None or set(values) != {"start_irr", "last_irr"}:
        raise AuditError(f"cannot read prefix and irrep range from {path}")
    return prefix_match.group(1), values["start_irr"], values["last_irr"]


def read_mode_map(
    path: Path, text: str | None = None
) -> tuple[int, dict[int, tuple[int, ...]]]:
    if text is None:
        text = path.read_text(errors="replace")
    totals = {int(value) for value in TOTAL_IRREPS_RE.findall(text)}
    if len(totals) != 1:
        raise AuditError(f"expected one total-irrep value in {path}, got {sorted(totals)}")

    mode_map: dict[int, tuple[int, ...]] = {}
    for match in MODE_RE.finditer(text):
        irr = int(match.group(1))
        modes = tuple(int(value) for value in match.group(2).split())
        previous = mode_map.setdefault(irr, modes)
        if previous != modes:
            raise AuditError(f"inconsistent mode mapping for irrep {irr} in {path}")
    if not mode_map:
        raise AuditError(f"no representation-to-mode mapping found in {path}")
    return totals.pop(), mode_map


def hash_record(path: Path, record_bytes: int, record_index: int) -> tuple[str, bool]:
    remaining = record_bytes
    digest = hashlib.sha256()
    nonzero = False
    zero_block = bytes(min(8 * 1024 * 1024, record_bytes))
    with path.open("rb") as handle:
        handle.seek(record_index * record_bytes)
        while remaining:
            block = handle.read(min(len(zero_block), remaining))
            if not block:
                raise AuditError(f"short read from {path} at record {record_index + 1}")
            digest.update(block)
            if not nonzero and block != zero_block[: len(block)]:
                nonzero = True
            remaining -= len(block)
    return digest.hexdigest(), nonzero


def audit(args: argparse.Namespace) -> dict[str, object]:
    errors: list[str] = []
    warnings: list[str] = []
    chunks: list[Chunk] = []
    reference_patterns: str | None = None
    total_irreps: int | None = None
    mode_map: dict[int, tuple[int, ...]] = {}
    final_dvscf = getattr(args, "final_dvscf", None)

    for chunk_dir in args.chunk:
        chunk_dir = chunk_dir.resolve()
        prefix, start_irr, last_irr = read_chunk_input(chunk_dir / "ph.in")
        output = chunk_dir / "ph.out"
        output_text = output.read_text(errors="replace")
        if "JOB DONE." not in output_text:
            errors.append(f"{chunk_dir}: ph.out has no JOB DONE marker")
        if "Error in routine" in output_text or "No convergence" in output_text:
            errors.append(f"{chunk_dir}: ph.out contains an error/nonconvergence marker")

        chunk_total, chunk_modes = read_mode_map(output, output_text)
        if total_irreps is None:
            total_irreps = chunk_total
        elif chunk_total != total_irreps:
            errors.append(
                f"{chunk_dir}: total irrep count {chunk_total} differs from {total_irreps}"
            )
        for irr, modes in chunk_modes.items():
            previous = mode_map.setdefault(irr, modes)
            if previous != modes:
                errors.append(f"{chunk_dir}: mode mapping for irrep {irr} is inconsistent")

        if start_irr not in chunk_modes or last_irr not in chunk_modes:
            raise AuditError(f"{chunk_dir}: requested irrep range is absent from ph.out")
        first_mode = min(chunk_modes[start_irr])
        last_mode = max(chunk_modes[last_irr])

        phsave = chunk_dir / "tmp" / "_ph0" / f"{prefix}.phsave"
        expected_dynmats = {
            phsave / f"dynmat.{args.q_index}.{irr}.xml"
            for irr in range(start_irr, last_irr + 1)
        }
        missing_dynmats = sorted(path.name for path in expected_dynmats if not path.is_file())
        if missing_dynmats:
            errors.append(f"{chunk_dir}: missing dynmat files {missing_dynmats}")

        pattern = phsave / f"patterns.{args.q_index}.xml"
        if not pattern.is_file():
            errors.append(f"{chunk_dir}: missing {pattern.name}")
        else:
            pattern_hash = hashlib.sha256(pattern.read_bytes()).hexdigest()
            if reference_patterns is None:
                reference_patterns = pattern_hash
            elif pattern_hash != reference_patterns:
                errors.append(f"{chunk_dir}: displacement pattern differs from first chunk")

        dvscf = (
            chunk_dir
            / "tmp"
            / "_ph0"
            / f"{prefix}.q_{args.q_index}"
            / f"{prefix}.dvscf1"
        )
        if not dvscf.is_file():
            raise AuditError(f"missing dvscf: {dvscf}")
        expected_size = last_mode * args.record_bytes
        if dvscf.stat().st_size != expected_size:
            errors.append(
                f"{chunk_dir}: dvscf size {dvscf.stat().st_size} != {expected_size}"
            )

        chunks.append(
            Chunk(
                path=str(chunk_dir),
                start_irr=start_irr,
                last_irr=last_irr,
                first_mode=first_mode,
                last_mode=last_mode,
                dvscf=str(dvscf),
                dvscf_size=dvscf.stat().st_size,
                dynmat_count=len(expected_dynmats) - len(missing_dynmats),
            )
        )

    assert total_irreps is not None
    chunks.sort(key=lambda item: item.start_irr)

    expected_irreps = set(range(chunks[0].start_irr, total_irreps + 1))
    missing_mode_map = sorted(expected_irreps - set(mode_map))
    out_of_range_mode_map = sorted(
        set(mode_map) - set(range(1, total_irreps + 1))
    )
    if missing_mode_map or out_of_range_mode_map:
        errors.append(
            f"chunk representation-to-mode map coverage: missing={missing_mode_map}, "
            f"out_of_range={out_of_range_mode_map}"
        )

    all_modes = [
        mode
        for irrep, modes in mode_map.items()
        if irrep in expected_irreps
        for mode in modes
    ]
    mode_counts = Counter(all_modes)
    duplicate_modes = sorted(mode for mode, count in mode_counts.items() if count > 1)
    total_modes = max(all_modes)
    first_chunk_mode = chunks[0].first_mode
    missing_modes = sorted(
        set(range(first_chunk_mode, total_modes + 1)) - set(all_modes)
    )
    if missing_modes or duplicate_modes:
        errors.append(
            f"global mode map is not one-to-one: missing={missing_modes}, "
            f"duplicates={duplicate_modes}"
        )

    base_size = args.base_dvscf.stat().st_size
    if base_size % args.record_bytes:
        errors.append("base dvscf size is not a multiple of record_bytes")
    base_modes = base_size // args.record_bytes

    inferred_base_last_irrep = (
        chunks[0].start_irr - 1
        if chunks[0].first_mode == base_modes + 1
        else None
    )

    expected_irr = chunks[0].start_irr
    expected_mode = base_modes + 1
    for index, chunk in enumerate(chunks):
        if index and chunk.start_irr != expected_irr:
            errors.append(
                f"irrep gap/overlap: expected {expected_irr}, got {chunk.start_irr}"
            )
        if chunk.first_mode != expected_mode:
            errors.append(
                f"mode gap/overlap at irrep {chunk.start_irr}: "
                f"expected mode {expected_mode}, got {chunk.first_mode}"
            )
        expected_irr = chunk.last_irr + 1
        expected_mode = chunk.last_mode + 1

    if chunks[-1].last_irr != total_irreps:
        errors.append(
            f"chunks end at irrep {chunks[-1].last_irr}, expected {total_irreps}"
        )
    expected_final_size = total_modes * args.record_bytes
    if final_dvscf and final_dvscf.stat().st_size != expected_final_size:
        errors.append(
            f"final dvscf size {final_dvscf.stat().st_size} != {expected_final_size}"
        )

    if args.final_phsave:
        actual = {
            int(match.group(1))
            for path in args.final_phsave.glob(f"dynmat.{args.q_index}.*.xml")
            if (match := re.fullmatch(
                rf"dynmat\.{args.q_index}\.(\d+)\.xml", path.name
            ))
        }
        expected = set(range(0, total_irreps + 1))
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        if missing or extra:
            errors.append(f"final phsave dynmat coverage: missing={missing}, extra={extra}")

    check_all_records = getattr(args, "check_all_records", False)
    check_boundary_records = getattr(args, "check_records", False)
    progress_every = getattr(args, "progress_every", 0)
    if progress_every < 0:
        raise AuditError("progress_every must not be negative")
    records_checked = 0
    if check_boundary_records or check_all_records:
        if final_dvscf is None:
            raise AuditError("record checks require --final-dvscf")
        segments = [("base", args.base_dvscf, 1, base_modes)] + [
            (f"chunk{index}", Path(chunk.dvscf), chunk.first_mode, chunk.last_mode)
            for index, chunk in enumerate(chunks, start=1)
        ]
        for label, source, first_mode, last_mode in segments:
            modes = (
                range(first_mode, last_mode + 1)
                if check_all_records
                else sorted({first_mode, last_mode})
            )
            for mode in modes:
                records_checked += 1
                if progress_every and records_checked % progress_every == 0:
                    print(
                        f"checked {records_checked} records; current={label} mode={mode}",
                        file=sys.stderr,
                        flush=True,
                    )
                source_hash, source_nonzero = hash_record(
                    source, args.record_bytes, mode - 1
                )
                final_hash, final_nonzero = hash_record(
                    final_dvscf, args.record_bytes, mode - 1
                )
                if not source_nonzero:
                    errors.append(f"{label}: source mode {mode} is all zero")
                if not final_nonzero:
                    errors.append(f"final dvscf: mode {mode} is all zero")
                if source_hash != final_hash:
                    errors.append(f"{label}: final mode {mode} differs from source")

    return {
        "ok": not errors,
        "q_index": args.q_index,
        "total_irreps": total_irreps,
        "total_modes": total_modes,
        "record_bytes": args.record_bytes,
        "base_modes": base_modes,
        "inferred_base_last_irrep": inferred_base_last_irrep,
        "pattern_sha256": reference_patterns,
        "record_check": "all" if check_all_records else (
            "boundaries" if check_boundary_records else "none"
        ),
        "records_checked": records_checked,
        "chunks": [asdict(chunk) for chunk in chunks],
        "errors": errors,
        "warnings": warnings,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--q-index", type=int, required=True)
    parser.add_argument("--record-bytes", type=int, required=True)
    parser.add_argument("--base-dvscf", type=Path, required=True)
    parser.add_argument("--final-dvscf", type=Path)
    parser.add_argument("--final-phsave", type=Path)
    parser.add_argument("--chunk", type=Path, action="append", required=True)
    checks = parser.add_mutually_exclusive_group()
    checks.add_argument(
        "--check-records",
        action="store_true",
        help="hash the first and last record of every contributed segment",
    )
    checks.add_argument(
        "--check-all-records",
        action="store_true",
        help="hash and compare every contributed record (I/O intensive; run in batch)",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=25,
        help="report record-check progress to stderr every N records; use 0 to disable",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    try:
        report = audit(parse_args(argv))
    except (AuditError, OSError) as exc:
        print(json.dumps({"ok": False, "fatal": str(exc)}, indent=2))
        return 2
    print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
