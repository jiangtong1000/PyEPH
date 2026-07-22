#!/usr/bin/env python3
"""Stage one q-point's QE patterns and dynmat files without hiding conflicts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


class CollectionError(RuntimeError):
    pass


def sha256_file(path: Path, block_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_bytes):
            digest.update(block)
    return digest.hexdigest()


def require_xml(path: Path) -> None:
    try:
        ET.parse(path)
    except ET.ParseError as exc:
        raise CollectionError(f"invalid XML {path}: {exc}") from exc


def load_manifest(path: Path) -> dict[str, object]:
    data = json.loads(path.read_text())
    required = {
        "q_index",
        "total_irreps",
        "output",
        "pattern_sources",
        "dynmat_sources",
    }
    missing = required - set(data)
    if missing:
        raise CollectionError(f"manifest is missing keys: {sorted(missing)}")
    return data


def require_string_list(data: dict[str, object], key: str) -> list[str]:
    values = data[key]
    if (
        not isinstance(values, list)
        or not values
        or any(not isinstance(value, str) or not value for value in values)
    ):
        raise CollectionError(f"{key} must be a nonempty list of paths")
    return values


def inspect(data: dict[str, object]) -> dict[str, object]:
    q_index = int(data["q_index"])
    total_irreps = int(data["total_irreps"])
    output = Path(str(data["output"])).resolve()
    if q_index < 1 or total_irreps < 1:
        raise CollectionError("q_index and total_irreps must be positive")
    partial = output.with_name(output.name + ".partial")
    if output.exists() or partial.exists():
        raise CollectionError(f"refusing to overwrite output or partial output: {output}")

    pattern_paths = [
        Path(value).resolve()
        for value in require_string_list(data, "pattern_sources")
    ]
    pattern_records = []
    for path in pattern_paths:
        if not path.is_file():
            raise CollectionError(f"missing pattern source: {path}")
        require_xml(path)
        pattern_records.append({"path": str(path), "sha256": sha256_file(path)})
    pattern_hashes = {record["sha256"] for record in pattern_records}
    if len(pattern_hashes) != 1:
        raise CollectionError("pattern sources disagree: " + json.dumps(pattern_records))

    source_dirs = [
        Path(value).resolve()
        for value in require_string_list(data, "dynmat_sources")
    ]
    candidates: dict[int, list[Path]] = {}
    name_re = re.compile(rf"dynmat\.{q_index}\.(\d+)\.xml")
    for source_dir in source_dirs:
        if not source_dir.is_dir():
            raise CollectionError(f"missing dynmat source directory: {source_dir}")
        for path in source_dir.glob(f"dynmat.{q_index}.*.xml"):
            match = name_re.fullmatch(path.name)
            if match is None:
                raise CollectionError(f"unexpected dynmat filename: {path}")
            candidates.setdefault(int(match.group(1)), []).append(path.resolve())

    expected = set(range(total_irreps + 1))
    actual = set(candidates)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing or extra:
        raise CollectionError(f"dynmat coverage mismatch: missing={missing}, extra={extra}")

    dynmat_records = []
    for irrep in sorted(expected):
        sources = sorted(candidates[irrep], key=str)
        hashed_sources = []
        for path in sources:
            require_xml(path)
            hashed_sources.append({"path": str(path), "sha256": sha256_file(path)})
        hashes = {record["sha256"] for record in hashed_sources}
        if len(hashes) != 1:
            raise CollectionError(
                f"conflicting dynmat.{q_index}.{irrep}.xml sources: "
                + json.dumps(hashed_sources)
            )
        dynmat_records.append(
            {
                "irrep": irrep,
                "sha256": hashed_sources[0]["sha256"],
                "selected_source": hashed_sources[0]["path"],
                "all_sources": [record["path"] for record in hashed_sources],
            }
        )

    provenance = data.get("provenance", {})
    if not isinstance(provenance, dict):
        raise CollectionError("provenance must be a JSON object")
    return {
        "q_index": q_index,
        "total_irreps": total_irreps,
        "output": str(output),
        "pattern_sha256": pattern_records[0]["sha256"],
        "pattern_sources": [record["path"] for record in pattern_records],
        "dynmat": dynmat_records,
        "provenance": provenance,
    }


def copy_and_fsync(source: Path, destination: Path) -> None:
    with source.open("rb") as source_handle, destination.open("xb") as output_handle:
        shutil.copyfileobj(source_handle, output_handle, length=8 * 1024 * 1024)
        output_handle.flush()
        os.fsync(output_handle.fileno())


def stage(report: dict[str, object]) -> dict[str, object]:
    output = Path(str(report["output"]))
    partial = output.with_name(output.name + ".partial")
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        partial.mkdir()
        q_index = int(report["q_index"])
        pattern_source = Path(str(report["pattern_sources"][0]))
        pattern_output = partial / f"patterns.{q_index}.xml"
        copy_and_fsync(pattern_source, pattern_output)
        if sha256_file(pattern_output) != report["pattern_sha256"]:
            raise CollectionError("staged pattern hash differs from source")

        for record in report["dynmat"]:
            irrep = int(record["irrep"])
            source = Path(str(record["selected_source"]))
            destination = partial / f"dynmat.{q_index}.{irrep}.xml"
            copy_and_fsync(source, destination)
            if sha256_file(destination) != record["sha256"]:
                raise CollectionError(f"staged {destination.name} hash differs from source")

        receipt = partial / "collection_receipt.json"
        with receipt.open("x") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(partial, output)
    except Exception:
        shutil.rmtree(partial, ignore_errors=True)
        raise
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    try:
        args = parse_args(argv)
        report = inspect(load_manifest(args.manifest))
        if not args.dry_run:
            stage(report)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    except (
        CollectionError,
        OSError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
