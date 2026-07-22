#!/usr/bin/env python3
"""Merge explicitly mapped QE dvscf mode records without overwriting inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path


class MergeError(RuntimeError):
    pass


def load_manifest(path: Path) -> dict[str, object]:
    data = json.loads(path.read_text())
    required = {"record_bytes", "total_modes", "output", "segments"}
    if not required.issubset(data):
        raise MergeError(f"manifest is missing keys: {sorted(required - set(data))}")
    return data


def validate_manifest(
    data: dict[str, object],
) -> tuple[int, int, Path, Path, list[dict[str, object]]]:
    record_bytes = int(data["record_bytes"])
    total_modes = int(data["total_modes"])
    output = Path(str(data["output"])).resolve()
    receipt = Path(
        str(data.get("receipt", output.with_name(output.name + ".receipt.json")))
    ).resolve()
    raw_segments = data["segments"]
    if not isinstance(raw_segments, list) or any(
        not isinstance(segment, dict) for segment in raw_segments
    ):
        raise MergeError("segments must be a JSON list of objects")
    segments = list(raw_segments)
    if record_bytes < 1 or total_modes < 1 or not segments:
        raise MergeError("record_bytes, total_modes, and segments must be nonzero")
    partial = output.with_name(output.name + ".partial")
    receipt_partial = receipt.with_name(receipt.name + ".partial")
    if output.exists() or partial.exists() or receipt.exists() or receipt_partial.exists():
        raise MergeError(f"refusing to overwrite output or partial output: {output}")
    if output == receipt:
        raise MergeError("output and receipt must be different paths")
    provenance = data.get("provenance", {})
    if not isinstance(provenance, dict):
        raise MergeError("provenance must be a JSON object")

    next_mode = 1
    for segment in segments:
        first = int(segment["first_mode"])
        last = int(segment["last_mode"])
        source = Path(str(segment["path"])).resolve()
        if first != next_mode or last < first:
            raise MergeError(
                f"segment {segment.get('label', source)} starts at {first}; expected {next_mode}"
            )
        if not source.is_file():
            raise MergeError(f"missing source: {source}")
        expected_source_size = last * record_bytes
        if source.stat().st_size != expected_source_size:
            raise MergeError(
                f"{source}: size {source.stat().st_size} != {expected_source_size}"
            )
        segment["path"] = str(source)
        next_mode = last + 1
    if next_mode - 1 != total_modes:
        raise MergeError(f"segments end at mode {next_mode - 1}; expected {total_modes}")
    return record_bytes, total_modes, output, receipt, segments


def copy_record(
    source,
    destination,
    size: int,
    block_bytes: int,
    digests: tuple[object, ...] = (),
) -> tuple[str, bool]:
    remaining = size
    digest = hashlib.sha256()
    nonzero = False
    zero_block = bytes(min(block_bytes, size))
    while remaining:
        block = source.read(min(block_bytes, remaining))
        if not block:
            raise MergeError("short read while copying a dvscf record")
        destination.write(block)
        digest.update(block)
        for aggregate in digests:
            aggregate.update(block)
        if not nonzero and block != zero_block[: len(block)]:
            nonzero = True
        remaining -= len(block)
    return digest.hexdigest(), nonzero


def merge(data: dict[str, object], block_bytes: int) -> dict[str, object]:
    record_bytes, total_modes, output, receipt, segments = validate_manifest(data)
    partial = output.with_name(output.name + ".partial")
    receipt_partial = receipt.with_name(receipt.name + ".partial")
    results = []
    output.parent.mkdir(parents=True, exist_ok=True)
    receipt.parent.mkdir(parents=True, exist_ok=True)
    output_digest = hashlib.sha256()
    try:
        with partial.open("xb") as destination:
            for segment in segments:
                source_path = Path(str(segment["path"]))
                first = int(segment["first_mode"])
                last = int(segment["last_mode"])
                hashes = []
                segment_digest = hashlib.sha256()
                with source_path.open("rb") as source:
                    source.seek((first - 1) * record_bytes)
                    for mode in range(first, last + 1):
                        digest, nonzero = copy_record(
                            source,
                            destination,
                            record_bytes,
                            block_bytes,
                            (output_digest, segment_digest),
                        )
                        if not nonzero:
                            raise MergeError(
                                f"{segment.get('label', source_path)} mode {mode} is all zero"
                            )
                        hashes.append(digest)
                results.append(
                    {
                        "label": segment.get("label", source_path.name),
                        "first_mode": first,
                        "last_mode": last,
                        "segment_sha256": segment_digest.hexdigest(),
                        "first_record_sha256": hashes[0],
                        "last_record_sha256": hashes[-1],
                    }
                )
            destination.flush()
            os.fsync(destination.fileno())
        expected_size = total_modes * record_bytes
        if partial.stat().st_size != expected_size:
            raise MergeError(f"merged size {partial.stat().st_size} != {expected_size}")
        result = {
            "output": str(output),
            "receipt": str(receipt),
            "size": expected_size,
            "record_bytes": record_bytes,
            "total_modes": total_modes,
            "output_sha256": output_digest.hexdigest(),
            "provenance": data.get("provenance", {}),
            "segments": results,
        }
        with receipt_partial.open("x") as handle:
            json.dump(result, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(partial, output)
        os.replace(receipt_partial, receipt)
    except Exception:
        partial.unlink(missing_ok=True)
        receipt_partial.unlink(missing_ok=True)
        raise
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--block-mib", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    try:
        args = parse_args(argv)
        data = load_manifest(args.manifest)
        record_bytes, total_modes, output, receipt, segments = validate_manifest(data)
        if args.dry_run:
            print(
                json.dumps(
                    {
                        "ok": True,
                        "record_bytes": record_bytes,
                        "total_modes": total_modes,
                        "output": str(output),
                        "receipt": str(receipt),
                        "segments": segments,
                    },
                    indent=2,
                )
            )
            return 0
        result = merge(data, args.block_mib * 1024 * 1024)
        print(json.dumps(result, indent=2))
        return 0
    except (MergeError, OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
