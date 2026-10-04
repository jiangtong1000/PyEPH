# Copyright (c) 2026, the PyEPH contributors.
# Adapted from jiangtong1000/PyEPH revision
# 6c4693acbb69a06a5bc8b0593abde2170ff38843 under BSD-3-Clause (see LICENSE).
"""Merge explicitly mapped QE dvscf mode records without overwriting inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

from ._files import (absolute_output, positive_integer, publish_file,
                     require_absent, write_json)


class MergeError(RuntimeError):
    pass


def load_manifest(path: Path) -> dict[str, object]:
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise MergeError("manifest must be a JSON object")
    required = {"record_bytes", "total_modes", "output", "segments"}
    if not required.issubset(data):
        raise MergeError(f"manifest is missing keys: {sorted(required - set(data))}")
    return data


def validate_manifest(
    data: dict[str, object],
) -> tuple[int, int, Path, Path, list[dict[str, object]]]:
    record_bytes = positive_integer(data["record_bytes"], "record_bytes")
    total_modes = positive_integer(data["total_modes"], "total_modes")
    output = absolute_output(Path(str(data["output"])))
    receipt = absolute_output(Path(
        str(data.get("receipt", output.with_name(output.name + ".receipt.json")))
    ))
    raw_segments = data["segments"]
    if not isinstance(raw_segments, list) or any(
        not isinstance(segment, dict) for segment in raw_segments
    ):
        raise MergeError("segments must be a JSON list of objects")
    segments = [dict(segment) for segment in raw_segments]
    if record_bytes < 1 or total_modes < 1 or not segments:
        raise MergeError("record_bytes, total_modes, and segments must be nonzero")
    partial = output.with_name(output.name + ".partial")
    receipt_partial = receipt.with_name(receipt.name + ".partial")
    try:
        require_absent(output, partial, receipt, receipt_partial)
    except FileExistsError as exc:
        raise MergeError(str(exc)) from exc
    if output == receipt:
        raise MergeError("output and receipt must be different paths")
    provenance = data.get("provenance", {})
    if not isinstance(provenance, dict):
        raise MergeError("provenance must be a JSON object")

    next_mode = 1
    for segment in segments:
        first = positive_integer(segment["first_mode"], "first_mode")
        last = positive_integer(segment["last_mode"], "last_mode")
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
    if size < 1 or block_bytes < 1:
        raise MergeError("record size and block_bytes must be positive")
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


def merge(data: dict[str, object], block_bytes: int = 8 * 1024 * 1024) -> dict[str, object]:
    """Copy explicit cumulative mode records into a new file and hash receipt.

    All records are streamed and checked for zero-filled holes. The receipt is
    published before the completed data file; both use no-replace hard links.
    A process interrupted between publications can leave a receipt without its
    output. Such evidence is never overwritten automatically.
    """
    if block_bytes < 1:
        raise MergeError("block_bytes must be positive")
    record_bytes, total_modes, output, receipt, segments = validate_manifest(data)
    output.parent.mkdir(parents=True, exist_ok=True)
    receipt.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".dvscf-", dir=output.parent) as data_dir, \
         tempfile.TemporaryDirectory(prefix=".receipt-", dir=receipt.parent) as receipt_dir:
        staged_output = Path(data_dir) / "data"
        staged_receipt = Path(receipt_dir) / "receipt.json"
        results = []
        output_digest = hashlib.sha256()
        with staged_output.open("xb") as destination:
            for segment in segments:
                source_path = Path(str(segment["path"]))
                first = int(segment["first_mode"])
                last = int(segment["last_mode"])
                segment_digest = hashlib.sha256()
                first_digest = last_digest = None
                with source_path.open("rb") as source:
                    if os.fstat(source.fileno()).st_size != last * record_bytes:
                        raise MergeError(f"source size changed: {source_path}")
                    source.seek((first - 1) * record_bytes)
                    for mode in range(first, last + 1):
                        digest, nonzero = copy_record(
                            source, destination, record_bytes, block_bytes,
                            (output_digest, segment_digest),
                        )
                        if not nonzero:
                            raise MergeError(
                                f"{segment.get('label', source_path)} mode {mode} is all zero"
                            )
                        if first_digest is None:
                            first_digest = digest
                        last_digest = digest
                results.append({
                    "label": segment.get("label", source_path.name),
                    "first_mode": first, "last_mode": last,
                    "segment_sha256": segment_digest.hexdigest(),
                    "first_record_sha256": first_digest,
                    "last_record_sha256": last_digest,
                })
            destination.flush()
            os.fsync(destination.fileno())
        expected_size = total_modes * record_bytes
        if staged_output.stat().st_size != expected_size:
            raise MergeError(f"merged size {staged_output.stat().st_size} != {expected_size}")
        result = {
            "output": str(output), "receipt": str(receipt), "size": expected_size,
            "record_bytes": record_bytes, "total_modes": total_modes,
            "output_sha256": output_digest.hexdigest(),
            "provenance": data.get("provenance", {}), "segments": results,
        }
        write_json(staged_receipt, result)
        require_absent(output, output.with_name(output.name + ".partial"),
                       receipt, receipt.with_name(receipt.name + ".partial"))
        publish_file(staged_receipt, receipt)
        try:
            publish_file(staged_output, output)
        except Exception:
            # Remove only the hard link created by this call, never another file.
            if receipt.exists() and receipt.samefile(staged_receipt):
                receipt.unlink()
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
        if args.block_mib < 1:
            raise MergeError("block-mib must be positive")
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
