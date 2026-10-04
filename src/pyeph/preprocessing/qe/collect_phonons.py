# Copyright (c) 2026, the PyEPH contributors.
# Adapted from jiangtong1000/PyEPH ph_collect.sh, revision
# 6c4693acbb69a06a5bc8b0593abde2170ff38843 under BSD-3-Clause (see LICENSE).
"""Collect completed QE phonons into the save layout consumed by qe2pert.x."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile

from ._files import (absolute_output, copy_and_fsync, path_component,
                     publish_directory, require_absent, sha256_file, write_json)
from .make_chunks import ConfigError, q_count_from_dyn0


class CollectionError(RuntimeError):
    pass


def _identical_source(
    paths: list[Path], label: str, *, allow_empty: bool = False,
) -> dict[str, object]:
    candidates = sorted({path.resolve() for path in paths
                         if path.is_file() and (allow_empty or path.stat().st_size > 0)}, key=str)
    if not candidates:
        raise CollectionError(f"no nonempty source for {label}")
    records = [{"path": str(path), "size": path.stat().st_size,
                "sha256": sha256_file(path)} for path in candidates]
    if len({(record["size"], record["sha256"]) for record in records}) != 1:
        raise CollectionError(f"conflicting contents for {label}: {records}")
    return {"selected_source": records[0]["path"], "size": records[0]["size"],
            "sha256": records[0]["sha256"], "all_sources": records}


def inspect(
    *, prefix: str, work_root: Path = Path("."), tmp_root: Path | None = None,
    output: Path | None = None, dyn0: Path | None = None,
) -> dict[str, object]:
    """Resolve all q/image files and hashes without creating any output.

    A consolidated ``_ph0/<prefix>.phsave`` is required. This checks the
    patterns and irrep-zero files for every q; use ``audit_chunks`` to certify
    the complete irrep/mode coverage of a repaired chunk calculation first.
    """
    path_component(prefix, "prefix")
    if prefix == "PREFIX":
        raise CollectionError("set prefix to the calculation prefix")
    work_root = Path(work_root).resolve()
    tmp_root = Path(tmp_root).resolve() if tmp_root is not None else work_root / "tmp"
    output = absolute_output(Path(output) if output is not None else work_root / "save")
    dyn0 = Path(dyn0).resolve() if dyn0 is not None else work_root / f"{prefix}.dyn0"
    require_absent(output, output.with_name(output.name + ".partial"))
    phsave = tmp_root / "_ph0" / f"{prefix}.phsave"
    if not phsave.is_dir():
        raise CollectionError(f"missing consolidated phsave: {phsave}")
    if output == phsave or phsave in output.parents:
        raise CollectionError("output must not be inside the source phsave")
    nq = q_count_from_dyn0(dyn0)
    if nq < 1:
        raise CollectionError("q-point count must be positive")

    files = []

    def add(paths: list[Path], relative: str, *, allow_empty: bool = False) -> dict[str, object]:
        record = _identical_source(paths, relative, allow_empty=allow_empty)
        record["destination"] = relative
        files.append(record)
        return record

    add([dyn0], f"{prefix}.dyn0")
    for path in sorted(phsave.rglob("*")):
        if path.is_symlink():
            raise CollectionError(f"source phsave must not contain symlinks: {path}")
        if path.is_file():
            add([path], str(Path(f"{prefix}.phsave") / path.relative_to(phsave)), allow_empty=True)
    dvscf_size = None
    for q in range(1, nq + 1):
        for name in (f"patterns.{q}.xml", f"dynmat.{q}.0.xml"):
            if not (phsave / name).is_file():
                raise CollectionError(f"missing required file: {phsave / name}")
        add([work_root / f"{prefix}.dyn{q}.xml"], f"{prefix}.dyn{q}.xml")
        if q == 1:
            candidates = [tmp_root / "_ph0" / f"{prefix}.dvscf1",
                          tmp_root / "_ph0" / f"{prefix}.q_1" / f"{prefix}.dvscf1"]
        else:
            candidates = [path for path in tmp_root.rglob(f"{prefix}.dvscf1")
                          if path.parent.name == f"{prefix}.q_{q}"]
        record = add(candidates, f"{prefix}.dvscf_q{q}")
        if dvscf_size is None:
            dvscf_size = record["size"]
        elif dvscf_size != record["size"]:
            raise CollectionError(f"dvscf q{q} has {record['size']} bytes; expected {dvscf_size}")
    return {"prefix": prefix, "q_points": nq, "output": str(output),
            "work_root": str(work_root), "tmp_root": str(tmp_root),
            "dvscf_bytes_per_q": dvscf_size, "files": files}


def stage(report: dict[str, object]) -> dict[str, object]:
    """Publish a complete, independently hash-checked collection directory."""
    output = absolute_output(Path(str(report["output"])))
    require_absent(output, output.with_name(output.name + ".partial"))
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".phonons-", dir=output.parent) as temp:
        bundle = Path(temp) / "bundle"
        bundle.mkdir()
        for record in report["files"]:
            relative = Path(record["destination"])
            if relative.is_absolute() or ".." in relative.parts:
                raise CollectionError(f"unsafe destination in report: {relative}")
            destination = bundle / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            copy_and_fsync(Path(record["selected_source"]), destination)
            if (destination.stat().st_size != record["size"]
                    or sha256_file(destination) != record["sha256"]):
                raise CollectionError(f"staged {relative} differs from inspected source")
        write_json(bundle / "collection_receipt.json", report)
        publish_directory(bundle, output)
    return report


def collect(
    *, prefix: str, work_root: Path = Path("."), tmp_root: Path | None = None,
    output: Path | None = None, dyn0: Path | None = None, dry_run: bool = False,
) -> dict[str, object]:
    """Inspect and optionally stage a standard completed phonon calculation."""
    report = inspect(prefix=prefix, work_root=work_root, tmp_root=tmp_root,
                     output=output, dyn0=dyn0)
    return report if dry_run else stage(report)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--work-root", type=Path, default=Path("."))
    parser.add_argument("--tmp-root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dyn0", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    try:
        report = collect(**vars(parser.parse_args(argv)))
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    except (CollectionError, ConfigError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
