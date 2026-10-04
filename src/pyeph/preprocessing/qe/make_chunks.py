# Copyright (c) 2026, the PyEPH contributors.
# Adapted from jiangtong1000/PyEPH revision
# 6c4693acbb69a06a5bc8b0593abde2170ff38843 under BSD-3-Clause (see LICENSE).
"""Create validated, reproducible per-q irrep chunk directories."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

from ._files import (path_component, positive_integer, publication_lock,
                     publish_directory, publish_file, require_absent,
                     sha256_file, write_json)


class ConfigError(RuntimeError):
    pass


def split_range(first: int, last: int, count: int) -> list[tuple[int, int]]:
    try:
        first = positive_integer(first, "first irrep")
        last = positive_integer(last, "last irrep")
        count = positive_integer(count, "chunks")
    except ValueError as exc:
        raise ConfigError(str(exc)) from exc
    size = last - first + 1
    if first < 1 or last < first:
        raise ConfigError("invalid irrep range")
    if count < 1 or count > size:
        raise ConfigError("chunks must be between 1 and the number of irreps")
    quotient, remainder = divmod(size, count)
    ranges = []
    lower = first
    for index in range(count):
        upper = lower + quotient + (index < remainder) - 1
        ranges.append((lower, upper))
        lower = upper + 1
    return ranges


def q_count_from_dyn0(path: Path) -> int:
    lines = path.read_text().splitlines()
    if len(lines) < 2:
        raise ConfigError(f"invalid dyn0 file: {path}")
    try:
        return int(lines[1].split()[0])
    except (IndexError, ValueError) as exc:
        raise ConfigError(f"cannot read q-point count from {path}") from exc


def require(path: Path, kind: str = "path") -> None:
    if not path.exists():
        raise ConfigError(f"missing required {kind}: {path}")


def local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def validate_scf_xml(path: Path) -> None:
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError as exc:
        raise ConfigError(f"invalid SCF XML {path}: {exc}") from exc
    tags = {local_name(element.tag) for element in root.iter()}
    required = {"creator", "atomic_structure"}
    if not required.issubset(tags):
        raise ConfigError(f"{path} does not look like a QE SCF XML file")


def validate_pattern(path: Path, q_index: int) -> int:
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError as exc:
        raise ConfigError(f"invalid displacement-pattern XML {path}: {exc}") from exc
    elements = {local_name(element.tag): element for element in root.iter()}
    required = {
        "IRREPS_INFO",
        "QPOINT_NUMBER",
        "NUMBER_IRR_REP",
        "DISPLACEMENT_PATTERN",
    }
    if not required.issubset(elements):
        raise ConfigError(f"{path} is missing QE displacement-pattern content")
    try:
        actual_q = int(elements["QPOINT_NUMBER"].text or "")
        total_irreps = int(elements["NUMBER_IRR_REP"].text or "")
    except ValueError as exc:
        raise ConfigError(f"invalid q/irrep metadata in {path}") from exc
    if actual_q != q_index or total_irreps < 1:
        raise ConfigError(
            f"{path}: q={actual_q}, irreps={total_irreps}; expected q={q_index}"
        )
    if not (elements["DISPLACEMENT_PATTERN"].text or "").strip():
        raise ConfigError(f"empty displacement pattern in {path}")
    return total_irreps


def replace_tokens(path: Path, values: dict[str, str]) -> None:
    text = path.read_text()
    missing = [token for token in values if token not in text]
    if missing:
        raise ConfigError(f"{path} is missing template tokens: {missing}")
    for token, value in values.items():
        text = text.replace(token, value)
    path.write_text(text)


def validate_template(
    template: Path, shared_save: Path, shared_hess: Path | None
) -> tuple[int, dict[str, object]]:
    for path in template.rglob("*"):
        if path.is_symlink():
            raise ConfigError(f"template must contain real files, not symlinks: {path}")
    required = [
        template / "ph.in",
        template / "submit.sh",
        template / "PREFIX.dyn0",
        template / "tmp" / "PREFIX.xml",
        template / "tmp" / "_ph0" / "PREFIX.phsave",
        shared_save,
    ]
    if shared_hess is not None:
        required.append(shared_hess)
    for path in required:
        require(path)
    if not shared_save.is_dir():
        raise ConfigError(f"shared save must be a directory: {shared_save}")
    if shared_hess is not None and not shared_hess.is_file():
        raise ConfigError(f"shared Hessian must be a file: {shared_hess}")

    ph_input = (template / "ph.in").read_text()
    submit_input = (template / "submit.sh").read_text()
    ph_tokens = {
        "@PREFIX@",
        "@Q_INDEX@",
        "@START_IRR@",
        "@LAST_IRR@",
        "@DFTD3_HESS_LINE@",
    }
    submit_tokens = {
        "@PREFIX@",
        "@Q_INDEX@",
        "@START_IRR@",
        "@LAST_IRR@",
        "@JOB_NAME@",
    }
    missing_ph_tokens = sorted(token for token in ph_tokens if token not in ph_input)
    missing_submit_tokens = sorted(
        token for token in submit_tokens if token not in submit_input
    )
    if missing_ph_tokens:
        raise ConfigError(f"ph.in is missing tokens: {missing_ph_tokens}")
    if missing_submit_tokens:
        raise ConfigError(f"submit.sh is missing tokens: {missing_submit_tokens}")

    dyn0 = template / "PREFIX.dyn0"
    scf_xml = template / "tmp" / "PREFIX.xml"
    validate_scf_xml(scf_xml)
    nq = q_count_from_dyn0(dyn0)
    if nq < 1:
        raise ConfigError("q-point count must be positive")
    pattern_hashes = {}
    pattern_irreps = {}
    for q_index in range(1, nq + 1):
        pattern = (
            template
            / "tmp"
            / "_ph0"
            / "PREFIX.phsave"
            / f"patterns.{q_index}.xml"
        )
        require(pattern, "displacement-pattern file")
        pattern_irreps[str(q_index)] = validate_pattern(pattern, q_index)
        pattern_hashes[str(q_index)] = sha256_file(pattern)
    return nq, {
        "ph_input_sha256": sha256_file(template / "ph.in"),
        "submit_script_sha256": sha256_file(template / "submit.sh"),
        "dyn0_sha256": sha256_file(dyn0),
        "scf_xml_sha256": sha256_file(scf_xml),
        "pattern_sha256": pattern_hashes,
        "pattern_irreps": pattern_irreps,
        "shared_save": str(shared_save),
        "shared_hess": str(shared_hess) if shared_hess else None,
        "shared_hess_sha256": sha256_file(shared_hess) if shared_hess else None,
    }


def build_chunk(
    source: Path,
    destination: Path,
    prefix: str,
    q_index: int,
    first_irr: int,
    last_irr: int,
    job_name: str,
    shared_save: Path,
    shared_hess: Path | None,
    provenance: dict[str, object],
) -> None:
    shutil.copytree(source, destination, symlinks=False)
    _, copied_provenance = validate_template(destination, shared_save, shared_hess)
    if copied_provenance != provenance:
        raise ConfigError("template changed after validation")
    (destination / "PREFIX.dyn0").rename(destination / f"{prefix}.dyn0")
    (destination / "tmp" / "PREFIX.xml").rename(
        destination / "tmp" / f"{prefix}.xml"
    )
    phsave = destination / "tmp" / "_ph0" / "PREFIX.phsave"
    phsave.rename(phsave.with_name(f"{prefix}.phsave"))

    values = {
        "@PREFIX@": prefix,
        "@Q_INDEX@": str(q_index),
        "@START_IRR@": str(first_irr),
        "@LAST_IRR@": str(last_irr),
        "@DFTD3_HESS_LINE@": (
            f"  dftd3_hess='{prefix}.hess'" if shared_hess is not None else ""
        ),
    }
    replace_tokens(destination / "ph.in", values)
    replace_tokens(
        destination / "submit.sh",
        {
            "@PREFIX@": prefix,
            "@Q_INDEX@": str(q_index),
            "@START_IRR@": str(first_irr),
            "@LAST_IRR@": str(last_irr),
            "@JOB_NAME@": job_name,
        },
    )

    tmp = destination / "tmp"
    save_link = tmp / f"{prefix}.save"
    save_link.symlink_to(shared_save)
    if shared_hess is not None:
        (destination / f"{prefix}.hess").symlink_to(shared_hess)

    metadata = {
        "prefix": prefix,
        "q_index": q_index,
        "start_irr": first_irr,
        "last_irr": last_irr,
        "shared_save": str(shared_save),
        "shared_hess": str(shared_hess) if shared_hess else None,
        "template_provenance": provenance,
    }
    (destination / "chunk.json").write_text(json.dumps(metadata, indent=2) + "\n")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--q-index", type=int, required=True)
    parser.add_argument("--start-irr", type=int, required=True)
    parser.add_argument("--last-irr", type=int, required=True)
    parser.add_argument("--chunks", type=int, required=True)
    parser.add_argument("--shared-save", type=Path, required=True)
    parser.add_argument("--shared-hess", type=Path)
    parser.add_argument("--directory-prefix", default="chunk")
    parser.add_argument("--job-prefix", default="ph-chunk")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def make_chunks(
    *, template: Path, output_dir: Path, prefix: str, q_index: int,
    start_irr: int, last_irr: int, chunks: int, shared_save: Path,
    shared_hess: Path | None = None, directory_prefix: str = "chunk",
    job_prefix: str = "ph-chunk", dry_run: bool = False,
) -> dict[str, object]:
    """Validate a prepared template and create complete per-q chunk directories.

    Publication is serialized for this output directory. Each completed chunk
    is renamed into place, and the manifest is published last. An interrupted
    multi-chunk publication can leave complete chunks without a manifest;
    inspect them before choosing a fresh output directory.
    """
    try:
        q_index = positive_integer(q_index, "q_index")
        start_irr = positive_integer(start_irr, "start_irr")
        last_irr = positive_integer(last_irr, "last_irr")
        for label, value in (("prefix", prefix), ("directory_prefix", directory_prefix),
                             ("job_prefix", job_prefix)):
            path_component(value, label)
    except ValueError as exc:
        raise ConfigError(str(exc)) from exc
    template = Path(template).resolve()
    output_dir = Path(output_dir).resolve()
    shared_save = Path(shared_save).resolve()
    shared_hess = Path(shared_hess).resolve() if shared_hess is not None else None
    if output_dir == template or template in output_dir.parents:
        raise ConfigError("output directory must not be inside the template")
    nq, provenance = validate_template(template, shared_save, shared_hess)
    if not 1 <= q_index <= nq:
        raise ConfigError(f"q-index must be between 1 and {nq}")
    total_irreps = provenance["pattern_irreps"][str(q_index)]
    if last_irr > total_irreps:
        raise ConfigError(f"last-irr exceeds the pattern's {total_irreps} irreps")
    ranges = split_range(start_irr, last_irr, chunks)
    names = [f"{directory_prefix}{index}" for index in range(1, chunks + 1)]
    manifest_path = output_dir / f"chunks.q{q_index}.json"
    outputs = [output_dir / name for name in names] + [manifest_path]
    outputs += [path.with_name(path.name + ".partial") for path in outputs]
    require_absent(*outputs)
    plan = {
        "prefix": prefix, "q_index": q_index, "template": str(template),
        "template_provenance": provenance,
        "chunks": [
            {"directory": name, "start_irr": first, "last_irr": last}
            for name, (first, last) in zip(names, ranges)
        ],
    }
    if dry_run:
        return plan
    output_dir.mkdir(parents=True, exist_ok=True)
    with publication_lock(output_dir / ".pyeph-chunks.lock"):
        require_absent(*outputs)
        with tempfile.TemporaryDirectory(prefix=".chunks-", dir=output_dir) as staging_name:
            staging = Path(staging_name)
            for index, (name, (first, last)) in enumerate(zip(names, ranges), start=1):
                build_chunk(template, staging / name, prefix, q_index, first, last,
                            f"{job_prefix}-{index}", shared_save, shared_hess, provenance)
            staged_manifest = staging / "manifest.json"
            write_json(staged_manifest, plan)
            for name in names:
                publish_directory(staging / name, output_dir / name)
            publish_file(staged_manifest, manifest_path)
    return plan


def main(argv: list[str] | None = None) -> int:
    try:
        args = parse_args(argv)
        plan = make_chunks(**vars(args))
        if args.dry_run:
            print(json.dumps(plan, indent=2))
        else:
            print(args.output_dir.resolve() / f"chunks.q{args.q_index}.json")
        return 0
    except (ConfigError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
