# Copyright (c) 2026, the PyEPH contributors.
# Adapted from jiangtong1000/PyEPH revision
# 6c4693acbb69a06a5bc8b0593abde2170ff38843 under BSD-3-Clause (see LICENSE).
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import pytest

from pyeph.preprocessing.qe import audit_chunks, collect_dynmat, make_chunks, merge_dvscf, collect_phonons

ROOT = Path(__file__).resolve().parents[1]


PH_OUT = """There are  3 irreducible representations
Representation # 1 mode # 1
Representation # 2 modes # 2 3
Representation # 3 mode # 4
JOB DONE.
"""


class ToolTests(unittest.TestCase):
    def test_split_range_rejects_empty_chunks(self) -> None:
        self.assertEqual(make_chunks.split_range(10, 14, 2), [(10, 12), (13, 14)])
        with self.assertRaises(make_chunks.ConfigError):
            make_chunks.split_range(1, 2, 3)

    def test_merge_uses_explicit_mode_offsets(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            base = root / "base.dvscf"
            chunk1 = root / "chunk1.dvscf"
            chunk2 = root / "chunk2.dvscf"
            output = root / "merged.dvscf"
            base.write_bytes(b"AAAA" + b"BBBB")
            chunk1.write_bytes(bytes(8) + b"CCCC" + b"DDDD")
            chunk2.write_bytes(bytes(16) + b"EEEE")
            manifest = {
                "record_bytes": 4,
                "total_modes": 5,
                "output": str(output),
                "segments": [
                    {"label": "base", "path": str(base), "first_mode": 1, "last_mode": 2},
                    {"label": "chunk1", "path": str(chunk1), "first_mode": 3, "last_mode": 4},
                    {"label": "chunk2", "path": str(chunk2), "first_mode": 5, "last_mode": 5},
                ],
            }
            result = merge_dvscf.merge(manifest, block_bytes=2)
            self.assertEqual(output.read_bytes(), b"AAAABBBBCCCCDDDDEEEE")
            receipt = root / "merged.dvscf.receipt.json"
            self.assertTrue(receipt.is_file())
            self.assertEqual(
                json.loads(receipt.read_text())["output_sha256"],
                result["output_sha256"],
            )
            self.assertEqual(len(result["segments"]), 3)

    def test_full_record_audit_detects_an_interior_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            base = root / "base.dvscf"
            final = root / "final.dvscf"
            base.write_bytes(b"AAAA")
            final.write_bytes(b"AAAABBBB" + bytes(4) + b"DDDD")

            chunk = root / "chunk"
            phsave = chunk / "tmp" / "_ph0" / "test.phsave"
            qdir = chunk / "tmp" / "_ph0" / "test.q_1"
            phsave.mkdir(parents=True)
            qdir.mkdir()
            (chunk / "ph.in").write_text(
                "&inputph\n prefix='test'\n start_irr=2, last_irr=3\n/\n"
            )
            (chunk / "ph.out").write_text(PH_OUT)
            for irrep in (2, 3):
                (phsave / f"dynmat.1.{irrep}.xml").write_text("<ok/>\n")
            (phsave / "patterns.1.xml").write_text("<patterns/>\n")
            (qdir / "test.dvscf1").write_bytes(bytes(4) + b"BBBBCCCCDDDD")

            final_phsave = root / "final.phsave"
            final_phsave.mkdir()
            (final_phsave / "patterns.1.xml").write_text("<patterns/>\n")
            for irrep in range(4):
                (final_phsave / f"dynmat.1.{irrep}.xml").write_text("<ok/>\n")

            common = {
                "q_index": 1,
                "record_bytes": 4,
                "base_dvscf": base,
                "final_dvscf": final,
                "final_phsave": final_phsave,
                "chunk": [chunk],
            }
            boundary_report = audit_chunks.audit(
                argparse.Namespace(**common, check_records=True, check_all_records=False)
            )
            self.assertTrue(boundary_report["ok"], json.dumps(boundary_report, indent=2))
            full_report = audit_chunks.audit(
                argparse.Namespace(**common, check_records=False, check_all_records=True)
            )
            self.assertFalse(full_report["ok"])
            self.assertEqual(full_report["record_check"], "all")
            self.assertTrue(any("mode 3" in error for error in full_report["errors"]))

    def test_generator_builds_tokenized_chunks_and_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            template = root / "template"
            phsave = template / "tmp" / "_ph0" / "PREFIX.phsave"
            phsave.mkdir(parents=True)
            (template / "PREFIX.dyn0").write_text("2 1 1\n2\n0 0 0\n0.5 0 0\n")
            (template / "tmp" / "PREFIX.xml").write_text(
                "<espresso><creator/><atomic_structure/></espresso>\n"
            )
            for q_index in (1, 2):
                (phsave / f"patterns.{q_index}.xml").write_text(
                    "<Root><IRREPS_INFO>"
                    f"<QPOINT_NUMBER>{q_index}</QPOINT_NUMBER>"
                    "<NUMBER_IRR_REP>9</NUMBER_IRR_REP>"
                    "<DISPLACEMENT_PATTERN>1.0 0.0</DISPLACEMENT_PATTERN>"
                    "</IRREPS_INFO></Root>\n"
                )
            (template / "ph.in").write_text(
                "prefix=@PREFIX@ q=@Q_INDEX@ start=@START_IRR@ "
                "last=@LAST_IRR@\n@DFTD3_HESS_LINE@\n"
            )
            (template / "submit.sh").write_text(
                "job=@JOB_NAME@ prefix=@PREFIX@ q=@Q_INDEX@ "
                "start=@START_IRR@ last=@LAST_IRR@\n"
            )
            shared_save = root / "real.save"
            shared_save.mkdir()
            shared_hess = root / "real.hess"
            shared_hess.write_bytes(b"hess")
            output = root / "runs"

            rc = make_chunks.main(
                [
                    "--template", str(template),
                    "--output-dir", str(output),
                    "--prefix", "sample",
                    "--q-index", "2",
                    "--start-irr", "5",
                    "--last-irr", "9",
                    "--chunks", "2",
                    "--shared-save", str(shared_save),
                    "--shared-hess", str(shared_hess),
                ]
            )
            self.assertEqual(rc, 0)
            self.assertIn("start=5 last=7", (output / "chunk1" / "ph.in").read_text())
            self.assertIn("start=8 last=9", (output / "chunk2" / "ph.in").read_text())
            self.assertEqual((output / "chunk1" / "tmp" / "sample.save").resolve(), shared_save.resolve())
            self.assertEqual((output / "chunk1" / "sample.hess").resolve(), shared_hess.resolve())
            self.assertTrue((output / "chunks.q2.json").is_file())
            plan = json.loads((output / "chunks.q2.json").read_text())
            self.assertIn("scf_xml_sha256", plan["template_provenance"])

            (phsave / "patterns.1.xml").write_text("<patterns/>\n")
            with self.assertRaisesRegex(make_chunks.ConfigError, "missing QE"):
                make_chunks.validate_template(template, shared_save, shared_hess)

    def test_audit_detects_contiguous_irreps_and_modes(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            base = root / "base.dvscf"
            final = root / "final.dvscf"
            base.write_bytes(b"AAAA")
            final.write_bytes(b"AAAABBBBCCCCDDDD")

            chunk_dirs = []
            for index, (start, last, content) in enumerate(
                [(2, 2, bytes(4) + b"BBBBCCCC"), (3, 3, bytes(12) + b"DDDD")],
                start=1,
            ):
                chunk = root / f"chunk{index}"
                phsave = chunk / "tmp" / "_ph0" / "test.phsave"
                qdir = chunk / "tmp" / "_ph0" / "test.q_1"
                phsave.mkdir(parents=True)
                qdir.mkdir()
                (chunk / "ph.in").write_text(
                    "&inputph\n prefix='test'\n"
                    f" start_irr={start}, last_irr={last}\n/\n"
                )
                (chunk / "ph.out").write_text(PH_OUT)
                for irr in range(start, last + 1):
                    (phsave / f"dynmat.1.{irr}.xml").write_text("<ok/>\n")
                (phsave / "patterns.1.xml").write_text("<patterns/>\n")
                (qdir / "test.dvscf1").write_bytes(content)
                chunk_dirs.append(chunk)

            final_phsave = root / "final.phsave"
            final_phsave.mkdir()
            (final_phsave / "patterns.1.xml").write_text("<patterns/>\n")
            for irr in range(4):
                (final_phsave / f"dynmat.1.{irr}.xml").write_text("<ok/>\n")

            args = argparse.Namespace(
                q_index=1,
                record_bytes=4,
                base_dvscf=base,
                final_dvscf=final,
                final_phsave=final_phsave,
                chunk=chunk_dirs,
                check_records=True,
            )
            report = audit_chunks.audit(args)
            self.assertTrue(report["ok"], json.dumps(report, indent=2))
            self.assertEqual(report["total_modes"], 4)
            self.assertEqual(report["inferred_base_last_irrep"], 1)

            final_pattern = final_phsave / "patterns.1.xml"
            final_pattern.write_text("<different-calculation/>\n")
            self.assertTrue(any("pattern differs" in error
                                for error in audit_chunks.audit(args)["errors"]))
            final_pattern.write_text("<patterns/>\n")
            false_file = final_phsave / "dynmat.1.3.xml"
            false_file.unlink()
            false_file.mkdir()
            self.assertTrue(any("missing=[3]" in error
                                for error in audit_chunks.audit(args)["errors"]))
            false_file.rmdir()
            false_file.write_text("<ok/>\n")

            args.final_dvscf = None
            args.check_records = False
            structural_report = audit_chunks.audit(args)
            self.assertTrue(
                structural_report["ok"], json.dumps(structural_report, indent=2)
            )

            base.write_bytes(b"AAAABBBB")
            split_report = audit_chunks.audit(args)
            self.assertFalse(split_report["ok"])
            self.assertTrue(
                any("mode gap/overlap" in error for error in split_report["errors"])
            )

    def test_collection_stages_identical_duplicates_and_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            source1 = root / "source1"
            source2 = root / "source2"
            source1.mkdir()
            source2.mkdir()
            pattern1 = root / "pattern1.xml"
            pattern2 = root / "pattern2.xml"
            pattern1.write_text("<patterns><q>2</q></patterns>\n")
            pattern2.write_text(pattern1.read_text())
            (source1 / "dynmat.2.0.xml").write_text("<dynmat id='0'/>\n")
            (source1 / "dynmat.2.1.xml").write_text("<dynmat id='1'/>\n")
            (source2 / "dynmat.2.1.xml").write_text("<dynmat id='1'/>\n")
            (source2 / "dynmat.2.2.xml").write_text("<dynmat id='2'/>\n")
            output = root / "bundle"
            data = {
                "q_index": 2,
                "total_irreps": 2,
                "output": str(output),
                "pattern_sources": [str(pattern1), str(pattern2)],
                "dynmat_sources": [str(source1), str(source2)],
                "provenance": {"qe_version": "test"},
            }

            report = collect_dynmat.inspect(data)
            collect_dynmat.stage(report)
            self.assertTrue((output / "patterns.2.xml").is_file())
            self.assertTrue((output / "dynmat.2.2.xml").is_file())
            receipt = json.loads((output / "collection_receipt.json").read_text())
            self.assertEqual(receipt["provenance"]["qe_version"], "test")
            self.assertEqual(len(receipt["dynmat"][1]["all_sources"]), 2)

    def test_collection_rejects_conflicts_and_missing_tail(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            source1 = root / "source1"
            source2 = root / "source2"
            source1.mkdir()
            source2.mkdir()
            pattern = root / "patterns.1.xml"
            pattern.write_text("<patterns/>\n")
            (source1 / "dynmat.1.0.xml").write_text("<dynmat id='0'/>\n")
            (source1 / "dynmat.1.1.xml").write_text("<dynmat source='a'/>\n")
            (source2 / "dynmat.1.1.xml").write_text("<dynmat source='b'/>\n")
            data = {
                "q_index": 1,
                "total_irreps": 2,
                "output": str(root / "bundle"),
                "pattern_sources": [str(pattern)],
                "dynmat_sources": [str(source1), str(source2)],
            }
            with self.assertRaisesRegex(collect_dynmat.CollectionError, "coverage mismatch"):
                collect_dynmat.inspect(data)

            (source2 / "dynmat.1.2.xml").write_text("<dynmat id='2'/>\n")
            with self.assertRaisesRegex(collect_dynmat.CollectionError, "conflicting"):
                collect_dynmat.inspect(data)

    def test_standard_collector_discovers_q_points_across_images(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            work = Path(name)
            prefix = "sample"
            phsave = work / "tmp" / "_ph0" / f"{prefix}.phsave"
            phsave.mkdir(parents=True)
            (phsave / "empty-marker").touch()
            (work / f"{prefix}.dyn0").write_text(
                "3 1 1\n3\n0.0 0.0 0.0\n0.5 0.0 0.0\n0.0 0.5 0.0\n"
            )
            for q_index in range(1, 4):
                (phsave / f"patterns.{q_index}.xml").write_text("<patterns/>\n")
                (phsave / f"dynmat.{q_index}.0.xml").write_text("<dynmat/>\n")
                (work / f"{prefix}.dyn{q_index}.xml").write_text(
                    f"<dynamical-matrix q='{q_index}'/>\n"
                )

            (work / "tmp" / "_ph0" / f"{prefix}.dvscf1").write_bytes(b"AAAA")
            for image, q_index, content in ((0, 2, b"BBBB"), (1, 3, b"CCCC")):
                qdir = work / "tmp" / f"_ph{image}" / f"{prefix}.q_{q_index}"
                qdir.mkdir(parents=True)
                (qdir / f"{prefix}.dvscf1").write_bytes(content)

            script = ROOT / "examples/qe_chunked_irreps/ph_collect.sh"
            env = os.environ.copy()
            env.update({"PREFIX": prefix, "WORK_ROOT": str(work), "PYTHON": sys.executable})
            conflict_dir = work / "tmp" / "_ph1" / f"{prefix}.q_2"
            conflict_dir.mkdir()
            conflict = conflict_dir / f"{prefix}.dvscf1"
            conflict.write_bytes(b"ZZZZ")
            rejected = subprocess.run(
                ["bash", str(script)],
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(rejected.returncode, 2)
            self.assertIn("conflicting contents", rejected.stderr)
            self.assertFalse((work / "save.partial").exists())
            conflict.unlink()

            completed = subprocess.run(
                ["bash", str(script)],
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual((work / "save" / f"{prefix}.dvscf_q1").read_bytes(), b"AAAA")
            self.assertEqual((work / "save" / f"{prefix}.dvscf_q2").read_bytes(), b"BBBB")
            self.assertEqual((work / "save" / f"{prefix}.dvscf_q3").read_bytes(), b"CCCC")
            self.assertEqual((work / "save" / f"{prefix}.phsave/empty-marker").read_bytes(), b"")

            repeated = subprocess.run(
                ["bash", str(script)],
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(repeated.returncode, 2)
            self.assertIn("refusing existing output", repeated.stderr)


@pytest.fixture
def chunk_request(tmp_path):
    template = tmp_path / "template"
    phsave = template / "tmp/_ph0/PREFIX.phsave"
    phsave.mkdir(parents=True)
    (template / "PREFIX.dyn0").write_text("1 1 1\n1\n0 0 0\n")
    (template / "tmp/PREFIX.xml").write_text("<root><creator/><atomic_structure/></root>")
    (phsave / "patterns.1.xml").write_text(
        "<Root><IRREPS_INFO><QPOINT_NUMBER>1</QPOINT_NUMBER>"
        "<NUMBER_IRR_REP>3</NUMBER_IRR_REP>"
        "<DISPLACEMENT_PATTERN>1 0 0</DISPLACEMENT_PATTERN></IRREPS_INFO></Root>"
    )
    (template / "ph.in").write_text(
        "@PREFIX@ @Q_INDEX@ @START_IRR@ @LAST_IRR@ @DFTD3_HESS_LINE@\n"
    )
    (template / "submit.sh").write_text(
        "@PREFIX@ @Q_INDEX@ @START_IRR@ @LAST_IRR@ @JOB_NAME@\n"
    )
    shared = tmp_path / "shared.save"
    shared.mkdir()
    return dict(template=template, output_dir=tmp_path / "runs", prefix="test",
                q_index=1, start_irr=1, last_irr=3, chunks=2, shared_save=shared)


@pytest.mark.parametrize("field,value", [
    ("prefix", "../outside"), ("prefix", "'quoted"),
    ("directory_prefix", "/absolute"), ("job_prefix", "job\ncommand"),
])
def test_generator_rejects_unsafe_names_before_creating_outputs(chunk_request, field, value):
    with pytest.raises(make_chunks.ConfigError, match="plain name"):
        make_chunks.make_chunks(**dict(chunk_request, **{field: value}))
    assert not chunk_request["output_dir"].exists()


def test_generator_checks_pattern_irrep_limit_and_output_location(chunk_request):
    with pytest.raises(make_chunks.ConfigError, match="exceeds"):
        make_chunks.make_chunks(**dict(chunk_request, last_irr=4))
    with pytest.raises(make_chunks.ConfigError, match="inside the template"):
        make_chunks.make_chunks(**dict(chunk_request, output_dir=chunk_request["template"] / "runs"))
    assert not chunk_request["output_dir"].exists()


def test_generator_dry_run_and_existing_partial_preservation(chunk_request):
    plan = make_chunks.make_chunks(**dict(chunk_request, dry_run=True))
    assert [part["last_irr"] for part in plan["chunks"]] == [2, 3]
    output = chunk_request["output_dir"]
    assert not output.exists()
    output.mkdir()
    partial = output / "chunk1.partial"
    partial.write_bytes(b"other process evidence")
    with pytest.raises(FileExistsError):
        make_chunks.make_chunks(**chunk_request)
    assert partial.read_bytes() == b"other process evidence"
    assert sorted(path.name for path in output.iterdir()) == ["chunk1.partial"]


def test_generator_rejects_writable_template_symlink(chunk_request, tmp_path):
    outside = tmp_path / "outside.sh"
    outside.write_text("important")
    script = chunk_request["template"] / "submit.sh"
    script.unlink()
    script.symlink_to(outside)
    with pytest.raises(make_chunks.ConfigError, match="symlinks"):
        make_chunks.make_chunks(**chunk_request)
    assert outside.read_text() == "important"


@pytest.mark.parametrize("filename", ["ph.in", "submit.sh"])
def test_generator_detects_changed_input_or_script(chunk_request, tmp_path, filename):
    _, provenance = make_chunks.validate_template(
        chunk_request["template"], chunk_request["shared_save"], None)
    path = chunk_request["template"] / filename
    path.write_text(path.read_text() + "\nchanged settings\n")
    with pytest.raises(make_chunks.ConfigError, match="changed after validation"):
        make_chunks.build_chunk(chunk_request["template"], tmp_path / "staged",
            "test", 1, 1, 3, "job", chunk_request["shared_save"], None, provenance)


@pytest.fixture
def merge_manifest(tmp_path):
    source = tmp_path / "source"
    source.write_bytes(b"AAAABBBB")
    return {"record_bytes": 4, "total_modes": 2, "output": str(tmp_path / "output"),
            "segments": [{"path": str(source), "first_mode": 1, "last_mode": 2}]}


@pytest.mark.parametrize("suffix", ["", ".partial", ".receipt.json", ".receipt.json.partial"])
def test_merge_preserves_existing_outputs_and_partials(merge_manifest, suffix):
    existing = Path(merge_manifest["output"] + suffix)
    existing.write_bytes(b"existing evidence")
    with pytest.raises(merge_dvscf.MergeError, match="existing"):
        merge_dvscf.merge(merge_manifest)
    assert existing.read_bytes() == b"existing evidence"


def test_merge_preserves_dangling_output_symlink(merge_manifest, tmp_path):
    target = Path(merge_manifest["output"])
    target.symlink_to(tmp_path / "absent")
    with pytest.raises(merge_dvscf.MergeError, match="existing"):
        merge_dvscf.merge(merge_manifest)
    assert target.is_symlink()
    assert not (tmp_path / "absent").exists()


@pytest.mark.parametrize("block_bytes", [0, -1])
def test_merge_rejects_nonpositive_block_size(merge_manifest, block_bytes):
    with pytest.raises(merge_dvscf.MergeError, match="positive"):
        merge_dvscf.merge(merge_manifest, block_bytes)
    assert not Path(merge_manifest["output"]).exists()


@pytest.mark.parametrize("field,value", [
    ("record_bytes", 4.9), ("total_modes", True), ("record_bytes", 0),
    ("first_mode", 1.9), ("last_mode", False),
])
def test_merge_rejects_noninteger_record_counts(merge_manifest, field, value):
    if field in {"first_mode", "last_mode"}:
        merge_manifest["segments"][0][field] = value
    else:
        merge_manifest[field] = value
    with pytest.raises(ValueError, match="positive integer"):
        merge_dvscf.merge(merge_manifest)
    assert not Path(merge_manifest["output"]).exists()


@pytest.mark.parametrize("field,value", [("q_index", 1.9), ("total_irreps", True)])
def test_dynmat_rejects_noninteger_indices(tmp_path, field, value):
    data = {"q_index": 1, "total_irreps": 1, "output": str(tmp_path / "output")}
    data[field] = value
    with pytest.raises(ValueError, match="positive integer"):
        collect_dynmat.inspect(data)


def test_merge_zero_record_failure_cleans_only_owned_staging(merge_manifest, tmp_path):
    source = Path(merge_manifest["segments"][0]["path"])
    source.write_bytes(b"AAAA" + bytes(4))
    before = {path.name for path in tmp_path.iterdir()}
    with pytest.raises(merge_dvscf.MergeError, match="all zero"):
        merge_dvscf.merge(merge_manifest)
    assert {path.name for path in tmp_path.iterdir()} == before
    assert source.read_bytes() == b"AAAA" + bytes(4)


def test_merge_publication_race_cannot_replace_other_output(merge_manifest, monkeypatch):
    publish = merge_dvscf.publish_file
    output = Path(merge_manifest["output"])

    def race(source, destination):
        if destination == output:
            output.write_bytes(b"another completed output")
        publish(source, destination)

    monkeypatch.setattr(merge_dvscf, "publish_file", race)
    with pytest.raises(FileExistsError):
        merge_dvscf.merge(merge_manifest)
    assert output.read_bytes() == b"another completed output"
    assert not output.with_name(output.name + ".receipt.json").exists()


@pytest.fixture
def dynmat_report(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    pattern = source / "patterns.1.xml"
    pattern.write_text("<pattern/>")
    for irrep in (0, 1):
        (source / f"dynmat.1.{irrep}.xml").write_text(f"<dynmat id='{irrep}'/>")
    return collect_dynmat.inspect({"q_index": 1, "total_irreps": 1,
        "output": str(tmp_path / "bundle"), "pattern_sources": [str(pattern)],
        "dynmat_sources": [str(source)]})


@pytest.mark.parametrize("suffix", ["", ".partial", ".publish-lock"])
def test_dynmat_staging_preserves_existing_directories(dynmat_report, suffix):
    existing = Path(dynmat_report["output"] + suffix)
    existing.mkdir()
    (existing / "evidence").write_text("retain")
    with pytest.raises(FileExistsError):
        collect_dynmat.stage(dynmat_report)
    assert (existing / "evidence").read_text() == "retain"


def test_dynmat_detects_source_change_after_inspection(dynmat_report, tmp_path):
    source = Path(dynmat_report["dynmat"][1]["selected_source"])
    source.write_text("<changed/>")
    with pytest.raises(collect_dynmat.CollectionError, match="differs"):
        collect_dynmat.stage(dynmat_report)
    assert source.read_text() == "<changed/>"
    assert not Path(dynmat_report["output"]).exists()
    assert not list(tmp_path.glob(".dynmat-*"))


@pytest.mark.parametrize("changes", [{"record_bytes": 0}, {"q_index": 0}, {"chunk": ()}])
def test_audit_rejects_invalid_configuration_without_reading_files(changes):
    config = dict(q_index=1, record_bytes=4, base_dvscf=Path("missing"), chunk=(Path("missing"),))
    with pytest.raises(audit_chunks.AuditError, match="positive/nonempty"):
        audit_chunks.audit(audit_chunks.AuditConfig(**dict(config, **changes)))


def test_audit_rejects_nonmonotone_mode_map_that_hides_a_missing_record(tmp_path):
    base = tmp_path / "base"
    base.write_bytes(b"AAAA")
    chunk = tmp_path / "chunk"
    phsave = chunk / "tmp/_ph0/test.phsave"
    phsave.mkdir(parents=True)
    qdir = chunk / "tmp/_ph0/test.q_1"
    qdir.mkdir()
    (chunk / "ph.in").write_text("prefix='test' start_irr=2, last_irr=3")
    (chunk / "ph.out").write_text(PH_OUT.replace("modes # 2 3", "modes # 2 4")
                                          .replace("3 mode # 4", "3 mode # 3"))
    (qdir / "test.dvscf1").write_bytes(bytes(4) + b"BBBBCCCC")
    (phsave / "patterns.1.xml").write_text("<patterns/>")
    for irrep in (2, 3):
        (phsave / f"dynmat.1.{irrep}.xml").write_text("<dynmat/>")
    report = audit_chunks.audit(audit_chunks.AuditConfig(1, 4, base, (chunk,)))
    assert not report["ok"]
    assert any("chunk mode mapping" in error for error in report["errors"])


def test_standard_collector_checks_staged_paths_and_changed_source(tmp_path):
    source = tmp_path / "source"
    source.write_bytes(b"data")
    record = {"destination": "../outside", "selected_source": str(source),
              "size": 4, "sha256": collect_dynmat.sha256_file(source)}
    report = {"output": str(tmp_path / "bundle"), "files": [record]}
    with pytest.raises(collect_phonons.CollectionError, match="unsafe destination"):
        collect_phonons.stage(report)
    record["destination"] = "safe"
    source.write_bytes(b"modified")
    with pytest.raises(collect_phonons.CollectionError, match="differs"):
        collect_phonons.stage(report)
    assert not (tmp_path / "bundle").exists()
    assert not (tmp_path / "outside").exists()


def test_five_cli_commands_complete_synthetic_file_workflow(chunk_request, tmp_path):
    """Exercise installed module commands from outside the repository directory."""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")

    def run(module, *arguments):
        completed = subprocess.run(
            [sys.executable, "-m", f"pyeph.preprocessing.qe.{module}", *map(str, arguments)],
            cwd=tmp_path, env=env, text=True, capture_output=True, check=False,
        )
        assert completed.returncode == 0, completed.stderr + completed.stdout
        return completed.stdout

    runs = chunk_request["output_dir"]
    run("make_chunks", "--template", chunk_request["template"], "--output-dir", runs,
        "--prefix", "test", "--q-index", 1, "--start-irr", 2, "--last-irr", 3,
        "--chunks", 2, "--shared-save", chunk_request["shared_save"])
    base = tmp_path / "base"
    base.write_bytes(b"AAAA")
    phsave_base = tmp_path / "base.phsave"
    phsave_base.mkdir()
    for irrep in (0, 1):
        (phsave_base / f"dynmat.1.{irrep}.xml").write_text("<dynmat/>")
    pattern = chunk_request["template"] / "tmp/_ph0/PREFIX.phsave/patterns.1.xml"
    chunks = [runs / "chunk1", runs / "chunk2"]
    sources = [phsave_base]
    segments = [{"path": str(base), "first_mode": 1, "last_mode": 1}]
    mode_output = PH_OUT.replace("modes # 2 3", "mode # 2").replace("3 mode # 4", "3 mode # 3")
    for mode, chunk in zip((2, 3), chunks):
        (chunk / "ph.in").write_text(f"prefix='test' start_irr={mode} last_irr={mode}")
        (chunk / "ph.out").write_text(mode_output)
        phsave = chunk / "tmp/_ph0/test.phsave"
        (phsave / f"dynmat.1.{mode}.xml").write_text("<dynmat/>")
        sources.append(phsave)
        qdir = chunk / "tmp/_ph0/test.q_1"
        qdir.mkdir()
        dvscf = qdir / "test.dvscf1"
        dvscf.write_bytes(bytes(4 * (mode - 1)) + (b"BBBB" if mode == 2 else b"CCCC"))
        segments.append({"path": str(dvscf), "first_mode": mode, "last_mode": mode})
    dynmat_manifest = tmp_path / "collect.json"
    bundle = tmp_path / "phsave-bundle"
    dynmat_manifest.write_text(json.dumps({"q_index": 1, "total_irreps": 3,
        "output": str(bundle), "pattern_sources": [str(pattern)],
        "dynmat_sources": list(map(str, sources))}))
    run("collect_dynmat", dynmat_manifest)
    merged = tmp_path / "merged"
    merge_json = tmp_path / "merge.json"
    merge_json.write_text(json.dumps({"record_bytes": 4, "total_modes": 3,
                                    "output": str(merged), "segments": segments}))
    run("merge_dvscf", merge_json)
    assert merged.read_bytes() == b"AAAABBBBCCCC"
    audit = json.loads(run("audit_chunks", "--q-index", 1, "--record-bytes", 4,
        "--base-dvscf", base, "--final-dvscf", merged, "--final-phsave", bundle,
        "--chunk", chunks[0], "--chunk", chunks[1], "--check-all-records"))
    assert audit["ok"] and audit["records_checked"] == 3
    work = tmp_path / "completed"
    work.mkdir()
    scratch = work / "tmp/_ph0"
    scratch.mkdir(parents=True)
    shutil.copytree(bundle, scratch / "test.phsave")
    shutil.copyfile(merged, scratch / "test.dvscf1")
    shutil.copyfile(chunk_request["template"] / "PREFIX.dyn0", work / "test.dyn0")
    (work / "test.dyn1.xml").write_text("<dynamical-matrix/>")
    result = json.loads(run("collect_phonons", "--prefix", "test", "--work-root", work))
    assert result["q_points"] == 1 and result["dvscf_bytes_per_q"] == 12
    assert (work / "save/test.dvscf_q1").read_bytes() == b"AAAABBBBCCCC"
    assert (work / "save/collection_receipt.json").is_file()


if __name__ == "__main__":
    unittest.main()
