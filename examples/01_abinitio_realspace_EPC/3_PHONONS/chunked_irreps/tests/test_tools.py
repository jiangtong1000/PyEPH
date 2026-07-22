from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import audit_chunks
import collect_dynmat
import make_chunks
import merge_dvscf


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
                    "<NUMBER_IRR_REP>2</NUMBER_IRR_REP>"
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
            self.assertEqual((output / "chunk1" / "tmp" / "sample.save").resolve(), shared_save)
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

            script = ROOT.parent / "ph_collect.sh"
            env = os.environ.copy()
            env.update({"PREFIX": prefix, "WORK_ROOT": str(work)})
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
            shutil.rmtree(work / "save.partial")
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

            repeated = subprocess.run(
                ["bash", str(script)],
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(repeated.returncode, 2)
            self.assertIn("refusing existing output", repeated.stderr)


if __name__ == "__main__":
    unittest.main()
