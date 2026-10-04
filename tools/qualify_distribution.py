"""Test source-archive fixtures against an already installed matching wheel.

Run with the intended dependency environment's Python. The new destination
receives tests/examples/benchmarks and records; it deliberately has no src tree.
The installed package must be outside both the source checkout and destination.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys
import tarfile
import tempfile
import zipfile

if __package__:
    from .release import parse_inventory, relative_path
else:
    from release import parse_inventory, relative_path

WHEEL_METADATA = {"WHEEL", "METADATA", "RECORD", "top_level.txt", "licenses/LICENSE"}
SDIST_GENERATED = {"PKG-INFO", "setup.cfg", "src/pyeph.egg-info/PKG-INFO",
                   "src/pyeph.egg-info/SOURCES.txt", "src/pyeph.egg-info/dependency_links.txt",
                   "src/pyeph.egg-info/requires.txt", "src/pyeph.egg-info/top_level.txt"}


def digest(payload):
    return hashlib.sha256(payload).hexdigest()


def extract_support(archive_path, destination):
    """Extract only inventory entries after validating archive paths and types."""
    with tarfile.open(archive_path, "r:gz") as archive:
        entries = {}
        top = None
        for member in archive:
            relative_path(member.name.rstrip("/"))
            name = PurePosixPath(member.name)
            if name.is_absolute() or any(x in ("", ".", "..") for x in name.parts):
                raise ValueError("source archive contains an invalid path")
            top = name.parts[0] if top is None else top
            if name.parts[0] != top or not (member.isfile() or member.isdir()):
                raise ValueError("source archive must have one root and only regular files")
            if member.isdir():
                continue
            relative = PurePosixPath(*name.parts[1:]).as_posix()
            if relative in entries:
                raise ValueError("source archive contains duplicate files")
            entries[relative] = archive.extractfile(member).read()
    names = set(parse_inventory(entries["release-files.txt"].decode()))
    missing = names - set(entries)
    if missing:
        raise ValueError(f"source archive is missing inventory entries: {sorted(missing)}")
    extra = set(entries)-names-SDIST_GENERATED
    if extra:
        raise ValueError(f"source archive has entries outside the inventory: {sorted(extra)}")
    runtime = {name.removeprefix("src/pyeph/"): digest(entries[name]) for name in names
               if name.startswith("src/pyeph/")}
    for name in sorted(names):
        if name.startswith("src/"):
            continue
        path = destination / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(entries[name])
    return runtime


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sdist", type=Path, required=True)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--pytest-args", nargs=argparse.REMAINDER,
                        default=["-q", "--durations=20"],
                        help="pytest arguments; place this option last")
    args = parser.parse_args()
    destination = args.destination.resolve()
    destination.mkdir(parents=True, exist_ok=False)
    runtime = extract_support(args.sdist, destination)
    with zipfile.ZipFile(args.wheel) as wheel:
        if len(set(wheel.namelist())) != len(wheel.namelist()):
            raise ValueError("wheel contains duplicate entries")
        metadata_roots = set()
        for name in wheel.namelist():
            path = relative_path(name.rstrip("/"))
            if name.startswith("pyeph/"):
                continue
            if len(path.parts) < 2 or not path.parts[0].startswith("pyeph-") or not path.parts[0].endswith(".dist-info"):
                raise ValueError("wheel contains an unsupported member outside runtime/metadata")
            metadata_roots.add(path.parts[0])
            if "/".join(path.parts[1:]) not in WHEEL_METADATA:
                raise ValueError("wheel contains an unsupported metadata member")
        if len(metadata_roots) != 1:
            raise ValueError("wheel must contain exactly one distribution metadata root")
        wheel_runtime = {name.removeprefix("pyeph/"): digest(wheel.read(name))
                         for name in wheel.namelist()
                         if name.startswith("pyeph/") and not name.endswith("/")}
        wheel_metadata = {PurePosixPath(name).name: wheel.read(name).decode()
                          for name in wheel.namelist()
                          if name.endswith((".dist-info/WHEEL", ".dist-info/METADATA"))}
        if wheel_runtime != runtime or set(wheel_metadata) != {"WHEEL", "METADATA"}:
            raise ValueError("wheel runtime or metadata differs from the source inventory")
    record = dict(schema="pyeph.distribution-qualification.v1",
                  started_utc=datetime.now(timezone.utc).isoformat(),
                  source_archive_sha256=digest(args.sdist.read_bytes()),
                  qualifier_sha256=digest(Path(__file__).read_bytes()),
                  wheel_sha256=digest(args.wheel.read_bytes()), wheel_metadata=wheel_metadata,
                  expected_runtime_sha256=runtime, python_executable=sys.executable,
                  pytest_args=["tests", *args.pytest_args],
                  pytest_environment={key: os.environ[key] for key in
                                      ("PYTEST_ADDOPTS", "PYTEST_PLUGINS",
                                       "PYTEST_DISABLE_PLUGIN_AUTOLOAD") if key in os.environ})
    (destination / "qualification-input.json").write_text(json.dumps(record, indent=2)+"\n")
    # Use isolated Python to remove the invoking checkout and PYTHONPATH. Insert
    # only copied test-support modules, never a runtime source directory.
    script = '''
import hashlib, importlib.metadata, json, os, platform, sys
from pathlib import Path
support, checkout = map(Path, sys.argv[1:3])
sys.path.insert(0, str(support))
import pyeph
root = Path(pyeph.__file__).resolve().parent
if root.is_relative_to(support) or root.is_relative_to(checkout):
    raise RuntimeError("qualification imported a source checkout, not an installed wheel")
record = json.loads((support/"qualification-input.json").read_text())
distribution = importlib.metadata.distribution("pyeph")
direct = json.loads(distribution.read_text("direct_url.json") or "{}")
if direct.get("dir_info", {}).get("editable"):
    raise RuntimeError("qualification rejects editable installs")
if Path(distribution.locate_file("pyeph/__init__.py")).resolve() != root/"__init__.py":
    raise RuntimeError("imported runtime is not owned by the installed distribution")
for name, expected in record["wheel_metadata"].items():
    if distribution.read_text(name) != expected:
        raise RuntimeError("installed distribution metadata differs from the wheel")
owned = {str(p) for p in distribution.files or ()}
if any("pyeph/"+name not in owned for name in record["expected_runtime_sha256"]):
    raise RuntimeError("installed distribution does not own every expected runtime file")
def runtime_hashes():
    return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in root.rglob("*") if p.is_file()
            and not ("__pycache__" in p.parts and p.suffix == ".pyc")}
actual = runtime_hashes()
if actual != record["expected_runtime_sha256"]:
    raise RuntimeError("installed wheel sources differ from the source archive")
record.update(installed_package=str(root), platform=platform.platform(),
              python=sys.version, versions={name: importlib.metadata.version(name)
              for name in ("pyeph", "jax", "jaxlib", "numpy", "scipy", "h5py", "pytest")})
import jax
record["devices"] = [str(device) for device in jax.devices()]
import pytest
class CollectionRecord:
    def __init__(self):
        self.deselected = 0
        self.selected = []
        self.collection_finished = False

    def pytest_deselected(self, items):
        self.deselected += len(items)

    def pytest_collection_finish(self, session):
        self.selected = [item.nodeid for item in session.items]
        self.collection_finished = True

selection = CollectionRecord()
# Pytest can prepend environment/configuration arguments to the supplied list.
# Preserve the recorded invocation; its additional inputs are recorded above.
code = pytest.main(list(record["pytest_args"]), plugins=[selection])
record["pytest_collection"] = dict(
    finished=selection.collection_finished, selected_count=len(selection.selected),
    deselected_count=selection.deselected,
    ordered_nodeids_sha256=hashlib.sha256(json.dumps(
        selection.selected, ensure_ascii=True, separators=(",", ":")).encode()).hexdigest())
record["pytest_exit_code"] = int(code)
record["runtime_unchanged"] = actual == runtime_hashes()
(support/"qualification.json").write_text(json.dumps(record, indent=2)+"\\n")
raise SystemExit(code if record["runtime_unchanged"] else 1)
'''
    environment = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
    environment["JAX_ENABLE_X64"] = "1"
    # -B prevents writes but still reads existing bytecode. A fresh, empty
    # prefix also prevents timestamp-valid stale installed caches from masking
    # different source bytes during the first import.
    bytecode = tempfile.mkdtemp(prefix="bytecode-", dir=destination)
    command = [sys.executable, "-I", "-B", "-X", f"pycache_prefix={bytecode}",
               "-c", script, str(destination),
               str(Path(__file__).resolve().parents[1]), *args.pytest_args]
    with (destination / "pytest.log").open("w") as log:
        process = subprocess.Popen(command, cwd=destination, env=environment,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        code = process.wait()
    raise SystemExit(code)


if __name__ == "__main__":
    main()
