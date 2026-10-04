"""Independent installation-isolation checks using generated distributions."""

import importlib.metadata
import io
import json
import os
from pathlib import Path
import py_compile
import subprocess
import tarfile
import venv
import zipfile

import pytest


@pytest.fixture
def distribution_fixture(tmp_path):
    environment = tmp_path / "environment"
    # Managed POSIX interpreters can locate libpython relative to the executable;
    # copying only that executable breaks its loader path. Preserve the origin.
    venv.EnvBuilder(with_pip=False, symlinks=os.name != "nt").create(environment)
    executable = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    query = subprocess.run([str(executable), "-c", "import site; print(site.getsitepackages()[0])"],
                           check=True, text=True, capture_output=True)
    site_packages = Path(query.stdout.strip())
    # Reuse numerical/test dependencies, keeping the generated installation
    # first. The fixture contains no runtime checkout or copied project source.
    dependencies = Path(importlib.metadata.distribution("pytest").locate_file(""))
    (site_packages / "fixture_dependencies.pth").write_text(str(dependencies) + "\n")
    metadata = b"Metadata-Version: 2.1\nName: pyeph\nVersion: 0.0.0\n"
    wheel_metadata = b"Wheel-Version: 1.0\nGenerator: fixture\nRoot-Is-Purelib: true\nTag: py3-none-any\n"
    entries = {
        "pyeph/__init__.py": b"MARKER = 1\n",
        "pyeph-0.0.0.dist-info/METADATA": metadata,
        "pyeph-0.0.0.dist-info/WHEEL": wheel_metadata,
    }
    entries["pyeph-0.0.0.dist-info/RECORD"] = "".join(
        f"{name},,\n" for name in [*entries, "pyeph-0.0.0.dist-info/RECORD"]
    ).encode()
    for name, data in entries.items():
        target = site_packages / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    wheel = tmp_path / "fixture.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        for name, data in entries.items():
            archive.writestr(name, data)
    files = {
        "src/pyeph/__init__.py": entries["pyeph/__init__.py"],
        "tests/test_probe.py": (
            "from pathlib import Path\n"
            "import pyeph\n"
            "def test_record_imported_code():\n"
            "    Path('executed-marker.txt').write_text(str(pyeph.MARKER))\n"
            "def test_second_probe():\n"
            "    assert pyeph.MARKER == 1\n"
        ).encode(),
    }
    files["release-files.txt"] = "".join(f"{name}\n" for name in [*files, "release-files.txt"]).encode()
    source = tmp_path / "fixture.tar.gz"
    with tarfile.open(source, "w:gz") as archive:
        for name, data in files.items():
            member = tarfile.TarInfo("fixture-0.0.0/" + name)
            member.size = len(data)
            archive.addfile(member, io.BytesIO(data))

    def qualify(*pytest_args, pytest_addopts=None):
        destination = tmp_path / "qualification"
        environment = {**os.environ, "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1"}
        environment.pop("PYTEST_ADDOPTS", None)
        environment.pop("PYTEST_PLUGINS", None)
        if pytest_addopts is not None:
            environment["PYTEST_ADDOPTS"] = pytest_addopts
        process = subprocess.run(
            [str(executable), str(Path(__file__).resolve().parents[1] / "tools/qualify_distribution.py"),
             "--sdist", str(source), "--wheel", str(wheel), "--destination", str(destination),
             *(["--pytest-args", *pytest_args] if pytest_args else [])],
            env=environment,
            capture_output=True, text=True, timeout=90,
        )
        return process, destination

    return site_packages, qualify, source, wheel


def test_generated_installed_wheel_qualifies_without_runtime_source_tree(distribution_fixture):
    _, qualify, _, _ = distribution_fixture
    process, destination = qualify()
    assert process.returncode == 0, process.stdout + process.stderr
    record = json.loads((destination / "qualification.json").read_text())
    assert record["runtime_unchanged"]
    assert record["pytest_args"] == ["tests", "-q", "--durations=20"]
    assert record["pytest_collection"]["finished"]
    assert record["pytest_collection"]["selected_count"] == 2
    assert record["pytest_collection"]["deselected_count"] == 0
    assert not (destination / "src").exists()
    assert (destination / "executed-marker.txt").read_text() == "1"


@pytest.mark.parametrize("selection_source", ["arguments", "environment"])
def test_qualification_records_actual_test_selection(distribution_fixture, selection_source):
    import hashlib

    _, qualify, _, _ = distribution_fixture
    if selection_source == "arguments":
        process, destination = qualify("-q", "-k", "record_imported")
    else:
        process, destination = qualify(pytest_addopts="-k record_imported")
    assert process.returncode == 0, process.stdout + process.stderr
    record = json.loads((destination / "qualification.json").read_text())
    inputs = json.loads((destination / "qualification-input.json").read_text())
    assert inputs["pytest_args"] == record["pytest_args"]
    assert inputs["pytest_environment"] == record["pytest_environment"]
    if selection_source == "arguments":
        assert record["pytest_args"] == ["tests", "-q", "-k", "record_imported"]
    else:
        assert record["pytest_environment"]["PYTEST_ADDOPTS"] == "-k record_imported"
    expected = hashlib.sha256(json.dumps(
        ["tests/test_probe.py::test_record_imported_code"],
        ensure_ascii=True, separators=(",", ":")).encode()).hexdigest()
    assert record["pytest_collection"] == dict(
        finished=True, selected_count=1, deselected_count=1, ordered_nodeids_sha256=expected)


@pytest.mark.parametrize("contamination", ["editable", "extra-data", "source", "metadata", "ownership"])
def test_qualification_rejects_installation_mismatch(distribution_fixture, contamination):
    site_packages, qualify, _, _ = distribution_fixture
    dist_info = site_packages / "pyeph-0.0.0.dist-info"
    if contamination == "editable":
        (dist_info / "direct_url.json").write_text(json.dumps({"dir_info": {"editable": True}}))
    elif contamination == "extra-data":
        (site_packages / "pyeph/unlisted.json").write_text("{}")
    elif contamination == "source":
        (site_packages / "pyeph/__init__.py").write_text("MARKER = 2\n")
    elif contamination == "metadata":
        (dist_info / "WHEEL").write_text("Wheel-Version: 1.0\nGenerator: changed\n")
    else:
        (dist_info / "RECORD").write_text("")
    process, destination = qualify()
    assert process.returncode != 0, process.stdout + process.stderr
    assert not (destination / "executed-marker.txt").exists()


def test_timestamp_valid_stale_bytecode_cannot_replace_inventoried_source(distribution_fixture):
    site_packages, qualify, _, _ = distribution_fixture
    source = site_packages / "pyeph/__init__.py"
    original = source.read_bytes()
    source.write_bytes(b"MARKER = 2\n")
    py_compile.compile(str(source), doraise=True, invalidation_mode=py_compile.PycInvalidationMode.TIMESTAMP)
    stamp = source.stat()
    source.write_bytes(original)
    os.utime(source, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
    process, destination = qualify()
    if process.returncode == 0:
        assert (destination / "executed-marker.txt").read_text() == "1", (
            "qualification succeeded while Python executed stale code that differs from the audited source"
        )


@pytest.mark.parametrize("contamination", ["wheel-extra", "duplicate-inventory"])
def test_qualification_uses_strict_artifact_inventory(distribution_fixture, contamination):
    _, qualify, source, wheel = distribution_fixture
    if contamination == "wheel-extra":
        with zipfile.ZipFile(wheel, "a") as archive:
            archive.writestr("unlisted.txt", "Unlisted artifact content")
    else:
        with tarfile.open(source, "r:gz") as archive:
            entries = [(member.name, archive.extractfile(member).read()) for member in archive]
        with tarfile.open(source, "w:gz") as archive:
            for name, data in entries:
                if name.endswith("release-files.txt"):
                    data += b"release-files.txt\n"
                member = tarfile.TarInfo(name)
                member.size = len(data)
                archive.addfile(member, io.BytesIO(data))
    process, destination = qualify()
    assert process.returncode != 0, process.stdout + process.stderr
    assert not (destination / "executed-marker.txt").exists()
