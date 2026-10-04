"""Publication boundary checks, including compressed and versioned content."""

import io
import json
import re
import subprocess
import sys
import tarfile
import zipfile

import h5py
import numpy as np
import pytest

from tools.release import ReleaseError, audit, audit_git, export, inspect_bytes, inventory


PATTERNS = [re.compile("private_fixture_token", re.IGNORECASE)]


def test_explicit_inventory_export_does_not_copy_unlisted_workspace(tmp_path):
    source = tmp_path/"source"
    source.mkdir()
    (source/"README.md").write_text("Public instructions.\n")
    (source/"private.md").write_text("private_fixture_token")
    (source/"release-files.txt").write_text("README.md\nrelease-files.txt\n")
    names = inventory(source/"release-files.txt")
    output = tmp_path/"release"
    report = export(source, output, names, PATTERNS)
    assert report == audit(output, names, PATTERNS)
    assert sorted(p.name for p in output.iterdir()) == names
    with pytest.raises(FileExistsError):
        export(source, output, names, PATTERNS)


@pytest.mark.parametrize("entry", ["../outside.py", "/outside.py", "a//b.py", "./a.py",
                                  "a\\b.py", ".git/config", "a.py\na.py"])
def test_inventory_rejects_ambiguous_paths_and_duplicates(tmp_path, entry):
    path = tmp_path/"files.txt"
    path.write_text(entry+"\n")
    with pytest.raises(ReleaseError):
        inventory(path)


def test_symlinked_parent_is_not_a_release_file(tmp_path):
    source = tmp_path/"source"
    source.mkdir()
    outside = tmp_path/"external"
    outside.mkdir()
    (outside/"code.py").write_text("value = 1\n")
    (source/"linked").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ReleaseError, match="symlink"):
        audit(source, ["linked/code.py"])


def test_nested_archive_content_is_inspected_and_match_is_not_disclosed():
    nested = io.BytesIO()
    with tarfile.open(fileobj=nested, mode="w:gz") as archive:
        payload = b"private_fixture_token"
        member = tarfile.TarInfo("data.json")
        member.size = len(payload)
        archive.addfile(member, io.BytesIO(payload))
    outer = io.BytesIO()
    with zipfile.ZipFile(outer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("nested.tar.gz", nested.getvalue())
    with pytest.raises(ReleaseError) as raised:
        inspect_bytes("fixture.zip", outer.getvalue(), PATTERNS)
    assert "restricted content" in str(raised.value)
    assert "private_fixture_token" not in str(raised.value)


def test_numeric_array_and_archive_license_are_supported():
    buffer = io.BytesIO()
    np.savez_compressed(buffer, values=np.arange(10))
    inspect_bytes("arrays.npz", buffer.getvalue(), PATTERNS)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("LICENSE", "License terms preserved.")
    inspect_bytes("source.zip", buffer.getvalue(), PATTERNS)


@pytest.mark.parametrize("payload", [
    b'{"source":"\\u0070rivate_fixture_token"}',
    b'{"source":"&#112;rivate_fixture_token"}',
    'ｐrivate_fixture_token'.encode(),
])
def test_encoded_text_is_inspected_semantically(payload):
    with pytest.raises(ReleaseError, match="restricted"):
        inspect_bytes("metadata.json", payload, PATTERNS)


def test_object_arrays_and_unknown_binary_formats_are_rejected():
    buffer = io.BytesIO()
    np.save(buffer, np.array([dict(value=1)], dtype=object))
    with pytest.raises(ValueError):
        inspect_bytes("unsafe.npy", buffer.getvalue())
    with pytest.raises(ReleaseError, match="unrecognized"):
        inspect_bytes("program.bin", b"binary")


@pytest.mark.parametrize("compressed", [False, True])
def test_structured_unicode_arrays_cannot_bypass_inspection(compressed):
    buffer = io.BytesIO()
    value = np.array([("private_fixture_token",)], dtype=[("note", "U40")])
    if compressed:
        np.savez_compressed(buffer, values=value)
    else:
        np.save(buffer, value)
    with pytest.raises(ReleaseError, match="structured"):
        inspect_bytes("values.npz" if compressed else "values.npy", buffer.getvalue(), PATTERNS)


@pytest.mark.parametrize("entry", ["../outside.py", "/outside.py"])
def test_archive_path_traversal_is_rejected(entry):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(entry, "print('hello')")
    with pytest.raises(ReleaseError):
        inspect_bytes("source.zip", buffer.getvalue())


def test_hdf5_compressed_strings_and_external_links_are_checked():
    buffer = io.BytesIO()
    with h5py.File(buffer, "w") as handle:
        handle.create_dataset("labels", data=np.array([b"private_fixture_token"], dtype="S40"),
                              compression="gzip")
    with pytest.raises(ReleaseError, match="restricted"):
        inspect_bytes("data.h5", buffer.getvalue(), PATTERNS)
    buffer = io.BytesIO()
    with h5py.File(buffer, "w") as handle:
        handle["linked"] = h5py.ExternalLink("missing.h5", "/values")
    with pytest.raises(ReleaseError, match="links"):
        inspect_bytes("data.h5", buffer.getvalue())


def test_hdf5_numeric_fixture_with_hard_link_cycle_is_bounded():
    buffer = io.BytesIO()
    with h5py.File(buffer, "w") as handle:
        group = handle.create_group("group")
        group.create_dataset("values", data=np.arange(12).reshape(3, 4), compression="gzip")
        group["cycle"] = group
        handle.attrs["metadata"] = json.dumps(dict(units="atomic"))
    inspect_bytes("data.h5", buffer.getvalue())


def test_hdf5_long_unicode_attribute_arrays_are_not_truncated_before_inspection():
    buffer = io.BytesIO()
    with h5py.File(buffer, "w") as handle:
        labels = np.full(2000, "ordinary", dtype=object)
        labels[1000] = "ｐrivate_fixture_token"
        handle.attrs.create("labels", labels, dtype=h5py.string_dtype("utf-8"))
    with pytest.raises(ReleaseError, match="restricted"):
        inspect_bytes("data.h5", buffer.getvalue(), PATTERNS)


def test_hdf5_string_expansion_budget_includes_attributes():
    buffer = io.BytesIO()
    with h5py.File(buffer, "w") as handle:
        handle.attrs["large"] = "x"*1000
    payload = buffer.getvalue()
    with pytest.raises(ReleaseError, match="string budget"):
        inspect_bytes("data.h5", payload, budget=[len(payload)+100])


def test_expansion_budget_is_enforced(monkeypatch):
    monkeypatch.setattr("tools.release.MAX_MEMBER_BYTES", 1024)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("huge.txt", b"x"*2048)
    with pytest.raises(ReleaseError, match="oversized"):
        inspect_bytes("source.zip", buffer.getvalue())


def _git(root, *args):
    return subprocess.run(["git", "-C", str(root), *args], check=True,
                          capture_output=True, text=True).stdout


def test_git_audit_includes_prior_history_and_checks_exact_index(tmp_path):
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.name", "Fixture Author")
    _git(tmp_path, "config", "user.email", "fixture@example.invalid")
    path = tmp_path/"README.md"
    path.write_text("private_fixture_token\n")
    _git(tmp_path, "add", "README.md")
    _git(tmp_path, "commit", "-qm", "Initial fixture")
    path.write_text("Public content\n")
    _git(tmp_path, "add", "README.md")
    _git(tmp_path, "commit", "-qm", "Revise fixture")
    audit_git(tmp_path, ["README.md"], PATTERNS)
    with pytest.raises(ReleaseError, match="restricted"):
        audit_git(tmp_path, ["README.md"], PATTERNS, ref="HEAD")
    (tmp_path/"extra.txt").write_text("extra")
    _git(tmp_path, "add", "extra.txt")
    with pytest.raises(ReleaseError, match="inventory"):
        audit_git(tmp_path, ["README.md"])


def test_git_audit_rejects_unstaged_file_change(tmp_path):
    _git(tmp_path, "init", "-q")
    path = tmp_path/"README.md"
    path.write_text("First version\n")
    _git(tmp_path, "add", "README.md")
    path.write_text("Second version\n")
    with pytest.raises(ReleaseError, match="working file"):
        audit_git(tmp_path, ["README.md"])


@pytest.mark.parametrize("change", ["remove", "modify"])
def test_git_target_reference_must_match_inventory_and_bytes(tmp_path, change):
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.name", "Fixture Author")
    _git(tmp_path, "config", "user.email", "fixture@example.invalid")
    (tmp_path/"README.md").write_text("Public content\n")
    (tmp_path/"extra.py").write_text("value = 1\n")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-qm", "Initial fixture")
    if change == "remove":
        _git(tmp_path, "rm", "--cached", "extra.py")
        names = ["README.md"]
    else:
        (tmp_path/"extra.py").write_text("value = 2\n")
        _git(tmp_path, "add", "extra.py")
        names = ["README.md", "extra.py"]
    audit_git(tmp_path, names)
    with pytest.raises(ReleaseError, match="release reference differs"):
        audit_git(tmp_path, names, ref="HEAD")


def test_git_history_tracks_all_paths_of_a_renamed_identical_blob(tmp_path):
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.name", "Fixture Author")
    _git(tmp_path, "config", "user.email", "fixture@example.invalid")
    path = tmp_path/"private_fixture_token.py"
    path.write_text("value = 1\n")
    _git(tmp_path, "add", path.name)
    _git(tmp_path, "commit", "-qm", "Initial fixture")
    _git(tmp_path, "mv", path.name, "public.py")
    _git(tmp_path, "commit", "-qm", "Rename fixture")
    audit_git(tmp_path, ["public.py"], PATTERNS)
    with pytest.raises(ReleaseError, match="history path"):
        audit_git(tmp_path, ["public.py"], PATTERNS, ref="HEAD")


def test_annotated_tag_metadata_is_part_of_release_audit(tmp_path):
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.name", "Fixture Author")
    _git(tmp_path, "config", "user.email", "fixture@example.invalid")
    (tmp_path/"README.md").write_text("Public content\n")
    _git(tmp_path, "add", "README.md")
    _git(tmp_path, "commit", "-qm", "Initial fixture")
    _git(tmp_path, "tag", "-a", "v1", "-m", "private_fixture_token")
    with pytest.raises(ReleaseError, match="tag metadata"):
        audit_git(tmp_path, ["README.md"], PATTERNS, ref="v1")


def test_shallow_checkout_cannot_claim_complete_history_audit(tmp_path):
    source = tmp_path/"source"
    source.mkdir()
    _git(source, "init", "-q")
    _git(source, "config", "user.name", "Fixture Author")
    _git(source, "config", "user.email", "fixture@example.invalid")
    (source/"README.md").write_text("private_fixture_token\n")
    _git(source, "add", "README.md")
    _git(source, "commit", "-qm", "Initial fixture")
    (source/"README.md").write_text("Public content\n")
    _git(source, "commit", "-qam", "Clean current tree")
    shallow = tmp_path/"shallow"
    _git(tmp_path, "clone", "--quiet", "--depth", "1", source.as_uri(), str(shallow))
    audit_git(shallow, ["README.md"], PATTERNS)
    with pytest.raises(ReleaseError, match="shallow"):
        audit_git(shallow, ["README.md"], PATTERNS, ref="HEAD")
    _git(shallow, "fetch", "--quiet", "--unshallow")
    with pytest.raises(ReleaseError, match="restricted"):
        audit_git(shallow, ["README.md"], PATTERNS, ref="HEAD")


def test_cli_cannot_add_unlisted_report_inside_export(tmp_path):
    import tools.release

    source = tmp_path/"source"
    source.mkdir()
    (source/"README.md").write_text("Public content\n")
    manifest = source/"release-files.txt"
    manifest.write_text("README.md\nrelease-files.txt\n")
    destination = tmp_path/"release"
    result = subprocess.run([sys.executable, tools.release.__file__, "export",
                             "--root", str(source), "--inventory", str(manifest),
                             "--destination", str(destination),
                             "--report", str(destination/"extra.json")], capture_output=True)
    assert result.returncode != 0
    assert not destination.exists()
