"""Build and inspect an explicit, non-executable source release inventory.

The inventory is the publication boundary. Unlisted workspace files are never
copied. An optional local JSON policy supplies additional restricted patterns;
that policy and its matches are not written into the release directory.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import html
import io
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import tarfile
import unicodedata
import zipfile


MAX_MEMBER_BYTES = 64 * 1024 * 1024
MAX_EXPANDED_BYTES = 256 * 1024 * 1024
MAX_DEPTH = 4
TEXT_SUFFIXES = {".py", ".md", ".txt", ".toml", ".json", ".yml", ".yaml",
                 ".in", ".sh", ".csv", ".svg", ".xml", ".f90", ".f", ".jl", ".cfg"}
SPECIAL_TEXT = {"LICENSE", "NOTICE", "MANIFEST.in", ".gitignore", "Makefile",
                "METADATA", "WHEEL", "RECORD", "PKG-INFO"}


class ReleaseError(ValueError):
    """The proposed publication does not satisfy its inventory or policy."""


def relative_path(value):
    if not isinstance(value, str):
        raise ReleaseError("inventory paths must be strings")
    path = PurePosixPath(value)
    if (not value or "\\" in value or any(c in value for c in "\x00\r\n")
            or path.is_absolute() or any(p in ("", ".", "..") for p in value.split("/"))
            or path.as_posix() != value or ".git" in path.parts):
        raise ReleaseError("inventory paths must be normalized relative POSIX file paths")
    return path


def inventory(path):
    return parse_inventory(Path(path).read_text())


def parse_inventory(text):
    rows = [line.strip() for line in text.splitlines()
            if line.strip() and not line.lstrip().startswith("#")]
    for row in rows:
        relative_path(row)
    if not rows or len(set(rows)) != len(rows):
        raise ReleaseError("inventory must be nonempty and contain no duplicates")
    return sorted(rows)


def source_file(root, name):
    relative_path(name)
    root = Path(root).resolve()
    path = root
    for part in PurePosixPath(name).parts:
        path = path / part
        if path.is_symlink():
            raise ReleaseError(f"symlinks are not release files: {name}")
    if not path.is_file() or not path.resolve().is_relative_to(root):
        raise ReleaseError(f"missing regular release file: {name}")
    return path


def policy_patterns(path=None):
    if path is None:
        return []
    data = json.loads(Path(path).read_text())
    if set(data) != {"restricted_patterns"} or not isinstance(data["restricted_patterns"], list):
        raise ReleaseError("policy requires a restricted_patterns list")
    if any(not isinstance(p, str) or not p for p in data["restricted_patterns"]):
        raise ReleaseError("restricted patterns must be nonempty strings")
    return [re.compile(p, re.IGNORECASE) for p in data["restricted_patterns"]]


def _check_text(text, patterns, location):
    def unescape(match):
        code = int(next(group for group in match.groups() if group is not None), 16)
        return chr(code) if code <= 0x10ffff else match.group()

    for _ in range(MAX_DEPTH + 1):
        if any(pattern.search(text) for pattern in patterns):
            # Locate the file without leaking matched private text into reports.
            raise ReleaseError(f"restricted content in {location}")
        decoded = unicodedata.normalize("NFKC", html.unescape(text))
        decoded = re.sub(r"\\u([0-9a-fA-F]{4})|\\U([0-9a-fA-F]{8})|\\x([0-9a-fA-F]{2})",
                         unescape, decoded)
        if decoded == text:
            return
        text = decoded
    raise ReleaseError(f"nested text encoding exceeds inspection limit: {location}")


def inspect_bytes(name, payload, patterns=(), *, depth=0, budget=None):
    """Inspect nested archives and text; reject unknown binary formats.

    Limits apply to actual expanded bytes, not just declared archive lengths.
    Archive names, link entries and HDF5 string metadata are inspected too.
    """
    if budget is None:
        budget = [MAX_EXPANDED_BYTES]
    if depth > MAX_DEPTH or len(payload) > MAX_MEMBER_BYTES:
        raise ReleaseError(f"release inspection limit exceeded: {name}")
    budget[0] -= len(payload)
    if budget[0] < 0:
        raise ReleaseError("expanded release inspection budget exceeded")
    _check_text(name, patterns, name)
    _check_text(payload.decode("latin1"), patterns, name)
    leaf = PurePosixPath(name.rsplit("!", 1)[-1])
    suffix = leaf.suffix.lower()
    if payload.startswith(b"PK\x03\x04") or suffix in (".zip", ".npz", ".whl"):
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            for member in archive.infolist():
                relative_path(member.filename.rstrip("/"))
                if member.is_dir():
                    continue
                if (member.external_attr >> 16) & 0o170000 == 0o120000:
                    raise ReleaseError(f"archive symlink: {name}")
                if member.file_size > MAX_MEMBER_BYTES or member.flag_bits & 1:
                    raise ReleaseError(f"oversized or encrypted archive member: {name}")
                with archive.open(member) as stream:
                    child = stream.read(MAX_MEMBER_BYTES + 1)
                inspect_bytes(f"{name}!{member.filename}", child, patterns,
                              depth=depth+1, budget=budget)
        return
    if payload.startswith(b"\x1f\x8b"):
        with gzip.GzipFile(fileobj=io.BytesIO(payload)) as stream:
            child = stream.read(MAX_MEMBER_BYTES + 1)
        inspect_bytes(name.removesuffix(".gz"), child, patterns, depth=depth+1, budget=budget)
        return
    if suffix == ".tar":
        with tarfile.open(fileobj=io.BytesIO(payload), mode="r:") as archive:
            for member in archive:
                relative_path(member.name.rstrip("/"))
                if member.isdir():
                    continue
                if not member.isfile() or member.size > MAX_MEMBER_BYTES:
                    raise ReleaseError(f"unsupported archive entry: {name}")
                stream = archive.extractfile(member)
                inspect_bytes(f"{name}!{member.name}", stream.read(MAX_MEMBER_BYTES + 1),
                              patterns, depth=depth+1, budget=budget)
        return
    if suffix == ".npy":
        import numpy as np

        array = np.load(io.BytesIO(payload), allow_pickle=False)
        if array.dtype.hasobject:
            raise ReleaseError(f"object array is not releasable: {name}")
        if array.dtype.fields is not None or array.dtype.kind == "V":
            raise ReleaseError(f"structured or opaque arrays are not releasable: {name}")
        if array.dtype.kind in "US":
            for value in array.reshape(-1):
                _check_text(str(value), patterns, name)
        return
    if suffix in (".h5", ".hdf5"):
        import h5py
        import numpy as np

        with h5py.File(io.BytesIO(payload), "r") as handle:
            def inspect_values(value):
                values = np.asarray(value)
                if values.dtype.fields is not None or values.dtype.kind == "V":
                    raise ReleaseError(f"unsupported HDF5 metadata: {name}")
                if values.dtype.kind not in "OSU":
                    return
                for scalar in values.reshape(-1):
                    if isinstance(scalar, bytes):
                        scalar = scalar.decode("utf-8")
                    if not isinstance(scalar, str):
                        raise ReleaseError(f"unsupported HDF5 string value: {name}")
                    expanded = len(scalar.encode("utf-8"))
                    budget[0] -= expanded
                    if expanded > MAX_MEMBER_BYTES or budget[0] < 0:
                        raise ReleaseError(f"expanded HDF5 string budget exceeded: {name}")
                    _check_text(scalar, patterns, name)

            def inspect_item(item):
                _check_text(item.name, patterns, name)
                for key, value in item.attrs.items():
                    _check_text(str(key), patterns, name)
                    inspect_values(value)
                if isinstance(item, h5py.Dataset) and item.dtype.kind in "OSU":
                    if item.size * max(item.dtype.itemsize, 1) > MAX_MEMBER_BYTES:
                        raise ReleaseError(f"oversized HDF5 string array: {name}")
                    inspect_values(item[()])
                if isinstance(item, h5py.Dataset):
                    if item.external or item.is_virtual or item.dtype.kind == "V":
                        raise ReleaseError(f"unsupported HDF5 storage: {name}")

            def walk(group):
                inspect_item(group)
                for key in group:
                    _check_text(group.name + "/" + key, patterns, name)
                    link = group.get(key, getlink=True)
                    if not isinstance(link, h5py.HardLink):
                        raise ReleaseError(f"HDF5 links are not releasable: {name}")
                    item = group[key]
                    if item.id in visited:
                        continue
                    visited.add(item.id)
                    if isinstance(item, h5py.Group):
                        walk(item)
                    else:
                        inspect_item(item)

            visited = {handle.id}
            walk(handle)
        return
    if suffix in TEXT_SUFFIXES or leaf.name in SPECIAL_TEXT:
        _check_text(payload.decode("utf-8"), patterns, name)
        return
    raise ReleaseError(f"unrecognized release file format: {name}")


def audit(root, names, patterns=()):
    report = {}
    budget = [MAX_EXPANDED_BYTES]
    for name in names:
        path = source_file(root, name)
        payload = path.read_bytes()
        inspect_bytes(name, payload, patterns, budget=budget)
        report[name] = {"sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload)}
    return report


def audit_git(root, names, patterns=(), ref=None):
    """Require an exact index inventory; optionally scan all reachable history."""
    def git(*args):
        return subprocess.check_output(["git", "-C", str(root), *args])

    tracked = git("ls-files", "-z").decode().split("\0")[:-1]
    if set(tracked) != set(names):
        raise ReleaseError("Git index differs from the explicit release inventory")
    for name in names:
        index = git("show", f":{name}")
        if index != source_file(root, name).read_bytes():
            raise ReleaseError(f"Git index differs from working file: {name}")
    if ref is None:
        return
    if git("rev-parse", "--is-shallow-repository").strip() == b"true":
        raise ReleaseError("complete Git history is required; shallow repositories cannot be audited")
    oid = git("rev-parse", "--verify", "--end-of-options", ref).decode().strip()
    while git("cat-file", "-t", oid).strip() == b"tag":
        tag = git("cat-file", "-p", oid).decode()
        _check_text(tag, patterns, "Git tag metadata")
        oid = tag.splitlines()[0].removeprefix("object ")
    if git("cat-file", "-t", oid).strip() != b"commit":
        raise ReleaseError("release reference must resolve to a commit")
    target_tree = git("ls-tree", "-r", "-z", "--full-tree", oid).split(b"\0")[:-1]
    target_files = {}
    for entry in target_tree:
        header, raw_name = entry.split(b"\t", 1)
        mode, kind, blob = header.split()
        if kind != b"blob" or mode not in (b"100644", b"100755"):
            raise ReleaseError("release reference contains a non-regular file")
        target_files[raw_name.decode()] = blob.decode()
    if set(target_files) != set(names):
        raise ReleaseError("release reference differs from the explicit inventory")
    for name, blob in target_files.items():
        if git("cat-file", "blob", blob) != source_file(root, name).read_bytes():
            raise ReleaseError(f"release reference differs from working file: {name}")
    _check_text(git("log", "--format=fuller", oid).decode(), patterns, "Git commit metadata")
    # A blob may have had several names. rev-list --objects supplies only one,
    # so inspecting its printed path misses renames of identical file contents.
    # Walk every reachable commit tree and inspect each historical path.
    seen = set()
    for tree in set(git("log", "--format=%T", oid).decode().splitlines()):
        entries = git("ls-tree", "-r", "-z", "--full-tree", tree).split(b"\0")[:-1]
        for entry in entries:
            header, raw_name = entry.split(b"\t", 1)
            mode, kind, raw_oid = header.split()
            name, oid = raw_name.decode(), raw_oid.decode()
            relative_path(name)
            _check_text(name, patterns, "Git history path")
            if kind != b"blob" or mode not in (b"100644", b"100755"):
                raise ReleaseError("Git history contains a non-regular file")
            if (oid, name) not in seen:
                inspect_bytes(name, git("cat-file", "blob", oid), patterns)
                seen.add((oid, name))


def export(root, destination, names, patterns=()):
    before = audit(root, names, patterns)
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    try:
        for name in names:
            target = destination / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_file(root, name), target)
        after = audit(destination, names, patterns)
        if before != after or audit(root, names, patterns) != before:
            raise ReleaseError("source changed while release was being exported")
        actual = {p.relative_to(destination).as_posix() for p in destination.rglob("*")
                  if p.is_file() or p.is_symlink()}
        if actual != set(names):
            raise ReleaseError("export contains files outside the release inventory")
    except BaseException:
        # Keep an incomplete export for diagnosis; never silently reuse it.
        raise
    return after


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("audit", "export"))
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--inventory", type=Path, default=Path("release-files.txt"))
    parser.add_argument("--policy", type=Path)
    parser.add_argument("--destination", type=Path)
    parser.add_argument("--git", action="store_true")
    parser.add_argument("--ref")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    names = inventory(args.inventory)
    patterns = policy_patterns(args.policy)
    report = audit(args.root, names, patterns)
    if args.git or args.ref:
        audit_git(args.root, names, patterns, args.ref)
    if args.command == "export":
        if args.destination is None:
            parser.error("export requires --destination")
        if args.report and args.report.resolve().is_relative_to(args.destination.resolve()):
            parser.error("audit report must remain outside the exported release")
        report = export(args.root, args.destination, names, patterns)
    if args.report:
        with args.report.open("x") as stream:
            json.dump({"schema": "pyeph.release-audit.v1", "files": report}, stream, indent=2)
            stream.write("\n")
    print(f"Audited {len(report)} explicitly listed release files.")


if __name__ == "__main__":
    main()
