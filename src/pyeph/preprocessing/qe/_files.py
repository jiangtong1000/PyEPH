"""Small filesystem primitives shared by the QE preparation commands."""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import os
from numbers import Integral
from pathlib import Path
import re
import shutil


def positive_integer(value: object, label: str) -> int:
    """Reject silently truncated record counts and offsets in JSON manifests."""
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{label} must be a positive integer")
    return int(value)


def sha256_file(path: Path, block_bytes: int = 8 * 1024 * 1024) -> str:
    if block_bytes < 1:
        raise ValueError("block_bytes must be positive")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_bytes):
            digest.update(block)
    return digest.hexdigest()


def path_component(value: str, label: str) -> str:
    """Validate names inserted into paths, QE inputs and shell templates."""
    if not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]*", value):
        raise ValueError(f"{label} must be a plain name using letters, digits, _, . or -")
    return value


def absolute_output(path: Path) -> Path:
    """Resolve the parent, preserving a final symlink for no-overwrite checks."""
    return path.absolute().parent.resolve() / path.name


def require_absent(*paths: Path) -> None:
    for path in paths:
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"refusing existing output: {path}")


@contextmanager
def publication_lock(path: Path):
    """Serialize cooperating publishers; never remove a pre-existing lock."""
    path.mkdir()
    try:
        yield
    finally:
        path.rmdir()


def publish_directory(source: Path, destination: Path) -> None:
    """Publish a complete directory by rename under a sibling exclusive lock.

    The lock serializes these tools. Other writers must respect it; portable
    Python has no atomic no-replace directory rename on every supported OS.
    """
    lock = destination.with_name(destination.name + ".publish-lock")
    with publication_lock(lock):
        require_absent(destination, destination.with_name(destination.name + ".partial"))
        source.rename(destination)


def publish_file(source: Path, destination: Path) -> None:
    """Atomically publish a staged regular file without replacing any target."""
    os.link(source, destination)


def copy_and_fsync(source: Path, destination: Path) -> None:
    with source.open("rb") as src, destination.open("xb") as dst:
        shutil.copyfileobj(src, dst, length=8 * 1024 * 1024)
        dst.flush()
        os.fsync(dst.fileno())


def write_json(path: Path, value: object) -> None:
    with path.open("x") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
