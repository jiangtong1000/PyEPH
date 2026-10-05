"""Durable, independently claimed work units over ``run_ensemble``.

The coordinator stores identities and status in SQLite and numerical result
shards in non-executable NPZ files. It launches no processes and never reclaims
work based on timeouts. This implementation requires local filesystem locking,
atomic rename and fsync semantics; shared network filesystems are unqualified.
"""

from contextlib import closing, contextmanager
from dataclasses import dataclass, replace
import hashlib
import json
import os
from pathlib import Path
import socket
import sqlite3
import tempfile
import time
import uuid

import numpy as np

from pyeph.execution.ensemble import (
    EnsembleResult, merge_ensembles, partition_ids, run_ensemble,
)
from pyeph.io.provenance import assert_matching_manifest, problem_manifest, validate_manifest


class ClaimLostError(RuntimeError):
    """The attempt was completed, failed or explicitly revoked by another caller."""


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sync_directory(path):
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _integer(value, name, minimum):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be nonempty")
    return value


def _encode_tree(value, arrays):
    # Store plain container structure explicitly, never Python/JAX tree pickles.
    if type(value) is dict:
        if not all(isinstance(key, str) for key in value):
            raise TypeError("campaign observable dictionaries require string keys")
        keys = sorted(value)
        return {"kind": "dict", "keys": keys,
                "items": [_encode_tree(value[key], arrays) for key in keys]}
    if type(value) in (list, tuple):
        return {"kind": type(value).__name__,
                "items": [_encode_tree(item, arrays) for item in value]}
    if value is None:
        return {"kind": "none"}
    array = np.asarray(value)
    if array.dtype.kind not in "biufc":
        raise TypeError("campaign observables require numerical leaves and plain containers")
    name = f"array_{len(arrays)}"
    arrays[name] = array
    return {"kind": "array", "name": name}


def _decode_tree(description, arrays):
    kind = description["kind"]
    if kind == "array":
        array = arrays[description["name"]]
        if array.dtype.kind not in "biufc":
            raise ValueError("invalid campaign array dtype")
        return array
    if kind == "none":
        return None
    items = [_decode_tree(item, arrays) for item in description["items"]]
    if kind == "dict":
        keys = description["keys"]
        if len(set(keys)) != len(keys) or not all(isinstance(key, str) for key in keys):
            raise ValueError("invalid campaign dictionary keys")
        return dict(zip(keys, items, strict=True))
    if kind == "list":
        return items
    if kind == "tuple":
        return tuple(items)
    raise ValueError("unsupported campaign container")


def _write_result(path, result, metadata):
    arrays = {"times": np.asarray(result.times),
              "trajectory_ids": np.asarray(result.trajectory_ids)}
    document = {**metadata, "mean": _encode_tree(result.mean, arrays),
                "m2": _encode_tree(result.m2, arrays)}
    arrays["metadata"] = np.asarray(_json(document))
    with open(path, "wb") as stream:
        np.savez(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())


@dataclass(frozen=True)
class CampaignClaim:
    """One fenced attempt; keep its token when manually recovering interruption."""

    campaign_id: str
    shard_id: int
    token: str
    trajectory_ids: tuple
    worker_id: str
    attempt: int


class Campaign:
    """A durable ensemble of fixed work units with caller-reconstructed physics.

    ``create`` freezes the IDs, their work-unit partition, preparation identity,
    problem provenance, integration length and observation interval. Reopening
    reads data only. Every ``run_next`` checks the supplied simulation again.
    """

    @classmethod
    def create(cls, path, simulation, trajectory_ids, steps, *, preparation_id,
               shard_size=32, initial_time=0., initial_step=0, time_dtype=None, artifact_ids=None,
               continuation_steps=None, provider_bundle_id=None):
        ids = partition_ids(trajectory_ids, 0, 1)
        _integer(steps, "steps", 0)
        _integer(shard_size, "shard_size", 1)
        _text(preparation_id, "preparation_id")
        _integer(initial_step, "initial_step", 0)
        continuation = None
        if continuation_steps is not None:
            from pyeph.execution import _continuation
            if shard_size != 1:
                raise ValueError("scalar continuation requires shard_size=1")
            continuation = _continuation.profile(simulation, continuation_steps, provider_bundle_id)
            continuation["dt"] = simulation.integrator.dt
        elif provider_bundle_id is not None:
            raise ValueError("provider_bundle_id requires scalar continuation")
        if (np.ndim(initial_time) != 0 or np.iscomplexobj(initial_time)
                or not np.isfinite(initial_time)):
            raise ValueError("initial_time must be finite and real")
        interval = simulation.execution.save_every
        first = interval - initial_step % interval
        offsets = np.arange(first, steps + 1, interval, dtype=np.int64)
        if steps and (not offsets.size or offsets[-1] != steps):
            offsets = np.append(offsets, steps)
        offsets = np.concatenate(([0], offsets))
        manifest = problem_manifest(simulation.problem, simulation.integrator,
                                    artifact_ids=artifact_ids)
        validate_manifest(manifest)
        use_x64 = manifest["payload"]["runtime"]["jax_enable_x64"]
        dtype = np.dtype(time_dtype if time_dtype is not None else ("float64" if use_x64 else "float32"))
        if dtype not in (np.dtype("float32"), np.dtype("float64")):
            raise ValueError("time_dtype must be float32 or float64")
        if dtype.itemsize == 8 and not use_x64:
            raise ValueError("float64 campaign times require JAX x64")
        if continuation is not None and dtype != np.dtype("float64"):
            raise ValueError("scalar continuation requires float64 time precision")
        initial_time = float(np.asarray(initial_time, dtype=dtype))
        output_times = np.asarray(initial_time + offsets * simulation.integrator.dt, dtype=dtype)
        if not np.isfinite(output_times).all() or np.any(np.diff(output_times) <= 0):
            raise ValueError("output times must be finite and distinct at the declared time precision")
        specification = {"schema": 1, "simulation_manifest": manifest,
                         "preparation_id": preparation_id, "steps": steps,
                         "save_every": interval, "initial_time": float(initial_time),
                         "initial_step": initial_step, "output_times": output_times.tolist(),
                         "time_dtype": dtype.name,
                         "trajectory_ids": ids.tolist(), "shard_size": shard_size}
        if continuation is not None:
            specification.update(schema=2, continuation=continuation)
        specification["campaign_id"] = hashlib.sha256(_json(specification).encode()).hexdigest()
        path = Path(path).resolve()
        path.mkdir(parents=True, exist_ok=False)
        (path / "results").mkdir()
        if continuation is not None:
            (path / "segments").mkdir()
            (path / "failures").mkdir()
        with closing(sqlite3.connect(path / "ledger.sqlite")) as connection, connection:
            connection.execute("PRAGMA synchronous=EXTRA")
            connection.executescript("""
                CREATE TABLE specification (document TEXT NOT NULL);
                CREATE TABLE shards (
                    id INTEGER PRIMARY KEY, status TEXT NOT NULL,
                    token TEXT, worker_id TEXT, attempt INTEGER NOT NULL DEFAULT 0,
                    result_file TEXT, result_sha256 TEXT);
                CREATE TABLE attempts (
                    token TEXT PRIMARY KEY, shard_id INTEGER NOT NULL,
                    worker_id TEXT NOT NULL, started REAL NOT NULL, ended REAL,
                    status TEXT NOT NULL, error TEXT);
            """)
            connection.execute("INSERT INTO specification VALUES (?)", (_json(specification),))
            connection.executemany("INSERT INTO shards (id,status) VALUES (?, 'pending')",
                                   [(i,) for i in range((len(ids) + shard_size - 1) // shard_size)])
            if continuation is not None:
                connection.execute("CREATE INDEX attempts_by_shard ON attempts (shard_id)")
                connection.execute("""CREATE TABLE segments (
                    shard_id INTEGER NOT NULL, generation INTEGER NOT NULL,
                    token TEXT NOT NULL, parent_sha256 TEXT,
                    payload_file TEXT NOT NULL, payload_sha256 TEXT NOT NULL,
                    start_step INTEGER NOT NULL, end_step INTEGER NOT NULL,
                    first_output_index INTEGER NOT NULL, next_output_index INTEGER NOT NULL,
                    PRIMARY KEY (shard_id, generation))""")
        _sync_directory(path)
        _sync_directory(path.parent)
        return cls(path)

    def __init__(self, path):
        self.path = Path(path).resolve()
        with self._connection() as connection:
            rows = connection.execute("SELECT document FROM specification").fetchall()
        if len(rows) != 1:
            raise ValueError("campaign requires one immutable specification")
        document = json.loads(rows[0][0])
        expected = {"schema", "simulation_manifest", "preparation_id", "steps", "save_every",
                    "trajectory_ids", "shard_size", "campaign_id", "initial_time",
                    "initial_step", "output_times", "time_dtype"}
        if document.get("schema") == 2:
            expected.add("continuation")
        if set(document) != expected or document["schema"] not in (1, 2):
            raise ValueError("unsupported campaign specification")
        digest = hashlib.sha256(_json({key: value for key, value in document.items()
                                     if key != "campaign_id"}).encode()).hexdigest()
        if digest != document["campaign_id"]:
            raise ValueError("campaign specification checksum mismatch")
        validate_manifest(document["simulation_manifest"])
        partition_ids(document["trajectory_ids"], 0, 1)
        _integer(document["steps"], "steps", 0)
        _integer(document["save_every"], "save_every", 1)
        _integer(document["shard_size"], "shard_size", 1)
        _text(document["preparation_id"], "preparation_id")
        _integer(document["initial_step"], "initial_step", 0)
        if document["time_dtype"] not in ("float32", "float64"):
            raise ValueError("invalid campaign time dtype")
        times = np.asarray(document["output_times"], dtype=document["time_dtype"])
        if (times.ndim != 1 or not times.size or not np.isfinite(times).all()
                or np.any(np.diff(times) <= 0)):
            raise ValueError("invalid campaign output time grid")
        self._document = document
        if document["schema"] == 2:
            from pyeph.execution import _continuation
            declaration = document["continuation"]
            if (set(declaration) != {"profile", "steps", "provider_bundle_id", "execution",
                                     "q_shape", "nstates", "dt"}
                    or declaration["profile"] != _continuation.PROFILE
                    or document["shard_size"] != 1 or document["time_dtype"] != "float64"):
                raise ValueError("invalid scalar continuation declaration")
            _integer(declaration["steps"], "continuation_steps", 1)
            _integer(declaration["nstates"], "nstates", 1)
            if (not isinstance(declaration["q_shape"], list) or not declaration["q_shape"]
                    or any(type(x) is not int or x < 1 for x in declaration["q_shape"])
                    or type(declaration["dt"]) not in (float, int)
                    or not np.isfinite(declaration["dt"]) or declaration["dt"] <= 0):
                raise ValueError("invalid continuation dimensions or time step")
            if declaration["provider_bundle_id"] is not None:
                _text(declaration["provider_bundle_id"], "provider_bundle_id")
            from dataclasses import asdict
            from pyeph.execution.runner import Execution
            if asdict(Execution(**declaration["execution"])) != declaration["execution"]:
                raise ValueError("invalid continuation execution policy")
            if declaration["execution"]["save_every"] != document["save_every"]:
                raise ValueError("continuation sampling policy mismatch")
            expected_times = (document["initial_time"] +
                (_continuation.grid(document)-document["initial_step"]) * declaration["dt"])
            if not np.array_equal(times, expected_times):
                raise ValueError("continuation declaration disagrees with integer output schedule")
        self.ledger()

    @property
    def specification(self):
        """A defensive JSON copy; changing this cannot mutate the campaign."""
        return json.loads(_json(self._document))

    @contextmanager
    def _connection(self, *, transaction=False):
        connection = sqlite3.connect((self.path / "ledger.sqlite").as_uri() + "?mode=rw",
                                     uri=True, timeout=30., isolation_level=None)
        connection.row_factory = sqlite3.Row
        try:
            connection.execute("PRAGMA synchronous=EXTRA")
            if connection.execute("PRAGMA journal_mode").fetchone()[0] != "delete":
                raise ValueError("campaign requires SQLite DELETE journal mode")
            if transaction:
                connection.execute("BEGIN IMMEDIATE")
            yield connection
            if transaction:
                connection.commit()
        except BaseException:
            if transaction:
                connection.rollback()
            raise
        finally:
            connection.close()

    def _ids(self, shard_id):
        size = self._document["shard_size"]
        return tuple(self._document["trajectory_ids"][shard_id * size:(shard_id + 1) * size])

    def ledger(self):
        """Read status/claim tokens and attempt diagnostics without loading results."""
        with self._connection() as connection:
            rows = [dict(row) | {"trajectory_ids": list(self._ids(row["id"]))}
                    for row in connection.execute("SELECT * FROM shards ORDER BY id")]
        size = self._document["shard_size"]
        count = (len(self._document["trajectory_ids"]) + size - 1) // size
        if ([row["id"] for row in rows] != list(range(count))
                or any(row["status"] not in {"pending", "running", "failed", "completed"}
                       for row in rows)):
            raise ValueError("campaign ledger work units disagree with its specification")
        return rows

    def attempts(self):
        """All attempt records, including failures retained after a successful retry."""
        with self._connection() as connection:
            return [dict(row) for row in connection.execute(
                "SELECT * FROM attempts ORDER BY shard_id, started, token")]

    def claim(self, *, worker_id=None):
        """Atomically claim the first pending unit; no pending work returns None.

        None does not imply completion: other units may be running or failed.
        Completed/failed units are never silently reclaimed.
        """
        worker = worker_id or f"{socket.gethostname()}:{os.getpid()}"
        _text(worker, "worker_id")
        token = uuid.uuid4().hex
        with self._connection(transaction=True) as connection:
            row = connection.execute(
                "SELECT * FROM shards WHERE status='pending' ORDER BY id LIMIT 1").fetchone()
            if row is None:
                return None
            attempt = row["attempt"] + 1
            connection.execute("UPDATE shards SET status='running',token=?,worker_id=?,attempt=? "
                               "WHERE id=?", (token, worker, attempt, row["id"]))
            connection.execute("INSERT INTO attempts VALUES (?,?,?,?,NULL,'running',NULL)",
                               (token, row["id"], worker, time.time()))
        return CampaignClaim(self._document["campaign_id"], row["id"], token,
                             self._ids(row["id"]), worker, attempt)

    def _check_claim(self, connection, claim):
        if (type(claim) is not CampaignClaim or claim.campaign_id != self._document["campaign_id"]
                or claim.trajectory_ids != self._ids(claim.shard_id)):
            raise ClaimLostError("claim does not belong to this campaign")
        row = connection.execute("SELECT * FROM shards WHERE id=?", (claim.shard_id,)).fetchone()
        if (row is None or row["status"] != "running" or row["token"] != claim.token
                or row["worker_id"] != claim.worker_id or row["attempt"] != claim.attempt):
            raise ClaimLostError("claim is no longer running with this attempt token")

    def _check_result(self, result, ids):
        result.validate()
        assert_matching_manifest(self._document["simulation_manifest"], result.simulation_manifest)
        if result.preparation_id != self._document["preparation_id"]:
            raise ValueError("campaign preparation mismatch")
        if not np.array_equal(result.trajectory_ids, ids):
            raise ValueError("campaign result must contain precisely its claimed IDs in order")
        times = np.asarray(result.times)
        dtype = np.dtype(self._document["time_dtype"])
        if times.dtype != dtype:
            raise ValueError("campaign result time dtype differs from its declared precision")
        expected = np.asarray(self._document["output_times"], dtype=dtype)
        tolerance = 32 * np.finfo(dtype).eps * max(1., float(np.max(abs(expected))))
        if times.shape != expected.shape or not np.allclose(times, expected, rtol=0, atol=tolerance):
            raise ValueError("campaign result does not match the declared output time grid")
        # Stored shards share the exact declared grid. Independently accepted
        # accumulation roundoff must not leave incompatible grids for merging.
        return replace(result, times=expected)

    def complete(self, claim, result):
        """Commit a complete work unit once, after atomically publishing its result.

        Results are fsynced before the database commit. A hard interruption
        between publication and commit leaves an unreferenced file, never a
        completed ledger entry. Recovery can rerun the unit safely.
        """
        result = self._check_result(result, claim.trajectory_ids)
        if self._document["schema"] == 2:
            self._check_continuation_result(claim.shard_id, result)
        with self._connection() as connection:
            self._check_claim(connection, claim)
        result_dir = self.path / "results"
        descriptor, temporary = tempfile.mkstemp(prefix=".partial-", suffix=".npz", dir=result_dir)
        os.close(descriptor)
        filename = f"{claim.shard_id:08d}-{claim.token}.npz"
        try:
            _write_result(temporary, result, {"schema": 1, "campaign_id": claim.campaign_id,
                                             "shard_id": claim.shard_id, "token": claim.token})
            checksum = _digest(temporary)
            with self._connection(transaction=True) as connection:
                self._check_claim(connection, claim)
                os.replace(temporary, result_dir / filename)
                _sync_directory(result_dir)
                connection.execute("UPDATE shards SET status='completed',result_file=?,"
                                   "result_sha256=? WHERE id=?",
                                   (filename, checksum, claim.shard_id))
                connection.execute("UPDATE attempts SET status='completed',ended=? WHERE token=?",
                                   (time.time(), claim.token))
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    def fail(self, claim, error):
        """Record a failed attempt without accepting a partial ensemble result."""
        message = _text(str(error), "error")
        with self._connection(transaction=True) as connection:
            self._check_claim(connection, claim)
            connection.execute("UPDATE shards SET status='failed' WHERE id=?", (claim.shard_id,))
            connection.execute("UPDATE attempts SET status='failed',ended=?,error=? WHERE token=?",
                               (time.time(), message, claim.token))

    def retry_failed(self, shard_id):
        """Make a failed unit pending again, preserving the previous attempt record."""
        _integer(shard_id, "shard_id", 0)
        with self._connection(transaction=True) as connection:
            cursor = connection.execute("UPDATE shards SET status='pending',token=NULL,worker_id=NULL "
                                        "WHERE id=? AND status='failed'", (shard_id,))
            if cursor.rowcount != 1:
                raise ValueError("retry requires a failed work unit")

    def recover(self, shard_id, *, expected_token, reason):
        """Explicitly revoke an interrupted worker and make its unit pending.

        Call only after independently confirming the worker stopped. Elapsed
        time or a missing heartbeat is not such evidence. The old token is
        fenced even if that worker unexpectedly returns and tries to publish.
        """
        _integer(shard_id, "shard_id", 0)
        _text(expected_token, "expected_token")
        _text(reason, "reason")
        with self._connection(transaction=True) as connection:
            cursor = connection.execute("UPDATE shards SET status='pending',token=NULL,worker_id=NULL "
                                        "WHERE id=? AND status='running' AND token=?",
                                        (shard_id, expected_token))
            if cursor.rowcount != 1:
                raise ClaimLostError("recovery token does not name the current running attempt")
            connection.execute("UPDATE attempts SET status='interrupted',ended=?,error=? WHERE token=?",
                               (time.time(), reason, expected_token))

    def run_next(self, simulation, initialize, *, preparation_id, batch_size=32,
                 worker_id=None, artifact_ids=None):
        """Validate the reconstructed calculation, claim, compute and durably commit.

        Returns the completed claim or None when no pending unit exists. A
        caught calculation error marks the unit failed and re-raises the original
        exception (including its retained state); a hard crash stays running.
        """
        if self._document["schema"] == 2:
            raise ValueError("scalar continuation campaigns require run_next_scalar")
        manifest = problem_manifest(simulation.problem, simulation.integrator,
                                    artifact_ids=artifact_ids)
        assert_matching_manifest(self._document["simulation_manifest"], manifest)
        if preparation_id != self._document["preparation_id"]:
            raise ValueError("campaign preparation mismatch")
        if simulation.execution.save_every != self._document["save_every"]:
            raise ValueError("campaign observation interval mismatch")
        _integer(batch_size, "batch_size", 1)
        claim = self.claim(worker_id=worker_id)
        if claim is None:
            return None

        def prepare(ids):
            state = initialize(ids)
            if np.asarray(state.time).dtype != np.dtype(self._document["time_dtype"]):
                raise ValueError("initializer time dtype differs from the campaign declaration")
            if (not np.all(np.asarray(state.time) == self._document["initial_time"])
                    or not np.all(np.asarray(state.step) == self._document["initial_step"])):
                raise ValueError("initializer time/step differs from the campaign declaration")
            return state

        try:
            result = run_ensemble(simulation, prepare, np.asarray(claim.trajectory_ids),
                                  self._document["steps"], batch_size=batch_size,
                                  preparation_id=preparation_id, artifact_ids=artifact_ids)
            self.complete(claim, result)
        except BaseException as error:
            try:
                self.fail(claim, f"{type(error).__name__}: {error}")
            except ClaimLostError:
                pass  # Preserve the original error if explicit recovery fenced us.
            except Exception as persistence_error:
                error.add_note(f"Campaign failure record could not be saved: {persistence_error!r}")
            raise
        return claim

    def segments(self, shard_id):
        """Read committed scalar progress; unreferenced files are not progress."""
        if self._document["schema"] != 2:
            raise ValueError("campaign has no scalar continuation profile")
        _integer(shard_id, "shard_id", 0)
        if shard_id >= len(self._document["trajectory_ids"]):
            raise ValueError("unknown continuation work unit")
        with self._connection() as connection:
            return [dict(row) for row in connection.execute(
                "SELECT * FROM segments WHERE shard_id=? ORDER BY generation", (shard_id,))]

    def _check_scalar_simulation(self, simulation, preparation_id, artifact_ids, bundle_id):
        from pyeph.execution import _continuation
        if self._document["schema"] != 2:
            raise ValueError("run_next_scalar requires a continuation campaign")
        expected = self._document["continuation"]
        actual = _continuation.profile(simulation, expected["steps"], bundle_id)
        actual["dt"] = simulation.integrator.dt
        if actual != expected or preparation_id != self._document["preparation_id"]:
            raise ValueError("continuation execution, provider bundle or preparation mismatch")
        manifest = problem_manifest(simulation.problem, simulation.integrator, artifact_ids=artifact_ids)
        assert_matching_manifest(self._document["simulation_manifest"], manifest)

    def _read_segments(self, shard_id, *, collect=False):
        """Verify the entire committed chain before exposing any restart state."""
        from pyeph.execution import _continuation
        from pyeph.io.checkpoint import array_fingerprint
        rows = self.segments(shard_id)
        with self._connection() as connection:
            attempts = {row["token"]: dict(row) for row in connection.execute(
                "SELECT token,shard_id FROM attempts WHERE shard_id=?", (shard_id,))}
        previous, state, first_state, pieces = None, None, None, []
        identity = self._ids(shard_id)[0]
        for row in rows:
            for name, size in (("token", 32), ("payload_sha256", 64)):
                value = row[name]
                if (not isinstance(value, str) or len(value) != size
                        or any(character not in "0123456789abcdef" for character in value)):
                    raise ValueError("invalid continuation token or checksum")
            generation, start, end, first, stop = _continuation.bounds(self._document, previous)
            filename = f"{shard_id:08d}-{generation:08d}-{row['token']}.h5"
            if (row["generation"] != generation or row["start_step"] != start
                    or row["end_step"] != end or row["first_output_index"] != first
                    or row["next_output_index"] != stop or row["payload_file"] != filename
                    or row["parent_sha256"] != (None if previous is None else previous["payload_sha256"])
                    or row["token"] not in attempts or attempts[row["token"]]["shard_id"] != shard_id):
                raise ValueError("continuation ledger chain mismatch")
            path = self.path / "segments" / filename
            if path.is_symlink() or _digest(path) != row["payload_sha256"]:
                raise ValueError("continuation payload checksum mismatch")
            state, _, auxiliary = _continuation.read_payload(
                path, _continuation.metadata(self._document, row, identity), row["payload_sha256"])
            _continuation.check_state(state, self._document, identity, end)
            _continuation.check_auxiliary(auxiliary, self._document, row)
            if first_state is None:
                if float(state.time) != self._document["initial_time"]:
                    raise ValueError("continuation initial time differs from declaration")
                first_state = state
            elif (np.asarray(state.step).dtype != np.asarray(first_state.step).dtype
                    or not np.array_equal(state.key, first_state.key)
                    or array_fingerprint(state.method_state) != array_fingerprint(first_state.method_state)):
                raise ValueError("continuation state layout or random key changed")
            if collect:
                pieces.append(auxiliary)
            previous = row
        return state, previous, first_state, pieces

    def _publish_segment(self, claim, simulation, state, auxiliary, predecessor, *, artifact_ids=None):
        from pyeph.execution import _continuation
        from pyeph.io.checkpoint import save_checkpoint
        self._check_scalar_simulation(simulation, self._document["preparation_id"], artifact_ids,
                                      self._document["continuation"]["provider_bundle_id"])
        generation, start, end, first, stop = _continuation.bounds(self._document, predecessor)
        row = {"shard_id": claim.shard_id, "generation": generation, "token": claim.token,
               "parent_sha256": None if predecessor is None else predecessor["payload_sha256"],
               "start_step": start, "end_step": end, "first_output_index": first,
               "next_output_index": stop}
        _continuation.check_state(state, self._document, claim.trajectory_ids[0], end)
        _continuation.check_auxiliary(auxiliary, self._document, row)
        directory = self.path / "segments"
        descriptor, temporary = tempfile.mkstemp(prefix=".partial-", suffix=".h5", dir=directory)
        os.close(descriptor)
        filename = f"{claim.shard_id:08d}-{generation:08d}-{claim.token}.h5"
        try:
            save_checkpoint(temporary, state, metadata=_continuation.metadata(
                self._document, row, claim.trajectory_ids[0]), auxiliary=auxiliary)
            row.update(payload_file=filename, payload_sha256=_digest(temporary))
            with self._connection(transaction=True) as connection:
                self._check_claim(connection, claim)
                latest = connection.execute("SELECT * FROM segments WHERE shard_id=? "
                    "ORDER BY generation DESC LIMIT 1", (claim.shard_id,)).fetchone()
                if (None if latest is None else dict(latest)) != predecessor:
                    raise ClaimLostError("continuation predecessor changed before publication")
                os.replace(temporary, directory / filename)
                _sync_directory(directory)
                columns = tuple(row)
                connection.execute(f"INSERT INTO segments ({','.join(columns)}) "
                    f"VALUES ({','.join('?' for _ in columns)})", tuple(row.values()))
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        return row

    def _check_continuation_result(self, shard_id, result):
        from pyeph.execution import _continuation
        _, previous, _, pieces = self._read_segments(shard_id, collect=True)
        if (previous is None or previous["end_step"] != self._document["initial_step"] + self._document["steps"]
                or previous["next_output_index"] != len(self._document["output_times"])):
            raise ValueError("continuation has not committed its full trajectory")
        expected = _continuation.assemble(self._document, self._ids(shard_id)[0], pieces)
        for name in ("mean", "m2"):
            actual, wanted = getattr(result, name), getattr(expected, name)
            if (type(actual) is not dict or set(actual) != set(wanted)
                    or any(np.asarray(actual[key]).dtype != np.dtype("float64")
                           or np.shape(actual[key]) != np.shape(wanted[key])
                           or not np.array_equal(actual[key], wanted[key]) for key in wanted)):
                raise ValueError("final result disagrees with committed continuation output")

    def run_next_scalar(self, simulation, initialize_one, *, preparation_id,
                        worker_id=None, artifact_ids=None, provider_bundle_id=None):
        """Resume one fixed scalar trajectory, committing state and output together.

        Initializers receive one integer ID. Only a complete successful segment
        advances progress; scientific failures remain failed until explicit retry.
        Interrupted claims require explicit recovery after worker termination.
        """
        from pyeph.execution import _continuation, _validation
        from pyeph.io.checkpoint import save_checkpoint
        self._check_scalar_simulation(simulation, preparation_id, artifact_ids, provider_bundle_id)
        claim = self.claim(worker_id=worker_id)
        if claim is None:
            return None
        committed = None
        try:
            state, previous, original, _ = self._read_segments(claim.shard_id)
            if state is None:
                state = initialize_one(claim.trajectory_ids[0])
                _continuation.check_state(state, self._document, claim.trajectory_ids[0],
                                          self._document["initial_step"])
                if float(state.time) != self._document["initial_time"]:
                    raise ValueError("initializer time differs from the continuation declaration")
                original = state
            self._check_scalar_simulation(simulation, preparation_id, artifact_ids, provider_bundle_id)
            _validation.validate_run_span(simulation.problem.nuclear_treatment,
                                          simulation.integrator, original, self._document["steps"])
            simulation._validate_state(state)
            committed = state if previous is not None else None
            terminal = self._document["initial_step"] + self._document["steps"]
            while previous is None or int(state.step) < terminal:
                count = (0 if previous is None else min(
                    self._document["continuation"]["steps"], terminal-int(state.step)))
                result = simulation.run(state, count)
                auxiliary = _continuation.select_output(result, self._document, previous)
                previous = self._publish_segment(claim, simulation, result.final_state,
                                                 auxiliary, previous, artifact_ids=artifact_ids)
                state = committed = result.final_state
            self._check_scalar_simulation(simulation, preparation_id, artifact_ids, provider_bundle_id)
            _, _, _, pieces = self._read_segments(claim.shard_id, collect=True)
            self.complete(claim, _continuation.assemble(self._document, claim.trajectory_ids[0], pieces))
        except BaseException as error:
            if committed is not None and hasattr(error, "last_valid_state"):
                try:
                    path = self.path / "failures" / f"{claim.shard_id:08d}-{claim.token}.h5"
                    save_checkpoint(path, committed, metadata={"campaign_id": claim.campaign_id,
                        "shard_id": claim.shard_id, "token": claim.token,
                        "error_type": type(error).__name__, "error": str(error)}, auxiliary={
                        "last_valid_state": _continuation.diagnostic_tree(error.last_valid_state),
                        "failed_state": _continuation.diagnostic_tree(error.failed_state),
                        "diagnostics": _continuation.diagnostic_tree(error.diagnostics)})
                    _sync_directory(path.parent)
                except Exception as persistence_error:
                    error.add_note(f"Continuation diagnostics could not be saved: {persistence_error!r}")
            try:
                self.fail(claim, f"{type(error).__name__}: {error}")
            except ClaimLostError:
                pass
            except Exception as persistence_error:
                error.add_note(f"Campaign failure record could not be saved: {persistence_error!r}")
            raise
        return claim

    def merge(self, *, require_complete=True):
        """Merge committed shards in fixed order, retaining one shard plus moments.

        Partial results require explicit opt-in and contain only completed IDs.
        Standard errors describe independent trajectories, not time origins.
        """
        if type(require_complete) is not bool:
            raise TypeError("require_complete must be a boolean")
        rows = self.ledger()
        if require_complete and any(row["status"] != "completed" for row in rows):
            raise ValueError("campaign is incomplete; inspect running, pending and failed units")
        merged = None
        for row in rows:
            if row["status"] != "completed":
                continue
            filename = f"{row['id']:08d}-{row['token']}.npz"
            if row["result_file"] != filename:
                raise ValueError("campaign result filename mismatch")
            path = self.path / "results" / filename
            if _digest(path) != row["result_sha256"]:
                raise ValueError("campaign result checksum mismatch")
            with np.load(path, allow_pickle=False) as arrays:
                metadata = json.loads(str(arrays["metadata"]))
                if (metadata.get("schema") != 1
                        or metadata.get("campaign_id") != self._document["campaign_id"]
                        or metadata.get("shard_id") != row["id"]
                        or metadata.get("token") != row["token"]):
                    raise ValueError("campaign result identity mismatch")
                result = EnsembleResult(arrays["times"], arrays["trajectory_ids"],
                                        _decode_tree(metadata["mean"], arrays),
                                        _decode_tree(metadata["m2"], arrays),
                                        self._document["simulation_manifest"],
                                        self._document["preparation_id"])
            result = self._check_result(result, self._ids(row["id"]))
            if self._document["schema"] == 2:
                self._check_continuation_result(row["id"], result)
            merged = result if merged is None else merge_ensembles(merged, result)
        if merged is None:
            raise ValueError("campaign has no completed results")
        return merged
