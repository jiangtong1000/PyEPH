"""Data validation for the campaign's deliberately narrow scalar profile.

Transactions remain in ``campaign.py``. Physical propagation and numerical
serialization use the existing Runner and checkpoint implementation.
"""

from dataclasses import asdict
import hashlib
from io import BytesIO
import json

import h5py
import numpy as np

from pyeph.core.state import TrajectoryState
from pyeph.dynamics.cpa import CPA
from pyeph.dynamics.ehrenfest import Ehrenfest
from pyeph.execution.ensemble import EnsembleResult
from pyeph.execution.runner import Execution
from pyeph.integrators.krylov import LanczosOptions
from pyeph.io.checkpoint import load_checkpoint
from pyeph.observables.population import ElectronicPopulation


PROFILE = "scalar_checked_population_v1"


def profile(simulation, steps, bundle_id):
    if type(steps) is not int or steps < 1:
        raise ValueError("continuation_steps must be a positive integer")
    if bundle_id is not None and (not isinstance(bundle_id, str) or not bundle_id.strip()):
        raise ValueError("provider_bundle_id must be a nonempty string or None")
    if (type(simulation.problem.method) not in (CPA, Ehrenfest)
            or not simulation.problem.model.spec.native_jax
            or not isinstance(simulation.integrator.electronic, LanczosOptions)
            or type(simulation.measurement) is not ElectronicPopulation
            or type(simulation.execution) is not Execution):
        raise ValueError("scalar continuation requires native checked CPA/Ehrenfest, "
                         "ElectronicPopulation and Execution")
    return {"profile": PROFILE, "steps": steps, "provider_bundle_id": bundle_id,
            "execution": asdict(simulation.execution),
            "q_shape": list(simulation.problem.model.spec.system.q_shape),
            "nstates": simulation.problem.model.spec.system.nstates}


def schedule(start, count, interval):
    """Runner's integer output steps, including its initial and forced final row."""
    end = start + count
    first = start + interval - start % interval
    values = np.arange(first, end + 1, interval, dtype=np.int64)
    if count and (not values.size or values[-1] != end):
        values = np.append(values, np.int64(end))
    return np.concatenate((np.asarray([start], dtype=np.int64), values))


def grid(document):
    return schedule(document["initial_step"], document["steps"], document["save_every"])


def output_count(document, end):
    """Number of global output rows through an accepted integer step, in O(1)."""
    start = document["initial_step"]
    terminal = start + document["steps"]
    interval = document["save_every"]
    if not start <= end <= terminal:
        raise ValueError("continuation output step outside declared span")
    count = 1 + end // interval - start // interval
    if end == terminal and terminal > start and terminal % interval:
        count += 1
    return count


def output_slice(document, first, stop):
    """Construct only the requested rows, without allocating the full grid."""
    start = document["initial_step"]
    terminal = start + document["steps"]
    interval = document["save_every"]
    if not 0 <= first <= stop <= output_count(document, terminal):
        raise ValueError("continuation output slice outside declared grid")
    indices = np.arange(first, stop, dtype=np.int64)
    result = np.empty(indices.shape, dtype=np.int64)
    regular_count = terminal // interval - start // interval
    regular = (indices > 0) & (indices <= regular_count)
    if np.any(regular):
        result[regular] = (start // interval + indices[regular]) * interval
    result[indices == 0] = start
    # Compute only genuine regular multiples above: the next multiple after a
    # forced final row could overflow int64 near the supported counter limit.
    result[indices > regular_count] = terminal
    return result


def check_times(actual, expected):
    actual, expected = np.asarray(actual), np.asarray(expected, dtype=np.float64)
    tolerance = 32 * np.finfo(np.float64).eps * max(1., float(np.max(abs(expected), initial=0.)))
    if (actual.dtype != np.dtype("float64") or actual.shape != expected.shape
            or not np.isfinite(actual).all()
            or not np.allclose(actual, expected, rtol=0, atol=tolerance)):
        raise ValueError("continuation time grid or precision mismatch")


def check_state(state, document, identity, step):
    """Structural checks only: never call a provider while verifying an archive."""
    declaration = document["continuation"]
    if type(state) is not TrajectoryState:
        raise ValueError("continuation requires a scalar TrajectoryState")
    for name, shape, dtype in (
            ("q", tuple(declaration["q_shape"]), "float64"),
            ("p", tuple(declaration["q_shape"]), "float64"),
            ("electronic", (declaration["nstates"],), "complex128"),
            ("time", (), "float64"), ("trajectory_id", (), "uint32"),
            ("key", (2,), "uint32")):
        value = np.asarray(getattr(state, name))
        if value.shape != shape or value.dtype != np.dtype(dtype) or not np.isfinite(value).all():
            raise ValueError(f"invalid scalar continuation state {name}")
    counter = np.asarray(state.step)
    if (counter.shape != () or counter.dtype not in (np.dtype("int32"), np.dtype("int64"))
            or int(counter) != step or int(state.trajectory_id) != identity):
        raise ValueError("continuation state counter, identity or method data mismatch")
    _plain_state_tree(state.method_state)
    dt = document["continuation"]["dt"]
    check_times(state.time, document["initial_time"] + (step - document["initial_step"]) * dt)


def _plain_state_tree(value):
    if type(value) is dict:
        if not all(isinstance(key, str) for key in value):
            raise ValueError("continuation method state requires string dictionary keys")
        for item in value.values():
            _plain_state_tree(item)
    elif type(value) in (tuple, list):
        for item in value:
            _plain_state_tree(item)
    elif value is not None:
        array = np.asarray(value)
        if array.dtype.kind not in "biufc" or not np.isfinite(array).all():
            raise ValueError("continuation method state requires finite numerical leaves")


def bounds(document, previous):
    if previous is None:
        return 0, document["initial_step"], document["initial_step"], 0, 1
    start = previous["end_step"]
    terminal = document["initial_step"] + document["steps"]
    if start >= terminal:
        raise ValueError("continuation is already at its terminal step")
    end = min(start + document["continuation"]["steps"], terminal)
    return (previous["generation"] + 1, start, end,
            previous["next_output_index"], output_count(document, end))


def metadata(document, row, identity):
    return {"schema": 1, "profile": PROFILE, "campaign_id": document["campaign_id"],
            "trajectory_id": identity, "simulation_manifest": document["simulation_manifest"],
            "preparation_id": document["preparation_id"],
            "continuation": document["continuation"],
            **{key: row[key] for key in ("shard_id", "generation", "token", "parent_sha256",
                "start_step", "end_step", "first_output_index", "next_output_index")}}


def check_auxiliary(auxiliary, document, row):
    if type(auxiliary) is not dict or set(auxiliary) != {"output_steps", "actual_times", "population", "norm"}:
        raise ValueError("unexpected continuation auxiliary structure")
    first, stop = row["first_output_index"], row["next_output_index"]
    expected_steps = output_slice(document, first, stop)
    steps = np.asarray(auxiliary["output_steps"])
    if steps.dtype != np.dtype("int64") or not np.array_equal(steps, expected_steps):
        raise ValueError("continuation integer output cursor mismatch")
    check_times(auxiliary["actual_times"], np.asarray(document["output_times"][first:stop]))
    for name, shape in (("population", (stop-first, document["continuation"]["nstates"])),
                        ("norm", (stop-first,))):
        value = np.asarray(auxiliary[name])
        if value.dtype != np.dtype("float64") or value.shape != shape or not np.isfinite(value).all():
            raise ValueError(f"invalid continuation {name} output")


def select_output(result, document, previous):
    generation, start, end, first, stop = bounds(document, previous)
    emitted = schedule(start, end-start, document["save_every"])
    times = np.asarray(result.times)
    if type(result.observables) is not dict or set(result.observables) != {"population", "norm"}:
        raise ValueError("continuation requires exact population observations")
    if times.shape != emitted.shape:
        raise ValueError("Runner output does not match its integer schedule")
    expected = document["initial_time"] + (emitted-document["initial_step"]) * document["continuation"]["dt"]
    check_times(times, expected)
    nstates = document["continuation"]["nstates"]
    for name, shape in (("population", (len(emitted), nstates)), ("norm", (len(emitted),))):
        value = np.asarray(result.observables[name])
        if value.shape != shape or value.dtype != np.dtype("float64") or not np.isfinite(value).all():
            raise ValueError("invalid Runner population output")
    wanted = output_slice(document, first, stop)
    indices = np.searchsorted(emitted, wanted)
    if np.any(indices >= len(emitted)) or not np.array_equal(emitted[indices], wanted):
        raise ValueError("Runner omitted a scheduled continuation output")
    return {"output_steps": wanted, "actual_times": times[indices],
            **{name: np.asarray(result.observables[name])[indices] for name in ("population", "norm")}}


def _unique_json(value):
    def pairs(items):
        result = {}
        for key, item in items:
            if key in result:
                raise ValueError("duplicate continuation checkpoint JSON key")
            result[key] = item
        return result
    return json.loads(value, object_pairs_hook=pairs)


def _check_tree(group, description):
    """Reject extra children and external links ignored by a generic tree reader."""
    kind = description["kind"]
    if kind == "array":
        expected = {"value"}
        fields = {"kind", "shape", "dtype", "sha256"}
    elif kind == "none":
        expected = set()
        fields = {"kind"}
    elif kind in ("dict", "tuple", "list"):
        expected = {str(i) for i in range(len(description["children"]))}
        fields = {"kind", "children"}
        if kind == "dict":
            fields.add("keys")
            keys = description["keys"]
            if (len(keys) != len(expected) or not all(isinstance(key, str) for key in keys)
                    or len(set(keys)) != len(keys)):
                raise ValueError("invalid continuation checkpoint dictionary")
    else:
        raise ValueError("unknown continuation checkpoint tree kind")
    if set(description) != fields or set(group) != expected or group.attrs:
        raise ValueError("unexpected continuation checkpoint tree contents")
    for name in expected:
        if not isinstance(group.get(name, getlink=True), h5py.HardLink):
            raise ValueError("continuation checkpoint must contain local data")
    if kind not in ("array", "none"):
        for i, child in enumerate(description["children"]):
            _check_tree(group[str(i)], child)
    elif kind == "array":
        value = group["value"]
        if (not isinstance(value, h5py.Dataset) or value.is_virtual or value.external
                or value.attrs):
            raise ValueError("continuation checkpoint arrays must be local numerical datasets")


def read_payload(path, expected_metadata, expected_sha256):
    # Hash, inspect and decode the same immutable bytes, even if the directory
    # entry changes concurrently. One segment is the bounded memory unit.
    snapshot = path.read_bytes()
    if hashlib.sha256(snapshot).hexdigest() != expected_sha256:
        raise ValueError("continuation payload checksum mismatch")
    with h5py.File(BytesIO(snapshot), "r") as handle:
        if (set(handle) != {"state", "auxiliary"}
                or set(handle.attrs) != {"schema_version", "state_type", "metadata", "manifest",
                                         "auxiliary_manifest"}
                or handle.attrs["schema_version"] != 2
                or handle.attrs["state_type"] != "TrajectoryState"
                or _unique_json(handle.attrs["metadata"]) != expected_metadata):
            raise ValueError("continuation payload metadata mismatch")
        for name, manifest in (("state", "manifest"), ("auxiliary", "auxiliary_manifest")):
            if not isinstance(handle.get(name, getlink=True), h5py.HardLink):
                raise ValueError("continuation payload must contain local trees")
            _check_tree(handle[name], _unique_json(handle.attrs[manifest]))
    return load_checkpoint(BytesIO(snapshot), expected_metadata=expected_metadata, with_auxiliary=True)


def assemble(document, identity, pieces):
    values = {name: np.concatenate([np.asarray(part[name]) for part in pieces], axis=0)
              for name in ("population", "norm")}
    return EnsembleResult(np.asarray(document["output_times"], dtype=np.float64),
        np.asarray([identity], dtype=np.uint32), values,
        {name: np.zeros_like(value) for name, value in values.items()},
        document["simulation_manifest"], document["preparation_id"])


def diagnostic_tree(value):
    """Convert declared exception diagnostics to the checkpoint's plain containers."""
    if hasattr(value, "_asdict"):
        return {key: diagnostic_tree(item) for key, item in value._asdict().items()}
    if isinstance(value, dict):
        return {key: diagnostic_tree(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(diagnostic_tree(item) for item in value)
    return value
