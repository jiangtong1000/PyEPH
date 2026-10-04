"""Campaign persistence and process failures, with a numerical ensemble reference."""

from dataclasses import replace
import json
import os
import subprocess
import sys

import jax.numpy as jnp
import numpy as np
import pytest

from pyeph import CoupledClassical, Ehrenfest, Execution, Integrator, Problem, Simulation
from pyeph.core.state import make_state, stack_states
from pyeph.execution.campaign import Campaign, ClaimLostError
from pyeph.execution.ensemble import EnsembleResult, run_ensemble
from pyeph.initialization import sample_harmonic
from pyeph.models.analytic import SpinBosonModel

PREPARATION = "harmonic-w1-T.1-spin0-seed28-v1"


def simulation(**execution):
    model = SpinBosonModel()
    policy = {"chunk_size": 5, "save_every": 3} | execution
    return Simulation(Problem(model, model.default_params(), CoupledClassical(1.), Ehrenfest()),
                      Integrator(.01), Execution(**policy))


def initialize(ids):
    q, p = sample_harmonic([1.], 1., .1, ids, seed=28)
    return stack_states([make_state(q[i], p[i], [1, 0], trajectory_id=int(identity), seed=28)
                         for i, identity in enumerate(ids)])


def campaign(tmp_path, ids=(19, 4, 12, 20, 7, 1, 38), **kwargs):
    return Campaign.create(tmp_path / "campaign", simulation(), ids, 12,
                           preparation_id=PREPARATION, shard_size=3, **kwargs)


def result_for(store, claim):
    """Persistence-only fixture; numerical correctness uses actual dynamics below."""
    specification = store.specification
    times = np.array(specification["output_times"])
    ids = np.array(claim.trajectory_ids)
    samples = ids[:, None] + np.arange(len(times))[None, :]
    return EnsembleResult(times, ids, {"value": samples.mean(axis=0)},
                          {"value": ((samples - samples.mean(axis=0)) ** 2).sum(axis=0)},
                          specification["simulation_manifest"], PREPARATION)


def test_unequal_work_units_resume_merge_and_numerical_reference(tmp_path):
    store = campaign(tmp_path)
    sim = simulation()
    first = store.run_next(sim, initialize, preparation_id=PREPARATION, batch_size=2)
    assert first.trajectory_ids == (19, 4, 12)
    with pytest.raises(ValueError, match="incomplete"):
        store.merge()
    assert store.merge(require_complete=False).count == 3
    reopened = Campaign(store.path)
    while reopened.run_next(sim, initialize, preparation_id=PREPARATION, batch_size=1):
        pass
    actual = reopened.merge()
    expected = run_ensemble(sim, initialize, store.specification["trajectory_ids"], 12,
                            preparation_id=PREPARATION, batch_size=4)
    assert [len(row["trajectory_ids"]) for row in reopened.ledger()] == [3, 3, 1]
    np.testing.assert_array_equal(actual.trajectory_ids, expected.trajectory_ids)
    for key in actual.mean:
        np.testing.assert_allclose(actual.mean[key], expected.mean[key], rtol=2e-13, atol=1e-15)
        np.testing.assert_allclose(actual.m2[key], expected.m2[key], rtol=2e-13, atol=1e-17)
        np.testing.assert_allclose(actual.standard_error[key], expected.standard_error[key],
                                   rtol=2e-13, atol=1e-15)
    assert reopened.run_next(sim, initialize, preparation_id=PREPARATION) is None
    assert len(reopened.attempts()) == 3


def test_merge_order_is_fixed_when_workers_complete_out_of_order(tmp_path):
    store = campaign(tmp_path)
    claims = [store.claim(worker_id=str(i)) for i in range(3)]
    for claim in reversed(claims):
        store.complete(claim, result_for(store, claim))
    merged = store.merge()
    np.testing.assert_array_equal(merged.trajectory_ids, store.specification["trajectory_ids"])
    samples = np.array(store.specification["trajectory_ids"])
    np.testing.assert_allclose(merged.mean["value"][0], samples.mean())
    np.testing.assert_allclose(merged.m2["value"][0], ((samples-samples.mean())**2).sum())
    np.testing.assert_array_equal(merged.mean["value"], Campaign(store.path).merge().mean["value"])


def test_failed_attempt_keeps_original_exception_and_retry_history(tmp_path):
    store = campaign(tmp_path, ids=(9, 4))
    failure = RuntimeError("reference input was unavailable")

    def fail(_):
        raise failure

    with pytest.raises(RuntimeError) as caught:
        store.run_next(simulation(), fail, preparation_id=PREPARATION)
    assert caught.value is failure
    assert store.ledger()[0]["status"] == "failed"
    assert store.claim() is None
    store.retry_failed(0)
    assert store.run_next(simulation(), initialize, preparation_id=PREPARATION).attempt == 2
    assert [item["status"] for item in store.attempts()] == ["failed", "completed"]
    assert "reference input was unavailable" in store.attempts()[0]["error"]
    with pytest.raises(ValueError, match="failed"):
        store.retry_failed(0)


def test_recovered_attempt_is_fenced_and_completed_work_cannot_duplicate(tmp_path):
    store = campaign(tmp_path, ids=(3, 5))
    previous = store.claim(worker_id="interrupted")
    original = result_for(store, previous)
    with pytest.raises(ClaimLostError, match="recovery token"):
        store.recover(0, expected_token="not-current", reason="confirmed process exit")
    store.recover(0, expected_token=previous.token, reason="confirmed process exit")
    current = store.claim(worker_id="replacement")
    for operation in (lambda: store.complete(previous, original),
                      lambda: store.fail(previous, "late error")):
        with pytest.raises(ClaimLostError):
            operation()
    store.complete(current, result_for(store, current))
    with pytest.raises(ClaimLostError):
        store.complete(current, result_for(store, current))
    assert store.merge().count == 2
    assert [item["status"] for item in store.attempts()] == ["interrupted", "completed"]


def test_physics_preparation_output_schedule_and_identity_are_strict(tmp_path):
    store = campaign(tmp_path)
    altered = simulation()
    altered.problem.params["delta"] = .7
    with pytest.raises(ValueError, match="params"):
        store.run_next(altered, initialize, preparation_id=PREPARATION)
    with pytest.raises(ValueError, match="preparation"):
        store.run_next(simulation(), initialize, preparation_id="new-seed")
    with pytest.raises(ValueError, match="observation interval"):
        store.run_next(simulation(save_every=4), initialize, preparation_id=PREPARATION)
    assert not store.attempts()
    claim = store.claim()
    result = result_for(store, claim)
    for changed, message in [(replace(result, trajectory_ids=[1, 2, 3]), "claimed IDs"),
                              (replace(result, times=result.times + .1), "output time grid"),
                              (replace(result, preparation_id="changed"), "preparation")]:
        with pytest.raises(ValueError, match=message):
            store.complete(claim, changed)
    store.complete(claim, result)
    copy = store.specification
    copy["steps"] = 0
    assert store.specification["steps"] == 12
    with pytest.raises(FileExistsError):
        Campaign.create(store.path, simulation(), [1], 12, preparation_id=PREPARATION)


def test_nonzero_initial_time_and_step_match_absolute_output_schedule(tmp_path):
    store = campaign(tmp_path, ids=(5, 8), initial_time=.7, initial_step=2)

    def shifted(ids):
        state = initialize(ids)
        return state._replace(time=jnp.full_like(state.time, .7),
                              step=jnp.full_like(state.step, 2))

    store.run_next(simulation(chunk_size=2), shifted, preparation_id=PREPARATION)
    np.testing.assert_allclose(store.merge().times, [.7, .71, .74, .77, .80, .82], atol=1e-15)


def test_wrong_initial_time_and_step_fail_before_dynamics(tmp_path):
    store = campaign(tmp_path, initial_step=1)
    with pytest.raises(ValueError, match="initializer time/step"):
        store.run_next(simulation(), initialize, preparation_id=PREPARATION)
    assert store.ledger()[0]["status"] == "failed"


def test_numeric_container_roundtrip_and_checksum(tmp_path):
    store = campaign(tmp_path, ids=(2, 4))
    claim = store.claim()
    result = result_for(store, claim)
    x = np.asarray(result.mean["value"], complex) * (1 + 2j)
    result = replace(result, mean={"complex": (x, [x[:, None], None])},
                     m2={"complex": (result.m2["value"], [result.m2["value"][:, None], None])})
    store.complete(claim, result)
    actual = store.merge()
    assert isinstance(actual.mean["complex"], tuple)
    assert isinstance(actual.mean["complex"][1], list)
    np.testing.assert_array_equal(actual.mean["complex"][0], x)
    path = store.path / "results" / store.ledger()[0]["result_file"]
    with path.open("r+b") as stream:
        stream.seek(-1, os.SEEK_END)
        last = stream.read(1)
        stream.seek(-1, os.SEEK_END)
        stream.write(bytes([last[0] ^ 1]))
    with pytest.raises(ValueError, match="checksum"):
        store.merge()


def subprocess_environment():
    return os.environ | {"JAX_ENABLE_X64": "1", "OPENBLAS_NUM_THREADS": "1"}


def test_independent_processes_contend_for_claims_without_duplicates(tmp_path):
    store = campaign(tmp_path, ids=range(31))
    script = """
import json, sys, time
from pyeph.execution.campaign import Campaign
campaign = Campaign(sys.argv[1])
claims = []
while True:
    claim = campaign.claim()
    if claim is None:
        break
    claims.append(list(claim.trajectory_ids))
    time.sleep(.01)
print(json.dumps(claims), flush=True)
"""
    processes = [subprocess.Popen([sys.executable, "-c", script, str(store.path)],
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                                   env=subprocess_environment()) for _ in range(4)]
    try:
        chunks = []
        for process in processes:
            stdout, stderr = process.communicate(timeout=45)
            assert process.returncode == 0, stderr
            chunks.extend(json.loads(stdout))
        ids = [identity for chunk in chunks for identity in chunk]
        assert sorted(ids) == list(range(31))
        assert len(ids) == len(set(ids))
        assert all(row["status"] == "running" for row in store.ledger())
        assert store.claim() is None
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=10)


CRASH_SCRIPT = """
from dataclasses import asdict
import json, os, sys
import numpy as np
import pyeph.execution.campaign as module
from pyeph.execution.ensemble import EnsembleResult
store = module.Campaign(sys.argv[1])
claim = store.claim(worker_id='crash-test')
print(json.dumps(asdict(claim)), flush=True)
specification = store.specification
times = np.array(specification['output_times'])
result = EnsembleResult(times, np.array(claim.trajectory_ids),
                        {'x': np.zeros(len(times))}, {'x': np.zeros(len(times))},
                        specification['simulation_manifest'], specification['preparation_id'])
if sys.argv[2] == 'before_publish':
    module._write_result = lambda *args: os._exit(77)
elif sys.argv[2] == 'after_publish':
    module._sync_directory = lambda *args: os._exit(77)
store.complete(claim, result)
os._exit(77)
"""


@pytest.mark.parametrize("stage", ["before_publish", "after_publish", "after_commit"])
def test_real_process_exit_at_publication_boundaries_is_recoverable(tmp_path, stage):
    store = campaign(tmp_path, ids=(8, 4))
    process = subprocess.run([sys.executable, "-c", CRASH_SCRIPT, str(store.path), stage],
                              capture_output=True, text=True, env=subprocess_environment(), timeout=45)
    assert process.returncode == 77, process.stderr
    claim_data = json.loads(process.stdout)
    reopened = Campaign(store.path)
    if stage == "after_commit":
        assert reopened.ledger()[0]["status"] == "completed"
        assert reopened.merge().count == 2
        assert reopened.claim() is None
    else:
        assert reopened.ledger()[0]["status"] == "running"
        with pytest.raises(ValueError, match="incomplete"):
            reopened.merge()
        # returncode proves this process terminated; elapsed time alone would not.
        reopened.recover(0, expected_token=claim_data["token"],
                          reason=f"worker exited with status {process.returncode}")
        replacement = reopened.claim()
        reopened.complete(replacement, result_for(reopened, replacement))
        assert reopened.merge().count == 2
        assert len(reopened.attempts()) == 2
        if stage == "after_publish":
            assert len(list((store.path / "results").glob("*.npz"))) == 2


def test_manifest_or_ledger_tampering_is_not_interpreted_as_completion(tmp_path):
    import sqlite3

    store = campaign(tmp_path)
    with sqlite3.connect(store.path / "ledger.sqlite") as connection:
        connection.execute("DELETE FROM shards WHERE id=2")
    with pytest.raises(ValueError, match="work units"):
        Campaign(store.path)


def test_accepted_accumulation_roundoff_uses_one_canonical_output_grid(tmp_path):
    store = campaign(tmp_path)
    claims = [store.claim() for _ in range(3)]
    eps = np.finfo(np.float64).eps
    for claim, offset in zip(claims, [20*eps, -20*eps, 0], strict=True):
        result = result_for(store, claim)
        store.complete(claim, replace(result, times=result.times+offset))
    merged = store.merge()
    np.testing.assert_array_equal(merged.times, store.specification["output_times"])
    assert merged.count == 7


def test_result_time_precision_cannot_weaken_grid_validation(tmp_path):
    store = campaign(tmp_path, ids=(2, 8))
    claim = store.claim()
    result = result_for(store, claim)
    for dtype in (np.float16, np.float32):
        with pytest.raises(ValueError, match="time dtype"):
            store.complete(claim, replace(result, times=(result.times+.015).astype(dtype)))
    assert store.ledger()[0]["status"] == "running"
    with pytest.raises(ValueError, match="output time grid"):
        store.complete(claim, replace(result, times=result.times+.015))


def test_time_precision_is_declared_and_insufficient_resolution_rejected(tmp_path):
    with pytest.raises(ValueError, match="float32 or float64"):
        campaign(tmp_path, time_dtype="float16")
    with pytest.raises(ValueError, match="distinct"):
        campaign(tmp_path, time_dtype="float32", initial_time=2**30)
    store = campaign(tmp_path, ids=(3, 4), time_dtype="float32")
    with pytest.raises(ValueError, match="initializer time dtype"):
        store.run_next(simulation(), initialize, preparation_id=PREPARATION)
    store.retry_failed(0)

    def float32_times(ids):
        state = initialize(ids)
        return state._replace(time=state.time.astype(jnp.float32))

    store.run_next(simulation(), float32_times, preparation_id=PREPARATION)
    assert store.merge().times.dtype == np.dtype("float32")
