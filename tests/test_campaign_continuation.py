"""Independent scalar continuation: physical references and real crash recovery."""

from dataclasses import replace
import hashlib
import os
import sqlite3
import subprocess
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pyeph import CPA, Ehrenfest, Execution, Integrator, Problem, Simulation
from pyeph.core.geometry import CoordinateBox
from pyeph.core.problem import CoupledClassical
from pyeph.core.state import make_state
from pyeph.execution.campaign import Campaign, ClaimLostError
from pyeph.execution.runner import SimulationError
from pyeph.integrators.krylov import LanczosOptions
from pyeph.io.checkpoint import load_checkpoint, save_checkpoint
from pyeph.models.analytic import SpinBosonModel
from pyeph.models.composite import SumModel
from pyeph.paths.harmonic import HarmonicBath

PREPARATION = 'fixed-provider-continuation-seed913-v1'
BUNDLE = 'fixed-provider-fixture-v1'
ARTIFACTS = {'model.models[1]': 'independent-example-reference-source-and-parameters-v1'}


def physics(profile='molecular', method='cpa', *, stride=4, chunk=2):
    """Reuse fixed-support application providers; no integration equations here."""
    if profile == 'molecular':
        from examples.oriented_fragments import fixture
        problem, initial = fixture(2, method)
        problem = replace(problem, measurement=None)
    else:
        from examples.perovskite import HarmonicReference, build_cspbi3
        carrier, params, q, masses = build_cspbi3(spinful=False)
        reference = HarmonicReference(replace(carrier.spec, name='continuation_tethers'))
        model = SumModel((carrier, reference), additive_probes=carrier.spec.probes)
        reference_params = dict(equilibrium=q, spring=jnp.full(q.shape, .002))
        q = q+.01*jnp.sin(jnp.arange(q.size).reshape(q.shape))
        p = .03*jnp.cos(jnp.arange(q.size).reshape(q.shape))
        c = jnp.arange(1, carrier.nstates+1)+.1j
        c = c/jnp.linalg.norm(c)
        nuclear = HarmonicBath(0., masses) if method == 'cpa' else CoupledClassical(masses)
        problem = Problem(model, (params, reference_params), nuclear, CPA() if method == 'cpa' else Ehrenfest())
        initial = make_state(q, p, c)
    sim = Simulation(problem, Integrator(.01, LanczosOptions(max_dimension=16)),
                     Execution(chunk_size=chunk, save_every=stride))
    return sim, initial


def state_for(initial, identity, *, step=2, time=.7):
    return make_state(initial.q, initial.p, initial.electronic, trajectory_id=identity,
                      seed=913, step=step, time=time,
                      method_state={'marker': np.array([17, 29], dtype=np.int64)})


def create(path, sim, *, ids=(13,), steps=9, segment=3, start=2, time=.7, artifacts=ARTIFACTS):
    return Campaign.create(path, sim, ids, steps, preparation_id=PREPARATION, shard_size=1,
        initial_step=start, initial_time=time, continuation_steps=segment,
        provider_bundle_id=BUNDLE, artifact_ids=artifacts)


def run(store, sim, initial, *, initialize=None, artifacts=ARTIFACTS):
    spec = store.specification
    if initialize is None:
        def initialize(identity):
            return state_for(initial, identity, step=spec['initial_step'], time=spec['initial_time'])
    return store.run_next_scalar(sim, initialize, preparation_id=PREPARATION,
                                 artifact_ids=artifacts, provider_bundle_id=BUNDLE)


def equal_state(a, b, *, exact):
    assert jax.tree.structure(a) == jax.tree.structure(b)
    for left, right in zip(jax.tree.leaves(a), jax.tree.leaves(b), strict=True):
        left, right = np.asarray(left), np.asarray(right)
        assert left.shape == right.shape and left.dtype == right.dtype
        if exact or left.dtype.kind in 'biu':
            assert left.tobytes() == right.tobytes()
        else:
            np.testing.assert_allclose(left, right, atol=3e-13, rtol=3e-13)


def payload(store, row):
    return store.path/'segments'/row['payload_file']


def schedule(start, steps, stride):
    terminal = start+steps
    return sorted({start, terminal, *range(((start//stride)+1)*stride, terminal+1, stride)})


def segment_reference(sim, initial, steps, segment):
    state = initial
    left = steps
    while left:
        count = min(left, segment)
        state = sim.run(state, count).final_state
        left -= count
    return state


@pytest.mark.parametrize('profile,method', [('molecular', 'cpa'), ('molecular', 'ehrenfest'),
                                          ('periodic', 'cpa'), ('periodic', 'ehrenfest')])
def test_fixed_application_providers_complete_once_and_match_segmented_reference(tmp_path, profile, method):
    sim, initial = physics(profile, method)
    store = create(tmp_path/'campaign', sim)
    run(store, sim, initial)
    rows = store.segments(0)
    assert len(rows) == 4
    assert [r['generation'] for r in rows] == list(range(4))
    all_steps = []
    for index, row in enumerate(rows):
        state, metadata, aux = load_checkpoint(payload(store, row), with_auxiliary=True)
        assert int(state.step) == row['end_step']
        assert int(state.trajectory_id) == 13
        all_steps.extend(np.asarray(aux['output_steps']).tolist())
        assert row['next_output_index']-row['first_output_index'] == len(aux['output_steps'])
        if index:
            assert row['start_step'] == rows[index-1]['end_step']
            assert row['parent_sha256'] == rows[index-1]['payload_sha256']
        else:
            assert row['start_step'] == row['end_step'] == 2
    assert all_steps == schedule(2, 9, 4)
    expected = segment_reference(sim, state_for(initial, 13), 9, 3)
    equal_state(state, expected, exact=True)
    long = sim.run(state_for(initial, 13), 9)
    equal_state(state, long.final_state, exact=False)
    merged = store.merge()
    assert merged.count == 1
    for key in merged.mean:
        np.testing.assert_allclose(merged.mean[key], long.observables[key], atol=3e-13, rtol=3e-13)
        np.testing.assert_array_equal(merged.m2[key], np.zeros_like(merged.m2[key]))
    assert run(store, sim, initial) is None


@pytest.mark.parametrize('start,steps,segment,stride', [(0, 0, 3, 5), (2, 1, 3, 5),
    (2, 9, 2, 7), (5, 8, 3, 4)])
def test_integer_schedule_has_no_segment_endpoints_or_duplicate_initial_rows(tmp_path, start, steps, segment, stride):
    sim, initial = physics(stride=stride)
    store = create(tmp_path/'campaign', sim, start=start, steps=steps, segment=segment)
    run(store, sim, initial)
    values = [load_checkpoint(payload(store, row), with_auxiliary=True)[2] for row in store.segments(0)]
    actual = np.concatenate([v['output_steps'] for v in values])
    expected = schedule(start, steps, stride)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_allclose(store.merge().times, .7+(np.array(expected)-start)*.01, atol=1e-15, rtol=0)
    assert len(actual) == len(set(actual.tolist()))
    if stride > segment and steps > stride:
        assert any(len(v['output_steps']) == 0 for v in values[1:])


CRASH_SCRIPT = r'''
import importlib.util, os, sys
from contextlib import contextmanager
import jax
jax.config.update('jax_enable_x64', True)
spec = importlib.util.spec_from_file_location('continuation_acceptance_support', sys.argv[2])
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)
from pyeph.execution import campaign as module
from pyeph.execution.campaign import Campaign
store=Campaign(sys.argv[1]); phase=sys.argv[3]
sim,initial=helper.physics(sys.argv[4],sys.argv[5])
original_publish=Campaign._publish_segment
original_replace=module.os.replace
original_sync=module._sync_directory
original_connection=Campaign._connection
original_complete=Campaign.complete
active=False

def replace_file(*args,**kwargs):
    final_payload = str(args[1]).endswith('.h5') and not os.path.basename(args[1]).startswith('.')
    if active and final_payload and phase=='before_hdf_rename': os._exit(77)
    value=original_replace(*args,**kwargs)
    if active and final_payload and phase=='after_hdf_rename': os._exit(77)
    return value

def sync(*args,**kwargs):
    if active and phase=='before_directory_fsync': os._exit(77)
    value=original_sync(*args,**kwargs)
    if active and phase=='after_directory_fsync': os._exit(77)
    return value

@contextmanager
def connection(self,*,transaction=False):
    with original_connection(self,transaction=transaction) as con:
        yield con
    if active and transaction and phase=='after_sqlite_commit':os._exit(77)

def publish(self,claim,simulation,state,*args,**kwargs):
    global active
    active=int(state.step)>self.specification['initial_step']
    return original_publish(self,claim,simulation,state,*args,**kwargs)

def complete(self,*args,**kwargs):
    if phase=='before_final_publish':os._exit(77)
    value=original_complete(self,*args,**kwargs)
    if phase=='after_final_publish':os._exit(77)
    return value
module.os.replace=replace_file
module._sync_directory=sync
Campaign._connection=connection
Campaign._publish_segment=publish
Campaign.complete=complete
helper.run(store,sim,initial)
raise AssertionError('crash boundary was not reached')
'''


def crash(store, phase, profile='molecular', method='cpa'):
    env = os.environ | {'JAX_ENABLE_X64': '1', 'OPENBLAS_NUM_THREADS': '1'}
    result = subprocess.run([sys.executable, '-c', CRASH_SCRIPT, str(store.path), __file__,
                             phase, profile, method], capture_output=True, text=True, env=env, timeout=90)
    assert result.returncode == 77, result.stdout+result.stderr
    return Campaign(store.path)


def recover(store):
    row = store.ledger()[0]
    store.recover(0, expected_token=row['token'], reason='independently observed subprocess exit77')


@pytest.mark.parametrize('phase', ['before_hdf_rename', 'after_hdf_rename', 'before_directory_fsync',
    'after_directory_fsync', 'after_sqlite_commit', 'before_final_publish', 'after_final_publish'])
def test_real_crashes_only_resume_committed_segments_and_never_reinitialize(tmp_path, phase):
    sim, initial = physics()
    store = create(tmp_path/'campaign', sim)
    interrupted = crash(store, phase)
    committed = interrupted.segments(0)
    expected_count = 4 if 'final_publish' in phase else 2 if phase=='after_sqlite_commit' else 1
    assert len(committed) == expected_count
    identities = [(r['generation'], r['payload_sha256']) for r in committed]
    if phase != 'after_final_publish':
        recover(interrupted)
        def forbid_initializer(_):
            raise AssertionError('committed generation0 must bypass initialization')
        run(interrupted, sim, initial, initialize=forbid_initializer)
    assert interrupted.merge().count == 1
    assert [(r['generation'],r['payload_sha256']) for r in interrupted.segments(0)[:len(committed)]] == identities
    final, _, _ = load_checkpoint(payload(interrupted, interrupted.segments(0)[-1]), with_auxiliary=True)
    equal_state(final, segment_reference(sim, state_for(initial,13),9,3), exact=True)
    assert interrupted.ledger()[0]['status']=='completed'


@pytest.mark.parametrize('profile,method', [('molecular','ehrenfest'),('periodic','cpa'),('periodic','ehrenfest')])
def test_real_application_restart_matches_same_segment_schedule(tmp_path, profile, method):
    sim, initial = physics(profile,method)
    store = create(tmp_path/'campaign',sim)
    store = crash(store,'after_sqlite_commit',profile,method)
    recover(store)
    run(store,sim,initial,initialize=lambda _: pytest.fail('unexpected reinitialization'))
    state,_,_=load_checkpoint(payload(store,store.segments(0)[-1]),with_auxiliary=True)
    equal_state(state,segment_reference(sim,state_for(initial,13),9,3),exact=True)


def publish_initial(store, sim, initial):
    claim = store.claim()
    state = state_for(initial, claim.trajectory_ids[0], step=store.specification['initial_step'],
                      time=store.specification['initial_time'])
    population = np.asarray(abs(state.electronic)**2)
    aux = {'output_steps': np.array([int(state.step)], dtype=np.int64),
           'actual_times': np.array([float(state.time)], dtype=np.float64),
           'population': population[None], 'norm': np.array([population.sum()])}
    row = store._publish_segment(claim, sim, state, aux, None, artifact_ids=ARTIFACTS)
    return claim, state, aux, row


def test_stale_worker_cannot_publish_generation_after_explicit_recovery(tmp_path):
    sim, initial = physics()
    store = create(tmp_path/'campaign', sim)
    old, state, aux, row = publish_initial(store, sim, initial)
    recover(store)
    replacement = store.claim()
    assert replacement.token != old.token
    with pytest.raises(ClaimLostError):
        store._publish_segment(old, sim, state, aux, None, artifact_ids=ARTIFACTS)
    assert store.segments(0) == [row]
    assert store.ledger()[0]['token'] == replacement.token


@pytest.mark.parametrize('fault', ['payload_bytes', 'output_index', 'parent', 'producer',
                                   'population_shape', 'output_dtype', 'metadata'])
def test_referenced_corruption_is_rejected_without_fallback_or_initializer(tmp_path, fault):
    sim, initial = physics()
    store = create(tmp_path/'campaign', sim)
    _, state, aux, row = publish_initial(store, sim, initial)
    path = payload(store, row)
    if fault == 'payload_bytes':
        path.write_bytes(path.read_bytes()+b'changed')
    elif fault in ('output_index', 'parent', 'producer'):
        column, value = {'output_index': ('next_output_index', 0),
                         'parent': ('parent_sha256', 'f'*64),
                         'producer': ('token', 'invented-producer')}[fault]
        with sqlite3.connect(store.path/'ledger.sqlite') as connection:
            connection.execute(f'UPDATE segments SET {column}=?', (value,))
    else:
        _, metadata, _ = load_checkpoint(path, with_auxiliary=True)
        if fault == 'population_shape':
            aux['population'] = aux['population'][0]
        elif fault == 'output_dtype':
            aux['output_steps'] = aux['output_steps'].astype(np.float64)
        else:
            metadata['unexpected'] = 'unvalidated data'
        save_checkpoint(path, state, metadata=metadata, auxiliary=aux)
        checksum = hashlib.sha256(path.read_bytes()).hexdigest()
        with sqlite3.connect(store.path/'ledger.sqlite') as connection:
            connection.execute('UPDATE segments SET payload_sha256=?', (checksum,))
    recover(store)
    with pytest.raises(ValueError):
        run(store, sim, initial, initialize=lambda _: pytest.fail('corrupt archive reinitialized'))
    assert len(store.segments(0)) == 1
    assert store.ledger()[0]['status'] == 'failed'


@pytest.mark.parametrize('fault', ['bundle', 'external_source', 'domain', 'chunk', 'preparation'])
def test_frozen_identity_mismatch_rejected_before_claim_or_initializer(tmp_path, fault):
    sim, initial = physics()
    store = create(tmp_path/'campaign', sim)
    bundle, artifacts, preparation = BUNDLE, ARTIFACTS, PREPARATION
    if fault == 'bundle':
        bundle += '-different'
    elif fault == 'external_source':
        artifacts = {'model.models[1]': 'different-implementation-identity'}
    elif fault == 'domain':
        q = np.asarray(initial.q)
        sim = Simulation(replace(sim.problem, geometry_guard=CoordinateBox(q-1., q+1.)),
                         sim.integrator, sim.execution)
    elif fault == 'chunk':
        sim = Simulation(sim.problem, sim.integrator, replace(sim.execution, chunk_size=3))
    else:
        preparation += '-different'
    with pytest.raises(ValueError):
        store.run_next_scalar(sim, lambda _: pytest.fail('identity mismatch reached initializer'),
            preparation_id=preparation, artifact_ids=artifacts, provider_bundle_id=bundle)
    assert store.ledger()[0]['status'] == 'pending'
    assert store.attempts() == []


@pytest.mark.parametrize('method', ['cpa', 'ehrenfest'])
def test_failed_later_chunk_retains_only_committed_segment_and_diagnostic(tmp_path, method):
    model = SpinBosonModel()
    params = model.default_params() | {'omega': jnp.zeros(1), 'coupling': jnp.zeros(1)}
    treatment = HarmonicBath(0., 1.) if method == 'cpa' else CoupledClassical(1.)
    problem = Problem(model, params, treatment, CPA() if method == 'cpa' else Ehrenfest(),
                      geometry_guard=CoordinateBox([-1.], [.075]))
    sim = Simulation(problem, Integrator(.01, LanczosOptions(max_dimension=2)),
                     Execution(chunk_size=2, save_every=2))
    initial = make_state([0.], [1.], [1.+0.j, 0.+0.j])
    store = create(tmp_path/'campaign', sim, steps=12, segment=4, start=0, time=0., artifacts=None)
    with pytest.raises(SimulationError) as caught:
        run(store, sim, initial, artifacts=None)
    rows = store.segments(0)
    assert [r['end_step'] for r in rows] == [0, 4]
    assert int(caught.value.last_valid_state.step) == 6
    committed, _, _ = load_checkpoint(payload(store, rows[-1]), with_auxiliary=True)
    assert int(committed.step) == 4
    assert store.ledger()[0]['status'] == 'failed'
    files = list((store.path/'failures').glob('*.h5'))
    assert len(files) == 1
    retained, _, diagnostic = load_checkpoint(files[0], with_auxiliary=True)
    equal_state(retained, committed, exact=True)
    assert set(diagnostic) == {'last_valid_state', 'failed_state', 'diagnostics'}
    assert int(diagnostic['last_valid_state']['step']) == 6
    assert store.claim() is None
    with pytest.raises(ValueError):
        store.merge()


def small_simulation(method='ehrenfest'):
    model = SpinBosonModel()
    nuclear = CoupledClassical(1.) if method == 'ehrenfest' else HarmonicBath(.6, 1.)
    problem = Problem(model, model.default_params(), nuclear,
                      Ehrenfest() if method == 'ehrenfest' else CPA())
    return Simulation(problem, Integrator(.01, LanczosOptions(max_dimension=2)),
                      Execution(chunk_size=2, save_every=4))


def diverse_state(identity):
    angle = .03*identity
    return make_state([.01*identity], [-.02*identity],
        [np.cos(angle)+0.j, 1j*np.sin(angle)], trajectory_id=identity,
        seed=913, step=2, time=.7)


CONCURRENT_SCRIPT = r'''
import importlib.util, sys, time
from pathlib import Path
import jax
jax.config.update('jax_enable_x64', True)
spec=importlib.util.spec_from_file_location('continuation_acceptance_support',sys.argv[2])
helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
store=helper.Campaign(sys.argv[1]);sim=helper.small_simulation(sys.argv[4])
ready=Path(sys.argv[3]); worker=sys.argv[5]; first=True

def initialize(identity):
    global first
    if first:
        first=False
        (ready/worker).write_text(str(identity))
        deadline=time.monotonic()+30
        while len(list(ready.iterdir()))<3:
            if time.monotonic()>deadline:raise RuntimeError('workers failed to overlap')
            time.sleep(.01)
    return helper.diverse_state(identity)
while store.run_next_scalar(sim,initialize,preparation_id=helper.PREPARATION,
                           worker_id=worker,provider_bundle_id=helper.BUNDLE):pass
'''


@pytest.mark.parametrize('method', ['cpa', 'ehrenfest'])
def test_concurrent_scalar_writers_reduce_independent_trajectory_statistics(tmp_path, method):
    ids = (19, 4, 12, 20, 7, 1, 38)
    sim = small_simulation(method)
    store = create(tmp_path/'campaign', sim, ids=ids, steps=7, segment=3, artifacts=None)
    ready = tmp_path/'ready'
    ready.mkdir()
    env = os.environ | {'JAX_ENABLE_X64': '1', 'OPENBLAS_NUM_THREADS': '1'}
    workers = [subprocess.Popen([sys.executable, '-c', CONCURRENT_SCRIPT, str(store.path),
        __file__, str(ready), method, f'worker-{i}'], stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, text=True, env=env) for i in range(3)]
    try:
        for worker in workers:
            stdout, stderr = worker.communicate(timeout=90)
            assert worker.returncode == 0, stdout+stderr
    finally:
        for worker in workers:
            if worker.poll() is None:
                worker.kill()
                worker.wait()
    # All three workers claimed distinct scalar IDs before any could initialize.
    assert len({p.read_text() for p in ready.iterdir()}) == 3
    ledger = store.ledger()
    assert len(ledger) == len(ids) == len(store.attempts())
    assert {row['worker_id'] for row in ledger} == {'worker-0', 'worker-1', 'worker-2'}
    assert all(row['status'] == 'completed' and row['attempt'] == 1 for row in ledger)
    actual = store.merge()
    assert actual.count == len(ids)
    np.testing.assert_array_equal(actual.trajectory_ids, ids)
    references = [sim.run(diverse_state(identity), 7) for identity in ids]
    for key in actual.mean:
        samples = np.stack([np.asarray(result.observables[key]) for result in references])
        mean = samples.mean(axis=0)
        m2 = np.sum((samples-mean)**2, axis=0)
        np.testing.assert_allclose(actual.mean[key], mean, atol=3e-13, rtol=3e-13)
        np.testing.assert_allclose(actual.m2[key], m2, atol=3e-14, rtol=3e-12)
        np.testing.assert_allclose(actual.standard_error[key],
                                   np.sqrt(m2/(len(ids)*(len(ids)-1))), atol=3e-14, rtol=3e-12)
    assert np.max(actual.m2['population']) > .01
    for shard, identity in enumerate(ids):
        rows = store.segments(shard)
        assert [row['generation'] for row in rows] == [0, 1, 2, 3]
        state, _, _ = load_checkpoint(payload(store, rows[-1]), with_auxiliary=True)
        equal_state(state, segment_reference(sim, diverse_state(identity), 7, 3), exact=True)


def test_initializer_parameter_mutation_rejected_before_provider_preflight(tmp_path, monkeypatch):
    sim = small_simulation()
    store = create(tmp_path/'campaign', sim, artifacts=None)

    def mutate(identity):
        sim.update_parameters(sim.problem.params | {'delta': .7})
        return diverse_state(identity)

    def forbid_provider(_):
        pytest.fail('mutated initializer reached provider preflight')

    monkeypatch.setattr(sim, '_validate_state', forbid_provider)
    with pytest.raises(ValueError, match='provenance'):
        run(store, sim, None, initialize=mutate, artifacts=None)
    assert store.segments(0) == []
    assert store.ledger()[0]['status'] == 'failed'


def test_historical_token_path_rejected_before_any_payload_read(tmp_path, monkeypatch):
    from pyeph.execution import campaign as module
    sim, initial = physics()
    store = create(tmp_path/'campaign', sim)
    claim, _, _, row = publish_initial(store, sim, initial)
    recover(store)
    malicious = '../'+'f'*29
    with sqlite3.connect(store.path/'ledger.sqlite') as connection:
        connection.execute('UPDATE segments SET token=?,payload_file=?',
                           (malicious, f'00000000-00000000-{malicious}.h5'))
        connection.execute('UPDATE attempts SET token=? WHERE token=?', (malicious, claim.token))
    monkeypatch.setattr(module, '_digest', lambda _: pytest.fail('unsafe token reached filesystem'))
    with pytest.raises(ValueError, match='token'):
        run(store, sim, initial, initialize=lambda _: pytest.fail('unsafe archive reinitialized'))
    assert len(store.segments(0)) == 1
    assert row['payload_sha256'] == store.segments(0)[0]['payload_sha256']


@pytest.mark.parametrize('dtype', [np.int64, np.float32])
def test_forged_final_moment_precision_is_rejected_despite_equal_zero_values(tmp_path, dtype):
    from pyeph.execution.ensemble import EnsembleResult
    sim, initial = physics()
    store = create(tmp_path/'campaign', sim, steps=0)
    claim, _, aux, _ = publish_initial(store, sim, initial)
    means = {name: aux[name] for name in ('population', 'norm')}
    result = EnsembleResult(np.array([.7]), np.array([13], dtype=np.uint32), means,
        {name: np.zeros_like(value, dtype=dtype) for name, value in means.items()},
        store.specification['simulation_manifest'], PREPARATION)
    with pytest.raises(ValueError, match='committed continuation output'):
        store.complete(claim, result)
    assert store.ledger()[0]['status'] == 'running'
    assert not list((store.path/'results').iterdir())


def test_cpa_hidden_path_excursion_keeps_generation_zero_only(tmp_path):
    from test_geometry_guard_adversarial import ReturningPath
    model = SpinBosonModel()
    problem = Problem(model, model.default_params(), ReturningPath(), CPA(),
                      geometry_guard=CoordinateBox([-.5], [.5]))
    sim = Simulation(problem, Integrator(1., LanczosOptions(max_dimension=2)),
                     Execution(chunk_size=1, save_every=1))
    initial = make_state([0.], [0.], [1.+0.j, 0.+0.j])
    artifacts = {'nuclear_treatment': 'analytic-returning-path-test-v1'}
    store = create(tmp_path/'campaign', sim, steps=1, start=0, time=0., artifacts=artifacts)
    with pytest.raises(SimulationError) as caught:
        run(store, sim, initial, artifacts=artifacts)
    assert [row['end_step'] for row in store.segments(0)] == [0]
    assert int(caught.value.last_valid_state.step) == 0
    assert store.ledger()[0]['status'] == 'failed'
    paths = list((store.path/'failures').glob('*.h5'))
    assert len(paths) == 1
    state, _, auxiliary = load_checkpoint(paths[0], with_auxiliary=True)
    assert int(state.step) == 0
    np.testing.assert_array_equal(auxiliary['diagnostics']['step_info']['attempted_q'], [1.])


def test_arithmetic_output_slices_equal_independent_integer_schedule():
    from pyeph.execution._continuation import output_count, output_slice
    for start in (0, 1, 2, 5, 13):
        for count in range(10):
            for stride in (1, 2, 4, 7, 10**30):
                document = {'initial_step': start, 'steps': count, 'save_every': stride}
                expected = schedule(start, count, stride)
                for end in range(start, start+count+1):
                    assert output_count(document, end) == sum(step <= end for step in expected)
                for first in range(len(expected)+1):
                    for stop in range(first, len(expected)+1):
                        actual = output_slice(document, first, stop)
                        assert actual.dtype == np.dtype('int64')
                        np.testing.assert_array_equal(actual, expected[first:stop])


def test_small_segment_does_not_materialize_full_campaign_schedule(monkeypatch):
    from pyeph.execution import _continuation as module
    document = {'initial_step': 2, 'steps': 10**12, 'save_every': 7,
                'continuation': {'steps': 3}}
    monkeypatch.setattr(module, 'grid', lambda _: pytest.fail('materialized full campaign grid'))
    np.testing.assert_array_equal(module.output_slice(document, 1, 4), [7, 14, 21])
    assert module.output_count(document, 5) == 1
    previous = {'generation': 0, 'end_step': 2, 'next_output_index': 1}
    assert module.bounds(document, previous) == (1, 2, 5, 1, 1)
