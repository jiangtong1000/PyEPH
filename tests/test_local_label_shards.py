"""Provider-owned shards retain identities, weights and derivative semantics."""
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from examples import local_label_shards as example


@pytest.fixture(scope='module', params=('molecular', 'periodic'))
def generated(request):
    item = example.profile(request.param)
    arrays, groups, oracle_error = example.generated_data(item)
    return item, arrays, groups, oracle_error


def dataset(tmp_path, generated):
    item, arrays, groups, _ = generated
    path = tmp_path/'labels'
    checksum = example.write_dataset(path, item, arrays, groups)
    index, masks = example.read_index(path, item, checksum)
    return path, index, masks


def test_independent_all_atom_derivatives_and_shape_valid_omission(generated):
    item, arrays, _, error = generated
    assert error < 3e-9
    assert example.derivative_errors(item, np.ones(3), arrays) < 3e-9
    incomplete = {key: value.copy() for key, value in arrays.items()}
    incomplete['reference_force'][:] = 0.
    assert example.derivative_errors(item, np.ones(3), incomplete) > 1e-7


def test_unequal_shards_use_global_family_counts_and_parameter_gradients(tmp_path, generated):
    item, _, _, _ = generated
    path, index, masks = dataset(tmp_path, generated)
    assert [len(row['ids']) for row in index['shards']] == [3, 5, 4]
    assert {name: int(mask.sum()) for name, mask in masks.items()} == dict(train=6, validation=3, test=3)
    rows = list(example.read_shards(path, item, index))
    joined = {key: np.concatenate([values[key] for _, values in rows]) for key in example.KEYS}
    theta = np.array([.91, 1.07, 1.12])
    streamed = example.aggregate(path, item, index, masks, theta)
    evaluate = example.make_loss(item)
    for name, mask in masks.items():
        value, gradient = evaluate(theta, joined, mask)
        np.testing.assert_allclose(streamed[name]['loss'], value/mask.sum(), atol=2e-13, rtol=0)
        np.testing.assert_allclose(streamed[name]['gradient'], gradient/mask.sum(), atol=2e-12, rtol=0)
    # Discrete objective AD is checked independently of its analytical reverse
    # pass. Small-fixture force labels are checked against separate NumPy code.
    for width in (2e-4, 1e-4):
        finite = []
        for direction in np.eye(3)*width:
            plus = float(evaluate(theta+direction, joined, masks['train'])[0])
            minus = float(evaluate(theta-direction, joined, masks['train'])[0])
            finite.append((plus-minus)/(2*width*masks['train'].sum()))
        np.testing.assert_allclose(finite, streamed['train']['gradient'], atol=2e-9, rtol=0)


@pytest.mark.parametrize('failure', ('gauge', 'baseline', 'reference', 'images', 'mixed', 'stale', 'missing'))
def test_rejects_wrong_conventions_and_shard_contents(tmp_path, generated, failure):
    item, _, _, _ = generated
    path, index, _ = dataset(tmp_path, generated)
    first = index['shards'][0]
    if failure == 'gauge':
        index['contract']['phase_convention'] = 'changed phase order'
    elif failure in ('baseline', 'reference'):
        index['contract'][failure+'_sha256'] = 'f'*64
    elif failure == 'images':
        index['contract']['graph']['edges'].reverse()
    elif failure == 'mixed':
        first['dataset_id'] = 'f'*64
    elif failure == 'stale':
        (path/first['file']).write_bytes((path/first['file']).read_bytes()+b'changed')
    else:
        with np.load(path/first['file'], allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in archive.files if key != 'carrier_gradient'}
        np.savez(path/first['file'], **arrays)
        first['sha256'] = example.digest((path/first['file']).read_bytes())
    example.write_json(path/'index.json', index)
    # Reseal the outer index to exercise semantic/payload checks, not just its hash.
    checksum = example.digest((path/'index.json').read_bytes())
    with pytest.raises(ValueError):
        changed, _ = example.read_index(path, item, checksum)
        list(example.read_shards(path, item, changed))


def test_two_force_contractions_do_not_identify_a_real_symmetric_derivative():
    states, derivative = example.contraction_witness()
    np.testing.assert_array_equal(derivative, derivative.T)
    measured = np.einsum('ki,ij,kj->k', states.conj(), derivative, states).real
    np.testing.assert_allclose(measured, 0., atol=2e-15, rtol=0)
    assert np.max(abs(np.linalg.eigvalsh(derivative))) > .3
    assert np.linalg.norm(derivative) == pytest.approx(1.)


def test_cli_never_adds_failure_record_to_an_existing_output(tmp_path):
    sentinel = tmp_path/'keep.txt'
    sentinel.write_text('preserve')
    result = subprocess.run([sys.executable, '-m', 'examples.local_label_shards', '--output', str(tmp_path)],
                            cwd=Path(example.__file__).resolve().parents[1], capture_output=True, text=True)
    assert result.returncode != 0
    assert sorted(path.name for path in tmp_path.iterdir()) == ['keep.txt']
    assert sentinel.read_text() == 'preserve'


@pytest.mark.parametrize('source,key', (
    ('native', 'action'), ('native', 'hopping'), ('oracle', 'carrier_gradient'),
    ('oracle', 'reference_energy'),
))
def test_generation_rejects_nonfinite_native_and_oracle_evidence(monkeypatch, generated, source, key):
    item, arrays, _, _ = generated
    native = {name: arrays[name][0].copy() for name in example.TARGETS}
    oracle = {name: native[name].copy() for name in
              ('action', 'carrier_gradient', 'reference_energy', 'reference_force')}
    (native if source == 'native' else oracle)[key] = np.full_like(native[key], np.nan)
    monkeypatch.setattr(example, 'predict', lambda *args: native)
    monkeypatch.setattr(example, 'independent', lambda *args: oracle)
    with pytest.raises(ValueError, match='nonfinite'):
        example.generated_data(item)


def test_validation_shear_geometries_are_distinct_and_follow_recorded_formula(generated):
    item, arrays, groups, _ = generated
    rows = arrays['q'][groups == 'validation-shear']
    assert len(np.unique(rows.reshape(3, -1), axis=0)) == 3
    shear = np.array([[0., .005, 0.], [0., 0., -.005], [.003, 0., 0.]])
    for q, scale in zip(rows, (.8, 1., 1.2)):
        np.testing.assert_allclose(q, item.origin+scale*(item.origin@shear), atol=1e-15, rtol=0)


@pytest.mark.parametrize('quantity', ('value', 'gradient'))
def test_joined_audit_cannot_swallow_nonfinite_evidence(quantity):
    masks = {'train': np.ones(2, dtype=bool)}
    result = {'train': {'loss': 0., 'gradient': np.zeros(3)}}
    value = np.nan if quantity == 'value' else 0.
    gradient = np.full(3, np.nan) if quantity == 'gradient' else np.zeros(3)
    def evaluate(*args):
        return value, gradient
    with pytest.raises(ValueError, match='nonfinite'):
        example.joined_error(result, evaluate, np.ones(3), {}, masks)


def test_derivative_audit_rejects_nonfinite_reference(monkeypatch):
    arrays = {'q': np.zeros((1, 2, 3)), 'states': np.ones((1, 2, 2)),
              'carrier_gradient': np.zeros((1, 2, 2, 3)), 'reference_force': np.zeros((1, 2, 3))}
    expected = {'carrier_gradient': np.zeros((2, 2, 3)), 'reference_force': np.full((2, 3), np.nan)}
    monkeypatch.setattr(example, 'independent', lambda *args: expected)
    with pytest.raises(ValueError, match='nonfinite'):
        example.derivative_errors(None, np.ones(3), arrays)


def test_oracle_comparison_cannot_accept_broadcast_shapes():
    with pytest.raises(ValueError, match='shape mismatch'):
        example.checked_error('action', np.ones((2, 3)), np.ones((1, 3)))
