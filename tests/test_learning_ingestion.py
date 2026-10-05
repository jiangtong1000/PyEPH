"""Real label and periodic ingress preserve conversions and converted identities."""

from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from pyeph.core import units
from pyeph.learning import bundle_identity, import_labels, load_bundle, load_labels, save_bundle


def source(tmp_path):
    path = tmp_path/'source'
    path.mkdir()
    q = np.array([[[.2, -.1, .3]], [[-.3, .25, .1]], [[.45, .2, -.2]]])
    x = q[:, 0, 0]
    h = np.zeros((3, 2, 2))
    h[:, 0, 0], h[:, 1, 1] = .2+.13*x, -.1+.06*x*x
    h[:, 0, 1] = h[:, 1, 0] = .07+.03*x
    derivative = np.zeros((3, 2, 2, 1, 3))
    derivative[:, 0, 0, 0, 0] = .13
    derivative[:, 1, 1, 0, 0] = .12*x
    derivative[:, 0, 1, 0, 0] = derivative[:, 1, 0, 0, 0] = .03
    arrays = dict(q=q, h_electron=h, electronic_gradient=derivative,
        neutral_energy=.2*np.sum(q*q, axis=(1, 2)), neutral_force=-.4*q,
        species=np.array([1]), fragment=np.array([0]), geometry_ids=np.array(['a', 'b', 'c']),
        groups=np.array(['train', 'validation', 'test']))
    np.savez(path/'raw-input.npz', **arrays)
    metadata = dict(schema='pyeph.fixed_basis_labels.v1', units=dict(energy='eV', length='angstrom'),
        basis_kind='fixed_effective_orthonormal', basis_id='generated:two-states',
        electronic_energy_definition='generated electron Hamiltonian', phase_convention='fixed real',
        neutral_reference='generated quadratic reference', label_scope='unit conversion test',
        sources=['independent scalar equations'], carrier='electron', arrays_file='raw-input.npz',
        arrays_sha256=hashlib.sha256((path/'raw-input.npz').read_bytes()).hexdigest())
    (path/'source.json').write_text(json.dumps(metadata))
    return path/'source.json', arrays


def contract(metadata):
    return dict(provider='test.converted', provider_version='1', code_hashes=metadata['ingestion']['importer_identity'],
        baseline_sha256=metadata['arrays_sha256'], dataset_sha256=bundle_identity(metadata),
        basis_kind=metadata['basis_kind'], basis_id=metadata['basis_id'], units=metadata['units'],
        carrier='electron', neutral_reference=metadata['neutral_reference'], scope='generated test',
        configuration=dict(ingestion=metadata['ingestion']['identity'], static=metadata['static_configuration']))


def test_saved_replay_and_explicit_raw_reconversion_ignore_changed_constants(tmp_path, monkeypatch):
    path, raw = source(tmp_path)
    static = dict(cell=np.diag([8., 9., 10.]).tolist(), cutoff=3., switch_on=2.)
    first = import_labels(path, tmp_path/'first', static_configuration=static)
    values, _ = load_labels(tmp_path/'first/labels.json')
    factors = {k: first['ingestion']['factors'][k]['value'] for k in ('energy_to_hartree', 'length_to_bohr')}
    saved = save_bundle(tmp_path/'bundle', {'h': values['h_electron']}, contract=contract(first),
        validation=dict(scope='fixture', checks=[dict(name='independent formula', passed=True)]))
    monkeypatch.setattr(units, 'BOHR_ANGSTROM', units.BOHR_ANGSTROM*1.000001)
    monkeypatch.setattr(units, 'HARTREE_EV', units.HARTREE_EV*1.000001)
    replay, _ = load_labels(tmp_path/'first/labels.json')
    for key in values:
        assert replay[key].dtype == values[key].dtype
        assert replay[key].tobytes() == values[key].tobytes()
    arrays, loaded = load_bundle(tmp_path/'bundle/bundle.json', expected_contract=contract(first))
    assert loaded == saved and arrays['h'].tobytes() == values['h_electron'].tobytes()
    repeated = import_labels(tmp_path/'first/raw/source.json', tmp_path/'repeated',
                             static_configuration=static, conversion_factors=factors)
    repeated_arrays, _ = load_labels(tmp_path/'repeated/labels.json')
    for key in values:
        assert repeated_arrays[key].tobytes() == values[key].tobytes()
    assert repeated['ingestion']['converted_configuration'] == first['static_configuration']
    second = import_labels(path, tmp_path/'second', static_configuration=static)
    assert second['arrays_sha256'] != first['arrays_sha256']
    assert second['ingestion']['identity'] != first['ingestion']['identity']
    with pytest.raises(ValueError, match='contract mismatch'):
        load_bundle(tmp_path/'bundle/bundle.json', expected_contract=contract(second))
    with pytest.raises(FileExistsError):
        import_labels(path, tmp_path/'first')
    assert (tmp_path/'first/raw/raw-input.npz').read_bytes() == path.with_name('raw-input.npz').read_bytes()
    np.testing.assert_array_equal(raw['q'], np.load(tmp_path/'first/raw/raw-input.npz')['q'])


def test_complete_derivative_chain_rule_against_independent_scalar_finite_difference(tmp_path):
    path, raw = source(tmp_path)
    metadata = import_labels(path, tmp_path/'converted')
    arrays, _ = load_labels(tmp_path/'converted/labels.json')
    fE = metadata['ingestion']['factors']['energy_to_hartree']['value']
    fL = metadata['ingestion']['factors']['length_to_bohr']['value']
    x = arrays['q'][:, 0, 0]
    def h(q):
        r = q/fL
        return np.array([[[.2+.13*v, .07+.03*v], [.07+.03*v, -.1+.06*v*v]] for v in r])*fE
    width = 1e-4
    fd = (h(x+width)-h(x-width))/(2*width)
    np.testing.assert_allclose(arrays['electronic_gradient'][..., 0, 0], fd, atol=2e-14, rtol=0)
    np.testing.assert_allclose(arrays['neutral_force'], -.4*arrays['q']*fE/fL**2, atol=2e-18, rtol=0)
    wrong = raw['electronic_gradient']*fE*fL
    assert np.max(abs(wrong-arrays['electronic_gradient'])) > 1e-3
    np.savez(tmp_path/'converted/labels.npz', **{**arrays, 'electronic_gradient': wrong})
    metadata['arrays_sha256'] = hashlib.sha256((tmp_path/'converted/labels.npz').read_bytes()).hexdigest()
    (tmp_path/'converted/labels.json').write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match='converted array identity'):
        load_labels(tmp_path/'converted/labels.json')


@pytest.mark.parametrize('field', ['cell', 'cutoff', 'q_shape', 'basis_id', 'factor'])
def test_conversion_static_and_factor_tampering_rejects_before_model_use(tmp_path, field):
    path, _ = source(tmp_path)
    meta = import_labels(path, tmp_path/'converted', static_configuration=dict(cell=np.eye(3).tolist(), cutoff=3.))
    if field == 'factor':
        meta['ingestion']['factors']['length_to_bohr']['hex'] = float(1.).hex()
        meta['ingestion']['identity'] = bundle_identity({k: v for k, v in meta['ingestion'].items() if k != 'identity'})
    else:
        meta['static_configuration'][field] = {'cell': np.eye(3).tolist(), 'cutoff': 4.,
                                              'q_shape': [2, 3], 'basis_id': 'other'}[field]
        if field == 'cell':
            meta['static_configuration']['cell'][0][0] += .1
    (tmp_path/'converted/labels.json').write_text(json.dumps(meta))
    with pytest.raises(ValueError, match='conversion|converted'):
        load_labels(tmp_path/'converted/labels.json')


@pytest.mark.parametrize('static,factor', [({'cutoff': 1e-200}, 1e-200),
    ({'cell': np.eye(3).tolist()}, 1e-200),
    ({'cutoff': 1.0000000000000002, 'switch_on': 1.}, 5e-324)])
def test_static_lengths_cannot_collapse_during_conversion(tmp_path, static, factor):
    path, _ = source(tmp_path)
    with pytest.raises(ValueError, match='converted|factor'):
        import_labels(path, tmp_path/'bad', static_configuration=static,
                      conversion_factors=dict(energy_to_hartree=factor, length_to_bohr=factor))


def test_periodic_profile_binds_actual_cell_cutoff_and_conversion_constants(monkeypatch):
    from examples import local_label_shards as example
    first = example.profile('periodic')
    before = example.contract(first)
    changed = deepcopy(first.ingestion)
    changed['converted_configuration']['graph']['cell'][0][0] += .1
    bad = replace(first, ingestion=changed)
    with pytest.raises(ValueError, match='checksum'):
        example.contract(bad)
    carrier = first.carrier
    graph = replace(carrier.graph, cutoff=carrier.graph.cutoff+.1)
    model = replace(first.problem.model, models=(replace(carrier, graph=graph), first.problem.model.models[1]))
    with pytest.raises(ValueError, match='static configuration'):
        example.contract(replace(first, problem=replace(first.problem, model=model)))
    monkeypatch.setattr(example.perovskite, 'BOHR_ANGSTROM', example.perovskite.BOHR_ANGSTROM*1.000001)
    second = example.contract(example.profile('periodic'))
    assert before['ingestion']['identity'] != second['ingestion']['identity']
    assert before['graph']['cell'] != second['graph']['cell']


def test_periodic_provenance_hashes_actual_float32_returned_arrays():
    root = Path(__file__).resolve().parents[1]
    script = '''
import hashlib,json
import jax
jax.config.update('jax_enable_x64', False)
import numpy as np
from examples.perovskite import build_cspbi3
from pyeph.learning.ingestion import validate_conversion
model, params, q, masses, evidence = build_cspbi3(spinful=np.bool_(False), with_provenance=True)
arrays = {**{key: np.asarray(value) for key, value in params.items()}, 'q': np.asarray(q), 'masses': np.asarray(masses)}
validate_conversion(evidence, arrays, configuration=evidence['converted_configuration'])
assert arrays['q'].dtype == np.float32
assert evidence['converted_arrays']['q']['sha256'] == hashlib.sha256(arrays['q'].tobytes()).hexdigest()
print(json.dumps({'validated': True}))
'''
    result = subprocess.run([sys.executable, '-c', script], cwd=root, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout+result.stderr
