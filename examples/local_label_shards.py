"""Provider-owned local label shards for two generated numerical profiles.

Run ``python -m examples.local_label_shards --output NEW_DIRECTORY``.
No fitting, physical teacher or material-accuracy claim. The reader and loss are
example-owned; neither is a universal dataset or training API. Atomic units,
fixed effective orthonormal basis, and explicit reference-plus-carrier energy.
"""
import argparse
from dataclasses import asdict, dataclass, replace
import hashlib
import io
import inspect
import json
from pathlib import Path
import platform
import zipfile

import jax
import jax.numpy as jnp
import numpy as np
import scipy

import pyeph
from pyeph import CoupledClassical, Ehrenfest, Problem
from pyeph.core.contracts import pure_state_weight
from pyeph.io.checkpoint import array_fingerprint
from pyeph.learning import bundle_identity, grouped_split, load_bundle, save_bundle
from pyeph.learning.ingestion import validate_conversion
from pyeph.models.composite import SumModel
from examples import oriented_fragments, perovskite


TARGETS = ('onsite', 'hopping', 'action', 'carrier_gradient', 'reference_energy', 'reference_force')
KEYS = {'q', 'states', *TARGETS}
SCHEMA = 'pyeph.example.local_label_shards.v1'
GROUPS = ('train-pose', 'train-internal', 'validation-shear', 'test-mixed')


def digest(data):
    return hashlib.sha256(data).hexdigest()


def json_data(value):
    if isinstance(value, dict):
        return {k: json_data(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_data(v) for v in value]
    if isinstance(value, (np.ndarray, jax.Array)):
        return np.asarray(value).tolist()
    return value


def write_json(path, value):
    path.write_text(json.dumps(json_data(value), indent=2, allow_nan=False)+'\n')


def finite_array(label, value):
    array = np.asarray(value)
    if array.dtype.kind not in 'buifc' or not np.isfinite(array).all():
        raise ValueError(f'nonfinite or nonnumeric {label}')
    return array


def checked_error(label, actual, expected):
    """Host audit: reject nonfinite evidence before reductions can hide it."""
    actual = finite_array(label+' actual', actual)
    expected = finite_array(label+' expected', expected)
    if actual.shape != expected.shape or actual.size == 0:
        raise ValueError(f'shape mismatch or empty {label}')
    error = float(np.max(np.abs(actual-expected)))
    if not np.isfinite(error):
        raise ValueError(f'nonfinite error for {label}')
    return error


@dataclass(frozen=True)
class Profile:
    name: str
    problem: object
    origin: object
    ingestion: object = None

    @property
    def carrier(self):
        return self.problem.model.models[0]

    def parameters(self, theta, xp=jnp):
        carrier, reference = ({k: xp.asarray(v) for k, v in p.items()}
                              for p in self.problem.params)
        carrier['onsite'] = theta[1]*carrier['onsite']
        if self.name == 'molecular':
            carrier['deformation'] = theta[1]*carrier['deformation']
            for key in ('pp_sigma', 'pp_pi'):
                carrier[key] = theta[0]*carrier[key]
        else:
            carrier['hopping'] = theta[0]*carrier['hopping']
        reference['spring'] = theta[2]*reference['spring']
        return carrier, reference


def profile(name):
    """Two known providers; no model class is reconstructed from artifact data."""
    if name == 'molecular':
        problem, initial = oriented_fragments.fixture()
        return Profile(name, problem, np.asarray(initial.q))
    if name != 'periodic':
        raise ValueError('choose molecular or periodic')
    carrier, params, q, masses, ingestion = perovskite.build_cspbi3(
        spinful=False, with_provenance=True)
    reference = perovskite.HarmonicReference(replace(carrier.spec, name='illustrative_tethers'))
    model = SumModel((carrier, reference), additive_probes=carrier.spec.probes)
    neutral = {'equilibrium': q, 'spring': jnp.full(q.shape, .002)}
    return Profile(name, Problem(model, (params, neutral), CoupledClassical(masses), Ehrenfest()),
                   np.asarray(q), ingestion)


def contract(item):
    carrier = item.carrier
    if item.ingestion is not None:
        converted = {**{key: np.asarray(value) for key, value in item.problem.params[0].items()},
                     "q": np.asarray(item.problem.params[1]["equilibrium"]),
                     "masses": np.asarray(item.problem.nuclear_treatment.masses)}
        configuration = dict(graph={**asdict(carrier.graph),
                                    "cell": np.asarray(carrier.graph.cell).tolist()},
                             centers=asdict(carrier.centers), basis_id=carrier.spec.system.basis_id,
                             q_shape=list(item.origin.shape), nstates=carrier.nstates)
        validate_conversion(item.ingestion, converted, configuration=configuration)
    return json_data(dict(profile=item.name, ingestion=item.ingestion, basis_id=carrier.spec.system.basis_id,
        basis_kind='fixed_effective_orthonormal', units={'energy': 'hartree', 'length': 'bohr'},
        carrier='hole' if carrier.charge > 0 else 'electron', charge=carrier.charge,
        graph=asdict(carrier.graph), centers=asdict(carrier.centers),
        provider=type(carrier.coefficient_provider).__name__,
        provider_configuration=asdict(carrier.coefficient_provider),
        site_order=list(range(carrier.graph.nsites)), atom_order=list(range(carrier.centers.natoms)),
        code_hashes={Path(path).name: digest(Path(path).read_bytes()) for path in
            (__file__, oriented_fragments.__file__, perovskite.__file__,
             inspect.getfile(type(carrier)), inspect.getfile(type(carrier.coefficient_provider)))},
        orbital_order=['effective-axial'] if item.name == 'molecular' else ['s', 'px', 'py', 'pz'],
        phase_convention='fixed ordered anchor phases' if item.name == 'molecular'
                         else 'fixed global Cartesian axes and provider orbital phases',
        baseline_sha256=array_fingerprint(item.problem.params[0]),
        reference_sha256=array_fingerprint(item.problem.params[1]),
        reference=type(item.problem.model.models[1]).__name__,
        blocks='physical blocks after final support once; full canonical image keys retain order',
        derivatives='positive d(c_dagger_H_c)/dq at each fixed saved c; negative reference force',
        adequacy='two state contractions are not the full matrix derivative'))


def independent(item, theta, q, states):
    """Reuse shipped NumPy equations, never production gradients or steps."""
    params = item.parameters(theta, np)
    if item.name == 'molecular':
        problem = replace(item.problem, params=params)
        h, energy, gradient, _ = oriented_fragments.numpy_quantities(problem, q)
        carrier_gradient = np.empty((len(states), *q.shape))
        width = 1e-4
        for index in np.ndindex(q.shape):
            shift = np.zeros_like(q)
            shift[index] = width
            plus = oriented_fragments.numpy_quantities(problem, q+shift)[0]
            minus = oriented_fragments.numpy_quantities(problem, q-shift)[0]
            plus2 = oriented_fragments.numpy_quantities(problem, q+2*shift)[0]
            minus2 = oriented_fragments.numpy_quantities(problem, q-2*shift)[0]
            derivative = (-plus2+8*plus-8*minus+minus2)/(12*width)
            carrier_gradient[(slice(None), *index)] = np.einsum(
                'ki,ij,kj->k', states.conj(), derivative, states).real
    else:
        h = perovskite.numpy_quantities(item.carrier, params[0], q)[0]
        carrier_gradient = np.array([perovskite.numpy_quantities(item.carrier, params[0], q, c)[1]
                                     for c in states])
        displacement = q-params[1]['equilibrium']
        energy = .5*np.sum(params[1]['spring']*displacement**2)
        gradient = params[1]['spring']*displacement
    return dict(action=(h@states.T).T, carrier_gradient=carrier_gradient,
                reference_energy=energy, reference_force=-gradient)


def predict(item, theta, q, states):
    params, reference = item.parameters(theta)
    carrier, nuclear = item.problem.model.models
    onsite, hopping = carrier.coefficients(params, q)
    return dict(onsite=onsite, hopping=hopping,
        action=jax.vmap(lambda c: carrier.apply(params, q, c))(states),
        carrier_gradient=jax.vmap(lambda c: carrier.contract_gradient(params, q, pure_state_weight(c)))(states),
        reference_energy=nuclear.reference_energy(reference, q),
        reference_force=-nuclear.reference_gradient(reference, q))


def generated_data(item):
    """Explicit generated groups, not evidence of physical out-of-domain transfer."""
    rng = np.random.default_rng(881 if item.name == 'molecular' else 882)
    groups = np.repeat(GROUPS, 3)
    positions = []
    mapping = np.asarray(item.carrier.centers.atom_site)
    for row, group in enumerate(groups):
        q = item.origin.copy()
        if group == 'train-pose':
            q += .025*rng.normal(size=(item.carrier.graph.nsites, 3))[mapping]
        elif group == 'train-internal':
            q += .02*rng.normal(size=q.shape)
        elif group == 'validation-shear':
            # Three distinct held-out shears: q = q0 + scale*q0@S.
            scale = .8+.2*(row-6)
            q += scale*(q@np.array([[0., .005, 0.], [0., 0., -.005], [.003, 0., 0.]]))
        else:
            q += .02*np.sin(np.arange(q.size).reshape(q.shape)+row)+.01*rng.normal(size=q.shape)
        positions.append(q)
    states = rng.normal(size=(12, 2, item.carrier.nstates))+1j*rng.normal(size=(12, 2, item.carrier.nstates))
    states /= np.linalg.norm(states, axis=-1, keepdims=True)
    arrays = {key: [] for key in KEYS}
    oracle_error = 0.
    for q, vectors in zip(positions, states):
        # Block values are generated by the known provider. Actions and forces
        # use independent NumPy equations to check the resulting actions/forces.
        # This is not an independent label for every individual image block.
        native = predict(item, jnp.ones(3), jnp.asarray(q), jnp.asarray(vectors))
        oracle = independent(item, np.ones(3), q, vectors)
        for key, value in native.items():
            finite_array('native '+key, value)
        for key, value in oracle.items():
            finite_array('oracle '+key, value)
            oracle_error = max(oracle_error, checked_error(key, native[key], value))
        row = dict(q=q, states=vectors, onsite=native['onsite'], hopping=native['hopping'], **oracle)
        for key, value in row.items():
            if not np.isfinite(np.asarray(value)).all():
                raise ValueError(f'nonfinite generated target: {key}')
            arrays[key].append(value)
    return {key: np.asarray(value) for key, value in arrays.items()}, groups, oracle_error


def write_dataset(directory, item, arrays, groups):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    expected = contract(item)
    dataset_id = array_fingerprint(arrays)
    index = dict(schema=SCHEMA, contract=expected, dataset_id=dataset_id,
                 family_rules=dict(zip(GROUPS, ('coherent fragment/site translations',
                    'independent atom displacements',
                    'q=q0+scale*q0@S; scale=0.8,1.0,1.2; S=[[0,.005,0],[0,0,-.005],[.003,0,0]]',
                    'trigonometric and random displacements'))),
                 validation_groups=[GROUPS[2]], test_groups=[GROUPS[3]], shards=[])
    order = np.array([0, 6, 9, 3, 1, 7, 10, 4, 2, 8, 11, 5])
    for number, rows in enumerate(np.split(order, (3, 8))):
        name = f'shard-{number}.npz'
        np.savez(directory/name, **{key: value[rows] for key, value in arrays.items()})
        index['shards'].append(dict(file=name, sha256=digest((directory/name).read_bytes()),
            dataset_id=dataset_id, contract_id=bundle_identity(expected),
            ids=[f'{item.name}-{row:02d}' for row in rows], groups=groups[rows].tolist()))
    write_json(directory/'index.json', index)
    return digest((directory/'index.json').read_bytes())


def read_index(directory, item, expected_sha256):
    payload = (Path(directory)/'index.json').read_bytes()
    if digest(payload) != expected_sha256:
        raise ValueError('index checksum mismatch')
    index = json.loads(payload)
    if index['schema'] != SCHEMA or index['contract'] != contract(item):
        raise ValueError('provider/basis/gauge/baseline/reference contract mismatch')
    ids = [value for shard in index['shards'] for value in shard['ids']]
    groups = np.array([value for shard in index['shards'] for value in shard['groups']])
    if len(ids) != len(set(ids)) or any(not value for value in ids):
        raise ValueError('duplicate or empty geometry IDs')
    splits = grouped_split(groups, validation_groups=index['validation_groups'],
                           test_groups=index['test_groups'])
    return index, {name: np.isin(np.arange(len(ids)), rows) for name, rows in splits.items()}


def read_shards(directory, item, index):
    """Hash/decode one payload snapshot; a host metadata index stays resident."""
    offset, seen = 0, set()
    n, b, e, a = (item.carrier.graph.nsites, item.carrier.graph.norbitals,
                  len(item.carrier.graph.edges), item.carrier.centers.natoms)
    for shard in index['shards']:
        if (shard['dataset_id'] != index['dataset_id']
                or shard['contract_id'] != bundle_identity(contract(item))):
            raise ValueError('mixed shard identity')
        name = shard['file']
        path = Path(directory)/name
        if Path(name).name != name or path.is_symlink() or name in seen:
            raise ValueError('unsafe or repeated shard path')
        seen.add(name)
        payload = path.read_bytes()
        if digest(payload) != shard['sha256']:
            raise ValueError('stale shard checksum')
        with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
            if set(archive.files) != KEYS or len(archive.files) != len(KEYS):
                raise ValueError('missing or unknown label arrays')
            arrays = {key: archive[key] for key in archive.files}
        count = len(shard['ids'])
        shapes = dict(q=(count, a, 3), states=(count, 2, n*b), onsite=(count, n, b, b),
                      hopping=(count, e, b, b), action=(count, 2, n*b),
                      carrier_gradient=(count, 2, a, 3), reference_energy=(count,),
                      reference_force=(count, a, 3))
        if len(shard['groups']) != count or count == 0:
            raise ValueError('invalid shard row metadata')
        for key, value in arrays.items():
            if (value.shape != shapes[key] or value.dtype.kind not in ('fc' if key in ('states', 'action') else 'f')
                    or not np.isfinite(value).all()):
                raise ValueError(f'incomplete/nonfinite numerical label: {key}')
        if not np.allclose(np.linalg.norm(arrays['states'], axis=-1), 1., atol=1e-13, rtol=0):
            raise ValueError('state normalization mismatch')
        if not np.allclose(arrays['onsite'], arrays['onsite'].conj().swapaxes(-1, -2), atol=1e-13, rtol=0):
            raise ValueError('onsite labels must be Hermitian')
        yield offset, arrays
        offset += count


def make_loss(item):
    def loss(theta, arrays, mask):
        predicted = jax.vmap(lambda q, c: predict(item, theta, q, c))(arrays['q'], arrays['states'])
        # Six means per geometry, normalized by fixed 0.1-Hartree and
        # 0.1-Hartree/bohr scales; equal shard means would misweight the data.
        rows = sum(jnp.mean(jnp.abs(predicted[key]-arrays[key]).reshape(len(mask), -1)**2, axis=1)/.1**2
                   for key in TARGETS)
        return jnp.sum(rows*mask)
    return jax.jit(jax.value_and_grad(loss))


def aggregate(directory, item, index, masks, theta):
    evaluate = make_loss(item)
    sums = {name: [0., np.zeros(3), 0] for name in masks}
    for offset, arrays in read_shards(directory, item, index):
        count = len(arrays['q'])
        for name, mask in masks.items():
            selected = mask[offset:offset+count]
            value, gradient = evaluate(theta, arrays, selected)
            if not np.isfinite(value) or not np.isfinite(gradient).all():
                raise ValueError('nonfinite loss or gradient')
            sums[name][0] += float(value)
            sums[name][1] += np.asarray(gradient)
            sums[name][2] += int(np.sum(selected))
    return {name: dict(loss=value/count, gradient=gradient/count, samples=count)
            for name, (value, gradient, count) in sums.items()}


def joined_error(result, evaluate, theta, joined, masks):
    """Compare a small joined audit with the separately streamed objective."""
    difference = 0.
    for split, mask in masks.items():
        value, gradient = evaluate(theta, joined, mask)
        finite_array('joined loss', value)
        finite_array('joined gradient', gradient)
        count = mask.sum()
        difference = max(difference,
            checked_error('joined loss', np.asarray(value)/count, result[split]['loss']),
            checked_error('joined gradient', np.asarray(gradient)/count, result[split]['gradient']))
    return difference


def derivative_errors(item, theta, arrays):
    """A shape-valid label may still omit baseline, frame or reference response."""
    errors = []
    for row, (q, states) in enumerate(zip(arrays['q'], arrays['states'])):
        expected = independent(item, theta, q, states)
        errors.append(max(checked_error(key, arrays[key][row], expected[key])
                          for key in ('carrier_gradient', 'reference_force')))
    return max(errors)


def contraction_witness():
    """Two constraints do not span a real-symmetric three-state derivative."""
    rng = np.random.default_rng(37)
    states = rng.normal(size=(2, 3))+1j*rng.normal(size=(2, 3))
    states /= np.linalg.norm(states, axis=1, keepdims=True)
    basis = []
    for i in range(3):
        for j in range(i, 3):
            matrix = np.zeros((3, 3))
            matrix[i, j] = matrix[j, i] = 1.
            basis.append(matrix)
    constraints = np.array([[np.vdot(c, matrix@c).real for matrix in basis] for c in states])
    _, _, vectors = np.linalg.svd(constraints, full_matrices=True)
    derivative = np.einsum('a,aij->ij', vectors[-1], basis)
    derivative /= np.linalg.norm(derivative)
    return states, derivative


def run(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    runtime = Path(pyeph.__file__).resolve().parent
    sources = {f'pyeph/{p.relative_to(runtime)}': p for p in runtime.rglob('*.py')}
    for module in (oriented_fragments, perovskite):
        sources[f'examples/{Path(module.__file__).name}'] = Path(module.__file__)
    sources['examples/local_label_shards.py'] = Path(__file__)
    hashes = {name: digest(path.read_bytes()) for name, path in sources.items()}
    with zipfile.ZipFile(output/'sources.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for name, path in sources.items():
            archive.write(path, name)
    write_json(output/'identity.json', dict(sources=hashes, platform=platform.platform(),
        python=platform.python_version(), versions=dict(pyeph=pyeph.__version__, jax=jax.__version__, numpy=np.__version__, scipy=scipy.__version__),
        scope='generated, unfitted numerical data/loss demonstration; no material or performance claim'))
    records = {}
    theta = np.array([.91, 1.07, 1.12])
    for name in ('molecular', 'periodic'):
        item = profile(name)
        arrays, groups, error = generated_data(item)
        checksum = write_dataset(output/name, item, arrays, groups)
        index, masks = read_index(output/name, item, checksum)
        result = aggregate(output/name, item, index, masks, theta)
        # Validation only: a second pass joins this small fixture for comparison.
        joined = {key: np.concatenate([rows[key] for _, rows in read_shards(output/name, item, index)]) for key in KEYS}
        evaluate = make_loss(item)
        difference = joined_error(result, evaluate, theta, joined, masks)
        np.savez(output/f'{name}-loss-evidence.npz', theta=theta,
                 **{split+'_gradient': row['gradient'] for split, row in result.items()})
        checks = [dict(name='independent_values_and_complete_contractions', passed=error < 3e-9),
                  dict(name='unequal_shard_aggregation', passed=difference < 2e-12)]
        records[name] = dict(index_sha256=checksum, independent_max_abs=error,
                             joined_max_abs=difference, splits=result, checks=checks)
        write_json(output/'report.json', records)
        if not all(check['passed'] for check in checks):
            raise ValueError('numerical check failed; preserve generated data and report')
        expected = dict(provider=f'{name}-shard-example', provider_version='1',
            basis_id=item.carrier.spec.system.basis_id, basis_kind='fixed_effective_orthonormal',
            units={'energy': 'hartree', 'length': 'bohr'}, carrier=contract(item)['carrier'],
            neutral_reference=contract(item)['reference'], baseline_sha256=contract(item)['baseline_sha256'],
            dataset_sha256=checksum, code_hashes=hashes,
            configuration=dict(profile=contract(item), numerical_versions=dict(jax=jax.__version__, numpy=np.__version__)),
            scope='unfitted candidate for a generated numerical profile')
        manifest = save_bundle(output/f'{name}-candidate', {'theta': theta}, contract=expected,
            validation=dict(scope='numerical oracle and aggregation checks only', checks=checks))
        restored, _ = load_bundle(output/f'{name}-candidate/bundle.json', expected_contract=expected)
        np.testing.assert_array_equal(restored['theta'], theta)
        records[name]['bundle_identity'] = manifest['identity']
    states, derivative = contraction_witness()
    np.savez(output/'nonspanning-witness.npz', states=states, derivative=derivative)
    records['contraction_witness'] = dict(
        measured_max_abs=float(np.max(abs(np.einsum('ki,ij,kj->k', states.conj(), derivative, states).real))),
        unobserved_max_abs=float(np.max(abs(np.linalg.eigvalsh(derivative)))))
    if (records['contraction_witness']['measured_max_abs'] > 2e-15
            or records['contraction_witness']['unobserved_max_abs'] < .3):
        raise ValueError('derivative-adequacy witness failed')
    if {name: digest(path.read_bytes()) for name, path in sources.items()} != hashes:
        raise ValueError('source changed during the example; retain failed evidence')
    records['source_unchanged'] = True
    records['scope'] = 'The candidate is not fitted. Two force contractions do not identify the full derivative.'
    write_json(output/'report.json', records)
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    jax.config.update('jax_enable_x64', True)
    try:
        result = run(args.output)
    except Exception as exc:
        # Keep all already-created inputs/results. Do not relabel a failure pass.
        if not isinstance(exc, FileExistsError) and args.output.is_dir() and not (args.output/'failure.json').exists():
            write_json(args.output/'failure.json', dict(type=type(exc).__name__, reason=str(exc)))
        raise
    print(json.dumps(json_data(result), indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
