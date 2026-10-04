"""Physical derivative and artifact gates for the real-label example provider."""

import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pyeph.core.contracts import ProbeContext, pure_state_weight


_EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
sys.path.insert(0, str(_EXAMPLES))
try:
    _SPEC = importlib.util.spec_from_file_location("molecular_residual_example", _EXAMPLES / "molecular_residual.py")
    example = importlib.util.module_from_spec(_SPEC)
    sys.modules[_SPEC.name] = example
    _SPEC.loader.exec_module(example)
    import molecular_surrogate_dynamics as dynamics
finally:
    sys.path.pop(0)


def fixture(separation=7.):
    monomer = np.array([[0., 0., 1.2], [0., 0., -1.2],
                        [0., 1.6, 2.1], [0., -1.6, 2.1],
                        [0., 1.6, -2.1], [0., -1.6, -2.1]])
    q = jnp.asarray(np.concatenate((monomer, monomer + [separation, .2, -.1])))
    models = example.make_models("test:ordered-ethylene", hidden=8)
    bp = example.baseline_params(jnp.array([.3, .03, -.01, -.02, .006]), q, .5)
    network = models[1].coefficient_provider.network
    nn = network.init_params(jax.random.key(9), zero_last=False)
    nn = nn | dict(q_center=example.distances(q), q_scale=jnp.ones(66))
    cp = dict(baseline=bp, network=nn)
    nr = models[2].network.init_params(jax.random.key(14), zero_last=False)
    rp = dict(offset=jnp.asarray(-155.), center=example.distances(q), scale=jnp.ones(66),
              polynomial=jnp.linspace(-.004, .005, 133), network=nr)
    return models, (bp, cp, rp), q


def assert_exact_parameters(actual, expected):
    assert jax.tree.structure(actual) == jax.tree.structure(expected)
    for left, right in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
        left, right = np.asarray(left), np.asarray(right)
        assert left.shape == right.shape and left.dtype == right.dtype
        assert np.isfinite(left).all() and np.isfinite(right).all()
        assert left.tobytes() == right.tobytes()


def assert_recomputed_equal(actual, expected):
    """Declared reconstruction allowance in the computed field's own units.

    This permits last-bit changes in device reductions; it is not a general
    error bound for unspecified device transcendental implementations.
    """
    actual, expected = np.asarray(actual), np.asarray(expected)
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    assert np.isfinite(actual).all() and np.isfinite(expected).all()
    allowance = 128*np.finfo(expected.real.dtype).eps
    scale = np.max(np.abs(expected))
    np.testing.assert_allclose(actual, expected, rtol=allowance,
                               atol=allowance*scale, equal_nan=False)


@pytest.mark.parametrize("separation", [7., 12.5])
def test_complete_energy_force_finite_differences_including_cutoff(separation):
    models, (_, cp, rp), q = fixture(separation)
    model = models[-1]
    model.validate_params((cp, rp))
    c = jnp.array([.8, .36+.48j])
    weight = pure_state_weight(c)
    gradient = model.reference_gradient((cp, rp), q) + model.contract_gradient((cp, rp), q, weight)
    energy = jax.jit(lambda x: model.reference_energy((cp, rp), x)
                     + jnp.vdot(c, model.apply((cp, rp), x, c)).real)
    expected = np.empty((12, 3))
    for atom, axis in np.ndindex(12, 3):
        displacement = np.zeros((12, 3))
        displacement[atom, axis] = 2e-5
        expected[atom, axis] = (energy(q+displacement)-energy(q-displacement))/4e-5
    np.testing.assert_allclose(gradient, expected, atol=1e-8, rtol=2e-5)
    # A nonanchor hydrogen contributes through descriptors and neutral energy.
    assert np.linalg.norm(gradient[5]) > 1e-5
    np.testing.assert_allclose(gradient.sum(axis=0), 0., atol=2e-14)
    np.testing.assert_allclose(jnp.cross(q, gradient).sum(axis=0), 0., atol=2e-13)


def test_rotation_fragment_exchange_phase_gauge_and_current():
    models, (bp, cp, rp), q = fixture()
    model = models[-1]
    h = model.dense((cp, rp), q)
    rotation = jnp.array([[.8, -.6, 0.], [.6, .8, 0.], [0., 0., 1.]])
    transformed = q @ rotation.T + jnp.array([3., -2., .8])
    np.testing.assert_allclose(model.dense((cp, rp), transformed), h, atol=1e-15)
    np.testing.assert_allclose(model.reference_energy((cp, rp), transformed),
                               model.reference_energy((cp, rp), q), atol=1e-13)
    np.testing.assert_allclose(model.dense((cp, rp), example.exchanged(q)), h[::-1, ::-1], atol=1e-15)
    np.testing.assert_allclose(model.reference_energy((cp, rp), example.exchanged(q)),
                               model.reference_energy((cp, rp), q), atol=1e-13)
    phase = np.diag([-1., 1.])
    changed = example.make_models("test:ordered-ethylene", 8, phases=(-1, 1))
    np.testing.assert_allclose(changed[-1].dense((cp, rp), q), phase@h@phase, atol=1e-15)
    centers = np.asarray(q).reshape(2, 6, 3).mean(axis=1)
    for axis, probe in enumerate(model.spec.probes):
        position = np.diag(centers[:, axis])
        expected = 1j*(h@position-position@h)
        actual = model.probe_apply((cp, rp), ProbeContext(q), probe, jnp.eye(2))
        np.testing.assert_allclose(actual, expected, atol=1e-15)


def test_plain_npz_export_restores_values_forces_and_rejects_tampering(tmp_path):
    models, params, q = fixture()
    payload = tmp_path / "parameters.npz"
    schema = example.save_parameters(payload, params)
    record = dict(artifact_schema=example.ARTIFACT_SCHEMA,
                  dataset=dict(basis_id="test:ordered-ethylene"), training=dict(hidden=8),
                  implementation_hashes=example.implementation_hashes(),
                  static_configuration=example.static_configuration(models),
                  parameter_schema=schema, parameters_sha256=hashlib.sha256(payload.read_bytes()).hexdigest())
    path = tmp_path / "report.json"
    path.write_text(json.dumps(record))
    restored, actual, _ = example.load_artifact(path)
    assert_exact_parameters(actual, params)
    cp, rp = params[1:]
    assert_recomputed_equal(restored[-1].dense(actual[1:], q), models[-1].dense((cp, rp), q))
    assert_recomputed_equal(restored[-1].reference_gradient(actual[1:], q),
                            models[-1].reference_gradient((cp, rp), q))
    changed = json.loads(json.dumps(record))
    changed["static_configuration"]["graph"]["cutoff"] += 1
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="static provider"):
        example.load_artifact(path)
    path.write_text(json.dumps(record | {"implementation_hashes": {}}))
    with pytest.raises(ValueError, match="implementation differs"):
        example.load_artifact(path)
    path.write_text(json.dumps(record))
    payload.write_bytes(payload.read_bytes()+b"changed")
    with pytest.raises(ValueError, match="checksum"):
        example.load_artifact(path)


def test_derivative_supervised_ridge_does_not_use_test_targets():
    # Independent polynomial truth with a known derivative, not neural outputs.
    x = np.linspace(-1, 1, 12)
    design = np.stack((np.ones_like(x), x, x*x), axis=1)
    derivative = np.zeros((12, 3, 12, 3))
    derivative[:, :, 0, 0] = np.stack((0*x, 1+0*x, 2*x), axis=1)
    target = .2+.7*x-.15*x*x
    gradient = np.zeros((12, 12, 3))
    gradient[:, 0, 0] = .7-.3*x
    a, report = example.ridge_fit(design, derivative, target, gradient, np.arange(8), np.array([8, 9]))
    np.testing.assert_allclose(a, [.2, .7, -.15], atol=2e-10)
    target[10:] += 10000
    gradient[10:] -= 300
    b, changed = example.ridge_fit(design, derivative, target, gradient, np.arange(8), np.array([8, 9]))
    np.testing.assert_array_equal(a, b)
    assert report == changed


def test_shared_bundle_restores_provider_complete_force_and_rejects_extra_weights(tmp_path):
    from pyeph.learning import save_bundle
    models, params, q = fixture()
    dataset = dict(basis_kind="fixed_effective_orthonormal", basis_id="test:ordered-ethylene",
                   units=dict(energy="hartree", length="bohr"), carrier="hole",
                   neutral_reference="fixture potential")
    contract = example.provider_bundle_contract(models, params, dataset)
    validation = dict(scope="synthetic provider reconstruction", checks=[dict(name="fixture", passed=True)])
    arrays = example.parameter_arrays(params)
    save_bundle(tmp_path/"bundle", arrays, contract=contract, validation=validation)
    restored, actual, _ = example.load_provider_bundle(
        tmp_path/"bundle/bundle.json", expected_contract=contract)
    assert_exact_parameters(actual, params)
    electronic = jnp.array([.8, .36+.48j])
    def gradient(model, parameters):
        return (model.reference_gradient(parameters, q)
                + model.contract_gradient(parameters, q, pure_state_weight(electronic)))
    assert_recomputed_equal(gradient(models[-1], params[1:]), gradient(restored[-1], actual[1:]))
    save_bundle(tmp_path/"extra", arrays | {"unknown/weight": np.ones(1)},
                contract=contract, validation=validation)
    with pytest.raises(ValueError, match="unknown provider"):
        example.load_provider_bundle(tmp_path/"extra/bundle.json", expected_contract=contract)


def test_domain_monitor_stops_real_runner_and_preserves_observed_geometry(tmp_path):
    from pyeph import Execution, Integrator, Simulation
    from pyeph.learning import DomainViolation, GeometryDomainMonitor
    models, params, q = fixture()
    problem, initial = dynamics.fixture(models[-1], params[1:], q)
    def descriptor(x):
        return np.array([x[0, 0]])
    watch = GeometryDomainMonitor(tmp_path, descriptor, [-1e-12], [1e-12],
                                  descriptor_id="fixture:first-coordinate",
                                  metadata=dict(model="synthetic-regression-fixture"),
                                  trajectory_ids=[71])
    simulation = Simulation(problem, Integrator(.5, "exponential_midpoint"),
                            Execution(chunk_size=1, save_every=1))
    with pytest.raises(DomainViolation) as caught:
        simulation.run(initial, 4, observer=watch)
    with np.load(caught.value.record_path.parent/"geometry.npz", allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved["times"], [.5])
        np.testing.assert_array_equal(saved["trajectory_ids"], [71])
        assert abs(saved["q"][0, 0, 0]) > 1e-12


def test_independent_numpy_surrogate_matrix_current_and_force():
    models, (_, cp, rp), q = fixture(12.5)
    problem, initial = dynamics.fixture(models[-1], (cp, rp), q)
    h, relative, current = dynamics.numpy_values(problem, q)
    np.testing.assert_allclose(h, models[-1].dense((cp, rp), q), atol=2e-15)
    np.testing.assert_allclose(relative+rp["offset"], models[-1].reference_energy((cp, rp), q), atol=3e-14)
    for axis, probe in enumerate(problem.model.spec.probes):
        np.testing.assert_allclose(current[axis], problem.model.probe_apply(
            problem.params, ProbeContext(q), probe, jnp.eye(2)), atol=2e-15)
    force = (-problem.model.reference_gradient(problem.params, q)
             - problem.model.contract_gradient(problem.params, q, pure_state_weight(initial.electronic)))
    np.testing.assert_allclose(dynamics.numpy_force(problem, q, initial.electronic), force, atol=2e-9)


@pytest.mark.parametrize("method", ["cpa", "ehrenfest"])
def test_surrogate_dynamics_refines_restarts_and_batches(tmp_path, method):
    models, (_, cp, rp), q = fixture()
    problem, initial = dynamics.fixture(models[-1], (cp, rp), q, method=method)
    report = dynamics.run_case(problem, initial, tmp_path, steps=4, dt=.5,
                               artifact_metadata=dict(model="synthetic-regression-fixture"))
    assert report["checkpoint_roundtrip_bitwise"]
    assert report["restart_numerically_qualified"]
    assert isinstance(report["restart_bitwise"], bool)
    assert report["errors"]["electronic"][1] < 1e-5


def test_restart_qualification_separates_storage_bytes_from_propagated_roundoff():
    from pyeph import make_state
    initial = make_state([1.], [.2], [1., 0.])
    changed = initial._replace(q=jnp.array([np.nextafter(1., 2.)]))
    report = dynamics.compare_restart_states(changed, initial)
    assert report["numerically_qualified"] and not report["bytewise_equal"]
    assert report["fields"][".q"]["max_absolute_error"] > 0
    with pytest.raises(AssertionError, match="bytes differ"):
        dynamics.compare_restart_states(changed, initial, exact=True)
    assert dynamics.compare_restart_states(initial, initial, exact=True)["bytewise_equal"]


@pytest.mark.parametrize("failure", ["tree", "shape", "dtype", "nonfinite", "step", "key", "float"])
def test_restart_qualification_rejects_invalid_state_evidence(failure):
    from pyeph import make_state
    initial = make_state([1.], [.2], [1., 0.])
    changed = {
        "tree": initial._replace(method_state={"extra": jnp.array(1)}),
        "shape": initial._replace(q=jnp.array([[1.]])),
        "dtype": initial._replace(q=initial.q.astype(jnp.float32)),
        "nonfinite": initial._replace(q=jnp.array([jnp.nan])),
        "step": initial._replace(step=initial.step+1),
        "key": initial._replace(key=initial.key.at[0].add(jnp.uint32(1))),
        "float": initial._replace(q=initial.q+1e-8),
    }[failure]
    with pytest.raises(AssertionError):
        dynamics.compare_restart_states(changed, initial)


def test_recomputed_force_allowance_does_not_weaken_parameter_or_field_checks():
    original = np.array([.01, -.02])
    rounded = np.nextafter(original, np.inf)
    assert_recomputed_equal(rounded, original)
    with pytest.raises(AssertionError):
        assert_exact_parameters({"weights": rounded}, {"weights": original})
    for bad in (original+1e-9, original.astype(np.float32), original[None],
                np.array([np.nan, -.02])):
        with pytest.raises(AssertionError):
            assert_recomputed_equal(bad, original)


@pytest.mark.parametrize("byte_ids", [False, True])
def test_cli_report_normalizes_geometry_ids_without_changing_label_identity(tmp_path, monkeypatch, byte_ids):
    """Exercise report/export and dynamics selection; numerical fitting/run is stubbed."""
    from pyeph.learning import bundle_identity, load_labels, save_bundle

    models, params, q = fixture()
    ids = (np.array([b"train-a", b"validation-b", b"test-c"]) if byte_ids else
           np.array(["train-alpha", "validation-β", "test-gamma"]))
    groups = np.array(["train", "validation", "test"], dtype="S" if byte_ids else "U")
    arrays = dict(q=np.repeat(np.asarray(q)[None], 3, axis=0), h_hole=np.zeros((3, 2, 2)),
                  electronic_gradient=np.zeros((3, 2, 2, 12, 3)),
                  neutral_energy=np.zeros(3), neutral_force=np.zeros((3, 12, 3)),
                  species=np.array([6, 6, 1, 1, 1, 1]*2), fragment=np.repeat([0, 1], 6),
                  geometry_ids=ids, groups=groups)
    # Distinct coordinates ensure the dynamics path chooses the held-out row.
    arrays["q"][1, :, 0] += .1
    arrays["q"][2, :, 0] += .2
    payload = tmp_path/"labels.npz"
    np.savez(payload, **arrays)
    payload_before = payload.read_bytes()
    metadata = dict(schema="pyeph.fixed_basis_labels.v1", basis_kind="fixed_effective_orthonormal",
                    basis_id="test:ordered-ethylene", carrier="hole",
                    units=dict(energy="hartree", length="bohr"),
                    electronic_energy_definition="synthetic report fixture", phase_convention="fixed",
                    neutral_reference="synthetic reference", label_scope="report serialization only",
                    sources=["independent generated reporting fixture"], arrays_file=payload.name,
                    arrays_sha256=hashlib.sha256(payload_before).hexdigest())
    manifest = tmp_path/"labels.json"
    manifest.write_text(json.dumps(metadata))
    manifest_before = manifest.read_bytes()
    loaded, checked_metadata = load_labels(manifest)
    assert loaded["geometry_ids"].dtype == ids.dtype
    assert loaded["geometry_ids"].tobytes() == ids.tobytes()
    contract = example.provider_bundle_contract(models, params, checked_metadata)
    existing = save_bundle(tmp_path/"existing-bundle", example.parameter_arrays(params),
                           contract=contract, validation=dict(scope="report fixture only",
                           checks=[dict(name="synthetic parameter fixture", passed=True)]))
    old_bundle = tmp_path/"existing-bundle/bundle.json"
    old_bytes = old_bundle.read_bytes()

    monkeypatch.setattr(example, "fit", lambda *args, **kwargs: (models, params, dict(hidden=8)))
    predictions = dict(h=arrays["h_hole"], dh=arrays["electronic_gradient"],
                       neutral_energy=arrays["neutral_energy"], neutral_force=arrays["neutral_force"])
    monkeypatch.setattr(example, "evaluate", lambda *args: (predictions, {}))
    output = tmp_path/"fit-report"
    monkeypatch.setattr(sys, "argv", ["molecular_residual.py", str(manifest), "--output", str(output),
                                      "--validation-groups", "validation", "--test-groups", "test"])
    example.main()

    report = json.loads((output/"report.json").read_text())
    expected_ids = ids.astype(str)
    assert report["splits"] == {name: [expected_ids[i]]
                                for i, name in enumerate(("train", "validation", "test"))}
    bundle = json.loads((output/"provider_bundle/bundle.json").read_text())
    assert bundle["contract"]["dataset_sha256"] == bundle_identity(checked_metadata)
    for name, values in report["splits"].items():
        assert values == bundle["validation"]["label_report"]["splits"][name]["geometry_ids"]
    selected = []

    def record_selection(problem, initial, output, **kwargs):
        selected.append(np.asarray(initial.q))
        assert kwargs["artifact_metadata"]["dataset"] == metadata["arrays_sha256"]
        return dict(method=type(problem.method).__name__.lower())

    monkeypatch.setattr(dynamics, "run_case", record_selection)
    dynamics_output = tmp_path/"dynamics-report"
    monkeypatch.setattr(sys, "argv", ["molecular_surrogate_dynamics.py", str(output/"report.json"),
                                      str(manifest), "--output", str(dynamics_output)])
    dynamics.main()
    dynamics_report = json.loads((dynamics_output/"report.json").read_text())
    assert dynamics_report["initial_geometry_id"] == expected_ids[2]
    assert [case["method"] for case in dynamics_report["cases"]] == ["cpa", "ehrenfest"]
    assert len(selected) == 2
    for q_selected in selected:
        np.testing.assert_array_equal(q_selected, arrays["q"][2])
    assert payload.read_bytes() == payload_before
    assert manifest.read_bytes() == manifest_before
    reloaded, _ = load_labels(manifest)
    assert reloaded["geometry_ids"].dtype == ids.dtype
    assert reloaded["geometry_ids"].tobytes() == ids.tobytes()
    assert old_bundle.read_bytes() == old_bytes
    assert json.loads(old_bytes)["identity"] == existing["identity"]


def test_legacy_artifact_loads_the_checksum_verified_snapshot(tmp_path, monkeypatch):
    models, params, _ = fixture()
    payload = tmp_path / "parameters.npz"
    schema = example.save_parameters(payload, params)
    original_bytes = payload.read_bytes()
    original_digest = hashlib.sha256(original_bytes).hexdigest()
    changed = (params[0], params[1], params[2] | dict(offset=params[2]["offset"] + 1.))
    replacement = tmp_path / "replacement.npz"
    example.save_parameters(replacement, changed)
    replacement_bytes = replacement.read_bytes()
    assert hashlib.sha256(replacement_bytes).hexdigest() != original_digest
    record = dict(artifact_schema=example.ARTIFACT_SCHEMA,
                  dataset=dict(basis_id="test:ordered-ethylene"), training=dict(hidden=8),
                  implementation_hashes=example.implementation_hashes(),
                  static_configuration=example.static_configuration(models),
                  parameter_schema=schema, parameters_sha256=original_digest)
    report = tmp_path / "report.json"
    report.write_text(json.dumps(record))
    read_bytes = Path.read_bytes
    checked_snapshots = []

    def replace_after_read(path):
        data = read_bytes(path)
        if path == payload:
            checked_snapshots.append(hashlib.sha256(data).hexdigest())
            payload.write_bytes(replacement_bytes)
        return data

    monkeypatch.setattr(Path, "read_bytes", replace_after_read)
    _, restored, loaded_report = example.load_artifact(report)
    assert checked_snapshots == [original_digest]
    assert read_bytes(payload) == replacement_bytes
    assert loaded_report["parameters_sha256"] == original_digest
    assert_exact_parameters(restored, params)
    # A later call must reject the replacement against the unchanged manifest.
    with pytest.raises(ValueError, match="parameter checksum mismatch"):
        example.load_artifact(report)
