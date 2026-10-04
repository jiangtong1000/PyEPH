from copy import deepcopy
from dataclasses import dataclass, replace
import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pyeph.core.problem import CoupledClassical, PrescribedPath, Problem
from pyeph.core.state import make_state
from pyeph.core.units import UnitSystem
from pyeph.dynamics.cpa import CPA
from pyeph.dynamics.ehrenfest import Ehrenfest
from pyeph.integrators.electronic import Integrator
from pyeph.io.checkpoint import load_checkpoint, save_checkpoint
from pyeph.io.provenance import (
    assert_matching_manifest,
    problem_manifest,
    recorded_path_manifest,
    validate_manifest,
)
from pyeph.models.analytic import SpinBosonModel
from pyeph.models.composite import ReferenceShiftModel
from pyeph.models.periodic import PeriodicBlockModel
from pyeph.paths.electronic import AdiabaticElectronicPath, FixedBasisElectronicPath
from pyeph.paths.harmonic import ConstantPath


def test_manifest_records_random_options_that_change_seeded_samples():
    option = "jax_threefry_partitionable"
    if not hasattr(jax.config, option):
        pytest.skip("this JAX version does not expose the partitionable PRNG option")
    original = getattr(jax.config, option)
    try:
        jax.config.update(option, False)
        before = problem_manifest(_problem(), Integrator(.1))
        sample_before = jax.random.normal(jax.random.PRNGKey(723), (17,))
        jax.config.update(option, True)
        after = problem_manifest(_problem(), Integrator(.1))
        sample_after = jax.random.normal(jax.random.PRNGKey(723), (17,))
        assert not np.array_equal(sample_before, sample_after)
        with pytest.raises(ValueError, match="runtime"):
            assert_matching_manifest(before, after)
    finally:
        jax.config.update(option, original)


def _problem():
    model = SpinBosonModel()
    return Problem(model, model.default_params(), CoupledClassical(1.0), Ehrenfest())


def test_native_manifest_is_deterministic_json_and_tracks_numerical_tree():
    problem, integrator = _problem(), Integrator(0.01)
    manifest = problem_manifest(problem, integrator)
    assert manifest["complete"] and manifest["unresolved"] == []
    assert validate_manifest(json.loads(json.dumps(manifest))) == manifest
    reversed_params = dict(reversed(list(problem.params.items())))
    assert_matching_manifest(manifest, problem_manifest(replace(problem, params=reversed_params), integrator))
    assert manifest["payload"]["params"]["sha256"]
    assert manifest["payload"]["model_spec"]["fields"]["unit_system"]["fields"] == {
        "energy_hartree": 1.0, "length_bohr": 1.0}
    assert manifest["payload"]["source"]["pyeph_python_file_count"] > 10
    assert manifest["payload"]["runtime"]["versions"]["jax"]


@pytest.mark.parametrize("change, section", [
    ("params", "params"), ("dtype", "params"), ("dt", "integrator"),
    ("substeps", "integrator"), ("mass", "nuclear_treatment"),
    ("basis", "model_spec"), ("units", "model_spec"), ("sector", "model_spec"),
])
def test_strict_matching_rejects_scientific_changes(change, section):
    problem, integrator = _problem(), Integrator(0.01)
    saved = problem_manifest(problem, integrator)
    if change == "params":
        problem = replace(problem, params={**problem.params, "delta": 0.031})
    elif change == "dtype":
        problem = replace(problem, params={**problem.params, "omega": np.array([1.], np.float32)})
    elif change == "dt":
        integrator = replace(integrator, dt=0.02)
    elif change == "substeps":
        integrator = replace(integrator, electronic_substeps=2)
    elif change == "mass":
        problem = replace(problem, nuclear_treatment=CoupledClassical(2.0))
    else:
        model = deepcopy(problem.model)
        spec = model.spec
        if change == "basis":
            spec = replace(spec, system=replace(spec.system, basis_id="new_orbital_order"))
        elif change == "units":
            spec = replace(spec, unit_system=UnitSystem(0.1, 2.0))
        else:
            spec = replace(spec, electronic_sector="different_carrier_sector")
        object.__setattr__(model, "spec", spec)
        problem = replace(problem, model=model)
    with pytest.raises(ValueError, match=section):
        assert_matching_manifest(saved, problem_manifest(problem, integrator))


@pytest.mark.parametrize("changed", ["edges", "cell", "cutoff"])
def test_static_periodic_topology_is_part_of_identity(changed):
    model = PeriodicBlockModel(2, 1, ((0, 1, 0, 0, 0),), np.eye(3)*8)
    problem = Problem(model, model.default_params(), CoupledClassical(1.0), Ehrenfest())
    saved = problem_manifest(problem, Integrator(0.1))
    changes = {"edges": {"edges": ((0, 1, 1, 0, 0),)},
               "cell": {"cell": np.eye(3)*9}, "cutoff": {"cutoff": 9.0}}
    replacement = replace(model, **changes[changed])
    with pytest.raises(ValueError, match="model"):
        assert_matching_manifest(saved, problem_manifest(replace(problem, model=replacement), Integrator(0.1)))


def test_nuclear_paths_and_method_change_identity():
    problem = replace(_problem(), nuclear_treatment=PrescribedPath(ConstantPath(jnp.array([0.]))),
                      method=CPA())
    saved = problem_manifest(problem, Integrator(0.1))
    changed = replace(problem, nuclear_treatment=PrescribedPath(ConstantPath(jnp.array([0.2]))))
    with pytest.raises(ValueError, match="nuclear_treatment"):
        assert_matching_manifest(saved, problem_manifest(changed, Integrator(0.1)))
    with pytest.raises(ValueError, match="method"):
        assert_matching_manifest(saved, problem_manifest(replace(problem, method=Ehrenfest()), Integrator(0.1)))


def test_callable_source_does_not_cover_captured_constants():
    def make_shift(scale):
        return lambda params, q: scale*jnp.sum(q*q)

    problem = _problem()
    shifted = ReferenceShiftModel(problem.model, make_shift(0.1))
    problem = replace(problem, model=shifted, params=(problem.params, None))
    manifest = problem_manifest(problem, Integrator(0.1))
    assert manifest["unresolved"] == ["model.shift_fn"]
    with pytest.raises(ValueError, match="model.shift_fn"):
        validate_manifest(manifest)
    assert_matching_manifest(manifest, manifest, strict=False)
    identified = problem_manifest(problem, Integrator(0.1),
                                  artifact_ids={"model.shift_fn": "shift-function-and-scale-v1"})
    validate_manifest(identified)
    updated = problem_manifest(problem, Integrator(0.1),
                               artifact_ids={"model.shift_fn": "shift-function-and-scale-v2"})
    with pytest.raises(ValueError, match="model"):
        assert_matching_manifest(identified, updated)
    with pytest.raises(ValueError, match="do not name"):
        problem_manifest(problem, Integrator(0.1), artifact_ids={"model.shft_fn": "typo"})


def test_custom_dataclass_fields_alone_do_not_certify_external_code():
    @dataclass(frozen=True)
    class ExternalModel:
        spec: object
        scale: float = 1.0

    problem = replace(_problem(), model=ExternalModel(_problem().model.spec))
    manifest = problem_manifest(problem, Integrator(0.1))
    assert manifest["unresolved"] == ["model"]
    identified = problem_manifest(problem, Integrator(0.1), artifact_ids={"model": "external-bundle-v1"})
    validate_manifest(identified)
    assert identified["payload"]["model"]["fields"]["scale"] == 1.0


def test_torch_captured_weights_require_provider_artifact_identity():
    torch = pytest.importorskip("torch")
    from pyeph.adapters.torch import TorchHamiltonianAdapter

    module = torch.nn.Linear(1, 2, dtype=torch.float64)
    provider = TorchHamiltonianAdapter(_problem().model.spec,
                                       lambda params, q: torch.diag(module(q)))
    problem = replace(_problem(), model=provider, params=None)
    manifest = problem_manifest(problem, Integrator(0.1))
    assert manifest["unresolved"] == ["model"]
    with pytest.raises(ValueError, match="incomplete"):
        assert_matching_manifest(manifest, manifest)
    identified = problem_manifest(problem, Integrator(0.1),
                                  artifact_ids={"model": "weights-code-baseline-probes-v1"})
    validate_manifest(identified)
    assert identified["payload"]["runtime"]["versions"]["torch"]
    # The adapter object is opaque; the caller MUST change this identity when
    # captured weights change. Do not pretend an arbitrary closure was hashed.
    with torch.no_grad():
        module.weight.add_(1.0)
    updated = problem_manifest(problem, Integrator(0.1),
                               artifact_ids={"model": "weights-code-baseline-probes-v2"})
    with pytest.raises(ValueError, match="model"):
        assert_matching_manifest(identified, updated)


def test_manifest_integrity_and_parameter_container_errors():
    manifest = problem_manifest(_problem(), Integrator(0.1))
    changed = deepcopy(manifest)
    changed["payload"]["params"]["sha256"] = "tampered"
    with pytest.raises(ValueError, match="checksum"):
        validate_manifest(changed)
    changed = deepcopy(manifest)
    changed["schema"] = 9000
    with pytest.raises(ValueError, match="schema"):
        validate_manifest(changed)
    with pytest.raises(TypeError, match="numerical leaves"):
        problem_manifest(replace(_problem(), params=object()), Integrator(0.1))
    with pytest.raises(ValueError, match="nonempty"):
        problem_manifest(_problem(), Integrator(0.1), artifact_ids={"model": ""})


def test_recorded_path_arrays_gauge_transport_and_method_policy():
    times = [0., 0.1, 0.2]
    path = AdiabaticElectronicPath(times, np.zeros((3, 2)), np.repeat(np.eye(2)[None], 2, axis=0))
    method = {"name": "recorded_cpa", "max_subspace_loss": 1e-8}
    manifest = recorded_path_manifest(path, method=method)
    validate_manifest(manifest)
    for replacement in (replace(path, transport_mode="polar"),
                        replace(path, basis_id="new_gauge"),
                        replace(path, unit_system=UnitSystem(0.2, 1.0)),
                        replace(path, energies=np.ones((3, 2))),
                        replace(path, times=[0., 0.2, 0.4]),
                        replace(path, overlaps=-np.repeat(np.eye(2)[None], 2, axis=0))):
        with pytest.raises(ValueError, match="path"):
            assert_matching_manifest(manifest, recorded_path_manifest(replacement, method=method))
    with pytest.raises(ValueError, match="method"):
        assert_matching_manifest(manifest, recorded_path_manifest(
            path, method={**method, "max_subspace_loss": 0.01}))
    dataset_only = recorded_path_manifest(path)
    assert "method" in dataset_only["unresolved"]
    with pytest.raises(ValueError, match="incomplete"):
        validate_manifest(dataset_only)


def test_fixed_recorded_integrator_and_checkpoint_metadata_roundtrip(tmp_path):
    path = FixedBasisElectronicPath([0., 1.], np.zeros((2, 2, 2)))
    policy = {"name": "recorded_cpa", "max_subspace_loss": 1e-8}
    saved = recorded_path_manifest(path, Integrator(0.1), method=policy)
    with pytest.raises(ValueError, match="integrator"):
        validate_manifest(recorded_path_manifest(path, method=policy))
    with pytest.raises(ValueError, match="integrator"):
        assert_matching_manifest(saved, recorded_path_manifest(path, Integrator(0.2), method=policy))
    manifest = problem_manifest(_problem(), Integrator(0.1))
    file = tmp_path / "strict.h5"
    save_checkpoint(file, make_state([0.], [0.], [1., 0.]), metadata={"provenance": manifest})
    _, metadata = load_checkpoint(file, expected_metadata={"provenance": manifest})
    assert_matching_manifest(metadata["provenance"], manifest)
    changed = problem_manifest(_problem(), Integrator(0.2))
    with pytest.raises(ValueError, match="metadata mismatch"):
        load_checkpoint(file, expected_metadata={"provenance": changed})
