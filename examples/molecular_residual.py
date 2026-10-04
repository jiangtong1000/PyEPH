"""Fit a local ethylene-dimer surrogate without changing the dynamics core.

The teacher supplies a constrained-RHF Koopmans Hamiltonian in a declared
effective two-state basis. This example fits a Slater--Koster-like baseline
and the last layer of a tanh network, using values and all atomic derivatives.
The neutral energy/force fit is separate. It is a local model study, not a
transferable molecular potential or exact AO/moving-basis dynamics.
"""

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np

import pyeph
from pyeph.learning import (
    bundle_identity, error_metrics, grouped_split, load_bundle, load_labels,
    save_bundle, validation_report,
)
from pyeph.models.base import AutoDiffModel
from pyeph.models.composite import SumModel
from pyeph.models.fragment import OrientedFragmentCoefficients
from pyeph.models.local import AtomCenterMap, LocalBlockGraph, LocalBlockModel, LocalCoefficients
from pyeph.models.neural import NeuralResidualModel


PAIRS = tuple(zip(*np.triu_indices(12, 1)))
EXCHANGE = tuple(range(6, 12)) + tuple(range(6))
ANCHORS = ((0, 1, 2), (6, 7, 8))
ARTIFACT_SCHEMA = "pyeph.ethylene_residual.v1"


def distances(q):
    """All labelled pair distances; valid only for the fixed 12-atom topology."""
    pairs = jnp.asarray(PAIRS)
    return jnp.linalg.norm(q[pairs[:, 1]] - q[pairs[:, 0]], axis=1)


def exchanged(q):
    return q[jnp.asarray(EXCHANGE)]


def invariant_matrix(network, params, q):
    """Enforce exchange of the two identical fragments, including state labels."""
    direct = network.dense(params, distances(q))
    swapped = network.dense(params, distances(exchanged(q)))
    return (direct + swapped[::-1, ::-1]) / 2


@dataclass(frozen=True)
class ResidualCoefficients:
    baseline: OrientedFragmentCoefficients
    network: NeuralResidualModel

    def __call__(self, params, q, geometry):
        base = self.baseline(params["baseline"], q, geometry)
        correction = invariant_matrix(self.network, params["network"], q)
        phases = jnp.asarray(self.baseline.phases)
        correction = correction * phases[:, None] * phases[None, :]
        i, j = geometry.pairs.T
        return LocalCoefficients(base.onsite + jnp.diag(correction)[:, None, None],
                                 base.hopping + correction[i, j, None, None])

    def validate_params(self, params):
        if not isinstance(params, dict) or set(params) != {"baseline", "network"}:
            raise ValueError("residual params must contain baseline and network")
        self.baseline.validate_params(params["baseline"])
        self.network.validate_params(params["network"])


def polynomial(params, q):
    """Exchange-invariant local distance polynomial for the neutral baseline."""
    def features(x):
        delta = (distances(x) - params["center"]) / params["scale"]
        return jnp.concatenate((jnp.ones(1), delta, delta**2))
    return (features(q) + features(exchanged(q))) / 2


@dataclass(frozen=True)
class NeutralReference(AutoDiffModel):
    spec: object
    network: NeuralResidualModel

    def apply(self, params, q, vectors):
        return jnp.zeros_like(vectors)

    def reference_energy(self, params, q):
        base = params["offset"] + polynomial(params, q) @ params["polynomial"]
        residual = (self.network.dense(params["network"], distances(q))[0, 0]
                    + self.network.dense(params["network"], distances(exchanged(q)))[0, 0]) / 2
        return base + residual

    def probe_apply(self, params, context, probe, vectors):
        if probe not in self.spec.probes:
            return super().probe_apply(params, context, probe, vectors)
        return jnp.zeros_like(vectors)

    def validate_params(self, params):
        self.network.validate_params(params["network"])
        for name, shape in (("offset", ()), ("center", (66,)), ("scale", (66,)),
                            ("polynomial", (133,))):
            value = np.asarray(params[name])
            if value.shape != shape or value.dtype.kind not in "iuf" or not np.isfinite(value).all():
                raise ValueError(f"neutral {name} must be finite real with shape {shape}")
        if np.any(np.asarray(params["scale"]) <= 0):
            raise ValueError("neutral distance scales must be positive")


def make_models(basis_id, hidden=96, phases=(1, 1)):
    """Use the existing sparse coefficient and scalar-reference contracts."""
    centers = AtomCenterMap(tuple(np.repeat(np.arange(2), 6)), (1/6,) * 12, 2)
    # The fit domain lies inside the fully active region. The same smooth
    # support acts on baseline and residual, including in their derivatives.
    graph = LocalBlockGraph(2, 1, ((0, 1),), switch_on=12., cutoff=14.)
    baseline = OrientedFragmentCoefficients(ANCHORS, phases)
    network = NeuralResidualModel(2, (66,), hidden_sizes=(hidden,))
    base = LocalBlockModel(graph, centers, baseline, charge=1., basis_id=basis_id)
    carrier = LocalBlockModel(graph, centers, ResidualCoefficients(baseline, network),
                              charge=1., basis_id=basis_id)
    neutral = NeutralReference(carrier.spec, NeuralResidualModel(1, (66,), hidden_sizes=(hidden,)))
    return base, carrier, neutral, SumModel((carrier, neutral), additive_probes=carrier.spec.probes)


def latent(matrix):
    return jnp.stack(((matrix[0, 0] + matrix[1, 1]) / 2,
                      (matrix[0, 0] - matrix[1, 1]) / 2, matrix[0, 1]))


def baseline_params(coefficients, reference, decay):
    lengths = jnp.asarray([[jnp.linalg.norm(reference[a] - reference[o]),
                            jnp.linalg.norm(reference[b] - reference[o])] for o, a, b in ANCHORS])
    lengths = jnp.broadcast_to(jnp.mean(lengths, axis=0), (2, 2))
    # One identical monomer law, two deformation coordinates, two radial laws.
    return dict(onsite=jnp.repeat(coefficients[0], 2),
                deformation=jnp.tile(coefficients[1:3], (2, 1)), bond_lengths=lengths,
                pp_sigma=coefficients[3], pp_pi=coefficients[4], decay=jnp.asarray(decay),
                reference_distance=jnp.asarray(7.))


def design_values(function, q):
    """Materialize tiny training Jacobians only; inference uses contractions."""
    values = np.asarray(jax.jit(jax.vmap(function))(q))
    gradients = np.asarray(jax.jit(jax.vmap(jax.jacfwd(function)))(q))
    return values, gradients


def ridge_fit(design, derivative, target, gradient, train, validation, *, length=.3):
    """Select ridge strength on validation families, leaving test rows unused.

    Each atomic derivative receives length/sqrt(36), so all 36 force labels
    together have a declared scale relative to the energy/value label.
    Singular-value filtering is applied through an SVD, not normal equations.
    """
    weight = length / 6
    a = np.concatenate((design[train], weight * derivative[train].transpose(0, 2, 3, 1).reshape(-1, design.shape[1])))
    b = np.concatenate((target[train], weight * gradient[train].reshape(-1)))
    u, s, vt = np.linalg.svd(a, full_matrices=False)
    projected = u.T @ b
    candidates = []
    for relative in (1e-12, 1e-10, 1e-8, 1e-6, 1e-4):
        penalty = relative * s[0]**2
        coefficients = vt.T @ (s / (s*s + penalty) * projected)
        error = design[validation] @ coefficients - target[validation]
        derror = np.einsum("bfij,f->bij", derivative[validation], coefficients) - gradient[validation]
        score = float(np.mean(error**2) + length**2 * np.mean(derror**2))
        candidates.append((score, relative, coefficients))
    score, relative, coefficients = min(candidates, key=lambda item: item[0])
    return coefficients, dict(relative_ridge=relative, validation_score=score,
                             derivative_length_bohr=length,
                             singular_values=s.tolist(), rows=len(b))


def hidden_features(params, q):
    x = (distances(q) - params["q_center"]) / params["q_scale"]
    for layer in params["layers"][:-1]:
        x = jnp.tanh(x @ layer["weight"] + layer["bias"])
    return jnp.concatenate((x, jnp.ones(1)))


def set_output(params, weights):
    layers = params["layers"][:-1] + (dict(weight=jnp.asarray(weights[:-1]),
                                         bias=jnp.asarray(weights[-1])),)
    return params | dict(layers=layers)


def fit(arrays, metadata, splits, *, hidden=96, seed=43):
    """Fit only on training families; select hyperparameters on validation."""
    if arrays["q"].shape[1:] != (12, 3) or arrays["h_hole"].shape[1:] != (2, 2):
        raise ValueError("this example requires twelve-atom, two-fragment hole labels")
    if np.iscomplexobj(arrays["h_hole"]) or np.iscomplexobj(arrays["electronic_gradient"]):
        raise ValueError("this local teacher must be real; do not discard complex labels")
    np.testing.assert_array_equal(arrays["species"], [6, 6, 1, 1, 1, 1] * 2)
    np.testing.assert_array_equal(arrays["fragment"], np.repeat([0, 1], 6))
    base, carrier, neutral, composed = make_models(metadata["basis_id"], hidden)
    q = jnp.asarray(arrays["q"])
    train, validation = splits["train"], splits["validation"]
    reference = q[train[0]]
    targets = np.asarray(jax.vmap(latent)(arrays["h_hole"]))
    gradients = np.asarray(jax.vmap(latent)(arrays["electronic_gradient"]))
    baseline_trials = []
    for decay in (.5, .8, 1.2):
        def design(x):
            return jax.jacfwd(lambda c: latent(base.dense(baseline_params(c, reference, decay), x)))(jnp.zeros(5))
        values, derivatives = design_values(design, q)
        # One common coefficient vector fits all three independent matrix entries.
        a = values[train].reshape(-1, 5)
        da = derivatives[train].transpose(0, 1, 3, 4, 2).reshape(-1, 5)
        weights = np.linalg.lstsq(np.concatenate((a, .05*da)),
                                  np.concatenate((targets[train].reshape(-1),
                                                  .05*gradients[train].reshape(-1))), rcond=1e-11)[0]
        prediction = np.einsum("btf,f->bt", values, weights)
        derivative = np.einsum("btfij,f->btij", derivatives, weights)
        score = np.mean((prediction[validation]-targets[validation])**2) + .09*np.mean((derivative[validation]-gradients[validation])**2)
        baseline_trials.append((float(score), decay, weights, prediction, derivative))
    score, decay, weights, base_prediction, base_gradient = min(baseline_trials, key=lambda item: item[0])
    bp = baseline_params(jnp.asarray(weights), reference, decay)

    network = carrier.coefficient_provider.network
    initial = network.init_params(jax.random.key(seed))
    features = np.asarray(jax.vmap(distances)(q[train]))
    swapped_features = np.asarray(jax.vmap(lambda x: distances(exchanged(x)))(q[train]))
    features = np.concatenate((features, swapped_features))
    first = initial["layers"][0] | dict(bias=.3*jax.random.normal(jax.random.key(seed+1), (hidden,), dtype=jnp.float64))
    initial = initial | dict(q_center=jnp.asarray(features.mean(axis=0)),
                             q_scale=jnp.asarray(np.maximum(features.std(axis=0), .15)),
                             layers=(first, initial["layers"][-1]))
    def even(x):
        return (hidden_features(initial, x) + hidden_features(initial, exchanged(x))) / 2
    def odd(x):
        return (hidden_features(initial, x) - hidden_features(initial, exchanged(x))) / 2
    phi_even, dphi_even = design_values(even, q)
    phi_odd, dphi_odd = design_values(odd, q)
    coefficients, residual_records = [], []
    for index in range(3):
        phi, dphi = (phi_odd, dphi_odd) if index == 1 else (phi_even, dphi_even)
        fitted, record = ridge_fit(phi, dphi, targets[:, index]-base_prediction[:, index],
                                  gradients[:, index]-base_gradient[:, index], train, validation)
        coefficients.append(fitted)
        residual_records.append(record)
    mean, difference, hopping = coefficients
    nn = set_output(initial, np.stack((mean+difference, hopping, hopping, mean-difference), axis=1))
    cp = dict(baseline=bp, network=nn)

    # A separate scalar potential: local distance polynomial plus tanh residual.
    rp = dict(offset=jnp.asarray(np.mean(arrays["neutral_energy"][train])),
              center=initial["q_center"], scale=initial["q_scale"])
    poly, dpoly = design_values(lambda x: polynomial(rp, x), q)
    energy = arrays["neutral_energy"] - float(rp["offset"])
    derivative = -arrays["neutral_force"]
    pw, polynomial_record = ridge_fit(poly, dpoly, energy, derivative, train, validation)
    neutral_base = poly @ pw
    neutral_base_gradient = np.einsum("bfij,f->bij", dpoly, pw)
    rw, neutral_record = ridge_fit(phi_even, dphi_even, energy-neutral_base,
                                   derivative-neutral_base_gradient, train, validation)
    rp = rp | dict(polynomial=jnp.asarray(pw), network=set_output(initial, rw[:, None]))
    composed.validate_params((cp, rp))
    report = dict(seed=seed, hidden=hidden,
                  training="fixed random tanh hidden layer; derivative-supervised affine output fit",
                  normalization="training geometries and their fragment exchange only; scale floor0.15bohr",
                  baseline=dict(decay=decay, coefficients=weights.tolist(), validation_score=score,
                                trials=[dict(decay=t[1], score=t[0]) for t in baseline_trials]),
                  carrier_residual=residual_records, neutral_polynomial=polynomial_record,
                  neutral_residual=neutral_record)
    return (base, carrier, neutral, composed), (bp, cp, rp), report


def evaluate(models, params, arrays, splits):
    base, carrier, neutral, _ = models
    bp, cp, rp = params
    q = jnp.asarray(arrays["q"])
    h_base, dh_base = design_values(lambda x: base.dense(bp, x), q)
    h, dh = design_values(lambda x: carrier.dense(cp, x), q)
    e, de = design_values(lambda x: neutral.reference_energy(rp, x), q)
    reference_baseline = rp | dict(network=set_output(rp["network"], np.zeros((rp["network"]["layers"][-1]["weight"].shape[0]+1, 1))))
    e_base, de_base = design_values(lambda x: neutral.reference_energy(reference_baseline, x), q)
    predictions = dict(h_base=h_base, dh_base=dh_base, h=h, dh=dh,
                       neutral_energy=e, neutral_force=-de,
                       neutral_energy_base=e_base, neutral_force_base=-de_base)
    report = {}
    for name, index in splits.items():
        report[name] = dict(samples=len(index))
        for label, value, target in (("matrix", h, arrays["h_hole"]),
                                     ("matrix_base", h_base, arrays["h_hole"]),
                                     ("gradient", dh, arrays["electronic_gradient"]),
                                     ("gradient_base", dh_base, arrays["electronic_gradient"]),
                                     ("hopping", h[:, 0, 1], arrays["h_hole"][:, 0, 1]),
                                     ("hopping_base", h_base[:, 0, 1], arrays["h_hole"][:, 0, 1]),
                                     ("hopping_gradient", dh[:, 0, 1], arrays["electronic_gradient"][:, 0, 1]),
                                     ("hopping_gradient_base", dh_base[:, 0, 1], arrays["electronic_gradient"][:, 0, 1]),
                                     ("neutral_energy", e, arrays["neutral_energy"]),
                                     ("neutral_energy_base", e_base, arrays["neutral_energy"]),
                                     ("neutral_force", -de, arrays["neutral_force"]),
                                     ("neutral_force_base", -de_base, arrays["neutral_force"]),
                                     ("gap", np.diff(np.linalg.eigvalsh(h), axis=-1),
                                      np.diff(np.linalg.eigvalsh(arrays["h_hole"]), axis=-1))):
            report[name][label] = error_metrics(value[index], target[index])
    return predictions, report


def parameter_arrays(params):
    """Provider-owned names for a known parameter PyTree; no generic unpickler."""
    flattened, _ = jax.tree_util.tree_flatten_with_path(params)
    def name(keys):
        return "/".join(str(key.key if hasattr(key, "key") else key.idx) for key in keys)
    return {name(keys): np.asarray(value) for keys, value in flattened}


def save_parameters(path, params):
    """Plain NPZ, no pickle or executable model checkpoint."""
    arrays = parameter_arrays(params)
    np.savez_compressed(path, **arrays)
    return {key: dict(shape=list(value.shape), dtype=str(value.dtype)) for key, value in arrays.items()}


def implementation_hashes():
    """Bind artifacts to the tested source; migration requires explicit re-export."""
    root = Path(pyeph.__file__).resolve().parent
    hashes = {f"pyeph/{path.relative_to(root)}": hashlib.sha256(path.read_bytes()).hexdigest()
              for path in sorted(root.rglob("*.py"))}
    hashes["molecular_residual.py"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return hashes


def static_configuration(models):
    base, carrier, _, _ = models
    config = dict(units=dict(energy="hartree", length="bohr"),
                  graph=asdict(carrier.graph), centers=asdict(carrier.centers),
                  baseline=asdict(base.coefficient_provider),
                  network=dict(hidden_sizes=carrier.coefficient_provider.network.hidden_sizes,
                               descriptors="66 labelled distances; identical-fragment exchange projection"),
                  neutral="separate symmetric distance polynomial plus scalar tanh residual",
                  basis_id=carrier.spec.system.basis_id)
    return json.loads(json.dumps(config))


def baseline_identity(params):
    """Content identity of the provider's declared numerical baseline."""
    baseline = {key: dict(shape=list(value.shape), dtype=value.dtype.str,
                          sha256=hashlib.sha256(value.tobytes()).hexdigest())
                for key, value in parameter_arrays(params).items()}
    return bundle_identity(baseline)


def provider_bundle_contract(models, params, dataset):
    """Bind the numerical baseline, conventions, dataset and trusted source."""
    return dict(provider="pyeph.examples.ethylene_residual", provider_version="1",
                code_hashes=implementation_hashes(), baseline_sha256=baseline_identity(params[0]),
                dataset_sha256=bundle_identity(dataset), configuration=static_configuration(models),
                basis_kind=dataset["basis_kind"], basis_id=dataset["basis_id"],
                units=dataset["units"], carrier=dataset["carrier"],
                neutral_reference=dataset["neutral_reference"],
                scope="local constrained-RHF ethylene surrogate; no material transferability claim")


def parameters_from_arrays(arrays):
    """Reconstruct only this provider's known tree and preserve scientific precision."""
    if not jax.config.x64_enabled:
        raise ValueError("molecular residual artifacts require explicit JAX_ENABLE_X64=1")
    baseline_keys = ("onsite", "deformation", "bond_lengths", "pp_sigma", "pp_pi", "decay",
                     "reference_distance")
    expected = {f"{prefix}/{key}" for prefix in ("0", "1/baseline") for key in baseline_keys}
    expected |= {f"2/{key}" for key in ("offset", "center", "scale", "polynomial")}
    for prefix in ("1/network", "2/network"):
        expected |= {f"{prefix}/{key}" for key in ("q_center", "q_scale")}
        expected |= {f"{prefix}/layers/{layer}/{key}"
                     for layer in range(2) for key in ("weight", "bias")}
    if set(arrays) != expected:
        raise ValueError("missing or unknown provider parameter keys")
    arrays = {name: jnp.asarray(value) for name, value in arrays.items()}
    def mapping(prefix):
        return {key[len(prefix)+1:]: value for key, value in arrays.items()
                if key.startswith(prefix+"/") and "/" not in key[len(prefix)+1:]}
    def network(prefix):
        return mapping(prefix) | dict(layers=tuple(mapping(f"{prefix}/layers/{i}") for i in range(2)))
    bp = mapping("0")
    cp = dict(baseline=mapping("1/baseline"), network=network("1/network"))
    rp = mapping("2") | dict(network=network("2/network"))
    restored = (bp, cp, rp)
    if set(parameter_arrays(restored)) != set(arrays):
        raise ValueError("unknown provider parameter keys")
    return restored


def load_provider_bundle(path, *, expected_contract):
    """A concrete provider owns reconstruction; the common loader only reads data."""
    if (expected_contract.get("provider") != "pyeph.examples.ethylene_residual"
            or expected_contract.get("provider_version") != "1"
            or expected_contract.get("code_hashes") != implementation_hashes()):
        raise ValueError("artifact implementation differs from this provider")
    arrays, record = load_bundle(path, expected_contract=expected_contract)
    config = expected_contract["configuration"]
    models = make_models(expected_contract["basis_id"], config["network"]["hidden_sizes"][0])
    if config != static_configuration(models):
        raise ValueError("artifact static provider configuration mismatch")
    params = parameters_from_arrays(arrays)
    if baseline_identity(params[0]) != expected_contract["baseline_sha256"]:
        raise ValueError("artifact numerical baseline identity mismatch")
    models[0].validate_params(params[0])
    models[-1].validate_params(params[1:])
    return models, params, record


def load_artifact(report_path):
    """Restore the declared native provider from a checksum-bound plain NPZ."""
    path = Path(report_path)
    report = json.loads(path.read_text())
    if report.get("artifact_schema") != ARTIFACT_SCHEMA:
        raise ValueError("unsupported molecular residual artifact schema")
    if not jax.config.x64_enabled:
        raise ValueError("molecular residual artifacts require explicit JAX_ENABLE_X64=1")
    if report.get("implementation_hashes") != implementation_hashes():
        raise ValueError("artifact implementation differs; validate and re-export explicitly")
    payload = path.parent / "parameters.npz"
    if hashlib.sha256(payload.read_bytes()).hexdigest() != report["parameters_sha256"]:
        raise ValueError("parameter checksum mismatch")
    with np.load(payload, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    schema = report["parameter_schema"]
    if set(arrays) != set(schema) or any(
            dict(shape=list(arrays[key].shape), dtype=str(arrays[key].dtype)) != description
            for key, description in schema.items()):
        raise ValueError("parameter schema mismatch")
    bp, cp, rp = parameters_from_arrays(arrays)
    models = make_models(report["dataset"]["basis_id"], report["training"]["hidden"])
    if report.get("static_configuration") != static_configuration(models):
        raise ValueError("artifact static provider configuration mismatch")
    models[-1].validate_params((cp, rp))
    models[0].validate_params(bp)
    return models, (bp, cp, rp), report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validation-groups", nargs="+", required=True)
    parser.add_argument("--test-groups", nargs="+", required=True)
    parser.add_argument("--hidden", type=int, default=96)
    parser.add_argument("--seed", type=int, default=43)
    args = parser.parse_args()
    if not jax.config.x64_enabled:
        raise RuntimeError("set JAX_ENABLE_X64=1 explicitly")
    source_identity = implementation_hashes()
    manifest_identity = hashlib.sha256(args.manifest.read_bytes()).hexdigest()
    arrays, metadata = load_labels(args.manifest)
    splits = grouped_split(arrays["groups"], validation_groups=args.validation_groups,
                           test_groups=args.test_groups)
    args.output.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    models, params, training = fit(arrays, metadata, splits, hidden=args.hidden, seed=args.seed)
    predictions, errors = evaluate(models, params, arrays, splits)
    if source_identity != implementation_hashes() or manifest_identity != hashlib.sha256(args.manifest.read_bytes()).hexdigest():
        raise RuntimeError("source or dataset manifest changed during fitting; preserve this failed run and rerun")
    schema = save_parameters(args.output/"parameters.npz", params)
    np.savez_compressed(args.output/"predictions.npz", **predictions, **{f"split_{k}": v for k, v in splits.items()})
    record = dict(artifact_schema=ARTIFACT_SCHEMA,
                  scope="local constrained-RHF effective ethylene-dimer model; no material transferability claim",
                  dataset=metadata, dataset_manifest_sha256=manifest_identity,
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  versions=dict(jax=jax.__version__, numpy=np.__version__), backend=jax.default_backend(),
                  jax_enable_x64=bool(jax.config.x64_enabled),
                  jax_default_matmul_precision=jax.config.jax_default_matmul_precision,
                  jax_random_configuration={name: getattr(jax.config, name, None) for name in (
                      "jax_default_prng_impl", "jax_threefry_partitionable", "jax_random_seed_offset")},
                  training=training, errors=errors, seconds=time.perf_counter()-start,
                  splits={k: arrays["geometry_ids"][v].astype(str).tolist() for k, v in splits.items()},
                  static_configuration=static_configuration(models), implementation_hashes=source_identity,
                  parameter_schema=schema,
                  parameters_sha256=hashlib.sha256((args.output/"parameters.npz").read_bytes()).hexdigest())
    (args.output/"report.json").write_text(json.dumps(record, indent=2)+"\n")
    targets = dict(matrix=arrays["h_hole"], gradient=arrays["electronic_gradient"],
                   neutral_energy=arrays["neutral_energy"], neutral_force=arrays["neutral_force"])
    fitted = dict(matrix=predictions["h"], gradient=predictions["dh"],
                  neutral_energy=predictions["neutral_energy"],
                  neutral_force=predictions["neutral_force"])
    label_report = validation_report(
        fitted, targets, splits, geometry_ids=arrays["geometry_ids"], groups=arrays["groups"],
        units=dict(matrix="hartree", gradient="hartree/bohr", neutral_energy="hartree",
                   neutral_force="hartree/bohr"), scope=record["scope"])
    models[0].validate_params(params[0])
    models[-1].validate_params(params[1:])
    save_bundle(args.output/"provider_bundle", parameter_arrays(params),
                contract=provider_bundle_contract(models, params, metadata),
                validation=dict(scope="parameter validity and finite family-held-out label metrics",
                                checks=[dict(name="parameter validation", passed=True),
                                        dict(name="family-disjoint finite label report", passed=True)],
                                label_report=label_report))
    print(json.dumps(dict(errors=errors, seconds=record["seconds"], output=str(args.output)), indent=2))


if __name__ == "__main__":
    main()
