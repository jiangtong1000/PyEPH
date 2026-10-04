# Label and learned-provider lifecycle

`pyeph.learning` contains host-side data, validation and artifact utilities.
Provider implementations still own descriptors, training, parameter trees and
model construction. The helpers do not add a trainer, select JAX precision,
import a teacher, or change the dynamics model contract.

The supported data profile is a fixed effective orthonormal electronic basis,
atomic coordinates in bohr, and energies in Hartree. A label set may contain
complex Hermitian matrices and their complete derivatives. Supporting this
data type does not imply that every dynamics method supports complex states.

This is a small dense data profile, not a universal teacher-output requirement.
When present, its full electronic derivative has shape
`(samples, states, states, atoms, 3)`, and the loader reads the complete NPZ into
memory. For 1,000 states and 1,000 atoms, one real float64 derivative tensor is
24 GB before copies. Large local/block providers need their own sharded label
formats and batch readers; those are not implemented by this loader. The
runtime can already apply local Hamiltonian blocks and differentiate scalar
energy contractions without constructing that global tensor.

Such future readers should preserve the shared identity and family-split
contracts while recording their own site/orbital order, local frames, periodic
images and phase/unitary conventions. A teacher's selected force contractions
must remain identified as contractions, rather than being presented as a full
matrix derivative. Elementwise matrix losses require a consistent electronic
representation across geometries; smooth eigenvalues alone do not supply it.

## Labels and whole-family validation

```python
from pyeph.learning import grouped_split, load_labels, validation_report

arrays, metadata = load_labels("labels.json")
splits = grouped_split(
    arrays["groups"],
    validation_groups=["validation_pose", "validation_deformation"],
    test_groups=["test_combined", "test_asymmetric"],
)
```

The existing `pyeph.fixed_basis_labels.v1` format is retained. Its JSON
manifest declares the basis identity, phase convention, electron/hole
convention, electronic energy definition, neutral reference, label scope and
source provenance. Teacher settings, source hashes and geometry identities
belong in that metadata; they are preserved without loading the teacher.
Dataset identity should bind the full manifest, not just its numerical arrays.
The loader does not infer missing conventions or convert units silently.

The sibling NPZ is checked against its SHA256 before loading, using the same
bytes for hashing and decoding. Pickle, redirected paths, symbolic links and
duplicate array names are rejected. Validation checks finite array shapes,
unique geometry IDs, physical atomic numbers, Hermiticity, gradient/force label
shapes and required companion energies. It does not numerically verify those
derivatives against the energy labels. Optional labels remain optional; they are not imputed.
These checks do not establish the physical accuracy of the teacher.

`grouped_split` assigns whole structural families or trajectory identities to
train, validation and test sets. Select fit parameters only with training
data, hyperparameters with validation data, and inspect test data only for the
final evaluation. Declared family IDs need scientific justification; a helper
cannot detect two differently named but physically duplicate configurations.

For predictions in the original row order:

```python
report = validation_report(
    {"matrix": predicted_h, "gradient": predicted_dh},
    {"matrix": arrays["h_hole"], "gradient": arrays["electronic_gradient"]},
    splits,
    geometry_ids=arrays["geometry_ids"],
    groups=arrays["groups"],
    units={"matrix": "hartree", "gradient": "hartree/bohr"},
    scope="agreement with the declared fixed-basis teacher",
)
```

Reports record each split's geometry IDs, family names, sample count and
elementwise RMSE/maximum absolute error. Splits must be disjoint, complete and
free of family leakage. Shapes must match exactly; broadcasting and nonfinite
labels are rejected. Complex RMSE uses squared modulus, including phase errors.
Provider-specific acceptance gates and symmetry/force tests remain necessary.

The compatibility module `examples/materials_data.py` forwards the older
example imports to these helpers. New code should use `pyeph.learning`.

## Data-only model bundles

`save_bundle(directory, arrays, contract=..., validation=...)` creates a fresh
directory containing `arrays.npz` and `bundle.json`. It never overwrites a
previous run. The manifest binds numerical shapes/dtypes and the payload hash
to:

- Trusted provider name/version and explicitly supplied implementation hashes.
- Numerical baseline identity and full dataset-manifest identity.
- Basis, carrier, units, neutral reference and declared scientific scope.
- Static reconstruction configuration and a scoped report of passing checks.

All stored parameters are finite numerical arrays. The format contains no
executable checkpoint, Python import target, pickle or automatic class loader.
JSON/NPZ hashes detect identity changes; they do not authenticate an untrusted
author or prove that a supplied validation report is true.

```python
from pyeph.learning import load_bundle, save_bundle

record = save_bundle(
    "new_model_bundle", parameter_arrays,
    contract=provider_contract,
    validation={
        "scope": "held-out teacher agreement and complete force finite differences",
        "checks": checked_results,  # nonempty named checks, each passed=True
        "label_report": report,
    },
)
restored, manifest = load_bundle(
    "new_model_bundle/bundle.json", expected_contract=provider_contract,
)
```

The caller must construct the expected contract independently from the
implementation and configuration it intends to run. Copying the contract out
of an arbitrary artifact and treating it as trusted defeats the compatibility
check. Hash every source/dependency that changes provider semantics, and
include needed implementation versions in the contract. The common utility
cannot discover hidden closures or external provider dependencies.

`load_bundle` returns NumPy arrays without a precision conversion. Trusted
provider code reconstructs only its own known parameter tree, rejects unknown
keys, validates parameters and explicitly selects device precision. Bind
`manifest['identity']` into existing problem/checkpoint artifact provenance.
The loader does not weaken the existing strict trajectory restart checks.

The [molecular fitting example](../examples/molecular_residual.py) writes a
`provider_bundle/` alongside its
existing fit report. `provider_bundle_contract` binds baseline parameters,
dataset metadata and the provider implementation; `load_provider_bundle`
reconstructs the known model and validates parameters. It conservatively hashes
the complete runtime tree. Its fitting-time validation covers parameter validity
and finite held-out label metrics, not dynamic or material accuracy. The local
neutral-potential and extrapolation limitations still apply.

That example requires caller-supplied twelve-atom ethylene-dimer labels, ordered
`[C,C,H,H,H,H]` for each of two consecutive fragments. Fragment IDs are six zeros
followed by six ones; ordered atoms `(0,1,2)` and `(6,7,8)` define the respective
orbital frames. Labels include real hole matrices, complete matrix derivatives,
neutral energies and neutral forces. It is a specific provider workflow, not a
trainer for arbitrary datasets. The separate
[molecular dynamics example](../examples/molecular_surrogate_dynamics.py) reads
its fitted artifact and checks CPA/Ehrenfest against independent NumPy/SciPy
equations for that same surrogate. Such agreement checks numerical execution;
charged-state teacher accuracy and transfer to other configurations require
additional reference data. A fit report's finite errors alone do not accept a
model for material prediction.

## Explicit revalidation after implementation changes

Ordinary loading rejects any contract difference. A deliberate code or provider
version change can be qualified with:

```python
from pyeph.learning import revalidate_bundle

new_record = revalidate_bundle(
    "old_model/bundle.json", "revalidated_model",
    expected_contract=new_contract,
    validator=validate_new_provider,
    reason="equivalent provider refactor checked against saved independent references",
)
```

`validate_new_provider(arrays, old_contract, new_contract)` is a caller-supplied
Python function, never executable data from the bundle. It must reconstruct
the changed provider and run the relevant gates: values, complete derivatives,
physical probes, symmetries and dynamics as required by the intended use. It
returns an explicit scope and named passing checks with reproducible evidence.
The generic helper cannot certify that those checks are scientifically adequate.

Only code hashes and provider version may change in this operation. Weights,
basis, units, baseline identity, dataset and configuration remain fixed. A
physical/model conversion instead needs a provider-owned conversion and fresh
export. Failed validation creates no new bundle. Successful revalidation saves
the source bundle identity, old contract, changed fields, reason and new check
results without modifying the source. Historical example artifacts retain their
original schema and strict loader; they are not silently migrated by this API.

## Optional geometry-domain observation

`GeometryDomainMonitor` fits the existing runner observer interface:

```python
from pyeph.learning import GeometryDomainMonitor

watch = GeometryDomainMonitor(
    "domain_diagnostics", descriptor, lower_bounds, upper_bounds,
    descriptor_id=descriptor_source_identity,
    metadata={"model_bundle": record["identity"], "dataset": dataset_identity},
    trajectory_ids=[71],
)
result = simulation.run(initial, steps, observer=watch)
```

The measurement must include atomic coordinates under `q`. The descriptor is
trusted host code returning a finite real vector for one geometry. Bounds
should be obtained from training data or a justified physical domain and then
fixed; no target-dependent fitting occurs inside the monitor. Use appropriate
invariant descriptors instead of orientation-dependent Cartesian bounds when
the physical model has rotational/translation symmetry.

At the first observed violating chunk, the monitor writes the offending
geometries, descriptors, bounds, times and stable trajectory IDs to a fresh,
checksum-bound diagnostic, then raises `DomainViolation`. The exception carries
`record_path`. Nonfinite coordinates, times or descriptors, and descriptor
execution errors, are saved as failures with diagnostic reasons. Batched runs require
explicit IDs matching batch order. Previously saved diagnostics are retained.

This is an envelope check, not calibrated epistemic uncertainty or an active
learning policy. Only sampled output is checked, after a chunk has executed;
the integrator may have advanced beyond the first violating sample. Set
`save_every=1, chunk_size=1` to check every accepted step. Internal integrator
stages remain unchecked. The geometry diagnostic is not a restart checkpoint;
retain application-owned checkpoints when restart is required. The monitor
does not launch teacher calculations or modify model parameters during a run.

## Verification

Focused checks cover label compatibility, family leakage, complex metrics,
contract mismatch/tampering, failed and successful revalidation, complete
provider force reconstruction, and a domain violation during an actual
Ehrenfest run. Reproduce with:

```sh
python -m pytest -q tests/test_learning_lifecycle.py tests/test_learning_domain.py \
  tests/test_materials_data.py tests/test_molecular_residual.py
```
