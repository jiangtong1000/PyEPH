# Checkpoints and explicit run identity

A checkpoint stores numerical state. Reconstruct the same model, parameters,
nuclear treatment and numerical method before resuming it. A matching shape or
model name is insufficient: changing an orbital order, periodic image edge,
force baseline or unit scale can produce a plausible but incorrect continuation.

`Simulation.save_checkpoint(path, state, artifact_ids=None)` and
`Simulation.load_checkpoint(path, artifact_ids=None)` provide the strict path:
they create and validate a manifest against the configured problem/integrator.
`RecordedCPA` provides equivalent methods for electronic-path states. The
lower-level functions below expose the identity explicitly for other workflows.

`pyeph.io.provenance.problem_manifest` creates a JSON-ready identity for a
`Problem` and its `Integrator`. `validate_manifest` rejects incomplete identities;
`assert_matching_manifest` rejects incompatible ones. These functions neither
execute provider callbacks nor reconstruct executable Python objects.

```python
from pyeph.io.checkpoint import load_checkpoint, save_checkpoint
from pyeph.io.provenance import (
    assert_matching_manifest,
    problem_manifest,
    validate_manifest,
)

# problem, integrator, and state are the objects used for this run.
identity = problem_manifest(problem, integrator)
validate_manifest(identity)
save_checkpoint("restart.h5", state, metadata={"simulation_manifest": identity})

# In a new process, reconstruct problem and integrator explicitly first.
expected = problem_manifest(problem, integrator)
validate_manifest(expected)
state, metadata = load_checkpoint(
    "restart.h5", expected_metadata={"simulation_manifest": expected}
)
assert_matching_manifest(metadata["simulation_manifest"], expected)
```

The final comparison is redundant when `expected_metadata` already compares the
whole manifest; it is useful when checking manifests obtained from other storage.
Existing low-level `save_checkpoint(..., metadata=...)` and `load_checkpoint`
remain available. Arbitrary caller metadata does **not** automatically constitute
a complete scientific restart identity.

`simulation_manifest` is the metadata key used by the public `Simulation`
checkpoint methods. Using this key keeps the low-level example compatible with
the public loader. A different key is allowed by the low-level functions, but
then its validation and loading remain the caller's responsibility.

The checkpoint includes positions, canonical momenta, electronic vector or
column block, time, counters, trajectory IDs, random keys, and numerical
method-specific state. Recorded electronic paths use their separate state type
and retain the frame index. State arrays have shape/dtype/content checksums;
loading an x64 checkpoint with JAX x64 disabled is rejected. Writes use an atomic
file replacement, and executable pickle serialization is never used.

## Streaming output is separate from restart

Given a configured `simulation` and its valid `initial` state:

```python
from pyeph.io.hdf5 import HDF5Observer

with HDF5Observer("observables.h5", metadata={"purpose": "example"}) as writer:
    result = simulation.run(initial, steps=100, observer=writer, collect=False)
simulation.save_checkpoint("restart.h5", result.final_state)
```

The observer writes sampled times and a nested dictionary of numeric measurement
arrays. Record trajectory IDs and experiment labels explicitly in its metadata
when needed. It does not write a restartable trajectory state.
Its default exclusive-create mode preserves an existing file; write a new output
segment after restarting and account for the repeated initial row when joining
segments. Native observable streams, checkpoints and legacy transport HDF5 files
have distinct schemas and purposes. Custom providers in this example need the
artifact identities described below when saving the strict checkpoint.

If a model callback, compiled kernel or device synchronization fails, the runner
raises `SimulationError` and retains `error.last_valid_state`. For a chunk
failure this is the synchronized chunk-entry state; none of that failed chunk's
observations have been published. Save that state with the same simulation's
checkpoint method. `error.failed_state` is `None` when execution raised before
returning a trustworthy candidate, and `error.__cause__` preserves the underlying
exception. An initial measurement failure retains the validated initial state.
Method-specific failures such as MASH event exhaustion also supply diagnostics.

Correct the cause before retrying; a strict checkpoint still requires the same
scientific identity. Changing a physical model or numerical configuration is an
explicitly reviewed migration, not an automatic strict resume. Host observer
exceptions retain their original type: a partially written output segment is
not guaranteed to be transactional and should not be blindly appended to.

## What the manifest checks

The manifest records:

- Every numerical parameter leaf's shape, dtype and SHA-256 content hash, together
  with the tree structure. Dictionary key order does not affect identity; list
  and tuple structure remain distinct. Use plain dictionaries with string keys,
  lists, tuples, numerical leaves and `None`; convert custom PyTree nodes
  explicitly before generating an identity.
- Native model dataclass fields, including cells, fixed edge/image graphs,
  cutoffs, composition, charges and model dimensions. Numerical arrays are
  hashed in full rather than converted into large JSON lists.
- Actual `ModelSpec` declarations: basis identity and kind, electronic sector,
  coordinate convention, force support, reference-plus-carrier convention and
  `UnitSystem` energy/length scales. The manifest records declarations; it does
  not verify that an external provider obeys them.
- Nuclear masses, prescribed path data, method, integrator time step/algorithm/
  substeps, and measurement configuration. These affect either propagation or
  the interpretation of retained method state.
- The PyEPH version and content hash of its complete installed Python source
  tree, Python/JAX/jaxlib/NumPy/SciPy/h5py versions, and JAX x64/matmul precision
  settings. The optional Torch adapter also records the Torch version. Available
  external module source is hashed as supporting evidence.

The source check is conservative: editing any installed PyEPH Python source
changes the identity, including an unrelated module. A strict resume after a
source or dependency upgrade is rejected; establish and document equivalence
before consciously using lower-level checkpoint APIs for such a migration.
No timestamp, working-directory path, machine name or memory address participates
in the identity. Equal manifests are compatibility evidence, not a promise of
bitwise equality across accelerator hardware or compiler environments.
Floating-point trajectories can also differ in their last bits on one GPU
when output policy or segment layout changes. Array storage/load integrity is
separate from numerical continuation: require exact checkpoint round trips for
all fields, including discrete state, and assess propagated values with declared
accuracy tolerances. In event-driven methods, small arithmetic differences near
a switching boundary can change the event sequence; a matching seed alone does
not guarantee identical trajectories across execution configurations.

Native static numerical configuration is copied at construction. This includes
nuclear masses, path data, scalar time steps, unit scales, model cutoffs and
method tolerances. Mutating an original NumPy input array cannot change those
objects after compilation; shape/probe collections are immutable tuples.
Numerical `params` remain explicit runtime inputs, and their current contents
are hashed when the manifest is created.

## Clock resolution for native geometry propagation

`Simulation.run` and `DifferentiableRollout.preflight` check the requested time
span before compiled propagation or initial output. Nonzero propagation supports
float32 and float64 state clocks; `make_state` selects these from the active JAX
real precision. Checked propagation still requires float64. A finite checkpoint
time alone does not establish that adding a small timestep advances its clock.

The shared conservative policy uses the stored clock dtype's machine epsilon
`eps`. For `N` requested steps, each trajectory must have a required clock
separation greater than

```text
R = 4 * eps * (abs(initial_time) + abs(N * dt) + abs(dt)).
```

The separation is `dt / (2 * electronic_substeps)` for prescribed CPA, including
its start/midpoint/end sampling, and `dt` for coupled nuclear propagation.
Ehrenfest's frozen-geometry electronic halves use local clocks. The prescribed
policy also applies to treatments that use relative elapsed time internally;
preflight does not infer custom path implementations. Both the first forward
macrostep and a backward macrostep from the requested endpoint must advance in
the stored dtype. Separations that round to zero or become nonfinite are rejected.
The work scales with the number of trajectories, not the requested step count.

All quantities above use the same canonical time unit as `Integrator.dt`, as
defined by the problem's `UnitSystem`; no absolute tolerance in seconds is
assumed. Negative origins and spans crossing zero are valid when resolved.
Zero-step requests skip the resolution policy and retain the usual state and
path-domain checks. Passing this policy is not a guarantee of integration,
path interpolation or event-localization accuracy.

If the origin is too large, shift it consistently with the physical path:
for `t_new = t_old - t_ref`, a prescribed path must satisfy
`q_new(t_new) = q_old(t_new + t_ref)`, with the same velocity convention.
Changing path data or the clock convention is an explicit migration with recorded
provenance, not a silent strict restart. Alternatively, select suitable precision
before creating the state and qualify the resulting numerical calculation.

## Callbacks, external providers and neural weights

A callable's source code cannot identify its closure, an external weight file,
mutable module parameters or process globals. Opaque providers and custom
dataclass implementations are therefore incomplete by default. Their source
hashes never silently clear that limitation.

```python
identity = problem_manifest(problem, integrator)
print(identity["unresolved"])
# TorchHamiltonianAdapter: ["model"]
# TorchReferenceModel: ["model.reference_fn"]
# ReferenceShiftModel: ["model.shift_fn"]

identity = problem_manifest(
    problem,
    integrator,
    artifact_ids={"model": "sha256:<complete-provider-bundle-digest>"},
)
validate_manifest(identity)
```

An artifact identity is an explicit caller assertion. For a Torch adapter its
scope includes callback implementation, all captured weights/buffers, feature
normalization, physical baseline, probe definitions and provider configuration.
Generate it from immutable saved artifacts, and change it whenever any of those
change. The adapter's in-memory module weights are **not** secretly traversed or
hashed by the manifest. Prefer passing ordinary numerical model parameters as
`problem.params`, where their contents are automatically hashed.

For a reference-shift callback, provide `{"model.shift_fn": "..."}`. A callable
inside a sum model has a path such as `model.models[1].shift_fn`. Custom dataclasses
retain hashes of their declared fields but also require an identity for external
class behavior; nested callable fields need their own identities. Unused or
misspelled artifact paths raise an error. Hashing source alone, or reusing a label
after changing its artifact, does not establish a reliable restart.

Column CPA transport stores its original density-factor/current insertions in
the propagated state. Its measurement preflight also checks the origin's
encoded model/parameter/probe identity, including on no-output runs. Resume
with the saved state rather than calling the initializer again. Its physical
origin digest omits the integrator, but the strict checkpoint manifest above
still includes it; these are different checks with different purposes.

Artifact IDs describe scientific identity; they are not compilation-cache keys.
Changing one does not retrace an existing runner or refresh constants captured
from a mutable provider. Reconstruct the provider and simulation when changing
opaque implementation or captured data. Ordinary numerical parameters can use
`simulation.update_parameters(new_params)`; subsequent manifests hash the new
values and reject old checkpoints. `RecordedCPA` configuration is immutable too:
use a new instance when changing its path, integrator or transport-loss policy.

`assert_matching_manifest(..., strict=False)` permits incomplete manifests only.
It still rejects every fingerprint mismatch; it is not an option to ignore
changed parameters or a changed time step. The manifest checksum detects an
accidental change to the JSON; it is not an authenticity signature.

## Checked propagation failures

The opt-in Lanczos path returns numerical diagnostics separately from physical
state. On failure, `SimulationError.last_valid_state` is the last complete
published chunk boundary. Save it with the same simulation's `save_checkpoint`.
`failed_state` is the retained input to the rejected macrostep; it may be later
than the published boundary because earlier steps in the failed chunk were
accepted internally. That chunk's output is discarded as a whole.

`error.diagnostics` contains the attempted time, stable trajectory IDs, first
failed macrostep index and `CheckedStepInfo` with detailed action statuses.
These temporary arrays do not enter the checkpoint schema. Solver options do
enter the scientific manifest, so increasing Krylov capacity or changing its
tolerances requires an explicit new numerical policy; strict loading never
silently accepts that change. There is no automatic dense fallback or discarded
trajectory. Execution exceptions still retain a chunk boundary with
`failed_state=None` and the original chained cause.

## Recorded electronic paths

`recorded_path_manifest` hashes the full time grid and Hamiltonians, or energies
and temporal overlaps, together with the basis and interpolation/transport
policy and the path's actual `UnitSystem` declaration. Both native electronic
path classes default to atomic units; declare other scales when constructing
the path. Supply the complete numerical method policy explicitly:

```python
from pyeph.io.provenance import recorded_path_manifest

identity = recorded_path_manifest(
    recorded.path,
    recorded.integrator,
    method={
        "name": "recorded_cpa",
        "max_subspace_loss": recorded.max_subspace_loss,
    },
)
validate_manifest(identity)
```

An adiabatic recorded path has `integrator=None`, because its interval endpoints
are fixed by the recorded time grid. Its raw/polar overlap transport choice and
the subspace-loss bound are part of the identity. Omitting `method` produces a
dataset-only identity marked incomplete for strict run resume. Temporal
overlaps remain basis-transport data; checkpointing them does not supply spatial
nonadiabatic coupling vectors or a nuclear-force model.

Execution chunk size, observation save frequency, observer output filenames and
future run length are deliberately outside `problem_manifest`'s signature.
Changing them does not identify a different propagator, although it can change
roundoff or the output sampling pattern. Keep important external provenance and
experiment labels in additional checkpoint metadata alongside the manifest.
