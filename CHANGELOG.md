# Changes

## 0.1.0.dev3 (unreleased)

- Add an explicit coordinate-domain boundary to scalar checked CPA/Ehrenfest
  stages, with retained rejection diagnostics and chunk rollback. Coordinate
  membership does not certify neighbor coverage or model accuracy.
- Add explicit fixed-basis label ingestion with saved conversion provenance,
  and bind the periodic example's dimensional inputs to its model contract.

This candidate has its own complete current/minimum macOS CPU checks and
selected optional-Torch checks; see [qualification scope](docs/QUALIFICATION.md)
for artifact identities, warnings and remaining platform coverage. The dev2
accelerator results do not qualify its changed runtime. Automatic neighbor
rebuilding remains unfinished.

## 0.1.0.dev2

- Correct host neighbor coverage at floating-point boundaries using exact
  float-input geometry. Check periodic search budgets before allocation and
  verify candidate completeness after coordinate rewrapping.
- Preserve byte-string geometry IDs through fitting reports and dynamics
  initialization. Verify and decode one parameter-file snapshot during legacy
  artifact loading.
- Add a generated neural-Hamiltonian benchmark for full-gradient windowed
  losses, independent discrete references and short-horizon Hessian-vector
  checks. It adds no new runtime API or material-accuracy claim.
- Record source-specific installed CPU, CI and accelerator qualification.
- Retain qualification test-selection arguments, relevant environment settings
  and ordered collection identity. Accept explicit pytest flag passthrough and
  report slow test phases to guide measured validation-workflow improvements.
- Add an example of provider-owned local label shards for molecular and
  periodic models, with complete force contractions, global structural splits,
  count-weighted losses and strict candidate-bundle reconstruction.
- Clarify the fixed-geometry linear operator contract and the separate energy
  functional needed by electronic-state-dependent self-consistent generators.

Host coverage checks remain separate from automatic guarding of internal
dynamics stages. Earlier qualification does not silently apply to changed
runtime files; see [qualification scope](docs/QUALIFICATION.md).

## 0.1.0.dev1

This development version extends the modular dynamics core while retaining the
maintained transport and preprocessing entry points.

- Add persistent independent-trajectory campaigns with explicit claims,
  recovery, checksummed shards and deterministic statistical merging.
- Add candidate-neighbor construction and conservative coverage checks for
  finite and fixed periodic cells; rebuilding remains an explicit operation.
- Add reusable label validation, structural splits, model bundles,
  revalidation records and sampled geometry-domain diagnostics.
- Add `Simulation.describe()` and document the public extension boundaries.
- Add a pure RK4 CPA/Ehrenfest sensitivity interface with explicit preflight,
  static-configuration checks and optional rematerialization.
- Add finite-chain Metropolis preparation for confined real nonlinear models,
  with strict chain restart and separately reported mixing limitations.
- Make publication contents explicit and test installed distributions against
  minimum/current numerical dependencies.

Exact trajectory and chain restarts remain bound to their original numerical,
source and runtime identities. An older checkpoint is not silently upgraded.
Unchanged model weights may use the separately validated bundle-revalidation
pathway; retain the original artifact and record the new validation.

Moving-basis feedback and electronic-space reduction remain research work.
Complex Hamiltonian support in CPA/Ehrenfest does not qualify SOC hopping or
individual-surface dynamics inside a degenerate manifold.
