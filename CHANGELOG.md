# Changes

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
