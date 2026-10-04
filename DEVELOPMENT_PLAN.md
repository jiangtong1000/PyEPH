# Development roadmap

The package supports effective single-carrier electron–nuclear models through
operator actions, contracted geometry derivatives and explicit physical
conventions. Molecular aggregates and periodic semiconductors are equal
application targets. Each model must identify its basis, units, nuclear
reference energy and carrier interpretation.

## Infrastructure milestones

1. **Versioned development and releases.** Maintain an explicit publication
   inventory, source history, minimum/current dependency checks, isolated wheel
   qualification and platform-specific results. Build from a fresh export. Audit
   the full remote-bound history and distributions before publication.
2. **Stable public boundaries.** Document model operations, method equations,
   preparation, measurements and execution separately. Provide readable
   calculation descriptions. Preserve exact restart identity and require an
   explicit validation record for any migration.
3. **Resumable campaigns.** Persist scientific/preparation identity, work-unit
   ownership, successes, failures and checksummed results. Recover interruptions
   explicitly. Merge independent trajectories deterministically with correct
   weights and uncertainty.
4. **Reusable learned-model lifecycle.** Standardize data conventions, structural
   holdouts, model bundles and validation reports. Keep provider reconstruction
   caller-owned. Add opt-in domain diagnostics and preserve offending geometries.
   Teacher execution and model fitting remain optional workflows.
5. **Atomic geometry and hardware qualification.** Construct and certify candidate
   graphs under motion and periodic images. Rebuild explicitly when coverage is
   lost. Qualify complete CPU/GPU workflows at matched numerical accuracy before
   promoting performance changes.

Infrastructure completion requires executable examples, independent failure and
invariant checks, installed-package verification and recorded remaining scope.
Test counts alone are not acceptance criteria. Host-specific results must not
be generalized to untested platforms.

## What to consolidate next

The model/method/execution separation is established. Extend those boundaries
through complete user workflows and explicit qualification before introducing
another shared abstraction.

| Area | Existing infrastructure | Next bounded improvement |
| --- | --- | --- |
| Release and portability | Explicit inventory, archive/history audit, installed-wheel harness, passing minimum/current Linux matrices and a complete A100 suite for the corrected runtime | Keep later support-source changes separately qualified; retain actual test selection, failures and hardware scope |
| Public calculation path | `Problem`, `Simulation`, method-owned preparation and measurements; readable configuration descriptions | Keep the migration and first-calculation examples reproducible from the released tree |
| Long trajectory campaigns | Persistent local claims, fenced retries and deterministic result merges | Add within-unit continuation and scheduler integration only with explicit checkpoint and filesystem contracts |
| Learned models | Dense fixed-basis labels plus a provider-owned local-block shard example, global family splits, model bundles, revalidation and sampled domain diagnostics | Complete two application training workflows with independent carrier and nuclear force labels, retaining charge-state and basis evidence |
| Atomic candidate graphs | Periodic construction, motion checks and explicit rebuild lineage | Implement the narrowly scoped [stage guard and recovery design](docs/NEIGHBOR_GUARD_DESIGN.md) |
| Smooth sensitivities | Pure CPA/Ehrenfest RK4 rollouts, shared host span validation, generated-model calibration and full-gradient window benchmarks with independent derivatives | Extend windowed losses and force/parameter derivative qualification to the application providers; measure actual memory and long-time conditioning |

Performance qualification must follow each method's actual operations. Sparse
or action-based CPA/Ehrenfest does not remove the complete-spectrum requirement
of the current mapping methods. Electronic-space reduction and complex
degenerate hopping need physical derivations as well as faster numerical kernels.

## Research milestones

- **General nonlinear canonical preparation.** Sample the joint classical-nuclear
  and quantum-electronic target including its electronic partition function.
  Report burn-in, mixing and sampling uncertainty; distinguish stationary
  distribution correctness from quantum dynamical accuracy.
- **Controlled electronic-space reduction.** Investigate dynamic state selection,
  omitted-space response and observable errors. Preserve the full method as an
  explicit reference. Changing mapping-space dimension is a change of physical
  approximation, not merely a faster eigensolver.
- **Complex and degenerate electronic manifolds.** Derive consistent subspace
  transport, geometric forces and momentum changes for the stated spin/basis
  conventions. Validate against independent small quantum references before
  exposing a general hopping capability.
- **Moving nonorthogonal orbital feedback.** Specify the metric and basis
  connection, complete spatial derivatives and total-energy meaning. Recorded
  temporal overlaps alone do not determine force-coupled dynamics.
- **Dynamics-aware learning.** First qualify differentiable smooth native
  CPA/Ehrenfest rollouts and observable losses. Event-time derivatives and
  acceptance branches require a separate formulation.
- **Material Hamiltonians.** Acquire independently validated carrier and nuclear
  force labels, then compare predictions on withheld configurations and
  trajectories. Numerical derivative consistency alone is insufficient.

Research progress is recorded as a derivation, runnable experiment, numerical
result or a specific unresolved obstruction. Experimental code must state its
scope and must not silently replace a qualified algorithm.

## Working discipline

Optimize complete measured calculations at matched error. Reuse existing
interfaces. Keep dynamic parameters explicit, preserve all geometry-dependent
force contributions, and fail with recoverable state when a numerical contract
is violated. Archive each qualification with source and input identity before
claiming broader support.
