# Development guidance

PyEPH separates physical models, dynamics methods, numerical execution and
measurement. Read the relevant implementation and callers before changing a
boundary. The implemented layout is in `docs/ARCHITECTURE.md`; current scope and
evidence are in `docs/QUALIFICATION.md` and `DEVELOPMENT_PLAN.md`.

## Reuse and clarity

Establish that a feature serves the requested calculation, look for an existing
implementation, and use the standard library or existing dependencies before
introducing another one.

- Prefer the existing model operations and `LocalBlockModel` provider boundary
  over a new universal Hamiltonian hierarchy.
- Keep a helper beside its callers until separate real uses justify sharing it.
- Optimize a measured complete workflow at matched numerical accuracy.
- Prefer readable equations and explicit conventions over compressed code.
- Preserve necessary physical validation, independent reference tests,
  provenance and failure diagnostics. Minimum line/file/test counts are not
  scientific acceptance criteria.

## Scientific and implementation requirements

- State the electronic basis, units, carrier convention and nuclear reference
  potential. Distinguish a parameterized model, fitted surrogate and validated
  material model in examples and reports.
- All geometry-dependent terms participate in forces, including baselines,
  descriptors, overlap/basis transformations and smooth cutoff factors.
- Use operator actions and contracted derivatives where supported. Dense
  spectra are method-specific requirements, not a universal model contract.
- Keep changing parameters in explicit PyTrees. Do not alter global JAX
  precision at import. Host validation and external callbacks stay explicit.
- MASH2 and MASHRM have distinct preparations and estimators. Do not claim
  complex/SOC, degenerate-spectrum or general canonical RM support from a
  passing CPA/Ehrenfest example.
- Add regression checks for a real failure or physical invariant; reuse the
  existing pytest infrastructure and independent oracles.

## Working evidence

Preserve existing benchmark records, qualified snapshots and distributions.
Write new run outputs to a fresh path and bind them to their actual inputs and
sources. Sibling repositories are reference inputs; do not modify them as part
of this package's development. Keep optional teacher/provider dependencies
isolated until they have a supported dependency contract.

Before finishing a change, check reuse, scientific scope, relevant tests and
measured performance claims. A green numerical check does not establish model
accuracy outside the tested domain.

## Publication boundary

Only files named in `release-files.txt` may enter a release or remote branch.
Audit the staged tree, complete reachable history, and built distributions
before publication. Internal comparison records and external reference bundles
remain local evidence. Do not remove required copyright or license notices;
exclude material that cannot satisfy the release scope with its notices intact.
Dependencies and scientific equations must be described accurately.
