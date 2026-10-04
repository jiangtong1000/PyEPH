# Physical model contracts

PyEPH composes physical operations rather than requiring every model to use one
Hamiltonian representation. A dense matrix, sparse EPC operator, molecular graph,
periodic orbital blocks or a learned residual can provide the same operations.
The model states what its electronic space, coordinates and energies mean;
the dynamics method requests the operations it needs.

## Native coupled dynamics

The current fixed orthonormal effective-basis profile uses

\[
\mathcal H(q,p)=\tfrac12\sum_i p_i^2/m_i+V_{\rm ref}(q)+h(q).
\]

The scalar reference and electronic correction must use the same nuclear
coordinates and define a consistent total energy. Eigenvalues of a raw orbital
Hamiltonian are not automatically charged-state total energies. Atomic units
are the default; declaring a unit system does not convert arrays.

| Operation | Meaning |
| --- | --- |
| `apply(params, q, vectors)` | Apply the carrier Hamiltonian to a vector or column block |
| `reference_energy(params, q)` | Common scalar nuclear reference potential |
| `contract_gradient(params, q, weight)` | Coordinate derivative of the weighted carrier energy with the supplied weight held fixed |
| `reference_gradient(params, q)` | Complete gradient of the reference potential |
| `probe(params, q, name)` | A declared physical observable operator |
| `prepare_action(params, q)` | Optional reusable operator action for one geometry |

A `LowRankWeight` avoids constructing a full density matrix. The Hamiltonian
representation remains model-owned. Dense spectra are required by the current
mapping methods, not by all CPA or Ehrenfest calculations. See the
[model extension guide](ADDING_MODELS.md) for executable signatures.

All geometry-dependent terms contribute to derivatives: baselines, learned
residuals, descriptors, orbital rotations within the declared effective model,
overlap transformations where explicitly supported, and cutoff factors.
Differentiating only the NN's final matrix output while freezing its geometric
inputs gives an incomplete force.

## Three supported model constructions

**Electron–phonon models.** Linear couplings can be stored as dense tensors,
sparse edges or real-space stencils. Preserve cell/image and mode conventions
at ingestion. Canonical coordinates and momenta differ from legacy scaled
quadratures; the compatibility boundary performs the declared conversion.
A prescribed harmonic bath and coupled classical nuclei are separate choices.

**Molecular aggregates.** Effective localized states carry site energies and
pair or orbital-block couplings. The reference potential, carrier sign, orbital
phases and site embedding must be explicit. Pair energy splittings alone do
not determine globally consistent signed couplings. Smoothness, permutation
covariance and loop phase consistency need validation beyond a dimer fit.

**Periodic orbital models.** Edges carry integer cell images and Hermitian
reverse conventions. Disorder is represented in an explicit supercell or an
approximation that states what momentum mixing it omits. Physical current
operators retain image displacements even when endpoints wrap to the same
finite-cell index. Variable-cell dynamics has no general native contract yet.

An additive model must share the same electronic space and avoid double
counting. A coordinate-dependent repartition between `V_ref` and the identity
part of `h` changes both gradients; their combined physical force must remain
consistent. Parameter arrays are dynamic PyTrees, while topology and basis
identity are static configuration.

## Method and measurement compatibility

CPA prescribes nuclear motion and needs electronic propagation along it.
Ehrenfest couples nuclei to the mean electronic force. Original two-state MASH
and real finite-state mapping dynamics have distinct preparation measures,
active-surface rules and estimators. A complex Hamiltonian supported by CPA or
Ehrenfest does not establish a complex/SOC hopping formulation.

Physical probes belong to the model. Their statistical estimators belong to
the selected method or workflow. Squared mapping amplitudes are not generic
physical populations. A product of valid one-time estimates is not necessarily
a valid two-time correlation estimator. Reduced quantum-bath transport includes
its declared dressing and correlation factors explicitly.

## Recorded and moving bases

Recorded electronic frames and cross-time overlaps have an electronic-only
propagation profile. [AO projection](AO_RECORDED_PATHS.md) additionally requires
the same-time metric and consistent coefficient conventions. Temporal overlaps
alone do not supply spatial force derivatives or a momentum-change direction.
A general moving nonorthogonal basis needs a basis connection and complete
energy/force meaning before it can provide nuclear feedback.

## Evidence required for an extension

Check Hermiticity, dimensions, units and declared invariances. Compare operator
actions to a small dense reference and complete force contractions to independent
finite differences. Test cutoffs and geometry-domain failures. Then establish
trajectory convergence, conservation where applicable, estimator correctness
and independent reference agreement. Material transferability requires withheld
structures and physical labels in addition to these numerical checks.
