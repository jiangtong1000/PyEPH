# Finite-chain canonical preparation for nonlinear models

`NativeCanonicalMetropolis` provides an explicit initialization workflow for
real, native, finite, fixed-orthonormal-basis electronic models. It leaves the
existing linear-model rejection sampler and transport workflow unchanged.
The nonlinear sampler requires JAX x64, a differentiable nuclear reference,
complete electronic forces and constant positive masses. It does not enable
complex/SOC or degenerate-spectrum dynamics.

## Target and transition

For inverse temperature beta and constant Cartesian/canonical masses, integrate
out momentum and sum over active electronic states. The coordinate target is

\[
\pi(q) \propto e^{-\beta V_{\rm ref}(q)}\sum_a e^{-\beta E_a(q)}.
\]

All geometry-dependent carrier terms contribute through the eigenvalues of the
complete effective Hamiltonian. Bare nuclear-reference sampling followed by an
electronic draw generally has the wrong coordinate distribution.

The proposal is `q' = q + proposal_scale * Normal(0,1)`, with fixed positive
componentwise scales. Its density is symmetric, so

\[
\alpha(q,q')=\min\{1,\exp[\log\pi(q')-\log\pi(q)]\}.
\]

Both accepted probability fluxes equal the symmetric proposal density times
`min(pi(q), pi(q'))`; the rejected probability stays at q. This establishes
invariance/detailed balance in exact arithmetic, subject to normalizability and
irreducibility. It does not establish convergence after a finite burn-in.
The calculation removes fixed energy origins and uses shifted log-sum-exp
arithmetic. This is ordinary Metropolis sampling, not a new sampling algorithm.
See [Metropolis et al., J. Chem. Phys. 21, 1087 (1953)](https://doi.org/10.1063/1.1699114).

The caller must establish confinement/normalizability in the chosen coordinate
space. A freely translating aggregate on infinite Cartesian space is not a
normalizable canonical coordinate distribution. Constraints, position-dependent
mass metrics, quotient coordinates and hard-wall domains require separate
measures and are unsupported here. Fixing arbitrary coordinates can change the
physical problem; declare the intended ensemble before doing so.

## Preparation and diagnostics

```python
from pyeph.workflows.canonical_metropolis import NativeCanonicalMetropolis

sampler = NativeCanonicalMetropolis(
    model, params, masses, beta, proposal_scale, reference_q,
    artifact_ids=artifact_ids,  # when the provider/model requires an explicit identity
)
prepared = sampler.sample(
    trajectory_ids, initial_q, initialization_id="sha256:<initialization-bundle>",
    burn_in=1000, production_steps=1000, thin=10, seed=28,
)
state = prepared.state
```

`reference_q` fixes numerical energy origins only; it is not the sampling center
or a harmonic approximation. `initial_q` is either one model-shaped coordinate
or a batch in the requested ID order. The caller-owned initialization identity
must cover the code, distribution, external inputs and any additional seed used
to construct those coordinates. It is not inferred from an opaque callback.

Every stable trajectory ID has a separate random stream. Proposal randomness is
indexed by ID and proposal count, while endpoint momentum, active surface,
mapping vector and future dynamics use a separate domain. Reordering IDs or
restarting a chain preserves its stream. Numerical batching can still change
last-bit arithmetic for some providers/hardware.

After the explicit warmup, the workflow retains a diagnostic sequence at the
requested thinning interval. `production_steps` must be divisible by `thin`.
It returns **one endpoint per ID**, not every retained chain position as an
independent trajectory. Retained positions within each chain remain correlated;
thinning alone does not prove independence or remove finite-warmup bias.

Diagnostics include cumulative acceptance rates, ordinary coordinate split-Rhat
and pooled lag-one coordinate correlation. Split-Rhat requires at least two
chains and four retained positions. High split-Rhat, zero within-chain variance
and extreme acceptance rates produce `MixingWarning`. These basic diagnostics
can miss unvisited metastable regions or slowly mixing nonlinear observables.
They are not a rank-normalized convergence analysis, an ESS estimator or a
certificate of equilibrium. Use dispersed physical initializations, longer
runs, and relevant additional observables. Independent-ID error bars quantify
random endpoint variation; they do not bound a shared finite-burn-in bias.

At each final endpoint, momentum has covariance `M/beta`, the active adiabatic
surface is Boltzmann-distributed, and the mapping vector is drawn uniformly on
the complex unit sphere conditioned on that surface having the largest
population. The fixed-basis vector is obtained from that endpoint's real
eigenvectors. This preserves the separate RM preparation and one-time estimator
convention; see [Runeson and Manolopoulos, J. Chem. Phys. 159, 094115 (2023)](https://arxiv.org/abs/2305.08835).

## Explicit continuation and failures

```python
chain = sampler.start(ids, initial_q, initialization_id=initialization_id, seed=28)
chain = sampler.advance(chain, 500).chain
sampler.save_chain("outputs/chain.npz", chain)
restored = sampler.load_chain("outputs/chain.npz")
run = sampler.advance(restored, 500, record_every=10)
prepared = sampler.finalize(run.chain, burn_in=500, thin=10)
```

Saving refuses to overwrite an existing artifact unless `overwrite=True` is
explicitly supplied. Failed checkpoints retain their diagnostic arrays and
receive the same structural validation as successful chains. Failed initial
targets may have nonfinite log densities; failed proposals may themselves be
nonfinite. Neither case permits propagation.

Checkpoints contain numerical arrays and JSON identities only; no model,
callable or executable serialization is loaded. Reconstruct the same sampler
before loading. Source, runtime, model, parameters, masses, beta, proposal scales
and energy origins are bound to strict provenance. `chain.subset(ids)` supports
explicit partitioning. There is no proposal-scale adaptation or hidden mutable
random generator.

A nonfinite coordinate, Hamiltonian, target density or asymmetric Hamiltonian is
an error. The workflow does not silently reject such proposals as zero-density
configurations. `MetropolisSamplingError.chain` retains all requested IDs, last
valid endpoints and the offending proposal coordinates. Failed diagnostics can
be checkpointed but failed chains cannot be advanced or finalized.

Degenerate intermediate Hamiltonians are valid in the coordinate partition
function. The endpoint gap check occurs only when constructing RM physical
states. An incompatible endpoint raises with the endpoint and attempted state
retained; it is never redrawn, filtered or replaced. Complete reference and
carrier forces, momenta, mapping coefficients and total energy are checked for
finiteness before returning physical states.

No general nonlinear canonical transport workflow is claimed. The prepared
states can run with the existing real MASHRM method, but a physical current,
its allowed RM correlation estimator, stationarity, finite-step errors and
sampling convergence must be justified for that calculation separately.
Multiplying one-time mapping population estimators does not create a valid
transport correlation.

## Evidence and resource scope

The [nonlinear canonical benchmark](../benchmarks/nonlinear_canonical.py) uses
one confined analytic two-state model with a quartic neutral reference,
nonlinear onsite and hopping terms, classical nuclei, atomic units and a fixed
orthonormal basis. Independent scalar eigenvalue formulas and adaptive
quadrature provide q, q-squared and active-state equilibrium references.
Warmup-length comparisons retain source identities, raw chain checkpoints and
mixing diagnostics. They validate this fixture, not arbitrary learned material
models or quantum nuclear sampling.

Regression tests additionally check actual Metropolis accept/reject decisions,
pairwise detailed balance, conditional mapping moments, Maxwell momentum
moments, source-bound chain restart/partitioning, constant energy shifts,
retained numerical failures, endpoint degeneracy and a deliberately poorly
mixing case. Sampling requires a complete dense spectrum per proposal and is
not a reduced-state or scalable atomistic equilibrium algorithm. Unrecorded
advancement retains only chain endpoints and counters; explicitly recorded
traces require memory proportional to records times chains times coordinates.
