# Two-state MASH implementation

`pyeph.dynamics.mash2.MASH2` implements a deliberately limited, testable MASH
method for real two-state Hamiltonians in a fixed orthonormal diabatic basis.
Its numerical scheme is named `fixed_diabatic_pc_midpoint`. It uses explicit
nuclear coordinates, canonical momenta, and positive diagonal masses. A model
must supply the complete scalar reference energy and contracted electronic
derivatives. A two-state analytic model or differentiable neural Hamiltonian
can meet this same contract.

This implementation does not support complex Hamiltonians, spin–orbit
coupling, degenerate surfaces, moving/nonorthogonal model bases, multistate
MASH, thermostats, decoherence corrections, coherent-state preparation, or
general electronic correlation estimators. These need separate derivations
and validation. Differentiable model forces do not imply a differentiable
whole MASH trajectory: discrete events and bounded searches are present.

## Scientific references and implemented scope

The underlying method and population weights follow Mannouch and Richardson,
*A mapping approach to surface hopping*, J. Chem. Phys. (2023),
[DOI:10.1063/5.0139734](https://doi.org/10.1063/5.0139734).
The inspected primary manuscript is
[arXiv:2212.11773v1](https://arxiv.org/pdf/2212.11773v1), particularly Eqs. 7–8
and 25–30: a unit spin selects the active surface; an equator crossing triggers
a NAC-directed momentum impulse; population measurements require their
associated initial sampling weight.

Event localization is motivated by Geuther, Asnaashari, and Richardson,
*Time-reversible implementation of MASH for efficient nonadiabatic molecular
dynamics*, J. Chem. Theory Comput. **21**, 2179–2188 (2025),
[DOI:10.1021/acs.jctc.4c01684](https://doi.org/10.1021/acs.jctc.4c01684).
The inspected primary manuscript is
[arXiv:2412.15976v1](https://arxiv.org/pdf/2412.15976v1), especially §3.2:
splitting nuclear propagation at a converged event avoids delaying the impulse
to the end of a step. That paper also discusses global diabatic models in §2.

Our fixed-basis midpoint electronic exponential and bounded bisection search
are explicit implementation choices. This is not an implementation of the
paper's particular `rev-pc-LD`, `rev-pc-NACs`, or overlap propagators. Tests
establish convergence on the stated examples; they do not establish
published population parity or a universal convergence theorem.

## Hamiltonian, mapping state, and measurements

Write the classical/electronic Hamiltonian as

\[
H(q,p)=\sum_a p_a^2/(2m_a)+V_{\rm ref}(q)+h(q),
\qquad h U=U\,\operatorname{diag}(E_-,E_+).
\]

The `electronic` array stores a normalized two-component complex vector in
the fixed diabatic basis. It is a convenient representation of a unit spin,
not a physical electronic wavefunction estimator. With `a = U.conj().T @ c`
in sorted lower/upper order,

\[
S_z=|a_+|^2-|a_-|^2,\quad
S_x=2\operatorname{Re}(a_+^*a_-),\quad
S_y=2\operatorname{Im}(a_+^*a_-).
\]

The active surface is lower for negative `Sz`, upper for positive `Sz`.
Its force and conserved continuum energy are

\[
F_a=-\nabla V_{\rm ref}-\nabla\operatorname{Tr}(P_a h),\qquad
\mathcal E=\sum_i p_i^2/(2m_i)+V_{\rm ref}+E_a,
\]

where the projector is held fixed in the derivative contraction. Using
`c.conj() @ h @ c` for this conserved energy would implement a different
physical model. The code requests a selected surface projector and, only
at an event, a selected NAC contraction. It does not store a full derivative
tensor in trajectory state.

`sample_adiabatic_population(..., active=0|1)` draws an initial adiabatic
population with `Sz = ±sqrt(U)` and uniform azimuth. Thus the population-to-
population weight is absorbed into the initial ensemble. Nuclear q,p sampling
is independent and supplied by the caller. A population mixture can be
prepared by independently drawing its initial surface from the intended
mixture probabilities before calling this constructor. Trajectory IDs and a
seed determine samples independently of batching.

`MASHPopulation()` measures the one-hot **active surface**. Ordinary squared
mapping amplitudes are not the physical population estimator. The measurement
also reports mapping norm, `Sz`, active-surface energy, and event diagnostics.
At an exactly localized event it records the outgoing active side; this is a
numerical one-sided convention at an isolated instant. The initial weighted
samples do not by themselves support coherence or arbitrary correlation
measurements. `mapping_state(..., spin=...)` constructs deterministic diagnostic
trajectories; it does not claim to prepare a quantum density matrix.

## Actual numerical step

Each outer timestep is divided into `event_substeps` intervals. Within one
interval, the following no-hop map is computed on the current active surface:

1. Velocity Verlet updates q,p using the initial and final surface forces.
2. The mapping vector is updated by the dense unitary exponential of
   `h((q_initial + q_final)/2)` over that interval's duration.
3. The final `Sz` is checked against the active hemisphere.

This smooth map is symmetric in time and second order for sufficiently smooth
Hamiltonians. It uses the arithmetic midpoint of endpoint positions; it is not
an exact solution for a varying Hamiltonian. `Integrator` must specify
`electronic="exponential_midpoint"` and `electronic_substeps=1`. Subdivision is
controlled explicitly by the MASH method.

If the endpoint changes hemisphere, bounded bisection repeatedly reevaluates
the no-hop map from the start of that interval to a trial duration. Once the
`Sz` residual meets `event_tolerance`, the impulse is applied at that geometry,
and the remaining interval is propagated from the event. The small residual
is projected onto the equator while preserving the relative adiabatic phase.
This changes the finite-tolerance spin by the localization error; it is not
an additional physical stochastic jump. In particular, a frustrated event
reverses the crossing direction through its momentum reflection.

For canonical momenta, define `v = p/sqrt(m)` and
`n = (d/sqrt(m))/norm(d/sqrt(m))`, where
`d = <u_lower | grad u_upper>`. Only `v_parallel = dot(v,n)` changes. An
allowed hop with energy increase `ΔE` uses
`v_parallel_new = sign(v_parallel)*sqrt(v_parallel**2 - 2*ΔE)`.
An energetically forbidden hop reflects that component. The perpendicular
mass-weighted momentum is unchanged. A zero NAC or vanishing incident
parallel component is reported as an unresolved event. No arbitrary NAC
direction, gap denominator floor, or random hop is inserted.

The interval scan processes the earliest **detected** crossing bracket before
later intervals. It cannot guarantee discovery of a pair of crossings hidden
inside one interval, nor that a nonmonotone unresolved bracket contains only
one root. The test suite contains an example where refinement exposes hidden
recrossings. Both timestep/subdivision and event-tolerance convergence are
required for a new model. Whole-trajectory exact reversibility at finite event
tolerance or finite subdivision is not claimed. The no-hop symmetric map is
tested separately under p→−p and c→conj(c).

## Bounds, diagnostics, and failure handling

Method configuration contains `event_substeps`, `max_events_per_step`,
`bisection_iterations`, `event_tolerance`, `gap_tolerance`, `real_tolerance`,
and `direction_tolerance`. Dimensional tolerances use the model-declared
atomic or reduced kernel units; the event tolerance is dimensionless. The default event tolerance requires
JAX x64. A host check rejects an event tolerance below eight coordinate-machine
epsilons; lower precision requires an explicit looser tolerance.

`method_state` contains active surface, status, event/attempt/accepted/frustrated
counters, total localization iterations, maximum event residual, and maximum
absolute impulse energy error. Successful and frustrated impulses count as
events; failed root searches and capacity-rejected detected crossings also
count as attempts. Search iterations include failed localization attempts.

| Status | Meaning |
| --- | --- |
| 0 | Successful step |
| 1 | Invalid/complex/non-Hermitian/degenerate Hamiltonian or invalid force |
| 2 | Mapping hemisphere disagrees with the stored active surface |
| 3 | Localization failed within the configured iteration budget |
| 4 | Undefined NAC/incident momentum direction at the event |
| 5 | More detected events than the configured per-step capacity |

Compiled kernels preserve the last finite accepted q,p,c and its physical time
after failure. Later steps leave that trajectory frozen. The runner invokes
`validate_result` at every chunk boundary; nonzero status raises `MASHError`,
a `SimulationError` with `failed_state` and the preceding whole-chunk
`last_valid_state`. Failed output is not silently presented as a successful
trajectory. In a batch, other trajectories may have progressed farther; the
previous chunk boundary is the synchronized restart point. Bounds must be
increased or the timestep reduced before rerunning a failed chunk.

Model parameters are runtime PyTrees. Mapping state and diagnostics are arrays
and support JIT, independent-trajectory batching, and checkpoint continuation.
The implementation prioritizes a small auditable event kernel. Bisection can
require many force evaluations; interpolated or safeguarded faster localizers
would be separate numerical changes requiring renewed validation.

## Reproducible check and measured limits

Run:

```sh
.venv/bin/python -m pytest tests/test_mash2.py -q
.venv/bin/python examples/mash2_tully.py
```

The tests include independent analytic moving-basis propagation for a rotating
constant-gap model, allowed and frustrated trajectories, impulse energy and
perpendicular-momentum invariance with unequal masses, sampler moments,
no-coupling harmonic motion, timestep/subdivision convergence, failed searches,
capacity exhaustion, and JIT/batch/restart consistency.

The example uses Tully model 1, 64 initial lower-population spins, fixed
q=−4 and p=20, mass 2000, and duration 1000 atomic time units. It saves JSON
and NPZ results under the working prefix `outputs/mash2_tully` by default;
choose another prefix with `--output`. The numbers below describe a historical
local CPU run whose execution archive is not included in the distribution.
The example command above generates new records for the current environment.
This is a monokinetic
classical ensemble with no initial nuclear Wigner distribution; it is a
numerical convergence benchmark, not a reproduction of a published quantum
scattering calculation.

On the development CPU with JAX 0.11.2 and x64, the recorded run gave:

| Outer dt (2 subdivisions) | Maximum sampled energy drift | Accepted / frustrated | Final lower / upper population |
| --- | --- | --- | --- |
| 1.0 | 1.571×10⁻⁷ | 63 / 0 | 0.578125 / 0.421875 |
| 0.5 | 3.942×10⁻⁸ | 63 / 0 | 0.578125 / 0.421875 |

All statuses were zero. Maximum norm error was 2.21×10⁻¹³ and maximum impulse
energy error 3.03×10⁻¹⁷. The binomial standard error of each reported final
population is approximately 0.062, so identical rounded populations alone
would be weak evidence; the independent trajectory tests and energy refinement
provide the stronger numerical checks. Maximum final-position change was
5.11×10⁻⁵. Cached ensemble runs took approximately 0.63 and 1.16 seconds on
the shared development CPU; compile-plus-first-run took 3.26 and 3.04 seconds.
These are local observations, not portable performance promises.

## Wavepacket distribution comparison

The separate [wavepacket driver](../benchmarks/mash2_wavepacket.py) uses the
modified Tully potential and Gaussian Wigner preparations defined in that
script. It generates new trajectories and an independent FFT quantum reference;
this is not a comparison with digitized published curves. The
[distribution analysis](../benchmarks/mash2_distribution_analysis.py) consumes
one quantum-reference pair and four low/high, dt1/dt05 trajectory pairs.

Run from the source root with base dependencies installed in `.venv`, keeping
the environment and source tree unchanged throughout. Use a fresh directory:
the generators overwrite matching output files, so the initial `mkdir` below
deliberately fails if that directory already exists.

```sh
set -eu
export JAX_ENABLE_X64=1
export PYTHONPATH=src
mkdir -p outputs
mkdir outputs/mash2-distributions-new
.venv/bin/python -m benchmarks.mash2_quantum_validation \
  --output outputs/mash2-distributions-new/mash2_quantum_validation
.venv/bin/python -m benchmarks.mash2_wavepacket --case low --mode mash \
  --mash-dt 1 --trajectories 1024 --batch-size 128 \
  --output outputs/mash2-distributions-new/mash2_wavepacket_low_dt1
.venv/bin/python -m benchmarks.mash2_wavepacket --case low --mode mash \
  --mash-dt .5 --trajectories 1024 --batch-size 128 \
  --output outputs/mash2-distributions-new/mash2_wavepacket_low_dt05
.venv/bin/python -m benchmarks.mash2_wavepacket --case high --mode mash \
  --mash-dt 1 --trajectories 1024 --batch-size 128 \
  --output outputs/mash2-distributions-new/mash2_wavepacket_high_dt1
.venv/bin/python -m benchmarks.mash2_wavepacket --case high --mode mash \
  --mash-dt .5 --trajectories 1024 --batch-size 128 \
  --output outputs/mash2-distributions-new/mash2_wavepacket_high_dt05
.venv/bin/python -m benchmarks.mash2_distribution_analysis \
  --results outputs/mash2-distributions-new \
  --output outputs/mash2-distributions-new/distribution_analysis.json
```

Each of the five input prefixes produces a JSON/NPZ pair. Keep the default
150 fs duration; the analysis rejects shorter runs. The trajectory driver uses
nuclear seed 4729, mapping seed 4730 and two event subdivisions. Matching counts,
source and runtime preserve paired initial samples across the requested
timesteps of 1 and 0.5 atomic units; the actual timesteps are adjusted to end at
exactly 150 fs.

The quantum generator evolves both preparations at all four declared grid/time
settings. It saves the `time_half` arrays—8192 points, requested timestep 0.25
atomic units, position box [−50, 90) bohr—for this analysis. The resulting
comparison reports method discrepancies and sampling uncertainty. This recipe
does not assert numerical agreement or supply an estimated execution time.
