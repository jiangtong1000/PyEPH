# Real finite-state RM mapping dynamics

`pyeph.dynamics.mashrm.MASHRM` implements the real finite-state formulation of
Runeson and Manolopoulos. It is a separate method from `MASH2`, with its own
preparation, observable estimator and event direction. Both names remain
explicit even for a two-state model.

This first implementation supports a fixed orthonormal model basis, a real
native JAX Hamiltonian, a complete isolated electronic spectrum, canonical
classical nuclei, and positive diagonal masses. It requires x64 without changing
JAX's global precision setting. Complete diagonalization is an explicit operation
requested by this method; sparse CPA/Ehrenfest models do not acquire a dense
Hamiltonian requirement. This is presently a small finite-state method, not a
large-band or spin–orbit implementation.

## Physical convention

The model supplies `V_ref(q)` and the carrier operator `h(q)`, with eigenpairs
`epsilon_a,u_a`. The stored mapping vector `c` is normalized and lives in the
fixed model basis. Define `z=U.conj().T @ c`. The active surface has the largest
`abs(z_a)**2`. Between events,

\[
\dot q_i=p_i/m_i,\qquad
\dot p_i=-\partial_i(V_{\rm ref}+\epsilon_a),\qquad
i\dot c=h(q)c.
\]

The omitted scalar-reference electronic phase has no effect on populations,
forces or impulses. It must be restored when comparing raw amplitudes with a
reference that evolves `(V_ref*I+h)c`. Conserved continuum energy is
`sum(p*p/(2*m))+V_ref+epsilon_active`, not the mapping expectation of `h`.

When populations `a,b` meet as the unique top pair, define

\[
\delta_i=\tfrac12\partial_i[c^\dagger(P_a-P_b)c],\qquad P_a=u_au_a^\dagger,
\]

holding the fixed-basis `c` fixed for this coordinate derivative. It contains
responses through **all** states, not only the pair NAC. The implementation
forms a rank-two `LowRankWeight` and requests one `contract_gradient`. It never
allocates a full electronic-matrix derivative tensor. The separate algebra
module preserves derivatives through its factors in larger smooth calculations;
the event-driven trajectory as a whole has no supported differentiable API.

In mass-weighted momentum space, change only the component along
`delta/sqrt(m)`. An allowed hop pays `epsilon_b-epsilon_a`; a forbidden hop
reflects this component and retains `a`. Coordinates and mapping amplitudes
are unchanged by the instantaneous impulse. No mapping projection,
renormalization, eigenvector phase tracking, gap floor or stochastic hop enters
this rule. Incoming and outgoing population-margin rates are checked explicitly.

These equations follow the [2023 RM formulation](https://arxiv.org/pdf/2305.08835)
and [2024 transport formulation, Eq. 26](https://arxiv.org/html/2406.19851v2).
The mapping measure and impulse depend on the declared full electronic space;
adding spectator states is not a neutral implementation detail.

## Preparation and observable meaning

`sample_population(...,population=k,basis="adiabatic"|"fixed")` draws uniformly
from the normalized complex sphere conditional on component `k` being largest
in the declared preparation basis. Swapping the largest component of an
unconditional sphere draw into position `k` implements this conditional measure
without rejection. Stable trajectory IDs determine the random draws independently
of batch partitioning. Nuclear q,p sampling is supplied separately.

The paper's conditional-sphere measure differs from the author's optional
focused sampler, even though both can reproduce the same initial density
moment. Coupled trajectories need the full preparation measure. The separate
`LinearEPCCanonical` workflow supplies joint canonical preparation for a
strictly confined real linear EPC model; see [RM transport](RM_TRANSPORT.md).

`mapping_state(...,mapping=c,basis=...,active=...)` constructs deterministic
diagnostic trajectories. It does not claim to sample a physical ensemble.
Active always refers to a sorted adiabatic surface; a fixed-basis prepared
population does not specify it. Tied initial largest populations require an
explicit active surface. After changing model parameters, reusing an initial
state is valid only if its active surface still satisfies the new model.

For `H_N=sum(1/k,k=1..N)`, set

\[
\alpha_N=(N-1)/(H_N-1),\qquad b_N=(1-\alpha_N)/N.
\]

`MASHRMPopulation(basis="fixed"|"adiabatic")` measures
`alpha_N*abs(c_in_basis)**2+b_N`. With `include_density=True`, it also returns
`alpha_N*outer(c_in_basis,c_in_basis.conj())+b_N*I`. Individual estimates can be
negative; neither squared amplitudes nor active-state indicators are the RM
population estimator. `active` remains an explicitly separate diagnostic.
Off-diagonal density elements in the adiabatic output basis use the instantaneous
eigensolver gauge; signs can change between geometries. Use fixed-basis density
for coherent time comparisons unless a separate gauge convention is supplied.
For a common prescribed unitary propagation, the conditional ensemble yields
the correct initial density moment and its unitary evolution in expectation.
That limit does not establish quantum accuracy for coupled nuclear motion.

The optional `mapping_observable` helper evaluates
`alpha_N*c.conj()@(O@c)+b_N*trace(O)` from an operator action and trace, including
complex coherence observables. Multiplying these one-time estimates is not a
general correlation-function prescription. Separate `LinearEPCCanonical`,
`RMVelocity` and `RMTransport` components implement canonical preparation and
single-origin velocity correlations under their explicit finite-model scope.
See [RM transport](RM_TRANSPORT.md) for the fourth-moment estimator, physical
probe restrictions and continuation semantics. Material mobility remains
separate work; existing CPA output must not be relabeled as RM transport.

## Numerical step and event bounds

Use `Integrator(dt, electronic="exponential_midpoint", electronic_substeps=1)`.
`MASHRM.event_substeps` divides the outer step. A fixed-active smooth segment
uses velocity Verlet for q,p and the dense exponential of `h((q0+q1)/2)` for c.
Every root evaluation recomputes this segment from its accepted start.

Each endpoint-bracketed active/competitor margin is localized independently.
Interior bisection requires `event_tolerance` on **both endpoint population
residuals** and `event_time_tolerance` on the final sign bracket. A prospective
impulse determines which side belongs to the outgoing surface: an accepted hop
uses the upper endpoint, while a reflection uses the lower endpoint. The smooth
state, spectrum and impulse are recomputed at that endpoint; the impulse outcome
must agree with the prospective one. The actual selected residual and time are
used in diagnostics and the remaining integration interval. The earliest root must
have a time bracket separated from the others; the associated pair must be
uniquely largest at the event. Ties among spectators below it are permitted.
Triple top ties, unresolved competing roots and grazing directions fail.

An incoming initial boundary resolves before drift only if both its population
residual and `abs(margin) <= abs(rate)*event_time_tolerance` are satisfied.
For a still-positive margin, this shortcut also requires a valid prospective
impulse and admissible outgoing ownership. A nonnegative outgoing margin is
admissible; a negative one must satisfy both the population bound and
`abs(post_margin) <= outgoing_rate*event_time_tolerance`. An uphill impulse can
slow the margin rate enough to invalidate the shortcut, in which case integration
continues to the actual crossing. This same ownership check applies to every
committed impulse.
A small positive margin farther away in time advances normally. A negative
margin farther away fails rather than silently starting on the wrong surface,
even if its rate points outward. No fixed population roundoff allowance bypasses
this time criterion at a slow crossing. An outgoing boundary can leave and later
return without a finite time nudge. An incoming terminal
boundary within the population tolerance and the local time criterion
`positive_margin <= (-incoming_rate)*event_time_tolerance` is resolved at the
endpoint only when the same prospective impulse and outgoing-ownership checks
pass. Otherwise a still-positive endpoint stays on the current surface until
a later actual crossing. Exact boundaries and sign-bracketed events must resolve
or fail; an invalid actual impulse is never silently deferred. The actual
population residual is retained. This terminal convention
is a local transverse-boundary tolerance, not an interior sign-bracket proof.
The instantaneous operation still leaves q,c unchanged. At resolved terminal
events, output uses the outgoing active surface. The initially supplied state
is reported as supplied before any step, including an explicitly incoming
initial boundary.

The incident parallel kinetic energy exactly at an upward-hop threshold, to
floating-point resolution, is rejected as unresolved. The test is
`abs(parallel**2-2*gap) <= 64*eps*max(parallel**2,2*abs(gap))`, for positive gap.
It avoids assigning a direction to a zero outgoing velocity based on roundoff.

All searches and event counts are bounded. Finite subdivisions do not prove
that an interval contains only one crossing or expose a leave-and-return pair
hidden between evaluations. A nonmonotone bracket may contain several roots;
incoming-rate checks reject an outgoing result but do not prove first passage.
Establish timestep, subdivision and tolerance convergence for each new model.
Choosing the outgoing side introduces a one-sided placement error bounded by
the retained time bracket; both endpoint population bounds still apply. No
exact finite-step reversibility, phase-volume preservation or universal
convergence theorem is claimed.

## Failure and integration contract

| Status | Meaning |
|---|---|
| 0 | Successful full step |
| 1 | Invalid/complex/non-Hermitian/degenerate model or force |
| 2 | Active surface disagrees with the largest mapping population |
| 3 | Residual/time localization requirement failed |
| 4 | Invalid, grazing, threshold or non-outgoing impulse |
| 5 | Per-step event capacity exhausted |
| 6 | Unresolved competing events or top triple tie |
| 7 | Integer diagnostic counter capacity exhausted |

On failure, q,p,c and time retain the last accepted segment start; rejected
proposals never replace them. Event diagnostics retain attempted work. The
nominal step advances only after the full step succeeds. Subsequent steps
freeze a failed lane. `MASHRMError`, a `SimulationError`, exposes this partial
state as `failed_state`; the runner supplies the preceding whole-chunk
`last_valid_state` and publishes none of the failed chunk's output.

Diagnostics include accepted/frustrated/total events, event attempts, summed
localization iterations, maximum event residual, maximum root bracket width,
and maximum impulse energy error. An attempt counts a root/impulse proposal or
capacity rejection; initial invalid model/ownership/grazing checks need not
count as attempted localization. Fields have canonical scalar dtypes and fixed
shapes, with a leading axis for batches.
Counters are int32. Proposed increments are checked with wide integer arithmetic;
overflow fails explicitly while retaining previous counters and physical state.
An unrepresentable amount of new diagnostic work is not reported as a successful
step or silently wrapped to a negative count.

The method uses existing structural Runner hooks, runtime parameter trees,
batching and strict HDF5 restart. It imposes no method dispatch table and adds
no Hamiltonian superclass. Under outer trajectory `vmap`, conditionals can
become selects: failed lanes retain their state, but this does not guarantee
that all pure provider computation for those lanes is skipped.
Callback providers, complex/SOC Hamiltonians,
nonorthogonal or moving bases, degeneracies, adaptive state spaces and general
thermostats are outside this method's present scope.

Platform and integrated release evidence are recorded in [qualification](QUALIFICATION.md).
