# Guarded candidate graphs during dynamics

**Status: proposed neighbor-coverage and recovery design; automatic rebuilds
are not implemented.** Existing [candidate snapshots](NEIGHBOR_GRAPHS.md) can
check a supplied geometry and construct a replacement graph. The separate
[coordinate-domain stage checks](COORDINATE_DOMAINS.md) reject internal CPA and
Ehrenfest coordinates outside an explicit box, but do not certify candidate
coverage. This document specifies the additional provider, arithmetic and
recovery requirements for a guarded neighbor workflow.

The objective is to stop **before** evaluating a local Hamiltonian, force or
measurement at a geometry whose candidate coverage is no longer certified,
then repeat the unpublished interval with a valid replacement graph. This
controls candidate omission. It does not control integration error, prove a
learned model accurate, or certify an entire continuous nuclear path.

## First supported calculation

The first implementation should require all of the following:

- Native JAX `CPA` or `Ehrenfest` with `LanczosOptions`, retaining the checked
  route's float64 nuclear arrays/time and complex128 electronic state.
- One trajectory; CPA may retain its existing vector or column-block state,
  while Ehrenfest retains one normalized electronic vector.
- One `LocalBlockModel` constructed from one `NeighborGraph` generation, with
  fixed site/orbital count, atom-to-center map, electronic basis, carrier
  convention, units and physical cutoff. The skin must be positive.
- A fixed finite or periodic cell. Atomic coordinates remain coherently
  unwrapped in the snapshot's image convention throughout an attempted segment.
  Cell changes and coordinate rewrapping are explicit host transitions.
- A pure, candidate-invariant coefficient provider with an identified parameter
  materialization rule for every newly included `(a,b,image)` edge.
- Measurements evaluate the model only at the supplied state's coordinates.
  This includes their initial-state/preflight hooks. A measurement that probes
  displaced or historical origin geometries needs its own coverage contract
  and is excluded initially.

Measurements must also be pure and valid at the extra segment-initial/final
evaluation points introduced by the workflow. Shape inference and conditional
branches trace provider/measurement Python code even when a numerical branch
will be rejected. The guard prevents rejected runtime evaluations; it does not
prevent tracing of pure code.

A plain `LocalBlockModel` has `Vref=0`; that convention must stay explicit. A
useful first extension may accept exactly
`SumModel((guarded_carrier, reference))`, with the carrier identified explicitly
and a declared graph-independent reference whose electronic action is zero.
The reference must supply its complete gradient in the same coordinates and
remain unchanged during rebuilds. This is an explicit supported composition,
not recursive discovery of every local model inside an arbitrary `SumModel`.
Until that composition is qualified, reject it rather than treating a guard on
one child as protection for the whole model. Multiple independently changing
candidate graphs require a later joint-coverage and rebuild policy.

Permanent bond graphs are excluded. A graph declaring a fixed physical network
does not mean "all pairs within a radius"; adding a distance-selected edge can
change its Hamiltonian. An empty hopping coefficient or a finite cutoff alone
does not establish eligibility for automatic neighbor rebuilding.

## Coverage and provider responsibility

For reference centers `R0`, current centers `R`, fixed cell and consistent
images, the candidate set includes every unique Hermitian pair within
`cutoff + skin` at construction. Let

\[
 D=\max_a\|R_a-R^0_a\|.
\]

Every pair distance changes by at most `2D`, so `2D <= skin` is the mathematical
sufficient condition for retaining every pair currently inside the physical
cutoff. No minimum-image reduction is allowed in displacement from `R0`.
Uniform translation can conservatively exhaust this certificate.

The device guard computes center displacements without evaluating coefficients,
forces, electronic actions or a teacher. It returns finite numerical evidence
and a scalar acceptance predicate. Nonfinite coordinates, center arithmetic or
distance calculations cannot produce success.

The implementation must reserve a documented conservative margin for candidate
construction, reference-center conversion and device arithmetic, accepting only
when `2D + margin <= skin`. The margin must have length units, enter provenance,
and account for the actual summation and norm operations. A guessed tolerance
that merely enlarges the allowed displacement is insufficient. Borderline or
poorly conditioned geometry must reject if coverage cannot be established;
large absolute translations must never make acceptance more permissive.
Independent near-boundary tests are required before describing this as a
numerical certificate. Host exhaustive search remains a diagnostic/oracle,
not an operation inside the compiled step.

Candidate invariance is a separate physical requirement. At any covered
geometry, adding/removing edges outside physical support must preserve:

1. The Hamiltonian action in the same electronic basis.
2. The complete reference energy and contracted electronic/reference forces.
3. Every requested probe, including the image-resolved current.

`LocalBlockModel` multiplies raw hopping blocks by its smooth support once.
That does not eliminate a candidate's influence on onsite messages, global
normalization, attention denominators, descriptors or a baseline potential.
For example, dividing messages by the number of candidate rows changes onsite
energies when an inactive edge is inserted. Such a provider is ineligible.
All candidate-dependent paths must lose their effect smoothly at the declared
support, with the derivative order required by the calculation. Forces require
consistent first coordinate derivatives; higher sensitivities need additional
qualification. A provider-internal neighbor list is another graph and is not
covered implicitly by this guard.

The host factory must therefore provide explicit eligibility evidence: source/
artifact identity, support convention, parameter-key convention and tests of
candidate invariance. Neither an arbitrary Boolean attribute nor passing a
single geometry test proves an unknown neural provider satisfies the contract.
The initial implementation should accept only deliberately qualified providers.

## Where the current methods evaluate geometry

The stage locations below follow the implemented
[`CPA.build_checked_step`](../src/pyeph/dynamics/cpa.py),
[`Ehrenfest.build_checked_step`](../src/pyeph/dynamics/ehrenfest.py) and
[`dynamics.checked`](../src/pyeph/dynamics/checked.py).

| Stage | Required coverage gate | What must be skipped after rejection |
| --- | --- | --- |
| Host entry | `initial.q`, before model preflight or initial measurement | Model/provider calls and initial publication |
| Checked CPA substep `i` | `point(state, (i+1/2) * dt/substeps).q`, after finite path checks and before `electronic_action` | Coefficient preparation, all Lanczos actions and later path/model stages |
| Checked CPA endpoint | `point(state, dt).q`, before accepting the macrostep | Endpoint measurement, next macrostep and state commitment |
| Checked Ehrenfest first half | `q_n`, before `fixed_geometry_actions` | All first-half actions and subsequent nuclear stages |
| Checked Ehrenfest first force | The same `q_n`, with the half-evolved electronic state | First force, kick and drift if no valid certificate exists |
| Checked Ehrenfest second force | `q_new = q_n + dt * p_half/M`, after finite drift checks and before the second force | Second force/kick and the second electronic half |
| Checked Ehrenfest second half | The same `q_new` | All second-half actions if coverage was not accepted |
| Accepted endpoint observation | The accepted state's `q`, under the endpoint certificate | Observation/publication if no valid endpoint certificate exists |

The Ehrenfest electronic halves each freeze their coordinates. A successful
certificate can be reused within one macrostep while `q`, graph and center map
are unchanged: it covers all actions at that geometry and the force there.
The first force holds the intermediate electronic state fixed when taking the
coordinate derivative; it remains the existing complete reference-plus-carrier
force. The guard does not alter that derivative or replace the force.

The CPA endpoint needs a gate even on an unsaved step: the accepted state may
become a checkpoint, an observation or the next macrostep's input. CPA's
midpoint action locations differ from its endpoint. Endpoint-only checks are
therefore insufficient even when every accepted state is saved.

Path evaluation itself remains the nuclear treatment's responsibility. A guard
can reject its returned geometry before calling the Hamiltonian; it cannot
cancel work already performed inside an arbitrary path callback. The initial
scope uses native prescribed treatments with the existing time-span preflight.

## Checked status and ownership

Keep the geometry predicate beside `NeighborGraph` and its small device-facing
data. Keep the physical stage placement in CPA/Ehrenfest. The workflow owns
host rebuilds; the runner owns rollback and publication. Do not implement this
as a Hamiltonian wrapper that substitutes NaNs or zero matrix elements after
coverage expires: that loses the failure reason and can evaluate a force on an
invalid graph before rejection.

Add an explicit neighbor-coverage failure to the checked diagnostic contract,
for example code `4`, without renumbering current codes `0`–`3`. A new fixed-shape
coverage record should retain the first failing phase/substep, stage evaluation
time, maximum center displacement, allowed displacement/margin, failed-lane mask
and failing geometry. Only one failing geometry is needed, not a stage history.
The immutable snapshot's identity/generation belongs in host failure metadata.
The numerical cost of carrying this diagnostic must be measured.

The recorded stage time identifies the evaluated nuclear geometry. For the
split Ehrenfest halves, the phase/substep also identifies the frozen-geometry
electronic operation; do not interpret its geometry time as continuous coupled
motion inside that half step.

These are transient diagnostics, not new physical `method_state` fields.
`dynamics.checked` should own their status names and formatting. The runner
should not accumulate a second independent table of coverage and solver codes.
Preserve the first failure: a later budget/finiteness check cannot replace a
coverage rejection. Do not overwrite the last successful action record and
mislabel it as a failed Lanczos action.

The checked method receives an optional immutable coverage guard through its
builder; existing unguarded calls retain their behavior. Internal block
construction passes that guard explicitly. No changing NN weights belong in
the guard: parameters remain runtime PyTrees. Guard evaluation stays outside
the differentiated provider energy/force computation.

For the first implementation, the workflow requires `LanczosOptions`, so the
runner's existing checked-route selection suffices. Guard failure must pass
through the same scalar acceptance gates as other checked failures. Every
physical state field, including momentum, electronic amplitudes, time, step,
trajectory ID and random key, returns to the macrostep input on rejection.
The checked scan then stops later macrosteps.

## Host segment transaction and recovery

An attempted segment is one bounded runner chunk. Its entry state is the last
committed state. Run the attempt with `observer=None` and collect its bounded
output internally; publish nothing until it succeeds. Initial output is also
buffered, since the existing runner normally evaluates it before propagation.
Set the runner chunk size to the attempted segment length, including the final
remainder, so its failure rollback identifies that same committed entry. Before
the first segment, validate the entire requested run span through the existing
shared host validator. A later segment must not discover a predictable path
domain, counter-overflow or clock-resolution failure after earlier publication.
The workflow checks initial coverage before calling runner preflight, which
can itself evaluate model coefficients.
The starting snapshot must certify the supplied initial geometry. Reject an
invalid starting snapshot before provider evaluation; construct a valid initial
generation explicitly before entering this workflow. Automatic retry applies
to a segment beginning from an already certified committed state.

The current checked runner already suppresses observations from a rejected
chunk. Its `SimulationError.last_valid_state` is the **chunk entry**;
`failed_state` may be the input of a later failing macrostep within that chunk.
Recovery uses the committed segment entry, not `failed_state`, the failing
geometry or a partially advanced electronic state.

On a coverage failure:

1. Retain failure evidence and discard all attempted output.
2. Rebuild candidates at the committed entry coordinates. Refreshing the anchor
   at the same skin may suffice. If one macrostep itself exhausts the skin,
   repeating an identical rebuild cannot make progress.
3. Apply only caller-authorized, bounded skin/capacity choices. A larger skin
   requires a newly constructed snapshot with explicit parent identity; the
   current `NeighborGraph.rebuild` only changes reference coordinates/capacity.
   Keep physical cutoff, time step, equations and preparation unchanged.
4. Construct replacement parameters by full edge key and a new model/runner.
   Validate provider outputs and compare old/new action, complete force and
   probes at the rollback geometry, where both graphs must represent the same
   physical model. Numerical equivalence uses declared tolerances.
5. Retry the whole segment from the identical entry state. Stop with its last
   committed checkpoint if the retry limit, edge capacity, image-search budget
   or candidate-invariance checks fail. Never prune edges to fit a budget.

Other failures retain their existing meanings. Do not rebuild automatically
after a rejected electronic action, invalid force, provider exception or I/O
failure. The first implementation must not silently reduce `dt` or accept an
uncertified stage to finish a run.

After success, publish rows on the original absolute-step output schedule.
Segment restarts introduce duplicate initial rows, and the runner saves each
call's final row even if it is off cadence. Remove those bookkeeping duplicates
and extra internal segment endpoints by global step index; retain the overall
initial/final rows. Do not deduplicate by approximate floating-point time.

This gives atomic **numerical acceptance** of one bounded segment. Arbitrary
observer side effects are not transactional. The first workflow should either
return accepted segments to its caller or persist immutable segment artifacts
before marking them committed; it must not promise rollback of partial HDF5
appends or external writes. An I/O exception stops publication and needs explicit
recovery. Durable multi-process claims remain the campaign layer's job.

## Parameters, snapshots and restart lineage

Candidate rows are not stable parameter identities. Canonical keys include both
sites and the full image vector. For retained keys, preserve the corresponding
physical parameters; for new keys, require the declared rule or source data
that supplies them. Never reuse an old array by row number, truncate it, or fill
new interactions with zero because no parameter was available.

Shared neural/radial weights can remain unchanged when the provider is proven
candidate-invariant. Site/atom ordering and electronic state dimension must
remain fixed. A host remap record should identify retained, added and removed
keys and the source of each newly materialized parameter. Parameters stored as
arrays must match the replacement graph's order and shapes.

Each committed generation records old/new snapshot identities, parent link,
reference coordinates, skin/capacity/numerical margin, model/provider/parameter
identities, remap evidence, rollback state checksum, global step/time, retry
reason and the accepted segment's output identity. Reference-anchor refreshes
remain new generations even when edge membership is unchanged.

A strict checkpoint loads with the exact generation/model manifest that wrote
it. Rebuilding is a subsequent explicit workflow transition to a different
manifest, with the old checkpoint retained. Do not relax manifest equality or
claim bitwise continuation across a graph-order/kernel change. Require exact
checkpoint storage/load integrity and physical equivalence within declared
floating-point tolerances. Source and runtime changes remain separate migrations.

The ordinary model manifest binds `CoordinateBox` configuration but does not
identify a `NeighborGraph` generation or its candidate/arithmetic/rebuild
lineage. An anchor or skin change can leave its graph and model manifest
identical. Therefore a
workflow checkpoint envelope must additionally bind snapshot identity, guard
arithmetic/margin policy, model/parameter identities, state checksum and absolute
output-step policy. Validate this envelope before ordinary checkpoint preflight
can evaluate the provider. Failed rebuild attempts belong in the attempt ledger;
they must not become parents in the chain of committed generations.

## Proposed public workflow sketch

The following is an interface sketch, **not runnable current API**. Introduce
one bounded workflow rather than a new universal Hamiltonian or callback registry:

```python
# Proposed name and arguments; not implemented.
for segment in run_guarded_neighbors(
    build_problem=build_for_candidates,
    candidates=initial_candidates,
    initial=initial_state,
    integrator=Integrator(dt, electronic=LanczosOptions(...)),
    steps=1000,
    segment_steps=16,
    save_every=8,
    rebuild_skins=(1.0, 2.0, 4.0),  # explicit retry choices, in coordinate units
    max_edges=20_000,
    max_retries_per_segment=3,
):
    save_accepted_segment(segment)  # caller-owned durable output
```

`build_for_candidates(snapshot, previous)` is a pure host factory returning the
`Problem`, provider/artifact identities and parameter-remap evidence. `previous`
contains the previous accepted generation and its parameters, or is `None` for
initial construction. Eligibility/support conventions must be supplied and
validated before this factory is used for a run. The workflow controls retries
and preserves state; the factory cannot advance nuclei, electronic state or
random streams. The initial snapshot's capacity and every allowed enlargement
must fit the declared edge budget. The first version does not accept an arbitrary
live observer during an attempted segment.

Validate the factory's unchanged physical configuration explicitly: basis,
atom/site ordering, cutoff and `switch_on`, units/carrier convention, method,
nuclear treatment/masses, preparation and measurement. Agreement at the rollback
geometry is a sanity check alongside provider invariance evidence; it is not
proof of equivalence at future geometries or for untested electronic states.

Each yielded record contains the accepted final state, globally filtered output
rows and generation/retry manifests. Yielding provides bounded numerical
segments, not automatic durable commitment. The caller must stop on a failed
write rather than advancing the iterator and treating the segment as stored.

## Shape changes, batching and the RK4 follow-up

Graph membership and center-reference data are static for one compiled segment.
A rebuild creates a new model/runner even if the edge count happens to match;
changed edge contents or reference centers invalidate captured constants.
Different counts may compile different executables. `update_parameters` does
not update topology. Existing `capacity` is a hard edge limit, not a padding
shape. Dummy edges, masked capacity arrays and compilation-cache reuse need a
separate implementation/qualification; they are not implied by this design.

The initial workflow is scalar. A later common-graph batch must cover the union
of all lane requirements with explicit per-lane reference centers and a union
capacity budget. One lane's failure rejects the whole batch and restores all
keys/IDs. Reduce coverage to a scalar **before** entering a vectorized provider
stage; `vmap` over whole conditional steps can execute both branches. Different
lane topologies cannot simply be stacked into unequal shapes. Independent
scalar runs or declared same-topology cohorts remain valid alternatives.

RK4 support is a distinct follow-up. Ordinary CPA evaluates each substep at
`t_i`, `t_i+h/2` twice and `t_i+h`; a guard must precede every new evaluation
geometry, as well as the accepted endpoint. Ordinary Ehrenfest still uses only
`q_n` and `q_new`, despite its RK4 electronic microsteps. Sharing those coverage
locations must not merge the methods' physical equations.

The current ordinary `build_step` cannot return a recoverable stage rejection,
and the runner selects checked blocks from `LanczosOptions`. Guarded RK4 needs
an explicit checked-step/status path independent of electronic solver choice.
It cannot be implemented safely by wrapping the output observer. For known
prescribed paths, a host can preflight all CPA evaluation geometries for a
bounded interval; that is a useful restricted alternative, not general coupled
Ehrenfest recovery. Retrospective endpoint inspection does not prevent an
invalid provider call that has already happened.

## Falsification and acceptance gates

| Gate | Required adversarial evidence |
| --- | --- |
| Certificate | Compare against independent finite/periodic image enumeration, including skew cells, self/multiple images, exact-boundary roundoff and precision-losing coordinate conventions; no accepted stage may omit an inside-cutoff pair. |
| CPA hidden stages | A prescribed path exits coverage at a midpoint and returns inside at the endpoint; reject before the midpoint provider action despite valid saved endpoints. |
| Ehrenfest drift | `q_n` is covered and `q_new` is not; reject before the second force and all later electronic actions. Preserve every initial state field. |
| Gate ordering | Test host initial failure, each action/force phase, endpoint failure and failure precedence. Verify rejected provider branches are not executed, rather than merely returning an error after evaluation. |
| Output transaction | Fail late in a chunk; no initial/intermediate rows from that attempt reach the consumer. Retry reproduces one global output schedule with no missing/duplicate rows. |
| Rebuild equivalence | Use a radial provider and an independently constructed sufficiently large fixed graph; compare complete CPA/Ehrenfest trajectories, currents, norm and applicable energy/convergence diagnostics at matched time-step accuracy. |
| Invalid provider | Insert an inactive edge into a candidate-count-normalized onsite model; reject eligibility/equivalence instead of silently changing the Hamiltonian. |
| Remapping | Reorder retained edges, add periodic images and require new per-edge parameters. Wrong image/row mappings and absent new-key data must fail. |
| Capacity/retries | Exhaust skin within one macrostep, edge capacity and image-search budgets. Bound retries, preserve the last committed state and never truncate interactions or change `dt`. |
| Restart lineage | Reload an exact committed generation, then perform an explicit rebuild transition. Wrong generation, weights, margin or parent/remap evidence must reject; checkpoint bytes must round-trip exactly. |
| Unsupported scope | Reject permanent bond graphs, undisclosed provider-internal lists, arbitrary composites, callback providers, variable cell, unsupported measurements and multi-lane inputs in version one. |
| Execution evidence | Test compiled and uncompiled paths; record guard/rebuild/compilation cost for a complete matched-accuracy workflow. CPU success alone establishes no accelerator behavior or speed claim. |

Any later multi-lane implementation additionally needs a failing lane whose
provider must never be entered, union coverage, all-lane rollback and stable
trajectory-ID/random-key tests. RK4 requires separate internal-stage tests.
Passing the first checked-Lanczos milestone does not claim those extensions,
automatic uncertainty control, general graph differentiability, or production
neighbor-list scaling.
