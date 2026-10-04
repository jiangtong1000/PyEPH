# CPA correlations from propagated columns

`pyeph.workflows.column_transport` evaluates physical current correlations with
Hamiltonian and current **actions**. It lets a sparse model retain its sparse
representation through the transport workflow. The existing dense thermal
workflow remains the small-system reference and supplies its own thermal
preparation.

For an explicitly supplied positive density factor `rho0 = L @ L.conj().T`,
with unit trace and K columns, the new workflow propagates

\[
A(t)=U(t)L,\qquad B_b(t)=U(t)J_b(0)L.
\]

It measures

\[
C_{ab}(t)=\sum_k A_k(t)^\dagger J_a(t)B_{b,k}(t)
        =\operatorname{Tr}\{J_a(t)U(t)J_b(0)\rho_0U(t)^\dagger\}.
\]

The result is complex and unsymmetrized, with axes `(current_probe,
origin_probe)`. It is neither a symmetrized correlation nor an imaginary-time
Kubo transform. The probes already contain their physical charge, displacement
and unit conventions; the workflow inserts no conversion factor.

The electronic state has shape `(nstates, K*(1+nprobes))`. The implementation
does not construct a full Hamiltonian, density, current matrix or propagator.
A caller can still choose a full-rank factor, in which case the column storage
is naturally quadratic. Use an action-based electronic integrator too: the
ordinary `exponential_midpoint` option materializes a matrix, whereas RK4 and
the checked Lanczos action can retain the model's structured representation.

## Preparation is explicit

`initialize_column_transport_state(problem,q,p,factor,...)` accepts any finite
factor of shape `(nstates,K)` with `sum(abs(factor)**2) == 1` within numerical
roundoff. Columns may be nonorthogonal, unequal in norm or zero. The initializer
does not silently normalize them. A pure initial state is simply a rank-one
factor. A finite-temperature factor supplied by the caller is valid, but this
interface does not construct or certify its thermal approximation.

For finite-temperature preparation, the separate
[action-filter helper](THERMAL_COLUMNS.md) constructs such a factor from a
declared spectral enclosure and random columns. Its accepted polynomial error
compares with the exact filter applied to the same columns. Trace-rank error,
nuclear sampling, and propagation error still need separate convergence checks.

`initialize_infinite_temperature_columns(...,trace_ids=...,seed=...)` provides
one narrowly specified stochastic preparation. For independent complex random
phases `z_k` with `E[z_k z_k†]=I`, it forms

\[
L_k=\frac{z_k}{\sqrt{N K}},\qquad
\mathbb{E}[L L^\dagger]=I/N.
\]

The correlation is therefore an unbiased random-phase estimate at **beta=0**
for the supplied nuclear path. Every realized factor has unit trace up to
roundoff. There is no finite-temperature argument or estimated partition
function in this routine. Applying a thermal filter and normalizing each random
state separately would define a different, generally biased estimator.

Random keys use explicit Threefry streams and depend on seed, trajectory ID and
individual trace ID. Reordering
or partitioning trace IDs preserves their phases; each partition's factor
normalization uses its own K. Combine partitions with disjoint trace IDs using
their trace counts as weights; repeating an ID adds no independent sample.
Different nuclear trajectories still need distinct stable
trajectory IDs and their own physical nuclear preparation. Both initializers
retain a two-word Threefry state key. Process RNG configuration still enters
the scientific manifest, so changing it rejects an existing origin even though
these phase draws explicitly select their algorithm.

Trace columns on one nuclear path are not independent nuclear trajectories.
The returned correlation is one path sample, containing trace noise when that
preparation is used. Estimate conditional trace noise from independent trace
replicates at fixed nuclei, and nuclear-ensemble uncertainty from independent
paths. The workflow does not report a standard error that conflates these two
levels. When several trace estimates share a nuclear path, treating all of them
as independent paths would underestimate nuclear sampling uncertainty.

## Minimal calculation

```python
import jax.numpy as jnp
from pyeph import Integrator, PrescribedPath, Simulation, configure_precision
from pyeph.models.aggregate import AggregateModel
from pyeph.paths.harmonic import ConstantPath
from pyeph.workflows.column_transport import (
    make_column_transport_problem,
    initialize_infinite_temperature_columns,
)

configure_precision(True)
q = jnp.array([[0., 0., 0.], [2., .2, .1], [.8, 2., .3]])
model = AggregateModel(3, ((0, 1), (0, 2), (1, 2)))
problem = make_column_transport_problem(
    model, model.default_params(), PrescribedPath(ConstantPath(q)),
    probes=("current_x", "current_y"),
)
initial = initialize_infinite_temperature_columns(
    problem, q, jnp.zeros_like(q), trace_ids=[3, 17, 42],
    seed=11, trajectory_id=5,
)
simulation = Simulation(problem, Integrator(.02, electronic="rk4"))
result = simulation.run(initial, 100)
print(result.observables["current_correlation"][-1])
```

For a model without current probes, an explicit pure
`probe_callback(params,context,name,vectors)` can implement their actions.
`context` includes nuclear coordinates, physical velocity when available and
time. The result must preserve the vector/block shape. Linearity, Hermiticity,
basis consistency and the physical meaning of a probe remain provider
contracts; checking a few action columns cannot certify a full operator.

## Origins, diagnostics and restart

The state keeps the original time, coordinates, momenta, trajectory identity,
factor digest, per-column squared norms and preparation metadata. Origin
configuration hashes include the numerical parameters, probes, model, nuclear
treatment and source/runtime evidence. The optional measurement preflight
checks that identity before every run and checkpoint, including zero-step and
no-output runs. Updating parameters requires an explicitly new preparation;
the workflow does not infer a physical quench.

Opaque provider/callback captures still require the caller's immutability
discipline. The encoded origin alone cannot detect changes to unrecorded
captured data. Strict `Simulation.save_checkpoint`/`load_checkpoint` additionally
require complete external artifact identities, as explained in
[the restart guide](RESTART.md). Do not call an initializer when resuming: use
the saved final state to retain the original current insertions and lag time.

The diagnostic `column_norm_squared_drift` compares each propagated column with
its own initial squared norm. These columns generally are not orthogonal or
normalized individually, so `U†U-I` would be an invalid diagnostic here. A
small norm drift also does not bound phase or correlation error; converge the
electronic and nuclear time resolution independently.

This workflow requires prescribed-path CPA in a fixed orthonormal basis.
State-dependent Ehrenfest or MASH nuclear trajectories do not share the linear
propagator used in the trace identity above. Their response estimators require
their own preparation and derivation.

Platform and integrated release evidence are recorded in [qualification](QUALIFICATION.md).
