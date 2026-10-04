# Canonical RM velocity correlations

This workflow composes a canonical preparation, `MASHRM`, and a physical
velocity provider. It implements the finite real-state RM prescription from
the [2024 transport paper](https://arxiv.org/html/2406.19851v2), with separate
tests of preparation, estimator algebra and workflow reduction.

## Canonical preparation

`LinearEPCCanonical` accepts exactly the native real `LinearEPCModel` with
exactly symmetric h0/couplings and strictly positive `omega`; it does not repair
input matrices implicitly. Its target is classical-nuclear joint canonical RM
equilibrium. For `h(q)=h0+sum(q_j*G_j)` and
`Vref=V0+sum(omega_j²*(q_j-qeq_j)²)/2`, the coordinate density is

\[
\pi(q)\propto e^{-\beta V_{\rm ref}(q)}Z_e(q),\qquad
Z_e(q)=\sum_a e^{-\beta\epsilon_a(q)}.
\]

Bare harmonic sampling omits the electronic partition-function weight. After
drawing q, momenta are independent Gaussians with covariance `M/beta`. The
active adiabatic surface is Boltzmann-distributed at that q. The complex mapping
vector is uniform on the sphere sector where that component is largest, then
rotated to the fixed basis. Harmonic stiffness is `omega**2`, without an extra
mass factor; actual oscillator frequency is `omega/sqrt(mass)`.

The bounded rejection sampler uses `x=q-qeq`, `g_j=||G_j||_2`,
`A²=sum((g_j/omega_j)**2)` and a Gaussian proposal of covariance
`diag(1/(beta*kappa*omega**2))`. Ordered Weyl bounds give
`Z_e(q)<=Z_e(qeq)*exp(beta*sum(g_j*abs(x_j)))`. Completing squares yields

\[
\log a=\log Z_e(q)-\log Z_e(q_{\rm eq})
-\frac{\beta(1-\kappa)}2\sum_j\omega_j^2x_j^2
-\frac{\beta A^2}{2(1-\kappa)}.
\]

Common energy offsets cancel before Boltzmann arithmetic. The default proposal
width minimizes the analytic envelope normalization. Exactly zero coupling
accepts the first proposal. This bound is specific to diagonal harmonic
stiffness; general positive-definite stiffness needs a different bound.
Exactness describes the target distribution in real arithmetic, not interval
certification. Strong coupling or low temperature can make rejection expensive.

Capacity exhaustion, nonfinite arithmetic or a detected envelope violation
raises `CanonicalSamplingError`. Its `.result` retains all IDs, statuses,
attempt counts and last/accepted coordinates. An accepted near-degenerate
spectrum fails the dynamics compatibility gate; it is **never discarded and
redrawn**, which would bias the coordinate distribution.

The sampler owns immutable input snapshots and versioned, purpose-separated
random streams. Reordering/partitioning stable uint32 IDs or increasing a
sufficient trial budget leaves successful draws unchanged.
`preparation_metadata(seed=...)` returns a fresh identity record without
drawing. It covers physical inputs, seed, algorithm, dependency versions and
numerical/RNG configuration. Changing a captured JAX setting requires a new
sampler. Trial budget is conservatively included in the identity, so different
budgets are not automatically merged even when individual successful draws agree.

## Velocity estimator

For fixed-basis c, active eigenvector u_a and physical velocity v,

\[
v_M=\sqrt{2/\Gamma_N}\,\mathrm{Re}
\big[(u_a^\dagger c)^*u_a^\dagger vc\big],\qquad
\Gamma_N=\frac{H_N}{N(N-1)}-
\frac{H_N^2+G_N}{(N+1)N(N-1)}.
\]

Here `H_N=sum(1/k)` and `G_N=sum(1/k**2)`. Gamma is the conditional fourth
moment `E[|c_a|²|c_b|² | a largest]`, b!=a; no extra 1/N enters. Its N=2,3,4
values are `1/6`, `47/432`, `67/864`. Matching only first moments with focused
sampling does not supply the needed fourth moment.

`RMVelocity` requires a finite Hermitian velocity with **zero adiabatic
diagonal**. Zero site-basis diagonal alone is insufficient. It checks full
small spectra and operators at every saved observation; Runner rejects any
invalid block before publication. The action-only `rm_velocity(c,u_a,v_c)`
helper assumes the physical restrictions have already been established.

The optional callback signature is
`(model, params, ProbeContext, probe_name, vectors) -> velocity_vectors`.
Otherwise the model's `probe_apply` supplies the action. Providers must define
physical units; probe names never convert charge or legacy current conventions.

`FixedPositionVelocity` supplies the finite-system commutator `i[h,X]`, hbar=1,
for a constant diagonal or full Hermitian position X. This does not define
periodic-image velocity or moving-center convective current. Wrapped positions
cannot silently replace a periodic transport operator.

## Public composition

```python
import numpy as np
from pyeph import configure_precision, Execution, Integrator, MASHRM
from pyeph.models.epc import LinearEPCModel
from pyeph.observables.transport.mashrm import FixedPositionVelocity, RMVelocity
from pyeph.workflows.mashrm_equilibrium import LinearEPCCanonical
from pyeph.workflows.mashrm_transport import RMTransport

configure_precision()  # Before creating double-precision arrays.
model = LinearEPCModel(3, 1)
params = model.create_params(
    [[-.4, .08, 0.], [.08, .1, .06], [0., .06, .8]],
    [[[.03, .01, 0.], [.01, -.02, .01], [0., .01, .01]]], omega=[.7],
)
sampler = LinearEPCCanonical(model, params, masses=[2.], beta=1.6)
velocity = RMVelocity(probe_callback=FixedPositionVelocity({"velocity": [0., 1., 2.]}))
transport = RMTransport(
    sampler, Integrator(.02, electronic="exponential_midpoint"), velocity,
    method=MASHRM(event_substeps=2), execution=Execution(chunk_size=16),
)
initial = transport.prepare(np.arange(128), seed=2026)
result = transport.run(initial, 50)
lag_times = result.statistics.times
correlation = result.statistics.mean  # (time, initial_probe, current_probe)
sampling_error = result.statistics.standard_error
transport.save_checkpoint("rm_transport.h5", result.final_state)
continued = transport.run(transport.load_checkpoint("rm_transport.h5"), 50)
```

For each independent trajectory k, the workflow forms
`C_k,ij(t)=v_k,i(0)*v_k,j(t)` **before** calculating means and Welford M2.
Multiplying ensemble-mean velocities is a different quantity. SEM uses
independent trajectories, not time origins, as samples. One trajectory gives
a mean and undefined sample variance/SEM. `merge_ensembles` combines disjoint
IDs only when preparation, scientific identity and lag grids match.

`observer(lag_times, values)` receives bounded host chunks. The added
`velocity_correlation` has axes `(time, trajectory, initial_probe, current_probe)`.
`collect=True` retains moments on the saved grid without full per-trajectory
history. `collect=False` retains no output history. With no observer and
`collect=False`, no velocity measurements are requested.

## Continuation and failure

`RMTransportState` contains the physical state plus an immutable
`RMCorrelationOrigin`. Origin IDs, v0 and t0 stay outside the exact MASHRM
method-state schema. ID order must match exactly. Parameter/probe changes
invalidate the origin. A complete manifest identifies model, parameters,
masses, units, method, integrator, measurement, source and runtime. Opaque
providers require explicit identities covering behavior and captured data.

Atomic HDF5 checkpoints store origin arrays in a separate checksummed auxiliary
tree. Such files use schema 2 so old readers reject them. The low-level loader
also refuses to discard auxiliary context unless `with_auxiliary=True` is
explicit. Ordinary files retain schema 1 and the existing two-value loading API.

A restart returns a **new segment**, preserving v0/t0 and the lag from the
original preparation. Adjacent segments both contain their shared boundary;
remove that duplicate explicitly when concatenating along time. The checkpoint
may end between regular saved steps; its forced terminal observation then adds
a sample absent from an unsplit run. Compare common times rather than assuming
identical output grids across arbitrary segment boundaries. The checkpoint
does not recover a writer's publication cursor or previously accumulated
statistics. It is not an automatic crash-consistent output database.

Runner/measurement failures raise `SimulationError` with physical states.
Wrap its `last_valid_state` with the unchanged origin for continuation. Host
correlation products or moments that exceed the output dtype raise `ValueError`
before the workflow observer sees that block; these postprocessing errors do
not carry a chunk state. External observer exceptions retain their original type.

Platform and integrated release evidence are recorded in [qualification](QUALIFICATION.md).
