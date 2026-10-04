# Differentiable smooth dynamics

`pyeph.execution.differentiable.DifferentiableRollout` exposes the native CPA
and Ehrenfest method steps as a pure JAX computation. It supports gradients
of trajectory observables with respect to explicit model parameters and
floating-point initial data. Its scope is fixed-step smooth-method sensitivity,
not differentiable hopping or a new dynamics approximation.

The operational `Simulation` API remains responsible for checked execution,
host diagnostics, streaming, checkpoints and restart. Construct and validate
a differentiable rollout outside transformations; use its pure call inside
`jax.jit`, `jax.jvp`, `jax.grad` or an outer `jax.vmap`.

## Interface

```python
from pyeph.execution.differentiable import DifferentiableRollout

rollout = DifferentiableRollout(
    problem, integrator, initial,
    steps=100,
    rematerialize=True,
)

# Host validation for a concrete changed parameter/state pair, outside AD/JIT.
rollout.preflight(new_parameters, new_initial)

# No host validation or conversion occurs in this call.
result = rollout(new_parameters, new_initial)
```

`initial` is the existing `TrajectoryState` PyTree. The constructor checks the
representative parameters/state and fixes whether the call handles a single
trajectory or a batch. `result` is a device PyTree with `final_state`, `times`
and `observables`. Observations use the problem's existing measurement, or
`ElectronicPopulation` when none is supplied. The first output is the initial
sample; subsequent samples follow every fixed step, including the final state.
Zero steps returns that initial sample unchanged.

Parameters remain explicit runtime PyTrees. Replacing them never requires
reimplementing the equations or recreating the rollout for the same structure.
Initial coordinates, momenta and electronic amplitudes also remain runtime
data. Method, model implementation, measurement, timestep, electronic substeps,
number of nuclear steps and nuclear-treatment configuration are static.
Treat these objects and their captured arrays as immutable. Explicit preflight
freshly traces frozen representative input shapes and checks that operations
and closed numerical constants have not changed, including derivative programs.
If static model data changes, construct a new rollout before compiling it.
The pure call intentionally performs no host guard; mutating a captured object
and calling an old compiled function without preflight is unsupported. This
process-local trace check is not a portable artifact or checkpoint identity.

## Observable losses and physical constraints

```python
import jax
import jax.numpy as jnp

def loss(parameters, q0):
    state = initial._replace(q=q0)
    prediction = rollout(parameters, state).observables["population"]
    return jnp.mean((prediction - target_population)**2)

value_and_gradient = jax.jit(jax.value_and_grad(loss, argnums=(0, 1)))
value, (parameter_gradient, initial_coordinate_gradient) = value_and_gradient(
    problem.params, initial.q,
)
```

The loss must be a real scalar for `jax.grad`. Differentiate selected floating
leaves rather than the entire state, whose IDs, counters and random-key fields
are discrete bookkeeping. Preserve Hermiticity, positive physical parameters
and normalized Ehrenfest initial electronic states through the parameterization
being optimized. For example, varying a normalized state's angle is meaningful;
unconstrained variations of its two amplitudes may leave the physical manifold.
For complex parameter derivatives, follow JAX's real-output complex cotangent
convention and verify directional derivatives before interpreting components.

The same complete force used by operational Ehrenfest is retained:

\[
F(R,c)=-\nabla_R V_{\rm ref}(R)-\nabla_R\langle c|H(R)|c\rangle.
\]

The inner coordinate derivative holds the electronic state fixed. The outer
trajectory derivative must still include the electronic state's response,
along with every baseline, descriptor and residual derivative. A callback that
provides a correct first force but lacks higher derivatives is insufficient
for force sensitivities. The native residual/composition regression exercises
parameter gradients through this feedback.

CPA does not gain nuclear feedback by differentiating it. With `HarmonicBath`,
each initial coordinate/momentum changes its own prescribed harmonic path.
With a shared `PrescribedPath`, the path's stored coordinates are static; merely
varying the state's initial `q` does not turn that stored path into a dynamic
parameter. Any such optimization must respect the selected path semantics.

## Batches and rematerialization

Pass `stack_states(...)` as the representative initial state to build a batched
rollout. Each trajectory advances independently through `jax.vmap`, sharing
the parameter PyTree. Output axes are `(time, trajectory, ...)`. Stable IDs must
be unique and initial step counters synchronized, as in the operational runner.
Initial times may differ between trajectories. Define ensemble losses with an
explicit average; the gradient then follows that exact finite sample estimator.

For separate parameter sets, an outer `jax.vmap` over a single-trajectory
rollout is possible. Validate representative concrete inputs first. Batching
does not introduce a new estimator or differentiate a parameter-dependent
sampling distribution automatically.

`rematerialize=True` applies `jax.checkpoint` to the scan body, trading
recomputation for saved intermediate reverse-mode data. It preserves values
and gradients in the tested cases. All requested observables are still returned
at every step; this interface does not claim bounded memory for long trajectories
or a measured speed/memory advantage on a particular device.

## Supported numerical boundary

- Native CPA and Ehrenfest only. Event-driven methods and custom method
  subclasses are rejected at construction.
- Fixed-step electronic RK4 only. The complete Ehrenfest scheme remains its
  existing second-order electronic/nuclear splitting, even with RK4 electronic
  half steps. CPA's electronic integration has fourth-order convergence for
  the smooth prescribed path used in the reference test.
- Checked Krylov solvers and dense eigendecomposition-based exponential steps
  are rejected. Eigenvector derivatives can be singular at degeneracies even
  when the matrix exponential is smooth. Supporting that path requires a
  separately qualified matrix-function derivative.
- Models must declare native JAX execution. Traced pure, I/O and debug host
  callbacks are rejected in primal, forward-JVP and reverse-gradient programs,
  including nested scan/call programs and derivative-only custom rules.
  Providers/measurements must support both forward and reverse differentiation.
  Construction does not infer the correctness of a custom derivative rule.
- Importing the module does not change precision. Select precision explicitly
  before constructing arrays. There are no silent device-to-host conversions
  in the differentiable call.

Preflight checks input compatibility, counter and clock capacity, conservative
clock resolution, the requested span against a prescribed path's known domain,
representative static computation identity, traceability and hidden callbacks.
Nonzero propagation requires float32 or float64 clocks. The shared
[clock-resolution policy](RESTART.md#clock-resolution-for-native-geometry-propagation)
includes CPA's internal midpoint spacing, checks every batch lane, and rejects
unresolvable absolute time origins before tracing. Zero-step requests skip the
resolution check. All clock intervals use the same canonical time unit as `dt`;
passing the check does not establish integration accuracy. This remains an
explicit host preflight: the pure differentiable call does not repeat it when
given new runtime inputs.

Preflight does not certify
the complete future trajectory or automatically
check finite values during transformed execution. Recheck optimized candidates
with operational runs, timestep refinement, domain diagnostics and the relevant
physical validation. Hard cutoff changes, state selection and nonsmooth custom
measurements can make a derivative invalid even when JAX returns a number.

## Runnable example and evidence

The public example uses only a generated two-state, one-mode parameterized
Hamiltonian, a harmonic nuclear reference and a fixed orthonormal carrier basis
in canonical atomic units. It requires no external model or archived dataset:

```sh
JAX_ENABLE_X64=1 python examples/differentiable_dynamics.py --output fresh_sensitivity_run
```

The example compares a coupling/initial-coordinate loss gradient with central
finite differences and checks population values against operational Ehrenfest.
Its output directory must be new. The example demonstrates numerical sensitivity;
it is not a material model or a claim of learned transport accuracy.

`tests/test_differentiable_rollout.py` additionally checks gradients against an
independent NumPy/SciPy solution of nonlinear CPA and Ehrenfest equations, including
timestep convergence, changes to all model parameters, initial coordinates,
momenta and the electronic preparation angle. It checks batched/single parity,
rematerialization, native NN feedback derivatives, exact electronic degeneracy
under RK4, and rejection of unsupported methods and callbacks.

## Fitting parameters through dynamics

The independent calibration benchmark fits two positive physical parameters
through batched rollouts, with separate CPA and Ehrenfest cases:

```sh
JAX_ENABLE_X64=1 python benchmarks/dynamics_calibration.py --output fresh_calibration_run
```

Its generated fixed-orthonormal-basis Hamiltonian is
`H(q) = (0.04 + g*q)*sigma_z + delta*sigma_x`, with harmonic reference
`V0(q) = 0.35^2*q^2/2`. Here `q` is a canonical normal coordinate with chosen
unit mass; `g` is energy per canonical length, `delta` is in Hartree, time is
in atomic units and hbar equals one. This is not an atomic-geometry label
contract. The bias, reference potential, mass, basis and positive coupling
convention are known and fixed.

The benchmark fits `(g, delta)` in bounded log coordinates, starting at
`(0.06, 0.15)` against noiseless targets generated at `(0.12, 0.09)`. An
independent SciPy DOP853 solver implements explicit NumPy equations without
calling production Hamiltonians, forces or integration steps. Three fixed
training initial conditions determine the loss; two other initial conditions
are held out. Observations are three Bloch components, plus nuclear position
and momentum for Ehrenfest. The fixed protocol uses 160 steps of 0.05 atomic
time and 21 observation times. No fit settings are chosen from holdout results.

The run checks finite numerical evidence, optimizer convergence, central finite
differences of the discrete loss, independent teacher gradients, teacher
refinement, parameter recovery and local sensitivity rank. A half-step audit
retains the fitted parameters without refitting. It compares integration errors
against a new teacher trajectory at those same parameters, separating numerical
error from parameter-estimation error. This distinction matters because a fit
can partly compensate for timestep error. All gates are declared before fitting.
Output must be new; protocol, inputs, source hashes, source payloads, trajectories,
optimizer history and failed numerical evidence are retained.

On local macOS arm64 CPU with Python 3.13.16 and JAX 0.11.2, the two cases
recovered the target parameters with maximum relative errors below `9e-10`
(CPA) and `1.4e-5` (Ehrenfest). Held-out scaled RMSEs were approximately
`3.2e-10` and `1.2e-5`. Ehrenfest's independent-gradient and same-parameter
integration errors decreased by approximately four when halving the timestep,
consistent with its second-order splitting. This demonstrates a complete
generated-model calibration workflow, not material accuracy, global parameter
identifiability or a new dynamics method. The local source-bound run archives
are not distributed; the command above regenerates the protocol.


## Full-gradient windows and higher sensitivities

The generated-model benchmark composes fixed-length `DifferentiableRollout`
windows with `jax.checkpoint` around their carry updates. It accumulates one
scalar objective and retains the complete state tangent between windows; it
does not require a new runtime API or a trainer. Run in explicit float64 mode
and choose a fresh output directory:

```sh
JAX_ENABLE_X64=1 python benchmarks/windowed_sensitivities.py \
  --output outputs/windowed_sensitivities_run
```

The protocol uses a real fixed-orthonormal two-state Hamiltonian: a linear
spin-boson term plus a generated tanh network with two eight-unit hidden layers.
There is one canonical normal coordinate with mass 1.4, Hartree energies and
atomic time. The nuclear reference is
`V_ref = omega**2 * (Q - Q_eq)**2 / 2`. CPA uses an independently prescribed
harmonic frequency 0.6; Ehrenfest uses the stated reference and full electronic
force. These are separate calculations, not identical nuclear trajectories.

For 64, 512 and 2,048 steps at `dt=0.02`, with 16 steps per window, the objective
averages `(population[1]-0.3)**2 + 0.07*Q**2 + 0.02*P**2` over all positive steps
and adds `0.03*Q_final**2` once. Each window excludes its initial observation,
so boundary samples are not counted twice. Clock anchoring changes floating
association; value and gradient equivalence is tested numerically, not claimed
bitwise. A deliberately detached-state control preserves the loss value while
breaking its gradient and must be detected.

At the shortest horizon, the protocol also checks first directional gradients
and Hessian-vector products with central-difference widths `1e-3` and `5e-4`,
refinement with a declared floating-point allowance, Hessian symmetry, and
monolithic/windowed Hessian-vector agreement. An independently coded NumPy tanh
matrix and analytic coordinate derivative feed an explicit reference using the
same discrete algorithms: time-dependent electronic RK4 for CPA, and RK4
half steps around the Ehrenfest velocity-Verlet update. No production force or
propagator is used by that reference. Its primal and scalar-loss directional
finite differences provide separate checks. Longer cases test value/gradient
equivalence only; these checks do not establish continuum accuracy, long-time
conditioning or material accuracy.

The recorded local CPU run on native Python 3.13 with JAX 0.11.2 passed all six
cases. At 2,048 steps the largest monolithic/windowed gradient difference was
`1.06e-15`. Detaching window states changed the short-horizon gradient by about
70% for CPA and 76% for Ehrenfest while preserving the objective value. These
numbers concern this generated fixture and stack; the script regenerates its
own complete numerical record rather than requiring a shipped historical
record.

Parameters, initial state, directions, finite-difference pairs, HVP arrays,
source snapshots and hashes are retained. Nonfinite arrays are saved with
explicit failure diagnostics before rejection. The report includes compiler
memory estimates for arguments, outputs, aliases and temporary buffers. For
the tested 2,048-step cases, temporary-buffer estimates changed from 3,294,400
to 13,824 bytes (CPA) and 6,145,952 to 14,696 bytes (Ehrenfest) between the
monolithic and windowed versions. These are compiler estimates for this small
fixture, not process peak RSS, accelerator evidence or a universal scaling
bound. Raw local timing samples are not performance claims.
