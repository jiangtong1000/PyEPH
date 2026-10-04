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
