# Extending a dynamics method

A method defines equations of motion, electronic-state meaning, nuclear feedback
and any event rules. It also determines valid initial preparation and physical
estimators. A different Hamiltonian provider or a different numerical splitting
of the same equations does not, by itself, define a new physics method.

The public boundary is structural: a class supplies `validate(problem)` and
`build_step(problem, integrator)`. It need not inherit a base class or register in
a central dispatch table. See the short implemented
[`CPA`](../src/pyeph/dynamics/cpa.py) and
[`Ehrenfest`](../src/pyeph/dynamics/ehrenfest.py) classes, then
[`MASH2`](../src/pyeph/dynamics/mash2.py) for method-specific preparation,
measurement and bounded event failures. The separate
[`MASHRM`](../src/pyeph/dynamics/mashrm.py) method implements multistate RM
preparation and estimators; these remain distinct from MASH2 even for two states.

## What a step receives and returns

`build_step` returns a pure function `step(state) -> state` for **one** trajectory.
Both states are `TrajectoryState` objects. Their array PyTree structure, shapes
and dtypes must remain stable under JAX tracing. The runner adds the independent
trajectory axis with `vmap`, advances steps with `lax.scan`, and schedules
measurements and output. The method does not implement its own host batch loop.

The conceptual contract is:

```text
MyMethod.validate(problem)
    reject unsupported physical model, nuclear treatment and estimator

MyMethod.build_step(problem, integrator) -> step
    use problem.model and runtime problem.params for model operations
    assemble compatible numerical kernels using static method/integrator policy

step(TrajectoryState) -> TrajectoryState
    advance q, p, electronic and any method_state
    advance time and step for a successful full step
    preserve trajectory_id and explicitly manage key if randomness is used
```

`problem.params` is passed dynamically into compiled blocks. Use the problem
received by `build_step`; do not close over a separately saved set of NN weights.
Static topology, tolerances and method choices belong in immutable Python
configuration. Dynamic arrays, active-state labels and method-owned counters
belong in `state.method_state`, which can be a nested array PyTree. All retained
method state is checkpointed. Temporary numerical-solver diagnostics instead
use the checked-step result described below; they do not replace user method state.

The assembled runner's `problem`, `integrator`, `execution` and `measurement`
properties are read-only. `update_parameters(params)` validates and replaces
only runtime numerical parameters, preserving the compiled-block cache. Build a
new simulation for changed static configuration. Custom provider/method objects
and captured data must remain immutable for the runner's lifetime; read-only
runner properties cannot enforce the internals of arbitrary external objects.

Keep I/O, Python mutation and global random draws out of `step`. Use JAX array
control flow for data-dependent branches and bounded loops. Stable trajectory IDs
and `execution.random.event_key` can define streams independent of batch/chunk
partitioning; a stochastic method must document its step/event/retry convention.

For a draw, use `event_key(state.key, state.step, event_id, typed=True)` and pass
the returned typed Threefry key to `jax.random`. The default `typed=False`
preserves the raw two-word return format for compatibility. Do not pass stored
raw keys directly to `jax.random.split` or other random operations: their
interpretation otherwise depends on the ambient JAX PRNG setting. If a method
advances its stored stream, first use `execution.random.as_threefry_key(state.key)`,
split the typed key, and store `jax.random.key_data(next_key)` in the returned
state. This preserves the `uint32[2]` state/checkpoint schema. Standalone sampling
functions receiving a caller's typed key may honor that key's implementation.

## Validation and failure hooks

The runner calls the following optional hooks when present:

| Hook | Where it runs | Responsibility |
|---|---|---|
| `validate_integrator(integrator)` | Host, during runner construction | Reject unsupported time-stepping policies before any initial output. |
| `validate_state(state, *, batch=False)` | Host, at run/checkpoint boundaries | Check state representation, normalization and method-specific initialization. `batch=True` means a leading trajectory axis. |
| `validate_initial_state(problem, state, *, batch=False)` | Host, after structural state validation at run/checkpoint boundaries | Check consistency with the current model and parameters before initial output or checkpoint publication. |
| `step_succeeded(state)` | JAX, per trajectory | Return a scalar boolean. False prevents the runner from replacing a partial-step failure's physical time with the nominal grid time. |
| `validate_result(state)` | Host, after each executed chunk | Inspect status/diagnostics and raise a meaningful failure before the chunk's output is delivered. |

`validate_initial_state` receives the current parameters because a state prepared
earlier can become physically inconsistent after `update_parameters`. MASHRM
uses it to check the real isolated spectrum, reference energy, active force and
largest-population ownership for every trajectory, including zero-step runs.
It never silently reselects the active surface; re-preparation is explicit.
An explicit pair boundary remains admissible for its event kernel to classify.
The hook is optional, and its presence should not be inferred for other methods.

For an event method, encode a failure in finite numerical state with a nonzero
status and stop subsequent physical updates. Merely returning `False` from
`step_succeeded` does not freeze dynamics or raise an error: the step and result
validator must implement those behaviors. Throw a `SimulationError`, or its
subclass, with `failed_state` and `last_valid_state=None`. The runner fills the
latter with the previous valid **chunk boundary**, not an inferred pre-event
state. A failure's actual last accepted time must remain unchanged. MASH uses
this pattern when localization or event capacity is exhausted.

The runner also checks ordinary state shapes and, by default, finite state values
at chunk boundaries. Such checks do not prove energy conservation, derivative
correctness or discovery of every possible crossing. Those are numerical and
scientific validation tasks for the new method.

Every returned chunk is synchronized before it can be published, including
`collect=False` runs with finite-value checks disabled. An exception during
compiled execution, a provider callback or synchronization becomes a
`SimulationError` with the original exception as `__cause__`, the chunk-entry
state as `last_valid_state`, and `failed_state=None`. Initial observation
evaluation has the same behavior with the validated initial state. Preflight
validation errors and host-side observer exceptions retain their original
types; an output-device failure is not a rejected dynamics step. Method-owned
`validate_result` errors retain their more specific status and failed state.

Measurements may implement
`validate_initial_state(problem, state, *, batch=False)` to check workflow-owned
origins and insertions against the current problem. This optional host preflight
runs after the method's state hooks at every run/checkpoint boundary, including
zero-step and no-output runs. It does not evaluate observations or modify the
state. A correlation origin prepared using different parameters can therefore
be rejected before any propagation, even when no values are requested.

Measurements may also implement a host-side `validate_observations(values)` hook.
The runner calls it after transferring each requested observation block and
before invoking observers or appending collected output. Initial values have
the trajectory shape; subsequent blocks add a leading saved-time axis. An
exception becomes `SimulationError` with the original exception as its cause.
For an initial failure, `last_valid_state` is the supplied state and
`failed_state` is `None`. A later failure retains the chunk-entry state and its
physically computed end state, while publishing none of that chunk. This hook
does not mutate method status or impose a universal finite-value policy on
arbitrary diagnostics. No observation evaluation or `validate_observations`
call runs when `collect=False` and no observer is supplied; the initial-state
preflight above still runs.

## Optional checked numerical steps

`Integrator(electronic=LanczosOptions(...), dt=...)` selects a separate numerical
acceptance path. A compatible method supplies
`build_checked_step(problem, integrator, *, batch=False)`, returning a pure
`step(state) -> (state, CheckedStepInfo)` function. CPA and Ehrenfest implement
this hook. Existing string integrators keep the ordinary `build_step` path.

The checked function must handle its batch axis itself: vectorize individual
action/force stages, reduce their statuses, and use scalar acceptance gates
between stages. Do not `vmap` an entire checked step; batched conditionals can
evaluate both branches and would undermine the intended runtime stopping rule.
Any rejected stage must return the complete macrostep input for every trajectory,
with fixed-shape diagnostics and no advanced nuclear, electronic or random state.

The runner stops subsequent macrosteps after the first rejection and delivers
none of the failed chunk's observations. It raises `SimulationError` with the
chunk-entry checkpoint state and the failed step's numerical diagnostics.
Finiteness and action-status acceptance are mandatory on this route, including
when optional legacy finite checks are disabled. External callback models and
RecordedCPA do not yet support it. See the [checked propagation
contract](CHECKED_PROPAGATION_DESIGN.md) for budget allocation, phase codes,
in-flight batch limitations and tests.

## Preparation and measurement belong to the method's meaning

Ehrenfest accepts a normalized electronic wavefunction and uses its expectation
value for the force. Current MASH2 instead stores a spinor encoding mapping
variables, uses an active adiabatic surface for nuclear motion, and requires a
population sampler and active-state population estimator. Their arrays can have
the same shape while representing different statistical experiments.
MASHRM instead pairs a conditional complex-sphere preparation with its affine
population/density estimator, including at two states. See its separate
[method and estimator contract](MASHRM.md); its samples and measurements are not
interchangeable with MASH2's.

A measurement supplies `validate(problem)` and
`evaluate(problem,state) -> array PyTree`. It must be pure and have a fixed output
schema: the runner may skip it on unsaved steps or entirely during no-output
execution. It must not advance random streams, update an accumulator hidden in a
Python object, or influence subsequent dynamics. Put necessary evolving
accumulators in explicit method/workflow state instead. The built-in HDF5
observer accepts nested dictionaries of numerical arrays; broader PyTrees need
an appropriate observer or conversion.

For simple wavefunction observables, `FunctionalMeasurement` is an optional
wrapper around a pure function. Methods whose electronic variables need a
special estimator should set `uses_mapping_estimator=True` so the default
amplitude-population measurement rejects them, and should validate their allowed
measurements explicitly. MASH2 additionally requires `supports_mash2=True` and
uses `MASHPopulation`. A marker does not establish scientific validity for an
arbitrary custom estimator.

The following complete calculation shows that substantive distinction using
the implemented method rather than introducing an unvalidated alternative:

```python
from pyeph import (CoupledClassical, Integrator, MASH2, MASHPopulation,
                   Problem, Simulation, configure_precision)
from pyeph.dynamics.mash2 import sample_adiabatic_population
from pyeph.models.analytic import TullyModel

configure_precision(enable_x64=True)
model = TullyModel(kind=1)
params = model.default_params()
problem = Problem(model, params, CoupledClassical(masses=2000.0),
                  MASH2(), measurement=MASHPopulation())
initial = sample_adiabatic_population(
    model, params, q=[-4.0], p=[20.0], active=0, seed=7, trajectory_id=0
)
simulation = Simulation(problem, Integrator(dt=0.5, electronic="exponential_midpoint"))
result = simulation.run(initial, steps=20)
print(result.observables["population"][-1])  # one trajectory's active-state indicator
```

An ensemble average of these indicators estimates the supported population
experiment. This short run is an API example, not a converged scattering
benchmark. See [MASH2's scope and validation](MASH2.md) and the
[independent event-driven reference](MASH2_REFERENCE.md). Do not replace its
sampler with a pure-state spin pole or reuse these samples for unspecified
coherence estimators.

## Integrating and testing an extension

Once a custom method implements this contract, pass its instance as
`Problem(..., method=your_method, measurement=your_measurement)` and supply its
prepared initial state to `Simulation.run`. No runner modification is needed for
the existing fixed-shape, fixed-step execution profile. Strict restarts require
an artifact identity for external method/measurement implementations, as
described in [the restart guide](RESTART.md).

Start with the method's uncoupled limit, an independently computed small-system
reference and time-step convergence. Check the appropriate energy/work balance,
preparation/estimator moments, batch versus single trajectories, and uninterrupted
versus restarted execution. An event method additionally needs accepted and
frustrated events, localization/subdivision convergence, explicit exhausted-bound
failures and physical failure times. Claim reversal symmetry only if the actual
numerical scheme passes its applicable reversal test.

Reuse numerical kernels in `integrators/` where their equations match the new
method. Adding an alternative integrator for Ehrenfest should retain Ehrenfest's
physical name and estimator; adding a distinct multistate mapping formulation
requires its own method, preparation and validation. Adaptive time steps,
variable electronic dimensions and growing event-state arrays exceed the
current runner contract and need a separate design rather than hidden changes
inside a step.
