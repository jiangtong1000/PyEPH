# Coordinate domains inside checked dynamics

`CoordinateBox` declares a closed domain for the coordinates supplied to a
model. Scalar checked CPA and Ehrenfest test this domain before their internal
model evaluations. This is useful when a calculation has an explicitly accepted
coordinate range. Selecting bounds does not establish that a learned model is
accurate inside them.

```python
from pyeph import CPA, Execution, Integrator, LanczosOptions, Problem, Simulation
from pyeph.core.geometry import CoordinateBox

problem = Problem(
    model, parameters, nuclear_treatment, CPA(),
    geometry_guard=CoordinateBox(lower_coordinates, upper_coordinates),
)
simulation = Simulation(
    problem,
    Integrator(dt, electronic=LanczosOptions()),
    Execution(chunk_size=16),
)
result = simulation.run(initial_state, steps)
```

Bounds must have exactly the model's coordinate shape and use its coordinate
units. They are copied into immutable configuration. Membership compares the
integer ordering of finite IEEE binary64 inputs, including signed zero and
subnormal values. It does not subtract large coordinates or introduce a
distance tolerance. Coordinate checks therefore establish membership of the
actual stage inputs, not bounds on errors inside a provider.

The first supported path requires native JAX models, one trajectory, float64
coordinates/momenta/time, complex128 electronic amplitudes, and checked
`LanczosOptions`. CPA may propagate an electronic column block for that one
trajectory. Ehrenfest retains its single-vector preparation. The measurement
must be the exact built-in `ElectronicPopulation` class, which is the default;
custom measurements and subclasses are rejected because they may evaluate
unrelated coordinates internally. Mapping methods, ordinary RK4, trajectory
batches and derivatives through rejection are unsupported. Do not wrap a
guarded scalar step in `vmap`: a vectorized conditional can execute branches
that a scalar conditional would skip.

JAX still traces both branches when compiling the scalar conditional. Providers
must therefore remain pure and traceable; the guard gates runtime numerical
evaluation, not Python side effects during tracing.

Host validation checks the initial coordinates before model and measurement
preflight at that geometry. CPA additionally checks every frozen electronic
midpoint and the accepted endpoint. Ehrenfest checks its initial geometry and
the new geometry after the nuclear drift, before the second force and electronic
half. The predicate is outside differentiation of the physical force.

A rejected macrostep retains its input state and the first rejected geometry,
time, phase and electronic substep in transient diagnostics. It does not turn
the Hamiltonian or force into zeros. The runner stops the chunk and raises
`SimulationError` with its committed chunk-entry state in `last_valid_state`.
Its `failed_state` may be the input of a later macrostep inside that unpublished
chunk; restart from `last_valid_state`. Earlier accepted chunks and the valid
overall initial observation may already have been published. The rejected
chunk publishes no rows.

The guard enters strict checkpoint provenance. Changing its bounds is a new
configuration, so a checkpoint written under different bounds is not silently
accepted by strict loading. Retain the original checkpoint and make any
scientifically justified transition explicit. The runner does not enlarge the
domain or change the time step automatically.

The opt-in [scalar campaign continuation](CAMPAIGNS.md) preserves accepted
segments under that same fixed domain. Its durable restart point may precede
the runner's last internally accepted chunk; recovery does not enlarge bounds.

Run the [analytic replay example](../examples/guarded_dynamics.py) with
`JAX_ENABLE_X64=1 python examples/guarded_dynamics.py`. It demonstrates rejection,
byte-exact chunk rollback, and an explicit calculation with wider bounds on a
globally defined analytic model. It compares that replay with the same
unguarded equations. It supplies no material-model validation.

## Neighbor graphs remain a separate requirement

A box alone does not certify that an omitted pair has zero Hamiltonian, force
and probe contributions. Such a claim additionally depends on candidate
construction, provider support, floating-point arithmetic and consistent
parameter materialization. `CoordinateBox` performs no neighbor search,
eligibility inference, edge remapping or graph rebuilding. The
[neighbor guard and recovery design](NEIGHBOR_GUARD_DESIGN.md) remains the plan
for that additional workflow; these stage boundaries are one prerequisite.
