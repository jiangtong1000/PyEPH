# Public API and compatibility

The supported construction path is `Problem` → `Simulation` → `run`. The
top-level package exports the main method, state, integrator and execution
objects. Domain-specific preparation, transport and data interfaces live in
their named modules. Files and attributes beginning with an underscore are
implementation details.

| Responsibility | Public boundary |
| --- | --- |
| Physical dimensions and conventions | `SystemSpec`, `ModelSpec`, `UnitSystem` |
| Hamiltonian | `reference_energy`, `apply`; optional `prepare_action` |
| Nuclear feedback | `reference_gradient`, `contract_gradient`, `LowRankWeight` |
| Dynamics | `validate(problem)`, `build_step(problem, integrator)` |
| Calculation | `Problem`, `Simulation`, `Integrator`, `Execution` |
| State | `TrajectoryState`, `make_state`, `stack_states`; method-owned preparation |
| Measurement | physical model probes and method-compatible estimators |
| Restart | `Simulation.save_checkpoint`, `Simulation.load_checkpoint` |
| Ensembles | `run_ensemble`, `partition_ids`, `merge_ensembles` |
| Durable work units | `execution.campaign.Campaign` |
| Smooth sensitivities | `execution.differentiable.DifferentiableRollout` |
| Nonlinear preparation | `workflows.canonical_metropolis.NativeCanonicalMetropolis` |

`simulation.describe()` returns a JSON-ready summary without evaluating the
Hamiltonian or exposing parameter arrays. It records basis, units, state count,
method scope, integration algorithm, measurement and execution policy. This is
a configuration description; strict restart still uses the full provenance
manifest. The description does not certify model accuracy or future geometries.

## Parameter and state ownership

Numerical parameters are runtime PyTrees. Call `update_parameters` between runs
to validate a replacement while retaining compiled blocks. Built-in static
configuration is copied or immutable; opaque provider closures and their
external state remain caller-owned. Reconstruct a simulation for changed topology, masses,
method, integrator or execution settings. Initial states own their arrays.

## Compatibility policy

The package is a prerelease. Public breaking changes require a documented
migration and version change; private helpers may change as implementation
evolves. Do not change the physical meaning of an existing method or estimator
under the same public name. File schemas have their own explicit versions.

Strict trajectory restart checks numerical state, parameters, static conventions,
source and runtime identity. It does not silently resume after an upgrade.
When migrating, preserve the original checkpoint, reconstruct both conventions,
execute the relevant equivalence checks and export a new artifact with its own
identity. Matching manifests do not guarantee bitwise results across accelerator
or compiler environments; see [restart scope](RESTART.md).
Model-bundle revalidation has a separate limited pathway for unchanged
weights after implementation changes; it does not relax trajectory restart.

Ordinary native CPA/Ehrenfest force derivatives and smooth kernels support
automatic differentiation within their tested scope. The host operational
runner, external callbacks, hopping switches and checked solver branches do not
acquire an end-to-end derivative guarantee from that fact.
