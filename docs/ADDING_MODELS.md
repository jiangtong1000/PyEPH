# Adding a Hamiltonian provider

A model supplies a physical energy definition and numerical operations. Dynamics methods do not need to know whether those operations came from an analytic expression, a site graph, orbital blocks, or a neural network. The initial native contract is a fixed orthonormal electronic basis with

\[
\hat H=T_{\rm nuc}I+V_{\rm ref}(Q)I+h(Q).
\]

For Ehrenfest, the corresponding trajectory energy is
`T_nuc + V_ref + c†h c` for normalized `c`. MASH instead uses the active adiabatic
surface energy; its mapping amplitudes do not define an Ehrenfest energy or
population estimator. Prescribed-path CPA need not conserve a coupled
electron–nuclear energy.

Coordinates use the declared canonical convention and a consistent unit system with hbar=1; atomic units are the default and `ModelSpec.unit_system` may declare reduced energy/length scales. A declaration does not convert arrays automatically; reduced time is in units of hbar divided by the declared energy scale. Nuclear masses belong to `CoupledClassical`, not the Hamiltonian object. A model of raw Kohn–Sham matrix elements, a moving atomic-orbital basis, or an uncalibrated foundation-model representation does not automatically satisfy this total-energy contract.

For `PrescribedPath(path)`, initialize `q` from `path.position(initial_time)`.
The runner checks every trajectory against the path at its own initial time
before producing observations, including on restart. It allows only numerical
roundoff based on coordinate precision and path velocity; it does not overwrite
the supplied coordinates. `HarmonicBath` instead advances each trajectory's own
initial `q,p`, so it has no shared-path coordinate requirement.

## Minimal native implementation

Static model configuration and graph topology live in the model object. Numeric fit coefficients and NN weights live in the dynamic `params` PyTree, which the runner passes to compiled kernels. Do not capture changing model weights or per-trajectory arrays in a Python closure.

```python
import jax.numpy as jnp
from pyeph.core.contracts import ModelSpec
from pyeph.core.system import SystemSpec
from pyeph.models.base import AutoDiffModel

class TwoStateModel(AutoDiffModel):
    spec = ModelSpec(
        SystemSpec(2, (1,), coordinate_kind="canonical"),
        name="two_state_example",
    )

    def apply(self, params, q, vectors):
        z = params["slope"] * q[0]
        v = params["coupling"]
        h = jnp.array([[z, v], [v, -z]])
        return h @ vectors

    def reference_energy(self, params, q):
        return 0.5 * params["spring"] * jnp.sum(q**2)
```

Complete the calculation with the public orchestration API:

```python
from pyeph import (CoupledClassical, Ehrenfest, Integrator, Problem,
                   Simulation, configure_precision, make_state)

configure_precision(enable_x64=True)  # before creating numerical arrays
model = TwoStateModel()
params = {"slope": 0.1, "coupling": 0.2, "spring": 0.7}
problem = Problem(model, params, CoupledClassical(masses=1.0), Ehrenfest())
initial = make_state(q=[0.4], p=[0.1], electronic=[1, 0])
simulation = Simulation(problem, Integrator(dt=0.01, electronic="exponential_midpoint"))
result = simulation.run(initial, steps=100)
print(result.observables["population"][-1])  # fixed-basis amplitude populations
```

This custom provider is sufficient for propagation. Strict checkpoints also
require an explicit artifact identity for its external implementation; see
[the restart guide](RESTART.md). Supplying numerical parameters separately
ensures their actual contents participate in restart identity.

For a later run with the same static model, call
`simulation.update_parameters(new_params)`. It validates the candidate problem
before replacing parameters; a rejected update leaves the existing configuration
intact. Cached kernels and initial measurements receive the new runtime values.
Direct mutation of a parameter dictionary still changes runtime inputs but
bypasses this preflight, so prefer the explicit update. Do not mutate model,
path, method or measurement objects after assembly. Changes to static
configuration require a new `Simulation`; changes to an opaque provider's code
or captured weights require reconstructing that provider as well.

Parameter updates are allowed only between runs. An observer cannot call
`update_parameters` or recursively run the same simulation; checkpoints from an
observer remain allowed. The guard is released after success or failure. Direct
mutation of a parameter dictionary or its leaves during a run is unsupported:
it bypasses the explicit API and can change the physical problem between chunks.

`apply` accepts either a state vector `(S,)` or a block of column vectors `(S,K)`. It must preserve complex amplitudes even when `h` itself is real. A sparse/edge model should apply its operator directly. `AutoDiffModel.dense` is only a convenience for small reference checks and dense algorithms; a large Hamiltonian need not be materialized for CPA or diabatic Ehrenfest.

An optional `prepare_action(params, q)` hook can move geometry-dependent work
outside repeated operator applications at the same geometry. It returns a pure
callable accepting the same vector/block shapes as `apply`. For an already dense
NN provider, the pattern is:

```python
def prepare_action(self, params, q):
    matrix = self.dense(params, q)
    return lambda vectors: matrix @ vectors
```

Sparse and periodic providers instead close over their onsite, edge or orbital
block coefficients; preparation does not require a dense Hamiltonian.
`pyeph.core.contracts.prepared_action(model, params, q)` selects this optional hook
or falls back to regular `apply` when the hook is absent or `None`. Malformed
hooks/results raise `TypeError`. If `apply` is overridden on a more-derived class or instance than its hook,
the helper conservatively uses that `apply`;
add a matching explicit hook to optimize the changed Hamiltonian. Overriding
`dense` or `elements` can retain inherited preparation when both ordinary and
prepared operations dispatch through those same formulas. Ordinary native
`apply` does not begin calling a subclass's preparation hook. Ordinary `apply`
evaluates its formula using each call's coordinates and parameters. A prepared
callable instead represents the supplied geometry/parameters for its local
action lifetime; changing parameters elsewhere does not refresh that callable.
Construct a new one when geometry or parameters change, and reuse it only for
vectors/columns belonging to that action.

Checked CPA/Ehrenfest uses preparation once per frozen electronic action,
sharing its coefficients across the Krylov vectors and electronic columns.
Ordinary Ehrenfest also prepares once inside each fixed-geometry electronic
half-step, sharing coefficients across its RK4 stages and electronic substeps.
The second half prepares anew at the updated nuclear coordinates. Ordinary CPA
continues evaluating its prescribed changing geometry at the integrator stages.
Batching prepares each trajectory's geometry independently. A sum prepares its
children without densifying them, and a reference shift prepares the base action
and scalar shift. Forces/VJPs remain on their regular derivative APIs; this hook
does not cache them or share data across nuclear steps or separate electronic
halves.
An all-zero vector retains its exact zero result and zero action estimate, but
preparation can still cost work before the kernel's zero-vector branch. No
per-lane cancellation of preparation is promised under batching.

Create and consume the callable **inside** the numerical function being compiled
or differentiated. Do not JIT/vmap the callable-producing hook by itself, return
its callable through a JIT boundary, store it on a model/Runner, or serialize it.
Its numerical arrays must remain connected to the supplied JAX coordinates and
parameters: no NumPy conversion or `stop_gradient`. Cached Runner blocks still
receive current parameter arrays on every run. Preparing outside a coordinate
derivative and then reusing that closed-over operator would omit its coordinate
dependence. Tests in `tests/test_prepared_actions.py` cover value/linearity and
AD parity, composition, subclass compatibility, parameter updates and runtime
stage counts; preparation alone makes no performance claim.

The mixin supplies `reference_gradient` and `contract_gradient`. The latter evaluates

\[
\nabla_Q\operatorname{Re}\operatorname{Tr}(W^\dagger h).
\]

`LowRankWeight(left,right)` means `W=left @ right.conj().T`; its scalar contraction is `Re sum(conj(left) * apply(right))`. `pure_state_weight(c)` gives the projector `c c†` without allocating a dense density matrix. Weights are held fixed during the instantaneous derivative. Complex off-diagonal derivatives can be selected with real and imaginary weights.

The native force helpers preserve outer derivatives through the electronic
weight. Their inner `jax.grad` varies only its coordinate argument, which holds
its closed-over weight fixed for the instantaneous force; an outer derivative
can still include changes in that weight with electronic amplitudes, coordinates
or parameters. `ReferenceShiftModel` preserves the same response of the weight
trace. Force values are unchanged by this distinction. Reference compensation
is invariant for normalized one-carrier states and norm-preserving variations;
its derivative in an unnormalized direction need not match the original model.

`tests/test_force_sensitivities.py` checks forward/reverse force Jacobians against
independent finite differences, including complex amplitudes and dense/low-rank
weights. A small smooth native RK4 Ehrenfest calculation also checks pure-step
sensitivities as the finite-difference increment is refined. This is bounded
kernel evidence. The separate [pure rollout interface](DIFFERENTIABLE_DYNAMICS.md)
supports qualified smooth CPA/Ehrenfest sensitivities; the operational Runner
performs host validation and NumPy collection. No trajectory-sensitivity guarantee is
made for Torch callbacks, MASH events, checked Lanczos acceptance/breakdown
branches, or dense eigensolves at degeneracies. Those require separate
formulations and tests; they are not implied by a differentiable NN provider.

Implement optional `validate_params(params)` and `validate_geometry(q)` for host-side preflight checks. Validate static shapes, graph indices, Hermiticity, nonfinite values, and representation declarations before entering JIT code. The native model constructors reject duplicate edges and fractional image/index values instead of silently rounding them.

## Existing constructions

Choose the provider boundary from the data you actually have:

| Input | Entry point |
|---|---|
| Native JAX onsite/hopping coefficients from all atoms | [`LocalBlockModel`](LOCAL_MODELS.md) |
| Torch onsite/hopping coefficients from all atoms | [`TorchLocalBlockModel`](TORCH_LOCAL_MODELS.md) |
| Torch scalar neutral nuclear potential | [`TorchReferenceModel`](#a-scalar-torch-reference-beside-a-native-carrier), composed with a carrier |
| A small full dense Torch Hamiltonian | `TorchHamiltonianAdapter`, described below |
| Saved moving-AO coefficients, metrics and cross-time overlaps | [`project_ao_path` and `RecordedCPA`](AO_RECORDED_PATHS.md), for electronic propagation without feedback forces |


| Construction | Dynamic data and intended use |
|---|---|
| `LinearEPCModel(nstates, nmodes)` | Dense `h0` and mode-coupling matrices for small references. `create_params` validates Hermiticity. |
| `EdgeEPCModel(nstates, nmodes, edges)` | Linear onsite and edge EPC in mass-weighted normal coordinates. Graph action avoids a global matrix per mode. |
| `AggregateModel(nsites, edges, ...)` | Smooth nonlinear site energies and transfer integrals on a fixed irregular graph. This is an infrastructure fixture, not a fitted molecular parametrization. |
| `PeriodicBlockModel(nsites, norbitals, edges, cell, ...)` | Explicit orbital blocks and integer cell-image edges, with a declared Bloch/Peierls convention. Supports complex Hermitian electronic dynamics. |
| `NeuralResidualModel(nstates, q_shape, ...)` | Small fully differentiable dense tanh network for interface demonstrations. Not equivariant or size-extensive. |
| `SumModel((baseline, residual))` | Adds every Hamiltonian and reference-potential term. The neural residual's reference energy is zero. |
| `ReferenceShiftModel(model, shift_fn)` | Applies `h'=h-f(Q)I`, `V_ref'=V_ref+f(Q)` as a consistent energy-reference transformation. |

Analytic fixtures expose `default_params()` to inspect their parameter schema; local coefficient providers define their own runtime parameter trees. The dense NN uses `init_params(key)` and starts with a zero output layer unless `zero_last=False` is requested. A residual can therefore be inserted without changing the initial baseline model.

`SumModel` requires matching coordinate shape, basis identity, electronic sector, energy convention and units. Its parameters are a tuple of child parameter PyTrees. It does not guess that two arbitrary probe definitions are additive: pass `additive_probes=(...)` only when that physical composition is valid. In particular, summing two Hamiltonians does not require summing their position operators.

Composition adds each provider's **supplied derivative operations**. It does not
differentiate the entire sum through every provider's internals. A native JAX
baseline and a Torch callback residual can therefore share one `SumModel`:

```python
from pyeph import Execution
from pyeph.models.composite import SumModel

# Given prepared native and Torch providers and their parameter PyTrees:
combined = SumModel((native_baseline, torch_residual))
combined_params = (baseline_params, residual_params)
execution = Execution(allow_host_callbacks=True, verify_external_gradients=True)
```

Each provider still supplies the complete gradient of its own coordinate-dependent
terms. The resulting energy, Hamiltonian and forces are additive; host callback
overhead remains. This does not create a cross-framework autodiff graph for
training the whole trajectory. `ReferenceShiftModel` likewise combines the base
provider's derivatives with the native JAX derivative of its shift function.
The shift must return one finite real scalar. A composed model advertises force
support only when each child declares support and supplies both derivative
operations; adding a wrapper cannot turn a values-only provider into a force
model.
Mixed-provider trajectories and compensated shifts are checked against native
references in `tests/test_mixed_composition.py`.

## A trained native residual example

For full atomic inputs and sparse local carrier blocks, see
[local coefficient providers](LOCAL_MODELS.md). That concrete family keeps the
center map and periodic image graph separate from the baseline/NN coefficient
function, without changing the dynamics contract. The example below instead
uses a small general dense matrix residual.

Run:

```sh
PYTHONPATH=src .venv/bin/python examples/neural_surrogate.py
```

The example fits the final affine layer of a native JAX tanh network to an analytic nonlinear two-state residual and its coordinate derivatives at 61 geometries. Its hidden features remain fixed random features; the output fit uses SVD least squares. It evaluates 60 distinct interpolation holdouts, then replaces the analytic residual with the NN inside the **same** `SumModel` and Ehrenfest runner. The physical harmonic/spin-boson baseline remains present with all its derivatives. The test is deliberately small and does not establish chemical transferability, extrapolation reliability, or a foundation-model result.

The script contains the architecture, fitting objective and synthetic-data
recipe. Run it to generate a new report and numerical parameter artifact.
Compare both label errors and complete trajectories; timestep error and fitted
model error are separate quantities.

## External PyTorch models

Install the optional dependency with `.venv/bin/python -m pip install -e '.[torch]'`. For a small full dense Hamiltonian, use `TorchHamiltonianAdapter`; sparse coefficient and scalar-reference routes are listed above. This dense adapter is an explicit **CPU host-callback** route: PyTorch evaluates the complete Hamiltonian/reference graphs and their own gradients, while JAX receives arrays through `jax.pure_callback(vmap_method="sequential")`. This can interoperate with compiled/batched CPA and Ehrenfest, but it is not a single device-resident JAX graph. No cross-framework autodiff or differentiability through the trajectory is claimed.

```python
import torch
from pyeph.adapters.torch import TorchHamiltonianAdapter
from pyeph.core.contracts import ModelSpec
from pyeph.core.system import SystemSpec
from pyeph.execution.runner import Execution

spec = ModelSpec(SystemSpec(2, (1,), coordinate_kind="canonical"),
                 name="torch_two_state")

def hamiltonian(params, q):
    z = params["slope"] * q[0]
    coupling = params["coupling"] + q.sum() * 0
    return torch.stack((torch.stack((z, coupling)),
                        torch.stack((coupling, -z))))

def reference(params, q):
    return 0.5 * params["spring"] * torch.sum(q**2)

model = TorchHamiltonianAdapter(spec, hamiltonian, reference)
execution = Execution(allow_host_callbacks=True, verify_external_gradients=True)
```

The runner requires the explicit `allow_host_callbacks=True` setting. The adapter uses dense `(S,S)` Hamiltonian arrays; its cost and host transfers must be included in end-to-end benchmarks. A batch invokes callbacks sequentially. For sparse fixed-basis onsite/hopping heads, use [`TorchLocalBlockModel`](TORCH_LOCAL_MODELS.md). Keeping an external model accelerator-resident still requires a separately validated execution path.

The callback receives a matching PyTree of CPU Torch parameter tensors and one differentiable coordinate tensor. Returned values must be Torch tensors, with the declared shape and Hermiticity. Reference energies must be real scalars. Callback functions must be pure and deterministic; put Torch neural modules in evaluation mode, disable dropout, and do not mutate hidden weights or caches during a run. Use precision consistently between coordinates, parameter tensors and any captured module.

All coordinate-dependent baseline and residual terms must be part of the Torch graph. The adapter rejects NumPy outputs and completely detached coordinate-dependent outputs. A legitimate constant can be declared with `coordinate_independent_hamiltonian=True` or `coordinate_independent_reference=True`. A default omitted reference is an exact graph-connected zero.

A subtler error is `h = attached_residual(q) + detached_baseline(q)`: the output still has `requires_grad=True`, but its derivatives are incomplete. `validate_complete_gradients(params,q)` therefore checks the complete Hamiltonian and reference energy against central differences for **every coordinate** at a supplied geometry. Set `Execution(verify_external_gradients=True)` to request the runner's preflight audit at the first initial geometry; this is **off by default**, because it can require many expensive external evaluations. A failure raises an error before production. Also call `model.validate_complete_gradients(params, q)` explicitly on representative training/holdout geometries. This is a local audit, not a proof over the model's entire domain; its cost grows with coordinate count and dense output size. Use float64 for tight derivative checks; the audit increases its finite-difference step and absolute tolerance when coordinates are float32.

Optional probe callbacks receive `(torch_params, ProbeContext)` and return a dense physical operator. The adapter does not infer currents or missing basis-connection terms. Native and Torch provider equality, low-rank/complex contractions, JIT/batch behavior, complete CPA/Ehrenfest trajectories, and a deliberately detached baseline are covered by [`test_torch_adapter.py`](../tests/test_torch_adapter.py).

## A scalar Torch reference beside a native carrier

When the external model supplies only the neutral nuclear potential, use
`TorchReferenceModel` from `pyeph.adapters.torch_reference`. It evaluates the
scalar `V_ref(params,q)` and its complete Torch coordinate gradient. Its
electronic action and contracted electronic gradient are exact native zeros,
so adding it to a sparse carrier does not construct or transfer a dummy dense
Hamiltonian.

For an already constructed `carrier` whose scalar nuclear reference is zero:

```python
import torch
from pyeph.adapters.torch_reference import TorchReferenceModel
from pyeph.models.composite import SumModel
from pyeph import Execution

def neutral_reference(params, q):
    # An illustrative scalar potential; a real NN uses its full Torch graph.
    return .5 * params["spring"] * torch.sum((q-params["reference_positions"])**2)

reference = TorchReferenceModel(
    carrier.spec, neutral_reference,
    zero_probes=("current_x", "current_y", "current_z"),
)
model = SumModel((carrier, reference), additive_probes=reference.spec.probes)
params = (carrier_params, reference_params)
execution = Execution(allow_host_callbacks=True, verify_external_gradients=True)
```

Both components must use the same coordinate array, electronic basis, sector
and unit declaration. Convert positions and scalar energies inside the Torch
graph when the external model uses different units, so its returned gradient
has the corresponding complete conversion factor. A complete xTB energy plus
a complete foundation-model energy would generally double-count the neutral
potential: choose one reference or train/use an explicit residual.

`zero_probes` declares only the electronic operator contributions that really
vanish for this scalar reference. Position operators are not automatically
additive or zero. A separate ionic observable would need its own definition.
The reference model's supplied `spec` is copied with external execution and
force support declared; the carrier's basis and unit conventions are retained.

This remains a CPU host-callback route with sequential external batch calls.
The full composition requires explicit callback execution permission. The
checked Lanczos path and MASHRM retain their native-model restrictions; tested
composition here is ordinary CPA/Ehrenfest. Torch owns the force graph, and
cross-framework parameter or trajectory differentiation is not implemented.
An arbitrary xTB executable is not made differentiable by this wrapper: a
specific energy/force adapter is separate work.

Host preflight checks finite energy and gradient values at all supplied initial
geometries. The optional complete-gradient audit also detects a detached
coordinate-dependent baseline hidden beside an attached NN residual. Strict
checkpoints require `model.reference_fn` for the standalone model, or a path
such as `model.models[1].reference_fn` for this sum, as an external artifact
identity. Numerical parameters remain explicit runtime inputs. See
[`test_torch_reference.py`](../tests/test_torch_reference.py) for complete-force
finite differences, unit conversion, parameter updates, trajectory parity and
strict restart checks.

## Checking a provider at actual coordinates

A model may define the optional host hook
`validate_at(params, q, *, batch=False)`. Runner calls it before initial output,
including a zero-step or no-output run, and before saving/accepting a checkpoint.
The hook owns geometry checks as well as any joint parameter/coordinate checks.
For a batch, q contains all initial geometries and the hook is called once; a
native provider can use `jax.vmap` rather than a Python loop over trajectories.
An invalid initial provider raises before any observer publication. Updated
parameters are passed on every new run.

Models without this hook retain the existing `validate_geometry(q)` behavior.
Parameter-only validation remains part of Problem construction and parameter
updates. `SumModel` and `ReferenceShiftModel` forward concrete preflight to their
components, preserving each child's geometry-only fallback. This operation is
optional; it adds no required Hamiltonian representation or method dispatch.

Preflight checks the supplied geometries, not every future coordinate in a
trajectory. Providers must remain pure, finite, Hermitian and consistent with
their declared basis/complexity across the integration domain. The normal
finite-state guard is not a general audit of arbitrary provider Hermiticity or
gradient completeness. Explicit numerical checks at representative distorted
geometries remain necessary. A differentiable output does not establish that
all physical baseline and feature derivatives were retained.

## Physical conventions to retain

- **Energy zero:** subtracting a coordinate-dependent trace/reference from `h` requires adding it to `V_ref`. Population invariance on a prescribed path does not imply force invariance.
- **Periodic edges:** cell vectors are rows, `d_abn=q_b+n@cell-q_a`. A stored edge and its generated conjugate reverse must be counted once. The Bloch action uses the lattice phase `exp(i k·n@cell)`; the Peierls current uses the full displacement.
- **Disorder:** primitive-cell independent k sectors do not represent arbitrary disorder scattering. Use the explicit supercell or retain off-diagonal momentum couplings.
- **Probes:** aggregate `current_*` is hopping charge current in a diagonal-position approximation. `lab_current_*` additionally needs site velocities. Neither is an ionic-current model. General AO/Wannier currents may require additional position/connection information.
- **Smoothness:** do not remove edges using a geometry-dependent hard magnitude threshold. Switch values and their derivatives smoothly at the spatial cutoff, preserve stable IDs, and report neighbor-list overflow.
- **Basis meaning:** these native providers use an effective fixed orthonormal basis. Raw moving/nonorthogonal AO data require overlaps, cross-time transport and consistent force terms. Complex storage alone does not validate SOC surface hopping.

For a new provider, start with independent dense-versus-structured action, contracted-gradient finite differences, reference-shift invariance, and an appropriate physical probe check. Then compare unchanged dynamics using that provider against a small analytic reference. Geometry symmetry, energy/norm convergence, state-space convergence and training-domain tests remain specific scientific responsibilities rather than consequences of adopting an API.
