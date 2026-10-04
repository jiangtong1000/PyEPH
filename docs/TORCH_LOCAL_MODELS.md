# Sparse local Hamiltonian heads in Torch

`pyeph.adapters.torch_local.TorchLocalBlockModel` accepts a Torch function that
returns onsite and hopping blocks from all atomic coordinates. It preserves
the graph representation of `LocalBlockModel`, while Torch computes complete
coordinate derivatives. This is useful when an existing Torch feature extractor
or Hamiltonian head would otherwise require a dense global matrix.

This adapter uses CPU host callbacks. It does not make a Torch network part of
a fused JAX program, supply GPU execution, or provide automatic derivatives
through a mixed-backend trajectory. Use the native JAX coefficient route when
the model is available in that form and compiled execution matters. Both routes
share the physical graph, basis, force and current contracts.

Install the optional dependency from the repository root with
`.venv/bin/python -m pip install -e '.[torch]'`. Importing the package or adapter
module does not require Torch; constructing a Torch adapter does. This extra
does not install a foundation-model checkpoint or its separate dependencies.

## Provider interface

The constructor takes the same graph, atomic center map, charge, complex-value
declaration and basis identity as the [native local model](LOCAL_MODELS.md):

```python
from pyeph.adapters.torch_local import TorchLocalBlockModel
from pyeph.models.local import AtomCenterMap, LocalBlockGraph, LocalCoefficients

def coefficients(params, q, geometry):
    # q and every numerical geometry field are CPU Torch tensors.
    # This is a simple interface fixture, not a fitted chemical model.
    internal = (q[0]-q[1]).square().sum()
    onsite = (params["site_energy"] + params["response"]*internal)[:, None, None]
    hopping = (params["transfer"]*(-params["decay"]*geometry.distances).exp())[:, None, None]
    return LocalCoefficients(onsite, hopping)

model = TorchLocalBlockModel(
    LocalBlockGraph(2, 1, ((0, 1),), switch_on=3., cutoff=5.),
    AtomCenterMap((0, 0, 1), (.5, .5, 1.), 2),
    coefficients,
)
```

Continue the same script with runtime weights and a short trajectory:

```python
from pyeph import (CoupledClassical, Ehrenfest, Execution, Integrator, Problem,
                   Simulation, configure_precision, make_state)
import jax.numpy as jnp

configure_precision()
params = {"site_energy": jnp.array([-.1, .1]), "response": jnp.array([.02, -.03]),
          "transfer": jnp.array([-.04]), "decay": jnp.array(.2)}
q = jnp.array([[-.2, 0., 0.], [.2, 0., 0.], [4., 0., 0.]])
initial = make_state(q, jnp.zeros_like(q), [1.+0.j, 0.+0.j])
problem = Problem(model, params, CoupledClassical(masses=1.), Ehrenfest())
simulation = Simulation(problem, Integrator(dt=.005),
                        Execution(allow_host_callbacks=True, verify_external_gradients=True))
result = simulation.run(initial, steps=20)
print(result.observables["population"][-1])
```

This executable interface example has three atoms, two carrier states and zero
common reference potential. Add a physically defined nuclear reference before
using this head for material feedback. The short run exercises the complete
gradient preflight; it does not establish a thermal or chemical model.

Parameters are numerical PyTrees supplied to the normal `Problem`. The provider
receives Torch copies of those leaves at every callback. Keep numerical weights
in that tree when they should change through `simulation.update_parameters`.
Captured neural modules or calculator settings are opaque artifacts: hold them
immutable, identify them explicitly, and construct a new model/runner when they
change. Enable evaluation mode and disable stochastic layers; repeated calls
must have the same mathematical values and derivatives.

`LocalCoefficients` must contain CPU Torch floating/complex tensors with shapes
`onsite[site,b,b]` and `hopping[edge,b,b]`. Onsite blocks must be Hermitian;
forward hopping blocks need not be. The adapter generates the Hermitian reverse.
Use `complex_valued=True` for complex blocks. Inputs and outputs follow the
native model's atomic-unit, fixed orthonormal carrier-basis convention.

The provider returns **raw hopping blocks**. The adapter applies the final
smooth support exactly once. Provider-side environment messages need their own
smooth support gates, just as in the native route. The fixed graph must cover
every allowed edge throughout the calculation. There is no neighbor search,
orbital/spin equivariance, trained checkpoint or chemistry implied by this
interface.

## Complete atomic derivatives

Torch reconstructs the weighted centers, periodic image displacements,
distances and cutoff support from `q` inside its own graph. The provider also
receives the original all-atom coordinates. A zero-center-weight atom can still
affect an internal descriptor and have a nonzero force. Passing precomputed
JAX/NumPy descriptors into a Torch network would detach their coordinate
dependence; this adapter does not use that shortcut.

For `LowRankWeight(L,R)`, the force operation differentiates
`Re Tr[(L R†)† H]`, holding the weight fixed. It first forms local cotangents

\[
G_i=L_iR_i^\dagger,\qquad
G_{ij}=L_iR_j^\dagger+R_iL_j^\dagger,
\]

then differentiates the real contraction with the physical onsite and supported
hopping blocks in Torch. This handles non-Hermitian weights and periodic
self-image edges. A shortcut using only twice one directed weight is valid
only for Hermitian weights. No global Hamiltonian, identity or
Hamiltonian-by-coordinate tensor is constructed on this route. A caller-supplied
dense weight is also accepted; only its relevant blocks are gathered.

The model's common reference energy is exactly zero. Compose a separate neutral
potential through `SumModel` and, when appropriate, `TorchReferenceModel`.
That reference must share the coordinate/basis convention. Declare its zero
electronic current contributions explicitly before adding physical probes.
See [scalar reference composition](ADDING_MODELS.md#a-scalar-torch-reference-beside-a-native-carrier).
Do not add two complete reference potentials unless their physical meaning
calls for that sum.

Declare a genuinely coordinate-independent complete Hamiltonian explicitly with
`coordinate_independent_hamiltonian=True`. This permits a disconnected zero
derivative; it never suppresses attached derivatives. An attached residual can
still conceal a detached baseline, so this flag cannot certify completeness.

The optional `validate_complete_gradients(params,q)` audit compares all real and
imaginary supported block derivatives with central finite differences, one
atomic coordinate at a time. It stores block-sized outputs rather than a full
coordinate Jacobian. The number of directions grows with atomic coordinate
count, and each direction still costs provider evaluations. Its Torch JVP audit
requires the provider's operations to support the necessary higher automatic
derivatives. This audit tests supplied geometries; it is not proof throughout
a training domain or future trajectory.

## Execution, currents and restart

Use `Execution(allow_host_callbacks=True)` for public dynamics, optionally adding
`verify_external_gradients=True` for the representative initial-geometry audit.
Ordinary CPA and Ehrenfest work with the same state and parameter-update APIs.
External-provider restrictions remain: checked Lanczos and the current MASHRM
profile require native JAX models. No new method capability is inferred simply
from accepting a sparse coefficient provider.

JAX performs sparse actions and physical hopping-current operations using the
transferred blocks. Currents retain full image displacements and the native
point-center convention; they omit ionic convection, intracenter dipoles and
moving-AO connection terms. `rewrapped(shifts)` only relabels the graph. The
caller must coherently shift the associated atoms and maintain any provider-owned
reference or descriptor data.

Callbacks use sequential `vmap`: a JAX trajectory batch does not imply one
batched Torch model evaluation. Ordinary Ehrenfest prepares the
block values once per fixed-geometry electronic half-step and reuses them across
RK4 stages/substeps. Its second half prepares at the updated geometry. Forces
still use their complete derivative callbacks. Ordinary CPA evaluates its
prescribed path at each required stage. Compiler elimination of pure calls and
actual provider invocation counts remain separate questions.

Strict checkpoints require an artifact ID at `model.coefficient_provider`, or
its exact nested composition path. It must cover provider code, captured weights
and settings. Runtime provenance also includes the Torch version. The adapter
does not retain a Torch autograd graph or a hidden geometry cache across calls.
Callback failures propagate through the ordinary runner failure boundary and
retain the last accepted chunk for diagnosis or checkpointing.

## Established checks

Independent tests compare complex sparse actions and all atomic forces with
NumPy dense assembly and finite differences, including nonsymmetric hopping,
non-Hermitian low-rank weights, periodic self-images, zero-center-weight atoms,
smooth cutoff transitions and coherent rewrapping. Peierls-phase finite
differences independently check physical currents. Sparse shape guards reject
global square intermediates in a 512-state case.

Public tests cover mixed scalar-reference composition, complete-gradient
preflight, a complex current-correlation oracle, strict origins/restart,
sequential trajectory batching, changed parameters and failed-provider state
retention. These validate the adapter contract. They do not validate a pretrained
foundation model, a fitted material, GPU throughput or cross-framework AD.

Platform and integrated release evidence are recorded in [qualification](QUALIFICATION.md).
