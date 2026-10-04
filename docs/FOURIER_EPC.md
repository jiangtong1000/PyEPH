# Optional Fourier evaluation of periodic Cartesian EPC

`RealSpaceEPC.compile_supercell(..., epc_backend="fft")` selects
`FourierCartesianEPCModel`, an alternative evaluator for the same periodic
linear Hamiltonian. `epc_backend="direct"` remains the default. The model keeps
the original hopping, Cartesian EPC and force-constant parameters; no spectral
parameter replacement or persistent coefficient cache is introduced.

```python
from pyeph.adapters.epr import read_epr

source = read_epr("DNTT_epr.h5", polar="short_range")
compiled = source.compile_supercell(
    (3, 3, 2), carrier="hole", hermiticity="project",
    epc_backend="fft", max_spectral_bytes=256*1024**2,
)
```

The source and physical policies in this example are the same as in the
[Cartesian EPC guide](AB_INITIO_EPC.md). Fourier evaluation does not add omitted
long-range terms, repair force constants or change the electronic basis.
The runnable ab initio example accepts `--epc-backend fft` and the optional
`--max-spectral-bytes` byte limit. Direct evaluation still accepts its existing
`--term-batch-size` tuning option.

## What changes computationally

Each directed electronic image channel `h` has a hopping value

`V[c,h] = h0[h] + sum_(r,a,mu) g[r,h,a,mu] q[c+r,a,mu]`.

The FFT evaluator computes this circular correlation over displacement-cell
translations. With the usual negative-phase forward FFT, it forms
`Gplus = ncells * ifftn(G)` and evaluates
`ifftn(sum_(a,mu) Gplus * fftn(q))`. Distinct electronic hopping images keep
their own unwrapped displacements. The inherited Hermitian Hamiltonian action,
Peierls action and current probes therefore retain their original meaning.

Real models return real hopping values at the natural promoted precision;
complex coefficients retain their complex values. Electronic vectors and
physical current actions can still be complex. The contracted real-coordinate
gradient uses the corresponding transpose with `Gplus[-k]`, not a conjugated
coefficient symbol. The reference nuclear potential and its forces retain
their existing IFC implementation.

The source-coefficient scatter and Fourier transform remain inside JAX with
dynamic parameters. Coordinate and original-coefficient differentiation remain
available. `Simulation.update_parameters` follows the normal validation and
compilation rules; there is no manually invalidated spectral cache. The
compiler can move fixed-parameter work outside the trajectory loop, but this
is a per-compiled-block optimization, not a promise of reuse across public calls.

No dynamics method or measurement needs a new interface. This model can use
the same CPA, coupled Ehrenfest and applicable real-Hamiltonian method
contracts. The [periodic harmonic bath](HARMONIC_BATHS.md) is a separate component:
Fourier EPC evaluation does not require prescribed harmonic nuclei.

## Choose according to time and memory

One dense complex coefficient symbol has shape
`(*mesh, hopping_channels, primitive_atoms, 3)`. It can be much larger than a
sparse source stencil on a large simulation mesh. For the q332 DNTT input,
one complex128 symbol requires 41.43 MB at mesh 3×3×2 and 331.44 MB at 6×6×4.
The direct evaluator keeps bounded term batches and can be preferable for
sparse stencils, short calculations or large target meshes.

Select the evaluator on the actual source, cell and workflow at matched
numerical accuracy, including temporary storage and force differentiation.

`max_spectral_bytes` limits **one coefficient symbol at its actual execution
precision**. It does not bound total device memory, FFT construction workspace,
autodiff storage, trajectory batches or other compiled executables. The default
is 256 MiB. A rejected allocation is explicit; the model does not silently
switch evaluation methods. Empty EPC arrays take the constant-Hamiltonian and
zero-electronic-gradient path without allocating a Fourier symbol.

The immutable mesh must match the declared cell count. Parameter validation
checks that every neighbor-map row is the corresponding periodic translation
in C-order cells, with z varying fastest. Arbitrary graph models and models
with unrelated subclass behavior are not automatically converted.

Backend, mesh, parameters and source are part of strict restart provenance.
Resume with the same implementation and backend. Selecting a mathematically
equivalent evaluator does not authorize changing a checkpoint's recorded
identity.

Restart tests should compare a full run with split/restarted propagation
using identical source parameters, initialization and output sampling.

Platform and integrated release evidence are recorded in [qualification](QUALIFICATION.md).
