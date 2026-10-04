# Importing electronic frames from a moving AO basis

`pyeph.adapters.ao_frames.project_ao_path` converts explicitly supplied AO data
into the existing electronic-only `RecordedCPA` profile. It is a host ingestion
step. It does not force AO Hamiltonians into the fixed-basis force-model
interface, or infer nuclear forces from temporal overlap data.

This route accepts explicitly supplied electronic frames.
The source program remains responsible for the physical Hamiltonian,
generalized eigenvectors and overlap integrals. No source program or model
checkpoint is imported by this adapter.

## Data and conventions

For AO basis columns `chi_k`, each coefficient row describes one ket:

\[
|\phi_{ka}\rangle=\sum_\mu|\chi_{k\mu}\rangle C_{ka\mu}.
\]

The required arrays are:

| Argument | Shape | Meaning |
|---|---|---|
| `times` | `(F,)` | Strictly increasing recorded times |
| `energies` | `(F,B)` | Real energies paired with the coefficient rows |
| `coefficients` | `(F,B,A)` | Row-wise ket coefficients, possibly complex |
| `metrics` | `(F,A,A)` | Same-time AO overlaps `chi_k† chi_k` |
| `cross_metrics` | `(F-1,A,A)` | Ordered cross-time overlaps `chi_k† chi_(k+1)` |
| `retained_bands` | `(K,)` or `(F,K)` | Explicit ordered, unique, zero-based row indices at each frame |

The initial adapter requires constant AO count A, available band count B and
retained dimension K. A different ordering at each frame is allowed when both
energies and coefficient rows follow the supplied indices. It never sorts
states, aligns phases or follows bands by energy automatically.

For the selected rows it checks

\[
C_k^* S_k C_k^T=I,
\]

including off-diagonal entries. Individual vector normalization is insufficient.
The electronic overlap is then

\[
O_{k,ab}=\langle\phi_{ka}|\phi_{k+1,b}\rangle
        =[C_k^* S_{k,k+1} C_{k+1}^T]_{ab}.
\]

Consequently old coefficients project into the next basis with **`O.conj().T`**.
A coefficient dot product without the cross-time AO metric generally has a
different physical meaning. The instantaneous metric `S_k` cannot substitute
for `S_(k,k+1)`.

## Checks, units and provenance

The adapter requires JAX x64 to be explicitly enabled by the caller. Inputs
must be finite and have the stated shapes. Each same-time metric must be
Hermitian and positive definite. With `S_k=L_k L_k†`, the full AO cross metric
must satisfy

\[
\sigma_{\max}\!\left(L_k^{-1}S_{k,k+1}L_{k+1}^{-\dagger}\right)\leq1
\]

within the existing numerical overlap tolerance. Checking only the selected
electronic overlap could conceal inconsistent AO data in discarded directions.
Ill-conditioned metrics remain a numerical responsibility; condition and
orthonormality diagnostics make that visible. The adapter does not symmetrize,
clip singular values or renormalize rejected inputs.

Declare `energy_unit="eV"` or `"hartree"`, and `time_unit="fs"` or `"atomic"`.
The returned path uses atomic units, including hbar=1. Conversion constants are
recorded with the evidence. AO overlap and orbital normalization conventions
must already agree; changing an energy unit cannot repair their mismatch.

`ao_basis_id` identifies the declared AO/spin/k ordering convention, `basis_id`
identifies the retained electronic space, and `source_identity` identifies the
source calculation or immutable export. All are explicit nonempty strings.
The returned path carries immutable projection evidence: exact input hashes,
shapes and dtypes, retained indices, unit conversions and identities. A digest
binds it to the actual returned path arrays and transport policy. The normal
strict `RecordedCPA` checkpoint manifest includes that evidence. Replacing
path data while retaining stale projection evidence is rejected.
The digest checks consistency, not authenticity: source identities are caller
assertions and are not authenticated against an external calculation.

These checks establish mathematical consistency of the supplied arrays. They
cannot certify physical AO ordering, whether each energy belongs to its stated
orbital, the accuracy of the overlap integrals, or sufficient band coverage.
Those require the source Hamiltonian/calculation and independent validation.

## Minimal use

```python
from pyeph import configure_precision
from pyeph.adapters.ao_frames import project_ao_path
from pyeph.dynamics.recorded import RecordedCPA

configure_precision(True)
projection = project_ao_path(
    times, energies, coefficients, metrics, cross_metrics,
    retained_bands=[4, 5, 6],
    energy_unit="eV", time_unit="fs",
    ao_basis_id="calculation-17:AO-spin-order-v1",
    basis_id="calculation-17:retained-bands-4-5-6",
    source_identity="sha256:<digest-of-immutable-export>",
    transport_mode="raw",
)
runner = RecordedCPA(projection.path, max_subspace_loss=1e-6)
initial = runner.initialize([1., 0., 0.])
result = runner.run(initial, len(times)-1)
runner.save_checkpoint("recorded.h5", result.final_state)
```

The arrays in this snippet come from a caller-controlled export. For `.npy`
inputs, use `np.load(path, allow_pickle=False)`, explicitly order the frames,
and preserve complex coefficients. Do not infer frame order lexicographically
from directory names such as `1`, `10`, `2`.

The [complete synthetic example](../examples/ao_recorded_path.py) constructs all
five input arrays, exports energies/times in eV/fs, compares with independent
fixed-basis SciPy propagation and checks exact checkpoint continuation in its
CPU qualification configuration:

```sh
JAX_PLATFORMS=cpu .venv/bin/python examples/ao_recorded_path.py --output-dir .cache/ao-demo
```

Run the example to generate its numerical report and compare full-space
propagation, truncation and checkpoint continuation under the stated inputs.
Exact continuation in this fixture does not establish bitwise agreement across
devices or compiler environments; see [restart scope](RESTART.md).

## Raw projection, polar transport and truncation

`transport_mode="raw"` retains the actual overlap projection, including loss
from an evolving truncated subspace. No amplitude renormalization occurs.
`"polar"` explicitly substitutes the unitary polar factor; it discards that
projection loss in evolution while preserving the raw diagnostics. A singular
overlap has no unique unitary polar map and is rejected for polar transport.
`RecordedCPA.max_subspace_loss` checks the **raw** diagnostic for either policy.

For one interval the worst possible projected norm-squared loss is
`max(0,1-sigma_min(O)**2)`. This bounds that interval's projection loss. It is
not an error bound against evolution in the full physical Hilbert space.
Reducing the time step can make adjacent subspaces nearly identical while
omitted-state transitions remain physically important. Repeated projection
can approach evolution confined to the retained subspace. Converge the retained
space and compare observables independently of the interval-loss diagnostic.

The example makes this distinction concrete. Keeping only the lowest two
states reduces the maximum interval loss from 2.10e-3 to 1.32e-4 when dt falls
from 0.08 to 0.02. Its final squared norm rises from 0.9703 to 0.9925, yet the physical
amplitude error remains about 0.63. The full-space reference has transferred
35.24% of its population into the discarded third state. A better norm alone
would give the wrong conclusion about this truncated physical model.

## Input acquisition requirements

Export instantaneous metrics, electronic coefficients and cross-time overlaps
with matching atom/orbital ordering and basis identity. The same-time metric
cannot be reconstructed merely from band energies and temporal overlaps.
Antisymmetrized or hbar/dt-scaled coupling matrices are not raw overlaps.
Casting low-precision eigenvectors to x64 does not restore orthonormality;
investigate Gram-check failures against the original metric and eigensolve.

This bridge supplies electronic propagation on the recorded nuclear path.
Spatial force-coupled dynamics and event momentum changes need additional
physical information and a separately validated method.
