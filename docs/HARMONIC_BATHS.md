# Harmonic nuclear paths in physical coordinates

CPA can use independent oscillator coordinates, coupled finite-system modes,
or periodic Cartesian force constants. These are prescribed nuclear paths:
electronic amplitudes do not change their motion. Ehrenfest instead propagates
canonical coordinates with the model's forces through `CoupledClassical`.

All bath quantities use the same declared `hbar=1` units as the model. A
Cartesian Hessian has energy/length² units, momenta are canonical, and masses
are the reduced masses defined by `UnitSystem`. The bath does not infer a
conversion from a file name or model type.

| Nuclear representation | Bath | Preparation and storage |
| --- | --- | --- |
| Independent oscillators already expressed in their canonical coordinates | `paths.harmonic.HarmonicBath` | Frequencies and masses only |
| General finite Cartesian Hessian, including disorder | `paths.normal_modes.NormalModeBath` | Dense mass-weighted eigensystem |
| Translation-invariant Cartesian force-constant stencil | `paths.periodic_harmonic.PeriodicHarmonicBath` | Primitive-cell eigensystems on the supercell Fourier mesh |

Use a dense bath when its flexibility or small-system speed is useful. The
periodic bath reduces storage growth with cell count, but it requires a
translation-invariant Hessian. It is not an approximation for a disordered
force-constant matrix, nor does it automatically make a disordered model
periodic. Electronic EPC terms and initial thermal displacements can still
differ among cells while the prescribed reference bath remains periodic.

## Dense normal modes

For Cartesian Hessian `K` and positive diagonal mass matrix `M`, preparation
diagonalizes

\[
 M^{-1/2} K M^{-1/2}=U\,\mathrm{diag}(\omega^2)U^T.
\]

The canonical modal variables are `Q = U.T @ sqrt(M) @ (q-q_eq)` and
`P = U.T @ invsqrt(M) @ p`. Each positive-frequency mode then evolves as an
ordinary unit-mass oscillator. The public state keeps physical `q,p`.

```python
from pyeph.paths.normal_modes import NormalModeBath, sample_normal_modes

bath = NormalModeBath.from_hessian(
    hessian, masses, equilibrium,
    frozen_modes=(), zero_tolerance=0.0,
)
q, p = sample_normal_modes(
    bath, temperature_in_energy_units, trajectory_ids, seed=1120,
    distribution="classical",
)
```

`equilibrium` fixes the coordinate shape. Atomic masses usually have shape
`(natoms,1)` and broadcast over `(natoms,3)`. The Hessian is a flat
`(equilibrium.size,equilibrium.size)` array in the same ordering. Its
diagonalization happens once on the host; `point(state, elapsed)` uses native
JAX operations and accepts JIT, automatic differentiation through the state,
and trajectory batching through `vmap`.

`from_hessian` diagonalizes in host double precision even when the supplied
Hessian is float32. The resulting arrays follow the caller's JAX precision.
The direct eigensystem constructor requires orthonormal vectors at that
execution precision; it rejects an approximate float32 basis used as if it
were double precision. It does not silently rotate a supplied basis.
The stored basis is checked using the largest row Euclidean norm of
`U.T @ U - I`, with allowance `max(64, 8*sqrt(ndof))*eps` at execution
precision. `orthogonality_error` records the measured value and
`orthogonality_tolerance` exposes the acceptance allowance. This is an explicit
size/precision policy, not a certified bound for a particular eigensolver.
It allows ordinary error accumulation in large valid mode bases while rejecting
inaccurate input bases. Frozen-momentum preflight uses four times that allowance
times `max(1, ||P_modes||_2)`, including the basis's own roundoff leakage.
It does not make independently diagonalized near-degenerate projectors identical.

## Periodic Cartesian force constants

The periodic bath consumes format-independent arrays:

- `mesh = (nx,ny,nz)` gives positive integer primitive-cell counts.
- `masses` contains one mass per primitive-cell atom.
- `ifc_atoms[t] = (a,b)` and `ifc_cells[t] = R` identify
  `K[(0,a),(R,b)] = ifc_values[t]`, a real `3×3` Cartesian Hessian block.
- The full directed stencil must include the transposed block at `(b,a,-R)`.
  Duplicate terms are summed. The validator checks image pairs before
  periodic wrapping, so an invalid stencil cannot pass just because its
  images alias on a small mesh.

```python
from pyeph.paths.periodic_harmonic import (
    PeriodicHarmonicBath, sample_periodic_harmonic,
)

bath = PeriodicHarmonicBath(
    mesh, primitive_masses, ifc_atoms, ifc_cells, ifc_values,
    frozen_below=None, zero_tolerance=0.0,
)
q, p = sample_periodic_harmonic(
    bath, temperature_in_energy_units, trajectory_ids, seed=1120,
    distribution="classical",
)
```

Coordinates have shape `(nx*ny*nz*natoms,3)`, in cell-major order with z
fastest, then atom and Cartesian component. The default equilibrium is zero,
appropriate when `q` denotes displacement from a reference crystal. An
explicit equilibrium array uses that same shape. `RealSpaceEPC` supplies the
IFC and primitive-mass arrays directly; no normal-mode EPC transformation is
needed to connect its Cartesian Hamiltonian to this bath.

The dynamical matrix uses positive phase `exp(+2π i k.R)` and the forward
coordinate FFT uses negative phase, both with orthonormal Fourier
normalization. Complex conjugate Fourier components represent one real
Cartesian field. Sampling starts from real white-noise fields, then applies
the appropriate matrix square roots in Fourier space; it does not sample
`k` and `-k` as independent complex oscillators.

If there are `C` cells and `d=3*natoms` coordinates per cell, dense mode
storage grows as `(C*d)²`, while periodic mode storage grows as `C*d²`.
The periodic implementation uses complex Fourier arrays, so its constants
and small-system performance differ from the real dense route. Construction
uses NumPy eigensolvers; trajectory motion uses JAX matrix products and FFTs.
Actual performance evidence is CPU-only so far.

## Zero, unstable, and deliberately constrained modes

Both baths retain the unmodified signed `squared_frequencies` for inspection.
They reject active negative values below the explicit `zero_tolerance`.
That tolerance has squared-frequency units; choosing it changes which
numerically small values are treated as free modes. No acoustic sum rule or
phonon stabilization is applied implicitly.

An active zero mode moves freely: `Q(t)=Q(0)+P(0)*t`. A thermal free-particle
position distribution is not normalizable. Sampling therefore requires
explicit `free_positions` whenever such modes exist:

- Dense modes accept positions in the active zero-mode coordinates, with
  optional leading trajectory axis.
- Periodic modes accept real physical displacement fields in the zero-mode
  subspace, shaped like one state or a trajectory batch. For example,
  uniform translations can fix the center of mass. Fields containing a
  nonzero positive-frequency component are rejected.

The dense bath's `frozen_modes` selects explicit mode indices. The periodic
bath's optional `frozen_below=w` freezes all squared frequencies `<=w²`,
including negative ones. The latter is an explicit spectral constraint, not
a repair of the source force constants. Its projector must respect conjugate
Fourier symmetry. Place a cutoff away from a degenerate spectral boundary.

Frozen modal positions stay at their incoming values; their momenta must be
zero within numerical projection precision. Native run and checkpoint
entry points validate this constraint. Thermal samplers set frozen positions
and momenta to zero around equilibrium. A frozen-coordinate trajectory does
not reproduce the unconstrained unstable or translational dynamics.

When independent eigensolvers produce slightly different acoustic
subspaces, their projectors can differ more than the orthogonality error of
either individual basis. A state prepared for one bath is not guaranteed to
satisfy the other's exact constraints. Reuse the same prepared bath for a
trajectory and its restart, and inspect the subspace difference when making
cross-representation comparisons.

## Sampling and validation

Temperature is an energy (`kB*T`), and both samplers index random streams by
explicit trajectory IDs. Repartitioning an ID set leaves its samples
unchanged. Classical positive-mode variances are `T/omega²` and `T` in
unit-mass coordinates. Wigner sampling uses
`coth(omega/(2*T))/(2*omega)` and
`omega*coth(omega/(2*T))/2`, including the zero-temperature limit. Active
free momenta always use Maxwell statistics; this does not define a quantum
thermal position ensemble for free motion.

`tests/test_normal_mode_bath.py` and
`tests/test_periodic_harmonic_bath.py` compare motion with independently
assembled canonical matrix exponentials and test physical covariance,
mass normalization, free/frozen modes, sampling partitions and input
precision. The periodic tests include complex Fourier phases, image aliasing
and a coupled two-atom basis.

Platform results are recorded in [qualification](QUALIFICATION.md). Compare
preparation, complete-run cost and storage separately when choosing a bath.
