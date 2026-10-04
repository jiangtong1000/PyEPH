# Orientation-sensitive effective molecular fragments

`OrientedFragmentCoefficients` supplies one real carrier state per fragment to
the existing `LocalBlockModel`. Atomic coordinates determine both fragment
centres and signed local normals. Rotating one fragment at a fixed centre can
therefore change its transfer integral. Rotating or translating the entire
system leaves the Hamiltonian unchanged, transforms forces and hopping
currents as vectors, and gives zero net internal force and torque.

This is a parameterized **effective axial-orbital baseline**, not a DNTT model,
an orbital-overlap identity, or a general molecular HOMO representation. The
two-centre angular decomposition follows [Slater and Koster, Physical Review
94, 1498 (1954)](https://doi.org/10.1103/PhysRev.94.1498). Its application to a
whole fragment, exponential radial dependence and numerical parameters require
independent calibration. The supplied dimer/trimer uses illustrative values.

## Equations and basis

Each fragment declares three ordered atom indices `(o, a, b)` and a constant
phase `s_i = ±1`. With `u = R_a - R_o` and `v = R_b - R_o`,

\[
 n_i=s_i\frac{u_i\times v_i}{|u_i\times v_i|},\qquad
 t_{ij}=e^{-\alpha(r-r_0)}
 \left[V_\pi\,n_i\cdot n_j+
 (V_\sigma-V_\pi)(n_i\cdot\hat d)(n_j\cdot\hat d)\right].
\]

`LocalBlockModel` multiplies the raw transfer by its graph's quintic switching
envelope once. The onsite term is
`onsite[i] + sum(deformation[i] * (anchor_bond_lengths - bond_lengths[i]))`.
All radial, normal, centre, deformation and cutoff derivatives remain in the
JAX graph. Atoms outside the anchor triplet can still contribute through their
centre weights. This limited baseline does not describe all intramolecular
distortions; a calibrated residual may add that dependence.

The electronic labels form a declared fixed orthonormal effective basis. The
normals are coordinate-dependent coefficient features. They do not introduce
a moving atomic-orbital basis or its overlap/connection terms. A constant phase
change produces `H' = S H S`; states, density matrices, current operators and
training labels must use the same transformation. There is no eigenvector
sign fixing or arbitrary sign choice at each geometry.

The runtime parameter mapping has exactly these keys:

| Key | Shape | Meaning |
| --- | --- | --- |
| `onsite` | `(N,)` | Effective carrier energies at reference anchor lengths |
| `deformation` | `(N,2)` | Onsite response to the two anchor-bond lengths |
| `bond_lengths` | `(N,2)` | Positive reference lengths |
| `pp_sigma`, `pp_pi` | scalar | Signed radial amplitudes at `reference_distance` |
| `decay` | scalar | Nonnegative inverse-length decay coefficient |
| `reference_distance` | scalar | Positive reference intermolecular distance |

The names `pp_sigma` and `pp_pi` describe the angular model; they are not
automatically atomic or molecular integrals extracted from a calculator.
Parameters remain dynamic PyTree leaves and can be differentiated or replaced
with `Simulation.update_parameters`.

## Composition and domain

```python
from pyeph.models.fragment import OrientedFragmentCoefficients
from pyeph.models.local import AtomCenterMap, LocalBlockGraph, LocalBlockModel

# Twelve atoms, six per fragment; anchor order and phases are part of the basis.
centers = AtomCenterMap((0,)*6 + (1,)*6, (1/6,)*12, 2)
provider = OrientedFragmentCoefficients(((0, 1, 2), (6, 7, 8)), phases=(1, 1))
graph = LocalBlockGraph(2, 1, ((0, 1),), switch_on=8., cutoff=12.)
carrier = LocalBlockModel(graph, centers, provider, charge=1.,
                          basis_id="declared-fragment-hole-basis")
# Supply params and real q[12,3], then carrier.validate_at(params, q).
```

The provider requires each anchor atom to belong to its declared fragment.
The ordered bonds must exceed `min_bond_length`; the sine of their angle must
exceed `min_sine`. These are explicit excluded-domain boundaries. Invalid
frames yield nonfinite coefficients, which host preflight and checked
propagation reject. They are not repaired by clipping a physical denominator.
A successful initial preflight does not guarantee that a later trajectory
avoids frame collapse. Recheck actual sampled geometries and use the existing
checked propagator when per-step failure retention is required.

Periodic image edges use the same provider. Each fragment's atoms must remain
coherently unwrapped; centre-image relabeling must move all its atoms together.
The electronic current uses the complete image-resolved hopping displacement.
It excludes moving-centre convection, ionic current, intra-fragment dipoles and
AO connections. A scalar neutral reference potential must be composed separately
for feedback dynamics, in the same coordinate and unit conventions.

## Runnable numerical qualification

```sh
JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 PYTHONPATH=src .venv/bin/python \
  examples/oriented_fragments.py \
  --output outputs/materials_20261004/molecular/example
```

Use a fresh output directory. The example runs a dimer and trimer with CPA and
Ehrenfest, each at two time steps, and compares them with independently assembled
NumPy Hamiltonians and SciPy DOP853 evolution. The Ehrenfest oracle includes
analytic neutral spring forces plus full-coordinate finite differences of the
carrier energy. Each method has its own reference; their trajectories need not
agree. The script retains initial parameters, coordinates, trajectories, raw
reference arrays, checkpoints and source hashes. This CPU qualification fixture
requires bitwise continuation in its recorded configuration. That assertion is
not a cross-platform guarantee; the separate platform benchmark reports exact
checkpoint storage and numerical continuation independently, as described in
[restart scope](RESTART.md). Every saved geometry is rechecked for frame validity.

The example uses Hartree, bohr, electron masses and atomic time, with charge
`+e`. Its parameters already describe effective **hole** energies. The spring
network is an illustrative neutral reference, not an xTB or first-principles
force field. CPA follows prescribed free nuclear motion; its total energy need
not be conserved. Ehrenfest uses the complete reference-plus-carrier force.

`tests/test_fragment.py` independently checks Cartesian tensor assembly,
complete coordinate finite differences, parameter derivatives, charge-current
phase derivatives, phases, rigid motions, atom/site ordering, periodic images,
two smooth cutoff derivatives and invalid frames. The workflow tests exercise
dynamic parameter updates, batch partitions, restart, ODE refinement and
retention of the last accepted state when an initially valid frame collapses
inside a checked CPA step.

These checks establish numerical consistency within this effective model.
They do not establish chemical transferability, fit quality, equilibrium MASH
transport or converged molecular mobility. Material force fields and signed fragment couplings require separate
labels and validation beyond these illustrative parameters.
