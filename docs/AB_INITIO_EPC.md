# General Cartesian EPC and ab initio ingestion

The native boundary is a periodic set of Cartesian derivatives, not a
Holstein–Peierls ansatz or an electronic-structure file format. The EPR reader
feeds this boundary; another code or an ML fitting workflow can supply the same
arrays directly. Dynamics continue to consume Hamiltonian actions, contracted
derivatives, a reference potential, and probes.

```text
PERTURBO EPR              Other first-principles/fitted Cartesian data
     │                                     │
     read_epr                              │
     └───────────── RealSpaceEPC ────────────┘
                      │ explicit units, image Hermiticity and carrier policy
                      compile_supercell(mesh)
                      │
                  CartesianEPCModel + dynamic array params
                      ├── Hamiltonian/block action and contracted force
                      ├── harmonic reference energy/force
                      └── image-resolved Peierls current
                          │
        ┌─────────────────┴──────────────────┐
        CPA + PeriodicHarmonicBath           Ehrenfest + CoupledClassical
        primitive-cell Fourier modes        full carrier force feedback
```

## Data and physical coordinates

An electronic channel stores `hopping_orbitals=(i,j)`, an integer image `Re`,
and `hopping_values = <0,i|h|Re,j>`. A derivative stores its electronic channel,
the displaced atom `a`, image `Rp`, and three Cartesian components
`g = <0,i|∂h/∂u(Rp,a)|Re,j>`. No normal-mode eigenvector or oscillator factor is
already absorbed into `g`. IFC records store `K[(0,a),(R,b)]`, before mass
division. Every array is available on `RealSpaceEPC` for independent inspection.

For each origin cell `L`, the directed electronic matrix element is

\[
A_{Li,L+R_e,j}(u)=h_{ij}(R_e)+
\sum_{a,R_p}g_{ij,a}(R_e,R_p)\cdot u_{L+R_p,a}.
\]

The runtime uses `H=(A+A†)/2`; ingestion either requires image Hermiticity or
explicitly authorizes this projection. The neutral reference is
`V0=1/2 ∑ u[L,a] K[a,b,R] u[L+R,b]`. Coordinates are displacements from the
reference crystal, ordered by `np.ndindex(mesh)` (z fastest), then atom and xyz.
Momenta are ordinary canonical Cartesian momenta; `qdot=p/m`.

`RealSpaceEPC.converted(UnitSystem(...))` converts energies, lengths, masses,
derivatives, and IFCs together. EPR uses Ry and Bohr: masses are measured in
**2 electron masses**, `g` in Ry/Bohr, and IFCs in Ry/Bohr². This matches
`UnitSystem(energy_hartree=.5, length_bohr=1)` directly. Converting to Hartree
units halves `h`, `g`, and IFCs, doubles the numerical masses, and preserves
the physical frequencies. No extra mass or oscillator scaling belongs in the
Cartesian model.

The `carrier='hole'` choice implements `H_hole=-H_electron.T` and charge `+1`;
the neutral reference IFCs stay unchanged. It assumes that the supplied
electronic subspace is appropriate for the carrier being modeled.

## Explicit source policies

The EPR conventions were checked against the primary PERTURBO source at
commit `0f993052cf57d5b57d8df83452a0cebef357c823`, including
`elph_matrix_wannier`, `elphon_coupling_matrix`, `epr_hdf5_io`,
`save_elph_mat_wann`, and `lattice_data`. Each ordered orbital pair has its own
electronic Wigner–Seitz cell. Stored EPC values already contain WS degeneracy
weights. Neither an extra degeneracy division nor a lower-orbital conjugation
is applied. Both Fourier phases use `exp(+2πi k·Re)` / `exp(+2πi q·Rp)`.
The two HDF5 Cartesian IFC axes reverse the Fortran array axes and are decoded
accordingly. See the [QE2PERT documentation](https://perturbo-code.github.io/mydoc_qe2pert.html)
for the upstream workflow.

- `read_epr(path)` rejects a source marked polar. `polar='short_range'`
  explicitly accepts omission of its long-range phonon and EPC corrections.
  This is a partial effective model; long-range polar physics is not implemented.
- `data.hermiticity_audit()` checks the **unaliased image stencil**, including
  missing reverse terms. For derivatives the reverse is
  `(i,j,Re,a,Rp) → (j,i,-Re,a,Rp-Re)` with conjugation.
- `compile_supercell(..., hermiticity='require')` rejects defects above the
  declared tolerance. `hermiticity='project'` authorizes `(A+A†)/2` and records
  maximum and relative L2 defects. Raw source arrays are preserved.
- Complex coefficients are retained. Optional `real_tolerance=...` explicitly
  accepts discarding bounded imaginary coefficients and reports separate
  absolute and relative removed norms for `h` and `g`. This is not permission
  to ignore spectral assumptions of real-Hamiltonian surface-hopping methods.

The small DNTT fixture and the distinct q332/k664 DNTT input have unaliased EPC
Hermiticity defects. Projection changes their full-union sparse coefficients
by approximately 4.03% and 1.23% in relative L2 norm. Their source-compatible
finite grids retain much smaller numerical residues: sampled-displacement
matrix changes were `3.2e-13 Ry` (2×2×2) and `8.2e-11 Ry` (3×3×2), while
off-grid interpolation can change. Projected off-grid data must not be described as primary-source
exact. These numbers concern the stencil norm, not an estimated error bar on
an observable.

## Harmonic motion, stability, and scale

`PeriodicHarmonicBath(mesh, masses, ifc_atoms, ifc_cells, ifc_values)` diagonalizes
only `3*natoms` matrices at the finite mesh's wavevectors. FFTs transform real
Cartesian fields to/from the corresponding mass-weighted modes. This avoids a
dense `(3*natoms*ncells)^2` supercell mode matrix. `NormalModeBath.from_hessian`
is the independently checked dense finite-system alternative.

Signed squared frequencies are retained. Physical negative modes fail unless
the caller explicitly freezes a subspace. `zero_tolerance` is a numerical
**squared-frequency** tolerance. `frozen_below=w` freezes all modes with
`omega² <= w²`, including negative modes; it changes the dynamics and its count
must be reported. Active zero modes move freely and need explicit initial
positions when sampling, because their thermal position distribution is not
normalizable. No frequency floor, acoustic sum-rule correction, or negative-mode
deletion is implicit.

Electronic and IFC coefficients remain primitive-cell stencils. EPC evaluation
uses bounded term batches; increasing the cell count grows the neighbor map
and states, not a replicated Cartesian EPC Jacobian. The general coupled IFC
force currently uses a vectorized term-by-cell workspace. The example's exact
thermal response propagates the full electronic `U`, with quadratic electronic
storage; this does not establish scalability of exact thermal traces to
arbitrary system sizes.

For action-based finite-temperature carrier columns, this model and the FFT
bath compose with `make_column_transport_problem`. Qualify numerical filtering
and propagation separately from finite-rank trace sampling.

`compile_supercell(..., term_batch_size=...)` controls the bounded EPC loop.
Larger batches may improve throughput at increased workspace cost. Keep 256
as the initial choice and tune complete representative calculations.

`compile_supercell(..., epc_backend="fft")` alternatively evaluates periodic
EPC translations with Fourier transforms. It preserves the original dynamic
coefficients, parameter derivatives and electronic image currents. The
[Fourier guide](FOURIER_EPC.md) explains the explicit memory limit and when
this optional evaluator is useful; direct bounded-batch evaluation remains
the default. The example below accepts `--epc-backend fft` for this choice.

Tune the evaluator against the actual cell and workflow at matched numerical
accuracy; the best memory/throughput choice depends on those inputs.

Currents retain each periodic image's unwrapped displacement
`Re@cell + center[j]-center[i]`. Collapsing multiple images to a minimum-image
edge would lose current information. The current is the Peierls derivative
`J=charge * ∂H(kappa)/∂kappa` of this effective fixed-Wannier-center model,
not a complete first-principles optical velocity operator.

## Executable workflow

This short calculation exercises a 2×2×2 DNTT cell and explicitly freezes modes
below 1 meV. That cutoff is a demonstration constraint, not a validated DNTT
approximation. The short output is a current correlation, not a mobility.

```bash
JAX_ENABLE_X64=1 PYTHONPATH=src .venv/bin/python examples/abinitio_epc.py \
  --epr tests/compatibility/data/preprocessing/DNTT_epr.h5 \
  --output run-dntt --mesh 2 2 2 --carrier hole \
  --polar short_range --hermiticity project \
  --zero-tolerance 1e-16 --freeze-below-mev 1 \
  --temperature-kelvin 300 --trajectories 4 --dt-fs .01 --steps 20
```

`trajectory.h5` streams per-trajectory currents and unitarity diagnostics;
`ensemble.h5` streams means and separate real/imaginary nuclear-sampling SEMs.
`segment_start_state.npz`, `checkpoint.h5`, and `run.json` preserve the segment's
starting state, complete restart, ingestion evidence, source hash, signed-mode policy,
runtime manifest, and units. SEM is undefined with one trajectory. Output
directories must be new. Resume into a new directory using
`--resume run-dntt/checkpoint.h5` and the same physical and ensemble arguments;
only steps, chunk size, and save frequency may change. The reference response
origin and thermal insertions remain in the checkpoint.

For coupled dynamics, compose the same compiled model and parameters with
`Problem(model, params, CoupledClassical(compiled.masses), Ehrenfest())` and a
normalized electronic vector. Harmonic bath constraints do not automatically
carry over to coupled nuclei. Real-Hamiltonian MASH variants additionally
require a justified real conversion and the method's spectral assumptions;
generic DNTT surface-hopping validity has not been established.

## Independent checks and remaining physical work

`tests/test_cartesian_epc.py` checks literal translated dense equations,
complex weights, contracted forces, multiple-image currents, coherent units,
empty EPC/IFC limits, and parameter differentiation. `tests/test_epr_adapter.py`
uses independently labeled complex synthetic WS data, validates storage axes,
and audits DNTT source-grid/off-grid behavior and acoustic numerical zeros.

`benchmarks/epr_cartesian_reference.py --epr FILE --output NEW_DIRECTORY`
assembles separate NumPy Gamma-cell equations and solves CPA/Ehrenfest with
SciPy DOP853. Both DNTT files show fourth-order CPA electronic convergence,
second-order coupled nuclear convergence, and bitwise checkpoint restart in the
recorded CPU configuration. Accelerator continuation has a separate numerical
qualification policy; see [restart scope](RESTART.md).
This verifies ingestion-to-dynamics equations over a short interval; it does
not validate long-time statistics. The Gamma-cell EPC uniform-translation sum
is nonzero (about `7.8e-4 Ry/Bohr` for these inputs); no acoustic correction is
silently applied. Basis/gauge conventions and any desired translation
constraint need a separate physical investigation.

`benchmarks/epr_finite_cell_reference.py` independently checks arbitrary finite
cell action, force and image-current equations. `benchmarks/epr_mapping_reference.py`
checks a real isolated two-state Gamma trajectory for MASH2 and MASHRM against
a separate smooth surface ODE, with explicit `real_tolerance=1e-9`. The default
complex model is rejected. This check requires zero events and makes no claim
about DNTT hopping statistics, crossings or degeneracies.

Long-range polar terms, source-grid convergence, linear-EPC validity over the
thermal displacement range, carrier equilibrium weighting, finite-size/time
convergence, and a transport plateau remain necessary before a materials
mobility claim. The larger independent dataset is available in the
[first-principles transport data repository](https://github.com/JoonhoLee-Group/first-principles-transport-data)
under its data license; this package does not copy that external file into its
distribution.
