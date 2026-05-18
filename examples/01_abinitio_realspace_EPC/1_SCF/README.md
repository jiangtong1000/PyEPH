# Step 1: SCF Calculation

**Program:** `pw.x`

**Purpose:** Self-consistent field calculation with relaxed geometry on a coarse k-grid to get the density.

## Notes on parameters
- `forc_conv_thr = 1.0d-5` we might need a tight enough force convergence threshold for subsequent phonon calculations.
- `nstep = 200`, same as above, enough steps
- `ibrav = 0`, just let QE figure out the symmetry by itself
- `ecutrho=4*ecutwfc`, the charge density cutoff should be 4 times the wavefunction cutoff
- `dftd3_threebody = .false.`, since the hessian is not supported
- `conv_thr = 1.0d-15`, the convergence threshold needs to be tight enough for subsequent phonon calculations.

## Parallelization

The `-nk` (k-point pools) used here must stay consistent with the phonon step.
The `pw.x` SCF is run **without** `-nimage`; each phonon image then inherits
the same per-image parallelization layout.

For example, if SCF uses:

```bash
mpirun -np 8 pw.x -nk 2 < scf.in > scf.out
```

then `ph.x` with 8 images needs `8 * 8 = 64` total ranks, with the same `-nk 2`
inside each image:

```bash
mpirun -np 64 ph.x -ni 8 -nk 2 < ph.in > ph.out
```

In general: `total_ranks = ranks_per_image * NI`, and `ranks_per_image` and `-nk`
must match the SCF.

See the [QE Phonon User Guide -- Parallelism](https://www.quantum-espresso.org/Doc/ph_user_guide/node17.html)
for the full explanation.