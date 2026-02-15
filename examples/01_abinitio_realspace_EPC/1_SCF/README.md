# Step 1: SCF Calculation

**Program:** `pw.x`

**Purpose:** Self-consistent field calculation with relaxed geometry on a coarse k-grid to get the density.

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