# Step 3: Phonons (DFPT)

**Program:** `ph.x`

**Purpose:** DFPT phonon calculation on a coarse q-grid. Typically the **most expensive** step.

The submit script runs two stages:
1. **Heavy DFPT calculation** with image parallelization (`-ni N`).
2. **Merge step** using `ph2.in` with `recover=.true.` (single image, collects results).

After both complete:

```bash
bash ph_collect.sh
```

This collects dynamical matrices and dvscf files into `save/` for step 6.

## Key Parameters

| Parameter | Description |
|-----------|-------------|
| `nq1, nq2, nq3` | q-point grid (must match in `ph.in` and `ph2.in`) |
| `-ni N` | Number of images; distributes q-points across N images |
| `-nk N` | k-point pool parallelization |
| `lqdir=.true.` | Store per-q-point data in separate subdirectories (`tmp/_ph*/DNTT.q_*/`) |
| `recover=.true.` | Enable restart from interrupted runs |
| `dftd3_hess` | Path to D3 Hessian file (remove if not using D3) |

## Image Parallelization

- Set `-ni` to distribute q-points across images. Each image handles `ceil(nq / ni)` q-points.
- Total MPI ranks = `ni * nk * npool_per_image`.
- `NPROC_IMAGE = SLURM_NTASKS / NIMAGE` is the number of MPI ranks per image.
- See the [QE Phonon User Guide -- Parallelism](https://www.quantum-espresso.org/Doc/ph_user_guide/node17.html)
for the full explanation.

## Restarting an Interrupted Job

When a phonon job is killed (walltime, node failure, etc.), `ph.x` leaves recover
files that record the state of each MPI rank. On restart, it reads these to resume. Use the `recover=.true.` to restart the job.

### The `touch` trick in `submit.sh`

The submit script should contain a block like this **before** the `mpirun` line:

```bash
PREFIX=DNTT          # must match prefix in ph.in / scf.in
OUTDIR=tmp
NIMAGE=9             # must match the -ni flag in the mpirun line below
NPROC_IMAGE=$((SLURM_NTASKS / NIMAGE))

# Remove stale recover/restart files everywhere (top-level + inside lqdir q-point dirs)
rm -f "$OUTDIR"/${PREFIX}.recover* "$OUTDIR"/${PREFIX}.restart_k*
find "$OUTDIR"/_ph*/${PREFIX}.q_*/ -name "${PREFIX}.recover*" -delete 2>/dev/null
find "$OUTDIR"/_ph*/${PREFIX}.q_*/ -name "${PREFIX}.restart_k*" -delete 2>/dev/null

# Create fresh recover/restart files
for i in $(seq 1 "$NPROC_IMAGE"); do
    suffix=""
    if [ "$i" -ne 1 ]; then suffix="$i"; fi
    touch "$OUTDIR/${PREFIX}.recover${suffix}"
    touch "$OUTDIR/${PREFIX}.restart_k${suffix}"
done
```

This ensures recover files always match the current parallelization layout.

### Common restart error

```
Error in routine check_initial_status (1):
    recover file found, change in start_q not allowed
```
or
```
Error termination. Backtrace: At line 21 of file check_restart_recover.f90 (unit = 99, file = './tmp/*.recover') Fortran runtime error: File cannot be deleted
```

Then  will re-detect which q-points are
already done from the saved `control_ph.xml` and `dynmat.*.xml` files in
`_ph*/PREFIX.phsave/`, and resume the incomplete ones.

## Merge Step (`ph2.in`)

After all images finish, run the merge with a **single image** (no `-ni`):

```bash
mpirun -np 24 ph.x -nk 8 < ph2.in > ph2.out
```

`ph2.in` should be identical to `ph.in` with `recover=.true.`. This collects
results from all image directories into a single set of dynamical matrices.

## After Completion

Run the collection script:

```bash
bash ph_collect.sh
```

Make sure `PH_NI` in `ph_collect.sh` matches your `-ni` flag. The script handles
the dvscf file location difference between `-ni 1` and `-ni N` modes.