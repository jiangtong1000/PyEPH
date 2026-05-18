# Step 3: Phonons (DFPT)

**Program:** `ph.x`

**Purpose:** DFPT phonon calculation on a coarse q-grid. Typically the most expensive step.

The standard submit script runs two stages:
1. Heavy DFPT calculation with image parallelization (`-ni N`).
2. Merge step using `ph2.in` with `recover=.true.` (single image, collects the image-parallel results).

After the main run is complete:

```bash
bash ph_collect.sh
```

This collects the phonon outputs needed for step 6.

## Key Parameters

| Parameter | Description |
|-----------|-------------|
| `tr2_ph` | Convergence threshold for the phonon calculation. Organic crystals often need a very tight value such as `1.0d-17` to avoid spurious negative frequencies. |
| `nq1, nq2, nq3` | q-point grid |
| `-ni N` | Number of images; distributes q-points across images |
| `-nk N` | k-point pool parallelization |
| `lqdir=.true.` | Store per-q-point data in separate subdirectories (`tmp/_ph*/PREFIX.q_*/`) |
| `recover=.true.` | Enable restart from interrupted runs |
| `dftd3_hess` | Path to D3 Hessian file (remove if not using D3) |

## Image Parallelization

- Set `-ni` to distribute q-points across images. Each image handles `ceil(nq / ni)` q-points.
- Total MPI ranks = `ni * nk * npool_per_image`.
- `NPROC_IMAGE = SLURM_NTASKS / NIMAGE` is the number of MPI ranks per image.
- See the [QE Phonon User Guide -- Parallelism](https://www.quantum-espresso.org/Doc/ph_user_guide/node17.html) for the full explanation.

## Restarting An Interrupted Main Job

When a phonon job is killed (walltime, node failure, etc.), `ph.x` leaves recover files that record the state of each MPI rank. On restart, it reads these to resume. Use `recover=.true.` to restart the job.

### The `touch` trick in `submit.sh`

The submit script should contain a block like this before the main `mpirun` line:

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

```text
Error in routine check_initial_status (1):
    recover file found, change in start_q not allowed
```

or

```text
Error termination. Backtrace: At line 21 of file check_restart_recover.f90 (unit = 99, file = './tmp/*.recover') Fortran runtime error: File cannot be deleted
```

If this happens, remove the stale recover/restart files and restart with the current parallelization layout. `ph.x` will re-detect which q-points are already done from the saved `control_ph.xml` and `dynmat.*.xml` files in `_ph*/PREFIX.phsave/`.

## Merge Step (`ph2.in`)

After all images finish, run the merge with a single image (no `-ni`):

```bash
mpirun -np 24 ph.x -nk 8 < ph2.in > ph2.out
```

`ph2.in` should be identical to `ph.in` with `recover=.true.`. This collects results from all image directories into a single set of dynamical matrices.

## Per-Q Irrep Chunking

Use this workflow when one or a few q-points are still incomplete and you want to split the remaining irreducible representations across several helper jobs.

This example keeps chunk jobs lightweight:
- Each chunk targets one q-point with `start_q=last_q=q`.
- Each chunk gets a complementary `start_irr,last_irr` range.
- Chunk jobs reuse the same displacement patterns as the main run.
- Chunk jobs do not rely on `control_ph.xml` or `status_run.xml`.

### File Roles

| File | Role |
|------|------|
| `dynmat.#q.#irr.xml` | Per-irrep contribution for q-point `q`. Collect these from all chunks after finishing all chunks. |
| `patterns.#iq.xml` | Shared displacement patterns. These must be identical across all chunks and the main run (before chunking). |
| `PREFIX.dynq.xml` | Final dynamical matrix for one q-point, produced only after the final per-q recollection run. |

## Chunk Workspace Layout

This directory now includes a reusable chunk-job skeleton:

```text
3_PHONONS_Chunked/
├── TEMPLATE/
│   ├── PREFIX.dyn0
│   ├── ph.in
│   ├── submit.sh
│   └── tmp/
│       ├── PREFIX.xml
│       └── _ph0/
│           └── PREFIX.phsave/
│               ├── patterns.1.xml
│               ├── ...
│               └── patterns.Nq.xml
├── SharedFiles/
│   └── PREFIX.save
│   └── PREFIX.hess
└── make_chunks.py
```

`TEMPLATE/` is intentionally lightweight. The committed files under `TEMPLATE/` are placeholders. Before real use, replace them with files copied from the main phonon run:
- `PREFIX.dyn0`
- `tmp/PREFIX.xml`
- `tmp/_ph0/PREFIX.phsave/patterns.1.xml ... patterns.NPH.xml`

This example ships with `patterns.1.xml ... patterns.8.xml` placeholders because the example q-grid uses `NPH=8`. If your system has a different number of irreducible q-points, replace the placeholder files and adjust the count accordingly.

## `make_chunks.py`

`make_chunks.py` is the reusable helper that clones `TEMPLATE/` into one directory per irrep chunk.

Edit the small config block at the top of the script:
- `PREFIX`
- `Q_INDEX`
- `START_IRR`, `END_IRR`
- `TOTAL_CHUNKS`
- `TEMPLATE_DIR`
- `SHARED_DIR`
- `CHUNK_DIR_PREFIX`
- `JOB_PREFIX`

Then run:

```bash
python3 make_chunks.py
```

The script will:
- clone `TEMPLATE/`
- rename placeholder `PREFIX` paths to the chosen prefix
- replace `tmp/PREFIX.save` and `tmp/PREFIX.hess` with symlinks into `SharedFiles/`
- rewrite `ph.in` `start_q,last_q`
- rewrite `ph.in` `start_irr,last_irr`
- rewrite the chunk job name in `submit.sh`

## Final Per-Q Recollection

After all chunks for one q-point finish:

1. Create a clean per-q collect directory, for example `_phq_collect`.
2. Gather the full set of `dynmat.q.*.xml` files there.
3. Copy the matching `patterns.1..NPH.xml` files into the collect directory.
4. For a true `recover=.true.` recollection, also copy `control_ph.xml` and the q-appropriate `status_run.xml` into `tmp/_ph0/PREFIX.phsave/`.
5. Provide the base `PREFIX.save`, `PREFIX.hess`, and `PREFIX.xml`.
6. Also provide the matching `PREFIX.q_q/` directory.
7. Run `ph.x` there with:
   - `start_q=last_q=q`
   - no `start_irr,last_irr`
   - single-image mode

Use `control_ph.xml` and `status_run.xml` from the canonical main `_phq/PREFIX.phsave/`. If these XML files are missing, `ph.x` may drop out of recover mode and start from scratch instead of performing a clean recollection.

That recollection run produces the authoritative `PREFIX.dynq.xml` for that q-point.

## Restore Into The Main `3_PHONONS` Tree

Before later downstream collection:

1. Copy the merged `dynmat.q.*.xml` files back into the main run:
   - `tmp/_ph{q-1}/PREFIX.phsave/`
2. Copy the final `PREFIX.dynq.xml` back into the main `3_PHONONS` directory.
3. Repeat for every q-point that was chunked.
4. Then run the usual downstream collection:

```bash
bash ph_collect.sh
```

`ph_collect.sh` does not know about chunk directories. It only reads the consolidated main `3_PHONONS` tree, so restore the merged per-q results there first.
