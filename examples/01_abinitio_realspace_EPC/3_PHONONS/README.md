# Step 3: Phonons (DFPT)

**Program:** Quantum ESPRESSO `ph.x`

This step calculates phonons and electron-phonon perturbations on the coarse q grid
needed by `qe2pert.x`. It is usually the most expensive part of the workflow.

## Standard workflow

1. Prepare `ph.in` from the same prefix, `outdir`, pseudopotentials, and coarse-grid SCF
   data used in step 1.
2. Run the heavy calculation with q-point image parallelism when useful.
3. After every q point is complete, run `ph2.in` with `recover=.true.` and one image to
   recollect and diagonalize the dynamical matrices.
4. Export the consolidated files for step 6 with `ph_collect.sh`.

`submit.sh` illustrates the first two runs. Scheduler resources, environment setup,
image count, and k-point pools must be chosen for the target calculation and cluster.

## Inputs that require a decision

| Setting | Guidance |
|---------|----------|
| `prefix`, `outdir` | Must refer to the matching SCF calculation. |
| `nq1`, `nq2`, `nq3` | Coarse q grid used by the downstream Perturbo workflow. |
| `tr2_ph` | Converge for the material; `1d-12` is the QE default, not a universal answer. |
| `epsil` | Use only where dielectric/Born-charge calculations are appropriate. |
| `dftd3_hess` | Keep only when the SCF uses the matching D3 correction and Hessian. |
| `fildvscf` | Required for the perturbation data consumed downstream. |
| `max_seconds` | Set below the scheduler wall time to give QE time to stop cleanly. |

Do not use `asr` or an unusually tight `tr2_ph` as a generic fix for unstable modes.
Test the physical and numerical convergence of the calculation.

## Q-point images are not q-point count

`-ni N` selects the number of QE images. It does not declare the number of irreducible
q points, and one image may process more than one q point. Read the q list from
`PREFIX.dyn0` or `control_ph.xml` after the preparatory run.

With nonblank `fildvscf`, use image parallelism to distribute q points. QE deliberately
does not distribute a q point's irreducible representations across images in this
configuration.

See the [QE PHonon parallelism guide](https://www.quantum-espresso.org/Doc/ph_user_guide/node17.html)
and [`ph.x` input reference](https://www.quantum-espresso.org/Doc/INPUT_PH.html).

## Restart policy

- Keep the SCF save tree and `PREFIX.phsave` from the same physical calculation.
- Use `recover=.true.` only with that matching scratch state.
- Use `max_seconds` to stop before the scheduler kills the job.
- If a raw recover file is corrupt, remove it as described by QE and retain the XML
  `phsave` data that records completed work.
- Do not pre-create empty `recover` or `restart_k` files as a general procedure.
- Do not change q/irrep selectors or parallel decomposition underneath incompatible
  raw restart files.

## Final recollection

Run the final `ph.x` pass with `recover=.true.`, the same physical input, no irrep
selectors, and one image. A missing `dynmat` contribution may cause this pass to do
expensive DFPT work rather than merely collect, so submit it with realistic resources
and verify `JOB DONE` in `ph2.out`.

## Export for `qe2pert.x`

The collector reads the q count from `PREFIX.dyn0`, discovers q-specific `dvscf` files
across image directories, rejects ambiguous sources and unequal final file sizes, and
promotes a clean staging tree only after all copies succeed.

Run this I/O-heavy step through the batch scheduler for large calculations:

```bash
PREFIX=my_material bash ph_collect.sh
```

The script refuses the literal `PREFIX` placeholder and existing `save/` or
`save.partial/` directories. Inspect the staged source paths it prints before using the
result in step 6.

## Advanced: split one q point by irrep

When a q point cannot finish as one job, use the validated
[per-q irrep chunking tutorial](chunked_irreps/). This is an advanced recovery workflow:

- QE supports collecting the per-irrep `dynmat.#q.#irr.xml` files.
- QE does not document merging partial-irrep `dvscf` binaries.
- Binary merging must use cumulative mode numbers, an explicit manifest, the exact QE
  I/O layout, and full source-to-output record validation.

Do not copy a `dvscf` file from an arbitrary chunk and do not use irrep numbers as
binary record offsets.
