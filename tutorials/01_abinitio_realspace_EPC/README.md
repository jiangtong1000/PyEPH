# Workflow: Build First-Principles EPH Hamiltonian

Construct an real-space first-principles EPH Hamiltonian from crystal structure.

**Software stack:** Quantum ESPRESSO 7.3, Wannier90, Perturbo, PyEPH

## Workflow Overview

```
  Phonon path:     Geometry -> 0_VcRelax -> 1_SCF -> 2_D3HESS -> 3_PHONONS
  Wannier path:    Geometry -> 0_VcRelax -> 1_SCF -> 4_NSCF   -> 5_WANN
  Merge:           3_PHONONS + 5_WANN -> 6_QE2PERT -> 7_PyEPH
```

| Step | Dir | Program | Purpose |
|------|-----|---------|---------|
| 0 | `0_VcRelax/` | `pw.x` | lattice + atomic relaxation (optional if already relaxed) |
| 1 | `1_SCF/` | `pw.x` | SCF with relaxed geometry on coarse k-grid |
| 2 | `2_D3HESS/` | `d3hess.x` | D3 dispersion Hessian (only if `vdw_corr = grimme-d3`) |
| 3 | `3_PHONONS/` | `ph.x` | DFPT phonon calculation on coarse q-grid |
| 4 | `4_NSCF/` | `pw.x` | NSCF on finer k-grid (for Wannierization) |
| 5 | `5_WANN/` | `wannier90.x` / `pw2wannier90.x` | Wannier90 localization |
| 6 | `6_QE2PERT/` | `qe2pert.x` | Process data, output `epr.h5` |
| 7 | `7_PyEPH/` | PyEPH | Localization of EPC and build real-space EPC Hamiltonian |

## Quick Start

```bash
# 1. Copy template to your project
cp -r 01_abinitio_realspace_EPC/ /path/to/my_project/

# 2. Find all TODOs and fill them in
grep -rn "TODO" /path/to/my_project/

# 3. Set up tmp/ symlinks after SCF (step 1) completes
cd /path/to/my_project && bash setup_links.sh

# 4. Submit each step in order
cd 0_VcRelax && sbatch submit.sh   # wait for completion
cd ../1_SCF && sbatch submit.sh     # wait, then run setup_links.sh
cd ../2_D3HESS && sbatch submit.sh
cd ../3_PHONONS && sbatch submit.sh  # after done: bash ph_collect.sh
cd ../4_NSCF && sbatch submit.sh
cd ../5_WANN && sbatch submit.sh
cd ../6_QE2PERT && sbatch submit.sh
cd ../7_PolarEPH && sbatch submit.sh
```

## Step-by-Step Notes

### Step 0: vc-relax

- **Skip if** you already have a relaxed geometry (e.g., from experiment or prior calc).
- Extract the relaxed CELL_PARAMETERS and ATOMIC_POSITIONS from `scf.out` (look
  for the last `CELL_PARAMETERS` and `ATOMIC_POSITIONS` blocks before "Final enthalpy").
- Use these relaxed coordinates for **all subsequent steps**.

### Step 1: SCF

- Uses the **relaxed geometry** from step 0.
- The `tmp/` directory created here is shared (via symlinks) with steps 2–6.
- After this completes, run `bash setup_links.sh` from the project root.

### Step 2: D3 Hessian

- Only needed if you use `vdw_corr = 'grimme-d3'` in the SCF.
- Produces `PREFIX.hess` which is consumed by `ph.x` in step 3.
- **Remove the `dftd3_hess` line from `ph.in`/`ph2.in` if you skip this step.**

### Step 3: Phonons (DFPT)

- This is typically the **most expensive** step.
- **Image parallelization** (`-ni N`): distributes q-points across N images.
  Set `-ni` equal to the number of q-points (e.g., 8 for a 2×2×2 grid).
- After ph.x finishes, **run `ph2.in`** (merge step with `recover=.true.`).
- After ph2 finishes, **run `bash ph_collect.sh`** to collect data into `save/`.
- The `ph_collect.sh` script handles the dvscf file location difference between
  `-ni 1` and `-ni N` modes. Make sure `PH_NI` matches your `-ni` flag.

### Step 4: NSCF

- Computes wavefunctions on a **uniform k-grid** (no symmetry reduction).
- K-points must be explicit (crystal coords). Generate with:
  ```
  wannier90/utility/kmesh.pl nk1 nk2 nk3
  ```
- `nbnd` should be large enough to cover the disentanglement energy window.

### Step 5: Wannierization

**Input files:**
- `PREFIX.win` — Wannier90 control file
- `pw2wan.in` — QE-to-Wannier90 interface

**Key parameters to tune:**
- `num_wann`: number of Wannier functions (= number of target bands)
- `dis_win_min/max`: disentanglement energy window (eV). Check the **highest
  occupied / lowest unoccupied** energies in `nscf.out` to set this.
- `dis_froz_min/max`: frozen window (bands in this range kept exactly)
- `projections`: initial guess. `random` works for many cases.
- `kpoint_path`: use [SeeK-path](https://www.materialscloud.org/work/tools/seekpath)
  to determine the correct high-symmetry path for your crystal.

**Troubleshooting:**

| Error | Solution |
|-------|----------|
| `kmesh_get_bvector: Not enough bvectors found` | Add `kmesh_tol = 1e-4` to `PREFIX.win` |
| Wannier spread not converging | Try different projections, adjust windows |
| Band structure looks wrong | Check `dis_win` range, compare with DFT bands |

### Step 6: qe2pert

- Reads Wannier90 output + phonon `save/` directory → produces `PREFIX_epr.h5`.
- `dft_band_min/max` must match the band range you Wannierized.
- Consistency check: `dft_band_max - dft_band_min + 1 == num_wann`.
- `phdir` must point to the `save/` directory created by `ph_collect.sh`.

### Step 7: PyEPH post-processing

- Reads `PREFIX_epr.h5` and computes EPH coupling matrices on a finer q-grid.
- Adjust `Nx`, `Ny` for your desired q-grid resolution.
- Uses MPI parallelization; can run on many cores.
- Output: `eph_data_{Nx}.h5` containing `gmat_raw`, phonon frequencies/modes.

## Consistency Checklist

Before running, verify these parameters match across files:

| Parameter | Must match in |
|-----------|---------------|
| `prefix` | All `.in` files, `.win`, `ph_collect.sh`, `setup_links.sh` |
| `ecutwfc/ecutrho` | `scf.in`, `nscf.in` |
| k-grid (`nk1,nk2,nk3`) | `scf.in`, `nscf.in`, `.win` (`mp_grid`), `qe2pert.in` |
| q-grid (`nq1,nq2,nq3`) | `ph.in`, `ph2.in` |
| `num_wann` | `.win`, `qe2pert.in`, `run_pyeph.py` |
| `dft_band_min/max` | `qe2pert.in` (must = `num_wann` bands) |
| Cell & positions | `scf.in`, `nscf.in`, `.win` (all must use relaxed geometry) |
| `nbnd` | `nscf.in`, `.win` (`num_bands`) |

## File Structure

```
01_build_hamiltonian/
├── README.md              ← this file
├── setup_links.sh         ← run after SCF to create tmp/ symlinks
├── 0_VcRelax/
│   ├── scf.in
│   └── submit.sh
├── 1_SCF/
│   ├── scf.in
│   └── submit.sh
├── 2_D3HESS/
│   ├── d3hess.in
│   └── submit.sh
├── 3_PHONONS/
│   ├── ph.in              ← main phonon calc
│   ├── ph2.in             ← merge step (recover=.true.)
│   ├── ph_collect.sh      ← collect dvscf into save/
│   └── submit.sh
├── 4_NSCF/
│   ├── nscf.in
│   └── submit.sh
├── 5_WANN/
│   ├── PREFIX.win          ← rename to your prefix
│   ├── pw2wan.in
│   └── submit.sh
├── 6_QE2PERT/
│   ├── qe2pert.in
│   └── submit.sh
└── 7_PolarEPH/
    ├── run_pyeph.py
    └── submit.sh
```
