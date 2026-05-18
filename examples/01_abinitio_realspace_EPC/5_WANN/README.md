# Step 5: Wannierization

**Programs:** `wannier90.x`, `pw2wannier90.x`

**Purpose:** Construct maximally localized Wannier functions from the NSCF wavefunctions.

## Input Files

| File | Description |
|------|-------------|
| `PREFIX.win` | Wannier90 control file |
| `pw2wan.in` | QE-to-Wannier90 interface input |

## Key Parameters to Tune

| Parameter | Description |
|-----------|-------------|
| `num_wann` | Number of Wannier functions (= number of target bands) |
| `dis_win_min/max` | Disentanglement energy window (eV) |
| `dis_froz_min/max` | Frozen window (bands kept exactly) |
| `projections` | Initial guess; `random` works for many cases |
| `mp_grid` | Must match the k-grid in `nscf.in` |
| `kpoint_path` | High-symmetry path for band interpolation |

Use `detect_wannier_window.py` to determine the energy windows in `nscf.out`.

The submit script typically runs:
1. `wannier90.x -pp PREFIX` (preprocessing: generates `PREFIX.nnkp`)
2. `mpirun -np 8 pw2wannier90.x -i pw2wan.in > pw2wan.out` (computes overlaps `PREFIX.mmn`, `PREFIX.amn`)
3. `wannier90.x PREFIX` (minimizes spread, produces `PREFIX_hr.dat`)

### Setting Energy Windows

Check the **highest occupied / lowest unoccupied** energies in `nscf.out` to guide
the `dis_win_min/max` and `dis_froz_min/max` settings.

## Troubleshooting

| Error | Solution |
|-------|----------|
| `kmesh_get_bvector: Not enough bvectors found` | Add `kmesh_tol = 1e-4` to `PREFIX.win` |
| Wannier spread not converging | Try different projections, adjust windows |
| Band structure looks wrong | Check `dis_win` range, compare with DFT bands |
