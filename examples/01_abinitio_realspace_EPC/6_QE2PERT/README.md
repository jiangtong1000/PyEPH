# Step 6: qe2pert

**Program:** `qe2pert.x`
**Purpose:** Combine Wannier90 output + phonon `save/` directory to produce `PREFIX_epr.h5`.


## Key Parameters

| Parameter | Description |
|-----------|-------------|
| `dft_band_min/max` | Must match the band range you Wannierized |
| `phdir` | Path to the `save/` directory created by `ph_collect.sh` in step 3 |

## Output

- `PREFIX_epr.h5` -- the electron-phonon matrix in the Wannier basis, used by PyEPH in step 7.
