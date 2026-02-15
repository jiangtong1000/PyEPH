# Step 2: D3 Dispersion Hessian

**Program:** `d3hess.x`

**Purpose:** Compute the DFT-D3 Hessian for van der Waals corrections.

## When to Skip

- Only needed if you use `vdw_corr = 'grimme-d3'` in VcRelax and SCF.
- If you skip this step, **remove the `dftd3_hess` line** from `ph.in` and `ph2.in` in step 3.

## How to Run

```bash
sbatch submit.sh
```

## Output

- `PREFIX.hess` -- consumed by `ph.x` in step 3.

## Notes
