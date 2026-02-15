# Step 4: NSCF Calculation

**Program:** `pw.x`

**Purpose:** Non-self-consistent calculation on a **uniform k-grid** (no symmetry reduction) for Wannierization.

## Key Parameters

| Parameter | Description |
|-----------|-------------|
| `calculation = 'bands'` | NSCF mode |
| `nbnd` |  |
| K_POINTS | Explicit list in crystal coordinates (no automatic grid) |

## Generating K-Points

Use the Wannier90 utility:

```bash
wannier90/utility/kmesh.pl nk1 nk2 nk3
```

Paste the output into `nscf.in` under `K_POINTS crystal`.
The k-grid here must match `mp_grid` in the Wannier90 `.win` file.