# Step 0: Variable-Cell Relaxation

**Program:** `pw.x`

**Purpose:** Relax lattice parameters and atomic positions.

## How to get xyz from .cif file
there is a util script under `pyeph/utils/cifio.py` to extract the `CELL_PARAMETERS` and `ATOMIC_POSITIONS` from a .cif file.

## While running
Running with symmetry is the default setting in QE.
Check the `scf.out` and search for `Sym. Ops.` which should be consistent with the output from running `cifio.py`. If not, something might be wrong.

## After Completion
- Check if the relaxation is converged, force and stress should be converged.
- Extract the relaxed `CELL_PARAMETERS` and `ATOMIC_POSITIONS` from `scf.out`.
  Look for the **last** `CELL_PARAMETERS` and `ATOMIC_POSITIONS` blocks.
- Use these relaxed coordinates for **all subsequent steps** (SCF, NSCF, Wannier `.win`, etc.).