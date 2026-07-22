# Template staging area

This example is intentionally incomplete and must not be submitted directly.

1. Replace `TODO_NQ*` in `ph.in` with the real q grid and use the converged `tr2_ph`.
2. Copy the real preparatory-run `dyn0` to `PREFIX.dyn0`.
3. Copy the real SCF XML to `tmp/PREFIX.xml`.
4. Copy all real preparatory-run patterns to
   `tmp/_ph0/PREFIX.phsave/patterns.#q.xml`.
5. Review the scheduler resources and export `QE_ENV` and `NPOOL` before submission.
6. Do not copy placeholder patterns, stale status files, or unrelated dynmat files.

`make_chunks.py` will refuse this directory until those required artifacts exist.
