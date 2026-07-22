# Advanced per-q irrep chunking for QE phonons

Use this recovery workflow when one q point cannot complete within a practical job and
must be divided into complementary irreducible-representation ranges.

This tutorial has two different support levels:

- QE officially supports GRID-style `dynmat.#q.#irr.xml` calculation and recollection.
- Merging partial-irrep `dvscf` direct-access files is an advanced, version-pinned
  extension. QE does not document that binary merge as a public workflow.

Read the [QE `ph.x` input reference](https://www.quantum-espresso.org/Doc/INPUT_PH.html)
and [PHonon parallelism guide](https://www.quantum-espresso.org/Doc/ph_user_guide/node17.html)
before using this procedure.

## Non-negotiable rules

1. Generate one authoritative displacement-pattern set and use byte-identical
   `patterns.#q.xml` files in every worker.
2. Derive binary offsets from cumulative mode numbers printed by `ph.x`, not irrep
   numbers. One irrep can contain multiple modes.
3. Keep the QE build, SCF save tree, pseudopotentials, D3 Hessian, input files, and
   pattern hashes fixed across workers.
4. Never copy the `dvscf` from an arbitrary chunk. A chunk file can have the expected
   extent while containing zero records outside its computed mode range.
5. Never overwrite the original partial file during a merge.

## Boundary of the supplied tools

The tools use only the Python standard library and intentionally refuse ambiguous or
existing outputs.

| Tool | Purpose |
|------|---------|
| `make_chunks.py` | Validate a real template and generate complementary jobs plus a manifest. |
| `audit_chunks.py` | Check outputs, irrep/mode coverage, patterns, `dynmat`, binary extents, and optional record hashes. |
| `merge_dvscf.py` | Merge explicit cumulative-mode segments into a new file and write a hash receipt. |
| `collect_dynmat.py` | Build a conflict-checked per-q pattern/`dynmat` bundle and receipt. |

These tools do not infer the direct-access record size, repair arbitrary QE scratch
state, submit retries, or construct a complete multi-q recollection tree.

Do not use the binary merger unchanged for PAW `_paw` companion files,
spin/noncollinear layouts, or a different QE I/O implementation. Extend and validate
those cases against the exact QE source first.

## 1. Generate authoritative patterns

Run a preparatory `ph.x` calculation following QE's GRID procedure, for example with
`start_irr=0, last_irr=0`, to create the displacement patterns without solving the
expensive representations. Preserve the resulting `PREFIX.dyn0`, SCF XML/save tree,
and `PREFIX.phsave/patterns.#q.xml` files together.

Record at least:

- QE version, build, and executable hash;
- complete phonon input and scheduler command;
- SCF XML/save provenance and pseudopotential hashes;
- optional D3 Hessian hash;
- target q point, total irreps, and representation-to-mode map; and
- SHA-256 of every pattern used by a worker.

## 2. Prepare a real worker template

Copy `TEMPLATE.example` to an untracked calculation directory named `TEMPLATE` and
populate it with real files:

```text
TEMPLATE/
|-- PREFIX.dyn0
|-- ph.in
|-- submit.sh
`-- tmp/
    |-- PREFIX.xml
    `-- _ph0/PREFIX.phsave/
        |-- patterns.1.xml
        |-- ...
        `-- patterns.Nq.xml
```

Do not commit or use placeholder patterns. Do not seed workers with unrelated
`control_ph.xml`, `status_run.xml`, or completed `dynmat` files. Supply the immutable
`PREFIX.save` directory and optional `PREFIX.hess` as explicit absolute paths.

## 3. Generate complementary chunks

Determine the unfinished irrep range from `ph.out`, then run a dry plan first:

```bash
python3 make_chunks.py \
  --template TEMPLATE \
  --output-dir runs/q3 \
  --prefix my_material \
  --q-index 3 \
  --start-irr 101 \
  --last-irr 180 \
  --chunks 8 \
  --shared-save /absolute/path/my_material.save \
  --shared-hess /absolute/path/my_material.hess \
  --directory-prefix q3_chunk \
  --job-prefix q3 \
  --dry-run
```

Remove `--shared-hess` when D3 is not used. Remove `--dry-run` only after checking the
plan. Review every generated `ph.in` and `submit.sh`; scheduler account, resources,
pool count, environment, and wall time are calculation-specific.

The generator writes `chunk.json` inside each worker and `chunks.q3.json` beside the
workers. Retain both as provenance.

## 4. Run and restart workers

- Use `recover=.true.` only with scratch data from the same physical worker.
- Preserve the worker `PREFIX.phsave` files.
- Put `max_seconds` below the scheduler wall time for a clean stop.
- Remove corrupt raw recover files only as QE documents.
- Do not pre-create empty `recover` or `restart_k` files as a portable recipe.

A retry must use the same q/irrep range, patterns, SCF state, QE build, and compatible
parallel layout. Record retries in the run metadata rather than renaming directories
into an ordering convention.

## 5. Audit completed chunks

Obtain `RECORD_BYTES` from several file extents divided by their known final cumulative
mode and cross-check it against the exact QE implementation. For QE 7.3 direct-access
I/O this depends on padded dense FFT dimensions and magnetization layout; do not infer
it from irrep count.

```bash
RECORD_BYTES=REPLACE_WITH_VERIFIED_VALUE

python3 audit_chunks.py \
  --q-index 3 \
  --record-bytes "$RECORD_BYTES" \
  --base-dvscf /path/to/original.partial.dvscf1 \
  --final-dvscf /path/to/existing.final.dvscf_q3 \
  --final-phsave /path/to/final/PREFIX.phsave \
  --check-all-records \
  --chunk /path/to/q3_chunk1 \
  --chunk /path/to/q3_chunk2
```

Before a final binary exists, omit record comparison by leaving out
`--check-records`/`--check-all-records`; the other structural checks still run.
`--check-records` is a fast boundary smoke test. `--check-all-records` compares every
contributed source record with the final binary and should run through the batch
scheduler for production-size data.

The command exits nonzero for structural, convergence, pattern, size, zero-record, or
source-to-final failures and prints a JSON report suitable for retention.

## 6. Merge `dvscf` from an explicit mode manifest

Copy `merge.example.json` to the calculation directory. Replace every placeholder with
the verified cumulative mode ranges from `ph.out`. Every mode from 1 through
`total_modes` must appear exactly once.

```bash
python3 merge_dvscf.py merge.json --dry-run
python3 merge_dvscf.py merge.json
```

The merger requires each source extent to end at its declared `last_mode`, streams and
checks every contributed record, rejects all-zero records, writes a new `.partial`
file, fsyncs it, verifies final size, and promotes it without overwriting an existing
path. It also creates `OUTPUT.receipt.json` with provenance, whole-output SHA-256, and
per-segment hashes.

Run `audit_chunks.py --check-all-records` against the promoted file as an independent
post-merge check. Retain the input manifest, receipt, and audit report.

## 7. Stage `dynmat` without hidden conflicts

Copy `collect.example.json` and list the authoritative pattern file from every worker
plus all original/chunk phsave directories that contribute the target q point:

```bash
python3 collect_dynmat.py collect.q3.json --dry-run
python3 collect_dynmat.py collect.q3.json
```

The tool requires exactly `dynmat.Q.0.xml` through `dynmat.Q.Nirr.xml`, parses every
XML file, rejects differing patterns or conflicting same-name files, accepts only
byte-identical duplicates, and creates a clean per-q bundle with
`collection_receipt.json`.

This bundle is not a complete runnable `PREFIX.phsave`.

## 8. Perform QE's final recollection

Prepare a separate scratch tree from the canonical main calculation. Install only the
validated per-q pattern and `dynmat` bundle alongside the matching SCF data and QE
metadata required for recovery. Review a collection input that targets the q point and
contains no irrep selectors, then run single-image `ph.x` with realistic resources.

Require all of the following before restoring results to the main calculation:

- `ph.x` ends with `JOB DONE`;
- the final `PREFIX.dynQ.xml` parses;
- the recollected phsave has the exact expected `0..Nirr` set;
- the merged `dvscf` passes the full source-to-final audit; and
- the collection input, output, manifests, and receipts are retained.

After every repaired q point is restored and the standard final recollection succeeds,
run the parent directory's validated `ph_collect.sh` to build the step-6 input tree.

## Tests

```bash
python3 -m unittest discover -s tests -v
```

The tests cover multidimensional irreps, an interior sparse gap that boundary-only
checks miss, missing-tail and duplicate-conflict collection cases, safe staging, and
merge/collection receipts.
