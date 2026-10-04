# QE phonon file preparation and collection

`pyeph.preprocessing.qe` contains host tools for the maintained PyEPH DFPT
workflow. They prepare chunk directories, inspect completed files, merge
explicitly mapped binary records, and collect data for `qe2pert.x`. They do
not run QE, submit jobs, choose physical convergence settings, or localize
phonons. The implementations use the Python standard library; importing the
containing `pyeph` package still imports its normal installed dependencies.

The code and synthetic tests were adapted from BSD-3-Clause PyEPH revision
`6c4693acbb69a06a5bc8b0593abde2170ff38843`. The original checkout is not a
runtime dependency. Thin command wrappers and editable input examples live
in [`examples/qe_chunked_irreps`](../examples/qe_chunked_irreps/).

## Operations and Python interfaces

| Module / command | Reusable operation | Contract |
| --- | --- | --- |
| `make_chunks` | `make_chunks(**configuration)` | Validate a prepared template and split a chosen q point's contiguous irrep range |
| `audit_chunks` | `audit(AuditConfig(...))` | Read-only irrep/mode, pattern, file-size and optional record-hash checks |
| `collect_dynmat` | `inspect(manifest)` then `stage(report)` | Gather one q point's identical patterns and `dynmat.q.0…N.xml` without hiding duplicate conflicts |
| `merge_dvscf` | `merge(manifest, block_bytes=...)` | Stream explicitly indexed cumulative mode records into a new binary file and hash receipt |
| `collect_phonons` | `inspect(...)`, `stage(report)` or `collect(...)` | Gather a completed multi-q calculation across image directories into the standard `save/` tree |

Run any module with `python -m pyeph.preprocessing.qe.<module> --help`.
The original four CLI argument sets remain supported by the corresponding
example wrappers. No plugin registry, electronic-structure subprocess layer,
or dynamics callback is involved.

## Prepare and audit chunks

Start with a **real preparatory QE calculation**. Copy the real `dyn0`, SCF
XML and all displacement patterns into the layout described by
[`TEMPLATE.example/README.md`](../examples/qe_chunked_irreps/TEMPLATE.example/README.md).
Supply the shared SCF save directory and, when applicable, the D3 Hessian
explicitly. The template must contain real files; the generator creates the
shared-data symlinks itself. Edit every `TODO` in the scheduler and phonon
input before using them on a cluster.

```bash
python -m pyeph.preprocessing.qe.make_chunks \
  --template /work/TEMPLATE --output-dir /work/q2-chunks \
  --prefix sample --q-index 2 --start-irr 5 --last-irr 9 --chunks 2 \
  --shared-save /work/scf/sample.save --dry-run
```

The requested upper bound must fit `NUMBER_IRR_REP` in that q point's
pattern file. Remove `--dry-run` to generate directories, tokenized inputs,
per-chunk provenance and a `chunks.q2.json` manifest. Prefixes, directory
prefixes and scheduler job prefixes accept only plain names; embedded shell
syntax and path separators are rejected.

After running the jobs through the user's existing QE environment, inspect
their outputs. `record_bytes` is an explicit, externally verified property
of the specific QE binary/file layout; the tool does not infer it from a
file length.

```bash
python -m pyeph.preprocessing.qe.audit_chunks \
  --q-index 2 --record-bytes 4096 --base-dvscf /work/base.dvscf \
  --chunk /work/q2-chunks/chunk1 --chunk /work/q2-chunks/chunk2
```

The audit reads each `ph.out` representation-to-mode map. Irreps can have
multiple modes, so an irrep number is **not** a binary-record offset. The
base file must end immediately before the first contributed chunk mode;
the chunks must then cover all remaining irreps and modes without overlaps.

## Collect XML and merge binary records

Edit [`collect.example.json`](../examples/qe_chunked_irreps/collect.example.json)
and [`merge.example.json`](../examples/qe_chunked_irreps/merge.example.json)
with explicit input/output paths and provenance. Paths in JSON resolve from
the process working directory; absolute paths are preferable for batch jobs.

```bash
python -m pyeph.preprocessing.qe.collect_dynmat collect.json --dry-run
python -m pyeph.preprocessing.qe.collect_dynmat collect.json
python -m pyeph.preprocessing.qe.merge_dvscf merge.json --dry-run
python -m pyeph.preprocessing.qe.merge_dvscf merge.json
```

XML collection requires complete irrep `0…N` coverage and byte-identical
duplicate files. Binary segments specify one-based cumulative
`first_mode`/`last_mode` ranges. Every source must have the expected extent,
the contributed ranges must exactly cover the target, and each contributed
record must contain nonzero bytes. A full-sized file with sparse zero holes
cannot pass merely because its final byte offset is correct.

After merging, repeat the audit with `--final-dvscf`, `--final-phsave` and
`--check-all-records`. The cheaper `--check-records` option checks only the
first and last record of each segment and can miss interior corruption.
Full-record checks can read very large files and should be scheduled as
ordinary batch I/O work.

## Collect the completed multi-q calculation

The standard collector replaces the old Linux-specific shell internals.
It requires the consolidated `tmp/_ph0/<prefix>.phsave`, `dyn0`, and
`<prefix>.dynN.xml` files. For q=1 it checks the two supported `_ph0` layouts;
for later q points it discovers `<prefix>.q_N/<prefix>.dvscf1` across scratch
image directories. Duplicate candidates must have identical sizes and
hashes. All q points must have the same dvscf extent.

```bash
python -m pyeph.preprocessing.qe.collect_phonons \
  --prefix sample --work-root /work/phonons --output /work/phonons/save
```

The resulting directory contains `<prefix>.phsave/`, `<prefix>.dyn0`,
`<prefix>.dynN.xml`, `<prefix>.dvscf_qN`, and a collection receipt with input
paths and file hashes. The compatibility wrapper also accepts the original
`PREFIX`, `WORK_ROOT`, `TMP_ROOT`, `SAVE_DIR`, and `DYN0` environment variables;
`PYTHON` optionally selects the installed interpreter:

```bash
PREFIX=sample WORK_ROOT=/work/phonons \
  bash examples/qe_chunked_irreps/ph_collect.sh
```

This standard collector checks required per-q patterns and irrep-zero
files. Full repaired-chunk mode coverage must first pass `audit_chunks`;
the collector does not reconstruct missing QE dynamical matrices.

## Publication and failure handling

Sources are read-only. Existing outputs, dangling output symlinks and known
`.partial` outputs are refused. A failed operation removes only its own
uniquely named staging directory; it never deletes another operation's
partial output. Copies are flushed and verified before publication.

Complete directories are published by a same-filesystem rename under an
exclusive sibling `.publish-lock`; cooperating writers must respect that
lock. Portable Python cannot guarantee no-replace directory rename against
an unrelated writer that ignores the lock. Binary files and receipts use
atomic no-replace hard links, requiring filesystem hard-link support.

The multi-chunk generator publishes its manifest last. An interruption can
leave complete chunk directories without the final manifest. Binary output
and receipt are separate files; an interruption can leave a receipt without
its data file. Existing evidence is retained for inspection and is not
automatically repaired or overwritten. File flushes do not imply a tested
power-loss recovery guarantee for every cluster filesystem.

## Qualification and remaining external boundaries

`tests/test_qe_preprocessing.py` ports all eight original test methods and
adds rejection/failure cases for unsafe paths, out-of-range irreps, existing
files and partial directories, source changes after inspection, binary
publication races, and nonpositive record sizes. The synthetic tests cover
multi-dimensional irreps, interior sparse-record damage, missing XML tails,
conflicting duplicates, cross-image q discovery, staging and receipts.
One synthetic integration test invokes all five module commands from a
temporary working directory, completing generation, collection, merging,
full-record audit and final multi-q staging without the old checkout.

One old generator fixture requested irreps 5–9 while declaring two irreps
in its XML. The migrated valid fixture declares nine; a separate test now
requires the inconsistent case to fail. MacOS `/var` versus `/private/var`
symlink aliases are normalized in the copied path assertions.

These tests do not execute QE, MPI inside QE, SLURM, D3, Wannier90 or
`qe2pert.x`. Partial-irrep binary merging remains an advanced, QE-version-
specific workflow. The caller must establish its record format and physical
compatibility; no PAW, noncollinear, spin-dependent or arbitrary scratch-file
repair claim follows from synthetic byte-record tests.

The separately licensed QCPBC optimizer remains in the compatibility
`post_qe2pert` boundary. Its optional import and some loss helpers are tested;
licensed optimizer convergence and saved localization outputs are not
qualified here. The general Cartesian EPR dynamics route is documented in
[`AB_INITIO_EPC.md`](AB_INITIO_EPC.md) and does not require that optimizer.
