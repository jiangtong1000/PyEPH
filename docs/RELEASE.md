# Releasing an explicit source inventory

`release-files.txt` is the publication boundary. Add a new file only after
reviewing its content, license and purpose. Local evidence, working outputs and
external reference bundles are outside this boundary. Required notices and
accurate dependency declarations remain part of any applicable release.

```sh
python tools/release.py audit --inventory release-files.txt
python tools/release.py export --inventory release-files.txt --destination /tmp/pyeph-source-release
```

The destination must be new. Exports reject symlinks and unlisted files. The
optional `--policy /path/to/local-policy.json` accepts a JSON object containing
`restricted_patterns`, a list of regular expressions. Keep that policy and
audit reports outside the exported directory. The scanner checks text and
supported numerical/archival formats recursively and rejects unknown binaries.
It complements source review and an explicit inventory; it does not determine
ownership or establish that an implementation is scientifically original.

Before a remote branch or tag is sent, stage only the inventory and run:

```sh
python tools/release.py audit --inventory release-files.txt --git --ref HEAD
```

The staged tree must exactly match both inventory and working files. History
checks inspect every reachable commit tree, historical filenames, blob contents,
commit messages and an annotated tag when supplied as the reference. A clean
current tree is insufficient if restricted material existed in earlier commits.
History audits reject shallow repositories; fetch complete history before
running the reference audit. The CI release gate uses a full-depth checkout.

## Distribution qualification

Build in the fresh export, never by repackaging a working environment:

```sh
python -m build --outdir /tmp/pyeph-distribution /tmp/pyeph-source-release
```

Inspect the actual wheel and source tarball with the same publication policy.
Install the wheel into a fresh environment, copy tests and fixtures from the
source distribution to an outside-checkout directory, and run them against the
installed package. Confirm import paths and runtime source hashes. Record
dependency versions, platform, warnings, skipped capabilities and test results.

For a current-dependency CPU check of the single wheel/source pair just built:

```sh
python -m venv /tmp/pyeph-wheel-check-env
/tmp/pyeph-wheel-check-env/bin/python -m pip install pytest /tmp/pyeph-distribution/*.whl
JAX_PLATFORMS=cpu /tmp/pyeph-wheel-check-env/bin/python \
  /tmp/pyeph-source-release/tools/qualify_distribution.py \
  --sdist /tmp/pyeph-distribution/*.tar.gz \
  --wheel /tmp/pyeph-distribution/*.whl \
  --destination /tmp/pyeph-installed-check
```

Use new environment and destination paths; the distribution directory must
contain exactly the intended wheel/source pair. The harness copies inventoried
support files without `src`, checks installed-wheel ownership and hashes, and
writes `qualification.json` and `pytest.log` in the destination. Reports retain
pytest arguments, relevant pytest environment settings, and the count and hash
of ordered selected test IDs. Selection describes collection, not completed or
passed tests; assess it together with the exit code and log. The default log
also includes the twenty slowest test phases.

For a focused check, put `--pytest-args -q -k EXPRESSION` last. The harness always
selects the `tests` root before these arguments; added positional paths do not
replace that root. Focused and collection-only runs are not full-suite evidence.
This CPU run does not qualify optional providers or another dependency/device
stack.

Use an explicit remote refspec after the audit. Do not push unrelated branches,
tags, caches, run outputs or local audit archives. A new release must not alter
the evidence attached to an earlier one.
