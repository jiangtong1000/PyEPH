# Qualification scope

This is a development release with bounded scientific and platform scope.

The `0.1.0.dev4` candidate adds fixed-provider scalar campaign continuation. Its
own installed macOS CPU qualification passed against one audited wheel and
source archive:

| Dependency profile | Selection | Result |
| --- | --- | --- |
| Current, Python 3.13 / JAX 0.11.2 | Complete released suite | 2120 passed, 8 skipped, 4 expected failures; no warnings |
| Minimum, Python 3.11 / JAX 0.6.2 | Complete released suite | 2120 passed, 8 skipped, 4 expected failures; 351 warnings retained |
| Optional Torch, Python 3.12 | Selected adapter and shared-integration checks | 120 passed; no warnings |

Both complete runs selected every collected test, with no deselections. The
eight skips require optional Torch; the four expected failures preserve the
documented archival golden-data discrepancies described below. Minimum-stack
warnings remain visible; passing tests do not establish a blanket explanation
for their backend cause.

The tested wheel SHA256 is
`da62e41ee0fd2d9fcbff0e437069ba6153440b8130e0f3058eeb96b6bf5e3743`
and source-archive SHA256 is
`c0283ab6ba2a717cf31f14cb93075341d5ef2a74dc69b1e627fe06d6a6827506`.
The archive contains 392 explicitly inventoried files. All 144 installed
runtime files and 248 copied support files remained unchanged. Fresh package
environments reused existing numerical dependencies; these are not fresh
dependency-resolution runs.

Continuation checks cover actual process crashes, concurrent workers,
same-segmentation restart agreement, independently computed ensemble moments,
corrupt artifacts, identity changes and numerical/domain rejection. Four
installed CLI cases cover molecular and periodic examples with CPA and
Ehrenfest, including identical completed results after reopening. These
parameterized examples do not establish material-model accuracy.

Later reporting-only Markdown edits are separate from that tested archive;
runtime, tests and executable support remain unchanged. These receipts do not
qualify dev4 on Linux, accelerators or shared network storage. The historical
dev3 and dev2 evidence below applies only to its recorded sources and artifacts.

The `0.1.0.dev3` candidate adds coordinate-domain stage checks and conversion
provenance. Its own installed macOS CPU qualification is summarized here:

| Dependency profile | Selection | Result |
| --- | --- | --- |
| Current, Python 3.13 / JAX 0.11.2 | Complete suite | 2078 passed, 8 skipped, 4 expected failures; no warnings |
| Minimum, Python 3.11 / JAX 0.6.2 | Complete suite | 2078 passed, 8 skipped, 4 expected failures; 351 warnings retained |
| Optional Torch, Python 3.12 | Selected adapter and shared-integration checks | 120 passed; no warnings |

The eight complete-suite skips require optional Torch. The four expected
failures preserve documented differences between archival golden data and its
pinned original engine. The minimum stack emits JAX casting and NumPy
matrix-product warnings. A bounded paired investigation reproduced twelve
perovskite matrix-product warnings on dev2 and dev3 with finite, byte-identical
captured operands and completed results. It does not establish the backend
cause or explain every warning in the full suite.

These runs used fresh package environments with existing numerical dependencies,
not fresh dependency resolutions. They tested the wheel SHA256
`90573641bd6bb9301038e48a18f32f3708326ce6fce7db573509c246ce20965c`
and source-archive SHA256
`a1e1a6925c24b36fe4048a8a9970b5e4040e0c543f191ea134f0d90ef2625cb7`.
All 143 runtime files remained unchanged. The source archive contains 389
explicitly inventoried files. The guard/replay and molecular/periodic local-label
examples also passed against that installed wheel on both CPU dependency stacks.
The local-label candidates are illustrative and unfitted.

Later qualification-report edits are separate from that tested source archive;
runtime, tests and executable support sources remain unchanged.

The published dev3 commit `ac8a98df30ae16dda193556bd205b4a57dc1f676`
subsequently passed its [Linux CI matrix](https://github.com/jiangtong1000/PyEPH/actions/runs/37252893103).
Each of the three installed CPU jobs passed 2078 tests, with eight optional-Torch
skips, four documented archival expected failures and no deselected tests.
The Python 3.11 minimum stack retained 57 JAX complex-cast warnings; the Python
3.12 and 3.13 current stacks reported none. Downloaded artifacts and qualification
receipts match the commit's 389 source files and 143 runtime files. The separate
optional-Torch CI job also succeeded. These dev3 results do not qualify changed
dev4 sources or accelerators. The historical dev2 platform results below apply
to their own artifacts.

Independent numerical checks, installed-package checks and material validation
answer different questions. The results below identify the snapshot actually
tested; they do not silently qualify later source changes.

The corrected dev2 runtime has passed complete installed suites on macOS CPU,
the minimum/current Linux matrix, and an A100 GPU. Each of these dev2 suites
passed 1997 tests, with eight optional-backend skips and four documented
archival expected failures. The minimum Linux stack retained 51 JAX warnings;
the current-stack runs reported none. The exact artifacts and qualifications
are identified below. Later example, test and reporting changes have their own
source qualification; unchanged runtime payloads alone do not qualify them.

## Installed CPU evidence, 4 October 2026

The following historical dev1 suites ran outside the checkout against one audited
wheel and source archive. Runtime files, distribution ownership and metadata
matched before execution, and runtime hashes remained unchanged afterward.

| Platform | Python / numerical stack | Passed | Skipped | Expected failures | Warnings |
| --- | --- | ---: | ---: | ---: | ---: |
| macOS 26.6.2 arm64 CPU | 3.12.8 / current, with optional Torch | 2047 | 0 | 4 | 0 |
| macOS 26.6.2 arm64 CPU | 3.11.17 / minimum | 1963 | 8 | 4 | 345 |
| macOS 26.6.2 arm64 CPU | 3.13.16 / current | 1963 | 8 | 4 | 0 |
| Linux x86_64, glibc 2.34 CPU | 3.12.10 / current | 1963 | 8 | 4 | 0 |

The current numerical stack is JAX/JAXlib 0.11.2, NumPy 2.5.3, SciPy 1.18.1,
h5py 3.16.0 and pytest 9.1.1. The first row additionally includes Torch 2.14.1.
The minimum stack is pinned in [requirements-minimum.txt](../requirements-minimum.txt).
The eight skips are missing optional Torch imports; parametrization makes the
test-count difference larger than eight. The four expected failures retain
historical transport golden files that already disagree with the pinned
original engine; the independent saved-sample engine comparisons pass. See
[fixture provenance](../tests/compatibility/data/greenkubo/README.md).

The minimum stack reports matrix-product warnings on this Apple
Silicon/Accelerate combination and JAX complex-to-real casting warnings during
real-parameter differentiation. Finite numerical reference checks pass. A
NumPy-only finite-matrix reproducer isolates the former warning family from
the dynamics implementation; the current NumPy stack does not reproduce it.
Warnings are retained, not suppressed. See [installation scope](INSTALLATION.md).

Python 3.13 used a fresh native interpreter and freshly resolved/downloaded
dependencies. The other two macOS runs used fresh package environments with
existing numerical dependency installations; they are not fresh-resolution
evidence. Linux used a fresh environment and dependency resolution with cached
wheel payloads.

These rows refer to wheel SHA256
`67f20811d88204630b43a5f669db54b98046c3e71baaa9a0db779bd6f94ba41d`
and source-archive SHA256
`b49b6fcc775a5d3567c9bca762655c3d7105b1596fe4d387ded453d1f4e291d8`.
Subsequent candidate changes are a separately checked whitespace cleanup,
portability corrections in tests/example diagnostics, documentation, and the
generated-model calibration benchmark. The numerical runtime equations did
not change in that delta. That later dev1 candidate passed a complete native
Python 3.13 CPU suite with 1972 passed, eight optional Torch skips, four archival
expected failures and no warnings. It reused the earlier freshly resolved
Python 3.13 dependencies. Its wheel SHA256 is
`63e60def06881395ebb308bf7ddffff1277dbde1e6a61540818f50edd3683ac0`
and source-archive SHA256 is
`b13a1a3cac8707a30b6e04da57e42e4ef6e9f23c6f98ea928d925176d87e393a`.

The first published dev1 commit,
`255f6a501b8240de866810ee50c52a65ae285d69`, subsequently passed its
[complete Linux CI matrix](https://github.com/jiangtong1000/PyEPH/actions/runs/37236870458).
Each installed-wheel job passed 1972 tests, with eight optional Torch skips
and four archival expected failures. Python 3.11.16 with the minimum stack
retained 51 JAX complex-to-real casting warnings; Python 3.12.14 and 3.13.15
with the current stack reported none. The separate optional-Torch job passed
its selected adapter/composition tests. Downloaded wheel/source archives and
qualification records agree on the 141 runtime-file hashes; the runtime
fingerprint is
`a70cd826ef81c3fb0e5c68f4edb4d2e781ed88fcdd2afe6fb061ed6f47f26e7b`.
Archive bytes differ between CI builds, so their individual hashes remain in
each retained qualification record.

Those passing suites preceded a targeted geometry audit that found host
neighbor-coverage errors at floating-point boundaries. The dev2 correction uses
exact float-input geometry for candidate and skin decisions, checks completeness
after rewrapping, and rejects excessive image searches before allocation. It
also fixes serialization of valid byte-string geometry IDs in a fitting report.
The audited dev2 wheel passed a complete installed Python 3.13.16 suite on
macOS arm64: 1997 passed, eight optional Torch skips, four archival expected
failures and no warnings. A new package environment reused the earlier resolved
numerical dependencies; this was not another fresh dependency resolution.
Its wheel SHA256 is
`deafbbc8944faf4e865bd06f93aaab54793ed8fa6a27ac4a88a51f186cb5a6bf`
and source-archive SHA256 is
`f9e644a41be223431025fb8b97c9f921df8a8964ef1753066aa45511ed3a8513`.
The corresponding published commit,
`a42198759cb91e4402d7c4f8881a43676bf08e39`, also passed its
[Linux matrix](https://github.com/jiangtong1000/PyEPH/actions/runs/37239687825).
All three installed-wheel jobs passed 1997 tests, with eight optional Torch
skips and four archival expected failures. The minimum Python 3.11 stack
retained 51 JAX casting warnings; current Python 3.12/3.13 reported none.
The separate selected optional-Torch checks passed. Each retained CI archive
matches that commit's 378-file inventory and the corrected 141-file runtime
fingerprint
`c43a67cf58cae3d86dd53c7fc50fd03d2796673927c1e3e14b826e5e6a3403fb`.
The full accelerator suite has also passed as described below. These results
do not establish automatic stage guarding or silently qualify later
support-source changes.

The subsequent support-source commit
`8529690e7be4e71c266a85bd1ce46b0855a10244` added the full-gradient window
benchmark and lifecycle follow-ups. Its [Linux CI matrix](https://github.com/jiangtong1000/PyEPH/actions/runs/37242207177)
passed 2003 tests in each installed-wheel job, with eight optional Torch skips
and four archival expected failures. The minimum stack retained 53 warnings;
current Python 3.12/3.13 reported none. The selected optional-Torch job passed.
The downloaded archives match that commit's 380-file inventory and the same
corrected runtime fingerprint. Later qualification-report and local-label
example changes require their own source checks.

The [CI workflow](../.github/workflows/tests.yml) builds an explicit source
export and tests its installed wheel on Linux with minimum/current stacks.
A workflow definition is not evidence of a successful run. Local run archives
are retained separately from the source release. To regenerate a source-bound
report, follow [the distribution procedure](RELEASE.md); the harness writes
the exact identities, dependency versions, exit code and full test log.

## Accelerator status

Small molecular and periodic CPA/Ehrenfest workflows have completed matched
CPU/A100 numerical gates, including independently referenced trajectories,
timestep refinement, batching, sampled output and checkpoint continuation.
These runs cover illustrative effective Hamiltonians, not validated materials
or general large-system speedup.

The first full A100 suite failed. Its records exposed a CUDA-only launch that
excluded the CPU backend needed by host diagnostics, platform-specific fixture
startup, overly exact comparisons of recomputed floating-point values, and
independent-oracle resolution. Corrections preserve exact stored state and
discrete fields while comparing propagated floating values with declared
numerical tolerances. The later dev1 candidate identified above then passed its
full installed A100 suite: 1972 tests passed, eight optional Torch imports were
skipped, four archival tests were expected failures, and no warnings were
reported. The launch verified a default A100 GPU backend, an available CPU
backend for host diagnostics, and float64 mode; installed runtime hashes stayed
unchanged. It used the same `63e60def...` wheel and `b13a1a3c...` source archive
identified above, with Python 3.12.10 and the current numerical stack.

That dev1 GPU result predates the host-geometry correction. The corrected dev2
wheel `deafbbc8...` and source archive `f9e644a4...`, identified in full above,
subsequently passed their own installed A100 suite: 1997 passed, eight optional
Torch skips, four archival expected failures, and no warnings. The default A100
GPU, CPU callback backend and float64 settings were verified, and all 141
runtime hashes remained unchanged. The full pytest run took 3023.40 seconds;
this is qualification duration, not a dynamics throughput benchmark.
The same corrected dev2 artifacts also passed 12 matched CPU/A100 workflow
pairs across molecular and periodic illustrative models with CPA and
Ehrenfest. Independent references, timestep refinement and numerical checkpoint
continuation passed. At batch size 64 the molecular cases were faster on CPU,
whereas the periodic cases were faster on this A100 allocation. GPU repetition
showed small floating-point variation without discrete-field changes. These
bounded results establish neither general GPU speedup nor material accuracy;
use the [workflow benchmark](../benchmarks/platform_qualification.py) to measure
the intended model and workload.

## Acceptance gates and remaining scope

| Area | Required evidence |
| --- | --- |
| Core methods | Independent equations, conservation/convergence and estimator tests |
| Providers | Complete derivative, unit, Hermiticity and basis-contract checks |
| Campaigns | Real concurrent claims, crash recovery, identity and deterministic merges |
| Neighbor graphs | Independent periodic enumeration, motion coverage and cutoff forces |
| Learned models | Withheld-family errors, strict bundle identity and domain diagnostics |
| Distribution | Explicit inventory, history/archive audit and outside-checkout wheel suite |
| Platforms | Separate CPU, dependency-stack and accelerator qualification |

The current methods have intentionally bounded physical scope. Native geometry
dynamics uses fixed orthonormal effective electronic states and a declared
reference-plus-carrier total energy. Real finite-state multistate mapping
dynamics requires a complete isolated spectrum; complex/SOC hopping and
degenerate-subspace dynamics remain research work. General material mobility
requires qualified preparation, current operators, statistics and model labels.

Candidate graphs are fixed during compiled propagation. Coverage diagnostics
and rebuilding have explicit host-side contracts. A declared descriptor domain
is a geometric diagnostic, not calibrated uncertainty. Torch callbacks and
independent-worker execution have separate hardware restrictions.

CPU correctness does not establish GPU throughput, multi-node scaling, quantum
accuracy or transferability to a new material. Published evidence must retain
those distinctions.
