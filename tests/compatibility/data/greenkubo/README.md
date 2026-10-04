# Green–Kubo takeover references

The original source is `https://github.com/jiangtong1000/PyEPH`, pinned to
revision `6c4693acbb69a06a5bc8b0593abde2170ff38843` (BSD-3-Clause; see root
LICENSE). `provenance.json` records source-file hashes, original dependency
versions, source revision, reference-file hash, unchanged archival-file hashes
and the original engine's own comparison against those archival files.

`reference.h5` was produced by the original source in an isolated environment:
Python 3.12, NumPy 2.3.5, SciPy 1.18.1, h5py 3.16.0, numba 0.68.0 and
llvmlite 0.50.0. The new package has no numba dependency. The executable
`generate_reference.py` records the exact case configurations and refuses to
overwrite the reference file. It uses the historical constructor/preparation
and historical propagator/estimator, not the new implementation.

Each case saves the actual `q0,p0` sampled by the original source, initial
Hamiltonian/density for the first trajectory, every trajectory's complex
current correlations and the final first-trajectory propagator. Nonlocal
cases additionally save every sampled real-space field. Native tests feed
these arrays directly into the facade; matching integer seeds alone is not
used as a reproducibility argument. The whole reference HDF5 file hash
therefore also pins the sampled array content.

The 14 cases are the nine maintained integration configurations, one
band-narrow-only configuration, and four dispersive half-grid cases
(Boltzmann/Wigner, gauge on/off). All use the original RK4 policy and
whole-H LF thermal preparation. The four 2D ordinary CPA cases have 324
electronic states, so this is more than a two-site identity test.

All nine `expected_*.h5` files are copied unchanged from the original checkout.
The original engine reproduces all five ordinary CPA archival fixtures to
about 1e-15, but **does not reproduce the four archival LF fixtures**:
maximum errors range from 0.0118 to 1.2785 across current components. The
original LF integration tests default to bypassing these comparisons. Those
four historical-file comparisons are explicit `xfail` cases; they do not
relax or skip native-versus-original-source trajectory comparisons. No new
reference replaces or silently repairs those archival values.

`compare_reference.py` runs the native package against `reference.h5` and
writes `native_comparison.json`, an acceptance report rather than a golden
file. It records tolerances, dtype, dependencies and absolute/relative errors.
No CPU/GPU speed claim follows from these numerical comparisons.
