# A native replacement for the original Holstein example

[`examples/holstein_peierls.py`](../examples/holstein_peierls.py) takes physical
model inputs through native CPA propagation, per-trajectory current correlations,
streamed output, checkpoint/restart, and finite-window transport analysis.
It uses `EdgeEPCModel`, `HarmonicBath`, `Problem`, `CPA`, and `Simulation` through
the native transport workflow builders. No `pyeph.greenkubo` compatibility class
is imported. Its small application-owned model adds the ring velocity and units;
there is no second propagation algorithm.

This is the same one-dimensional, one-orbital **periodic model family** used in
the original PyEPH `examples/02_holstein/run.py`, with independent bond Peierls
oscillators and an optional local Holstein mode on each site. It is not a fitted
disordered molecular aggregate or perovskite. The demonstrated result is a
finite-time correlation and its running integral, not a validated material mobility.

## Run and inspect

From the repository root, using an environment with PyEPH's base dependencies:

```bash
JAX_ENABLE_X64=true PYTHONPATH=src python examples/holstein_peierls.py \
  --output results/holstein_cpa
```

The compact defaults are 8 sites, 8 trajectories, 200 steps, and `dt=0.02`, with
the original physical values: hopping 100 meV, Holstein frequency 50 meV,
reorganization energy 100 meV, Peierls frequency 6.2 meV, and temperature
183.84761032548917 K. Peierls disorder is disabled by default, as in the old
script. For a coupled Holstein--Peierls example, add `--peierls-fraction 0.25`.
Use a fresh output directory: existing results are never silently overwritten.

```bash
# Exact original workload size and endpoint convention: 5000 samples, 4999 steps.
JAX_ENABLE_X64=true PYTHONPATH=src python examples/holstein_peierls.py \
  --original --output results/holstein_original_size

# Integrated-out local quantum Holstein mode plus classical bond modes.
JAX_ENABLE_X64=true PYTHONPATH=src python examples/holstein_peierls.py \
  --mode lf --peierls-fraction 0.25 --output results/holstein_lf

# Only the narrowed band/current approximation; omit the LF bath-dressed estimator.
JAX_ENABLE_X64=true PYTHONPATH=src python examples/holstein_peierls.py \
  --mode band --peierls-fraction 0.25 --output results/holstein_band
```

Each output contains:

| File | Contents |
|---|---|
| `initial_samples.npz` | Actual canonical `q0,p0`, oscillator frequencies, nuclear temperature, and stable trajectory IDs. |
| `trajectory.h5` | Every observed time, each trajectory's complex unsymmetrized velocity correlation, and its unitary error. Data is appended in chunks. |
| `checkpoint.h5` | Full trajectory state, original thermal density/current insertions, and strict scientific/source provenance. |
| `analysis.npz` | Mean complex correlation, separate real/imaginary SEMs, running real integral and its SEM, reduced time and femtoseconds. |
| `run.json` | All inputs, physical conventions, source/configuration manifest, data hashes, segment history, and selected final diagnostics. |

The integration kernel retains a full electronic evolution matrix for each
trajectory. This is an exact thermal-trace reference workflow for modest state
counts. It is distinct from the [column transport](COLUMN_TRANSPORT.md) route,
which requires separately controlled electronic trace sampling.

## Hamiltonian and coordinate conventions

Take (E_0=J>0), one lattice spacing as the length unit, and

\[
\widetilde\omega_H=\omega_H/E_0,\quad
\widetilde g_H=\sqrt{\lambda_H\omega_H}/E_0,\quad
\widetilde T=k_BT/E_0,\quad t_0=\hbar/E_0.
\]

Here \(\lambda_H\) is the supplied reorganization energy. Tildes are omitted
below. Canonical unit-mass coordinates obey

\[
Q_a(t)=Q_a(0)\cos\omega_at+
       P_a(0)\sin(\omega_at)/\omega_a,
\quad X_a=\sqrt{2\omega_a}\,Q_a.
\]

In CPA, both branches are classical and the electronic matrix is

\[
H_{ii}=g_HX_{H,i},\qquad
H_{i,i+1}=H_{i+1,i}=-1+g_PX_{P,i},
\]

with site indices modulo (N). Nuclear arrays are mode-major, then site; the
two branches have independent canonical thermal samples with
\(\langle Q_a^2\rangle=T/\omega_a^2\) and
\(\langle P_a^2\rangle=T\). The electronic density is prepared separately on
each sampled geometry as \(e^{-\beta H}/\operatorname{Tr}e^{-\beta H}\).
The nuclear distribution is not reweighted by the carrier partition function.
Agreement with old CPA does not establish that this approximate joint ensemble
is stationary under the chosen dynamics.

For direct takeover, the supplied Peierls fraction (dJ) retains the historical
conversion

\[
g_P=dJ\sqrt{\tanh[\omega_P/(2T)]}.
\]

That conversion calibrates a Wigner oscillator variance, whereas the original
1D workflow uses classical Boltzmann sampling. Consequently the actual
classical hopping RMS is

\[
\sigma_J=dJ\sqrt{\frac{2T}{\omega_P}
                         \tanh\frac{\omega_P}{2T}},
\]

in units of (E_0). It equals `dJ` only in the classical low-frequency limit.
Both the coupling and actual RMS are recorded in `run.json`. No parameter is
silently relabeled as an exact classical RMS.

The velocity is the derivative of a **uniform bond Peierls phase**. A directed
nearest-neighbor bond has displacement +1, including the bond from (N-1) to
0; its reverse has -1. Thus (v_{ij}=i d_{ij}H_{ij}). This avoids a spurious
length (N-1) for the boundary bond from a naive finite-position commutator.
At least three sites are required; a two-site periodic multigraph needs a
different edge representation.

## LF and band-only choices

| Mode | Holstein branch | Electronic propagation | Velocity correlation |
|---|---|---|---|
| `cpa` | Explicit independent classical oscillators | Bare (H(Q)) | Bare velocity correlation |
| `lf` | Integrated-out identical independent local quantum baths | Offdiagonal elements multiplied by (f=e^{-\phi(0)}) | Bare velocities plus the matching LF bath factor |
| `band` | Same narrowing factor | Same narrowed Hamiltonian | Narrowed velocities; no fluctuating LF bath factor |

The native local LF implementation specifies

\[
\phi(t)=(g_H/\omega_H)^2
 [\coth(\beta\omega_H/2)\cos\omega_Ht-i\sin\omega_Ht].
\]

The Peierls branch stays explicit and classical in all three modes. The
example selects the branches by their physical role; it does not infer that a
frequency cutoff justifies a quantum/classical approximation. The old default
cutoffs, (2\omega_H) for CPA and \(\omega_H/2\) for LF, give this same split
for the supplied frequencies. The original 1D helper accepts a `model_type`
argument but constructs the bond model regardless; this example exposes only
the bond model it actually implements.

`--thermal-policy offdiagonal` prepares from the Hamiltonian used for
propagation. `--thermal-policy legacy_full` reproduces the old preparation that
scales the entire bare matrix by (f), while still propagating with narrowed
offdiagonals. These are explicit distinct conventions when retained diagonal
disorder is present. In this single-Holstein-branch LF example there is no
remaining onsite disorder, so they coincide. No constant LF relaxation-energy
shift is inserted; this matches the original convention and such a common
scalar shift does not change this correlation.

## Restart and matched samples

```bash
JAX_ENABLE_X64=true PYTHONPATH=src python examples/holstein_peierls.py \
  --steps 100 --output results/part1

JAX_ENABLE_X64=true PYTHONPATH=src python examples/holstein_peierls.py \
  --resume results/part1 --steps 100 --chunk-size 32 --output results/part2
```

The second command inherits the first run's physical inputs, IDs, seed, and
integrator. Only additional `steps` and `chunk_size` can change. Its analysis
joins both streams and checks and removes the one duplicated boundary sample.
The correlation origin stays at the original preparation, rather than being
reinitialized at the restart. Checkpoint restore is exact; changing chunk and
restart boundaries can change subsequent floating-point clock expressions by
an ULP, so continuation comparisons use a stated numerical tolerance.

All previous streams and the sample archive are content-hashed and checked
before resuming. Keep the referenced segment files at their recorded paths.
This example uses absolute local paths and is not a portable archive format.
Strict checkpoint loading also checks source identity: editing the example or
runtime package requires an explicit migration, not silent resume. Resume with
the same entry point, because Python model-class identities form part of the
manifest. `run.json` is written only after a completed segment; an interrupted
new segment leaves the last completed directory usable as its restart point.

For a timestep comparison, replay actual samples rather than assuming equal
seeds imply identical data across codes or versions:

```bash
JAX_ENABLE_X64=true PYTHONPATH=src python examples/holstein_peierls.py \
  --dt 0.01 --steps 400 --samples results/holstein_cpa/initial_samples.npz \
  --output results/holstein_half_dt
```

The default trajectory IDs are 0 through 7. `--first-id` and `--trajectories`
select another stable ID interval. Sampling is independent of how that interval
is partitioned. A saved archive must have the requested IDs, mode frequencies,
and nuclear temperature. Supplying samples deliberately fixes the nuclear
draws; it does not change the original seed stored in the electronic state key.

## What the transport analysis establishes

The saved complex quantity is

\[
C_r(t)=\operatorname{Tr}[v_r(t)U_r(t)v_r(0)\rho_r(0)U_r^\dagger(t)].
\]

The analysis forms a trapezoidal running integral **for each independent
nuclear trajectory**,

\[
I_r(t)=\int_0^t \operatorname{Re}C_r(s)\,ds,
\]

and then reports the mean and sample standard deviation divided by
\(\sqrt{N_\mathrm{traj}}\). It does not add time-point error bars in quadrature:
those samples are correlated. Correlation real and imaginary parts have
separate SEMs. One trajectory has undefined sampling SEM, stored as NaN in
arrays and `null` in the JSON scalar summary. This differs from the old output's
spread across rank averages, which is zero for a single rank.

`mean_beta_integral` is \(\beta\overline I(t)\). A DC mobility interpretation
requires a justified equilibrium/linear-response estimator, a stationary
ensemble, a converged time window, and finite-size checks. The clean-ring test
deliberately has constant correlation and a linearly growing integral: the
workflow does not misidentify ballistic propagation as a diffusion plateau.

No physical length is assigned by default. With an explicitly supplied
`--spacing-angstrom a`, the optional `finite_window_mobility_proxy_cm2_per_Vs`
multiplies \(\beta\overline I(t)\) by
\(e a^2/\hbar\), including the metre-to-centimetre conversion. This is a named
finite-window proxy, not a claim that the required DC limit exists. UnitSystem
uses a nominal 1 Angstrom length solely as internal metadata when the spacing
is unknown; `physical_length_known=false` prevents a physical length conversion
in the output.

Before interpreting transport, compare at least:

1. The same saved samples at `dt`, `dt/2`, and `dt/4`, at matched physical times.
2. Longer correlation windows; inspect the remaining tail and running integral.
3. Larger rings, which can alter recurrences and localization lengths.
4. More independent trajectory IDs; sample SEM does not include steps 1--3 or
   model/approximation error and need not shrink monotonically in a small sample.

## Reproducible validation

`tests/test_holstein_peierls_example.py` covers explicit initialization, output
conventions and split/restarted calculations. The maintained compatibility
fixtures retain original inputs, numerical results and provenance under
`tests/compatibility/data`. New runs must keep preparation, propagation and
transport-analysis convergence distinct. See [qualification](QUALIFICATION.md)
for platform and release evidence.
