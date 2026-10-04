# Native thermal and LF-CPA transport

The transport recipes use the same `Problem`, `CPA`, `TrajectoryState` and
`Simulation` as other native dynamics. They retain the exact small-system
density-matrix/full-propagator route from PyEPH while separating thermal
preparation, physical current probes, electronic propagation and the estimator.
They do not require the previous PyEPH checkout at runtime.

The ordinary recipe evaluates

\[
 C_\alpha(t)=\operatorname{Tr}\left[J_\alpha(t)U(t,t_0)
 J_\alpha(t_0)\rho_0 U^\dagger(t,t_0)\right],\qquad
 \rho_0=\frac{e^{-\beta H(t_0)}}{\operatorname{Tr}e^{-\beta H(t_0)}}.
\]

`beta` is inverse energy, not inverse Kelvin. Native kernels use `hbar=1` in
one consistent unit system; current matrices must include the user's charge,
length and energy conventions. Returned correlations are complex and
unsymmetrized. They are not automatically a mobility or a response estimator
for Ehrenfest or MASH trajectories.

## Choosing a transport route

| Scientific calculation | Public workflow | Main scaling or convergence consideration |
| --- | --- | --- |
| Ordinary CPA with the complete thermal density | `workflows.transport` | Dense preparation and a full propagator; useful as a small-system reference and direct migration route |
| Ordinary CPA with an explicit density factor | [Column transport](COLUMN_TRANSPORT.md) | Propagates `K*(1+nprobes)` columns through model actions; an exact full-rank factor is still quadratic in size |
| Ordinary CPA with sampled finite-temperature columns | [Thermal preparation](THERMAL_COLUMNS.md), then column transport | Bound filter error, then converge trace rank/seeds, nuclear samples and timestep separately |
| Local LF reduced-bath transport | `workflows.polaron_transport`, with the [compact estimator](COMPACT_LF.md) by default | Retains the full propagator; compact contraction preserves the selected LF physics and full current-edge support |
| Real multistate RM transport | [RM transport](RM_TRANSPORT.md) | Separate joint canonical preparation, dynamics and velocity estimator; the CPA trace formula does not apply |

The first three rows differ in electronic preparation or representation within
prescribed-path CPA. LF and RM introduce different physical approximations and
estimators. Choose them from the intended model and observable, rather than
from a runtime comparison alone. All routes require physical current operators
and explicit unit conventions.

## Ordinary CPA example

```python
import numpy as np
import jax.numpy as jnp

from pyeph import configure_precision
from pyeph.core.state import stack_states
from pyeph.execution.runner import Execution
from pyeph.integrators.electronic import Integrator
from pyeph.models.epc import LinearEPCModel
from pyeph.paths.harmonic import HarmonicBath
from pyeph.simulation import Simulation
from pyeph.workflows.transport import (
    make_transport_problem, initialize_transport_state,
)

configure_precision()
model = LinearEPCModel(nstates=2, nmodes=1)
params = model.create_params(
    h0=np.array([[0.2, -0.3], [-0.3, -0.1]]),
    coupling=np.array([[[0.2, 0.0], [0.0, -0.2]]]),
    omega=np.array([0.8]),
)

# Fixed-centre electronic hopping current, charge=hbar=1. This example
# supplies a callback because LinearEPCModel does not infer physical probes.
centres = jnp.array([0.0, 1.0])
displacement = centres[None, :] - centres[:, None]
def current_probe(params, context, probe):
    return 1j * displacement * model.dense(params, context.q)

problem = make_transport_problem(
    model, params, HarmonicBath([0.8]),
    probes=("current_x",), probe_callback=current_probe,
)
initial = stack_states([
    initialize_transport_state(problem, [q], [p], beta=2.0,
                               trajectory_id=i, seed=1120)
    for i, (q, p) in enumerate([(0.2, 0.1), (-0.3, 0.2)])
])
simulation = Simulation(problem, Integrator(dt=0.01),
                        Execution(chunk_size=128))
result = simulation.run(initial, steps=100)
correlations = result.observables["current_correlation"]
# shape: (101 times, 2 trajectories, 1 probe)
mean_correlation = correlations.mean(axis=1)
```

A model with declared physical probes can instead implement
`probe_apply(params, ProbeContext(q, velocity, time), name, identity)`.
`probe_callback(params, context, name)` is an explicit alternative and must be
JAX compatible. It returns a full physical Hermitian current matrix.
Initial currents are checked for shape, finiteness and Hermiticity.

`HarmonicBath` advances independent canonical initial conditions from each
state. `PrescribedPath` uses the path stored in the problem; initial coordinates
must match that path at the initial time. Probe velocities come from canonical
`p/m` for a harmonic bath or the prescribed path's `velocity(time)`.
A custom prescribed treatment can expose `velocity(state)`; otherwise velocity
is `None` and a velocity-dependent probe must reject that missing information.

The output key `current_correlation` has a final axis in requested probe order.
`unitary_error` records `max(abs(U.conj().T @ U - I))`. These are per-trajectory
observations. Neither the measurement nor the runner silently averages initial
density matrices or current insertions across trajectories.

## LF-CPA and the thermal-initialization decision

```python
from pyeph.workflows.polaron_transport import (
    make_polaron_transport_problem, initialize_polaron_transport_state,
)

lf_problem = make_polaron_transport_problem(
    model, params, HarmonicBath([0.8]),
    frequencies=[2.0], couplings=[0.4], beta=2.0,
    hopping_pairs=[(0, 1), (1, 0)],
    thermal_policy="offdiagonal",
    probes=("current_x",), probe_callback=current_probe,
)
lf_initial = initialize_polaron_transport_state(
    lf_problem, [0.2], [0.1], trajectory_id=0,
)
lf_result = Simulation(lf_problem, Integrator(dt=0.01)).run(lf_initial, 100)
```

This reproduces the declared local, independent, identical-site harmonic
quantum-bath approximation. Frequencies and couplings have energy units;
frequency arrays must be positive. The problem has one quantum-bath temperature
and enforces the same temperature for thermal electronic preparation.
An empty quantum bath reduces to ordinary CPA.

The propagation Hamiltonian has off-diagonal elements scaled by
`exp(-phi(0))`; diagonal site energies are preserved. The full LF correlation
also contains

\[
\begin{aligned}
\phi(t)&=\sum_\nu |g_\nu/\omega_\nu|^2
\left[\coth(\beta\omega_\nu/2)\cos(\omega_\nu t)
-i\sin(\omega_\nu t)\right],\\
F_{ijkl}(t)&=\exp\{[-2+\delta_{ij}+\delta_{kl}]\phi(0)
-[\delta_{ik}-\delta_{jk}-\delta_{il}+\delta_{jl}]\phi(t)\}.
\end{aligned}
\]

The current probe still returns the **bare** physical current. The workflow
applies the LF bath factor in the estimator; do not additionally narrow that
current or multiply the result by another band-narrowing factor. The model
wrapper is explicitly force-incompatible. This workflow does not derive
nonlocal Fröhlich dressing, site-dependent quantum baths, or an LF-Ehrenfest
response theory.

`hopping_pairs` contains unique **directed** current edges. Include both
directions of a Hermitian hopping and the union of all allowed edges over all
coordinates and probes. Preparation rejects missing nonzero initial edges;
the caller remains responsible for a valid topology at later times. The default
`estimator="compact"` contracts the sector-zero terms as an ordinary trace and
stores only shared-endpoint corrections. Topology storage is linear for a
bounded-degree graph; the full propagator and dense density contraction remain.
`estimator="full"` retains the reference Cartesian square of edges, with quadratic
edge-count construction and storage. The explicit quadruple evaluator still
supports arbitrary subsets; the compact estimator represents the complete
square of its declared support. Both preserve diagonal current edges.

Choose thermal preparation deliberately:

| Policy | Initial density Hamiltonian | Propagation Hamiltonian |
| --- | --- | --- |
| `offdiagonal` (default) | Off-diagonal elements narrowed, diagonal retained | Same |
| `legacy_full` | Entire bare Hamiltonian multiplied by the narrowing factor | Off-diagonal elements narrowed, diagonal retained |

The older PyEPH `simulation.py` used whole-matrix scaling for initialization,
while `propagator.py` narrowed only off-diagonal elements during evolution.
These conventions differ for nonconstant diagonal energies. `legacy_full`
is available for reproducibility; `offdiagonal` is the internally consistent
choice for the stated effective Hamiltonian. The tests explicitly expose the
difference rather than hiding it inside a migration tolerance.

## Legacy conventions and checkpoints

Historical `build_jx_jy` matrices omit the imaginary factor. Pass
`current_convention="legacy_without_i"` when importing those matrices.
The workflow converts each imported matrix once; do not also apply the legacy
minus sign to its returned correlation. Native physical-current input uses
`current_convention="physical"`.

Legacy oscillator fields satisfy `X=sqrt(2*w)*Q` and
`Y=sqrt(2/w)*P` for unit-mass canonical `Q,P`.
`paths.harmonic.legacy_quadratures_to_canonical` performs this conversion.
The EPC derivative must be converted as well: a legacy `g*X` term becomes
`g*sqrt(2*w)*Q`. Passing the legacy fields directly to the unit-mass canonical
bath changes both the dynamics and sampling distribution.

Each state saves `rho0`, `currents0`, `time0` and `beta` under
`method_state["transport"]`. The electronic array stores `U(t,t0)`. Save the
complete state with `io.checkpoint.save_checkpoint`, reconstruct the identical
problem and resume the loaded state; do not prepare a new thermal state at
restart. The LF phase uses elapsed time since saved `time0`.
Checkpoint metadata should include model/parameter hashes, probe definition,
units, coordinate convention, bath parameters and `thermal_policy`. Python
model/callback code is not serialized.

## Validation status

`tests/test_transport_workflow.py` compares native results with independent
SciPy matrix exponentials, NumPy time-dependent RK4, and explicit LF
four-index contractions. It checks complex Hermitian inputs, per-trajectory
thermal preparation, independent harmonic paths, probe velocity context,
single/batched agreement, output chunking, streaming, HDF5 checkpoint/restart,
legacy current conversion and the two LF thermal policies.

Kernel tests separately cover low-temperature stability, beta zero/infinity,
batching/JIT, gauge transformations, finite-difference gradients, empty baths
and diagonal disorder. These tests establish the stated small-system equations.
They do not establish speed parity with legacy PyEPH. The separate takeover
audit below compares native dynamics with pinned original-source trajectories,
actual saved initial samples and the nine archived HDF5 regression fixtures.

## Native QE2PERT preprocessing migration

The twelve numerical modules under `post_qe2pert` have been migrated from
`jiangtong1000/PyEPH` revision
`6c4693acbb69a06a5bc8b0593abde2170ff38843`, retaining their BSD-3-Clause
attribution and public module paths. They run entirely from this package.
`PostQE2Pert`, `ElectronBands`, `PhononDispersion`, `CalcEphMatReciprocal` and
`parse_qpoint_path` retain their package-level imports; `CalcEphMatMixed`,
localization, symmetrization and unwrapping retain their submodule imports.

The maintained extraction example is also available as a native command:

```bash
USE_MPI=false python -m pyeph.post_qe2pert.extract \
  --epr-file tests/compatibility/data/preprocessing/DNTT_epr.h5 \
  --nx 2 --ny 2 --output outputs/dntt_eph.h5
```

It extracts real-space EPC arrays and phonon modes for later localization;
it refuses to overwrite an existing output. It does not run electronic
structure programs or perform QCPBC optimization. The native command requires
an explicit input path, since an installed package has no adjacent QE workflow
directory from which to infer one.

Preprocessing uses NumPy, SciPy and h5py on the host. The localization losses
use JAX; importing them no longer changes device selection, CPU count or
global JAX precision. Call `pyeph.configure_precision()` explicitly before
numerical use when double precision is required. MPI is optional and loaded
only when requested by an operation; `USE_MPI=false` guarantees serial mode.
QCPBC remains a separately licensed optional dependency and is imported only
when its optimizer is called. Neither mpi4py nor QCPBC is necessary for the
serial extraction and regression tests.

`tests/compatibility/test_preprocessing.py` verifies four electronic-band
points, four polar phonon points and one reciprocal-space k/q EPC pair against
the original DNTT reference data. It also checks complex Hermitian packing,
an independent mixed-space inverse Fourier formula, mass normalization,
skew-cell minimum images, localization density, optional-dependency import
behavior and the native extraction command's output/TRS/no-overwrite contract.
The test data directory contains the original 4.17 MB EPR input, reduced
reference arrays, and SHA256/source-revision provenance. External programs
were not rerun to regenerate the original references. MPI execution and
QCPBC optimization have not been validated in this migration.

One numerical setup defect was repaired: polar interpolation previously
referenced undefined `nrx`/`nrx_ph` arrays for q meshes with a singleton axis.
The arrays are now constructed before that axis is zeroed, and a focused test
covers the branch. Other material-specific assumptions are retained and
should not be interpreted as general three-dimensional material support:

- The polar correction explicitly rejects `system_2d=True`; the copied
  three-dimensional polar formulas have been checked only on this DNTT fixture.
- `CalcEphMatMixed.extract_gmat_raw` warns and then discards an imaginary part.
  Its output is for the original real EPC convention, not a general complex
  spin-orbit coupling model.
- `ifft_to_real_space` is an unnormalized Fourier sum and fills the electronic
  upper triangle. Downstream normalization and Hermitian completion must follow
  the stored-data convention.
- `eph_pht.symmetrize_electronic_eph` and `unwrap_epc.assign_atoms_to_mol` retain
  a two-orbital/two-molecule cell specialization. The latter's lattice-metric
  minimum-image operation is tested independently; this does not make the
  two-molecule grouping a general aggregate partitioner.
- The extraction command retains the original two-dimensional MP grid and
  conjugate-partner ordering. Its output remains in the preprocessing units:
  phonon frequencies in Ry and q points in fractional reciprocal coordinates.
  Converting this data to a canonical nuclear dynamics model is a separate step.

## Maintained-workflow compatibility checklist

This inventory uses the old repository README, the two maintained examples,
its CI workflow and the `greenkubo/tests` and `post_qe2pert/test` suites at the
revision above. It distinguishes implemented native equations from executing
an unchanged historical script.

| Old maintained workflow or entrypoint | Native status | Remaining compatibility validation |
| --- | --- | --- |
| EPR bands, polar phonons, reciprocal EPC | Baseline and optimized complete 511-point band/phonon grids and 100 EPC pairs pass saved references; optimized serial/MPI arrays are identical, including empty-work rank handling | Broader materials and MPI configurations remain separate |
| Step 7 DNTT real-space extraction | Native `post_qe2pert.extract` command, public `utils` wrappers, serial 2×2 tests and byte-identical serial/two-rank 2×3 files | Broader MPI configurations; external example/scheduler files remain separate |
| Skew-cell molecule unwrapping | Original API migrated; metric-MIC and intact-pair tests pass | Original full unwrapping suite, general many-molecule grouping |
| QCPBC EPC localization | Code migrated and optional import checked | Licensed optimizer execution, convergence and saved-output regression |
| QE relaxation/SCF, D3 Hessian, DFPT, NSCF, Wannier90 and QE2PERT stages | External workflow templates remain in the source repository | Template/tool migration and actual external-program execution are separate work |
| DFPT chunked irreps: make/collect/merge/audit tools | Package-owned host tools and five CLIs; 43 tests pass on Python 3.11 and 3.12, including the full file workflow; see [DFPT guide](DFPT_CHUNKS.md) | External QE/scheduler execution and version-specific binary record conventions |
| 1D CPA Holstein/Peierls transport | Old constructors/builders delegate to native CPA; original-source trajectory and `expected_1D_CPA` pass | Long production example timing and convergence are separate from compatibility |
| 2D optical/bond Peierls CPA, zigzag on/off | All four original 324-state cases and their archived expected files pass | GPU and MPI execution not tested |
| 2D Holstein+Peierls local LF-CPA | All four pinned original-source trajectories pass from identical samples; both thermal policies independently tested | Four old LF archives disagree with the original source itself; retained as explicit archival discrepancies |
| Band-narrow-only transport | Native off-diagonal narrowing and a narrowed current at each insertion; source trajectory and complex-current formula pass | This approximation remains distinct from the full LF bath-dressed estimator |
| Classical Boltzmann/Wigner bath sampling | Historical local sampling order retained; canonical mapping, zero-classical-mode and saved-sample replay tested | Native repartition-independent sampling is distinct from the historical rank RNG |
| Dispersive/nonlocal classical phonons and MP half grids | Four original-source cases pass: Boltzmann/Wigner × gauge on/off; field, U and C(t) parity | Original even-grid/x-major field convention retained; no claim of a new general phonon interpolation theory |
| Rank-wise HDF5 current dumps and `analysis.merge_outputs` | Serial layout and an actual two-rank full-LF run pass identical-sample partition, merge and restart checks | Broader MPI configurations and performance remain untested; rank-mean standard deviation is not trajectory uncertainty |

The nine old Green–Kubo expected HDF5 files are present and were copied
unchanged with hashes. Executing the pinned old source reproduces all five
ordinary CPA archives to about 1e-15 but disagrees with all four LF archives:
maximum absolute current errors range from 0.0118 to 1.2785 across components.
The old LF integration tests default to `test_mode=False`, which bypasses
those comparisons. The native suite marks only those archival comparisons
`xfail`, with the original-source discrepancy as the reason; comparisons with
newly archived pinned-source trajectories still run and pass for all cases.
No archival file was regenerated or adjusted to obtain a passing result.

## Green–Kubo compatibility implementation

The historical constructors now prepare topology, arrays and I/O views,
then execute the shared `Problem`/`CPA`/`Simulation` path. The old propagation
loop and Numba estimator are not runtime dependencies of this package:

| Historical class/function | Native responsibility and proposed adapter |
| --- | --- |
| `BravaisLattice2D` | Host topology builder: retain old site ordering and explicit Cartesian centres; produce fixed edge indices and image displacements |
| `ElectronPhononHamiltonian(tmat,gmat,lattice)` | Compile dictionaries into the structured `LatticeEPCModel`: sparse directed edges, sparse EPC terms and the union of static/EPC support; current displacements share those immutable edge indices |
| `ClassicPhononBath` | Host Boltzmann/Wigner initial sampling plus canonical coordinate conversion; per-trajectory `HarmonicBath` motion, without mutable global `qfield` |
| `ClassicalPhononNonlocal` | Independent real/imaginary half-grid canonical modes, native harmonic motion and an explicit conjugation/Fourier map into the original EPC field order |
| `QuantumPhononBath` | Frequency/coupling/beta inputs to the LF workflow; `band_narrow_only` explicitly narrows both current insertions instead of using full LF factors |
| `DensityMatrixUnitaryPropagator` | Integrator configuration plus `initialize_transport_state`; save full U and thermal/current payload in `TrajectoryState` |
| `GreenKuboSimulation` | Compatibility recipe that builds native `Problem`, initializer and `Simulation`, then an optional historical-format observer |
| `estimator.current_from_density_*` | Thin convention adapter to `observables.transport` kernels; the historical minus sign follows from converting both currents by `i` |
| `typical_model_helper.build_*` | Host convenience builders composing topology, EPC parameters, bath split and initialization; do not introduce an independent propagation loop |
| `MPIRandomContext` | Execution/seed adapter; keep historical NumPy rank streams only for fixture reproduction, and use stable trajectory identifiers for native repartitioning |
| `analysis.merge_outputs` / sparse HDF5 utilities | I/O compatibility reader/writer, independent of the propagation algorithm; preserve metadata and distinguish trajectory uncertainty from rank averages |
| `vpt.compute_f` / `vpt_opt.compute_f` | Original host optimizer migrated, retaining its real symmetric zero-diagonal restriction; separate from the fixed LF workflow and not claimed validated by the dynamics oracle |

The historical import surface
`pyeph.greenkubo.{lattice,hamiltonian,hamiltonian_2,phonon,propagator,simulation,
estimator,typical_model_helper,mp_qmesh,mpi_random,analysis,utils,vpt,vpt_opt}`
and `pyeph.utils.{grid,constants,logger,fake_mpi}` is now provided. The old
`pyeph.lib.setup_jax.configure_jax_backend` is not provided. Use explicit
`pyeph.configure_precision()` before creating arrays, or set `JAX_ENABLE_X64=1`
before launch; select devices through JAX configuration. Global import-time
device/precision configuration was not restored.

Additional source utilities outside those maintained tests are
`utils.cifio.cif_to_qe_blocks` (optional pymatgen; ordered structures only) and
`utils.analyze_qcpbc_opt.extract_loss_history_qcpbc` (text-log parser). Their
public paths are migrated; pymatgen is imported only inside CIF conversion,
whose actual external-library execution has not been validated.
`legacy.wannier_phonon` spread optimization and
`legacy.eph_mat_mixed_simple` are explicitly retained legacy implementations
in the source repository, not prerequisites of the native preprocessing.
The README's spectral-function, optical-conductivity and mobility claims do
not correspond to separately tested postprocessing entrypoints in the audited
maintained suites; current correlations alone should not be presented as a
validated implementation of all those derived observables.

## Takeover precision, replay and numerical evidence

Legacy transport entrypoints require explicit double precision to preserve
their original NumPy64 inputs. Imports leave JAX configuration unchanged;
constructing a compatibility simulation with x64 disabled raises an
actionable error before samples or Hamiltonian parameters are converted.
Set `JAX_ENABLE_X64=true` before starting Python, or call
`pyeph.configure_precision()` before constructing the simulation. Fresh-process
tests cover both branches, so the suite's global x64 setting cannot hide a
default-float32 regression. This restriction belongs to the legacy facade;
native models remain free to declare another deliberate precision policy.

The established constructors remain usable:

```python
from pyeph import configure_precision
from pyeph.greenkubo.typical_model_helper import build_1d_Holstein_Peierls_model
from pyeph.greenkubo.propagator import DensityMatrixUnitaryPropagator
from pyeph.greenkubo.simulation import GreenKuboSimulation

configure_precision()
ham, classical, quantum, lattice, temperature = build_1d_Holstein_Peierls_model(
    J=1.0, dJ=0.2, wH=[0.5], gH=[0.15], wP=0.2,
    temperature=0.3, nsites=6, cpa_cutoff=0.25,
)
propagator = DensityMatrixUnitaryPropagator(lattice.nsites, 3, 0.01, 0.1, temperature)
simulation = GreenKuboSimulation(lattice, ham, classical, quantum, propagator,
                                thermal_policy="legacy_full")
result = simulation.run("outputs/legacy_transport", dump_interval=4, collect=True)
```

The compatibility recipe defaults to `legacy_full`; the modern LF recipe
continues to default to `offdiagonal`. Historical `time_range=arange(0,T,dt)`
excludes T. The facade therefore runs `len(time_range)-1` native steps,
including the initial observation. It preserves the original final HDF5
dataset names and sample-count-times-dt metadata. A new output directory is
required, avoiding silent overwrites of existing scientific data.
The historical writer requires `Execution(save_every=1)` and rejects sparse
observation schedules; use the native workflow for those schedules. Rank files
record every actual observation time. Merged output records `initial_time` and
`final_time`; resumed segments also include an explicit `time` dataset, so a
segment beginning at nonzero time is never mislabeled as starting at zero.
Rank merging rejects inconsistent time grids.

Each run also saves `initial_samples_<rank>.h5` with the actual legacy X,Y
arrays, canonical Q,P, coordinate/order conventions, unit scales and thermal
policy. The archive also records globally unique trajectory IDs, offset by
`rank * trajectories_per_rank`. Historical draws still use NumPy rank streams;
these draws are not invariant under changing MPI partition sizes. The native
ID-based initializer provides that separate guarantee. Exact original X,Y
arrays are retained in checkpoint payloads and restored without reconstructing
them through inverse dynamics. Replay via
`GreenKuboSimulation.load_initial_samples(path)` and the
constructor's `initial_samples=(X,Y)` option. Matching integer seeds alone is
not used to assert equivalence. Explicit `BravaisLattice2D(unit_system=...)`
records physical scales. Without them, metadata labels the identity reduced
scales as **unknown physical scales**, rather than inferring eV, meV or Angstrom.

The source oracle was generated in a separate environment using the original
source and NumPy 2.3.5, SciPy 1.18.1, h5py 3.16.0, numba 0.68.0 and
llvmlite 0.50.0. Source/dependency versions, source-file hashes, original archive
hashes and saved-sample-containing HDF5 hashes are recorded in
`tests/compatibility/data/greenkubo/provenance.json`. The new execution uses
JAX RK4 with one electronic substep and the exact saved original samples.
Fourteen complete trajectories are compared: nine maintained integration
configurations, band-narrow-only and four nonlocal distribution/gauge cases.

| Quantity | Maximum absolute error across 14 cases | Maximum relative max-norm error | Acceptance rtol / atol |
| --- | --- | --- | --- |
| Initial H, first trajectory | 2.22e-16 | 1.53e-16 | 1e-10 / 1e-11 |
| Initial rho, first trajectory | 1.28e-15 | 9.22e-15 | 1e-9 / 1e-11 |
| C(t), every trajectory and both axes | 1.26e-14 | 4.28e-15 | 2e-8 / 1e-9 |
| Final U, first trajectory | 1.11e-16 | 1.11e-16 | 1e-9 / 2e-11 |

Relative max-norm error means max(abs(error))/max(abs(reference)). The report
also includes elementwise relative errors above a stated reference threshold;
near-zero density entries can have much larger relative errors while their
absolute errors remain tiny. Full per-case results and the native dependency
versions are in `native_comparison.json` beside the fixtures. These are
numerical parity measurements, not timing or performance claims.

Independent tests additionally cover complex Hermitian static/EPC terms,
off-diagonal narrowing with diagonal disorder, finite-difference derivatives,
EPC-only edges and hopping zero crossings, both LF thermal policies, zero
classical modes, a zero narrowing factor, safe current indexing, sparse HDF5
helpers, streamed legacy output, sample replay and checkpoint/restart.
The unchanged legacy static/current/estimator and MP-grid unit files also
passed 23 tests with NumPy seed 0 set externally for their unseeded random
inputs. Trial non-Hermitian EPC blocks may be inspected by host compatibility
objects, but are rejected before compiling native dynamics.

The optional `tests/mpi/greenkubo_partition.py` acceptance program was launched
with two real MPI ranks using mpi4py 4.1.2 and MPICH 5.0.2 on this CPU host.
Two trajectories per rank in a four-state full local LF-CPA case matched a
serial four-trajectory run using the identical saved samples, with maximum
absolute current error **2.22e-16** over four steps (`dt=0.01`). It also checked
rank-offset IDs, exact original sample restoration after checkpoint load,
resumed times `[0.02, 0.03, 0.04]`, merged means and the rank-mean standard
deviation. The runnable program writes its dependency versions and results to
`acceptance.json` under the chosen fresh output directory. This small runtime
acceptance does not establish MPI scaling or distributed preprocessing parity.

Remaining restrictions are explicit: the facade retains the old 2D lattice,
static hopping cutoff (default 1e-3), local identical-site LF bath split, and
the nonlocal bath's even MP mesh and x-major real-space field order. It removes
coordinate-dependent EPC pruning so sparse-current support cannot drift with
geometry. General disordered/perovskite models should use the native graph or
periodic-block interfaces rather than treat these historical conventions as
universal physical assumptions. GPU execution and speed parity have not been
established by this audit.
