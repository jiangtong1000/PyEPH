# Finite-temperature preparation using Hamiltonian actions

`pyeph.thermal` is an opt-in preparation helper for the existing column CPA
workflow. It accepts a linear Hermitian action on an `(N, K)` block and an
explicit spectral enclosure. It does not require a dense Hamiltonian,
diagonalization, a new model base class, or changes to trajectory/checkpoint
schemas. Forward working storage is proportional to the column block and
the operator's own working storage. Applying the same idea to another
dynamics method requires that method's independently specified preparation.

```python
import jax
from pyeph.thermal import (
    ThermalFilterPlan, thermal_random_columns,
    prepare_thermal_columns, require_success,
)
from pyeph.workflows.column_transport import initialize_column_transport_state

# The caller establishes a full-spectrum enclosure at this initial geometry.
plan = ThermalFilterPlan(
    beta, lower, upper, polynomial_atol=1e-12,
    action_id="provider-weights-parameters-and-q0-artifact-id",
    bounds_id="spectral-enclosure-argument-and-input-artifact-id",
)
omega = thermal_random_columns(nstates, [2, 7, 12], seed=71, trajectory_id=5)
prepare = jax.jit(lambda params, q, columns: prepare_thermal_columns(
    lambda block: problem.model.apply(params, q, block), columns, plan, rtol=1e-6,
))
prepared = prepare(problem.params, q0, omega)
factor = require_success(prepared)  # mandatory host acceptance before dynamics
initial = initialize_column_transport_state(
    problem, q0, p0, factor, trajectory_id=5, seed=71,
)
```

The action, beta, and bounds must use the same energy convention. Beta has
inverse-energy units; the factor is dimensionless. Bounds are required even
for an opaque action or neural network. A caller may use a justified analytic
enclosure, such as Gershgorin row sums for explicit matrix elements. Finite
action probes and approximate Lanczos extrema do not certify an enclosure.
The helper does not discover models, tighten bounds, change temperature,
or infer the identity of a captured function. Incorrect bounds or a nonlinear
or non-Hermitian action invalidate its mathematical error statements.

Models with a `prepare_action` hook can reuse fixed-geometry coefficients.
Construct the action inside the compiled function while keeping parameters
and coordinates as arguments. This avoids reevaluating a network inside
the recurrence and requires no additional thermal API.

## Filter and conditional error bound

For `L <= spectrum(H) <= U`, the helper approximates

\[
F=\exp[-\beta(H-LI)/2],\quad
A=(H-cI)/a,\quad c=(L+U)/2,\quad a=(U-L)/2,\quad
z=\beta(U-L)/4.
\]

Its Chebyshev coefficients are `ive(0,z)` and
`2*(-1)**n*ive(n,z)`, where `ive(n,z)=exp(-z)*I_n(z)` for nonnegative z.
The scaled coefficients avoid evaluating a large unscaled Bessel function.
See the [DLMF generating function](https://dlmf.nist.gov/10.35) and
[SciPy's scaled Bessel definition](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.ive.html).
The recurrence makes exactly `degree` block actions. Beta zero, or a flat
interval declaring `H=L*I`, gives the identity without an action call.

For degree `K`, put `m=K+1`. The discarded coefficient sum is
`2*sum(exp(-z)*I_n(z), n>K)`. These coefficients are the probabilities of
a symmetric integer variable with moment-generating function
`exp(z*(cosh(t)-1))`, directly from the Bessel generating function. Applying
Markov's inequality to both tails and minimizing at `t=asinh(m/z)` gives

\[
\|F-P_K(A)\|_2\le\tau_K
=\min\left\{1,2\exp\left[
\frac{m^2}{\sqrt{z^2+m^2}+z}-m\operatorname{asinh}(m/z)
\right]\right\}.
\]

The fraction avoids subtracting nearly equal positive numbers. At z zero,
the tail is exactly zero. Otherwise a positive tail that underflows is never
reported as an exact zero in host plan metadata. The execution-relative bound
uses the working dtype's smallest normal
as a conservative floor for a positive tail. Only a true identity plan reports
an exact zero. A positive beta and nonflat interval whose scaled width
underflows host precision is rejected.
Degree selection is on the host; coefficients and bounds are evaluated with ordinary float64 arithmetic,
not interval arithmetic. This is a conditional exact-arithmetic truncation
bound, not a roundoff-inclusive certificate.

For `Y=P_K(A)@omega`, let `d=||Y||F`,
`eta=tau_K*||omega||F`, and `epsilon_poly=eta/d`.
The returned factor is `Y/d`, with one normalization over **all columns**.
The helper accepts the polynomial budget only when `epsilon_poly <= rtol < 1`.
Conditionally, its normalized-factor error is at most `2*epsilon_poly`, and
the trace-norm error of its density is at most `min(2,4*epsilon_poly)`.
These compare with the exact filtered factor **for the same omega**; they
do not bound finite-column thermal sampling error.

The separate `heuristic_precision_screen` is
`eps_execution*(degree+1)**2*||omega||F/||Y||F` and must also be at most `rtol`.
It uses the actual result dtype. This screen catches severe normalization
amplification, including many excessively loose lower bounds. Passing it
does not certify errors in an opaque action, rounded coefficients, recurrence,
or normalization. A tiny polynomial tail alone is insufficient. Failure has
an explicit status and a NaN factor; `require_success` raises an exception
retaining all diagnostics. Improve the established bounds or execution
precision, or deliberately adjust the requested tolerance; no automatic
retry conceals a failed preparation.

## Normalization, sampling, and differentiation

Max-scaled Frobenius normalization and log norms avoid unnecessary overflow
or underflow from squaring a very large or small filtered block. A usable
factor and finite log partition can exist when the raw scaled partition is
zero or infinity. Separate underflow/overflow flags report this; host JSON
diagnostics use `null` for an overflowed raw value and retain the log value.

With independent unit-phase or sign columns, `E[omega@omega.conj().T]/K=I`.
Consequently `||F@omega||F**2/K` is an unbiased estimate of
`exp(beta*L)*Z` for an exact filter. `log_partition_estimate` subtracts
`beta*L` from its logarithm. The logarithmic estimate and the normalized
thermal ratio generally have finite-rank bias. Polynomial approximation
adds its own error. `prepare_thermal_columns` also accepts generic numeric
blocks; their norm statistics are **not** automatically unbiased partition
estimates. Arbitrary rescaling of a generic block preserves the factor but
changes the reported partition statistic.

Do not normalize each filtered column separately. To combine independently
prepared blocks at the same geometry, weight each normalized block by
`sqrt(softmax(log_norm_squared))` before concatenating. Equivalently pool the
unnormalized numerator and denominator. Never use electronic partition
weights to combine independent nuclear trajectories unless that weighting is
explicitly part of the intended physical ensemble.

The random-column generator uses explicit Threefry keys derived from the
seed, global trajectory ID, namespace `0x54484D46`, and global column ID.
Permuted or partitioned IDs give bitwise-identical columns before
normalization. IDs must be unique within one block. A scalar ID requests one
column. Each geometry can construct its own block; batch preparation is an
explicit `jax.vmap` over the single-geometry kernel. Each geometry retains
its own normalization and diagnostics. IDs do not make independent samples
out of duplicated data, and floating reductions may change with block shape.

Automatic differentiation is supported through the captured action's
parameters and coordinates with fixed bounds, beta, degree, and columns,
as long as all varied operators satisfy the same enclosure. Host planning,
random IDs, and the acceptance decision are not differentiable. The
accepted factor is smooth away from a zero norm; rejected factors are NaN.
Reverse mode may retain storage proportional to `degree*N*K`, unlike the
forward recurrence's small number of blocks.

## Recipe, provenance, and validation

Run the native complex disordered-ring example from the repository root:

```sh
JAX_ENABLE_X64=1 PYTHONPATH=src python examples/thermal_columns.py \
  --output /tmp/thermal-ring --nstates 12 --ncolumns 4 --beta 3 --steps 40
```

It writes the original random columns and converted factor, a preparation
JSON record, streamed current correlation, and a strict checkpoint. The toy
uses atomic model units with hbar, charge magnitude, and ring spacing equal
to one; its short correlation is not a material mobility prediction.
`examples/thermal_columns.py` binds the recipe to `factor_digest` **after**
`initialize_column_transport_state` performs its dtype conversion. Plan
metadata records caller-owned action/bounds IDs, coefficient hash, degree,
and budget. The recipe adds seed/IDs, diagnostics, actual factor shape/dtype,
and the simulation manifest. Retain this record alongside the checkpoint.
An arbitrary factor remains the existing explicit-factor preparation kind;
no thermal metadata is silently inserted into its payload. Strict resume
loads the original state/current insertions instead of regenerating a factor.

The independent driver can be run with:

```sh
JAX_ENABLE_X64=1 PYTHONPATH=src python benchmarks/thermal_columns.py \
  --output /tmp/thermal-validation --seeds 96
```

It compares preparation with SciPy `expm`, a complete-basis factor with the
dense thermal density, public CPA correlations with an independent complex
DOP853 ODE, and split/checkpoint/restarted propagation with an uninterrupted
run. Timestep halving separates dynamics error from preparation error. It
saves raw independent trace-seed estimates at ranks 1, 4, 16, and 64 for one
fixed geometry. The mean's standard error describes repeated trace seeds;
it is neither a bound on finite-rank bias nor evidence of nuclear convergence.
The earlier isolated prototype and its evidence remain separate artifacts.

`--timing` additionally measures matched CPU preparations using the same H,
beta, random columns, and ratio estimator. Plan construction, RNG, compilation,
first call, and repeated warm calls are separated. Dense eigensolve filtering
is compared with the sparse action filter, excluding dense-H assembly and
current insertion on both sides; these are preparation timings, not complete
transport speedups. Run timings without competing CPU workloads. Always
include compilation in a one-off preparation cost: warm reuse requires fixed
plan values and block shapes, even though Hamiltonian parameters and
coordinates remain dynamic. Different bounds may require a new compilation.
Qualify polynomial tolerance, working precision, trace rank/seed variation,
nuclear ensemble size, timestep, system size, and physical model assumptions
separately. General background on stochastic traces and Chebyshev methods
is available in the primary papers on
[random phase trace estimation](https://arxiv.org/abs/cond-mat/0401202) and
[finite-temperature Chebyshev response calculations](https://arxiv.org/abs/cond-mat/0212485).

Platform and integrated release evidence are recorded in [qualification](QUALIFICATION.md).
