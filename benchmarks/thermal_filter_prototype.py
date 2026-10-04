"""Isolated matrix-action thermal-column prototype; not a PyEPH runtime API.

Let Hermitian H have spectrum in caller-declared [L,U], c=(L+U)/2,
a=(U-L)/2, A=(H-cI)/a and z=beta*a/2. Then

  F=exp[-beta*(H-LI)/2]
   =ive(0,z) I + 2 sum(n>=1) (-1)^n ive(n,z) T_n(A).

Proof of the conditional exact-arithmetic truncation bound:
|T_n(x)|<=1 on [-1,1]. The omitted coefficient norm is
2 sum(n>K) exp(-z) I_n(z). The nonnegative masses
p_n=exp(-z) I_|n|(z), n in Z, sum to one and have moment-generating
function exp[z(cosh(t)-1)]. Applying Markov's inequality to both tails,
then choosing t=asinh((K+1)/z), gives the bound implemented below.
This is the symmetric Skellam distribution, but the proof needs only its
Bessel generating function. For z=0 the series is exactly the identity.

Sources verified 2026-10-03:
* NIST DLMF10.35.1,10.35.2: https://dlmf.nist.gov/10.35
* SciPy exponentially scaled Bessel implementation:
  https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.ive.html
* Iitaka/Ebisuzaki, random-phase trace estimates, Phys.Rev.E69,057701:
  https://arxiv.org/abs/cond-mat/0401202
* Their earlier Boltzmann/Chebyshev response method:
  https://arxiv.org/abs/cond-mat/0212485

No finite set of opaque action probes certifies Hermiticity or spectral bounds.
The caller owns those contracts; Lanczos extremal estimates are not bounds.
The mathematical bound excludes floating coefficient, action and recurrence
errors; its numerical evaluation is also ordinary floating arithmetic, not
interval arithmetic. Bounds and beta are fixed host planning parameters;
AD below concerns the action parameters at fixed bounds, beta and degree.

For independent unit-phase/sign columns Omega, Y=F Omega, E||Y||F²/r is the
scaled partition Z_L=Tr exp[-beta(H-LI)]. A single pooled normalization
B=Y/||Y||F produces rho_hat=B B†. Observable ratios are generally biased at
finite rank, although numerator and denominator trace estimates are unbiased
for exact F. Per-column normalization gives a different estimator and is
deliberately not used. Never pool partition weights across nuclear geometries.
Polynomial, trace-sampling, nuclear-sampling and dynamics-timestep errors are
distinct. The global norm can amplify filter errors at low temperature or
with a loose lower bound; the returned norm/error diagnostics expose this.

Run ``JAX_ENABLE_X64=1 python benchmarks/thermal_filter_prototype.py --output DIR``.
The report is a small correctness/rank study, not a performance benchmark.
"""

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import platform

import jax
import jax.numpy as jnp
import numpy as np
import scipy
from scipy.linalg import expm
from scipy.special import ive


def _real(value, name):
    array = np.asarray(value)
    if array.shape or array.dtype.kind not in "iuf" or not np.isfinite(array):
        raise ValueError(f"{name} must be a finite real scalar")
    return float(array)


def log_tail_bound(z, degree):
    """Log of a two-sided Chernoff upper bound, capped by probability one.

    The algebraically equivalent first term n²/(hypot(z,n)+z) avoids
    cancellation in sqrt(z²+n²)-z. This remains a floating evaluation of
    an exact-arithmetic inequality, not a validated numerical certificate.
    """
    z = _real(z, "z")
    if z < 0 or isinstance(degree, bool) or not isinstance(degree, (int, np.integer)) or degree < 0:
        raise ValueError("z and integer degree must be nonnegative")
    if z == 0:
        return -np.inf
    n = float(degree+1)
    # Preserve small n/z accurately; avoid its overflow for tiny positive z.
    ratio = n/z
    t = np.arcsinh(ratio) if np.isfinite(ratio) else np.log(n+np.hypot(n, z))-np.log(z)
    exponent = n*(n/(np.hypot(z, n)+z))-n*t
    return min(0., float(np.log(2.)+exponent))


@dataclass(frozen=True, eq=False)
class FilterPlan:
    beta: float
    lower: float
    upper: float
    coefficients: object
    log_truncation_bound: float

    @property
    def degree(self):
        return len(self.coefficients)-1

    @property
    def truncation_bound(self):
        if self.log_truncation_bound == -np.inf:
            return 0.
        # Do not report an exactly zero mathematical tail after underflow.
        return max(float(np.nextafter(0., 1.)), float(np.exp(self.log_truncation_bound)))

    def report(self):
        return dict(beta=self.beta, spectral_interval=[self.lower, self.upper],
            degree=self.degree, conditional_exact_arithmetic_truncation_bound=self.truncation_bound,
            log_conditional_exact_arithmetic_truncation_bound=(None if self.truncation_bound == 0.
                                                              else self.log_truncation_bound),
            coefficient_and_recurrence_roundoff_bound=None,
            bounds_status="caller-declared, not certified by opaque-action probes",
            bound_evaluation="ordinary float64, not interval arithmetic",
            coefficient_implementation="scipy.special.ive, host float64")


def make_plan(beta, lower, upper, *, tolerance=1e-10, max_degree=100000):
    beta, lower, upper, tolerance = (_real(v, n) for v, n in (
        (beta, "beta"), (lower, "lower"), (upper, "upper"), (tolerance, "tolerance")))
    if beta < 0 or lower > upper or not 1e-300 <= tolerance < 1:
        raise ValueError("require beta>=0, lower<=upper and1e-300<=tolerance<1")
    if isinstance(max_degree, bool) or not isinstance(max_degree, int) or max_degree < 0:
        raise ValueError("max_degree must be a nonnegative integer")
    z = beta*((upper-lower)/4)
    if not np.isfinite(z):
        raise ValueError("scaled spectral width must be finite")
    if z == 0:
        coefficients, log_bound = np.array([1.]), -np.inf
    else:
        target = np.log(tolerance)
        low, high = -1, 0
        while log_tail_bound(z, high) > target and high < max_degree:
            low, high = high, min(max_degree, max(1, 2*high))
        if log_tail_bound(z, high) > target:
            raise ValueError("requested truncation bound exceeds max_degree budget")
        while high-low > 1:
            middle = (low+high)//2
            if log_tail_bound(z, middle) <= target:
                high = middle
            else:
                low = middle
        orders = np.arange(high+1)
        coefficients = ive(orders, z)*np.where(orders == 0, 1., 2*(-1.)**orders)
        if not np.isfinite(coefficients).all():
            raise FloatingPointError("scaled Bessel coefficients are nonfinite")
        log_bound = log_tail_bound(z, high)
    coefficients.setflags(write=False)
    return FilterPlan(beta, lower, upper, coefficients, log_bound)


def random_columns(nstates, column_ids, *, seed=0, trajectory_id=0, kind="complex"):
    """Stable identity-indexed unit-modulus columns, with independent streams."""
    for value, name in ((nstates, "nstates"), (seed, "seed"), (trajectory_id, "trajectory_id")):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"{name} must be an integer")
    if nstates < 1 or not 0 <= seed < 2**32 or not 0 <= trajectory_id < 2**32:
        raise ValueError("positive nstates and uint32 seed/trajectory_id required")
    ids = np.asarray(column_ids)
    if (ids.ndim != 1 or not ids.size or ids.dtype.kind not in "iu"
            or np.any(ids < 0) or np.any(ids >= 2**32) or len(np.unique(ids)) != len(ids)):
        raise ValueError("column_ids must be nonempty unique uint32-compatible integers")
    if kind not in ("real", "complex"):
        raise ValueError("kind must be real or complex")
    key = jax.random.fold_in(jax.random.key(int(seed), impl="threefry2x32"), np.uint32(trajectory_id))
    key = jax.random.fold_in(key, np.uint32(0x54484D46))
    keys = jax.vmap(lambda identity: jax.random.fold_in(key, identity))(jnp.asarray(ids, jnp.uint32))
    dtype = jnp.result_type(1.)

    def one(key):
        if kind == "real":
            return jnp.where(jax.random.bernoulli(key, shape=(nstates,)), 1., -1.).astype(dtype)
        phase = 2*jnp.pi*jax.random.uniform(key, (nstates,), dtype=dtype)
        return jnp.exp(1j*phase)

    return jax.vmap(one)(keys).T


def filter_columns(action, columns, plan):
    """Chebyshev forward recurrence: O(degree) block actions, O(N*rank) state.

    No identity, dense Hamiltonian or density matrix is requested. The caller's
    action determines its own memory/time cost. Reverse AD may retain a
    degree*N*rank loop tape; no gradient-memory optimization is claimed.
    A flat interval declares H=L I;
    opaque actions cannot verify that assertion and are not evaluated then.
    """
    columns = jnp.asarray(columns)
    if columns.ndim != 2 or min(columns.shape) < 1 or columns.dtype.kind not in "iufc":
        raise ValueError("columns must be a nonempty numeric(nstates,rank) block")
    coefficients = jnp.asarray(plan.coefficients)
    previous = columns.astype(jnp.result_type(columns, coefficients))
    result = coefficients[0]*previous
    if plan.degree == 0:
        return result
    center, radius = plan.lower/2+plan.upper/2, (plan.upper-plan.lower)/2

    def scaled_action(value):
        applied = jnp.asarray(action(value))
        if applied.shape != value.shape:
            raise ValueError("Hamiltonian action must preserve column-block shape")
        return (applied-center*value)/radius

    current = scaled_action(previous)
    previous = previous.astype(current.dtype)
    result = (result+coefficients[1]*current).astype(current.dtype)

    def advance(index, state):
        previous, current, result = state
        following = 2*scaled_action(current)-previous
        return current, following, result+coefficients[index]*following

    return jax.lax.fori_loop(2, plan.degree+1, advance, (previous, current, result))[2]


def prepare_columns(action, columns, plan):
    """Return one globally normalized factor and inspectable numerical diagnostics.

    Invalid/underflowed normalization produces valid=False and NaN factor;
    callers must check valid before supplying the factor to a dynamics API.
    relative_filter_bound uses the computed norm in its denominator and is a
    conditional exact-arithmetic diagnostic, not a roundoff-inclusive guarantee.
    valid checks normalization only, not thermal accuracy. Taking the logarithm
    of the partition estimate also creates a finite-column Jensen bias.
    """
    columns = jnp.asarray(columns)
    filtered = filter_columns(action, columns, plan)
    weights = jnp.sum(jnp.abs(filtered)**2, axis=0)
    norm_squared = jnp.sum(weights)
    valid = jnp.isfinite(norm_squared) & (norm_squared > 0) & jnp.all(jnp.isfinite(filtered))
    factor = jnp.where(valid, filtered/jnp.sqrt(norm_squared), jnp.nan)
    absolute_bound = plan.truncation_bound*jnp.linalg.norm(columns)
    return dict(factor=factor, filtered=filtered, column_norm_squared=weights,
                scaled_partition_estimate=norm_squared/columns.shape[1],
                log_partition_estimate=jnp.log(norm_squared/columns.shape[1])-plan.beta*plan.lower,
                conditional_filter_error_frobenius_bound=absolute_bound,
                conditional_relative_filter_bound=absolute_bound/jnp.sqrt(norm_squared), valid=valid)


def ring_action(onsite, hopping, vectors):
    """Toy Hermitian periodic chain; only N and N*rank arrays occur."""
    return (onsite[:, None]*vectors + hopping[:, None]*jnp.roll(vectors, -1, axis=0)
            + jnp.conj(jnp.roll(hopping, 1))[:, None]*jnp.roll(vectors, 1, axis=0))


def study(output):
    """Small fixed-geometry rank study, independent seeds; no timing claims."""
    if not jax.config.x64_enabled:
        raise ValueError("this diagnostic study requires JAX_ENABLE_X64=1")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(107)
    n = 12
    onsite = rng.normal(size=n)*.7
    hopping = rng.normal(size=n)*.2+1j*rng.normal(size=n)*.15
    radius = abs(hopping)+abs(np.roll(hopping, 1))
    lower, upper = float(np.min(onsite-radius)-.01), float(np.max(onsite+radius)+.01)
    plan = make_plan(3., lower, upper, tolerance=1e-12)
    # Dense arrays exist only in this explicitly small independent reference.
    h = np.diag(onsite).astype(complex)
    for i in range(n):
        h[i, (i+1) % n] += hopping[i]
        h[(i+1) % n, i] += hopping[i].conjugate()
    exact_filter = expm(-plan.beta*(h-lower*np.eye(n))/2)
    density = exact_filter@exact_filter
    partition = np.trace(density).real
    observable = np.diag(np.linspace(-1, 1, n))
    exact_mean = np.trace(density@observable).real/partition
    ranks, seeds = (1, 4, 16, 64), range(96)
    rows, arrays = {}, dict(h=h, observable=observable)
    def action(vectors):
        return ring_action(jnp.asarray(onsite), jnp.asarray(hopping), vectors)

    for rank in ranks:
        prepared = jax.jit(lambda columns: prepare_columns(action, columns, plan))
        estimates, partitions, filter_errors = [], [], []
        for seed in seeds:
            omega = random_columns(n, np.arange(rank), seed=seed, trajectory_id=7)
            result = prepared(omega)
            if not bool(result["valid"]):
                raise FloatingPointError("invalid thermal factor")
            y, factor = map(np.asarray, (result["filtered"], result["factor"]))
            estimates.append(float(np.vdot(factor, observable@factor).real))
            partitions.append(float(result["scaled_partition_estimate"]))
            filter_errors.append(float(np.linalg.norm(y-exact_filter@omega)/np.linalg.norm(omega)))
        arrays[f"rank_{rank}_observable"] = np.array(estimates)
        arrays[f"rank_{rank}_partition"] = np.array(partitions)
        error = np.asarray(estimates)-exact_mean
        rows[str(rank)] = dict(observable_mean=float(np.mean(estimates)),
            empirical_bias=float(np.mean(error)), empirical_rmse=float(np.sqrt(np.mean(error**2))),
            across_seed_mean_sem=float(np.std(estimates, ddof=1)/np.sqrt(len(estimates))),
            partition_mean=float(np.mean(partitions)), exact_scaled_partition=float(partition),
            maximum_filter_error_per_input_frobenius_norm=max(filter_errors))
    np.savez_compressed(output/"arrays.npz", **arrays)
    report = dict(plan=plan.report(), fixed_geometry=True, exact_observable=float(exact_mean),
        controlled_independent_seeds=list(seeds), nested_column_ids=True, ranks=rows,
        rank_label="number of random columns; may exceed Hilbert dimension in this small statistical study",
        estimator="pooled ratio within one nuclear geometry; finite-rank biased",
        uncertainty="empirical trace-seed variation only; no nuclear ensemble or dynamics here",
        floating_errors="observed dense-expm discrepancies, not certified error bounds",
        prototype_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        versions=dict(python=platform.python_version(), jax=jax.__version__,
                      numpy=np.__version__, scipy=scipy.__version__), jax_enable_x64=jax.config.x64_enabled)
    (output/"report.json").write_text(json.dumps(report, indent=2)+"\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    print(json.dumps(study(parser.parse_args().output), indent=2))
