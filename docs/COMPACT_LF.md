# Exact compact local Lang–Firsov correlations

`make_polaron_transport_problem(..., estimator="compact")` is the default LF-CPA
factory. It computes the same complete edge-pair correlation as
`estimator="full"`, with the same full electronic evolution matrix, thermal
preparation, current conventions, output fields, and unitarity diagnostic.
The improvement is in the correlation contraction and its static topology.

```python
from pyeph.workflows.polaron_transport import make_polaron_transport_problem

problem = make_polaron_transport_problem(
    model, params, nuclear_treatment, frequencies, couplings, beta,
    hopping_pairs=directed_current_support,
    estimator="compact",                  # default; "full" is the reference
    thermal_policy="offdiagonal",
)
```

The LF bath remains local, independent, and identical on each electronic site.
This estimator does not add quantum-bath forces to feedback dynamics. The
`offdiagonal` and `legacy_full` initial thermal conventions retain their existing
meanings; changing the estimator does not change either convention.

## Support and compatibility

`hopping_pairs` declares the unique directed union of all supported current
edges over the entire trajectory and every requested probe. Diagonal current
entries must be included when present. The factory checks for unsupported
nonzero initial currents; it cannot infer the support of later geometries.
Current values outside the declared support never enter the compact sum.

A support S means the complete Cartesian product S×S. The existing
`observables.transport.polaron.lf_current_correlation` function and direct
`PolaronTransportMeasurement(..., quad_indices=..., sector_indices=...)`
constructor still evaluate exactly their supplied quadruples. In particular,
a custom quadruple subset is never silently expanded to S×S.

The validated topology is available separately for direct contractions:

```python
from pyeph.observables.transport.polaron_compact import (
    build_compact_lf_sectors,
    lf_current_correlation_compact,
)

topology = build_compact_lf_sectors(directed_current_support, nstates)
correlation = lf_current_correlation_compact(U, rho0, Jt, J0, topology, phi0, phit)
```

`CompactLFSectors(support_pairs, nstates)` performs the same validation and owns
immutable JAX index arrays. Its derived correction indices cannot be supplied
or omitted by callers. Capture the topology as static configuration when using
`jax.jit`; matrix values and bath factors can remain differentiable arguments.
Their leading dimensions broadcast, including trajectory and probe axes.

A measurement may receive `compact_topology=topology` or explicit quadruple
and sector arrays, never both. Direct compact topology must match the model's
electronic dimension. The factory's returned `Problem` type and the transport
state payload are unchanged; the compact measurement has `quad_indices=None`
and `sector_indices=None` because the topology owns its correction arrays.

Strict checkpoints still identify source and measurement configuration. A
checkpoint made with the full estimator is not accepted as an identical compact
run, and a checkpoint predating a source change is not silently migrated.
Reconstructing the same compact configuration supports strict restart across
execution chunk sizes.

## Exact contraction

Let V = conjugate(U) rho0ᵀ and
D(J)ᵢⱼ = exp[(-1+δᵢⱼ) phi0] Jᵢⱼ on S, with zero values outside S. The baseline is

```
C_base = sum_(k,l in S) D(J0)_kl sum_i [D(Jt) U]_ik V_il.
```

It accounts for sector-zero pairs, including pairs involving a diagonal
current. The remaining terms have
n = δᵢₖ − δⱼₖ − δᵢₗ + δⱼₗ ≠ 0 and necessarily share a vertex. Both edges of such a
pair are off diagonal. Their correction is

```
Jt_ij J0_kl U_jk V_il * (exp[-2 phi0 - n phit] - exp[-2 phi0]).
```

The code combines the exponents before exponentiating. This avoids multiplying
an underflowed narrowing factor by an overflowing exponential at strong
coupling. An explicit multiway `einsum` preserves the intended complex
contraction on the minimum supported JAX compiler, including captured constant
inputs; a retained minimal reproduction documents why chained multiplication
was not used.

For N states, E directed edges, and M nonzero-sector pairs, topology construction
visits O(N+E+Σᵥdegree(v)²) candidate data and sorts each incident candidate set
for deterministic ordering. At bounded degree, its cost and M grow linearly
with E. Dense supports can still have quadratic pair count. The baseline applies its masked
current by edge scatter; the full contraction costs O(N³+EN+M). The propagated
U, initial density, and currents remain dense. This is an exact full-U estimator,
not a stochastic trace or a claim of linear electronic memory scaling.

Platform and integrated release evidence are recorded in [qualification](QUALIFICATION.md).
