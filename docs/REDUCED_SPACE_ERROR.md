# Controlled fixed-space reduction: a research experiment

The [fixed-space error benchmark](../benchmarks/reduced_space_error.py) examines
a precise prerequisite for reducing electronic cost: how to bound the error
introduced by a **fixed** electronic subspace during prescribed-path propagation.
It is a standalone experiment, not automatic truncation in the runtime and not
a reduced MASH formulation.

## A residual with a direct physical meaning

Let the full electronic Hamiltonian h(t) be Hermitian, with hbar=1, and let the
columns of the fixed matrix P be orthonormal. The full and projected equations
are

\[
 i\dot\psi=h\psi,\qquad i\dot c=P^\dagger hP c,\qquad \phi=Pc.
\]

The lifted projected state generally fails the full equation by the residual

\[
 r=(I-PP^\dagger)hPc,\qquad i\dot\phi=h\phi-r.
\]

Writing the error as e=psi-phi gives `i de/dt = h e + r`. Variation of constants
and unitarity of the full propagator give

\[
 \|e(t)\|\le\|e(0)\|+\int_0^t\|r(s)\|\,ds.
\]

This is a continuum bound under the stated assumptions. It includes any initial
component outside the subspace. Renormalizing a projected initial state changes
that initial mismatch and cannot be done without accounting for it.

For any Hermitian observable O,

\[
 |\langle\psi|O|\psi\rangle-\langle\phi|O|\phi\rangle|
 \le (\|\psi\|+\|\phi\|)\,\|O\|\,\|e\|.
\]

Population projectors have operator norm one. The benchmark also compares the
particle-current operator `J(t)=i[h(t),X]` for a fixed diagonal position X,
using unit carrier charge. An electrical current requires the declared charge
factor. This does not introduce moving-center, ionic or intra-orbital currents.

## An enclosure and a sampled indicator are different evidence

If

\[
 h(t)=\sum_k f_k(t)H_k,\qquad |f_k(t)|\le a_k
\]

throughout the full time interval, a computable conservative bound is

\[
 \|e(t)\|\le\|e(0)\|+t\|c(0)\|\sum_k a_k\|(I-PP^\dagger)H_kP\|.
\]

The coefficient bounds must hold between sampled frames. In this experiment the
coefficients are exactly a constant and a sine, whose absolute values are at
most one analytically. The implementation specializes P to a coordinate-axis
selector, which is exactly orthonormal. Its omitted/retained cross-block is
therefore obtained by indexing, avoiding a rounded dense projector product.
A Frobenius norm upper bound replaces each spectral norm. Scalar positive
products, sums and square roots are rounded outward by one binary64 ULP.

That numerical enclosure applies to the supplied matrix entries and the stated
analytical coefficient bounds. It does not certify upstream fitted Hamiltonian
errors or the numerical ODE solver. The independent full/projected SciPy solves
use tight tolerances and a maximum step; the regression allowance for their
finite numerical error is reported separately from the continuum inequality.

The benchmark also integrates **sampled** residual norms with a trapezoid rule.
This is an indicator, not an upper bound: sampling can miss excursions between
frames. It is deliberately never labeled a certified estimate.

A failure case makes the distinction explicit. Take `h(t)=sin(t)*sigma_x`,
retain only the first basis state, and start in that state. The projected
Hamiltonian is zero, so the reduced state has exactly constant norm. Residual
samples at 0 and pi are both zero, but the integrated rotation angle is two.
The final full-state error is `2*sin(1)`, and substantial population has left
the retained state. The coefficient-enclosed bound remains valid while the
coarse sampled indicator misses the error.

## What the cases establish

The generated models use a fixed orthonormal basis, atomic units and a
prescribed smooth Hamiltonian. They are parameterized numerical fixtures.

- Weak omitted coupling with a large gap tests a regime where projection works.
- Strong omitted coupling with the same large gap separates coupling magnitude
  from resonant leakage.
- Strong coupling near resonance demonstrates large population/current errors
  despite small numerical drift in the reduced wavefunction norm.
- The half-sine case demonstrates that a sampled residual integral can fail.
- An independent constant two-state solution and a nonzero initial omitted
  amplitude validate the equations and initial-error term.

Run into a fresh output directory:

```sh
python benchmarks/reduced_space_error.py --output outputs/reduced_space_study
```

The output contains full/lifted states, residual samples, physical observable
comparisons, coefficient enclosures, solver settings/evaluation counts and the
benchmark source identity. Norm conservation is a numerical integration check;
it does not certify the physical accuracy of an electronic truncation.

The enclosure can be quite conservative and does not exploit off-resonant
phase cancellation. A small bound is useful evidence of a small error. A large
bound says that this particular argument is inconclusive, even when the actual
projection error is small.

## Remaining theory before a production reduction method

This derivation compares two electronic states under the same prescribed h(t).
If nuclei respond differently to their electronic states, the full and reduced
Hamiltonians are evaluated along different paths; feedback, force error and
stability bounds must enter the analysis. A moving P(t) adds connection and
subspace-motion terms. Adding/removing states also changes state preparation
and conservation requirements.

MASHRM sampling and observable estimators depend on the electronic dimension.
Changing that dimension is a change of approximation, not an automatic
acceleration justified by this residual. Neither canonical partition-function
errors, discarded-state thermal weight, hopping events nor correlated
force/transport errors are resolved by this experiment. Those are separate
research milestones before adaptive dynamics can be advertised.
