# Degenerate-subspace transport and the obstruction to individual labels

This independent research fixture proves gauge-covariant adiabatic transport of
an isolated complex two-dimensional electronic subspace. It also demonstrates
why individual eigenvector populations cannot define a physical active state
inside a degenerate block. It adds no hopping API, nuclear force prescription,
SOC material model or extension of the real MASH implementation.

The geometry is established theory: [Kato's adiabatic transport](https://doi.org/10.1143/JPSJ.5.435)
and the [Wilczek–Zee connection for degenerate states](https://doi.org/10.1103/PhysRevLett.52.2111).
A modern operator derivation of the projector commutator appears in Appendix E
of [Albert et al., Geometry and Response of Lindbladians](https://doi.org/10.1103/PhysRevX.6.041031).
The calculation below independently implements the finite-dimensional equations
for a fully declared analytic fixture. Its purpose is to establish requirements
and failure cases before attempting a manifold dynamics method.

## Model and actual time-reversal symmetry

Use atomic units and a fixed four-dimensional orthonormal basis ordered as
\((L\uparrow,L\downarrow,R\uparrow,R\downarrow)\). There is one carrier, no
scalar reference energy, and no nuclear feedback. Three prescribed model
coordinates \(q=(x,y,z)\) enter linearly with unit energy-per-coordinate coupling.
They are illustrative control coordinates, not a validated atomistic geometry.

With the ordinary spin Pauli matrices, define

\[
H(q)=\begin{pmatrix}\Delta I_2&T(q)\\T(q)^\dagger&-\Delta I_2\end{pmatrix},
\qquad T(q)=xI_2+iy\sigma_y+iz\sigma_z,\qquad\Delta=0.4.
\]

The complex spin-dependent coupling is an effective model. Merely having such
matrix entries does not identify an electronic-structure SOC approximation.
Here the antiunitary time-reversal operator is explicitly

\[
\Theta=J\mathcal K,\qquad J=I_{\rm site}\otimes i\sigma_y,
\qquad JJ^*=-I_4,
\]

where \(\mathcal K\) conjugates coefficients in the declared fixed basis.
The quaternion relation \((i\sigma_y)T^*(i\sigma_y)^\dagger=T\) gives
\(JH^*J^\dagger=H\) for every real \(q\). Thus \(\Theta^2=-1\), and an
eigenvector and its orthogonal time-reversed partner have the same eigenvalue.
The tests verify these identities and the partners directly; the word Kramers
is not inferred just from numerically repeated eigenvalues.

In fact,

\[
TT^\dagger=(x^2+y^2+z^2)I_2,\quad H^2=E^2I_4,\qquad
E=\sqrt{\Delta^2+x^2+y^2+z^2}.
\]

There are two rank-2 clusters with energies \(-E\) and \(+E\), separated by
at least 0.8. The lower projector and an analytic frame are

\[
P=\frac{I_4-H/E}{2},\qquad
F=\frac{1}{\sqrt{2E(E+\Delta)}}
\begin{pmatrix}-T\\(E+\Delta)I_2\end{pmatrix},
\qquad F^\dagger F=I_2,\quad FF^\dagger=P.
\]

The projector is independent of the arbitrary choice \(F\to FG(q)\), with
\(G\in U(2)\). Such a change can alter a particular paired-column convention
for \(\Theta\); its physical fixed-space action remains the one declared above.
No division by a within-doublet energy gap occurs.

## Prescribed paths and the equation being tested

The base point is \(q_0=(0.55,0,0.25)\). For \(s\in[0,1]\), the two closed
paths are

\[
q_A(s)=q_0+0.42(\cos2\pi s-1,\;\sin2\pi s,\;0),
\]
\[
q_B(s)=q_0+0.36(0,\;\sin2\pi s,\;\cos2\pi s-1).
\]

Allowing the third coefficient to vary enables a test of two noncommuting
holonomies at the same base point. A single loop cannot establish
noncommutativity by itself.

Analytic differentiation gives

\[
\partial_iP=-\frac12\left(\frac{\partial_iH}{E}
                          -\frac{Hq_i}{E^3}\right),\qquad
P'=\sum_i q_i'(s)\partial_iP.
\]

The independent full-space ODE is

\[
Y'=[P',P]Y,\qquad Y(0)=P(0).
\]

This is **adiabatic geometric parallel transport with the common dynamical
phase removed**. It is not exact finite-speed Schrödinger evolution under
\(H(q(t))\). Since \([P',P]\) is anti-Hermitian and
\([[P',P],P]=P'\), the transported operator obeys

\[
Y^\dagger Y=P(0),\qquad YY^\dagger=P(s),\qquad PY=Y.
\]

It is a partial isometry between the doublets, represented throughout in the
full fixed four-dimensional space. SciPy DOP853 integrates this equation using
analytic projectors and derivatives, without an eigenvector frame or overlap
transport. A tighter integration independently checks this reference.

## Discrete whole-subspace transport and gauge covariance

At grid points, take any orthonormal frames \(F_n\) spanning \(P_n\). Let

\[
M_n=F_{n+1}^\dagger F_n=L_n\Sigma_nR_n^\dagger,
\qquad V_n=L_nR_n^\dagger,
\]
\[
Y_n=F_nV_{n-1}\cdots V_0F_0^\dagger.
\]

The unitary polar factors retain the complete 2-by-2 overlap information. Under
independent frame changes \(F_n\to F_nG_n\),
\(V_n\to G_{n+1}^\dagger V_nG_n\); all internal and endpoint gauges cancel
in the physical \(Y_n\). The benchmark checks both a smooth coordinate-dependent
\(U(2)\) gauge and arbitrary discontinuous sampled unitary gauges, including
different endpoint frames. It rejects a rank-deficient overlap rather than
creating a unitary factor with a singular-value floor.

For a closed loop, the doublet holonomy in an initial frame is
\(W=F_0^\dagger Y_1F_0\). The matrix transforms by conjugation,
\(W\to G_0^\dagger WG_0\); its trace and eigenvalues are gauge invariant.
The code compares the full physical transport operator, these invariants, and
spin polarizations \(\langle I_{\rm site}\otimes\sigma_i\rangle\).
The reported polarizations equal \(2\langle S_i\rangle/\hbar\), not spin
angular momenta themselves. The initial physical spinor is
\(F_0(\sqrt{0.7},i\sqrt{0.3})^T\).

Gauge-invariant Wilson-loop observables have established experimental examples;
see [Sugawa et al., Wilson loop and Wilczek–Zee phase from a non-Abelian gauge
field](https://doi.org/10.1038/s41534-021-00483-2). The present model and loops do
not reproduce that experiment or its five-dimensional parameter space.

## Concrete information lost by individual labels or block averaging

At one fixed geometry, choose the physical state \(\psi=F(1,0)^T\). The frames
\(F\), \(F(I-i\sigma_y)/\sqrt2\), and \(F(-i\sigma_y)\) give individual
eigenvector populations

\[
(1,0),\qquad(1/2,1/2),\qquad(0,1)
\]

for exactly the same \(\psi\), \(H\), energy and projector. Selecting an active
individual eigenvector from these populations depends on an unphysical frame
choice. Rephasing or matching columns one by one cannot remove the general
\(U(2)\) freedom. A within-block gap formula is undefined.

The cluster population \(\psi^\dagger P\psi=1\) avoids that ambiguity, but it
does not determine the internal state. For example, the Hermitian curvature
matrix in a frame is

\[
\Omega_{ij}=iF^\dagger[\partial_iP,\partial_jP]F.
\]

At the declared base point, \(\Omega_{xy}\) has eigenvalues approximately
\(\pm0.620006\). Its two eigenstates have the same cluster population and
electronic energy, yet opposite curvature expectations. Replacing a pure
doublet state by \(P/2\) also erases spin polarization: it gives zero where
\(F(1,0)^T\) gives unit \(\sigma_z\) polarization.

There is an important distinction here. In this exactly degenerate model,
\(F^\dagger(\partial_iH)F=-(q_i/E)I_2\), so the scalar potential force from
\(-E\) is the same for every internal state. The obstruction is not an arbitrary
choice of this scalar gradient. That gradient alone cannot specify internal
transport, spin observables, transitions, or geometric nuclear feedback.
Curvature enters appropriately derived adiabatic effective theories; this
prescribed-path proof does **not** insert a Berry force into nuclear equations.

## Finite-speed dynamics is a separate calculation

For a physical traversal time \(T\), the exact electronic equation on the
prescribed path is

\[
i\frac{d\psi}{ds}=T H(q(s))\psi.
\]

The adiabatic approximation within the lower doublet instead has
\(\psi_{\rm ad}(s)=\exp[-iT\int_0^s\epsilon_-(u)du]Y(s)\psi(0)\).
Comparing physical density matrices removes this common dynamical phase.
The exact equation can leave the lower doublet; the geometric transport
equation cannot. The finite-speed comparison records this leakage rather than
projecting it out or renormalizing the state. Energy need not be conserved:
the prescribed coordinates do external work, and no nuclear energy is evolved.

## Requirements for a future manifold dynamics method

The next method design must supply all of the following together:

- Smooth constant-rank projectors or complete block overlaps, isolated
  inter-cluster gaps, and explicit rules for changing cluster rank or closing
  gaps. Individual eigenvector continuity is insufficient.
- A covariant internal state, such as block amplitudes/density matrices,
  inter-block coherences where the approximation needs them, and a declared
  transformation law for time reversal and physical spin observables.
- A justified preparation and estimator. Summing raw mapping populations into
  blocks does not derive a new mapping ensemble or establish equilibrium,
  detailed balance or agreement with quantum dynamics.
- Physical energy and force operators, a scalar reference, and a consistent
  treatment of the connection/curvature if geometric nuclear terms are used.
  Nuclear canonical versus kinetic momentum must be specified in such a theory.
  A block trace or an arbitrary mean force does not define this approximation.
- Gauge-covariant inter-cluster transitions, an energy and momentum treatment
  derived with the electronic approximation, and tests of spin-dependent
  observables against independent quantum benchmarks. A complex coupling matrix
  is not by itself a unique real momentum-rescaling direction.

There is already relevant method development: [Bian et al., Modeling
Spin-Dependent Nonadiabatic Dynamics with Electronic Degeneracy: A Phase-Space
Surface-Hopping Method](https://doi.org/10.1021/acs.jpclett.2c01802) addresses
Berry-curvature effects in a singlet–triplet model. It is prior art for the
broader problem, not an implementation included here. A future contribution
must identify a distinct controlled approximation and validate its physical
predictions beyond this geometry proof.

## Reproduction and scope

```sh
python -m pytest -q tests/test_degenerate_subspaces.py
python -m benchmarks.degenerate_subspaces --output outputs/degenerate-subspaces-new-run
```

The standalone benchmark uses NumPy and SciPy and does not alter native runtime
precision or interfaces. It saves explicit parameters, the initial spinor and
time-reversal convention, raw transport/exact-dynamics arrays, the benchmark
source, hashes and dependency versions in a fresh directory. Tests also verify
that native individual-surface validation continues to reject these doublets.

The recorded maximum entrywise errors of polar transport relative to the
full-space ODE were:

| Loop | 32 steps | 64 steps | 128 steps |
| --- | ---: | ---: | ---: |
| A, xy | 1.89e-3 | 4.71e-4 | 1.18e-4 |
| B, yz | 1.40e-3 | 3.50e-4 | 8.75e-5 |

This is second-order convergence for the declared loops. Changing to the smooth
coordinate-dependent gauge changed transport entries by at most 2.6e-15 and
spin polarizations by at most 4.3e-15. Tightening the independent ODE changed
its entries by at most 1.4e-15. The holonomy eigenphases were approximately
\(\pm0.762738\) and \(\pm0.355856\); the Frobenius norm
\(\|W_AW_B-W_BW_A\|\) was 0.615107. These noncommuting transformations act on
the internal state even though each doublet stays exactly degenerate.

The separate exact Schrödinger calculation had maximum upper-doublet
populations 0.147524, 0.020648 and 0.001041 at traversal times 4, 20 and 100.
Its density-matrix differences from geometric adiabatic transport decreased
from 0.3130 to 0.0880 to 0.0187. Increasing traversal time improves this
approximation in the tested sequence; making the geometric transport grid
finer at fixed physical speed cannot remove its omitted transitions.

This is evidence for a controlled adiabatic subspace approximation on known
smooth prescribed paths, plus its finite-speed limitation. It does not qualify
complex/SOC MASH, degenerate hopping, thermal transport, learned degeneracies,
truncated moving AO spaces, changing ranks, a nuclear force law or a realistic
material calculation.
