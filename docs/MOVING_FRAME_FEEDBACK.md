# Complete moving-frame feedback proof

This research calculation verifies the spatial connection and physical
Ehrenfest force for a **known invertible change of coordinates within a complete
fixed electronic Hilbert space**. It is deliberately separate from the native
fixed-orthonormal model interface and from the recorded AO ingestion bridge.
It does not turn arbitrary Hamiltonian and overlap labels into a force provider.

The underlying connection geometry is established theory. In particular,
Artacho and O'Regan distinguish changing the basis within a space from changing
the physical space itself in
[Quantum mechanics in an evolving Hilbert space, PRB 95, 115155 (2017)](https://doi.org/10.1103/PhysRevB.95.115155).
For the broader time-dependent-discretization and physical force problem, see
[Ojanperä et al., Nonadiabatic Ehrenfest molecular dynamics within the
projector augmented-wave method](https://doi.org/10.1063/1.3700800).
The equations below are derived directly for the restricted complete-space
case; this calculation is a correctness foundation, not a new dynamics method
or a reproduction of a material calculation from either paper.

## Defined physical problem

Use atomic units, a normalized single-carrier state \(\psi\) in a complete fixed
orthonormal three-state space, and two classical coordinates \(q_i\) with
mechanical momenta \(p_i=M_i\dot q_i\). The total energy is

\[
E=\sum_i\frac{p_i^2}{2M_i}+V_{\mathrm{ref}}(q)+\psi^\dagger h(q)\psi.
\]

The supplied smooth Hermitian \(h(q)\) is the physical carrier Hamiltonian.
The scalar reference is a declared harmonic-plus-quartic potential. These are
illustrative effective-model parameters, not quasiparticle eigenvalues being
identified with many-electron total energies. There is no electronic population
renormalization or nuclear thermostat in the test.

Let the columns of the square, smooth, invertible matrix \(B(q)\) be the moving
orbitals expressed in the fixed physical space. Coefficients are **columns**:

\[
\psi=Bc,\quad S=B^\dagger B,\quad H=B^\dagger hB,\quad
A_i=B^\dagger\partial_iB,\quad \Gamma_i=S^{-1}A_i=B^{-1}\partial_iB.
\]

These column coefficients are a different array convention from the existing
recorded AO bridge's row-ket export convention. No implicit transpose, ordering
conversion or orbital correspondence is inferred between the two.

## Electronic propagation and force

Substituting \(\psi=Bc\) into \(i\dot\psi=h\psi\) gives

\[
iS\dot c=\left(H-i\sum_i\dot q_iA_i\right)c,
\qquad
\dot c=-iS^{-1}Hc-\sum_i\dot q_i\Gamma_i c.
\]

Metric compatibility is
\(\partial_iS=A_i+A_i^\dagger\). Consequently, differentiating
\(c^\dagger Sc\), including the time dependence of \(S\), gives zero.
The implementation uses linear solves; it neither floors the metric spectrum
nor silently orthogonalizes a singular input.

The force must differentiate the energy while holding the **physical state
\(\psi\)** fixed. Since

\[
\partial_iH=(\partial_iB)^\dagger hB+B^\dagger(\partial_i h)B
              +B^\dagger h(\partial_iB),
\]

the physical derivative in moving coordinates is

\[
D_iH\equiv\partial_iH-\Gamma_i^\dagger H-H\Gamma_i
       =B^\dagger(\partial_i h)B.
\]

Thus the complete force for this defined problem is

\[
\dot p_i=F_i=-\partial_iV_{\mathrm{ref}}-c^\dagger D_iHc.
\]

Using \(-c^\dagger\partial_iHc\) instead holds the coordinate vector \(c\)
fixed while its physical state moves with the basis. It produces a different,
incorrect force for this problem. In the finite-difference force test, every
displaced geometry instead uses \(c(q')=B(q')^{-1}\psi\).

These coupled equations conserve the stated total energy in continuous time.
The reported finite-step energy residuals measure the chosen numerical
integrators; neither fourth-order Runge–Kutta nor the native second-order
Ehrenfest splitting is claimed to conserve energy exactly at finite step.

## Frame covariance and missing connection information

For an arbitrary smooth invertible relabeling \(B'=BG(q)\), use

\[
c'=G^{-1}c,\quad S'=G^\dagger SG,\quad H'=G^\dagger HG,
\]
\[
A_i'=G^\dagger A_iG+G^\dagger S\partial_iG,\qquad
\Gamma_i'=G^{-1}\Gamma_iG+G^{-1}\partial_iG.
\]

The force operator transforms covariantly,
\(D_i'H'=G^\dagger D_iHG\), so physical forces, norms and energies are
unchanged. The second trajectory fixture uses a coordinate-dependent,
nonunitary complex \(G\), testing more than constant sign or phase alignment.

Knowing \(S\) and \(\partial_iS\) fixes only the Hermitian combination
\(A_i+A_i^\dagger\). Replacing \(A_i\) with
\(\tfrac12\partial_iS\) discards independent information. A unitary moving
frame already has \(S=I\) and \(\partial_iS=0\) while its connection need
not vanish. Norm conservation alone therefore cannot certify physical dynamics.

There is also a stronger identifiability counterexample. Choose constant
\(H_0=\operatorname{diag}(0.3,-0.3)\). One physical realization has
\(B_1=I,h_1=H_0\). Another has
\(B_2=U(q),h_2=U(q)H_0U(q)^\dagger\), where \(U\) rotates with rate 0.7.
Both give the same raw matrices \(H=H_0,S=I\) at every geometry and the same
zero derivatives of those matrices. For raw coefficients
\(c=(1,1)/\sqrt2\), their physical forces are respectively 0 and \(-0.42\).
The full embedding/connection distinguishes these physical realizations;
raw \(H,S\) labels do not.

Instantaneous symmetric orthogonalization is not a solution to this missing
information. With \(F=B S^{-1/2}\), the orthogonalized Hamiltonian still lives
in a moving orthonormal frame \(F\) and requires its connection. A true fixed
frame can be recovered here only because the complete physical embedding
\(B\) is explicitly known.

## Numerical evidence and reproduction

Run from the repository with a **fresh output directory**:

```sh
python -m benchmarks.moving_frame_feedback --output outputs/moving_frame_proof_new
pytest tests/test_moving_frame_feedback.py
```

The command explicitly selects JAX x64 at entry; importing the benchmark does
not change global precision. The output contains the benchmark source, exact
parameter/initial-state arrays, raw moving and fixed-reference trajectories,
SHA-256 identities, dependency versions and a JSON report.

The recorded proof_1 report (local execution record; regenerate with the command above)
uses duration 4 in atomic time units and three timesteps:

| Check | dt 0.08 | dt 0.04 | dt 0.02 |
| --- | ---: | ---: | ---: |
| Moving-frame maximum coordinate error | 9.24e-7 | 5.75e-8 | 3.58e-9 |
| Regauged maximum coordinate error | 9.00e-7 | 5.60e-8 | 3.49e-9 |
| Native fixed-frame final maximum component error | 9.13e-4 | 2.28e-4 | 5.70e-5 |

The moving equations converge at fourth order under the test's full coupled
RK4 integration; the existing native Ehrenfest splitting converges at second
order to the same independent fixed-frame DOP853 reference. These are different
integrators, so the table is **not** a performance or method-quality comparison.
Tightening the independent reference changed its final components by at most
6.11e-16. At dt 0.02, the moving calculation's maximum metric-norm defect was
8.09e-11 and its total-energy drift was 1.21e-10.

The intentionally incomplete metric-only connection gives a metric-norm defect
of only 6.63e-11, yet a maximum coordinate error of 0.145, electronic component
error of 0.371, and energy drift of 0.0531. The negative control makes clear why
smooth Hamiltonians, overlap normalization and stable short trajectories do
not establish correct feedback.

Twelve focused tests additionally check generalized eigenvalues, full force
finite differences, instantaneous physical-state propagation, covariance,
singular/rectangular-frame rejection, and a degenerate Hamiltonian without
choosing eigenvectors or dividing by gaps.

## What remains before an AO feedback interface

1. **Physical spatial connection.** Supply orbital derivatives or independently
   meaningful nearby-geometry cross metrics with coherent orbital/spin ordering.
   One recorded temporal overlap constrains a directional combination of the
   spatial connections, not every spatial component needed for a force.
2. **A defined physical energy functional.** Declare neutral and carrier/excited
   reference energies and their complete derivatives. A learned one-particle
   Hamiltonian does not automatically define a many-electron charged-state
   total-energy surface or its self-consistent response.
3. **Changing physical subspaces.** Rectangular \(B\), discarded states and a
   moving AO span are excluded here. Their projection errors and additional
   geometric terms must be derived and tested, not replaced by a metric floor
   or a polar overlap repair.
4. **Numerical qualification.** Add conditioning limits, force/probe conventions,
   complete derivative labels, checkpoint representation and coupled-trajectory
   controls for the actual provider before exposing a native feedback contract.

## Complex and degenerate manifolds

Complex matrices and exact degeneracy present no algebraic singularity for the
matrix-based Ehrenfest equations in this proof. This does not establish SOC
material accuracy, spin-resolved transport, or a complex/degenerate MASH method.

For the complete smooth frame, the connection is locally pure gauge:

\[
\partial_i\Gamma_j-\partial_j\Gamma_i+[\Gamma_i,\Gamma_j]=0.
\]

The test verifies this cancellation with noncommuting coordinate generators.
It does **not** imply that Berry curvature of a physical band or projected
degenerate manifold vanishes. Projection changes that geometric problem.

Inside a degenerate manifold, arbitrary coordinate-dependent unitary rotations
change individual eigenvector populations. An active-state rule based on the
largest such population is not automatically invariant under those rotations.
A future manifold method needs projector/internal-density dynamics, geometric
forces, preparation and observables, and a derived transfer/momentum rule.
Simply averaging the forces within a block or allowing complex coefficient
arrays does not provide that derivation. Current real MASH implementations are
unchanged by this research benchmark.
