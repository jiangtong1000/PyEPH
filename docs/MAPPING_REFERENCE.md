# Independent real three-state mapping reference

`benchmarks/mapping_reference.py` supplies a public numerical oracle for the
implemented real finite-state Runeson–Manolopoulos mapping dynamics. It uses
fresh analytic fixtures and SciPy DOP853. Its reference path does not call the
package's diagonalizer, force derivatives, event locator or impulse routines.
The native path is the system under test and uses the ordinary `Simulation`
interface, including its production forces, electronic propagator and events.

The underlying dynamics is described by Runeson, Drayton and Manolopoulos,
[Charge transport in organic semiconductors from the mapping approach to
surface hopping, J. Chem. Phys. 161, 144102 (2024)](https://doi.org/10.1063/5.0226001)
([open manuscript](https://arxiv.org/abs/2406.19851)). This benchmark implements
an independent numerical reference for those equations; it does not propose a
new mapping method or reproduce a material calculation from that work.

## Explicit physical model

All quantities use atomic units. There are two classical canonical coordinates
\(q=(x,y)\), momenta \(p\), and masses \(M=(1.3,0.8)\). The carrier occupies a
three-dimensional fixed orthonormal basis. The model is an illustrative
parameterization, with no fitted or validated material interpretation.

Define real plane rotations by the block
\(R_{ab}(\gamma)_{ab}=\begin{pmatrix}\cos\gamma&-\sin\gamma\\
\sin\gamma&\cos\gamma\end{pmatrix}\), with the other direction unchanged:

\[
U(q)=R_{01}(\theta)R_{12}(\phi),\qquad
\theta=0.7x+0.17\sin y,\quad \phi=0.3y+0.11x,
\]
\[
\epsilon(q)=(-0.3+0.04x,\;0.1-0.02x,\;0.65+0.03y),\qquad
h(q)=U(q)\operatorname{diag}(\epsilon(q))U(q)^T,
\]
\[
V_{\rm ref}(q)=\tfrac12(0.04x^2+0.048y^2).
\]

The sampled domain has three distinct ordered eigenvalues. The scalar reference
participates in forces and total energy; its removable global electronic phase
is omitted, consistently with the native convention. All changing model
parameters, including a constant electronic basis rotation, are explicit data.

The normalized fixed-basis mapping vector \(c\) obeys
\(\dot c=-ih(q)c\). With \(P_a=u_au_a^T\), the active surface has the largest
raw mapping population \(c^\dagger P_ac\). Between events,

\[
\dot q_i=p_i/M_i,\qquad
\dot p_i=-\partial_iV_{\rm ref}-\partial_i\epsilon_a.
\]

These raw mapping populations select events. They are not the RM population
estimator. The fixtures supply individual deterministic vectors; this work
does not test conditional-sphere sampling, population correlations or ensemble
equilibrium.

## Independent event and impulse construction

An incoming pair boundary is a downward zero of
\(g_{ab}=c^\dagger(P_a-P_b)c\). Analytic derivatives of the declared rotations
give

\[
\partial_iP_a=(\partial_i u_a)u_a^T+u_a(\partial_i u_a)^T,
\quad
\delta_i=\tfrac12c^\dagger(\partial_iP_a-\partial_iP_b)c,
\quad
\dot g_{ab}=2\sum_i\delta_i p_i/M_i.
\]

The commutator contribution vanishes because the spectral projectors commute
with \(h\). The partial derivative holds **fixed-basis \(c\) fixed**. All three
components contribute, including the spectator. The tests remove the spectator
without renormalization as a negative control; the resulting impulse direction
changes by more than 0.03 in these fixtures.

Let \(v=p/\sqrt M\), \(n=(\delta/\sqrt M)/|\delta/\sqrt M|\),
\(v_\parallel=n\cdot v\), and \(\Delta=\epsilon_b-\epsilon_a\). The reference
changes only the normal component of mass-weighted momentum:

\[
v_\parallel'=\begin{cases}
\operatorname{sgn}(v_\parallel)\sqrt{v_\parallel^2-2\Delta},
  &v_\parallel^2>2\Delta\quad\text{(accepted)},\\
-v_\parallel,&v_\parallel^2<2\Delta\quad\text{(frustrated)},
\end{cases}
\quad
p'=\sqrt M\,[v+(v_\parallel'-v_\parallel)n].
\]

An accepted event selects \(b\); a frustrated event retains \(a\).
Coordinates and electronic coefficients are unchanged at the impulse.
The tangential momentum and \(T+V_{\rm ref}+\epsilon_{\rm active}\) are
conserved. The reference rejects grazing, exact-threshold and simultaneous
largest-population ties instead of assigning them an unvalidated prescription.

Both fixtures declare a boundary at \(q=(-0.23,0.31)\), with adiabatic mapping
populations \((0.47,0.47,0.06)\) and phases \((0,0.27,-0.55)\). Incoming active
surface 0 competes with surface 1. The normal kinetic energy is respectively
1.8 or 0.25 times the positive energy gap; the orthogonal mass-weighted
momentum is 0.2. The continuous active-0 equations are integrated backward by
0.173 to construct an initial state strictly inside its active region. Forward
integration then reaches the declared event at \(t=0.173\), away from all tested
output-grid points. Both the declaration and the actual initial arrays are
saved. These values were chosen directly from the analytic model.

DOP853 integrates the full coupled equations, stops at its terminal population
root, applies the independent impulse and restarts with the outgoing active
surface. Dense output supplies samples on the native output grid. Neither path
renormalizes the mapping vector. A tighter tolerance and smaller maximum step
check the reference itself.

## Checks and reproduction

Run from the repository root with the project's test dependencies installed:

```sh
python -m pytest -q tests/test_mapping_reference.py
python -m benchmarks.mapping_reference --output outputs/mapping-reference-new-run
```

The command explicitly selects JAX double precision; importing the module
does not change global precision. It refuses an existing result directory.
The output records the exact initial arrays, parameter arrays, complete native
and reference traces, event rows, the benchmark source, runtime source snapshots
and SHA-256 hashes, plus NumPy/SciPy/JAX/Python versions. A runtime source change
during the run invalidates the qualification rather than silently mixing it.

The tests check:

- The analytic eigensystem against the independently written native matrix;
  analytic projector derivatives against coordinate finite differences.
- Analytic active-surface forces and full spectator-dependent impulse directions
  against the native force contractions.
- Accepted and frustrated events, unmodified coordinates/electronic state,
  mass-weighted tangential momentum, impulse energy and outgoing direction.
- Full native trajectory convergence at timesteps 0.08, 0.04 and 0.02, with two
  event subdivisions, against the independently integrated coupled trajectory.
- Mapping norm, total-energy drift and event outcomes throughout both runs.
- Covariance of the independent and native trajectories under a nontrivial
  constant real orthogonal electronic basis transformation, including impulses.

The recorded 0.8-time-unit run used the three timesteps above. Its largest
coordinate errors over the sampled trajectory were:

| Event | dt = 0.08 | dt = 0.04 | dt = 0.02 |
| --- | ---: | ---: | ---: |
| Accepted | 5.46e-6 | 1.39e-6 | 3.52e-7 |
| Frustrated | 3.00e-6 | 7.59e-7 | 1.90e-7 |

At dt = 0.02, the largest electronic-component errors were 7.71e-7 and 1.24e-7,
and the largest total-energy drifts were 8.25e-9 and 1.03e-8, respectively.
Mapping norm defects remained below 5.4e-15 across the timestep study. A constant
real basis rotation changed the complete native trajectory by at most 6.7e-15
after rotating the electronic coefficients back. Tightening the independent
reference changed its sampled trajectory by at most 1.4e-15. These are observed
errors for the declared fixtures, not universal accuracy bounds or performance
claims. The source and input hashes in a saved report identify its exact run.

This covers isolated real pair events with unequal masses and a participating
spectator. It does not establish complex/SOC dynamics, degenerate manifolds,
arbitrarily fast unresolved recrossings, adaptive state truncation, canonical RM
equilibrium or agreement with exact quantum dynamics. Event-subdivision and
timestep studies remain necessary for a new physical calculation. The code is
a small qualification fixture, not a general-purpose reference solver API.
