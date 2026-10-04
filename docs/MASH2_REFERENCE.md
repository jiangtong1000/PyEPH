# Independent event-driven MASH reference

The [reference script](../benchmarks/mash2_reference.py) compares complete
production `MASH2` trajectories against separately written NumPy/SciPy equations
with adaptive integration and event roots. Three deterministic trajectories cover
a nonlinear rotating electronic basis, a varying gap, repeated allowed hops,
repeated frustrated hops, and a Tully avoided crossing. Historical local CPU
records contain event times, jump momenta, energy residuals, reference
self-refinement and production convergence. Those execution archives are not
included in the distribution; regenerate records with the reference driver:

```sh
PYTHONPATH=src .venv/bin/python benchmarks/mash2_reference.py \
  --output outputs/mash2_reference_rerun.json
PYTHONPATH=src .venv/bin/python -m pytest tests/test_mash2_reference.py -q
```

This establishes numerical agreement with the specified **classical MASH
equations** on small examples. It does not establish quantum accuracy, reproduce
published scattering/population curves, or validate multistate MASH. The initial
spins are selected diagnostic points, not a weighted population ensemble. See
[MASH2.md](MASH2.md) for the implemented population sampler, estimator and primary
scientific references.

## Independent equations and event handling

The reference uses one nuclear coordinate and real traceless two-state matrices

\[
h(q)=\begin{pmatrix}z(q)&v(q)\\v(q)&-z(q)\end{pmatrix},
\quad \rho(q)=\sqrt{z^2+v^2},\quad E_\pm=\pm\rho.
\]

With active-surface sign \(s=-1,+1\), each smooth segment obeys

\[
\dot q=p/m,\qquad
\dot p=-V_{\rm ref}'-s\frac{zz'+vv'}{\rho},\qquad
\dot c=-i h(q)c.
\]

The complex vector represents the mapping spin in the fixed diabatic basis.
The gauge-independent equator coordinate is evaluated without instantaneous
eigenvectors:

\[
S_z=c^\dagger\frac{h}{\rho}c,\qquad
\dot S_z=\frac{p}{m}c^\dagger\partial_q\!\left(\frac{h}{\rho}\right)c.
\]

The electronic commutator in this derivative vanishes because \(h\) commutes
with \(h/\rho\). The active hemisphere has \(sS_z\ge0\); its outward crossing
has negative time derivative. SciPy's terminal event uses that direction.

At a crossing, \(\Delta E=-2s\rho\). In one dimension a nonzero NAC spans the
entire nuclear momentum direction, so an allowed impulse is independently
implemented as

\[
p' = \operatorname{sign}(p)\sqrt{p^2-2m\Delta E}.
\]

If the radicand is negative, the reference reflects \(p'=-p\) and retains the
surface. It leaves the fixed-diabatic mapping vector continuous. Every jump logs
the incoming/outgoing signed equator rates and the energy balance; unresolved
grazing crossings are rejected. The analytic fixtures have nonzero gaps and
nonzero derivative coupling at their events.

The reference calls **no production step, surface-force, derivative-coupling,
root-search or impulse routine**. Its Hamiltonian entries, entry derivatives,
reference potential and surface forces are explicit NumPy formulas. The
production comparison uses the public MASH state constructor for state fields,
then supplies the independently constructed initial mapping vector. Its analytic
initial eigenvector gauge is fixed by `atan2(v,z)` so the reference does not
depend on an eigensolver's arbitrary column signs.

SciPy DOP853 integrates the six real variables
\((q,p,\operatorname{Re}c_1,\operatorname{Re}c_2,
\operatorname{Im}c_1,\operatorname{Im}c_2)\). At an event, the reference
restarts on the outgoing branch. It checks that branch's signed derivative is
positive and suppresses only the zero at **exactly** the segment's initial time;
it does not advance time by a finite epsilon. An initially equatorial trajectory
gets an immediate impulse only when its specified side is incoming. A separate
test verifies that this initial event occurs once and that an already outgoing
trajectory receives no spurious impulse.
If an event falls exactly at the requested final time, both the final state and
the last sampled row use the outgoing surface and momentum. An endpoint test
checks this convention around neighboring floating-point stop times.

The rotating reference uses `rtol=2e-11`, `atol=2e-13`, and maximum step 0.05.
The Tully reference uses `rtol=2e-12`, `atol=2e-14`, and maximum step 1. Each
case is recomputed with both tolerance values divided by ten and its maximum
step halved. The stricter Tully settings resolve its change in Hamiltonian
curvature at q=0 before the crossing. On the tested Linux CPU, the original
Tully settings (`2e-11`, `2e-13`, maximum step 2) changed the event time by
**2.44e-8** under refinement, exceeding the unchanged **1e-8** self-audit gate.
The new baseline/audit pair differs by **5.66e-11**. Further independent
tolerance/step refinements down to `rtol=3e-14`, `atol=3e-16`, and maximum step
0.125 agree with the new audit event time within **8.87e-12**, and with its final
state within **7.39e-13**. These changes affect only the independent oracle.

The historical local CPU baseline used the original Tully defaults. Across its
three cases, the maximum final-state change was **1.62e-10**, maximum event-time
change **1.37e-9**, maximum sampled energy drift **7.85e-13**, and maximum
mapping-norm error **2.00e-15**. All trajectory numbers and tables below retain
that historical configuration. Their production errors are substantially
larger than those reference-resolution estimates. Adaptive event detection
remains numerical; refinement provides evidence for these resolved cases and
does not prove detection of every possible paired crossing in an arbitrary
model.

## Nonlinear and Tully cases

The confined fixture is

\[
\theta(q)=1.3q+0.35\sin q,\quad
\rho(q)=0.2+0.035\cos q,\quad
h=\rho(\cos\theta\,\sigma_z+\sin\theta\,\sigma_x),\quad
V_{\rm ref}=0.125q^2.
\]

It starts at \(q=-1\), mass 1, and runs to time 25. Both diagnostic spins start
on the lower surface with \(S_z=-0.4\) and azimuth zero. With initial momentum
1.6 it has **nine events: eight accepted and one frustrated**. With momentum
0.45 it has **five events, all frustrated**. For example, the mixed trajectory's
frustrated event occurs at time **10.0913485053**, with position **−3.2290862305**
and momentum **−0.309349521 → +0.309349521**. Its gap is **0.330267759**, so the
available kinetic energy cannot pay for the attempted upward hop.

The third case independently implements Tully model 1 with
\(a=0.01,b=1.6,c=0.005,d=1\), initial \(q=-4,p=20\), mass 2000, and final time
1000. It uses the same selected lower-hemisphere spin. Its one accepted crossing
occurs at time **479.254910706**, position **0.7549995183**, and momentum
**19.75624494 → 18.16104927**. These are single classical mapping trajectories;
they are not a nuclear wavepacket or a quantum scattering experiment.

Production runs use `fixed_diabatic_pc_midpoint`, one event subdivision per
outer step and event tolerance `1e-11`. This is the production velocity-Verlet /
fixed-basis midpoint-exponential method, whereas the independent reference uses
adaptive ODE integration. No production smooth-step map is reused in its oracle.

| Case | Outer dt values | Final-state errors at those dt values | Observed orders |
|---|---|---|---|
| Nonlinear, accepted + frustrated | 0.1 / 0.05 / 0.025 | 8.706e-3 / 2.238e-3 / 5.680e-4 | 1.960 / 1.978 |
| Nonlinear, all frustrated | 0.1 / 0.05 / 0.025 | 1.109e-3 / 2.714e-4 / 6.807e-5 | 2.030 / 1.995 |
| Tully 1, accepted | 2 / 1 / 0.5 | 2.586e-5 / 6.488e-6 / 1.625e-6 | 1.995 / 1.998 |

The error is the maximum difference among the six real state components in
this fixture's units and fixed gauge. It is a numerical regression metric, not
a universal physical error norm. Production and reference agree on every final
accepted/frustrated count, final surface and sampled active surface. Every
reference event lies inside the outer-step bracket inferred from production
event counters. The production state records counters and equator residuals,
not exact event timestamps; bracket agreement must not be presented as an
independent direct comparison of localized production event times.

Full sampled q/p histories also converge. The finest-step maximum position
errors are **6.03e-4**, **1.68e-4**, and **4.34e-7** respectively. The JSON reports
momentum errors both on all sample points and with a two-step neighborhood of
reference events omitted. That distinction matters when an impulse's small
time displacement puts two solutions on different sides of a momentum jump;
no such active-side mismatch occurred in this selected grid.

Maximum sampled energy drift decreases from **8.15e-4 to 5.08e-5** in the mixed
case, **1.03e-4 to 6.40e-6** in the all-frustrated case, and **1.28e-6 to 8.09e-8**
in Tully. Maximum production mapping-norm error is **5.91e-14** and maximum
impulse energy residual **2.23e-16** across these runs. All statuses are zero.

## Localization and subdivision checks

At fixed outer dt 0.05 in the mixed case, changing event tolerance independently
of the smooth-step resolution gives:

| Event tolerance | Maximum equator residual | Final-state change relative to tolerance 1e-11 |
|---|---:|---:|
| 1e-5 | 8.66e-6 | 3.58e-5 |
| 1e-8 | 8.33e-9 | 1.62e-7 |

Both runs retain all nine events. Tight localization does not eliminate the
remaining smooth-propagation error: dt refinement is still necessary. An outer
step 0.1 with four event subdivisions reproduces the final state of outer step
0.025 with one subdivision to the checked precision in this fixture. This checks
the subdivision plumbing; it does not guarantee detection of arbitrary hidden
recrossings or resolve an insufficient per-step event capacity.

The historical local CPU report was generated with JAX 0.11.2, SciPy 1.18.1 and
x64. No performance comparison is made. General multidimensional impulse
directions, unequal masses, complex/SOC Hamiltonians, conical degeneracies,
moving bases, and multistate extensions remain outside this independent oracle;
other unit tests address some of those implementation components separately.
