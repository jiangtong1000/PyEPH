"""Runeson–Manolopoulos preparation and one-time observable algebra.

This is the real multistate RM formulation's mapping convention, including for
N=2; it is not the original MASH2 weighted-spin sampler/active-state estimator.
No dynamics, equilibrium sampling, VACF, or correlation estimator is supplied.

The preparation is uniform on the normalized complex sphere, conditional on a
chosen component having the largest squared magnitude (2023 paper, Sec. III.E,
Eq. 37). Its probability measure is the author's ``theta`` sampler. The pinned
author code also offers ``focused`` sampling, and defaults to it, but that fixes
the magnitudes and is a different measure. Matching the same first density
moment alone does not make a preparation equivalent for coupled trajectories.

Instead of rejection, we draw one uniform sphere vector and swap its largest
component into the requested position. Each original largest-component sector
maps bijectively to the target sector by a coordinate permutation; permutations
preserve sphere measure. Summing the N identical sector contributions therefore
gives exactly the same conditional measure in continuous arithmetic. Ties have
zero measure; floating-point argmax uses JAX's usual first-index convention.

Writing H_N=sum_{k=1}^N 1/k, the conditional mean largest population is H_N/N;
each other population has mean (N-H_N)/(N*(N-1)). Uniform phases give zero mean
coherences. Thus E[alpha*c*c† + b*I] = |population><population| with the constants
below. For any prescribed common unitary evolution this moment evolves exactly
to U*rho*U† in expectation. This says nothing about exact coupled nuclear motion.

References: https://arxiv.org/pdf/2305.08835 (v3, Eqs. 29, 34–37 and Appendix C);
author model.py lines 518–540 at commit 41c64fc4a6a35d23329f773afe42975734c63111.
"""

import math

import jax
import jax.numpy as jnp

from pyeph.core._configuration import integer_scalar


def rm_coefficients(nstates):
    """Return Python scalars ``(alpha_N, b_N)`` for a static integer N >= 2."""
    nstates = integer_scalar(nstates, "nstates")
    if nstates < 2:
        raise ValueError("RM mapping requires at least two electronic states")
    # Sum H_N-1 directly to avoid subtracting nearly equal numbers.
    alpha = (nstates-1)/math.fsum(1.0/k for k in range(2, nstates+1))
    return alpha, (1.0-alpha)/nstates


def sample_population_conditional(key, nstates, *, population=0):
    """Draw one RM conditional-sphere vector, shape ``(nstates,)``.

    ``population`` names the prepared population in the caller's chosen basis;
    it need not be the eventual active adiabatic surface. Both integer arguments
    are static when jitted. The result has complex dtype corresponding to JAX's
    current default real precision; this function does not set global precision.

    The supplied key is used purely and is not advanced or derived from a seed.
    Use explicit per-trajectory keys with ``jax.vmap`` to obtain ``(batch,N)``
    samples; keeping those keys fixed preserves preparation across partitions.
    A caller can rotate the returned vector by its declared basis matrix.
    There is no focused sampler or generic density/coherence preparation here.
    """
    nstates = integer_scalar(nstates, "nstates")
    population = integer_scalar(population, "population")
    if nstates < 2:
        raise ValueError("RM mapping requires at least two electronic states")
    if not 0 <= population < nstates:
        raise ValueError("population must be an electronic index in [0,nstates)")
    normals = jax.random.normal(key, (2, nstates), dtype=jnp.result_type(1.0))
    c = normals[0]+1j*normals[1]
    largest = jnp.argmax(normals[0]**2+normals[1]**2)
    permutation = jnp.arange(nstates).at[largest].set(population).at[population].set(largest)
    return c[permutation]/jnp.sqrt(jnp.sum(normals**2))


def _vector(c):
    c = jnp.asarray(c)
    if c.ndim != 1 or c.shape[0] < 2:
        raise ValueError("RM mapping amplitudes must be one vector with N>=2; use vmap for batches")
    return c


def mapping_populations(c):
    """One-time RM population estimators ``alpha*abs(c)**2+b``, shape ``(N,)``.

    Values may be negative. They are neither squared amplitudes nor active-state
    indicators. Inputs are not normalized silently; the caller supplies a finite
    unit vector and dynamics/state validation owns that invariant.
    """
    c = _vector(c)
    alpha, b = rm_coefficients(c.shape[0])
    return alpha*jnp.abs(c)**2+b


def mapping_density(c):
    """RM density-matrix estimator ``alpha*c*c†+b*I``, shape ``(N,N)``.

    This optional dense diagnostic need not be formed to measure populations or
    operator actions. Its trace is one for unit c, but an individual estimator
    is not a positive density matrix. Element ``[n,m]`` estimates rho_nm; the
    mapped observable ``|n><m|`` is therefore element ``[m,n]``.
    """
    c = _vector(c)
    alpha, b = rm_coefficients(c.shape[0])
    return alpha*jnp.outer(c, c.conj())+b*jnp.eye(c.shape[0], dtype=c.dtype)


def mapping_observable(c, operator_c, operator_trace):
    """One-time estimator ``alpha*c†(O c)+b*Tr(O)`` from an operator action.

    ``operator_c`` has shape ``(N,)`` and ``operator_trace`` is scalar, both in
    the same basis as c. No dense operator is required. The result can be complex
    for coherence operators; no imaginary part is discarded. A product of these
    one-time estimators is not generally a valid dynamical correlation recipe.
    """
    c = _vector(c)
    operator_c, operator_trace = jnp.asarray(operator_c), jnp.asarray(operator_trace)
    if operator_c.shape != c.shape or operator_trace.ndim != 0:
        raise ValueError("operator action must match the mapping vector and its trace must be scalar")
    alpha, b = rm_coefficients(c.shape[0])
    return alpha*jnp.vdot(c, operator_c)+b*operator_trace
