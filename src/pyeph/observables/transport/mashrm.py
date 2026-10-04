"""RM velocity algebra and validated one-time velocity measurements.

The 2024 RM symmetrized VACF uses E[v_M(0)*v_M(t)] with its own conditional
fourth-moment normalization. It is not a product of generic RM one-time
observable estimators. This module does not prepare equilibrium or accumulate
correlations. Reference: https://arxiv.org/html/2406.19851v2, Eqs. 30--33 and
Appendix A. The static-H identity does not establish exact coupled dynamics.
"""

from collections.abc import Mapping
from dataclasses import dataclass
import math
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from pyeph.core._configuration import boolean_scalar, integer_scalar, real_scalar
from pyeph.core.contracts import ProbeContext
from pyeph.dynamics.mashrm import MASHRM
from pyeph.representations.adiabatic import diagonalize


def rm_gamma(nstates):
    """Conditional moment E[|c_a|^2 |c_b|^2 | a is largest], b != a.

    The conditional measure is uniform on a sector of the complex unit sphere,
    not a focused preparation or the unconditional sphere. No extra 1/N enters
    this expression. Return a host float for the static number of states N>=2.
    """
    nstates = integer_scalar(nstates, "nstates")
    if nstates < 2:
        raise ValueError("RM velocity requires at least two electronic states")
    harmonic = math.fsum(1./k for k in range(1, nstates+1))
    square_sum = math.fsum(1./(k*k) for k in range(1, nstates+1))
    return ((nstates+1)*harmonic-harmonic*harmonic-square_sum)/(
        (nstates+1)*nstates*(nstates-1))


def rm_velocity(c, active_vector, velocity_c):
    """Action-only real RM velocity for one normalized fixed-basis vector.

    ``active_vector`` is the active adiabatic eigenvector and ``velocity_c`` is
    v@c in the same fixed basis. This algebra assumes a Hermitian velocity and
    is basis/eigenvector-phase invariant. The correlation prescription further
    requires a zero adiabatic diagonal and the conditional-sphere measure.
    No matrix, normalization, charge conversion or physical validity check is
    hidden here; RMVelocity owns the small-system observation validation.
    """
    c, active_vector, velocity_c = map(jnp.asarray, (c, active_vector, velocity_c))
    if c.ndim != 1 or c.shape[0] < 2 or active_vector.shape != c.shape or velocity_c.shape != c.shape:
        raise ValueError("RM velocity expects three equally sized vectors with N>=2")
    coefficient = jnp.vdot(active_vector, c)
    return jnp.sqrt(2./rm_gamma(c.shape[0]))*jnp.real(
        jnp.conj(coefficient)*jnp.vdot(active_vector, velocity_c))


@dataclass(frozen=True)
class FixedPositionVelocity:
    """Named finite fixed-position commutators v=i[h,X], with hbar=1.

    Construct with ``FixedPositionVelocity({'velocity': positions})`` or a
    sequence of (name, operator) pairs. Each position operator is a diagonal
    vector or a full Hermitian matrix. Arrays are copied into immutable JAX
    storage. The result has physical velocity units when h and X use the
    model's consistent energy/length units; there is no charge factor.

    This is a finite-position commutator. It does not define a periodic image
    velocity or add moving-center convective current. Such probes need an
    explicit model/provider and their own physical interpretation.
    """

    operators: Any

    def __post_init__(self):
        items = tuple(self.operators.items()) if isinstance(self.operators, Mapping) else tuple(self.operators)
        if not items or any(not isinstance(item, (tuple, list)) or len(item) != 2 for item in items):
            raise ValueError("position operators must be nonempty (name, operator) pairs")
        names = tuple(item[0] for item in items)
        if any(not isinstance(name, str) or not name for name in names) or len(set(names)) != len(names):
            raise ValueError("position operator names must be unique nonempty strings")
        stored = []
        for name, value in items:
            array = np.asarray(value)
            if (array.ndim not in (1, 2) or array.shape[0] < 2
                    or (array.ndim == 2 and array.shape[0] != array.shape[1])
                    or array.dtype.kind not in "iufc" or not np.isfinite(array).all()):
                raise ValueError("each position must be a finite diagonal vector or square matrix")
            if array.ndim == 1:
                if np.iscomplexobj(array) and np.any(array.imag != 0):
                    raise ValueError("a diagonal position operator must be real")
                array = np.real(array)
            elif not np.allclose(array, array.conj().T, atol=1e-12, rtol=1e-12):
                raise ValueError("a full position operator must be Hermitian")
            if (not jax.config.x64_enabled
                    and ((array.dtype.kind == "f" and array.dtype.itemsize > 4)
                         or (array.dtype.kind == "c" and array.dtype.itemsize > 8)
                         or (array.dtype.kind in "iu" and array.dtype.itemsize > 4))):
                raise ValueError("enable JAX_ENABLE_X64=1 before constructing double-precision positions")
            stored.append((name, jnp.array(array, copy=True)))
        object.__setattr__(self, "operators", tuple(stored))

    @property
    def probes(self):
        return tuple(name for name, _ in self.operators)

    def __call__(self, model, params, context, probe, vectors):
        candidates = [operator for name, operator in self.operators if name == probe]
        if not candidates:
            raise ValueError(f"no fixed position operator for probe {probe!r}")
        position = candidates[0]
        vectors = jnp.asarray(vectors)
        n = model.spec.system.nstates
        if position.shape[0] != n or vectors.ndim not in (1, 2) or vectors.shape[0] != n:
            raise ValueError("position operator and vectors must match the model state count")

        def apply_position(value):
            if position.ndim == 1:
                return position*value if value.ndim == 1 else position[:, None]*value
            return position @ value

        return 1j*(model.apply(params, context.q, apply_position(vectors))
                   - apply_position(model.apply(params, context.q, vectors)))


_VELOCITY_MESSAGES = {
    1: "nonfinite physical velocity operator",
    2: "physical velocity operator is not Hermitian",
    3: "physical velocity has a nonzero adiabatic diagonal",
    4: "invalid, nonreal, or nonisolated RM Hamiltonian",
    5: "invalid mapping vector or active surface",
}


@dataclass(frozen=True)
class RMVelocity:
    """Validated RM velocity samples, one output axis per named probe.

    An optional pure action callback has signature
    ``(model, params, ProbeContext, probe, vectors) -> velocity_vectors``.
    Otherwise ``model.probe_apply`` is used. Its result must represent physical
    velocity, with consistent model units and hbar=1. Probe names never imply
    charge conversion. A charge-current callback produces charge-scaled data
    and is the caller's responsibility; use an explicit conversion provider.

    Complete small spectra and dense probe validation are intentional here.
    Each observation checks finiteness, Hermiticity and zero adiabatic diagonal.
    Relative-plus-absolute bounds are tolerance*(1+max(abs(operator))). Invalid
    probes produce a velocity_status code and NaN velocity. The host Runner
    calls validate_observations before publication; evaluate does not change
    the physical trajectory or its method status.
    """

    probes: tuple[str, ...] = ("velocity",)
    probe_callback: Any = None
    hermiticity_tolerance: float = 1e-10
    diagonal_tolerance: float = 1e-10
    gap_tolerance: float = 1e-10
    real_tolerance: float = 1e-12
    norm_tolerance: float = 1e-8
    include_nuclei: bool = False
    supports_mashrm = True

    def __post_init__(self):
        probes = tuple(self.probes)
        if (not probes or any(not isinstance(probe, str) or not probe for probe in probes)
                or len(set(probes)) != len(probes)):
            raise ValueError("probes must be unique nonempty names")
        object.__setattr__(self, "probes", probes)
        if self.probe_callback is not None and not callable(self.probe_callback):
            raise TypeError("probe_callback must be an operator-action callable")
        for name in ("hermiticity_tolerance", "diagonal_tolerance", "gap_tolerance",
                     "real_tolerance", "norm_tolerance"):
            value = real_scalar(getattr(self, name), name)
            if value <= 0:
                raise ValueError(f"{name} must be finite and positive")
            object.__setattr__(self, name, value)
        object.__setattr__(self, "include_nuclei", boolean_scalar(self.include_nuclei, "include_nuclei"))

    def validate(self, problem):
        if not isinstance(problem.method, MASHRM):
            raise ValueError("RMVelocity requires MASHRM")
        if self.probe_callback is None:
            missing = set(self.probes)-set(problem.model.spec.probes)
            if missing:
                raise ValueError(f"model does not define physical velocity probes: {sorted(missing)}")
        elif isinstance(self.probe_callback, FixedPositionVelocity):
            if set(self.probes)-set(self.probe_callback.probes):
                raise ValueError("fixed position provider is missing a requested probe")
            if any(operator.shape[0] != problem.model.spec.system.nstates
                   for _, operator in self.probe_callback.operators):
                raise ValueError("fixed position operators must match the model state count")

    def evaluate(self, problem, state):
        model, params = problem.model, problem.params
        c = state.electronic
        n = model.spec.system.nstates
        identity = jnp.eye(n, dtype=c.dtype)
        h = model.apply(params, state.q, identity)
        data = diagonalize(h, gap_tolerance=self.gap_tolerance)
        spectrum_ok = (jnp.all(jnp.isfinite(data.energies))
                       & jnp.all(jnp.isfinite(data.vectors)) & ~jnp.any(data.near_degenerate)
                       & (jnp.max(abs(jnp.imag(h))) <= self.real_tolerance))
        active = state.method_state["active"]
        safe_active = jnp.clip(active, 0, n-1)
        norm = jnp.vdot(c, c).real
        populations = abs(data.vectors.conj().T @ c)**2
        state_ok = (jnp.all(jnp.isfinite(c)) & (abs(norm-1.) <= self.norm_tolerance)
                    & (active >= 0) & (active < n)
                    & (jnp.max(populations)-populations[safe_active]
                       <= problem.method.event_tolerance))
        context = ProbeContext(state.q, state.p/jnp.asarray(problem.nuclear_treatment.masses), state.time)
        velocities, statuses = [], []
        for probe in self.probes:
            if self.probe_callback is None:
                matrix = model.probe_apply(params, context, probe, identity)
            else:
                matrix = self.probe_callback(model, params, context, probe, identity)
            matrix = jnp.asarray(matrix)
            if matrix.shape != (n, n):
                raise ValueError("physical velocity action must preserve the electronic column-block shape")
            finite = jnp.all(jnp.isfinite(matrix))
            scale = 1+jnp.max(abs(matrix))
            hermitian = jnp.max(abs(matrix-matrix.conj().T)) <= self.hermiticity_tolerance*scale
            adiabatic_diagonal = jnp.sum(data.vectors.conj()*(matrix @ data.vectors), axis=0)
            offdiagonal = jnp.max(abs(adiabatic_diagonal)) <= self.diagonal_tolerance*scale
            status = jnp.select((~spectrum_ok, ~state_ok, ~finite, ~hermitian, ~offdiagonal),
                                (4, 5, 1, 2, 3), default=0).astype(jnp.int32)
            value = rm_velocity(c, data.vectors[:, safe_active], matrix @ c)
            velocities.append(jnp.where(status == 0, value, jnp.nan))
            statuses.append(status)
        energy = (jnp.sum(state.p**2/(2*jnp.asarray(problem.nuclear_treatment.masses)))
                  + model.reference_energy(params, state.q)+data.energies[safe_active])
        result = dict(velocity=jnp.stack(velocities), velocity_status=jnp.stack(statuses),
                      energy=energy, mapping_norm=norm, **state.method_state)
        if self.include_nuclei:
            result.update(q=state.q, p=state.p)
        return result

    def validate_observations(self, values):
        """Reject bad probes/data on the host, before observers publish a chunk."""
        required = {"velocity", "velocity_status", "energy", "mapping_norm", "active"}
        if not isinstance(values, dict) or not required <= values.keys():
            raise ValueError("RM velocity observations are missing required fields")
        velocity, status = np.asarray(values["velocity"]), np.asarray(values["velocity_status"])
        if (velocity.ndim < 1 or velocity.shape[-1] != len(self.probes)
                or status.shape != velocity.shape or not np.issubdtype(status.dtype, np.integer)):
            raise ValueError("invalid RM velocity/status observation shapes or dtypes")
        if np.any(status):
            messages = "; ".join(f"{int(code)}: {_VELOCITY_MESSAGES.get(int(code), 'unknown velocity failure')}"
                                 for code in np.unique(status[status != 0]))
            raise ValueError(f"RM velocity validation failed ({messages})")
        for name, raw in values.items():
            value = np.asarray(raw)
            if value.dtype.kind not in "biuf" or not np.isfinite(value).all():
                raise ValueError(f"RM velocity observation {name} must be finite and real")
        for name in ("energy", "mapping_norm", "active"):
            if np.asarray(values[name]).shape != velocity.shape[:-1]:
                raise ValueError(f"RM velocity observation {name} has an inconsistent trajectory shape")
        active = np.asarray(values["active"])
        if not np.issubdtype(active.dtype, np.integer) or np.any(active < 0):
            raise ValueError("RM velocity observation active must be a nonnegative integer")
        if "status" in values and np.any(values["status"]):
            raise ValueError("RM velocity observations contain failed dynamics states")
        if np.any(abs(np.asarray(values["mapping_norm"])-1.) > self.norm_tolerance):
            raise ValueError("RM velocity observation has a nonunit mapping norm")
