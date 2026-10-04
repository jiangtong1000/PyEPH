"""Exact thermal Green--Kubo preparation on independent prescribed CPA paths.

This workflow deliberately propagates the full electronic evolution matrix.
It is a small-system reference route, not a stochastic-trace approximation.
Per-trajectory initial densities and current insertions live in the state so
batching and checkpoint/restart never replace them with an ensemble average.
"""

from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
import numpy as np

from pyeph.core.contracts import ProbeContext
from pyeph.core.problem import Problem
from pyeph.core.state import make_state
from pyeph.dynamics.cpa import CPA
from pyeph.observables.transport.greenkubo import (
    current_correlation,
    legacy_current_to_physical,
    thermal_density_matrix,
)


def _velocity(problem, state):
    treatment = problem.nuclear_treatment
    if hasattr(treatment, "path") and callable(getattr(treatment.path, "velocity", None)):
        return treatment.path.velocity(state.time)
    if hasattr(treatment, "masses"):
        return state.p / jnp.asarray(treatment.masses)
    # A custom prescribed treatment may expose its own velocity operation.
    if callable(getattr(treatment, "velocity", None)):
        return treatment.velocity(state)
    return None


@dataclass(frozen=True)
class TransportMeasurement:
    """Correlate model-defined physical current components on CPA paths.

    By default currents come from ``model.probe_apply(params, context, name,
    identity)``.  An optional pure JAX ``probe_callback(params, context, name)``
    can supply the full matrix when a physical model has not implemented that
    method.  The callback must document its units, basis and current definition.
    ``current_convention='legacy_without_i'`` converts both initial and final
    imported matrices to physical currents exactly once on entry.
    """

    probes: tuple[str, ...] = ("current_x",)
    probe_callback: Any = None
    current_convention: str = "physical"

    def __post_init__(self):
        probes = tuple(self.probes)
        if not probes or len(set(probes)) != len(probes) or not all(isinstance(p, str) for p in probes):
            raise ValueError("probes must be a nonempty tuple of unique probe names")
        if self.current_convention not in {"physical", "legacy_without_i"}:
            raise ValueError("current_convention must be physical or legacy_without_i")
        if self.probe_callback is not None and not callable(self.probe_callback):
            raise TypeError("probe_callback must be callable")
        object.__setattr__(self, "probes", probes)

    def validate(self, problem):
        if not isinstance(problem.method, CPA) or not getattr(problem.nuclear_treatment, "prescribed", False):
            raise ValueError("thermal trace transport requires CPA on prescribed paths")
        if self.probe_callback is None:
            missing = set(self.probes) - set(problem.model.spec.probes)
            if missing:
                raise ValueError(f"model does not define required physical probes: {sorted(missing)}")

    def currents(self, problem, state):
        context = ProbeContext(state.q, _velocity(problem, state), state.time)
        identity = jnp.eye(problem.model.spec.system.nstates, dtype=state.electronic.dtype)
        matrices = []
        for probe in self.probes:
            if self.probe_callback is None:
                current = problem.model.probe_apply(problem.params, context, probe, identity)
            else:
                current = self.probe_callback(problem.params, context, probe)
            if self.current_convention == "legacy_without_i":
                current = legacy_current_to_physical(current)
            matrices.append(jnp.asarray(current))
        return jnp.stack(matrices)

    def initial_hamiltonian(self, problem, state):
        return problem.model.apply(problem.params, state.q, state.electronic)

    def validate_preparation_beta(self, beta):
        """Optional recipe-specific relation between preparation and bath temperature."""

    def validate_preparation_currents(self, currents):
        """Optional recipe-specific validation of the initial current operators."""

    def evaluate(self, problem, state):
        payload = _transport_payload(state, problem.model.spec.system.nstates, len(self.probes))
        if self.probe_callback is None and type(self).currents is TransportMeasurement.currents:
            # Tr[J(t) U J(0) rho0 U†] = <U, J(t)[U(J(0)rho0)]>_F.
            # Keep current probes as actions and associate the fixed origin
            # product first, so JAX can hoist it out of the trajectory scan.
            # No new checkpoint payload or approximation is introduced.
            context = ProbeContext(state.q, _velocity(problem, state), state.time)
            inserted = state.electronic @ (payload["currents0"] @ payload["rho0"])
            values = []
            for index, probe in enumerate(self.probes):
                acted = problem.model.probe_apply(problem.params, context, probe, inserted[index])
                if self.current_convention == "legacy_without_i":
                    acted = legacy_current_to_physical(acted)
                values.append(jnp.vdot(state.electronic, acted))
            correlation = jnp.stack(values)
        else:
            # Matrix callbacks and existing subclasses overriding currents()
            # retain their original interpretation and evaluation path.
            correlation = current_correlation(
                state.electronic, payload["rho0"], self.currents(problem, state), payload["currents0"]
            )
        return _observation(state, correlation)


def _transport_payload(state, nstates, nprobes):
    if state.electronic.shape != (nstates, nstates):
        raise ValueError("thermal transport requires a full (nstates, nstates) propagator")
    if not isinstance(state.method_state, dict) or "transport" not in state.method_state:
        raise ValueError("initialize thermal transport with initialize_transport_state")
    payload = state.method_state["transport"]
    if payload["rho0"].shape != (nstates, nstates) or payload["currents0"].shape != (nprobes, nstates, nstates):
        raise ValueError("initial transport payload does not match the model and probes")
    return payload


def _observation(state, correlation):
    u = state.electronic
    unitarity = jnp.max(jnp.abs(u.conj().T @ u - jnp.eye(u.shape[0], dtype=u.dtype)))
    return {"current_correlation": correlation, "unitary_error": unitarity}


def make_transport_problem(
    model, params, nuclear_treatment, *, probes=("current_x",), probe_callback=None,
    current_convention="physical",
):
    """Assemble and validate a native CPA current-correlation problem.

    Current output has a final probe axis in ``probes`` order.  For independent
    harmonic initial conditions use ``HarmonicBath``; a ``PrescribedPath``
    instead shares its configured path among trajectories.
    """
    measurement = TransportMeasurement(tuple(probes), probe_callback, current_convention)
    return Problem(model, params, nuclear_treatment, CPA(), measurement).validate()


def initialize_transport_state(
    problem, q, p, beta, *, time=0.0, trajectory_id=0, seed=0,
):
    """Prepare one thermal trajectory with ``U(0)=I`` and immutable insertions.

    ``q,p`` must follow the nuclear treatment's canonical coordinate convention.
    ``beta`` is inverse electronic energy and may be zero or positive infinity.
    Initialize trajectories separately and call ``stack_states`` to batch them.
    The saved ``time0`` makes an eventual restart retain the original response
    origin; restarting must pass the returned final state without reinitializing.
    """
    problem.validate()
    if not isinstance(problem.measurement, TransportMeasurement):
        raise TypeError("problem must use a TransportMeasurement")
    beta_array = np.asarray(beta)
    if beta_array.ndim != 0 or np.isnan(beta_array) or beta_array < 0:
        raise ValueError("beta must be one nonnegative inverse temperature")
    problem.measurement.validate_preparation_beta(beta)
    nstates = problem.model.spec.system.nstates
    state = make_state(q, p, jnp.eye(nstates, dtype=jnp.result_type(1j)),
                       time=time, trajectory_id=trajectory_id, seed=seed)
    if state.q.shape != problem.model.spec.system.q_shape:
        raise ValueError("initial coordinates do not match the model")
    prescribed_q = problem.nuclear_treatment.point(state, 0.0)[0]
    if not np.allclose(np.asarray(prescribed_q), np.asarray(state.q), rtol=1e-10, atol=1e-12):
        raise ValueError("initial coordinates must agree with the prescribed path at initial time")
    hamiltonian = problem.measurement.initial_hamiltonian(problem, state)
    _validate_hermitian(hamiltonian, (nstates, nstates), "initial Hamiltonian")
    currents = problem.measurement.currents(problem, state)
    _validate_hermitian(currents, (len(problem.measurement.probes), nstates, nstates), "initial currents")
    problem.measurement.validate_preparation_currents(currents)
    payload = {
        "rho0": thermal_density_matrix(hamiltonian, beta),
        "currents0": currents,
        "time0": state.time,
        "beta": jnp.asarray(beta, dtype=state.q.dtype),
    }
    return state._replace(method_state={"transport": payload})


def _validate_hermitian(matrix, shape, name):
    array = np.asarray(matrix)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite with shape {shape}")
    if not np.allclose(array, array.conj().swapaxes(-1, -2), rtol=1e-10, atol=1e-12):
        raise ValueError(f"{name} must use the physical Hermitian convention")


__all__ = ["TransportMeasurement", "make_transport_problem", "initialize_transport_state"]
