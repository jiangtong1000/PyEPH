"""Mean-field feedback with a reversible electronic/nuclear splitting."""

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from pyeph.core.contracts import prepared_action, pure_state_weight
from pyeph.core.problem import CoupledClassical
from pyeph.dynamics.checked import (
    PHASE_EHRENFEST_FIRST,
    PHASE_EHRENFEST_SECOND,
    PHASE_FORCE_FIRST,
    PHASE_FORCE_SECOND,
    check_finite,
    finish_checked_step,
    fixed_geometry_actions,
    initial_checked_info,
    validate_checked_model,
)
from pyeph.integrators.electronic import propagate


def _build_force(model, params):
    """Complete mean-field force shared by ordinary and checked stepping."""
    def force(q, electronic):
        return -model.reference_gradient(params, q) - model.contract_gradient(
            params, q, pure_state_weight(electronic))

    return force


@dataclass(frozen=True)
class Ehrenfest:
    """Pure electronic-state Ehrenfest; statistical mixtures use separate trajectories.

    The electronic half steps bracket a velocity-Verlet nuclear step with the
    intermediate electronic state held fixed. Dense exponential electronic steps
    make this symmetric and norm preserving; RK4 provides a matrix-free option.
    This is an order-two splitting, not an assertion of exact energy conservation.
    """

    def validate(self, problem):
        if not isinstance(problem.nuclear_treatment, CoupledClassical):
            raise ValueError("Ehrenfest requires coupled canonical classical nuclei")
        if not problem.model.spec.force_support:
            raise ValueError("Ehrenfest requires complete reference and electronic derivatives")
        for name in ("reference_gradient", "contract_gradient"):
            if not callable(getattr(problem.model, name, None)):
                raise TypeError(f"force model is missing {name}")

    def validate_state(self, state, *, batch=False):
        import numpy as np

        expected_ndim = 2 if batch else 1
        if state.electronic.ndim != expected_ndim:
            raise ValueError("this Ehrenfest method requires one pure electronic vector per trajectory")
        norm = np.sum(np.abs(np.asarray(state.electronic))**2, axis=-1)
        if not np.allclose(norm, 1.0, atol=1e-8, rtol=1e-8):
            raise ValueError("Ehrenfest initial electronic states must be normalized")

    def build_step(self, problem, integrator):
        model, params = problem.model, problem.params
        mass = jnp.asarray(problem.nuclear_treatment.masses)
        dt = integrator.dt

        def electronic_step(q, state, duration):
            action = prepared_action(model, params, q)
            return propagate(lambda t, x: action(x), 0.0, state, duration,
                             algorithm=integrator.electronic,
                             substeps=integrator.electronic_substeps)

        force = _build_force(model, params)

        def step(state):
            c = electronic_step(state.q, state.electronic, dt / 2)
            p_half = state.p + dt / 2 * force(state.q, c)
            q_new = state.q + dt * p_half / mass
            p_new = p_half + dt / 2 * force(q_new, c)
            c_new = electronic_step(q_new, c, dt / 2)
            return state._replace(q=q_new, p=p_new, electronic=c_new,
                                  time=state.time + dt, step=state.step + 1)

        return step

    def build_checked_step(self, problem, integrator, *, batch=False):
        """Checked electronic halves with scalar gates before nuclear stages.

        The two halves share the entry state's fixed action budget. This is
        still the same order-two nuclear/electronic splitting; the action budget
        does not bound errors in the force feedback or nuclear trajectory.
        """
        self.validate(problem)
        validate_checked_model(problem, integrator)
        model, params = problem.model, problem.params
        mass = jnp.asarray(problem.nuclear_treatment.masses)
        dt, options = integrator.dt, integrator.electronic
        substeps = integrator.electronic_substeps

        force = _build_force(model, params)
        if batch:
            force = jax.vmap(force)

        def half(q, electronic, info, phase):
            return fixed_geometry_actions(model, params, q, electronic, info, dt / 2,
                                          options, substeps, phase=phase, batch=batch)

        def step(state):
            expected_ndim = 2 if batch else 1
            if state.electronic.ndim != expected_ndim:
                raise ValueError("checked Ehrenfest requires one pure electronic vector per trajectory")
            info = initial_checked_info(state, options, batch=batch)
            c, info = half(state.q, state.electronic, info, PHASE_EHRENFEST_FIRST)

            def nuclear_stage(candidate, diagnostic, *, phase, drift):
                """Half kick, optionally followed by drift, with force/update gates."""
                def evaluate(carry):
                    current, checked = carry
                    f = force(current.q, c)
                    checked = check_finite(checked, f, phase=phase, batch=batch)

                    def update(_):
                        p = current.p + dt / 2 * f
                        if drift:
                            q = current.q + dt * p / mass
                            updated = current._replace(q=q, p=p, electronic=c)
                            return updated, check_finite(checked, p, q, phase=phase, batch=batch)
                        return current._replace(p=p), check_finite(
                            checked, p, phase=phase, batch=batch)

                    return jax.lax.cond(checked.code == 0, update,
                                        lambda _: (current, checked), operand=None)

                return jax.lax.cond(diagnostic.code == 0, evaluate, lambda x: x,
                                    (candidate, diagnostic))

            candidate, info = nuclear_stage(state, info, phase=PHASE_FORCE_FIRST, drift=True)
            candidate, info = nuclear_stage(candidate, info, phase=PHASE_FORCE_SECOND, drift=False)

            def second_half(carry):
                candidate, diagnostic = carry
                c_new, diagnostic = half(candidate.q, c, diagnostic, PHASE_EHRENFEST_SECOND)
                return candidate._replace(electronic=c_new, time=state.time + dt,
                                          step=state.step + 1), diagnostic

            candidate, info = jax.lax.cond(info.code == 0, second_half, lambda x: x,
                                          (candidate, info))
            return finish_checked_step(state, candidate, info, batch=batch)

        return step


def total_energy(model, params, state, masses):
    kinetic = jnp.sum(state.p**2 / (2 * jnp.asarray(masses)))
    electronic = jnp.real(jnp.vdot(state.electronic, model.apply(params, state.q, state.electronic)))
    return kinetic + model.reference_energy(params, state.q) + electronic
