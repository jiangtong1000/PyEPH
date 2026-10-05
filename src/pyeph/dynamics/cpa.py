"""Electronic evolution along an imposed nuclear trajectory (classical path approximation)."""

from dataclasses import dataclass

import jax

from pyeph.dynamics.checked import (
    PHASE_CPA,
    PHASE_ENDPOINT,
    check_finite,
    check_geometry,
    electronic_action,
    finish_checked_step,
    initial_checked_info,
    record_action,
    record_geometry,
    validate_checked_model,
)
from pyeph.integrators.electronic import propagate


@dataclass(frozen=True)
class CPA:
    def validate(self, problem):
        if not getattr(problem.nuclear_treatment, "prescribed", False):
            raise ValueError("CPA requires a prescribed nuclear path")
        if not callable(getattr(problem.nuclear_treatment, "point", None)):
            raise TypeError("a prescribed nuclear treatment must provide point(state, elapsed)")

    def build_step(self, problem, integrator):
        if problem.geometry_guard is not None:
            raise ValueError("coordinate guards require scalar checked Lanczos propagation")
        model, params = problem.model, problem.params
        treatment = problem.nuclear_treatment

        def step(state):
            def apply(t, x):
                return model.apply(params, treatment.point(state, t - state.time)[0], x)
            electronic = propagate(
                apply, state.time, state.electronic, integrator.dt,
                algorithm=integrator.electronic, substeps=integrator.electronic_substeps,
            )
            time = state.time + integrator.dt
            q, p = treatment.point(state, integrator.dt)
            return state._replace(q=q, p=p,
                                  electronic=electronic, time=time, step=state.step + 1)

        return step

    def build_checked_step(self, problem, integrator, *, batch=False):
        """Frozen-midpoint actions with a transactional macrostep budget.

        This controls electronic action approximation, not the error from the
        time-dependent midpoint rule. Batched trajectories accept or roll back
        together; a failed action prevents later path and model stages.
        """
        self.validate(problem)
        validate_checked_model(problem, integrator)
        model, params = problem.model, problem.params
        guard = problem.geometry_guard
        if guard is not None and batch:
            raise ValueError("coordinate guards support scalar checked trajectories only")
        treatment = problem.nuclear_treatment
        options, dt = integrator.electronic, integrator.dt
        substeps = integrator.electronic_substeps
        duration = dt / substeps
        point = jax.vmap(treatment.point, in_axes=(0, None)) if batch else treatment.point

        def step(state):
            info = initial_checked_info(state, options, batch=batch, guard=guard)
            allocation = info.macrostep_budget / substeps

            def body(index, carry):
                def advance(carry):
                    electronic, diagnostic = carry
                    q, p = point(state, (index + 0.5) * duration)
                    diagnostic = record_geometry(diagnostic, q,
                        state.time + (index + 0.5) * duration)
                    diagnostic = check_finite(diagnostic, q, p, phase=PHASE_CPA, batch=batch)
                    diagnostic = check_geometry(diagnostic, guard, q,
                        state.time + (index + 0.5) * duration, phase=PHASE_CPA, substep=index)

                    def propagate(carry):
                        electronic, diagnostic = carry
                        action = electronic_action(model, params, q, electronic, duration,
                                                   options, allocation, batch=batch)
                        return action.value, record_action(
                            diagnostic, action, phase=PHASE_CPA, substep=index, batch=batch)

                    return jax.lax.cond(diagnostic.code == 0, propagate, lambda x: x,
                                        (electronic, diagnostic))

                return jax.lax.cond(carry[1].code == 0, advance, lambda x: x, carry)

            electronic, info = jax.lax.fori_loop(0, substeps, body, (state.electronic, info))

            def endpoint(_):
                q, p = point(state, dt)
                diagnostic = record_geometry(info, q, state.time + dt)
                diagnostic = check_finite(diagnostic, q, p, phase=PHASE_ENDPOINT, batch=batch)
                diagnostic = check_geometry(diagnostic, guard, q, state.time + dt,
                                            phase=PHASE_ENDPOINT)
                return state._replace(q=q, p=p, electronic=electronic,
                                      time=state.time + dt, step=state.step + 1), diagnostic

            candidate, info = jax.lax.cond(info.code == 0, endpoint, lambda _: (state, info),
                                            operand=None)
            return finish_checked_step(state, candidate, info, batch=batch)

        return step
