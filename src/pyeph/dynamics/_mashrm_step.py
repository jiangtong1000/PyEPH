"""Private smooth propagation and isolated-pair events for real RM MASH.

The immutable context binds one step builder's model, parameters, and numerical
options. Trajectory state and diagnostics remain explicit function arguments;
there is no persistent trajectory state or spectrum cache here. This is the RM
formulation only: original two-state MASH uses different mapping preparation.
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from pyeph.integrators.electronic import exponential_action
from pyeph.integrators.events import BracketedCrossing, bisect_bracketed_crossing
from pyeph.integrators.impulses import energy_conserving_impulse
from pyeph.representations.adiabatic import diagonalize, surface_force
from pyeph.representations.mapping_rm import rm_impulse_direction


def build_step(method, problem, integrator):
    """Bind one RM step without retaining parameters across runtime builds."""
    n = problem.model.spec.system.nstates
    kernel = _MASHRMStep(method, problem.model, problem.params,
                        jnp.asarray(problem.nuclear_treatment.masses), n,
                        jnp.arange(n), integrator.dt,
                        integrator.dt/method.event_substeps)
    return kernel.step


@dataclass(frozen=True, eq=False)
class _MASHRMStep:
    """Scientific operations used by the bounded RM event controller."""

    method: object
    model: object
    params: object
    mass: object
    nstates: int
    indices: object
    dt: float
    width: float

    def surfaces_at(self, q):
        h = self.model.apply(self.params, q, jnp.eye(self.nstates, dtype=q.dtype))
        data = diagonalize(h, gap_tolerance=self.method.gap_tolerance)
        valid = ((jnp.max(jnp.abs(jnp.imag(h))) <= self.method.real_tolerance)
                 & jnp.all(jnp.isfinite(data.energies)) & ~jnp.any(data.near_degenerate)
                 & jnp.isfinite(self.model.reference_energy(self.params, q)))
        return h, data, valid

    def populations(self, data, c):
        return jnp.abs(data.vectors.conj().T @ c)**2

    def mark(self, state, code):
        return state._replace(method_state={**state.method_state, "status": jnp.int32(code)})

    def smooth(self, state, duration):
        """Velocity Verlet on one surface with a midpoint mapping exponential."""
        active = state.method_state["active"]
        _, start, valid_start = self.surfaces_at(state.q)
        f0 = surface_force(self.model, self.params, state.q, start, active)
        half = state.p + duration/2*f0.value
        q = state.q + duration*half/self.mass
        _, end, valid_end = self.surfaces_at(q)
        f1 = surface_force(self.model, self.params, q, end, active)
        p = half + duration/2*f1.value
        hmid, _, valid_mid = self.surfaces_at((state.q+q)/2)
        c = exponential_action(hmid, state.electronic, duration)
        valid = (valid_start & valid_end & valid_mid & f0.valid & f1.valid
                 & jnp.all(jnp.isfinite(q)) & jnp.all(jnp.isfinite(p))
                 & jnp.all(jnp.isfinite(c)))
        return state._replace(q=q, p=p, electronic=c, time=state.time+duration), end, valid

    def direction(self, state, data, competitor):
        result = rm_impulse_direction(self.model, self.params, state.q, data, state.electronic,
            state.method_state["active"], competitor, gap_tolerance=self.method.gap_tolerance)
        rate = 2*jnp.sum(result.value*state.p/self.mass)
        return result, rate

    def record_attempt(self, start, code, roots):
        old = start.method_state
        attempts = old["event_attempts"].astype(jnp.int64)+1
        iterations = (old["localization_iterations"].astype(jnp.int64)
                      + jnp.sum(roots.iterations, dtype=jnp.int64))
        overflow = ((attempts > np.iinfo(np.int32).max)
                    | (iterations > np.iinfo(np.int32).max))
        code = jnp.where(overflow, 7, code)
        residual = jnp.max(jnp.where(jnp.isfinite(roots.residual), abs(roots.residual), 0.))
        bracket = jnp.max(jnp.where(jnp.isfinite(roots.upper-roots.lower),
                                   roots.upper-roots.lower, 0.))
        return self.mark(start, code)._replace(method_state={**old, "status": jnp.int32(code),
            "event_attempts": jnp.where(overflow, old["event_attempts"], attempts).astype(jnp.int32),
            "localization_iterations": jnp.where(overflow, old["localization_iterations"],
                                                 iterations).astype(jnp.int32),
            "max_event_residual": jnp.maximum(old["max_event_residual"], residual),
            "max_event_bracket_width": jnp.maximum(old["max_event_bracket_width"], bracket)})

    def inspect_impulse(self, event, data, competitor):
        """Prospective impulse, without changing state or counting an event."""
        active = event.method_state["active"]
        pops = self.populations(data, event.electronic)
        pair_floor = jnp.minimum(pops[active], pops[competitor])
        unique = jnp.all(jnp.where((self.indices == active) | (self.indices == competitor), True,
                                  pair_floor-pops > self.method.event_tolerance))
        delta, rate = self.direction(event, data, competitor)
        impulse = energy_conserving_impulse(event.p, self.mass, delta.value,
            data.energies[competitor]-data.energies[active],
            direction_tolerance=self.method.direction_tolerance)
        gap = data.energies[competitor]-data.energies[active]
        kinetic_margin = impulse.parallel_before**2-2*gap
        threshold_uncertainty = 64*jnp.finfo(event.q.dtype).eps*jnp.maximum(
            impulse.parallel_before**2, 2*abs(gap))
        at_threshold = (gap > 0) & (abs(kinetic_margin) <= threshold_uncertainty)
        outgoing = 2*jnp.sum(delta.value*impulse.momentum/self.mass)
        outgoing = jnp.where(impulse.accepted, -outgoing, outgoing)
        impulse_valid = (delta.valid & impulse.valid & ~at_threshold
                         & jnp.isfinite(rate) & jnp.isfinite(outgoing)
                         & jnp.all(jnp.isfinite(impulse.momentum))
                         & jnp.isfinite(impulse.energy_error) & (rate < -self.method.rate_tolerance)
                         & (outgoing > self.method.rate_tolerance))
        margin = pops[active]-pops[competitor]
        post_margin = jnp.where(impulse.accepted, -margin, margin)
        # This is the same outgoing ownership policy used at interval
        # entry. A slower accepted impulse can make an otherwise localized
        # negative post-margin inadmissible; no amplitude projection fixes it.
        admissible = ((post_margin >= 0)
                      | ((post_margin >= -self.method.event_tolerance)
                         & (abs(post_margin) <= outgoing*self.method.event_time_tolerance)))
        return impulse, unique, impulse_valid, admissible, margin, rate

    def positive_shortcut(self, event, data, competitor):
        """A still-positive margin is optional, never a genuine crossing."""
        _, unique, valid, admissible, margin, rate = self.inspect_impulse(event, data, competitor)
        # The caller establishes strict positivity. Repeating that sign
        # decision after a fused reevaluation can differ by an ulp at an
        # exact boundary; all actual residual/time/impulse bounds still apply.
        return (unique & valid & admissible
                & (abs(margin) <= self.method.event_tolerance)
                & (abs(margin) <= -rate*self.method.event_time_tolerance))

    def apply_event(self, start, event, data, competitor, roots, valid, count, remaining,
                    expected_acceptance=None, selection_valid=True):
        """Commit a valid pair impulse or retain the incoming state with diagnostics."""
        active = start.method_state["active"]
        impulse, unique, impulse_valid, admissible, margin, _ = self.inspect_impulse(
            event, data, competitor)
        if expected_acceptance is not None:
            impulse_valid = impulse_valid & (impulse.accepted == expected_acceptance)
        impulse_valid = impulse_valid & admissible & selection_valid
        valid = valid & (abs(margin) <= self.method.event_tolerance)
        code = jnp.select((~valid, ~unique, ~impulse_valid), (3, 6, 4), default=0)
        attempt = self.record_attempt(start, code, roots)
        # The attempts >= events invariant also guards every event counter:
        # no accepted/frustrated increment can overflow before attempts do.
        code = attempt.method_state["status"]
        old = attempt.method_state
        diagnostics = {**old,
            "active": jnp.where(impulse.accepted, competitor, active).astype(jnp.int32),
            "events": old["events"]+1,
            "accepted": old["accepted"]+impulse.accepted.astype(jnp.int32),
            "frustrated": old["frustrated"]+(~impulse.accepted).astype(jnp.int32),
            "max_impulse_energy_error": jnp.maximum(old["max_impulse_energy_error"],
                                                     abs(impulse.energy_error))}
        updated = event._replace(p=impulse.momentum, method_state=diagnostics)
        return jax.lax.cond(code == 0,
            lambda: (updated, remaining, count+1),
            lambda: (attempt, jnp.zeros_like(remaining), count))

    def locate(self, start, duration, candidate, terminal, endpoint_margins, count):
        """Localize all candidates, require isolation, then select the outgoing side."""
        active = start.method_state["active"]

        def one(competitor):
            def value_at(t):
                trial, data, ok = self.smooth(start, t)
                pops = self.populations(data, trial.electronic)
                margin = pops[active]-pops[competitor]
                # Only a previously validated outgoing boundary can have a
                # tiny negative initial margin. Do not rediscover t=0.
                margin = jnp.where(t == 0, jnp.maximum(margin, 0.), margin)
                # Treat an incoming endpoint already within both the
                # population and local time tolerance as right-continuous.
                # Its actual residual is retained below; q,c are unchanged.
                margin = jnp.where((t == duration) & terminal[competitor], 0., margin)
                return jnp.where(ok, margin, jnp.nan)

            root = jax.lax.cond(candidate[competitor],
                lambda: bisect_bracketed_crossing(value_at, duration,
                    iterations=self.method.bisection_iterations,
                    population_tolerance=self.method.event_tolerance,
                    time_tolerance=self.method.event_time_tolerance, both_endpoints=True),
                lambda: BracketedCrossing(jnp.asarray(jnp.inf, duration.dtype),
                    jnp.zeros_like(duration), jnp.asarray(jnp.inf, duration.dtype),
                    jnp.asarray(jnp.inf, duration.dtype), jnp.int32(0), jnp.bool_(True)))
            return root._replace(residual=jnp.where(terminal[competitor],
                endpoint_margins[competitor], root.residual))

        # lax.map retains scalar conditional cancellation for inactive pairs.
        roots = jax.lax.map(one, self.indices)
        competitor = jnp.argmin(roots.time).astype(jnp.int32)
        isolated = jnp.all(jnp.where(candidate & (self.indices != competitor),
            roots.lower > roots.upper[competitor]+self.method.event_time_tolerance, True))
        resolved = jnp.all(roots.converged) & isolated
        event, data, valid = self.smooth(start, roots.time[competitor])

        def resolved_event():
            provisional, _, provisional_valid, _, _, _ = self.inspect_impulse(event, data, competitor)
            # Select a side owned by the outgoing active surface. Both
            # endpoints satisfy the residual bound; competition continues
            # to use the complete original brackets, not these point times.
            selected_time = jnp.where(provisional.accepted,
                roots.upper[competitor], roots.lower[competitor])
            selected, selected_data, selected_valid = self.smooth(start, selected_time)
            selected_pops = self.populations(selected_data, selected.electronic)
            residual = selected_pops[active]-selected_pops[competitor]
            selected_roots = roots._replace(
                time=roots.time.at[competitor].set(selected_time),
                residual=roots.residual.at[competitor].set(residual))
            return self.apply_event(start, selected, selected_data, competitor, selected_roots,
                valid & selected_valid, count, jnp.maximum(0., duration-selected_time),
                expected_acceptance=provisional.accepted, selection_valid=provisional_valid)

        return jax.lax.cond(resolved, resolved_event,
            lambda: (self.record_attempt(start, jnp.where(isolated, 3, 6), roots),
                     jnp.zeros_like(duration), count))

    def exhausted(self, state, remaining, count):
        attempts = state.method_state["event_attempts"]
        overflow = attempts == np.iinfo(np.int32).max
        failed = self.mark(state, jnp.where(overflow, 7, 5))
        failed = failed._replace(method_state={**failed.method_state,
            "event_attempts": jnp.where(overflow, attempts, attempts+1)})
        return failed, jnp.zeros_like(remaining), count

    def advance(self, s, left, events):
        """Propagate a smooth trial interval and locate any competing pair events."""
        active = s.method_state["active"]
        trial, end, good = self.smooth(s, left)
        end_pops = self.populations(end, trial.electronic)
        end_margins = end_pops[active]-end_pops

        def terminal_pair(b):
            def classify():
                # A positive terminal residual is only a shortcut.
                # If its prospective impulse is invalid or leaves
                # unresolved outgoing ownership, retain the current
                # surface until an actual sign bracket is reached.
                return self.positive_shortcut(trial, end, b)

            return jax.lax.cond((b != active) & (end_margins[b] > 0)
                & (end_margins[b] <= self.method.event_tolerance), classify,
                lambda: jnp.bool_(False))

        terminal = jax.lax.map(terminal_pair, self.indices)
        candidate = (self.indices != active) & ((end_margins <= 0.) | terminal)
        return jax.lax.cond(good,
            lambda: jax.lax.cond(jnp.any(candidate),
                lambda: jax.lax.cond(events < self.method.max_events_per_step,
                    lambda: self.locate(s, left, candidate, terminal, end_margins, events),
                    lambda: self.exhausted(s, left, events)),
                lambda: (trial, jnp.zeros_like(left), events)),
            lambda: (self.mark(s, 1), jnp.zeros_like(left), events))

    def boundary(self, s, left, events, data, competitor, margins):
        """Classify an interval starting on a single population boundary."""
        delta, rate = self.direction(s, data, competitor)
        close_time = (abs(margins[competitor]) <=
                      abs(rate)*self.method.event_time_tolerance)
        incoming = rate < -self.method.rate_tolerance
        invalid = (~delta.valid | ~jnp.isfinite(rate) | (abs(rate) <= self.method.rate_tolerance)
                   | ((margins[competitor] < 0) & ~close_time))

        def instant():
            zero = jnp.zeros_like(left)
            root = BracketedCrossing(zero, margins[competitor], zero, zero,
                                   jnp.int32(0), jnp.bool_(True))
            return jax.lax.cond(events < self.method.max_events_per_step,
                lambda: self.apply_event(s, s, data, competitor, root, jnp.bool_(True),
                                    events, left),
                lambda: self.exhausted(s, left, events))

        instant_ok = incoming & close_time & jax.lax.cond(
            margins[competitor] > 0,
            lambda: self.positive_shortcut(s, data, competitor),
            lambda: jnp.bool_(True))

        return jax.lax.cond(invalid,
            lambda: (self.mark(s, 4), jnp.zeros_like(left), events),
            lambda: jax.lax.cond(instant_ok, instant,
                lambda: self.advance(s, left, events)))

    def advance_interval(self, carry):
        """Check surface ownership before taking an event or a smooth segment."""
        s, left, events = carry
        active = s.method_state["active"]
        _, data, valid = self.surfaces_at(s.q)
        pops = self.populations(data, s.electronic)
        margins = pops[active]-pops
        near = (self.indices != active) & (abs(margins) <= self.method.event_tolerance)
        near_count = jnp.sum(near)
        competitor = jnp.argmax(near).astype(jnp.int32)
        consistent = jnp.all(margins >= -self.method.event_tolerance)
        code = jnp.select((~valid, ~consistent, near_count > 1), (1, 2, 6), default=0)

        return jax.lax.cond(code == 0,
            lambda: jax.lax.cond(near_count == 1,
                lambda: self.boundary(s, left, events, data, competitor, margins),
                lambda: self.advance(s, left, events)),
            lambda: (self.mark(s, code), jnp.zeros_like(left), events))

    def interval(self, state, count):
        """Consume one subdivision while retaining the last valid local segment."""
        def condition(carry):
            s, left, _ = carry
            return (left > 0) & (s.method_state["status"] == 0)

        result, _, count = jax.lax.while_loop(condition, self.advance_interval,
            (state, jnp.asarray(self.width, state.q.dtype), count))
        return result, count

    def step(self, state):
        """Advance the macrostep clock only if every event subdivision succeeds."""
        original_time, original_step = state.time, state.step

        def body(_, carry):
            s, count = carry
            return jax.lax.cond(s.method_state["status"] == 0,
                lambda: self.interval(s, count), lambda: (s, count))

        state, _ = jax.lax.fori_loop(0, self.method.event_substeps, body, (state, jnp.int32(0)))
        success = state.method_state["status"] == 0
        return state._replace(time=jnp.where(success, original_time+self.dt, state.time),
                              step=jnp.where(success, original_step+1, state.step))
