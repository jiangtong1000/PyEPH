"""Host-side trajectory preflight shared by execution and checkpoint entry points."""

import numpy as np

from pyeph.core.problem import PrescribedPath
from pyeph.core.state import TrajectoryState
from pyeph.core.validation import validate_model_at


def validate_run_span(treatment, integrator, state, steps):
    """Check a requested fixed span on the host, before tracing or publication.

    Callers own step-count normalization and state compatibility. This shared
    check covers counter/clock capacity, conservative clock resolution and a
    prescribed path's known domain; it does not certify integration accuracy
    or the model at future geometries.
    """
    counter = np.asarray(state.step)
    if counter.dtype.kind not in "iu" or not counter.size:
        raise ValueError("step counter must contain integers")
    if steps > np.iinfo(counter.dtype).max - int(np.max(counter)):
        raise ValueError("requested steps would overflow the state step counter")
    times = np.asarray(state.time)
    if steps and times.dtype not in (np.dtype("float32"), np.dtype("float64")):
        raise ValueError("nonzero propagation requires float32 or float64 time coordinates")
    with np.errstate(over="ignore", invalid="ignore"):
        end_times = times + steps * integrator.dt
    if np.iscomplexobj(end_times) or not np.isfinite(end_times).all():
        raise ValueError("requested propagation would overflow the time coordinate "
                         "or has a nonfinite/nonreal time")
    if steps:
        _validate_clock_resolution(treatment, integrator, times, end_times, steps)
    path = getattr(treatment, "path", None)
    if callable(getattr(path, "validate_span", None)):
        for start, stop in zip(times.reshape(-1), end_times.reshape(-1), strict=True):
            path.validate_span(float(start), float(stop))


def _validate_clock_resolution(treatment, integrator, times, end_times, steps):
    """Conservative, O(batch) policy for distinct scheduled physical times.

    Macro clocks are anchored, and CPA electronic substeps use bounded
    multiply/add expressions rather than a repeatedly accumulated clock. Four
    epsilons times the span's magnitude allow for those expressions. CPA needs
    distinct start/midpoint/end nodes; frozen-geometry coupled electronic steps
    use local clocks, so their absolute-time requirement is the macro interval.
    This policy also applies to prescribed treatments that internally use
    relative elapsed time; host preflight does not infer their implementation.
    """
    dt = integrator.dt
    separation = dt
    if getattr(treatment, "prescribed", False):
        separation = dt / integrator.electronic_substeps / 2
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        typed_dt = np.asarray(dt, dtype=times.dtype)
        separation = np.asarray(separation, dtype=times.dtype)
        first = times + typed_dt
        before_last = end_times - typed_dt
    # Work in float64 and scale terms separately: summing large unscaled time
    # coordinates can overflow although the rounding allowance remains finite.
    factor = 4 * float(np.finfo(times.dtype).eps)
    allowance = (factor * np.abs(times.astype(np.float64))
                 + factor * abs(steps * dt) + factor * abs(dt))
    unresolved = ((not np.isfinite(separation)) | (separation <= allowance)
                  | (first <= times) | (before_last >= end_times))
    if np.any(unresolved):
        index = int(np.flatnonzero(np.asarray(unresolved).reshape(-1))[0])
        origin = float(times.reshape(-1)[index])
        raise ValueError(
            f"requested propagation exceeds {times.dtype} clock resolution: "
            f"required time separation {float(separation):.17g} at origin {origin:.17g}; "
            "shift the time origin while preserving the prescribed path's physical time convention"
        )


def validate_state(problem, measurement, execution, state, *, checked):
    """Validate host inputs in evaluation order, before output or compilation."""
    if not isinstance(state, TrajectoryState):
        raise TypeError("geometry dynamics requires a TrajectoryState")
    batch = np.ndim(state.time) == 1
    if np.ndim(state.time) not in (0, 1):
        raise ValueError("state time must be scalar or a leading trajectory axis")
    shape = state.q.shape[1:] if batch else state.q.shape
    if shape != problem.model.spec.system.q_shape or state.p.shape != state.q.shape:
        raise ValueError("state coordinates do not match the model")
    if problem.geometry_guard is not None:
        if batch:
            raise ValueError("coordinate guards support scalar checked trajectories only")
        problem.geometry_guard.require(state.q)
    if any(np.iscomplexobj(x) for x in (state.q, state.p, state.time)):
        raise ValueError("nuclear coordinates, momenta and time must be real")
    axis = 1 if batch else 0
    if (np.ndim(state.electronic) not in (axis + 1, axis + 2) or
            state.electronic.shape[axis] != problem.model.spec.system.nstates):
        raise ValueError("state electronic dimension does not match the model")
    if checked:
        if any(getattr(state, name).dtype != np.dtype("float64")
               for name in ("q", "p", "time")):
            raise ValueError("checked propagation requires float64 coordinates, momenta and time; "
                             "enable JAX x64 before creating the state")
        if state.electronic.dtype != np.dtype("complex128"):
            raise ValueError("checked propagation requires complex128 electronic states; "
                             "enable JAX x64 before creating the state")
        if state.electronic.ndim == axis + 2 and not state.electronic.shape[-1]:
            raise ValueError("checked propagation requires at least one electronic column")
    counter_shape = np.shape(state.time)
    if batch and (state.q.shape[0] != state.time.size or
                  state.electronic.shape[0] != state.time.size):
        raise ValueError("state arrays must share the leading trajectory axis")
    for name in ("step", "trajectory_id"):
        counter = np.asarray(getattr(state, name))
        if (counter.shape != counter_shape or
                not np.issubdtype(counter.dtype, np.integer) or np.any(counter < 0)):
            raise ValueError(f"{name} must contain nonnegative integers matching the time shape")
    if np.any(np.asarray(state.trajectory_id, dtype=np.uint64) >= 2**32):
        raise ValueError("trajectory IDs must be representable as uint32")
    if np.shape(state.key) != counter_shape + (2,) or np.asarray(state.key).dtype != np.uint32:
        raise ValueError("random keys must be uint32 pairs matching the time shape")
    for x in (state.q, state.p, state.electronic, state.time):
        if not np.isfinite(np.asarray(x)).all():
            raise ValueError("initial state contains nonfinite values")
    if isinstance(problem.nuclear_treatment, PrescribedPath):
        _validate_prescribed_state(problem.nuclear_treatment, state, batch=batch)
    validate_nuclei = getattr(problem.nuclear_treatment, "validate_initial_state", None)
    if callable(validate_nuclei):
        validate_nuclei(state, batch=batch)
    validate_model_at(problem.model, problem.params, state.q, batch=batch)
    if execution.verify_external_gradients:
        verify = getattr(problem.model, "validate_complete_gradients", None)
        if not callable(verify):
            raise ValueError("this model does not provide the requested derivative preflight")
        verify(problem.params, state.q[0] if batch else state.q)
    if batch:
        if state.time.size == 0:
            raise ValueError("empty trajectory batches are unsupported")
        if np.unique(np.asarray(state.trajectory_id)).size != state.time.size:
            raise ValueError("batch trajectory IDs must be unique")
        if np.unique(np.asarray(state.step)).size != 1:
            raise ValueError("this runner requires synchronized step counters in a batch")
    if callable(getattr(problem.method, "validate_state", None)):
        problem.method.validate_state(state, batch=batch)
    if callable(getattr(problem.method, "validate_initial_state", None)):
        problem.method.validate_initial_state(problem, state, batch=batch)
    # Workflow-owned origins and insertions remain meaningful even when
    # observations are disabled. Check them once on the host before any
    # propagation, initial output, or checkpoint acceptance.
    if callable(getattr(measurement, "validate_initial_state", None)):
        measurement.validate_initial_state(problem, state, batch=batch)
    return batch


def _validate_prescribed_state(treatment, state, *, batch):
    if not np.issubdtype(state.q.dtype, np.floating):
        raise ValueError("prescribed path state coordinates must have a real floating dtype")
    coordinates = np.asarray(state.q)
    times = np.asarray(state.time)
    samples = zip(coordinates, times) if batch else ((coordinates, times),)
    epsilon = np.finfo(state.q.dtype).eps
    for q, time in samples:
        expected = np.asarray(treatment.path.position(time))
        velocity = np.asarray(treatment.path.velocity(time))
        if any(x.shape != q.shape or np.iscomplexobj(x) or not np.isfinite(x).all()
               for x in (expected, velocity)):
            raise ValueError("prescribed path position and velocity must be finite real "
                             "arrays matching the state coordinate shape")
        # Adjacent floating-point clock expressions can differ by an
        # ULP at a resumed endpoint. Convert that clock uncertainty to
        # coordinate uncertainty using the path's physical velocity.
        scale = np.maximum(1., np.maximum(np.abs(q), np.abs(expected)))
        tolerance = 32 * epsilon * (scale + np.abs(velocity)*max(1., abs(float(time))))
        if np.any(np.abs(q - expected) > tolerance):
            raise ValueError("initial coordinates must agree with the prescribed path "
                             "at each trajectory's initial time")
