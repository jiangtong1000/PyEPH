# Checked electronic propagation integration

**Status: implemented, 2026-10-03.** CPA and Ehrenfest now have an optional
checked Hermitian Lanczos path, with transactional failure handling in Runner.
The contracts below cover numerical action acceptance and transactional
state publication. Platform results belong to [qualification](QUALIFICATION.md).

The existing array-only `propagate`, ordinary method steps, MASH2 path, physical
state and checkpoint schema retain their interfaces. Numerical action diagnostics
live in a transient scan carry and an exception, outside user-owned
`TrajectoryState.method_state`. The implementation is fixed-step and fail-closed;
it does not automatically enlarge the Krylov space or retry a rejected step.

## 1. Configuration: one small solver option object

Existing calls remain unchanged:

```python
Integrator(dt=.01)
Integrator(dt=.01, electronic="exponential_midpoint", electronic_substeps=2)
```

The new call is:

```python
Integrator(dt=.01,
           electronic=LanczosOptions(max_dimension=32, atol=1e-12, rtol=1e-10),
           electronic_substeps=2)
```

`LanczosOptions` is the numerical module's existing frozen native dataclass,
reused directly rather than wrapped in another option class. It contains the algorithm-specific
configuration, including breakdown, orthogonality and projected-Hermiticity
tolerances. Its scalar inputs are normalized with the existing static-configuration
helpers. `Integrator.electronic` accepts the two existing strings or this one
option type. This avoids adding six solver-specific fields to `Integrator` or
introducing an arbitrary solver plugin registry.

An `Integrator.electronic_name` property returns the existing string unchanged,
or `"lanczos_midpoint"`, which is the `RunResult.metadata` label; the metadata
field remains a string. Native dataclass provenance already encodes nested
options, so their actual tolerances/capacity enter strict run identity. The
option object's `atol/rtol` refer to a macrostep's electronic action budget,
defined below, not to the entire trajectory's physical error.

Before initial observations, Runner rejects a checked option for a method without
`build_checked_step`. The implemented scope is native JAX CPA/Ehrenfest with
**float64 q, p and time, and complex128 electronic arrays**. Both Runner preflight
and direct checked hooks reject unsupported input precision rather than silently
converting it. The numerical action assumes a pure linear Hermitian operator.
External Torch/host-callback providers are explicitly unsupported by this checked
route; their existing string-integrator route is unchanged. MASH2 retains its
dense-exponential requirement. `RecordedCPA` explicitly rejects the new option,
and the legacy Green–Kubo constructor continues selecting its existing string.

## 2. One additional fixed-shape diagnostic record

The production action returns candidate values plus diagnostics, retaining
the prototype's separated components. The total is named **`error_estimate`**:
the truncation/recurrence terms have conditional mathematical bounds, but the
roundoff allowance is heuristic. Success is an approximation-control decision,
not a certified floating-point enclosure or a whole-dynamics accuracy claim.

`dynamics.checked` reuses the numerical module's `LanczosResult` and adds one
checked-step record, with numerical array leaves:

```python
CheckedStepInfo(
    code, phase, substep,                 # scalar int32 control/context
    failed_trajectories,                  # shape (B,), or scalar if unbatched
    action,                              # existing LanczosResult, last action
    accumulated_error_estimate,           # sum across attempted action stages
    macrostep_budget,
)
```

`CheckedStepInfo.code` has four method-level meanings: `0` success, `1` rejected
electronic action, `2` nonfinite nuclear endpoint/force/candidate, `3` accumulated
action estimate exceeds the original macrostep budget. Preserve the
action's more detailed status inside `action.status`; a force failure has its own
method status rather than a fictitious Krylov breakdown. `phase` identifies CPA action, first Ehrenfest
half, first force, second force, second half or endpoint validation. `substep` is
zero-based within a segment, with `-1` for an empty record or a non-action failure.
The phase codes are `0` initial, `1` CPA, `2` first Ehrenfest half, `3` first
force/drift, `4` second force/kick, `5` second half and `6` endpoint.

Use the numerical result's natural diagnostic shapes: scalar for a vector,
`(K,)` for a column block, `(B,)` for batched vectors, `(B,K)` for batched blocks.
The accumulated estimate and budget use the same shapes. `action.value` keeps
the corresponding electronic shape and is a numerical diagnostic candidate,
not a physical checkpoint state. Integer status/iterations use int32; numerical
estimates use the electronic array's real dtype. Non-action failures leave the
last action record unchanged and mark the failing trajectories separately.

An `empty_checked_info(state, batch=...)` helper allocates zero diagnostics,
false masks, phase `0` and substep `-1` from shapes/dtypes alone; `action.value` reuses
the input electronic array. It does not call a model, measurement,
force routine or `eval_shape` on a numerical step. This initializes the scan's
fixed PyTree before tracing, works when `steps=0`, and leaves arbitrary existing
`method_state` trees untouched. No Python strings, changing dictionaries, optional
array shapes or executable callbacks belong in these records.

## 3. Budget division must use the macrostep input

For each trajectory and column, fix

\[
\tau_k = \mathrm{atol}+\mathrm{rtol}\,\|c_k(t_n)\|_2
\]

once at macrostep entry. With `N=electronic_substeps`:

- CPA has `N` midpoint-frozen actions of duration `dt/N`; each gets absolute
  budget `tau_k/N`.
- Ehrenfest has `N` actions in **each** half, hence `2N` actions of duration
  `dt/(2N)`; each gets `tau_k/(2N)`.

Do not give each Ehrenfest half the full macrostep budget. Do not recompute a
fresh relative allowance from a drifting approximate norm at every substep.
The action kernel accepts the explicit dynamic argument:

```python
action = lanczos_action(apply, vectors, duration, options,
                        absolute_budget=per_column_budget)
candidate = action.value
```

The budget is a JAX numerical input, not a newly constructed dataclass inside
JIT. Standalone action calls may continue deriving a budget from the supplied
vector and options when `absolute_budget` is omitted. A zero vector remains an
exact zero action even when its allocated budget is zero. Reject invalid
nontrivial budgets through action status.

Accumulate the returned estimates by column and require the final sum to be no
greater than the original `tau_k` before accepting the macrostep; code `3`
reports any excess. Fixed absolute allocation avoids norm-dependent allowances;
the final check also makes the policy explicit in finite arithmetic. The triangle
inequality motivates this division for products of frozen Hermitian unitary
actions in exact arithmetic. The heuristic roundoff allowance prevents claiming
a rigorous floating-point guarantee. For a block, the corresponding aggregate
diagnostic is the Euclidean norm of the column estimates (a Frobenius error
measure); there is no division by the number of independent trajectories.

This budget controls only the electronic exponential approximations. It does
not control CPA midpoint time-ordering error, Ehrenfest splitting/force errors,
model-fitting error, or feedback amplification through the nuclear path. Time
step, substep, model and observable convergence still require separate checks.

## 4. A batch-aware checked method step

CPA and Ehrenfest expose the optional method hook:

```python
checked_step = method.build_checked_step(problem, integrator, batch=batch)
candidate_state, step_info = checked_step(macrostep_input)
```

The existing `build_step(problem, integrator)` and its single-trajectory contract
remain unchanged. The new hook handles the leading batch axis itself because
**`vmap(single_trajectory_checked_step)` is unsafe for the intended gating**:
batched conditionals can become selects and evaluate both branches. Instead,
vectorize each completed action/force stage, reduce its status across all
trajectories/columns, then use a **scalar** `lax.cond` before the next stage.

The shared checked-segment helper is a bounded loop over electronic substeps:

```python
carry = (input_vectors, empty_info)
for j in fixed_range(N):                 # lax.fori_loop or scan
    carry = lax.cond(
        carry.info.code == 0,           # one scalar for the whole batch
        attempt_midpoint_action_and_merge_info,
        identity,
        carry,
    )
```

An action's status is reduced across `(B,K)` before another substep is allowed.
The first failure record is latched; later loop iterations cannot overwrite it.
The action itself remains vectorized across columns/trajectories. CPA samples
its nuclear treatment at elapsed `(j+.5)*dt/N` relative to each macrostep input,
so different trajectory time origins retain their current meaning.

CPA pseudocode:

```python
info = empty_checked_info(state)
budget = macrostep_budget(state.electronic)
c, info = checked_segment(midpoint_path_action, state.electronic,
                           duration=dt, substeps=N, budget=budget/N)

def finish_if_success():
    q, p = point_for_all_trajectories(state, elapsed=dt)
    # Mark failure if endpoint data are nonfinite.
    return completed_macrostep_or_original_state(q, p, c, state, info)

return lax.cond(info.code == 0, finish_if_success,
                lambda: (state, info))
```

Validate midpoint coordinates before invoking the Hamiltonian. On a failed
action do not sample a later endpoint merely to manufacture a failed state.
The accepted time and step advance only after the entire macrostep succeeds.

Ehrenfest pseudocode, with a scalar success gate before every following stage:

```text
electronic half 1 at q0, N actions, each budget tau/(2N)
  -> force(q0, c_half); check finite across batch
  -> p_half and q1; check finite before another provider evaluation
  -> force(q1, c_half); check finite across batch
  -> p1; check finite
  -> electronic half 2 at q1, N actions, each budget tau/(2N)
  -> assemble and check complete candidate state
  -> accept q1,p1,c1,time+dt,step+1 for every trajectory together
```

Any failure returns the **whole macrostep input state**, including q, p,
electronic amplitudes, time, step, random key and user method state, together
with the failure diagnostics. A successful first electronic half is not a
checkpointable half-time physical trajectory state. A rejected second half must
not leave updated nuclei behind. Nuclear/force finiteness is a mandatory
acceptance condition in this checked path, independent of the optional legacy
`Execution.check_finite` policy.

### Precise boundary for “no calls after failure”

After a completed action reports failure, no subsequent force, action, endpoint
or measurement stage is evaluated. After a rejected macrostep, no later
macrostep is evaluated anywhere in the batch. Use scalar conditionals outside
all batch maps to enforce this at runtime.

This does **not** promise cancellation of already in-flight compiled work inside
one batched Lanczos action. The production kernel's vmapped lanes can perform
masked/zero matvecs while other lanes are active. A declared pure linear
Hermitian `apply` makes those extra calls scientifically harmless; opaque
side-effecting providers are outside that contract. If collective early stopping
inside the Krylov loop becomes necessary, add a scalar all-lanes gate there as a
separately measured kernel optimization. Do not serialize every lane merely to
claim cancellation. The production integration accepts native JAX providers;
external callback action support needs its own explicit validation and is rejected
by the current preflight.

Both branches are still traced during compilation. No Python side-effect or
call-count guarantee is made for tracing a provider; providers must already be
pure. The guarantee concerns numerical runtime evaluation after returned status.

## 5. Runner: a separate checked block, one output-buffer path

The existing `_block` handles string integrators/MASH2. The private
`_checked_block` is selected only by the new option and uses the same fixed
sampling indices. It returns `(state, (times, values), block_info)` and shares the
existing cache with a distinct `("checked", batch, nsteps, indices)` key.

`block_info` contains the latched `CheckedStepInfo` and first failed macrostep index
(`-1` initially). It is transient; do not add it to physical checkpoints or
default measurement output. Initialize it by the shape-only helper above.

The sparse buffer strategy handles **all collected checked output**,
including dense sampling, using one checked loop for both dense and sparse
checked output. Existing dense fast paths remain unchanged.
With no output requested, allocate no observation buffers and do not trace or
evaluate the measurement.

```python
def attempt(carry, index):
    before, _, cursor, times, values = carry
    start_time = chunk_input.time + index*dt
    before = before._replace(time=start_time)
    candidate, info = checked_step(before)  # already batch-aware

    def accept():
        accepted = candidate._replace(time=chunk_input.time + (index+1)*dt)
        # checked_step has advanced step/key/method_state correctly.
        # This observe call exists only inside the successful scalar branch.
        cursor1, times1, values1 = save_if_selected(accepted, index, buffers)
        return accepted, success_block_info, cursor1, times1, values1

    def reject():
        return before, failed_block_info(info, index), cursor, times, values

    return lax.cond(info.code == 0, accept, reject)

def body(carry, index):
    return lax.cond(carry.block_info.failed_macro_index < 0,
                    lambda: attempt(carry, index), lambda: carry), None

final, _ = lax.scan(body, initial_carry, arange(nsteps))
```

`save_if_selected` uses another scalar conditional with the existing constant
save mask. Output shapes can be prepared with `eval_shape(observe, input)` only
when output is requested. On success, dense/sparse schedules, absolute-step
sampling and final-row inclusion match current behavior. On failure, any
accepted rows earlier in that chunk remain private and are discarded together
with unused buffer rows. The host never sees duplicate frozen-state rows.

The checked branch does not use MASH2's method-owned `step_succeeded` hook.
Its acceptance decision directly controls clock/counter stamping. Leave the
existing `_set_time`/MASH2 path intact, including its partial-event diagnostic
times and existing exceptions.

## 6. Host acceptance, exceptions and checkpoint authority

In `_run`, the selected block is invoked and synchronized within the narrow
execution exception wrapper, including `block_info` in `jax.block_until_ready`.
Its compact status is inspected **before** publishing or collecting that chunk's
observations. Status checking is mandatory even with `collect=False` and
`check_finite=False`.

A returned nonzero status raises `SimulationError` with:

- `last_valid_state=previous_state`, the complete synchronized state at chunk
  entry. Its actual time/step are the authoritative safe restart boundary.
- `failed_state=returned_state`, the macrostep input retained at the first
  rejected attempt, possibly after earlier accepted but unpublished macrosteps
  in this chunk. It is useful diagnostic geometry, not a published checkpoint
  boundary; automatic recovery uses `last_valid_state`.
- A `diagnostics` attribute containing host `block_info`, stable trajectory IDs
  and attempted macrostep start time. Ordinary execution exceptions retain its
  default value `None`; no new exception hierarchy was introduced.

An execution/callback exception before a trustworthy result is returned keeps
the existing `failed_state=None` and chained cause. Preflight validation and host
observer exceptions remain outside this wrapper. Do not classify an output
write failure as an integrator rejection; output streams are not atomic
checkpoints.

The physical state and HDF5 checkpoint schema remain unchanged. Krylov vectors
and temporary diagnostic histories need not be serialized; reconstruct each
action on retry. Solver options enter the existing strict manifest. Changing
options, source or dependency versions therefore follows the current explicit
restart compatibility policy; schema compatibility does not imply identical
manifests across a source upgrade. Never silently retry with a dense method,
renormalize a rejected vector, or discard failed ensemble members.

## 7. Implemented scope and acceptance tests

Runtime changes are limited to the action/options and numerical records,
`Integrator` validation/name, CPA/Ehrenfest optional hooks, Runner's checked
block/status gate, and an explicit unsupported-option check in `RecordedCPA`.
Model, nuclear-state, observable, ensemble and checkpoint APIs are unchanged.
The historical experimental benchmark retains its old field names and its own
recorded source identity.

The acceptance tests cover:

1. Existing string-integrator and MASH2 tests keep identical state/output
   semantics, including dense/sparse/no-output and restart.
2. Finite but unconverged action rejection works; a second-half Ehrenfest failure
   rolls back every physical field and random key to macrostep entry.
3. One bad trajectory or column rejects the entire macrostep; later stages,
   measurements and macrosteps are not evaluated. Use runtime callback guards
   or deliberate downstream failures to test the scalar gating, not Python
   trace-time counters.
4. Diagnostics have fixed shapes for vectors/blocks and singleton/multiple
   trajectories. User `method_state` remains unchanged by numerical diagnostics.
5. Nonzero time origins/step counters survive failure; the error's chunk-entry
   checkpoint resumes deterministically, with no failed-chunk output published.
6. CPA divides the input-based budget over `N` actions; Ehrenfest divides over
   `2N`, and rejects a final accumulated excess. Include unnormalized and zero
   CPA columns and a dynamic budget under JIT.
7. Numerical/provider failures are detected without output and with optional
   finite checks disabled; observer exceptions retain their separate behavior.
8. Small independent SciPy comparisons establish CPA midpoint convergence and
   Ehrenfest parity with separately computed frozen exponentials and a Verlet
   split. Complete-run accuracy and timing must be measured separately; action
   microbenchmarks alone do not establish faster or more accurate dynamics.

The direct method tests are in `tests/test_checked_methods.py`; public failure,
restart, sampling and parameter-update tests are in `tests/test_checked_runner.py`
and the independent stage-callback review is in `tests/test_checked_runner_review.py`.
Runtime callback instrumentation checks stage evaluation after compiled execution,
not Python trace counts. The direct method suite also covers nonfinite midpoint,
endpoint, force and overflowed drift, unsupported precision, divided budgets that
reject an otherwise acceptable single action, and final accumulated-budget rejection.
This is numerical/operational validation on CPU, not validation of material models
or of quantum nuclear physics.

This is a fixed-step, fail-closed integration. Adaptive time stepping, automatic
Krylov retries, partial-chunk publication/recovery, a general solver registry,
moving-AO dynamics and new MASH algorithms remain separate work.
