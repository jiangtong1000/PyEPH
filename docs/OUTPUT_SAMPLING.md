# Sampled output execution

`Execution(save_every=k)` controls actual measurement evaluation and device
output storage. Earlier runner versions evaluated measurements at every step,
stored the entire chunk on device, and selected rows afterward. The current
runner evaluates and stores only scheduled observations. Nuclear/electronic
propagation and per-chunk validity checks are unchanged.

## Public behavior

The output schedule uses the absolute trajectory step counter. Initial and
final observations are always retained when output has a consumer; intermediate
steps are retained when `absolute_step % save_every == 0`. This makes a restart's
scheduled intermediate output independent of where the restart occurs. A
restart still has its own initial observation, which can duplicate the preceding
run's final row when concatenating segments.

- `collect=True` returns sampled host arrays, optionally also sending each
  output chunk to an observer.
- `collect=False, observer=callback` streams sampled host arrays without
  accumulating them in `RunResult`.
- `collect=False, observer=None` is **no-output execution**: initial and later
  measurements are neither traced nor evaluated. The returned observations and
  times are empty. Model, state, method, finite-state, and failure validations
  still run, and the final trajectory state remains available.

Measurements are pure functions of the supplied problem/state. They must not
provide hidden state updates or side effects required by propagation. Skipping
unconsumed or unscheduled measurements is therefore part of the execution
contract. The observer is the separate host interface for output side effects.

## Compiled block behavior

The private `_block(batch, nsteps, sample_indices=None)` retains all-step output
by default, preserving the kernel entry point used by existing benchmarks.
An explicit selection is a sorted vector of unique zero-based indices of
post-step states. The compiled-function cache includes this schedule.

There are three paths:

1. Dense output retains a conventional `lax.scan` with observations in its
   returned arrays. No extra scatter-buffer overhead is introduced.
2. Sparse output uses abstract shape evaluation to allocate compact time/value
   buffers. A scalar `lax.cond` evaluates and writes a measurement only at a
   selected step. The scan returns no per-step output stack.
3. An empty selection traces no measurement code and allocates no observation
   buffers. Its private output is empty times plus `{}`.

Runtime model parameters remain arguments to the compiled function. Nested
observable PyTrees, complex arrays, scalar values, integer/bool diagnostics,
batch axes, and shape-declared host callbacks are supported. The sampling
decision stays outside trajectory `vmap`, so sparse measurement evaluation
is conditional for the entire selected batch rather than an elementwise
selection that computes both branches.

Output memory scales with the number of selected rows in a chunk. A chunk still
stores the current trajectory state and the integrator's temporary work. With
`collect=True`, all selected host output is accumulated across chunks; use an
observer with `collect=False` to bound that accumulation. Compilation is cached
per distinct chunk length and output pattern; aligning a regular chunk size
with the sampling stride can reduce the number of cached patterns.

For a long calculation, pass the intended step count to one public `run` and
let `Execution.chunk_size` control compilation and output chunks. Use an
observer with `collect=False` when the complete host history is unnecessary.
Calling `run(..., steps=1)` repeatedly also repeats host preflight and workflow
origin/provenance validation. Those checks are intentional; increasing chunk
size is not a request to remove them or to trust a cached physical identity.

Failure behavior is preserved. For example, a failed MASH trajectory retains
its last physical event time while other batch members may advance. Method
status is checked before any failed chunk is sent to the observer, including
chunks containing no scheduled output. No-output execution does not suppress
these failures.

## Verification

Run:

```sh
.venv/bin/python -m pytest tests/test_output_sampling.py -q
```

The 17 sampling tests cover exact absolute schedules across different chunk
sizes and restarts, sparse versus dense trajectory equivalence, nested output,
batched and unbatched runs, JIT and eager entry points, zero-output chunks,
runtime measurement invocation counts, host callbacks, runtime parameters,
and batched MASH failure times. A deliberately failing measurement proves that
no-output execution skips both initial and later measurement tracing while
state validation remains active. Existing dynamics, ensemble, and MASH suites
also passed after the change.

## Reusing the initial measurement kernel

With `Execution(jit=True)`, the initial measurement now uses a cached JIT
function, just as later observations run inside compiled trajectory blocks.
The cache holds executable code; current numerical parameters and trajectory
state remain runtime arguments. Validated `update_parameters` calls and
repeated runs therefore update the initial observation without capturing old
values. Scalar and batched observers use separate cache entries.
`Execution(jit=False)` retains eager evaluation. No-output execution still
never constructs or traces this observer. Preflight, measurement validation,
host callbacks, finite checks, failure handling, and output publication keep
their existing order.

`tests/test_output_sampling.py` covers output schedules, zero-step observations,
cache reuse with changed parameters and no-output execution. Complete-run
performance still needs measurement for the intended model and workload.
