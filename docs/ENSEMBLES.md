# Independent trajectory ensembles

`execution.ensemble.run_ensemble` initializes and propagates a fixed list of
trajectory identities in batches. The same IDs produce the same random draws
when the initializer uses the package's identity-based sampling functions.
Changing batch size or dividing IDs across processes preserves those draws.

Package trajectory and event streams explicitly use Threefry, including when
the ambient JAX PRNG implementation is `rbg`. States and checkpoints keep the
existing `uint32[2]` key format. `trajectory_keys(seed, ids)` returns those raw
pairs for compatibility; use `trajectory_keys(seed, ids, typed=True)` when
passing keys to JAX random operations. Typed keys retain their implementation
under JAX transformations and cannot be reinterpreted by an ambient setting.
Other JAX random settings, including the seed offset and Threefry partitioning
policy, remain part of the scientific provenance; this does not waive checkpoint
compatibility checks when process configuration changes.

```python
import numpy as np
from pyeph import make_state, stack_states
from pyeph.execution.ensemble import run_ensemble
from pyeph.initialization import sample_harmonic

def initialize(ids):
    q, p = sample_harmonic([1.0], 1.0, 0.1, ids, seed=28)
    return stack_states([
        make_state(q[i], p[i], [1, 0], trajectory_id=int(identity), seed=28)
        for i, identity in enumerate(ids)
    ])

# `simulation` is a configured Simulation, for example the README spin-boson.
ensemble = run_ensemble(
    simulation, initialize, np.arange(128), steps=1000, batch_size=16,
    preparation_id="classical-w1-T.1-spin0-seed28-v1",
)
print(ensemble.mean["population"][-1])
print(ensemble.standard_error["population"][-1])
```

Initialization returns a stacked state even for a batch of one. The caller's
`preparation_id` must change when initialization code, distribution, parameters,
or seed change. It is an explicit identity for a potentially opaque callback;
the library cannot infer the contents of its closure. The model, integrator,
bath, measurement, basis, units and code also have a strict scientific manifest,
using the same rules as [checkpoint restart](RESTART.md).

The output is an **unweighted independent-sample average**. Variance uses `N-1`
and standard error is `sqrt(variance/N)`. A one-sample standard error is NaN.
For complex data the variance is the squared complex modulus about the mean.
Importance weights require a separate estimator; time-series autocorrelation
and material/model error are not included in this sampling uncertainty.

Memory contains one trajectory batch's sampled outputs and the mean/M2 arrays
on the requested time grid. `batch_size` bounds simultaneous trajectories; it
does not bound the number of saved times. Use `Execution.save_every` to select
the output interval. A `batch_observer(ids, result)` callback can save that
batch's final states or raw results before the next batch replaces them.

Tune batch size against complete-run throughput and memory; a larger batch
is not guaranteed to improve either. [Campaigns](CAMPAIGNS.md) add durable
work ownership and recovery above these ensemble operations.

## Explicit partitioning

`partition_ids(ids, rank, size)` makes round-robin partitions. No MPI import or
process startup occurs. Each worker/device can run its own configured simulation
on its assigned IDs; empty partitions skip execution. Merge completed results
with `merge_ensembles(left, right)`. Merging rejects overlapping IDs, changed
preparations or scientific manifests, and different physical output grids.
Means and M2 use the parallel Welford formula, so ranks need not have equal
sample counts. Averages over rank averages are insufficient for unequal ranks.

This interface supplies local batching and the statistical merge operation.
Actual multi-process/device orchestration is a separate layer; a partitioning
unit test is not evidence of MPI or GPU execution. Floating-point reduction
order can change last-bit results across partitions.
