# Durable trajectory campaigns

`pyeph.execution.campaign.Campaign` adds durable work-unit bookkeeping above
`run_ensemble`. Each independent worker reconstructs its own `Simulation` and
initializer, claims pending IDs, computes a small ensemble, and atomically
publishes its numerical moments. No scheduler, MPI launcher, model object,
compiled function or executable Python serialization is stored.

This is a local-filesystem implementation. Its SQLite transactions, atomic
renames and file/directory `fsync` calls require a filesystem with those
semantics. Network filesystems and multi-node shared-storage locking are not
qualified. Multiple independent processes on the same qualified local
filesystem are supported.

## Create, work and merge

```python
from pyeph.execution.campaign import Campaign

# simulation and initialize(ids) are reconstructed by the application.
# initialize returns a stacked state, with random draws determined by each ID.
campaign = Campaign.create(
    "outputs/transport_campaign", simulation, trajectory_ids=range(1000),
    steps=20000, preparation_id="sha256:<initializer-and-preparation-bundle>",
    shard_size=32,
)
```

Creation refuses to replace an existing directory. The immutable specification
binds the model, basis, physical units, carrier convention, reference potential,
parameters, nuclear treatment, measurement, method, integrator, source and
runtime through the existing strict problem manifest. It additionally binds
the preparation identity, ordered trajectory IDs, work-unit partition,
integration length, output time grid and time precision. Opaque providers still require explicit
`artifact_ids`; see [restart provenance](RESTART.md).

The preparation identity is a caller assertion covering the initialization
code, input files, distribution parameters and seed. It cannot be inferred from
a callback's Python source alone. Different workers must use that same bundle.
The default declaration is `initial_time=0., initial_step=0`. Nonzero starts
must be explicitly declared at creation; every initialized state is checked.
`time_dtype` defaults to float64 when JAX x64 is enabled, otherwise float32;
applications using another supported time precision must declare it explicitly.
A lower-precision result cannot broaden the grid acceptance tolerance. Accepted
accumulation roundoff is mapped to the exact declared time grid when saved, so
independently valid shards remain mergeable. Unresolvable adjacent output times
are rejected during creation.
A campaign reruns interrupted work units from their declared preparation, not
from an arbitrary unverified intermediate state.

Each process opens the directory and runs any number of work units:

```python
campaign = Campaign("outputs/transport_campaign")
while campaign.run_next(
    simulation, initialize, preparation_id=preparation_id, batch_size=8,
    worker_id="worker-1",
) is not None:
    pass
```

`run_next` checks the reconstructed scientific manifest and output interval
before claiming work. Batch size and execution chunk size may change; IDs and
preparation do not. Successful calls return a `CampaignClaim`. `None` means no
pending unit exists, which can also mean another worker is running or a failed
unit needs attention. Inspect `campaign.ledger()` to distinguish these states.
`campaign.attempts()` retains old failures after successful retries.

```python
ensemble = campaign.merge()  # requires every unit to be completed
print(ensemble.count, ensemble.standard_error)
```

Merging checks file checksums, shard identity, exact trajectory IDs, the frozen
physical time grid and existing ensemble invariants. It uses the existing
count-weighted parallel Welford calculation in fixed work-unit order, independent
of completion order. Unequal work-unit sizes are valid. Different numerical
batching may still change last-bit arithmetic; this is not a bitwise-invariance
claim across execution configurations. Partial statistics require explicit
`campaign.merge(require_complete=False)` and include only completed IDs.

## Failure and interruption

A caught computation error marks the unit failed and re-raises the original
exception, including a `SimulationError`'s retained state and diagnostics. The
attempt ledger saves the error type/message. If a detailed failure checkpoint
is needed, catch that original exception and use the existing checkpoint API.
A retry restarts the whole bounded work unit:

```python
campaign.retry_failed(shard_id=3)
```

Failed units are never silently retried. A hard process crash can leave a unit
`running`. Confirm that the specific worker has stopped before explicitly
revoking its attempt:

```python
row = campaign.ledger()[3]
campaign.recover(
    row["id"], expected_token=row["token"],
    reason="Worker process exited with status 137; confirmed by its process handle",
)
```

A timeout, an unchanged progress message or elapsed wall time alone is not
proof of termination. There is no automatic lease expiry. Recovery records the
reason, makes the unit pending, and fences the old token. Even if an old worker
returns unexpectedly, it cannot commit over the replacement's result. Duplicate
completion, stale failure reports and retries of completed units are rejected.

The data publication order is write temporary NPZ → file `fsync` → atomic
rename → result-directory `fsync` → SQLite completion commit. Only committed
files are merged. An interruption before database commit may leave an
unreferenced temporary/result file; recovery safely recomputes that unit.
These orphan files are deliberately ignored rather than treated as completed
work. They can be removed after all workers have stopped by comparing files to
`result_file` entries in the ledger. Never delete a file being written by an
active worker. Keep the entire campaign directory together when archiving it;
make a consistent backup only while workers are stopped.

## Resource and scientific scope

Worker memory is one execution batch and the work unit's mean/M2 arrays on the
sampled time grid. Merge memory is accumulated moments plus one result shard;
it does not load all trajectory time series. Metadata and result-disk size grow
with the number of work units. `Execution.save_every` controls sampled output
size; `shard_size` controls recomputation lost after interruption. Failed retry
attempt records and orphan files are retained for diagnosis, so repeated
failures can increase disk use.

The result remains an unweighted average over independent trajectories.
Standard errors are trajectory sampling errors, not independent-time-origin
errors, model uncertainty or a convergence certificate. This coordinator does
not broaden any dynamics method's physical domain. It does not resume within a
partially computed work unit or distribute a Hamiltonian over workers.

The executable [campaign example](../examples/campaign.py) uses a parameterized
spin-boson Hamiltonian in a fixed orthonormal two-state basis with atomic units,
a single propagated electronic amplitude vector and a harmonic neutral
reference. Its sampled classical nuclei begin at temperature 0.1 and mass 1;
it is an interface/numerical demonstration, not a validated material model.

Focused tests exercise independent-process claim contention, real process exits
before publication, after publication and after commit, explicit recovery and
stale-token rejection. A separate actual Ehrenfest calculation compares unequal
work units and changed batching against an ordinary ensemble calculation.
