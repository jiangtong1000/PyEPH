# MPI trajectory ensembles

[`examples/mpi_ensemble.py`](../examples/mpi_ensemble.py) runs independent native
Ehrenfest trajectories on actual MPI processes, merges their statistics, and
checks the result against a serial run. The core ensemble API supplies local
batching and statistical merging; the example supplies process orchestration.
One complete Hamiltonian and trajectory batch live on each rank. This does not
distribute a single Hamiltonian or trajectory across ranks.

## Run the example

With PyEPH and its `mpi` extra installed in a Python environment backed by a
compatible MPI runtime, run from the repository root:

```sh
JAX_ENABLE_X64=1 mpiexec -n 2 python -m mpi4py examples/mpi_ensemble.py
```

The `mpi` extra installs `mpi4py`; an MPI runtime and matching launcher must
also be available. `python -m mpi4py` aborts the MPI job if an unhandled Python
exception occurs, which avoids leaving other ranks waiting in a collective.

This checkout was validated with a project-local MPI environment:

```sh
FI_TCP_IFACE=lo0 JAX_ENABLE_X64=1 .cache/mpi-venv/bin/mpiexec -n 2 \
  .cache/mpi-venv/bin/python -m mpi4py examples/mpi_ensemble.py
```

`FI_TCP_IFACE=lo0` selected the macOS loopback interface for this single-host
MPICH test. It is not a cluster network setting. The MPI environment resides
under `.cache/mpi-venv`; no system MPI installation was modified. The command
needs permission to create local sockets when run inside a restrictive sandbox.

Defaults run both seven trajectories and one trajectory on two ranks. Options
include `--counts 7 1`, `--steps 56`, `--batch-size 2`, and
`--output outputs/mpi_ensemble.json`. New runs write this working output. The
historical validation used the single-host macOS CPU configuration above; its
execution report is not included in the distribution. The serial reference
deliberately uses batch size three. The script requires at least two MPI ranks.

## Orchestration and identity

Each rank constructs the same `Simulation`, then calls
`partition_ids(global_ids, rank, size)`. Sampling uses the global trajectory ID
and a common seed, so rank assignment and local batch size do not change random
draws. This example uses nonconsecutive IDs beginning with `17, 54, 91, ...`,
Wigner harmonic sampling with seed 724, and a two-mode spin-boson Hamiltonian.

Nonempty ranks call `run_ensemble`; empty ranks skip this call but still enter
every MPI collective. `run_ensemble` requires a nonempty ID list. Rank zero
reconstructs validated `EnsembleResult` objects from gathered numerical and
JSON fields, then applies `merge_ensembles`. The merge uses counts, means and
M2; averaging rank means would be wrong for the default four/three partition.
Only rank zero writes the final JSON report.

The example also gathers each final state by global ID for validation. A large
production ensemble can send only the mean/M2 summaries, stream per-rank output,
or store final states through `batch_observer`; gathering every final state on
rank zero is not required by the ensemble API. This recipe uses trusted-process
mpi4py object collectives to transfer arrays and metadata, not executable model
objects or checkpoint files.

Merge validation rejects overlapping trajectory IDs, inconsistent scientific
manifests or initialization identities, and incompatible output grids. The
explicit `preparation_id` covers initialization code, seed, distribution and
parameters. `FunctionalMeasurement` is opaque to automatic provenance, so the
example supplies a `measurement.function` artifact identity derived from its
script hash. That identity covers this top-level function, which uses only the
supplied state and library operations. For another callback, hashing its source
alone does not identify captured weights, external files or mutable closures;
the caller must identify all such artifacts. See [restart provenance](RESTART.md).

The manifest also records package source and runtime versions. Keep the code
and artifacts unchanged during a distributed run; strict validation can reject
otherwise numerically identical results if the package source changes between
rank execution and merging.

Platform and integrated release evidence are recorded in [qualification](QUALIFICATION.md).
