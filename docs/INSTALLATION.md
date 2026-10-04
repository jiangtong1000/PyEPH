# Installation and platform scope

Use Python 3.11 or newer in an isolated environment. For source development:

```sh
python -m venv .venv
.venv/bin/python -m pip install -e '.[test]'
JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 .venv/bin/python -m pytest
JAX_ENABLE_X64=1 .venv/bin/python examples/linear_epc.py
```

Base dependencies are JAX, NumPy, SciPy and h5py. Importing PyEPH does not change
JAX precision. Enable x64 before creating arrays or compiling functions. The
compatibility transport facade and default mapping-event tolerances require it.

| Optional capability | Requirement |
| --- | --- |
| Torch energy or Hamiltonian providers | `pip install -e '.[torch]'` and explicit host-callback execution opt-in |
| Independent MPI workers | `pip install -e '.[mpi]'`, a compatible MPI runtime and launcher |
| Plotting examples | `pip install -e '.[plot]'` |
| Legacy CIF helper | Install `pymatgen` separately when needed |
| Legacy localization optimizer | Install `pyqcpbc` separately when needed |

The optional helpers import their dependencies only when called. Their presence
does not qualify every combination of external versions, accelerator and model.
A scalar force-field provider supplies nuclear energy and forces; an electronic
Hamiltonian still needs its own [physical contract](MODEL_CONTRACTS.md).

`requirements-minimum.txt` pins the declared numerical dependency floor and
pytest for compatibility checks. `requirements-tested.txt` records one complete
local environment; it is evidence for that environment, not a portable lockfile.
See [qualification](QUALIFICATION.md) for actual platform results. Availability
of a JAX device alone does not establish speedup, memory scaling or identical
floating-point trajectories on that device.

On Apple Silicon with Accelerate, the minimum NumPy 2.0.0 stack can emit
spurious matrix-product warnings even for exact finite products. Treating those
warnings as errors can interrupt valid calculations. The qualified current
stack, including NumPy 2.5.3, avoids the reproduced warnings. See
[qualification](QUALIFICATION.md) and the [upstream NumPy fix](https://github.com/numpy/numpy/pull/29223).

## Explicit accelerator selection

Install an accelerator-enabled JAX build compatible with the device and driver,
following the [JAX installation instructions](https://docs.jax.dev/en/latest/installation.html).
PyEPH's base dependency does not select a CUDA distribution. Preserve the actual
JAX/plugin/driver versions in each qualification record.

For NVIDIA runs that use host diagnostic callbacks, initialize both backends
and verify that GPU remains the default:

```sh
JAX_ENABLE_X64=1 JAX_PLATFORMS=cuda,cpu python - <<'PY'
import jax
assert jax.default_backend() == "gpu"
assert jax.devices()[0].platform == "gpu"
assert jax.devices("cpu")[0].platform == "cpu"
print(jax.devices())
PY
```

JAX initializes the explicitly named platforms and uses the first as default;
a requested initialization failure raises instead of silently selecting CPU.
See [JAX platform configuration](https://docs.jax.dev/en/latest/config_options.html#platforms).
Keep the same environment variables on the actual workflow command. The CPU
backend permits host diagnostics; its availability does not qualify optional
external providers on GPU. For a deliberate CPU run, use `JAX_PLATFORMS=cpu`.

`benchmarks/platform_qualification.py` checks independently referenced molecular
and periodic CPA/Ehrenfest fixtures, validates every timed output, and separates
checkpoint byte integrity from numerical continuation. Its `--device` option
requires the requested device. `benchmarks/platform_reproducibility.py` repeats
unchanged inputs through warmed computations and records byte differences
separately from numerical errors; finite repeatability tests cannot prove
universal determinism. Neither script validates a material parameterization.

## Distribution checks

Follow the [release procedure](RELEASE.md) to build from a new inventory-controlled
export. Install its wheel into a fresh environment and execute the source
archive's tests outside the checkout. The source distribution includes the
listed independent numerical references and compatibility fixtures. It does not
include every internal experiment or historical execution archive.

The installed smoke test checks module imports, precision ownership, model
updates, checkpoint continuation, propagation methods and force contractions:

```sh
python -m pip install --no-deps --target /tmp/pyeph-wheel /tmp/pyeph-distribution/*.whl
python -I tests/packaging/check_installed.py --target /tmp/pyeph-wheel
```

Use new directories and select exactly one intended wheel. This smoke reuses
the invoking environment's dependencies; a full platform qualification also
runs the complete exported suite and records skips, warnings and source identity.
