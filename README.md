<p align="center">
  <img src="./logo.png" width="280" alt="PyEPH logo">
</p>

<h1 align="center">PyEPH</h1>

<p align="center">
  <strong>First-principles electron–phonon Hamiltonians and nonperturbative charge dynamics</strong>
</p>

<p align="center">
  <a href="https://github.com/jiangtong1000/PyEPH/actions/workflows/ci.yml"><img src="https://github.com/jiangtong1000/PyEPH/actions/workflows/ci.yml/badge.svg" alt="CI status"></a>
  <a href="#installation"><img src="https://img.shields.io/badge/Python-3.10%2B-blue.svg" alt="Python 3.10+"></a>
  <a href="#testing"><img src="https://img.shields.io/badge/tests-serial%20%7C%20MPI-informational.svg" alt="Serial and MPI tests"></a>
</p>

PyEPH is a research software package for constructing real-space electron–phonon Hamiltonians and simulating finite-temperature quantum dynamics. It connects first-principles outputs from [Quantum ESPRESSO](https://www.quantum-espresso.org/), [Wannier90](https://www.wannier.org/), and [PERTURBO](https://perturbo-code.github.io/) to nonperturbative quantum-classical calculations of spectral functions, charge mobility, and optical conductivity.

The package supports both model Hamiltonians and general first-principles Hamiltonians, with current ab initio workflows validated primarily for organic molecular crystals.

## Capabilities

- **First-principles Hamiltonians:** interpolate electronic bands, phonons, and electron–phonon couplings in a fully real-space representation based on maximally localized Wannier functions.
- **Model systems:** construct one- and two-dimensional Holstein, Peierls, and Holstein–Peierls models.
- **Quantum dynamics:** simulate nonperturbative real-time dynamics and evaluate Green–Kubo transport observables.
- **Electron–phonon localization:** optimize real-space electron–phonon couplings and analyze molecular or atomic contributions.
- **High-performance execution:** use MPI parallelization, Numba acceleration, sparse matrices, HDF5 output, and JAX-based optimization.
- **DFPT workflow guidance:** document image parallelization, restart, collection, and per-q-point irreducible-representation chunking.

## Scientific workflows

| Workflow | Purpose |
|---|---|
| [First-principles electron–phonon Hamiltonian](examples/01_abinitio_realspace_EPC/) | Quantum ESPRESSO → Wannier90 → PERTURBO → PyEPH |
| [Holstein transport model](examples/02_holstein/) | Run a Green–Kubo charge-transport simulation for a lattice model |
| [Phonon calculations](examples/01_abinitio_realspace_EPC/3_PHONONS/) | Configure DFPT parallelization, restart interrupted calculations, and collect per-q-point results |

Each first-principles workflow stage contains its own README with the required inputs, important numerical parameters, and troubleshooting notes.

<p align="center">
  <img src="examples/workflow_hamiltonian_paper.png" width="760" alt="PyEPH first-principles workflow">
</p>

## Installation

PyEPH currently supports **Python 3.10 or newer**. The repository is used directly through `PYTHONPATH` while packaging metadata is being prepared.

```bash
git clone https://github.com/jiangtong1000/PyEPH.git
cd PyEPH

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

export PYTHONPATH="$PWD:${PYTHONPATH:-}"
python -c "import pyeph; print('PyEPH import successful')"
```

For MPI execution, install an MPI implementation on the system and use the MPI dependency set:

```bash
python -m pip install -r requirements-mpi.txt
mpirun --version
```

The first-principles workflow additionally requires Quantum ESPRESSO, Wannier90, and PERTURBO. Their versions and runtime configuration should be kept consistent across all workflow stages.

## Quick validation

After installation, run the lightweight regression tests:

```bash
python -m pytest -q pyeph/post_qe2pert/test/test_unwrap_epc.py
python -m pytest -q pyeph/greenkubo/tests/test_mp_grid.py
```

## Example: Holstein transport

The included Holstein example runs a Green–Kubo transport calculation and writes trajectory data under `currents_output/`:

```bash
cd examples/02_holstein
python run.py 0
```

The command-line argument selects a temperature from the list defined in `run.py`. Production calculations can be launched with the accompanying Slurm template and MPI.

## First-principles workflow

The complete ab initio workflow is documented in [`examples/01_abinitio_realspace_EPC/`](examples/01_abinitio_realspace_EPC/). Its main stages are:

1. structural relaxation and self-consistent DFT;
2. DFPT phonons and optional dispersion-correction Hessians;
3. non-self-consistent DFT and Wannier localization;
4. Quantum ESPRESSO-to-PERTURBO conversion;
5. PyEPH interpolation and electron–phonon localization;
6. real-time dynamics and transport analysis.

The example inputs are templates rather than universal production settings. Convergence thresholds, reciprocal-space grids, pseudopotentials, and parallelization parameters must be validated for each material.

## Repository structure

```text
PyEPH/
├── pyeph/
│   ├── greenkubo/       # Hamiltonians, phonons, propagators, and transport
│   ├── post_qe2pert/    # First-principles interpolation and EPC localization
│   ├── legacy/          # Retained compatibility implementations
│   └── utils/           # Shared numerical and I/O utilities
├── examples/            # Model and first-principles workflows
└── .github/workflows/   # Serial and MPI continuous integration
```

## Testing

The continuous-integration workflow runs serial Green–Kubo tests, first-principles post-processing tests, and a multi-process MPI check. To run the main serial suites locally:

```bash
python -m pytest -q pyeph/greenkubo/tests
python -m pytest -q pyeph/post_qe2pert/test
```

## Development status

PyEPH is active research software. The version-controlled workflows and regression tests document the behavior used in current studies, while the Python API may continue to evolve. For archival calculations, record the exact Git commit and retain all input files, dependency versions, random seeds, and scheduler settings.

## Citation

This repository accompanies the following manuscript:

> Tong Jiang and Joonho Lee, “First-Principles Origins of Charge Transport in Molecular Semiconductors,” submitted (2026).

which is currently under review and does not yet have a public preprint or DOI yet.