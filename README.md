# PyEPH

Python infrastructure for effective electron–nuclear dynamics and charge transport.
Physical models, nuclear treatments, dynamics methods, numerical integration,
measurements and execution are separate components. Models retain dense, sparse,
edge or orbital-block structure and expose operator actions and contracted forces.

This is a development release. Supported mathematical contracts do not certify a
material model outside its validated domain. See [qualification](docs/QUALIFICATION.md)
and the [development roadmap](DEVELOPMENT_PLAN.md).

## Install

Use Python 3.11 or newer in an isolated environment:

```sh
python -m venv .venv
.venv/bin/python -m pip install -e '.[test]'
JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 .venv/bin/python -m pytest
```

The runtime depends on JAX, NumPy, SciPy and h5py. Optional extras are `torch`,
`mpi` and `plot`. Importing the package does not change global precision. Set
`JAX_ENABLE_X64=1` before launch or explicitly configure precision before arrays
or compiled functions are created. The import name is `pyeph`; use separate
virtual environments when comparing different package versions.

## A small calculation

```python
from pyeph import (CoupledClassical, Ehrenfest, Integrator, Problem,
                   Simulation, configure_precision, make_state)
from pyeph.models.analytic import SpinBosonModel

configure_precision(enable_x64=True)
model = SpinBosonModel(nmodes=1)
problem = Problem(model, model.default_params(), CoupledClassical(masses=1.0), Ehrenfest())
simulation = Simulation(problem, Integrator(dt=0.01, electronic="exponential_midpoint"))
initial = make_state(q=[0.2], p=[0.1], electronic=[1, 0])
print(simulation.describe())
result = simulation.run(initial, steps=1000)
print(result.observables["population"][-1])
```

Native kernels use a consistent declared unit system with hbar=1. Atomic units
are the default. Unit declarations do not convert user arrays automatically.
Model parameters are dynamic PyTrees. Update them between runs with
`simulation.update_parameters(new_params)`; changing static model or method
configuration requires constructing a new simulation.

## Choose a workflow

| Task | Guide |
| --- | --- |
| Install optional providers and check platform scope | [Installation](docs/INSTALLATION.md) |
| Add a Hamiltonian or differentiable residual | [Model interface](docs/ADDING_MODELS.md) |
| Add a dynamics method or estimator | [Method interface](docs/ADDING_METHODS.md) |
| Use the supported public API | [API and compatibility](docs/PUBLIC_API.md) |
| Migrate an existing PyEPH CPA transport calculation | [Transport migration](docs/TRANSPORT_MIGRATION.md) |
| Resume independent trajectory jobs | [Campaigns](docs/CAMPAIGNS.md) |
| Build finite or periodic candidate graphs | [Neighbor coverage](docs/NEIGHBOR_GRAPHS.md) |
| Reject internal checked stages outside a declared coordinate domain | [Coordinate domains](docs/COORDINATE_DOMAINS.md) |
| Load labels, validate fits and preserve model identity | [Model lifecycle](docs/MODEL_LIFECYCLE.md) |
| Save and restore a trajectory | [Restart](docs/RESTART.md) |
| Differentiate smooth trajectory observables | [Sensitivity rollouts](docs/DIFFERENTIABLE_DYNAMICS.md) |
| Prepare nonlinear canonical nuclear samples | [Finite-chain sampling](docs/NONLINEAR_CANONICAL.md) |
| Develop or qualify a change | [Contributor guide](CONTRIBUTING.md) |
| Build an auditable source release | [Release procedure](docs/RELEASE.md) |

Implemented methods include prescribed-path CPA, coupled Ehrenfest, original
real two-state MASH and separate real finite-state Runeson–Manolopoulos mapping
dynamics. Their preparation and observable estimators are distinct. The finite
multistate implementation requires a complete isolated real spectrum. Complex
Hamiltonian support in CPA/Ehrenfest does not establish SOC hopping capability.

Maintained charge-transport and preprocessing APIs remain available under
`pyeph.greenkubo`, `pyeph.post_qe2pert` and `pyeph.preprocessing`. Native workflows
also support Hamiltonian-column correlations and explicit phonon dressing.
Recorded electronic frames have a separate electronic-only propagation profile;
loading them does not provide nuclear feedback forces.

Native JAX models support complete coordinate derivatives. Optional Torch
providers use an explicit host-callback boundary and require execution opt-in.
A differentiable provider does not imply differentiability through hopping
switches or the public host-side simulation runner.

Batching, bounded output chunks, HDF5 streaming, stable trajectory IDs and strict
source-bound restart are available. A campaign coordinates independent work
units; it does not distribute a single Hamiltonian across workers. Published
hardware qualification is separate from the execution API.

## Release contents

`release-files.txt` names every file intended for version control and release.
The release tool audits this inventory and can additionally apply a local
publication policy. Development evidence outside that inventory remains local.
Scientific citations and required dependency/license notices are preserved.
