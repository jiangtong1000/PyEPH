# Developing PyEPH

Start with `Problem` and a complete runnable example. A model supplies physical
operations; a method supplies equations and estimator requirements; the runner
handles batching and output. Keep helpers beside their callers until distinct
uses justify sharing them.

## Development loop

1. State the physical or operational failure and the expected behavior.
2. Read the relevant implementation, callers and tests before editing.
3. Add a meaningful regression or independent invariant check. Use explicit
   units, basis conventions and numerical tolerances with a physical rationale.
4. Run focused checks, then the applicable integration suite and linting.
5. Record limitations in the guide for that capability. Update the release
   inventory only for files intended to be published.

```sh
JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 .venv/bin/python -m pytest tests/test_dynamics.py -q
.venv/bin/python -m ruff check src tests tools
```

The full CPU suite is the integration gate. Installed-wheel tests must run from a
directory outside the editable checkout. Optional providers, different dependency
stacks and accelerator execution have separate gates; a skipped test is not
qualification of that capability.

## Public contracts

- [Models](docs/ADDING_MODELS.md): `reference_energy`, `apply`, and force
  contractions where required. Preserve vector and block actions.
- [Methods](docs/ADDING_METHODS.md): `validate` and pure `build_step`; preparation
  and measurement follow the physical method.
- [API](docs/PUBLIC_API.md): supported entry points and compatibility policy.
- [Release](docs/RELEASE.md): publication inventory and isolated builds.

Avoid a global backend switch or a universal symbolic Hamiltonian hierarchy.
Changing numerical parameters must not require mutating captured static objects.
Do not remove physical checks, provenance or independent oracles to make a
benchmark faster. Report compilation and steady execution separately.

## Scientific scope

An analytic fixture, a trained surrogate and a validated material model are
different deliverables. Force agreement verifies derivatives of the supplied
model; it does not establish the model's physical accuracy. New scientific
methods remain separate until their preparation, equations, observables and
limits have independent evidence.
