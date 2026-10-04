"""Static configuration cannot drift away from compiled physics or provenance."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pyeph import CPA, CoupledClassical, Ehrenfest, Integrator, Problem, Simulation, make_state
from pyeph.core.problem import PrescribedPath
from pyeph.core.contracts import ModelSpec, ProbeContext
from pyeph.core.system import SystemSpec
from pyeph.core.units import UnitSystem
from pyeph.dynamics.mash2 import MASH2, MASHPopulation
from pyeph.io.provenance import assert_matching_manifest, problem_manifest
from pyeph.models.analytic import SpinBosonModel, TullyModel
from pyeph.models.aggregate import AggregateModel
from pyeph.models.periodic import PeriodicBlockModel
from pyeph.observables.population import FunctionalMeasurement
from pyeph.paths.electronic import AdiabaticElectronicPath, FixedBasisElectronicPath
from pyeph.paths.harmonic import ConstantPath, HarmonicBath, HarmonicPath
from pyeph.paths.nuclear import RecordedNuclearPath


def free_model():
    model = SpinBosonModel()
    params = model.default_params() | {
        "omega": jnp.array([0.]), "coupling": jnp.array([0.]), "delta": 0.}
    return model, params


def assert_state_equal(left, right):
    for a, b in zip(jax.tree.leaves(left), jax.tree.leaves(right), strict=True):
        np.testing.assert_allclose(a, b, atol=2e-14, rtol=2e-14)


def test_mass_snapshot_preserves_cached_physics_and_restart_identity(tmp_path):
    mass = np.array([1.])
    model, params = free_model()
    problem = Problem(model, params, CoupledClassical(mass), Ehrenfest())
    integrator = Integrator(.1, electronic="exponential_midpoint")
    simulation = Simulation(problem, integrator)
    initial = make_state([0.], [1.], [1., 0.])
    before = simulation.run(initial, 1).final_state
    identity = problem_manifest(problem, integrator)

    mass[0] = 2.  # A caller still owns this mutable array, but the model does not.
    np.testing.assert_array_equal(problem.nuclear_treatment.masses, [1.])
    cached = simulation.run(initial, 1).final_state
    fresh = Simulation(problem, integrator).run(initial, 1).final_state
    assert_state_equal(before, cached)
    assert_state_equal(cached, fresh)
    np.testing.assert_allclose(cached.q, [.1])
    assert_matching_manifest(identity, problem_manifest(problem, integrator))
    checkpoint = tmp_path / "mass.h5"
    simulation.save_checkpoint(checkpoint, cached)
    assert_state_equal(Simulation(problem, integrator).load_checkpoint(checkpoint), cached)

    changed = replace(problem, nuclear_treatment=CoupledClassical(mass))
    np.testing.assert_allclose(Simulation(changed, integrator).run(initial, 1).final_state.q, [.05])
    with pytest.raises(ValueError, match="nuclear_treatment"):
        Simulation(changed, integrator).load_checkpoint(checkpoint)


@pytest.mark.parametrize("treatment", ["constant", "bath"])
def test_prescribed_static_arrays_do_not_change_between_cached_and_fresh_runs(treatment):
    q = np.array([.4])
    omega, masses = np.array([1.2]), np.array([2.])
    nuclear = (PrescribedPath(ConstantPath(q)) if treatment == "constant"
               else HarmonicBath(omega, masses))
    model, params = free_model()
    params["coupling"] = jnp.array([.7])
    problem = Problem(model, params, nuclear, CPA())
    integrator = Integrator(.1, electronic="exponential_midpoint")
    simulation = Simulation(problem, integrator)
    initial = make_state(q, [.3], [2**-.5, 2**-.5])
    before = simulation.run(initial, 3).final_state
    identity = problem_manifest(problem, integrator)
    q[:] = 4.
    omega[:] = 2.4
    masses[:] = 5.
    cached = simulation.run(initial, 3).final_state
    fresh = Simulation(problem, integrator).run(initial, 3).final_state
    assert_state_equal(before, cached)
    assert_state_equal(cached, fresh)
    assert_matching_manifest(identity, problem_manifest(problem, integrator))


def test_dynamic_numpy_parameters_still_update_an_existing_compiled_kernel():
    model, params = free_model()
    params["delta"] = np.array(.2)
    problem = Problem(model, params, PrescribedPath(ConstantPath([0.])), CPA())
    simulation = Simulation(problem, Integrator(.1, electronic="exponential_midpoint"))
    initial = make_state([0.], [0.], [1., 0.])
    before = simulation.run(initial, 2).final_state
    params["delta"][...] = .8  # Numeric params are explicit runtime arguments.
    after = simulation.run(initial, 2).final_state
    np.testing.assert_allclose(abs(before.electronic[1])**2, np.sin(.2*.2)**2, atol=2e-14)
    np.testing.assert_allclose(abs(after.electronic[1])**2, np.sin(.8*.2)**2, atol=2e-14)


def test_harmonic_path_copies_all_arrays_and_origin():
    arrays = [np.array([.4]), np.array([.2]), np.array([.7]), np.array([2.])]
    origin = np.array(.3)
    path = HarmonicPath(*arrays, origin=origin)
    expected_q, expected_v = np.array(path.position(.9)), np.array(path.velocity(.9))
    for array in arrays:
        array[:] = 8.
    origin[...] = 5.
    assert path.origin == .3
    np.testing.assert_array_equal(path.position(.9), expected_q)
    np.testing.assert_array_equal(path.velocity(.9), expected_v)


def test_recorded_paths_retain_independent_numeric_snapshots():
    times = np.array([0., 1., 2.])
    positions = np.array([[0.], [.1], [.2]])
    velocities = np.ones_like(positions)*.1
    hamiltonians = np.repeat(np.array([[[0., .2j], [-.2j, 1.]]]), 3, axis=0)
    energies = np.tile([-.2, .2], (3, 1))
    overlaps = np.repeat(np.eye(2, dtype=complex)[None], 2, axis=0)
    nuclear = RecordedNuclearPath(times, positions, velocities)
    fixed = FixedBasisElectronicPath(times, hamiltonians)
    adiabatic = AdiabaticElectronicPath(times, energies, overlaps)
    expected = {"nuclear": np.array(nuclear.position(.4)),
                "fixed": np.array(fixed.sample_frame(.4).hamiltonian),
                "energies": np.array(adiabatic.energies),
                "overlaps": np.array(adiabatic.overlaps)}
    for array in (times, positions, velocities, hamiltonians, energies, overlaps):
        array[...] = 9.
    for path in (nuclear, fixed, adiabatic):
        np.testing.assert_array_equal(path.times, [0., 1., 2.])
    np.testing.assert_array_equal(nuclear.position(.4), expected["nuclear"])
    np.testing.assert_array_equal(fixed.sample_frame(.4).hamiltonian, expected["fixed"])
    np.testing.assert_array_equal(adiabatic.energies, expected["energies"])
    np.testing.assert_array_equal(adiabatic.overlaps, expected["overlaps"])


def test_new_static_snapshots_preserve_shapes_and_explicit_array_precision():
    masses = np.array([[2.], [3.]], dtype=np.float32)
    coupled = CoupledClassical(masses)
    coupled.validate((2, 3))
    assert coupled.masses.shape == (2, 1) and coupled.masses.dtype == np.float32
    bath = HarmonicBath(np.array([.5], dtype=np.float32), np.array(2., dtype=np.float32))
    bath.validate((1,))
    assert bath.frequencies.dtype == np.float32 and bath.masses.dtype == np.float32


@pytest.mark.parametrize("factory", [
    lambda: HarmonicPath([1j], [0.], [1.]),
    lambda: HarmonicPath([0.], [0.], [1.], origin=np.inf),
    lambda: ConstantPath([1j]),
    lambda: HarmonicBath([1j]),
])
def test_static_paths_reject_invalid_real_coordinate_data(factory):
    with pytest.raises(ValueError, match="real"):
        factory()


def test_mutable_scalar_timestep_cannot_diverge_from_compiled_time_or_identity():
    dt = np.array(.1)
    substeps = np.array(1)
    integrator = Integrator(dt, electronic="exponential_midpoint", electronic_substeps=substeps)
    model, params = free_model()
    problem = Problem(model, params, CoupledClassical(1.), Ehrenfest())
    initial = make_state([0.], [1.], [1., 0.])
    simulation = Simulation(problem, integrator)
    before = simulation.run(initial, 2).final_state
    identity = problem_manifest(problem, integrator)
    dt[...] = .3
    substeps[...] = 3
    assert type(integrator.dt) is float and integrator.dt == .1
    assert type(integrator.electronic_substeps) is int and integrator.electronic_substeps == 1
    after = simulation.run(initial, 2).final_state
    assert_state_equal(before, after)
    np.testing.assert_allclose(after.time, .2)
    assert_matching_manifest(identity, problem_manifest(problem, integrator))


@pytest.mark.parametrize("periodic", [False, True])
def test_mutable_scalar_model_configuration_cannot_change_eager_or_compiled_operators(periodic):
    cutoff, switch_on, charge = np.array(8.), np.array(6.), np.array(-1.)
    complex_valued = np.array(False)
    kwargs = dict(cutoff=cutoff, switch_on=switch_on, charge=charge, complex_valued=complex_valued)
    model = (PeriodicBlockModel(np.array(2), np.array(1), ((0, 1, 0, 0, 0),), np.eye(3)*20, **kwargs)
             if periodic else AggregateModel(np.array(2), ((0, 1),), **kwargs))
    params = model.default_params()
    params["hopping"] = jnp.ones_like(params["hopping"])*.04
    q, vectors = jnp.array([[0., 0., 0.], [7., 0., 0.]]), jnp.eye(2, dtype=complex)
    action = jax.jit(model.apply)
    current = jax.jit(lambda p, x, v: model.probe_apply(p, ProbeContext(x), "current_x", v))
    before, current_before = np.array(action(params, q, vectors)), np.array(current(params, q, vectors))
    problem = Problem(model, params, CoupledClassical(1.), Ehrenfest())
    identity = problem_manifest(problem, Integrator(.1))
    cutoff[...] = 16.
    switch_on[...] = 12.
    charge[...] = 1.
    complex_valued[...] = True
    assert (model.cutoff, model.switch_on, model.charge, model.complex_valued) == (8., 6., -1., False)
    np.testing.assert_array_equal(action(params, q, vectors), before)
    np.testing.assert_allclose(model.apply(params, q, vectors), before, atol=1e-15)
    np.testing.assert_array_equal(current(params, q, vectors), current_before)
    np.testing.assert_allclose(model.probe_apply(params, ProbeContext(q), "current_x", vectors),
                               current_before, atol=1e-15)
    assert_matching_manifest(identity, problem_manifest(problem, Integrator(.1)))


def test_shapes_probes_capabilities_and_units_are_independent_static_values():
    shape = [np.array(1)]
    probes = ["current_x"]
    force, native, complex_valued = np.array(True), np.array(True), np.array(False)
    energy, length = np.array(.1), np.array(2.)
    units = UnitSystem(energy, length)
    spec = ModelSpec(SystemSpec(np.array(2), shape), probes=probes, force_support=force,
                     native_jax=native, complex_valued=complex_valued, unit_system=units)
    measurement = FunctionalMeasurement(lambda p, s: s.q, required_probes=probes)
    shape[0][...] = 9
    probes.append("current_y")
    force[...] = False
    native[...] = False
    complex_valued[...] = True
    energy[...] = .7
    length[...] = 3.
    assert spec.system.q_shape == (1,) and spec.system.nstates == 2
    assert spec.probes == measurement.required_probes == ("current_x",)
    assert (spec.force_support, spec.native_jax, spec.complex_valued) == (True, True, False)
    assert units.energy_hartree == .1 and units.length_bohr == 2.
    assert type(units.energy_hartree) is float and type(spec.native_jax) is bool


def test_mash_tolerances_and_analytic_branch_are_scalar_snapshots():
    names = ("event_tolerance", "gap_tolerance", "real_tolerance", "direction_tolerance")
    tolerances = {name: np.array(1e-8) for name in names}
    subdivisions, include_nuclei, kind = np.array(2), np.array(True), np.array(1)
    method = MASH2(event_substeps=subdivisions, **tolerances)
    measurement = MASHPopulation(include_nuclei)
    model = TullyModel(kind)
    q = jnp.array([.7])
    before = np.array(jax.jit(model.dense)(None, q))
    for value in tolerances.values():
        value[...] = .5
    subdivisions[...] = 16
    include_nuclei[...] = False
    kind[...] = 3
    assert method.event_substeps == 2 and measurement.include_nuclei is True
    assert all(type(getattr(method, n)) is float and getattr(method, n) == 1e-8 for n in names)
    assert model.kind == 1 and type(model.kind) is int
    np.testing.assert_allclose(model.dense(None, q), before, atol=1e-15)


@pytest.mark.parametrize("factory", [
    lambda: Integrator(np.array([.1])),
    lambda: Integrator(.1 + 0j),
    lambda: Integrator(.1, electronic_substeps=np.array(1.)),
    lambda: UnitSystem(np.array([1.]), 1.),
    lambda: UnitSystem(1., 1j),
    lambda: AggregateModel(2, ((0, 1),), cutoff=np.array([8.])),
    lambda: PeriodicBlockModel(2, 1, ((0, 1, 0, 0, 0),), np.eye(3), charge=1j),
    lambda: MASH2(event_tolerance=np.array([1e-8])),
    lambda: MASH2(gap_tolerance=1j),
    lambda: ModelSpec(SystemSpec(2, (1,)), native_jax=np.array([True])),
    lambda: SystemSpec(2, [np.array([1])]),
    lambda: TullyModel(np.array([1])),
])
def test_static_scalar_fields_reject_nonscalar_or_wrong_numeric_domains(factory):
    with pytest.raises(ValueError, match="scalar"):
        factory()
