"""Smoke a separately installed wheel, without importing this checkout's src.

Run with the dependency environment's Python and isolated mode:
python -I tests/packaging/check_installed.py --target .cache/wheel-acceptance
Install the built wheel into that target with pip --no-deps --target first.
"""

import argparse
import importlib
import json
from pathlib import Path
import pkgutil
import sys
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, required=True)
    args = parser.parse_args()
    target = args.target.resolve()
    checkout_src = Path(__file__).resolve().parents[2]/"src"
    sys.path[:] = [str(target)] + [entry for entry in sys.path if entry != str(checkout_src)]
    import jax
    import numpy as np

    initial_precision = jax.config.x64_enabled
    import pyeph

    if not Path(pyeph.__file__).is_relative_to(target):
        raise RuntimeError("this smoke must use the independently installed wheel")
    # walk_packages alone omits implicit namespace directories such as adapters.
    # Include every installed Python source module, without importing optional
    # dependencies merely to discover the modules that wrap them.
    modules = {entry.name for entry in pkgutil.walk_packages(pyeph.__path__, prefix="pyeph.")}
    package_root = Path(pyeph.__file__).parent
    for source in package_root.rglob("*.py"):
        relative = source.relative_to(package_root).with_suffix("")
        parts = relative.parts[:-1] if relative.name == "__init__" else relative.parts
        if parts:
            modules.add("pyeph." + ".".join(parts))
    modules = sorted(modules)
    for name in modules:
        importlib.import_module(name)
    assert jax.config.x64_enabled == initial_precision, "an import changed global precision"
    pyeph.configure_precision(True)
    from pyeph.dynamics.mash2 import sample_adiabatic_population
    from pyeph.core.contracts import prepared_action, pure_state_weight
    from pyeph.models.analytic import SpinBosonModel
    from pyeph.models.neural import NeuralResidualModel
    import jax.numpy as jnp

    model = SpinBosonModel()
    params = model.default_params()
    initial = pyeph.make_state([.1], [.2], [1.+0.j, 0.+0.j])
    problem = pyeph.Problem(model, params, pyeph.CoupledClassical(1.), pyeph.Ehrenfest())

    def force_at_angle(angle):
        c = jnp.array([jnp.cos(angle), jnp.sin(angle)], dtype=jnp.complex128)
        return (-model.reference_gradient(params, initial.q)
                - model.contract_gradient(params, initial.q, pure_state_weight(c)))[0]

    np.testing.assert_allclose(jax.grad(force_at_angle)(.3), 2*np.sin(.6), atol=2e-14)
    simulation = pyeph.Simulation(problem, pyeph.Integrator(.001), pyeph.Execution(save_every=5))
    result = simulation.run(initial, 20)
    with tempfile.TemporaryDirectory(prefix="pyeph-wheel-") as directory:
        checkpoint = Path(directory)/"restart.h5"
        simulation.save_checkpoint(checkpoint, result.final_state)
        loaded = simulation.load_checkpoint(checkpoint)
        for expected, actual in zip(jax.tree.leaves(result.final_state), jax.tree.leaves(loaded), strict=True):
            np.testing.assert_array_equal(actual, expected)
    simulation.update_parameters({**params, "delta": .2})
    changed = simulation.run(initial, 20)
    assert not np.allclose(changed.final_state.electronic, result.final_state.electronic)

    checked = pyeph.Simulation(problem, pyeph.Integrator(.001, pyeph.LanczosOptions()))
    checked_result = checked.run(initial, 5)
    assert checked_result.metadata["electronic_integrator"] == "lanczos_midpoint"
    from pyeph.execution.runner import SimulationError

    limited = pyeph.Simulation(
        problem, pyeph.Integrator(.001, pyeph.LanczosOptions(max_dimension=1)),
        pyeph.Execution(check_finite=False))
    try:
        limited.run(initial, 5, collect=False)
    except SimulationError as error:
        assert int(error.diagnostics["step_info"].code) == 1
        assert int(error.last_valid_state.step) == 0
    else:
        raise AssertionError("installed checked solver accepted an unconverged action")

    neural = NeuralResidualModel(2, (1,), hidden_sizes=(4,), complex_valued=True)
    neural_params = neural.init_params(jax.random.key(7), zero_last=False)
    prepared = jax.jit(lambda p, q, c: prepared_action(neural, p, q)(c))
    np.testing.assert_allclose(prepared(neural_params, initial.q, initial.electronic),
                               neural.apply(neural_params, initial.q, initial.electronic), atol=2e-14)
    neural_run = pyeph.Simulation(
        pyeph.Problem(neural, neural_params, pyeph.CoupledClassical(1.), pyeph.Ehrenfest()),
        pyeph.Integrator(.001, pyeph.LanczosOptions())).run(initial, 3, collect=False)
    np.testing.assert_allclose(jnp.linalg.norm(neural_run.final_state.electronic), 1., atol=2e-14)

    mash_problem = pyeph.Problem(model, params, pyeph.CoupledClassical(1.), pyeph.MASH2(),
                                pyeph.MASHPopulation())
    mash = pyeph.Simulation(mash_problem, pyeph.Integrator(.001, "exponential_midpoint"))
    mapping = sample_adiabatic_population(model, params, [.1], [.2], active=0, trajectory_id=1)
    mapped = mash.run(mapping, 3)
    assert int(mapped.final_state.method_state["status"]) == 0

    from pyeph.dynamics.mashrm import sample_population
    from pyeph.models.epc import LinearEPCModel

    rm_model = LinearEPCModel(3, 2)
    rm_params = rm_model.create_params(np.diag([-.4, .1, .8]), np.zeros((2, 3, 3)),
                                      omega=[.2, .3])
    rm_problem = pyeph.Problem(rm_model, rm_params, pyeph.CoupledClassical([1., 2.]),
        pyeph.MASHRM(event_substeps=1), pyeph.MASHRMPopulation(include_density=True))
    rm_initial = sample_population(rm_model, rm_params, [.1, -.2], [.2, .3], seed=12)
    rm = pyeph.Simulation(rm_problem, pyeph.Integrator(.01, "exponential_midpoint"))
    rm_result = rm.run(rm_initial, 3)
    assert int(rm_result.final_state.method_state["status"]) == 0
    np.testing.assert_allclose(rm_result.observables["mapping_norm"], 1., atol=2e-14)
    np.testing.assert_allclose(rm_result.observables["population"].sum(axis=-1), 1., atol=5e-14)
    with tempfile.TemporaryDirectory(prefix="pyeph-wheel-rm-") as directory:
        path = Path(directory)/"restart.h5"
        rm.save_checkpoint(path, rm_result.final_state)
        loaded = rm.load_checkpoint(path)
        for expected, actual in zip(jax.tree.leaves(rm_result.final_state),
                                    jax.tree.leaves(loaded), strict=True):
            np.testing.assert_array_equal(actual, expected)
    # New parameter values can change sorted surface ownership without changing
    # any static shapes; reject before publishing a zero-step initial row.
    rm.update_parameters({**rm_params, "h0": jnp.diag(jnp.array([.8, .1, -.4]))})
    try:
        rm.run(rm_initial, 0)
    except ValueError as error:
        assert "largest mapping population" in str(error)
    else:
        raise AssertionError("installed RM method published inconsistent initial ownership")

    from pyeph.observables.transport.mashrm import FixedPositionVelocity, RMVelocity
    from pyeph.workflows.mashrm_equilibrium import LinearEPCCanonical
    from pyeph.workflows.mashrm_transport import RMTransport

    canonical = LinearEPCCanonical(rm_model, rm_params, [1., 2.], beta=1.5)
    velocity = RMVelocity(probe_callback=FixedPositionVelocity({
        "velocity": [[0., .2, .1], [.2, 1., .3], [.1, .3, 2.]]}))
    transport = RMTransport(canonical, pyeph.Integrator(.01, "exponential_midpoint"), velocity,
                           method=pyeph.MASHRM(event_substeps=1),
                           execution=pyeph.Execution(chunk_size=2))
    thermal = transport.prepare([19, 3, 7], seed=2026)
    prefix = transport.run(thermal, 2)
    uninterrupted = transport.run(thermal, 4)
    with tempfile.TemporaryDirectory(prefix="pyeph-wheel-rm-transport-") as directory:
        path = Path(directory)/"transport.h5"
        transport.save_checkpoint(path, prefix.final_state)
        restored = transport.load_checkpoint(path)
        np.testing.assert_array_equal(restored.origin.velocity0, thermal.origin.velocity0)
        resumed = transport.run(restored, 2)
        np.testing.assert_allclose(resumed.statistics.mean[-1], uninterrupted.statistics.mean[-1],
                                   atol=2e-14)
        for expected, actual in zip(jax.tree.leaves(uninterrupted.final_state.state),
                                    jax.tree.leaves(resumed.final_state.state), strict=True):
            np.testing.assert_allclose(actual, expected, atol=2e-14)
    assert np.isfinite(uninterrupted.statistics.standard_error).all()

    from pyeph.core.contracts import LowRankWeight
    from pyeph.models.local import AtomCenterMap, LocalBlockGraph, LocalBlockModel, LocalCoefficients

    def local_provider(p, q, geometry):
        internal = jnp.sum((q-geometry.centers[geometry.atom_site])**2, axis=1)
        scalar = jnp.zeros(2).at[geometry.atom_site].add(internal)
        onsite = p["onsite"] + .03*scalar[:, None, None]*jnp.eye(2)
        hopping = p["hopping"]*jnp.exp(-.2*geometry.distances)[:, None, None]
        return LocalCoefficients(onsite, hopping)

    local = LocalBlockModel(LocalBlockGraph(2, 2, ((0, 1),), switch_on=1., cutoff=3.),
                            AtomCenterMap((0, 0, 1), (.5, .5, 1.), 2), local_provider)
    local_params = dict(onsite=jnp.array([[[.1, .01], [.01, .3]], [[.4, .02], [.02, .5]]]),
                        hopping=jnp.array([[[.1, .02], [-.01, .07]]]))
    q = jnp.array([[0., 0., 0.], [.2, .1, 0.], [1.5, .2, 0.]])
    c = jnp.array([1., 0., 0., 0.], dtype=jnp.complex128)
    local.validate_at(local_params, q)
    action = jax.jit(lambda p, x, v: prepared_action(local, p, x)(v))
    np.testing.assert_allclose(action(local_params, q, c), local.apply(local_params, q, c), atol=2e-14)
    weight = LowRankWeight(c[:, None], c[:, None])
    gradient = local.contract_gradient(local_params, q, weight)
    displacement = jnp.zeros_like(q).at[0, 0].set(1e-5)
    finite_difference = (jnp.vdot(c, local.apply(local_params, q+displacement, c)).real
                         - jnp.vdot(c, local.apply(local_params, q-displacement, c)).real)/2e-5
    np.testing.assert_allclose(gradient[0, 0], finite_difference, atol=3e-12)
    local_simulation = pyeph.Simulation(
        pyeph.Problem(local, local_params, pyeph.CoupledClassical(jnp.ones((3, 1))), pyeph.Ehrenfest()),
        pyeph.Integrator(.001, pyeph.LanczosOptions(max_dimension=4)))
    local_result = local_simulation.run(pyeph.make_state(q, jnp.zeros_like(q), c), 3, collect=False)
    np.testing.assert_allclose(jnp.linalg.norm(local_result.final_state.electronic), 1., atol=2e-14)

    from pyeph.paths.harmonic import ConstantPath
    from pyeph.workflows.column_transport import (
        initialize_infinite_temperature_columns,
        make_column_transport_problem,
    )

    column_problem = make_column_transport_problem(
        local, local_params, pyeph.PrescribedPath(ConstantPath(q)),
        probes=("current_x", "current_y"))
    columns = initialize_infinite_temperature_columns(
        column_problem, q, jnp.zeros_like(q), trace_ids=[3, 11], trajectory_id=19, seed=8)
    assert columns.electronic.shape == (4, 6)
    column_simulation = pyeph.Simulation(column_problem, pyeph.Integrator(.001, "rk4"))
    column_full = column_simulation.run(columns, 4)
    column_prefix = column_simulation.run(columns, 2)
    with tempfile.TemporaryDirectory(prefix="pyeph-wheel-columns-") as directory:
        artifact_ids = {"model.coefficient_provider": "installed-smoke-local-v1"}
        path = Path(directory)/"restart.h5"
        column_simulation.save_checkpoint(path, column_prefix.final_state, artifact_ids=artifact_ids)
        restored = column_simulation.load_checkpoint(path, artifact_ids=artifact_ids)
        column_resumed = column_simulation.run(restored, 2)
        np.testing.assert_array_equal(restored.method_state["column_transport"]["time0"], 0.)
        np.testing.assert_allclose(column_resumed.observables["current_correlation"][-1],
                                   column_full.observables["current_correlation"][-1], atol=2e-14)
    np.testing.assert_allclose(column_full.observables["column_norm_squared_drift"], 0., atol=2e-14)

    from pyeph.adapters.ao_frames import project_ao_path
    from pyeph.dynamics.recorded import RecordedCPA

    ao = np.array([[[1., 0.], [0., 1.]], [[1.1, .1j], [.2, .9]],
                   [[.9, .2j], [.1j, 1.2]]], dtype=np.complex128)
    rows = np.swapaxes(np.linalg.inv(ao), -1, -2)
    adjoint = ao.conj().swapaxes(-1, -2)
    projected = project_ao_path(
        [0., .1, .2], np.tile([-.1, .2], (3, 1)), rows,
        adjoint@ao, adjoint[:-1]@ao[1:], retained_bands=[0, 1],
        energy_unit="hartree", time_unit="atomic", ao_basis_id="smoke-two-AOs",
        basis_id="smoke-two-eigenstates", source_identity="installed-AO-smoke-v1")
    recorded = RecordedCPA(projected.path)
    amplitudes = np.array([1., 1j])/np.sqrt(2.)
    recorded_initial = recorded.initialize(amplitudes)
    recorded_full = recorded.run(recorded_initial, 2)
    np.testing.assert_allclose(recorded_full.final_state.electronic,
                               np.exp(-.2j*np.array([-.1, .2]))*amplitudes, atol=2e-14)
    with tempfile.TemporaryDirectory(prefix="pyeph-wheel-ao-") as directory:
        path = Path(directory)/"recorded.h5"
        prefix = recorded.run(recorded_initial, 1)
        recorded.save_checkpoint(path, prefix.final_state)
        restored = recorded.load_checkpoint(path)
        continued = recorded.run(restored, 1)
        np.testing.assert_array_equal(continued.final_state.electronic,
                                      recorded_full.final_state.electronic)
    assert projected.evidence.source_identity == "installed-AO-smoke-v1"

    torch_reference_tested = False
    torch_local_tested = False
    if importlib.util.find_spec("torch") is not None:
        from pyeph.adapters.torch_reference import TorchReferenceModel
        from pyeph.models.composite import SumModel

        def reference_energy(p, coordinates):
            return .5*p["spring"]*(coordinates**2).sum()

        reference = TorchReferenceModel(local.spec, reference_energy)
        reference_params = {"spring": jnp.asarray(.1)}
        np.testing.assert_allclose(reference.reference_gradient(reference_params, q), .1*q,
                                   atol=2e-14)
        external = pyeph.Simulation(
            pyeph.Problem(SumModel((local, reference)), (local_params, reference_params),
                          pyeph.CoupledClassical(jnp.ones((3, 1))), pyeph.Ehrenfest()),
            pyeph.Integrator(.001, "rk4"),
            pyeph.Execution(allow_host_callbacks=True, verify_external_gradients=True))
        external_result = external.run(pyeph.make_state(q, jnp.zeros_like(q), c), 3, collect=False)
        assert np.isfinite(np.asarray(external_result.final_state.q)).all()
        torch_reference_tested = True

        from pyeph.adapters.torch_local import TorchLocalBlockModel
        from pyeph.io.provenance import problem_manifest

        def torch_coefficients(p, coordinates, geometry):
            return LocalCoefficients(p["onsite"]*(1.+.01*coordinates.square().sum()),
                                     p["hopping"]*(-.1*geometry.distances).exp()[:, None, None])

        external_carrier = TorchLocalBlockModel(local.graph, local.centers, torch_coefficients)
        external_carrier.validate_complete_gradients(local_params, q)
        sparse_problem = pyeph.Problem(external_carrier, local_params,
                                       pyeph.CoupledClassical(jnp.ones((3, 1))), pyeph.Ehrenfest())
        sparse_simulation = pyeph.Simulation(sparse_problem, pyeph.Integrator(.001, "rk4"),
                                              pyeph.Execution(allow_host_callbacks=True))
        sparse_result = sparse_simulation.run(pyeph.make_state(q, jnp.zeros_like(q), c), 3, collect=False)
        assert np.isfinite(np.asarray(sparse_result.final_state.q)).all()
        sparse_manifest = problem_manifest(sparse_problem, sparse_simulation.integrator,
                                            artifact_ids={"model.coefficient_provider": "installed-torch-local-v1"})
        assert sparse_manifest["payload"]["runtime"]["versions"]["torch"]
        torch_local_tested = True
    print(json.dumps(dict(installed_file=pyeph.__file__, imported_submodules=len(modules),
                          import_preserves_precision=True, checkpoint_exact=True,
                          cached_parameter_update=True, mash_status=0,
                          checked_lanczos_acceptance=True, checked_lanczos_rejection=True,
                          native_force_sensitivity=True,
                          prepared_neural_action=True,
                          mashrm_status=0, mashrm_checkpoint_exact=True,
                          initial_model_state_preflight=True,
                          rm_canonical_preparation=True, rm_velocity_correlation=True,
                          rm_origin_checkpoint_continuation=True,
                          local_atomic_preflight=True, local_prepared_action=True,
                          local_atomic_force_finite_difference=True,
                          local_checked_ehrenfest=True,
                          column_transport_origins=True, column_transport_restart=True,
                          ao_frame_projection=True, ao_provenance_checkpoint=True,
                          scalar_torch_reference=torch_reference_tested,
                          sparse_torch_coefficients=torch_local_tested,
                          ehrenfest_final_norm=float(np.vdot(result.final_state.electronic,
                                                             result.final_state.electronic).real)), indent=2))


if __name__ == "__main__":
    main()
