"""Zero-temperature LF is valid physics and must have a strict JSON identity."""

import json

import jax
import numpy as np

from pyeph import Integrator, PrescribedPath, Simulation
from pyeph.io.provenance import problem_manifest, validate_manifest
from pyeph.models.aggregate import AggregateModel
from pyeph.paths.harmonic import ConstantPath
from pyeph.workflows.polaron_transport import (
    initialize_polaron_transport_state,
    make_polaron_transport_problem,
)


def test_zero_temperature_lf_strict_checkpoint_matches_uninterrupted(tmp_path):
    model = AggregateModel(2, ((0, 1),))
    params = model.default_params()
    q = np.array([[0., 0., 0.], [2., 0., 0.]])
    problem = make_polaron_transport_problem(
        model, params, PrescribedPath(ConstantPath(q)), [1.7], [.4], np.inf,
        hopping_pairs=[[0, 1], [1, 0]])
    integrator = Integrator(.01, "exponential_midpoint")
    manifest = problem_manifest(problem, integrator)
    validate_manifest(manifest)
    json.loads(json.dumps(manifest, allow_nan=False))
    encoded_beta = manifest["payload"]["measurement"]["fields"]["beta"]
    assert encoded_beta == {"kind": "float", "value": "+infinity"}

    initial = initialize_polaron_transport_state(problem, q, np.zeros_like(q))
    rho = np.asarray(initial.method_state["transport"]["rho0"])
    energies, rotation = np.linalg.eigh(np.asarray(problem.model.apply(params, initial.q, np.eye(2))))
    assert energies[0] < energies[1]
    np.testing.assert_allclose(rho, np.outer(rotation[:, 0], rotation[:, 0].conj()), atol=1e-14)
    simulation = Simulation(problem, integrator)
    whole = simulation.run(initial, 12)
    half = simulation.run(initial, 6)
    path = tmp_path / "zero-temperature.h5"
    simulation.save_checkpoint(path, half.final_state)
    resumed = simulation.run(simulation.load_checkpoint(path), 6)
    assert np.all(np.isfinite(whole.observables["current_correlation"]))
    np.testing.assert_allclose(resumed.observables["current_correlation"],
                               whole.observables["current_correlation"][6:], atol=2e-15)
    for expected, actual in zip(jax.tree.leaves(whole.final_state),
                                jax.tree.leaves(resumed.final_state), strict=True):
        np.testing.assert_allclose(actual, expected, atol=2e-15)
