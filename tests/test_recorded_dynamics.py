import numpy as np
import pytest
from scipy.linalg import expm

from pyeph.dynamics.recorded import RecordedCPA
from pyeph.execution.runner import Execution
from pyeph.integrators.electronic import Integrator
from pyeph.io.checkpoint import load_checkpoint, save_checkpoint
from pyeph.paths.electronic import AdiabaticElectronicPath, FixedBasisElectronicPath


def test_fixed_recorded_evolution_nonzero_start_and_restart(tmp_path):
    h = np.array([[0.2, 0.13j], [-0.13j, -0.1]])
    path = FixedBasisElectronicPath([5., 6., 7.], np.stack([h] * 3))
    sim = RecordedCPA(path, Integrator(0.01, "exponential_midpoint"), Execution(chunk_size=13))
    initial = sim.initialize([1, 0])
    first = sim.run(initial, 60)
    checkpoint = tmp_path / "electronic.h5"
    save_checkpoint(checkpoint, first.final_state, metadata={"path_hash": "constant-fixture"})
    restored, _ = load_checkpoint(checkpoint, expected_metadata={"path_hash": "constant-fixture"})
    final = sim.run(restored, 40)
    np.testing.assert_allclose(final.final_state.electronic, expm(-1j * h) @ [1, 0], atol=4e-14)
    np.testing.assert_allclose(final.observables["norm"], 1., atol=6e-14)
    with pytest.raises(ValueError):
        sim.run(final.final_state, 101)


def test_adiabatic_gauge_phases_and_permutations_exact_constant_physics():
    energies = np.array([0.1, 0.3])
    u0 = np.eye(2, dtype=complex)
    u1 = np.array([[0, np.exp(0.3j)], [np.exp(-0.7j), 0]])
    u2 = np.diag(np.exp(1j * np.array([0.6, -0.4])))
    path = AdiabaticElectronicPath([0., 0.3, 0.8], [energies, energies[::-1], energies],
                                  np.stack([u0.conj().T @ u1, u1.conj().T @ u2]))
    sim = RecordedCPA(path)
    initial = np.array([1., 1j]) / np.sqrt(2)
    result = sim.run(sim.initialize(initial), 2)
    physical = u2 @ result.final_state.electronic
    np.testing.assert_allclose(physical, np.exp(-0.8j * energies) * initial, atol=1e-14)
    np.testing.assert_allclose(result.observables["norm"], 1., atol=1e-14)


def test_raw_subspace_projection_loss_is_not_renormalized():
    path = AdiabaticElectronicPath([0., 1.], [[0., 0.], [0., 0.]],
                                  [np.diag([0.8, 1.])])
    sim = RecordedCPA(path)
    initial = sim.initialize([1, 0])
    with pytest.raises(ValueError, match="subspace loss"):
        sim.run(initial, 1)
    permitted = RecordedCPA(path, max_subspace_loss=0.4).run(initial, 1)
    np.testing.assert_allclose(permitted.final_state.electronic, [0.8, 0])
    np.testing.assert_allclose(permitted.observables["norm"][-1], 0.64)
    np.testing.assert_allclose(permitted.observables["maximum_subspace_loss"][-1], 0.36)


def test_polar_transport_still_checks_raw_loss_and_missing_overlap():
    path = AdiabaticElectronicPath([0., 1.], [[0., 0.], [0., 0.]],
                                  [np.diag([0.8, 1.])], transport_mode="polar")
    sim = RecordedCPA(path)
    with pytest.raises(ValueError, match="subspace loss"):
        sim.run(sim.initialize([1, 0]), 1)
    permitted = RecordedCPA(path, max_subspace_loss=None).run(sim.initialize([1, 0]), 1)
    np.testing.assert_allclose(permitted.observables["norm"], 1.)
    with pytest.raises(ValueError, match="energies alone"):
        RecordedCPA(AdiabaticElectronicPath([0., 1.], [[0., 0.], [0., 0.]]))


def test_recorded_streaming_blocks_and_frame_only_rejection():
    h = np.array([[0., 0.2], [0.2, 0.]])
    path = FixedBasisElectronicPath([0., 1.], [h, h])
    sim = RecordedCPA(path, Integrator(0.02), Execution(chunk_size=7, save_every=3))
    chunks = []
    initial = sim.initialize(np.eye(2))
    result = sim.run(initial, 20, observer=lambda t, v: chunks.append((t, v)), collect=False)
    np.testing.assert_allclose(result.final_state.electronic, expm(-0.4j * h), atol=3e-12)
    np.testing.assert_allclose(np.concatenate([x[0] for x in chunks]),
                               np.array([0, 3, 6, 9, 12, 15, 18, 20]) * 0.02)
    assert result.observables == {}
    with pytest.raises(ValueError, match="interpolation"):
        RecordedCPA(FixedBasisElectronicPath([0., 1.], [h, h], interpolation="frames_only"),
                    Integrator(0.02))


def test_decimal_grid_endpoint_roundoff_and_invalid_loaded_state():
    path = FixedBasisElectronicPath([0., .1, .2, .3], np.zeros((4, 2, 2)))
    sim = RecordedCPA(path, Integrator(.1))
    initial = sim.initialize([1, 0])
    result = sim.run(initial, 3)
    np.testing.assert_allclose(result.final_state.electronic, [1, 0])
    assert np.isfinite(path.sample_frame(0.1 * 3).hamiltonian).all()
    assert np.isnan(path.sample_frame(.3 + 1e-8).hamiltonian).all()
    with pytest.raises(ValueError, match="finite vector/block"):
        sim.run(initial._replace(electronic=np.ones(3)), 0)
