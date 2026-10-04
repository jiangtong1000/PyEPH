"""Initialization owns state and coordinate-map inputs, including host buffers."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pyeph import Integrator, make_state
from pyeph.core.coordinates import MassWeightedModes
from pyeph.dynamics.recorded import RecordedCPA
from pyeph.paths.electronic import FixedBasisElectronicPath


@pytest.mark.parametrize("block", [False, True])
def test_make_state_snapshots_primary_arrays_and_time(block):
    q = np.arange(32, dtype=np.float64) / 10
    p = q + .7
    c = (q + 1j*p).reshape(16, 2) if block else q + 1j*p
    time = np.array(.4)
    expected = tuple(np.array(x, copy=True) for x in (q, p, c, time))
    state = make_state(q, p, c, time=time)
    jax.block_until_ready(state)
    for array in (q, p, c, time):
        array[...] = 9
    for actual, wanted in zip((state.q, state.p, state.electronic, state.time), expected):
        np.testing.assert_array_equal(actual, wanted)


def test_method_state_snapshots_nested_containers_and_array_leaves():
    weights = np.arange(8, dtype=np.float64)
    payload = {"nested": [weights, {"counter": np.array(17, dtype=np.uint64)}],
               "jax": jnp.array([2., 3.]), "absent": None}
    state = make_state([0.], [0.], [1., 0.], method_state=payload)
    weights[:] = -1
    payload["nested"][1]["counter"][...] = 19
    payload["nested"].append("later")
    payload["jax"] = jnp.array([5., 6.])
    payload["new"] = 1
    assert len(state.method_state["nested"]) == 2
    assert "new" not in state.method_state
    np.testing.assert_array_equal(state.method_state["nested"][0], np.arange(8))
    assert int(state.method_state["nested"][1]["counter"]) == 17
    assert not state.method_state["nested"][0].flags.writeable
    np.testing.assert_array_equal(state.method_state["jax"], [2., 3.])
    assert state.method_state["absent"] is None


def test_host_method_state_snapshot_preserves_wide_integer_with_x64_disabled():
    previous = jax.config.x64_enabled
    counter = np.array(2**40 + 7, dtype=np.uint64)
    try:
        jax.config.update("jax_enable_x64", False)
        state = make_state([0.], [0.], [1., 0.], method_state={"counter": counter})
        counter[...] = 0
        assert isinstance(state.method_state["counter"], np.ndarray)
        assert state.method_state["counter"].dtype == np.dtype("uint64")
        assert int(state.method_state["counter"]) == 2**40 + 7
        assert state.q.dtype == jnp.float32
    finally:
        jax.config.update("jax_enable_x64", previous)


@pytest.mark.parametrize("block", [False, True])
def test_recorded_initialize_snapshots_electronic_input(block):
    path = FixedBasisElectronicPath([0., 1.], np.zeros((2, 16, 16)))
    simulation = RecordedCPA(path, Integrator(.1))
    coefficients = np.arange(16, dtype=np.complex128) + .3j
    if block:
        coefficients = np.column_stack((coefficients, coefficients.conj()))
    expected = coefficients.copy()
    state = simulation.initialize(coefficients)
    jax.block_until_ready(state)
    coefficients[...] = 0
    np.testing.assert_array_equal(state.electronic, expected)
    result = simulation.run(state, 1)
    np.testing.assert_allclose(result.final_state.electronic, expected, atol=1e-14)


def test_mass_weighted_modes_snapshot_caller_arrays_and_compiled_transform():
    reference = np.arange(24, dtype=np.float64).reshape(8, 3)
    masses = np.linspace(1., 2., 8)
    modes = np.eye(24, 3)
    coordinates = MassWeightedModes(reference, masses, modes)
    q = jnp.array([.1, -.2, .3])
    compiled = jax.jit(coordinates.to_cartesian)
    expected = np.asarray(compiled(q)).copy()
    for array in (reference, masses, modes):
        array[...] = 7
    np.testing.assert_array_equal(coordinates.to_cartesian(q), expected)
    np.testing.assert_array_equal(compiled(q), expected)
    np.testing.assert_array_equal(jax.jit(coordinates.to_cartesian)(q), expected)
    np.testing.assert_allclose(coordinates.to_modes(expected), q, atol=1e-14)
