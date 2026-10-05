"""Reject an internal coordinate-domain crossing, then explicitly replay.

Run: JAX_ENABLE_X64=1 python examples/guarded_dynamics.py

This generated, globally analytic spin-boson model uses atomic units and a
mass-weighted classical coordinate. Its complete reference potential and
electronic Hamiltonian remain the same in every run. A box is a declared
coordinate domain; it supplies neither a neighbor-coverage certificate nor a
learned model's accuracy or uncertainty estimate.

The wider-box run below is an explicit new calculation on this known analytic
model. An unknown fitted provider needs independent domain validation before
its allowed coordinates can be enlarged.
"""

from dataclasses import replace
import json

import jax
import jax.numpy as jnp
import numpy as np

from pyeph import Execution, Integrator, Simulation
from pyeph.core.geometry import CoordinateBox
from pyeph.core.problem import CoupledClassical, Problem
from pyeph.core.state import make_state
from pyeph.dynamics.cpa import CPA
from pyeph.dynamics.ehrenfest import Ehrenfest
from pyeph.execution.runner import SimulationError
from pyeph.integrators.krylov import LanczosOptions
from pyeph.models.analytic import SpinBosonModel
from pyeph.paths.harmonic import HarmonicBath


def run(method_name):
    """Return bounded rejection/replay evidence for one dynamics method."""
    if method_name not in {"cpa", "ehrenfest"}:
        raise ValueError("method must be cpa or ehrenfest")
    if not jax.config.x64_enabled:
        raise ValueError("enable JAX x64 before running this checked example")
    model = SpinBosonModel(nmodes=1)
    params = dict(omega=jnp.array([0.5]), coupling=jnp.array([0.05]),
                  q_eq=jnp.zeros(1), bias=0.01, delta=0.02,
                  reference_offset=0.0)
    method = CPA() if method_name == "cpa" else Ehrenfest()
    nuclei = HarmonicBath([0.5]) if method_name == "cpa" else CoupledClassical(1.0)
    initial = make_state([0.0], [1.0], [1.0, 0.0], trajectory_id=7, seed=13)
    problem = Problem(model, params, nuclei, method,
                      geometry_guard=CoordinateBox([-0.12], [0.12]))
    integrator = Integrator(0.1, electronic=LanczosOptions(
        max_dimension=2, atol=1e-12, rtol=1e-11))
    execution = Execution(chunk_size=4, save_every=1)
    published = []

    def observe(times, values):
        published.extend(np.asarray(times).tolist())

    try:
        Simulation(problem, integrator, execution).run(initial, 4, observer=observe)
    except SimulationError as error:
        info = error.diagnostics["step_info"] if error.diagnostics else None
        if info is None or int(info.code) != 4:
            raise
        rollback = error.last_valid_state
        for actual, expected in zip(jax.tree.leaves(rollback),
                                    jax.tree.leaves(initial), strict=True):
            if np.asarray(actual).tobytes() != np.asarray(expected).tobytes():
                raise AssertionError("a rejected chunk changed the committed state")
        if published != [0.0]:
            raise AssertionError("a rejected chunk published observations")
        rejection_phase = int(info.phase)
        rejected_macrostep = int(error.diagnostics["failed_macro_index"])
    else:
        raise AssertionError("the narrow domain should reject an internal stage")

    # This is a deliberate configuration change, not automatic domain recovery.
    wider = replace(problem, geometry_guard=CoordinateBox([-2.0], [2.0]))
    resumed = Simulation(wider, integrator, execution).run(rollback, 4)
    reference = Simulation(replace(problem, geometry_guard=None),
                           integrator, execution).run(initial, 4)
    maximum_difference = 0.0
    for actual, expected in zip(jax.tree.leaves(resumed.final_state),
                                jax.tree.leaves(reference.final_state), strict=True):
        left, right = np.asarray(actual), np.asarray(expected)
        if left.dtype.kind in "iu":
            np.testing.assert_array_equal(left, right)
        else:
            np.testing.assert_allclose(left, right, rtol=1e-12, atol=1e-13)
            maximum_difference = max(maximum_difference,
                                     float(np.max(np.abs(left-right))))
    np.testing.assert_array_equal(resumed.times, reference.times)
    np.testing.assert_allclose(resumed.observables["population"],
                               reference.observables["population"],
                               rtol=1e-12, atol=1e-13)
    return dict(method=method_name, rejected_phase=rejection_phase,
                rejected_macrostep=rejected_macrostep,
                published_times_before_rejection=published,
                all_rollback_fields_byte_exact=True,
                final_step=int(resumed.final_state.step),
                final_time=float(resumed.final_state.time),
                maximum_state_difference=maximum_difference,
                scope="scalar checked coordinate domain; explicit analytic-model replay")


if __name__ == "__main__":
    print(json.dumps([run(name) for name in ("cpa", "ehrenfest")], indent=2))
