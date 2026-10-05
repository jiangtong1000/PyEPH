"""Readable declared calculation scope without evaluating the physical model."""

from dataclasses import asdict

import jax

from pyeph.dynamics.cpa import CPA
from pyeph.dynamics.ehrenfest import Ehrenfest
from pyeph.dynamics.mash2 import MASH2
from pyeph.dynamics.mashrm import MASHRM
from pyeph.integrators.krylov import LanczosOptions


def describe_simulation(simulation):
    """Return JSON-ready configuration, separate from a restart identity.

    No provider is evaluated, weights are not exposed, and no device is selected.
    The report records declarations and implemented method scope; it does not
    certify model accuracy, runtime graph coverage, or future trajectories.
    """
    problem = simulation.problem
    spec = problem.model.spec
    system = spec.system
    method = problem.method
    electronic = simulation.integrator.electronic
    if type(method) is MASHRM:
        scope = dict(electronic_spectrum="complete_isolated_real",
                     preparation="conditional mapping sphere or qualified canonical workflow",
                     trajectory_derivatives="unsupported",
                     equilibrium="workflow-specific")
    elif type(method) is MASH2:
        scope = dict(electronic_spectrum="two_isolated_real_states",
                     preparation="method-specific mapping sphere",
                     trajectory_derivatives="unsupported",
                     equilibrium="workflow-specific")
    elif type(method) in (CPA, Ehrenfest):
        scope = dict(electronic_spectrum="operator_action_sufficient",
                     preparation="caller-owned physical ensemble",
                     trajectory_derivatives="pure native smooth kernels only",
                     equilibrium="not guaranteed by selecting this method")
    else:
        scope = dict(electronic_spectrum="custom method contract",
                     preparation="custom method contract",
                     trajectory_derivatives="custom method contract",
                     equilibrium="custom method contract")
    return {
        "schema": "pyeph.calculation-description.v1",
        "model": {"type": type(problem.model).__name__, "name": spec.name,
                  "electronic_states": system.nstates, "coordinates": list(system.q_shape),
                  "coordinate_kind": system.coordinate_kind, "basis_id": system.basis_id,
                  "basis_kind": spec.basis_kind, "electronic_sector": spec.electronic_sector,
                  "energy_convention": spec.energy_convention,
                  "forces_declared": spec.force_support, "complex_declared": spec.complex_valued,
                  "provider_execution": "native_jax" if spec.native_jax else "host_callback",
                  "probes": list(spec.probes)},
        "units": {"energy_hartree": spec.unit_system.energy_hartree,
                  "length_bohr": spec.unit_system.length_bohr,
                  "time_fs": spec.unit_system.time_fs,
                  "automatic_input_conversion": False},
        "nuclear_treatment": type(problem.nuclear_treatment).__name__,
        "method": {"type": type(method).__name__, **scope},
        "integrator": {"dt": simulation.integrator.dt,
                       "electronic": ("checked_lanczos" if isinstance(electronic, LanczosOptions)
                                      else electronic),
                       "electronic_linear_algebra": (
                           "dense_eigendecomposition" if electronic == "exponential_midpoint"
                           else "operator_action"),
                       "electronic_substeps": simulation.integrator.electronic_substeps},
        "geometry_guard": (None if problem.geometry_guard is None else {
            "type": "CoordinateBox", "lower": list(problem.geometry_guard.lower),
            "upper": list(problem.geometry_guard.upper), "shape": list(problem.geometry_guard.shape),
            "scope": "declared coordinate domain only; no neighbor coverage or automatic recovery",
            "execution": "scalar checked CPA/Ehrenfest; ElectronicPopulation only"}),
        "measurement": type(simulation.measurement).__name__,
        "execution": asdict(simulation.execution),
        "precision": {"jax_x64_enabled": bool(jax.config.x64_enabled)},
        "restart": "strict numerical, configuration, source and runtime identity",
    }
