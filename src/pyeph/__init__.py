"""Electron–nuclear dynamics with explicit physical contracts."""

__version__ = "0.1.0.dev2"

from pyeph.core.problem import CoupledClassical, PrescribedPath, Problem
from pyeph.core.state import TrajectoryState, make_state, stack_states
from pyeph.dynamics.cpa import CPA
from pyeph.dynamics.ehrenfest import Ehrenfest
from pyeph.dynamics.mash2 import MASH2, MASHPopulation
from pyeph.dynamics.mashrm import MASHRM, MASHRMPopulation
from pyeph.dynamics.recorded import RecordedCPA
from pyeph.execution.runner import Execution
from pyeph.integrators.electronic import Integrator
from pyeph.integrators.krylov import LanczosOptions
from pyeph.simulation import Simulation

__all__ = ["Problem", "CoupledClassical", "PrescribedPath", "TrajectoryState",
           "make_state", "stack_states", "CPA", "Ehrenfest", "MASH2", "MASHPopulation",
           "MASHRM", "MASHRMPopulation",
           "RecordedCPA", "Execution",
           "Integrator", "LanczosOptions", "Simulation", "configure_precision"]


def configure_precision(enable_x64: bool = True) -> None:
    """Configure JAX precision explicitly, before array creation/compilation."""
    import jax

    jax.config.update("jax_enable_x64", enable_x64)
