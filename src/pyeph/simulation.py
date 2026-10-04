"""Public orchestration entry point, deliberately thin over the runner."""

from pyeph.execution.runner import Execution, RunResult, Runner


class Simulation(Runner):
    """Assemble a validated Problem, Integrator and Execution policy, then run it."""


__all__ = ["Simulation", "Execution", "RunResult"]
