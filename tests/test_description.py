"""Calculation descriptions expose conventions without executing a provider."""

from dataclasses import replace
import json

import pytest

from pyeph import (CoupledClassical, Ehrenfest, Integrator, MASHRM, MASHRMPopulation,
                   Problem, Simulation)
from pyeph.core.units import UnitSystem
from pyeph.models.analytic import SpinBosonModel


def test_description_does_not_evaluate_model_or_expose_parameters(monkeypatch):
    model = SpinBosonModel(nmodes=1)
    simulation = Simulation(Problem(model, model.default_params(), CoupledClassical(1.),
                                    Ehrenfest()), Integrator(.01))
    def forbidden(*args, **kwargs):
        raise AssertionError("description must not call a model")
    monkeypatch.setattr(type(model), "apply", forbidden)
    monkeypatch.setattr(type(model), "reference_energy", forbidden)
    result = simulation.describe()
    assert result["model"]["basis_kind"] == "fixed_orthonormal"
    assert result["method"]["electronic_spectrum"] == "operator_action_sufficient"
    assert result["model"]["electronic_states"] == 2
    assert "params" not in result
    json.dumps(result, allow_nan=False)


def test_mapping_scope_and_reduced_units_are_explicit():
    class ReducedSpinBoson(SpinBosonModel):
        def __post_init__(self):
            super().__post_init__()
            object.__setattr__(self, "spec", replace(self.spec, unit_system=UnitSystem(.2, 3.)))

    model = ReducedSpinBoson(nmodes=1)
    simulation = Simulation(Problem(model, model.default_params(), CoupledClassical(1.),
                                    MASHRM(), MASHRMPopulation()),
                            Integrator(.01, "exponential_midpoint"))
    result = simulation.describe()
    assert result["method"]["electronic_spectrum"] == "complete_isolated_real"
    assert result["method"]["trajectory_derivatives"] == "unsupported"
    assert result["units"]["automatic_input_conversion"] is False
    assert result["integrator"]["electronic_linear_algebra"] == "dense_eigendecomposition"
    assert result["units"]["time_fs"] == pytest.approx(UnitSystem(.2, 3.).time_fs)
