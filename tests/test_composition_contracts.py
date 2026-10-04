import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pyeph import CoupledClassical, Ehrenfest, Integrator, Problem, Simulation, make_state
from pyeph.core.contracts import pure_state_weight
from pyeph.execution.runner import SimulationError
from pyeph.models.analytic import SpinBosonModel
from pyeph.models.composite import ReferenceShiftModel, SumModel


@pytest.mark.parametrize("shift", [lambda p, q: q, lambda p, q: 1j*jnp.sum(q)])
def test_reference_shift_cannot_broadcast_a_vector_or_break_hermiticity(shift):
    base = SpinBosonModel()
    model = ReferenceShiftModel(base, shift)
    params = (base.default_params(), None)
    q, c = jnp.array([.2]), jnp.array([1., 0.])
    with pytest.raises(ValueError, match="one real scalar"):
        jax.jit(model.apply)(params, q, c)
    with pytest.raises(ValueError, match="one real scalar"):
        model.reference_energy(params, q)
    with pytest.raises(ValueError, match="one real scalar"):
        model.contract_gradient(params, q, pure_state_weight(c))


def test_nonfinite_shift_is_rejected_as_failed_dynamics():
    base = SpinBosonModel()
    model = ReferenceShiftModel(base, lambda p, q: jnp.asarray(jnp.inf))
    params = (base.default_params(), None)
    assert np.isnan(model.reference_energy(params, jnp.array([.2])))
    simulation = Simulation(Problem(model, params, CoupledClassical(1.), Ehrenfest()), Integrator(.01))
    with pytest.raises(SimulationError, match="nonfinite"):
        simulation.run(make_state([.2], [.1], [1, 0]), 1, collect=False)


@pytest.mark.parametrize("wrapper", ["sum", "shift"])
def test_wrappers_do_not_invent_force_capability_from_only_a_spec_flag(wrapper):
    native = SpinBosonModel()
    class ValuesOnly:
        spec = native.spec

        def apply(self, params, q, vectors):
            return native.apply(params, q, vectors)

        def reference_energy(self, params, q):
            return native.reference_energy(params, q)

    model = (SumModel((ValuesOnly(), native)) if wrapper == "sum" else
             ReferenceShiftModel(ValuesOnly(), lambda p, q: jnp.sum(q*q)))
    params = ((native.default_params(), native.default_params()) if wrapper == "sum"
              else (native.default_params(), None))
    assert not model.spec.force_support
    with pytest.raises(ValueError, match="complete reference and electronic derivatives"):
        Simulation(Problem(model, params, CoupledClassical(1.), Ehrenfest()), Integrator(.01))
