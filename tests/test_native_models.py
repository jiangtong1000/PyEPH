"""Independent checks of the model contracts and physically defined probes."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pyeph.core.contracts import LowRankWeight, ProbeContext, pure_state_weight
from pyeph.models.aggregate import AggregateModel, smooth_switch
from pyeph.models.analytic import SpinBosonModel, TullyModel
from pyeph.models.composite import ReferenceShiftModel, SumModel
from pyeph.models.epc import EdgeEPCModel, LinearEPCModel
from pyeph.models.neural import NeuralResidualModel
from pyeph.models.periodic import PeriodicBlockModel


def finite_gradient(function, q, delta=1e-6):
    q = np.array(q, dtype=float)
    gradient = np.zeros_like(q)
    for index in np.ndindex(q.shape):
        step = np.zeros_like(q)
        step[index] = delta
        gradient[index] = (function(jnp.asarray(q + step)) - function(jnp.asarray(q - step))) / (2 * delta)
    return gradient


def check_action_and_gradient(model, params, q):
    rng = np.random.default_rng(71)
    block = jnp.asarray(rng.normal(size=(model.nstates, 3)) + 1j * rng.normal(size=(model.nstates, 3)))
    matrix = np.asarray(model.dense(params, q))
    np.testing.assert_allclose(matrix, matrix.conj().T, atol=1e-13)
    for v in (block, block[:, 0]):
        np.testing.assert_allclose(model.apply(params, q, v), matrix @ v, atol=1e-13)
        np.testing.assert_allclose(jax.jit(model.apply)(params, q, v), matrix @ v, atol=1e-13)
    left, right = block[:, :2], block[:, 1:]
    weight = LowRankWeight(left, right)
    dense_weight = left @ right.conj().T
    gradient = model.contract_gradient(params, q, weight)
    expected = finite_gradient(lambda x: np.real(np.vdot(dense_weight, model.dense(params, x))), q)
    np.testing.assert_allclose(gradient, expected, rtol=2e-6, atol=2e-9)
    np.testing.assert_allclose(gradient, model.contract_gradient(params, q, dense_weight), atol=1e-12)
    np.testing.assert_allclose(model.reference_gradient(params, q),
                               finite_gradient(lambda x: model.reference_energy(params, x), q), atol=1e-9)


@pytest.mark.parametrize("kind", (1, 2, 3))
def test_tully_models_action_force_and_known_values(kind):
    model = TullyModel(kind)
    q = jnp.array([0.37])
    check_action_and_gradient(model, None, q)
    h0 = model.dense(None, jnp.array([0.0]))
    if kind == 1:
        np.testing.assert_allclose(h0, [[0, 0.005], [0.005, 0]])
        slope = jax.jacfwd(lambda x: model.dense(None, x))(jnp.array([0.0]))
        np.testing.assert_allclose(slope[:, :, 0], [[0.016, 0], [0, -0.016]], atol=1e-14)
    elif kind == 2:
        np.testing.assert_allclose(h0, [[0, 0.015], [0.015, -0.05]])
    else:
        np.testing.assert_allclose(h0, [[0.0006, 0.1], [0.1, -0.0006]])


def test_spin_boson_canonical_force():
    model = SpinBosonModel(3)
    p = model.default_params() | dict(omega=jnp.array([0.2, 0.3, 0.7]),
                                     coupling=jnp.array([0.1, -0.2, 0.07]),
                                     q_eq=jnp.array([0.1, 0.0, -0.2]))
    q = jnp.array([0.4, -0.2, 0.7])
    check_action_and_gradient(model, p, q)
    np.testing.assert_allclose(model.reference_gradient(p, q), p["omega"]**2 * (q - p["q_eq"]))


def epc_fixture():
    model = EdgeEPCModel(nstates=3, nmodes=2, edges=((0, 1), (0, 2), (1, 2)), complex_valued=True)
    p = model.default_params() | dict(
        onsite=jnp.array([0.1, -0.2, 0.7]), hopping=jnp.array([0.05j, 0.03-0.04j, 0.1]),
        onsite_coupling=jnp.array([[0.1, 0.0, -0.1], [0.2, 0.3, 0.4]]),
        hopping_coupling=jnp.array([[0.01j, 0.02, 0.03j], [-0.03, 0.01j, 0.04]]),
        omega=jnp.array([0.3, 0.7]))
    return model, p, jnp.array([0.1, -0.2])


def test_edge_epc_matches_independent_dense_reference():
    model, p, q = epc_fixture()
    h0 = np.diag(np.asarray(p["onsite"])).astype(complex)
    g = np.array([np.diag(row) for row in p["onsite_coupling"]], dtype=complex)
    for index, (i, j) in enumerate(model.edges):
        h0[i, j], h0[j, i] = p["hopping"][index], p["hopping"][index].conj()
        g[:, i, j] = p["hopping_coupling"][:, index]
        g[:, j, i] = p["hopping_coupling"][:, index].conj()
    dense = LinearEPCModel(nstates=3, nmodes=2, complex_valued=True)
    dp = dense.create_params(h0, g, omega=p["omega"])
    np.testing.assert_allclose(model.dense(p, q), h0 + np.tensordot(q, g, axes=1))
    np.testing.assert_allclose(model.dense(p, q), dense.dense(dp, q))
    check_action_and_gradient(model, p, q)
    check_action_and_gradient(dense, dp, q)


def test_complex_offdiagonal_derivatives_not_discarded():
    model, p, q = epc_fixture()
    w = jnp.zeros((3, 3), dtype=complex).at[0, 1].set(1j)
    np.testing.assert_allclose(model.contract_gradient(p, q, w), [0.01, 0.0], atol=1e-14)


def aggregate_fixture():
    model = AggregateModel(3, ((0, 1), (0, 2), (1, 2)), cutoff=5, switch_on=3,
                           complex_valued=True)
    p = model.default_params() | dict(hopping=jnp.array([0.03+0.02j, 0.04-0.01j, -0.03j]),
                                     environment=jnp.array([[0.1, -0.2], [0.03, 0.02], [-0.1, 0.1]]),
                                     spring=jnp.array([0.1, 0.2, 0.3]))
    q = jnp.array([[0.2, 0.1, 0.3], [2.1, 0.4, 0.2], [0.7, 2.8, 1.1]])
    return model, p, q


def test_aggregate_nonlinear_action_complete_forces():
    model, p, q = aggregate_fixture()
    check_action_and_gradient(model, p, q)
    # Electronic energy depends on relative geometry; the external harmonic
    # reference is intentionally anchored and is not claimed invariant here.
    np.testing.assert_allclose(model.dense(p, q), model.dense(p, q + jnp.array([1.2, -0.5, 0.7])), atol=1e-13)
    angle = 0.31
    rot = jnp.array([[jnp.cos(angle), -jnp.sin(angle), 0], [jnp.sin(angle), jnp.cos(angle), 0], [0, 0, 1]])
    np.testing.assert_allclose(model.dense(p, q), model.dense(p, q @ rot), atol=1e-13)


def test_aggregate_currents_continuity_and_moving_centers():
    model, p, q = aggregate_fixture()
    c = jnp.array([1, 2j, 1-1j]) / jnp.sqrt(7.)
    h = model.dense(p, q)
    dc = -1j * h @ c
    dpop = 2 * jnp.real(c.conj() * dc)
    velocity = jnp.array([[0.1, 0.2, 0.3], [-0.1, 0.1, 0.0], [0.2, 0.0, -0.3]])
    for a, name in enumerate("xyz"):
        hopping = model.probe_apply(p, ProbeContext(q), f"current_{name}", c)
        lab = model.probe_apply(p, ProbeContext(q, velocity), f"lab_current_{name}", c)
        np.testing.assert_allclose(jnp.vdot(c, hopping).real, model.charge * jnp.dot(dpop, q[:, a]), atol=1e-13)
        np.testing.assert_allclose(jnp.vdot(c, lab).real,
                                   model.charge * (jnp.dot(dpop, q[:, a]) + jnp.dot(abs(c)**2, velocity[:, a])), atol=1e-13)
    with pytest.raises(ValueError, match="requires coordinate velocities"):
        model.probe_apply(p, ProbeContext(q), "lab_current_x", c)


def test_single_translating_site_has_laboratory_current_only():
    model = AggregateModel(1, ())
    q = jnp.array([[2., 1., 0.]])
    c = jnp.array([1.+0j])
    np.testing.assert_allclose(model.probe_apply(None, ProbeContext(q), "current_x", c), 0)
    np.testing.assert_allclose(model.probe_apply(None, ProbeContext(q, jnp.array([[0.4, 0., 0.]])), "lab_current_x", c), -0.4)


@pytest.mark.parametrize("r", (3.0, 5.0))
def test_cutoff_has_continuous_first_and_second_derivatives(r):
    def f(x):
        return smooth_switch(x, 3., 5.)
    assert abs(float(jax.grad(f)(r))) < 1e-12
    assert abs(float(jax.grad(jax.grad(f))(r))) < 1e-12
    for dx in (-1e-5, 1e-5):
        assert abs(float(jax.grad(f)(r + dx))) < 1e-8


def periodic_fixture():
    model = PeriodicBlockModel(2, 2, ((0, 1, 0, 0, 0), (0, 1, -1, 0, 0), (1, 1, 0, 1, 0)),
                               ((4., 0., 0.), (0., 4., 0.), (0., 0., 5.)), cutoff=6., switch_on=5.)
    p = model.default_params()
    p["onsite"] = jnp.array([[[0.1, 0.02j], [-0.02j, -0.1]], [[0.2, 0.01], [0.01, 0.3]]])
    p["hopping"] = jnp.array([[[0.02, 0.01j], [0.03j, 0.04]], [[0.01j, -0.01], [0.02, 0.01j]], [[0.02, 0], [0.01j, 0.03]]])
    p["decay"] = jnp.array([0.3, 0.2, 0.1])
    p["onsite_derivative"] = p["onsite_derivative"].at[0, 1, 0, 0].set(0.03)
    p["spring"] = jnp.array([0.1, 0.2])
    q = jnp.array([[0.1, 0.2, 0.3], [1.8, 0.3, 0.1]])
    return model, p, q


def test_periodic_blocks_action_complete_derivatives_and_current():
    model, p, q = periodic_fixture()
    check_action_and_gradient(model, p, q)
    identity = jnp.eye(model.nstates)
    for axis, name in enumerate("xyz"):
        step = jnp.eye(3)[axis] * 1e-6
        finite = model.charge * (model.apply_peierls(p, q, step, identity) - model.apply_peierls(p, q, -step, identity)) / 2e-6
        current = model.probe_apply(p, ProbeContext(q), f"current_{name}", identity)
        np.testing.assert_allclose(current, finite, atol=1e-11, rtol=1e-8)
        np.testing.assert_allclose(current, current.conj().T, atol=1e-13)


def test_periodic_cell_wrapping_preserves_operator_and_current():
    model, p, q = periodic_fixture()
    shifts = jnp.array([[1, 0, 0], [-1, 1, 0]])
    wrapped = model.rewrapped(shifts)
    offset = shifts @ jnp.asarray(model.cell)
    wp = p | dict(reference_positions=p["reference_positions"] + offset)
    np.testing.assert_allclose(model.dense(p, q), wrapped.dense(wp, q + offset), atol=1e-13)
    np.testing.assert_allclose(model.reference_energy(p, q), wrapped.reference_energy(wp, q + offset), atol=1e-13)
    for name in "xyz":
        np.testing.assert_allclose(model.probe_apply(p, ProbeContext(q), f"current_{name}", jnp.eye(4)),
                                   wrapped.probe_apply(wp, ProbeContext(q + offset), f"current_{name}", jnp.eye(4)), atol=1e-13)


def test_periodic_pristine_supercell_matches_bloch_grid_and_disorder_mixes_k():
    n, t, a = 5, 0.17, 2.
    unit = PeriodicBlockModel(1, 1, ((0, 0, 1, 0, 0),), ((a, 0, 0), (0, 8, 0), (0, 0, 8)))
    up = unit.default_params() | dict(hopping=jnp.array([[[t]]]))
    k = 2 * np.pi * np.arange(n) / (n * a)
    band = np.array([unit.dense_bloch(up, jnp.zeros((1, 3)), jnp.array([ki, 0, 0]))[0, 0].real for ki in k])
    edges = tuple((i, i+1, 0, 0, 0) for i in range(n-1)) + ((0, n-1, -1, 0, 0),)
    supercell = PeriodicBlockModel(n, 1, edges, ((n*a, 0, 0), (0, 8, 0), (0, 0, 8)))
    sp = supercell.default_params() | dict(hopping=jnp.full((n, 1, 1), t))
    q = jnp.zeros((n, 3)).at[:, 0].set(jnp.arange(n) * a)
    h = supercell.dense(sp, q)
    np.testing.assert_allclose(np.linalg.eigvalsh(h), np.sort(band), atol=1e-13)
    u = np.exp(1j * np.outer(np.arange(n) * a, k)) / np.sqrt(n)
    np.testing.assert_allclose(u.conj().T @ h @ u, np.diag(band), atol=1e-13)
    dp = sp | dict(onsite=sp["onsite"].at[2, 0, 0].set(0.3))
    momentum = u.conj().T @ supercell.dense(dp, q) @ u
    assert np.max(np.abs(momentum - np.diag(np.diag(momentum)))) > 0.05


def test_baseline_neural_residual_and_compensated_reference_shift():
    base, p, q = aggregate_fixture()
    nn = NeuralResidualModel(nstates=3, q_shape=(3, 3), hidden_sizes=(5,), complex_valued=True)
    np0 = nn.init_params(jax.random.key(4))
    model = SumModel((base, nn))
    np.testing.assert_allclose(model.dense((p, np0), q), base.dense(p, q), atol=0)
    neural_params = nn.init_params(jax.random.key(4), zero_last=False)
    params = (p, neural_params)
    check_action_and_gradient(model, params, q)
    shift = ReferenceShiftModel(model, lambda sp, x: sp["a"] * jnp.sum(x**2))
    sp = (params, dict(a=0.2))
    c = jnp.array([1, 2j, 1-1j]) / jnp.sqrt(7.)
    w = pure_state_weight(c)
    def total_gradient(m, p):
        return m.reference_gradient(p, q) + m.contract_gradient(p, q, w)
    np.testing.assert_allclose(total_gradient(model, params), total_gradient(shift, sp), atol=1e-12)
    original_energy = model.reference_energy(params, q) + jnp.vdot(c, model.apply(params, q, c)).real
    shifted_energy = shift.reference_energy(sp, q) + jnp.vdot(c, shift.apply(sp, q, c)).real
    np.testing.assert_allclose(original_energy, shifted_energy, atol=1e-12)
    # Negative control: trace centering without compensating V_ref changes forces.
    assert np.linalg.norm(model.contract_gradient(params, q, w) - shift.contract_gradient(sp, q, w)) > 0.1
    assert model.spec.probes == ()  # H-only residual does not invent a current.


def test_invalid_static_models_rejected():
    with pytest.raises(ValueError, match="duplicate"):
        AggregateModel(2, ((0, 1), (0, 1)))
    with pytest.raises(ValueError, match="canonical"):
        EdgeEPCModel(2, 1, ((1, 0),))
    with pytest.raises(ValueError, match="nonzero image"):
        PeriodicBlockModel(1, 1, ((0, 0, 0, 0, 0),), np.eye(3))
    with pytest.raises(ValueError, match="coordinates and basis"):
        SumModel((TullyModel(), NeuralResidualModel(2, (1,))))
    with pytest.raises(ValueError, match="Hermitian"):
        LinearEPCModel(2, 1).create_params([[0, 1], [0, 0]], np.zeros((1, 2, 2)))
    with pytest.raises(ValueError, match="exact integers"):
        PeriodicBlockModel(2, 1, ((0, 1, 0.2, 0, 0),), np.eye(3))
    with pytest.raises(ValueError, match="exact integers"):
        AggregateModel(2, ((0.5, 1),))


@pytest.mark.parametrize("kind", (1, 2, 3))
def test_tully_large_coordinates_have_finite_values_and_gradients(kind):
    model = TullyModel(kind)
    q = jnp.array([[-1000.], [0.], [1000.]])
    values = jax.jit(jax.vmap(lambda x: model.dense(None, x)))(q)
    gradients = jax.jit(jax.vmap(jax.jacfwd(lambda x: model.dense(None, x))))(q)
    assert np.all(np.isfinite(values))
    assert np.all(np.isfinite(gradients))


def test_aggregate_rephasing_preserves_physical_observables_and_forces():
    model, p, q = aggregate_fixture()
    phase = jnp.exp(1j * jnp.array([0.2, -0.7, 1.3]))
    e = np.asarray(model.edges)
    pp = p | dict(hopping=p["hopping"] * phase[e[:, 0]] * phase[e[:, 1]].conj())
    d = jnp.diag(phase)
    np.testing.assert_allclose(model.dense(pp, q), d @ model.dense(p, q) @ d.conj().T, atol=1e-13)
    c = jnp.array([1, 2j, 1-1j]) / jnp.sqrt(7.)
    np.testing.assert_allclose(model.contract_gradient(pp, q, pure_state_weight(phase*c)),
                               model.contract_gradient(p, q, pure_state_weight(c)), atol=1e-13)
    for name in "xyz":
        jp = model.probe_apply(pp, ProbeContext(q), f"current_{name}", phase*c)
        j = model.probe_apply(p, ProbeContext(q), f"current_{name}", c)
        np.testing.assert_allclose(jnp.vdot(phase*c, jp), jnp.vdot(c, j), atol=1e-13)


def test_parameter_validation_prevents_broadcasting_and_false_real_declaration():
    aggregate = AggregateModel(2, ((0, 1),))
    p = aggregate.default_params()
    with pytest.raises(ValueError, match="onsite.*shape"):
        aggregate.validate_params(p | dict(onsite=jnp.zeros(1)))
    with pytest.raises(ValueError, match="hopping.*real"):
        aggregate.validate_params(p | dict(hopping=jnp.ones(1, dtype=complex)))
    aggregate.validate_params(p)
    with pytest.raises(ValueError, match="coincide"):
        aggregate.validate_geometry(jnp.zeros((2, 3)))
    periodic, pp, q = periodic_fixture()
    periodic.validate_params(pp)
    with pytest.raises(ValueError, match="Hermitian"):
        periodic.validate_params(pp | dict(onsite=pp["onsite"].at[0, 0, 1].set(2.)))
    nn = NeuralResidualModel(2, (1,))
    pn = nn.init_params(jax.random.key(0))
    nn.validate_params(pn)
    with pytest.raises(ValueError, match="positive"):
        nn.validate_params(pn | dict(q_scale=jnp.zeros(1)))


def test_fixed_graph_zero_crossing_and_batch_match_single_actions():
    model, p, q = epc_fixture()
    qs = jnp.stack([q, -q, q*0])
    c = jnp.array([1., 2j, -1.]) / jnp.sqrt(6.)
    actions = jax.jit(jax.vmap(lambda x: model.apply(p, x, c)))(qs)
    np.testing.assert_allclose(actions, jnp.stack([model.apply(p, x, c) for x in qs]), atol=1e-13)
    # The edge remains present when its coefficient crosses zero.
    pp = p | dict(hopping=p["hopping"].at[0].set(0),
                  hopping_coupling=p["hopping_coupling"].at[:, 0].set(jnp.array([1j, 0.])))
    left = model.dense(pp, jnp.array([-1e-7, 0.]))[0, 1]
    middle = model.dense(pp, jnp.array([0., 0.]))[0, 1]
    right = model.dense(pp, jnp.array([1e-7, 0.]))[0, 1]
    np.testing.assert_allclose([left, middle, right], [-1e-7j, 0, 1e-7j], atol=1e-16)
