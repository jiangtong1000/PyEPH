"""Nonlinear canonical target, independent quadrature and explicit chain recovery."""

from dataclasses import dataclass, replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import expit

from pyeph import Execution, Integrator, MASHRM, MASHRMPopulation, Problem, Simulation, CoupledClassical
from pyeph.core.contracts import ModelSpec
from pyeph.core.system import SystemSpec
from pyeph.dynamics.mashrm_mapping import mapping_populations
from pyeph.models.base import AutoDiffModel
from pyeph.workflows.canonical_metropolis import (
    MixingWarning, MetropolisSamplingError, NativeCanonicalMetropolis,
)


@dataclass(frozen=True)
class ConfinedNonlinear(AutoDiffModel):
    spec = ModelSpec(SystemSpec(2, (1,), coordinate_kind="canonical"), name="confined-nonlinear")
    invalid_outside: float = np.inf
    singular_force: bool = False

    def apply(self, params, q, vectors):
        x = q[0]
        bias = params["bias"]+.5*jnp.tanh(x)+.05*x*x
        coupling = params["delta"]+.1*jnp.cos(x)
        h = jnp.array([[bias, coupling], [coupling, -bias]])+params["shift"]*jnp.eye(2)
        return jnp.where(abs(x) <= self.invalid_outside, h, jnp.nan) @ vectors

    def reference_energy(self, params, q):
        x = q[0]
        value = .3*x*x+.12*x**4+params["offset"]
        return value + (jnp.sqrt(x*x) if self.singular_force else 0.)


def sampler(**options):
    params = {"bias": .2, "delta": .3, "shift": 0., "offset": 0.}
    params.update(options.pop("params", {}))
    model = options.pop("model", ConfinedNonlinear())
    return NativeCanonicalMetropolis(model, params, 1.7, 1.4,
                                      options.pop("proposal_scale", .7), [0.],
                                      artifact_ids={"model": "analytic-confined-nonlinear-v1"}, **options)


def scalar_reference(x):
    """Independent explicit two-by-two eigenvalue formula; no package/model calls."""
    bias = .2+.5*np.tanh(x)+.05*x*x
    coupling = .3+.1*np.cos(x)
    radius = np.sqrt(bias*bias+coupling*coupling)
    reference = .3*x*x+.12*x**4
    log_weight = -1.4*reference + np.logaddexp(1.4*radius, -1.4*radius)
    return log_weight, expit(2*1.4*radius)


def quadrature():
    norm = quad(lambda x: np.exp(scalar_reference(x)[0]), -10, 10, epsabs=1e-11)[0]
    moments = [quad(lambda x: x**power*np.exp(scalar_reference(x)[0]), -10, 10,
                     epsabs=1e-11)[0]/norm for power in (1, 2, 4)]
    active = quad(lambda x: scalar_reference(x)[1]*np.exp(scalar_reference(x)[0]),
                   -10, 10, epsabs=1e-11)[0]/norm
    tail_norm = quad(lambda x: np.exp(scalar_reference(x)[0]), -12, 12, epsabs=1e-11)[0]
    assert abs(tail_norm/norm-1) < 2e-12
    return (*moments, active)


def test_target_and_pairwise_detailed_balance_against_independent_scalar_formula():
    s = sampler()
    origin = scalar_reference(0)[0]
    for x, y in [(-1.2, .7), (.3, 1.8), (-.1, -.1), (-2.1, 2.3)]:
        lx, ly = s.log_density([x]), s.log_density([y])
        assert lx-s.log_density([0.]) == pytest.approx(scalar_reference(x)[0]-origin, abs=1e-14)
        log_proposal = -.5*((y-x)/.7)**2-np.log(.7*np.sqrt(2*np.pi))
        forward = scalar_reference(x)[0]+log_proposal+min(0, ly-lx)
        backward = scalar_reference(y)[0]+log_proposal+min(0, lx-ly)
        assert forward == pytest.approx(backward, abs=2e-14)


def test_nonlinear_coordinate_active_mapping_and_momentum_moments_match_quadrature():
    s = sampler()
    count = 4096
    sample = s.sample(np.arange(count), [0.], initialization_id="point-origin-v1",
                       burn_in=500, production_steps=500, thin=10, seed=8129)
    mean, second, fourth, population = quadrature()
    q = np.asarray(sample.state.q)[:, 0]
    assert abs(q.mean()-mean) < 6*np.sqrt((second-mean**2)/count)
    assert abs(np.mean(q*q)-second) < 6*np.sqrt((fourth-second**2)/count)
    active = np.asarray(sample.state.method_state["active"])
    assert abs(np.mean(active == 0)-population) < 6*np.sqrt(population*(1-population)/count)
    # p is freshly and independently drawn, rather than sampled by the q chain.
    momentum = np.asarray(sample.state.p)[:, 0]
    variance = 1.7/1.4
    assert abs(momentum.mean()) < 6*np.sqrt(variance/count)
    assert abs(np.mean(momentum**2)-variance) < 6*variance*np.sqrt(2/count)
    _, _, vectors, _ = s._density(sample.state.q)
    adiabatic = jnp.einsum("bji,bj->bi", vectors, sample.state.electronic)
    mapped = np.asarray(jax.vmap(mapping_populations)(adiabatic))[:, 0]
    assert abs(mapped.mean()-population) < 6*np.sqrt((population+1/12-population**2)/count)
    np.testing.assert_array_equal(np.argmax(abs(adiabatic)**2, axis=1), active)
    assert sample.diagnostics["certifies_equilibrium"] is False
    assert np.max(sample.diagnostics["split_rhat"]) < 1.05


def test_partition_reordering_and_saved_chain_continuation_match_exactly(tmp_path):
    s = sampler()
    ids = np.array([18, 7, 91, 23, 4])
    start = s.start(ids, [-.4], initialization_id="point-minus.4-v1", seed=192)
    full = s.advance(start, 41).chain
    prefix = s.advance(start, 13).chain
    s.save_chain(tmp_path / "chain.npz", prefix)
    restored = s.load_chain(tmp_path / "chain.npz")
    resumed = s.advance(restored, 28).chain
    for name in ("q", "attempts", "accepted", "log_density"):
        np.testing.assert_array_equal(getattr(resumed, name), getattr(full, name))
    prepared = s.finalize(full, burn_in=13)
    for chosen in ([91, 18], [4, 23, 7]):
        partition = s.advance(prefix.subset(chosen), 28).chain
        expected = full.subset(chosen)
        np.testing.assert_array_equal(partition.q, expected.q)
        result = s.finalize(partition, burn_in=13)
        lookup = [np.where(ids == value)[0][0] for value in chosen]
        for a, b in zip(jax.tree.leaves(result.state), jax.tree.leaves(prepared.state), strict=True):
            np.testing.assert_array_equal(a, np.asarray(b)[lookup])
        assert result.metadata["preparation_id"] == prepared.metadata["preparation_id"]


def test_invalid_proposal_is_failure_with_endpoint_and_offending_coordinate_retained(tmp_path):
    s = sampler(model=ConfinedNonlinear(invalid_outside=.1), proposal_scale=100.)
    start = s.start([7, 8], [0.], initialization_id="point-origin-v1", seed=41)
    with pytest.raises(MetropolisSamplingError, match="proposal") as caught:
        s.advance(start, 20)
    chain = caught.value.chain
    np.testing.assert_array_equal(chain.q, start.q)
    assert np.all(chain.status == 1)
    assert np.all(abs(np.asarray(chain.failed_q)[:, 0]) > .1)
    np.testing.assert_array_equal(chain.attempts, 1)
    s.save_chain(tmp_path / "failed.npz", chain)
    restored = s.load_chain(tmp_path / "failed.npz")
    np.testing.assert_array_equal(restored.failed_q, chain.failed_q)
    with pytest.raises(MetropolisSamplingError, match="failed chains"):
        s.finalize(restored)


@dataclass(frozen=True)
class DegenerateModel(AutoDiffModel):
    spec = ConfinedNonlinear.spec

    def apply(self, params, q, vectors):
        return jnp.zeros_like(vectors)

    def reference_energy(self, params, q):
        return jnp.sum(q*q)/2


def test_degenerate_geometry_is_allowed_in_target_but_endpoint_is_never_redrawn():
    s = sampler(model=DegenerateModel())
    chain = s.start([5, 19], [0.], initialization_id="point-origin-v1", seed=4)
    chain = s.advance(chain, 20).chain
    assert np.all(chain.status == 0)
    with pytest.raises(MetropolisSamplingError, match="no redraw") as caught:
        s.finalize(chain)
    np.testing.assert_array_equal(caught.value.chain.q, chain.q)
    np.testing.assert_array_equal(caught.value.chain.attempts, 20)
    assert np.all(caught.value.state.method_state["status"] == 1)


def test_finite_potential_with_invalid_force_fails_only_at_finalization():
    s = sampler(model=ConfinedNonlinear(singular_force=True))
    chain = s.start([1], [0.], initialization_id="point-origin-v1")
    with pytest.raises(MetropolisSamplingError, match="invalid force") as caught:
        s.finalize(chain)
    np.testing.assert_array_equal(caught.value.chain.q, [[0.]])


def test_bad_mixing_emits_warning_and_never_claims_equilibrium():
    s = sampler(proposal_scale=1e-12)
    with pytest.warns(MixingWarning):
        result = s.sample([1, 2], [[-2.], [2.]], initialization_id="two-separated-starts-v1",
                          burn_in=0, production_steps=8, thin=2)
    assert result.diagnostics["warnings"]
    assert result.diagnostics["certifies_equilibrium"] is False
    assert result.diagnostics["split_rhat"][0] > 10


def test_sampler_identity_immutability_and_corrupted_cached_density(tmp_path):
    s = sampler()
    start = s.start([1, 2], [0.], initialization_id="origin")
    with pytest.raises(AttributeError, match="immutable"):
        s.beta = 20
    params = s.params
    params["bias"] = 9
    assert float(s.params["bias"]) == .2
    with pytest.raises(ValueError, match="density"):
        s.advance(replace(start, log_density=jnp.array([jnp.nan, 0.])), 1)
    with pytest.raises(MetropolisSamplingError, match="cached"):
        s.advance(replace(start, log_density=start.log_density+1), 1)
    s.save_chain(tmp_path / "chain.npz", start)
    other = sampler(params={"bias": .3})
    with pytest.raises(ValueError, match="identity"):
        other.load_chain(tmp_path / "chain.npz")


def test_endpoint_runs_in_existing_real_mash_without_extending_transport_contract():
    s = sampler()
    chain = s.advance(s.start([19, 7], [0.], initialization_id="origin", seed=87), 20).chain
    state = s.finalize(chain).state
    run = Simulation(Problem(s.model, s.params, CoupledClassical(s.masses), MASHRM(),
                             MASHRMPopulation(include_nuclei=True)),
                      Integrator(.001, electronic="exponential_midpoint"), Execution(chunk_size=2)).run(state, 2)
    np.testing.assert_allclose(run.observables["mapping_norm"], 1., atol=2e-14)
    assert np.isfinite(run.observables["energy"]).all()


def test_actual_transition_matches_independent_acceptance_decisions():
    s = sampler()
    ids = np.arange(128)
    positions = np.linspace(-2, 2, len(ids))[:, None]
    seed = 846
    chain = s.start(ids, positions, initialization_id="fixed-linear-grid-v1", seed=seed)
    result = s.advance(chain, 1).chain
    expected = []
    accepted = []
    for identifier, position in zip(ids, positions[:, 0], strict=True):
        # Only the documented counter-key schedule is shared. Target weights
        # use the independent analytic eigenvalue formula, not sampler code.
        key = jax.random.fold_in(jax.random.fold_in(
            jax.random.key(seed, impl="threefry2x32"), 0x524D4D43), np.uint32(identifier))
        normal, uniform = jax.random.split(jax.random.fold_in(key, np.uint32(0)))
        proposal = position+.7*float(jax.random.normal(normal, (1,), dtype=jnp.float64)[0])
        ratio = scalar_reference(proposal)[0]-scalar_reference(position)[0]
        take = np.log1p(-float(jax.random.uniform(uniform, dtype=jnp.float64))) <= min(0, ratio)
        expected.append(proposal if take else position)
        accepted.append(int(take))
    np.testing.assert_allclose(result.q[:, 0], expected, rtol=0, atol=5e-16)
    np.testing.assert_array_equal(result.accepted, accepted)
    assert 0 < sum(accepted) < len(ids)


def test_constant_reference_and_electronic_shifts_cancel_from_coordinate_target():
    s = sampler()
    shifted = sampler(params={"shift": 1024., "offset": -2048.})
    for q in ([-1.7], [.2], [1.8]):
        assert shifted.log_density(q) == pytest.approx(s.log_density(q), abs=5e-12)


def test_failed_checkpoints_are_validated_and_not_overwritten(tmp_path):
    import json
    from pyeph.io.checkpoint import array_fingerprint

    s = sampler(model=ConfinedNonlinear(invalid_outside=.1))
    with pytest.raises(MetropolisSamplingError) as caught:
        s.start([7, 8], [[1.], [2.]], initialization_id="invalid-start-diagnostic")
    failed = caught.value.chain
    assert not np.isfinite(failed.log_density).any()
    path = tmp_path/"failed.npz"
    s.save_chain(path, failed)
    original = path.read_bytes()
    with pytest.raises(FileExistsError):
        s.save_chain(path, failed)
    assert path.read_bytes() == original
    np.testing.assert_array_equal(s.load_chain(path).q, failed.q)
    changes = {"q": np.full((2, 1), np.nan), "trajectory_ids": np.array([7, 7], dtype=np.uint32),
               "attempts": np.array([-1, 0], dtype=np.int64),
               "status": np.array([1, 19], dtype=np.int32),
               "failed_q": np.zeros((3, 1)), "accepted": np.array([0., 0.])}
    with np.load(path, allow_pickle=False) as saved:
        metadata = json.loads(str(saved["metadata"]))
        arrays = {key: saved[key] for key in saved.files if key != "metadata"}
    for key, value in changes.items():
        altered = arrays | {key: value}
        meta = metadata | {"arrays_sha256": array_fingerprint(altered)}
        bad = tmp_path/f"bad_{key}.npz"
        np.savez(bad, metadata=np.asarray(json.dumps(meta)), **altered)
        with pytest.raises(ValueError):
            s.load_chain(bad)


def test_checkpoint_rejects_unexpected_metadata_arrays_and_duplicate_keys(tmp_path):
    import io
    import json
    import zipfile

    s = sampler()
    chain = s.start([4], [0.], initialization_id="origin")
    path = tmp_path/"chain.npz"
    s.save_chain(path, chain)
    with np.load(path, allow_pickle=False) as saved:
        values = {key: saved[key] for key in saved.files}
    meta = json.loads(str(values["metadata"])) | {"unexpected": True}
    metadata_path = tmp_path/"metadata.npz"
    np.savez(metadata_path, **(values | {"metadata": np.asarray(json.dumps(meta))}))
    with pytest.raises(ValueError, match="metadata fields"):
        s.load_chain(metadata_path)
    extra_path = tmp_path/"extra.npz"
    np.savez(extra_path, **(values | {"unexpected": np.zeros(1)}))
    with pytest.raises(ValueError, match="unexpected or duplicate"):
        s.load_chain(extra_path)
    raw = io.BytesIO()
    np.save(raw, np.asarray(chain.q))
    with zipfile.ZipFile(path, "a") as archive, pytest.warns(UserWarning, match="Duplicate"):
        archive.writestr("q.npy", raw.getvalue())
    with pytest.raises(ValueError, match="unexpected or duplicate"):
        s.load_chain(path)


@dataclass(frozen=True)
class MalformedForceModel(ConfinedNonlinear):
    failure: str = "reference_shape"

    def reference_gradient(self, params, q):
        if self.failure == "reference_shape":
            return jnp.zeros(())
        if self.failure == "reference_complex":
            return jnp.ones_like(q)*(1+1j)
        return super().reference_gradient(params, q)

    def contract_gradient(self, params, q, weight):
        if self.failure == "carrier_shape":
            return jnp.zeros((2,))
        if self.failure == "carrier_complex":
            return jnp.ones_like(q)*(1+1j)
        return super().contract_gradient(params, q, weight)


@pytest.mark.parametrize("failure", ["reference_shape", "reference_complex", "carrier_shape", "carrier_complex"])
def test_malformed_complete_forces_reject_endpoint_with_chain_retained(failure):
    s = sampler(model=MalformedForceModel(failure=failure))
    chain = s.start([19], [0.], initialization_id="origin")
    with pytest.raises(MetropolisSamplingError, match="endpoint evaluation") as caught:
        s.finalize(chain)
    assert "actually real and model-shaped" in str(caught.value.__cause__)
    np.testing.assert_array_equal(caught.value.chain.q, chain.q)
