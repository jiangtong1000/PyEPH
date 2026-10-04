"""Independent classical canonical preparation for a real linear EPC model.

The coordinate density is exp(-beta*Vref(q))*Tr exp(-beta*h(q)). A Gaussian
rejection envelope accounts for the carrier contribution, unlike bare phonon
sampling followed by an electronic Boltzmann draw. Exactness refers to this
probability distribution in real arithmetic, not a certified floating-point
enclosure or quantum nuclear thermal sampling.

For x=q-q_eq, g_j=||G_j||_2 and A²=sum_j(g_j/omega_j)², ordered Weyl gives
Z_e(q) <= Z_e(q_eq)*exp(beta*sum_j g_j*|x_j|). Completing diagonal harmonic
squares bounds the unnormalized target by B*exp(-beta*kappa*x.K.x/2), with
log B=-beta*V0+log Z_e(q_eq)+beta*A²/[2*(1-kappa)]. Gaussian normalization
does not enter the pointwise acceptance ratio. Its optimal width has
kappa=2/[2+r+sqrt(r*(r+4))], r=beta*A²/nmodes.
"""

from dataclasses import asdict, dataclass, field
import hashlib
import json

import jax
import jax.numpy as jnp
import numpy as np
import scipy
from scipy.special import logsumexp

from pyeph.core._configuration import integer_scalar, real_scalar
from pyeph.core.state import TrajectoryState
from pyeph.dynamics.mashrm import _method_state
from pyeph.models.epc import LinearEPCModel


STATUS_MESSAGES = {
    0: "success",
    1: "proposal capacity exhausted",
    2: "nonfinite or unrepresentable sampling arithmetic",
    3: "rejection envelope violated beyond floating-point tolerance",
    4: "accepted coordinate has a nonisolated electronic spectrum",
}
_VERSION = "linear_epc_canonical_v1"
_DOMAIN = 0x524D4551  # RM equilibrium, separate from population preparation.


@dataclass(frozen=True)
class CanonicalEnsemble:
    """Batched preparation and per-ID diagnostics, in requested ID order.

All statuses are zero on a successful public return. On failure this record is
available as ``CanonicalSamplingError.result``. Failed rows retain the last
proposal (including an accepted incompatible geometry), but have invalid
physical-state placeholders and must not be propagated.
    """

    state: TrajectoryState
    attempts: object
    status: object
    log_acceptance: object
    metadata: dict


class CanonicalSamplingError(RuntimeError):
    """Bounded preparation failure retaining every requested trajectory ID."""

    def __init__(self, result):
        self.result = result
        ids, status = np.asarray(result.state.trajectory_id), np.asarray(result.status)
        bad = np.flatnonzero(status)
        details = "; ".join(f"ID {int(ids[i])}: {STATUS_MESSAGES[int(status[i])]}"
                            for i in bad[:8])
        super().__init__(f"canonical sampling failed for {len(bad)} trajectory IDs: {details}")


def _real_array(value, shape, name):
    value = np.asarray(value)
    if value.dtype.kind not in "iuf" or value.shape != shape or not np.isfinite(value).all():
        raise ValueError(f"{name} must be finite and real with shape {shape}")
    result = np.array(value, dtype=np.float64, copy=True)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} is not representable in float64")
    return result


def _fingerprint(values):
    result = hashlib.sha256()
    for name, value in sorted(values.items()):
        array = np.asarray(value)
        result.update(json.dumps([name, array.dtype.str, array.shape]).encode())
        result.update(array.tobytes(order="C"))
    return result.hexdigest()


def _jax_configuration():
    return {name: getattr(jax.config, name, None) for name in (
        "jax_enable_x64", "jax_default_matmul_precision", "jax_default_prng_impl",
        "jax_threefry_partitionable", "jax_random_seed_offset", "jax_high_dynamic_range_gumbel")}


@dataclass(frozen=True, init=False, eq=False)
class LinearEPCCanonical:
    """Immutable, bounded rejection sampler for exactly ``LinearEPCModel``.

``omega`` must be strictly positive and h0/couplings exactly real symmetric.
Coordinates are canonical: K=diag(omega**2), without an extra mass factor;
momenta have covariance M/beta. JAX x64 is required, never enabled implicitly.

``kappa=None`` minimizes the analytic envelope normalization. Exactly zero
coupling gives kappa=1 and accepts the first Gaussian proposal. An explicit
width must lie in (0,1), or equal 1 for zero coupling. The optional gap tolerance
is an AFTER-acceptance dynamics compatibility check. Failing it raises, rather
than changing the equilibrium distribution by drawing another coordinate.

The parameter dictionary returned by ``params`` is fresh; its JAX values and
all stored configuration are owned immutable snapshots. Construct a new sampler
for a different Hamiltonian or changed JAX numerical/RNG configuration. The
compiled draw has no mutable RNG state. Metadata conservatively includes the
proposal budget in preparation_id: increasing a sufficient budget preserves
individual draws, but records with different budgets have different identities.
    """

    model: LinearEPCModel
    masses: object
    beta: float
    kappa: float
    max_trials: int
    gap_tolerance: float
    _params: tuple = field(repr=False)
    _metadata: str = field(repr=False)
    _configuration: str = field(repr=False)
    _draw: object = field(repr=False)

    def __init__(self, model, params, masses, beta, *, kappa=None, max_trials=10000,
                 gap_tolerance=1e-10):
        if type(model) is not LinearEPCModel or model.complex_valued or model.nstates < 2:
            raise ValueError("canonical preparation requires exactly real LinearEPCModel with N>=2")
        if not jax.config.x64_enabled:
            raise ValueError("canonical RM sampling requires JAX x64")
        beta = real_scalar(beta, "beta")
        gap_tolerance = real_scalar(gap_tolerance, "gap_tolerance")
        max_trials = integer_scalar(max_trials, "max_trials")
        if beta <= 0 or gap_tolerance <= 0 or not 1 <= max_trials <= np.iinfo(np.int32).max:
            raise ValueError("beta/gap_tolerance must be positive; max_trials a positive int32")
        p = model.default_params() if params is None else params
        n, d = model.nstates, model.nmodes
        shapes = dict(h0=(n, n), coupling=(d, n, n), omega=(d,), q_eq=(d,))
        owned = {name: _real_array(p[name], shape, name) for name, shape in shapes.items()}
        owned["reference_offset"] = np.asarray(real_scalar(
            p.get("reference_offset", 0.), "reference_offset"))
        for name in ("h0", "coupling"):
            if not np.array_equal(owned[name], owned[name].swapaxes(-1, -2)):
                raise ValueError(f"{name} must be exactly symmetric for the canonical envelope")
        if np.any(owned["omega"] <= 0):
            raise ValueError("all omega must be strictly positive")
        try:
            mass = np.broadcast_to(np.asarray(masses), (d,))
        except ValueError as exc:
            raise ValueError("masses must broadcast to (nmodes,)") from exc
        mass = _real_array(mass, (d,), "masses")
        if np.any(mass <= 0):
            raise ValueError("masses must be positive")

        # Ordered Weyl bounds every eigenvalue by its value at q_eq minus
        # sum_j ||G_j||_2 |x_j|. Diagonal K permits componentwise completion of
        # squares; g.T@inv(K)@g is NOT this bound for a general SPD matrix.
        with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
            curvature = owned["omega"]**2
            hstar = owned["h0"]+np.einsum("a,aij->ij", owned["q_eq"], owned["coupling"])
        if not np.isfinite(hstar).all() or not np.all(np.isfinite(curvature) & (curvature > 0)):
            raise ValueError("harmonic curvature and centered Hamiltonian must be representable")
        eigstar = np.linalg.eigvalsh(hstar)
        shift = eigstar[0]
        centered_h0 = owned["h0"]-shift*np.eye(n)
        centered_star = centered_h0+np.einsum("a,aij->ij", owned["q_eq"], owned["coupling"])
        g = np.max(np.abs(np.linalg.eigvalsh(owned["coupling"])), axis=1)
        coupled = bool(np.any(owned["coupling"] != 0))
        # A small outward cushion limits roundoff underestimation. This is not
        # interval certification: every actual log acceptance is still checked.
        g = np.where(g > 0, np.nextafter(g*(1+32*n*np.finfo(float).eps), np.inf), 0.)
        with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
            a2 = np.sum((g/owned["omega"])**2)
            beta_a2 = beta*a2
        if (not np.isfinite(eigstar).all() or not np.isfinite(centered_star).all()
                or not np.isfinite(a2) or not np.isfinite(beta_a2)
                or (coupled and (a2 <= 0 or beta_a2 <= 0))):
            raise ValueError("electronic envelope is not representable in float64")
        if kappa is None:
            r = beta_a2/d
            # sqrt(r)*sqrt(r+4) avoids r*(r+4) overflow at otherwise finite r.
            kappa = (1. if not coupled else
                     1./(1.+r/2.+np.sqrt(r)*np.sqrt(r+4.)/2.))
            if coupled:
                kappa = min(kappa, np.nextafter(1., 0.))
        else:
            kappa = real_scalar(kappa, "kappa")
        if not 0 < kappa <= 1 or (coupled and kappa == 1):
            raise ValueError("kappa must be in (0,1), or 1 for exactly zero coupling")
        with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
            sigma_q = 1./np.sqrt(beta)/np.sqrt(kappa)/owned["omega"]
            sigma_p = np.sqrt(mass)/np.sqrt(beta)
            penalty = beta_a2/(2*(1-kappa)) if kappa < 1 else 0.
            star_logits = -beta*np.linalg.eigvalsh(centered_star)
        if (not np.all(np.isfinite(sigma_q) & (sigma_q > 0))
                or not np.all(np.isfinite(sigma_p) & (sigma_p > 0))
                or not np.isfinite(penalty) or not np.isfinite(star_logits).all()):
            raise ValueError("sampling scales and shifted Boltzmann factors must be representable")
        snapshot = {name: jnp.array(value, dtype=jnp.float64, copy=True)
                    for name, value in owned.items()}
        for name, value in dict(model=model, masses=jnp.array(mass, copy=True), beta=beta,
                                kappa=kappa, max_trials=max_trials,
                                gap_tolerance=gap_tolerance,
                                _params=tuple(snapshot.items())).items():
            object.__setattr__(self, name, value)
        metadata = dict(algorithm=_VERSION, target="classical_nuclei_rm_canonical",
                        beta=beta, kappa=kappa, max_trials=max_trials,
                        gap_tolerance=gap_tolerance, a_squared=float(a2),
                        envelope_penalty=float(penalty), zero_coupling=not coupled,
                        nstates=n, nmodes=d, unit_system=asdict(model.spec.unit_system),
                        params_sha256=_fingerprint(owned), masses_sha256=_fingerprint({"m": mass}),
                        rng="jax_threefry_fold_in_domain_id_trial", status_messages=STATUS_MESSAGES,
                        jax_version=jax.__version__, numpy_version=np.__version__,
                        scipy_version=scipy.__version__,
                        jax_configuration=_jax_configuration())
        object.__setattr__(self, "_metadata", json.dumps(metadata, sort_keys=True))
        object.__setattr__(self, "_configuration", json.dumps(_jax_configuration(), sort_keys=True))
        object.__setattr__(self, "_draw", self._build_draw(
            centered_h0, shift, sigma_q, sigma_p, float(logsumexp(star_logits)), penalty))

    @property
    def params(self):
        return dict(self._params)

    def _build_draw(self, centered_h0, energy_shift, sigma_q, sigma_p, log_zstar, penalty):
        p = self.params
        n, d = self.model.nstates, self.model.nmodes
        beta, kappa = self.beta, self.kappa
        h0, sigma_q, sigma_p = map(jnp.asarray, (centered_h0, sigma_q, sigma_p))
        mass, limit, gap_tol = self.masses, self.max_trials, self.gap_tolerance

        def one(identifier, master):
            root = jax.random.fold_in(jax.random.fold_in(master, _DOMAIN), identifier)
            q_key, reject_key, p_key, active_key, sphere_key, future_key = jax.random.split(root, 6)
            # Last candidate is retained on failure; it is never a usable state.
            initial = (jnp.int32(0), jnp.int32(-1), jnp.full((d,), jnp.nan),
                       jnp.full((n,), jnp.nan), jnp.full((n, n), jnp.nan), jnp.float64(jnp.nan))

            def condition(carry):
                attempt, status, *_ = carry
                return (status == -1) & (attempt < limit)

            def proposal(carry):
                attempt, _, _, _, _, _ = carry
                z = jax.random.normal(jax.random.fold_in(q_key, attempt), (d,), dtype=jnp.float64)
                x = sigma_q*z
                q = p["q_eq"]+x
                h = h0+jnp.einsum("a,aij->ij", q, p["coupling"])
                energies, vectors = jnp.linalg.eigh(
                    jnp.where(jnp.all(jnp.isfinite(h)), h, jnp.zeros_like(h)),
                    symmetrize_input=False)
                logits = -beta*energies
                active_logits = -beta*(energies-energies[0])
                # Reference offset and common electronic shift cancel before
                # arithmetic. Express the harmonic term in standard normals.
                harmonic = (1-kappa)/(2*kappa)*jnp.sum(z*z)
                loga = (jnp.float64(0.) if kappa == 1. else
                        jax.scipy.special.logsumexp(logits)-log_zstar-harmonic-penalty)
                rounding = 64*jnp.finfo(jnp.float64).eps*(
                    1+jnp.max(jnp.abs(logits))+abs(log_zstar)+harmonic+penalty)
                finite = (jnp.all(jnp.isfinite(q)) & jnp.all(jnp.isfinite(h))
                          & jnp.all(jnp.isfinite(energies)) & jnp.all(jnp.isfinite(vectors))
                          & jnp.all(jnp.isfinite(logits)) & jnp.isfinite(loga)
                          & jnp.all(jnp.isfinite(active_logits))
                          & jnp.isfinite(rounding))
                # 1-u lies in (0,1], avoiding log(0) without a clipped uniform.
                uniform = jax.random.uniform(jax.random.fold_in(reject_key, attempt),
                                             dtype=jnp.float64)
                accepted = jnp.log1p(-uniform) <= jnp.minimum(loga, 0.)
                isolated = jnp.all(jnp.diff(energies) > gap_tol)
                status = jnp.where(~finite, 2, jnp.where(loga > rounding, 3,
                    jnp.where(accepted, jnp.where(isolated, 0, 4), -1))).astype(jnp.int32)
                return attempt+1, status, q, energies, vectors, loga

            attempts, status, q, energies, vectors, loga = jax.lax.while_loop(
                condition, proposal, initial)
            status = jnp.where(status == -1, 1, status).astype(jnp.int32)

            def finish(_):
                momentum = sigma_p*jax.random.normal(p_key, (d,), dtype=jnp.float64)
                # Remove the per-coordinate scalar energy BEFORE adding Gumbel
                # noise; a large common logit can otherwise quantize that noise.
                active = jax.random.categorical(
                    active_key, -beta*(energies-energies[0])).astype(jnp.int32)
                normal = jax.random.normal(sphere_key, (2, n), dtype=jnp.float64)
                mapping = normal[0]+1j*normal[1]
                largest = jnp.argmax(jnp.abs(mapping)**2)
                permutation = jnp.arange(n).at[largest].set(active).at[active].set(largest)
                adiabatic = mapping[permutation]/jnp.linalg.norm(mapping)
                electronic = vectors@adiabatic
                x = q-p["q_eq"]
                reference = .5*jnp.sum((p["omega"]*x)**2)+p["reference_offset"]
                force = -(p["omega"]**2)*x-jnp.einsum(
                    "i,aij,j->a", vectors[:, active], p["coupling"], vectors[:, active])
                total = jnp.sum((momentum/jnp.sqrt(mass))**2)/2+reference+energy_shift+energies[active]
                valid = (jnp.all(jnp.isfinite(momentum)) & jnp.all(jnp.isfinite(electronic))
                         & jnp.all(jnp.isfinite(force)) & jnp.isfinite(total))
                return momentum, electronic, active, jnp.where(valid, 0, 2).astype(jnp.int32)

            def failed(_):
                return jnp.full((d,), jnp.nan), jnp.full((n,), jnp.nan+0j), jnp.int32(0), status

            momentum, electronic, active, status = jax.lax.cond(status == 0, finish, failed, None)
            method = _method_state(active, jnp.float64)
            # Do not disguise failed sampler rows as successful RM states if a
            # caller deliberately accesses the retained exception diagnostics.
            method["status"] = jnp.where(status == 0, 0, 1).astype(jnp.int32)
            state = TrajectoryState(q, momentum, electronic, jnp.float64(0), jnp.int64(0),
                                    identifier, jax.random.key_data(future_key), method)
            return state, attempts, status, loga

        return jax.jit(jax.vmap(one, in_axes=(0, None)))

    def preparation_metadata(self, *, seed=0):
        """Return fresh validated identity metadata without drawing any samples.

This is also used by ``sample``. A workflow can compare restored preparation
metadata against this value without consuming randomness or redoing rejection.
        """
        if not jax.config.x64_enabled:
            raise ValueError("canonical RM sampling requires JAX x64")
        if json.dumps(_jax_configuration(), sort_keys=True) != self._configuration:
            raise ValueError("JAX configuration changed; construct a new canonical sampler")
        seed = integer_scalar(seed, "seed")
        if not 0 <= seed < 2**32:
            raise ValueError("seed must be a uint32 integer")
        metadata = json.loads(self._metadata)
        metadata["seed"] = seed
        metadata["preparation_id"] = hashlib.sha256(
            json.dumps(metadata, sort_keys=True).encode()).hexdigest()
        return metadata

    def sample(self, trajectory_ids, *, seed=0):
        """Draw all requested unique uint32 IDs, or raise with full diagnostics.

Reordering/partitioning IDs and increasing a sufficient max_trials leaves each
draw unchanged. No failed ID is dropped; accepted nonisolated spectra are never
redrawn. The sampler's key domain is separate from native population preparation
and reserves an unused per-ID stream in the returned state's ``key``.
        """
        metadata = self.preparation_metadata(seed=seed)
        ids = np.asarray(trajectory_ids)
        if (ids.ndim != 1 or not ids.size or ids.dtype.kind not in "iu"
                or np.any(ids < 0) or np.any(ids >= 2**32) or len(np.unique(ids)) != len(ids)):
            raise ValueError("trajectory_ids must be nonempty unique uint32 integers")
        key = jax.random.key(metadata["seed"], impl="threefry2x32")
        state, attempts, status, loga = jax.block_until_ready(self._draw(
            jnp.asarray(ids, dtype=jnp.uint32), key))
        result = CanonicalEnsemble(state, attempts, status, loga, metadata)
        if np.any(np.asarray(status)):
            raise CanonicalSamplingError(result)
        return result
