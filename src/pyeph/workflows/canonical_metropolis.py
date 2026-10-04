"""Explicit Metropolis preparation for confined real finite electronic models.

The invariant coordinate density is exp(-beta*Vref(q))*Tr exp(-beta*h(q)).
Finite chains do not constitute exact equilibrium draws. No thermostat or
surface-hopping dynamics is used to manufacture initial nuclear coordinates.
"""

from dataclasses import dataclass, replace
import hashlib
import json
import os
from pathlib import Path
import tempfile
import warnings

import jax
import jax.numpy as jnp
import numpy as np

from pyeph.core._configuration import integer_scalar, real_scalar
from pyeph.core.contracts import pure_state_weight
from pyeph.core.problem import CoupledClassical, Problem
from pyeph.core.state import TrajectoryState
from pyeph.dynamics.mashrm import MASHRM, MASHRMPopulation, _method_state
from pyeph.execution.ensemble import partition_ids
from pyeph.io.checkpoint import array_fingerprint
from pyeph.io.provenance import assert_matching_manifest, problem_manifest, validate_manifest

_DOMAIN = 0x524D4D43
_FINAL_DOMAIN = 0x524D4D46


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value):
    return hashlib.sha256(_json(value).encode()).hexdigest()


def _positive_array(value, shape, name):
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be real")
    array = np.array(np.broadcast_to(raw, shape), dtype=np.float64, copy=True)
    if not np.isfinite(array).all() or np.any(array <= 0):
        raise ValueError(f"{name} must be finite and positive")
    return jnp.asarray(array)


@dataclass(frozen=True)
class MetropolisChain:
    """One independent random stream per ID; arrays are immutable JAX snapshots.

    ``failed_q`` retains the offending proposal on numerical failure. Failed
    chains cannot be advanced or finalized. Their last valid q remains in q.
    """

    trajectory_ids: object
    q: object
    log_density: object
    attempts: object
    accepted: object
    status: object
    failed_q: object
    seed: int
    initialization_id: str
    sampler_id: str

    def __post_init__(self):
        for name in ("trajectory_ids", "q", "log_density", "attempts", "accepted", "status", "failed_q"):
            object.__setattr__(self, name, jnp.array(getattr(self, name), copy=True))

    def subset(self, trajectory_ids):
        ids = partition_ids(trajectory_ids, 0, 1)
        lookup = {int(value): i for i, value in enumerate(np.asarray(self.trajectory_ids))}
        if any(int(value) not in lookup for value in ids):
            raise ValueError("subset contains an unknown trajectory ID")
        indices = np.array([lookup[int(value)] for value in ids])
        values = {name: jnp.array(np.asarray(getattr(self, name))[indices], copy=True)
                  for name in ("trajectory_ids", "q", "log_density", "attempts", "accepted",
                               "status", "failed_q")}
        return replace(self, **values)


@dataclass(frozen=True)
class MetropolisRun:
    chain: MetropolisChain
    coordinates: object
    log_density: object
    recorded_attempts: object


@dataclass(frozen=True)
class MetropolisPreparation:
    state: TrajectoryState
    chain: MetropolisChain
    diagnostics: dict
    metadata: dict


class MetropolisSamplingError(RuntimeError):
    """Retain all chains and the last valid endpoints rather than redrawing IDs."""

    def __init__(self, message, chain, *, state=None):
        self.chain, self.state = chain, state
        super().__init__(message)


class MixingWarning(UserWarning):
    """Finite-chain diagnostics detected a mixing concern, not a proof of bias."""


def mixing_diagnostics(run):
    """Ordinary split-Rhat and pooled lag-one correlation, not an ESS estimate.

    These coordinate diagnostics can miss slow nonlinear observables. The
    retained sequence within each chain is correlated even after thinning.
    """
    x = np.asarray(run.coordinates)
    samples, chains = x.shape[:2]
    flat = x.reshape(samples, chains, -1)
    notes = []
    rhat = None
    lag_one = None
    if samples >= 4 and chains >= 2:
        half = samples // 2
        split = np.concatenate((flat[:half], flat[-half:]), axis=1)
        within = np.var(split, axis=0, ddof=1).mean(axis=0)
        between = half * np.var(split.mean(axis=0), axis=0, ddof=1)
        variance = (half-1)/half*within + between/half
        ratio = np.divide(variance, within, out=np.full_like(within, np.inf), where=within > 0)
        rhat = np.sqrt(ratio)
        if np.any(rhat > 1.05):
            notes.append("coordinate split-Rhat exceeds 1.05 or within-chain variance is zero")
        centered = flat-flat.mean(axis=0, keepdims=True)
        numerator = (centered[:-1]*centered[1:]).sum(axis=(0, 1))
        denominator = (centered*centered).sum(axis=(0, 1))
        lag_one = np.divide(numerator, denominator, out=np.full_like(numerator, np.nan),
                            where=denominator > 0)
    else:
        notes.append("split-Rhat requires at least four retained positions and two chains")
    attempts, accepted = np.asarray(run.chain.attempts), np.asarray(run.chain.accepted)
    rates = np.divide(accepted, attempts, out=np.zeros_like(accepted, dtype=float), where=attempts > 0)
    if np.any((rates < .1) | (rates > .9)):
        notes.append("one or more chain acceptance rates lie outside [0.1,0.9]")
    return {"retained_positions_per_chain": samples, "chains": chains,
            "acceptance_rate": rates, "split_rhat": rhat, "lag_one_correlation": lag_one,
            "warnings": notes, "certifies_equilibrium": False,
            "uncertainty_scope": "chain diagnostics do not bound finite-burn-in bias"}


class NativeCanonicalMetropolis:
    """Fixed-scale random-walk Metropolis in unconstrained Cartesian/canonical q.

    The caller must establish confinement and normalizability. Free translations,
    constrained coordinates, position-dependent mass metrics and hard walls need
    separately derived measures and are unsupported. Nonfinite proposals are
    numerical failures, never silently interpreted as zero target probability.

    Native real fixed orthonormal models with complete forces and N>=2 are
    supported. All calculations require x64. No gap restriction enters the
    Markov target; isolated spectra are checked only when endpoints become
    physical RM states. Model behavior outside declared params must have an
    explicit artifact identity, following ordinary strict provenance rules.
    """

    def __setattr__(self, name, value):
        if getattr(self, "_sealed", False):
            raise AttributeError("sampler configuration is immutable; construct a new sampler")
        object.__setattr__(self, name, value)

    def __init__(self, model, params, masses, beta, proposal_scale, reference_q, *,
                 gap_tolerance=1e-10, real_tolerance=1e-12, artifact_ids=None):
        if not jax.config.x64_enabled:
            raise ValueError("canonical Metropolis sampling requires JAX x64")
        self.beta = real_scalar(beta, "beta")
        if self.beta <= 0:
            raise ValueError("beta must be positive")
        self.model = model
        self.method = MASHRM(gap_tolerance=gap_tolerance, real_tolerance=real_tolerance)
        leaves, self._params_tree = jax.tree.flatten(params)
        self._params_leaves = tuple(jnp.array(value, copy=True) for value in leaves)
        self.q_shape = tuple(model.spec.system.q_shape)
        self.masses = _positive_array(masses, self.q_shape, "masses")
        self.proposal_scale = _positive_array(proposal_scale, self.q_shape, "proposal_scale")
        self._artifacts = dict(artifact_ids or {})
        self.problem = Problem(model, self.params, CoupledClassical(self.masses),
                               self.method, MASHRMPopulation()).validate()
        reference = np.asarray(reference_q)
        if (reference.shape != self.q_shape or reference.dtype.kind not in "iuf"
                or not np.isfinite(reference).all()):
            raise ValueError("reference_q must be finite real model-shaped coordinates")
        self.reference_q = jnp.array(reference, dtype=jnp.float64, copy=True)
        self.nstates = model.spec.system.nstates
        h = np.asarray(model.apply(self.params, self.reference_q, jnp.eye(self.nstates)))
        v = np.asarray(model.reference_energy(self.params, self.reference_q))
        if (h.shape != (self.nstates, self.nstates) or h.dtype.kind not in "iuf"
                or not np.isfinite(h).all() or not np.allclose(h, h.T, atol=real_tolerance, rtol=0)
                or v.shape != () or v.dtype.kind not in "iuf" or not np.isfinite(v)):
            raise ValueError("reference Hamiltonian/potential must be finite, actually real and symmetric")
        self._energy_origin = float(np.linalg.eigvalsh(h)[0])
        self._reference_origin = float(v)
        manifest = problem_manifest(self.problem, None, artifact_ids=self._artifacts)
        validate_manifest(manifest)
        self._identity_json = _json({"schema": 1, "algorithm": "native_canonical_metropolis_v1",
                                    "simulation": manifest, "beta": self.beta,
                                    "proposal_scale": np.asarray(self.proposal_scale).tolist(),
                                    "reference_q": reference.tolist(),
                                    "energy_origin": self._energy_origin,
                                    "reference_origin": self._reference_origin})
        self._density = jax.jit(jax.vmap(self._evaluate))
        self._move = jax.jit(jax.vmap(self._build_move(), in_axes=(0, 0, 0, 0, 0, 0, 0, None, None)),
                             static_argnums=(8,))
        self._finish = jax.jit(jax.vmap(self._build_finish(), in_axes=(0, 0, 0, None)))
        self._sealed = True

    @property
    def params(self):
        return self._params_tree.unflatten(self._params_leaves)

    @property
    def identity(self):
        return json.loads(self._identity_json)

    @property
    def fingerprint(self):
        return _digest(self.identity)

    def _check_identity(self):
        if not jax.config.x64_enabled:
            raise ValueError("canonical Metropolis sampling requires JAX x64")
        assert_matching_manifest(self.identity["simulation"], problem_manifest(
            self.problem, None, artifact_ids=self._artifacts))
        if (self.beta != self.identity["beta"] or
                not np.array_equal(self.proposal_scale, self.identity["proposal_scale"])):
            raise ValueError("sampler settings changed; construct a new sampler")

    def _evaluate(self, q):
        h = jnp.asarray(self.model.apply(self.params, q, jnp.eye(self.nstates, dtype=jnp.float64)))
        reference = jnp.asarray(self.model.reference_energy(self.params, q))
        if h.dtype.kind not in "iuf" or reference.dtype.kind not in "iuf":
            raise ValueError("sampling requires actually real Hamiltonian and reference potential")
        shifted = h-self._energy_origin*jnp.eye(self.nstates)
        energies, vectors = jnp.linalg.eigh(shifted, symmetrize_input=False)
        logits = -self.beta*(energies-energies[0])
        log_density = (-self.beta*((reference-self._reference_origin)+energies[0])
                       + jax.scipy.special.logsumexp(logits))
        valid = (jnp.all(jnp.isfinite(q)) & jnp.all(jnp.isfinite(h))
                 & jnp.all(jnp.abs(h-h.T) <= self.method.real_tolerance)
                 & jnp.isfinite(reference) & jnp.all(jnp.isfinite(energies))
                 & jnp.all(jnp.isfinite(vectors)) & jnp.isfinite(log_density)
                 & jnp.all(jnp.isfinite(logits)))
        return log_density, energies, vectors, valid

    def log_density(self, coordinates):
        """Unnormalized log target, with fixed additive energy origins removed."""
        self._check_identity()
        q = np.asarray(coordinates)
        if q.shape != self.q_shape or q.dtype.kind not in "iuf" or not np.isfinite(q).all():
            raise ValueError("coordinates must be finite real model-shaped data")
        value, _, _, valid = jax.device_get(self._evaluate(jnp.asarray(q, dtype=jnp.float64)))
        if not valid:
            raise ValueError("invalid Hamiltonian, reference potential or target arithmetic")
        return float(value)

    def start(self, trajectory_ids, initial_q, *, initialization_id, seed=0):
        self._check_identity()
        ids = partition_ids(trajectory_ids, 0, 1)
        seed = integer_scalar(seed, "seed")
        if not 0 <= seed < 2**32:
            raise ValueError("seed must be a uint32 integer")
        if not isinstance(initialization_id, str) or not initialization_id.strip():
            raise ValueError("initialization_id must identify initial-coordinate preparation")
        raw = np.asarray(initial_q)
        if raw.dtype.kind not in "iuf":
            raise ValueError("initial_q must be finite real model-shaped or batch-shaped data")
        q = np.array(np.broadcast_to(raw, (len(ids),)+self.q_shape), dtype=np.float64, copy=True)
        if raw.dtype.kind not in "iuf" or not np.isfinite(q).all():
            raise ValueError("initial_q must be finite real model-shaped or batch-shaped data")
        values, _, _, valid = self._density(jnp.asarray(q))
        chain = MetropolisChain(jnp.asarray(ids), jnp.asarray(q), values,
                                jnp.zeros(len(ids), dtype=jnp.int64),
                                jnp.zeros(len(ids), dtype=jnp.int64),
                                jnp.where(valid, 0, 1).astype(jnp.int32),
                                jnp.where(valid.reshape((-1,)+(1,)*len(self.q_shape)),
                                          jnp.nan, jnp.asarray(q)),
                                seed, initialization_id, self.fingerprint)
        if not np.asarray(valid).all():
            raise MetropolisSamplingError("invalid initial target evaluation", chain)
        return chain

    def _validate_structure(self, chain):
        if type(chain) is not MetropolisChain or chain.sampler_id != self.fingerprint:
            raise ValueError("chain belongs to a different sampler")
        ids = partition_ids(chain.trajectory_ids, 0, 1)
        seed = integer_scalar(chain.seed, "chain seed")
        if not 0 <= seed < 2**32 or not isinstance(chain.initialization_id, str) or not chain.initialization_id.strip():
            raise ValueError("invalid chain seed or initialization identity")
        n = len(ids)
        shapes = {"trajectory_ids": (n,), "q": (n,)+self.q_shape,
                  "failed_q": (n,)+self.q_shape, "log_density": (n,),
                  "attempts": (n,), "accepted": (n,), "status": (n,)}
        dtypes = {"trajectory_ids": "uint32", "q": "float64", "failed_q": "float64",
                  "log_density": "float64", "attempts": "int64", "accepted": "int64",
                  "status": "int32"}
        for name in shapes:
            array = np.asarray(getattr(chain, name))
            if array.shape != shapes[name] or array.dtype != np.dtype(dtypes[name]):
                raise ValueError(f"invalid chain {name} shape or dtype")
        if (not np.isfinite(chain.q).all() or np.any(chain.attempts < 0)
                or np.any(chain.attempts >= 2**32) or np.any(chain.accepted < 0)
                or np.any(chain.accepted > chain.attempts)
                or np.any((np.asarray(chain.status) != 0) & (np.asarray(chain.status) != 1))):
            raise ValueError("invalid chain coordinates, counters or status")
        successful = np.asarray(chain.status) == 0
        if (not np.isfinite(np.asarray(chain.log_density)[successful]).all()
                or not np.isnan(np.asarray(chain.failed_q)[successful]).all()):
            raise ValueError("invalid successful-chain density or failure diagnostics")
        # A failed initial target may have a nonfinite log density; a failed
        # proposal may itself be nonfinite. These are diagnostic arrays only.

    def _validate_chain(self, chain):
        self._check_identity()
        self._validate_structure(chain)
        if np.any(chain.status):
            raise MetropolisSamplingError("failed chains cannot be advanced or finalized", chain)
        values, _, _, valid = self._density(chain.q)
        tolerance = 32*np.finfo(float).eps*np.maximum(1, np.abs(values))
        if not np.asarray(valid).all() or np.any(abs(values-chain.log_density) > tolerance):
            raise MetropolisSamplingError("cached chain density does not match its target", chain)

    def _build_move(self):
        def one(identifier, q, log_density, attempts, accepted, status, failed_q, seed, steps):
            root = jax.random.fold_in(jax.random.fold_in(
                jax.random.key(seed, impl="threefry2x32"), _DOMAIN), identifier)

            def move(_, carry):
                q, log_density, attempts, accepted, status, failed_q = carry
                key = jax.random.fold_in(root, attempts.astype(jnp.uint32))
                normal_key, uniform_key = jax.random.split(key)
                proposal = q+self.proposal_scale*jax.random.normal(
                    normal_key, self.q_shape, dtype=jnp.float64)
                candidate, _, _, valid = self._evaluate(proposal)
                ratio = candidate-log_density
                valid &= jnp.isfinite(ratio)
                accept = jnp.log1p(-jax.random.uniform(uniform_key, dtype=jnp.float64)) <= jnp.minimum(0., ratio)
                running = status == 0
                take = running & valid & accept
                return (jnp.where(take, proposal, q), jnp.where(take, candidate, log_density),
                        attempts+running.astype(jnp.int64), accepted+take.astype(jnp.int64),
                        jnp.where(running & ~valid, 1, status),
                        jnp.where(running & ~valid, proposal, failed_q))

            return jax.lax.fori_loop(0, steps, move,
                                     (q, log_density, attempts, accepted, status, failed_q))
        return one

    def advance(self, chain, steps, *, record_every=None):
        """Advance an explicit number of proposals; optionally retain thinned diagnostics.

        No proposal-scale adaptation occurs. Records contain the entry position
        and every requested interval plus the final endpoint. They are correlated
        within each ID. Memory scales with explicitly requested retained records.
        """
        self._validate_chain(chain)
        steps = integer_scalar(steps, "steps")
        if steps < 0 or np.any(np.asarray(chain.attempts, dtype=object)+steps >= 2**32):
            raise ValueError("proposal counter must remain below 2**32")
        stride = max(1, steps) if record_every is None else integer_scalar(record_every, "record_every")
        if stride < 1:
            raise ValueError("record_every must be positive")
        coordinates, values, counters = [chain.q], [chain.log_density], [chain.attempts]
        remaining = steps
        while remaining:
            count = min(stride, remaining)
            try:
                output = jax.block_until_ready(self._move(
                    chain.trajectory_ids, chain.q, chain.log_density, chain.attempts,
                    chain.accepted, chain.status, chain.failed_q, chain.seed, count))
            except Exception as error:
                raise MetropolisSamplingError("target evaluation failed; input chains retained", chain) from error
            chain = replace(chain, **dict(zip(
                ("q", "log_density", "attempts", "accepted", "status", "failed_q"), output, strict=True)))
            if np.any(chain.status):
                raise MetropolisSamplingError("nonfinite/asymmetric proposal; last valid chains retained", chain)
            if record_every is not None or remaining == count:
                coordinates.append(chain.q)
                values.append(chain.log_density)
                counters.append(chain.attempts)
            remaining -= count
        return MetropolisRun(chain, jnp.stack(coordinates), jnp.stack(values), jnp.stack(counters))

    def _build_finish(self):
        def one(identifier, q, attempts, seed):
            root = jax.random.fold_in(jax.random.fold_in(
                jax.random.key(seed, impl="threefry2x32"), _FINAL_DOMAIN), identifier)
            root = jax.random.fold_in(root, attempts.astype(jnp.uint32))
            p_key, active_key, sphere_key, future_key = jax.random.split(root, 4)
            _, energies, vectors, valid = self._evaluate(q)
            valid &= jnp.all(jnp.diff(energies) > self.method.gap_tolerance)
            active = jax.random.categorical(active_key, -self.beta*(energies-energies[0])).astype(jnp.int32)
            normals = jax.random.normal(sphere_key, (2, self.nstates), dtype=jnp.float64)
            mapping = normals[0]+1j*normals[1]
            largest = jnp.argmax(abs(mapping)**2)
            permutation = jnp.arange(self.nstates).at[largest].set(active).at[active].set(largest)
            electronic = vectors@(mapping[permutation]/jnp.linalg.norm(mapping))
            momentum = (jnp.sqrt(self.masses)/jnp.sqrt(self.beta))*jax.random.normal(
                p_key, self.q_shape, dtype=jnp.float64)
            vector = vectors[:, active]
            reference_gradient = jnp.asarray(self.model.reference_gradient(self.params, q))
            carrier_gradient = jnp.asarray(self.model.contract_gradient(
                self.params, q, pure_state_weight(vector)))
            for gradient in (reference_gradient, carrier_gradient):
                if gradient.shape != self.q_shape or gradient.dtype.kind not in "iuf":
                    raise ValueError("complete reference/carrier gradients must be actually real and model-shaped")
            force = -reference_gradient-carrier_gradient
            total_energy = (jnp.sum((momentum/jnp.sqrt(self.masses))**2)/2
                            + self.model.reference_energy(self.params, q)
                            + energies[active]+self._energy_origin)
            valid &= (jnp.all(jnp.isfinite(reference_gradient))
                      & jnp.all(jnp.isfinite(carrier_gradient))
                      & jnp.all(jnp.isfinite(force)) & jnp.all(jnp.isfinite(momentum))
                      & jnp.all(jnp.isfinite(electronic)) & jnp.isfinite(total_energy))
            method = _method_state(active, jnp.float64)
            method["status"] = jnp.where(valid, 0, 1).astype(jnp.int32)
            return TrajectoryState(q, momentum, electronic, jnp.float64(0), jnp.int64(0),
                                   identifier, jax.random.key_data(future_key), method)
        return one

    def finalize(self, chain, *, diagnostics=None, burn_in=0, thin=1):
        """Draw p/active/mapping once at each endpoint; never redraw a bad spectrum."""
        self._validate_chain(chain)
        burn_in, thin = integer_scalar(burn_in, "burn_in"), integer_scalar(thin, "thin")
        if burn_in < 0 or thin < 1 or np.any(chain.attempts < burn_in):
            raise ValueError("invalid declared burn-in or thinning")
        try:
            state = jax.block_until_ready(self._finish(
                chain.trajectory_ids, chain.q, chain.attempts, chain.seed))
        except Exception as error:
            raise MetropolisSamplingError("endpoint evaluation failed; chains retained", chain) from error
        if np.any(np.asarray(state.method_state["status"])):
            raise MetropolisSamplingError("endpoint is degenerate or has invalid force/state; no redraw",
                                          chain, state=state)
        notes = {"warnings": ["no chain mixing diagnostics supplied"], "certifies_equilibrium": False}
        diagnostics = notes if diagnostics is None else {**diagnostics, "certifies_equilibrium": False}
        metadata = {"sampler": self.identity, "seed": chain.seed,
                    "initialization_id": chain.initialization_id, "burn_in": burn_in, "thin": thin,
                    "attempts": np.asarray(chain.attempts).tolist(),
                    "equilibrium_claim": "finite-chain approximation; convergence must be assessed"}
        metadata["preparation_id"] = _digest({key: value for key, value in metadata.items() if key != "attempts"}
                                            | {"attempt_counts": sorted(set(metadata["attempts"]))})
        return MetropolisPreparation(state, chain, diagnostics, metadata)

    def sample(self, trajectory_ids, initial_q, *, initialization_id, burn_in, production_steps,
               thin=1, seed=0):
        """Independent per-ID chain endpoints, with explicit warmup and diagnostic run."""
        burn_in = integer_scalar(burn_in, "burn_in")
        production_steps = integer_scalar(production_steps, "production_steps")
        thin = integer_scalar(thin, "thin")
        if burn_in < 0 or production_steps < 1 or thin < 1 or production_steps % thin:
            raise ValueError("burn_in must be nonnegative; production_steps positive and divisible by thin")
        chain = self.start(trajectory_ids, initial_q, initialization_id=initialization_id, seed=seed)
        chain = self.advance(chain, burn_in).chain
        run = self.advance(chain, production_steps, record_every=thin)
        diagnostics = mixing_diagnostics(run)
        for note in diagnostics["warnings"]:
            warnings.warn(note, MixingWarning, stacklevel=2)
        return self.finalize(run.chain, diagnostics=diagnostics, burn_in=burn_in, thin=thin)

    def save_chain(self, path, chain, *, overwrite=False):
        """Atomic numerical/JSON checkpoint; failed-chain diagnostics can also be saved."""
        self._check_identity()
        self._validate_structure(chain)
        if type(overwrite) is not bool:
            raise ValueError("overwrite must be boolean")
        if not np.any(chain.status):
            self._validate_chain(chain)
        arrays = {name: np.asarray(getattr(chain, name)) for name in (
            "trajectory_ids", "q", "log_density", "attempts", "accepted", "status", "failed_q")}
        metadata = {"identity": self.identity, "seed": chain.seed,
                    "initialization_id": chain.initialization_id,
                    "arrays_sha256": array_fingerprint(arrays)}
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                np.savez(stream, metadata=np.asarray(_json(metadata)), **arrays)
                stream.flush()
                os.fsync(stream.fileno())
            if overwrite:
                os.replace(temporary, path)
            else:
                # Same-filesystem link atomically refuses an existing artifact.
                os.link(temporary, path)
                os.unlink(temporary)
            directory = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    def load_chain(self, path):
        self._check_identity()
        with np.load(path, allow_pickle=False) as data:
            expected = {"trajectory_ids", "q", "log_density", "attempts", "accepted", "status", "failed_q", "metadata"}
            if len(data.files) != len(set(data.files)) or set(data.files) != expected:
                raise ValueError("chain checkpoint has unexpected or duplicate arrays")
            raw_metadata = data["metadata"]
            if raw_metadata.shape != () or raw_metadata.dtype.kind != "U":
                raise ValueError("chain checkpoint metadata must be one JSON string")
            metadata = json.loads(str(raw_metadata))
            arrays = {name: data[name] for name in data.files if name != "metadata"}
        if (not isinstance(metadata, dict)
                or set(metadata) != {"identity", "seed", "initialization_id", "arrays_sha256"}):
            raise ValueError("chain checkpoint has unexpected metadata fields")
        if metadata["identity"] != self.identity or metadata["arrays_sha256"] != array_fingerprint(arrays):
            raise ValueError("chain checkpoint identity or array checksum mismatch")
        chain = MetropolisChain(**{name: jnp.asarray(value) for name, value in arrays.items()},
                                seed=metadata["seed"], initialization_id=metadata["initialization_id"],
                                sampler_id=self.fingerprint)
        self._validate_structure(chain)
        if not np.any(chain.status):
            self._validate_chain(chain)
        return chain
