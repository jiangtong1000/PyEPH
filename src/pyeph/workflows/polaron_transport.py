"""Local Lang--Firsov dressing plus the corresponding CPA transport estimator.

The reduced quantum bath contributes both a narrowed electronic Hamiltonian
and a bath-dressed correlation.  It is not inserted as a nuclear feedback
force.  The bare model supplies physical currents before the LF bath factor.
"""

from dataclasses import dataclass, replace
from typing import Any

import jax.numpy as jnp
import numpy as np

from pyeph.core._configuration import real_scalar
from pyeph.core.contracts import prepared_action
from pyeph.core.problem import Problem
from pyeph.dynamics.cpa import CPA
from pyeph.models.polaron import band_narrow_hamiltonian, lf_band_narrowing, lf_phi
from pyeph.observables.transport.polaron import build_lf_sectors, lf_current_correlation
from pyeph.observables.transport.polaron_compact import (
    CompactLFSectors,
    build_compact_lf_sectors,
    lf_current_correlation_compact,
)
from pyeph.workflows.transport import (
    TransportMeasurement,
    _observation,
    _transport_payload,
    initialize_transport_state,
)


def _diagonal_operation(model):
    """Respect an apply override that has not supplied its own diagonal hook."""
    diagonal = getattr(model, "diagonal", None)
    if diagonal is None:
        return None
    if not callable(diagonal):
        raise TypeError("model.diagonal must be callable or None")
    # Inspect the instance first, then the method resolution order. A diagonal
    # declared beside or before apply describes that Hamiltonian; an older
    # inherited diagonal must not bypass a newer apply implementation.
    definitions = (getattr(model, "__dict__", {}), *(cls.__dict__ for cls in type(model).__mro__))
    for namespace in definitions:
        if "diagonal" in namespace:
            return diagonal
        if "apply" in namespace:
            return None
    return None


@dataclass(frozen=True)
class PolaronDressedModel:
    """Local LF dressing with unchanged bare physical current probes.

    Models may provide ``diagonal(params, q)`` returning the (nstates,) diagonal
    of the same Hamiltonian as ``apply``. With that operation, dressing retains
    sparse or block actions: f H v + (1-f) diag(H) v. Models without a suitable
    diagonal retain the dense reference implementation. A subclass or instance
    overriding only apply also retains that reference path, so an inherited
    diagonal cannot silently change the Hamiltonian being narrowed.

    This wrapper is deliberately force-incompatible.  Generalizing reduced
    quantum-bath dynamics to feedback needs a separately specified theory.
    """

    base_model: Any
    factor: Any

    def __post_init__(self):
        # This wrapper is static JIT configuration. A NumPy scalar array must
        # not remain mutable while compiled kernels retain its previous value.
        factor = real_scalar(self.factor, "LF narrowing factor")
        if not 0 <= factor <= 1:
            raise ValueError("LF narrowing factor must be between zero and one")
        object.__setattr__(self, "factor", factor)

    @property
    def spec(self):
        return replace(self.base_model.spec, name=f"{self.base_model.spec.name}+local_lf", force_support=False)

    def bare_hamiltonian(self, params, q):
        identity = jnp.eye(self.spec.system.nstates, dtype=jnp.result_type(q, 1j))
        return self.base_model.apply(params, q, identity)

    def apply(self, params, q, vectors):
        return self.prepare_action(params, q)(vectors)

    def prepare_action(self, params, q):
        """Prepare one dressed geometry without a persistent numerical cache."""
        diagonal = _diagonal_operation(self.base_model)
        if diagonal is None:
            hamiltonian = band_narrow_hamiltonian(self.bare_hamiltonian(params, q), self.factor)
            return lambda vectors: hamiltonian @ vectors
        onsite = jnp.asarray(diagonal(params, q))
        if onsite.shape != (self.spec.system.nstates,):
            raise ValueError("model.diagonal must return shape (nstates,)")
        action = prepared_action(self.base_model, params, q)

        def dressed(vectors):
            local = onsite[:, None] if vectors.ndim == 2 else onsite
            return self.factor*action(vectors) + (1-self.factor)*local*vectors

        return dressed

    def reference_energy(self, params, q):
        return self.base_model.reference_energy(params, q)

    def probe_apply(self, params, context, probe, vectors):
        return self.base_model.probe_apply(params, context, probe, vectors)


@dataclass(frozen=True)
class PolaronTransportMeasurement(TransportMeasurement):
    frequencies: Any = ()
    couplings: Any = ()
    beta: float = 1.0
    quad_indices: Any = None
    sector_indices: Any = None
    thermal_policy: str = "offdiagonal"
    compact_topology: Any = None

    def __post_init__(self):
        super().__post_init__()
        frequencies, couplings = np.asarray(self.frequencies), np.asarray(self.couplings)
        if frequencies.ndim != 1 or couplings.shape != frequencies.shape:
            raise ValueError("LF bath frequencies and couplings require equal one-dimensional mode arrays")
        if (np.iscomplexobj(frequencies) or not np.isfinite(frequencies).all()
                or np.any(frequencies <= 0) or not np.isfinite(couplings).all()):
            raise ValueError("LF frequencies must be finite and positive; couplings must be finite")
        beta = real_scalar(self.beta, "LF inverse temperature", allow_infinite=True)
        if beta < 0 or (frequencies.size and beta == 0):
            raise ValueError("LF inverse temperature must be positive for a nonempty bath")
        if self.compact_topology is None:
            quads, sectors = np.asarray(self.quad_indices), np.asarray(self.sector_indices)
            if (quads.ndim != 2 or quads.shape[1] != 4 or sectors.shape != (len(quads),)
                    or quads.dtype.kind not in "iu" or sectors.dtype.kind not in "iu"):
                raise ValueError("LF quad_indices and sector_indices require integer shapes (n,4) and (n,)")
            if (np.any(quads < 0) or np.any(quads > np.iinfo(np.int32).max)
                    or np.any(sectors < -2) or np.any(sectors > 2)):
                raise ValueError("LF indices must be nonnegative int32 values and sectors must lie in [-2,2]")
            object.__setattr__(self, "quad_indices", jnp.array(quads, copy=True))
            object.__setattr__(self, "sector_indices", jnp.array(sectors, copy=True))
        else:
            if self.quad_indices is not None or self.sector_indices is not None:
                raise ValueError("compact_topology cannot be combined with explicit LF quadruples or sectors")
            if not isinstance(self.compact_topology, CompactLFSectors):
                raise TypeError("compact_topology must be a validated CompactLFSectors instance")
        if self.thermal_policy not in {"offdiagonal", "legacy_full"}:
            raise ValueError("thermal_policy must be offdiagonal or legacy_full")
        for name, value in (("frequencies", frequencies), ("couplings", couplings)):
            object.__setattr__(self, name, jnp.array(value, copy=True))
        object.__setattr__(self, "beta", beta)

    def validate(self, problem):
        super().validate(problem)
        if not isinstance(problem.model, PolaronDressedModel):
            raise TypeError("LF transport requires its corresponding PolaronDressedModel")
        if self.thermal_policy not in {"offdiagonal", "legacy_full"}:
            raise ValueError("thermal_policy must be offdiagonal or legacy_full")
        if self.compact_topology is not None:
            if self.compact_topology.nstates != problem.model.spec.system.nstates:
                raise ValueError("compact LF topology and model electronic dimensions differ")
        elif np.any(np.asarray(self.quad_indices) >= problem.model.spec.system.nstates):
            raise ValueError("LF quad_indices contain an out-of-range electronic state")

    def initial_hamiltonian(self, problem, state):
        if self.thermal_policy == "legacy_full":
            return problem.model.factor * problem.model.bare_hamiltonian(problem.params, state.q)
        return super().initial_hamiltonian(problem, state)

    def validate_preparation_beta(self, beta):
        if float(beta) != self.beta:
            raise ValueError("LF thermal preparation must use the problem's quantum-bath inverse temperature")

    def validate_preparation_currents(self, currents):
        currents = np.asarray(currents)
        supported = np.zeros(currents.shape[-2:], dtype=bool)
        pairs = (np.asarray(self.quad_indices)[:, :2] if self.compact_topology is None
                 else np.asarray(self.compact_topology.support_pairs))
        if len(pairs):
            supported[pairs[:, 0], pairs[:, 1]] = True
        if np.any(np.abs(currents[..., ~supported]) > 1e-12):
            raise ValueError("LF hopping_pairs omit nonzero initial current edges")

    def evaluate(self, problem, state):
        payload = _transport_payload(state, problem.model.spec.system.nstates, len(self.probes))
        phi0 = jnp.real(lf_phi(self.frequencies, self.couplings, self.beta, 0.0))
        phit = lf_phi(self.frequencies, self.couplings, self.beta, state.time - payload["time0"])
        currents = self.currents(problem, state)
        if self.compact_topology is None:
            correlation = lf_current_correlation(
                state.electronic, payload["rho0"], currents, payload["currents0"],
                self.quad_indices, self.sector_indices, phi0, phit,
            )
        else:
            correlation = lf_current_correlation_compact(
                state.electronic, payload["rho0"], currents, payload["currents0"],
                self.compact_topology, phi0, phit,
            )
        return _observation(state, correlation)


def make_polaron_transport_problem(
    model, params, nuclear_treatment, frequencies, couplings, beta, *, hopping_pairs,
    thermal_policy="offdiagonal", probes=("current_x",), probe_callback=None,
    current_convention="physical", estimator="compact",
):
    """Assemble LF-CPA with a declared initial-state convention.

    ``offdiagonal`` prepares the density from the same narrowed Hamiltonian
    used in propagation, retaining onsite disorder.  ``legacy_full`` reproduces
    historical PyEPH preparation from ``factor * entire_H`` while propagation
    still narrows only offdiagonals.  This difference is consequential for
    nonconstant diagonal energies and is never selected implicitly.

    The bath is local, independent, identical on all electronic sites, and at
    one inverse temperature per problem.  ``hopping_pairs`` must contain the
    unique directed union of all supported bare current edges at every time,
    not just the nonzero edges of one sampled geometry. ``estimator='compact'``
    uses shared-vertex corrections and an ordinary-trace baseline, retaining
    diagonal current edges. ``estimator='full'`` keeps the reference Cartesian
    square of edges, with quadratic edge-count construction/storage. Both are
    exact trace estimators with the same full-U state and diagnostic outputs.
    """
    frequencies, couplings = np.asarray(frequencies), np.asarray(couplings)
    if frequencies.ndim != 1 or couplings.shape != frequencies.shape:
        raise ValueError("LF bath frequencies and couplings require equal one-dimensional mode arrays")
    if not np.isfinite(frequencies).all() or np.any(frequencies <= 0) or not np.isfinite(couplings).all():
        raise ValueError("LF frequencies must be finite and positive; couplings must be finite")
    if np.ndim(beta) != 0 or np.isnan(beta) or beta < 0 or (len(frequencies) and beta == 0):
        raise ValueError("LF inverse temperature must be positive for a nonempty bath")
    if not isinstance(estimator, str) or estimator not in ("compact", "full"):
        raise ValueError("LF estimator must be compact or full")
    factor = lf_band_narrowing(frequencies, couplings, beta)
    topology = None
    quads = sectors = None
    if estimator == "compact":
        topology = build_compact_lf_sectors(hopping_pairs, model.spec.system.nstates)
    else:
        quads, sectors = build_lf_sectors(hopping_pairs, nstates=model.spec.system.nstates)
    dressed = PolaronDressedModel(model, factor)
    measurement = PolaronTransportMeasurement(
        probes=tuple(probes), probe_callback=probe_callback, current_convention=current_convention,
        frequencies=jnp.asarray(frequencies), couplings=jnp.asarray(couplings), beta=float(beta),
        quad_indices=quads, sector_indices=sectors, thermal_policy=thermal_policy,
        compact_topology=topology,
    )
    return Problem(dressed, params, nuclear_treatment, CPA(), measurement).validate()


def initialize_polaron_transport_state(problem, q, p, *, time=0.0, trajectory_id=0, seed=0):
    """Initialize LF transport using the problem's bath temperature and policy."""
    if not isinstance(problem.measurement, PolaronTransportMeasurement):
        raise TypeError("problem must be assembled with make_polaron_transport_problem")
    return initialize_transport_state(problem, q, p, problem.measurement.beta,
                                      time=time, trajectory_id=trajectory_id, seed=seed)


__all__ = [
    "PolaronDressedModel", "PolaronTransportMeasurement",
    "make_polaron_transport_problem", "initialize_polaron_transport_state",
]
