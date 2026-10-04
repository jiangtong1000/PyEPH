"""Linear electron-phonon models in canonical mass-weighted coordinates.

Legacy scaled quadratures must be converted before evaluating these models.
Dense tensors are a small reference implementation; EdgeEPCModel preserves
the graph and never allocates a matrix for each mode during operator action.
"""

from dataclasses import dataclass, field

import jax.numpy as jnp
import numpy as np

from pyeph.core.contracts import ModelSpec
from pyeph.core._configuration import boolean_scalar, integer_scalar
from pyeph.core.system import SystemSpec
from pyeph.models.base import AutoDiffModel


def _check_hermitian(a, shape, name):
    a = np.asarray(a)
    if a.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {a.shape}")
    if not np.all(np.isfinite(a)):
        raise ValueError(f"{name} must be finite")
    if not np.allclose(a, a.conj().swapaxes(-1, -2), atol=1e-12, rtol=1e-10):
        raise ValueError(f"{name} must be Hermitian")
    return jnp.asarray(a)


def _harmonic_reference(p, q):
    return (0.5 * jnp.sum((p["omega"] * (q - p["q_eq"])) ** 2)
            + p.get("reference_offset", 0.0))


@dataclass(frozen=True, init=False)
class LinearEPCModel(AutoDiffModel):
    """Dense reference h(Q)=h0+sum_a Q_a G_a; no coordinate rescaling."""

    nstates_config: int
    nmodes: int
    complex_valued: bool = False
    spec: ModelSpec = field(init=False)

    def __init__(self, nstates, nmodes, complex_valued=False):
        object.__setattr__(self, "nstates_config", nstates)
        object.__setattr__(self, "nmodes", nmodes)
        object.__setattr__(self, "complex_valued", complex_valued)
        self.__post_init__()

    def __post_init__(self):
        for name in ("nstates_config", "nmodes"):
            object.__setattr__(self, name, integer_scalar(getattr(self, name), name))
        object.__setattr__(self, "complex_valued", boolean_scalar(self.complex_valued, "complex_valued"))
        if self.nstates_config < 1 or self.nmodes < 1:
            raise ValueError("nstates and nmodes must be positive")
        object.__setattr__(self, "spec", ModelSpec(
            system=SystemSpec(nstates=self.nstates_config, q_shape=(self.nmodes,),
                              coordinate_kind="normal_mode"),
            name="linear_epc", complex_valued=self.complex_valued))

    def create_params(self, h0, coupling, omega=None, q_eq=None):
        h0 = _check_hermitian(h0, (self.nstates, self.nstates), "h0")
        coupling = _check_hermitian(
            coupling, (self.nmodes, self.nstates, self.nstates), "coupling")
        if not self.complex_valued and (jnp.iscomplexobj(h0) or jnp.iscomplexobj(coupling)):
            raise ValueError("complex parameters require complex_valued=True")
        omega = np.ones(self.nmodes) if omega is None else np.asarray(omega)
        q_eq = np.zeros(self.nmodes) if q_eq is None else np.asarray(q_eq)
        if omega.shape != (self.nmodes,) or q_eq.shape != (self.nmodes,):
            raise ValueError("omega and q_eq must have shape (nmodes,)")
        if np.any(omega < 0) or not np.all(np.isfinite(omega)):
            raise ValueError("omega must be finite and nonnegative")
        return dict(h0=h0, coupling=coupling, omega=jnp.asarray(omega),
                    q_eq=jnp.asarray(q_eq), reference_offset=0.0)

    def default_params(self):
        return self.create_params(np.zeros((self.nstates, self.nstates)),
                                  np.zeros((self.nmodes, self.nstates, self.nstates)))

    def dense(self, params, q):
        p = self.default_params() if params is None else params
        return p["h0"] + jnp.einsum("a,aij->ij", q, p["coupling"])

    def apply(self, params, q, vectors):
        return self.dense(params, q) @ vectors

    def prepare_action(self, params, q):
        """Construct this dense reference model's operator once at a geometry."""
        matrix = self.dense(params, q)
        return lambda vectors: matrix @ vectors

    def reference_energy(self, params, q):
        p = self.default_params() if params is None else params
        return _harmonic_reference(p, q)

    def validate_params(self, params):
        p = self.default_params() if params is None else params
        self.create_params(p["h0"], p["coupling"], p["omega"], p["q_eq"])
        if np.asarray(p.get("reference_offset", 0.)).shape != ():
            raise ValueError("reference_offset must be a scalar")


@dataclass(frozen=True, init=False)
class EdgeEPCModel(AutoDiffModel):
    """Linear onsite and hopping EPC on canonical unordered edges.

    Parameter keys: onsite (S,), hopping (E,), onsite_coupling (D,S),
    hopping_coupling (D,E), omega (D,), q_eq (D,). Complex hoppings are
    allowed when declared; onsite energies and onsite derivatives are real.
    """

    nstates_config: int
    nmodes: int
    edges: tuple
    complex_valued: bool = False
    spec: ModelSpec = field(init=False)

    def __init__(self, nstates, nmodes, edges, complex_valued=False):
        object.__setattr__(self, "nstates_config", nstates)
        object.__setattr__(self, "nmodes", nmodes)
        object.__setattr__(self, "edges", edges)
        object.__setattr__(self, "complex_valued", complex_valued)
        self.__post_init__()

    def __post_init__(self):
        for name in ("nstates_config", "nmodes"):
            object.__setattr__(self, name, integer_scalar(getattr(self, name), name))
        object.__setattr__(self, "complex_valued", boolean_scalar(self.complex_valued, "complex_valued"))
        raw_edges = tuple(tuple(edge) for edge in self.edges)
        if any(x != int(x) for edge in raw_edges for x in edge):
            raise ValueError("edge indices must be exact integers")
        edges = tuple(tuple(int(x) for x in edge) for edge in raw_edges)
        if self.nstates_config < 1 or self.nmodes < 1:
            raise ValueError("nstates and nmodes must be positive")
        if any(len(e) != 2 or not 0 <= e[0] < e[1] < self.nstates_config for e in edges):
            raise ValueError("edges must be unique canonical pairs 0 <= i < j < nstates")
        if len(set(edges)) != len(edges):
            raise ValueError("duplicate graph edges would double-count hoppings")
        object.__setattr__(self, "edges", edges)
        object.__setattr__(self, "spec", ModelSpec(
            system=SystemSpec(nstates=self.nstates_config, q_shape=(self.nmodes,),
                              coordinate_kind="normal_mode"),
            name="edge_epc", complex_valued=self.complex_valued))

    def default_params(self):
        dtype = jnp.complex128 if self.complex_valued else jnp.float64
        return dict(onsite=jnp.zeros(self.nstates), hopping=jnp.zeros(len(self.edges), dtype),
                    onsite_coupling=jnp.zeros((self.nmodes, self.nstates)),
                    hopping_coupling=jnp.zeros((self.nmodes, len(self.edges)), dtype),
                    omega=jnp.ones(self.nmodes), q_eq=jnp.zeros(self.nmodes),
                    reference_offset=0.0)

    def elements(self, params, q):
        p = self.default_params() if params is None else params
        return (p["onsite"] + q @ p["onsite_coupling"],
                p["hopping"] + q @ p["hopping_coupling"])

    def diagonal(self, params, q):
        """Return onsite values without constructing the hopping operator."""
        return self.elements(params, q)[0]

    def apply(self, params, q, vectors):
        return self._action(*self.elements(params, q), vectors)

    def prepare_action(self, params, q):
        """Retain only onsite/edge values; no dense Hamiltonian is constructed."""
        onsite, hopping = self.elements(params, q)
        return lambda vectors: self._action(onsite, hopping, vectors)

    def _action(self, onsite, hopping, vectors):
        scalar = vectors.ndim == 1
        v = vectors[:, None] if scalar else vectors
        out = onsite[:, None] * v
        out = out.astype(jnp.result_type(out, hopping))
        if self.edges:
            edge = jnp.asarray(self.edges)
            i, j = edge[:, 0], edge[:, 1]
            out = out.at[i].add(hopping[:, None] * v[j])
            out = out.at[j].add(hopping.conj()[:, None] * v[i])
        return out[:, 0] if scalar else out

    def reference_energy(self, params, q):
        p = self.default_params() if params is None else params
        return _harmonic_reference(p, q)

    def validate_params(self, params):
        p = self.default_params() if params is None else params
        shapes = dict(onsite=(self.nstates,), hopping=(len(self.edges),),
                      onsite_coupling=(self.nmodes, self.nstates),
                      hopping_coupling=(self.nmodes, len(self.edges)),
                      omega=(self.nmodes,), q_eq=(self.nmodes,))
        for key, shape in shapes.items():
            a = np.asarray(p[key])
            if a.shape != shape or not np.all(np.isfinite(a)):
                raise ValueError(f"{key} must be finite with shape {shape}")
            if np.iscomplexobj(a) and (not self.complex_valued or key not in {"hopping", "hopping_coupling"}):
                raise ValueError(f"{key} must be real under the declared model")
        if np.any(np.asarray(p["omega"]) < 0):
            raise ValueError("omega must be nonnegative")
