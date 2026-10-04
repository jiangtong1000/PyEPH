"""Compatibility propagation configuration and views over native state.

There is no legacy time-integration engine here: all evolution delegates to
the native electronic integrator or the shared Simulation/CPA workflow.
"""

from collections.abc import Mapping

import jax
import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp

from pyeph.integrators.electronic import Integrator, exponential_action, rk4_step
from pyeph.models.polaron import band_narrow_hamiltonian
from pyeph.observables.transport.greenkubo import thermal_density_matrix
from pyeph.observables.transport.polaron import lf_current_correlation

from .estimator import (
    _prepare_sector_arrays, _stack_to_dense, current_from_density_no_polaron,
    current_from_density_polaron, get_sectors_for_polaron_transform,
)
from ._precision import require_legacy_precision


def scale_offdiag(matrix, factor):
    if sp.issparse(matrix):
        result = matrix.astype(np.result_type(matrix.dtype, factor)).copy()
        result.data *= factor
        result.setdiag(matrix.diagonal())
        result.eliminate_zeros()
        return result
    return np.asarray(band_narrow_hamiltonian(matrix, factor))


def band_narrow(heps, factor):
    return [scale_offdiag(matrix, factor) for matrix in heps]


def _to_dense_list(mats, dtype=np.complex128):
    return list(_stack_to_dense(mats, dtype))


def integrate_unitary_rk4(u_t0, h0, hmid, hfinal, dt, ntraj):
    """Adapt pre-sampled Hamiltonians to the shared native RK4 primitive."""
    require_legacy_precision()
    first, middle, last = map(lambda x: jnp.asarray(_stack_to_dense(x)), (h0, hmid, hfinal))
    if len(first) != ntraj:
        raise ValueError("ntraj does not match Hamiltonian batch")

    def step(u, a, b, c):
        def apply_at(t, vectors):
            matrix = jnp.where(t == 0, a, jnp.where(t == dt, c, b))
            return matrix @ vectors
        return rk4_step(apply_at, 0., u, dt)
    result = np.asarray(jax.vmap(step)(jnp.asarray(u_t0), first, middle, last))
    if isinstance(u_t0, np.ndarray):
        u_t0[...] = result
        return u_t0
    return result


def integrate_unitary_exact(u_t0, h0, dt, ntraj):
    require_legacy_precision()
    h = jnp.asarray(_stack_to_dense(h0))
    if len(h) != ntraj:
        raise ValueError("ntraj does not match Hamiltonian batch")
    result = np.asarray(jax.vmap(lambda a, u: exponential_action(a, u, dt))(h, jnp.asarray(u_t0)))
    if isinstance(u_t0, np.ndarray):
        u_t0[...] = result
        return u_t0
    return result


class _StaticBathFactors(Mapping):
    """Lazy legacy dict view; avoid an eager Python object for every edge pair."""
    def __init__(self, pairs, phi0):
        self.pairs = tuple(tuple(int(x) for x in pair) for pair in pairs)
        self.support = frozenset(self.pairs)
        self.phi0 = phi0

    def __getitem__(self, key):
        i, j, k, ell = key
        if (i, j) not in self.support or (k, ell) not in self.support:
            raise KeyError(key)
        return np.exp((-2 + (i == j) + (k == ell))*self.phi0)

    def __iter__(self):
        return (a+b for a in self.pairs for b in self.pairs)

    def __len__(self):
        return len(self.pairs)**2


class UnitaryPropagator:
    def __init__(self, nsites, ntraj_per_rank, time_step, total_time, temperature):
        if any(not isinstance(n, (int, np.integer)) or n < 1 for n in (nsites, ntraj_per_rank)):
            raise ValueError("nsites and ntraj must be positive integers")
        if not np.isfinite(total_time) or total_time <= 0 or not np.isfinite(temperature) or temperature <= 0:
            raise ValueError("total_time and temperature must be finite and positive")
        self.integrator = Integrator(float(time_step))
        self.nsites, self.ntraj = int(nsites), int(ntraj_per_rank)
        self.time_step, self.total_time = float(time_step), float(total_time)
        self.time_range = np.arange(0, total_time, time_step)
        self.beta, self.time = 1/float(temperature), 0.


class DensityMatrixUnitaryPropagator(UnitaryPropagator):
    def __init__(self, nsites, ntraj, time_step, total_time, temperature):
        super().__init__(nsites, ntraj, time_step, total_time, temperature)
        self.polaron_prefactor, self.polaron_transform, self.band_narrow_only = 1., False, False
        self.calculate_current = self.calculate_current_no_polaron
        self.F0, self._sectors, self._sector_arrays, self.sec_weights = None, None, None, None
        self._problem, self._state = None, None

    @property
    def sectors(self):
        if self._sectors is None and self.F0 is not None:
            self._sectors = get_sectors_for_polaron_transform(self._ham.hopping_pairs)
        return self._sectors

    def _arrays(self):
        if self._sector_arrays is None:
            self._sector_arrays = _prepare_sector_arrays(self.sectors, self.F0)
        return self._sector_arrays

    @property
    def sector_quad_idx(self):
        return self._arrays()[0]

    @property
    def sector_F0_vals(self):
        return self._arrays()[1]

    @property
    def sector_offsets(self):
        return self._arrays()[2]

    def build(self, ham, quantum_ph):
        if quantum_ph is None:
            raise ValueError("build requires a quantum phonon bath")
        self._ham, self._quantum = ham, quantum_ph
        self.polaron_transform, self.band_narrow_only = True, quantum_ph.band_narrow_only
        self.polaron_prefactor = quantum_ph.polaron_prefactor
        self.sec_weights = quantum_ph.sector_weights
        self.calculate_current = (self.calculate_current_no_polaron if self.band_narrow_only
                                  else self.calculate_current_polaron)
        self.F0 = _StaticBathFactors(ham.hopping_pairs, quantum_ph.phi0)
        self._sectors, self._sector_arrays = None, None

    def initialize_density_matrix(self, heps, jx_0, jy_0):
        require_legacy_precision()
        h = _stack_to_dense(heps)
        if h.shape != (self.ntraj, self.nsites, self.nsites):
            raise ValueError("initial Hamiltonian shape does not match propagator")
        self.rho0 = np.asarray(thermal_density_matrix(h, self.beta))
        self.jx_0, self.jy_0 = _to_dense_list(jx_0), _to_dense_list(jy_0)
        self.j_rho0_x_T = (np.asarray(self.jx_0) @ self.rho0).swapaxes(-1, -2)
        self.j_rho0_y_T = (np.asarray(self.jy_0) @ self.rho0).swapaxes(-1, -2)
        self.u_t = np.tile(np.eye(self.nsites, dtype=complex), (self.ntraj, 1, 1))

    def _sync_native(self, problem, state):
        self._problem, self._state = problem, state
        self.time = float(np.asarray(state.time).reshape(-1)[0])
        self.u_t = np.asarray(state.electronic)
        payload = state.method_state["transport"]
        self.rho0 = np.asarray(payload["rho0"])
        # Compatibility views omit i. Band-narrow-only payload currents are
        # scaled, so restore bare currents before applying its historical f².
        currents = np.asarray(payload.get("bare_currents0", payload["currents0"]))/1j
        self.jx_0, self.jy_0 = currents[:, 0], currents[:, 1]
        self.j_rho0_x_T = (self.jx_0 @ self.rho0).swapaxes(-1, -2)
        self.j_rho0_y_T = (self.jy_0 @ self.rho0).swapaxes(-1, -2)

    def evolve(self, ham, classic_ph, quantum_ph):
        from pyeph.simulation import Simulation
        from .simulation import _assemble

        if self._problem is None:
            problem, state = _assemble(ham, classic_ph, quantum_ph, self, thermal_policy="legacy_full")
            self._sync_native(problem, state)
        result = Simulation(self._problem, self.integrator).run(self._state, 1, collect=False)
        self._sync_native(self._problem, result.final_state)
        classic_ph.update_position(self.time)
        ham.heps = ham.build_ep_variation_matrix(classic_ph.qfield)
        if quantum_ph is not None:
            quantum_ph.update_phit(self.time)
            self.sec_weights = quantum_ph.sector_weights

    def calculate_current_no_polaron(self, jx_t, jy_t):
        result = []
        for insertion, currents in ((self.j_rho0_x_T, jx_t), (self.j_rho0_y_T, jy_t)):
            values = np.asarray([current_from_density_no_polaron(insertion[i], self.u_t[i], currents[i])
                                 for i in range(self.ntraj)])
            result.append(values*self.polaron_prefactor**2 if self.band_narrow_only else values)
        return tuple(result)

    def calculate_current_polaron(self, jx_t, jy_t, use_python=False):
        if np.array_equal(self.sec_weights, self._quantum.sector_weights):
            # Keep exponents combined for strong coupling. The native workflow
            # already follows this path; this also stabilizes the old method.
            sector = np.repeat(np.arange(-2, 3), np.diff(self.sector_offsets))
            return tuple(np.asarray(lf_current_correlation(
                self.u_t, self.rho0, 1j*_stack_to_dense(current), 1j*_stack_to_dense(initial),
                self.sector_quad_idx, sector, self._quantum.phi0, self._quantum.phit,
            )) for current, initial in ((jx_t, self.jx_0), (jy_t, self.jy_0)))
        return tuple(np.asarray(current_from_density_polaron(
            self.u_t, self.rho0, current, initial, self.F0, self.sectors,
            self.sec_weights, use_python=use_python, quad_idx=self.sector_quad_idx,
            F0_vals=self.sector_F0_vals, sector_offsets=self.sector_offsets,
        )) for current, initial in ((jx_t, self.jx_0), (jy_t, self.jy_0)))
