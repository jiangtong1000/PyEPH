"""
Construct Wannier-like phonon modes by minimizing the Marzari-Vanderbilt spread
functional to find the best gauge for localizing the phonon modes.
"""
import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsp_linalg
import numpy, h5py, scipy
from pyeph.utils.grid import generate_half_qgrids, rgrid_2d_full
from pyeph.lib.setup_jax import configure_jax_backend

# Things to improve:
# 1. Proskrutes / Parallel transport
# 2. Link-overlap cost (Delta R free?)
# 3. Riemannian optimizer to replace expm
# 4. After convergence, round l to nearest integer (Holstein, SSH)

def assert_trs(modes_all, q_hbz, q_minus, partner_hbz_for_minus):
    assert len(q_minus) == len(partner_hbz_for_minus)
    assert len(q_hbz) + len(q_minus) == len(modes_all)
    for iqm, minus_q in enumerate(partner_hbz_for_minus):
        assert minus_q < len(q_hbz)
        assert jnp.allclose(modes_all[iqm+len(q_hbz)], modes_all[minus_q].conj())

def base_gauge_with_proskrutes(modes_hbz, hbz_qgrids):
    nmodes = modes_hbz.shape[-1]
    nq_hbz = len(hbz_qgrids)
    base = numpy.zeros((nq_hbz, nmodes, nmodes), dtype=numpy.complex128)
    modes_ref = modes_hbz[0]  # sould be Gamma? not required?

    for iq in range(nq_hbz):
        overlap = modes_hbz[iq].conj().T @ modes_ref
        u, _, vh = scipy.linalg.svd(overlap, full_matrices=False, lapack_driver="gesvd")
        base[iq] = u @ vh
    return jnp.asarray(base)

def trs_grid(Nx, Ny):
    q_hbz, q_minus, q_full, partner_hbz_for_minus = generate_half_qgrids(Nx, Ny)
    return q_hbz, q_minus, q_full, partner_hbz_for_minus, rgrid_2d_full(Nx, Ny)

def _params_to_unitary(params_hbz, U_base, partner_hbz_for_minus):
    params_hbz = jnp.asarray(params_hbz, jnp.complex128)
    U_base = jnp.asarray(U_base, jnp.complex128)
    skew = 0.5 * (params_hbz - jnp.swapaxes(params_hbz.conj(), -1, -2))
    U_delta = jax.vmap(jsp_linalg.expm)(skew)
    U_hbz = jnp.einsum("qij,qjk->qik", U_base, U_delta)
    U_minus = jnp.conj(U_hbz[partner_hbz_for_minus])
    return jnp.concatenate([U_hbz, U_minus], axis=0)


def compute_wannier_amplitudes(
    eigenvectors,
    q_vectors,
    delta_r_vectors,
    masses,
    gauge
):
    """
    Evaluate ``W_{a,alpha,mu}(Delta R)`` for a set of phonon eigenvectors.
    .. math::
    W_{a\alpha,\mu}(\Delta\mathbf R)
      = \frac{1}{N_{\mathbf q}} \sum_{\mathbf q}
        e^{-i\mathbf q\cdot\Delta\mathbf R}
        \frac{\widetilde e^{(\mu)}_{a\alpha}(\mathbf q)}{\sqrt{M_a}},
    \qquad
    \widetilde e^{(\mu)}(\mathbf q) = \sum_s e^{(s)}(\mathbf q) U_{s\mu}(\mathbf q).
    and probability density ``rho_mu(Delta R)`` from Wannier amplitudes.
    """

    eigvecs = jnp.asarray(eigenvectors, dtype=jnp.complex128) # (N_q, N_modes, N_modes)
    n_q, n_modes, _ = eigvecs.shape
    assert n_modes == 3 * len(masses)
    q_vecs = jnp.asarray(q_vectors, dtype=jnp.float64) # (N_q, 3)
    delta_r = jnp.asarray(delta_r_vectors, dtype=jnp.float64) # (N_R, 3)
    
    masses = jnp.asarray(masses, dtype=jnp.float64)
    mass_per_component = jnp.repeat(masses, 3)
    sqrt_mass_per_component = jnp.sqrt(mass_per_component)

    gauge = jnp.asarray(gauge, dtype=jnp.complex128)
    assert gauge.shape == (n_q, n_modes, n_modes)

    rotated_eigvecs = jnp.einsum("qjk, qkl -> qjl", eigvecs, gauge)
    rotated_mass_weighted = rotated_eigvecs / sqrt_mass_per_component[None, :, None]

    phases = jnp.exp(-1j * (q_vecs @ delta_r.T)) # (N_q, N_R)
    wannier_amplitudes = jnp.einsum("qij, qr -> rij", rotated_mass_weighted, phases) / n_q # (N_R, N_modes, N_modes)
    
    rho = jnp.einsum(
        "i,rij->rj", mass_per_component, jnp.abs(wannier_amplitudes)**2
        ).real # (N_R, N_modes), this is already normalized if we have enough grids
    rho = rho / rho.sum(axis=0, keepdims=True)
    return rho

def compute_marzari_vanderbilt_spread(
    delta_r_vectors, rho
):
    """
    Center: ``\bar{Delta R}_mu`` given the probability distribution.
    \bar{\Delta\mathbf R}_\mu=\sum_{\Delta\mathbf R}\Delta\mathbf R\,\rho_\mu(\Delta\mathbf R).
    
    Second moment: ``\sum_{\Delta\mathbf R} \lVert\Delta\mathbf R\rVert^2 \rho_\mu(\Delta\mathbf R)``
    
    Args:
        delta_r_vectors: (N_R, 3). Lattice vectors.
        rho: (N_R, N_modes). Probability density for each modes at each lattice vector.

    Returns:
        total_spread: float. Total spread.
    """

    delta_r = jnp.asarray(delta_r_vectors, dtype=jnp.float64)
    rho = jnp.asarray(rho, dtype=jnp.float64) #(N_R, N_modes)

    assert delta_r.ndim == 2 and delta_r.shape[1] == 3
    assert rho.ndim == 2 and rho.shape[0] == delta_r.shape[0]

    centers = jnp.einsum("Ra, Rm -> am", delta_r, rho) #(3, N_modes)
    norm_sq = jnp.sum(delta_r ** 2, axis=1)
    second_moments = jnp.einsum("R, Rm -> m", norm_sq, rho)
    spreads = second_moments - jnp.sum(centers ** 2, axis=0)
    total_spread = jnp.sum(spreads)
    return total_spread

def minimize_wannier_spread(
    eigenvectors,
    q_vectors,
    delta_r_vectors,
    masses,
    q_hbz,
    q_minus,
    partner_hbz_for_minus,
    fname,
    learning_rate=0.05,
    max_iter=200,
    tol=1e-9,
    initial_params=None
):
    """Minimize the Marzari-Vanderbilt spread over unitary gauges using JAX autodiff.

    Args:
        eigenvectors: (N_q, N_modes, N_modes) phonon eigenvectors (unweighted) in reciprocal space.
        q_vectors: (N_q, 3) Brillouin-zone sampling points.
        delta_r_vectors: (N_R, 3) lattice vectors used for the Wannier transform.
        masses: (N_atoms,) atomic masses.
        learning_rate: gradient-descent step size.
        max_iter: maximum number of optimization steps.
        tol: stop when consecutive spreads differ by less than this threshold.
        initial_params: optional complex array with the same shape as the gauge parameters.

    Returns:
        gauge_opt: optimized unitary gauge matrices of shape (N_q, N_modes, N_modes).
        spread_history: 1D array tracking the spread after each iteration.
    """

    configure_jax_backend(verbose=True)

    eigvecs = jnp.asarray(eigenvectors, dtype=jnp.complex128)
    q_vecs = jnp.asarray(q_vectors, dtype=jnp.float64)
    delta_r = jnp.asarray(delta_r_vectors, dtype=jnp.float64)
    masses = jnp.asarray(masses, dtype=jnp.float64)
    partner_hbz_for_minus = jnp.asarray(partner_hbz_for_minus, dtype=jnp.int32)
    assert_trs(eigenvectors, q_hbz, q_minus, partner_hbz_for_minus)

    n_modes = eigvecs.shape[-1]
    U_base_gauge = base_gauge_with_proskrutes(eigenvectors[:len(q_hbz)], q_hbz)
    
    if initial_params is None:
        params = numpy.zeros((len(q_hbz), n_modes, n_modes), dtype=numpy.complex128)
    else:
        params = jnp.asarray(initial_params, dtype=jnp.complex128)
        if params.shape != (len(q_hbz), n_modes, n_modes):
            raise ValueError("initial_params must have shape (N_q, N_modes, N_modes)")

    learning_rate = jnp.asarray(learning_rate, dtype=jnp.float64)

    def objective(param_array):
        gauge = _params_to_unitary(param_array, U_base_gauge, partner_hbz_for_minus)
        rho = compute_wannier_amplitudes(
            eigvecs,
            q_vecs,
            delta_r,
            masses,
            gauge,
        )
        loss = compute_marzari_vanderbilt_spread(delta_r, rho)
        return loss, rho

    value_and_grad = jax.jit(jax.value_and_grad(objective, has_aux=True))

    loss_track = []
    rho_track = []
    best_params = params
    best_loss = 1e6
    prev_loss = 1e6

    for _ in range(int(max_iter)):
        (loss, rho), grad = value_and_grad(params)
        print(f"Loss: {loss}")
        loss_real = float(jnp.real(loss))
        loss_track.append(loss_real)
        rho_track.append(rho)
        with h5py.File(fname, "a") as f:
            if 'loss' not in f:
                f['loss'] = loss_track
            if 'rho' not in f:
                f['rho'] = rho_track
            else:
                del f['loss']
                del f['rho']
                f['loss'] = loss_track
                f["rho"] = rho_track
        if loss_real < best_loss:
            best_loss = loss_real
            best_params = params
            with h5py.File(fname, "a") as f:
                if 'params' not in f:
                    f['params'] = best_params
                else:
                    del f['params']
                    f["params"] = best_params

        if abs(prev_loss - loss_real) < tol:
            break
        prev_loss = loss_real
        grad = 0.5 * (grad - jnp.swapaxes(grad.conj(), -1, -2))
        params = params + learning_rate * grad

    gauge_opt = _params_to_unitary(best_params, U_base_gauge, partner_hbz_for_minus)
    loss_track = jnp.asarray(loss_track, dtype=jnp.float64)
    return gauge_opt, loss_track, rho_track

def minimize_wannier_spread_pyqcpbc(
    eigenvectors,
    q_vectors,
    delta_r_vectors,
    masses,
    q_hbz,
    q_minus,
    partner_hbz_for_minus,
    max_iter=200,
    tol=1e-9,
    initial_params=None,
    algorithm="l_bfgs_ls",
    solver_options = None,
):
    """Minimize the spread using pyqcpbc's optimizer infrastructure.

    Args:
        eigenvectors: (N_q, N_modes, N_modes) phonon eigenvectors.
        q_vectors: (N_q, 3) Brillouin-zone sampling points.
        delta_r_vectors: (N_R, 3) lattice vectors used for the Wannier transform.
        masses: (N_atoms,) atomic masses.
        q_hbz: list of q-points in the half Brillouin zone.
        q_minus: list of time-reversed partners.
        partner_hbz_for_minus: mapping from indices in ``q_minus`` to ``q_hbz``.
        max_iter: maximum number of optimizer iterations.
        tol: tolerance passed to the pyqcpbc minimizer.
        initial_params: optional complex array for the initial gauge parameters.
        algorithm: pyqcpbc optimization algorithm name.
        solver_options: optional dict of extra ``solver.cfg`` options.

    Returns:
        gauge_opt: optimized unitary gauge matrices of shape (N_q, N_modes, N_modes).
        loss_track: 1D array tracking the spread whenever the objective is evaluated.
        rho_track: list of probability densities corresponding to ``loss_track``.
    """
    
    from typing import Any, Dict, List, Optional
    from pyqcpbc.OPT import objective_function as qcpbc_objf
    from pyqcpbc.OPT import minimizer as qcpbc_minimizer

    eigvecs = jnp.asarray(eigenvectors, dtype=jnp.complex128)
    q_vecs = jnp.asarray(q_vectors, dtype=jnp.float64)
    delta_r = jnp.asarray(delta_r_vectors, dtype=jnp.float64)
    masses = jnp.asarray(masses, dtype=jnp.float64)
    partner_hbz_for_minus = jnp.asarray(partner_hbz_for_minus, dtype=jnp.int32)
    assert_trs(eigenvectors, q_hbz, q_minus, partner_hbz_for_minus)

    n_modes = eigvecs.shape[-1]
    U_base_gauge = base_gauge_with_proskrutes(eigenvectors[: len(q_hbz)], q_hbz)

    if initial_params is None:
        params0 = numpy.zeros((len(q_hbz), n_modes, n_modes), dtype=numpy.complex128)
    else:
        params0 = jnp.asarray(initial_params, dtype=jnp.complex128)
        if params0.shape != (len(q_hbz), n_modes, n_modes):
            raise ValueError("initial_params must have shape (N_q, N_modes, N_modes)")
        params0 = numpy.asarray(params0)

    total_size = params0.size
    params_shape = params0.shape

    def _pack_complex(matrix):
        arr = numpy.asarray(matrix)
        return numpy.concatenate([arr.real.ravel(), arr.imag.ravel()])

    def _unpack_complex(vector):
        half = vector.size // 2
        real = vector[:half].reshape(params_shape)
        imag = vector[half:].reshape(params_shape)
        return real + 1j * imag

    def objective(param_array):
        gauge = _params_to_unitary(param_array, U_base_gauge, partner_hbz_for_minus)
        rho = compute_wannier_amplitudes(
            eigvecs,
            q_vecs,
            delta_r,
            masses,
            gauge,
        )
        loss = compute_marzari_vanderbilt_spread(delta_r, rho)
        return loss, rho

    value_and_grad = jax.jit(jax.value_and_grad(objective, has_aux=True))

    loss_history = []
    rho_history = []

    class WannierSpreadObjective(qcpbc_objf):
        def __init__(self, initial_params_complex):
            super().__init__(2 * total_size)
            self.params_vec = _pack_complex(initial_params_complex)
            self._origin_vec = self.params_vec.copy()
            self._best_loss = numpy.inf
            self._best_params_vec = self.params_vec.copy()
            self._loss = None
            self._grad_vec = None
            self._dirty = True

        def _evaluate(self):
            params_complex = jnp.asarray(_unpack_complex(self.params_vec), dtype=jnp.complex128)
            (loss, rho), grad = value_and_grad(params_complex)
            loss_real = float(jnp.real(loss))
            grad = 0.5 * (grad - jnp.swapaxes(grad.conj(), -1, -2))
            grad_vec = _pack_complex(grad)
            self._loss = loss_real
            self._grad_vec = grad_vec
            self._dirty = False
            loss_history.append(loss_real)
            rho_history.append(numpy.asarray(rho))
            if loss_real < self._best_loss:
                self._best_loss = loss_real
                self._best_params_vec = self.params_vec.copy()

        def _ensure_evaluated(self):
            if self._dirty or self._loss is None:
                self._evaluate()

        def get_value(self):
            self._ensure_evaluated()
            return self._loss

        def grad(self):
            self._ensure_evaluated()
            return self._grad_vec

        def precond(self, x, shift):
            return x

        def update_params(self, step):
            self.params_vec = self.params_vec + step.ravel()
            self._dirty = True

        def save_new_origin(self):
            self._origin_vec = self.params_vec.copy()

        def back_to_origin(self):
            self.params_vec = self._origin_vec.copy()
            self._dirty = True

        @property
        def best_params_vec(self):
            return self._best_params_vec

        @property
        def best_loss(self):
            return self._best_loss

    objective_wrapper = WannierSpreadObjective(params0)
    _ = objective_wrapper.get_value()

    solver = qcpbc_minimizer()
    solver.cfg("algorithm", algorithm)
    solver.cfg("maxiter", int(max_iter))
    solver.cfg("tol", float(tol))
    if solver_options:
        for key, value in solver_options.items():
            solver.cfg(key, value)

    solver.run(objective_wrapper)

    best_params_complex = _unpack_complex(objective_wrapper.best_params_vec)
    gauge_opt = _params_to_unitary(best_params_complex, U_base_gauge, partner_hbz_for_minus)
    loss_track = jnp.asarray(loss_history, dtype=jnp.float64)
    rho_track = [jnp.asarray(rho) for rho in rho_history]
    return gauge_opt, loss_track, rho_track
