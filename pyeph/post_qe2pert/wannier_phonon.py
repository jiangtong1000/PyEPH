"""
Construct Wannier-like phonon modes by minimizing the Marzari-Vanderbilt spread
functional to find the best gauge for localizing the phonon modes.
"""
import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsp_linalg
import numpy, h5py
from pyeph.utils.grid import generate_half_qgrids, rgrid_2d_full
from pyeph.lib.setup_jax import configure_jax_backend
configure_jax_backend(verbose=True)

# Things to improve:
# 1. Proskrutes / Parallel transport
# 2. Link-overlap cost (Delta R free?)
# 3. Riemannian optimizer to replace expm
# 4. After convergence, round l to nearest integer (Holstein, SSH)


def trs_grid(Nx, Ny):
    q_hbz, q_minus, q_full, partner_hbz_for_minus = generate_half_qgrids(Nx, Ny)
    return q_hbz, q_minus, q_full, partner_hbz_for_minus, rgrid_2d_full(Nx, Ny)

def _params_to_unitary(params_hbz, q_minus, partner_hbz_for_minus):
    """
    Expand HBZ params to full-BZ with TRS: params_full = [params_hbz, params_hbz[partner]^*],
    then exp(skew-Hermitian) per q to get a unitary gauge for all q.
    """
    params_hbz = jnp.asarray(params_hbz, dtype=jnp.complex128)
    partner_hbz_for_minus = jnp.asarray(partner_hbz_for_minus, dtype=jnp.int32)
    n_hbz, n_modes, _ = params_hbz.shape
    n_minus = len(q_minus)
    n_full  = n_hbz + n_minus

    # Build full params with TRS conjugation
    params_full = jnp.zeros((n_full, n_modes, n_modes), dtype=jnp.complex128)
    params_full = params_full.at[:n_hbz].set(params_hbz)
    params_full = params_full.at[n_hbz:].set(params_hbz[partner_hbz_for_minus].conj())

    # Project to skew-Hermitian and exponentiate -> unitary
    skew_full = 0.5 * (params_full - jnp.swapaxes(params_full.conj(), -1, -2))
    U_full = jax.vmap(jsp_linalg.expm)(skew_full)
    return U_full

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

    eigvecs = jnp.asarray(eigenvectors, dtype=jnp.complex128)
    q_vecs = jnp.asarray(q_vectors, dtype=jnp.float64)
    delta_r = jnp.asarray(delta_r_vectors, dtype=jnp.float64)
    masses = jnp.asarray(masses, dtype=jnp.float64)

    n_modes = eigvecs.shape[-1]
    
    if initial_params is None:
        params = jnp.zeros((len(q_hbz), n_modes, n_modes), dtype=jnp.complex128)
    else:
        params = jnp.asarray(initial_params, dtype=jnp.complex128)
        if params.shape != (len(q_hbz), n_modes, n_modes):
            raise ValueError("initial_params must have shape (N_q, N_modes, N_modes)")

    learning_rate = jnp.asarray(learning_rate, dtype=jnp.float64)

    def objective(param_array):
        gauge = _params_to_unitary(param_array, q_minus, partner_hbz_for_minus)
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
        with h5py.File("wannier_phonon.h5", "a") as f:
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
            with h5py.File("wannier_phonon.h5", "a") as f:
                if 'params' not in f:
                    f['params'] = best_params
                else:
                    del f['params']
                    f["params"] = best_params

        if abs(prev_loss - loss_real) < tol:
            break
        prev_loss = loss_real
        params = params - learning_rate * grad

    gauge_opt = _params_to_unitary(best_params, q_minus, partner_hbz_for_minus)
    loss_track = jnp.asarray(loss_track, dtype=jnp.float64)
    return gauge_opt, loss_track, rho_track