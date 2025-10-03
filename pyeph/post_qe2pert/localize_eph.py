import jax
import jax.numpy as jnp
import numpy
import numpy as np
import h5py
import os

from pyeph.post_qe2pert.wannier_phonon import assert_trs
from pyeph.utils.constants import ryd_to_mev
from pyeph.lib.setup_jax import configure_jax_backend
configure_jax_backend()

def zero_out_negative_freqs(freqs, eigvecs, phfreq_cutoff):
    """
    Zero out negative frequencies and eigenvectors.
    """
    for iq in range(eigvecs.shape[0]):
        mask = freqs[iq] < phfreq_cutoff
        eigvecs[iq, :, mask] = 0.0
        freqs[iq, mask] = phfreq_cutoff
    return eigvecs, freqs

def compute_density(eph_real, delta_r_vectors):
    g2 = jnp.abs(eph_real) ** 2
    g2_r = jnp.sum(g2, axis=(0, 1, 2, 4))
    p = g2_r / jnp.sum(g2_r)
    r = jnp.linalg.norm(delta_r_vectors, axis=1)
    return p, r

def compute_eph_mat_real_space(
    eigvecs,
    freqs,
    masses,
    eph_raw,
    rph,
    delta_r_vectors,
    q_hbz,
    q_minus,
    partner_hbz_for_minus,
    fname,
    phfreq_cutoff=1.5/ryd_to_mev,
    max_iter = 500,
    tol = 1e-9
):
    """
    Localize the e-ph matrix in real space by optimizing per-mode phase factors.

    Args:
        eigvecs: (nq, nmodes, nmodes), phonon eigenvectors (unweighted).
        freqs: (nq, nmodes), phonon frequencies.
        masses: (natom,), atomic masses.
        eph_raw: (nband, nband, natom, 3, nre, nq), raw e-ph matrix.
        rph: (nrph, 3), Wigner-Seitz lattice vectors for phonon modes.
        delta_r_vectors: (n_delta_r, 3), delta_r vectors over which to maximize.
        q_hbz: (nq_hbz, 3), half-Brillouin zone q-points.
        q_minus: (nq_minus, 3), minus-BZ q-points.
        partner_hbz_for_minus: (nq_minus,), partner index for minus-BZ q-points.
        fname: file name to save the results.
        phfreq_cutoff: phonon frequency cutoff.
        max_iter: maximum number of iterations.
        tol: tolerance for the convergence.

    Returns:
        best_eph_real: (nband, nband, nre, n_delta_r, nmodes), localized e-ph matrix.
        best_gauge: (nq, nmodes, nmodes), diagonal phase gauge matrices.
        metric_history: (n_steps,), convergence history of the localization metric.
    """
    (nband, _, natom, _, nre_ws, nrph_ws) = eph_raw.shape
    assert_trs(eigvecs, q_hbz, q_minus, partner_hbz_for_minus)
    
    nq_hbz = q_hbz.shape[0]
    nq_minus = q_minus.shape[0]
    nq = nq_hbz + nq_minus
    nmodes = eigvecs.shape[1]
    assert eigvecs.shape == (nq, nmodes, nmodes)
    assert freqs.shape == (nq, nmodes)
    assert masses.shape == (natom,)
    assert rph.shape == (nrph_ws, 3)
    
    eigvecs, freqs = zero_out_negative_freqs(freqs, eigvecs, phfreq_cutoff)

    eigvecs = jnp.asarray(eigvecs, dtype=jnp.complex128)
    freqs = jnp.asarray(freqs, dtype=jnp.float64)
    masses = jnp.asarray(masses, dtype=jnp.float64)
    eph_raw = jnp.asarray(eph_raw, dtype=jnp.complex128)
    rph = jnp.asarray(rph, dtype=jnp.float64)
    delta_r_vectors = jnp.asarray(delta_r_vectors, dtype=jnp.float64)
    q_hbz = jnp.asarray(q_hbz, dtype=jnp.float64)
    q_minus = jnp.asarray(q_minus, dtype=jnp.float64)
    q_vectors = jnp.concatenate([q_hbz, q_minus], axis=0)
    partner_hbz_for_minus = jnp.asarray(partner_hbz_for_minus, dtype=jnp.int32)
    n_delta_r = delta_r_vectors.shape[0]

    rng = numpy.random.default_rng()
    params_hbz = rng.standard_normal((nq_hbz, nmodes)) * 1e-3
    params_hbz = jnp.asarray(params_hbz, dtype=jnp.float64)
    learning_rate = jnp.asarray(0.05, dtype=jnp.float64)

    mass_per_component = jnp.repeat(masses, 3)
    sqrt_mass_per_component = 1.0 / jnp.sqrt(mass_per_component)
    inv_sqrt_freqs = 1.0 / jnp.sqrt(2 * freqs)
    exp_iq_delta_r = jnp.exp(-1j * 2.0 * jnp.pi * delta_r_vectors @ q_vectors.T)
    exp_iqr = jnp.exp(1j * 2.0 * jnp.pi * rph @ q_vectors.T)
    eph_raw_rot = jnp.einsum("ijkaep, pq->ijkaeq", eph_raw, exp_iqr)
    eph_raw_rot = eph_raw_rot.reshape((nband, nband, natom * 3, nre_ws, nq))
    
    def init_eph_real():
        rotated_eigvecs = jnp.einsum("qvu, v, qu->qvu", eigvecs, sqrt_mass_per_component, inv_sqrt_freqs)
        eph_real = jnp.zeros((nband, nband, nre_ws, n_delta_r, nmodes), dtype=jnp.complex128)
        for iw in range(nband):
            for jw in range(nband):
                eph_mixed = jnp.einsum("veq, qvu->euq", eph_raw_rot[iw, jw], rotated_eigvecs)
                block = jnp.einsum("euq, rq->eru", eph_mixed, exp_iq_delta_r)
                eph_real = eph_real.at[iw, jw].set(block)
        eph_real = jnp.abs(eph_real)
        total_g2 = jnp.sum(eph_real**2, axis=3) # (nband, nband, nre_ws, nmodes)
        return eph_real, total_g2
    
    eph_real, total_g2 = init_eph_real()
    p_init, r_init = compute_density(eph_real, delta_r_vectors)
    with h5py.File(fname, "w") as f:
        f.create_dataset("init_eph_real", data=eph_real)
        f.create_dataset("init_p", data=p_init)
        f.create_dataset("init_r", data=r_init)
    
    def build_phase_factors(param_array):
        phases_hbz = jnp.exp(1j * param_array)
        phases_minus = jnp.conj(phases_hbz[partner_hbz_for_minus])
        return jnp.concatenate([phases_hbz, phases_minus], axis=0)
    
    def compute_eph_real(param_array):
        phase_factors = build_phase_factors(param_array)
        rotated_eigvecs = jnp.einsum("qvu, qu, v, qu->qvu", eigvecs, phase_factors, sqrt_mass_per_component, inv_sqrt_freqs)
        eph_real = jnp.zeros((nband, nband, nre_ws, n_delta_r, nmodes), dtype=jnp.complex128)
        # only track the upper triangle since the e-ph matrix is symmetric
        for jw in range(nband):
            for iw in range(jw + 1):
                eph_mixed = jnp.einsum("veq, qvu->euq", eph_raw_rot[iw, jw], rotated_eigvecs)
                block = jnp.einsum("euq, rq->eru", eph_mixed, exp_iq_delta_r)
                eph_real = eph_real.at[iw, jw].set(block)
        return eph_real
    
    def objective(param_array):
        eph_real = compute_eph_real(param_array)
        p, r = compute_density(eph_real, delta_r_vectors)
        Er  = jnp.sum(p * r)
        Er2 = jnp.sum(p * r**2)
        return Er2 - Er * Er
    
    @jax.jit
    def grad_function(param_array):
        grad = jax.grad(objective)(param_array)
        return grad

    metric_history = []
    best_metric = float("inf")
    best_params = params_hbz
    prev_metric = float("inf")
    
    for _ in range(max_iter):
        grad = grad_function(params_hbz)
        params_hbz = params_hbz - learning_rate * grad
        metric = objective(params_hbz)
        if abs(prev_metric - metric) < tol:
            break
        prev_metric = metric
        print(f"Metric: {metric:.6e}")
        metric_history.append(metric)

        if metric < best_metric:
            best_metric = metric
            best_params = params_hbz
            with h5py.File(fname, "a") as f:
                if "eph_real" in f:
                    del f["eph_real"]
                    del f["best_phase_factors"]
                    del f["best_p"]
                eph_real = compute_eph_real(best_params)
                assert jnp.allclose(jnp.sum(eph_real**2, axis=3), total_g2), "Total g2 should be conserved"
                p, r = compute_density(eph_real, delta_r_vectors)
                best_phase_factors = build_phase_factors(best_params)
                f.create_dataset("best_phase_factors", data=best_phase_factors)
                f.create_dataset("eph_real", data=eph_real)
                f.create_dataset("best_p", data=p)

    best_eph_real = compute_eph_real(best_params)
    best_phase_factors = build_phase_factors(best_params)
    metric_history = jnp.asarray(metric_history, dtype=jnp.float64)
    return best_eph_real, best_phase_factors, metric_history