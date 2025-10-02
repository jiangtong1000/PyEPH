import jax
import jax.numpy as jnp
import numpy
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
        return eph_real
    
    with h5py.File(fname, "w") as f:
        f.create_dataset("init_eph_real", data=init_eph_real())
    
    def build_phase_factors(param_array):
        phases_hbz = jnp.exp(1j * param_array)
        phases_minus = jnp.conj(phases_hbz[partner_hbz_for_minus])
        return jnp.concatenate([phases_hbz, phases_minus], axis=0)

    def objective(param_array):
        phase_factors = build_phase_factors(param_array)
        rotated_eigvecs = jnp.einsum("qvu, qu, v, qu->qvu", eigvecs, phase_factors, sqrt_mass_per_component, inv_sqrt_freqs)
        eph_real = jnp.zeros((nband, nband, nre_ws, n_delta_r, nmodes), dtype=jnp.complex128)
        for jw in range(nband):
            for iw in range(jw + 1):
                eph_mixed = jnp.einsum("veq, qvu->euq", eph_raw_rot[iw, jw], rotated_eigvecs)
                block = jnp.einsum("euq, rq->eru", eph_mixed, exp_iq_delta_r)
                eph_real = eph_real.at[iw, jw].set(block)

        eph_real = jnp.abs(eph_real)
        metric = -jnp.sum(eph_real)
        return metric, eph_real

    value_and_grad = jax.jit(jax.value_and_grad(objective, has_aux=True))

    metric_history = []
    best_metric = float("inf")
    best_params = params_hbz
    prev_metric = float("inf")
    
    for _ in range(max_iter):
        (metric, eph_real), grad = value_and_grad(params_hbz)
        print(f"Metric: {metric:.6e}")
        metric_history.append(metric)

        if metric < best_metric:
            best_metric = metric
            best_params = params_hbz
            with h5py.File(fname, "a") as f:
                if "eph_real" in f:
                    del f["eph_real"]
                    del f["best_phase_factors"]
                best_phase_factors = build_phase_factors(best_params)
                f.create_dataset("best_phase_factors", data=best_phase_factors)
                f.create_dataset("eph_real", data=eph_real)

        if abs(prev_metric - metric) < tol:
            break
        prev_metric = metric
        params_hbz = params_hbz - learning_rate * grad

    _, best_eph_real = objective(best_params)
    best_phase_factors = build_phase_factors(best_params)
    metric_history = jnp.asarray(metric_history, dtype=jnp.float64)
    return best_eph_real, best_phase_factors, metric_history