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
    """
    We should normalize each mode separately.
    Since we want each mode to be localized, 
    and each each mode g squared is conserved (normalized) by itself.
    This avoids the unequal weight for different modes since g carries the frequency factor.
    """
    g2 = jnp.abs(eph_real) ** 2 # (nband, nband, nre, n_delta_r, nmodes)
    p = jnp.sum(g2, axis=(0, 1, 2, 4))
    p = p / jnp.sum(p)
    r = jnp.linalg.norm(delta_r_vectors, axis=1)
    return p, r

def compute_density_each_band_and_mode(eph_real, delta_r_vectors):
    g2 = jnp.abs(eph_real) ** 2  # (nband, nband, nre, n_delta_r, nmodes)
    g2_norm = jnp.sum(g2, axis=3)  # (nband, nband, nre, nmodes)
    mask = g2_norm > 1e-3 * jnp.max(g2_norm)
    safe_g2_norm = jnp.where(mask, g2_norm, jnp.ones_like(g2_norm))
    p = jnp.einsum('ijeru, ijeu->ijeur', g2, 1.0 / safe_g2_norm)
    p = jnp.where(mask[..., None], p, 0.0)
    p = jnp.sum(p, axis=(0, 1, 2, 3))
    r = jnp.linalg.norm(delta_r_vectors, axis=1)
    p = p / jnp.sum(p)
    return p, r

def localize_reorganization_energy(
    eph_real,
    freqs,
    delta_re_ws,
    delta_r_vectors
):
    freq = jnp.mean(freqs, axis=0)
    inv_omega = 1.0 / freq
    g2 = jnp.abs(eph_real) ** 2 # (nband, nband, nre, n_delta_rp, nmodes)
    g2 = g2.at[1,0].set(0.0)
    reorg = jnp.einsum('ijeru,u->ijer', g2, inv_omega)
    
    # apply mask
    reorg_denom = jnp.sum(reorg, axis=-1)
    mask = reorg_denom * ryd_to_mev > 1
    reorg = jnp.where(mask[..., None], reorg, 0.0)
    
    # # perform normalization for each hopping, not sure if we want this
    # inv_reorg_denom = jnp.where(mask, 1.0 / (reorg_denom + 1e-10), 0.0)
    # reorg = jnp.einsum('ijer, ije->ijer', reorg, inv_reorg_denom)

    reorg = reorg.sum(axis=(0,1)) # (nre, n_delta_rp)
    
    rp_norm = jnp.linalg.norm(delta_r_vectors, axis=1)
    rep_diff_norm = jnp.linalg.norm(delta_re_ws[:, None, :] - delta_r_vectors[None, :, :], axis=-1)
    eq_mask = jnp.isclose(rep_diff_norm, 0.0, atol=1e-6, rtol=0.0)
    weight = jnp.where(eq_mask, -1.0, rp_norm[None, :]+rep_diff_norm)
    arg_rph_zero = jnp.argmin(rp_norm)
    weight = weight.at[:, arg_rph_zero].set(-1.0)
    # weight = weight ** 0.5
    
    loss = jnp.sum(reorg * weight)
    return loss