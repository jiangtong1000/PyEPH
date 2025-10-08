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

def localize_reorganization_energy(eph_real, freqs, delta_re_ws, delta_r_vectors):
    freq = jnp.mean(freqs, axis=0)
    g2 = jnp.abs(eph_real) ** 2
    reorg_energy = jnp.einsum("ijeru, u->ijeru", g2, 1/freq)
    reorg_energy = reorg_energy.sum(axis=(0, 1, 4)) # (nRe, n_delta_r)
    diff_re_rph = delta_re_ws[:, None, :] / 2 - delta_r_vectors[None, :, :]
    diff_re_rph_norm2 = jnp.linalg.norm(diff_re_rph, axis=-1) # (nRe, n_delta_r)
    loss = jnp.sum(reorg_energy * diff_re_rph_norm2)
    return loss

# compute_density = compute_density_each_band_and_mode