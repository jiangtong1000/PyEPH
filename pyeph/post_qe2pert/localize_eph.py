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
    g2_r = jnp.sum(g2, axis=(0, 1, 2))
    p = jnp.einsum('ru, u->ru', g2_r, 1.0/jnp.sum(g2_r, axis=0))
    r = jnp.linalg.norm(delta_r_vectors, axis=1)
    return p, r