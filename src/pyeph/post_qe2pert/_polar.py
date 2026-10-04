"""Bounded reciprocal-space contraction for the maintained 3D polar formula.

The cutoff and transpose conventions reproduce PyEPH revision
6c4693acbb69a06a5bc8b0593abde2170ff38843 (BSD-3-Clause). This is host NumPy
preprocessing, independent of the native trajectory model interface.
"""

import numpy as np


def long_range_dynamical_matrix(qpoint, *, reciprocal_basis, positions, born_charges,
                               dielectric, radius, alpha, cutoff, volume, block_size=512):
    """Return packed (3,3,natoms*(natoms+1)//2) long-range atom-pair blocks.

    Each reciprocal vector contributes
    w_G exp[2pi i G.(tau_a-tau_b)] (Z_a G)_i (Z_b G)_j.
    Only the spatial phase is conjugated, preserving the original Born-tensor
    transpose convention. Blocks bound temporary storage by O(block_size*natoms
    + natoms**2); there is no all-G by atom-pair tensor.
    """
    if isinstance(block_size, bool) or not isinstance(block_size, int) or block_size < 1:
        raise ValueError("block_size must be a positive integer")
    radius = np.asarray(radius, dtype=np.int64)
    counts = 2*radius+1
    number = int(np.prod(counts))
    nat = len(positions)
    # A common origin cancels analytically. Remove it before phase evaluation
    # so a large cell-origin translation does not magnify trigonometric error.
    relative_positions = positions-positions[:1] if nat else positions
    result = np.zeros((3*nat, 3*nat), dtype=np.complex128)
    falph, ggmax = 4*alpha, cutoff*4*alpha
    for start in range(0, number, block_size):
        index = np.arange(start, min(number, start+block_size), dtype=np.int64)
        reciprocal = np.stack((index//(counts[1]*counts[2]),
                               (index//counts[2]) % counts[1], index % counts[2]), axis=1)-radius
        crystal = reciprocal+np.asarray(qpoint)
        cartesian = crystal @ reciprocal_basis
        qeq = np.einsum("gi,ij,gj->g", cartesian, dielectric, cartesian)
        # Preserve the scalar reference's hard-cutoff membership. Reordered
        # reductions can otherwise change an O(1) near-Gamma contribution.
        # Absolute products also cover cancellation and anisotropic inputs.
        absolute_cartesian = np.abs(crystal) @ np.abs(reciprocal_basis)
        absolute_products = np.einsum(
            "gi,ij,gj->g", absolute_cartesian, np.abs(dielectric), absolute_cartesian)
        roundoff = 64*np.finfo(np.float64).eps*absolute_products
        near_cutoff = ((np.abs(qeq-1e-14) <= roundoff)
                       | (np.abs(qeq-ggmax) <= roundoff)
                       | ~np.isfinite(qeq) | ~np.isfinite(roundoff))
        for local in np.flatnonzero(near_cutoff):
            scalar_cartesian = reciprocal_basis.T @ (np.asarray(qpoint)+reciprocal[local])
            cartesian[local] = scalar_cartesian
            qeq[local] = scalar_cartesian @ dielectric @ scalar_cartesian
        # Retain the legacy skip predicate, including invalid-input propagation.
        selected = ~((qeq < 1e-14) | (qeq > ggmax))
        cartesian, qeq = cartesian[selected], qeq[selected]
        if not len(qeq):
            continue
        weight = np.exp(-qeq/falph)/qeq
        phase = np.exp(2j*np.pi*(cartesian @ relative_positions.T))
        charge = np.einsum("aij,gj->gai", born_charges, cartesian)
        left = (charge*phase[..., None]).reshape(len(qeq), 3*nat)
        right = (charge*phase.conj()[..., None]).reshape(len(qeq), 3*nat)
        result += left.T @ (weight[:, None]*right)
    ja, ia = np.tril_indices(nat)
    blocks = result.reshape(nat, 3, nat, 3)[ia, :, ja, :].transpose(1, 2, 0)
    return blocks*(8*np.pi/volume)
