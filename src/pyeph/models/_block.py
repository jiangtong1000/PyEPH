"""Shared action for fixed, uniformly sized Hermitian site blocks."""

import jax.numpy as jnp


def block_action(onsite, hopping, edges, vectors):
    """Apply onsite blocks plus each unique edge and its conjugate transpose.

    The first two entries of each static edge are its site indices. A self-image
    edge contributes both T and T† to the same site block. No dense Hamiltonian
    or coordinate derivative tensor is constructed.
    """
    nsites, norbitals = onsite.shape[:2]
    scalar = vectors.ndim == 1
    v = (vectors[:, None] if scalar else vectors).reshape(nsites, norbitals, -1)
    out = jnp.einsum("iab,ibk->iak", onsite, v)
    out = out.astype(jnp.result_type(out, hopping))
    if edges:
        e = jnp.asarray(edges)
        i, j = e[:, 0], e[:, 1]
        out = out.at[i].add(jnp.einsum("eab,ebk->eak", hopping, v[j]))
        out = out.at[j].add(jnp.einsum("eba,ebk->eak", hopping.conj(), v[i]))
    out = out.reshape(nsites * norbitals, -1)
    return out[:, 0] if scalar else out
