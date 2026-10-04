"""Periodic Cartesian EPC evaluated as a Fourier correlation over nuclear cells."""

from dataclasses import dataclass, field
from math import prod

import jax.numpy as jnp
import numpy as np

from pyeph.core._configuration import integer_scalar
from pyeph.models.cartesian_epc import CartesianEPCModel


_FFT_AXES = (0, 1, 2)


@dataclass(frozen=True)
class FourierCartesianEPCModel(CartesianEPCModel):
    """An opt-in Fourier evaluator with the original dynamic EPC parameters.

    Only the displacement-cell translation is transformed. Each electronic
    image channel and its unwrapped current displacement remain separate.
    Parameter gradients pass through the source-coefficient scatter and FFT;
    there is no cached spectrum that can become stale after parameter updates.

    ``mesh`` uses C-order cell enumeration (z fastest). ``max_spectral_bytes``
    bounds one dense complex coefficient grid at the actual evaluation dtype.
    FFT construction, executable workspaces, trajectory batches and other live
    arrays require additional memory. Use CartesianEPCModel's direct chunks
    when a dense spectrum is unsuitable for a large or sparse target cell.
    """

    mesh: tuple = field(kw_only=True)
    max_spectral_bytes: int = field(default=256*1024**2, kw_only=True)

    def __post_init__(self):
        super().__post_init__()
        try:
            raw_mesh = tuple(self.mesh)
        except TypeError as error:
            raise ValueError("mesh must contain three positive integer dimensions") from error
        if len(raw_mesh) != 3:
            raise ValueError("mesh must contain three positive integer dimensions")
        mesh = tuple(integer_scalar(value, "mesh dimension") for value in raw_mesh)
        if any(value < 1 for value in mesh) or prod(mesh) != self.ncells:
            raise ValueError("mesh dimensions must be positive and their product must equal ncells")
        limit = integer_scalar(self.max_spectral_bytes, "max_spectral_bytes")
        if limit < 1:
            raise ValueError("max_spectral_bytes must be positive")
        object.__setattr__(self, "mesh", mesh)
        object.__setattr__(self, "max_spectral_bytes", limit)

    def _spectral_dtype(self, params, dtype):
        spectral = jnp.result_type(dtype, jnp.complex64)
        required = self.ncells*params["hopping"].size*self.natoms*3*np.dtype(spectral).itemsize
        if required > self.max_spectral_bytes:
            raise ValueError(f"Fourier EPC spectral kernel needs {required} bytes, above "
                             f"max_spectral_bytes={self.max_spectral_bytes}; use epc_backend='direct' "
                             "or explicitly raise the spectral limit")
        return spectral

    def _fourier_coefficients(self, params, dtype):
        """Gplus(k,h,a,mu) = sum_r g(r,h,a,mu) exp(+ik.r)."""
        dtype = self._spectral_dtype(params, dtype)
        channels = params["epc_channels"].reshape(-1)
        atoms = params["epc_atoms"].reshape(-1)
        offsets = params["epc_offsets"].reshape(-1)
        # Validation establishes canonical origin cell0 and translation maps.
        cells = params["neighbors"][offsets, 0]
        coefficients = jnp.asarray(params["epc_values"], dtype=dtype).reshape(-1, 3)
        shape = (self.ncells, params["hopping"].size, self.natoms, 3)
        grid = jnp.zeros(shape, dtype=dtype).at[cells, channels, atoms].add(coefficients)
        grid = grid.reshape(*self.mesh, *shape[1:])
        return self.ncells*jnp.fft.ifftn(grid, axes=_FFT_AXES)

    def elements(self, params, q):
        """Evaluate complex or real image amplitudes at Cartesian displacement q."""
        dtype = jnp.result_type(params["hopping"], params["epc_values"], q)
        if params["epc_values"].shape[0] == 0 or not jnp.issubdtype(dtype, jnp.inexact):
            # Constant and exact-integer data need no Fourier allocation.
            return super().elements(params, q)
        coefficients = self._fourier_coefficients(params, dtype)
        displacement = jnp.asarray(q, dtype=coefficients.dtype).reshape(*self.mesh, self.natoms, 3)
        transformed = jnp.fft.fftn(displacement, axes=_FFT_AXES)
        values = jnp.einsum("...ham,...am->...h", coefficients, transformed)
        values = jnp.fft.ifftn(values, axes=_FFT_AXES).reshape(self.ncells, -1)
        if not jnp.issubdtype(dtype, jnp.complexfloating):
            values = values.real.astype(dtype)
        return values+jnp.asarray(params["hopping"], dtype=dtype)

    def contract_gradient(self, params, q, weight):
        """Apply the real-coordinate transpose without a dense derivative tensor."""
        if params["epc_values"].shape[0] == 0:
            return super().contract_gradient(params, q, weight)
        weights = self._electronic_weights(params, weight).reshape(*self.mesh, -1)
        dtype = jnp.result_type(params["hopping"], params["epc_values"], q, weights)
        coefficients = self._fourier_coefficients(params, dtype)
        weights = jnp.asarray(weights, dtype=coefficients.dtype)
        # Existing electronic weights include the Hermitian half-factors.
        # Correlation's transpose uses Gplus(-k), not its complex conjugate.
        reverse = coefficients
        for axis, length in enumerate(self.mesh):
            reverse = jnp.take(reverse, (-jnp.arange(length)) % length, axis=axis)
        transformed = jnp.fft.fftn(weights, axes=_FFT_AXES)
        gradient = jnp.einsum("...ham,...h->...am", reverse, transformed)
        return jnp.fft.ifftn(gradient, axes=_FFT_AXES).real.reshape(q.shape).astype(q.dtype)

    def validate_params(self, params):
        """Validate original stencils and canonical periodic maps on every update."""
        super().validate_params(params)
        if params["epc_values"].shape[0]:
            dtype = jnp.result_type(params["hopping"], params["epc_values"])
            self._spectral_dtype(params, dtype)
        neighbors = np.asarray(params["neighbors"])
        cells = np.array(list(np.ndindex(self.mesh)))
        shifts = cells[neighbors[:, 0]]
        coordinates = (shifts[:, None]+cells[None]) % self.mesh
        expected = np.ravel_multi_index(coordinates.transpose(2, 0, 1), self.mesh)
        if not np.array_equal(neighbors, expected):
            raise ValueError("Fourier EPC requires C-order periodic translation maps")
