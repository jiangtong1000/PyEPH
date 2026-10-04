"""Optional sparse local Torch coefficients with explicit CPU host callbacks.

Torch owns complete atomic derivatives, including the center map, periodic
displacements, provider features and final hopping support. No autodiff graph
crosses the callback boundary. This is a fixed orthonormal carrier model, not
an AO Hamiltonian adapter or a foundation-model implementation.
"""

from dataclasses import dataclass, field, replace

import jax
import jax.numpy as jnp
import numpy as np

from pyeph.core._configuration import boolean_scalar, real_scalar
from pyeph.core.contracts import LowRankWeight
from pyeph.models.local import LocalBlockModel, LocalCoefficients, LocalGeometry


@dataclass(frozen=True)
class TorchLocalBlockModel(LocalBlockModel):
    """Adapt ``provider(torch_params, q_tensor, torch_geometry)`` to local blocks.

The provider returns LocalCoefficients containing CPU Torch floating/complex
tensors: Hermitian onsite (N,b,b) and RAW hopping (E,b,b). Combine every carrier
baseline and residual inside that graph. The adapter applies final support
once; provider-side neighbor messages still need their own support gates.
All coordinates/energies are atomic units. Vref=0: compose a separate neutral
reference in exactly the same coordinate/basis convention when appropriate.

Sparse actions/current probes remain native JAX and transfer only local block
values. Forces use local block cotangents and a complete Torch geometry graph,
without an S-by-S Hamiltonian or its derivative tensor. Current conventions,
fixed graph coverage, Gamma-only periodic scope, rewrapping restrictions and
point-orbital limitations are inherited from LocalBlockModel.

The provider must be deterministic and side-effect free (eval mode, no dropout
or mutable scientific caches). Numerical params remain runtime inputs. New
captured weights/provider code require a new model/Runner and explicit strict
checkpoint identity for model.coefficient_provider. CPU pure_callback uses
sequential vmap; no Torch batched throughput, GPU residency, JAX autodiff or
whole-trajectory differentiation is promised. Existing external-provider
method/execution restrictions apply. The preparation hook can reuse values
within a caller's geometry action; ordinary RK4 does not promise such reuse.

Declare a genuinely constant full Hamiltonian with
coordinate_independent_hamiltonian=True. This only permits disconnected zero
derivatives, never suppresses existing gradients or repairs partially detached
terms. validate_complete_gradients is an opt-in representative-geometry audit.
"""

    coordinate_independent_hamiltonian: bool = field(default=False, kw_only=True)
    audit_atol: float = field(default=2e-6, kw_only=True)
    audit_rtol: float = field(default=2e-4, kw_only=True)
    audit_step: float = field(default=1e-5, kw_only=True)
    execution_mode = "host_callback"

    def __post_init__(self):
        super().__post_init__()
        object.__setattr__(self, "coordinate_independent_hamiltonian", boolean_scalar(
            self.coordinate_independent_hamiltonian, "coordinate_independent_hamiltonian"))
        for name in ("audit_atol", "audit_rtol", "audit_step"):
            value = real_scalar(getattr(self, name), name)
            if value < 0 or (name == "audit_step" and value == 0):
                raise ValueError("audit tolerances must be nonnegative and audit_step positive")
            object.__setattr__(self, name, value)
        try:
            import torch
        except ImportError as exc:
            raise ImportError("TorchLocalBlockModel requires the optional 'torch' dependency") from exc
        object.__setattr__(self, "_torch", torch)
        object.__setattr__(self, "spec", replace(self.spec, name="torch_local_blocks", native_jax=False))

    def _coordinates(self, q):
        q = jnp.asarray(q)
        if q.shape != self.spec.system.q_shape or q.dtype not in (jnp.float32, jnp.float64):
            raise ValueError("local Torch coordinates require q_shape and float32/float64 dtype")
        return q

    def _inputs(self, params, q):
        torch = self._torch
        tp = jax.tree.map(lambda x: torch.as_tensor(np.array(x, copy=True)), params)
        tq = torch.tensor(np.array(q, copy=True), requires_grad=True)
        return tp, tq

    def _torch_geometry(self, q):
        torch = self._torch
        if not bool(torch.isfinite(q).all()):
            raise ValueError("atomic coordinates must be finite")
        sites = torch.tensor(self.centers.atom_site, dtype=torch.long)
        weights = torch.tensor(self.centers.weights, dtype=q.dtype)
        centers = torch.zeros((self.graph.nsites, 3), dtype=q.dtype).index_add(
            0, sites, weights[:, None]*q)
        edges = torch.tensor(self.graph.edges, dtype=torch.long).reshape(-1, 5)
        pairs = edges[:, :2]
        displacement = centers[pairs[:, 1]]-centers[pairs[:, 0]]
        if self.graph.cell is not None:
            displacement = displacement + edges[:, 2:].to(q.dtype) @ torch.tensor(
                self.graph.cell, dtype=q.dtype)
        distances = torch.linalg.vector_norm(displacement, dim=-1)
        if (not bool(torch.isfinite(centers).all()) or not bool(torch.isfinite(distances).all())
                or bool(torch.any(distances == 0))):
            raise ValueError("mapped centers/image distances must be finite and edges noncoincident")
        if self.graph.cutoff is None:
            support = torch.ones_like(distances)
        else:
            u = torch.clamp((distances-self.graph.switch_on)
                            /(self.graph.cutoff-self.graph.switch_on), 0., 1.)
            support = 1.-10.*u**3+15.*u**4-6.*u**5
        return LocalGeometry(centers, pairs, displacement, distances, support, sites, weights)

    def _raw(self, params, q, geometry):
        torch = self._torch
        result = self.coefficient_provider(params, q, geometry)
        if not isinstance(result, LocalCoefficients):
            raise TypeError("coefficient_provider must return LocalCoefficients of Torch tensors")
        n, b, e = self.graph.nsites, self.graph.norbitals, len(self.graph.edges)
        for name, value, shape in zip(("onsite", "hopping"), result, ((n, b, b), (e, b, b))):
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"{name} must be a Torch tensor, not a detached NumPy value")
            if tuple(value.shape) != shape or value.device.type != "cpu":
                raise ValueError(f"{name} requires CPU tensor shape {shape}")
            if not value.is_floating_point() and not value.is_complex():
                raise ValueError(f"{name} requires floating or complex coefficients")
            if value.is_complex() and not self.complex_valued:
                raise ValueError("complex coefficients contradict complex_valued=False")
            if not bool(torch.isfinite(value).all()):
                raise ValueError(f"{name} coefficients must be finite")
        onsite = result.onsite
        real = onsite.real
        imag = onsite.imag if onsite.is_complex() else torch.zeros_like(real)
        scale = torch.maximum(torch.ones_like(real[..., :1, :1]), torch.maximum(
            real.abs().amax(dim=(-2, -1), keepdim=True), imag.abs().amax(dim=(-2, -1), keepdim=True)))
        real, imag = real/scale, imag/scale
        tolerance = 64*torch.finfo(q.dtype).eps
        if (bool(torch.any((real-real.transpose(-1, -2)).abs() > tolerance))
                or bool(torch.any((imag+imag.transpose(-1, -2)).abs() > tolerance))):
            raise ValueError("onsite coefficients must be Hermitian; no symmetrization is applied")
        return result

    def _physical(self, params, q):
        geometry = self._torch_geometry(q)
        onsite, hopping = self._raw(params, q, geometry)
        return LocalCoefficients(onsite, hopping*geometry.support[:, None, None])

    @staticmethod
    def _array(value, dtype, name):
        with np.errstate(over="ignore", invalid="ignore"):
            result = np.asarray(value.detach().cpu().numpy(), dtype=dtype)
        if not np.isfinite(result).all():
            raise ValueError(f"{name} is nonfinite in callback output dtype")
        return result

    def _coefficients(self, params, q, geometry):
        q = self._coordinates(q)
        dtype = jnp.result_type(q, 1j if self.complex_valued else 1.)
        n, b, e = self.graph.nsites, self.graph.norbitals, len(self.graph.edges)
        structure = LocalCoefficients(jax.ShapeDtypeStruct((n, b, b), dtype),
                                      jax.ShapeDtypeStruct((e, b, b), dtype))

        def callback(p, x):
            tp, tq = self._inputs(p, x)
            raw = self._raw(tp, tq, self._torch_geometry(tq))
            return LocalCoefficients(*(self._array(value, dtype, name)
                                       for name, value in zip(("onsite", "hopping"), raw)))

        raw = jax.pure_callback(callback, structure, params, q, vmap_method="sequential")
        return LocalCoefficients(raw.onsite, raw.hopping*geometry.support[:, None, None])

    def _block_cotangents(self, weight):
        n, b = self.graph.nsites, self.graph.norbitals
        edges = jnp.asarray(self.graph.edges, dtype=jnp.int32).reshape(-1, 5)
        i, j = edges[:, 0], edges[:, 1]
        if isinstance(weight, LowRankWeight):
            left, right = jnp.asarray(weight.left), jnp.asarray(weight.right)
            if (left.ndim != 2 or left.shape[0] != self.nstates or left.shape != right.shape
                    or left.dtype.kind not in "iufc" or right.dtype.kind not in "iufc"):
                raise ValueError("weight factors must have equal numeric (nstates,rank) shapes")
            left, right = left.reshape(n, b, -1), right.reshape(n, b, -1)
            onsite = jnp.einsum("iak,ibk->iab", left, right.conj())
            hopping = (jnp.einsum("eak,ebk->eab", left[i], right[j].conj())
                       + jnp.einsum("eak,ebk->eab", right[i], left[j].conj()))
        else:
            dense = jnp.asarray(weight)
            if dense.shape != (self.nstates, self.nstates) or dense.dtype.kind not in "iufc":
                raise ValueError("dense weight must have numeric shape (nstates,nstates)")
            blocks = dense.reshape(n, b, n, b)
            sites = jnp.arange(n)
            onsite = blocks[sites, :, sites, :]
            hopping = blocks[i, :, j, :]+blocks[j, :, i, :].conj().swapaxes(-1, -2)
        return LocalCoefficients(onsite, hopping)

    def _gradient(self, scalar, q):
        torch = self._torch
        if not bool(torch.isfinite(scalar)):
            raise ValueError("contracted local energy is nonfinite")
        gradient = None
        if scalar.requires_grad:
            gradient, = torch.autograd.grad(scalar, q, allow_unused=True)
        if gradient is None:
            if not self.coordinate_independent_hamiltonian:
                raise ValueError("local Hamiltonian is disconnected from coordinates; declare genuine constants explicitly")
            gradient = torch.zeros_like(q)
        if tuple(gradient.shape) != tuple(q.shape) or not bool(torch.isfinite(gradient).all()):
            raise ValueError("contracted local gradient must be finite and match coordinates")
        return gradient

    def contract_gradient(self, params, q, weight):
        q = self._coordinates(q)
        cotangents = self._block_cotangents(weight)

        def callback(p, x, weights):
            torch = self._torch
            tp, tq = self._inputs(p, x)
            physical = self._physical(tp, tq)
            tw = tuple(torch.as_tensor(np.array(w, copy=True)) for w in weights)
            if not all(bool(torch.isfinite(w).all()) for w in tw):
                raise ValueError("local block cotangents must be finite")
            scalar = sum(torch.sum(w.conj()*block).real for w, block in zip(tw, physical))
            return self._array(self._gradient(scalar, tq), x.dtype, "contracted local gradient")

        return jax.pure_callback(callback, jax.ShapeDtypeStruct(q.shape, q.dtype),
                                 params, q, cotangents, vmap_method="sequential")

    def validate_params(self, params):
        for value in jax.tree.leaves(params):
            array = np.asarray(value)
            if array.dtype.kind not in "iufc" or not np.isfinite(array).all():
                raise ValueError("local Torch params must be finite numerical arrays")
        super().validate_params(params)

    def validate_at(self, params, q, *, batch=False):
        """Sequential CPU preflight; no full-domain physics/gradient certification."""
        batch = boolean_scalar(batch, "batch")
        self.validate_params(params)
        values = self._validate_geometry(q, batch)
        values = values if batch else values[None]
        for coordinate in values:
            x = np.asarray(self._coordinates(coordinate))
            self._validate_geometry(x, False)
            tp, tq = self._inputs(params, x)
            physical = self._physical(tp, tq)
            dtype = jnp.result_type(x, 1j if self.complex_valued else 1.)
            for name, value in zip(("onsite", "hopping"), physical):
                self._array(value, dtype, name)
            # This probes connection/finiteness only; finite differences below
            # are needed to detect a detached term hidden beside an attached one.
            scalar = sum(value.real.mean() if value.numel() else value.real.sum()
                         for value in physical)
            self._array(self._gradient(scalar, tq), x.dtype, "preflight gradient")

    def validate_complete_gradients(self, params, q):
        """Compare full supported block JVPs to physical-value finite differences.

The test uses O((N+E)b²) outputs per direction, never a dense H/Jacobian.
Native value evaluation also checks agreement between the independent small
        JAX/Torch geometry implementations. The number of directions is linear
        in atomic coordinate count; each JVP/provider evaluation has its own
        graph/model cost and requires Torch higher-AD operator support.
"""
        self.validate_at(params, q)
        x = np.asarray(self._coordinates(q))
        tp, tq = self._inputs(params, x)
        torch = self._torch

        def outputs(value):
            blocks = self._physical(tp, value)
            return torch.cat(tuple(part.reshape(-1) for block in blocks for part in
                                   (block.real, block.imag if block.is_complex() else torch.zeros_like(block))))

        def native_outputs(value):
            blocks = tuple(np.asarray(block) for block in self.coefficients(params, value))
            return np.concatenate(tuple(part.reshape(-1) for block in blocks
                                        for part in (block.real, block.imag)))

        eps = np.finfo(x.dtype).eps
        for index in range(x.size):
            direction = torch.zeros_like(tq)
            direction.reshape(-1)[index] = 1.
            _, derivative = torch.autograd.functional.jvp(outputs, tq, direction)
            h = max(self.audit_step, eps**(1/3))*max(1., abs(float(x.reshape(-1)[index])))
            dx = np.zeros_like(x)
            dx.reshape(-1)[index] = h
            finite = (native_outputs(x+dx)-native_outputs(x-dx))/(2*h)
            actual = derivative.detach().numpy()
            if (not np.isfinite(finite).all() or not np.isfinite(actual).all()
                    or not np.allclose(actual, finite, atol=max(self.audit_atol, 200*eps), rtol=self.audit_rtol)):
                raise ValueError(f"incomplete local derivatives at coordinate {index}; "
                                 "check detached baseline/residual or geometry terms")
