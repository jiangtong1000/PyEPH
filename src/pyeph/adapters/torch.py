"""Optional CPU PyTorch provider with explicit JAX host-callback boundaries.

PyTorch owns all geometry differentiation. ``jax.pure_callback`` transfers
evaluated arrays and complete contracted gradients; it does not transfer an
autograd graph. Differentiation through this adapter with ``jax.grad`` is not
supported. This dense adapter prioritizes correctness and interoperability,
not accelerator-resident trajectory performance.
"""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np

from pyeph.core.contracts import LowRankWeight, ProbeContext
from pyeph.models.base import AutoDiffModel


class TorchHamiltonianAdapter(AutoDiffModel):
    """Adapt ``hamiltonian_fn(torch_params, q_tensor) -> H_tensor``.

    ``reference_fn`` has the same arguments and returns a real scalar. Every
    coordinate-dependent baseline and residual must be inside these Torch
    graphs. Legitimately constant outputs require explicit corresponding
    ``coordinate_independent_*`` declarations. Functions must be deterministic
    and pure: set neural modules to eval mode and avoid mutable caches/RNG.

    Optional probe functions receive ``(torch_params, ProbeContext)`` and return
    dense operators. Their physical meaning is supplied by the caller. No
    current operator is inferred from arbitrary Hamiltonian entries.
    """

    execution_mode = "host_callback"

    def __init__(self, spec, hamiltonian_fn, reference_fn=None, probes=None, *,
                 coordinate_independent_hamiltonian=False,
                 coordinate_independent_reference=False,
                 audit_atol=2e-6, audit_rtol=2e-4, audit_step=1e-5):
        try:
            import torch
        except ImportError as exc:
            raise ImportError("TorchHamiltonianAdapter requires the optional 'torch' dependency") from exc
        self._torch = torch
        self.hamiltonian_fn = hamiltonian_fn
        self.reference_fn = reference_fn
        self.probes = {} if probes is None else dict(probes)
        self.spec = replace(spec, native_jax=False, probes=tuple(self.probes))
        self.coordinate_independent_hamiltonian = coordinate_independent_hamiltonian
        self.coordinate_independent_reference = coordinate_independent_reference
        if audit_atol < 0 or audit_rtol < 0 or audit_step <= 0:
            raise ValueError("derivative audit tolerances must be nonnegative and step positive")
        self.audit_atol, self.audit_rtol, self.audit_step = audit_atol, audit_rtol, audit_step

    def _inputs(self, params, q):
        torch = self._torch
        # Copy callback inputs: a shared JAX buffer must never be modified by
        # a framework with mutable tensor storage.
        tp = jax.tree.map(lambda x: torch.as_tensor(np.array(x, copy=True)), params)
        tq = torch.tensor(np.array(q, copy=True), requires_grad=True)
        return tp, tq

    def _check_tensor(self, value, shape, name, *, constant=False, derivative=False):
        torch = self._torch
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{name} must return a PyTorch tensor, not a detached NumPy value")
        if tuple(value.shape) != shape:
            raise ValueError(f"{name} has shape {tuple(value.shape)}; expected {shape}")
        if value.device.type != "cpu":
            raise ValueError("this host-callback adapter requires CPU Torch tensors")
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"{name} returned nonfinite values")
        if derivative and not value.requires_grad and not constant:
            raise ValueError(f"{name} is detached from coordinates; declare genuine constants explicitly")
        return value

    def _hamiltonian(self, params, q, *, derivative=False):
        h = self._check_tensor(self.hamiltonian_fn(params, q), (self.nstates, self.nstates),
                               "Hamiltonian", constant=self.coordinate_independent_hamiltonian,
                               derivative=derivative)
        torch = self._torch
        tolerance = 1e-10 if h.dtype in (torch.float64, torch.complex128) else 1e-5
        if not bool(torch.allclose(h, h.mH, atol=tolerance, rtol=tolerance)):
            raise ValueError("Torch Hamiltonian must be Hermitian")
        if not self.spec.complex_valued and h.is_complex():
            raise ValueError("complex Torch Hamiltonian requires complex_valued=True")
        return h

    def _reference(self, params, q, *, derivative=False):
        value = q.sum()*0 if self.reference_fn is None else self.reference_fn(params, q)
        value = self._check_tensor(value, (), "reference energy",
                                    constant=self.coordinate_independent_reference,
                                    derivative=derivative)
        if value.is_complex():
            raise ValueError("reference energy must be real")
        return value

    def _array(self, tensor, dtype):
        return np.asarray(tensor.detach().cpu().numpy(), dtype=dtype)

    def _h_dtype(self, q):
        return jnp.result_type(q, 1j if self.spec.complex_valued else 1.)

    def dense(self, params, q):
        dtype = self._h_dtype(q)
        def callback(p, x):
            tp, tq = self._inputs(p, x)
            return self._array(self._hamiltonian(tp, tq), dtype)
        return jax.pure_callback(callback, jax.ShapeDtypeStruct((self.nstates, self.nstates), dtype),
                                 params, q, vmap_method="sequential")

    def apply(self, params, q, vectors):
        return self.dense(params, q) @ vectors

    def reference_energy(self, params, q):
        dtype = jnp.asarray(q).dtype
        def callback(p, x):
            tp, tq = self._inputs(p, x)
            return self._array(self._reference(tp, tq), dtype)
        return jax.pure_callback(callback, jax.ShapeDtypeStruct((), dtype), params, q,
                                 vmap_method="sequential")

    def _gradient(self, scalar, q, *, constant=False):
        if not scalar.requires_grad:
            if not constant:
                raise ValueError("geometry-dependent scalar is detached from its Torch graph")
            return self._torch.zeros_like(q)
        (gradient,) = self._torch.autograd.grad(scalar, q, allow_unused=True)
        if gradient is None:
            if not constant:
                raise ValueError("Torch output depends on parameters but is disconnected from coordinates")
            return self._torch.zeros_like(q)
        return gradient

    def reference_gradient(self, params, q):
        if not self.spec.force_support:
            raise ValueError("this Torch provider does not declare force support")
        dtype = jnp.asarray(q).dtype
        def callback(p, x):
            tp, tq = self._inputs(p, x)
            value = self._reference(tp, tq, derivative=True)
            g = self._gradient(value, tq, constant=self.coordinate_independent_reference)
            return self._array(g, dtype)
        return jax.pure_callback(callback, jax.ShapeDtypeStruct(q.shape, dtype), params, q,
                                 vmap_method="sequential")

    def contract_gradient(self, params, q, weight):
        if not self.spec.force_support:
            raise ValueError("this Torch provider does not declare force support")
        dtype = jnp.asarray(q).dtype
        factored = isinstance(weight, LowRankWeight)
        weight = jax.tree.map(jax.lax.stop_gradient, weight)
        def callback(p, x, w):
            torch = self._torch
            tp, tq = self._inputs(p, x)
            h = self._hamiltonian(tp, tq, derivative=True)
            if factored:
                left, right = (torch.as_tensor(np.array(a, copy=True)) for a in w)
                dtype_h = torch.promote_types(h.dtype, right.dtype)
                scalar = torch.sum(left.conj() * (h.to(dtype_h) @ right.to(dtype_h))).real
            else:
                tw = torch.as_tensor(np.array(w, copy=True))
                scalar = torch.sum(tw.conj() * h).real
            g = self._gradient(scalar, tq, constant=self.coordinate_independent_hamiltonian)
            return self._array(g, dtype)
        return jax.pure_callback(callback, jax.ShapeDtypeStruct(q.shape, dtype), params, q, weight,
                                 vmap_method="sequential")

    def probe_apply(self, params, context, probe, vectors):
        if probe not in self.probes:
            return super().probe_apply(params, context, probe, vectors)
        dtype = jnp.result_type(context.q, 1j)
        def callback(p, ctx):
            tp, tq = self._inputs(p, ctx.q)
            def convert(v):
                return None if v is None else self._torch.as_tensor(np.array(v, copy=True))
            tc = ProbeContext(tq, convert(ctx.velocity), convert(ctx.time))
            value = self._check_tensor(self.probes[probe](tp, tc),
                                       (self.nstates, self.nstates), f"probe {probe}")
            return self._array(value, dtype)
        matrix = jax.pure_callback(callback, jax.ShapeDtypeStruct((self.nstates, self.nstates), dtype),
                                   params, context, vmap_method="sequential")
        return matrix @ vectors

    def validate_geometry(self, q):
        q = np.asarray(q)
        if q.shape != self.spec.system.q_shape or not np.all(np.isfinite(q)) or np.iscomplexobj(q):
            raise ValueError("Torch geometry must match the declared finite real coordinate shape")

    def validate_params(self, params):
        for value in jax.tree.leaves(params):
            if not np.all(np.isfinite(np.asarray(value))):
                raise ValueError("Torch callback parameters must contain finite numerical arrays")

    def validate_complete_gradients(self, params, q):
        """Preflight every coordinate's H and V_ref derivative by central difference.

        This detects attached residuals hiding detached baseline terms at the
        audited geometry. It is a local numerical audit, not a proof over an
        arbitrary model's full domain. Cost scales with coordinate count and
        dense output size; run it before production, never inside a JIT loop.
        """
        self.validate_geometry(q)
        q_array = np.asarray(q)
        tp, tq = self._inputs(params, q_array)
        if not self.spec.force_support:
            self._hamiltonian(tp, tq)
            self._reference(tp, tq)
            return
        torch = self._torch
        self._hamiltonian(tp, tq, derivative=True)
        self._reference(tp, tq, derivative=True)
        def outputs(x):
            h = self._hamiltonian(tp, x)
            ref = self._reference(tp, x)
            imaginary = h.imag if h.is_complex() else torch.zeros_like(h)
            return torch.cat((h.real.reshape(-1), imaginary.reshape(-1), ref.reshape(1)))
        for i in range(tq.numel()):
            direction = torch.zeros_like(tq)
            direction.reshape(-1)[i] = 1
            _, derivative = torch.autograd.functional.jvp(outputs, tq, direction, create_graph=False)
            eps = np.finfo(q_array.dtype).eps
            step = max(self.audit_step, eps**(1/3)) * max(1., abs(float(tq.reshape(-1)[i].detach())))
            finite = (outputs(tq+step*direction)-outputs(tq-step*direction))/(2*step)
            atol = max(self.audit_atol, 200*eps)
            if not torch.allclose(derivative, finite, atol=atol, rtol=self.audit_rtol):
                error = float(torch.max(torch.abs(derivative-finite)).detach())
                raise ValueError(f"incomplete Torch geometry derivatives at coordinate {i}: maximum error {error:.3g}; "
                                 "check detached baseline/residual terms")
