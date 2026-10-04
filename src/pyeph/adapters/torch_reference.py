"""Optional reference-only CPU Torch provider with explicit host callbacks.

Torch owns differentiation of the scalar reference potential. No autograd graph
crosses the JAX boundary, and no electronic matrix is constructed. This adapter
does not integrate a particular xTB or foundation model by itself.
"""

from dataclasses import dataclass, field, replace
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from pyeph.core._configuration import boolean_scalar, real_scalar
from pyeph.core.contracts import LowRankWeight, ModelSpec
from pyeph.models.base import AutoDiffModel


@dataclass(frozen=True)
class TorchReferenceModel(AutoDiffModel):
    """Adapt ``reference_fn(torch_params, q_tensor) -> real scalar tensor``.

    Supply the same ModelSpec coordinates, basis, sector, energy convention and
    units as the carrier model. The provider must return energy in those units;
    input/output conversions belong INSIDE its Torch graph. All baseline and
    residual energy terms must remain attached to coordinates. Declare genuine
    constants explicitly with ``coordinate_independent_reference=True``.

    ``apply`` and ``contract_gradient`` are exact native zeros. ``zero_probes``
    explicitly declares electronic probe contributions that are zero for this
    reference, allowing valid SumModel probe addition without dense identities
    or callbacks. Position/other operators are not inherited automatically.

    Only reference_energy and reference_gradient use CPU pure_callback, with
    sequential vmap semantics. No GPU residency, batched Torch inference,
    cross-framework AD, or full-trajectory differentiation is promised. Current
    host-callback execution permission and native-only method restrictions still
    apply. Numerical params are runtime inputs; replace the model/Simulation if
    provider code, captured weights or calculator configuration changes. Strict
    checkpoints require an explicit artifact identity for ``model.reference_fn``
    (or its nested composition path), covering code and captured artifacts.

    The callable must be deterministic and free of externally visible side
    effects. Configure neural modules for inference and manage any calculator
    caches so repeated evaluations have identical energies and complete forces.
    ``validate_complete_gradients`` is an opt-in local finite-difference audit;
    it is not proof over all future geometries.
    """

    spec: ModelSpec
    reference_fn: Any
    zero_probes: tuple = field(default=(), kw_only=True)
    coordinate_independent_reference: bool = field(default=False, kw_only=True)
    audit_atol: float = field(default=2e-6, kw_only=True)
    audit_rtol: float = field(default=2e-4, kw_only=True)
    audit_step: float = field(default=1e-5, kw_only=True)
    execution_mode = "host_callback"

    def __post_init__(self):
        if not isinstance(self.spec, ModelSpec):
            raise TypeError("spec must be a ModelSpec matching the carrier representation")
        if not callable(self.reference_fn):
            raise TypeError("reference_fn must be callable")
        if isinstance(self.zero_probes, str):
            raise ValueError("zero_probes must be a collection of explicitly zero probe names")
        probes = tuple(self.zero_probes)
        if len(set(probes)) != len(probes):
            raise ValueError("zero_probes must not contain duplicates")
        object.__setattr__(self, "zero_probes", probes)
        object.__setattr__(self, "coordinate_independent_reference", boolean_scalar(
            self.coordinate_independent_reference, "coordinate_independent_reference"))
        for name in ("audit_atol", "audit_rtol", "audit_step"):
            value = real_scalar(getattr(self, name), name)
            if value < 0 or (name == "audit_step" and value == 0):
                raise ValueError("audit tolerances must be nonnegative and audit_step positive")
            object.__setattr__(self, name, value)
        object.__setattr__(self, "spec", replace(
            self.spec, name="torch_reference", native_jax=False, force_support=True, probes=probes))
        try:
            import torch
        except ImportError as exc:
            raise ImportError("TorchReferenceModel requires the optional 'torch' dependency") from exc
        # A runtime module handle is not scientific state and is intentionally
        # absent from dataclass fields/provenance; configuration is above.
        object.__setattr__(self, "_torch", torch)

    def _coordinates(self, q):
        q = jnp.asarray(q)
        if q.shape != self.spec.system.q_shape or q.dtype not in (jnp.float32, jnp.float64):
            raise ValueError("reference coordinates must match q_shape with float32 or float64 dtype")
        return q

    def _vectors(self, vectors):
        vectors = jnp.asarray(vectors)
        if (vectors.ndim not in (1, 2) or vectors.shape[0] != self.nstates
                or vectors.dtype.kind not in "iufc"):
            raise ValueError("electronic vectors must have numeric shape (nstates,) or (nstates,ncolumns)")
        return vectors

    def apply(self, params, q, vectors):
        return jnp.zeros_like(self._vectors(vectors))

    def contract_gradient(self, params, q, weight):
        q = self._coordinates(q)
        if isinstance(weight, LowRankWeight):
            left, right = jnp.asarray(weight.left), jnp.asarray(weight.right)
            if left.ndim != 2 or left.shape[0] != self.nstates or right.shape != left.shape:
                raise ValueError("weight factors must have equal (nstates,rank) shapes")
        elif jnp.asarray(weight).shape != (self.nstates, self.nstates):
            raise ValueError("dense weight must have shape (nstates,nstates)")
        return jnp.zeros_like(q)

    def probe_apply(self, params, context, probe, vectors):
        if probe not in self.spec.probes:
            return super().probe_apply(params, context, probe, vectors)
        return jnp.zeros_like(self._vectors(vectors))

    def _inputs(self, params, q):
        torch = self._torch
        # Never share mutable Torch storage with a JAX callback input buffer.
        tp = jax.tree.map(lambda x: torch.as_tensor(np.array(x, copy=True)), params)
        tq = torch.tensor(np.array(q, copy=True), requires_grad=True)
        return tp, tq

    def _reference(self, params, q, *, derivative=False):
        torch = self._torch
        value = self.reference_fn(params, q)
        if not isinstance(value, torch.Tensor):
            raise TypeError("reference_fn must return a PyTorch tensor, not a detached scalar/NumPy value")
        if tuple(value.shape) != () or value.device.type != "cpu" or not value.is_floating_point():
            raise ValueError("reference_fn must return a real floating scalar CPU tensor")
        if not bool(torch.isfinite(value)):
            raise ValueError("reference_fn returned nonfinite energy")
        if derivative and not value.requires_grad and not self.coordinate_independent_reference:
            raise ValueError("reference energy is detached from coordinates; declare genuine constants explicitly")
        return value

    def _gradient(self, value, q):
        torch = self._torch
        if value.requires_grad:
            (gradient,) = torch.autograd.grad(value, q, allow_unused=True)
        else:
            gradient = None
        if gradient is None:
            if not self.coordinate_independent_reference:
                raise ValueError("reference energy is disconnected from coordinates")
            gradient = torch.zeros_like(q)
        if (not isinstance(gradient, torch.Tensor) or tuple(gradient.shape) != tuple(q.shape)
                or gradient.device.type != "cpu" or not gradient.is_floating_point()
                or not bool(torch.isfinite(gradient).all())):
            raise ValueError("reference gradient must be finite real CPU data matching q_shape")
        return gradient

    @staticmethod
    def _array(value, dtype, name):
        with np.errstate(over="ignore", invalid="ignore"):
            array = np.asarray(value.detach().cpu().numpy(), dtype=dtype)
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} is not finite in the callback output dtype")
        return array

    def reference_energy(self, params, q):
        q = self._coordinates(q)

        def callback(p, x):
            tp, tq = self._inputs(p, x)
            return self._array(self._reference(tp, tq), x.dtype, "reference energy")

        return jax.pure_callback(callback, jax.ShapeDtypeStruct((), q.dtype), params, q,
                                 vmap_method="sequential")

    def reference_gradient(self, params, q):
        q = self._coordinates(q)

        def callback(p, x):
            tp, tq = self._inputs(p, x)
            value = self._reference(tp, tq, derivative=True)
            return self._array(self._gradient(value, tq), x.dtype, "reference gradient")

        return jax.pure_callback(callback, jax.ShapeDtypeStruct(q.shape, q.dtype), params, q,
                                 vmap_method="sequential")

    def validate_params(self, params):
        for value in jax.tree.leaves(params):
            array = np.asarray(value)
            if array.dtype.kind not in "iufc" or not np.all(np.isfinite(array)):
                raise ValueError("reference params must be finite numerical arrays")

    def validate_geometry(self, q):
        values = np.asarray(self._coordinates(q))
        if not np.all(np.isfinite(values)):
            raise ValueError("reference coordinates must be finite")

    def validate_at(self, params, q, *, batch=False):
        batch = boolean_scalar(batch, "batch")
        self.validate_params(params)
        values = np.asarray(q)
        if batch:
            if values.ndim != len(self.spec.system.q_shape)+1 or values.shape[0] < 1:
                raise ValueError("batched coordinates need one nonempty leading trajectory axis")
        else:
            values = values[None]
        # This external provider intentionally follows its sequential callback
        # batch policy. No native/GPU batch throughput is implied by preflight.
        for coordinate in values:
            self.validate_geometry(coordinate)
            native = np.asarray(self._coordinates(coordinate))
            tp, tq = self._inputs(params, native)
            value = self._reference(tp, tq, derivative=True)
            self._array(value, native.dtype, "reference energy")
            self._array(self._gradient(value, tq), native.dtype, "reference gradient")

    def validate_complete_gradients(self, params, q):
        """Audit the COMPLETE scalar energy against all coordinate derivatives.

        In particular this catches a detached baseline hidden beside an
        attached residual. Cost is linear in coordinate count in external
        evaluations; call on representative geometries, not each nuclear step.
        """
        self.validate_at(params, q)
        q_array = np.asarray(self._coordinates(q))
        tp, tq = self._inputs(params, q_array)
        value = self._reference(tp, tq, derivative=True)
        gradient = self._gradient(value, tq).detach()
        eps = np.finfo(q_array.dtype).eps
        atol = max(self.audit_atol, 200*eps)
        for index in range(tq.numel()):
            direction = self._torch.zeros_like(tq)
            direction.reshape(-1)[index] = 1
            step = max(self.audit_step, eps**(1/3))*max(1., abs(float(tq.reshape(-1)[index].detach())))
            plus = self._reference(tp, tq+step*direction)
            minus = self._reference(tp, tq-step*direction)
            finite = (plus-minus)/(2*step)
            if not bool(self._torch.isfinite(finite)):
                raise ValueError("finite-difference reference audit returned nonfinite derivative")
            dtype = self._torch.promote_types(gradient.dtype, finite.dtype)
            actual, expected = gradient.reshape(-1)[index].to(dtype), finite.to(dtype)
            if not self._torch.allclose(actual, expected, atol=atol, rtol=self.audit_rtol):
                error = float(abs(actual-expected).detach())
                raise ValueError(f"incomplete reference derivatives at coordinate {index}: error {error:.3g}; "
                                 "check detached baseline/residual terms")
