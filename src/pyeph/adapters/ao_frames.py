"""Validate moving-AO frames and project them to a recorded adiabatic path.

Rows are *ket coefficients*: phi[a] = sum_mu chi[mu] C[a,mu]. Therefore
G = C.conj() @ S @ C.T and O01 = C0.conj() @ S01 @ C1.T. S01 contains
<chi_mu(t0)|chi_nu(t1)>; it is not a same-time metric or an antisymmetrized
NAC. RecordedCPA transports coefficients with O01.conj().T.

The instantaneous metric and cross-time metric must both be supplied with
their declared coefficient ordering. This helper does not load files, infer
AO/spin ordering, verify eigenpairs without H, or provide forces. These array
conventions are independent of the program used to produce the input data.
"""

from dataclasses import dataclass, field
import hashlib
import json

import jax
import numpy as np
from scipy.linalg import solve_triangular

from pyeph.core.units import ATOMIC_TIME_FS, HARTREE_EV, UnitSystem
from pyeph.paths.electronic import AdiabaticElectronicPath

_TOLERANCE = 1e-10
_RANK_TOLERANCE = 1e-12


def _text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _digest(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class AOArrayEvidence:
    """Exact numerical input bytes, dtype and shape (C-order serialization)."""

    name: str
    shape: tuple
    dtype: str
    sha256: str

    def __post_init__(self):
        _text(self.name, "array name")
        _text(self.dtype, "array dtype")
        if (not isinstance(self.shape, tuple)
                or any(type(x) is not int or x < 0 for x in self.shape)):
            raise TypeError("evidence shape must be an immutable tuple of dimensions")
        if (not isinstance(self.sha256, str) or len(self.sha256) != 64
                or any(x not in "0123456789abcdef" for x in self.sha256)):
            raise ValueError("evidence sha256 must be a hexadecimal SHA-256 digest")


def _array_evidence(name, array):
    return AOArrayEvidence(name, tuple(array.shape), array.dtype.str,
                           hashlib.sha256(array.tobytes(order="C")).hexdigest())


def _array_identity(array):
    a = np.asarray(array)
    e = _array_evidence("array", a)
    return {"shape": e.shape, "dtype": e.dtype, "sha256": e.sha256}


def _path_digest(times, energies, overlaps, basis_id, transport_mode, unit_system):
    return _digest({"times": _array_identity(times), "energies": _array_identity(energies),
                    "overlaps": _array_identity(overlaps), "basis_id": basis_id,
                    "transport_mode": transport_mode,
                    "energy_hartree": unit_system.energy_hartree,
                    "length_bohr": unit_system.length_bohr})


@dataclass(frozen=True)
class AOProjectionEvidence:
    """Immutable source assertions and hashes bound to the final native path.

The caller identifies the external dataset/exporter and the common ordered AO
labels (including spin/k sector) with source_identity and ao_basis_id. Hashes
identify the supplied arrays; they do not certify these external assertions.
Strict RecordedCPA checkpoints automatically include this dataclass.
"""

    source_identity: str
    ao_basis_id: str
    energy_unit: str
    time_unit: str
    energy_to_hartree: float
    time_to_atomic: float
    retained_bands: tuple
    input_arrays: tuple
    projected_path_sha256: str
    schema_version: int = 1
    coefficient_convention: str = "row_kets"
    cross_metric_convention: str = "bra_previous_ket_next"
    validation_tolerance: float = _TOLERANCE
    rank_tolerance: float = _RANK_TOLERANCE

    def __post_init__(self):
        for name in ("source_identity", "ao_basis_id"):
            _text(getattr(self, name), name)
        if (not isinstance(self.energy_unit, str) or not isinstance(self.time_unit, str)
                or self.energy_unit not in {"eV", "hartree"}
                or self.time_unit not in {"fs", "atomic"}):
            raise ValueError("evidence must declare supported source units")
        if (type(self.energy_to_hartree) is not float or type(self.time_to_atomic) is not float
                or not np.isfinite([self.energy_to_hartree, self.time_to_atomic]).all()
                or min(self.energy_to_hartree, self.time_to_atomic) <= 0):
            raise ValueError("evidence conversion factors must be finite positive Python floats")
        if (not isinstance(self.retained_bands, tuple)
                or not all(isinstance(row, tuple) and row
                           and all(type(x) is int and x >= 0 for x in row)
                           for row in self.retained_bands)):
            raise TypeError("retained bands evidence must be an immutable tuple of index tuples")
        if (not isinstance(self.input_arrays, tuple)
                or not all(isinstance(x, AOArrayEvidence) for x in self.input_arrays)):
            raise TypeError("input evidence must be a tuple of AOArrayEvidence")
        AOArrayEvidence("projected_path", (), "digest", self.projected_path_sha256)
        if (type(self.schema_version) is not int or self.schema_version != 1
                or type(self.coefficient_convention) is not str
                or self.coefficient_convention != "row_kets"
                or type(self.cross_metric_convention) is not str
                or self.cross_metric_convention != "bra_previous_ket_next"
                or type(self.validation_tolerance) is not float
                or type(self.rank_tolerance) is not float
                or self.validation_tolerance != _TOLERANCE
                or self.rank_tolerance != _RANK_TOLERANCE):
            raise ValueError("unsupported AO projection evidence convention or tolerances")


@dataclass(frozen=True)
class AOProjectionDiagnostics:
    """Host double-precision validation results, stored as immutable tuples.

All frames/intervals retain their supplied order. Gram errors check *all*
retained offdiagonal and diagonal entries. Raw overlap loss is the largest
possible squared-norm loss, and does not establish physical band completeness.
"""

    metric_min_eigenvalues: tuple
    metric_condition_numbers: tuple
    metric_hermiticity_errors: tuple
    retained_gram_errors: tuple
    cross_metric_max_singular_values: tuple
    overlap_singular_values: tuple
    maximum_subspace_loss: tuple
    rank_deficient: tuple


@dataclass(frozen=True, eq=False)
class ProjectedAOElectronicPath(AdiabaticElectronicPath):
    """A native recorded path carrying its validated AO projection evidence.

Construct through project_ao_path. Replacing times, energies, overlaps, basis,
units or transport mode while retaining old evidence is rejected. The digest
is a consistency check, not an authenticity signature for caller assertions.
"""

    projection_evidence: AOProjectionEvidence = field(kw_only=True)

    def __post_init__(self):
        if not jax.config.jax_enable_x64:
            raise ValueError("AO projection requires jax_enable_x64=True; no implicit downcast")
        if not isinstance(self.projection_evidence, AOProjectionEvidence):
            raise TypeError("projection_evidence must be AOProjectionEvidence")
        _text(self.basis_id, "basis_id")
        super().__post_init__()
        if self.overlaps is None or not isinstance(self.unit_system, UnitSystem):
            raise ValueError("a projected AO path requires overlaps and an explicit UnitSystem")
        actual = _path_digest(self.times, self.energies, self.overlaps, self.basis_id,
                              self.transport_mode, self.unit_system)
        if actual != self.projection_evidence.projected_path_sha256:
            raise ValueError("projected path differs from its bound AO projection evidence")


@dataclass(frozen=True)
class AOPathProjection:
    path: ProjectedAOElectronicPath
    diagnostics: AOProjectionDiagnostics

    @property
    def evidence(self):
        return self.path.projection_evidence


def _numeric(value, name, *, real=False):
    a = np.asarray(value)
    if a.dtype.kind not in "iufc" or (real and np.iscomplexobj(a)):
        raise ValueError(f"{name} must contain {'real ' if real else ''}numerical values")
    if not np.isfinite(a).all():
        raise ValueError(f"{name} must be finite")
    # Snapshot both numerical provenance and subsequent calculations together.
    return np.array(a, copy=True)


def _finite(value, description):
    if not np.isfinite(value).all():
        raise ValueError(f"nonfinite {description}; input conditioning/scale exceeds this profile")
    return value


def project_ao_path(times, energies, coefficients, metrics, cross_metrics, *, retained_bands,
                    energy_unit, time_unit, ao_basis_id, basis_id, source_identity,
                    transport_mode="raw"):
    """Validate and project numerical AO frames into an atomic-unit native path.

Input shapes are times(F), energies(F,B), row-ket coefficients(F,B,A),
instantaneous metrics(F,A,A), and forward cross_metrics(F-1,A,A). The AO
count and number of retained states are fixed. retained_bands is an explicit
ordered, zero-based (K,) or (F,K) integer array; E and C are selected together.
No energy sorting, state matching, phase fixing, normalization, metric floors,
or contraction repair is performed. Supplied E/C eigenpair association and AO
ordering are external assertions: without H or AO labels they cannot be tested.

energy_unit is 'eV' or 'hartree'; time_unit is 'fs' or 'atomic'. Times and
energies are converted to atomic units using the recorded conversion factors.
JAX x64 is required. Single-precision exports are checked at the same fixed
1e-10 tolerance; casting them to double does not recover lost accuracy.

The full AO cross Gram is checked using singular values of L0^-1 S01 L1^-H,
where S=L L^H. This catches inconsistent discarded AO directions as well as
retained-state expansion. Valid raw overlaps preserve actual projection loss;
explicit polar transport retains that raw-loss diagnostic and requires full
rank. Neither mode supplies spatial NACs, forces, or a continuous-time basis.
This is a host ingestion helper, not a JIT/differentiable transformation.
"""
    if not jax.config.jax_enable_x64:
        raise ValueError("AO projection requires jax_enable_x64=True; no implicit downcast")
    for name, value in (("ao_basis_id", ao_basis_id), ("basis_id", basis_id),
                        ("source_identity", source_identity)):
        _text(value, name)
    if (not isinstance(energy_unit, str) or not isinstance(time_unit, str)
            or energy_unit not in {"eV", "hartree"} or time_unit not in {"fs", "atomic"}):
        raise ValueError("declare energy_unit='eV'/'hartree' and time_unit='fs'/'atomic'")
    if transport_mode not in {"raw", "polar"}:
        raise ValueError("transport_mode must be 'raw' or 'polar'")
    originals = {"times": _numeric(times, "times", real=True),
                 "energies": _numeric(energies, "energies", real=True),
                 "coefficients": _numeric(coefficients, "coefficients"),
                 "metrics": _numeric(metrics, "metrics"),
                 "cross_metrics": _numeric(cross_metrics, "cross_metrics")}
    bands_input = np.asarray(retained_bands)
    if bands_input.dtype.kind not in "iu":
        raise ValueError("retained_bands must be explicitly ordered integer indices")
    originals["retained_bands"] = np.array(bands_input, copy=True)
    t, e, c, s, cross = (originals[name] for name in
                         ("times", "energies", "coefficients", "metrics", "cross_metrics"))
    if t.ndim != 1 or len(t) < 2:
        raise ValueError("times must have shape (frames,) with at least two frames")
    frames = len(t)
    if e.ndim != 2 or e.shape[0] != frames or not e.shape[1]:
        raise ValueError("energies must have shape (frames, bands)")
    if c.ndim != 3 or c.shape[:2] != e.shape or not c.shape[2]:
        raise ValueError("row-ket coefficients must have shape (frames, bands, AOs)")
    aos = c.shape[2]
    if s.shape != (frames, aos, aos):
        raise ValueError("instantaneous metrics must have shape (frames, AOs, AOs)")
    if cross.shape != (frames-1, aos, aos):
        raise ValueError("cross_metrics must have shape (intervals, AOs, AOs)")
    if bands_input.ndim == 1:
        bands = np.broadcast_to(bands_input, (frames, len(bands_input)))
    elif bands_input.ndim == 2 and bands_input.shape[0] == frames:
        bands = bands_input
    else:
        raise ValueError("retained_bands must have shape (retained,) or (frames, retained)")
    if (not bands.shape[1] or bands.shape[1] > aos or np.any(bands >= e.shape[1])
            or np.any(bands < 0) or any(len(np.unique(row)) != len(row) for row in bands)):
        raise ValueError("retained_bands must be nonempty, unique per frame, and in range")
    bands = np.asarray(bands, dtype=np.int64)
    energy_factor = 1.0 / HARTREE_EV if energy_unit == "eV" else 1.0
    time_factor = 1.0 / ATOMIC_TIME_FS if time_unit == "fs" else 1.0
    input_evidence = tuple(_array_evidence(name, value) for name, value in originals.items())

    with np.errstate(over="ignore", invalid="ignore", divide="ignore", under="ignore"):
        t = _finite(np.asarray(t, dtype=np.float64)*time_factor, "converted times")
        if not np.isfinite(np.diff(t)).all() or np.any(np.diff(t) <= 0):
            raise ValueError("converted times must be finite and strictly increasing")
        e = _finite(np.asarray(e, dtype=np.float64)*energy_factor, "converted energies")
        c = _finite(np.asarray(c, dtype=np.complex128), "double-precision coefficients")
        s = _finite(np.asarray(s, dtype=np.complex128), "double-precision metrics")
        cross = _finite(np.asarray(cross, dtype=np.complex128), "double-precision cross metrics")
        selected_c = np.take_along_axis(c, bands[:, :, None], axis=1)
        selected_e = np.take_along_axis(e, bands, axis=1)
        cholesky, min_eigen, condition, hermitian, gram_errors = [], [], [], [], []
        for k in range(frames):
            scale = max(float(np.max(np.abs(s[k].real))), float(np.max(np.abs(s[k].imag))))
            if scale == 0:
                raise ValueError(f"metric at frame {k} is not positive definite")
            scaled = s[k]/scale
            defect = float(np.max(np.abs(scaled-scaled.conj().T)))
            if defect > _TOLERANCE:
                raise ValueError(f"metric at frame {k} is not Hermitian (scaled defect {defect:.3g})")
            eigenvalues = _finite(np.linalg.eigvalsh(scaled), f"metric spectrum at frame {k}")
            if eigenvalues[0] <= 0:
                raise ValueError(f"metric at frame {k} is not positive definite")
            smallest = float(eigenvalues[0]*scale)
            cond = float(eigenvalues[-1]/eigenvalues[0])
            if smallest <= 0 or not np.isfinite([smallest, cond]).all():
                raise ValueError(f"metric at frame {k} is too ill-conditioned for double precision")
            try:
                factor = np.linalg.cholesky(scaled)*np.sqrt(scale)
            except np.linalg.LinAlgError as exc:
                raise ValueError(f"metric at frame {k} is not positive definite") from exc
            cholesky.append(_finite(factor, f"Cholesky metric at frame {k}"))
            gram = _finite(selected_c[k].conj() @ s[k] @ selected_c[k].T,
                           f"retained Gram matrix at frame {k}")
            gram_defect = float(np.max(np.abs(gram-np.eye(bands.shape[1]))))
            if gram_defect > _TOLERANCE:
                raise ValueError(f"retained metric orthonormality fails at frame {k} "
                                 f"(Gram defect {gram_defect:.3g}; input dtype "
                                 f"{originals['coefficients'].dtype}); no normalization is applied")
            min_eigen.append(smallest)
            condition.append(cond)
            hermitian.append(defect)
            gram_errors.append(gram_defect)

        overlaps, cross_max, singular_values, losses, deficient = [], [], [], [], []
        for k in range(frames-1):
            left = _finite(solve_triangular(cholesky[k], cross[k], lower=True),
                           f"left-whitened cross metric at interval {k}")
            whitened = _finite(solve_triangular(cholesky[k+1], left.conj().T,
                                                lower=True).conj().T,
                               f"whitened cross metric at interval {k}")
            singular = _finite(np.linalg.svd(whitened, compute_uv=False),
                               f"cross-metric singular spectrum at interval {k}")
            if singular[0] > 1 + _TOLERANCE:
                raise ValueError(f"full AO cross metric is not a contraction at interval {k} "
                                 f"(whitened maximum singular value {singular[0]:.16g})")
            overlap = _finite(selected_c[k].conj() @ cross[k] @ selected_c[k+1].T,
                              f"projected overlap at interval {k}")
            spectrum = _finite(np.linalg.svd(overlap, compute_uv=False),
                               f"retained overlap spectrum at interval {k}")
            if spectrum[0] > 1 + _TOLERANCE:
                raise ValueError(f"retained overlap is not a contraction at interval {k}; "
                                 "no clipping or repair is applied")
            rank_deficient = bool(spectrum[-1] <= _RANK_TOLERANCE)
            if transport_mode == "polar" and rank_deficient:
                raise ValueError(f"polar transport requires full-rank overlap at interval {k}")
            overlaps.append(overlap)
            cross_max.append(float(singular[0]))
            singular_values.append(tuple(float(x) for x in spectrum))
            losses.append(float(max(0.0, 1.0-spectrum[-1]**2)))
            deficient.append(rank_deficient)

    overlaps = np.asarray(overlaps, dtype=np.complex128)
    units = UnitSystem()
    evidence = AOProjectionEvidence(
        source_identity, ao_basis_id, energy_unit, time_unit, float(energy_factor), float(time_factor),
        tuple(tuple(int(x) for x in row) for row in bands), input_evidence,
        _path_digest(t, selected_e, overlaps, basis_id, transport_mode, units))
    path = ProjectedAOElectronicPath(t, selected_e, overlaps, basis_id, transport_mode, units,
                                    projection_evidence=evidence)
    diagnostics = AOProjectionDiagnostics(tuple(min_eigen), tuple(condition), tuple(hermitian),
                                         tuple(gram_errors), tuple(cross_max),
                                         tuple(singular_values), tuple(losses), tuple(deficient))
    return AOPathProjection(path, diagnostics)
