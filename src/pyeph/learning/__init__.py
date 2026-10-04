"""Host-side data and artifact utilities for explicitly reconstructed providers.

Training algorithms and model construction remain provider-owned. Importing
these helpers does not select JAX precision or load a teacher dependency.
"""

from .bundles import bundle_identity, load_bundle, revalidate_bundle, save_bundle
from .domain import DomainViolation, GeometryDomainMonitor
from .labels import grouped_split, load_labels, validate_labels
from .reports import error_metrics, validation_report

__all__ = [
    "DomainViolation", "GeometryDomainMonitor", "bundle_identity", "error_metrics",
    "grouped_split", "load_bundle", "load_labels", "revalidate_bundle", "save_bundle",
    "validate_labels", "validation_report",
]
