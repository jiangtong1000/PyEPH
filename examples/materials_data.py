"""Compatibility imports for older label-example commands.

New applications should import these host helpers from ``pyeph.learning``.
The v1 fixed-basis label schema and existing datasets remain unchanged.
"""

from pyeph.learning.labels import SCHEMA, grouped_split, load_labels, validate_labels

__all__ = ["SCHEMA", "grouped_split", "load_labels", "validate_labels"]
