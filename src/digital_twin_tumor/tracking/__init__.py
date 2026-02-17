"""Lesion tracking module for the Digital Twin Tumor system.

Matches lesions across time-points and builds longitudinal identity graphs.

Exports the core tracking components:
  - :class:`LesionGraph` -- DAG-based identity graph (ADR-010)
  - :class:`LesionMatcher` -- Three-stage lesion matching pipeline
  - :func:`register_volumes` -- Rigid image registration
"""

from __future__ import annotations

from digital_twin_tumor.tracking.identity_graph import LesionGraph
from digital_twin_tumor.tracking.matching import LesionMatcher
from digital_twin_tumor.tracking.registration import register_volumes

__all__ = [
    "LesionGraph",
    "LesionMatcher",
    "register_volumes",
]
