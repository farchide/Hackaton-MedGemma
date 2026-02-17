"""Ingestion layer -- DICOM and NIfTI loaders with PHI de-identification.

Public API::

    from digital_twin_tumor.ingestion import load_dicom_series, load_nifti, PHIGate

Individual sub-modules can also be imported directly when only one loader
is needed (and its dependencies are available)::

    from digital_twin_tumor.ingestion.nifti_loader import load_nifti

Lazy imports are used so that missing optional dependencies (pydicom,
SimpleITK, nibabel) only raise when the corresponding function is
actually called, not at package import time.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from digital_twin_tumor.ingestion.dicom_loader import load_dicom_series as load_dicom_series
    from digital_twin_tumor.ingestion.nifti_loader import load_nifti as load_nifti
    from digital_twin_tumor.ingestion.phi_gate import PHIGate as PHIGate

__all__ = [
    "load_dicom_series",
    "load_nifti",
    "PHIGate",
]


def __getattr__(name: str) -> object:
    """Lazy-load public symbols on first access."""
    if name == "load_dicom_series":
        from digital_twin_tumor.ingestion.dicom_loader import load_dicom_series
        return load_dicom_series
    if name == "load_nifti":
        from digital_twin_tumor.ingestion.nifti_loader import load_nifti
        return load_nifti
    if name == "PHIGate":
        from digital_twin_tumor.ingestion.phi_gate import PHIGate
        return PHIGate
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
