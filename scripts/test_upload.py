"""Test script for the Digital Twin Tumor Upload & Preprocess pipeline.

Discovers all sample medical images (DICOM and NIfTI) under
``data/sample_images/`` and runs them through the ingestion and
preprocessing pipeline, reporting success or failure for each.

Usage:
    PYTHONPATH=src python scripts/test_upload.py

Exit codes:
    0 -- all files processed successfully
    1 -- one or more failures occurred
"""
from __future__ import annotations

import logging
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
from digital_twin_tumor.ingestion.dicom_loader import (
    load_dicom_series,
    load_single_dicom,
)
from digital_twin_tumor.ingestion.nifti_loader import load_nifti
from digital_twin_tumor.preprocessing.normalize import normalize_volume
from digital_twin_tumor.domain.models import ProcessedVolume

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)-7s %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_DIR = PROJECT_ROOT / "data" / "sample_images"
DICOM_DIR = SAMPLE_DIR / "dicom"
NIFTI_DIR = SAMPLE_DIR / "nifti"

SEPARATOR = "-" * 72


# ---------------------------------------------------------------------------
# Result tracking
# ---------------------------------------------------------------------------

@dataclass
class TestResult:
    """Outcome of loading and preprocessing a single file or directory."""

    name: str
    source_type: str  # "dicom_series", "dicom_single", "nifti"
    path: str
    success: bool = False
    shape: tuple[int, ...] | None = None
    spacing: tuple[float, float, float] | None = None
    modality: str = ""
    metadata_keys: list[str] | None = None
    normalized: bool = False
    error: str = ""


# ---------------------------------------------------------------------------
# Test runners
# ---------------------------------------------------------------------------

def _format_spacing(vol: ProcessedVolume) -> tuple[float, float, float]:
    """Extract spacing as a plain tuple."""
    return (vol.spacing.x, vol.spacing.y, vol.spacing.z)


def test_dicom_series(directory: Path) -> TestResult:
    """Test loading an entire DICOM series from a directory."""
    name = directory.name
    result = TestResult(
        name=name,
        source_type="dicom_series",
        path=str(directory),
    )
    try:
        vol = load_dicom_series(directory)
        result.success = True
        result.shape = tuple(vol.pixel_data.shape)
        result.spacing = _format_spacing(vol)
        result.modality = vol.modality
        result.metadata_keys = sorted(vol.metadata.keys())

        # Test normalization
        try:
            norm_vol = normalize_volume(vol)
            result.normalized = True
            _validate_normalized(norm_vol)
        except Exception as norm_exc:
            result.normalized = False
            result.error = f"Normalization failed: {norm_exc}"

    except Exception as exc:
        result.success = False
        result.error = f"{type(exc).__name__}: {exc}"

    return result


def test_dicom_single(filepath: Path) -> TestResult:
    """Test loading a single DICOM file."""
    name = filepath.name
    result = TestResult(
        name=name,
        source_type="dicom_single",
        path=str(filepath),
    )
    try:
        vol = load_single_dicom(filepath)
        result.success = True
        result.shape = tuple(vol.pixel_data.shape)
        result.spacing = _format_spacing(vol)
        result.modality = vol.modality
        result.metadata_keys = sorted(vol.metadata.keys())

        # Test normalization
        try:
            norm_vol = normalize_volume(vol)
            result.normalized = True
            _validate_normalized(norm_vol)
        except Exception as norm_exc:
            result.normalized = False
            result.error = f"Normalization failed: {norm_exc}"

    except Exception as exc:
        result.success = False
        result.error = f"{type(exc).__name__}: {exc}"

    return result


def test_nifti(filepath: Path) -> TestResult:
    """Test loading a NIfTI file."""
    name = filepath.name
    result = TestResult(
        name=name,
        source_type="nifti",
        path=str(filepath),
    )
    try:
        vol = load_nifti(filepath)
        result.success = True
        result.shape = tuple(vol.pixel_data.shape)
        result.spacing = _format_spacing(vol)
        result.modality = vol.modality
        result.metadata_keys = sorted(vol.metadata.keys())

        # Test normalization -- determine appropriate modality
        try:
            norm_vol = normalize_volume(vol)
            result.normalized = True
            _validate_normalized(norm_vol)
        except ValueError as ve:
            # May fail if modality not CT/MR -- try forcing CT
            if "Unsupported modality" in str(ve):
                vol_ct = ProcessedVolume(
                    pixel_data=vol.pixel_data,
                    spacing=vol.spacing,
                    origin=vol.origin,
                    direction=vol.direction,
                    modality="CT",
                    metadata=vol.metadata,
                )
                norm_vol = normalize_volume(vol_ct)
                result.normalized = True
                result.error = "Normalization required modality override to CT"
                _validate_normalized(norm_vol)
            else:
                raise
        except Exception as norm_exc:
            result.normalized = False
            result.error = f"Normalization failed: {norm_exc}"

    except Exception as exc:
        result.success = False
        result.error = f"{type(exc).__name__}: {exc}"

    return result


def _validate_normalized(vol: ProcessedVolume) -> None:
    """Sanity-check that normalized volume has values in [0, 1]."""
    data = vol.pixel_data
    if data.size == 0:
        return
    vmin, vmax = float(np.min(data)), float(np.max(data))
    if vmin < -0.01 or vmax > 1.01:
        logger.warning(
            "Normalized volume outside [0,1]: min=%.4f, max=%.4f",
            vmin,
            vmax,
        )


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def discover_dicom_series(base: Path) -> list[Path]:
    """Find directories that contain .dcm files (series directories)."""
    series_dirs: list[Path] = []
    if not base.exists():
        return series_dirs
    for d in sorted(base.iterdir()):
        if d.is_dir():
            dcm_files = list(d.glob("*.dcm")) + list(d.glob("*.DCM"))
            if dcm_files:
                series_dirs.append(d)
            # Also recurse one level
            for sub in sorted(d.iterdir()):
                if sub.is_dir():
                    sub_dcm = list(sub.glob("*.dcm")) + list(sub.glob("*.DCM"))
                    if sub_dcm:
                        series_dirs.append(sub)
    return series_dirs


def discover_single_dicoms(base: Path) -> list[Path]:
    """Find individual .dcm files in the single_files directory."""
    single_dir = base / "single_files"
    if not single_dir.exists():
        return []
    return sorted(single_dir.glob("*.dcm"))


def discover_nifti_files(base: Path) -> list[Path]:
    """Find all NIfTI files (.nii, .nii.gz) excluding masks."""
    if not base.exists():
        return []
    files = sorted(base.glob("*.nii")) + sorted(base.glob("*.nii.gz"))
    # Include masks too -- they should still load correctly
    return files


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def print_result(result: TestResult) -> None:
    """Print a formatted test result."""
    status = "PASS" if result.success else "FAIL"
    icon = "[+]" if result.success else "[-]"

    print(f"  {icon} {result.name} ({result.source_type})")
    print(f"      Status:     {status}")
    print(f"      Path:       {result.path}")

    if result.success:
        print(f"      Shape:      {result.shape}")
        print(f"      Spacing:    ({result.spacing[0]:.3f}, {result.spacing[1]:.3f}, {result.spacing[2]:.3f}) mm")
        print(f"      Modality:   {result.modality}")
        print(f"      Normalized: {'Yes' if result.normalized else 'No'}")
        if result.metadata_keys:
            print(f"      Metadata:   {len(result.metadata_keys)} keys")
    if result.error:
        print(f"      Error:      {result.error}")
    print()


def print_summary(results: list[TestResult]) -> None:
    """Print an overall summary."""
    total = len(results)
    passed = sum(1 for r in results if r.success)
    failed = total - passed
    normalized = sum(1 for r in results if r.normalized)

    print(SEPARATOR)
    print("SUMMARY")
    print(SEPARATOR)
    print(f"  Total files tested:   {total}")
    print(f"  Passed:               {passed}")
    print(f"  Failed:               {failed}")
    print(f"  Normalized:           {normalized}")
    print()

    if failed > 0:
        print("  FAILED ITEMS:")
        for r in results:
            if not r.success:
                print(f"    - {r.name}: {r.error}")
        print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    """Run all upload/preprocessing tests and return exit code."""
    print(SEPARATOR)
    print("Digital Twin Tumor -- Upload & Preprocess Pipeline Test")
    print(SEPARATOR)
    print()

    results: list[TestResult] = []

    # 1. DICOM series
    series_dirs = discover_dicom_series(DICOM_DIR)
    if series_dirs:
        print(f"Found {len(series_dirs)} DICOM series directories:")
        for d in series_dirs:
            print(f"  - {d.relative_to(SAMPLE_DIR)}")
        print()

        print("Testing DICOM series loading...")
        print(SEPARATOR)
        for d in series_dirs:
            result = test_dicom_series(d)
            results.append(result)
            print_result(result)
    else:
        print("No DICOM series directories found.\n")

    # 2. Single DICOM files
    single_dicoms = discover_single_dicoms(DICOM_DIR)
    if single_dicoms:
        print(f"Found {len(single_dicoms)} single DICOM files:")
        for f in single_dicoms:
            print(f"  - {f.name}")
        print()

        print("Testing single DICOM loading...")
        print(SEPARATOR)
        for f in single_dicoms:
            result = test_dicom_single(f)
            results.append(result)
            print_result(result)
    else:
        print("No single DICOM files found.\n")

    # 3. NIfTI files
    nifti_files = discover_nifti_files(NIFTI_DIR)
    if nifti_files:
        print(f"Found {len(nifti_files)} NIfTI files:")
        for f in nifti_files:
            print(f"  - {f.name}")
        print()

        print("Testing NIfTI loading...")
        print(SEPARATOR)
        for f in nifti_files:
            result = test_nifti(f)
            results.append(result)
            print_result(result)
    else:
        print("No NIfTI files found.\n")

    # Summary
    print_summary(results)

    # Return code
    failures = sum(1 for r in results if not r.success)
    return 1 if failures > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
