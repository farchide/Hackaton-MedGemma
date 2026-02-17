"""DICOM series and single-file loader.

Uses ``pydicom`` for metadata extraction and ``SimpleITK`` for robust
series reading (handles sorting, multi-frame, and orientation).  All
volumes are reoriented to RAS+ before being returned as a
:class:`ProcessedVolume`.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from digital_twin_tumor.domain.models import ProcessedVolume, VoxelSpacing

# ---------------------------------------------------------------------------
# Optional dependency imports -- fail gracefully with clear messages.
# ---------------------------------------------------------------------------
try:
    import pydicom
    from pydicom.dataset import Dataset
except ImportError as _exc:  # pragma: no cover
    raise ImportError(
        "pydicom is required for DICOM loading. Install it with: "
        "pip install 'pydicom>=2.4'"
    ) from _exc

try:
    import SimpleITK as sitk
except ImportError as _exc:  # pragma: no cover
    raise ImportError(
        "SimpleITK is required for DICOM series loading. Install it with: "
        "pip install 'SimpleITK>=2.3'"
    ) from _exc

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# DICOM tags we attempt to extract (tag keyword -> default value)
# ---------------------------------------------------------------------------
_DICOM_TAG_DEFAULTS: dict[str, Any] = {
    "PatientID": "",
    "StudyDate": "",
    "SeriesDate": "",
    "Modality": "CT",
    "Manufacturer": "",
    "SliceThickness": 0.0,
    "PixelSpacing": [1.0, 1.0],
    "SpacingBetweenSlices": 0.0,
    "ImageOrientationPatient": [1, 0, 0, 0, 1, 0],
    "StudyDescription": "",
    "SeriesDescription": "",
    "SeriesInstanceUID": "",
    "StudyInstanceUID": "",
    "Rows": 0,
    "Columns": 0,
    "BitsAllocated": 16,
    "WindowCenter": None,
    "WindowWidth": None,
    "RescaleIntercept": 0.0,
    "RescaleSlope": 1.0,
    "BodyPartExamined": "",
    "SequenceName": "",
    "ScanningSequence": "",
    "MagneticFieldStrength": None,
    "RepetitionTime": None,
    "EchoTime": None,
    "ProtocolName": "",
    "AccessionNumber": "",
    "InstitutionName": "",
    "PatientName": "",
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_dicom_series(directory: str | Path) -> ProcessedVolume:
    """Read an entire DICOM series from *directory* and return a ProcessedVolume.

    The series is read via SimpleITK's ``ImageSeriesReader`` which handles
    file sorting, multi-frame DICOM, and vendor-specific quirks.  The
    resulting image is reoriented to RAS+ using ``DICOMOrient``.

    Parameters
    ----------
    directory:
        Path to a directory containing the DICOM files for **one** series.

    Returns
    -------
    ProcessedVolume
        Volume with ``pixel_data`` in RAS+ orientation and extracted metadata.

    Raises
    ------
    FileNotFoundError
        If *directory* does not exist.
    ValueError
        If no DICOM files are found in *directory*.
    RuntimeError
        If SimpleITK fails to read the series.
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"DICOM directory not found: {directory}")

    # Discover series IDs in the directory
    reader = sitk.ImageSeriesReader()
    series_ids = reader.GetGDCMSeriesIDs(str(directory))
    if not series_ids:
        raise ValueError(f"No DICOM series found in {directory}")

    # Use the first series if multiple are present
    if len(series_ids) > 1:
        logger.warning(
            "Multiple DICOM series found in %s (%d series). Using the first one.",
            directory,
            len(series_ids),
        )
    series_id = series_ids[0]
    dicom_filenames = reader.GetGDCMSeriesFileNames(str(directory), series_id)

    if not dicom_filenames:
        raise ValueError(f"No DICOM files for series {series_id} in {directory}")

    # Read the series into a SimpleITK image
    reader.SetFileNames(dicom_filenames)
    reader.MetaDataDictionaryArrayUpdateOn()
    reader.LoadPrivateTagsOn()

    try:
        sitk_image = reader.Execute()
    except RuntimeError as exc:
        raise RuntimeError(
            f"SimpleITK failed to read DICOM series from {directory}: {exc}"
        ) from exc

    # Reorient to RAS+
    sitk_image = sitk.DICOMOrient(sitk_image, "RAS")

    # Extract metadata from the first file in the series
    first_file = Path(dicom_filenames[0])
    try:
        ds = pydicom.dcmread(str(first_file), stop_before_pixels=True)
    except Exception:
        logger.warning("Could not read DICOM header from %s for metadata", first_file)
        ds = Dataset()

    metadata = extract_metadata(ds)
    metadata["source_directory"] = str(directory)
    metadata["series_id"] = series_id
    metadata["num_files"] = len(dicom_filenames)

    return _sitk_image_to_volume(sitk_image, metadata)


def load_single_dicom(path: str | Path) -> ProcessedVolume:
    """Load a single DICOM file and return it as a 2-D ProcessedVolume.

    Parameters
    ----------
    path:
        Path to a single ``.dcm`` file.

    Returns
    -------
    ProcessedVolume
        Volume with ``pixel_data`` of shape ``(1, H, W)`` and extracted
        metadata.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    RuntimeError
        If the file cannot be read.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"DICOM file not found: {path}")

    try:
        ds = pydicom.dcmread(str(path))
    except Exception as exc:
        raise RuntimeError(f"Failed to read DICOM file {path}: {exc}") from exc

    metadata = extract_metadata(ds)
    metadata["source_file"] = str(path)

    # Get pixel data as numpy array
    if not hasattr(ds, "pixel_array"):
        raise ValueError(f"DICOM file {path} does not contain pixel data")

    pixel_array = ds.pixel_array.astype(np.float32)

    # Apply rescale slope / intercept if available
    slope = float(getattr(ds, "RescaleSlope", 1.0) or 1.0)
    intercept = float(getattr(ds, "RescaleIntercept", 0.0) or 0.0)
    pixel_array = pixel_array * slope + intercept

    # Ensure 3-D shape (D, H, W) -- single slice means D=1
    if pixel_array.ndim == 2:
        pixel_array = pixel_array[np.newaxis, :, :]
    elif pixel_array.ndim == 3:
        # Multi-frame DICOM: already (frames, H, W)
        pass
    else:
        logger.warning(
            "Unexpected pixel array shape %s in %s; attempting reshape",
            pixel_array.shape,
            path,
        )

    # Build spacing
    pixel_spacing = getattr(ds, "PixelSpacing", [1.0, 1.0])
    if pixel_spacing is None:
        pixel_spacing = [1.0, 1.0]
    row_spacing = float(pixel_spacing[0])
    col_spacing = float(pixel_spacing[1])
    slice_thickness = float(getattr(ds, "SliceThickness", 1.0) or 1.0)

    spacing = VoxelSpacing(x=col_spacing, y=row_spacing, z=slice_thickness)

    return ProcessedVolume(
        pixel_data=pixel_array,
        spacing=spacing,
        origin=(0.0, 0.0, 0.0),
        direction=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
        modality=metadata.get("Modality", "CT"),
        metadata=metadata,
    )


def extract_metadata(ds: Dataset) -> dict[str, Any]:
    """Extract relevant DICOM tags from *ds*, using safe defaults for missing tags.

    Parameters
    ----------
    ds:
        A ``pydicom.Dataset`` (may be partially populated).

    Returns
    -------
    dict[str, Any]
        Dictionary of tag keyword -> extracted value.
    """
    result: dict[str, Any] = {}

    for tag_name, default in _DICOM_TAG_DEFAULTS.items():
        value = getattr(ds, tag_name, None)
        if value is None:
            result[tag_name] = default
            continue

        # Convert pydicom types to plain Python types for serialisation
        try:
            if tag_name == "PixelSpacing":
                result[tag_name] = [float(v) for v in value]
            elif tag_name == "ImageOrientationPatient":
                result[tag_name] = [float(v) for v in value]
            elif tag_name in ("PatientName",):
                result[tag_name] = str(value)
            elif isinstance(value, (int, float, str)):
                result[tag_name] = value
            elif hasattr(value, "original_string"):
                # DSfloat, IS, etc.
                result[tag_name] = str(value)
            else:
                result[tag_name] = str(value)
        except (TypeError, ValueError):
            logger.warning(
                "Failed to convert DICOM tag %s (value=%r); using default",
                tag_name,
                value,
            )
            result[tag_name] = default

    return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _sitk_image_to_volume(
    image: sitk.Image,
    metadata: dict[str, Any],
) -> ProcessedVolume:
    """Convert a SimpleITK Image to a :class:`ProcessedVolume`.

    Parameters
    ----------
    image:
        A 3-D SimpleITK image (already in RAS+ orientation).
    metadata:
        Pre-extracted DICOM metadata dictionary.

    Returns
    -------
    ProcessedVolume
    """
    # Convert to numpy -- SimpleITK uses (x, y, z) ordering; GetArrayFromImage
    # returns (z, y, x) which maps to our (D, H, W) convention.
    pixel_data = sitk.GetArrayFromImage(image).astype(np.float32)

    sitk_spacing = image.GetSpacing()  # (x, y, z)
    spacing = VoxelSpacing(
        x=float(sitk_spacing[0]),
        y=float(sitk_spacing[1]),
        z=float(sitk_spacing[2]),
    )

    origin = tuple(float(v) for v in image.GetOrigin())
    direction = tuple(float(v) for v in image.GetDirection())

    modality = metadata.get("Modality", "CT")

    # Record the original image dimensions in metadata
    metadata["original_shape"] = list(pixel_data.shape)
    metadata["orientation"] = "RAS"

    return ProcessedVolume(
        pixel_data=pixel_data,
        spacing=spacing,
        origin=origin,  # type: ignore[arg-type]
        direction=direction,
        modality=modality,
        metadata=metadata,
    )
