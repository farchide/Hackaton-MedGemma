"""NIfTI file loader (.nii and .nii.gz).

Uses ``nibabel`` for I/O and ``nibabel.as_closest_canonical()`` for
reorientation to RAS+.  Handles both 3-D and 4-D NIfTI files (for 4-D
the first volume/time-point is extracted).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from digital_twin_tumor.domain.models import ProcessedVolume, VoxelSpacing

# ---------------------------------------------------------------------------
# Optional dependency import
# ---------------------------------------------------------------------------
try:
    import nibabel as nib
    from nibabel.orientations import aff2axcodes, axcodes2ornt, ornt_transform
except ImportError as _exc:  # pragma: no cover
    raise ImportError(
        "nibabel is required for NIfTI loading. Install it with: "
        "pip install 'nibabel>=5.0'"
    ) from _exc

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_nifti(path: str | Path) -> ProcessedVolume:
    """Load a NIfTI file and return a RAS+-oriented ProcessedVolume.

    Supports ``.nii`` and ``.nii.gz`` compressed files.  For 4-D NIfTI
    volumes (e.g. time-series or multi-echo) the **first volume** along
    the fourth axis is used.

    Parameters
    ----------
    path:
        Path to a NIfTI file.

    Returns
    -------
    ProcessedVolume
        Volume with ``pixel_data`` in RAS+ orientation and metadata
        extracted from the NIfTI header.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    ValueError
        If the file cannot be parsed as NIfTI.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"NIfTI file not found: {path}")

    try:
        img = nib.load(str(path))
    except Exception as exc:
        raise ValueError(f"Failed to load NIfTI file {path}: {exc}") from exc

    # Validate that this is a Nifti1Image (or Nifti2Image, both work)
    if not isinstance(img, (nib.Nifti1Image, nib.Nifti2Image)):
        logger.warning(
            "Loaded image type is %s; expected Nifti1Image or Nifti2Image.",
            type(img).__name__,
        )

    # Reorient to RAS+ using nibabel's canonical form
    original_orientation = "".join(aff2axcodes(img.affine))
    img_canonical = nib.as_closest_canonical(img)

    # Get data as float32
    data = np.asarray(img_canonical.dataobj, dtype=np.float32)

    # Handle 4-D NIfTI: take the first volume
    if data.ndim == 4:
        logger.info(
            "4-D NIfTI detected (shape=%s); extracting first volume.",
            data.shape,
        )
        data = data[:, :, :, 0]
    elif data.ndim == 2:
        # Rare: 2-D NIfTI -- add a depth dimension
        data = data[np.newaxis, :, :]
    elif data.ndim != 3:
        raise ValueError(
            f"Unexpected NIfTI data dimensionality: {data.ndim} "
            f"(shape={data.shape}). Expected 2, 3, or 4 dimensions."
        )

    # Transpose to (D, H, W) -- nibabel loads as (W, H, D) in voxel order,
    # but after canonical reorientation the axes are (i, j, k) = (R, A, S).
    # We want depth (S=axial slices) first -> (k, j, i) = (D, H, W).
    data = np.transpose(data, (2, 1, 0))

    # Extract spacing from the affine
    affine = img_canonical.affine.copy()
    metadata = extract_nifti_metadata(img_canonical)
    metadata["source_file"] = str(path)
    metadata["original_orientation"] = original_orientation

    voxel_sizes = _voxel_sizes_from_affine(affine)
    spacing = VoxelSpacing(
        x=float(voxel_sizes[0]),
        y=float(voxel_sizes[1]),
        z=float(voxel_sizes[2]),
    )

    # Build direction cosines from the affine (3x3 rotation normalised by spacing)
    rotation = affine[:3, :3].copy()
    for i in range(3):
        norm = np.linalg.norm(rotation[:, i])
        if norm > 0:
            rotation[:, i] /= norm
    direction = tuple(float(v) for v in rotation.flatten())

    origin = tuple(float(v) for v in affine[:3, 3])

    return ProcessedVolume(
        pixel_data=data,
        spacing=spacing,
        origin=origin,  # type: ignore[arg-type]
        direction=direction,
        modality=metadata.get("modality", "MR"),
        metadata=metadata,
    )


def extract_nifti_metadata(img: nib.Nifti1Image | nib.Nifti2Image) -> dict[str, Any]:
    """Extract metadata from a NIfTI image header.

    Parameters
    ----------
    img:
        A nibabel NIfTI image (already reoriented or not).

    Returns
    -------
    dict[str, Any]
        Dictionary containing voxel dimensions, data type, orientation
        codes, and header fields.
    """
    header = img.header
    affine = img.affine

    # Orientation codes
    orientation_codes = "".join(aff2axcodes(affine))

    # Voxel sizes from the affine (more reliable than header zooms for
    # reoriented images)
    voxel_sizes = _voxel_sizes_from_affine(affine)

    # Header-level information (safely extracted)
    result: dict[str, Any] = {
        "voxel_sizes_mm": [float(v) for v in voxel_sizes],
        "orientation": orientation_codes,
        "data_dtype": str(header.get_data_dtype()),
        "data_shape": list(img.shape),
    }

    # Extract additional NIfTI-1 header fields if available
    _safe_set(result, "descrip", header, "descrip")
    _safe_set(result, "qform_code", header, "qform_code")
    _safe_set(result, "sform_code", header, "sform_code")
    _safe_set(result, "xyzt_units", header, "xyzt_units")

    # Attempt to extract intent code (used in some research NIfTIs)
    try:
        intent_code = int(header["intent_code"])
        result["intent_code"] = intent_code
    except (KeyError, TypeError, ValueError):
        pass

    # Temporal resolution for 4-D volumes
    zooms = header.get_zooms()
    if len(zooms) >= 4:
        result["temporal_resolution"] = float(zooms[3])

    return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _voxel_sizes_from_affine(affine: np.ndarray) -> np.ndarray:
    """Compute per-axis voxel sizes from a 4x4 affine matrix.

    Parameters
    ----------
    affine:
        4x4 affine transformation matrix.

    Returns
    -------
    np.ndarray
        Array of shape ``(3,)`` with voxel sizes in mm.
    """
    return np.sqrt(np.sum(affine[:3, :3] ** 2, axis=0))


def _safe_set(
    target: dict[str, Any],
    key: str,
    header: Any,
    field_name: str,
) -> None:
    """Safely extract a header field and store it in *target*.

    Silently skips if the field is missing or cannot be converted.
    """
    try:
        raw = header[field_name]
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8", errors="replace").strip("\x00 ")
        elif isinstance(raw, np.ndarray):
            raw = raw.tolist()
        target[key] = raw
    except (KeyError, TypeError, ValueError):
        pass
