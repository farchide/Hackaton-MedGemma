"""Isotropic resampling for 3-D medical imaging volumes.

Implements the resampling strategy described in ADR-007:

- **Image data** is resampled with B-spline interpolation (order 3) to
  preserve intensity fidelity.
- **Masks** use nearest-neighbour interpolation (order 0) to preserve
  label integrity.
- Spacing and origin metadata are updated to reflect the new geometry.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.ndimage import zoom

from digital_twin_tumor.domain.models import ProcessedVolume, VoxelSpacing


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------


def _compute_zoom_factors(
    original_spacing: Tuple[float, float, float],
    target_spacing: Tuple[float, float, float],
) -> Tuple[float, float, float]:
    """Return per-axis zoom factors to resample from *original* to *target*.

    A zoom factor > 1 means the output will have **more** voxels along that
    axis (finer spacing); < 1 means fewer voxels (coarser spacing).

    Parameters
    ----------
    original_spacing:
        Current (z, y, x) voxel spacing in mm.
    target_spacing:
        Desired (z, y, x) voxel spacing in mm.

    Returns
    -------
    tuple[float, float, float]
        Per-axis zoom factors.
    """
    factors: list[float] = []
    for orig, tgt in zip(original_spacing, target_spacing):
        if tgt <= 0:
            raise ValueError(f"Target spacing must be positive, got {tgt}.")
        factors.append(orig / tgt)
    return (factors[0], factors[1], factors[2])


def _adjust_origin(
    origin: Tuple[float, float, float],
    original_spacing: Tuple[float, float, float],
    target_spacing: Tuple[float, float, float],
) -> Tuple[float, float, float]:
    """Adjust the volume origin so that the physical extent remains centred.

    When voxel spacing changes, the centre of the first voxel shifts by
    half the difference between old and new spacing.

    Parameters
    ----------
    origin:
        Original (x, y, z) world-coordinate origin.
    original_spacing:
        Current voxel spacing in mm.
    target_spacing:
        New voxel spacing in mm.

    Returns
    -------
    tuple[float, float, float]
        Adjusted origin.
    """
    return (
        origin[0] + 0.5 * (target_spacing[0] - original_spacing[0]),
        origin[1] + 0.5 * (target_spacing[1] - original_spacing[1]),
        origin[2] + 0.5 * (target_spacing[2] - original_spacing[2]),
    )


# ---------------------------------------------------------------------------
# Mask resampling (standalone utility)
# ---------------------------------------------------------------------------


def resample_mask(
    mask: np.ndarray,
    original_spacing: VoxelSpacing,
    target_spacing: tuple[float, float, float],
) -> np.ndarray:
    """Resample a binary mask to new voxel spacing using nearest-neighbour.

    Parameters
    ----------
    mask:
        3-D binary (or integer label) array.
    original_spacing:
        Current voxel spacing.
    target_spacing:
        Desired ``(z, y, x)`` voxel spacing in mm.

    Returns
    -------
    np.ndarray
        Resampled mask with the same dtype as the input.
    """
    if mask.ndim != 3:
        raise ValueError(f"Expected a 3-D mask, got {mask.ndim}-D array.")

    orig_tuple = (original_spacing.z, original_spacing.y, original_spacing.x)
    factors = _compute_zoom_factors(orig_tuple, target_spacing)

    # Check if resampling is actually needed.
    if all(abs(f - 1.0) < 1e-6 for f in factors):
        return mask.copy()

    input_dtype = mask.dtype
    resampled = zoom(mask.astype(np.float64), factors, order=0, mode="nearest")

    # Restore original dtype (important for integer label masks).
    if np.issubdtype(input_dtype, np.integer) or input_dtype == bool:
        resampled = np.round(resampled).astype(input_dtype)

    return resampled


# ---------------------------------------------------------------------------
# Volume resampling
# ---------------------------------------------------------------------------


def resample_volume(
    volume: ProcessedVolume,
    target_spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> ProcessedVolume:
    """Resample a :class:`ProcessedVolume` to isotropic (or arbitrary) spacing.

    Image data uses B-spline interpolation (order 3) for smooth intensity
    preservation.  The associated mask (if present) is resampled with
    nearest-neighbour interpolation (order 0) to preserve label integrity.

    Parameters
    ----------
    volume:
        The processed volume to resample.
    target_spacing:
        Desired ``(z, y, x)`` voxel spacing in millimetres.  Defaults to
        1.0 mm isotropic.

    Returns
    -------
    ProcessedVolume
        A **new** volume with resampled ``pixel_data``, updated ``spacing``,
        adjusted ``origin``, and resampled ``mask`` (if one was attached).

    Raises
    ------
    ValueError
        If any target spacing component is non-positive, or the input
        volume has fewer than 3 dimensions.
    """
    if volume.pixel_data.ndim < 3:
        raise ValueError(
            f"Expected a 3-D volume, got {volume.pixel_data.ndim}-D array."
        )

    orig_spacing = (volume.spacing.z, volume.spacing.y, volume.spacing.x)
    factors = _compute_zoom_factors(orig_spacing, target_spacing)

    # Short-circuit if spacing already matches within tolerance.
    if all(abs(f - 1.0) < 1e-6 for f in factors):
        return ProcessedVolume(
            pixel_data=volume.pixel_data.copy(),
            spacing=volume.spacing,
            origin=volume.origin,
            direction=volume.direction,
            modality=volume.modality,
            metadata=dict(volume.metadata),
            mask=volume.mask.copy() if volume.mask is not None else None,
            selected_slices=list(volume.selected_slices),
        )

    # Resample image data with B-spline (order=3).
    resampled_data = zoom(
        volume.pixel_data.astype(np.float32),
        factors,
        order=3,
        mode="nearest",
    ).astype(np.float32)

    # Resample mask with nearest-neighbour (order=0) if present.
    resampled_mask: np.ndarray | None = None
    if volume.mask is not None:
        resampled_mask = resample_mask(
            volume.mask,
            volume.spacing,
            target_spacing,
        )

    # Build updated spacing.
    new_spacing = VoxelSpacing(
        x=target_spacing[2],
        y=target_spacing[1],
        z=target_spacing[0],
    )

    # Adjust origin for voxel-centre shift.
    new_origin = _adjust_origin(volume.origin, orig_spacing, target_spacing)

    return ProcessedVolume(
        pixel_data=resampled_data,
        spacing=new_spacing,
        origin=new_origin,
        direction=volume.direction,
        modality=volume.modality,
        metadata=dict(volume.metadata),
        mask=resampled_mask,
        selected_slices=[],  # Slices indices are invalidated after resampling.
    )
