"""Volume and surface-area computation for 3-D tumour masks.

Implements ADR-004: voxel-based volume computation, volume change
tracking, and surface-area estimation via marching cubes or voxel
face counting.
"""

from __future__ import annotations

import logging

import numpy as np

from digital_twin_tumor.domain.models import VoxelSpacing

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_volume(mask_3d: np.ndarray, spacing: VoxelSpacing) -> float:
    """Compute tumour volume from a 3-D binary mask.

    Parameters
    ----------
    mask_3d:
        Binary mask with shape ``(D, H, W)`` where D = number of
        slices.  Nonzero voxels are treated as foreground.
    spacing:
        Physical voxel spacing in millimetres.

    Returns
    -------
    float
        Volume in cubic millimetres (mm^3).
    """
    if mask_3d.size == 0:
        return 0.0

    voxel_count = int((mask_3d > 0).sum())
    voxel_volume_mm3 = float(spacing.x) * float(spacing.y) * float(spacing.z)
    return voxel_count * voxel_volume_mm3


def compute_volume_change(
    current_mm3: float,
    previous_mm3: float,
) -> dict[str, float | str]:
    """Compute volume change between two time-points.

    Parameters
    ----------
    current_mm3:
        Current volume in mm^3.
    previous_mm3:
        Previous (baseline or prior) volume in mm^3.

    Returns
    -------
    dict
        Keys:
        - ``absolute_change``: current - previous (mm^3).
        - ``percent_change``: relative change (%).  ``0.0`` when
          previous volume is zero.
        - ``direction``: ``"growing"``, ``"shrinking"``, or
          ``"stable"``.  Stable is defined as <5% absolute change.
    """
    absolute = current_mm3 - previous_mm3

    if previous_mm3 > 0:
        percent = (absolute / previous_mm3) * 100.0
    else:
        # Avoid division by zero.  If previous was 0 and current > 0,
        # the percent change is not meaningful.
        percent = 0.0 if current_mm3 == 0.0 else 100.0

    # Classify direction using a 5% threshold for "stable"
    if abs(percent) < 5.0:
        direction = "stable"
    elif absolute > 0:
        direction = "growing"
    else:
        direction = "shrinking"

    return {
        "absolute_change": absolute,
        "percent_change": percent,
        "direction": direction,
    }


def mask_to_surface_area(
    mask_3d: np.ndarray,
    spacing: VoxelSpacing,
) -> float:
    """Approximate the surface area of a 3-D binary mask.

    Attempts to use marching cubes (via ``skimage.measure.marching_cubes``)
    for an accurate triangulated surface.  Falls back to voxel-face
    counting when marching cubes is unavailable or the mask is too
    small.

    Parameters
    ----------
    mask_3d:
        Binary mask with shape ``(D, H, W)``.
    spacing:
        Physical voxel spacing in millimetres.

    Returns
    -------
    float
        Estimated surface area in mm^2.
    """
    if mask_3d.size == 0 or mask_3d.max() == 0:
        return 0.0

    binary = (mask_3d > 0).astype(np.uint8)
    sp = (float(spacing.z), float(spacing.y), float(spacing.x))

    # Marching cubes requires at least 2x2x2 foreground region
    if all(s >= 2 for s in binary.shape):
        try:
            return _surface_area_marching_cubes(binary, sp)
        except Exception as exc:  # noqa: BLE001
            logger.debug(
                "Marching cubes failed (%s); falling back to face counting.",
                exc,
            )

    return _surface_area_face_counting(binary, sp)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _surface_area_marching_cubes(
    binary: np.ndarray,
    spacing: tuple[float, float, float],
) -> float:
    """Surface area via skimage marching cubes + mesh_surface_area.

    Parameters
    ----------
    binary:
        3-D binary uint8 array.
    spacing:
        Physical spacing ``(z, y, x)`` in mm.

    Returns
    -------
    float
        Surface area in mm^2.
    """
    from skimage.measure import marching_cubes, mesh_surface_area

    # Pad with a layer of zeros so the surface is closed
    padded = np.pad(binary, pad_width=1, mode="constant", constant_values=0)

    verts, faces, _, _ = marching_cubes(
        padded.astype(np.float64),
        level=0.5,
        spacing=spacing,
    )
    area = mesh_surface_area(verts, faces)
    return float(area)


def _surface_area_face_counting(
    binary: np.ndarray,
    spacing: tuple[float, float, float],
) -> float:
    """Approximate surface area by counting exposed voxel faces.

    Each voxel is an axis-aligned box with face areas determined by the
    physical spacing.  A face is "exposed" (contributing to the surface)
    when its neighbour in that direction is background (or out-of-bounds).

    Parameters
    ----------
    binary:
        3-D binary uint8 array with shape ``(D, H, W)``.
    spacing:
        Physical spacing ``(z, y, x)`` in mm.

    Returns
    -------
    float
        Surface area in mm^2.
    """
    dz, dy, dx = spacing

    # Face areas for each axis direction
    # z-face (normal along z): area = dy * dx
    # y-face (normal along y): area = dz * dx
    # x-face (normal along x): area = dz * dy
    face_z = dy * dx
    face_y = dz * dx
    face_x = dz * dy

    total = 0.0

    # For each axis, count boundaries between foreground and background
    # Z-axis
    diff_z = np.diff(binary.astype(np.int8), axis=0)
    total += np.abs(diff_z).sum() * face_z
    # Add top and bottom boundary faces
    total += binary[0].sum() * face_z   # top face exposed
    total += binary[-1].sum() * face_z  # bottom face exposed

    # Y-axis
    diff_y = np.diff(binary.astype(np.int8), axis=1)
    total += np.abs(diff_y).sum() * face_y
    total += binary[:, 0, :].sum() * face_y
    total += binary[:, -1, :].sum() * face_y

    # X-axis
    diff_x = np.diff(binary.astype(np.int8), axis=2)
    total += np.abs(diff_x).sum() * face_x
    total += binary[:, :, 0].sum() * face_x
    total += binary[:, :, -1].sum() * face_x

    return float(total)
