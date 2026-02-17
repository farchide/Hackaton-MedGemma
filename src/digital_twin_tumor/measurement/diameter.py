"""Diameter measurement utilities for RECIST 1.1 response assessment.

Implements ADR-009: longest-diameter and short-axis measurements using
contour analysis from scikit-image and pairwise distance computations
from scipy.spatial.
"""

from __future__ import annotations

import logging

import numpy as np
from scipy.spatial.distance import pdist, squareform
from skimage.measure import find_contours, label, regionprops

from digital_twin_tumor.domain.models import VoxelSpacing

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_feret_diameter(
    contour_points: np.ndarray,
    spacing: tuple[float, float] = (1.0, 1.0),
) -> float:
    """Compute the Feret diameter (maximum caliper distance) of a contour.

    Parameters
    ----------
    contour_points:
        Array of shape ``(N, 2)`` with (row, col) coordinates.
    spacing:
        Physical pixel spacing ``(row_spacing, col_spacing)`` in mm.

    Returns
    -------
    float
        Feret diameter in mm.  Returns ``0.0`` when fewer than two
        points are provided.
    """
    if contour_points.shape[0] < 2:
        return 0.0

    # Scale contour by physical spacing
    scaled = contour_points.copy().astype(np.float64)
    scaled[:, 0] *= spacing[0]
    scaled[:, 1] *= spacing[1]

    # For small contours use brute-force pairwise distance
    if scaled.shape[0] <= 500:
        distances = pdist(scaled)
        if distances.size == 0:
            return 0.0
        return float(distances.max())

    # For larger contours use convex hull to reduce computation
    try:
        from scipy.spatial import ConvexHull

        hull = ConvexHull(scaled)
        hull_pts = scaled[hull.vertices]
        distances = pdist(hull_pts)
        if distances.size == 0:
            return 0.0
        return float(distances.max())
    except Exception:  # noqa: BLE001
        # Fall back to brute force on subsampled points
        step = max(1, scaled.shape[0] // 500)
        subsampled = scaled[::step]
        distances = pdist(subsampled)
        if distances.size == 0:
            return 0.0
        return float(distances.max())


def measure_longest_diameter(
    mask_2d: np.ndarray,
    spacing: tuple[float, float] = (1.0, 1.0),
) -> float:
    """Compute the longest diameter (Feret diameter) of a 2-D binary mask.

    Parameters
    ----------
    mask_2d:
        2-D binary mask (H, W) with nonzero values for the lesion.
    spacing:
        Physical pixel spacing ``(row_spacing, col_spacing)`` in mm.

    Returns
    -------
    float
        Longest diameter in millimetres.  Returns ``0.0`` for empty
        masks or single-pixel masks.
    """
    if mask_2d.size == 0 or mask_2d.max() == 0:
        return 0.0

    binary = (mask_2d > 0).astype(np.uint8)
    contour_points = _extract_contour_points(binary)

    if contour_points.shape[0] < 2:
        # Single pixel – diameter is roughly the pixel diagonal
        nonzero = np.argwhere(binary)
        if nonzero.shape[0] <= 1:
            return 0.0
        return compute_feret_diameter(nonzero.astype(np.float64), spacing)

    return compute_feret_diameter(contour_points, spacing)


def measure_short_axis(
    mask_2d: np.ndarray,
    spacing: tuple[float, float] = (1.0, 1.0),
) -> float:
    """Measure the short axis perpendicular to the longest diameter.

    Primarily used for lymph-node assessment per RECIST 1.1.

    Algorithm:
        1. Find the longest diameter axis endpoints.
        2. Compute a direction perpendicular to that axis.
        3. Project all mask pixels onto the perpendicular direction.
        4. Return the extent (max - min projection) as the short axis.

    Parameters
    ----------
    mask_2d:
        2-D binary mask (H, W).
    spacing:
        Physical pixel spacing ``(row_spacing, col_spacing)`` in mm.

    Returns
    -------
    float
        Short-axis diameter in mm.  Returns ``0.0`` for empty or
        single-pixel masks.
    """
    if mask_2d.size == 0 or mask_2d.max() == 0:
        return 0.0

    binary = (mask_2d > 0).astype(np.uint8)
    contour_points = _extract_contour_points(binary)

    if contour_points.shape[0] < 2:
        return 0.0

    # Find the two endpoints of the longest diameter
    scaled = contour_points.copy().astype(np.float64)
    scaled[:, 0] *= spacing[0]
    scaled[:, 1] *= spacing[1]

    # Use pairwise distances or convex hull
    if scaled.shape[0] <= 500:
        dist_matrix = squareform(pdist(scaled))
    else:
        try:
            from scipy.spatial import ConvexHull

            hull = ConvexHull(scaled)
            hull_pts = scaled[hull.vertices]
            dist_matrix = squareform(pdist(hull_pts))
            # Re-map to hull indices
            idx = np.unravel_index(dist_matrix.argmax(), dist_matrix.shape)
            p1 = hull_pts[idx[0]]
            p2 = hull_pts[idx[1]]
            return _short_axis_from_endpoints(binary, p1, p2, spacing)
        except Exception:  # noqa: BLE001
            step = max(1, scaled.shape[0] // 500)
            scaled_sub = scaled[::step]
            dist_matrix = squareform(pdist(scaled_sub))

    idx = np.unravel_index(dist_matrix.argmax(), dist_matrix.shape)
    p1 = scaled[idx[0]]
    p2 = scaled[idx[1]]

    return _short_axis_from_endpoints(binary, p1, p2, spacing)


def measure_3d_diameters(
    mask_3d: np.ndarray,
    spacing: VoxelSpacing,
) -> dict[str, float]:
    """Measure diameters on the axial slice with the largest cross-section.

    Parameters
    ----------
    mask_3d:
        3-D binary mask with shape ``(D, H, W)`` where D is the number
        of axial slices.
    spacing:
        Physical voxel spacing.

    Returns
    -------
    dict
        Keys: ``longest_diameter_mm``, ``short_axis_mm``,
        ``axial_slice_index``.  All values are ``0.0`` / ``0`` for
        empty masks.
    """
    result: dict[str, float] = {
        "longest_diameter_mm": 0.0,
        "short_axis_mm": 0.0,
        "axial_slice_index": 0,
    }

    if mask_3d.size == 0 or mask_3d.max() == 0:
        return result

    # Find the axial slice with the most foreground pixels
    slice_areas = np.array([
        (mask_3d[z] > 0).sum() for z in range(mask_3d.shape[0])
    ])

    if slice_areas.max() == 0:
        return result

    best_slice = int(slice_areas.argmax())
    axial_mask = mask_3d[best_slice]

    # In-plane spacing: (row = y, col = x)
    plane_spacing = (float(spacing.y), float(spacing.x))

    ld = measure_longest_diameter(axial_mask, plane_spacing)
    sa = measure_short_axis(axial_mask, plane_spacing)

    result["longest_diameter_mm"] = ld
    result["short_axis_mm"] = sa
    result["axial_slice_index"] = float(best_slice)
    return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _extract_contour_points(binary: np.ndarray) -> np.ndarray:
    """Extract contour coordinates from a 2-D binary mask.

    Uses :func:`skimage.measure.find_contours` at the 0.5 level and
    concatenates all contour fragments into a single ``(N, 2)`` array
    of ``(row, col)`` coordinates.

    Parameters
    ----------
    binary:
        2-D binary mask (uint8, 0/1).

    Returns
    -------
    np.ndarray
        Shape ``(N, 2)``.  Empty ``(0, 2)`` when no contour is found.
    """
    contours = find_contours(binary, level=0.5)
    if not contours:
        return np.empty((0, 2), dtype=np.float64)
    return np.concatenate(contours, axis=0)


def _short_axis_from_endpoints(
    binary: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray,
    spacing: tuple[float, float],
) -> float:
    """Compute the short axis perpendicular to the longest diameter.

    Uses a ray-casting approach: for each row or column along the
    perpendicular through the mask centroid, measure the chord length
    of the mask intersection.  Returns the maximum chord as the
    short-axis diameter.

    Parameters
    ----------
    binary:
        2-D binary mask (uint8).
    p1, p2:
        Endpoints of the longest diameter in *physical* coordinates
        (row_mm, col_mm).
    spacing:
        Physical pixel spacing ``(row_spacing, col_spacing)`` in mm.

    Returns
    -------
    float
        Short-axis length in mm.
    """
    direction = p2 - p1
    length = np.linalg.norm(direction)
    if length < 1e-12:
        return 0.0

    # Unit direction along longest diameter (in physical coords)
    ld_dir = direction / length

    # Perpendicular direction (90-degree rotation in 2-D)
    perp = np.array([-ld_dir[1], ld_dir[0]])

    # Compute centroid in physical coordinates
    rows, cols = np.nonzero(binary)
    if rows.size == 0:
        return 0.0
    centroid_mm = np.array([
        rows.mean() * spacing[0],
        cols.mean() * spacing[1],
    ])

    # Sample points along the perpendicular line through the centroid
    # and find where it intersects the mask boundary.
    # Use a generous range to ensure we cover the full extent.
    max_extent = max(binary.shape[0] * spacing[0], binary.shape[1] * spacing[1])
    t_values = np.linspace(-max_extent, max_extent, num=2000)

    inside_distances: list[float] = []
    for t in t_values:
        point_mm = centroid_mm + t * perp
        # Convert back to pixel coordinates
        r = point_mm[0] / spacing[0]
        c = point_mm[1] / spacing[1]
        ri, ci = int(round(r)), int(round(c))
        if 0 <= ri < binary.shape[0] and 0 <= ci < binary.shape[1]:
            if binary[ri, ci] > 0:
                inside_distances.append(t)

    if len(inside_distances) < 2:
        return 0.0

    # The short axis is the distance between the first and last
    # intersections (in mm, since t is parametrised in mm).
    return float(max(inside_distances) - min(inside_distances))
