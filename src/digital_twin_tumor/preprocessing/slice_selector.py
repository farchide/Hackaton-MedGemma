"""Intelligent slice selection for MedGemma 2-D reasoning.

Provides strategies for extracting representative 2-D axial slices from
3-D medical imaging volumes.  Selected slices are converted to 512x512
uint8 arrays suitable for PNG export and MedGemma ingestion (ADR-007).

Available strategies:

- **Lesion-centric**: selects slices centred on a lesion's centroid.
- **Volume-representative**: evenly spaced through the volume, skipping
  empty slices.
- **MIP (Maximum Intensity Projection)**: collapses a 3-D volume along
  one axis.
"""
from __future__ import annotations

from typing import List

import numpy as np

from digital_twin_tumor.domain.models import ProcessedVolume

# Optional dependency -- PIL for high-quality resize.
try:
    from PIL import Image as _PILImage  # type: ignore[import-untyped]

    _HAS_PIL = True
except ImportError:  # pragma: no cover
    _HAS_PIL = False

# Optional dependency -- skimage for resize fallback.
try:
    from skimage.transform import resize as _skimage_resize  # type: ignore[import-untyped]

    _HAS_SKIMAGE = True
except ImportError:  # pragma: no cover
    _HAS_SKIMAGE = False


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _resize_2d(image: np.ndarray, size: int) -> np.ndarray:
    """Resize a 2-D array to ``(size, size)`` using the best available backend.

    Parameters
    ----------
    image:
        2-D float array.
    size:
        Target height **and** width in pixels.

    Returns
    -------
    np.ndarray
        Resized float array of shape ``(size, size)``.
    """
    if image.shape[0] == size and image.shape[1] == size:
        return image.copy()

    if _HAS_PIL:
        # PIL expects uint8 or float [0,1] -> scale to uint8 for resizing.
        img_min = float(image.min())
        img_max = float(image.max())
        if img_max - img_min < 1e-8:
            return np.zeros((size, size), dtype=np.float32)
        scaled = ((image - img_min) / (img_max - img_min) * 255.0).astype(np.uint8)
        pil_img = _PILImage.fromarray(scaled, mode="L")
        pil_resized = pil_img.resize((size, size), _PILImage.LANCZOS)
        result = np.asarray(pil_resized, dtype=np.float32) / 255.0
        # Re-map to original dynamic range.
        return result * (img_max - img_min) + img_min

    if _HAS_SKIMAGE:
        return _skimage_resize(
            image.astype(np.float64),
            (size, size),
            anti_aliasing=True,
            preserve_range=True,
        ).astype(np.float32)

    # Pure-numpy fallback: nearest-neighbour via fancy indexing.
    row_idx = (np.arange(size) * image.shape[0] / size).astype(int)
    col_idx = (np.arange(size) * image.shape[1] / size).astype(int)
    row_idx = np.clip(row_idx, 0, image.shape[0] - 1)
    col_idx = np.clip(col_idx, 0, image.shape[1] - 1)
    return image[np.ix_(row_idx, col_idx)].astype(np.float32)


def _to_uint8(arr: np.ndarray) -> np.ndarray:
    """Scale a float array to uint8 [0, 255].

    Parameters
    ----------
    arr:
        Input array of arbitrary numeric type.

    Returns
    -------
    np.ndarray
        Uint8 array with values in [0, 255].
    """
    arr_f = arr.astype(np.float64)
    v_min = float(arr_f.min())
    v_max = float(arr_f.max())
    if v_max - v_min < 1e-8:
        return np.zeros(arr.shape, dtype=np.uint8)
    scaled = (arr_f - v_min) / (v_max - v_min) * 255.0
    return np.clip(scaled, 0, 255).astype(np.uint8)


def _nonzero_fraction(slice_2d: np.ndarray) -> float:
    """Return the fraction of non-zero pixels in a 2-D array."""
    if slice_2d.size == 0:
        return 0.0
    return float(np.count_nonzero(slice_2d) / slice_2d.size)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def slice_to_png(
    slice_2d: np.ndarray,
    size: int = 512,
) -> np.ndarray:
    """Convert a 2-D slice to a PNG-ready uint8 array.

    The slice is resized to ``(size, size)`` and mapped to [0, 255] uint8.

    Parameters
    ----------
    slice_2d:
        2-D numeric array representing a single axial slice.
    size:
        Output height **and** width in pixels.

    Returns
    -------
    np.ndarray
        Uint8 array of shape ``(size, size)`` ready for PNG encoding.

    Raises
    ------
    ValueError
        If *slice_2d* is not 2-D.
    """
    if slice_2d.ndim != 2:
        raise ValueError(f"Expected a 2-D array, got {slice_2d.ndim}-D.")

    resized = _resize_2d(slice_2d.astype(np.float32), size)
    return _to_uint8(resized)


def select_lesion_centric(
    volume: ProcessedVolume,
    centroid: tuple[float, float, float],
    n_slices: int = 4,
) -> list[np.ndarray]:
    """Select axial slices centred on a lesion centroid.

    The centre slice is placed at the z-coordinate of *centroid*.
    Additional slices are distributed symmetrically above and below the
    centre, spaced by the volume's z-spacing.

    Each returned slice is a 512x512 uint8 array suitable for PNG export.

    Parameters
    ----------
    volume:
        The processed volume to slice.
    centroid:
        ``(x, y, z)`` physical-space centroid of the lesion.
    n_slices:
        Total number of slices to return (including the centre slice).
        Must be >= 1.

    Returns
    -------
    list[np.ndarray]
        List of uint8 arrays, each of shape ``(512, 512)``.
    """
    if n_slices < 1:
        raise ValueError(f"n_slices must be >= 1, got {n_slices}.")

    data = volume.pixel_data
    if data.ndim < 3 or data.shape[0] == 0:
        return []

    depth = data.shape[0]
    z_spacing = volume.spacing.z if volume.spacing.z > 0 else 1.0

    # Convert physical z-coordinate to slice index.
    # Assume origin[2] corresponds to index 0 along the depth axis.
    origin_z = volume.origin[2] if len(volume.origin) > 2 else 0.0
    centre_idx = int(round((centroid[2] - origin_z) / z_spacing))
    centre_idx = max(0, min(centre_idx, depth - 1))

    # Distribute slices symmetrically around the centre using integer steps.
    # We grow outward from the centre, alternating above/below, until we
    # have enough slices or exhaust the volume.
    requested = min(n_slices, depth)
    seen: set[int] = {centre_idx}
    unique_indices: list[int] = [centre_idx]

    distance = 1
    while len(unique_indices) < requested:
        # Try below centre, then above centre at each distance.
        for candidate in (centre_idx - distance, centre_idx + distance):
            if 0 <= candidate < depth and candidate not in seen:
                seen.add(candidate)
                unique_indices.append(candidate)
                if len(unique_indices) >= requested:
                    break
        distance += 1
        # Safety: if distance exceeds depth, all reachable indices are found.
        if distance > depth:
            break

    unique_indices.sort()

    slices: list[np.ndarray] = []
    for idx in unique_indices:
        slices.append(slice_to_png(data[idx]))

    return slices


def select_volume_representative(
    volume: ProcessedVolume,
    n_slices: int = 4,
    min_content_fraction: float = 0.05,
) -> list[np.ndarray]:
    """Select evenly-spaced representative slices, avoiding empty ones.

    Slices are first filtered to those with at least *min_content_fraction*
    non-zero content.  From the surviving candidates, *n_slices* are chosen
    at evenly-spaced positions.

    Each returned slice is a 512x512 uint8 array suitable for PNG export.

    Parameters
    ----------
    volume:
        The processed volume to slice.
    n_slices:
        Number of slices to return.
    min_content_fraction:
        Minimum fraction of non-zero pixels for a slice to be considered
        non-empty (default 5 %).

    Returns
    -------
    list[np.ndarray]
        List of uint8 arrays, each of shape ``(512, 512)``.
    """
    if n_slices < 1:
        raise ValueError(f"n_slices must be >= 1, got {n_slices}.")

    data = volume.pixel_data
    if data.ndim < 3 or data.shape[0] == 0:
        return []

    depth = data.shape[0]

    # Filter to non-empty slices.
    candidate_indices: list[int] = []
    for z in range(depth):
        if _nonzero_fraction(data[z]) >= min_content_fraction:
            candidate_indices.append(z)

    # Fallback: if too few candidates, use all slices.
    if len(candidate_indices) < n_slices:
        candidate_indices = list(range(depth))

    # Select evenly spaced indices from candidates.
    n_candidates = len(candidate_indices)
    if n_candidates <= n_slices:
        selected = candidate_indices
    else:
        step = (n_candidates - 1) / max(n_slices - 1, 1)
        selected = [
            candidate_indices[int(round(i * step))] for i in range(n_slices)
        ]

    slices: list[np.ndarray] = []
    for idx in selected:
        slices.append(slice_to_png(data[idx]))

    return slices


def create_mip(
    volume: ProcessedVolume,
    axis: int = 1,
) -> np.ndarray:
    """Compute a Maximum Intensity Projection along the given axis.

    Parameters
    ----------
    volume:
        The processed volume.
    axis:
        Projection axis.  Common choices:

        - ``0`` -- axial MIP (collapse depth).
        - ``1`` -- coronal MIP (collapse height / anterior-posterior).
        - ``2`` -- sagittal MIP (collapse width / left-right).

    Returns
    -------
    np.ndarray
        2-D float32 MIP image.

    Raises
    ------
    ValueError
        If *axis* is out of range for a 3-D volume.
    """
    data = volume.pixel_data
    if data.ndim < 3:
        raise ValueError(f"Expected a 3-D volume, got {data.ndim}-D array.")

    if axis < 0 or axis >= data.ndim:
        raise ValueError(
            f"axis {axis} is out of range for a {data.ndim}-D volume."
        )

    mip = np.max(data, axis=axis).astype(np.float32)
    return mip
