"""Modality-aware intensity normalisation for CT and MRI volumes.

Implements the normalisation strategies described in ADR-007:

- **CT**: Hounsfield Unit windowing with tissue-specific presets, rescaled
  to [0, 1] float32.
- **MRI**: Percentile-clipped z-score normalisation using foreground
  statistics, rescaled to [0, 1] float32.

The public entry-point :func:`normalize_volume` dispatches automatically
based on the volume's ``modality`` field.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Dict, Tuple

import numpy as np

from digital_twin_tumor.domain.models import ProcessedVolume

# Optional dependency -- used for Otsu threshold if available.
try:
    from skimage.filters import threshold_otsu as _threshold_otsu  # type: ignore[import-untyped]

    _HAS_SKIMAGE = True
except ImportError:  # pragma: no cover
    _HAS_SKIMAGE = False

# ---------------------------------------------------------------------------
# CT window presets (ADR-007 Table)
# ---------------------------------------------------------------------------

CT_WINDOW_PRESETS: Dict[str, Tuple[float, float]] = {
    "soft_tissue": (-150.0, 250.0),
    "lung": (-1000.0, 200.0),
    "bone": (-500.0, 1500.0),
    "brain": (0.0, 80.0),
    "liver": (-20.0, 200.0),
}

_VALID_WINDOWS = frozenset(CT_WINDOW_PRESETS.keys())


# ---------------------------------------------------------------------------
# CT normalisation
# ---------------------------------------------------------------------------


def normalize_ct(
    volume: np.ndarray,
    window: str = "soft_tissue",
) -> np.ndarray:
    """Apply HU windowing and rescale a CT volume to [0, 1] float32.

    Parameters
    ----------
    volume:
        3-D numpy array of CT Hounsfield Unit values.
    window:
        Name of the windowing preset.  Must be one of ``"soft_tissue"``,
        ``"lung"``, ``"bone"``, ``"brain"``, or ``"liver"``.

    Returns
    -------
    np.ndarray
        Float32 array in [0, 1] with the same shape as *volume*.

    Raises
    ------
    ValueError
        If *window* is not a recognised preset name.
    """
    if window not in _VALID_WINDOWS:
        raise ValueError(
            f"Unknown CT window preset '{window}'. "
            f"Choose from: {sorted(_VALID_WINDOWS)}"
        )

    low, high = CT_WINDOW_PRESETS[window]

    # Guard against degenerate window (should never happen with our presets).
    if high <= low:
        raise ValueError(f"Window high ({high}) must be greater than low ({low}).")

    out = np.array(volume, dtype=np.float32, copy=True)
    np.clip(out, low, high, out=out)

    # Rescale [low, high] -> [0, 1]
    out -= low
    out /= high - low
    return out


# ---------------------------------------------------------------------------
# MRI normalisation
# ---------------------------------------------------------------------------


def _foreground_mask(volume: np.ndarray) -> np.ndarray:
    """Compute a boolean foreground mask for an MRI volume.

    Strategy:
    1. If *skimage* is available, use Otsu thresholding on the positive
       voxels.
    2. Otherwise, fall back to a simple ``> 0`` mask.

    Parameters
    ----------
    volume:
        3-D MRI intensity array.

    Returns
    -------
    np.ndarray
        Boolean array with the same shape as *volume*.
    """
    positive = volume > 0

    # If no positive voxels, return all-False mask.
    if not np.any(positive):
        return np.zeros(volume.shape, dtype=bool)

    if _HAS_SKIMAGE:
        try:
            thresh = _threshold_otsu(volume[positive])
            return volume > thresh
        except (ValueError, RuntimeError):
            # Otsu can fail on degenerate distributions (e.g. all same value).
            return positive

    return positive


def normalize_mri(
    volume: np.ndarray,
    percentile_low: float = 1.0,
    percentile_high: float = 99.0,
) -> np.ndarray:
    """Z-score normalise an MRI volume and rescale to [0, 1] float32.

    Processing steps:
    1. Compute a foreground mask (Otsu or ``> 0``).
    2. Clip intensities to the ``[percentile_low, percentile_high]`` range
       computed over foreground voxels.
    3. Z-score normalise using foreground mean and standard deviation.
    4. Rescale the result to [0, 1] using the min/max of the z-scored data.

    Parameters
    ----------
    volume:
        3-D MRI intensity array.
    percentile_low:
        Lower clipping percentile (default 1.0).
    percentile_high:
        Upper clipping percentile (default 99.0).

    Returns
    -------
    np.ndarray
        Float32 array in [0, 1] with the same shape as *volume*.
    """
    out = np.array(volume, dtype=np.float32, copy=True)

    fg_mask = _foreground_mask(out)

    # Edge case: no foreground voxels -- return zeros.
    if not np.any(fg_mask):
        return np.zeros_like(out, dtype=np.float32)

    fg_values = out[fg_mask]

    # Step 2: Percentile clipping
    p_low = float(np.percentile(fg_values, percentile_low))
    p_high = float(np.percentile(fg_values, percentile_high))

    # Guard against constant-valued foreground.
    if p_high <= p_low:
        return np.zeros_like(out, dtype=np.float32)

    np.clip(out, p_low, p_high, out=out)

    # Step 3: Z-score using foreground stats
    fg_mean = float(np.mean(out[fg_mask]))
    fg_std = float(np.std(out[fg_mask]))

    if fg_std < 1e-8:
        return np.zeros_like(out, dtype=np.float32)

    out -= fg_mean
    out /= fg_std

    # Step 4: Rescale z-scored values to [0, 1]
    v_min = float(out.min())
    v_max = float(out.max())

    if v_max - v_min < 1e-8:
        return np.zeros_like(out, dtype=np.float32)

    out -= v_min
    out /= v_max - v_min
    return out


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------


def normalize_volume(
    volume: ProcessedVolume,
    *,
    ct_window: str = "soft_tissue",
    mri_percentile_low: float = 1.0,
    mri_percentile_high: float = 99.0,
) -> ProcessedVolume:
    """Normalise a :class:`ProcessedVolume` based on its modality.

    Dispatches to :func:`normalize_ct` for CT volumes and
    :func:`normalize_mri` for MRI volumes.

    Parameters
    ----------
    volume:
        The processed volume to normalise.
    ct_window:
        CT windowing preset (ignored for MRI).
    mri_percentile_low:
        Lower percentile for MRI clipping (ignored for CT).
    mri_percentile_high:
        Upper percentile for MRI clipping (ignored for CT).

    Returns
    -------
    ProcessedVolume
        A **new** volume with normalised ``pixel_data``.  All other fields
        are preserved.

    Raises
    ------
    ValueError
        If the volume modality is not ``"CT"`` or ``"MR"``/``"MRI"``.
    """
    modality = volume.modality.upper().strip()

    if modality == "CT":
        normalised = normalize_ct(volume.pixel_data, window=ct_window)
    elif modality in ("MR", "MRI"):
        normalised = normalize_mri(
            volume.pixel_data,
            percentile_low=mri_percentile_low,
            percentile_high=mri_percentile_high,
        )
    else:
        raise ValueError(
            f"Unsupported modality '{volume.modality}'. "
            "Expected 'CT' or 'MR'/'MRI'."
        )

    # Build a new ProcessedVolume preserving all other attributes.
    return ProcessedVolume(
        pixel_data=normalised,
        spacing=volume.spacing,
        origin=volume.origin,
        direction=volume.direction,
        modality=volume.modality,
        metadata=volume.metadata,
        mask=volume.mask,
        selected_slices=list(volume.selected_slices),
    )
