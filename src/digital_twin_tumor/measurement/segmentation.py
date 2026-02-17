"""Segmentation engine with MedSAM support and Otsu fallback.

Implements ADR-004: configurable segmentation backends with lazy model
loading. When MedSAM is unavailable the ``FallbackSegmentation`` provides
a classical-CV pipeline based on Otsu thresholding and morphological
operations so the system never hard-fails in a hackathon environment.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from scipy import ndimage as ndi
from skimage import morphology
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Fallback (classical CV) segmentation
# ---------------------------------------------------------------------------

class FallbackSegmentation:
    """Classical segmentation when MedSAM is not available.

    Pipeline:
        1. Crop to the bounding-box ROI.
        2. Otsu threshold on the ROI.
        3. Morphological opening to remove small objects.
        4. Keep the largest connected component.
        5. Morphological closing to fill holes.
    """

    def segment_roi(
        self,
        image: np.ndarray,
        bbox: tuple[int, int, int, int],
        *,
        opening_radius: int = 3,
        closing_radius: int = 5,
    ) -> np.ndarray:
        """Segment a region of interest using Otsu + morphology.

        Parameters
        ----------
        image:
            2-D grayscale image (H, W).
        bbox:
            Bounding box as ``(x_min, y_min, x_max, y_max)`` in pixel
            coordinates.
        opening_radius:
            Disk radius for the opening step (noise removal).
        closing_radius:
            Disk radius for the closing step (hole filling).

        Returns
        -------
        np.ndarray
            Binary mask with the same shape as *image*.
        """
        x_min, y_min, x_max, y_max = bbox
        h, w = image.shape[:2]

        # Clamp bbox to image bounds
        x_min = max(0, int(x_min))
        y_min = max(0, int(y_min))
        x_max = min(w, int(x_max))
        y_max = min(h, int(y_max))

        if x_max <= x_min or y_max <= y_min:
            logger.warning("Invalid bbox %s – returning empty mask.", bbox)
            return np.zeros(image.shape[:2], dtype=np.uint8)

        roi = image[y_min:y_max, x_min:x_max]

        # Handle constant-valued ROI (Otsu would fail)
        if roi.max() == roi.min():
            return np.zeros(image.shape[:2], dtype=np.uint8)

        # Normalise to 0-255 uint8 for Otsu
        roi_norm = roi.astype(np.float64)
        roi_norm = (roi_norm - roi_norm.min()) / (roi_norm.max() - roi_norm.min())
        roi_uint8 = (roi_norm * 255).astype(np.uint8)

        # 1. Otsu threshold
        try:
            thresh = threshold_otsu(roi_uint8)
        except ValueError:
            return np.zeros(image.shape[:2], dtype=np.uint8)

        binary = roi_uint8 > thresh

        # 2. Morphological opening
        selem_open = morphology.disk(opening_radius)
        binary = morphology.opening(binary, selem_open)

        # 3. Keep largest connected component
        binary = _keep_largest_component(binary)

        # 4. Morphological closing
        selem_close = morphology.disk(closing_radius)
        binary = morphology.closing(binary, selem_close)

        # Place ROI mask back into full-image mask
        full_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        full_mask[y_min:y_max, x_min:x_max] = binary.astype(np.uint8)
        return full_mask

    def segment_with_points(
        self,
        image: np.ndarray,
        points: list[tuple[int, int]],
        labels: list[int],
        *,
        radius: int = 15,
    ) -> np.ndarray:
        """Fallback point-prompt segmentation.

        Constructs a bounding box from the foreground points (label=1)
        then delegates to :meth:`segment_roi`.

        Parameters
        ----------
        image:
            2-D grayscale image.
        points:
            List of ``(x, y)`` pixel coordinates.
        labels:
            Corresponding labels (1 = foreground, 0 = background).
        radius:
            Pixel margin added around the foreground convex hull to
            form the bounding box.

        Returns
        -------
        np.ndarray
            Binary mask with the same shape as *image*.
        """
        fg_points = [p for p, lbl in zip(points, labels) if lbl == 1]
        if not fg_points:
            logger.warning("No foreground points provided – returning empty mask.")
            return np.zeros(image.shape[:2], dtype=np.uint8)

        xs = [p[0] for p in fg_points]
        ys = [p[1] for p in fg_points]
        bbox = (
            min(xs) - radius,
            min(ys) - radius,
            max(xs) + radius,
            max(ys) + radius,
        )
        return self.segment_roi(image, bbox)


# ---------------------------------------------------------------------------
# Main segmentation engine
# ---------------------------------------------------------------------------

class SegmentationEngine:
    """Configurable segmentation engine with MedSAM support.

    Falls back to :class:`FallbackSegmentation` when the requested model
    is not installed.

    Parameters
    ----------
    model_name:
        Name of the segmentation model backend.  ``"medsam"`` attempts
        to load the MedSAM model; any other value uses the fallback.
    device:
        PyTorch device string (``"cpu"``, ``"cuda"``, ``"cuda:0"``, etc.).
    """

    def __init__(self, model_name: str = "medsam", device: str = "cpu") -> None:
        self.model_name = model_name
        self.device = device
        self._model: Any = None
        self._model_loaded: bool = False
        self._fallback = FallbackSegmentation()

    # -- public API --------------------------------------------------------

    def segment_with_box(
        self,
        image: np.ndarray,
        bbox: tuple[int, int, int, int],
    ) -> np.ndarray:
        """Segment a lesion using a bounding-box prompt.

        Parameters
        ----------
        image:
            2-D grayscale or RGB image.
        bbox:
            ``(x_min, y_min, x_max, y_max)`` in pixel coordinates.

        Returns
        -------
        np.ndarray
            Binary mask (0/1) with the same spatial shape as *image*.
        """
        if image.size == 0:
            return np.zeros(image.shape[:2], dtype=np.uint8)

        self._load_model()

        if self._model is not None:
            return self._medsam_predict_box(image, bbox)

        # Fallback path
        img_2d = _to_grayscale(image)
        return self._fallback.segment_roi(img_2d, bbox)

    def segment_with_points(
        self,
        image: np.ndarray,
        points: list[tuple[int, int]],
        labels: list[int],
    ) -> np.ndarray:
        """Segment a lesion using point prompts.

        Parameters
        ----------
        image:
            2-D grayscale or RGB image.
        points:
            List of ``(x, y)`` coordinates.
        labels:
            Per-point labels: ``1`` = foreground, ``0`` = background.

        Returns
        -------
        np.ndarray
            Binary mask (0/1) with the same spatial shape as *image*.
        """
        if image.size == 0:
            return np.zeros(image.shape[:2], dtype=np.uint8)
        if len(points) != len(labels):
            raise ValueError(
                f"points ({len(points)}) and labels ({len(labels)}) "
                "must have the same length."
            )

        self._load_model()

        if self._model is not None:
            return self._medsam_predict_points(image, points, labels)

        img_2d = _to_grayscale(image)
        return self._fallback.segment_with_points(img_2d, points, labels)

    def refine_mask(
        self,
        mask: np.ndarray,
        brush_adds: np.ndarray | None = None,
        brush_removes: np.ndarray | None = None,
    ) -> np.ndarray:
        """Apply manual brush refinements to an existing mask.

        Parameters
        ----------
        mask:
            Existing binary mask (0/1).
        brush_adds:
            Binary mask of pixels to add (set to 1).
        brush_removes:
            Binary mask of pixels to remove (set to 0).

        Returns
        -------
        np.ndarray
            Refined binary mask.
        """
        refined = mask.copy().astype(np.uint8)

        if brush_adds is not None:
            if brush_adds.shape != mask.shape:
                raise ValueError(
                    f"brush_adds shape {brush_adds.shape} does not match "
                    f"mask shape {mask.shape}."
                )
            refined = np.maximum(refined, brush_adds.astype(np.uint8))

        if brush_removes is not None:
            if brush_removes.shape != mask.shape:
                raise ValueError(
                    f"brush_removes shape {brush_removes.shape} does not "
                    f"match mask shape {mask.shape}."
                )
            refined[brush_removes.astype(bool)] = 0

        return refined

    # -- internal helpers --------------------------------------------------

    def _load_model(self) -> None:
        """Lazy-load the segmentation model.

        Attempts to import and instantiate the MedSAM model.  On any
        import or runtime error the engine falls back to classical CV.
        """
        if self._model_loaded:
            return
        self._model_loaded = True

        if self.model_name != "medsam":
            logger.info(
                "Model '%s' is not MedSAM – using fallback segmentation.",
                self.model_name,
            )
            return

        try:
            # Attempt MedSAM import (requires segment-anything + weights)
            from segment_anything import sam_model_registry, SamPredictor  # type: ignore[import-untyped]
            import torch  # type: ignore[import-untyped]

            sam = sam_model_registry["vit_b"](checkpoint="medsam_vit_b.pth")
            sam.to(self.device)
            self._model = SamPredictor(sam)
            logger.info("MedSAM model loaded on device '%s'.", self.device)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Could not load MedSAM (%s). Falling back to classical "
                "segmentation.",
                exc,
            )
            self._model = None

    def _medsam_predict_box(
        self,
        image: np.ndarray,
        bbox: tuple[int, int, int, int],
    ) -> np.ndarray:
        """Run MedSAM box-prompt prediction."""
        import torch  # type: ignore[import-untyped]

        img_rgb = _to_rgb(image)
        self._model.set_image(img_rgb)

        box_np = np.array([[bbox[0], bbox[1], bbox[2], bbox[3]]])
        box_torch = torch.as_tensor(box_np, dtype=torch.float, device=self.device)
        transformed_box = self._model.transform.apply_boxes_torch(
            box_torch, img_rgb.shape[:2]
        )

        masks, scores, _ = self._model.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_box,
            multimask_output=False,
        )
        mask = masks[0, 0].cpu().numpy().astype(np.uint8)
        return mask

    def _medsam_predict_points(
        self,
        image: np.ndarray,
        points: list[tuple[int, int]],
        labels: list[int],
    ) -> np.ndarray:
        """Run MedSAM point-prompt prediction."""
        img_rgb = _to_rgb(image)
        self._model.set_image(img_rgb)

        point_coords = np.array(points, dtype=np.float32)
        point_labels = np.array(labels, dtype=np.int32)

        masks, scores, _ = self._model.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=False,
        )
        mask = masks[0].astype(np.uint8)
        return mask


# ---------------------------------------------------------------------------
# Module-level utilities
# ---------------------------------------------------------------------------

def _to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert an image to 2-D grayscale if needed."""
    if image.ndim == 2:
        return image
    if image.ndim == 3 and image.shape[2] == 3:
        # ITU-R BT.601 luminance weights
        return (
            0.2989 * image[..., 0]
            + 0.5870 * image[..., 1]
            + 0.1140 * image[..., 2]
        ).astype(image.dtype)
    if image.ndim == 3 and image.shape[2] == 1:
        return image[..., 0]
    return image.mean(axis=-1).astype(image.dtype) if image.ndim == 3 else image


def _to_rgb(image: np.ndarray) -> np.ndarray:
    """Ensure image is 3-channel RGB uint8 for SAM."""
    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)
    elif image.ndim == 3 and image.shape[2] == 1:
        image = np.repeat(image, 3, axis=2)

    if image.dtype != np.uint8:
        img_f = image.astype(np.float64)
        if img_f.max() > img_f.min():
            img_f = (img_f - img_f.min()) / (img_f.max() - img_f.min()) * 255
        image = img_f.astype(np.uint8)

    return image


def _keep_largest_component(binary: np.ndarray) -> np.ndarray:
    """Return a mask containing only the largest connected component.

    Parameters
    ----------
    binary:
        2-D boolean or uint8 mask.

    Returns
    -------
    np.ndarray
        Boolean mask with only the largest connected component.
    """
    labelled, num_features = ndi.label(binary)
    if num_features == 0:
        return np.zeros_like(binary, dtype=bool)

    # argmax on bincount (skip background label 0)
    component_sizes = np.bincount(labelled.ravel())
    # Ignore background at index 0
    if len(component_sizes) <= 1:
        return np.zeros_like(binary, dtype=bool)
    component_sizes[0] = 0
    largest_label = component_sizes.argmax()
    return labelled == largest_label


def compute_segmentation_confidence(mask: np.ndarray) -> float:
    """Compute confidence score for a segmentation mask.

    Combines multiple heuristics:
    - Compactness: ratio of area to perimeter squared (higher = more regular)
    - Size: penalize very small masks (likely noise) and very large (likely leak)
    - Edge sharpness: gradient magnitude along mask boundary

    Parameters
    ----------
    mask:
        2-D binary mask.

    Returns
    -------
    float
        Confidence in [0, 1].
    """
    if mask is None or mask.size == 0:
        return 0.0

    binary = mask.astype(bool)
    area = int(np.sum(binary))
    if area == 0:
        return 0.0

    total_pixels = binary.size

    # Compactness score (isoperimetric ratio)
    # Approximate perimeter by counting boundary pixels
    eroded = binary.copy()
    eroded[1:, :] &= binary[:-1, :]
    eroded[:-1, :] &= binary[1:, :]
    eroded[:, 1:] &= binary[:, :-1]
    eroded[:, :-1] &= binary[:, 1:]
    perimeter = area - int(np.sum(eroded))
    perimeter = max(perimeter, 1)
    compactness = min(1.0, (4 * np.pi * area) / (perimeter ** 2))

    # Size score - penalize extremes
    size_ratio = area / total_pixels
    if size_ratio < 0.001:
        size_score = size_ratio / 0.001  # linearly ramp up
    elif size_ratio > 0.5:
        size_score = max(0.0, 1.0 - (size_ratio - 0.5) / 0.5)
    else:
        size_score = 1.0

    # Connected component score - single component is more confident
    try:
        from scipy.ndimage import label as _label
        labeled, n_components = _label(binary)
        component_score = 1.0 / max(n_components, 1)
    except ImportError:
        component_score = 0.5

    # Weighted combination
    confidence = 0.4 * compactness + 0.3 * size_score + 0.3 * component_score
    return round(min(1.0, max(0.0, confidence)), 4)
