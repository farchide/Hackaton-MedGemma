"""Generate synthetic NIfTI test volumes with realistic tumor features.

Creates CT-like volumes in Hounsfield Units with embedded spherical lesions,
realistic noise, and corresponding segmentation masks. Produces a baseline
scan and a follow-up scan with a slightly larger tumor to simulate
longitudinal imaging.

Usage:
    PYTHONPATH=src python scripts/generate_test_volumes.py

Output files (under data/sample_images/nifti/):
    - synthetic_tumor_ct.nii.gz         Baseline CT volume (128x128x64)
    - synthetic_tumor_mask.nii.gz       Binary segmentation mask for baseline
    - synthetic_tumor_ct_followup.nii.gz Follow-up CT volume (larger tumor)
    - synthetic_tumor_mask_followup.nii.gz Binary segmentation mask for follow-up
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "sample_images" / "nifti"

# Volume geometry
VOLUME_SHAPE = (128, 128, 64)  # (W, H, D) in nibabel/NIfTI convention
VOXEL_SIZE_MM = (1.0, 1.0, 2.5)  # typical CT spacing (x, y, z)

# Hounsfield Unit ranges for different tissues
HU_AIR = -1000.0
HU_LUNG = -700.0
HU_FAT = -100.0
HU_SOFT_TISSUE = 40.0
HU_LIVER = 60.0
HU_TUMOR = 80.0  # slightly hyperdense relative to liver
HU_BONE_CORTICAL = 1000.0

# Tumor parameters
TUMOR_CENTER_BASELINE = (64, 64, 32)  # voxel coordinates (centered)
TUMOR_RADIUS_BASELINE_MM = 15.0  # ~15 mm radius = ~30 mm diameter
TUMOR_RADIUS_FOLLOWUP_MM = 19.0  # ~27% volume increase (growth)

# Noise
NOISE_STD_HU = 12.0  # realistic CT noise level


# ---------------------------------------------------------------------------
# Volume generation helpers
# ---------------------------------------------------------------------------

def _build_affine(voxel_size: tuple[float, float, float]) -> np.ndarray:
    """Build a RAS+ affine matrix from voxel sizes."""
    affine = np.eye(4)
    affine[0, 0] = voxel_size[0]
    affine[1, 1] = voxel_size[1]
    affine[2, 2] = voxel_size[2]
    # Center the volume so origin is near the middle
    affine[0, 3] = -voxel_size[0] * VOLUME_SHAPE[0] / 2
    affine[1, 3] = -voxel_size[1] * VOLUME_SHAPE[1] / 2
    affine[2, 3] = -voxel_size[2] * VOLUME_SHAPE[2] / 2
    return affine


def _create_ellipsoid_body(shape: tuple[int, int, int]) -> np.ndarray:
    """Create a simplified body phantom with tissue-like HU values.

    Returns a float32 array with shape *shape* containing Hounsfield Unit
    values for air (background), soft tissue (body), and a central
    liver-like organ region.
    """
    volume = np.full(shape, HU_AIR, dtype=np.float32)

    w, h, d = shape
    cx, cy, cz = w // 2, h // 2, d // 2

    # Coordinate grids
    zz, yy, xx = np.meshgrid(
        np.arange(d), np.arange(h), np.arange(w), indexing="ij"
    )
    # Note: meshgrid with ij indexing gives (D, H, W), but our volume is
    # (W, H, D) in NIfTI convention.  Transpose to match.
    xx = xx.transpose(2, 1, 0)  # (W, H, D)
    yy = yy.transpose(2, 1, 0)
    zz = zz.transpose(2, 1, 0)

    # Body ellipsoid: ~80% of volume extent
    body_rx, body_ry, body_rz = w * 0.40, h * 0.35, d * 0.42
    body_dist = (
        ((xx - cx) / body_rx) ** 2
        + ((yy - cy) / body_ry) ** 2
        + ((zz - cz) / body_rz) ** 2
    )
    body_mask = body_dist <= 1.0
    volume[body_mask] = HU_SOFT_TISSUE

    # Liver region: offset right and slightly inferior
    liver_cx = cx + w * 0.12
    liver_cy = cy + h * 0.05
    liver_cz = cz - d * 0.05
    liver_rx, liver_ry, liver_rz = w * 0.18, h * 0.15, d * 0.20
    liver_dist = (
        ((xx - liver_cx) / liver_rx) ** 2
        + ((yy - liver_cy) / liver_ry) ** 2
        + ((zz - liver_cz) / liver_rz) ** 2
    )
    liver_mask = liver_dist <= 1.0
    volume[liver_mask] = HU_LIVER

    # Spine: small bright cylinder along z-axis in posterior region
    spine_cx, spine_cy = cx, cy - h * 0.28
    spine_r = w * 0.04
    spine_dist = ((xx - spine_cx) ** 2 + (yy - spine_cy) ** 2)
    spine_mask = (spine_dist <= spine_r ** 2) & body_mask
    volume[spine_mask] = HU_BONE_CORTICAL

    return volume


def _add_spherical_tumor(
    volume: np.ndarray,
    center: tuple[int, int, int],
    radius_mm: float,
    voxel_size: tuple[float, float, float],
    hu_value: float = HU_TUMOR,
) -> tuple[np.ndarray, np.ndarray]:
    """Insert a spherical tumor into *volume* and return (modified_volume, mask).

    The tumor has a smooth edge with slight intensity variation to simulate
    heterogeneous enhancement.

    Parameters
    ----------
    volume:
        3-D array of HU values, shape (W, H, D).
    center:
        Tumor center in voxel coordinates (x, y, z).
    radius_mm:
        Tumor radius in millimetres.
    voxel_size:
        Voxel dimensions in mm (x, y, z).
    hu_value:
        Peak HU value for the tumor core.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (volume_with_tumor, binary_mask) both of shape (W, H, D).
    """
    w, h, d = volume.shape
    cx, cy, cz = center

    # Radius in voxel coordinates per axis
    rx = radius_mm / voxel_size[0]
    ry = radius_mm / voxel_size[1]
    rz = radius_mm / voxel_size[2]

    zz, yy, xx = np.meshgrid(
        np.arange(d), np.arange(h), np.arange(w), indexing="ij"
    )
    xx = xx.transpose(2, 1, 0)
    yy = yy.transpose(2, 1, 0)
    zz = zz.transpose(2, 1, 0)

    # Normalised distance from tumor center
    dist = np.sqrt(
        ((xx - cx) / rx) ** 2
        + ((yy - cy) / ry) ** 2
        + ((zz - cz) / rz) ** 2
    )

    # Binary mask: inside the sphere
    mask = (dist <= 1.0).astype(np.uint8)

    # Smooth intensity falloff at the edge (sigmoid-like)
    # Core is brighter, periphery slightly dimmer
    intensity_map = np.clip(1.0 - dist, 0.0, 1.0)

    # Add internal heterogeneity using Perlin-like noise (simplified)
    rng = np.random.default_rng(seed=42)
    heterogeneity = rng.normal(0, 8.0, size=volume.shape).astype(np.float32)

    vol_out = volume.copy()
    tumor_voxels = mask.astype(bool)
    vol_out[tumor_voxels] = (
        hu_value * intensity_map[tumor_voxels]
        + (1.0 - intensity_map[tumor_voxels]) * volume[tumor_voxels]
        + heterogeneity[tumor_voxels]
    )

    return vol_out, mask


def _add_noise(volume: np.ndarray, std: float, seed: int = 123) -> np.ndarray:
    """Add Gaussian noise to a volume."""
    rng = np.random.default_rng(seed=seed)
    noise = rng.normal(0, std, size=volume.shape).astype(np.float32)
    return volume + noise


def _save_nifti(
    data: np.ndarray,
    affine: np.ndarray,
    path: Path,
    dtype: type = np.float32,
) -> None:
    """Save a numpy array as a NIfTI file."""
    img = nib.Nifti1Image(data.astype(dtype), affine)
    img.header.set_zooms(tuple(abs(affine[i, i]) for i in range(3)))
    nib.save(img, str(path))
    logger.info("Saved: %s (shape=%s, dtype=%s)", path, data.shape, dtype.__name__)


# ---------------------------------------------------------------------------
# Main generation pipeline
# ---------------------------------------------------------------------------

def generate_all() -> list[Path]:
    """Generate all synthetic test volumes and return list of output paths."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    affine = _build_affine(VOXEL_SIZE_MM)
    outputs: list[Path] = []

    # --- Baseline scan ---
    logger.info("Generating baseline body phantom...")
    body = _create_ellipsoid_body(VOLUME_SHAPE)

    logger.info(
        "Inserting baseline tumor (center=%s, radius=%.1f mm)...",
        TUMOR_CENTER_BASELINE,
        TUMOR_RADIUS_BASELINE_MM,
    )
    baseline_vol, baseline_mask = _add_spherical_tumor(
        body, TUMOR_CENTER_BASELINE, TUMOR_RADIUS_BASELINE_MM, VOXEL_SIZE_MM
    )
    baseline_vol = _add_noise(baseline_vol, NOISE_STD_HU, seed=111)

    # Compute tumor volume for logging
    voxel_volume_mm3 = VOXEL_SIZE_MM[0] * VOXEL_SIZE_MM[1] * VOXEL_SIZE_MM[2]
    baseline_tumor_vol = float(np.sum(baseline_mask)) * voxel_volume_mm3
    logger.info(
        "Baseline tumor volume: %.1f mm3 (%.1f cm3)",
        baseline_tumor_vol,
        baseline_tumor_vol / 1000,
    )

    path_ct = OUTPUT_DIR / "synthetic_tumor_ct.nii.gz"
    path_mask = OUTPUT_DIR / "synthetic_tumor_mask.nii.gz"
    _save_nifti(baseline_vol, affine, path_ct, np.float32)
    _save_nifti(baseline_mask, affine, path_mask, np.uint8)
    outputs.extend([path_ct, path_mask])

    # --- Follow-up scan (tumor growth) ---
    logger.info(
        "Inserting follow-up tumor (center=%s, radius=%.1f mm)...",
        TUMOR_CENTER_BASELINE,
        TUMOR_RADIUS_FOLLOWUP_MM,
    )
    followup_vol, followup_mask = _add_spherical_tumor(
        body, TUMOR_CENTER_BASELINE, TUMOR_RADIUS_FOLLOWUP_MM, VOXEL_SIZE_MM
    )
    followup_vol = _add_noise(followup_vol, NOISE_STD_HU, seed=222)

    followup_tumor_vol = float(np.sum(followup_mask)) * voxel_volume_mm3
    growth_pct = (
        (followup_tumor_vol - baseline_tumor_vol) / baseline_tumor_vol * 100
        if baseline_tumor_vol > 0
        else 0.0
    )
    logger.info(
        "Follow-up tumor volume: %.1f mm3 (%.1f cm3) [+%.1f%%]",
        followup_tumor_vol,
        followup_tumor_vol / 1000,
        growth_pct,
    )

    path_ct_fu = OUTPUT_DIR / "synthetic_tumor_ct_followup.nii.gz"
    path_mask_fu = OUTPUT_DIR / "synthetic_tumor_mask_followup.nii.gz"
    _save_nifti(followup_vol, affine, path_ct_fu, np.float32)
    _save_nifti(followup_mask, affine, path_mask_fu, np.uint8)
    outputs.extend([path_ct_fu, path_mask_fu])

    # --- Simple MR-like volume (for MR normalization testing) ---
    logger.info("Generating synthetic MR-like volume...")
    mr_body = _create_ellipsoid_body(VOLUME_SHAPE)
    # Remap HU values to arbitrary MR intensity (no HU in MRI)
    mr_body[mr_body <= HU_AIR + 10] = 0.0  # air -> 0
    mr_body[(mr_body > HU_AIR + 10) & (mr_body < HU_SOFT_TISSUE)] = 200.0
    mr_body[(mr_body >= HU_SOFT_TISSUE) & (mr_body < HU_LIVER)] = 400.0
    mr_body[mr_body >= HU_LIVER] = 600.0
    # Insert tumor with MR-like intensity
    mr_vol, mr_mask = _add_spherical_tumor(
        mr_body,
        TUMOR_CENTER_BASELINE,
        TUMOR_RADIUS_BASELINE_MM,
        VOXEL_SIZE_MM,
        hu_value=800.0,  # bright on T1+contrast
    )
    mr_vol = _add_noise(mr_vol, 20.0, seed=333)
    mr_vol = np.clip(mr_vol, 0, None)  # MR intensities are non-negative

    path_mr = OUTPUT_DIR / "synthetic_tumor_mr.nii.gz"
    path_mr_mask = OUTPUT_DIR / "synthetic_tumor_mr_mask.nii.gz"
    _save_nifti(mr_vol, affine, path_mr, np.float32)
    _save_nifti(mr_mask, affine, path_mr_mask, np.uint8)
    outputs.extend([path_mr, path_mr_mask])

    logger.info("All synthetic volumes generated successfully.")
    return outputs


if __name__ == "__main__":
    try:
        paths = generate_all()
        print(f"\nGenerated {len(paths)} files:")
        for p in paths:
            size_kb = p.stat().st_size / 1024
            print(f"  {p.relative_to(p.parent.parent.parent.parent)}  ({size_kb:.0f} KB)")
    except Exception as exc:
        logger.error("Failed to generate volumes: %s", exc, exc_info=True)
        sys.exit(1)
