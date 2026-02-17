"""Tests for measurement: diameter, volume, segmentation, and RECIST."""

from __future__ import annotations

import numpy as np
import numpy.testing as npt
import pytest

from digital_twin_tumor.domain.models import Lesion, Measurement, VoxelSpacing
from digital_twin_tumor.measurement.diameter import (
    measure_longest_diameter,
    measure_short_axis,
)
from digital_twin_tumor.measurement.recist import RECISTClassifier
from digital_twin_tumor.measurement.segmentation import SegmentationEngine
from digital_twin_tumor.measurement.volume import (
    compute_volume,
    compute_volume_change,
)


# =====================================================================
# Helper: create geometric masks
# =====================================================================


def _circle_mask(radius: int = 20, center: tuple[int, int] = (50, 50),
                 shape: tuple[int, int] = (100, 100)) -> np.ndarray:
    """Create a 2D binary circle mask."""
    mask = np.zeros(shape, dtype=np.uint8)
    yy, xx = np.ogrid[:shape[0], :shape[1]]
    dist = np.sqrt((yy - center[0]) ** 2 + (xx - center[1]) ** 2)
    mask[dist <= radius] = 1
    return mask


def _ellipse_mask(a: int = 30, b: int = 15,
                  center: tuple[int, int] = (50, 50),
                  shape: tuple[int, int] = (100, 100)) -> np.ndarray:
    """Create a 2D binary ellipse mask with semi-axes a (row) and b (col)."""
    mask = np.zeros(shape, dtype=np.uint8)
    yy, xx = np.ogrid[:shape[0], :shape[1]]
    dist = ((yy - center[0]) / a) ** 2 + ((xx - center[1]) / b) ** 2
    mask[dist <= 1.0] = 1
    return mask


def _sphere_mask(radius: int = 10, center: tuple[int, int, int] = (32, 32, 32),
                 shape: tuple[int, int, int] = (64, 64, 64)) -> np.ndarray:
    """Create a 3D binary sphere mask."""
    mask = np.zeros(shape, dtype=np.uint8)
    zz, yy, xx = np.ogrid[:shape[0], :shape[1], :shape[2]]
    dist = np.sqrt(
        (zz - center[0]) ** 2 + (yy - center[1]) ** 2 + (xx - center[2]) ** 2
    )
    mask[dist <= radius] = 1
    return mask


# =====================================================================
# measure_longest_diameter
# =====================================================================


class TestMeasureLongestDiameter:
    """Test longest diameter computation on known geometric shapes."""

    def test_circle_diameter(self):
        """A circle of radius 20 with spacing 1.0 should have LD ~40 mm."""
        mask = _circle_mask(radius=20, center=(50, 50), shape=(100, 100))
        ld = measure_longest_diameter(mask, spacing=(1.0, 1.0))
        # Allow some tolerance for pixelation
        assert abs(ld - 40.0) < 3.0, f"Expected ~40.0, got {ld}"

    def test_circle_with_spacing(self):
        """A circle of radius 10 pixels with 2.0mm spacing -> LD ~40 mm."""
        mask = _circle_mask(radius=10, center=(50, 50), shape=(100, 100))
        ld = measure_longest_diameter(mask, spacing=(2.0, 2.0))
        assert abs(ld - 40.0) < 4.0, f"Expected ~40.0, got {ld}"

    def test_empty_mask_returns_zero(self):
        mask = np.zeros((50, 50), dtype=np.uint8)
        assert measure_longest_diameter(mask) == 0.0

    def test_single_pixel_very_small(self):
        """A single pixel should produce a very small diameter (sub-pixel contour)."""
        mask = np.zeros((50, 50), dtype=np.uint8)
        mask[25, 25] = 1
        result = measure_longest_diameter(mask)
        # find_contours produces a small contour around the pixel, so
        # the diameter is approximately 1.0 (one pixel diagonal), not zero.
        assert result < 2.0


# =====================================================================
# measure_short_axis
# =====================================================================


class TestMeasureShortAxis:
    """Test short-axis measurement on an elliptical mask."""

    def test_ellipse_short_axis(self):
        """An ellipse with semi-axes 30 (row) and 15 (col) should have SA ~30 mm."""
        mask = _ellipse_mask(a=30, b=15, center=(50, 50), shape=(100, 100))
        sa = measure_short_axis(mask, spacing=(1.0, 1.0))
        # Short axis should be ~2*b = 30
        assert sa > 0.0
        assert abs(sa - 30.0) < 8.0, f"Expected ~30.0, got {sa}"

    def test_circle_short_axis_close_to_diameter(self):
        """For a circle, short axis should be close to the diameter."""
        mask = _circle_mask(radius=15, center=(50, 50), shape=(100, 100))
        sa = measure_short_axis(mask, spacing=(1.0, 1.0))
        assert sa > 0.0
        assert abs(sa - 30.0) < 6.0, f"Expected ~30.0, got {sa}"

    def test_empty_mask_returns_zero(self):
        mask = np.zeros((50, 50), dtype=np.uint8)
        assert measure_short_axis(mask) == 0.0


# =====================================================================
# compute_volume
# =====================================================================


class TestComputeVolume:
    """Test volume computation on known shapes."""

    def test_sphere_volume(self, sample_mask):
        """A sphere of radius 10 with 1mm spacing -> V ~ 4/3 pi r^3 = 4189."""
        spacing = VoxelSpacing(x=1.0, y=1.0, z=1.0)
        vol = compute_volume(sample_mask, spacing)
        # Count how many voxels are in the sphere
        expected_voxels = int(sample_mask.sum())
        assert vol == expected_voxels * 1.0  # spacing product is 1.0
        # Should be close to 4/3 * pi * 10^3 = 4188.79
        assert abs(vol - 4189.0) < 200.0, f"Expected ~4189, got {vol}"

    def test_with_anisotropic_spacing(self, sample_mask):
        spacing = VoxelSpacing(x=1.0, y=1.0, z=2.0)
        vol = compute_volume(sample_mask, spacing)
        # Each voxel is 2.0 mm^3 instead of 1.0
        expected_voxels = int(sample_mask.sum())
        assert vol == expected_voxels * 2.0

    def test_empty_mask_returns_zero(self):
        mask = np.zeros((10, 10, 10), dtype=np.uint8)
        spacing = VoxelSpacing(x=1.0, y=1.0, z=1.0)
        assert compute_volume(mask, spacing) == 0.0


# =====================================================================
# compute_volume_change
# =====================================================================


class TestComputeVolumeChange:
    """Test volume change classification."""

    def test_growing(self):
        result = compute_volume_change(current_mm3=200.0, previous_mm3=100.0)
        assert result["absolute_change"] == 100.0
        assert result["percent_change"] == 100.0
        assert result["direction"] == "growing"

    def test_shrinking(self):
        result = compute_volume_change(current_mm3=50.0, previous_mm3=100.0)
        assert result["absolute_change"] == -50.0
        assert result["percent_change"] == -50.0
        assert result["direction"] == "shrinking"

    def test_stable(self):
        result = compute_volume_change(current_mm3=101.0, previous_mm3=100.0)
        assert abs(result["percent_change"]) < 5.0
        assert result["direction"] == "stable"

    def test_zero_previous(self):
        result = compute_volume_change(current_mm3=100.0, previous_mm3=0.0)
        assert result["percent_change"] == 100.0

    def test_both_zero(self):
        result = compute_volume_change(current_mm3=0.0, previous_mm3=0.0)
        assert result["percent_change"] == 0.0
        assert result["direction"] == "stable"


# =====================================================================
# SegmentationEngine (fallback)
# =====================================================================


class TestSegmentationEngineFallback:
    """Test fallback segmentation with box prompt."""

    def test_segment_with_box_produces_mask(self):
        engine = SegmentationEngine(model_name="fallback")
        # Create a synthetic image with a bright blob
        image = np.zeros((100, 100), dtype=np.float32)
        image[30:70, 30:70] = 1.0
        bbox = (25, 25, 75, 75)
        mask = engine.segment_with_box(image, bbox)
        assert mask.shape == (100, 100)
        assert mask.dtype == np.uint8
        # Should have some segmented pixels in the ROI
        assert mask[30:70, 30:70].sum() > 0

    def test_empty_image_returns_empty(self):
        engine = SegmentationEngine(model_name="fallback")
        mask = engine.segment_with_box(np.array([]), (0, 0, 10, 10))
        assert mask.size == 0


# =====================================================================
# RECISTClassifier
# =====================================================================


class TestRECISTSelectTargets:
    """Test target lesion selection."""

    def test_selects_up_to_max_total(self):
        lesions = [
            Lesion(
                lesion_id=f"L{i}", organ="liver",
                longest_diameter_mm=15.0, short_axis_mm=10.0,
            )
            for i in range(10)
        ]
        targets = RECISTClassifier.select_targets(
            lesions, max_total=5, max_per_organ=10
        )
        assert len(targets) <= 5

    def test_respects_max_per_organ(self):
        lesions = [
            Lesion(
                lesion_id=f"L{i}", organ="liver",
                longest_diameter_mm=15.0,
            )
            for i in range(5)
        ]
        targets = RECISTClassifier.select_targets(
            lesions, max_total=10, max_per_organ=2
        )
        assert len(targets) == 2

    def test_excludes_small_non_nodal(self):
        lesions = [
            Lesion(lesion_id="small", organ="liver", longest_diameter_mm=5.0),
            Lesion(lesion_id="big", organ="liver", longest_diameter_mm=15.0),
        ]
        targets = RECISTClassifier.select_targets(lesions)
        assert len(targets) == 1
        assert targets[0].lesion_id == "big"

    def test_nodal_uses_short_axis(self):
        lesions = [
            Lesion(
                lesion_id="node1", organ="lymph node",
                longest_diameter_mm=25.0, short_axis_mm=16.0,
            ),
            Lesion(
                lesion_id="node2", organ="lymph node",
                longest_diameter_mm=20.0, short_axis_mm=12.0,
            ),
        ]
        targets = RECISTClassifier.select_targets(lesions)
        # node1 has SA >= 15mm, node2 does not
        assert len(targets) == 1
        assert targets[0].lesion_id == "node1"

    def test_empty_input(self):
        assert RECISTClassifier.select_targets([]) == []


class TestRECISTClassifyResponse:
    """Test RECIST 1.1 response classification."""

    def test_complete_response(self):
        result = RECISTClassifier.classify_response(
            current_sum=0.0, baseline_sum=50.0, nadir_sum=20.0,
        )
        assert result.category == "CR"

    def test_partial_response(self):
        # >= 30% decrease from baseline
        result = RECISTClassifier.classify_response(
            current_sum=30.0, baseline_sum=50.0, nadir_sum=30.0,
        )
        # (30-50)/50 = -40%, which is <= -30%
        assert result.category == "PR"

    def test_stable_disease(self):
        result = RECISTClassifier.classify_response(
            current_sum=40.0, baseline_sum=50.0, nadir_sum=35.0,
        )
        # (40-50)/50 = -20%, which is > -30% (not PR)
        # (40-35)/35 = 14.3%, which is < 20% (not PD)
        assert result.category == "SD"

    def test_progressive_disease_from_nadir(self):
        result = RECISTClassifier.classify_response(
            current_sum=50.0, baseline_sum=40.0, nadir_sum=30.0,
        )
        # (50-30)/30 = 66.7% >= 20% and (50-30)=20 >= 5mm
        assert result.category == "PD"

    def test_progressive_disease_new_lesions(self):
        result = RECISTClassifier.classify_response(
            current_sum=40.0, baseline_sum=50.0, nadir_sum=40.0,
            new_lesions=True,
        )
        assert result.category == "PD"

    def test_percent_changes_calculated(self):
        result = RECISTClassifier.classify_response(
            current_sum=40.0, baseline_sum=50.0, nadir_sum=35.0,
        )
        expected_baseline = ((40 - 50) / 50) * 100
        expected_nadir = ((40 - 35) / 35) * 100
        assert abs(result.percent_change_from_baseline - expected_baseline) < 0.1
        assert abs(result.percent_change_from_nadir - expected_nadir) < 0.1


class TestRECISTClassifyOverall:
    """Test overall RECIST decision table."""

    def test_cr_cr_no_new(self):
        assert RECISTClassifier.classify_overall("CR", "CR", False) == "CR"

    def test_cr_non_cr_no_new(self):
        result = RECISTClassifier.classify_overall("CR", "non-CR/non-PD", False)
        assert result == "PR"

    def test_pd_any_no_new(self):
        assert RECISTClassifier.classify_overall("PD", "CR", False) == "PD"

    def test_any_pd_no_new(self):
        assert RECISTClassifier.classify_overall("CR", "PD", False) == "PD"

    def test_new_lesions_always_pd(self):
        assert RECISTClassifier.classify_overall("CR", "CR", True) == "PD"

    def test_ne_targets(self):
        assert RECISTClassifier.classify_overall("NE", "CR", False) == "NE"

    def test_sd_non_pd(self):
        assert RECISTClassifier.classify_overall("SD", "CR", False) == "SD"

    def test_pr_non_pd(self):
        assert RECISTClassifier.classify_overall("PR", "CR", False) == "PR"


class TestClassifyIRECIST:
    """Test iRECIST immunotherapy extension."""

    def test_not_on_immunotherapy_passthrough(self):
        result = RECISTClassifier.classify_irecist(
            "PD", previous_category=None, on_immunotherapy=False,
        )
        assert result == "PD"

    def test_first_pd_on_immunotherapy_is_iupd(self):
        result = RECISTClassifier.classify_irecist(
            "PD", previous_category=None, on_immunotherapy=True,
        )
        assert result == "iUPD"

    def test_confirmed_pd_after_iupd(self):
        result = RECISTClassifier.classify_irecist(
            "PD", previous_category="iUPD", on_immunotherapy=True,
        )
        assert result == "iCPD"

    def test_non_pd_after_iupd_resets(self):
        result = RECISTClassifier.classify_irecist(
            "SD", previous_category="iUPD", on_immunotherapy=True,
        )
        assert result == "iSD"

    def test_cr_on_immunotherapy(self):
        result = RECISTClassifier.classify_irecist(
            "CR", previous_category=None, on_immunotherapy=True,
        )
        assert result == "iCR"

    def test_pr_on_immunotherapy(self):
        result = RECISTClassifier.classify_irecist(
            "PR", previous_category="iPR", on_immunotherapy=True,
        )
        assert result == "iPR"
