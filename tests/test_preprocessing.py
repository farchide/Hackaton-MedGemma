"""Tests for preprocessing: normalization, resampling, and slice selection."""

from __future__ import annotations

import numpy as np
import numpy.testing as npt
import pytest

from digital_twin_tumor.domain.models import ProcessedVolume, VoxelSpacing
from digital_twin_tumor.preprocessing.normalize import (
    CT_WINDOW_PRESETS,
    normalize_ct,
    normalize_mri,
    normalize_volume,
)
from digital_twin_tumor.preprocessing.resample import resample_volume
from digital_twin_tumor.preprocessing.slice_selector import (
    create_mip,
    select_volume_representative,
    slice_to_png,
)


# =====================================================================
# normalize_ct
# =====================================================================


class TestNormalizeCT:
    """Test CT HU windowing normalization."""

    @pytest.mark.parametrize("window", list(CT_WINDOW_PRESETS.keys()))
    def test_output_in_zero_one(self, window):
        rng = np.random.default_rng(42)
        # Generate HU-like data spanning a broad range
        volume = rng.uniform(-1024, 2000, size=(16, 16, 16)).astype(np.float32)
        result = normalize_ct(volume, window=window)
        assert result.dtype == np.float32
        assert float(result.min()) >= 0.0 - 1e-6
        assert float(result.max()) <= 1.0 + 1e-6

    @pytest.mark.parametrize("window", list(CT_WINDOW_PRESETS.keys()))
    def test_shape_preserved(self, window):
        volume = np.zeros((10, 20, 30), dtype=np.float32)
        result = normalize_ct(volume, window=window)
        assert result.shape == (10, 20, 30)

    def test_known_values_soft_tissue(self):
        low, high = CT_WINDOW_PRESETS["soft_tissue"]
        # Value at the window minimum should map to 0
        vol_low = np.full((2, 2, 2), low, dtype=np.float32)
        result_low = normalize_ct(vol_low, window="soft_tissue")
        npt.assert_allclose(result_low, 0.0, atol=1e-6)

        # Value at the window maximum should map to 1
        vol_high = np.full((2, 2, 2), high, dtype=np.float32)
        result_high = normalize_ct(vol_high, window="soft_tissue")
        npt.assert_allclose(result_high, 1.0, atol=1e-6)

    def test_invalid_window_raises(self):
        volume = np.zeros((4, 4, 4), dtype=np.float32)
        with pytest.raises(ValueError, match="Unknown CT window"):
            normalize_ct(volume, window="invalid_preset")

    def test_clipping_below_window(self):
        low, high = CT_WINDOW_PRESETS["bone"]
        vol = np.full((2, 2, 2), low - 500, dtype=np.float32)
        result = normalize_ct(vol, window="bone")
        npt.assert_allclose(result, 0.0, atol=1e-6)

    def test_clipping_above_window(self):
        low, high = CT_WINDOW_PRESETS["bone"]
        vol = np.full((2, 2, 2), high + 500, dtype=np.float32)
        result = normalize_ct(vol, window="bone")
        npt.assert_allclose(result, 1.0, atol=1e-6)


# =====================================================================
# normalize_mri
# =====================================================================


class TestNormalizeMRI:
    """Test MRI z-score normalization."""

    def test_output_in_zero_one(self):
        rng = np.random.default_rng(42)
        volume = rng.uniform(100, 5000, size=(16, 16, 16)).astype(np.float32)
        result = normalize_mri(volume)
        assert result.dtype == np.float32
        assert float(result.min()) >= -1e-6
        assert float(result.max()) <= 1.0 + 1e-6

    def test_shape_preserved(self):
        volume = np.ones((8, 12, 10), dtype=np.float32) * 500.0
        # Add variation to avoid degenerate case
        volume[4, 6, 5] = 1000.0
        result = normalize_mri(volume)
        assert result.shape == (8, 12, 10)

    def test_zero_volume_returns_zeros(self):
        volume = np.zeros((4, 4, 4), dtype=np.float32)
        result = normalize_mri(volume)
        npt.assert_array_equal(result, np.zeros_like(volume))

    def test_constant_volume_returns_zeros(self):
        volume = np.full((4, 4, 4), 42.0, dtype=np.float32)
        result = normalize_mri(volume)
        npt.assert_array_equal(result, np.zeros_like(volume, dtype=np.float32))


# =====================================================================
# normalize_volume (dispatcher)
# =====================================================================


class TestNormalizeVolume:
    """Test the modality dispatcher."""

    def test_dispatches_ct(self, sample_volume):
        # sample_volume modality is "CT"
        result = normalize_volume(sample_volume)
        assert result.pixel_data.shape == sample_volume.pixel_data.shape
        assert result.modality == "CT"

    def test_dispatches_mri(self):
        rng = np.random.default_rng(42)
        data = rng.uniform(100, 5000, size=(16, 16, 16)).astype(np.float32)
        vol = ProcessedVolume(pixel_data=data, modality="MRI")
        result = normalize_volume(vol)
        assert result.modality == "MRI"
        assert float(result.pixel_data.min()) >= -1e-6
        assert float(result.pixel_data.max()) <= 1.0 + 1e-6

    def test_unsupported_modality_raises(self):
        vol = ProcessedVolume(
            pixel_data=np.zeros((4, 4, 4), dtype=np.float32), modality="PET"
        )
        with pytest.raises(ValueError, match="Unsupported modality"):
            normalize_volume(vol)


# =====================================================================
# resample_volume
# =====================================================================


class TestResampleVolume:
    """Test isotropic resampling."""

    def test_identity_resampling(self, sample_volume):
        """Resampling with same spacing should produce identical data."""
        result = resample_volume(sample_volume, target_spacing=(1.0, 1.0, 1.0))
        assert result.pixel_data.shape == sample_volume.pixel_data.shape
        npt.assert_allclose(
            result.pixel_data, sample_volume.pixel_data, atol=1e-5
        )

    def test_upsample(self):
        """Resampling to finer spacing should produce larger volume."""
        data = np.random.default_rng(42).random((8, 8, 8)).astype(np.float32)
        vol = ProcessedVolume(
            pixel_data=data,
            spacing=VoxelSpacing(x=2.0, y=2.0, z=2.0),
        )
        result = resample_volume(vol, target_spacing=(1.0, 1.0, 1.0))
        # Each dimension should roughly double
        for orig, res in zip(data.shape, result.pixel_data.shape):
            assert res >= orig

    def test_downsample(self):
        """Resampling to coarser spacing should produce smaller volume."""
        data = np.random.default_rng(42).random((16, 16, 16)).astype(np.float32)
        vol = ProcessedVolume(
            pixel_data=data,
            spacing=VoxelSpacing(x=1.0, y=1.0, z=1.0),
        )
        result = resample_volume(vol, target_spacing=(2.0, 2.0, 2.0))
        for orig, res in zip(data.shape, result.pixel_data.shape):
            assert res <= orig

    def test_spacing_updated(self):
        data = np.random.default_rng(42).random((8, 8, 8)).astype(np.float32)
        vol = ProcessedVolume(
            pixel_data=data,
            spacing=VoxelSpacing(x=2.0, y=2.0, z=2.0),
        )
        result = resample_volume(vol, target_spacing=(1.0, 1.0, 1.0))
        assert result.spacing.x == 1.0
        assert result.spacing.y == 1.0
        assert result.spacing.z == 1.0

    def test_mask_resampled_when_present(self):
        data = np.random.default_rng(42).random((8, 8, 8)).astype(np.float32)
        mask = np.zeros((8, 8, 8), dtype=np.uint8)
        mask[3:6, 3:6, 3:6] = 1
        vol = ProcessedVolume(
            pixel_data=data,
            spacing=VoxelSpacing(x=2.0, y=2.0, z=2.0),
            mask=mask,
        )
        result = resample_volume(vol, target_spacing=(1.0, 1.0, 1.0))
        assert result.mask is not None
        assert result.mask.shape == result.pixel_data.shape

    def test_2d_raises(self):
        vol = ProcessedVolume(
            pixel_data=np.zeros((8, 8), dtype=np.float32)
        )
        with pytest.raises(ValueError, match="3-D volume"):
            resample_volume(vol, target_spacing=(1.0, 1.0, 1.0))


# =====================================================================
# select_volume_representative
# =====================================================================


class TestSelectVolumeRepresentative:
    """Test evenly-spaced representative slice selection."""

    def test_returns_correct_number(self, sample_volume):
        slices = select_volume_representative(sample_volume, n_slices=4)
        assert len(slices) == 4

    def test_returns_fewer_when_volume_small(self):
        data = np.random.default_rng(42).random((3, 32, 32)).astype(np.float32)
        vol = ProcessedVolume(pixel_data=data)
        slices = select_volume_representative(vol, n_slices=10)
        # At most 3 slices when depth is 3
        assert len(slices) <= 3

    def test_slices_are_512x512_uint8(self, sample_volume):
        slices = select_volume_representative(sample_volume, n_slices=2)
        for s in slices:
            assert s.shape == (512, 512)
            assert s.dtype == np.uint8


# =====================================================================
# create_mip
# =====================================================================


class TestCreateMIP:
    """Test Maximum Intensity Projection."""

    def test_returns_2d(self, sample_volume):
        mip = create_mip(sample_volume, axis=0)
        assert mip.ndim == 2
        assert mip.dtype == np.float32

    @pytest.mark.parametrize("axis", [0, 1, 2])
    def test_correct_output_shape(self, sample_volume, axis):
        mip = create_mip(sample_volume, axis=axis)
        expected_shape = list(sample_volume.pixel_data.shape)
        del expected_shape[axis]
        assert mip.shape == tuple(expected_shape)

    def test_mip_values_are_max(self, sample_volume):
        mip = create_mip(sample_volume, axis=0)
        expected = np.max(sample_volume.pixel_data, axis=0).astype(np.float32)
        npt.assert_array_equal(mip, expected)

    def test_invalid_axis_raises(self, sample_volume):
        with pytest.raises(ValueError, match="out of range"):
            create_mip(sample_volume, axis=5)


# =====================================================================
# slice_to_png
# =====================================================================


class TestSliceToPng:
    """Test 2D slice to PNG-ready conversion."""

    def test_output_shape_and_dtype(self):
        slice_2d = np.random.default_rng(42).random((32, 32)).astype(np.float32)
        result = slice_to_png(slice_2d, size=512)
        assert result.shape == (512, 512)
        assert result.dtype == np.uint8

    def test_output_range(self):
        slice_2d = np.random.default_rng(42).random((32, 32)).astype(np.float32)
        result = slice_to_png(slice_2d, size=512)
        assert int(result.min()) >= 0
        assert int(result.max()) <= 255

    def test_3d_input_raises(self):
        with pytest.raises(ValueError, match="2-D"):
            slice_to_png(np.zeros((8, 8, 3), dtype=np.float32))
