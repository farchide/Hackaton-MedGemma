"""Tests for the Upload & Preprocess tab pipeline.

Covers:
- NIfTI loading (.nii and .nii.gz)
- Normalisation (CT windowing, MRI z-score, PET fallback)
- Slice extraction and bounds checking
- Gradio 6 file-path resolution (NamedString, str, Path)
- NIfTI suffix detection helper
- Slider update on upload
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np
import numpy.testing as npt
import pytest

from digital_twin_tumor.domain.models import ProcessedVolume, VoxelSpacing
from digital_twin_tumor.ingestion.nifti_loader import load_nifti
from digital_twin_tumor.preprocessing.normalize import normalize_volume
from digital_twin_tumor.ui.app import _is_nifti, _resolve_upload_path
from digital_twin_tumor.ui.components import extract_slice


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def synthetic_nifti_gz(tmp_path):
    """Create a small synthetic .nii.gz file and return its path."""
    data = np.random.default_rng(42).uniform(0, 1000, (64, 64, 20)).astype(
        np.float32,
    )
    data[25:40, 25:40, 8:14] = 1500.0
    affine = np.diag([1.0, 1.0, 2.0, 1.0])
    img = nib.Nifti1Image(data, affine)
    path = tmp_path / "scan.nii.gz"
    nib.save(img, str(path))
    return path


@pytest.fixture()
def synthetic_nifti(tmp_path):
    """Create a small synthetic .nii file and return its path."""
    data = np.random.default_rng(42).uniform(0, 1000, (64, 64, 20)).astype(
        np.float32,
    )
    affine = np.diag([1.0, 1.0, 2.0, 1.0])
    img = nib.Nifti1Image(data, affine)
    path = tmp_path / "scan.nii"
    nib.save(img, str(path))
    return path


# =====================================================================
# NIfTI loading
# =====================================================================


class TestLoadNifti:
    """Test NIfTI loader with real nibabel round-trip."""

    def test_load_nii_gz(self, synthetic_nifti_gz):
        vol = load_nifti(synthetic_nifti_gz)
        assert vol.pixel_data.ndim == 3
        assert vol.pixel_data.shape == (20, 64, 64)
        assert vol.pixel_data.dtype == np.float32

    def test_load_nii(self, synthetic_nifti):
        vol = load_nifti(synthetic_nifti)
        assert vol.pixel_data.ndim == 3
        assert vol.pixel_data.shape == (20, 64, 64)

    def test_spacing_extracted(self, synthetic_nifti_gz):
        vol = load_nifti(synthetic_nifti_gz)
        assert vol.spacing.x == pytest.approx(1.0, abs=0.01)
        assert vol.spacing.y == pytest.approx(1.0, abs=0.01)
        assert vol.spacing.z == pytest.approx(2.0, abs=0.01)

    def test_modality_default_mr(self, synthetic_nifti_gz):
        vol = load_nifti(synthetic_nifti_gz)
        assert vol.modality == "MR"

    def test_metadata_has_source(self, synthetic_nifti_gz):
        vol = load_nifti(synthetic_nifti_gz)
        assert "source_file" in vol.metadata
        assert str(synthetic_nifti_gz) in vol.metadata["source_file"]

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_nifti(tmp_path / "nonexistent.nii.gz")


# =====================================================================
# Normalisation
# =====================================================================


class TestNormalisePipeline:
    """Test normalisation of NIfTI-loaded volumes."""

    def test_mri_normalise_range(self, synthetic_nifti_gz):
        vol = load_nifti(synthetic_nifti_gz)
        norm = normalize_volume(vol)
        assert norm.pixel_data.dtype == np.float32
        assert float(norm.pixel_data.min()) >= -1e-6
        assert float(norm.pixel_data.max()) <= 1.0 + 1e-6

    def test_mri_normalise_preserves_shape(self, synthetic_nifti_gz):
        vol = load_nifti(synthetic_nifti_gz)
        norm = normalize_volume(vol)
        assert norm.pixel_data.shape == vol.pixel_data.shape

    def test_mri_normalise_preserves_spacing(self, synthetic_nifti_gz):
        vol = load_nifti(synthetic_nifti_gz)
        norm = normalize_volume(vol)
        assert norm.spacing == vol.spacing

    def test_ct_window_soft_tissue(self):
        data = np.random.default_rng(42).uniform(-1000, 2000, (16, 16, 16)).astype(
            np.float32,
        )
        vol = ProcessedVolume(
            pixel_data=data,
            spacing=VoxelSpacing(1, 1, 1),
            modality="CT",
        )
        norm = normalize_volume(vol, ct_window="soft_tissue")
        assert float(norm.pixel_data.min()) >= -1e-6
        assert float(norm.pixel_data.max()) <= 1.0 + 1e-6

    def test_ct_window_lung(self):
        data = np.random.default_rng(42).uniform(-1000, 2000, (16, 16, 16)).astype(
            np.float32,
        )
        vol = ProcessedVolume(
            pixel_data=data,
            spacing=VoxelSpacing(1, 1, 1),
            modality="CT",
        )
        norm = normalize_volume(vol, ct_window="lung")
        assert float(norm.pixel_data.min()) >= -1e-6
        assert float(norm.pixel_data.max()) <= 1.0 + 1e-6


# =====================================================================
# Slice extraction
# =====================================================================


class TestExtractSlice:
    """Test extract_slice with normalised volumes."""

    def test_middle_slice(self, synthetic_nifti_gz):
        vol = load_nifti(synthetic_nifti_gz)
        norm = normalize_volume(vol)
        mid = norm.pixel_data.shape[0] // 2
        slc = extract_slice(norm.pixel_data, mid)
        assert slc.shape == (64, 64)
        assert slc.dtype == np.uint8
        assert int(slc.max()) <= 255
        assert int(slc.min()) >= 0

    def test_first_slice(self, synthetic_nifti_gz):
        vol = load_nifti(synthetic_nifti_gz)
        slc = extract_slice(vol.pixel_data, 0)
        assert slc.shape == (64, 64)

    def test_last_slice(self, synthetic_nifti_gz):
        vol = load_nifti(synthetic_nifti_gz)
        slc = extract_slice(vol.pixel_data, vol.pixel_data.shape[0] - 1)
        assert slc.shape == (64, 64)

    def test_negative_index_clamped(self, synthetic_nifti_gz):
        vol = load_nifti(synthetic_nifti_gz)
        slc = extract_slice(vol.pixel_data, -5)
        # Should clamp to 0
        slc0 = extract_slice(vol.pixel_data, 0)
        npt.assert_array_equal(slc, slc0)

    def test_over_range_clamped(self, synthetic_nifti_gz):
        vol = load_nifti(synthetic_nifti_gz)
        slc = extract_slice(vol.pixel_data, 9999)
        slcN = extract_slice(vol.pixel_data, vol.pixel_data.shape[0] - 1)
        npt.assert_array_equal(slc, slcN)

    def test_empty_volume_returns_placeholder(self):
        slc = extract_slice(np.zeros((0, 64, 64), dtype=np.float32), 0)
        assert slc.shape == (256, 256)
        assert slc.dtype == np.uint8


# =====================================================================
# Path resolution (Gradio 6 compatibility)
# =====================================================================


class TestResolveUploadPath:
    """Test _resolve_upload_path with various input types."""

    def test_plain_string(self):
        p = _resolve_upload_path("/tmp/test.nii.gz")
        assert isinstance(p, Path)
        assert str(p) == "/tmp/test.nii.gz"

    def test_path_object(self):
        p = _resolve_upload_path(Path("/tmp/test.nii.gz"))
        assert str(p) == "/tmp/test.nii.gz"

    def test_named_string(self):
        from gradio.utils import NamedString

        p = _resolve_upload_path(NamedString("/tmp/test.nii.gz"))
        assert str(p) == "/tmp/test.nii.gz"

    def test_preserves_absolute_path(self):
        """Ensure Path.name (basename only) is never used."""
        p = _resolve_upload_path(Path("/a/b/c/scan.nii.gz"))
        assert str(p) == "/a/b/c/scan.nii.gz"


# =====================================================================
# NIfTI suffix detection
# =====================================================================


class TestIsNifti:
    """Test _is_nifti helper."""

    def test_nii(self):
        assert _is_nifti(Path("/tmp/scan.nii")) is True

    def test_nii_gz(self):
        assert _is_nifti(Path("/tmp/scan.nii.gz")) is True

    def test_dcm_rejected(self):
        assert _is_nifti(Path("/tmp/scan.dcm")) is False

    def test_tar_gz_rejected(self):
        assert _is_nifti(Path("/tmp/data.tar.gz")) is False

    def test_plain_gz_rejected(self):
        assert _is_nifti(Path("/tmp/data.gz")) is False

    def test_zip_rejected(self):
        assert _is_nifti(Path("/tmp/data.zip")) is False

    def test_case_insensitive(self):
        assert _is_nifti(Path("/tmp/scan.NII")) is True
        assert _is_nifti(Path("/tmp/scan.NII.GZ")) is True
        assert _is_nifti(Path("/tmp/scan.Nii.Gz")) is True
