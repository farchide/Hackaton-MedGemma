"""Tests for the dataset registry, PROTEASLoader, and download instructions.

Uses mock filesystem structures so no real data downloads are required.
"""
from __future__ import annotations

import os

import pytest

from digital_twin_tumor.data.datasets import (
    DATASETS,
    DatasetRegistry,
    PROTEASLoader,
    dataset_download_instructions,
)


# ---------------------------------------------------------------------------
# DatasetRegistry
# ---------------------------------------------------------------------------


class TestDatasetRegistry:
    def test_list_datasets_returns_list(self):
        datasets = DatasetRegistry.list_datasets()
        assert isinstance(datasets, list)
        assert len(datasets) >= 1

    def test_list_datasets_has_required_keys(self):
        datasets = DatasetRegistry.list_datasets()
        for entry in datasets:
            assert "key" in entry
            assert "name" in entry
            assert "modality" in entry

    def test_get_dataset_known(self):
        info = DatasetRegistry.get_dataset("proteas_brain_met")
        assert info is not None
        assert info["name"] == "PROTEAS Brain Metastases"

    def test_get_dataset_unknown(self):
        info = DatasetRegistry.get_dataset("nonexistent_dataset")
        assert info is None

    def test_validate_directory_unknown_dataset(self):
        result = DatasetRegistry.validate_directory("no_such_dataset", "/tmp")
        assert result["valid"] is False

    def test_validate_directory_missing_dir(self, tmp_path):
        result = DatasetRegistry.validate_directory(
            "proteas_brain_met", str(tmp_path / "nonexistent"),
        )
        assert result["valid"] is False

    def test_dataset_info_structure(self):
        """Every registered dataset must have the required metadata keys."""
        required_keys = {
            "name", "source", "url", "modality", "longitudinal",
            "patients", "description", "download_instructions",
        }
        for key, info in DATASETS.items():
            missing = required_keys - set(info.keys())
            assert not missing, f"Dataset {key} missing keys: {missing}"


# ---------------------------------------------------------------------------
# PROTEASLoader with mock data directory
# ---------------------------------------------------------------------------


class TestPROTEASLoader:
    @pytest.fixture()
    def mock_proteas_dir(self, tmp_path):
        """Create a minimal BIDS-like directory for PROTEAS."""
        root = tmp_path / "proteas"
        root.mkdir()

        for sub_id in ["sub-001", "sub-002"]:
            sub_dir = root / sub_id
            sub_dir.mkdir()
            for ses in ["ses-baseline", "ses-fu6w"]:
                ses_dir = sub_dir / ses
                anat_dir = ses_dir / "anat"
                anat_dir.mkdir(parents=True)
                # Create a dummy NIfTI file (just the name; no valid content)
                (anat_dir / f"{sub_id}_{ses}_T1w.nii.gz").touch()

        return root

    def test_init_raises_for_missing_dir(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            PROTEASLoader(str(tmp_path / "nonexistent"))

    def test_list_patients(self, mock_proteas_dir):
        loader = PROTEASLoader(str(mock_proteas_dir))
        patients = loader.list_patients()
        ids = [p["patient_id"] for p in patients]
        assert "sub-001" in ids
        assert "sub-002" in ids

    def test_load_patient(self, mock_proteas_dir):
        loader = PROTEASLoader(str(mock_proteas_dir))
        data = loader.load_patient("sub-001")
        assert data["patient_id"] == "sub-001"
        assert data["num_timepoints"] == 2
        assert len(data["timepoints"]) == 2

    def test_load_patient_not_found(self, mock_proteas_dir):
        loader = PROTEASLoader(str(mock_proteas_dir))
        with pytest.raises(FileNotFoundError):
            loader.load_patient("sub-999")

    def test_timepoints_sorted_by_week(self, mock_proteas_dir):
        loader = PROTEASLoader(str(mock_proteas_dir))
        data = loader.load_patient("sub-001")
        weeks = [tp["week"] for tp in data["timepoints"]]
        assert weeks == sorted(weeks)

    def test_treatment_info_returned(self, mock_proteas_dir):
        loader = PROTEASLoader(str(mock_proteas_dir))
        data = loader.load_patient("sub-001")
        assert "treatment_info" in data
        assert isinstance(data["treatment_info"], list)


# ---------------------------------------------------------------------------
# download_instructions
# ---------------------------------------------------------------------------


class TestDownloadInstructions:
    def test_returns_markdown(self):
        text = dataset_download_instructions()
        assert isinstance(text, str)
        assert "# Public Dataset Download Instructions" in text

    def test_contains_all_datasets(self):
        text = dataset_download_instructions()
        for key, info in DATASETS.items():
            assert info["name"] in text
