"""Shared pytest fixtures for the Digital Twin Tumor test suite."""

from __future__ import annotations

from datetime import date, datetime

import numpy as np
import pytest

from digital_twin_tumor.domain.models import (
    Lesion,
    Measurement,
    ProcessedVolume,
    TherapyEvent,
    TimePoint,
    VoxelSpacing,
)
from digital_twin_tumor.storage.audit import AuditLogger
from digital_twin_tumor.storage.sqlite_backend import SQLiteBackend


# ---------------------------------------------------------------------------
# Volume / mask fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def sample_volume() -> ProcessedVolume:
    """A ProcessedVolume with random 64x64x64 float32 data."""
    rng = np.random.default_rng(seed=0)
    data = rng.random((64, 64, 64), dtype=np.float32)
    return ProcessedVolume(
        pixel_data=data,
        spacing=VoxelSpacing(x=1.0, y=1.0, z=1.0),
        origin=(0.0, 0.0, 0.0),
        modality="CT",
    )


@pytest.fixture()
def sample_mask() -> np.ndarray:
    """A 64x64x64 binary mask with a sphere of radius 10 centred at (32,32,32)."""
    mask = np.zeros((64, 64, 64), dtype=np.uint8)
    zz, yy, xx = np.ogrid[:64, :64, :64]
    dist = np.sqrt((zz - 32) ** 2 + (yy - 32) ** 2 + (xx - 32) ** 2)
    mask[dist <= 10] = 1
    return mask


# ---------------------------------------------------------------------------
# Domain object fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def sample_lesion() -> Lesion:
    """A Lesion with centroid (32,32,32), volume ~4189 mm3, diameter 20 mm."""
    return Lesion(
        lesion_id="lesion-001",
        timepoint_id="tp-001",
        centroid=(32.0, 32.0, 32.0),
        volume_mm3=4189.0,
        longest_diameter_mm=20.0,
        short_axis_mm=18.0,
        organ="liver",
        is_target=True,
        confidence=0.95,
    )


@pytest.fixture()
def sample_measurement() -> Measurement:
    """A Measurement with typical values."""
    return Measurement(
        measurement_id="meas-001",
        lesion_id="lesion-001",
        timepoint_id="tp-001",
        diameter_mm=20.0,
        volume_mm3=4189.0,
        method="auto",
        reviewer="system",
        timestamp=datetime(2025, 1, 15, 10, 0, 0),
    )


@pytest.fixture()
def sample_therapy_event() -> TherapyEvent:
    """A TherapyEvent with start/end dates."""
    return TherapyEvent(
        therapy_id="therapy-001",
        patient_id="patient-001",
        start_date=date(2025, 1, 1),
        end_date=date(2025, 3, 1),
        therapy_type="chemotherapy",
        dose="100mg",
    )


@pytest.fixture()
def sample_timepoint() -> TimePoint:
    """A TimePoint with typical values."""
    return TimePoint(
        timepoint_id="tp-001",
        patient_id="patient-001",
        scan_date=date(2025, 1, 15),
        modality="CT",
        therapy_status="on_treatment",
    )


# ---------------------------------------------------------------------------
# Storage fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def tmp_db(tmp_path) -> SQLiteBackend:
    """An SQLiteBackend backed by a temporary database file."""
    db_path = str(tmp_path / "test.db")
    backend = SQLiteBackend(db_path=db_path)
    yield backend
    backend.close()


@pytest.fixture()
def audit_logger(tmp_db, tmp_path) -> AuditLogger:
    """An AuditLogger with a temp database and JSONL output."""
    jsonl_path = str(tmp_path / "audit.jsonl")
    return AuditLogger(backend=tmp_db, jsonl_path=jsonl_path)
