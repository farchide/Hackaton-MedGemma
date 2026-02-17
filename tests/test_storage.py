"""Tests for storage: SQLite backend and audit logging."""

from __future__ import annotations

import json
import os
from datetime import date, datetime

import pytest

from digital_twin_tumor.domain.models import (
    AuditEvent,
    GrowthModelResult,
    Lesion,
    Measurement,
    Patient,
    RECISTResponse,
    TimePoint,
)
from digital_twin_tumor.storage.audit import AuditLogger
from digital_twin_tumor.storage.sqlite_backend import SQLiteBackend


# =====================================================================
# SQLiteBackend CRUD
# =====================================================================


class TestSQLiteBackendPatient:
    """Test patient CRUD operations."""

    def test_save_and_get_patient(self, tmp_db):
        patient = Patient(patient_id="P001", metadata={"age": 55})
        tmp_db.save_patient(patient)
        loaded = tmp_db.get_patient("P001")
        assert loaded is not None
        assert loaded.patient_id == "P001"
        assert loaded.metadata["age"] == 55

    def test_get_nonexistent_patient(self, tmp_db):
        loaded = tmp_db.get_patient("DOES_NOT_EXIST")
        assert loaded is None

    def test_upsert_patient(self, tmp_db):
        p1 = Patient(patient_id="P001", metadata={"version": 1})
        p2 = Patient(patient_id="P001", metadata={"version": 2})
        tmp_db.save_patient(p1)
        tmp_db.save_patient(p2)
        loaded = tmp_db.get_patient("P001")
        assert loaded.metadata["version"] == 2


class TestSQLiteBackendTimepoint:
    """Test timepoint persistence."""

    def test_save_and_get_timepoints(self, tmp_db):
        patient = Patient(patient_id="P001")
        tmp_db.save_patient(patient)

        tp = TimePoint(
            timepoint_id="TP1", patient_id="P001",
            scan_date=date(2025, 1, 15), modality="CT",
        )
        tmp_db.save_timepoint(tp)

        tps = tmp_db.get_timepoints("P001")
        assert len(tps) == 1
        assert tps[0].timepoint_id == "TP1"
        assert tps[0].scan_date == date(2025, 1, 15)
        assert tps[0].modality == "CT"


class TestSQLiteBackendMeasurement:
    """Test measurement persistence."""

    def test_save_and_get_measurements(self, tmp_db):
        # Set up prerequisite patient and timepoint
        patient = Patient(patient_id="P001")
        tmp_db.save_patient(patient)
        tp = TimePoint(timepoint_id="TP1", patient_id="P001")
        tmp_db.save_timepoint(tp)

        m = Measurement(
            measurement_id="M001", lesion_id="L001", timepoint_id="TP1",
            diameter_mm=20.0, volume_mm3=4189.0, method="auto",
            reviewer="system", timestamp=datetime(2025, 1, 15, 10, 0),
        )
        tmp_db.save_measurement(m)

        loaded = tmp_db.get_measurements("L001")
        assert len(loaded) == 1
        assert loaded[0].measurement_id == "M001"
        assert loaded[0].diameter_mm == 20.0
        assert loaded[0].volume_mm3 == 4189.0
        assert loaded[0].method == "auto"


class TestSQLiteBackendLesion:
    """Test lesion persistence."""

    def test_save_and_get_lesion(self, tmp_db):
        patient = Patient(patient_id="P001")
        tmp_db.save_patient(patient)
        tp = TimePoint(timepoint_id="TP1", patient_id="P001")
        tmp_db.save_timepoint(tp)

        lesion = Lesion(
            lesion_id="L001", timepoint_id="TP1",
            centroid=(10.0, 20.0, 30.0), volume_mm3=4189.0,
            longest_diameter_mm=20.0, short_axis_mm=18.0,
            is_target=True, organ="liver", confidence=0.95,
        )
        tmp_db.save_lesion(lesion)

        lesions = tmp_db.get_lesions("TP1")
        assert len(lesions) == 1
        assert lesions[0].lesion_id == "L001"
        assert lesions[0].centroid == (10.0, 20.0, 30.0)
        assert lesions[0].volume_mm3 == 4189.0
        assert lesions[0].is_target is True
        assert lesions[0].organ == "liver"


class TestSQLiteBackendRECIST:
    """Test RECIST response persistence."""

    def test_save_and_get_recist(self, tmp_db):
        patient = Patient(patient_id="P001")
        tmp_db.save_patient(patient)
        tp = TimePoint(timepoint_id="TP1", patient_id="P001")
        tmp_db.save_timepoint(tp)

        resp = RECISTResponse(
            timepoint_id="TP1", sum_of_diameters=30.0,
            baseline_sum=50.0, nadir_sum=30.0,
            percent_change_from_baseline=-40.0,
            percent_change_from_nadir=0.0, category="PR",
        )
        tmp_db.save_recist(resp)

        history = tmp_db.get_recist_history("P001")
        assert len(history) == 1
        assert history[0].category == "PR"
        assert history[0].sum_of_diameters == 30.0


class TestSQLiteBackendGrowthModel:
    """Test growth model persistence."""

    def test_save_growth_model(self, tmp_db):
        patient = Patient(patient_id="P001")
        tmp_db.save_patient(patient)

        result = GrowthModelResult(
            model_name="exponential",
            parameters={"V0": 100.0, "r": 0.05},
            aic=50.0, bic=55.0, akaike_weight=0.7,
        )
        # Should not raise
        tmp_db.save_growth_model("P001", result)


class TestSQLiteBackendAudit:
    """Test audit event persistence."""

    def test_save_and_get_audit_event(self, tmp_db):
        event = AuditEvent(
            event_id="EVT001", user_id="user-1",
            action_type="measurement_created",
            patient_id="P001", timepoint_id="TP1",
            lesion_id="L001",
            before_state={}, after_state={"diameter_mm": 20.0},
            metadata={"source": "test"},
        )
        tmp_db.save_audit_event(event)

        log = tmp_db.get_audit_log(patient_id="P001")
        assert len(log) == 1
        assert log[0].event_id == "EVT001"
        assert log[0].action_type == "measurement_created"
        assert log[0].after_state["diameter_mm"] == 20.0

    def test_get_audit_log_all(self, tmp_db):
        for i in range(5):
            event = AuditEvent(
                event_id=f"EVT{i:03d}", user_id="user-1",
                action_type="test_action",
            )
            tmp_db.save_audit_event(event)

        log = tmp_db.get_audit_log()
        assert len(log) == 5

    def test_get_audit_log_with_limit(self, tmp_db):
        for i in range(10):
            event = AuditEvent(
                event_id=f"EVT{i:03d}", user_id="user-1",
                action_type="test_action",
            )
            tmp_db.save_audit_event(event)

        log = tmp_db.get_audit_log(limit=3)
        assert len(log) == 3


class TestSQLiteBackendQuery:
    """Test generic SQL query interface."""

    def test_raw_query(self, tmp_db):
        patient = Patient(patient_id="P001")
        tmp_db.save_patient(patient)

        rows = tmp_db.query("SELECT * FROM patients WHERE patient_id = ?", ("P001",))
        assert len(rows) == 1
        assert rows[0]["patient_id"] == "P001"


# =====================================================================
# AuditLogger
# =====================================================================


class TestAuditLoggerLog:
    """Test core audit logging."""

    def test_log_creates_event_with_uuid(self, audit_logger):
        event = audit_logger.log(
            user_id="test-user",
            action_type="test_action",
            patient_id="P001",
        )
        assert isinstance(event, AuditEvent)
        assert len(event.event_id) > 0
        assert event.user_id == "test-user"
        assert event.action_type == "test_action"
        assert isinstance(event.timestamp, datetime)

    def test_log_persists_to_backend(self, audit_logger, tmp_db):
        audit_logger.log(
            user_id="test-user", action_type="persist_test",
            patient_id="P001",
        )
        log = tmp_db.get_audit_log(patient_id="P001")
        assert len(log) >= 1
        assert any(e.action_type == "persist_test" for e in log)

    def test_log_appends_to_jsonl(self, audit_logger, tmp_path):
        audit_logger.log(
            user_id="test-user", action_type="jsonl_test",
        )
        jsonl_path = str(tmp_path / "audit.jsonl")
        assert os.path.exists(jsonl_path)
        with open(jsonl_path) as f:
            lines = f.readlines()
        assert len(lines) >= 1
        parsed = json.loads(lines[0])
        assert parsed["action_type"] == "jsonl_test"


class TestAuditLoggerOverride:
    """Test human override audit logging."""

    def test_log_override_records_before_after(self, audit_logger):
        event = audit_logger.log_override(
            user_id="clinician-1",
            lesion_id="L001",
            field="diameter_mm",
            old_value=20.0,
            new_value=22.0,
            reason="Manual re-measurement",
        )
        assert event.action_type == "human_override"
        assert event.before_state == {"diameter_mm": 20.0}
        assert event.after_state == {"diameter_mm": 22.0}
        assert event.metadata["reason"] == "Manual re-measurement"


class TestAuditLoggerExportJsonl:
    """Test JSONL export."""

    def test_export_creates_valid_jsonl(self, audit_logger, tmp_path):
        # Create some audit events
        for i in range(3):
            audit_logger.log(
                user_id="user-1",
                action_type=f"action_{i}",
                patient_id="P001",
            )

        export_path = str(tmp_path / "export.jsonl")
        audit_logger.export_jsonl(export_path, patient_id="P001")

        assert os.path.exists(export_path)
        with open(export_path) as f:
            lines = f.readlines()
        assert len(lines) == 3

        # Validate each line is valid JSON
        for line in lines:
            parsed = json.loads(line.strip())
            assert "event_id" in parsed
            assert "timestamp" in parsed
            assert "action_type" in parsed

    def test_export_all_patients(self, audit_logger, tmp_path):
        audit_logger.log(user_id="u1", action_type="a1", patient_id="P001")
        audit_logger.log(user_id="u1", action_type="a2", patient_id="P002")

        export_path = str(tmp_path / "all_export.jsonl")
        audit_logger.export_jsonl(export_path, patient_id=None)
        assert os.path.exists(export_path)
        with open(export_path) as f:
            lines = f.readlines()
        assert len(lines) == 2
