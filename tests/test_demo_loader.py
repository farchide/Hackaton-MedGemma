"""Tests for the demo data loader module."""
from __future__ import annotations

import json
import os
import tempfile

import numpy as np
import pytest

from digital_twin_tumor.data.demo_loader import DemoLoader, _recist_color
from digital_twin_tumor.domain.models import (
    GrowthModelResult,
    Lesion,
    Measurement,
    Patient,
    RECISTResponse,
    SimulationResult,
    TherapyEvent,
    TimePoint,
)
from digital_twin_tumor.storage.sqlite_backend import SQLiteBackend


@pytest.fixture
def demo_db(tmp_path):
    """Create a small demo database with one patient scenario."""
    db_path = str(tmp_path / "test_demo.db")
    db = SQLiteBackend(db_path)

    # Patient
    patient = Patient(
        patient_id="P001",
        metadata={
            "name": "Test Patient 1",
            "scenario": "Classic Responder",
            "cancer_type": "NSCLC",
            "stage": "IIIB",
            "age": 62,
            "sex": "M",
            "ECOG_PS": 1,
            "histology": "Adenocarcinoma",
        },
    )
    db.save_patient(patient)

    # Timepoints
    from datetime import date
    tps = [
        TimePoint(
            timepoint_id=f"TP{i}",
            patient_id="P001",
            scan_date=date(2025, 1 + i * 2, 6),
            modality="CT",
            therapy_status="pre" if i == 0 else "on",
            metadata={"week": i * 6, "scan_number": i + 1},
        )
        for i in range(3)
    ]
    for tp in tps:
        db.save_timepoint(tp)

    # Lesions per timepoint
    diameters = [35.0, 25.0, 15.0]
    for i, tp in enumerate(tps):
        d = diameters[i]
        vol = (4.0 / 3.0) * 3.14159 * (d / 2.0) ** 3
        lesion = Lesion(
            lesion_id=f"L001_tp{i}",
            timepoint_id=tp.timepoint_id,
            centroid=(5.0, -3.0, 2.0),
            volume_mm3=vol,
            longest_diameter_mm=d,
            short_axis_mm=d * 0.7,
            is_target=True,
            organ="lung_right",
            confidence=0.92,
        )
        db.save_lesion(lesion)

        meas = Measurement(
            measurement_id=f"M{i}",
            lesion_id=f"L001_tp{i}",
            timepoint_id=tp.timepoint_id,
            diameter_mm=d,
            volume_mm3=vol,
            method="auto",
            reviewer="AI_pipeline",
        )
        db.save_measurement(meas)

    # RECIST responses
    baseline_sum = 35.0
    for i, tp in enumerate(tps):
        d = diameters[i]
        pct = ((d - baseline_sum) / baseline_sum) * 100
        cat = "SD" if abs(pct) < 30 else "PR"
        r = RECISTResponse(
            timepoint_id=tp.timepoint_id,
            sum_of_diameters=d,
            baseline_sum=baseline_sum,
            nadir_sum=min(diameters[: i + 1]),
            percent_change_from_baseline=pct,
            percent_change_from_nadir=0.0,
            category=cat,
        )
        db.save_recist(r)

    # Therapy
    therapy = TherapyEvent(
        patient_id="P001",
        start_date=date(2025, 1, 20),
        end_date=date(2025, 7, 1),
        therapy_type="chemotherapy",
        dose="Carbo AUC5 + Pem 500",
        metadata={"drug_name": "Carboplatin + Pemetrexed"},
    )
    db.save_therapy_event(therapy)

    # Growth model
    gm = GrowthModelResult(
        model_name="exponential",
        parameters={"r": -0.05, "V0": 22000.0},
        aic=120.5,
        bic=122.3,
        akaike_weight=0.6,
    )
    db.save_growth_model("P001", gm)

    # Simulation result
    times = np.linspace(0, 30, 20)
    vols = 22000.0 * np.exp(-0.05 * times)
    sim = SimulationResult(
        scenario_name="Natural history",
        time_points=times,
        predicted_volumes=vols,
        lower_bound=vols * 0.85,
        upper_bound=vols * 1.15,
        parameters={"r": -0.05},
    )
    db.save_simulation_result("P001", sim)

    db.close()
    return db_path


@pytest.fixture
def loader(demo_db):
    """Create a DemoLoader from the test database."""
    dl = DemoLoader(demo_db)
    yield dl
    dl.close()


class TestDemoLoaderBasics:
    def test_list_patients(self, loader):
        patients = loader.list_patients()
        assert len(patients) == 1
        assert patients[0]["patient_id"] == "P001"
        assert "Classic Responder" in patients[0]["scenario"]

    def test_get_patient_choices(self, loader):
        choices = loader.get_patient_choices()
        assert len(choices) == 1
        assert "3 scans" in choices[0]

    def test_get_patient_ids(self, loader):
        ids = loader.get_patient_ids()
        assert ids == ["P001"]


class TestLoadPatient:
    def test_load_existing(self, loader):
        data = loader.load_patient("P001")
        assert data["patient"].patient_id == "P001"
        assert len(data["timepoints"]) == 3
        assert len(data["recist_responses"]) == 3
        assert len(data["therapy_events"]) == 1
        assert len(data["growth_models"]) >= 1
        assert len(data["simulation_results"]) >= 1

    def test_load_nonexistent(self, loader):
        data = loader.load_patient("NONEXISTENT")
        assert data == {}

    def test_lesions_per_timepoint(self, loader):
        data = loader.load_patient("P001")
        for tp in data["timepoints"]:
            lesions = data["lesions"][tp.timepoint_id]
            assert len(lesions) >= 1

    def test_measurements_per_timepoint(self, loader):
        data = loader.load_patient("P001")
        for tp in data["timepoints"]:
            meas = data["measurements"][tp.timepoint_id]
            assert len(meas) >= 1


class TestBuildUIState:
    def test_state_structure(self, loader):
        state = loader.build_ui_state("P001")
        assert "volume" in state
        assert "measurements" in state
        assert "recist_responses" in state
        assert "growth_results" in state
        assert "tracking_graph_json" in state
        assert "timepoints" in state
        assert "therapy_events" in state
        assert "patient_metadata" in state

    def test_measurements_flat(self, loader):
        state = loader.build_ui_state("P001")
        ms = state["measurements"]
        assert len(ms) == 3
        for m in ms:
            assert "lesion_id" in m
            assert "diameter_mm" in m
            assert "volume_mm3" in m
            assert "week" in m

    def test_recist_as_dicts(self, loader):
        state = loader.build_ui_state("P001")
        rs = state["recist_responses"]
        assert len(rs) == 3
        for r in rs:
            assert "category" in r
            assert "sum_of_diameters" in r
            assert "percent_change_from_baseline" in r

    def test_tracking_graph(self, loader):
        state = loader.build_ui_state("P001")
        gj = state["tracking_graph_json"]
        assert gj is not None
        gd = json.loads(gj)
        assert "nodes" in gd
        assert "edges" in gd
        assert len(gd["nodes"]) == 3
        assert len(gd["edges"]) == 2  # 3 observations, 2 edges

    def test_growth_results_have_weights(self, loader):
        state = loader.build_ui_state("P001")
        gm = state["growth_results"]
        assert len(gm) >= 1
        for g in gm:
            assert "weight" in g
            assert g["weight"] > 0

    def test_volume_generated(self, loader):
        state = loader.build_ui_state("P001")
        vol = state["volume"]
        assert vol is not None
        assert isinstance(vol, np.ndarray)
        assert vol.ndim == 3
        assert vol.shape == (64, 64, 32)

    def test_therapy_events(self, loader):
        state = loader.build_ui_state("P001")
        te = state["therapy_events"]
        assert len(te) == 1
        assert te[0]["therapy_type"] == "chemotherapy"
        assert "drug_name" in te[0]

    def test_empty_for_nonexistent(self, loader):
        state = loader.build_ui_state("NONEXISTENT")
        assert state == {}


class TestDashboardHTML:
    def test_returns_html(self, loader):
        html = loader.get_patient_dashboard_html("P001")
        assert "Classic Responder" in html
        assert "NSCLC" in html
        assert "chemotherapy" in html.lower()

    def test_nonexistent_patient(self, loader):
        html = loader.get_patient_dashboard_html("NONEXISTENT")
        assert "not found" in html.lower()


class TestTimelineHTML:
    def test_returns_timeline(self, loader):
        html = loader.get_timeline_html("P001")
        assert "Wk 0" in html
        assert "Wk 6" in html
        assert "Wk 12" in html

    def test_shows_recist_categories(self, loader):
        html = loader.get_timeline_html("P001")
        assert "SD" in html or "PR" in html

    def test_empty_for_nonexistent(self, loader):
        html = loader.get_timeline_html("NONEXISTENT")
        assert html == ""


class TestRecistColor:
    def test_known_categories(self):
        assert _recist_color("CR") == "#10b981"
        assert _recist_color("PR") == "#22d3ee"
        assert _recist_color("SD") == "#f59e0b"
        assert _recist_color("PD") == "#ef4444"

    def test_unknown_category(self):
        assert _recist_color("UNKNOWN") == "#94a3b8"

    def test_irecist_categories(self):
        assert _recist_color("iUPD") == "#f97316"
        assert _recist_color("iCPD") == "#ef4444"
