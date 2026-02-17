"""Tests for evaluation metrics: ForecastingBacktest, CalibrationMetrics,
MeasurementMetrics, RECISTAgreement, TrackingMetrics, and format_metrics_table.

Uses synthetic patient data so tests run fast with no database or model
loading.
"""
from __future__ import annotations

from datetime import date

import numpy as np
import pytest

from digital_twin_tumor.domain.models import (
    Lesion,
    Measurement,
    Patient,
    TimePoint,
)
from digital_twin_tumor.evaluation.metrics import (
    CalibrationMetrics,
    ForecastingBacktest,
    MeasurementMetrics,
    RECISTAgreement,
    TrackingMetrics,
    format_metrics_table,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def synthetic_patient_data():
    """Build a minimal patient data dict with known exponential growth."""
    weeks = [0, 6, 12, 18, 24]
    V0 = 1000.0
    r = 0.05
    volumes = [V0 * np.exp(r * w) for w in weeks]

    timepoints = [
        TimePoint(
            timepoint_id=f"TP{i}",
            patient_id="P001",
            scan_date=date(2025, 1 + i * 2, 1),
            modality="CT",
            therapy_status="pre" if i == 0 else "on",
            metadata={"week": weeks[i], "scan_number": i + 1},
        )
        for i in range(len(weeks))
    ]

    # One target lesion per timepoint
    lesions: dict[str, list[Lesion]] = {}
    measurements: dict[str, list[Measurement]] = {}
    for i, tp in enumerate(timepoints):
        d = (volumes[i] * 6 / np.pi) ** (1 / 3)  # sphere diameter
        les = Lesion(
            lesion_id=f"L001_tp{i}",
            timepoint_id=tp.timepoint_id,
            volume_mm3=volumes[i],
            longest_diameter_mm=d,
            is_target=True,
            organ="lung_right",
            confidence=0.92,
        )
        lesions[tp.timepoint_id] = [les]
        measurements[tp.timepoint_id] = [
            Measurement(
                measurement_id=f"M{i}",
                lesion_id=f"L001_tp{i}",
                timepoint_id=tp.timepoint_id,
                diameter_mm=d,
                volume_mm3=volumes[i],
                method="auto",
            ),
        ]

    return {
        "patient": Patient(patient_id="P001"),
        "timepoints": timepoints,
        "lesions": lesions,
        "measurements": measurements,
        "therapy_events": [],
    }


# ---------------------------------------------------------------------------
# ForecastingBacktest
# ---------------------------------------------------------------------------


class TestForecastingBacktest:
    def test_run_backtest_returns_dict(self, synthetic_patient_data):
        bt = ForecastingBacktest()
        result = bt.run_backtest(synthetic_patient_data)
        assert isinstance(result, dict)
        assert "mae_mm3" in result
        assert "best_model" in result

    def test_backtest_has_per_model_results(self, synthetic_patient_data):
        bt = ForecastingBacktest()
        result = bt.run_backtest(synthetic_patient_data)
        assert "per_model" in result
        assert len(result["per_model"]) >= 1

    def test_backtest_with_too_few_points(self):
        """Fewer than 3 timepoints should return skip_reason."""
        bt = ForecastingBacktest()
        tp = TimePoint(
            timepoint_id="TP0", patient_id="P001",
            metadata={"week": 0},
        )
        data = {
            "timepoints": [tp],
            "lesions": {
                "TP0": [Lesion(
                    lesion_id="L0", timepoint_id="TP0",
                    volume_mm3=100.0, is_target=True,
                )],
            },
        }
        result = bt.run_backtest(data)
        assert "skip_reason" in result


# ---------------------------------------------------------------------------
# CalibrationMetrics
# ---------------------------------------------------------------------------


class TestCalibrationMetrics:
    def test_coverage_all_covered(self):
        preds = np.array([10.0, 20.0, 30.0])
        actuals = np.array([10.0, 20.0, 30.0])
        lower = np.array([5.0, 15.0, 25.0])
        upper = np.array([15.0, 25.0, 35.0])
        cov = CalibrationMetrics.compute_coverage(preds, actuals, lower, upper)
        assert cov == 1.0

    def test_coverage_none_covered(self):
        preds = np.array([10.0, 20.0])
        actuals = np.array([100.0, 200.0])
        lower = np.array([5.0, 15.0])
        upper = np.array([15.0, 25.0])
        cov = CalibrationMetrics.compute_coverage(preds, actuals, lower, upper)
        assert cov == 0.0

    def test_coverage_empty(self):
        cov = CalibrationMetrics.compute_coverage(
            np.array([]), np.array([]), np.array([]), np.array([]),
        )
        assert cov == 0.0

    def test_ece_perfect_calibration(self):
        probs = np.array([0.1, 0.9, 0.1, 0.9])
        outcomes = np.array([0, 1, 0, 1])
        ece = CalibrationMetrics.compute_ece(probs, outcomes, n_bins=5)
        assert 0.0 <= ece <= 1.0


# ---------------------------------------------------------------------------
# MeasurementMetrics
# ---------------------------------------------------------------------------


class TestMeasurementMetrics:
    def test_diameter_mae_zero_for_identical(self):
        d = np.array([10.0, 20.0, 30.0])
        assert MeasurementMetrics.diameter_mae(d, d) == 0.0

    def test_diameter_mae_positive(self):
        pred = np.array([11.0, 22.0])
        true = np.array([10.0, 20.0])
        mae = MeasurementMetrics.diameter_mae(pred, true)
        assert mae == pytest.approx(1.5, abs=0.01)

    def test_volume_mae_zero_for_identical(self):
        v = np.array([1000.0, 2000.0])
        assert MeasurementMetrics.volume_mae(v, v) == 0.0

    def test_volume_mae_positive(self):
        pred = np.array([1100.0, 2200.0])
        true = np.array([1000.0, 2000.0])
        mae = MeasurementMetrics.volume_mae(pred, true)
        assert mae == pytest.approx(150.0, abs=0.01)

    def test_repeatability_coefficient(self):
        pairs = [(20.0, 20.5), (30.0, 29.8), (15.0, 15.3)]
        rc = MeasurementMetrics.repeatability_coefficient(pairs)
        assert rc > 0.0

    def test_repeatability_single_pair(self):
        """Single pair should return 0.0."""
        rc = MeasurementMetrics.repeatability_coefficient([(10.0, 10.5)])
        assert rc == 0.0


# ---------------------------------------------------------------------------
# RECISTAgreement
# ---------------------------------------------------------------------------


class TestRECISTAgreement:
    def test_perfect_agreement(self):
        cats = ["CR", "PR", "SD", "PD"]
        result = RECISTAgreement.compute_agreement(cats, cats)
        assert result["accuracy"] == 1.0
        assert result["cohens_kappa"] == 1.0

    def test_partial_agreement(self):
        pred = ["CR", "PR", "SD"]
        true = ["CR", "SD", "SD"]
        result = RECISTAgreement.compute_agreement(pred, true)
        assert 0.0 < result["accuracy"] < 1.0

    def test_confusion_matrix_structure(self):
        pred = ["CR", "PR"]
        true = ["CR", "CR"]
        result = RECISTAgreement.compute_agreement(pred, true)
        cm = result["confusion_matrix"]
        assert isinstance(cm, dict)
        assert "CR" in cm

    def test_empty_sequences(self):
        result = RECISTAgreement.compute_agreement([], [])
        assert result["accuracy"] == 0.0
        assert result["n"] == 0


# ---------------------------------------------------------------------------
# TrackingMetrics
# ---------------------------------------------------------------------------


class TestTrackingMetrics:
    def test_perfect_matching(self):
        pred = [("L1", "L2"), ("L3", "L4")]
        true = [("L1", "L2"), ("L3", "L4")]
        assert TrackingMetrics.match_accuracy(pred, true) == 1.0

    def test_no_matching(self):
        pred = [("L1", "L5")]
        true = [("L1", "L2")]
        assert TrackingMetrics.match_accuracy(pred, true) == 0.0

    def test_id_switches_zero(self):
        graph = {
            "nodes": [
                {"id": "n1", "lesion_id": "L1_tp0", "timepoint_id": "TP0"},
                {"id": "n2", "lesion_id": "L1_tp1", "timepoint_id": "TP1"},
            ],
            "edges": [
                {"source": "n1", "target": "n2", "confidence": 0.9},
            ],
        }
        assert TrackingMetrics.id_switches(graph) == 0

    def test_id_switches_detected(self):
        graph = {
            "nodes": [
                {"id": "n1", "lesion_id": "L1_tp0", "timepoint_id": "TP0"},
                {"id": "n2", "lesion_id": "L2_tp1", "timepoint_id": "TP1"},
            ],
            "edges": [
                {"source": "n1", "target": "n2", "confidence": 0.8},
            ],
        }
        assert TrackingMetrics.id_switches(graph) == 1


# ---------------------------------------------------------------------------
# format_metrics_table
# ---------------------------------------------------------------------------


class TestFormatMetricsTable:
    def test_returns_markdown_table(self):
        metrics = {
            "aggregate": {
                "mae_mm3_mean": 150.0,
                "mape_pct_mean": 12.5,
                "rmse_mm3_mean": 180.0,
                "coverage_80_mean": 0.75,
                "coverage_80_std": 0.05,
            },
            "patients": {"P001": {"tracking_id_switches": 0}},
        }
        table = format_metrics_table(metrics)
        assert isinstance(table, str)
        assert "Subsystem" in table
        assert "Growth Forecasting" in table
        assert "|" in table
