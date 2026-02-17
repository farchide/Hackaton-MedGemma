"""Tests for the MedGemma reasoning layer: model loading, prompt building,
template fallback, and safety disclaimers.

All tests mock heavy dependencies (torch, transformers) so they run fast
without a GPU.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from digital_twin_tumor.reasoning.prompts import (
    SAFETY_DISCLAIMER,
    SYSTEM_PROMPT,
    build_growth_context,
    build_measurement_context,
    build_therapy_context,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def sample_measurements():
    """List of measurement dicts as returned by DemoLoader.build_ui_state."""
    return [
        {
            "timepoint_id": "TP0",
            "diameter_mm": 35.0,
            "volume_mm3": 22449.0,
            "week": 0,
            "lesion_id": "L001",
        },
        {
            "timepoint_id": "TP1",
            "diameter_mm": 25.0,
            "volume_mm3": 8181.0,
            "week": 6,
            "lesion_id": "L001",
        },
    ]


@pytest.fixture()
def sample_therapy_events():
    return [
        {
            "therapy_type": "chemotherapy",
            "drug_name": "Carboplatin + Pemetrexed",
            "dose": "Carbo AUC5",
            "start_date": "2025-01-20",
            "end_date": "2025-07-01",
        },
    ]


@pytest.fixture()
def sample_growth_results():
    return [
        {"model_name": "exponential", "aic": 120.5, "weight": 0.6},
        {"model_name": "logistic", "aic": 122.3, "weight": 0.3},
    ]


@pytest.fixture()
def sample_recist_responses():
    return [
        {"category": "SD", "sum_of_diameters": 35.0, "percent_change_from_baseline": 0.0},
        {"category": "PR", "sum_of_diameters": 25.0, "percent_change_from_baseline": -28.6},
    ]


@pytest.fixture()
def sample_timepoints():
    return [
        {"week": 0, "scan_date": "2025-01-06", "therapy_status": "pre"},
        {"week": 6, "scan_date": "2025-03-06", "therapy_status": "on"},
    ]


# ---------------------------------------------------------------------------
# build_measurement_context
# ---------------------------------------------------------------------------


class TestBuildMeasurementContext:
    def test_returns_string(self, sample_measurements, sample_recist_responses, sample_timepoints):
        result = build_measurement_context(
            sample_measurements, sample_recist_responses, sample_timepoints,
        )
        assert isinstance(result, str)
        assert len(result) > 0

    def test_contains_header_row(self, sample_measurements, sample_recist_responses, sample_timepoints):
        result = build_measurement_context(
            sample_measurements, sample_recist_responses, sample_timepoints,
        )
        assert "Sum of Diameters" in result

    def test_empty_returns_fallback(self):
        result = build_measurement_context([], [], [])
        assert "No measurement data" in result

    def test_contains_recist_trajectory(self, sample_measurements, sample_recist_responses, sample_timepoints):
        result = build_measurement_context(
            sample_measurements, sample_recist_responses, sample_timepoints,
        )
        assert "RECIST Trajectory" in result
        assert "SD" in result


# ---------------------------------------------------------------------------
# build_therapy_context
# ---------------------------------------------------------------------------


class TestBuildTherapyContext:
    def test_returns_string_with_data(self, sample_therapy_events):
        result = build_therapy_context(sample_therapy_events)
        assert "Chemotherapy" in result
        assert "Carboplatin" in result

    def test_empty_returns_fallback(self):
        result = build_therapy_context([])
        assert "No therapy events" in result


# ---------------------------------------------------------------------------
# build_growth_context
# ---------------------------------------------------------------------------


class TestBuildGrowthContext:
    def test_returns_string_with_data(self, sample_growth_results):
        result = build_growth_context(sample_growth_results)
        assert "exponential" in result
        assert "0.600" in result

    def test_empty_returns_fallback(self):
        result = build_growth_context([])
        assert "No growth model" in result

    def test_best_model_noted(self, sample_growth_results):
        result = build_growth_context(sample_growth_results)
        assert "Best model" in result


# ---------------------------------------------------------------------------
# SYSTEM_PROMPT and SAFETY_DISCLAIMER constants
# ---------------------------------------------------------------------------


class TestPromptConstants:
    def test_system_prompt_not_empty(self):
        assert len(SYSTEM_PROMPT) > 0

    def test_system_prompt_mentions_research(self):
        assert "RESEARCH" in SYSTEM_PROMPT.upper()

    def test_safety_disclaimer_present(self):
        assert len(SAFETY_DISCLAIMER) > 0
        assert "DISCLAIMER" in SAFETY_DISCLAIMER

    def test_safety_disclaimer_mentions_not_clinical(self):
        assert "NOT" in SAFETY_DISCLAIMER
        assert "clinical" in SAFETY_DISCLAIMER.lower()


# ---------------------------------------------------------------------------
# MedGemmaReasoner -- mocked initialization
# ---------------------------------------------------------------------------


class TestMedGemmaReasonerInit:
    @patch("digital_twin_tumor.reasoning.medgemma._TORCH_AVAILABLE", False)
    def test_falls_back_without_torch(self):
        from digital_twin_tumor.reasoning.medgemma import MedGemmaReasoner
        reasoner = MedGemmaReasoner()
        assert reasoner.is_available is False
        assert "PyTorch" in (reasoner._load_error or "")

    @patch("digital_twin_tumor.reasoning.medgemma._TORCH_AVAILABLE", True)
    @patch("digital_twin_tumor.reasoning.medgemma._TRANSFORMERS_AVAILABLE", False)
    def test_falls_back_without_transformers(self):
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        with patch.dict("sys.modules", {"torch": mock_torch}):
            from digital_twin_tumor.reasoning.medgemma import MedGemmaReasoner
            reasoner = MedGemmaReasoner()
            assert reasoner.is_available is False
            assert "transformers" in (reasoner._load_error or "")


# ---------------------------------------------------------------------------
# Template fallback narrative generation
# ---------------------------------------------------------------------------


class TestTemplateFallback:
    @patch("digital_twin_tumor.reasoning.medgemma._TORCH_AVAILABLE", False)
    def test_tumor_board_fallback_includes_disclaimer(self):
        from digital_twin_tumor.reasoning.medgemma import MedGemmaReasoner
        reasoner = MedGemmaReasoner()
        result = reasoner.generate_tumor_board_summary({
            "patient_metadata": {"cancer_type": "NSCLC", "stage": "IIIB"},
            "measurements": [],
            "recist_responses": [],
            "growth_results": [],
            "therapy_events": [],
            "timepoints": [],
        })
        assert "DISCLAIMER" in result
        assert "template engine" in result.lower()

    @patch("digital_twin_tumor.reasoning.medgemma._TORCH_AVAILABLE", False)
    def test_tumor_board_fallback_includes_patient_info(self):
        from digital_twin_tumor.reasoning.medgemma import MedGemmaReasoner
        reasoner = MedGemmaReasoner()
        result = reasoner.generate_tumor_board_summary({
            "patient_metadata": {"cancer_type": "NSCLC", "stage": "IIIB"},
        })
        assert "NSCLC" in result
        assert "IIIB" in result

    @patch("digital_twin_tumor.reasoning.medgemma._TORCH_AVAILABLE", False)
    def test_counterfactual_fallback(self):
        from digital_twin_tumor.reasoning.medgemma import MedGemmaReasoner
        reasoner = MedGemmaReasoner()
        result = reasoner.generate_counterfactual_interpretation(
            baseline_trajectory={
                "name": "natural_history",
                "times": [0, 5, 10],
                "volumes": [100, 150, 200],
            },
            counterfactual_trajectories=[{
                "name": "early_treatment",
                "times": [0, 5, 10],
                "volumes": [100, 120, 130],
                "lower": [90, 110, 120],
                "upper": [110, 130, 140],
            }],
            therapy_info="Chemotherapy started week 2",
        )
        assert "DISCLAIMER" in result
        assert "natural_history" in result or "hypothetical" in result.lower()

    @patch("digital_twin_tumor.reasoning.medgemma._TORCH_AVAILABLE", False)
    def test_image_analysis_fallback(self):
        from digital_twin_tumor.reasoning.medgemma import MedGemmaReasoner
        reasoner = MedGemmaReasoner()
        result = reasoner.analyze_imaging_slice(
            image=np.zeros((64, 64), dtype=np.float32),
        )
        assert "unavailable" in result.lower() or "requires" in result.lower()
