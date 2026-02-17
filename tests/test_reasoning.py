"""Tests for reasoning: prompt templates, safety enforcement, narrative generation."""

from __future__ import annotations

from datetime import date, datetime

import numpy as np
import pytest

from digital_twin_tumor.domain.models import (
    GrowthModelResult,
    Measurement,
    NarrativeResult,
    RECISTResponse,
    SimulationResult,
    TherapyEvent,
    UncertaintyEstimate,
)
from digital_twin_tumor.reasoning.medgemma_client import FallbackClient
from digital_twin_tumor.reasoning.narrative import NarrativeGenerator
from digital_twin_tumor.reasoning.prompt_templates import (
    DISCLAIMER,
    SYSTEM_PROMPT,
    build_counterfactual_summary,
    build_growth_model_summary,
    build_measurement_table,
    build_recist_summary,
    build_therapy_timeline,
    build_uncertainty_summary,
    compose_user_prompt,
)
from digital_twin_tumor.reasoning.safety import (
    PROHIBITED_PHRASES,
    enforce_safety,
    ensure_disclaimer,
    sanitize_narrative,
    validate_grounding,
)


# =====================================================================
# Prompt template builders
# =====================================================================


class TestPromptTemplateBuilders:
    """Test that all prompt template builders produce non-empty strings."""

    def test_build_measurement_table_with_data(self, sample_measurement):
        result = build_measurement_table([sample_measurement])
        assert isinstance(result, str)
        assert len(result) > 0
        assert "Timepoint" in result
        assert "20.0" in result

    def test_build_measurement_table_empty(self):
        result = build_measurement_table([])
        assert "No measurement data" in result

    def test_build_therapy_timeline_with_data(self, sample_therapy_event):
        result = build_therapy_timeline([sample_therapy_event])
        assert isinstance(result, str)
        assert len(result) > 0
        assert "chemotherapy" in result

    def test_build_therapy_timeline_empty(self):
        result = build_therapy_timeline([])
        assert "No therapy events" in result

    def test_build_uncertainty_summary(self):
        est = UncertaintyEstimate(
            sigma_manual=1.5, sigma_auto=0.5, sigma_scan=0.3,
            total_sigma=1.63, reliability="HIGH",
        )
        result = build_uncertainty_summary(est)
        assert "1.63" in result
        assert "HIGH" in result

    def test_build_growth_model_summary_with_data(self):
        results = [
            GrowthModelResult(
                model_name="exponential", aic=100.0, bic=105.0,
                akaike_weight=0.7, parameters={"V0": 100.0, "r": 0.05},
            ),
        ]
        text = build_growth_model_summary(results)
        assert "exponential" in text
        assert len(text) > 0

    def test_build_growth_model_summary_empty(self):
        text = build_growth_model_summary([])
        assert "No growth model" in text

    def test_build_counterfactual_summary_with_data(self):
        sim = SimulationResult(
            scenario_name="natural_history",
            time_points=np.array([0.0, 5.0, 10.0]),
            predicted_volumes=np.array([100.0, 150.0, 200.0]),
            parameters={"V0": 100.0},
        )
        text = build_counterfactual_summary([sim])
        assert "natural_history" in text
        assert "200.0" in text

    def test_build_counterfactual_summary_empty(self):
        text = build_counterfactual_summary([])
        assert "No simulation" in text

    def test_build_recist_summary_with_data(self):
        resp = RECISTResponse(
            category="PR", sum_of_diameters=30.0,
            baseline_sum=50.0, nadir_sum=30.0,
            percent_change_from_baseline=-40.0,
            percent_change_from_nadir=0.0,
        )
        text = build_recist_summary([resp])
        assert "PR" in text
        assert "30.0" in text

    def test_build_recist_summary_empty(self):
        text = build_recist_summary([])
        assert "No RECIST" in text

    def test_system_prompt_not_empty(self):
        assert len(SYSTEM_PROMPT) > 0
        assert "research" in SYSTEM_PROMPT.lower()

    def test_disclaimer_not_empty(self):
        assert len(DISCLAIMER) > 0
        assert "DISCLAIMER" in DISCLAIMER


# =====================================================================
# compose_user_prompt
# =====================================================================


class TestComposeUserPrompt:
    """Test full prompt composition."""

    def test_includes_all_sections(self, sample_measurement, sample_therapy_event):
        uncertainty = UncertaintyEstimate(
            sigma_manual=1.5, sigma_auto=0.5, sigma_scan=0.3,
            total_sigma=1.63, reliability="HIGH",
        )
        prompt = compose_user_prompt(
            measurements=[sample_measurement],
            therapy_events=[sample_therapy_event],
            growth_results=[],
            simulations=[],
            uncertainty=uncertainty,
            recist_responses=[],
        )
        assert "## Tumor Measurement Data" in prompt
        assert "## Therapy Timeline" in prompt
        assert "## Growth Model Analysis" in prompt
        assert "## Counterfactual Simulations" in prompt
        assert "## Measurement Uncertainty" in prompt
        assert "## RECIST 1.1 Response Assessment" in prompt
        # Should include the task instruction
        assert "clinical validation" in prompt.lower()


# =====================================================================
# Safety: enforce_safety
# =====================================================================


class TestEnforceSafety:
    """Test prohibited phrase removal."""

    def test_removes_i_recommend(self):
        text = "Based on the data, I recommend further imaging."
        cleaned = enforce_safety(text)
        assert "I recommend" not in cleaned
        assert "The data suggests" in cleaned

    def test_removes_prescribe(self):
        text = "The oncologist should prescribe this medication."
        cleaned = enforce_safety(text)
        assert "prescribe" not in cleaned.lower()

    def test_case_insensitive(self):
        text = "I RECOMMEND doing something."
        cleaned = enforce_safety(text)
        assert "recommend" not in cleaned.lower()

    def test_clean_text_unchanged(self):
        text = "The observed trajectory shows stable disease."
        cleaned = enforce_safety(text)
        assert cleaned == text

    def test_multiple_prohibited_phrases(self):
        text = "I recommend a treatment plan with this diagnosis is confirmed."
        cleaned = enforce_safety(text)
        assert "I recommend" not in cleaned
        assert "treatment plan" not in cleaned
        assert "diagnosis is" not in cleaned


# =====================================================================
# Safety: validate_grounding
# =====================================================================


class TestValidateGrounding:
    """Test grounding validation against measurement data."""

    def test_grounded_with_diameter(self, sample_measurement):
        narrative = "The tumor measures 20.0 mm in longest diameter."
        assert validate_grounding(narrative, [sample_measurement]) is True

    def test_grounded_with_volume(self, sample_measurement):
        narrative = "The volume was calculated as 4189.0 mm3."
        assert validate_grounding(narrative, [sample_measurement]) is True

    def test_ungrounded_text(self, sample_measurement):
        narrative = "The tumor appears to be growing."
        assert validate_grounding(narrative, [sample_measurement]) is False

    def test_empty_measurements(self):
        assert validate_grounding("Any text", []) is False

    def test_integer_value_grounding(self):
        m = Measurement(diameter_mm=20.0, volume_mm3=4000.0)
        narrative = "The diameter is 20 mm."
        assert validate_grounding(narrative, [m]) is True


# =====================================================================
# Safety: ensure_disclaimer
# =====================================================================


class TestEnsureDisclaimer:
    """Test disclaimer appending."""

    def test_appends_when_missing(self):
        text = "This is a narrative."
        result = ensure_disclaimer(text, DISCLAIMER)
        assert result.endswith(DISCLAIMER)

    def test_does_not_duplicate(self):
        text = f"This is a narrative.\n\n{DISCLAIMER}"
        result = ensure_disclaimer(text, DISCLAIMER)
        # Should appear exactly once
        assert result.count(DISCLAIMER.strip()) == 1


# =====================================================================
# FallbackClient
# =====================================================================


class TestFallbackClient:
    """Test template-based fallback client."""

    def test_generate_produces_output(self):
        client = FallbackClient()
        output = client.generate(
            system_prompt="system",
            user_prompt="## Tumor Measurement Data\nSome data here.\n## Therapy Timeline\nSome therapy.\n",
        )
        assert isinstance(output, str)
        assert len(output) > 0

    def test_generate_includes_disclaimer(self):
        client = FallbackClient()
        output = client.generate(
            system_prompt="system",
            user_prompt="## Tumor Measurement Data\nData.\n## Therapy Timeline\nTherapy.\n",
        )
        assert "DISCLAIMER" in output

    def test_is_loaded_always_true(self):
        client = FallbackClient()
        assert client.is_loaded() is True

    def test_model_id(self):
        client = FallbackClient()
        assert client.model_id == "fallback"


# =====================================================================
# NarrativeGenerator (end-to-end with FallbackClient)
# =====================================================================


class TestNarrativeGenerator:
    """Test end-to-end narrative generation with FallbackClient."""

    def test_generate_narrative_returns_result(self, sample_measurement, sample_therapy_event):
        client = FallbackClient()
        gen = NarrativeGenerator(client=client)
        uncertainty = UncertaintyEstimate(
            sigma_manual=1.5, sigma_auto=0.5, sigma_scan=0.3,
            total_sigma=1.63, reliability="HIGH",
        )
        result = gen.generate_narrative(
            measurements=[sample_measurement],
            therapy_events=[sample_therapy_event],
            growth_results=[],
            simulations=[],
            uncertainty=uncertainty,
            recist_responses=[],
        )
        assert isinstance(result, NarrativeResult)
        assert len(result.text) > 0
        assert result.disclaimer == DISCLAIMER

    def test_safety_check_passed(self, sample_measurement, sample_therapy_event):
        """FallbackClient should not produce prohibited phrases."""
        client = FallbackClient()
        gen = NarrativeGenerator(client=client)
        uncertainty = UncertaintyEstimate()
        result = gen.generate_narrative(
            measurements=[sample_measurement],
            therapy_events=[sample_therapy_event],
            growth_results=[],
            simulations=[],
            uncertainty=uncertainty,
            recist_responses=[],
        )
        assert result.safety_check is True

    def test_narrative_text_contains_disclaimer(self, sample_measurement, sample_therapy_event):
        client = FallbackClient()
        gen = NarrativeGenerator(client=client)
        uncertainty = UncertaintyEstimate()
        result = gen.generate_narrative(
            measurements=[sample_measurement],
            therapy_events=[sample_therapy_event],
            growth_results=[],
            simulations=[],
            uncertainty=uncertainty,
            recist_responses=[],
        )
        assert "DISCLAIMER" in result.text
