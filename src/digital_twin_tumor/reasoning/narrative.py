"""Narrative generation orchestrator.

Coordinates prompt building, model invocation, and safety enforcement
to produce a :class:`NarrativeResult` from structured domain data.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from digital_twin_tumor.domain.models import (
    GrowthModelResult,
    Measurement,
    NarrativeResult,
    RECISTResponse,
    SimulationResult,
    TherapyEvent,
    UncertaintyEstimate,
)
from digital_twin_tumor.reasoning.medgemma_client import (
    FallbackClient,
    MedGemmaClient,
)
from digital_twin_tumor.reasoning.prompt_templates import (
    DISCLAIMER,
    SYSTEM_PROMPT,
    compose_user_prompt,
)
from digital_twin_tumor.reasoning.safety import sanitize_narrative

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _auto_detect_client() -> MedGemmaClient | FallbackClient:
    """Detect available backend and return the appropriate client.

    Tries to instantiate :class:`MedGemmaClient` (requires ``torch`` and
    ``transformers``).  Falls back to :class:`FallbackClient` when
    dependencies are missing or no GPU is available.

    Returns
    -------
    MedGemmaClient | FallbackClient
        The best available client.
    """
    try:
        import torch  # noqa: F811

        if not torch.cuda.is_available():
            logger.info(
                "No CUDA GPU detected; using FallbackClient for narrative "
                "generation."
            )
            return FallbackClient()

        from transformers import AutoModelForCausalLM  # noqa: F811

        logger.info("GPU and transformers available; using MedGemmaClient.")
        return MedGemmaClient()
    except ImportError:
        logger.info(
            "torch/transformers not installed; using FallbackClient."
        )
        return FallbackClient()


# ---------------------------------------------------------------------------
# NarrativeGenerator
# ---------------------------------------------------------------------------


class NarrativeGenerator:
    """Orchestrates narrative generation from domain data.

    Parameters
    ----------
    client:
        A :class:`MedGemmaClient` or :class:`FallbackClient`. When ``None``,
        the best available backend is auto-detected.
    """

    def __init__(
        self, client: MedGemmaClient | FallbackClient | None = None
    ) -> None:
        self.client: MedGemmaClient | FallbackClient = (
            client if client is not None else _auto_detect_client()
        )

    @staticmethod
    def extract_structured_output(
        narrative_text: str,
        measurements: list[Measurement],
        recist_responses: list[RECISTResponse],
        uncertainty: UncertaintyEstimate,
    ) -> dict[str, Any]:
        """Extract structured fields from narrative for machine consumption.

        Provides a structured summary alongside the free-text narrative
        containing key findings, uncertainty statement, and clinical context.

        Parameters
        ----------
        narrative_text:
            The generated narrative text.
        measurements:
            Source measurements.
        recist_responses:
            RECIST classifications.
        uncertainty:
            Measurement uncertainty estimate.

        Returns
        -------
        dict
            Keys: key_findings (list[str]), uncertainty_statement (str),
            recist_trajectory (str), trend (str), data_quality (str).
        """
        # Key findings from measurements
        key_findings: list[str] = []

        if measurements:
            latest = measurements[-1]
            key_findings.append(
                f"Latest measurement: diameter {latest.diameter_mm:.1f} mm, "
                f"volume {latest.volume_mm3:.1f} mm\u00b3"
            )

            if len(measurements) >= 2:
                first = measurements[0]
                vol_change = latest.volume_mm3 - first.volume_mm3
                if first.volume_mm3 > 0:
                    pct = (vol_change / first.volume_mm3) * 100
                    direction = "increased" if pct > 0 else "decreased"
                    key_findings.append(
                        f"Volume has {direction} by {abs(pct):.1f}% from first measurement"
                    )

        # RECIST trajectory
        recist_traj = "N/A"
        if recist_responses:
            categories = [r.category for r in recist_responses]
            recist_traj = " -> ".join(categories)
            key_findings.append(f"RECIST trajectory: {recist_traj}")

        # Trend detection
        trend = "stable"
        if len(measurements) >= 3:
            vols = [m.volume_mm3 for m in measurements[-3:]]
            if all(vols[i] < vols[i + 1] for i in range(len(vols) - 1)):
                trend = "increasing"
            elif all(vols[i] > vols[i + 1] for i in range(len(vols) - 1)):
                trend = "decreasing"

        # Uncertainty statement
        uncertainty_stmt = (
            f"Measurement uncertainty: {uncertainty.total_sigma:.2f} mm "
            f"(reliability: {uncertainty.reliability}). "
        )
        if uncertainty.reliability == "LOW":
            uncertainty_stmt += "Results should be interpreted with significant caution."
        elif uncertainty.reliability == "MEDIUM":
            uncertainty_stmt += "Results provide reasonable directional guidance."
        else:
            uncertainty_stmt += "Results are well-supported by available data."

        # Data quality
        n_pts = len(measurements)
        if n_pts >= 5:
            data_quality = "Good: sufficient data points for reliable analysis"
        elif n_pts >= 3:
            data_quality = "Moderate: minimum data for trend analysis"
        else:
            data_quality = "Limited: insufficient data for robust conclusions"

        return {
            "key_findings": key_findings,
            "uncertainty_statement": uncertainty_stmt,
            "recist_trajectory": recist_traj,
            "trend": trend,
            "data_quality": data_quality,
        }

    def generate_narrative(
        self,
        measurements: list[Measurement],
        therapy_events: list[TherapyEvent],
        growth_results: list[GrowthModelResult],
        simulations: list[SimulationResult],
        uncertainty: UncertaintyEstimate,
        recist_responses: list[RECISTResponse],
        evidence_images: list[np.ndarray] | None = None,
    ) -> NarrativeResult:
        """Generate a safety-checked narrative from structured input data.

        Steps:
          1. Build structured prompt from templates.
          2. Call the client's ``generate()`` method.
          3. Run the full safety pipeline (:func:`sanitize_narrative`).
          4. Package results into a :class:`NarrativeResult`.

        Parameters
        ----------
        measurements:
            Longitudinal tumor measurements.
        therapy_events:
            Therapy administration events.
        growth_results:
            Growth model fitting results.
        simulations:
            What-if simulation results.
        uncertainty:
            Combined measurement uncertainty estimate.
        recist_responses:
            RECIST 1.1 response classifications per time-point.
        evidence_images:
            Optional list of numpy image arrays for multimodal input.

        Returns
        -------
        NarrativeResult
            The generated narrative with grounding and safety metadata.
        """
        # Step 1: build structured prompt
        user_prompt = compose_user_prompt(
            measurements=measurements,
            therapy_events=therapy_events,
            growth_results=growth_results,
            simulations=simulations,
            uncertainty=uncertainty,
            recist_responses=recist_responses,
        )

        # Step 2: generate raw narrative
        generation_params: dict[str, Any] = {
            "model_id": self.client.model_id,
            "device": self.client.device,
            "client_type": type(self.client).__name__,
        }

        logger.info(
            "Generating narrative with %s", type(self.client).__name__
        )
        raw_narrative = self.client.generate(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt,
            images=evidence_images,
        )

        # Step 3: safety pipeline
        cleaned_text, grounding_ok, safety_ok = sanitize_narrative(
            narrative=raw_narrative,
            measurements=measurements,
            disclaimer=DISCLAIMER,
        )

        if not grounding_ok:
            logger.warning(
                "Narrative grounding check failed: no measurement values "
                "found in generated text."
            )
        if not safety_ok:
            logger.warning(
                "Narrative contained prohibited phrases that were replaced."
            )

        # Step 3.5: Log generation event
        try:
            from digital_twin_tumor.domain.events import EventBus, NARRATIVE_GENERATED
            bus = EventBus()
            bus.publish(NARRATIVE_GENERATED, {
                "client_type": type(self.client).__name__,
                "grounding_ok": grounding_ok,
                "safety_ok": safety_ok,
                "text_length": len(cleaned_text),
            })
        except Exception:
            logger.debug("Event publishing failed", exc_info=True)

        # Step 4: structured output extraction
        structured = self.extract_structured_output(
            cleaned_text, measurements, recist_responses, uncertainty,
        )

        # Step 5: package result
        return NarrativeResult(
            text=cleaned_text,
            disclaimer=DISCLAIMER,
            evidence_slices=[],
            grounding_check=grounding_ok,
            safety_check=safety_ok,
            generation_params={**generation_params, "structured_output": structured},
        )
