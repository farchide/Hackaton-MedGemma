"""Reasoning module for AI-powered narrative generation.

Exports the primary public interfaces:
  - :class:`NarrativeGenerator` -- orchestrates narrative generation
  - :class:`MedGemmaClient` -- HuggingFace MedGemma model client
  - :class:`FallbackClient` -- template-based fallback (no GPU required)
  - :class:`MedGemmaReasoner` -- high-level MedGemma reasoning layer
  - Prompt templates and context builders from :mod:`prompts`
"""

from __future__ import annotations

from digital_twin_tumor.reasoning.medgemma_client import (
    FallbackClient,
    MedGemmaClient,
)
from digital_twin_tumor.reasoning.medgemma import MedGemmaReasoner
from digital_twin_tumor.reasoning.narrative import NarrativeGenerator
from digital_twin_tumor.reasoning.prompts import (
    COUNTERFACTUAL_TEMPLATE,
    IMAGING_ANALYSIS_TEMPLATE,
    SAFETY_DISCLAIMER,
    SYSTEM_PROMPT as MEDGEMMA_SYSTEM_PROMPT,
    TUMOR_BOARD_TEMPLATE,
    build_growth_context,
    build_measurement_context,
    build_therapy_context,
)

__all__ = [
    "FallbackClient",
    "MedGemmaClient",
    "MedGemmaReasoner",
    "NarrativeGenerator",
    "COUNTERFACTUAL_TEMPLATE",
    "IMAGING_ANALYSIS_TEMPLATE",
    "MEDGEMMA_SYSTEM_PROMPT",
    "SAFETY_DISCLAIMER",
    "TUMOR_BOARD_TEMPLATE",
    "build_growth_context",
    "build_measurement_context",
    "build_therapy_context",
]
