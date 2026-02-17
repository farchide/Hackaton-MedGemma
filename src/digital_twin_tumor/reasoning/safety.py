"""Safety enforcement for AI-generated narrative text.

Implements ADR-003 (narrative safety) and ADR-011 (grounding validation).
All safety checks are performed in code -- they do not rely on prompt
engineering alone.
"""

from __future__ import annotations

import re

from digital_twin_tumor.domain.models import Measurement

# ---------------------------------------------------------------------------
# Prohibited phrases and their safe replacements
# ---------------------------------------------------------------------------

PROHIBITED_PHRASES: list[str] = [
    "I recommend",
    "I suggest prescribing",
    "you should take",
    "diagnosis is",
    "prognosis is",
    "treatment plan",
    "prescribe",
    "administer",
    "clinical recommendation",
]

_SAFE_REPLACEMENTS: dict[str, str] = {
    "I recommend": "The data suggests",
    "I suggest prescribing": "The data is consistent with",
    "you should take": "the following observation was noted",
    "diagnosis is": "the observed pattern is consistent with",
    "prognosis is": "the projected trajectory indicates",
    "treatment plan": "research observation",
    "prescribe": "note for clinical review",
    "administer": "evaluate in clinical context",
    "clinical recommendation": "research observation",
}

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def validate_grounding(narrative: str, measurements: list[Measurement]) -> bool:
    """Check that the narrative references at least one actual measurement value.

    The narrative is considered *grounded* if at least one numeric value from
    the measurement list (diameter or volume, rounded to 1 decimal) appears
    in the text.

    Parameters
    ----------
    narrative:
        The generated narrative text.
    measurements:
        Source measurements used to generate the narrative.

    Returns
    -------
    bool
        ``True`` if at least one measurement value is found in the text.
    """
    if not measurements:
        return False

    for m in measurements:
        # Check diameter (formatted to 1 decimal place)
        diameter_str = f"{m.diameter_mm:.1f}"
        if diameter_str in narrative:
            return True

        # Check volume (formatted to 1 decimal place)
        volume_str = f"{m.volume_mm3:.1f}"
        if volume_str in narrative:
            return True

        # Also check integer representations for whole numbers
        if m.diameter_mm == int(m.diameter_mm):
            if str(int(m.diameter_mm)) in narrative:
                return True
        if m.volume_mm3 == int(m.volume_mm3):
            if str(int(m.volume_mm3)) in narrative:
                return True

    return False


def enforce_safety(narrative: str) -> str:
    """Strip prohibited phrases from the narrative and replace with safe alternatives.

    Replacements are case-insensitive. If a prohibited phrase is found, it is
    replaced with the corresponding safe alternative from
    :data:`_SAFE_REPLACEMENTS`.

    Parameters
    ----------
    narrative:
        Raw narrative text, potentially containing prohibited phrases.

    Returns
    -------
    str
        Cleaned narrative with all prohibited phrases replaced.
    """
    cleaned = narrative
    for phrase in PROHIBITED_PHRASES:
        # Build a case-insensitive regex that preserves surrounding text.
        pattern = re.compile(re.escape(phrase), re.IGNORECASE)
        replacement = _SAFE_REPLACEMENTS.get(phrase, "research observation")
        cleaned = pattern.sub(replacement, cleaned)
    return cleaned


def ensure_disclaimer(narrative: str, disclaimer: str) -> str:
    """Append the disclaimer to the narrative if it is not already present.

    Parameters
    ----------
    narrative:
        The narrative text (possibly already containing the disclaimer).
    disclaimer:
        The disclaimer string to append.

    Returns
    -------
    str
        Narrative guaranteed to end with the disclaimer.
    """
    if disclaimer.strip() in narrative:
        return narrative
    return f"{narrative.rstrip()}\n\n{disclaimer}"


COUNTERFACTUAL_DISCLAIMER: str = (
    "IMPORTANT: Counterfactual simulations are hypothetical projections based "
    "on mathematical models. They represent 'what-if' scenarios and do NOT "
    "predict actual clinical outcomes. Treatment decisions must never be "
    "based on these simulations alone."
)


def ensure_counterfactual_disclaimer(narrative: str) -> str:
    """Append counterfactual disclaimer if simulation results are referenced.

    Checks for keywords indicating counterfactual content and appends
    the disclaimer if found and not already present.

    Parameters
    ----------
    narrative:
        The narrative text to check.

    Returns
    -------
    str
        Narrative with counterfactual disclaimer if applicable.
    """
    counterfactual_keywords = [
        "scenario", "counterfactual", "what-if", "simulation",
        "natural history", "treatment holiday", "resistance",
        "dose escalation", "earlier treatment", "later treatment",
    ]
    has_counterfactual = any(kw in narrative.lower() for kw in counterfactual_keywords)

    if has_counterfactual and COUNTERFACTUAL_DISCLAIMER not in narrative:
        return f"{narrative.rstrip()}\n\n{COUNTERFACTUAL_DISCLAIMER}"
    return narrative


_INJECTION_PATTERNS: list[str] = [
    "ignore previous instructions",
    "ignore all previous",
    "disregard the above",
    "forget everything",
    "new instructions:",
    "system prompt:",
    "you are now",
    "act as",
    "pretend you are",
    "jailbreak",
]


def detect_prompt_injection(text: str) -> bool:
    """Detect potential prompt injection attempts in input text.

    Parameters
    ----------
    text:
        User-provided text to scan.

    Returns
    -------
    bool
        True if injection patterns are detected.
    """
    lower = text.lower()
    return any(pattern in lower for pattern in _INJECTION_PATTERNS)


def sanitize_narrative(
    narrative: str,
    measurements: list[Measurement],
    disclaimer: str,
) -> tuple[str, bool, bool]:
    """Run the full safety pipeline on a narrative.

    Steps:
      1. :func:`enforce_safety` -- replace prohibited phrases.
      2. :func:`validate_grounding` -- verify data references.
      3. :func:`ensure_disclaimer` -- append disclaimer if missing.

    Parameters
    ----------
    narrative:
        Raw narrative text from the language model.
    measurements:
        Source measurements for grounding validation.
    disclaimer:
        The disclaimer to append.

    Returns
    -------
    tuple[str, bool, bool]
        A 3-tuple of ``(cleaned_text, grounding_ok, safety_ok)`` where
        *safety_ok* is ``True`` when no prohibited phrase remained after
        cleaning (i.e. the original text either had none, or they were
        successfully replaced).
    """
    # Step 1: enforce safety
    cleaned = enforce_safety(narrative)

    # Determine whether the *original* text was already safe (no prohibited
    # phrases present before cleaning).
    safety_ok = _is_safe(cleaned)

    # Step 2: validate grounding against cleaned text
    grounding_ok = validate_grounding(cleaned, measurements)

    # Step 3: append disclaimer
    final = ensure_disclaimer(cleaned, disclaimer)

    # Step 4: counterfactual disclaimer
    final = ensure_counterfactual_disclaimer(final)

    return final, grounding_ok, safety_ok


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _is_safe(text: str) -> bool:
    """Return ``True`` if the text contains none of the prohibited phrases."""
    lower = text.lower()
    for phrase in PROHIBITED_PHRASES:
        if phrase.lower() in lower:
            return False
    return True
