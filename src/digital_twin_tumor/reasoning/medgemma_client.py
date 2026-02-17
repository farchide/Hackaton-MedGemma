"""MedGemma model client and template-based fallback.

Implements ADR-003 model integration. The :class:`MedGemmaClient` wraps
Google's MedGemma-4B-IT via HuggingFace ``transformers``. When GPU or model
access is unavailable, :class:`FallbackClient` produces deterministic
template-based summaries using the same structured input data.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from digital_twin_tumor.domain.models import (
    GrowthModelResult,
    Measurement,
    RECISTResponse,
    SimulationResult,
    TherapyEvent,
    UncertaintyEstimate,
)
from digital_twin_tumor.reasoning.prompt_templates import DISCLAIMER

# ---------------------------------------------------------------------------
# Conditional heavy imports
# ---------------------------------------------------------------------------

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    _TORCH_AVAILABLE = False

try:
    from transformers import AutoModelForCausalLM, AutoProcessor
    _TRANSFORMERS_AVAILABLE = True
except ImportError:  # pragma: no cover
    _TRANSFORMERS_AVAILABLE = False

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# MedGemmaClient
# ---------------------------------------------------------------------------


class MedGemmaClient:
    """Client for Google MedGemma-4B-IT via HuggingFace Transformers.

    Parameters
    ----------
    model_id:
        HuggingFace model identifier.
    device:
        Target device (``"auto"``, ``"cuda"``, ``"cpu"``).
    precision:
        Quantization / precision mode (``"auto"``, ``"bf16"``, ``"fp16"``,
        ``"int4"``).
    """

    def __init__(
        self,
        model_id: str = "google/medgemma-4b-it",
        device: str = "auto",
        precision: str = "auto",
    ) -> None:
        self.model_id = model_id
        self.device = device
        self.precision = precision
        self._model: Any = None
        self._processor: Any = None

    # -- lifecycle ----------------------------------------------------------

    def _load_model(self) -> None:
        """Load the model and processor from HuggingFace.

        Raises
        ------
        RuntimeError
            If ``torch`` or ``transformers`` are not installed.
        """
        if not _TORCH_AVAILABLE:
            raise RuntimeError(
                "PyTorch is required for MedGemmaClient. "
                "Install it with: pip install torch"
            )
        if not _TRANSFORMERS_AVAILABLE:
            raise RuntimeError(
                "HuggingFace Transformers is required for MedGemmaClient. "
                "Install it with: pip install transformers"
            )

        dtype = self._resolve_dtype()
        quantization_config = self._resolve_quantization()

        logger.info(
            "Loading MedGemma model %s (dtype=%s, device=%s)",
            self.model_id,
            dtype,
            self.device,
        )

        load_kwargs: dict[str, Any] = {
            "device_map": self.device if self.device != "cpu" else None,
        }
        if quantization_config is not None:
            load_kwargs["quantization_config"] = quantization_config
        elif dtype is not None:
            load_kwargs["torch_dtype"] = dtype

        self._processor = AutoProcessor.from_pretrained(self.model_id)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_id, **load_kwargs
        )

        # Move to device explicitly when not using device_map
        if self.device == "cpu":
            self._model = self._model.to("cpu")

        logger.info("MedGemma model loaded successfully.")

    def unload(self) -> None:
        """Explicitly free the model from GPU memory."""
        if self._model is not None:
            del self._model
            self._model = None
        if self._processor is not None:
            del self._processor
            self._processor = None
        if _TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("MedGemma model unloaded.")

    def is_loaded(self) -> bool:
        """Return ``True`` if the model is currently loaded."""
        return self._model is not None and self._processor is not None

    # -- generation ---------------------------------------------------------

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        images: list[np.ndarray] | None = None,
        max_new_tokens: int = 1024,
        temperature: float = 0.3,
        top_p: float = 0.9,
    ) -> str:
        """Generate text from the model.

        Parameters
        ----------
        system_prompt:
            System-level instruction.
        user_prompt:
            User-facing structured prompt.
        images:
            Optional list of numpy image arrays for multimodal input.
        max_new_tokens:
            Maximum number of tokens to generate.
        temperature:
            Sampling temperature (lower is more deterministic).
        top_p:
            Nucleus sampling threshold.

        Returns
        -------
        str
            Generated text response.
        """
        if not self.is_loaded():
            self._load_model()

        # Build chat messages in MedGemma format
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
        ]

        # Build user content (text + optional images)
        user_content: list[dict[str, Any]] = []
        if images:
            for img in images:
                user_content.append({"type": "image", "image": img})
        user_content.append({"type": "text", "text": user_prompt})

        messages.append({"role": "user", "content": user_content})

        # Process inputs
        inputs = self._processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )

        # Move inputs to model device
        if _TORCH_AVAILABLE and self._model is not None:
            model_device = next(self._model.parameters()).device
            inputs = {k: v.to(model_device) for k, v in inputs.items()}

        # Generate
        input_len = inputs["input_ids"].shape[-1]
        with torch.inference_mode():
            output = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=temperature > 0,
            )

        # Decode only new tokens
        new_tokens = output[0][input_len:]
        response = self._processor.decode(new_tokens, skip_special_tokens=True)
        return response.strip()

    # -- internal helpers ---------------------------------------------------

    def _resolve_dtype(self) -> Any:
        """Determine the torch dtype based on precision setting."""
        if not _TORCH_AVAILABLE:
            return None
        if self.precision == "bf16":
            return torch.bfloat16
        if self.precision == "fp16":
            return torch.float16
        if self.precision == "auto":
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
                return torch.bfloat16
            if torch.cuda.is_available():
                return torch.float16
            return torch.float32
        # int4 is handled via quantization config, not dtype
        return None

    def _resolve_quantization(self) -> Any:
        """Build quantization config for int4 mode, if applicable."""
        if self.precision != "int4":
            return None
        if not _TRANSFORMERS_AVAILABLE:
            return None
        try:
            from transformers import BitsAndBytesConfig

            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16 if _TORCH_AVAILABLE else None,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        except ImportError:
            logger.warning(
                "bitsandbytes not available; falling back to full precision."
            )
            return None


# ---------------------------------------------------------------------------
# FallbackClient
# ---------------------------------------------------------------------------


class FallbackClient:
    """Template-based fallback client when MedGemma is unavailable.

    Produces deterministic summaries by filling measurement values, growth
    model results, and RECIST categories into a fixed template. No ML model
    is required.

    Parameters
    ----------
    model_id:
        Ignored; accepted for interface compatibility.
    device:
        Ignored.
    precision:
        Ignored.
    """

    def __init__(
        self,
        model_id: str = "fallback",
        device: str = "cpu",
        precision: str = "none",
    ) -> None:
        self.model_id = model_id
        self.device = device
        self.precision = precision

    def _load_model(self) -> None:
        """No-op for the fallback client."""

    def unload(self) -> None:
        """No-op for the fallback client."""

    def is_loaded(self) -> bool:
        """The fallback client is always available."""
        return True

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        images: list[np.ndarray] | None = None,
        max_new_tokens: int = 1024,
        temperature: float = 0.3,
        top_p: float = 0.9,
    ) -> str:
        """Generate a template-based summary from the structured prompt.

        Parses the structured sections from the user prompt and fills in a
        fixed report template. No language model is invoked.

        Parameters
        ----------
        system_prompt:
            Ignored (safety is enforced in code).
        user_prompt:
            Structured prompt assembled by :func:`compose_user_prompt`.
        images:
            Ignored by the fallback client.
        max_new_tokens:
            Ignored.
        temperature:
            Ignored.
        top_p:
            Ignored.

        Returns
        -------
        str
            Template-based narrative summary.
        """
        # The user prompt is already well-structured markdown.  We wrap it in
        # a fixed report template so the output looks like a proper narrative.
        sections = _extract_sections(user_prompt)

        parts: list[str] = [
            "# Tumor Response Research Summary",
            "",
            "## Observed Measurements",
            sections.get("tumor measurement data", "No measurement data."),
            "",
            "## Therapy History",
            sections.get("therapy timeline", "No therapy data recorded."),
            "",
            "## Growth Model Analysis",
            sections.get("growth model analysis", "No growth model data."),
            "",
            "## Counterfactual Simulations",
            sections.get(
                "counterfactual simulations", "No simulation data."
            ),
            "",
            "## Measurement Uncertainty",
            sections.get(
                "measurement uncertainty", "No uncertainty data."
            ),
            "",
            "## RECIST 1.1 Response Assessment",
            sections.get(
                "recist 1.1 response assessment", "No RECIST data."
            ),
            "",
            "## Research Notes",
            "The above data has been compiled from automated and manual "
            "measurement pipelines. All numeric values are derived directly "
            "from the source measurements and model outputs. These findings "
            "require review and validation by qualified medical professionals "
            "before any clinical interpretation.",
            "",
            DISCLAIMER,
        ]
        return "\n".join(parts)


def _extract_sections(prompt: str) -> dict[str, str]:
    """Parse markdown ``## Heading`` sections from a structured prompt.

    Parameters
    ----------
    prompt:
        The full user prompt text.

    Returns
    -------
    dict[str, str]
        Mapping of lower-cased heading text to section body.
    """
    sections: dict[str, str] = {}
    current_heading: str | None = None
    buffer: list[str] = []

    for line in prompt.splitlines():
        if line.startswith("## "):
            # Save previous section
            if current_heading is not None:
                sections[current_heading] = "\n".join(buffer).strip()
            current_heading = line[3:].strip().lower()
            buffer = []
        else:
            buffer.append(line)

    # Save last section
    if current_heading is not None:
        sections[current_heading] = "\n".join(buffer).strip()

    return sections
