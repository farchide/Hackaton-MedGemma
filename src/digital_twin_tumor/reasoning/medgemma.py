"""MedGemma reasoning layer for Digital Twin Tumor Response Assessment.

Provides :class:`MedGemmaReasoner` wrapping Google MedGemma-4B-IT for
tumor board summaries, imaging analysis, and counterfactual interpretation.
Falls back to template-based narratives when the model is unavailable.

MedGemma is part of Google's Health AI Developer Foundations (HAI-DEF).
Official repository and notebooks: https://github.com/google-health/medgemma

Supported models:
- google/medgemma-4b-it: 4B multimodal (Gemma 3 + MedSigLIP 400M image encoder)
- google/medgemma-27b-text-it: 27B text-only for complex clinical reasoning
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from digital_twin_tumor.reasoning.prompts import (
    COUNTERFACTUAL_TEMPLATE,
    IMAGING_ANALYSIS_TEMPLATE,
    SAFETY_DISCLAIMER,
    SYSTEM_PROMPT,
    TUMOR_BOARD_TEMPLATE,
    build_growth_context,
    build_measurement_context,
    build_therapy_context,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Supported MedGemma model variants
# ---------------------------------------------------------------------------
SUPPORTED_MODELS: dict[str, dict[str, str]] = {
    "google/medgemma-4b-it": {
        "parameters": "4B",
        "type": "multimodal",
        "architecture": "Gemma 3 + MedSigLIP (400M image encoder)",
        "medqa_score": "64.4%",
        "description": "Multimodal model for imaging analysis and text reasoning",
        "image_encoder": "MedSigLIP (400M)",
    },
    "google/medgemma-27b-text-it": {
        "parameters": "27B",
        "type": "text-only",
        "architecture": "Gemma 3",
        "medqa_score": "87.7%",
        "description": "Text-only model for complex clinical reasoning and synthesis",
        "image_encoder": "N/A",
    },
}

# Conditional heavy imports
_TORCH_AVAILABLE = False
_TRANSFORMERS_AVAILABLE = False
_BNB_AVAILABLE = False

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    pass
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    _TRANSFORMERS_AVAILABLE = True
except ImportError:
    pass
try:
    import bitsandbytes  # noqa: F401
    _BNB_AVAILABLE = True
except ImportError:
    pass

# RECIST category interpretation map
_RECIST_NOTES: dict[str, str] = {
    "CR": "The data shows a complete response pattern with no measurable disease.",
    "PR": "The data shows a partial response pattern with measurable tumor reduction.",
    "SD": "The data shows stable disease with no significant change in tumor burden.",
    "PD": "The data shows progressive disease with increasing tumor burden.",
}


class MedGemmaReasoner:
    """Reasoning layer powered by Google MedGemma-4B-IT with template fallback.

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier.
    device : str
        Target device (``"auto"``, ``"cuda"``, ``"cpu"``).
    quantize : bool
        Whether to use 4-bit quantization via bitsandbytes.
    """

    def __init__(
        self,
        model_name: str = "google/medgemma-4b-it",
        device: str = "auto",
        quantize: bool = True,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.quantize = quantize
        self._model: Any = None
        self._tokenizer: Any = None
        self._loaded = False
        self._load_error: str | None = None
        # NOTE: _try_load_model() is NOT called here -- loading is lazy
        # so that ZeroGPU (HF Spaces) can allocate GPU inside @spaces.GPU.

    # -- lifecycle ----------------------------------------------------------

    def _ensure_loaded(self) -> None:
        """Lazily trigger model loading on first use (ZeroGPU compatible)."""
        if not self._loaded and self._load_error is None:
            self._try_load_model()

    def _try_load_model(self) -> None:
        """Attempt to load the MedGemma model with graceful fallback."""
        if not _TORCH_AVAILABLE:
            self._load_error = "PyTorch not installed"
            logger.warning("MedGemma: PyTorch not available. Using template fallback.")
            return
        if not _TRANSFORMERS_AVAILABLE:
            self._load_error = "transformers library not installed"
            logger.warning("MedGemma: transformers not available. Using template fallback.")
            return

        has_cuda = torch.cuda.is_available()
        if not has_cuda and self.device == "auto":
            self._load_error = "No CUDA GPU detected"
            logger.info("MedGemma: No GPU detected. Using template fallback.")
            return

        try:
            logger.info("Loading MedGemma: %s (device=%s, quantize=%s)",
                        self.model_name, self.device, self.quantize)
            load_kwargs: dict[str, Any] = {}
            if self.quantize and _BNB_AVAILABLE and has_cuda:
                load_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4",
                )
                logger.info("Using 4-bit NF4 quantization via bitsandbytes.")
            elif has_cuda:
                load_kwargs["torch_dtype"] = torch.bfloat16
            if self.device != "cpu":
                load_kwargs["device_map"] = self.device

            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_name, **load_kwargs)
            if self.device == "cpu":
                self._model = self._model.to("cpu")
            self._loaded = True
            logger.info("MedGemma model loaded successfully.")
        except Exception as exc:
            self._load_error = str(exc)
            self._model = None
            self._tokenizer = None
            self._loaded = False
            logger.warning("MedGemma load failed: %s. Using template fallback.", exc)

    @property
    def is_available(self) -> bool:
        """Return True if the MedGemma model is loaded and ready."""
        return self._loaded and self._model is not None

    def get_model_info(self) -> dict[str, Any]:
        """Return metadata about the currently configured MedGemma model.

        Returns
        -------
        dict
            Model metadata including variant info, availability status,
            and capabilities. Useful for agentic workflows that need to
            inspect tool capabilities before invocation.
        """
        model_meta = SUPPORTED_MODELS.get(self.model_name, {})
        return {
            "model_name": self.model_name,
            "is_available": self.is_available,
            "load_error": self._load_error,
            "quantized": self.quantize,
            "device": self.device,
            "parameters": model_meta.get("parameters", "unknown"),
            "type": model_meta.get("type", "unknown"),
            "architecture": model_meta.get("architecture", "unknown"),
            "medqa_score": model_meta.get("medqa_score", "unknown"),
            "image_encoder": model_meta.get("image_encoder", "N/A"),
            "supported_models": list(SUPPORTED_MODELS.keys()),
            "repository": "https://github.com/google-health/medgemma",
        }

    def _generate_text(self, system_prompt: str, user_prompt: str,
                       max_new_tokens: int = 1024, temperature: float = 0.3) -> str:
        """Run model inference with chat template format."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        inputs = self._tokenizer.apply_chat_template(
            messages, add_generation_prompt=True,
            tokenize=True, return_dict=True, return_tensors="pt",
        )
        model_device = next(self._model.parameters()).device
        inputs = {k: v.to(model_device) for k, v in inputs.items()}
        input_len = inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            output = self._model.generate(
                **inputs, max_new_tokens=max_new_tokens,
                temperature=temperature, do_sample=True, top_p=0.9,
            )
        new_tokens = output[0][input_len:]
        return self._tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    # -- public API ---------------------------------------------------------

    def generate_tumor_board_summary(self, patient_data: dict[str, Any]) -> str:
        """Generate a tumor board summary from patient state data.

        Falls back to template-based summary when MedGemma is unavailable.
        """
        self._ensure_loaded()
        if not self.is_available:
            return self._template_tumor_board_summary(patient_data)

        user_prompt = TUMOR_BOARD_TEMPLATE.format(
            patient_context=self._format_patient_context(
                patient_data.get("patient_metadata", {})),
            measurement_context=build_measurement_context(
                patient_data.get("measurements", []),
                patient_data.get("recist_responses", []),
                patient_data.get("timepoints", [])),
            therapy_context=build_therapy_context(
                patient_data.get("therapy_events", [])),
            growth_context=build_growth_context(
                patient_data.get("growth_results", [])),
            simulation_context=self._format_simulation_context(
                patient_data.get("simulation_results_full", [])),
        )
        try:
            response = self._generate_text(SYSTEM_PROMPT, user_prompt)
            return f"{response}\n\n---\n\n{SAFETY_DISCLAIMER}"
        except Exception as exc:
            logger.warning("MedGemma generation failed: %s. Using template.", exc)
            return self._template_tumor_board_summary(patient_data)

    def analyze_imaging_slice(self, image: np.ndarray, context: str = "") -> str:
        """Analyze a 2D imaging slice using MedGemma multimodal capability.

        Uses the MedSigLIP (400M) image encoder within MedGemma-4B to extract
        clinically meaningful visual features from CT/MRI slices before the
        language model generates its analysis. MedSigLIP is specifically trained
        on medical imaging data, providing superior feature extraction compared
        to general-purpose vision encoders.

        Note: MedGemma 1.5 expands this to support 3D CT/MRI volumes and
        longitudinal imaging comparison across timepoints.
        """
        self._ensure_loaded()
        if not self.is_available:
            err = f" ({self._load_error})" if self._load_error else ""
            return (
                f"Image analysis requires MedGemma model loaded on GPU. "
                f"Currently unavailable{err}. Ensure a CUDA-capable GPU "
                f"and MedGemma model weights are accessible."
            )
        ctx = f"**Clinical context:** {context}\n\n" if context else ""
        user_prompt = IMAGING_ANALYSIS_TEMPLATE.format(context=ctx)
        try:
            messages: list[dict[str, Any]] = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": user_prompt},
                ]},
            ]
            inputs = self._tokenizer.apply_chat_template(
                messages, add_generation_prompt=True,
                tokenize=True, return_dict=True, return_tensors="pt",
            )
            model_device = next(self._model.parameters()).device
            inputs = {k: v.to(model_device) for k, v in inputs.items()}
            input_len = inputs["input_ids"].shape[-1]
            with torch.inference_mode():
                output = self._model.generate(
                    **inputs, max_new_tokens=512, temperature=0.3, do_sample=True)
            new_tokens = output[0][input_len:]
            resp = self._tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            return f"{resp}\n\n{SAFETY_DISCLAIMER}"
        except Exception as exc:
            logger.warning("Image analysis failed: %s", exc)
            return f"Image analysis error: {exc}. Verify image format and model."

    def generate_counterfactual_interpretation(
        self, baseline_trajectory: dict[str, Any],
        counterfactual_trajectories: list[dict[str, Any]],
        therapy_info: dict[str, Any] | str,
    ) -> str:
        """Interpret counterfactual simulation results in clinical language."""
        self._ensure_loaded()
        baseline_summary = _format_trajectory(baseline_trajectory)
        cf_summaries = "\n\n".join(
            _format_trajectory(cf) for cf in counterfactual_trajectories)
        therapy_text = (
            therapy_info if isinstance(therapy_info, str)
            else build_therapy_context(
                therapy_info.get("events", []) if isinstance(therapy_info, dict)
                else [])
        )
        if not self.is_available:
            return self._template_counterfactual(
                baseline_trajectory, counterfactual_trajectories, therapy_text)
        try:
            user_prompt = COUNTERFACTUAL_TEMPLATE.format(
                baseline_summary=baseline_summary,
                counterfactual_summaries=cf_summaries, therapy_info=therapy_text)
            response = self._generate_text(SYSTEM_PROMPT, user_prompt)
            return f"{response}\n\n---\n\n{SAFETY_DISCLAIMER}"
        except Exception as exc:
            logger.warning("Counterfactual interpretation failed: %s", exc)
            return self._template_counterfactual(
                baseline_trajectory, counterfactual_trajectories, therapy_text)

    # -- template fallbacks -------------------------------------------------

    def _template_tumor_board_summary(self, patient_data: dict[str, Any]) -> str:
        """Template-based tumor board summary using actual measurement data."""
        meta = patient_data.get("patient_metadata", {})
        ms = patient_data.get("measurements", [])
        rs = patient_data.get("recist_responses", [])
        gm = patient_data.get("growth_results", [])
        evts = patient_data.get("therapy_events", [])
        tps = patient_data.get("timepoints", [])

        s: list[str] = [
            "# Tumor Board Research Summary", "",
            "> Generated by template engine (MedGemma model not available). "
            "This is a research summary, NOT a clinical assessment.", "",
        ]
        # Patient profile
        if meta:
            s.extend(["## Clinical Summary", ""])
            for key, label in [("cancer_type", "Cancer Type"), ("stage", "Stage"),
                               ("age", "Age"), ("sex", "Sex"),
                               ("histology", "Histology"), ("ECOG_PS", "ECOG PS"),
                               ("scenario", "Scenario")]:
                s.append(f"- **{label}:** {meta.get(key, 'N/A')}")
            s.append("")
        # Treatment
        if evts:
            s.extend(["## Treatment History", ""])
            for ev in evts:
                drug = ev.get("drug_name", ev.get("dose", "N/A"))
                s.append(f"- **{ev.get('therapy_type', '?').title()}:** "
                         f"{drug} ({ev.get('start_date', '?')} to "
                         f"{ev.get('end_date', 'ongoing')})")
            s.append("")
        # Response table
        if tps and rs:
            s.extend(["## Response Assessment", "",
                       "| Week | RECIST | Sum (mm) | Change (%) | Status |",
                       "|------|--------|----------|------------|--------|"])
            for i, tp in enumerate(tps):
                if i < len(rs):
                    r = rs[i]
                    s.append(f"| {tp.get('week', '?')} | **{r.get('category', 'N/A')}** "
                             f"| {float(r.get('sum_of_diameters', 0)):.1f} "
                             f"| {float(r.get('percent_change_from_baseline', 0)):+.1f}% "
                             f"| {tp.get('therapy_status', 'N/A')} |")
            s.append("")
        # RECIST trajectory
        if rs:
            cats = [r.get("category", "?") for r in rs]
            note = _RECIST_NOTES.get(cats[-1], "Response category undetermined.")
            s.extend(["### RECIST Trajectory", "", " -> ".join(cats), "",
                       f"*{note}*", ""])
        # Trend
        if ms and len(ms) >= 2:
            v0, v1 = float(ms[0].get("volume_mm3", 0)), float(ms[-1].get("volume_mm3", 0))
            if v0 > 0:
                pct = ((v1 - v0) / v0) * 100
                direction = "increased" if pct > 0 else "decreased"
                s.extend(["## Trend Analysis", "",
                           f"Tumor volume {direction} by {abs(pct):.1f}% from "
                           f"{v0:.1f} to {v1:.1f} mm3 over {len(ms)} observations "
                           f"across {len(tps)} timepoints.", ""])
        # Growth models
        if gm:
            s.extend(["## Growth Model Analysis", ""])
            sorted_gm = sorted(gm, key=lambda g: float(g.get("weight", 0)), reverse=True)
            for g in sorted_gm:
                s.append(f"- **{g.get('model_name', '?')}:** AIC = "
                         f"{float(g.get('aic', 0)):.1f}, weight = "
                         f"{float(g.get('weight', 0)):.3f}")
            best = sorted_gm[0]
            s.append(f"\nBest fit: {best.get('model_name', '?')} "
                     f"(weight = {float(best.get('weight', 0)):.3f}). "
                     "Model selection uncertainty should be considered.")
            s.append("")
        # Uncertainty + Key findings
        s.extend(["## Uncertainty Discussion", "",
                   "- Measurement variability between manual and automated methods",
                   "- Growth model selection uncertainty",
                   "- Limited temporal sampling may miss rapid changes",
                   "- Simulations are mathematical projections, not predictions", "",
                   "## Key Findings", ""])
        if ms:
            s.append(f"- {len(ms)} measurements across {len(tps)} timepoints")
        if rs:
            lat = rs[-1]
            s.append(f"- Latest RECIST: **{lat.get('category', '?')}** "
                     f"({float(lat.get('percent_change_from_baseline', 0)):+.1f}%)")
        if gm:
            bm = max(gm, key=lambda g: float(g.get("weight", 0)))
            s.append(f"- Best model: {bm.get('model_name', '?')} "
                     f"(weight = {float(bm.get('weight', 0)):.3f})")
        if not ms and not rs and not gm:
            s.append("- Insufficient data for key findings")
        s.extend(["", "---", "", SAFETY_DISCLAIMER])
        return "\n".join(s)

    def _template_counterfactual(
        self, baseline: dict[str, Any],
        counterfactuals: list[dict[str, Any]], therapy_text: str,
    ) -> str:
        """Template-based counterfactual interpretation."""
        s: list[str] = [
            "# Counterfactual Simulation Interpretation", "",
            "> Template-generated. These are hypothetical, NOT clinical predictions.", "",
            "## Baseline Trajectory", "", _format_trajectory(baseline), "",
        ]
        if counterfactuals:
            s.extend(["## Counterfactual Scenarios", ""])
            for cf in counterfactuals:
                s.extend([f"### {cf.get('name', 'Unknown')}",
                          _format_trajectory(cf), ""])
        s.extend(["## Therapy Context", "", therapy_text, "",
                   "## Interpretation Notes", "",
                   "- Simulations are mathematical 'what-if' projections.",
                   "- Differences may suggest timing sensitivity but are not causal.",
                   "- Uncertainty bounds reflect model parameter uncertainty only.",
                   "- Treatment decisions must never be based on simulations alone.",
                   "", "---", "", SAFETY_DISCLAIMER])
        return "\n".join(s)

    # -- internal helpers ---------------------------------------------------

    @staticmethod
    def _format_patient_context(metadata: dict[str, Any]) -> str:
        """Format patient metadata into prompt context."""
        if not metadata:
            return "**Patient:** No demographic data available."
        lines = ["**Patient Profile:**"]
        for key, label in [("cancer_type", "Cancer Type"), ("stage", "Stage"),
                           ("age", "Age"), ("sex", "Sex"),
                           ("histology", "Histology"), ("ECOG_PS", "ECOG PS"),
                           ("scenario", "Clinical Scenario")]:
            val = metadata.get(key)
            if val is not None:
                lines.append(f"- **{label}:** {val}")
        return "\n".join(lines)

    @staticmethod
    def _format_simulation_context(sim_results: list[Any]) -> str:
        """Format simulation results into prompt context."""
        if not sim_results:
            return "**Counterfactual Simulations:** No simulation data available."
        lines: list[str] = ["**Counterfactual Simulation Results:**", ""]
        for sim in sim_results:
            if isinstance(sim, dict):
                name = sim.get("name", sim.get("scenario_name", "Unknown"))
                times = sim.get("times", sim.get("time_points", []))
                vols = sim.get("volumes", sim.get("predicted_volumes", []))
            elif hasattr(sim, "scenario_name"):
                name = sim.scenario_name
                times = getattr(sim, "time_points", [])
                vols = getattr(sim, "predicted_volumes", [])
            else:
                continue
            n = len(times) if hasattr(times, "__len__") else 0
            if n > 0:
                va = np.asarray(vols)
                fv = float(va[-1]) if va.size > 0 else 0.0
                lines.append(f"- **{name}:** {n} points, final vol = {fv:.1f} mm3")
        return "\n".join(lines)


def _format_trajectory(trajectory: dict[str, Any]) -> str:
    """Format a trajectory dict into a readable summary."""
    name = trajectory.get("name", "Unknown")
    times = trajectory.get("times", [])
    volumes = trajectory.get("volumes", [])
    if not hasattr(times, "__len__") or len(times) == 0:
        return f"**{name}:** No trajectory data."
    ta, va = np.asarray(times), np.asarray(volumes)
    n = len(ta)
    v0 = float(va[0]) if va.size > 0 else 0.0
    vf = float(va[-1]) if va.size > 0 else 0.0
    span = float(ta[-1] - ta[0]) if n > 1 else 0.0
    lines = [f"**{name}:**",
             f"- Time span: {span:.1f} weeks ({n} points)",
             f"- Starting volume: {v0:.1f} mm3",
             f"- Final volume: {vf:.1f} mm3"]
    if v0 > 0:
        lines.append(f"- Volume change: {((vf - v0) / v0) * 100:+.1f}%")
    lower, upper = trajectory.get("lower"), trajectory.get("upper")
    if lower is not None and upper is not None:
        la, ua = np.asarray(lower), np.asarray(upper)
        if la.size > 0 and ua.size > 0:
            lines.append(f"- 95% CI at endpoint: [{float(la[-1]):.1f}, "
                         f"{float(ua[-1]):.1f}] mm3")
    return "\n".join(lines)
