# ADR-003: MedGemma Integration Strategy

| Field       | Value                                              |
|-------------|----------------------------------------------------|
| **Status**  | Accepted                                           |
| **Date**    | 2026-02-11                                         |
| **Authors** | Digital Twin Tumor Team                            |
| **Scope**   | MedGemma model selection, integration architecture,|
|             | prompt strategy, safety constraints, and fallbacks |

---

## Context

The MedGemma Impact Challenge requires use of at least one HAI-DEF model, with
"Effective use of HAI-DEF models" weighted at 20% of the total score. The
judging description specifies: "whether the submission proposes an application
that uses HAI-DEF models to their fullest potential, where other solutions would
likely be less effective."

Our system must make MedGemma non-optional and central to the value proposition.
The spec establishes that MedGemma should serve as the "reasoning + narrative +
safety" layer, NOT as a segmentation or measurement tool. This separation is
deliberate: measurement accuracy requires specialized models (MedSAM, nnU-Net)
or human annotation, while MedGemma excels at multimodal understanding and
text generation grounded in medical knowledge.

MedGemma is available in several variants:
- **MedGemma 1.5 4B multimodal**: Expanded CT/MRI 3D support, longitudinal
  imaging, and localization capabilities
- **MedGemma 1.0 4B multimodal**: Earlier version with medical image + text
- **MedGemma 1.0 27B text-only**: Larger model for text-heavy reasoning
- **MedGemma 1.0 27B multimodal**: Full multimodal at larger scale

MedGemma is built on Gemma 3 and is explicitly described as a "developer model"
requiring validation for specific use cases. It is not clinical-grade and not
intended for direct diagnosis or patient management without independent
verification.

We also have access to **MedSigLIP**, recommended by Google's model card for
"image-based applications such as classification or retrieval" when generation
is not needed.

---

## Decision

### Primary Model: MedGemma 1.5 4B Multimodal

We use **MedGemma 1.5 4B multimodal** (`google/medgemma-4b-it`) as the primary
reasoning model, loaded via HuggingFace Transformers.

```python
from transformers import AutoProcessor, AutoModelForImageTextToText
import torch

MODEL_ID = "google/medgemma-4b-it"

processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
```

### MedGemma's Role: Reasoning and Narrative Generation

MedGemma serves exactly one role in our architecture: **Layer 6 (Reasoning)**.
It receives structured data from upstream layers and produces human-readable
clinical narratives. It does NOT perform:

- Segmentation (handled by MedSAM / human annotation in Layer 3)
- Measurement (handled by geometric computation in Layer 3)
- Growth model fitting (handled by scipy optimization in Layer 5)
- Lesion tracking (handled by spatial matching in Layer 4)

This separation is critical because:

1. **Measurement reliability**: Lesion measurements must be deterministic and
   auditable. An LLM's stochastic output is inappropriate for numerical
   measurement that feeds into mathematical models.
2. **Uncertainty propagation**: Our twin engine produces calibrated uncertainty
   bands via bootstrap. Mixing LLM uncertainty with statistical uncertainty
   would be methodologically unsound.
3. **Auditability**: Judges and clinicians need to trace every number back to
   its source. MedGemma generates narratives FROM measurements, not the
   measurements themselves.

### Input/Output Contract

**Input to MedGemma** (assembled by Layer 6 from upstream outputs):

```python
@dataclass
class ReasoningInput:
    # Structured measurement data (from Layer 3 + 4)
    measurement_table: list[LesionMeasurement]
    response_category: str               # e.g., "Partial Response"
    sum_of_diameters: TimeSeriesData

    # Twin engine results (from Layer 5)
    growth_parameters: dict[str, GrowthParams]
    counterfactual_summary: CounterfactualResult
    uncertainty_summary: UncertaintySummary

    # Selected evidence images (2-4 key slices)
    evidence_slices: list[np.ndarray]     # Preprocessed image arrays
    slice_annotations: list[str]          # What each slice shows

    # Context
    therapy_timeline: list[TherapyEvent]
    patient_context: str                  # Deidentified summary
```

**Output from MedGemma**:

```python
@dataclass
class ClinicalNarrative:
    tumor_board_summary: str        # 200-400 word narrative
    key_findings: list[str]         # Bullet points
    uncertainty_statement: str      # Explicit uncertainty language
    counterfactual_insight: str     # "What-if" interpretation
    safety_disclaimer: str          # Always present, always identical
    evidence_references: list[str]  # Which slices/measurements cited
    generation_metadata: GenerationMetadata  # Temperature, tokens, time
```

### Prompt Engineering Strategy

We use structured prompts with explicit sections, measurement tables, and safety
constraints. Prompts are assembled programmatically, never free-form.

#### System Prompt (Fixed, Hardcoded)

```
You are a medical AI assistant generating structured summaries of tumor
measurement data for research purposes. You are part of a Digital Twin
Tumor system that tracks longitudinal imaging changes.

CRITICAL CONSTRAINTS:
1. You MUST NOT provide diagnosis or treatment recommendations.
2. You MUST base ALL statements on the provided measurement data.
3. You MUST acknowledge uncertainty explicitly when present.
4. You MUST include the safety disclaimer verbatim in every response.
5. You MUST use hedging language ("measurements suggest", "data indicates",
   "consistent with", "the model estimates") rather than definitive claims.
6. You MUST NOT extrapolate beyond what the measurements support.
7. If measurement confidence is below 0.7, you MUST flag it as unreliable.

SAFETY DISCLAIMER (include verbatim at the end of every response):
"This summary is generated by an AI system for research and development
purposes only. It is NOT intended for clinical decision-making. All
findings require independent verification by qualified medical
professionals. MedGemma is a developer model and has not been validated
for clinical use."
```

#### User Prompt Template

```
## Longitudinal Tumor Assessment Summary

### Patient Context
{patient_context}

### Therapy Timeline
{therapy_timeline_formatted}

### Measurement Table
| Lesion | Timepoint | Diameter (mm) | Volume (mm3) | Confidence | Source   |
|--------|-----------|---------------|--------------|------------|----------|
{measurement_rows}

### Response Assessment
- Sum of diameters (baseline): {sum_baseline} mm
- Sum of diameters (current): {sum_current} mm
- Percent change: {pct_change}%
- Category: {response_category}

### Digital Twin Model Results
- Primary model: {model_form} (R2={model_r2:.3f})
- Growth rate estimate: {growth_rate} +/- {growth_rate_ci}
- Therapy sensitivity: {sensitivity} +/- {sensitivity_ci}
- Prediction horizon: {horizon_weeks} weeks

### Uncertainty Summary
- Measurement uncertainty (median CV): {measurement_cv}%
- Model uncertainty (95% CI width): {model_ci_width} mm
- Forecasting coverage: {coverage_pct}% of observed values within bands

### Counterfactual Analysis
- Scenario: {counterfactual_description}
- Projected difference at {projection_time}: {projected_diff} mm
  ({projected_diff_pct}% change from factual trajectory)

### Evidence Images
{evidence_image_descriptions}

---

Based on the above data, generate a structured tumor board summary with:
1. Key findings (3-5 bullet points grounded in measurements)
2. Trend interpretation (what the longitudinal data shows)
3. Uncertainty caveats (what we are and are not confident about)
4. Counterfactual insight (what the simulation suggests, with caveats)
5. Safety disclaimer (verbatim from instructions)
```

### Prompt Assembly in Code

```python
# src/reasoning/prompt_builder.py
from dataclasses import dataclass
from src.common.models import ReasoningInput

SYSTEM_PROMPT = """..."""  # Fixed, as shown above
USER_TEMPLATE = """..."""  # Template, as shown above

class PromptBuilder:
    """Assembles structured prompts from typed ReasoningInput."""

    def build(self, input_data: ReasoningInput) -> list[dict]:
        """Build a chat-format prompt with system + user messages."""
        measurement_rows = self._format_measurement_table(
            input_data.measurement_table
        )
        user_content = self._fill_template(input_data, measurement_rows)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
        ]

        # Add evidence images as interleaved content
        if input_data.evidence_slices:
            user_parts = []
            for i, (img, desc) in enumerate(zip(
                input_data.evidence_slices,
                input_data.slice_annotations
            )):
                user_parts.append({"type": "image", "image": img})
                user_parts.append({
                    "type": "text",
                    "text": f"Evidence slice {i+1}: {desc}"
                })
            user_parts.append({"type": "text", "text": user_content})
            messages.append({"role": "user", "content": user_parts})
        else:
            messages.append({"role": "user", "content": user_content})

        return messages

    def _format_measurement_table(self, measurements):
        """Format measurements as a markdown table."""
        rows = []
        for m in measurements:
            rows.append(
                f"| {m.lesion_id} | {m.timepoint.label} | "
                f"{m.diameter_mm:.1f} | {m.volume_mm3:.0f} | "
                f"{m.confidence:.2f} | {m.provenance.value} |"
            )
        return "\n".join(rows)
```

### Safety: Hard-Coded Refusal Behaviors

Safety is not optional or prompt-dependent. We enforce it in code.

```python
# src/reasoning/safety.py

IMMUTABLE_DISCLAIMER = (
    "This summary is generated by an AI system for research and development "
    "purposes only. It is NOT intended for clinical decision-making. All "
    "findings require independent verification by qualified medical "
    "professionals. MedGemma is a developer model and has not been validated "
    "for clinical use."
)

PROHIBITED_PHRASES = [
    "I recommend",
    "you should",
    "the patient needs",
    "start treatment",
    "stop treatment",
    "prescribe",
    "administer",
    "this confirms the diagnosis",
    "this rules out",
]

def enforce_safety(narrative: str) -> str:
    """Post-process MedGemma output to enforce safety constraints."""
    # Strip any prohibited clinical directive language
    for phrase in PROHIBITED_PHRASES:
        if phrase.lower() in narrative.lower():
            narrative = narrative.replace(
                phrase,
                "[REMOVED: clinical directive not permitted]"
            )

    # Ensure disclaimer is always present (append if missing)
    if IMMUTABLE_DISCLAIMER not in narrative:
        narrative = narrative.rstrip() + "\n\n---\n\n" + IMMUTABLE_DISCLAIMER

    return narrative


def validate_grounding(narrative: str, measurements: list) -> float:
    """Check that narrative references actual measurement values."""
    referenced_values = 0
    total_values = len(measurements)
    for m in measurements:
        if f"{m.diameter_mm:.1f}" in narrative:
            referenced_values += 1
    return referenced_values / max(total_values, 1)
```

### Generation Parameters

```python
GENERATION_CONFIG = {
    "max_new_tokens": 1024,
    "temperature": 0.3,       # Low for factual consistency
    "top_p": 0.9,
    "top_k": 50,
    "do_sample": True,
    "repetition_penalty": 1.1,
}
```

We use low temperature (0.3) because:
- Medical narratives must be factually consistent with input data
- We want deterministic-leaning output for reproducibility
- Higher temperatures risk hallucinated measurements or unsupported claims
- The structured prompt already constrains the output format

### MedSigLIP Integration for Evidence Retrieval

MedSigLIP serves a supporting role: given a set of image slices across
timepoints, it identifies the most informative slices to pass to MedGemma
as visual evidence.

```python
# src/reasoning/evidence_retrieval.py
from transformers import AutoModel, AutoProcessor
import torch

SIGLIP_MODEL_ID = "google/medsiglip"

class EvidenceRetriever:
    """Select most relevant slices for MedGemma visual grounding."""

    def __init__(self):
        self.processor = AutoProcessor.from_pretrained(SIGLIP_MODEL_ID)
        self.model = AutoModel.from_pretrained(
            SIGLIP_MODEL_ID,
            torch_dtype=torch.float16,
        )

    def select_evidence_slices(
        self,
        slices: list[np.ndarray],
        query: str,
        top_k: int = 4,
    ) -> list[int]:
        """Return indices of top-k slices most relevant to the query."""
        inputs = self.processor(
            text=[query],
            images=slices,
            return_tensors="pt",
            padding=True,
        )
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Cosine similarity between text and each image
        logits = outputs.logits_per_text[0]
        top_indices = logits.argsort(descending=True)[:top_k]
        return top_indices.tolist()
```

This serves two purposes:
1. **Quality**: MedGemma receives only the most informative slices, not the
   entire volume. This improves narrative quality and reduces token usage.
2. **Judging**: Demonstrates effective use of multiple HAI-DEF models (MedGemma
   + MedSigLIP), scoring higher on the "HAI-DEF use" criterion.

#### RuVector-Powered Evidence Retrieval

Rather than computing cosine similarity at query time across all slices,
MedSigLIP embeddings are pre-indexed in RuVector's HNSW index at ingestion
time. This moves the O(n) similarity scan to a sub-millisecond approximate
nearest neighbor lookup, enabling instant evidence retrieval even as the
embedding corpus grows across patients and timepoints.

```python
# Store MedSigLIP embeddings in RuVector
from ruvector import VectorDB, VectorEntry, SearchQuery

evidence_db = VectorDB(dimensions=768)  # MedSigLIP embedding dimension

# Index all slice embeddings at ingestion time
for slice_idx, embedding in enumerate(slice_embeddings):
    evidence_db.insert(VectorEntry(
        id=f"{patient_id}_{timepoint}_{slice_idx}",
        vector=embedding.tolist(),
        metadata={"patient_id": patient_id, "timepoint": timepoint, "slice_idx": slice_idx}
    ))

# At query time, retrieve top-k evidence slices in <1ms
results = evidence_db.search(SearchQuery(
    vector=query_embedding.tolist(), k=4, include_vectors=False
))
```

This approach has three advantages over the in-memory cosine similarity
baseline:

1. **Speed**: HNSW provides sub-millisecond retrieval regardless of corpus
   size, compared to linear-time cosine similarity over all slices.
2. **Persistence**: Embeddings survive process restarts. A clinician can close
   the application and return the next day without re-encoding all slices.
3. **Cross-patient retrieval**: The HNSW index spans all patients, enabling
   the evidence panel to surface visually similar slices from other patients
   in the same cohort -- a feature that would require a full re-scan with the
   in-memory approach.

### Fallback Strategy: MedGemma 27B Text-Only

If GPU resources allow (48+ GB VRAM or multi-GPU), we optionally use the 27B
text-only variant for offline batch processing:

```python
FALLBACK_MODEL_ID = "google/medgemma-27b-text-it"

def load_fallback_model():
    """Load 27B text model for batch summarization if GPU allows."""
    try:
        model = AutoModelForCausalLM.from_pretrained(
            FALLBACK_MODEL_ID,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            load_in_4bit=True,  # Quantize to fit in ~16GB
        )
        return model
    except (torch.cuda.OutOfMemoryError, RuntimeError):
        return None  # Graceful fallback to 4B multimodal
```

The 27B model is used ONLY for:
- Batch generation of narratives across all timepoints (background job)
- Comparative analysis across multiple lesions (longer context)
- It is never in the critical path of the interactive demo

### Why NOT Fine-Tuning

We explicitly choose prompt engineering over fine-tuning for this hackathon.

| Factor               | Fine-Tuning                          | Prompt Engineering              |
|----------------------|--------------------------------------|---------------------------------|
| Time required        | 3-5 days (data prep + train + eval)  | 1-2 days (iterate prompts)     |
| Data required        | Hundreds of example narratives       | Zero training examples          |
| Validation burden    | Must prove no regression on safety   | Safety enforced in code         |
| Reproducibility      | Requires training infrastructure     | Deterministic prompt assembly   |
| Competition risk     | Model may not converge in time       | Always produces output          |
| Clinical safety      | Risk of removing safety behaviors    | Safety is post-processing code  |

Fine-tuning is the right choice for a production system but wrong for a 13-day
hackathon where:
- We have no curated tumor-board narrative dataset to train on
- Validation of fine-tuned medical model behavior is a multi-week effort
- The "Novel Task Prize" is for fine-tuning; we are targeting the Main Track
- Prompt engineering with structured inputs already produces high-quality output

Post-hackathon, if this becomes a research artifact, we would fine-tune on a
curated dataset of (measurement table, expert narrative) pairs.

### GPU Memory Planning

MedGemma 1.5 4B in bfloat16 requires approximately 8 GB VRAM for weights alone,
plus KV cache overhead during generation. Our memory budget:

| Component             | VRAM Estimate  | When Loaded              |
|-----------------------|----------------|--------------------------|
| MedGemma 1.5 4B bf16 | ~8 GB          | Layer 6 (Reasoning)      |
| MedSAM               | ~2.5 GB        | Layer 3 (Measurement)    |
| MedSigLIP            | ~1.5 GB        | Layer 6 (Evidence)       |
| KV Cache + overhead   | ~2-4 GB        | During generation        |
| RuVector (CPU)        | ~0 GB GPU / ~500MB RAM | All phases (background) |
| Total peak            | ~14-16 GB      |                          |

Note: RuVector runs entirely on CPU and does not consume GPU VRAM. Its ~500 MB
RAM footprint covers the HNSW index, GNN layers, and connection overhead. This
means adding RuVector-powered evidence retrieval has zero impact on the GPU
memory budget for MedGemma, MedSAM, and MedSigLIP.

Strategy for a 16 GB GPU (T4 on HuggingFace Spaces):
1. Load MedSAM during annotation phase; unload before reasoning
2. Load MedSigLIP + MedGemma for reasoning phase
3. Use `torch.cuda.empty_cache()` between phases
4. If memory is tight, quantize MedGemma to int8 via bitsandbytes (~4 GB)

```python
# src/reasoning/model_manager.py
import gc
import torch

class ModelManager:
    """Manage GPU memory across model lifecycle phases."""

    def __init__(self):
        self._loaded_models: dict[str, Any] = {}

    def unload(self, model_name: str) -> None:
        if model_name in self._loaded_models:
            del self._loaded_models[model_name]
            gc.collect()
            torch.cuda.empty_cache()

    def load_reasoning_models(self) -> None:
        """Load MedGemma + MedSigLIP, unloading measurement models."""
        self.unload("medsam")
        # ... load MedGemma and MedSigLIP
```

### Integration with Upstream Layers

The reasoning layer subscribes to the `twin.fitted` event from the EventBus
(ADR-001). When the twin engine completes simulation, the reasoning layer:

1. Collects all `LesionMeasurement` objects from Layer 3/4
2. Collects `TwinSimulationResult` from Layer 5
3. Uses MedSigLIP to select 2-4 evidence slices from `PreprocessedVolume`
4. Assembles the structured prompt via `PromptBuilder`
5. Runs MedGemma inference
6. Post-processes output through `enforce_safety()`
7. Returns `ClinicalNarrative` dataclass
8. Publishes `narrative.generated` event

```python
# src/reasoning/engine.py
class ReasoningEngine:
    def generate(self, reasoning_input: ReasoningInput) -> ClinicalNarrative:
        # 1. Select evidence slices
        evidence_indices = self.retriever.select_evidence_slices(
            reasoning_input.evidence_slices,
            query="tumor response longitudinal change",
            top_k=4,
        )
        selected_slices = [
            reasoning_input.evidence_slices[i] for i in evidence_indices
        ]

        # 2. Build prompt
        messages = self.prompt_builder.build(reasoning_input)

        # 3. Run inference
        inputs = self.processor(
            messages,
            return_tensors="pt",
        ).to(self.model.device)

        outputs = self.model.generate(**inputs, **GENERATION_CONFIG)
        raw_narrative = self.processor.decode(
            outputs[0], skip_special_tokens=True
        )

        # 4. Enforce safety (code-level, not prompt-level)
        safe_narrative = enforce_safety(raw_narrative)

        # 5. Validate grounding
        grounding_score = validate_grounding(
            safe_narrative,
            reasoning_input.measurement_table,
        )

        return ClinicalNarrative(
            tumor_board_summary=safe_narrative,
            key_findings=self._extract_findings(safe_narrative),
            uncertainty_statement=self._extract_uncertainty(safe_narrative),
            counterfactual_insight=self._extract_counterfactual(safe_narrative),
            safety_disclaimer=IMMUTABLE_DISCLAIMER,
            evidence_references=[
                reasoning_input.slice_annotations[i]
                for i in evidence_indices
            ],
            generation_metadata=GenerationMetadata(
                model_id=MODEL_ID,
                temperature=GENERATION_CONFIG["temperature"],
                grounding_score=grounding_score,
                generation_time_seconds=elapsed,
            ),
        )
```

---

## Consequences

### Positive

- **Central HAI-DEF usage**: MedGemma is the sole generator of patient-facing
  narratives. Without it, the system produces numbers but not understanding.
  This satisfies the "fullest potential" language in the judging criteria.
- **Multi-model HAI-DEF**: Using both MedGemma (generation) and MedSigLIP
  (retrieval) demonstrates breadth of HAI-DEF utilization.
- **Safety by design**: Hard-coded safety in Python code cannot be bypassed by
  prompt injection or model drift. The disclaimer is always present.
- **Reproducible**: Prompt templates are version-controlled. Same input always
  produces similar output (low temperature + structured prompt).
- **Grounded narratives**: Every statement in the narrative traces back to a
  measurement value or model output. The `validate_grounding()` function
  provides a quantitative check.
- **Self-improving evidence retrieval**: RuVector's GNN layers (40+ attention
  mechanisms including MultiHeadAttention and GRUCell) refine the HNSW index
  over time. As more cases are processed and evidence slices are selected or
  rejected by clinicians, the retrieval quality improves automatically. This
  means the evidence panel becomes more clinically relevant with continued use.

### Negative

- **4B model limitations**: The 4B model has less reasoning depth than the 27B.
  Complex multi-lesion scenarios with conflicting trends may produce shallow
  narratives. Mitigated by providing highly structured input so the model
  summarizes rather than reasons from scratch.
- **No fine-tuning**: Prompt engineering has a ceiling. The model may occasionally
  produce awkward phrasing or miss nuances that fine-tuning would address.
  Acceptable for a hackathon demo.
- **Latency**: MedGemma generation adds 5-15 seconds per narrative on a T4 GPU.
  Mitigated by running generation after the user clicks "Generate Summary"
  (not blocking the main interaction loop).
- **Image token cost**: Passing multiple evidence slices consumes many tokens,
  reducing available context for the measurement table. Mitigated by limiting
  to 4 slices and using MedSigLIP to select the most informative ones.

### Risks

| Risk                             | Likelihood | Impact | Mitigation                                 |
|----------------------------------|------------|--------|--------------------------------------------|
| MedGemma hallucinates values     | Medium     | High   | `validate_grounding()` check; structured   |
|                                  |            |        | prompts constrain output format            |
| Safety disclaimer removed by     | Low        | High   | `enforce_safety()` is post-processing code |
| model or prompt injection        |            |        | that always appends disclaimer             |
| GPU OOM loading MedGemma         | Medium     | High   | Sequential model loading; int8 quantization|
|                                  |            |        | fallback via bitsandbytes                  |
| MedGemma API changes in          | Low        | Medium | Pin transformers version; test on load     |
| transformers update              |            |        |                                            |
| Narrative quality too low        | Low        | Medium | Pre-computed demo narratives as fallback;  |
| for demo                         |            |        | iterate prompt engineering                 |

---

## Alternatives Considered

### 1. MedGemma for Segmentation

Use MedGemma's visual understanding to directly segment lesions.

- **Rejected because**: MedGemma is a generative language model, not a
  segmentation model. It can describe what it sees in an image, but it does not
  output pixel masks. Segmentation requires MedSAM or nnU-Net, which produce
  actual contour geometries. Using MedGemma for segmentation would be an
  ineffective use of the model and would score poorly on the "appropriate use"
  criterion.

### 2. MedGemma 27B Multimodal as Primary

Use the larger model for better reasoning quality.

- **Rejected because**: The 27B multimodal model requires ~54 GB VRAM in bf16,
  which exceeds single-GPU capacity on most accessible hardware (T4: 16 GB,
  A100: 40/80 GB). Even with 4-bit quantization (~14 GB), inference is
  significantly slower, harming demo interactivity. The 4B model provides
  adequate narrative quality when given highly structured input, and the 27B
  text-only variant is available as an optional batch fallback.

### 3. Fine-Tuning on Synthetic Tumor Board Narratives

Generate synthetic training data and fine-tune MedGemma.

- **Rejected because**: Fine-tuning a medical model carries validation burden
  that exceeds our hackathon timeline. We would need to verify that fine-tuning
  did not degrade safety behaviors, did not introduce hallucinations, and
  actually improved narrative quality. The "Novel Task Prize" rewards fine-tuning,
  but we are targeting the Main Track where execution quality matters more.
  Post-hackathon, this is the recommended next step.

### 4. GPT-4V / Claude as Reasoning Model

Use a proprietary multimodal model for higher-quality narratives.

- **Rejected because**: The competition mandates use of HAI-DEF models. Using a
  non-HAI-DEF model as the primary reasoning engine would fail the mandatory
  requirement. Additionally, proprietary models cannot be deployed offline or
  on-premise, which undermines the "privacy-focused tools" philosophy stated
  in the competition description.

### 5. Rule-Based Narrative Generation

Generate narratives from templates without any LLM.

- **Rejected because**: Template-based narratives are rigid and cannot adapt to
  unusual measurement patterns, complex multi-lesion scenarios, or nuanced
  uncertainty communication. This would score poorly on "Effective use of
  HAI-DEF models" (20%) and would not demonstrate the unique value of MedGemma
  in synthesizing multimodal evidence into coherent clinical language.

---

## References

- MedGemma 1.5 Model Card: Expanded CT/MRI 3D support, longitudinal imaging
- MedGemma on HuggingFace: `google/medgemma-4b-it`
- MedSigLIP: Recommended by Google for image-based retrieval/classification
- Gemma 3 Memory Estimates: Google documentation on VRAM requirements
- HuggingFace Transformers: `AutoModelForImageTextToText` API
- Competition Rules: "Use of at least one of HAI-DEF models is mandatory"
- bitsandbytes: 4-bit/8-bit quantization for transformer models
