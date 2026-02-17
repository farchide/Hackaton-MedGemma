# Digital Twin: Tumor Response Assessment

### Project Name

Digital Twin: Tumor Response Assessment -- AI-Powered Longitudinal Oncology Decision Support

### Your Team

[Team members, specialties, and roles to be filled by team]

### Problem Statement

**Longitudinal tumor response assessment is one of the most consequential yet error-prone tasks in oncology.** Clinicians must track multiple lesions across multiple imaging timepoints, apply RECIST 1.1 criteria under time pressure, and synthesize measurements into treatment decisions -- all while managing substantial measurement uncertainty.

The scale is staggering: 19.3 million new cancer cases are diagnosed annually worldwide (GLOBOCAN 2020), and virtually every patient receiving systemic therapy requires serial imaging response assessment. Yet the tools available to clinicians are fragmented: PACS viewers lack longitudinal reasoning, measurements are recorded in spreadsheets, and RECIST classifications are computed manually. Published inter-reader variability for RECIST categorization ranges from 10-30% disagreement, meaning patients may receive different response classifications depending on who reads their scan.

Three critical gaps remain unaddressed by existing tools:

1. **No uncertainty quantification.** A 19% increase in sum-of-diameters is classified as Stable Disease, while 21% is Progressive Disease. Without accounting for the 1-3mm measurement noise inherent in CT, these binary cutoffs create false confidence.
2. **No counterfactual reasoning.** Clinicians ask "what if we had started treatment earlier?" or "what if resistance develops at week 12?" -- but no tool lets them explore these scenarios against a patient-specific model.
3. **No integrated temporal synthesis.** Tumor board discussions require a narrative that connects measurements, treatment history, response trajectory, and uncertainty into a coherent clinical picture. Today this synthesis happens entirely in the clinician's head.

**Our solution is a patient-specific digital twin** that integrates imaging measurements with treatment context to: (a) classify response using RECIST 1.1 and iRECIST, (b) fit patient-specific growth models with full uncertainty quantification, (c) run counterfactual "what-if" simulations, and (d) generate evidence-grounded tumor board summaries via MedGemma. The target users are oncologists, radiologists, and clinical trial teams who currently lack tools for structured longitudinal reasoning.

**Impact:** If deployed in even 1% of the 1,400+ NCI-registered cancer centers, this system could standardize response assessment for tens of thousands of patients annually, reduce inter-observer variability, and surface treatment timing insights that are invisible in current workflows.

### Overall Solution

MedGemma serves as the **reasoning and communication layer** of the digital twin -- not a peripheral feature, but the intelligence that transforms raw measurements into actionable clinical insight. Critically, MedGemma is deployed as an **executable agentic tool** within our clinical pipeline: the system orchestrates MedGemma as a callable instrument alongside DICOM ingestion, RECIST classification, and growth modeling -- mirroring the agentic architecture where AI models and clinical data stores function as tools for an intelligent agent.

**Architecture: DICOM Ingestion --> State Estimation --> Simulation --> MedGemma Reasoning --> Safety**

The system flows through six layers. Layers 1-5 produce structured data: measurements, RECIST classifications, growth model fits, uncertainty estimates, and counterfactual simulations. MedGemma (Layer 6) is the only component that can synthesize all of this into a coherent, evidence-grounded narrative. Without MedGemma, the system produces numbers and charts. With MedGemma, it produces a tumor board summary.

**Key design principles that differentiate this system:**

- **DICOM-native ingestion.** Our pipeline accepts DICOM files directly (via `ingestion/dicom_loader.py`), parsing DICOM metadata, applying PHI scanning and de-identification before any data enters the reasoning pipeline. This aligns with MedGemma 1.5's expanded DICOM support via DICOMweb and enables deployment within existing radiology workflows without format conversion.
- **Privacy-preserving by design.** A de-identification gate sits at the system boundary: PHI is detected and stripped before measurements enter the digital twin state. All reasoning occurs on de-identified structured data, enabling privacy-preserving clinical workflow integration.
- **Longitudinal imaging analysis.** This is the core use case MedGemma 1.5 was specifically expanded to address -- 3D CT/MRI volume analysis and longitudinal imaging across multiple timepoints. Our digital twin tracks lesions across serial imaging sessions, fits patient-specific growth models, and reasons over the full temporal trajectory. This is not single-scan inference; it is exactly the kind of longitudinal, multi-timepoint analysis that MedGemma 1.5's architecture was designed for.
- **MedGemma as a configurable agentic tool.** The `MedGemmaReasoner` class exposes MedGemma as an executable tool with defined inputs (structured patient data, imaging slices) and outputs (grounded narratives, safety-checked summaries). The agentic workflow tab in the UI allows users to configure and invoke MedGemma as one step in a multi-stage pipeline, making it a true tool in an agentic system rather than a standalone model.

**Specifically, MedGemma performs four non-optional functions:**

1. **Evidence-grounded narrative generation.** MedGemma receives structured measurement tables, growth model parameters with AIC weights, RECIST trajectories, and three-tier uncertainty estimates. It produces a clinical summary that explicitly references source measurements (validated by our grounding checker). This is not image captioning -- it is multi-step reasoning over temporal data.

2. **Counterfactual interpretation.** When the simulation engine generates a "therapy 4 weeks earlier" scenario, MedGemma translates the numerical projections into clinical language: describing the magnitude of difference, contextualizing it against measurement uncertainty, and flagging when the difference falls within noise bounds.

3. **Safety-constrained output.** MedGemma's narrative passes through a code-enforced safety pipeline: prohibited phrases (e.g., "I recommend", "prescribe", "diagnosis is") are detected and replaced, a non-clinical disclaimer is injected, and counterfactual disclaimers are appended when simulation content is referenced. This is defense-in-depth: prompt engineering plus code enforcement.

4. **Agentic workflow orchestration.** MedGemma acts as the intelligent agent that decides which measurements to highlight, which growth model to emphasize based on AIC weights, and how to frame uncertainty. The NarrativeGenerator class orchestrates prompt composition, model invocation, grounding validation, and safety enforcement as a complete agentic pipeline.

**Why MedGemma and not a general LLM?** MedGemma's medical pretraining means it understands RECIST criteria, tumor growth dynamics, and immunotherapy response patterns without extensive prompt engineering. Its multimodal capability (MedGemma 4B) allows future integration of evidence image slices directly into the reasoning context. Its open-weight nature enables deployment in environments where data cannot leave the facility.

#### Agentic Workflow

Our 6-stage pipeline is fundamentally an **agentic system** where MedGemma orchestrates the clinical reasoning workflow. Rather than using MedGemma as a passive inference endpoint, we deploy it as an **executable tool** that an intelligent agent invokes with structured clinical context -- the same architectural pattern used in production clinical AI systems where models and data stores (like FHIR Stores) serve as callable tools for an agent.

**Agent architecture:**

| Stage | Agent Action | Tool Invoked | Output |
|-------|-------------|-------------|--------|
| 1. Ingest | Accept DICOM/NIfTI, scan for PHI | `dicom_loader` (DICOM-native) | De-identified imaging data |
| 2. Preprocess | Normalize imaging modality | `normalize` | Standardized CT/MRI arrays |
| 3. Measure | Segment lesions, compute RECIST | `recist`, `segmentation` | Structured measurements |
| 4. Track | Link lesion identities across time | `tracking` graph matcher | Longitudinal lesion graph |
| 5. Simulate | Fit growth models, run counterfactuals | `growth_models`, `simulation` | Trajectories with uncertainty |
| 6. Reason | Synthesize narrative with safety checks | **MedGemma** (agentic tool) | Grounded tumor board summary |

**What makes this agentic (not just a pipeline):**

1. **MedGemma decides what to emphasize.** Given the full structured context (measurements, growth models with AIC weights, uncertainty tiers, counterfactual trajectories), MedGemma autonomously determines which findings are clinically significant, which growth model to highlight based on AIC weights, and how to frame uncertainty -- it is not merely filling a template.

2. **Tool-use pattern.** The `MedGemmaReasoner` class exposes MedGemma with defined tool interfaces: `generate_tumor_board_summary(patient_data)`, `analyze_imaging_slice(image, context)`, and `generate_counterfactual_interpretation(baseline, counterfactuals, therapy)`. Each is a discrete tool invocation with structured inputs and validated outputs -- the same pattern as function-calling agents.

3. **Safety as an agent constraint.** The agentic pipeline enforces safety constraints at each stage: PHI scanning at ingestion, grounding validation after generation, prohibited phrase detection, and counterfactual disclaimers. These are not post-hoc filters; they are agent-level constraints that shape the reasoning process.

4. **Configurable orchestration.** The Agentic Workflow tab in the Gradio UI allows users to configure which stages execute, adjust parameters, and observe the agent's reasoning trace -- making the agentic nature transparent and controllable.

This agentic architecture directly enables the longitudinal analysis that MedGemma 1.5 was designed for: rather than analyzing a single scan in isolation, the agent tracks lesions across multiple timepoints, fits patient-specific models, runs counterfactual scenarios, and synthesizes the full temporal narrative -- all orchestrated through tool invocations with MedGemma as the reasoning engine.

### Technical Details

**Architecture (6 layers, fully implemented):**

| Layer | Function | Implementation |
|-------|----------|---------------|
| 1. Ingestion | DICOM/NIfTI with PHI scanning | `ingestion/dicom_loader.py`, `nifti_loader.py` |
| 2. Preprocessing | CT windowing, MRI normalization | `preprocessing/normalize.py` |
| 3. Measurement | Interactive segmentation, RECIST 1.1/iRECIST | `measurement/recist.py`, `segmentation.py` |
| 4. Tracking | Lesion identity graph across timepoints | `tracking/` module with graph-based matching |
| 5. Digital Twin Engine | Multi-model growth ensemble, counterfactual simulation | `twin_engine/growth_models.py`, `simulation.py` |
| 6. MedGemma Reasoning | Evidence-grounded narratives with safety | `reasoning/narrative.py`, `medgemma.py` |

**MedGemma Model Configuration:**

| Variant | Model ID | Parameters | Use Case |
|---------|----------|------------|----------|
| Multimodal (primary) | `google/medgemma-4b-it` | 4B + MedSigLIP 400M | Imaging analysis, multimodal reasoning over CT/MRI slices |
| Text-only (complex reasoning) | `google/medgemma-27b-text-it` | 27B | Complex clinical text synthesis, tumor board narratives (87.7% MedQA) |

- **MedSigLIP** (400M parameters): The image encoder within MedGemma-4B that provides medical image understanding. When our `analyze_imaging_slice()` method processes CT/MRI data, MedSigLIP extracts clinically meaningful visual features before the language model generates its analysis. This is purpose-built medical vision -- not a general-purpose image encoder.
- **MedGemma 1.5 capabilities**: Expanded support for 3D CT/MRI volumes and longitudinal imaging analysis -- precisely our use case of tracking tumor response across serial imaging timepoints.
- **FHIR-compatible data structures**: Our SQLite backend stores patient data in structures that map directly to FHIR resources (Patient, Observation, ImagingStudy, MedicationAdministration). This enables interoperability with clinical FHIR stores, allowing the digital twin to consume and produce data in the standard clinical data exchange format.
- **Official reference**: Architecture and integration patterns follow the official `google-health/medgemma` repository notebooks.

**Key Technical Innovations:**

**Multi-model growth ensemble with AIC-based model averaging.** We fit Exponential, Logistic, and Gompertz models using multi-start L-BFGS-B optimization (50 Latin Hypercube initial points per model) in log-space. AIC weights determine the ensemble: delta-AIC is computed, weights are proportional to exp(-0.5 * delta), and the best model drives predictions. Treatment effects are modeled as piecewise-constant multiplicative modifiers on growth rate, with sensitivity parameters estimated per lesion. This enables heterogeneous response modeling where one lesion responds while another progresses.

**Counterfactual simulation engine with 7 scenario types.** The `SimulationEngine` supports: natural history (no treatment), earlier/later treatment (therapy timeline shifts), regimen switch (sensitivity modification), treatment holiday (gap simulation with event splitting), acquired resistance (exponential decay of treatment effect after onset), dose escalation, and combination therapy. Each scenario modifies the therapy event list and re-solves the growth ODE, producing full trajectories with uncertainty bands.

**Three-tier uncertainty quantification.** Tier 1: measurement uncertainty composed from manual (sigma=1.5mm), automated segmentation, and scan-rescan variability sources via root-sum-of-squares. Tier 2: parametric bootstrap (1000 iterations) for growth model parameter uncertainty with prediction interval computation. Tier 3: model-form uncertainty from ensemble disagreement across Exponential/Logistic/Gompertz. Combined reliability is classified as HIGH/MEDIUM/LOW based on timepoint count and bootstrap CV.

**RECIST 1.1 + iRECIST implementation.** Full classification engine: target selection (2 per organ, 5 total), sum-of-diameters with nodal/non-nodal distinction (short-axis vs. longest-diameter), response classification (CR/PR/SD/PD with 30%/20%+5mm thresholds), overall response combining targets + non-targets + new lesions, and iRECIST extension for immunotherapy pseudo-progression (iUPD/iCPD with confirmation logic).

**Evaluation Results (5 synthetic patient scenarios, backtested):**

| Metric | Value |
|--------|-------|
| RECIST Classification Accuracy | 100% (synthetic ground truth) |
| Growth Model Convergence Rate | >95% (across 3 model types x 5 patients) |
| 80% Prediction Interval Coverage | Calibrated via 1000-sample bootstrap |
| Runtime per patient (full pipeline) | <2s on CPU |
| Counterfactual scenarios per patient | 7 (natural history through combination therapy) |
| Patient scenarios covered | 5 (classic responder, mixed response, pseudo-progression, rapid progression, surgical + adjuvant) |

**Deployment:**
- **Frontend:** Gradio web application with 7 tabs (Dashboard, Upload, Annotate, Track, Simulate, Narrate, Agentic Workflow)
- **Model:** MedGemma 4B-IT via HuggingFace Transformers; 4-bit NF4 quantization (~3GB VRAM) with graceful CPU fallback using template-based narratives
- **Storage:** SQLite backend for patient data, measurements, growth models, and simulations
- **Deployment:** Docker-ready; runs on a single consumer GPU or CPU-only

**Safety and Ethics:**
- Prominent non-clinical research disclaimers in UI and all generated narratives
- Code-enforced safety pipeline: prohibited phrase detection/replacement, grounding validation against source measurements, prompt injection detection, counterfactual disclaimers
- De-identification gate for user-uploaded DICOM with PHI scanning at the system boundary
- Full audit trail: every measurement records method (manual/semi-auto/auto), reviewer, and confidence
- Aligned with WHO AI Ethics Guidance and FDA GMLP Principles for AI/ML medical devices
- **MedGemma-specific safety alignment**: Our safety disclaimers and output constraints align with MedGemma's own model card warnings and Google's Responsible AI commitments for Health AI Developer Foundations (HAI-DEF). MedGemma's model card explicitly states it is "not intended for direct clinical use" -- our system enforces this at the code level through prohibited phrase detection ("I recommend", "prescribe", "diagnosis is") and mandatory research-only disclaimers on every output
- Privacy-preserving architecture: all reasoning occurs on de-identified structured data; imaging data never leaves the facility boundary when deployed on-premise with MedGemma's open weights

### Links

- **Code:** [GitHub Repository URL]
- **Video:** [Video URL -- 3 minutes]
- **Live Demo:** [Gradio App URL]

### References

1. Eisenhauer, E.A. et al. "New response evaluation criteria in solid tumours: Revised RECIST guideline (version 1.1)." *European Journal of Cancer* 45.2 (2009): 228-247.
2. Seymour, L. et al. "iRECIST: guidelines for response criteria for use in trials testing immunotherapeutics." *Lancet Oncology* 18.3 (2017): e143-e152.
3. Yang, L. et al. "MedGemma: Medical AI Foundation Models." Google Health AI Developer Foundations (HAI-DEF), 2025. Models: `google/medgemma-4b-it`, `google/medgemma-27b-text-it`. Repository: github.com/google-health/medgemma.
4. Laird, A.K. "Dynamics of tumour growth." *British Journal of Cancer* 18.3 (1964): 490-502.
5. WHO. "Ethics and governance of artificial intelligence for health." World Health Organization, 2021.
6. FDA et al. "Good Machine Learning Practice for Medical Device Development: Guiding Principles." 2021.
7. Sung, H. et al. "Global cancer statistics 2020: GLOBOCAN estimates." *CA: A Cancer Journal for Clinicians* 71.3 (2021): 209-249.
8. Zhao, B. et al. "Evaluating variability in tumor measurements from same-day repeat CT scans (RIDER)." *Translational Oncology* 2.4 (2009): 293-301.

---

*Citation: Fereshteh Mahvar et al. The MedGemma Impact Challenge. https://kaggle.com/competitions/med-gemma-impact-challenge, 2026. Kaggle.*
