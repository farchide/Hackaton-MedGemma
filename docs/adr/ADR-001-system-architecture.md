# ADR-001: Overall System Architecture

| Field       | Value                                      |
|-------------|--------------------------------------------|
| **Status**  | Accepted                                   |
| **Date**    | 2026-02-11                                 |
| **Authors** | Digital Twin Tumor Team                    |
| **Scope**   | End-to-end system structure and data flow  |

---

## Context

We are building "Digital Twin Tumor," a patient-specific, continuously updatable
model for the MedGemma Impact Challenge (Kaggle hackathon, deadline Feb 24 2026).
The system must integrate longitudinal imaging measurements with treatment context
to: (a) summarize what changed, (b) quantify uncertainty, and (c) run
counterfactual "what-if" simulations.

The competition evaluates submissions on five weighted criteria:

- Effective use of HAI-DEF models (20%)
- Problem domain clarity (15%)
- Impact potential (15%)
- Product feasibility (20%)
- Execution and communication (30%)

We have approximately 13 days remaining. The team needs an architecture that is
fast to implement, easy to demo in a 3-minute video, and clean enough to
communicate technical depth in a 3-page writeup.

The system has natural conceptual boundaries: data ingestion, image preprocessing,
human-guided measurement, lesion tracking across time, mathematical simulation,
and AI-powered reasoning. These boundaries must be reflected in code organization
without introducing premature distributed-system complexity.

---

## Decision

We adopt a **layered pipeline architecture with six bounded contexts**, implemented
as a Python monolith with clean module boundaries under `src/`. Data flows
between layers via typed Python dataclasses and Protocol-based interfaces.

### The Six Layers

```
src/
  ingestion/       Layer 1 - Ingestion and Governance
  preprocessing/   Layer 2 - Image Preprocessing
  measurement/     Layer 3 - Human-in-the-Loop Measurement
  tracking/        Layer 4 - Lesion Tracking and Identity Graph
  twin_engine/     Layer 5 - Digital Twin Engine
  reasoning/       Layer 6 - MedGemma Reasoning Layer
```

### ASCII Architecture Diagram

```
+============================================================================+
|                          CLIENT / GRADIO UI                                |
|   Upload  -->  Annotate  -->  Track  -->  Simulate  -->  Narrate           |
+====+===============+============+============+===============+=============+
     |               |            |            |               |
     v               v            v            v               v
+----+----+   +------+-----+  +--+---+  +-----+------+  +-----+------+
| Layer 1 |   |  Layer 2   |  |Layer3|  |  Layer 4   |  |  Layer 5   |
|Ingestion|-->|Preprocess  |->|Measur|->| Tracking   |->|Twin Engine |
|  & Gov  |   |            |  | ment |  | & Identity |  | (Gompertz) |
+---------+   +------------+  +--+---+  +-----+------+  +-----+------+
                                  |            |               |
                                  |  +---------+-------+       |
                                  +->| Lesion Identity  |      |
                                     | Graph (NetworkX) |      |
                                     +-----------------+       |
                                                               v
                                                    +----------+----------+
                                                    |      Layer 6        |
                                                    | MedGemma Reasoning  |
                                                    | (Narrative + Safety)|
                                                    +----------+----------+
                                                               |
                                                               v
                                                    +----------+----------+
                                                    | Tumor Board Summary |
                                                    | + Uncertainty Report|
                                                    | + Safety Disclaimers|
                                                    +---------------------+

Data Flow (typed dataclasses at each boundary):

  ImagingStudy --> PreprocessedVolume --> LesionMeasurement -->
  TrackedLesionSet --> TwinSimulationResult --> ClinicalNarrative
```

### Layer Responsibilities

**Layer 1 -- Ingestion and Governance (`src/ingestion/`)**

- Accept DICOM (via `pydicom`) or NIfTI (via `nibabel`) imaging files
- Parse and validate therapy logs (CSV/JSON: regimen labels, start dates, doses)
- PHI scanning gate: reject or warn on non-deidentified DICOM headers
- Output: `ImagingStudy` dataclass with metadata, file references, therapy timeline
- Key libraries: `pydicom`, `nibabel`, `pandas`

**Layer 2 -- Preprocessing (`src/preprocessing/`)**

- CT: intensity windowing, resampling to isotropic voxels
- MRI: sequence normalization, orientation standardization
- Slice/patch selection policy for MedGemma input (controlled number of slices)
- Output: `PreprocessedVolume` dataclass with normalized arrays and spatial metadata
- Key libraries: `numpy`, `scipy.ndimage`, `SimpleITK`

**Layer 3 -- Human-in-the-Loop Measurement (`src/measurement/`)**

- Interactive lesion selection (enforce RECIST 1.1 guardrails: 2 per organ, 5 total)
- Diameter endpoint placement (click-to-measure tool)
- Optional MedSAM-assisted segmentation with always-editable contours
- Record provenance: was each measurement manual, AI-suggested, or AI-confirmed
- Output: `LesionMeasurement` dataclass with geometry, confidence, provenance
- Key libraries: `gradio` (UI), `segment-anything` (MedSAM), `opencv-python`

**Layer 4 -- Lesion Tracking and Identity Graph (`src/tracking/`)**

- Represent lesions as a temporal graph: nodes = (timepoint, lesion), edges = matches
- Simple spatial registration + nearest-centroid matching across timepoints
- New lesion detection and labeling (aligned with iRECIST concepts)
- Output: `TrackedLesionSet` dataclass with identity-resolved longitudinal series
- Key libraries: `networkx`, `scipy.spatial`, `numpy`

**Layer 5 -- Digital Twin Engine (`src/twin_engine/`)**

- Fit patient-specific Gompertz growth models per lesion with treatment effect terms
- Multi-start optimization via `scipy.optimize.minimize` with bootstrap uncertainty
- Model ensemble: exponential + logistic + Gompertz for epistemic uncertainty
- Counterfactual simulator: shift therapy timing, adjust sensitivity, swap regimens
- Output: `TwinSimulationResult` dataclass with trajectories, uncertainty bands, breakpoints
- Key libraries: `scipy.optimize`, `scipy.integrate`, `numpy`, `dataclasses`

**Layer 6 -- MedGemma Reasoning (`src/reasoning/`)**

- Convert structured measurement tables + uncertainty data into prompts
- Generate tumor-board-ready narratives with safety constraints
- Hard-coded refusal behaviors and non-clinical disclaimers in every output
- Optional MedSigLIP evidence retrieval for visual grounding
- Output: `ClinicalNarrative` dataclass with text, citations, confidence, disclaimers
- Key libraries: `transformers`, `torch`, `accelerate`

### Inter-Layer Data Contracts

All boundaries use Python `dataclasses` with type annotations. Each dataclass is
defined in `src/common/models.py` so that every layer imports from a single
source of truth. We use `typing.Protocol` for interface definitions so layers
depend on abstractions, not implementations.

```python
# Example boundary contracts (simplified)
@dataclass
class ImagingStudy:
    patient_id: str
    timepoints: list[Timepoint]
    therapy_log: list[TherapyEvent]
    provenance: DataProvenance

@dataclass
class PreprocessedVolume:
    array: np.ndarray          # (D, H, W) float32
    spacing: tuple[float, float, float]
    orientation: str
    timepoint: Timepoint

@dataclass
class LesionMeasurement:
    lesion_id: str
    timepoint: Timepoint
    diameter_mm: float
    volume_mm3: float | None
    contour: np.ndarray | None
    confidence: float
    provenance: MeasurementProvenance

@dataclass
class TwinSimulationResult:
    lesion_id: str
    observed: list[TimeValue]
    predicted: list[TimeValue]
    counterfactual: list[TimeValue]
    uncertainty_bands: UncertaintyBands
    model_form: str
    parameters: dict[str, float]
```

### Why Monolith-First

This is a 13-day hackathon. A monolith with clean internal boundaries gives us:

1. **Speed**: No service discovery, no network serialization, no Docker orchestration
2. **Debuggability**: Single process, single stack trace, print-statement debugging
3. **Demo simplicity**: `python app.py` launches everything; judges can reproduce
4. **Refactorability**: Clean module boundaries mean we can extract services later
   if this becomes a research artifact post-hackathon

The module boundaries are strict enough that no layer imports implementation
details from another layer -- only shared dataclasses from `src/common/`.

### Mapping Layers to Judging Criteria

| Layer                  | Primary Criteria Served                        | How                                                              |
|------------------------|------------------------------------------------|------------------------------------------------------------------|
| Ingestion & Governance | Product feasibility (20%)                      | Shows real data handling, PHI awareness, deployment readiness     |
| Preprocessing          | Product feasibility (20%)                      | Demonstrates handling of real medical imaging formats             |
| Measurement (HITL)     | Problem domain (15%), Impact (15%)             | Clinician-centric workflow; trust and auditability                |
| Tracking               | Problem domain (15%), Execution (30%)          | Longitudinal reasoning is the core clinical problem              |
| Twin Engine            | Impact potential (15%), Effective HAI-DEF (20%) | Counterfactuals are the "wow" feature; uncertainty is rigorous   |
| MedGemma Reasoning     | Effective HAI-DEF (20%), Execution (30%)       | Central, non-optional use of MedGemma; narrative quality         |

### Event-Driven Internal Communication

While this is a monolith, we use a lightweight event pattern internally via a
simple `EventBus` class (publish/subscribe with Python callables). This serves
two purposes:

1. **Audit trail**: Every state transition emits an event that is logged, enabling
   the "audit log" feature showing what was manual vs. automated
2. **Decoupling**: Layers react to events rather than calling each other directly,
   keeping the dependency graph clean

```python
# Lightweight event bus (no external dependency)
class EventBus:
    def __init__(self) -> None:
        self._handlers: dict[str, list[Callable]] = defaultdict(list)

    def subscribe(self, event_type: str, handler: Callable) -> None:
        self._handlers[event_type].append(handler)

    def publish(self, event_type: str, payload: Any) -> None:
        for handler in self._handlers[event_type]:
            handler(payload)
```

Events include: `study.ingested`, `volume.preprocessed`, `lesion.measured`,
`lesion.identity_confirmed`, `twin.fitted`, `narrative.generated`.

---

## Consequences

### Positive

- **Fast iteration**: Single codebase, single language, no deployment overhead
- **Clear responsibility**: Each module owns a bounded context with defined inputs
  and outputs; new team members can work on one layer without understanding all six
- **Demo-friendly**: The entire system runs as one process behind a Gradio UI,
  making it trivial to demonstrate and reproduce
- **Extensible**: Post-hackathon, any layer can be extracted into a service; the
  dataclass contracts become API schemas
- **Auditable**: Event-driven logging creates a complete provenance trail, which
  is critical for medical AI credibility

### Negative

- **Single point of failure**: If MedGemma inference crashes, the whole process
  dies. Mitigated by wrapping inference in try/except with graceful degradation
  (show measurements without narrative)
- **Memory pressure**: Loading MedGemma 4B + MedSAM in one process requires
  careful GPU memory management. Mitigated by sequential model loading and
  explicit `torch.cuda.empty_cache()` between stages
- **Scalability ceiling**: A monolith cannot serve many concurrent users.
  Acceptable for a hackathon demo; documented as a post-hackathon migration path
- **Testing complexity**: Integration tests must set up the full pipeline.
  Mitigated by Protocol-based interfaces that allow mocking at each boundary

### Risks

| Risk                            | Likelihood | Impact | Mitigation                                      |
|---------------------------------|------------|--------|--------------------------------------------------|
| GPU OOM with multiple models    | Medium     | High   | Sequential loading; quantization; slice limits   |
| Layer coupling creeps in        | Medium     | Medium | CI check for cross-layer imports; code review    |
| Demo crashes during recording   | Low        | High   | Pre-cache results for demo cases; graceful errors|
| Dataclass schema changes break  | Medium     | Low    | All models in one file; grep for usage           |

---

## Alternatives Considered

### 1. Microservices Architecture

Each layer as a separate service communicating via REST/gRPC.

- **Rejected because**: Massive overhead for a 13-day hackathon. Docker compose,
  service discovery, network debugging, and deployment complexity would consume
  days that should go to feature development. The team is small and co-located
  in a single repo.

### 2. Jupyter Notebook Pipeline

Implement each layer as a notebook, chained together.

- **Rejected because**: Notebooks are poor for code reuse, testing, and building
  a Gradio UI. They also make version control painful and do not produce a
  deployable application. However, we will use notebooks for evaluation and
  metrics reporting (separate from the application).

### 3. Pipeline Framework (Prefect / Airflow / Luigi)

Use an orchestration framework to manage layer execution.

- **Rejected because**: Adds a heavy dependency and operational complexity
  (scheduler, database, worker processes) without proportional benefit for a
  single-user demo application. Our lightweight EventBus achieves the decoupling
  benefits without the infrastructure cost.

### 4. Flat Script Architecture (No Module Boundaries)

Single `app.py` with all logic in one file.

- **Rejected because**: While maximally fast to start, it becomes unmaintainable
  after ~500 lines. Parallel development by team members becomes impossible.
  The judging criteria explicitly reward code quality under "Execution and
  communication" (30%), so clean organization directly impacts scoring.

---

## References

- MedGemma 1.5 Model Card (HuggingFace): CT/MRI preprocessing requirements
- RECIST 1.1 Guidelines: Target lesion selection and measurement rules
- iRECIST: Immunotherapy response criteria (confirmatory scan guidance)
- TumorTwin framework: Bidirectional update principle for digital twins
- PROTEAS Dataset (Scientific Data, Nature): Longitudinal brain metastases MRI
- Competition evaluation criteria: Kaggle MedGemma Impact Challenge overview page
