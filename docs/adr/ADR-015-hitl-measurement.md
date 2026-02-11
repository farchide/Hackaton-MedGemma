# ADR-015: Human-in-the-Loop Measurement Workflow

## Status

Accepted

## Date

2026-02-11

## Context

Tumor measurement in clinical practice is an inherently human activity. Radiologists
select which lesions to track, place calipers on specific slices, and make judgment
calls when anatomy is ambiguous. Fully automated pipelines that bypass this process
face two problems:

1. **Trust.** Clinicians will not adopt a tool whose measurements they cannot inspect,
   override, or understand. Black-box outputs are rejected at the point of care.
2. **Accuracy in edge cases.** Automated segmentation fails on ill-defined boundaries
   (e.g., liver lesions with poor contrast, post-treatment necrotic cores, lymph nodes
   adjacent to vessels). Human correction is not optional -- it is necessary for
   clinical-grade measurements.

Our competitive advantage in the MedGemma Impact Challenge is not raw segmentation
accuracy (many teams will use MedSAM or similar). It is the **workflow** -- making
the human-AI interaction feel like a natural extension of the radiologist's existing
practice, while capturing every action for auditability.

Key constraints:

- **Gradio as the UI framework.** We are committed to Gradio (ADR-009) for Kaggle
  compatibility. This limits us to components available in Gradio's ecosystem:
  ImageEditor, Slider, Button, Dropdown, and custom HTML/JS components.
- **RECIST 1.1 compliance.** The workflow must enforce (or at minimum, advise on)
  RECIST 1.1 rules: maximum 2 target lesions per organ, maximum 5 total targets,
  minimum 10 mm for measurable lesions, short-axis measurement for lymph nodes.
- **Longitudinal context.** The measurement tool must present prior timepoint
  measurements alongside the current scan to enable consistent tracking.

## Decision

We implement a **clinician-like interactive measurement workflow** as the core
differentiator of the Digital Twin Tumor system. The workflow follows six explicit
steps, each corresponding to a UI state and a logged action. Every human action
and automated suggestion is recorded with full provenance.

### Workflow Steps

#### Step 1: Case Upload and Preprocessing

**User action:** Upload DICOM folder or NIfTI file(s) for one or more timepoints.

**System behavior:**
- Run PHI gate (ADR-007) to strip protected health information.
- Preprocess: normalize intensity, resample to isotropic spacing, extract axial slices.
- Display the first timepoint's axial slices in the Gradio ImageEditor with a
  slice navigation slider.
- If multiple timepoints are uploaded, display them in chronological order in the
  timeline navigator.

**Logged:** `case_uploaded`, `preprocessing_completed`, scan metadata, spacing, dimensions.

#### Step 2: Select Target Lesions

**User action:** Click on lesions in the displayed slices to mark them as targets.
Each click places a numbered pin on the image.

**System behavior:**
- Enforce RECIST guardrails as advisory warnings (not hard blocks):
  - Warn if > 2 targets selected in the same organ.
  - Warn if > 5 total targets selected.
  - Warn if a selected lesion appears smaller than 10 mm.
- The user can override any warning with a single click ("Override: I confirm this
  selection"). Overrides are logged but permitted.
- Display a target lesion summary panel showing: lesion number, organ, slice number,
  estimated size (from quick auto-measurement).

**Logged:** `lesion_selected`, `recist_warning_issued`, `recist_warning_overridden`,
lesion coordinates, organ label, override reason (if provided).

#### Step 3: Measure Diameter

**User action:** For each target lesion, click two endpoints on the axial slice to
define the longest-axis diameter. For lymph nodes, also define the short-axis.

**System behavior:**
- Display a line overlay between the two clicked points with the measured distance
  in millimeters (computed using the scan's pixel spacing metadata).
- Auto-suggest measurement endpoints based on the segmentation mask (if already
  computed in Step 4) or on intensity gradients. The user can accept the suggestion
  with one click or place their own endpoints.
- Show the prior timepoint's measurement line overlaid in a different color for
  comparison (if longitudinal data is available).
- Compute and display: longest diameter (mm), short axis (mm, lymph nodes only),
  percentage change from prior timepoint.

**Logged:** `diameter_measured`, endpoint coordinates (pixel and mm), computed diameter,
measurement method (`manual` | `semi_auto` | `auto`), prior diameter for comparison.

#### Step 4: Segment Lesion

**User action:** Provide a segmentation prompt -- either a bounding box drawn around
the lesion, or a set of positive/negative point clicks -- to initialize MedSAM-based
segmentation. Then refine the resulting contour if needed.

**System behavior:**
- Run MedSAM (or SAM 2) with the user's prompt to generate a binary segmentation mask.
- Display the mask as an editable semi-transparent overlay on the slice.
- Provide tools for mask refinement:
  - **Brush:** paint to add regions to the mask.
  - **Eraser:** paint to remove regions from the mask.
  - **Threshold adjust:** slider to adjust the confidence threshold for the SAM output.
- Compute volume from the 3D mask (propagated across adjacent slices if the user
  confirms the 2D mask on the key slice).
- Display: mask area (mm^2 on current slice), estimated 3D volume (mm^3), Dice
  overlap with auto-generated mask (as a quality indicator).

**Logged:** `segmentation_prompted`, `segmentation_generated`, `mask_edited`,
prompt type (box/points), prompt coordinates, mask version number, edit tool used,
pixels changed.

#### Step 5: Lock Lesion Identity

**User action:** For longitudinal cases, confirm that the system's proposed lesion
match across timepoints is correct, or reassign it.

**System behavior:**
- Display side-by-side views of the same lesion across all available timepoints,
  aligned by registration.
- Show the system's proposed match with a confidence score. Matches below a
  configurable threshold (default: 0.7) are highlighted in amber for review.
- Provide a dropdown to reassign a lesion to a different prior-timepoint lesion or
  to mark it as "new" (not previously seen).
- Show a lesion identity graph: a visual timeline showing each tracked lesion as a
  horizontal line with measurements at each timepoint.

**Logged:** `identity_confirmed`, `identity_reassigned`, `identity_marked_new`,
proposed match ID, final match ID, confidence score, reason for reassignment (free text).

#### Step 6: Flag New Lesions

**User action:** Review system-detected candidate new lesions and confirm or dismiss
each one.

**System behavior:**
- Run automated new lesion detection by comparing current scan to prior timepoints
  (lesions present now but not matched to any prior lesion).
- Display candidates with bounding boxes and confidence scores.
- User clicks "Confirm New" or "Dismiss" for each candidate.
- Confirmed new lesions are added to the tracking graph and may trigger a RECIST
  response category of PD (progressive disease) if they are unequivocal.

**Logged:** `new_lesion_candidate_shown`, `new_lesion_confirmed`, `new_lesion_dismissed`,
candidate coordinates, confidence score, user decision.

### Audit Log Schema

Every action is recorded as a structured event:

```python
@dataclass(frozen=True)
class AuditEvent:
    event_id: str                   # UUID
    timestamp: datetime             # UTC
    user_id: str                    # Session identifier
    action_type: str                # One of the logged action types above
    patient_id: str
    timepoint_id: str | None
    lesion_id: str | None
    before_state: dict[str, Any]    # State before the action
    after_state: dict[str, Any]     # State after the action
    metadata: dict[str, Any]        # Action-specific data (coordinates, etc.)
```

Audit events are stored in an append-only list in session state and can be exported
as JSON for review. They are never deleted during a session.

### Edit History and Versioning

Every segmentation mask edit creates a new version. The system maintains a version
stack per lesion:

```python
@dataclass(frozen=True)
class MaskVersion:
    version: int
    mask: np.ndarray
    created_at: datetime
    created_by: Literal["auto", "user_edit"]
    parent_version: int | None
    change_summary: str             # e.g., "brush: added 142 pixels"
```

The user can revert to any previous version via a version history dropdown. Reverting
does not delete later versions; it creates a new version that duplicates the selected
historical state (branch-style history, not destructive undo).

### RECIST Auto-Classification

After all measurements are locked, the system computes the RECIST 1.1 response
category:

- **CR (Complete Response):** All target lesions disappeared.
- **PR (Partial Response):** Sum of diameters decreased by >= 30% from baseline.
- **PD (Progressive Disease):** Sum of diameters increased by >= 20% from nadir AND
  absolute increase >= 5 mm, OR unequivocal new lesions confirmed.
- **SD (Stable Disease):** Neither PR nor PD criteria met.

The computed category is displayed as an **advisory banner**, not as a final
determination. The human must click "Confirm Response" or select an override
category with a free-text reason. This ensures the system never makes a clinical
decision autonomously.

### UI Component Mapping

| Workflow Step | Primary Gradio Component | Supporting Components |
|---------------|--------------------------|----------------------|
| Case Upload | `gr.File` (multi-file) | `gr.Markdown` (status) |
| Slice Display | `gr.Image` / `gr.ImageEditor` | `gr.Slider` (slice nav) |
| Target Selection | `gr.ImageEditor` (point mode) | `gr.Dataframe` (summary table) |
| Diameter Measurement | Custom JS overlay on `gr.Image` | `gr.Number` (readout), `gr.Plot` (comparison) |
| Segmentation | `gr.ImageEditor` (brush/box mode) | `gr.Slider` (threshold), `gr.Button` (accept/redo) |
| Identity Lock | `gr.Gallery` (multi-timepoint) | `gr.Dropdown` (reassign), `gr.Plot` (identity graph) |
| New Lesion Flag | `gr.Gallery` (candidates) | `gr.Button` (confirm/dismiss) |
| RECIST Response | `gr.Markdown` (advisory banner) | `gr.Radio` (override category), `gr.Textbox` (reason) |
| Audit Trail | `gr.JSON` / `gr.Dataframe` | `gr.Button` (export) |

### Confidence and Provenance Display

Every measurement displayed in the UI carries a provenance badge:

- **Manual** (blue): Diameter endpoints or mask drawn entirely by the user.
- **Semi-auto** (green): System-suggested, user-confirmed or user-edited.
- **Auto** (amber): Fully automated, not yet reviewed by the user.

This transparency is critical for trust. Reviewers and judges can immediately see
which measurements were human-verified.

### Python Libraries

| Library | Role in HITL Workflow |
|---------|----------------------|
| gradio >= 4.10 | UI framework, ImageEditor, Blocks layout |
| numpy >= 1.26 | Array operations for masks and coordinates |
| Pillow >= 10.0 | Image manipulation for overlay rendering |
| matplotlib >= 3.8 | Measurement line rendering, identity graph plots |
| uuid (stdlib) | Audit event and lesion ID generation |
| datetime (stdlib) | Timestamp management |

## Consequences

### Positive

- **Clinical credibility.** The workflow mirrors how radiologists actually measure
  tumors, making the demo immediately recognizable to medical reviewers.
- **Full auditability.** Every measurement can be traced back to whether it was human
  or machine-generated, with exact timestamps and before/after states.
- **Error recovery.** Version history with non-destructive revert means no work is
  ever lost, even if the user makes mistakes.
- **RECIST compliance.** Guardrails prevent common protocol violations while remaining
  overridable (advisory, not blocking), respecting clinical autonomy.
- **Evaluation support.** The audit log directly feeds Layer 1 and Layer 2 evaluation
  metrics (ADR-013), since we know the provenance of every measurement.

### Negative

- **UI complexity.** Six workflow steps with multiple interaction modes require
  significant Gradio engineering, which is the most time-constrained resource in a
  hackathon.
- **Gradio limitations.** Some interactions (e.g., precise caliper placement) may feel
  imprecise in a browser-based tool compared to dedicated DICOM viewers like 3D Slicer.
  Mitigation: provide keyboard shortcuts and zoom controls.
- **State management overhead.** Maintaining version stacks, audit logs, and multi-
  timepoint state in Gradio's session state requires careful engineering to avoid
  memory leaks. Mitigation: cap version history at 50 versions per lesion and
  serialize to disk for long sessions.

### Risks

- If Gradio's ImageEditor does not support the overlay precision needed for caliper
  placement, we may need to fall back to a custom JavaScript component injected via
  `gr.HTML`. This is feasible but adds complexity. Mitigation: prototype the diameter
  tool in the first sprint.
- Audit log size could grow large for complex cases with many lesions and timepoints.
  Mitigation: export to file and clear in-memory log periodically; keep only the
  last 1000 events in the UI display.

## Alternatives Considered

### 1. Fully Automated Pipeline (No Human in the Loop)

Run segmentation and measurement end-to-end without user interaction. Rejected because
it eliminates the project's core differentiator and produces measurements that
clinicians cannot trust or verify. Automated results are still generated as
suggestions -- the HITL workflow wraps them with verification.

### 2. Desktop Application (3D Slicer Plugin)

Build the measurement tool as a 3D Slicer extension for precise caliper placement
and 3D visualization. Rejected because it cannot run in Kaggle, requires local
installation, and limits the audience to users who have 3D Slicer installed. We
acknowledge that a production system would benefit from integration with clinical
DICOM viewers.

### 3. Annotation-Only Workflow (No RECIST Logic)

Provide measurement tools without RECIST classification or guardrails. Rejected
because the RECIST classification is what connects individual measurements to
clinical decision-making. Without it, the tool is a generic annotation editor
rather than a clinical decision support system.

### 4. Blocking RECIST Enforcement

Prevent users from violating RECIST rules (e.g., hard-block selection of > 5
targets). Rejected because clinical practice sometimes requires deviation from
standard protocols (e.g., tracking a sixth lesion for research purposes). Advisory
warnings with logged overrides strike the right balance between guidance and
autonomy.

### 5. Separate Measurement and Review Phases

Split the workflow into a measurement phase (automated) and a review phase (human
edits). Rejected because this sequential approach is slower and less engaging than
an integrated workflow where the human and AI collaborate on each lesion in real
time. The integrated approach also produces richer audit trails since every
AI suggestion is immediately evaluated by the human.
