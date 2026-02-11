# ADR-009: RECIST/iRECIST Response Criteria Implementation

## Status

Accepted

## Date

2026-02-11

## Context

Tumor response assessment in clinical trials follows standardized criteria that define how
lesions are measured, tracked, and classified. The two relevant standards for this project
are:

- **RECIST 1.1** (Response Evaluation Criteria In Solid Tumors, version 1.1): The dominant
  standard for solid tumor response assessment in clinical trials. Defines rules for target
  lesion selection, measurement methodology, and response categorization.

- **iRECIST** (immune RECIST): An extension of RECIST 1.1 designed to handle the unique
  response patterns seen with immunotherapy (pseudoprogression, delayed response). Introduces
  unconfirmed and confirmed progression categories with a confirmatory scan requirement.

Anchoring the Digital Twin Tumor system to RECIST/iRECIST provides several strategic benefits:

1. **Clinical legibility**: Oncologists and radiologists immediately understand RECIST-based
   outputs. This makes the demo instantly credible to the hackathon judges, several of whom
   are clinical research scientists at Google.
2. **Structured measurement logic**: RECIST provides unambiguous rules for what to measure
   and how, eliminating ad-hoc measurement decisions.
3. **Response vocabulary**: CR/PR/SD/PD categories provide a shared language for the
   MedGemma narrative generation layer.
4. **Evaluation framework**: The PROTEAS dataset provides segmented lesions with longitudinal
   follow-ups, enabling validation of our RECIST classification against derived ground truth.

Critical safety constraint: iRECIST explicitly states that these guidelines are intended for
consistent data handling in clinical trials and are **not intended to guide clinical treatment
decisions**. Our system must mirror this limitation prominently in all outputs.

## Decision

We implement RECIST 1.1 measurement logic as the core response assessment engine, with an
optional iRECIST extension for immunotherapy-treated cases. The implementation uses Python
dataclasses for type safety and auditability, with automated classification supplemented
by mandatory manual override capability.

### RECIST 1.1 Core Implementation

#### Target Lesion Selection Rules

```python
RECIST_CONSTRAINTS = {
    "max_target_lesions_total": 5,
    "max_target_lesions_per_organ": 2,
    "min_measurable_size_mm": 10.0,          # Non-nodal: >= 10mm longest diameter
    "min_measurable_node_short_axis_mm": 15.0,  # Nodal: >= 15mm short axis
    "node_normal_threshold_mm": 10.0,         # Nodes < 10mm short axis = normal
}
```

Selection logic:
1. At baseline, identify all measurable lesions (non-nodal >= 10mm longest diameter,
   nodal >= 15mm short axis).
2. Select up to 2 target lesions per organ, up to 5 total, prioritizing:
   - Largest lesions (most reliably measurable)
   - Lesions representative of all involved organs
   - Lesions amenable to reproducible measurement (well-defined borders)
3. All remaining measurable and non-measurable lesions are classified as non-target.
4. The system **suggests** target lesion selection but the user **must confirm** via the
   HITL interface. The 2-per-organ / 5-total constraint is enforced as a default guardrail
   with explicit override for research use cases.

#### Measurement Methodology

```python
@dataclass
class LesionMeasurement:
    lesion_id: str                    # Unique, stable identifier across timepoints
    timepoint_id: str                 # Study date or timepoint label
    lesion_type: str                  # "non_nodal" | "nodal" | "new"
    organ: str                        # Anatomical location
    is_target: bool                   # Target vs non-target classification

    # Primary measurement
    longest_diameter_mm: Optional[float]    # Non-nodal target: longest diameter
    short_axis_mm: Optional[float]         # Nodal target: short axis perpendicular to longest
    measurement_method: str                # "manual" | "semi_auto" | "auto"

    # Measurement endpoints (for auditability)
    endpoint_1_ras: Optional[Tuple[float, float, float]]  # RAS+ coordinates
    endpoint_2_ras: Optional[Tuple[float, float, float]]  # RAS+ coordinates
    slice_index: int                       # Axial slice where measurement was taken

    # Segmentation-derived (optional, enriched)
    volume_mm3: Optional[float]            # From segmentation mask voxel count * voxel volume
    segmentation_source: Optional[str]     # "MedSAM" | "nnUNet" | "manual" | "edited"

    # Uncertainty (from ADR-008)
    measurement_sigma_mm: float            # Total measurement uncertainty
    confidence_flag: str                   # "high" | "medium" | "low"

    # Audit trail
    user_edited: bool                      # Whether user modified the auto measurement
    timestamp: str                         # ISO 8601 timestamp of measurement
```

Measurement rules:
- **Non-nodal lesions**: Longest diameter in the axial plane. The system auto-computes
  this from the segmentation mask (maximum Feret diameter) and presents it for user
  confirmation/adjustment.
- **Nodal lesions**: Short axis perpendicular to the longest axis. The system computes
  both axes and highlights the short axis measurement.
- **Non-measurable / too small to measure**: Recorded as "present" with no numeric
  measurement, or recorded as 0mm if completely disappeared (CR for that lesion).
- **New lesions**: Detected by the lesion tracking system or flagged by the user. New
  lesions are recorded but do not enter the sum of diameters calculation for existing
  targets. Their presence affects overall response (potential PD).

#### Sum of Diameters and Response Categories

```python
@dataclass
class TimePointResponse:
    timepoint_id: str
    target_measurements: List[LesionMeasurement]
    non_target_assessment: str          # "CR" | "non-CR/non-PD" | "PD" | "not_evaluated"
    new_lesions_detected: bool
    new_lesion_details: List[LesionMeasurement]

    # Computed values
    sum_of_diameters_mm: float          # Sum of LD (non-nodal) + SA (nodal) for targets
    baseline_sum_mm: float              # Sum at baseline (for PR calculation)
    nadir_sum_mm: float                 # Smallest sum recorded so far (for PD calculation)
    percent_change_from_baseline: float # For PR: must be <= -30%
    percent_change_from_nadir: float    # For PD: must be >= +20%
    absolute_change_from_nadir_mm: float  # For PD: must also be >= +5mm

    # Classification
    target_response: str                # "CR" | "PR" | "SD" | "PD"
    overall_response: str               # Integrating target + non-target + new lesions
    response_auto_classified: bool      # True if auto-classified, False if manually set
    response_override_reason: Optional[str]  # If user overrode auto classification
```

Response classification logic:

```python
def classify_target_response(tp: TimePointResponse) -> str:
    """RECIST 1.1 target lesion response classification."""
    # CR: All target lesions disappeared (non-nodal = 0mm, nodal < 10mm SA)
    all_non_nodal_gone = all(
        m.longest_diameter_mm == 0.0
        for m in tp.target_measurements if m.lesion_type == "non_nodal"
    )
    all_nodes_normal = all(
        m.short_axis_mm < RECIST_CONSTRAINTS["node_normal_threshold_mm"]
        for m in tp.target_measurements if m.lesion_type == "nodal"
    )
    if all_non_nodal_gone and all_nodes_normal:
        return "CR"

    # PD: >= 20% increase from nadir AND >= 5mm absolute increase
    if (tp.percent_change_from_nadir >= 20.0 and
            tp.absolute_change_from_nadir_mm >= 5.0):
        return "PD"

    # PR: >= 30% decrease from baseline
    if tp.percent_change_from_baseline <= -30.0:
        return "PR"

    # SD: Neither PR nor PD
    return "SD"
```

Overall response integrates target, non-target, and new lesion assessments:

| Target | Non-Target | New Lesions | Overall Response |
|---|---|---|---|
| CR | CR | No | CR |
| CR | non-CR/non-PD | No | PR |
| PR | non-PD | No | PR |
| SD | non-PD | No | SD |
| PD | any | any | PD |
| any | PD | any | PD |
| any | any | Yes | PD |

### iRECIST Extension (Optional, Immunotherapy Cases)

#### Unconfirmed vs Confirmed Progression

iRECIST modifies the progression logic to account for pseudoprogression under immunotherapy:

```python
@dataclass
class iRECISTAssessment:
    timepoint_id: str
    recist_response: str               # Standard RECIST classification
    irecist_response: str              # "iCR" | "iPR" | "iSD" | "iUPD" | "iCPD"
    prior_iupd_timepoint: Optional[str]  # If confirming a prior iUPD
    confirmation_window_weeks: int     # Expected: 4-8 weeks
    confirmation_scan_due: Optional[str]  # Date by which confirmatory scan is needed
    is_immunotherapy_active: bool       # Whether patient is on immunotherapy
```

iRECIST progression flow:
1. If RECIST classifies PD on a first occurrence during immunotherapy:
   classify as **iUPD** (immune Unconfirmed Progressive Disease).
2. Schedule a confirmatory scan at 4-8 weeks.
3. At confirmatory scan:
   - If progression is confirmed (further increase or no decrease): classify as **iCPD**
     (immune Confirmed Progressive Disease).
   - If lesions have decreased back to SD/PR/CR levels: reclassify as **iSD/iPR/iCR**.
4. If new lesions appear at the first assessment: still classify as iUPD (not iCPD) and
   require confirmation.

```python
def classify_irecist(current: TimePointResponse, history: List[TimePointResponse],
                     therapy_log: TherapyLog) -> iRECISTAssessment:
    recist = classify_target_response(current)

    if not therapy_log.is_immunotherapy_active(current.timepoint_id):
        # Not on immunotherapy: iRECIST mirrors RECIST
        return iRECISTAssessment(
            irecist_response="i" + recist,
            # ... other fields
        )

    if recist == "PD":
        # Check if there was a prior iUPD
        prior_iupd = find_prior_iupd(history)
        if prior_iupd is None:
            # First PD on immunotherapy: unconfirmed
            return iRECISTAssessment(
                irecist_response="iUPD",
                confirmation_window_weeks=6,  # 4-8 weeks, default 6
                # ...
            )
        else:
            # Confirmatory scan shows continued/worsened PD
            return iRECISTAssessment(
                irecist_response="iCPD",
                prior_iupd_timepoint=prior_iupd.timepoint_id,
                # ...
            )
    else:
        # Reset: if prior iUPD existed but now not PD, reclassify
        return iRECISTAssessment(
            irecist_response="i" + recist,
            # ...
        )
```

### Safety Constraints

Every output that includes response classification must carry the following:

1. **Prominent disclaimer**: Displayed in the UI header of any response classification panel:
   > "Response classifications are computed for research and educational purposes only.
   > iRECIST criteria are designed for consistent clinical trial data handling and are
   > explicitly NOT intended to guide clinical treatment decisions (per iRECIST guidelines).
   > This system has not been validated for clinical use."

2. **MedGemma narrative safety**: When MedGemma generates a tumor board summary, the prompt
   template includes a hard constraint:
   ```
   SAFETY: You must include the statement "This assessment is for research purposes
   only and should not be used for clinical decision-making" in every response that
   includes response classification. Do not provide treatment recommendations.
   ```

3. **UI visual safety cues**:
   - Response classifications displayed with an amber "RESEARCH USE ONLY" badge.
   - iRECIST classifications additionally flagged with "NOT FOR CLINICAL DECISIONS".
   - Manual override button always visible with audit logging of any override.

### Validation Against PROTEAS Dataset

The PROTEAS brain metastases dataset provides longitudinal segmentation masks from which
we can derive diameter measurements and validate our RECIST classification pipeline:

1. **Measurement validation**: Compute longest diameter from PROTEAS ground-truth masks
   and compare against our pipeline's auto-measured diameters. Report:
   - Mean absolute error (MAE) in mm
   - Correlation coefficient
   - Bland-Altman plot (agreement analysis)

2. **Classification validation**: Derive RECIST response categories from PROTEAS
   ground-truth measurements and compare against our pipeline's automated classifications.
   Report:
   - Confusion matrix (CR/PR/SD/PD)
   - Cohen's kappa for inter-rater agreement (treating ground-truth as reference)
   - Per-category precision and recall

3. **Longitudinal tracking validation**: Verify that lesion identities are maintained
   correctly across timepoints (no ID switches) on the PROTEAS cases.

### Data Model Summary

```python
@dataclass
class ResponseAssessment:
    """Complete response assessment for a patient across all timepoints."""
    patient_id: str
    target_lesions: List[str]              # Lesion IDs selected as targets
    baseline_timepoint: str
    timepoint_responses: List[TimePointResponse]
    irecist_enabled: bool
    irecist_assessments: Optional[List[iRECISTAssessment]]

    # Best overall response
    best_response: str                     # Best response achieved during follow-up
    best_response_timepoint: str

    # Audit
    created_by: str                        # User who initiated the assessment
    last_modified: str                     # ISO 8601 timestamp
    assessment_version: int                # Incremented on each modification
```

### UI Design for Response Classification

1. **Color-coded response badges**:
   - CR (Complete Response): Green
   - PR (Partial Response): Blue
   - SD (Stable Disease): Gray
   - PD (Progressive Disease): Red
   - iUPD (Unconfirmed Progression): Orange with "UNCONFIRMED" label
   - iCPD (Confirmed Progression): Red with "CONFIRMED" label

2. **Waterfall plot**: Per-lesion percent change from baseline, sorted by magnitude.
   Standard oncology visualization for comparing individual lesion responses.

3. **Spider plot**: Per-lesion percent change over time, showing heterogeneous response
   patterns (some lesions shrinking while others grow).

4. **Sum of diameters timeline**: Line plot with RECIST thresholds overlaid:
   - Dashed green line at -30% (PR threshold)
   - Dashed red line at +20% (PD threshold)
   - Nadir marker highlighted

5. **Manual override panel**: For each timepoint, the user can:
   - Override the automated response classification with a dropdown
   - Provide a free-text reason (required for override)
   - All overrides are logged in the audit trail

## Consequences

### Positive

- **Clinical credibility**: RECIST alignment makes the system immediately recognizable to
  oncology professionals. This directly supports the "product feasibility" judging criterion.
- **Structured measurements**: The RECIST framework eliminates ambiguity about what to
  measure and how, producing consistent data for the digital twin growth models.
- **Auditability**: Every measurement, classification, and override is logged with full
  provenance, supporting the "human-in-the-loop" design philosophy.
- **iRECIST readiness**: The optional immunotherapy extension demonstrates awareness of
  modern response assessment challenges, adding depth to the submission.
- **Evaluation clarity**: RECIST classifications provide a discrete, well-defined output
  that can be objectively evaluated against ground truth.

### Negative

- **RECIST limitations**: RECIST measures only unidimensional diameter, which may not capture
  complex morphological changes (e.g., necrosis, cavitation). Volume measurements from
  segmentation masks are provided as supplementary data but are not part of RECIST.
- **Disease-specific gaps**: RECIST 1.1 is designed for solid tumors. Brain metastases
  (our primary dataset) are sometimes assessed with RANO criteria instead. We document this
  limitation and note that RECIST is used as a general framework for the hackathon.
- **iRECIST complexity**: The confirmation logic adds state management complexity to the
  assessment pipeline. Mitigated by clear dataclass structure and explicit state transitions.
- **Small target count**: With max 5 target lesions, the sum of diameters can be dominated
  by a single large lesion, potentially missing mixed responses.

### Risks

- Users may misinterpret automated RECIST classifications as clinically valid assessments.
  Mitigated by prominent disclaimers, amber badges, and the mandatory safety language in
  MedGemma narratives.
- The PROTEAS dataset uses brain MRI, where RANO criteria are more standard than RECIST.
  Our RECIST implementation may not perfectly align with how these lesions would be assessed
  clinically. This is documented as a known limitation in the evaluation notebook.

## Alternatives Considered

### 1. RANO Criteria (Response Assessment in Neuro-Oncology)

Considered because the primary dataset (PROTEAS) involves brain metastases. RANO uses
bidimensional measurements (product of longest diameter and perpendicular) rather than
unidimensional. Rejected because: (a) RECIST 1.1 is more widely known and applicable
across tumor types, (b) the unidimensional approach is simpler to implement and validate,
and (c) brain metastases from extracranial primaries are often still assessed with RECIST
in trial settings. RANO support is documented as a post-hackathon extension.

### 2. WHO Criteria (Bidimensional)

Rejected. The older WHO response criteria use bidimensional measurements and have been
largely superseded by RECIST 1.1 in modern clinical trials. Implementing WHO criteria
would add complexity without adding credibility.

### 3. Volumetric Response Criteria

Considered as a more modern approach that leverages our segmentation capabilities. Volumetric
criteria are not yet standardized and different thresholds exist in the literature (e.g.,
65% volume decrease for PR vs 30% diameter decrease in RECIST). We compute and display
volumetric measurements as supplementary data but use RECIST as the primary classification
framework because it is the established standard that judges will recognize.

### 4. Fully Automated Classification Without Manual Override

Rejected. Fully automated RECIST classification without human oversight would be
irresponsible and would undermine the human-in-the-loop design philosophy that is central
to the project. The spec explicitly states that "automation can be partial; reliability and
auditability wins in regulated contexts." Every classification is presented as a suggestion
with a mandatory confirm/override mechanism.

### 5. Skip iRECIST, Implement RECIST Only

Considered for simplicity. Rejected because immunotherapy response patterns are one of the
most clinically relevant challenges in oncology today, and demonstrating awareness of
pseudoprogression and confirmation logic significantly strengthens the submission. The
iRECIST extension is implemented as an optional module that activates only when the therapy
log indicates immunotherapy, keeping the default path simple.
