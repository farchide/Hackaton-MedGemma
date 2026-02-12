# ADR-004: Segmentation Model Stack

| Field       | Value                                      |
|-------------|--------------------------------------------|
| **Status**  | Accepted                                   |
| **Date**    | 2026-02-11                                 |
| **Authors** | Digital Twin Tumor Team                    |
| **Deciders**| Full team                                  |

---

## Context

The Digital Twin Tumor project requires volumetric segmentation of tumors from
longitudinal imaging studies (MRI for brain metastases, CT for lung lesions).
Segmentation is the foundational step: every downstream volume measurement,
growth-model fit, and counterfactual simulation depends on accurate, reproducible
delineation of tumor boundaries.

Three competing requirements shape the decision:

1. **Clinical auditability.** In regulated and research contexts, a clinician must
   be able to inspect, correct, and formally accept or reject every segmentation
   mask. Pure auto-segmentation without human oversight is insufficient for any
   pathway that touches clinical decision support.

2. **Scalability for the hackathon demo.** We need an automated baseline that can
   process an entire dataset (e.g., PROTEAS 40-patient cohort) without manual
   intervention, so reviewers can see population-level results.

3. **Reproducibility.** Every mask must carry provenance metadata indicating how
   it was produced, what model version generated it, and whether a human edited
   it. Longitudinal consistency matters: a segmentation drift between timepoints
   that is caused by model inconsistency rather than true biological change would
   corrupt growth-model fitting.

The project targets Kaggle notebook execution, so model weights must be
downloadable or bundled within the competition environment. GPU memory is
constrained to a single T4 or P100 (16 GB VRAM).

---

## Decision

We adopt a **two-model architecture with a human-in-the-loop review layer**:

### Primary: MedSAM (Interactive Segmentation)

MedSAM (Medical Segment Anything Model) serves as the primary segmentation tool.
It is a foundation model fine-tuned on over one million medical image-mask pairs
across multiple modalities.

**Interaction workflow:**

1. The user provides a prompt: a bounding box around the suspected lesion, or one
   or more click points (positive/negative).
2. MedSAM generates a candidate mask for the target slice.
3. The mask is propagated to adjacent slices using a slice-by-slice inference
   loop with the previous slice's mask as an additional prompt.
4. The user reviews the 3D mask in a lightweight viewer, edits individual slices
   if needed, and marks the segmentation as `accepted` or `rejected`.
5. Every accepted mask is stored as a versioned NIfTI file with provenance
   metadata attached as a sidecar JSON.

**Why MedSAM over vanilla SAM:**

- Pre-trained on medical imaging data; substantially better zero-shot
  performance on CT/MRI than the general-domain SAM checkpoint.
- Accepts the same prompt types (box, point) so the interaction model is
  identical.
- Weights are publicly available under Apache 2.0.

### Automated Baseline: nnU-Net

nnU-Net v2 serves as the fully automated baseline for benchmarking and for
generating initial masks on known datasets where pre-trained nnU-Net models
exist (e.g., brain tumor segmentation from BraTS-trained checkpoints).

**Role in the pipeline:**

- Run nnU-Net inference on the full dataset to produce auto-segmentation masks.
- Use these masks as the starting point for MedSAM-guided refinement when
  human review is available.
- Compare MedSAM-interactive masks against nnU-Net-auto masks to quantify
  the value of human-in-the-loop correction (Dice, Hausdorff distance).

### Fallback: SAM2

SAM2 (Segment Anything Model 2) is retained as a fallback for non-medical
imaging contexts or for cases where MedSAM weights cannot be loaded in the
execution environment. SAM2 provides video-aware segmentation that could be
useful for tracking lesions across a temporal sequence of slices, but its
medical-domain performance is inferior to MedSAM without fine-tuning.

---

## Mask Storage and Versioning

### Storage Format

All segmentation masks are stored as **compressed NIfTI files** (`.nii.gz`).
NIfTI is chosen over DICOM-SEG for simplicity and broad library support in
the Python scientific ecosystem.

Each mask file is accompanied by a sidecar JSON with the following schema:

```json
{
  "mask_id": "uuid-v4",
  "patient_id": "PROTEAS-012",
  "timepoint": "2024-03-15",
  "version": 3,
  "parent_version": 2,
  "method": "medsam_interactive",
  "model_checkpoint": "medsam_vit_b_20230423",
  "prompts": [
    {"type": "box", "slice": 45, "coords": [120, 80, 200, 160]}
  ],
  "reviewer": "clinician_A",
  "status": "accepted",
  "created_at": "2026-02-11T14:32:00Z",
  "dice_vs_previous_version": 0.94,
  "volume_mm3": 4523.7
}
```

### Versioning Strategy

Every edit to a mask creates a **new version**. Versions are stored as
independent NIfTI files, not deltas, to ensure each version is self-contained
and can be loaded without replaying a chain of edits.

A lightweight diff is computed between consecutive versions:

- **Voxel-level diff:** number of voxels added, removed, and unchanged.
- **Dice coefficient** between version N and version N-1.
- **Volume delta** in mm^3.

This diff metadata is stored in the sidecar JSON and enables auditing of how
a segmentation evolved through the review process.

File naming convention:

```
data/processed/{dataset_name}/{patient_id}/{timepoint}/
    mask_v001.nii.gz
    mask_v001.json
    mask_v002.nii.gz
    mask_v002.json
    mask_final.nii.gz -> symlink to accepted version
```

---

### Vector-Indexed Segmentation Embeddings

In addition to file-based mask storage, segmentation embeddings are indexed in
RuVector for similarity-based retrieval. Each accepted mask is processed to
extract a compact vector representation -- either a MedSigLIP embedding of the
masked region or a shape descriptor vector (volume, surface area, sphericity,
elongation, compactness, centroid position) -- and stored in RuVector's HNSW
index.

This enables three capabilities:

**(a) Historical similarity lookup for quality comparison.** When a clinician
reviews a new segmentation, the system retrieves the k most similar historical
masks (by shape or visual appearance) and displays them alongside. This
provides context: "This mask is consistent with previous segmentations of
similar-sized lesions" or "This mask is unusually elongated compared to
historical cases." This is particularly useful for catching segmentation
errors that are geometrically plausible but clinically unusual.

**(b) Anomalous segmentation detection.** By computing the distance from a new
mask embedding to its nearest neighbors in the HNSW index, the system can flag
outlier segmentations. A mask whose nearest-neighbor distance exceeds a
threshold (calibrated from the existing corpus) triggers a warning:
"This segmentation appears unusual -- please verify." This serves as an
automated quality gate before the mask is accepted and passed to downstream
growth model fitting.

**(c) Evidence panel population.** The MedGemma evidence panel (Layer 6, see
ADR-003) can be populated with visually similar segmented regions from other
patients or other timepoints. RuVector's Cypher graph queries enable
relationship-aware retrieval: for example, "find masks from the same organ
with similar volume that showed treatment response."

```python
# Index a segmentation mask embedding in RuVector
from ruvector import VectorDB, VectorEntry, SearchQuery

mask_db = VectorDB(dimensions=128)  # Shape descriptor dimension

# After mask acceptance, compute shape descriptor and index
shape_vector = compute_shape_descriptor(mask_array, spacing)
mask_db.insert(VectorEntry(
    id=f"{patient_id}_{timepoint}_{lesion_id}_v{version}",
    vector=shape_vector.tolist(),
    metadata={
        "patient_id": patient_id,
        "lesion_id": lesion_id,
        "timepoint": timepoint,
        "volume_mm3": volume,
        "method": provenance.method,
        "status": "accepted",
    }
))

# Retrieve similar masks for quality comparison
similar = mask_db.search(SearchQuery(
    vector=shape_vector.tolist(), k=5, include_vectors=False
))
```

---

## Python Libraries

| Library              | Version   | Purpose                                      |
|----------------------|-----------|----------------------------------------------|
| `segment-anything`   | >=1.0     | SAM/SAM2 inference engine                    |
| MedSAM (custom)      | latest    | Medical SAM checkpoint and wrapper utilities |
| `nnunetv2`           | >=2.2     | Automated segmentation baseline              |
| `monai`              | >=1.3     | Medical image transforms, metrics (Dice, HD) |
| `SimpleITK`          | >=2.3     | NIfTI I/O, resampling, registration          |
| `nibabel`            | >=5.0     | NIfTI read/write as a lighter alternative    |
| `numpy`              | >=1.24    | Array operations                             |
| `torch`              | >=2.0     | Model inference backend                      |
| `ruvector`           | >=0.1     | Vector indexing for mask embeddings and similarity retrieval |

---

## Consequences

### Positive

- **Auditability.** Every mask has a clear provenance chain: which model, which
  prompts, which human reviewer, and what edits were applied. This is essential
  for any regulated or research context.
- **Flexibility.** The two-model stack allows fully automated processing for
  large-scale demos and careful human-guided segmentation for high-stakes cases.
- **Reproducibility.** Versioned masks with sidecar metadata mean any result can
  be traced back to the exact segmentation that produced it.
- **Kaggle compatibility.** Both MedSAM and nnU-Net can run on a single T4 GPU.
  MedSAM's ViT-B variant requires approximately 1.5 GB VRAM for inference.
- **"Find similar lesion" for QA.** RuVector-indexed segmentation embeddings
  enable automatic retrieval of historically similar masks, providing a
  quality assurance mechanism that flags anomalous segmentations before they
  propagate to growth model fitting.

### Negative

- **Complexity.** Maintaining two segmentation backends increases code surface
  and testing burden.
- **Storage overhead.** Storing every mask version as a full NIfTI file is
  space-inefficient. For the hackathon scale (tens of patients, a few timepoints
  each), this is acceptable. For production, delta-based storage would be needed.
- **MedSAM limitations.** MedSAM was trained on 2D slices; 3D consistency
  requires our own slice-propagation logic, which may introduce artifacts at
  volume boundaries.

---

## Alternatives Considered

### 1. Pure nnU-Net (Fully Automated)

Rejected because fully automated segmentation without human review is
inappropriate for a project that emphasizes clinical auditability. nnU-Net
produces excellent results on in-distribution data but provides no mechanism
for clinician interaction or correction.

### 2. Pure Manual Segmentation with ITK-SNAP

Rejected because manual segmentation does not scale to the hackathon demo
requirements. Processing 40 patients with multiple timepoints manually is
infeasible within the competition timeline.

### 3. MONAI Auto3DSeg

Considered as an alternative automated pipeline. Rejected in favor of nnU-Net
because nnU-Net has a longer track record in medical segmentation challenges,
pre-trained models are more widely available, and the self-configuring pipeline
reduces setup time. MONAI's segmentation transforms and metrics are still used
as utilities.

### 4. SAM2 as Primary (Instead of MedSAM)

Rejected because SAM2's training data is dominated by natural images and video.
While SAM2 has stronger temporal consistency features, its zero-shot performance
on medical imaging is substantially worse than MedSAM. Fine-tuning SAM2 on
medical data within the hackathon timeline is not feasible.

### 5. Storing Masks as DICOM-SEG

Rejected for the hackathon phase. DICOM-SEG is the standard for clinical PACS
integration, but the Python tooling (`highdicom`) adds complexity without
benefit in a Kaggle notebook environment. NIfTI with sidecar JSON is simpler
and sufficient. Migration to DICOM-SEG can be added post-hackathon if clinical
deployment is pursued.

### 6. RuVector for Mask Similarity

Use RuVector's HNSW index to store segmentation mask embeddings (shape
descriptors and/or MedSigLIP visual embeddings) for similarity-based retrieval,
anomaly detection, and cross-patient evidence panel population.

- **Accepted** because:
  - **Quality assurance**: Similarity search over historical masks provides an
    automated quality gate. Anomalous segmentations (unusually shaped, wrong
    organ boundary, segmentation leak) can be detected by their distance from
    the nearest-neighbor cluster in embedding space.
  - **Cross-patient context**: Clinicians benefit from seeing similar lesions
    from other patients, especially for rare lesion morphologies. RuVector's
    Cypher graph queries enable relationship-aware retrieval (e.g., same organ,
    similar stage, same treatment).
  - **Zero GPU cost**: RuVector runs on CPU, so mask embedding indexing and
    retrieval do not compete with MedSAM or MedGemma for GPU VRAM.
  - **GNN improvement**: RuVector's GNN layers learn from retrieval patterns,
    improving the relevance of similar-mask suggestions over time as more cases
    are processed and clinician feedback is recorded.
  - **Alternatives considered**: FAISS provides fast HNSW search but lacks
    graph queries and GNN self-improvement. Storing embeddings in PostgreSQL
    with pgvector provides basic similarity search but without graph-aware
    retrieval or self-improving index quality.

---

## References

- Ma, J., et al. "Segment Anything in Medical Images." Nature Communications, 2024.
- Isensee, F., et al. "nnU-Net: a self-configuring method for deep learning-based
  biomedical image segmentation." Nature Methods, 2021.
- Kirillov, A., et al. "Segment Anything." ICCV, 2023.
- MONAI Project. https://monai.io/
