# ADR-007: Medical Imaging Processing Pipeline

## Status

Accepted

## Date

2026-02-11

## Context

The Digital Twin Tumor system must ingest medical imaging data from heterogeneous sources
-- CT scans, MRI sequences across multiple weightings, and potentially mixed vendor formats --
and produce standardized, analysis-ready representations. The downstream consumers of this
pipeline include: (a) the segmentation models (MedSAM, nnU-Net), (b) the measurement engine
for RECIST/iRECIST diameter and volume extraction, (c) MedGemma for visual reasoning on
representative slices, and (d) the 3D visualization layer for clinical context.

The primary datasets for this hackathon are:

- **PROTEAS brain metastases MRI dataset** (Scientific Data): T1, T1-CE, T2, FLAIR sequences
  with longitudinal follow-ups at 6 weeks, 3/6/9/12 months.
- **RIDER Lung CT** (TCIA): same-day repeat CT scans for measurement uncertainty calibration.
- **Synthetic longitudinal augmentations**: morphologically modified masks from real cases.

Key constraints shaping this decision:

1. MedGemma 1.5 accepts 2D image inputs (slices) and has expanded 3D CT/MRI support, but
   preprocessing is required per Google's documentation. Full 3D volume inference is not
   practical within the hackathon timeline.
2. Patient data privacy must be enforced even when using de-identified public datasets, as
   the system is designed to accept arbitrary DICOM uploads.
3. Multiple MRI sequences require sequence-aware normalization -- a single normalization
   strategy would destroy contrast information critical for lesion identification.
4. Processing must be fast enough for interactive demo use (target: < 30 seconds per volume
   on a single GPU workstation).

## Decision

We adopt a standardized, modality-aware preprocessing pipeline with the following stages,
implemented as a deterministic, reproducible Python pipeline with full metadata provenance.

### 1. PHI Scanning Gate (Mandatory First Stage)

Before any pixel data is read, DICOM files pass through a PHI scanning gate:

```python
PHI_TAGS_TO_CHECK = [
    (0x0010, 0x0010),  # PatientName
    (0x0010, 0x0020),  # PatientID
    (0x0010, 0x0030),  # PatientBirthDate
    (0x0010, 0x1000),  # OtherPatientIDs
    (0x0008, 0x0080),  # InstitutionName
    (0x0008, 0x0081),  # InstitutionAddress
    (0x0008, 0x1070),  # OperatorsName
    (0x0008, 0x0090),  # ReferringPhysicianName
    (0x0010, 0x1001),  # OtherPatientNames
]
```

- If any PHI tag contains non-anonymized data (heuristic: non-empty string that is not a
  known anonymization placeholder like "ANONYMOUS" or "DEIDENTIFIED"), the pipeline halts
  and returns an error with the specific tags flagged.
- NIfTI files bypass PHI scanning (they do not carry DICOM header metadata by design).
- This gate implements the "do not upload PHI" safety requirement from the project spec,
  aligned with HIPAA Safe Harbor de-identification guidance.

### 2. Format Ingestion

| Input Format | Library | Conversion Target |
|---|---|---|
| DICOM series | `pydicom` + `SimpleITK` | `SimpleITK.Image` |
| NIfTI (.nii, .nii.gz) | `nibabel` | `SimpleITK.Image` (via numpy intermediary) |
| DICOM-SEG (segmentation objects) | `pydicom-seg` | Binary mask as `numpy.ndarray` |

Metadata extracted during ingestion and carried through the pipeline:

```python
@dataclass
class VolumeMetadata:
    patient_id: str            # De-identified ID only
    study_date: str            # YYYYMMDD
    series_uid: str
    modality: str              # CT | MR
    sequence_type: Optional[str]  # T1 | T1CE | T2 | FLAIR (MRI only)
    manufacturer: str
    slice_thickness_mm: float
    pixel_spacing_mm: Tuple[float, float]
    image_orientation: str     # Original orientation before reorientation
    window_center: Optional[float]   # CT only
    window_width: Optional[float]    # CT only
    magnetic_field_strength: Optional[float]  # MRI only, Tesla
    repetition_time_ms: Optional[float]       # MRI only
    echo_time_ms: Optional[float]             # MRI only
    original_shape: Tuple[int, int, int]
    affine_transform: np.ndarray  # 4x4 affine matrix
```

### 3. Orientation Standardization

All volumes are reoriented to **RAS+** (Right-Anterior-Superior) orientation:

- Use `SimpleITK.DICOMOrient(image, "RAS")` for DICOM-sourced images.
- Use `nibabel.as_closest_canonical()` followed by explicit RAS reorientation for NIfTI.
- The original orientation is recorded in metadata for provenance.
- All downstream coordinate systems (lesion centroids, bounding boxes, measurement
  endpoints) operate in RAS+ space.

### 4. Resampling to Isotropic Voxels

All volumes are resampled to isotropic voxel spacing:

- **CT volumes**: resample to 1.0 mm isotropic using B-spline interpolation (order 3)
  for intensity, nearest-neighbor for associated masks.
- **MRI volumes**: resample to 1.0 mm isotropic (matching PROTEAS dataset native resolution
  where possible; 0.5 mm for high-resolution sequences if memory permits).
- Resampling uses `SimpleITK.Resample()` with the following settings:
  - Interpolator: `sitkBSpline` for intensity, `sitkNearestNeighbor` for labels
  - Default pixel value: minimum intensity value of the volume
  - Output direction: identity (RAS+ aligned)

### 5. Modality-Specific Intensity Normalization

#### CT Normalization

CT values are in Hounsfield Units (HU) and require windowing for different tissue contexts:

```python
CT_WINDOW_PRESETS = {
    "soft_tissue": {"center": 40, "width": 400},    # [-160, 240] HU
    "lung":        {"center": -600, "width": 1500},  # [-1350, 150] HU
    "bone":        {"center": 400, "width": 1800},   # [-500, 1300] HU
    "brain":       {"center": 40, "width": 80},      # [0, 80] HU
    "liver":       {"center": 60, "width": 160},     # [-20, 140] HU
}
```

Processing steps:
1. Clip raw HU values to [-1024, 3071] to remove scanner artifacts.
2. Apply the appropriate window preset based on anatomical region (auto-detected from
   DICOM BodyPartExamined tag, with manual override).
3. Rescale windowed values to [0.0, 1.0] float32.
4. Store the raw HU array alongside windowed arrays (multiple windows can be generated
   for the same volume to support different visualization contexts).

#### MRI Sequence-Aware Normalization

MRI intensities are arbitrary and sequence-dependent. We apply sequence-specific normalization:

```python
MRI_NORMALIZATION = {
    "T1":   {"method": "percentile", "low": 0.5, "high": 99.5, "bias_correct": True},
    "T1CE": {"method": "percentile", "low": 0.5, "high": 99.5, "bias_correct": True},
    "T2":   {"method": "percentile", "low": 1.0, "high": 99.0, "bias_correct": True},
    "FLAIR": {"method": "percentile", "low": 1.0, "high": 99.0, "bias_correct": True},
}
```

Processing steps:
1. **Bias field correction**: Apply N4ITK bias field correction via
   `SimpleITK.N4BiasFieldCorrectionImageFilter()` to compensate for B1 field inhomogeneity.
   This is critical for consistent intensity comparison across timepoints.
2. **Brain extraction** (MRI only): Optional skull stripping using a precomputed brain mask
   or simple thresholding for the PROTEAS dataset (which provides brain masks).
3. **Percentile normalization**: Clip to [p_low, p_high] percentiles of non-zero voxels,
   then rescale to [0.0, 1.0] float32.
4. **Z-score normalization** (alternative for nnU-Net input): zero-mean, unit-variance
   within the foreground mask, stored as a secondary representation.

Sequence type is detected from DICOM tags (SequenceName, ScanningSequence, ProtocolName)
with a heuristic classifier, or explicitly set from dataset metadata for known datasets.

### 6. Slice and Patch Selection for MedGemma

MedGemma processes 2D slices, not full 3D volumes. Our selection policy:

1. **Lesion-centric selection** (primary): For each tracked lesion, select the axial slice
   at the lesion centroid (computed from the segmentation mask or manual annotation).
   Additionally select slices at centroid +/- N slices (default N=2) for context.

2. **Volume-representative selection** (fallback, no lesion annotations): Select axial
   slices at 25th, 50th, and 75th percentiles of the volume's z-extent, filtered to
   exclude slices with < 10% foreground content.

3. **Multi-sequence montage** (MRI): For brain MRI, compose a 2x2 montage of T1, T1-CE,
   T2, FLAIR at the same slice location, providing MedGemma with multi-contrast context
   in a single image.

4. **Output format for MedGemma**: PNG images at 512x512 resolution, with intensity
   values mapped to 8-bit grayscale (or RGB for montages). Metadata embedded in the
   filename convention: `{patient_id}_{study_date}_{sequence}_{slice_idx}.png`.

### 7. Volume Rendering (Contextual, Not Primary)

For UI context and the video demo, provide lightweight 3D views:

- Maximum Intensity Projection (MIP) in axial, coronal, sagittal planes.
- Generated using `scikit-image` projection utilities or `matplotlib` volume rendering.
- These are for **context only** -- all measurements are performed on 2D axial slices.
- Lesion bounding boxes overlaid on MIP views for spatial orientation.

### 8. Output Format

The pipeline produces a standardized output structure per study:

```python
@dataclass
class ProcessedVolume:
    pixel_data: np.ndarray          # Shape: (D, H, W), float32, normalized
    raw_data: Optional[np.ndarray]  # Shape: (D, H, W), original units (HU or raw)
    affine: np.ndarray              # 4x4 RAS+ affine transform
    metadata: VolumeMetadata        # Full metadata dict
    masks: Dict[str, np.ndarray]    # Named binary masks (lesions, brain, etc.)
    selected_slices: List[int]      # Indices of MedGemma-selected slices
    mip_projections: Dict[str, np.ndarray]  # Axial/coronal/sagittal MIPs
```

Persisted to disk as:
- `.npz` files for arrays (compressed numpy)
- `.json` sidecar for metadata
- `.png` files for MedGemma-ready slices

### 9. Image Embedding Storage in RuVector

After preprocessing, a compact embedding vector is generated for each processed volume
using MedSigLIP. These embeddings are batch-inserted into RuVector's HNSW index,
enabling three key capabilities:

1. **Rapid similar-case retrieval for MedGemma context**: When composing a prompt for
   MedGemma's visual reasoning, the system retrieves the k most similar prior scans
   (from the same patient or across the population) to provide comparative context.
   This is critical for longitudinal change detection and differential assessment.

2. **Anomaly detection for preprocessing quality control**: Embeddings of newly
   preprocessed volumes are compared against the population distribution. Volumes whose
   embeddings fall far from the population manifold (low cosine similarity to all
   neighbors) are flagged for manual review, catching preprocessing failures such as
   incorrect orientation, failed bias correction, or corrupt input data.

3. **Longitudinal drift detection across timepoints**: For a single patient, the
   sequence of per-timepoint embeddings forms a trajectory in embedding space. Sudden
   large jumps in this trajectory that do not correlate with expected treatment effects
   may indicate scan quality issues or protocol changes that affect downstream
   measurements.

```python
# After preprocessing, store embedding in RuVector
embedding = medsiglib_encoder(processed_volume.selected_slices)
ruvector_client.insert(VectorEntry(
    id=f"{metadata.patient_id}_{metadata.study_date}",
    vector=embedding,
    metadata={
        "patient_id": metadata.patient_id,
        "modality": metadata.modality,
        "sequence_type": metadata.sequence_type,
    }
))
```

For batch processing of entire datasets, RuVector's batch insert capability
(10K+ vectors per operation) ensures that initial dataset ingestion completes
efficiently:

```python
# Batch insert all embeddings after dataset preprocessing
entries = [
    VectorEntry(
        id=f"{vol.metadata.patient_id}_{vol.metadata.study_date}",
        vector=medsiglib_encoder(vol.selected_slices),
        metadata={"patient_id": vol.metadata.patient_id, "modality": vol.metadata.modality},
    )
    for vol in all_processed_volumes
]
ruvector_client.batch_insert(entries)
```

### Python Library Stack

| Library | Version | Purpose |
|---|---|---|
| `SimpleITK` | >= 2.3 | DICOM I/O, resampling, N4 bias correction, orientation |
| `nibabel` | >= 5.0 | NIfTI I/O, affine handling |
| `pydicom` | >= 2.4 | DICOM tag reading, PHI scanning |
| `scikit-image` | >= 0.22 | Image processing, morphology, projections |
| `torchio` | >= 0.19 | Data augmentation, transforms (for synthetic data) |
| `numpy` | >= 1.24 | Array operations |
| `Pillow` | >= 10.0 | PNG export for MedGemma slices |
| `ruvector` | >= 0.1 | Vector embedding storage and similarity search |

## Consequences

### Positive

- **Reproducibility**: deterministic pipeline with recorded parameters ensures that any
  measurement can be traced back to the exact preprocessing applied.
- **Modality agnostic downstream**: all consumers receive the same `ProcessedVolume`
  interface regardless of whether the input was CT DICOM, MRI NIfTI, or mixed.
- **PHI safety**: the hard gate prevents accidental processing of identifiable data,
  which is critical for the demo and aligns with HIPAA guidance.
- **MedGemma compatibility**: the slice selection policy produces appropriately sized
  2D inputs that match MedGemma's expected input format.
- **Measurement validity**: isotropic resampling and consistent orientation ensure that
  diameter and volume measurements are geometrically correct.
- **Similar-case retrieval**: embedding each processed volume in RuVector enables rapid
  retrieval of morphologically similar cases, enriching MedGemma's reasoning context
  with relevant comparators and supporting population-level analysis.

### Negative

- **Resampling artifacts**: B-spline interpolation can introduce ringing near sharp
  boundaries; mitigated by using nearest-neighbor for mask resampling.
- **Information loss from windowing**: CT windowing discards intensity information
  outside the window range; mitigated by preserving raw HU arrays alongside windowed.
- **MRI normalization sensitivity**: percentile-based normalization can be affected by
  large lesions that shift the intensity distribution; mitigated by using foreground
  masks that exclude obvious outliers.
- **2D slice limitation**: passing only 2D slices to MedGemma loses 3D spatial context;
  mitigated by selecting multiple slices per lesion and providing MIP views.
- **Processing time**: N4 bias field correction is the slowest step (~10-20s per volume);
  acceptable for hackathon but would need GPU acceleration for production.

### Risks

- Sequence type auto-detection may fail for non-standard DICOM headers from unfamiliar
  scanners. Mitigation: manual override in the UI and explicit labeling for known datasets.
- PHI scanning heuristics may produce false positives on legitimately anonymized data that
  retains non-empty tag values. Mitigation: configurable whitelist of known-safe patterns.

## Alternatives Considered

### 1. Full 3D Volume Processing for MedGemma

Rejected. While MedGemma 1.5 has expanded 3D support, the hackathon timeline does not
permit debugging 3D input formatting issues. The 2D slice approach is explicitly recommended
in Google's preprocessing notebooks and provides sufficient information for lesion-level
analysis. Full 3D can be added post-hackathon.

### 2. MONAI as Primary Preprocessing Framework

Considered but not selected as the primary stack. MONAI provides excellent medical imaging
transforms but adds a heavy dependency (PyTorch-based) and overlaps significantly with
SimpleITK for our needs. We use `torchio` (lighter weight) for augmentation transforms only.
MONAI remains an option for future integration if we need its training pipeline features.

### 3. No Bias Field Correction for MRI

Rejected. Longitudinal comparison of MRI intensities across timepoints is fundamentally
unreliable without bias field correction. The PROTEAS dataset includes scans from multiple
timepoints where B1 inhomogeneity varies, making N4 correction essential for consistent
measurements.

### 4. Fixed Resolution (0.5mm) for All Modalities

Rejected. 0.5mm isotropic would quadruple memory usage compared to 1.0mm without meaningful
benefit for the RECIST-scale measurements (longest diameter) we perform. The 1.0mm default
balances measurement precision with memory constraints on a single GPU workstation.

### 5. Cloud-Based DICOM Parsing (e.g., Google Cloud Healthcare API)

Rejected for hackathon. Adds network dependency, authentication complexity, and latency.
Local processing with pydicom + SimpleITK is sufficient and keeps the demo self-contained.
Cloud DICOM support is a post-hackathon enhancement for production deployment.
