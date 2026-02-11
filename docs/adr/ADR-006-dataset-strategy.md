# ADR-006: Dataset Strategy and Data Pipeline

| Field       | Value                                      |
|-------------|--------------------------------------------|
| **Status**  | Accepted                                   |
| **Date**    | 2026-02-11                                 |
| **Authors** | Digital Twin Tumor Team                    |
| **Deciders**| Full team                                  |

---

## Context

The Digital Twin Tumor project needs longitudinal imaging data with tumor
segmentations and treatment context to demonstrate its core capabilities:
patient-specific growth modeling, segmentation with provenance tracking, and
counterfactual simulation.

Key constraints:

1. **Longitudinal requirement.** The growth-model engine (ADR-005) needs
   multiple imaging timepoints per patient (minimum 3, ideally 5+) to fit
   patient-specific parameters. Most public medical imaging datasets are
   single-timepoint.

2. **Treatment context.** Counterfactual simulation requires knowledge of what
   treatment was administered and when. Very few public datasets include
   structured therapy logs alongside imaging.

3. **Measurement uncertainty calibration.** The state-space model in ADR-005
   requires an estimate of imaging measurement noise (sigma_obs). Calibrating
   this requires same-day repeat scans or known phantoms.

4. **Kaggle environment.** Data must be downloadable within the notebook
   environment or uploaded as a competition dataset. Total storage budget is
   approximately 20 GB for the notebook session.

5. **Licensing.** All data used must have licenses compatible with the
   competition (public domain, CC-BY, or equivalent research-use licenses).

6. **Volume.** We need enough patients to demonstrate population-level
   statistics (parameter distributions, ensemble performance), not just
   single-patient case studies.

---

## Decision

We adopt a **three-tier dataset strategy** that combines a primary longitudinal
dataset, a measurement-uncertainty calibration dataset, and synthetic
augmentation.

---

### Tier A: PROTEAS -- Primary Longitudinal Dataset

**Dataset:** PROTEAS (Prospective Registry of Outcomes in Patients Treated with
Stereotactic Radiosurgery for Brain Metastases)

**Source:** The Cancer Imaging Archive (TCIA)

**Key characteristics:**

| Property                | Value                              |
|-------------------------|------------------------------------|
| Patients                | ~40                                |
| Modality                | Brain MRI (T1 post-contrast)       |
| Timepoints per patient  | 3-8 follow-up scans                |
| Segmentations           | Expert tumor delineations provided |
| Treatment context       | SRS treatment dates and doses      |
| Format                  | DICOM + NIfTI segmentations        |
| License                 | TCIA Restricted License Agreement  |

**Why PROTEAS:**

- It is one of the very few publicly available datasets with true longitudinal
  imaging, expert segmentations, AND treatment context.
- The brain metastases use case is clinically compelling: SRS is a targeted
  therapy where monitoring tumor response is critical.
- 40 patients with multiple timepoints provides sufficient volume for
  population-level analysis.
- Expert segmentations serve as ground truth for benchmarking our segmentation
  stack (ADR-004).

**Limitations:**

- The license requires a signed data use agreement, which must be completed
  before the hackathon.
- Brain MRI only; does not cover the CT lung use case.
- Some patients may have fewer than 3 timepoints after filtering for quality.

---

### Tier B: RIDER Lung CT -- Measurement Uncertainty Calibration

**Dataset:** RIDER (Reference Image Database to Evaluate Therapy Response)
Lung CT subset

**Source:** The Cancer Imaging Archive (TCIA)

**Key characteristics:**

| Property                | Value                              |
|-------------------------|------------------------------------|
| Patients                | 32                                 |
| Modality                | Chest CT                           |
| Scans per patient       | 2 (same-day repeat, ~15 min apart) |
| Segmentations           | Tumor annotations available        |
| Treatment context       | Not applicable (test-retest study) |
| Format                  | DICOM                              |
| License                 | TCIA Data Usage Policy (open)      |

**Role in the pipeline:**

The RIDER dataset serves a specific, critical purpose: **calibrating the
observation noise parameter sigma_obs** in the state-space model (ADR-005).

By segmenting the same tumor on two same-day scans, we measure the
test-retest variability of our segmentation pipeline:

```
sigma_obs = std(V_scan1 - V_scan2) / sqrt(2)
```

This provides an empirical, modality-specific estimate of measurement
uncertainty that feeds directly into the growth model likelihood function.

**Secondary use:**

- Demonstrates the segmentation pipeline on CT data (complementing PROTEAS MRI).
- Validates that our segmentation stack (ADR-004) produces consistent results
  across repeat acquisitions.

---

### Tier C: Synthetic Longitudinal Augmentation

For patients with too few real timepoints, or to demonstrate system
capabilities beyond the available real data, we generate **synthetic
longitudinal sequences** that are explicitly labeled as synthetic.

**Generation methods:**

1. **Morphological augmentation.** Starting from a real segmentation mask at
   timepoint t0:
   - Apply calibrated morphological dilation (simulating growth) or erosion
     (simulating shrinkage) using `scipy.ndimage.binary_dilation` /
     `binary_erosion` with structuring elements sized to match clinically
     plausible growth rates.
   - Perturb the intensity within the dilated region using statistics sampled
     from the real tumor voxels (mean, std, texture).

2. **ODE-based volume trajectory.** Use the growth models from ADR-005 with
   known parameters to generate a ground-truth volume trajectory. Then
   deform the real baseline mask to match each target volume using
   `SimpleITK` thin-plate-spline or B-spline registration with controlled
   deformation fields.

3. **Noise injection.** Add realistic measurement noise calibrated from Tier B
   (RIDER) to the synthetic observations.

**Labeling:** Every synthetic data item carries a `provenance` field set to
`"synthetic"` with the generation method, source patient, and parameters
recorded. Synthetic data is never mixed with real data without explicit
flagging in all downstream outputs.

**Storage:**

```
data/synthetic/{method}/{patient_id}/
    timepoint_001.nii.gz
    timepoint_001_mask.nii.gz
    timepoint_001.json      # provenance metadata
    ...
    trajectory_params.json  # ground-truth growth parameters
```

---

## Data Loading Pipeline

### Architecture

```
DICOM/NIfTI files
    |
    v
[Reader Layer] -- pydicom, nibabel, SimpleITK
    |
    v
[Standardization] -- resampling, orientation (RAS+), intensity normalization
    |
    v
[Internal Format] -- xarray.Dataset with spatial + temporal dimensions
    |
    v
[Registry] -- pandas DataFrame cataloging all available data
    |
    v
[Consumer APIs] -- growth engine, segmentation pipeline, visualization
```

### Reader Layer

The reader layer abstracts over input formats:

- **DICOM:** `pydicom` for metadata extraction, `SimpleITK.ReadImage` with
  the DICOM series reader for volume assembly from individual slices.
- **NIfTI:** `nibabel.load` for `.nii` / `.nii.gz` files.
- **Segmentation masks:** loaded via the same path, with label encoding
  preserved (0 = background, 1+ = lesion IDs).

### Standardization

All loaded volumes undergo a standard preprocessing pipeline:

1. **Orientation:** Reorient to RAS+ (Right-Anterior-Superior) using
   `nibabel.as_closest_canonical` or `SimpleITK.DICOMOrient`.
2. **Resampling:** Resample to a target isotropic resolution (1mm^3 for MRI,
   1mm^3 for CT) using `SimpleITK.Resample` with B-spline interpolation for
   images and nearest-neighbor for masks.
3. **Intensity normalization:** Per-volume z-score normalization for MRI
   (to handle scanner variability). For CT, clip to the soft-tissue window
   [-150, 250] HU and scale to [0, 1].

### Internal Format

The standardized data is stored as an `xarray.Dataset`:

```python
import xarray as xr

patient_data = xr.Dataset(
    {
        "image": (["time", "z", "y", "x"], image_array),   # float32
        "mask": (["time", "z", "y", "x"], mask_array),      # uint8
    },
    coords={
        "time": pd.DatetimeIndex([...]),                     # acquisition dates
        "z": np.arange(n_slices) * voxel_spacing_z,         # mm
        "y": np.arange(n_rows) * voxel_spacing_y,           # mm
        "x": np.arange(n_cols) * voxel_spacing_x,           # mm
    },
    attrs={
        "patient_id": "PROTEAS-012",
        "modality": "MRI_T1_POST",
        "dataset": "PROTEAS",
        "provenance": "real",
        "voxel_spacing_mm": [1.0, 1.0, 1.0],
    },
)
```

xarray is chosen over raw numpy because it provides named dimensions, coordinate
labels (especially important for the time axis), metadata attributes, and
natural support for selection and slicing operations.

### Dataset Registry

A central registry (pandas DataFrame, serialized as CSV) catalogs all available
data:

```
patient_id | dataset  | timepoint  | modality | image_path        | mask_path         | provenance | license
PROTEAS-012| PROTEAS  | 2023-01-15 | MRI_T1   | data/processed/...| data/processed/...| real       | TCIA-RLA
RIDER-005  | RIDER    | 2024-06-01 | CT       | data/processed/...| data/processed/...| real       | TCIA-open
SYN-001    | SYNTHETIC| 2024-01-01 | MRI_T1   | data/synthetic/...| data/synthetic/...| synthetic  | N/A
```

This registry is the single source of truth for what data is available and
is queried by all downstream components.

---

## Directory Structure

```
data/
    raw/
        PROTEAS/
            {patient_id}/
                {series_uid}/
                    *.dcm
        RIDER/
            {patient_id}/
                scan_1/
                    *.dcm
                scan_2/
                    *.dcm
    processed/
        PROTEAS/
            {patient_id}/
                {timepoint}/
                    image.nii.gz
                    mask_v001.nii.gz
                    mask_v001.json
                    metadata.json
        RIDER/
            {patient_id}/
                scan_1/
                    image.nii.gz
                    mask_v001.nii.gz
                scan_2/
                    image.nii.gz
                    mask_v001.nii.gz
    synthetic/
        morphological/
            {patient_id}/
                ...
        ode_based/
            {patient_id}/
                ...
    registry.csv
```

---

## Python Libraries

| Library      | Version   | Purpose                                       |
|--------------|-----------|-----------------------------------------------|
| `pydicom`    | >=2.4     | DICOM file reading and metadata extraction    |
| `nibabel`    | >=5.0     | NIfTI file I/O                                |
| `SimpleITK`  | >=2.3     | Volume assembly, resampling, registration     |
| `xarray`     | >=2023.6  | Internal data representation with named dims  |
| `pandas`     | >=2.0     | Dataset registry, therapy log management      |
| `numpy`      | >=1.24    | Array operations                              |
| `scipy`      | >=1.11    | Morphological operations for synthetic data   |
| `requests`   | >=2.31    | TCIA API data download                        |
| `tqdm`       | >=4.65    | Progress bars for data loading                |

---

## Consequences

### Positive

- **Longitudinal coverage.** PROTEAS provides the rare combination of serial
  imaging + segmentations + treatment context needed for growth modeling.
- **Uncertainty calibration.** RIDER provides an empirical, defensible estimate
  of measurement noise rather than an arbitrary assumption.
- **Scalability.** Synthetic augmentation allows us to demonstrate the system
  on arbitrarily many patients and timepoints, with known ground-truth
  parameters for validation.
- **Transparency.** The three-tier labeling (real/calibration/synthetic) with
  provenance metadata ensures that no synthetic data is ever mistaken for real
  clinical data.
- **Structured pipeline.** The DICOM-to-xarray pipeline with standardized
  preprocessing reduces the risk of subtle bugs from inconsistent orientations,
  spacings, or intensity scales across datasets.

### Negative

- **PROTEAS access.** Requires a signed Data Use Agreement with TCIA. If the
  agreement is not completed in time, we fall back to Tier C (synthetic only)
  for the hackathon demo, which weakens the clinical narrative.
- **Limited modalities.** PROTEAS is MRI-only, RIDER is CT-only. We do not
  have a single dataset that covers both modalities with longitudinal data.
- **Storage.** Raw DICOM data for PROTEAS + RIDER may approach 10-15 GB.
  Processed NIfTI files are smaller but still require careful management in the
  Kaggle environment.
- **xarray overhead.** xarray adds a dependency and a learning curve compared
  to raw numpy. The benefits of named dimensions and coordinate-aware
  operations outweigh this cost for a multi-temporal, multi-patient dataset.

---

## Alternatives Considered

### 1. BraTS (Brain Tumor Segmentation Challenge)

Rejected as the primary dataset because BraTS is single-timepoint per patient.
It provides excellent segmentation ground truth but cannot support longitudinal
growth modeling. BraTS-trained nnU-Net models are still used for automated
baseline segmentation (ADR-004).

### 2. NSCLC-Radiomics / LUNG1

Considered for the lung CT use case. Rejected because while it includes
treatment outcome data, it lacks multiple imaging timepoints per patient -- most
patients have only a pre-treatment and one follow-up scan, which is insufficient
for growth curve fitting.

### 3. Purely Synthetic Data

Rejected as the sole data strategy because the MedGemma Impact Challenge
evaluates clinical relevance and real-world applicability. A demo built
entirely on synthetic data would be unconvincing to clinical reviewers.
Synthetic augmentation is used as a complement, not a replacement.

### 4. Private Clinical Data

Rejected due to IRB/ethics requirements, de-identification burden, and the
need for all data to be accessible in the Kaggle environment. Using exclusively
public datasets with clear licensing simplifies compliance.

### 5. Raw NumPy Arrays Instead of xarray

Considered for simplicity. Rejected because managing the time dimension,
patient metadata, and spatial coordinates across multiple datasets with raw
numpy arrays and dictionaries leads to fragile, error-prone code. xarray
provides the minimal structure needed without the overhead of a full database.

### 6. MONAI Dataset/Transform Pipeline

Considered for data loading and preprocessing. While MONAI provides excellent
medical imaging transforms, its Dataset classes are optimized for training
deep learning models (random access by index, on-the-fly augmentation). Our
use case is different: we need patient-centric, time-ordered access for growth
modeling. A custom xarray-based pipeline better fits this access pattern.
MONAI transforms are still used for specific operations (e.g., resampling,
intensity normalization) where they outperform SimpleITK.

---

## References

- PROTEAS Dataset. The Cancer Imaging Archive (TCIA).
  https://www.cancerimagingarchive.net/
- RIDER Lung CT. The Cancer Imaging Archive (TCIA).
  https://wiki.cancerimagingarchive.net/display/Public/RIDER+Lung+CT
- Zhao, B., et al. "Evaluating Variability in Tumor Measurements from
  Same-day Repeat CT Scans of Patients with Non-Small Cell Lung Cancer."
  Radiology, 2009.
- Clark, K., et al. "The Cancer Imaging Archive (TCIA): Maintaining and
  Operating a Public Information Repository." Journal of Digital Imaging, 2013.
