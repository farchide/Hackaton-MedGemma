"""Synthetic data generator for the Digital Twin Tumor Response Assessment system.

Produces realistic longitudinal oncology data for five patient scenarios
covering common clinical trajectories: classic responder, mixed response,
pseudo-progression, rapid progression, and surgical + adjuvant therapy.

All data is reproducible via fixed numpy RNG seeds. The generator uses the
actual domain models, RECIST classifier, growth models, simulation engine,
and uncertainty module so that demo data exercises the full system pipeline.
"""

from __future__ import annotations

import logging
import math
import uuid
from datetime import date, datetime, timedelta
from typing import Any

import numpy as np

from digital_twin_tumor.domain.models import (
    GrowthModelResult,
    Lesion,
    Measurement,
    Patient,
    RECISTResponse,
    SimulationResult,
    TherapyEvent,
    TimePoint,
    UncertaintyEstimate,
)
from digital_twin_tumor.measurement.recist import RECISTClassifier
from digital_twin_tumor.storage.sqlite_backend import SQLiteBackend
from digital_twin_tumor.twin_engine.growth_models import (
    ExponentialGrowth,
    GompertzGrowth,
    LogisticGrowth,
)
from digital_twin_tumor.twin_engine.simulation import SimulationEngine
from digital_twin_tumor.twin_engine.uncertainty import (
    compute_measurement_uncertainty,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_BASELINE_DATE = date(2025, 1, 6)  # Monday, start of study

_METHODS = ["auto", "semi-auto", "manual"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _uuid() -> str:
    return str(uuid.uuid4())


def _diameter_to_volume(diameter_mm: float) -> float:
    """Approximate volume from longest diameter using sphere model."""
    r = diameter_mm / 2.0
    return (4.0 / 3.0) * math.pi * r ** 3


def _volume_to_diameter(volume_mm3: float) -> float:
    """Approximate diameter from volume using sphere model."""
    if volume_mm3 <= 0:
        return 0.0
    r = ((3.0 * volume_mm3) / (4.0 * math.pi)) ** (1.0 / 3.0)
    return 2.0 * r


def _add_noise(value: float, rng: np.random.Generator, sigma: float = 1.5) -> float:
    """Add measurement variability noise."""
    return max(0.0, value + rng.normal(0.0, sigma))


def _jitter_centroid(
    base: tuple[float, float, float],
    rng: np.random.Generator,
    sigma: float = 1.0,
) -> tuple[float, float, float]:
    """Add slight positional jitter to centroid between scans."""
    return (
        base[0] + rng.normal(0.0, sigma),
        base[1] + rng.normal(0.0, sigma),
        base[2] + rng.normal(0.0, sigma),
    )


def _scan_date(week: int) -> date:
    """Compute scan date from week offset relative to baseline."""
    return _BASELINE_DATE + timedelta(weeks=week)


def _therapy_event(
    patient_id: str,
    therapy_type: str,
    drug_name: str,
    dose: str,
    start_week: int,
    end_week: int | None = None,
    cycle_info: str = "",
) -> TherapyEvent:
    """Create a TherapyEvent with week-based metadata for growth models."""
    meta: dict[str, Any] = {
        "drug_name": drug_name,
        "start_week": float(start_week),
    }
    if end_week is not None:
        meta["end_week"] = float(end_week)
    if cycle_info:
        meta["cycle_info"] = cycle_info
    return TherapyEvent(
        patient_id=patient_id,
        start_date=_scan_date(start_week),
        end_date=_scan_date(end_week) if end_week is not None else None,
        therapy_type=therapy_type,
        dose=dose,
        metadata=meta,
    )


# ---------------------------------------------------------------------------
# 3D Volume generation
# ---------------------------------------------------------------------------


def generate_demo_volumes(
    patient_id: str,
    timepoint_id: str,
    lesions: list[Lesion],
    shape: tuple[int, int, int] = (64, 64, 32),
    voxel_mm: float = 2.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Generate a synthetic 3D volume with embedded tumor blobs.

    Each lesion is rendered as a 3D Gaussian blob centered at a position
    derived from its centroid, with spread proportional to its diameter.
    Background tissue is modeled as low-amplitude Gaussian noise.

    Parameters
    ----------
    patient_id:
        Patient identifier (for logging).
    timepoint_id:
        Timepoint identifier (for logging).
    lesions:
        Lesions to embed as bright blobs in the volume.
    shape:
        Volume dimensions (D, H, W) in voxels.
    voxel_mm:
        Isotropic voxel spacing in millimetres.
    rng:
        Numpy random generator for reproducibility.

    Returns
    -------
    np.ndarray
        3D float64 array of shape *shape* with values in [0, 1].
    """
    if rng is None:
        rng = np.random.default_rng(42)

    vol = rng.normal(0.15, 0.05, size=shape).clip(0.0, 0.4)

    d, h, w = shape
    zz, yy, xx = np.mgrid[0:d, 0:h, 0:w]

    for lesion in lesions:
        cx, cy, cz = lesion.centroid
        # Map centroid (mm) to voxel coordinates, centered in volume
        vx = w / 2.0 + cx / voxel_mm
        vy = h / 2.0 + cy / voxel_mm
        vz = d / 2.0 + cz / voxel_mm

        # Clamp to volume bounds
        vx = np.clip(vx, 2, w - 3)
        vy = np.clip(vy, 2, h - 3)
        vz = np.clip(vz, 2, d - 3)

        # Sigma from diameter
        diam_voxels = max(lesion.longest_diameter_mm / voxel_mm, 1.5)
        sigma = diam_voxels / 2.5

        # Intensity proportional to size (larger tumors brighter)
        intensity = min(0.9, 0.5 + lesion.longest_diameter_mm / 100.0)

        dist_sq = (
            (xx - vx) ** 2 + (yy - vy) ** 2 + (zz - vz) ** 2
        )
        blob = intensity * np.exp(-dist_sq / (2.0 * sigma ** 2))
        vol = np.maximum(vol, blob)

    return vol.clip(0.0, 1.0)


# ---------------------------------------------------------------------------
# Per-patient scenario generators
# ---------------------------------------------------------------------------


def _generate_patient_1(rng: np.random.Generator) -> dict[str, Any]:
    """Classic Responder -- Lung Cancer.

    6 timepoints over 24 weeks. Two target lesions in lung, one non-target.
    Progressive shrinkage under chemotherapy: PR at week 6, near-CR by week 18.
    """
    patient_id = _uuid()
    patient = Patient(
        patient_id=patient_id,
        metadata={
            "name": "Demo Patient 1",
            "scenario": "Classic Responder",
            "age": 62,
            "sex": "M",
            "cancer_type": "Non-Small Cell Lung Cancer (NSCLC)",
            "stage": "IIIB",
            "histology": "Adenocarcinoma",
            "ECOG_PS": 1,
        },
    )

    # Scan weeks: baseline, 6, 10, 14, 18, 24
    scan_weeks = [0, 6, 10, 14, 18, 24]

    # Therapy: Carboplatin + Pemetrexed starting week 2, cycles every 3 weeks
    therapies = [
        _therapy_event(
            patient_id, "chemotherapy",
            "Carboplatin AUC5 + Pemetrexed 500mg/m2",
            "Carboplatin AUC5, Pemetrexed 500mg/m2 q3w",
            start_week=2, end_week=24,
            cycle_info="Cycles 1-8 q21d",
        ),
    ]

    # Lesion trajectories (diameter in mm over time)
    # Lesion 1: Right upper lobe, 35mm baseline -> shrinks
    # Lesion 2: Left lower lobe, 22mm baseline -> shrinks
    # Non-target: Mediastinal lymph node, 12mm (stable, below target threshold)
    lesion_1_diameters = [35.0, 28.0, 22.0, 16.0, 10.0, 6.0]
    lesion_2_diameters = [22.0, 18.0, 14.0, 10.0, 6.0, 3.0]
    nontarget_diameters = [12.0, 11.0, 10.0, 9.0, 8.0, 7.0]

    # Base centroids (mm offset from volume center)
    centroid_1 = (12.0, -8.0, 5.0)   # Right upper lobe
    centroid_2 = (-10.0, 6.0, -4.0)  # Left lower lobe
    centroid_nt = (0.0, -2.0, 8.0)   # Mediastinal

    timepoints = []
    all_lesions: dict[str, list[Lesion]] = {}
    all_measurements: dict[str, list[Measurement]] = {}
    all_volumes: dict[str, np.ndarray] = {}

    # Stable canonical IDs for lesion tracking
    lesion_1_canonical = _uuid()
    lesion_2_canonical = _uuid()
    lesion_nt_canonical = _uuid()

    for i, week in enumerate(scan_weeks):
        tp_id = _uuid()
        therapy_status = "pre" if week < 2 else "on"
        tp = TimePoint(
            timepoint_id=tp_id,
            patient_id=patient_id,
            scan_date=_scan_date(week),
            modality="CT",
            therapy_status=therapy_status,
            metadata={"week": week, "scan_number": i + 1},
        )
        timepoints.append(tp)

        d1 = _add_noise(lesion_1_diameters[i], rng, sigma=1.0)
        d2 = _add_noise(lesion_2_diameters[i], rng, sigma=0.8)
        d_nt = _add_noise(nontarget_diameters[i], rng, sigma=0.5)

        method = _METHODS[i % len(_METHODS)]
        conf = 0.92 + rng.uniform(-0.03, 0.03)

        les1 = Lesion(
            lesion_id=f"{lesion_1_canonical}_tp{i}",
            timepoint_id=tp_id,
            centroid=_jitter_centroid(centroid_1, rng),
            volume_mm3=_diameter_to_volume(d1),
            longest_diameter_mm=round(d1, 1),
            short_axis_mm=round(d1 * 0.7, 1),
            is_target=True,
            organ="right_upper_lobe_lung",
            confidence=round(conf, 3),
        )
        les2 = Lesion(
            lesion_id=f"{lesion_2_canonical}_tp{i}",
            timepoint_id=tp_id,
            centroid=_jitter_centroid(centroid_2, rng),
            volume_mm3=_diameter_to_volume(d2),
            longest_diameter_mm=round(d2, 1),
            short_axis_mm=round(d2 * 0.65, 1),
            is_target=True,
            organ="left_lower_lobe_lung",
            confidence=round(conf, 3),
        )
        les_nt = Lesion(
            lesion_id=f"{lesion_nt_canonical}_tp{i}",
            timepoint_id=tp_id,
            centroid=_jitter_centroid(centroid_nt, rng),
            volume_mm3=_diameter_to_volume(d_nt),
            longest_diameter_mm=round(d_nt, 1),
            short_axis_mm=round(d_nt * 0.8, 1),
            is_target=False,
            organ="mediastinal_lymph_node",
            confidence=round(conf - 0.05, 3),
        )

        lesions_at_tp = [les1, les2, les_nt]
        all_lesions[tp_id] = lesions_at_tp

        meas1 = Measurement(
            lesion_id=les1.lesion_id,
            timepoint_id=tp_id,
            diameter_mm=les1.longest_diameter_mm,
            volume_mm3=les1.volume_mm3,
            method=method,
            reviewer="AI_pipeline",
            metadata={"short_axis_mm": les1.short_axis_mm},
        )
        meas2 = Measurement(
            lesion_id=les2.lesion_id,
            timepoint_id=tp_id,
            diameter_mm=les2.longest_diameter_mm,
            volume_mm3=les2.volume_mm3,
            method=method,
            reviewer="AI_pipeline",
            metadata={"short_axis_mm": les2.short_axis_mm},
        )
        all_measurements[tp_id] = [meas1, meas2]

        vol = generate_demo_volumes(
            patient_id, tp_id, lesions_at_tp, rng=rng,
        )
        all_volumes[tp_id] = vol

    return {
        "patient": patient,
        "timepoints": timepoints,
        "lesions": all_lesions,
        "measurements": all_measurements,
        "therapies": therapies,
        "volumes": all_volumes,
        "scan_weeks": scan_weeks,
        "target_diameter_trajectories": {
            "lesion_1": lesion_1_diameters,
            "lesion_2": lesion_2_diameters,
        },
    }


def _generate_patient_2(rng: np.random.Generator) -> dict[str, Any]:
    """Mixed Response -- Liver Cancer (HCC).

    8 timepoints over 32 weeks. Three target lesions in liver.
    Lesion 1 shrinks (PR), Lesion 2 stable (SD), Lesion 3 grows.
    New lesion appears at week 16. Overall: SD -> PD from new lesion.
    """
    patient_id = _uuid()
    patient = Patient(
        patient_id=patient_id,
        metadata={
            "name": "Demo Patient 2",
            "scenario": "Mixed Response",
            "age": 58,
            "sex": "F",
            "cancer_type": "Hepatocellular Carcinoma (HCC)",
            "stage": "IVA",
            "histology": "HCC, moderately differentiated",
            "ECOG_PS": 1,
        },
    )

    scan_weeks = [0, 4, 8, 12, 16, 20, 26, 32]

    therapies = [
        _therapy_event(
            patient_id, "immunotherapy",
            "Atezolizumab 1200mg + Bevacizumab 15mg/kg",
            "Atezo 1200mg + Bev 15mg/kg q3w",
            start_week=4, end_week=32,
            cycle_info="Cycles 1-10 q21d",
        ),
    ]

    # Lesion 1: right lobe, 45mm -> shrinks to ~25mm
    l1_d = [45.0, 44.0, 38.0, 33.0, 30.0, 27.0, 25.0, 24.0]
    # Lesion 2: left lobe, 30mm -> stable ~28-32mm
    l2_d = [30.0, 31.0, 29.0, 30.0, 31.0, 29.0, 28.0, 29.0]
    # Lesion 3: caudate, 18mm -> grows to 28mm
    l3_d = [18.0, 19.0, 20.0, 22.0, 24.0, 26.0, 27.0, 28.0]
    # New lesion appears at week 16 (index 4)
    new_d = [0.0, 0.0, 0.0, 0.0, 8.0, 12.0, 15.0, 18.0]

    c1 = (8.0, -5.0, 2.0)
    c2 = (-6.0, -3.0, 0.0)
    c3 = (2.0, 4.0, -3.0)
    c_new = (10.0, 2.0, -5.0)

    l1_cid = _uuid()
    l2_cid = _uuid()
    l3_cid = _uuid()
    new_cid = _uuid()

    timepoints = []
    all_lesions: dict[str, list[Lesion]] = {}
    all_measurements: dict[str, list[Measurement]] = {}
    all_volumes: dict[str, np.ndarray] = {}

    for i, week in enumerate(scan_weeks):
        tp_id = _uuid()
        tp = TimePoint(
            timepoint_id=tp_id,
            patient_id=patient_id,
            scan_date=_scan_date(week),
            modality="CT",
            therapy_status="pre" if week < 4 else "on",
            metadata={"week": week, "scan_number": i + 1},
        )
        timepoints.append(tp)

        d1 = _add_noise(l1_d[i], rng, sigma=1.2)
        d2 = _add_noise(l2_d[i], rng, sigma=1.0)
        d3 = _add_noise(l3_d[i], rng, sigma=0.8)
        method = _METHODS[i % len(_METHODS)]
        conf = 0.90 + rng.uniform(-0.03, 0.03)

        les1 = Lesion(
            lesion_id=f"{l1_cid}_tp{i}",
            timepoint_id=tp_id,
            centroid=_jitter_centroid(c1, rng),
            volume_mm3=_diameter_to_volume(d1),
            longest_diameter_mm=round(d1, 1),
            short_axis_mm=round(d1 * 0.6, 1),
            is_target=True,
            organ="liver_right_lobe",
            confidence=round(conf, 3),
        )
        les2 = Lesion(
            lesion_id=f"{l2_cid}_tp{i}",
            timepoint_id=tp_id,
            centroid=_jitter_centroid(c2, rng),
            volume_mm3=_diameter_to_volume(d2),
            longest_diameter_mm=round(d2, 1),
            short_axis_mm=round(d2 * 0.65, 1),
            is_target=True,
            organ="liver_left_lobe",
            confidence=round(conf, 3),
        )
        les3 = Lesion(
            lesion_id=f"{l3_cid}_tp{i}",
            timepoint_id=tp_id,
            centroid=_jitter_centroid(c3, rng),
            volume_mm3=_diameter_to_volume(d3),
            longest_diameter_mm=round(d3, 1),
            short_axis_mm=round(d3 * 0.55, 1),
            is_target=True,
            organ="liver_caudate_lobe",
            confidence=round(conf, 3),
        )

        lesions_at_tp = [les1, les2, les3]
        measurements_at_tp = [
            Measurement(
                lesion_id=les1.lesion_id, timepoint_id=tp_id,
                diameter_mm=les1.longest_diameter_mm,
                volume_mm3=les1.volume_mm3, method=method,
                reviewer="AI_pipeline",
            ),
            Measurement(
                lesion_id=les2.lesion_id, timepoint_id=tp_id,
                diameter_mm=les2.longest_diameter_mm,
                volume_mm3=les2.volume_mm3, method=method,
                reviewer="AI_pipeline",
            ),
            Measurement(
                lesion_id=les3.lesion_id, timepoint_id=tp_id,
                diameter_mm=les3.longest_diameter_mm,
                volume_mm3=les3.volume_mm3, method=method,
                reviewer="AI_pipeline",
            ),
        ]

        # New lesion from week 16 onwards
        if new_d[i] > 0:
            d_new = _add_noise(new_d[i], rng, sigma=0.5)
            les_new = Lesion(
                lesion_id=f"{new_cid}_tp{i}",
                timepoint_id=tp_id,
                centroid=_jitter_centroid(c_new, rng),
                volume_mm3=_diameter_to_volume(d_new),
                longest_diameter_mm=round(d_new, 1),
                short_axis_mm=round(d_new * 0.6, 1),
                is_target=False,
                is_new=(i == 4),  # New at first appearance
                organ="liver_segment_VII",
                confidence=round(conf - 0.08, 3),
            )
            lesions_at_tp.append(les_new)

        all_lesions[tp_id] = lesions_at_tp
        all_measurements[tp_id] = measurements_at_tp
        all_volumes[tp_id] = generate_demo_volumes(
            patient_id, tp_id, lesions_at_tp, rng=rng,
        )

    return {
        "patient": patient,
        "timepoints": timepoints,
        "lesions": all_lesions,
        "measurements": all_measurements,
        "therapies": therapies,
        "volumes": all_volumes,
        "scan_weeks": scan_weeks,
        "new_lesion_week": 16,
    }


def _generate_patient_3(rng: np.random.Generator) -> dict[str, Any]:
    """Pseudo-progression then Response -- Melanoma, iRECIST.

    7 timepoints over 28 weeks. Two target lesions.
    Immunotherapy starts week 0. Initial increase (iUPD at week 8),
    then dramatic shrinkage. Demonstrates why iRECIST matters.
    """
    patient_id = _uuid()
    patient = Patient(
        patient_id=patient_id,
        metadata={
            "name": "Demo Patient 3",
            "scenario": "Pseudo-progression (iRECIST)",
            "age": 45,
            "sex": "F",
            "cancer_type": "Metastatic Melanoma",
            "stage": "IV",
            "histology": "BRAF V600E mutant melanoma",
            "ECOG_PS": 0,
            "biomarkers": {"BRAF_V600E": True, "PD_L1_TPS": "80%"},
        },
    )

    scan_weeks = [0, 4, 8, 12, 16, 22, 28]

    therapies = [
        _therapy_event(
            patient_id, "immunotherapy",
            "Pembrolizumab 200mg",
            "Pembrolizumab 200mg q3w",
            start_week=0, end_week=28,
            cycle_info="Cycles 1-10 q21d",
        ),
    ]

    # Pseudo-progression: initial increase then dramatic shrinkage
    # Lesion 1: 25mm -> 28 -> 32 (pseudo) -> 24 -> 15 -> 8 -> 3
    l1_d = [25.0, 27.0, 32.0, 24.0, 15.0, 8.0, 3.0]
    # Lesion 2: 20mm -> 22 -> 26 (pseudo) -> 18 -> 10 -> 5 -> 0 (disappeared)
    l2_d = [20.0, 22.0, 26.0, 18.0, 10.0, 5.0, 0.0]

    c1 = (5.0, -3.0, 2.0)   # Subcutaneous trunk
    c2 = (-8.0, 4.0, -1.0)  # Axillary

    l1_cid = _uuid()
    l2_cid = _uuid()

    timepoints = []
    all_lesions: dict[str, list[Lesion]] = {}
    all_measurements: dict[str, list[Measurement]] = {}
    all_volumes: dict[str, np.ndarray] = {}

    for i, week in enumerate(scan_weeks):
        tp_id = _uuid()
        tp = TimePoint(
            timepoint_id=tp_id,
            patient_id=patient_id,
            scan_date=_scan_date(week),
            modality="CT",
            therapy_status="on",
            metadata={"week": week, "scan_number": i + 1},
        )
        timepoints.append(tp)

        d1 = _add_noise(l1_d[i], rng, sigma=0.8)
        d2 = max(0.0, _add_noise(l2_d[i], rng, sigma=0.7))
        method = _METHODS[i % len(_METHODS)]
        conf = 0.94 + rng.uniform(-0.02, 0.02)

        les1 = Lesion(
            lesion_id=f"{l1_cid}_tp{i}",
            timepoint_id=tp_id,
            centroid=_jitter_centroid(c1, rng),
            volume_mm3=_diameter_to_volume(d1),
            longest_diameter_mm=round(d1, 1),
            short_axis_mm=round(d1 * 0.7, 1),
            is_target=True,
            organ="subcutaneous_trunk",
            confidence=round(conf, 3),
        )

        lesions_at_tp = [les1]
        meas_list = [
            Measurement(
                lesion_id=les1.lesion_id, timepoint_id=tp_id,
                diameter_mm=les1.longest_diameter_mm,
                volume_mm3=les1.volume_mm3, method=method,
                reviewer="AI_pipeline",
            ),
        ]

        if d2 > 0:
            les2 = Lesion(
                lesion_id=f"{l2_cid}_tp{i}",
                timepoint_id=tp_id,
                centroid=_jitter_centroid(c2, rng),
                volume_mm3=_diameter_to_volume(d2),
                longest_diameter_mm=round(d2, 1),
                short_axis_mm=round(d2 * 0.65, 1),
                is_target=True,
                organ="axillary_soft_tissue",
                confidence=round(conf, 3),
            )
            lesions_at_tp.append(les2)
            meas_list.append(
                Measurement(
                    lesion_id=les2.lesion_id, timepoint_id=tp_id,
                    diameter_mm=les2.longest_diameter_mm,
                    volume_mm3=les2.volume_mm3, method=method,
                    reviewer="AI_pipeline",
                ),
            )

        all_lesions[tp_id] = lesions_at_tp
        all_measurements[tp_id] = meas_list
        all_volumes[tp_id] = generate_demo_volumes(
            patient_id, tp_id, lesions_at_tp, rng=rng,
        )

    return {
        "patient": patient,
        "timepoints": timepoints,
        "lesions": all_lesions,
        "measurements": all_measurements,
        "therapies": therapies,
        "volumes": all_volumes,
        "scan_weeks": scan_weeks,
        "is_irecist": True,
    }


def _generate_patient_4(rng: np.random.Generator) -> dict[str, Any]:
    """Rapid Progression -- Pancreatic Cancer.

    5 timepoints over 16 weeks. Two target lesions.
    Exponential growth despite chemotherapy; PD at every assessment.
    New lesions appear at week 10. Doubling time ~6 weeks.
    """
    patient_id = _uuid()
    patient = Patient(
        patient_id=patient_id,
        metadata={
            "name": "Demo Patient 4",
            "scenario": "Rapid Progression",
            "age": 71,
            "sex": "M",
            "cancer_type": "Pancreatic Ductal Adenocarcinoma",
            "stage": "IV",
            "histology": "Poorly differentiated adenocarcinoma",
            "ECOG_PS": 2,
        },
    )

    scan_weeks = [0, 4, 8, 12, 16]

    therapies = [
        _therapy_event(
            patient_id, "chemotherapy",
            "Gemcitabine 1000mg/m2 + nab-Paclitaxel 125mg/m2",
            "Gem 1000 + nab-Pac 125 mg/m2 d1,8,15 q28d",
            start_week=2, end_week=16,
            cycle_info="Cycles 1-4 q28d",
        ),
    ]

    # Exponential growth, doubling ~6 weeks -> r = ln(2)/6 ~ 0.116/week
    # Lesion 1: 40mm -> 48 -> 57 -> 68 -> 80
    l1_d = [40.0, 48.0, 57.0, 68.0, 80.0]
    # Lesion 2: 28mm -> 34 -> 40 -> 48 -> 57
    l2_d = [28.0, 34.0, 40.0, 48.0, 57.0]
    # New lesions from week 10 (index 2-ish, first seen at 10 weeks)
    # We detect them at scan_week 12 (index 3)
    new1_d = [0.0, 0.0, 0.0, 10.0, 16.0]
    new2_d = [0.0, 0.0, 0.0, 8.0, 14.0]

    c1 = (4.0, 2.0, -1.0)
    c2 = (-3.0, 5.0, 1.0)
    c_n1 = (8.0, -4.0, 3.0)
    c_n2 = (-6.0, -2.0, -4.0)

    l1_cid = _uuid()
    l2_cid = _uuid()
    n1_cid = _uuid()
    n2_cid = _uuid()

    timepoints = []
    all_lesions: dict[str, list[Lesion]] = {}
    all_measurements: dict[str, list[Measurement]] = {}
    all_volumes: dict[str, np.ndarray] = {}

    for i, week in enumerate(scan_weeks):
        tp_id = _uuid()
        tp = TimePoint(
            timepoint_id=tp_id,
            patient_id=patient_id,
            scan_date=_scan_date(week),
            modality="CT",
            therapy_status="pre" if week < 2 else "on",
            metadata={"week": week, "scan_number": i + 1},
        )
        timepoints.append(tp)

        d1 = _add_noise(l1_d[i], rng, sigma=1.5)
        d2 = _add_noise(l2_d[i], rng, sigma=1.2)
        method = _METHODS[i % len(_METHODS)]
        conf = 0.88 + rng.uniform(-0.04, 0.04)

        les1 = Lesion(
            lesion_id=f"{l1_cid}_tp{i}",
            timepoint_id=tp_id,
            centroid=_jitter_centroid(c1, rng),
            volume_mm3=_diameter_to_volume(d1),
            longest_diameter_mm=round(d1, 1),
            short_axis_mm=round(d1 * 0.65, 1),
            is_target=True,
            organ="pancreas_body",
            confidence=round(conf, 3),
        )
        les2 = Lesion(
            lesion_id=f"{l2_cid}_tp{i}",
            timepoint_id=tp_id,
            centroid=_jitter_centroid(c2, rng),
            volume_mm3=_diameter_to_volume(d2),
            longest_diameter_mm=round(d2, 1),
            short_axis_mm=round(d2 * 0.6, 1),
            is_target=True,
            organ="pancreas_tail",
            confidence=round(conf, 3),
        )

        lesions_at_tp = [les1, les2]
        meas_list = [
            Measurement(
                lesion_id=les1.lesion_id, timepoint_id=tp_id,
                diameter_mm=les1.longest_diameter_mm,
                volume_mm3=les1.volume_mm3, method=method,
                reviewer="AI_pipeline",
            ),
            Measurement(
                lesion_id=les2.lesion_id, timepoint_id=tp_id,
                diameter_mm=les2.longest_diameter_mm,
                volume_mm3=les2.volume_mm3, method=method,
                reviewer="AI_pipeline",
            ),
        ]

        # New lesions from week 12 scan onwards
        if new1_d[i] > 0:
            dn1 = _add_noise(new1_d[i], rng, sigma=0.5)
            dn2 = _add_noise(new2_d[i], rng, sigma=0.5)
            n1 = Lesion(
                lesion_id=f"{n1_cid}_tp{i}",
                timepoint_id=tp_id,
                centroid=_jitter_centroid(c_n1, rng),
                volume_mm3=_diameter_to_volume(dn1),
                longest_diameter_mm=round(dn1, 1),
                short_axis_mm=round(dn1 * 0.6, 1),
                is_target=False, is_new=(i == 3),
                organ="liver_segment_IV",
                confidence=round(conf - 0.1, 3),
            )
            n2 = Lesion(
                lesion_id=f"{n2_cid}_tp{i}",
                timepoint_id=tp_id,
                centroid=_jitter_centroid(c_n2, rng),
                volume_mm3=_diameter_to_volume(dn2),
                longest_diameter_mm=round(dn2, 1),
                short_axis_mm=round(dn2 * 0.55, 1),
                is_target=False, is_new=(i == 3),
                organ="peritoneum",
                confidence=round(conf - 0.12, 3),
            )
            lesions_at_tp.extend([n1, n2])

        all_lesions[tp_id] = lesions_at_tp
        all_measurements[tp_id] = meas_list
        all_volumes[tp_id] = generate_demo_volumes(
            patient_id, tp_id, lesions_at_tp, rng=rng,
        )

    return {
        "patient": patient,
        "timepoints": timepoints,
        "lesions": all_lesions,
        "measurements": all_measurements,
        "therapies": therapies,
        "volumes": all_volumes,
        "scan_weeks": scan_weeks,
    }


def _generate_patient_5(rng: np.random.Generator) -> dict[str, Any]:
    """Surgical + Adjuvant -- Colorectal Cancer with liver metastases.

    7 timepoints over 36 weeks. Two liver mets pre-surgery.
    Surgery at week 4 removes both. Small recurrence at week 20.
    Adjuvant chemo shrinks recurrence. CR -> recurrence -> PR.
    """
    patient_id = _uuid()
    patient = Patient(
        patient_id=patient_id,
        metadata={
            "name": "Demo Patient 5",
            "scenario": "Surgical + Adjuvant",
            "age": 55,
            "sex": "M",
            "cancer_type": "Colorectal Cancer (Liver Metastases)",
            "stage": "IV (oligometastatic)",
            "histology": "Moderately differentiated adenocarcinoma",
            "ECOG_PS": 0,
            "biomarkers": {"MSI": "MSS", "KRAS": "G12D", "BRAF": "WT"},
        },
    )

    scan_weeks = [0, 4, 8, 14, 20, 28, 36]

    therapies = [
        _therapy_event(
            patient_id, "surgery",
            "Hepatic metastasectomy",
            "Bilateral liver resection",
            start_week=4, end_week=4,
        ),
        _therapy_event(
            patient_id, "chemotherapy",
            "FOLFOX (5-FU + Leucovorin + Oxaliplatin)",
            "FOLFOX q14d, 12 cycles",
            start_week=22, end_week=36,
            cycle_info="Cycles 1-6 q14d adjuvant",
        ),
    ]

    # Pre-surgery: two liver mets
    # Post-surgery: CR (both removed)
    # Recurrence at week 20: small 8mm lesion
    # Adjuvant chemo shrinks it
    # Lesion 1: 32mm -> 32 (pre-op scan) -> 0 -> 0 -> 0 -> 0 -> 0
    l1_d = [32.0, 32.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    # Lesion 2: 25mm -> 25 -> 0 -> 0 -> 0 -> 0 -> 0
    l2_d = [25.0, 25.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    # Recurrence: appears at week 20
    rec_d = [0.0, 0.0, 0.0, 0.0, 8.0, 6.0, 3.0]

    c1 = (6.0, -4.0, 1.0)
    c2 = (-5.0, 3.0, -2.0)
    c_rec = (8.0, -6.0, 3.0)  # Near original site

    l1_cid = _uuid()
    l2_cid = _uuid()
    rec_cid = _uuid()

    timepoints = []
    all_lesions: dict[str, list[Lesion]] = {}
    all_measurements: dict[str, list[Measurement]] = {}
    all_volumes: dict[str, np.ndarray] = {}

    for i, week in enumerate(scan_weeks):
        tp_id = _uuid()
        if week < 4:
            therapy_status = "pre"
        elif week == 4:
            therapy_status = "peri"
        elif week < 22:
            therapy_status = "post_surgery"
        else:
            therapy_status = "on"

        tp = TimePoint(
            timepoint_id=tp_id,
            patient_id=patient_id,
            scan_date=_scan_date(week),
            modality="CT",
            therapy_status=therapy_status,
            metadata={"week": week, "scan_number": i + 1},
        )
        timepoints.append(tp)

        method = _METHODS[i % len(_METHODS)]
        conf = 0.91 + rng.uniform(-0.03, 0.03)
        lesions_at_tp = []
        meas_list = []

        # Original lesions (pre/peri surgery only)
        if l1_d[i] > 0:
            d1 = _add_noise(l1_d[i], rng, sigma=1.0)
            les1 = Lesion(
                lesion_id=f"{l1_cid}_tp{i}",
                timepoint_id=tp_id,
                centroid=_jitter_centroid(c1, rng),
                volume_mm3=_diameter_to_volume(d1),
                longest_diameter_mm=round(d1, 1),
                short_axis_mm=round(d1 * 0.65, 1),
                is_target=True,
                organ="liver_segment_V",
                confidence=round(conf, 3),
            )
            lesions_at_tp.append(les1)
            meas_list.append(Measurement(
                lesion_id=les1.lesion_id, timepoint_id=tp_id,
                diameter_mm=les1.longest_diameter_mm,
                volume_mm3=les1.volume_mm3, method=method,
                reviewer="AI_pipeline",
            ))

        if l2_d[i] > 0:
            d2 = _add_noise(l2_d[i], rng, sigma=0.8)
            les2 = Lesion(
                lesion_id=f"{l2_cid}_tp{i}",
                timepoint_id=tp_id,
                centroid=_jitter_centroid(c2, rng),
                volume_mm3=_diameter_to_volume(d2),
                longest_diameter_mm=round(d2, 1),
                short_axis_mm=round(d2 * 0.6, 1),
                is_target=True,
                organ="liver_segment_VIII",
                confidence=round(conf, 3),
            )
            lesions_at_tp.append(les2)
            meas_list.append(Measurement(
                lesion_id=les2.lesion_id, timepoint_id=tp_id,
                diameter_mm=les2.longest_diameter_mm,
                volume_mm3=les2.volume_mm3, method=method,
                reviewer="AI_pipeline",
            ))

        # Recurrence
        if rec_d[i] > 0:
            d_rec = _add_noise(rec_d[i], rng, sigma=0.5)
            les_rec = Lesion(
                lesion_id=f"{rec_cid}_tp{i}",
                timepoint_id=tp_id,
                centroid=_jitter_centroid(c_rec, rng),
                volume_mm3=_diameter_to_volume(d_rec),
                longest_diameter_mm=round(d_rec, 1),
                short_axis_mm=round(d_rec * 0.6, 1),
                is_target=True,
                is_new=(i == 4),
                organ="liver_segment_V",
                confidence=round(conf - 0.06, 3),
            )
            lesions_at_tp.append(les_rec)
            meas_list.append(Measurement(
                lesion_id=les_rec.lesion_id, timepoint_id=tp_id,
                diameter_mm=les_rec.longest_diameter_mm,
                volume_mm3=les_rec.volume_mm3, method=method,
                reviewer="AI_pipeline",
            ))

        all_lesions[tp_id] = lesions_at_tp
        all_measurements[tp_id] = meas_list
        all_volumes[tp_id] = generate_demo_volumes(
            patient_id, tp_id, lesions_at_tp, rng=rng,
        )

    return {
        "patient": patient,
        "timepoints": timepoints,
        "lesions": all_lesions,
        "measurements": all_measurements,
        "therapies": therapies,
        "volumes": all_volumes,
        "scan_weeks": scan_weeks,
    }


# ---------------------------------------------------------------------------
# RECIST classification for a patient scenario
# ---------------------------------------------------------------------------


def _classify_patient_recist(
    scenario: dict[str, Any],
) -> list[RECISTResponse]:
    """Compute RECIST responses for all timepoints in a scenario.

    Uses the actual RECISTClassifier to compute sum-of-diameters and
    classify each timepoint relative to baseline and nadir.
    """
    classifier = RECISTClassifier()
    timepoints = scenario["timepoints"]
    all_meas = scenario["measurements"]
    all_les = scenario["lesions"]
    is_irecist = scenario.get("is_irecist", False)

    if not timepoints:
        return []

    # Baseline is the first timepoint
    baseline_tp_id = timepoints[0].timepoint_id
    baseline_meas = all_meas.get(baseline_tp_id, [])
    baseline_lesions = [
        l for l in all_les.get(baseline_tp_id, []) if l.is_target
    ]
    baseline_sum = classifier.compute_sum_of_diameters(
        baseline_meas, baseline_lesions,
    )

    responses: list[RECISTResponse] = []
    nadir_sum = baseline_sum
    prev_irecist_cat: str | None = None

    for tp in timepoints:
        tp_id = tp.timepoint_id
        meas = all_meas.get(tp_id, [])
        lesions = [l for l in all_les.get(tp_id, []) if l.is_target]

        current_sum = classifier.compute_sum_of_diameters(meas, lesions)

        # Check for new lesions at this timepoint
        new_lesions = any(l.is_new for l in all_les.get(tp_id, []))

        recist = classifier.classify_response(
            current_sum=current_sum,
            baseline_sum=baseline_sum,
            nadir_sum=nadir_sum,
            new_lesions=new_lesions,
        )

        # Apply iRECIST if applicable
        final_category = recist.category
        if is_irecist:
            final_category = classifier.classify_irecist(
                current_category=recist.category,
                previous_category=prev_irecist_cat,
                on_immunotherapy=True,
            )
            prev_irecist_cat = final_category

        response = RECISTResponse(
            timepoint_id=tp_id,
            sum_of_diameters=round(current_sum, 1),
            baseline_sum=round(baseline_sum, 1),
            nadir_sum=round(nadir_sum, 1),
            percent_change_from_baseline=recist.percent_change_from_baseline,
            percent_change_from_nadir=recist.percent_change_from_nadir,
            category=final_category,
        )
        responses.append(response)

        # Update nadir
        if current_sum < nadir_sum:
            nadir_sum = current_sum

    return responses


# ---------------------------------------------------------------------------
# Growth model fitting for a patient scenario
# ---------------------------------------------------------------------------


def _fit_growth_models(
    scenario: dict[str, Any],
) -> list[GrowthModelResult]:
    """Fit exponential, logistic, and Gompertz growth models.

    Uses sum-of-diameters volume trajectory across timepoints.
    Returns fitted results for each model that converges.
    """
    classifier = RECISTClassifier()
    timepoints = scenario["timepoints"]
    all_meas = scenario["measurements"]
    all_les = scenario["lesions"]
    scan_weeks = scenario["scan_weeks"]

    times = []
    volumes = []
    for i, tp in enumerate(timepoints):
        tp_id = tp.timepoint_id
        meas = all_meas.get(tp_id, [])
        target_lesions = [l for l in all_les.get(tp_id, []) if l.is_target]
        total_vol = sum(l.volume_mm3 for l in target_lesions)
        if total_vol > 0:
            times.append(float(scan_weeks[i]))
            volumes.append(total_vol)

    if len(times) < 2:
        return []

    t_arr = np.array(times, dtype=np.float64)
    v_arr = np.array(volumes, dtype=np.float64)

    results: list[GrowthModelResult] = []
    for model_cls in [ExponentialGrowth, LogisticGrowth, GompertzGrowth]:
        try:
            model = model_cls()
            result = model.fit(t_arr, v_arr, therapy_events=scenario["therapies"])
            results.append(result)
        except Exception as exc:
            logger.warning("Growth model %s failed: %s", model_cls.name, exc)

    return results


# ---------------------------------------------------------------------------
# Simulation scenarios for a patient
# ---------------------------------------------------------------------------


def _run_simulations(
    scenario: dict[str, Any],
    growth_results: list[GrowthModelResult],
) -> list[SimulationResult]:
    """Run counterfactual simulations using the best growth model."""
    if not growth_results:
        return []

    # Pick best model by AIC
    best = min(growth_results, key=lambda r: r.aic)
    scan_weeks = scenario["scan_weeks"]
    max_week = max(scan_weeks)

    # Extend prediction to 1.5x the observation window
    sim_times = np.linspace(0, max_week * 1.5, 50)
    engine = SimulationEngine()

    sim_results: list[SimulationResult] = []
    therapies = scenario["therapies"]

    try:
        sim_results.append(engine.run_natural_history(best, sim_times))
    except Exception as exc:
        logger.warning("Natural history sim failed: %s", exc)

    if therapies:
        try:
            sim_results.append(
                engine.run_earlier_treatment(best, therapies, 2.0, sim_times)
            )
        except Exception as exc:
            logger.warning("Earlier treatment sim failed: %s", exc)

        try:
            sim_results.append(
                engine.run_later_treatment(best, therapies, 4.0, sim_times)
            )
        except Exception as exc:
            logger.warning("Later treatment sim failed: %s", exc)

    return sim_results


# ---------------------------------------------------------------------------
# Uncertainty estimates for a patient
# ---------------------------------------------------------------------------


def _compute_uncertainties(
    scenario: dict[str, Any],
) -> list[dict[str, Any]]:
    """Compute measurement uncertainty for each timepoint."""
    timepoints = scenario["timepoints"]
    all_meas = scenario["measurements"]
    results = []

    for tp in timepoints:
        tp_id = tp.timepoint_id
        meas = all_meas.get(tp_id, [])
        if not meas:
            continue

        method = meas[0].method if meas else "manual"
        sigma_auto = 0.8 if method in ("auto", "semi-auto") else 0.0
        sigma_scan = 0.5 if len(meas) > 1 else 0.0

        ue = compute_measurement_uncertainty(
            sigma_manual=1.5,
            sigma_auto=sigma_auto,
            sigma_scan=sigma_scan,
        )
        results.append({
            "timepoint_id": tp_id,
            "uncertainty": ue,
        })

    return results


# ---------------------------------------------------------------------------
# Store a complete patient scenario to the database
# ---------------------------------------------------------------------------


def _store_scenario(
    db: SQLiteBackend,
    scenario: dict[str, Any],
    recist_responses: list[RECISTResponse],
    growth_results: list[GrowthModelResult],
    sim_results: list[SimulationResult],
) -> None:
    """Persist all data for a single patient scenario to SQLite."""
    patient = scenario["patient"]
    pid = patient.patient_id

    db.save_patient(patient)

    for tp in scenario["timepoints"]:
        db.save_timepoint(tp)

    for tp_id, lesions in scenario["lesions"].items():
        for lesion in lesions:
            db.save_lesion(lesion)

    for tp_id, measurements in scenario["measurements"].items():
        for meas in measurements:
            db.save_measurement(meas)

    for therapy in scenario["therapies"]:
        db.save_therapy_event(therapy)

    for response in recist_responses:
        db.save_recist(response)

    for result in growth_results:
        db.save_growth_model(pid, result)

    for sim in sim_results:
        db.save_simulation_result(pid, sim)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def generate_all_demo_data(
    db_path: str = ".cache/demo.db",
    seed: int = 42,
    verbose: bool = True,
) -> dict[str, Any]:
    """Generate all demo patient scenarios and store to SQLite.

    Creates five patient journeys with complete longitudinal data:
    measurements, lesions, RECIST classifications, growth model fits,
    simulation scenarios, uncertainty estimates, and 3D volumes.

    Parameters
    ----------
    db_path:
        Path to the SQLite database file.
    seed:
        Random seed for reproducible data generation.
    verbose:
        If True, print progress messages to stdout.

    Returns
    -------
    dict
        Summary containing all generated scenarios, RECIST histories,
        growth models, and simulation results keyed by patient ID.
    """
    rng = np.random.default_rng(seed)
    db = SQLiteBackend(db_path)

    generators = [
        ("Patient 1: Classic Responder (NSCLC)", _generate_patient_1),
        ("Patient 2: Mixed Response (HCC)", _generate_patient_2),
        ("Patient 3: Pseudo-progression (Melanoma, iRECIST)", _generate_patient_3),
        ("Patient 4: Rapid Progression (Pancreatic)", _generate_patient_4),
        ("Patient 5: Surgical + Adjuvant (CRC Liver Mets)", _generate_patient_5),
    ]

    all_results: dict[str, Any] = {}

    for label, gen_fn in generators:
        if verbose:
            print(f"\n{'='*60}")
            print(f"  Generating: {label}")
            print(f"{'='*60}")

        # Generate raw scenario data
        scenario = gen_fn(rng)
        pid = scenario["patient"].patient_id

        if verbose:
            n_tp = len(scenario["timepoints"])
            n_les = sum(len(v) for v in scenario["lesions"].values())
            print(f"  Patient ID : {pid[:12]}...")
            print(f"  Timepoints : {n_tp}")
            print(f"  Lesions    : {n_les} total observations")

        # Compute RECIST classifications
        if verbose:
            print("  Computing RECIST classifications...")
        recist_responses = _classify_patient_recist(scenario)

        if verbose:
            cats = [r.category for r in recist_responses]
            weeks = scenario["scan_weeks"]
            print(f"  RECIST     : {' -> '.join(cats)}")
            for w, r in zip(weeks, recist_responses):
                pct = r.percent_change_from_baseline
                print(
                    f"    Week {w:3d}: {r.category:5s} "
                    f"(sum={r.sum_of_diameters:6.1f}mm, "
                    f"chg={pct:+6.1f}%)"
                )

        # Fit growth models
        if verbose:
            print("  Fitting growth models...")
        growth_results = _fit_growth_models(scenario)

        if verbose and growth_results:
            for gr in growth_results:
                print(
                    f"    {gr.model_name:12s}: "
                    f"AIC={gr.aic:10.2f}  params={gr.parameters}"
                )

        # Run simulations
        if verbose:
            print("  Running counterfactual simulations...")
        sim_results = _run_simulations(scenario, growth_results)

        if verbose:
            for sr in sim_results:
                print(f"    Scenario: {sr.scenario_name}")

        # Compute uncertainty
        uncertainties = _compute_uncertainties(scenario)

        # Store everything to database
        if verbose:
            print("  Storing to database...")
        _store_scenario(db, scenario, recist_responses, growth_results, sim_results)

        all_results[pid] = {
            "scenario": scenario,
            "recist_responses": recist_responses,
            "growth_results": growth_results,
            "sim_results": sim_results,
            "uncertainties": uncertainties,
        }

    db.close()

    if verbose:
        _print_summary(all_results)

    return all_results


def _print_summary(all_results: dict[str, Any]) -> None:
    """Print a formatted summary table of all generated data."""
    print(f"\n{'='*80}")
    print("  DEMO DATA GENERATION SUMMARY")
    print(f"{'='*80}")

    header = (
        f"{'Scenario':<40s} | {'TPs':>3s} | "
        f"{'Lesions':>7s} | {'Best Model':<12s} | {'RECIST Trajectory'}"
    )
    print(header)
    print("-" * 80)

    for pid, data in all_results.items():
        scenario = data["scenario"]
        meta = scenario["patient"].metadata
        name = meta.get("scenario", "Unknown")
        cancer = meta.get("cancer_type", "")
        short_label = f"{name} ({cancer[:15]})"

        n_tp = len(scenario["timepoints"])
        n_les = sum(len(v) for v in scenario["lesions"].values())

        growth = data["growth_results"]
        best_model = "N/A"
        if growth:
            best = min(growth, key=lambda r: r.aic)
            best_model = best.model_name

        recist = data["recist_responses"]
        trajectory = " -> ".join(r.category for r in recist)

        print(
            f"{short_label:<40s} | {n_tp:3d} | "
            f"{n_les:7d} | {best_model:<12s} | {trajectory}"
        )

    print(f"\n  Total patients: {len(all_results)}")
    total_tp = sum(
        len(d["scenario"]["timepoints"]) for d in all_results.values()
    )
    total_les = sum(
        sum(len(v) for v in d["scenario"]["lesions"].values())
        for d in all_results.values()
    )
    total_sims = sum(len(d["sim_results"]) for d in all_results.values())
    print(f"  Total timepoints: {total_tp}")
    print(f"  Total lesion observations: {total_les}")
    print(f"  Total simulations: {total_sims}")
    print(f"{'='*80}")
