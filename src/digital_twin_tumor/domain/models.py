"""Domain models for the Digital Twin Tumor Response Assessment system.

All models are frozen dataclasses to enforce immutability. Mutable default
values (e.g. numpy arrays, dicts, lists) use ``field(default_factory=...)``.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _uuid() -> str:
    """Generate a new UUID4 string."""
    return str(uuid.uuid4())


def _now() -> datetime:
    """Return current UTC timestamp."""
    return datetime.now(UTC).replace(tzinfo=None)


def _empty_array() -> np.ndarray:
    """Return an empty float64 array."""
    return np.empty(0, dtype=np.float64)


def _empty_dict() -> dict[str, Any]:
    """Return an empty dictionary."""
    return {}


def _empty_list() -> list[Any]:
    """Return an empty list."""
    return []


# ---------------------------------------------------------------------------
# Core patient / scan models
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Patient:
    """Represents a de-identified patient."""

    patient_id: str = field(default_factory=_uuid)
    metadata: dict[str, Any] = field(default_factory=_empty_dict)


@dataclass(frozen=True)
class VoxelSpacing:
    """Physical spacing between voxels in millimetres (x, y, z)."""

    x: float = 1.0
    y: float = 1.0
    z: float = 1.0


@dataclass(frozen=True)
class TimePoint:
    """A single imaging time-point for a patient."""

    timepoint_id: str = field(default_factory=_uuid)
    patient_id: str = ""
    scan_date: date | None = None
    modality: str = "CT"
    therapy_status: str = "pre"
    metadata: dict[str, Any] = field(default_factory=_empty_dict)


@dataclass(frozen=True)
class ImagingStudy:
    """Represents a complete imaging study with metadata from DICOM/NIfTI.

    Maps to ADR-001 ImagingStudy data contract.
    """

    study_id: str = field(default_factory=_uuid)
    patient_id: str = ""
    timepoint_id: str = ""
    modality: str = "CT"
    series_description: str = ""
    manufacturer: str = ""
    slice_thickness_mm: float = 0.0
    pixel_spacing: tuple[float, float] = (1.0, 1.0)
    image_orientation: str = "RAS"
    acquisition_date: date | None = None
    num_slices: int = 0
    affine_matrix: np.ndarray = field(default_factory=lambda: np.eye(4))
    metadata: dict[str, Any] = field(default_factory=_empty_dict)


@dataclass(frozen=True)
class VolumeMetadata:
    """Rich metadata extracted during volume ingestion.

    Captures scanner parameters, sequence info, and provenance.
    """

    source_format: str = "DICOM"
    source_path: str = ""
    scanner_model: str = ""
    magnetic_field_strength: float = 0.0
    sequence_type: str = ""
    contrast_agent: bool = False
    window_center: float = 40.0
    window_width: float = 400.0
    bits_stored: int = 16
    rescale_slope: float = 1.0
    rescale_intercept: float = 0.0
    phi_stripped: bool = False
    checksum: str = ""


@dataclass(frozen=True)
class ProcessedVolume:
    """Preprocessed 3-D image volume ready for analysis.

    ``pixel_data`` is stored as a numpy array with shape (D, H, W) for
    single-channel volumes.
    """

    pixel_data: np.ndarray = field(default_factory=_empty_array)
    spacing: VoxelSpacing = field(default_factory=VoxelSpacing)
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0)
    direction: tuple[float, ...] = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    modality: str = "CT"
    metadata: dict[str, Any] = field(default_factory=_empty_dict)
    mask: np.ndarray | None = None
    selected_slices: list[int] = field(default_factory=_empty_list)


# ---------------------------------------------------------------------------
# Lesion / measurement models
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Lesion:
    """A single lesion identified in a time-point."""

    lesion_id: str = field(default_factory=_uuid)
    timepoint_id: str = ""
    centroid: tuple[float, float, float] = (0.0, 0.0, 0.0)
    volume_mm3: float = 0.0
    longest_diameter_mm: float = 0.0
    short_axis_mm: float = 0.0
    mask_path: str = ""
    is_target: bool = False
    is_new: bool = False
    organ: str = ""
    confidence: float = 0.0


@dataclass(frozen=True)
class Measurement:
    """A single measurement of a lesion at a time-point."""

    measurement_id: str = field(default_factory=_uuid)
    lesion_id: str = ""
    timepoint_id: str = ""
    diameter_mm: float = 0.0
    volume_mm3: float = 0.0
    method: str = "auto"
    reviewer: str = ""
    timestamp: datetime = field(default_factory=_now)
    metadata: dict[str, Any] = field(default_factory=_empty_dict)


@dataclass(frozen=True)
class DataProvenance:
    """Tracks the origin and lineage of a data artifact."""

    artifact_id: str = field(default_factory=_uuid)
    source_module: str = ""
    operation: str = ""
    input_ids: list[str] = field(default_factory=_empty_list)
    output_id: str = ""
    parameters: dict[str, Any] = field(default_factory=_empty_dict)
    timestamp: datetime = field(default_factory=_now)
    version: str = "1.0.0"


# ---------------------------------------------------------------------------
# Therapy
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TherapyEvent:
    """A therapy administration event."""

    therapy_id: str = field(default_factory=_uuid)
    patient_id: str = ""
    start_date: date | None = None
    end_date: date | None = None
    therapy_type: str = ""
    dose: str = ""
    metadata: dict[str, Any] = field(default_factory=_empty_dict)


# ---------------------------------------------------------------------------
# Growth / simulation models
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GrowthModelResult:
    """Result of fitting a growth model to longitudinal volume data."""

    model_name: str = ""
    parameters: dict[str, float] = field(default_factory=_empty_dict)
    aic: float = 0.0
    bic: float = 0.0
    akaike_weight: float = 0.0
    residuals: np.ndarray = field(default_factory=_empty_array)
    fitted_values: np.ndarray = field(default_factory=_empty_array)


@dataclass(frozen=True)
class UncertaintyEstimate:
    """Combined uncertainty from multiple sources."""

    sigma_manual: float = 1.5
    sigma_auto: float = 0.0
    sigma_scan: float = 0.0
    total_sigma: float = 0.0
    reliability: str = "MEDIUM"


@dataclass(frozen=True)
class SimulationResult:
    """Output of a what-if growth simulation."""

    scenario_name: str = ""
    time_points: np.ndarray = field(default_factory=_empty_array)
    predicted_volumes: np.ndarray = field(default_factory=_empty_array)
    lower_bound: np.ndarray = field(default_factory=_empty_array)
    upper_bound: np.ndarray = field(default_factory=_empty_array)
    parameters: dict[str, float] = field(default_factory=_empty_dict)


@dataclass(frozen=True)
class TrackedLesionSet:
    """Aggregated set of tracked lesions across timepoints (ADR-010).

    Groups all observations of a logical lesion across the longitudinal
    timeline with volume trajectory and match confidence history.
    """

    canonical_id: str = field(default_factory=_uuid)
    observation_ids: list[str] = field(default_factory=_empty_list)
    timepoint_ids: list[str] = field(default_factory=_empty_list)
    volumes: np.ndarray = field(default_factory=_empty_array)
    diameters: np.ndarray = field(default_factory=_empty_array)
    confidences: list[float] = field(default_factory=_empty_list)
    is_target: bool = False
    organ: str = ""
    first_seen: str = ""
    last_seen: str = ""
    status: str = "active"


# ---------------------------------------------------------------------------
# RECIST
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RECISTResponse:
    """RECIST 1.1 classification for a time-point."""

    timepoint_id: str = ""
    sum_of_diameters: float = 0.0
    baseline_sum: float = 0.0
    nadir_sum: float = 0.0
    percent_change_from_baseline: float = 0.0
    percent_change_from_nadir: float = 0.0
    category: str = "SD"
    is_confirmed: bool = False


# ---------------------------------------------------------------------------
# Narrative / LLM output
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class NarrativeResult:
    """Structured output from MedGemma narrative generation."""

    text: str = ""
    disclaimer: str = (
        "AI-generated summary for research purposes only. "
        "Not a substitute for professional medical judgement."
    )
    evidence_slices: list[int] = field(default_factory=_empty_list)
    grounding_check: bool = False
    safety_check: bool = False
    generation_params: dict[str, Any] = field(default_factory=_empty_dict)


# ---------------------------------------------------------------------------
# Audit
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AuditEvent:
    """Immutable audit log entry."""

    event_id: str = field(default_factory=_uuid)
    timestamp: datetime = field(default_factory=_now)
    user_id: str = ""
    session_id: str = ""
    action_type: str = ""
    component: str = ""
    patient_id: str = ""
    timepoint_id: str = ""
    lesion_id: str = ""
    before_state: dict[str, Any] = field(default_factory=_empty_dict)
    after_state: dict[str, Any] = field(default_factory=_empty_dict)
    metadata: dict[str, Any] = field(default_factory=_empty_dict)
    data_hash: str = ""


# ---------------------------------------------------------------------------
# Infrastructure configuration models
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DatabaseConfig:
    """Database connection configuration."""

    host: str = "localhost"
    port: int = 5432
    database: str = "digital_twin"
    user: str = ""
    password: str = ""
    ruvector_url: str = "http://localhost:8080"


@dataclass(frozen=True)
class AppConfig:
    """Application configuration loaded from YAML with environment overlays.

    Configuration is resolved in order:
      1. ``config/default.yaml``
      2. Environment-specific overlay (e.g. ``config/kaggle.yaml``)
      3. Environment variables prefixed with ``DTT_``
    """

    data: dict[str, Any] = field(default_factory=_empty_dict)

    # -- factory -----------------------------------------------------------

    @staticmethod
    def load(
        default_path: str | Path = "config/default.yaml",
        overlay_path: str | Path | None = None,
        env_prefix: str = "DTT_",
    ) -> AppConfig:
        """Load configuration from YAML files and environment variables.

        Parameters
        ----------
        default_path:
            Path to the base configuration file.
        overlay_path:
            Optional path to an environment-specific overlay.
        env_prefix:
            Prefix for environment variable overrides.  A variable named
            ``DTT_COMPUTE__PRECISION`` maps to ``config["compute"]["precision"]``.

        Returns
        -------
        AppConfig
            Frozen configuration object exposing the merged dictionary via
            ``data`` and typed helpers.
        """
        import os

        merged: dict[str, Any] = {}

        # 1. Load default
        default = Path(default_path)
        if default.exists():
            with open(default, "r", encoding="utf-8") as fh:
                raw = yaml.safe_load(fh) or {}
            merged = _deep_merge(merged, raw)

        # 2. Load overlay
        if overlay_path is not None:
            overlay = Path(overlay_path)
            if overlay.exists():
                with open(overlay, "r", encoding="utf-8") as fh:
                    raw = yaml.safe_load(fh) or {}
                merged = _deep_merge(merged, raw)

        # 3. Apply environment variable overrides
        for key, value in os.environ.items():
            if key.startswith(env_prefix):
                parts = key[len(env_prefix):].lower().split("__")
                _set_nested(merged, parts, _coerce(value))

        return AppConfig(data=merged)

    # -- typed accessors ---------------------------------------------------

    def get(self, dotted_key: str, default: Any = None) -> Any:
        """Retrieve a value using dot-separated path, e.g. ``compute.precision``.

        Parameters
        ----------
        dotted_key:
            Dot-separated path into the config tree.
        default:
            Value returned when the key is missing.
        """
        parts = dotted_key.split(".")
        node: Any = self.data
        for part in parts:
            if isinstance(node, dict) and part in node:
                node = node[part]
            else:
                return default
        return node

    def section(self, name: str) -> dict[str, Any]:
        """Return a top-level section as a dict (empty dict if missing)."""
        val = self.data.get(name)
        if isinstance(val, dict):
            return dict(val)
        return {}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge *overlay* into *base* (non-destructive)."""
    merged = dict(base)
    for key, value in overlay.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _set_nested(d: dict[str, Any], parts: list[str], value: Any) -> None:
    """Set a value in a nested dict using a list of keys."""
    for part in parts[:-1]:
        d = d.setdefault(part, {})
    if parts:
        d[parts[-1]] = value


def _coerce(value: str) -> Any:
    """Best-effort coercion from string to bool / int / float / str."""
    if value.lower() in ("true", "yes", "1"):
        return True
    if value.lower() in ("false", "no", "0"):
        return False
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value
