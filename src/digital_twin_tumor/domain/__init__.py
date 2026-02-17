"""Domain layer -- models, protocols, and events.

Re-exports all public domain types for convenient access::

    from digital_twin_tumor.domain import Patient, Lesion, RECISTResponse
"""

from __future__ import annotations

from digital_twin_tumor.domain.events import (
    AUDIT_LOGGED,
    IDENTITY_CONFIRMED,
    LESION_MEASURED,
    MEASUREMENT_OVERRIDDEN,
    NARRATIVE_GENERATED,
    RECIST_CLASSIFIED,
    SIMULATION_COMPLETED,
    STUDY_INGESTED,
    TWIN_FITTED,
    VOLUME_PREPROCESSED,
    Event,
    EventBus,
)
from digital_twin_tumor.domain.models import (
    AppConfig,
    AuditEvent,
    DatabaseConfig,
    DataProvenance,
    GrowthModelResult,
    ImagingStudy,
    Lesion,
    Measurement,
    NarrativeResult,
    Patient,
    ProcessedVolume,
    RECISTResponse,
    SimulationResult,
    TherapyEvent,
    TimePoint,
    TrackedLesionSet,
    UncertaintyEstimate,
    VolumeMetadata,
    VoxelSpacing,
)
from digital_twin_tumor.domain.protocols import (
    GrowthModelProtocol,
    IngestionProtocol,
    MeasurementProtocol,
    PHIGateProtocol,
    PreprocessingProtocol,
    RECISTProtocol,
    ReasoningProtocol,
    SegmentationProtocol,
    SimulationProtocol,
    StorageProtocol,
    TrackingProtocol,
)

__all__ = [
    # Models
    "AppConfig",
    "AuditEvent",
    "DatabaseConfig",
    "DataProvenance",
    "GrowthModelResult",
    "ImagingStudy",
    "Lesion",
    "Measurement",
    "NarrativeResult",
    "Patient",
    "ProcessedVolume",
    "RECISTResponse",
    "SimulationResult",
    "TherapyEvent",
    "TimePoint",
    "TrackedLesionSet",
    "UncertaintyEstimate",
    "VolumeMetadata",
    "VoxelSpacing",
    # Event constants (ADR-001)
    "AUDIT_LOGGED",
    "IDENTITY_CONFIRMED",
    "LESION_MEASURED",
    "MEASUREMENT_OVERRIDDEN",
    "NARRATIVE_GENERATED",
    "RECIST_CLASSIFIED",
    "SIMULATION_COMPLETED",
    "STUDY_INGESTED",
    "TWIN_FITTED",
    "VOLUME_PREPROCESSED",
    # Events
    "Event",
    "EventBus",
    # Protocols
    "GrowthModelProtocol",
    "IngestionProtocol",
    "MeasurementProtocol",
    "PHIGateProtocol",
    "PreprocessingProtocol",
    "RECISTProtocol",
    "ReasoningProtocol",
    "SegmentationProtocol",
    "SimulationProtocol",
    "StorageProtocol",
    "TrackingProtocol",
]
