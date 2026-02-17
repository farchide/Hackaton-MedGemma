"""Protocol interfaces for the Digital Twin Tumor Response Assessment system.

Each protocol defines the contract that concrete implementations must satisfy.
Using :class:`typing.Protocol` enables structural subtyping -- implementations
do not need to explicitly inherit from these classes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import networkx as nx
import numpy as np

from digital_twin_tumor.domain.models import (
    GrowthModelResult,
    Lesion,
    Measurement,
    NarrativeResult,
    ProcessedVolume,
    RECISTResponse,
    SimulationResult,
    VoxelSpacing,
)


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------

@runtime_checkable
class IngestionProtocol(Protocol):
    """Load raw imaging data from DICOM or NIfTI sources."""

    def load_dicom(self, path: str | Path) -> ProcessedVolume:
        """Load a DICOM series from *path* and return a processed volume.

        Parameters
        ----------
        path:
            Directory containing DICOM files for a single series.

        Returns
        -------
        ProcessedVolume
            The loaded and minimally processed volume.
        """
        ...

    def load_nifti(self, path: str | Path) -> ProcessedVolume:
        """Load a NIfTI file from *path* and return a processed volume.

        Parameters
        ----------
        path:
            Path to a ``.nii`` or ``.nii.gz`` file.

        Returns
        -------
        ProcessedVolume
            The loaded and minimally processed volume.
        """
        ...


# ---------------------------------------------------------------------------
# PHI Gate (Protected Health Information)
# ---------------------------------------------------------------------------

@runtime_checkable
class PHIGateProtocol(Protocol):
    """Scan and strip protected health information from DICOM datasets."""

    def scan(self, dicom_dataset: Any) -> list[str]:
        """Return a list of DICOM tag names that contain PHI.

        Parameters
        ----------
        dicom_dataset:
            A ``pydicom.Dataset`` instance.

        Returns
        -------
        list[str]
            Tag names that were detected as containing PHI.
        """
        ...

    def strip(self, dicom_dataset: Any) -> Any:
        """Remove PHI from a DICOM dataset and return the cleaned copy.

        Parameters
        ----------
        dicom_dataset:
            A ``pydicom.Dataset`` instance.

        Returns
        -------
        pydicom.Dataset
            A new dataset with PHI tags removed or replaced.
        """
        ...


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

@runtime_checkable
class PreprocessingProtocol(Protocol):
    """Normalize, resample, and slice-select processed volumes."""

    def normalize(self, volume: ProcessedVolume) -> ProcessedVolume:
        """Apply intensity normalization appropriate for the modality.

        Parameters
        ----------
        volume:
            Input processed volume.

        Returns
        -------
        ProcessedVolume
            Volume with normalized pixel intensities.
        """
        ...

    def resample(
        self, volume: ProcessedVolume, target_spacing: VoxelSpacing
    ) -> ProcessedVolume:
        """Resample the volume to the requested isotropic spacing.

        Parameters
        ----------
        volume:
            Input processed volume.
        target_spacing:
            Desired voxel spacing in millimetres.

        Returns
        -------
        ProcessedVolume
            Resampled volume with updated spacing metadata.
        """
        ...

    def select_slices(
        self, volume: ProcessedVolume, lesion_centroid: tuple[float, float, float]
    ) -> list[np.ndarray]:
        """Select representative 2-D slices centred on a lesion.

        Parameters
        ----------
        volume:
            Input processed volume.
        lesion_centroid:
            (x, y, z) centroid of the lesion of interest.

        Returns
        -------
        list[np.ndarray]
            A list of 2-D arrays (one per selected slice).
        """
        ...


# ---------------------------------------------------------------------------
# Segmentation
# ---------------------------------------------------------------------------

@runtime_checkable
class SegmentationProtocol(Protocol):
    """Segment structures from 2-D or 3-D images."""

    def segment(self, image: np.ndarray, prompt: str) -> np.ndarray:
        """Return a binary mask for the requested structure.

        Parameters
        ----------
        image:
            2-D or 3-D image array.
        prompt:
            Text prompt or label describing the structure to segment.

        Returns
        -------
        np.ndarray
            Binary mask with the same spatial dimensions as *image*.
        """
        ...


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------

@runtime_checkable
class MeasurementProtocol(Protocol):
    """Extract quantitative measurements from segmentation masks."""

    def measure_diameter(self, mask: np.ndarray) -> float:
        """Compute the longest in-plane diameter of a binary mask.

        Parameters
        ----------
        mask:
            2-D or 3-D binary mask.

        Returns
        -------
        float
            Longest diameter in millimetres.
        """
        ...

    def measure_volume(self, mask: np.ndarray, spacing: VoxelSpacing) -> float:
        """Compute the volume of a binary mask in cubic millimetres.

        Parameters
        ----------
        mask:
            3-D binary mask.
        spacing:
            Voxel spacing used to convert voxel counts to physical volume.

        Returns
        -------
        float
            Volume in mm^3.
        """
        ...


# ---------------------------------------------------------------------------
# RECIST classification
# ---------------------------------------------------------------------------

@runtime_checkable
class RECISTProtocol(Protocol):
    """Classify time-point response according to RECIST 1.1 criteria."""

    def classify(
        self,
        measurements: list[Measurement],
        baseline: list[Measurement],
        nadir: list[Measurement],
    ) -> RECISTResponse:
        """Determine the RECIST 1.1 response category.

        Parameters
        ----------
        measurements:
            Current time-point measurements.
        baseline:
            Baseline time-point measurements.
        nadir:
            Nadir (smallest recorded sum) measurements.

        Returns
        -------
        RECISTResponse
            The computed RECIST classification.
        """
        ...


# ---------------------------------------------------------------------------
# Lesion tracking
# ---------------------------------------------------------------------------

@runtime_checkable
class TrackingProtocol(Protocol):
    """Track lesions across time-points."""

    def match_lesions(
        self,
        current: list[Lesion],
        previous: list[Lesion],
    ) -> dict[str, str]:
        """Match current lesions to previous lesions by proximity.

        Parameters
        ----------
        current:
            Lesions detected at the current time-point.
        previous:
            Lesions detected at the previous time-point.

        Returns
        -------
        dict[str, str]
            Mapping of current lesion IDs to matched previous lesion IDs.
        """
        ...

    def build_graph(self, matches: dict[str, str]) -> nx.DiGraph:
        """Build a directed graph of lesion identity over time.

        Parameters
        ----------
        matches:
            Mapping produced by :meth:`match_lesions`.

        Returns
        -------
        nx.DiGraph
            Graph where nodes are lesion IDs and edges indicate identity.
        """
        ...


# ---------------------------------------------------------------------------
# Growth modelling
# ---------------------------------------------------------------------------

@runtime_checkable
class GrowthModelProtocol(Protocol):
    """Fit parametric growth curves to longitudinal volume data."""

    def fit(self, times: np.ndarray, volumes: np.ndarray) -> GrowthModelResult:
        """Fit the model to observed *(times, volumes)* data.

        Parameters
        ----------
        times:
            1-D array of time values (e.g. days since baseline).
        volumes:
            1-D array of corresponding tumor volumes.

        Returns
        -------
        GrowthModelResult
            Fitted model with parameters and goodness-of-fit statistics.
        """
        ...

    def predict(self, times: np.ndarray) -> np.ndarray:
        """Predict volumes at the given *times* using fitted parameters.

        Parameters
        ----------
        times:
            1-D array of future time values.

        Returns
        -------
        np.ndarray
            Predicted volumes.
        """
        ...


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

@runtime_checkable
class SimulationProtocol(Protocol):
    """Run what-if growth simulations under different scenarios."""

    def run_scenario(
        self,
        name: str,
        model: GrowthModelResult,
        params: dict[str, Any],
    ) -> SimulationResult:
        """Execute a named simulation scenario.

        Parameters
        ----------
        name:
            Human-readable scenario identifier.
        model:
            Fitted growth model to use as the basis.
        params:
            Scenario-specific parameter overrides.

        Returns
        -------
        SimulationResult
            Predicted trajectory with uncertainty bounds.
        """
        ...


# ---------------------------------------------------------------------------
# Reasoning / narrative generation
# ---------------------------------------------------------------------------

@runtime_checkable
class ReasoningProtocol(Protocol):
    """Generate clinician-facing narrative summaries via LLM."""

    def generate_narrative(self, input_data: dict[str, Any]) -> NarrativeResult:
        """Produce a structured narrative report.

        Parameters
        ----------
        input_data:
            Dictionary containing measurements, RECIST classification,
            growth model outputs, and evidence slices.

        Returns
        -------
        NarrativeResult
            LLM-generated narrative with grounding and safety metadata.
        """
        ...


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------

@runtime_checkable
class StorageProtocol(Protocol):
    """Persist and retrieve data from the backing store."""

    def save(self, key: str, data: Any) -> None:
        """Save *data* under the given *key*.

        Parameters
        ----------
        key:
            Unique storage key.
        data:
            Serialisable data to persist.
        """
        ...

    def load(self, key: str) -> Any:
        """Load data previously stored under *key*.

        Parameters
        ----------
        key:
            Storage key.

        Returns
        -------
        Any
            The deserialised data, or ``None`` if not found.
        """
        ...

    def query(self, sql: str) -> list[Any]:
        """Execute a raw SQL query and return all rows.

        Parameters
        ----------
        sql:
            SQL statement to execute.

        Returns
        -------
        list[Any]
            Rows returned by the query.
        """
        ...
