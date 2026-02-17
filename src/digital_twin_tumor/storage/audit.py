"""Audit logging system for the Digital Twin Tumor system (ADR-011).

Provides dual-output audit logging: every event is persisted to the SQLite
backend *and* optionally appended to a JSONL file for external consumption.
All events are immutable :class:`AuditEvent` domain objects with UUIDs and
UTC timestamps.
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any

from digital_twin_tumor.domain.models import (
    AuditEvent,
    Measurement,
    RECISTResponse,
)

if TYPE_CHECKING:
    from digital_twin_tumor.storage.sqlite_backend import SQLiteBackend


def _now() -> datetime:
    """Return current UTC timestamp."""
    from datetime import UTC
    return datetime.now(UTC).replace(tzinfo=None)


def _uuid() -> str:
    """Generate a new UUID4 string."""
    return str(uuid.uuid4())


class AuditLogger:
    """Dual-output audit logger writing to a storage backend and JSONL file.

    Parameters
    ----------
    backend:
        Optional :class:`SQLiteBackend` (or compatible) for database
        persistence.  When ``None``, events are only written to the JSONL
        file (if configured).
    jsonl_path:
        Optional path to a JSONL file.  Each event is appended as a single
        line of JSON.  Parent directories are created automatically.
    """

    def __init__(
        self,
        backend: SQLiteBackend | None = None,
        jsonl_path: str | None = None,
    ) -> None:
        self._backend = backend
        self._jsonl_path = jsonl_path
        if jsonl_path is not None:
            os.makedirs(os.path.dirname(jsonl_path) or ".", exist_ok=True)

    # ------------------------------------------------------------------
    # Core logging
    # ------------------------------------------------------------------

    def log(
        self,
        user_id: str,
        action_type: str,
        patient_id: str | None = None,
        timepoint_id: str | None = None,
        lesion_id: str | None = None,
        before_state: dict | None = None,
        after_state: dict | None = None,
        metadata: dict | None = None,
    ) -> AuditEvent:
        """Create and persist an audit event.

        Parameters
        ----------
        user_id:
            Identifier of the user or system component performing the action.
        action_type:
            Descriptive label for the action (e.g. ``"measurement_created"``).
        patient_id:
            Optional patient context.
        timepoint_id:
            Optional timepoint context.
        lesion_id:
            Optional lesion context.
        before_state:
            Snapshot of state before the action (for change tracking).
        after_state:
            Snapshot of state after the action.
        metadata:
            Any additional structured data about the action.

        Returns
        -------
        AuditEvent
            The persisted audit event with generated UUID and timestamp.
        """
        event = AuditEvent(
            event_id=_uuid(),
            timestamp=_now(),
            user_id=user_id,
            action_type=action_type,
            patient_id=patient_id or "",
            timepoint_id=timepoint_id or "",
            lesion_id=lesion_id or "",
            before_state=before_state or {},
            after_state=after_state or {},
            metadata=metadata or {},
        )
        self._persist(event)
        return event

    # ------------------------------------------------------------------
    # Convenience methods
    # ------------------------------------------------------------------

    def log_measurement(
        self, user_id: str, measurement: Measurement, method: str
    ) -> AuditEvent:
        """Log a measurement creation or update action.

        Parameters
        ----------
        user_id:
            Identifier of the user or system performing the measurement.
        measurement:
            The :class:`Measurement` domain object being recorded.
        method:
            The measurement method (e.g. ``"auto"``, ``"manual"``).

        Returns
        -------
        AuditEvent
            The persisted audit event.
        """
        return self.log(
            user_id=user_id,
            action_type="measurement_recorded",
            timepoint_id=measurement.timepoint_id,
            lesion_id=measurement.lesion_id,
            after_state={
                "measurement_id": measurement.measurement_id,
                "diameter_mm": measurement.diameter_mm,
                "volume_mm3": measurement.volume_mm3,
                "method": method,
                "reviewer": measurement.reviewer,
            },
            metadata={"method": method},
        )

    def log_override(
        self,
        user_id: str,
        lesion_id: str,
        field: str,
        old_value: Any,
        new_value: Any,
        reason: str,
    ) -> AuditEvent:
        """Log a human override of an AI-generated value.

        Parameters
        ----------
        user_id:
            Identifier of the clinician performing the override.
        lesion_id:
            The lesion being modified.
        field:
            Name of the field being overridden.
        old_value:
            Previous value of the field.
        new_value:
            New value after override.
        reason:
            Clinical rationale for the override.

        Returns
        -------
        AuditEvent
            The persisted audit event.
        """
        return self.log(
            user_id=user_id,
            action_type="human_override",
            lesion_id=lesion_id,
            before_state={field: old_value},
            after_state={field: new_value},
            metadata={"field": field, "reason": reason},
        )

    def log_recist_confirmation(
        self,
        user_id: str,
        response: RECISTResponse,
        confirmed: bool,
        reason: str | None = None,
    ) -> AuditEvent:
        """Log a RECIST classification confirmation or rejection.

        Parameters
        ----------
        user_id:
            Identifier of the clinician confirming the result.
        response:
            The :class:`RECISTResponse` being reviewed.
        confirmed:
            Whether the classification was confirmed by the clinician.
        reason:
            Optional rationale, especially useful when rejecting.

        Returns
        -------
        AuditEvent
            The persisted audit event.
        """
        return self.log(
            user_id=user_id,
            action_type="recist_confirmation",
            timepoint_id=response.timepoint_id,
            after_state={
                "category": response.category,
                "sum_of_diameters": response.sum_of_diameters,
                "confirmed": confirmed,
            },
            metadata={
                "confirmed": confirmed,
                "reason": reason or "",
                "percent_change_from_baseline": response.percent_change_from_baseline,
                "percent_change_from_nadir": response.percent_change_from_nadir,
            },
        )

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def get_session_log(self, session_id: str) -> list[AuditEvent]:
        """Retrieve audit events associated with a specific session.

        Parameters
        ----------
        session_id:
            The session identifier stored in event metadata.

        Returns
        -------
        list[AuditEvent]
            All events whose metadata contains the given session ID.
        """
        if self._backend is None:
            return []
        rows = self._backend.query(
            "SELECT * FROM audit_log WHERE json_extract(metadata, '$.session_id') = ? "
            "ORDER BY timestamp",
            (session_id,),
        )
        return [
            AuditEvent(
                event_id=r["event_id"],
                timestamp=datetime.fromisoformat(r["timestamp"]),
                user_id=r["user_id"],
                action_type=r["action_type"],
                patient_id=r["patient_id"],
                timepoint_id=r["timepoint_id"],
                lesion_id=r["lesion_id"],
                before_state=json.loads(r["before_state"] or "{}"),
                after_state=json.loads(r["after_state"] or "{}"),
                metadata=json.loads(r["metadata"] or "{}"),
            )
            for r in rows
        ]

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export_jsonl(self, path: str, patient_id: str | None = None) -> None:
        """Export audit events to a JSONL file.

        Parameters
        ----------
        path:
            Destination file path.  Parent directories are created
            automatically.
        patient_id:
            If provided, only export events for this patient.
        """
        if self._backend is None:
            return
        events = self._backend.get_audit_log(patient_id=patient_id, limit=100_000)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            for event in events:
                fh.write(json.dumps(_event_to_dict(event), default=str) + "\n")

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _persist(self, event: AuditEvent) -> None:
        """Write the event to both the database backend and JSONL file."""
        if self._backend is not None:
            self._backend.save_audit_event(event)
        if self._jsonl_path is not None:
            with open(self._jsonl_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(_event_to_dict(event), default=str) + "\n")


def _event_to_dict(event: AuditEvent) -> dict[str, Any]:
    """Convert an :class:`AuditEvent` to a plain dictionary for serialization."""
    return {
        "event_id": event.event_id,
        "timestamp": event.timestamp.isoformat(),
        "user_id": event.user_id,
        "action_type": event.action_type,
        "patient_id": event.patient_id,
        "timepoint_id": event.timepoint_id,
        "lesion_id": event.lesion_id,
        "before_state": event.before_state,
        "after_state": event.after_state,
        "metadata": event.metadata,
    }
