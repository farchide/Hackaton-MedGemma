"""PostgreSQL storage backend for the Digital Twin Tumor system.

Same interface as :class:`SQLiteBackend` but backed by PostgreSQL with JSONB
columns.  Requires ``psycopg`` (psycopg3); if unavailable, instantiation
raises a clear :class:`ImportError`.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from datetime import date, datetime
from typing import Any, Generator

from digital_twin_tumor.domain.models import (
    AuditEvent,
    DatabaseConfig,
    GrowthModelResult,
    Lesion,
    Measurement,
    Patient,
    RECISTResponse,
    TimePoint,
)

try:
    import psycopg
    from psycopg.rows import dict_row
    _HAS_PSYCOPG = True
except ImportError:
    _HAS_PSYCOPG = False


def _ser_date(d: date | datetime | None) -> str | None:
    """ISO-8601 string from date/datetime, or None."""
    return d.isoformat() if d is not None else None


def _parse_dt(val: str | None) -> datetime | None:
    """Parse ISO-8601 string to datetime."""
    return datetime.fromisoformat(val) if val else None


def _parse_d(val: str | None) -> date | None:
    """Parse ISO-8601 string to date."""
    if val is None:
        return None
    parsed = datetime.fromisoformat(val)
    return parsed.date() if isinstance(parsed, datetime) else parsed


def _jdump(obj: Any) -> str:
    """Serialize to JSON string."""
    return json.dumps(obj, default=str)


def _ensure_dict(val: Any) -> dict:
    """Ensure a value is a dict; parse JSON strings if needed."""
    if val is None:
        return {}
    if isinstance(val, dict):
        return val
    if isinstance(val, str):
        return json.loads(val)
    return {}


class PostgresBackend:
    """PostgreSQL-based storage for the Digital Twin Tumor system.

    Requires ``psycopg`` (psycopg 3).  Raises :class:`ImportError` with
    installation instructions if the library is missing.
    """

    def __init__(self, config: DatabaseConfig) -> None:
        if not _HAS_PSYCOPG:
            raise ImportError(
                "psycopg (psycopg3) is required for PostgreSQL support. "
                "Install it with: pip install 'psycopg[binary]'"
            )
        self._config = config
        conninfo = (
            f"host={config.host} port={config.port} dbname={config.database} "
            f"user={config.user} password={config.password}"
        )
        self._conn = psycopg.connect(conninfo, row_factory=dict_row)
        self._conn.autocommit = False
        self._init_tables()

    @contextmanager
    def _cursor(self) -> Generator[Any, None, None]:
        """Yield a cursor; commit on success, rollback on failure."""
        cur = self._conn.cursor()
        try:
            yield cur
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise

    def _init_tables(self) -> None:
        """Create all required tables if they do not already exist."""
        with self._cursor() as cur:
            cur.execute(
                "CREATE TABLE IF NOT EXISTS patients ("
                "patient_id TEXT PRIMARY KEY, "
                "metadata JSONB DEFAULT '{}'::jsonb, "
                "created_at TIMESTAMPTZ NOT NULL DEFAULT NOW())"
            )
            cur.execute(
                "CREATE TABLE IF NOT EXISTS timepoints ("
                "timepoint_id TEXT PRIMARY KEY, "
                "patient_id TEXT NOT NULL REFERENCES patients(patient_id), "
                "scan_date TIMESTAMPTZ, modality TEXT DEFAULT 'CT', "
                "therapy_status TEXT DEFAULT 'pre', "
                "metadata JSONB DEFAULT '{}'::jsonb)"
            )
            cur.execute(
                "CREATE TABLE IF NOT EXISTS measurements ("
                "measurement_id TEXT PRIMARY KEY, lesion_id TEXT, "
                "timepoint_id TEXT NOT NULL REFERENCES timepoints(timepoint_id), "
                "diameter_mm REAL DEFAULT 0.0, volume_mm3 REAL DEFAULT 0.0, "
                "method TEXT DEFAULT 'auto', reviewer TEXT DEFAULT '', "
                "timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(), "
                "metadata JSONB DEFAULT '{}'::jsonb)"
            )
            cur.execute(
                "CREATE TABLE IF NOT EXISTS lesions ("
                "lesion_id TEXT PRIMARY KEY, "
                "timepoint_id TEXT NOT NULL REFERENCES timepoints(timepoint_id), "
                "centroid_x REAL DEFAULT 0.0, centroid_y REAL DEFAULT 0.0, "
                "centroid_z REAL DEFAULT 0.0, volume_mm3 REAL DEFAULT 0.0, "
                "longest_diameter_mm REAL DEFAULT 0.0, short_axis_mm REAL DEFAULT 0.0, "
                "is_target BOOLEAN DEFAULT FALSE, is_new BOOLEAN DEFAULT FALSE, "
                "organ TEXT DEFAULT '', confidence REAL DEFAULT 0.0)"
            )
            cur.execute(
                "CREATE TABLE IF NOT EXISTS recist_responses ("
                "id SERIAL PRIMARY KEY, "
                "timepoint_id TEXT NOT NULL REFERENCES timepoints(timepoint_id), "
                "sum_of_diameters REAL DEFAULT 0.0, baseline_sum REAL DEFAULT 0.0, "
                "nadir_sum REAL DEFAULT 0.0, pct_change_baseline REAL DEFAULT 0.0, "
                "pct_change_nadir REAL DEFAULT 0.0, category TEXT DEFAULT 'SD', "
                "is_confirmed BOOLEAN DEFAULT FALSE)"
            )
            cur.execute(
                "CREATE TABLE IF NOT EXISTS growth_models ("
                "id SERIAL PRIMARY KEY, "
                "patient_id TEXT NOT NULL REFERENCES patients(patient_id), "
                "model_name TEXT DEFAULT '', "
                "parameters JSONB DEFAULT '{}'::jsonb, "
                "aic REAL DEFAULT 0.0, bic REAL DEFAULT 0.0, "
                "akaike_weight REAL DEFAULT 0.0, "
                "fitted_at TIMESTAMPTZ NOT NULL DEFAULT NOW())"
            )
            cur.execute(
                "CREATE TABLE IF NOT EXISTS audit_log ("
                "event_id TEXT PRIMARY KEY, "
                "timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(), "
                "user_id TEXT DEFAULT '', action_type TEXT DEFAULT '', "
                "patient_id TEXT DEFAULT '', timepoint_id TEXT DEFAULT '', "
                "lesion_id TEXT DEFAULT '', "
                "before_state JSONB DEFAULT '{}'::jsonb, "
                "after_state JSONB DEFAULT '{}'::jsonb, "
                "metadata JSONB DEFAULT '{}'::jsonb)"
            )

    # -- Patient -----------------------------------------------------------

    def save_patient(self, patient: Patient) -> None:
        """Persist a patient record, upserting on conflict."""
        with self._cursor() as cur:
            cur.execute(
                "INSERT INTO patients (patient_id, metadata, created_at) "
                "VALUES (%s, %s::jsonb, NOW()) ON CONFLICT (patient_id) "
                "DO UPDATE SET metadata = EXCLUDED.metadata",
                (patient.patient_id, _jdump(patient.metadata)),
            )

    def get_patient(self, patient_id: str) -> Patient | None:
        """Retrieve a patient by ID, or None if not found."""
        with self._cursor() as cur:
            cur.execute("SELECT * FROM patients WHERE patient_id = %s", (patient_id,))
            row = cur.fetchone()
        if row is None:
            return None
        return Patient(patient_id=row["patient_id"], metadata=_ensure_dict(row["metadata"]))

    # -- TimePoint ---------------------------------------------------------

    def save_timepoint(self, tp: TimePoint) -> None:
        """Persist a timepoint, upserting on conflict."""
        with self._cursor() as cur:
            cur.execute(
                "INSERT INTO timepoints (timepoint_id, patient_id, scan_date, "
                "modality, therapy_status, metadata) "
                "VALUES (%s, %s, %s, %s, %s, %s::jsonb) "
                "ON CONFLICT (timepoint_id) DO UPDATE SET "
                "patient_id=EXCLUDED.patient_id, scan_date=EXCLUDED.scan_date, "
                "modality=EXCLUDED.modality, therapy_status=EXCLUDED.therapy_status, "
                "metadata=EXCLUDED.metadata",
                (tp.timepoint_id, tp.patient_id, _ser_date(tp.scan_date),
                 tp.modality, tp.therapy_status, _jdump(tp.metadata)),
            )

    def get_timepoints(self, patient_id: str) -> list[TimePoint]:
        """Retrieve all timepoints for a patient ordered by scan date."""
        with self._cursor() as cur:
            cur.execute(
                "SELECT * FROM timepoints WHERE patient_id = %s ORDER BY scan_date",
                (patient_id,),
            )
            rows = cur.fetchall()
        return [
            TimePoint(
                timepoint_id=r["timepoint_id"], patient_id=r["patient_id"],
                scan_date=_parse_d(str(r["scan_date"])) if r["scan_date"] else None,
                modality=r["modality"], therapy_status=r["therapy_status"],
                metadata=_ensure_dict(r["metadata"]),
            )
            for r in rows
        ]

    # -- Measurement -------------------------------------------------------

    def save_measurement(self, m: Measurement) -> None:
        """Persist a measurement, upserting on conflict."""
        with self._cursor() as cur:
            cur.execute(
                "INSERT INTO measurements (measurement_id, lesion_id, timepoint_id, "
                "diameter_mm, volume_mm3, method, reviewer, timestamp, metadata) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb) "
                "ON CONFLICT (measurement_id) DO UPDATE SET "
                "lesion_id=EXCLUDED.lesion_id, timepoint_id=EXCLUDED.timepoint_id, "
                "diameter_mm=EXCLUDED.diameter_mm, volume_mm3=EXCLUDED.volume_mm3, "
                "method=EXCLUDED.method, reviewer=EXCLUDED.reviewer, "
                "timestamp=EXCLUDED.timestamp, metadata=EXCLUDED.metadata",
                (m.measurement_id, m.lesion_id, m.timepoint_id, m.diameter_mm,
                 m.volume_mm3, m.method, m.reviewer, _ser_date(m.timestamp),
                 _jdump(m.metadata)),
            )

    def get_measurements(self, lesion_id: str) -> list[Measurement]:
        """Retrieve all measurements for a lesion ordered by timestamp."""
        with self._cursor() as cur:
            cur.execute(
                "SELECT * FROM measurements WHERE lesion_id = %s ORDER BY timestamp",
                (lesion_id,),
            )
            rows = cur.fetchall()
        return [
            Measurement(
                measurement_id=r["measurement_id"], lesion_id=r["lesion_id"],
                timepoint_id=r["timepoint_id"], diameter_mm=r["diameter_mm"],
                volume_mm3=r["volume_mm3"], method=r["method"], reviewer=r["reviewer"],
                timestamp=_parse_dt(str(r["timestamp"])) if r["timestamp"] else datetime.utcnow(),
                metadata=_ensure_dict(r["metadata"]),
            )
            for r in rows
        ]

    # -- Lesion ------------------------------------------------------------

    def save_lesion(self, lesion: Lesion) -> None:
        """Persist a lesion, upserting on conflict."""
        cx, cy, cz = lesion.centroid
        with self._cursor() as cur:
            cur.execute(
                "INSERT INTO lesions (lesion_id, timepoint_id, centroid_x, centroid_y, "
                "centroid_z, volume_mm3, longest_diameter_mm, short_axis_mm, "
                "is_target, is_new, organ, confidence) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) "
                "ON CONFLICT (lesion_id) DO UPDATE SET "
                "timepoint_id=EXCLUDED.timepoint_id, centroid_x=EXCLUDED.centroid_x, "
                "centroid_y=EXCLUDED.centroid_y, centroid_z=EXCLUDED.centroid_z, "
                "volume_mm3=EXCLUDED.volume_mm3, "
                "longest_diameter_mm=EXCLUDED.longest_diameter_mm, "
                "short_axis_mm=EXCLUDED.short_axis_mm, is_target=EXCLUDED.is_target, "
                "is_new=EXCLUDED.is_new, organ=EXCLUDED.organ, "
                "confidence=EXCLUDED.confidence",
                (lesion.lesion_id, lesion.timepoint_id, cx, cy, cz, lesion.volume_mm3,
                 lesion.longest_diameter_mm, lesion.short_axis_mm, lesion.is_target,
                 lesion.is_new, lesion.organ, lesion.confidence),
            )

    def get_lesions(self, timepoint_id: str) -> list[Lesion]:
        """Retrieve all lesions for a given timepoint."""
        with self._cursor() as cur:
            cur.execute(
                "SELECT * FROM lesions WHERE timepoint_id = %s", (timepoint_id,)
            )
            rows = cur.fetchall()
        return [
            Lesion(
                lesion_id=r["lesion_id"], timepoint_id=r["timepoint_id"],
                centroid=(r["centroid_x"], r["centroid_y"], r["centroid_z"]),
                volume_mm3=r["volume_mm3"], longest_diameter_mm=r["longest_diameter_mm"],
                short_axis_mm=r["short_axis_mm"], is_target=bool(r["is_target"]),
                is_new=bool(r["is_new"]), organ=r["organ"], confidence=r["confidence"],
            )
            for r in rows
        ]

    # -- RECIST Response ---------------------------------------------------

    def save_recist(self, response: RECISTResponse) -> None:
        """Persist a RECIST response record."""
        with self._cursor() as cur:
            cur.execute(
                "INSERT INTO recist_responses (timepoint_id, sum_of_diameters, "
                "baseline_sum, nadir_sum, pct_change_baseline, pct_change_nadir, "
                "category, is_confirmed) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
                (response.timepoint_id, response.sum_of_diameters, response.baseline_sum,
                 response.nadir_sum, response.percent_change_from_baseline,
                 response.percent_change_from_nadir, response.category,
                 response.is_confirmed),
            )

    def get_recist_history(self, patient_id: str) -> list[RECISTResponse]:
        """Retrieve RECIST history for a patient across all timepoints."""
        with self._cursor() as cur:
            cur.execute(
                "SELECT r.* FROM recist_responses r "
                "JOIN timepoints t ON r.timepoint_id = t.timepoint_id "
                "WHERE t.patient_id = %s ORDER BY t.scan_date",
                (patient_id,),
            )
            rows = cur.fetchall()
        return [
            RECISTResponse(
                timepoint_id=r["timepoint_id"],
                sum_of_diameters=r["sum_of_diameters"],
                baseline_sum=r["baseline_sum"], nadir_sum=r["nadir_sum"],
                percent_change_from_baseline=r["pct_change_baseline"],
                percent_change_from_nadir=r["pct_change_nadir"],
                category=r["category"], is_confirmed=bool(r["is_confirmed"]),
            )
            for r in rows
        ]

    # -- Growth Model ------------------------------------------------------

    def save_growth_model(self, patient_id: str, result: GrowthModelResult) -> None:
        """Persist a growth model fit result."""
        with self._cursor() as cur:
            cur.execute(
                "INSERT INTO growth_models (patient_id, model_name, parameters, "
                "aic, bic, akaike_weight, fitted_at) "
                "VALUES (%s, %s, %s::jsonb, %s, %s, %s, NOW())",
                (patient_id, result.model_name, _jdump(result.parameters),
                 result.aic, result.bic, result.akaike_weight),
            )

    # -- Audit -------------------------------------------------------------

    def save_audit_event(self, event: AuditEvent) -> None:
        """Persist an audit event to the audit_log table."""
        with self._cursor() as cur:
            cur.execute(
                "INSERT INTO audit_log (event_id, timestamp, user_id, action_type, "
                "patient_id, timepoint_id, lesion_id, before_state, after_state, "
                "metadata) VALUES (%s, %s, %s, %s, %s, %s, %s, "
                "%s::jsonb, %s::jsonb, %s::jsonb)",
                (event.event_id, _ser_date(event.timestamp), event.user_id,
                 event.action_type, event.patient_id, event.timepoint_id,
                 event.lesion_id, _jdump(event.before_state),
                 _jdump(event.after_state), _jdump(event.metadata)),
            )

    def get_audit_log(
        self, patient_id: str | None = None, limit: int = 100
    ) -> list[AuditEvent]:
        """Retrieve audit events, optionally filtered by patient (most recent first)."""
        with self._cursor() as cur:
            if patient_id is not None:
                cur.execute(
                    "SELECT * FROM audit_log WHERE patient_id = %s "
                    "ORDER BY timestamp DESC LIMIT %s",
                    (patient_id, limit),
                )
            else:
                cur.execute(
                    "SELECT * FROM audit_log ORDER BY timestamp DESC LIMIT %s",
                    (limit,),
                )
            rows = cur.fetchall()
        return [
            AuditEvent(
                event_id=r["event_id"],
                timestamp=_parse_dt(str(r["timestamp"])) if r["timestamp"] else datetime.utcnow(),
                user_id=r["user_id"], action_type=r["action_type"],
                patient_id=r["patient_id"], timepoint_id=r["timepoint_id"],
                lesion_id=r["lesion_id"], before_state=_ensure_dict(r["before_state"]),
                after_state=_ensure_dict(r["after_state"]),
                metadata=_ensure_dict(r["metadata"]),
            )
            for r in rows
        ]

    # -- Generic query -----------------------------------------------------

    def query(self, sql: str, params: tuple = ()) -> list[dict]:
        """Execute raw SQL with ``%s`` placeholders, returning list of dicts."""
        with self._cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()
        return list(rows)

    # -- Lifecycle ---------------------------------------------------------

    def close(self) -> None:
        """Close the underlying database connection."""
        self._conn.close()
