"""SQLite storage backend for the Digital Twin Tumor system.

Provides a lightweight persistence layer using SQLite with WAL mode for
concurrent access.  Complex fields are serialized as JSON text.  All queries
use parameterized ``?`` placeholders to prevent SQL injection.
"""
from __future__ import annotations

import json, os, sqlite3
from contextlib import contextmanager
from datetime import UTC, date, datetime
from typing import Any, Generator

import numpy as np

from digital_twin_tumor.domain.models import (
    AuditEvent, GrowthModelResult, Lesion, Measurement, NarrativeResult,
    Patient, RECISTResponse, SimulationResult, TherapyEvent, TimePoint,
)

_TABLES = [
    ("patients",
     "patient_id TEXT PRIMARY KEY, metadata TEXT DEFAULT '{}', "
     "created_at TEXT NOT NULL"),
    ("timepoints",
     "timepoint_id TEXT PRIMARY KEY, "
     "patient_id TEXT NOT NULL REFERENCES patients(patient_id), "
     "scan_date TEXT, modality TEXT DEFAULT 'CT', "
     "therapy_status TEXT DEFAULT 'pre', metadata TEXT DEFAULT '{}'"),
    ("measurements",
     "measurement_id TEXT PRIMARY KEY, lesion_id TEXT, "
     "timepoint_id TEXT NOT NULL REFERENCES timepoints(timepoint_id), "
     "diameter_mm REAL DEFAULT 0.0, volume_mm3 REAL DEFAULT 0.0, "
     "method TEXT DEFAULT 'auto', reviewer TEXT DEFAULT '', "
     "timestamp TEXT NOT NULL, metadata TEXT DEFAULT '{}'"),
    ("lesions",
     "lesion_id TEXT PRIMARY KEY, "
     "timepoint_id TEXT NOT NULL REFERENCES timepoints(timepoint_id), "
     "centroid_x REAL DEFAULT 0.0, centroid_y REAL DEFAULT 0.0, "
     "centroid_z REAL DEFAULT 0.0, volume_mm3 REAL DEFAULT 0.0, "
     "longest_diameter_mm REAL DEFAULT 0.0, short_axis_mm REAL DEFAULT 0.0, "
     "is_target BOOLEAN DEFAULT 0, is_new BOOLEAN DEFAULT 0, "
     "organ TEXT DEFAULT '', confidence REAL DEFAULT 0.0"),
    ("recist_responses",
     "id INTEGER PRIMARY KEY AUTOINCREMENT, "
     "timepoint_id TEXT NOT NULL REFERENCES timepoints(timepoint_id), "
     "sum_of_diameters REAL DEFAULT 0.0, baseline_sum REAL DEFAULT 0.0, "
     "nadir_sum REAL DEFAULT 0.0, pct_change_baseline REAL DEFAULT 0.0, "
     "pct_change_nadir REAL DEFAULT 0.0, category TEXT DEFAULT 'SD', "
     "is_confirmed BOOLEAN DEFAULT 0"),
    ("growth_models",
     "id INTEGER PRIMARY KEY AUTOINCREMENT, "
     "patient_id TEXT NOT NULL REFERENCES patients(patient_id), "
     "model_name TEXT DEFAULT '', parameters TEXT DEFAULT '{}', "
     "aic REAL DEFAULT 0.0, bic REAL DEFAULT 0.0, "
     "akaike_weight REAL DEFAULT 0.0, fitted_at TEXT NOT NULL"),
    ("audit_log",
     "event_id TEXT PRIMARY KEY, timestamp TEXT NOT NULL, "
     "user_id TEXT DEFAULT '', action_type TEXT DEFAULT '', "
     "patient_id TEXT DEFAULT '', timepoint_id TEXT DEFAULT '', "
     "lesion_id TEXT DEFAULT '', before_state TEXT DEFAULT '{}', "
     "after_state TEXT DEFAULT '{}', metadata TEXT DEFAULT '{}', "
     "session_id TEXT DEFAULT '', component TEXT DEFAULT '', "
     "data_hash TEXT DEFAULT ''"),
    ("therapy_events",
     "therapy_id TEXT PRIMARY KEY, "
     "patient_id TEXT NOT NULL REFERENCES patients(patient_id), "
     "start_date TEXT, end_date TEXT, "
     "therapy_type TEXT DEFAULT '', dose TEXT DEFAULT '', "
     "metadata TEXT DEFAULT '{}'"),
    ("simulation_results",
     "id INTEGER PRIMARY KEY AUTOINCREMENT, "
     "patient_id TEXT NOT NULL REFERENCES patients(patient_id), "
     "scenario_name TEXT DEFAULT '', time_points TEXT DEFAULT '[]', "
     "predicted_volumes TEXT DEFAULT '[]', lower_bound TEXT DEFAULT '[]', "
     "upper_bound TEXT DEFAULT '[]', parameters TEXT DEFAULT '{}', "
     "created_at TEXT NOT NULL"),
    ("narrative_results",
     "id INTEGER PRIMARY KEY AUTOINCREMENT, "
     "patient_id TEXT NOT NULL REFERENCES patients(patient_id), "
     "timepoint_id TEXT DEFAULT '', text TEXT DEFAULT '', "
     "disclaimer TEXT DEFAULT '', grounding_check BOOLEAN DEFAULT 0, "
     "safety_check BOOLEAN DEFAULT 0, generation_params TEXT DEFAULT '{}', "
     "created_at TEXT NOT NULL"),
]

_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_timepoints_patient ON timepoints(patient_id)",
    "CREATE INDEX IF NOT EXISTS idx_measurements_lesion ON measurements(lesion_id)",
    "CREATE INDEX IF NOT EXISTS idx_measurements_timepoint ON measurements(timepoint_id)",
    "CREATE INDEX IF NOT EXISTS idx_lesions_timepoint ON lesions(timepoint_id)",
    "CREATE INDEX IF NOT EXISTS idx_audit_patient ON audit_log(patient_id)",
    "CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON audit_log(timestamp)",
    "CREATE INDEX IF NOT EXISTS idx_therapy_patient ON therapy_events(patient_id)",
]


def _ser(d: date | datetime | None) -> str | None:
    return d.isoformat() if d is not None else None

def _pdt(val: str | None) -> datetime | None:
    return datetime.fromisoformat(val) if val else None

def _pd(val: str | None) -> date | None:
    if val is None:
        return None
    p = datetime.fromisoformat(val)
    return p.date() if isinstance(p, datetime) else p

def _jd(obj: Any) -> str:
    return json.dumps(obj, default=str)

def _jl(text: str | None) -> Any:
    return json.loads(text) if text else {}

def _now_iso() -> str:
    return datetime.now(UTC).replace(tzinfo=None).isoformat()


class SQLiteBackend:
    """SQLite-based storage for the Digital Twin Tumor system."""

    def __init__(self, db_path: str = ".cache/digital_twin.db",
                 max_connections: int = 5) -> None:
        self._db_path = db_path
        self._max_connections = max_connections
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.row_factory = sqlite3.Row
        self._init_tables()

    @contextmanager
    def _cursor(self) -> Generator[sqlite3.Cursor, None, None]:
        cur = self._conn.cursor()
        try:
            yield cur
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise

    def _init_tables(self) -> None:
        with self._cursor() as cur:
            for name, cols in _TABLES:
                cur.execute(f"CREATE TABLE IF NOT EXISTS {name} ({cols})")
            for idx in _INDEXES:
                cur.execute(idx)

    # -- Patient -----------------------------------------------------------
    def save_patient(self, patient: Patient) -> None:
        with self._cursor() as cur:
            cur.execute(
                "INSERT INTO patients (patient_id, metadata, created_at) "
                "VALUES (?, ?, ?) ON CONFLICT(patient_id) DO UPDATE SET "
                "metadata=excluded.metadata, created_at=excluded.created_at",
                (patient.patient_id, _jd(patient.metadata), _now_iso()))

    def get_patient(self, patient_id: str) -> Patient | None:
        with self._cursor() as cur:
            cur.execute("SELECT * FROM patients WHERE patient_id = ?", (patient_id,))
            row = cur.fetchone()
        if row is None:
            return None
        return Patient(patient_id=row["patient_id"], metadata=_jl(row["metadata"]))

    def list_patients(self, limit: int = 100) -> list[Patient]:
        with self._cursor() as cur:
            cur.execute("SELECT * FROM patients LIMIT ?", (limit,))
            rows = cur.fetchall()
        return [Patient(patient_id=r["patient_id"], metadata=_jl(r["metadata"]))
                for r in rows]

    def delete_patient(self, patient_id: str) -> None:
        """Cascade delete a patient and all related records."""
        with self._cursor() as cur:
            tp_sub = "(SELECT timepoint_id FROM timepoints WHERE patient_id = ?)"
            for tbl in ("measurements", "lesions", "recist_responses"):
                cur.execute(f"DELETE FROM {tbl} WHERE timepoint_id IN {tp_sub}",
                            (patient_id,))
            for tbl in ("timepoints", "growth_models", "therapy_events",
                        "simulation_results", "narrative_results", "audit_log"):
                cur.execute(f"DELETE FROM {tbl} WHERE patient_id = ?", (patient_id,))
            cur.execute("DELETE FROM patients WHERE patient_id = ?", (patient_id,))

    # -- TimePoint ---------------------------------------------------------
    def save_timepoint(self, tp: TimePoint) -> None:
        with self._cursor() as cur:
            cur.execute(
                "INSERT INTO timepoints (timepoint_id, patient_id, scan_date, "
                "modality, therapy_status, metadata) VALUES (?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(timepoint_id) DO UPDATE SET patient_id=excluded.patient_id, "
                "scan_date=excluded.scan_date, modality=excluded.modality, "
                "therapy_status=excluded.therapy_status, metadata=excluded.metadata",
                (tp.timepoint_id, tp.patient_id, _ser(tp.scan_date),
                 tp.modality, tp.therapy_status, _jd(tp.metadata)))

    def get_timepoints(self, patient_id: str) -> list[TimePoint]:
        with self._cursor() as cur:
            cur.execute(
                "SELECT * FROM timepoints WHERE patient_id = ? ORDER BY scan_date",
                (patient_id,))
            rows = cur.fetchall()
        return [TimePoint(
            timepoint_id=r["timepoint_id"], patient_id=r["patient_id"],
            scan_date=_pd(r["scan_date"]), modality=r["modality"],
            therapy_status=r["therapy_status"], metadata=_jl(r["metadata"]))
            for r in rows]

    # -- Measurement -------------------------------------------------------
    def _ensure_timepoint(self, timepoint_id: str) -> None:
        """Create stub timepoint (and patient) if missing to satisfy FK."""
        with self._cursor() as cur:
            cur.execute("SELECT 1 FROM timepoints WHERE timepoint_id = ?",
                        (timepoint_id,))
            if cur.fetchone() is not None:
                return
            stub_pid = f"_stub_{timepoint_id}"
            cur.execute(
                "INSERT OR IGNORE INTO patients (patient_id, metadata, created_at) "
                "VALUES (?, '{}', ?)", (stub_pid, _now_iso()))
            cur.execute(
                "INSERT OR IGNORE INTO timepoints "
                "(timepoint_id, patient_id, scan_date, modality) VALUES (?, ?, ?, 'CT')",
                (timepoint_id, stub_pid, _now_iso()))

    def save_measurement(self, m: Measurement) -> None:
        self._ensure_timepoint(m.timepoint_id)
        with self._cursor() as cur:
            cur.execute(
                "INSERT INTO measurements (measurement_id, lesion_id, timepoint_id, "
                "diameter_mm, volume_mm3, method, reviewer, timestamp, metadata) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(measurement_id) DO UPDATE SET "
                "lesion_id=excluded.lesion_id, timepoint_id=excluded.timepoint_id, "
                "diameter_mm=excluded.diameter_mm, volume_mm3=excluded.volume_mm3, "
                "method=excluded.method, reviewer=excluded.reviewer, "
                "timestamp=excluded.timestamp, metadata=excluded.metadata",
                (m.measurement_id, m.lesion_id, m.timepoint_id, m.diameter_mm,
                 m.volume_mm3, m.method, m.reviewer, _ser(m.timestamp), _jd(m.metadata)))

    def _row_to_measurement(self, r: sqlite3.Row) -> Measurement:
        return Measurement(
            measurement_id=r["measurement_id"], lesion_id=r["lesion_id"],
            timepoint_id=r["timepoint_id"], diameter_mm=r["diameter_mm"],
            volume_mm3=r["volume_mm3"], method=r["method"], reviewer=r["reviewer"],
            timestamp=_pdt(r["timestamp"]) or datetime.now(UTC).replace(tzinfo=None),
            metadata=_jl(r["metadata"]))

    def get_measurements(self, lesion_id: str) -> list[Measurement]:
        with self._cursor() as cur:
            cur.execute("SELECT * FROM measurements WHERE lesion_id = ? ORDER BY timestamp",
                        (lesion_id,))
            return [self._row_to_measurement(r) for r in cur.fetchall()]

    def get_measurements_by_timepoint(self, timepoint_id: str) -> list[Measurement]:
        with self._cursor() as cur:
            cur.execute("SELECT * FROM measurements WHERE timepoint_id = ? ORDER BY timestamp",
                        (timepoint_id,))
            return [self._row_to_measurement(r) for r in cur.fetchall()]

    def delete_measurement(self, measurement_id: str) -> None:
        with self._cursor() as cur:
            cur.execute("DELETE FROM measurements WHERE measurement_id = ?",
                        (measurement_id,))

    # -- Lesion ------------------------------------------------------------
    def save_lesion(self, lesion: Lesion) -> None:
        cx, cy, cz = lesion.centroid
        with self._cursor() as cur:
            cur.execute(
                "INSERT INTO lesions (lesion_id, timepoint_id, centroid_x, centroid_y, "
                "centroid_z, volume_mm3, longest_diameter_mm, short_axis_mm, is_target, "
                "is_new, organ, confidence) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(lesion_id) DO UPDATE SET "
                "timepoint_id=excluded.timepoint_id, centroid_x=excluded.centroid_x, "
                "centroid_y=excluded.centroid_y, centroid_z=excluded.centroid_z, "
                "volume_mm3=excluded.volume_mm3, "
                "longest_diameter_mm=excluded.longest_diameter_mm, "
                "short_axis_mm=excluded.short_axis_mm, is_target=excluded.is_target, "
                "is_new=excluded.is_new, organ=excluded.organ, "
                "confidence=excluded.confidence",
                (lesion.lesion_id, lesion.timepoint_id, cx, cy, cz, lesion.volume_mm3,
                 lesion.longest_diameter_mm, lesion.short_axis_mm, lesion.is_target,
                 lesion.is_new, lesion.organ, lesion.confidence))

    def get_lesions(self, timepoint_id: str) -> list[Lesion]:
        with self._cursor() as cur:
            cur.execute("SELECT * FROM lesions WHERE timepoint_id = ?", (timepoint_id,))
            rows = cur.fetchall()
        return [Lesion(
            lesion_id=r["lesion_id"], timepoint_id=r["timepoint_id"],
            centroid=(r["centroid_x"], r["centroid_y"], r["centroid_z"]),
            volume_mm3=r["volume_mm3"], longest_diameter_mm=r["longest_diameter_mm"],
            short_axis_mm=r["short_axis_mm"], is_target=bool(r["is_target"]),
            is_new=bool(r["is_new"]), organ=r["organ"], confidence=r["confidence"])
            for r in rows]

    # -- RECIST Response ---------------------------------------------------
    def save_recist(self, response: RECISTResponse) -> None:
        with self._cursor() as cur:
            cur.execute(
                "INSERT INTO recist_responses (timepoint_id, sum_of_diameters, "
                "baseline_sum, nadir_sum, pct_change_baseline, pct_change_nadir, "
                "category, is_confirmed) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (response.timepoint_id, response.sum_of_diameters, response.baseline_sum,
                 response.nadir_sum, response.percent_change_from_baseline,
                 response.percent_change_from_nadir, response.category,
                 response.is_confirmed))

    def get_recist_history(self, patient_id: str) -> list[RECISTResponse]:
        with self._cursor() as cur:
            cur.execute(
                "SELECT r.* FROM recist_responses r "
                "JOIN timepoints t ON r.timepoint_id = t.timepoint_id "
                "WHERE t.patient_id = ? ORDER BY t.scan_date", (patient_id,))
            rows = cur.fetchall()
        return [RECISTResponse(
            timepoint_id=r["timepoint_id"], sum_of_diameters=r["sum_of_diameters"],
            baseline_sum=r["baseline_sum"], nadir_sum=r["nadir_sum"],
            percent_change_from_baseline=r["pct_change_baseline"],
            percent_change_from_nadir=r["pct_change_nadir"],
            category=r["category"], is_confirmed=bool(r["is_confirmed"]))
            for r in rows]

    # -- Growth Model ------------------------------------------------------
    def save_growth_model(self, patient_id: str, result: GrowthModelResult) -> None:
        with self._cursor() as cur:
            cur.execute(
                "INSERT INTO growth_models (patient_id, model_name, parameters, "
                "aic, bic, akaike_weight, fitted_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (patient_id, result.model_name, _jd(result.parameters),
                 result.aic, result.bic, result.akaike_weight, _now_iso()))

    # -- Audit -------------------------------------------------------------
    def save_audit_event(self, event: AuditEvent, session_id: str = "",
                         component: str = "", data_hash: str = "") -> None:
        with self._cursor() as cur:
            cur.execute(
                "INSERT INTO audit_log (event_id, timestamp, user_id, action_type, "
                "patient_id, timepoint_id, lesion_id, before_state, after_state, "
                "metadata, session_id, component, data_hash) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (event.event_id, _ser(event.timestamp), event.user_id,
                 event.action_type, event.patient_id, event.timepoint_id,
                 event.lesion_id, _jd(event.before_state), _jd(event.after_state),
                 _jd(event.metadata), session_id, component, data_hash))

    def get_audit_log(self, patient_id: str | None = None,
                      limit: int = 100) -> list[AuditEvent]:
        """Return audit entries (session_id/component/data_hash merged into metadata)."""
        with self._cursor() as cur:
            if patient_id is not None:
                cur.execute(
                    "SELECT * FROM audit_log WHERE patient_id = ? "
                    "ORDER BY timestamp DESC LIMIT ?", (patient_id, limit))
            else:
                cur.execute("SELECT * FROM audit_log ORDER BY timestamp DESC LIMIT ?",
                            (limit,))
            rows = cur.fetchall()
        results: list[AuditEvent] = []
        for r in rows:
            meta = _jl(r["metadata"])
            meta.update({"session_id": r["session_id"], "component": r["component"],
                         "data_hash": r["data_hash"]})
            results.append(AuditEvent(
                event_id=r["event_id"],
                timestamp=_pdt(r["timestamp"]) or datetime.now(UTC).replace(tzinfo=None),
                user_id=r["user_id"], action_type=r["action_type"],
                patient_id=r["patient_id"], timepoint_id=r["timepoint_id"],
                lesion_id=r["lesion_id"], before_state=_jl(r["before_state"]),
                after_state=_jl(r["after_state"]), metadata=meta))
        return results

    # -- TherapyEvent ------------------------------------------------------
    def save_therapy_event(self, event: TherapyEvent) -> None:
        with self._cursor() as cur:
            cur.execute(
                "INSERT INTO therapy_events (therapy_id, patient_id, start_date, "
                "end_date, therapy_type, dose, metadata) "
                "VALUES (?, ?, ?, ?, ?, ?, ?) ON CONFLICT(therapy_id) DO UPDATE SET "
                "patient_id=excluded.patient_id, start_date=excluded.start_date, "
                "end_date=excluded.end_date, therapy_type=excluded.therapy_type, "
                "dose=excluded.dose, metadata=excluded.metadata",
                (event.therapy_id, event.patient_id, _ser(event.start_date),
                 _ser(event.end_date), event.therapy_type, event.dose,
                 _jd(event.metadata)))

    def get_therapy_events(self, patient_id: str) -> list[TherapyEvent]:
        with self._cursor() as cur:
            cur.execute("SELECT * FROM therapy_events WHERE patient_id = ? "
                        "ORDER BY start_date", (patient_id,))
            rows = cur.fetchall()
        return [TherapyEvent(
            therapy_id=r["therapy_id"], patient_id=r["patient_id"],
            start_date=_pd(r["start_date"]), end_date=_pd(r["end_date"]),
            therapy_type=r["therapy_type"], dose=r["dose"],
            metadata=_jl(r["metadata"]))
            for r in rows]

    # -- SimulationResult --------------------------------------------------
    def save_simulation_result(self, patient_id: str, result: SimulationResult) -> None:
        with self._cursor() as cur:
            cur.execute(
                "INSERT INTO simulation_results (patient_id, scenario_name, "
                "time_points, predicted_volumes, lower_bound, upper_bound, "
                "parameters, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (patient_id, result.scenario_name,
                 json.dumps(result.time_points.tolist()),
                 json.dumps(result.predicted_volumes.tolist()),
                 json.dumps(result.lower_bound.tolist()),
                 json.dumps(result.upper_bound.tolist()),
                 _jd(result.parameters), _now_iso()))

    def get_simulation_results(self, patient_id: str) -> list[SimulationResult]:
        with self._cursor() as cur:
            cur.execute("SELECT * FROM simulation_results WHERE patient_id = ? "
                        "ORDER BY created_at", (patient_id,))
            rows = cur.fetchall()
        return [SimulationResult(
            scenario_name=r["scenario_name"],
            time_points=np.array(json.loads(r["time_points"])),
            predicted_volumes=np.array(json.loads(r["predicted_volumes"])),
            lower_bound=np.array(json.loads(r["lower_bound"])),
            upper_bound=np.array(json.loads(r["upper_bound"])),
            parameters=_jl(r["parameters"]))
            for r in rows]

    # -- NarrativeResult ---------------------------------------------------
    def save_narrative_result(self, patient_id: str, result: NarrativeResult,
                              timepoint_id: str = "") -> None:
        with self._cursor() as cur:
            cur.execute(
                "INSERT INTO narrative_results (patient_id, timepoint_id, text, "
                "disclaimer, grounding_check, safety_check, generation_params, "
                "created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (patient_id, timepoint_id, result.text, result.disclaimer,
                 result.grounding_check, result.safety_check,
                 _jd(result.generation_params), _now_iso()))

    # -- Health check / generic query / lifecycle --------------------------
    def health_check(self) -> bool:
        """Verify database connection is alive."""
        try:
            self._conn.execute("SELECT 1")
            return True
        except sqlite3.Error:
            return False

    def query(self, sql: str, params: tuple = ()) -> list[dict]:
        """Execute raw SQL with ``?`` placeholders, returning list of dicts."""
        with self._cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()
        return [dict(r) for r in rows]

    def close(self) -> None:
        """Close the underlying database connection."""
        self._conn.close()
