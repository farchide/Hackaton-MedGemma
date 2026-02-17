"""Load pre-generated demo patient data from the SQLite database.

Provides a high-level API for the Gradio UI to load complete patient
scenarios including measurements, RECIST classifications, growth models,
simulation results, and regenerated 3D volumes.
"""
from __future__ import annotations

import json
import logging
from datetime import date
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
)
from digital_twin_tumor.storage.sqlite_backend import SQLiteBackend

logger = logging.getLogger(__name__)


class DemoLoader:
    """Load demo patient scenarios from a pre-populated SQLite database.

    Parameters
    ----------
    db_path:
        Path to the demo database (created by ``generate_all_demo_data``).
    """

    def __init__(self, db_path: str = ".cache/demo.db") -> None:
        self._db = SQLiteBackend(db_path)

    def close(self) -> None:
        """Close the database connection."""
        self._db.close()

    # ------------------------------------------------------------------
    # Patient listing
    # ------------------------------------------------------------------

    def list_patients(self) -> list[dict[str, str]]:
        """Return a list of patient summaries for the UI selector.

        Each entry contains ``patient_id``, ``label``, ``scenario``,
        ``cancer_type``, ``timepoint_count``.
        """
        patients = self._db.list_patients(limit=50)
        result: list[dict[str, str]] = []
        for p in patients:
            meta = p.metadata
            scenario = meta.get("scenario", "Unknown")
            cancer = meta.get("cancer_type", "")
            name = meta.get("name", p.patient_id[:12])
            tps = self._db.get_timepoints(p.patient_id)
            result.append({
                "patient_id": p.patient_id,
                "label": f"{name}: {scenario}",
                "scenario": scenario,
                "cancer_type": cancer,
                "timepoint_count": str(len(tps)),
            })
        return result

    def get_patient_choices(self) -> list[str]:
        """Return dropdown-friendly list: 'label | patient_id'."""
        patients = self.list_patients()
        return [f"{p['label']} ({p['timepoint_count']} scans)" for p in patients]

    def get_patient_ids(self) -> list[str]:
        """Return patient IDs in same order as get_patient_choices."""
        return [p["patient_id"] for p in self.list_patients()]

    # ------------------------------------------------------------------
    # Full patient load
    # ------------------------------------------------------------------

    def load_patient(self, patient_id: str) -> dict[str, Any]:
        """Load all data for a single patient.

        Returns a dict containing:
        - patient: Patient domain object
        - timepoints: list of TimePoint objects sorted by scan_date
        - lesions: dict mapping timepoint_id -> list of Lesion objects
        - measurements: dict mapping timepoint_id -> list of Measurement objects
        - recist_responses: list of RECISTResponse objects
        - growth_models: list of dicts with model info
        - simulation_results: list of SimulationResult objects
        - therapy_events: list of TherapyEvent objects
        """
        patient = self._db.get_patient(patient_id)
        if patient is None:
            return {}

        timepoints = self._db.get_timepoints(patient_id)
        recist = self._db.get_recist_history(patient_id)
        therapies = self._db.get_therapy_events(patient_id)
        simulations = self._db.get_simulation_results(patient_id)

        # Growth models (raw query since no dedicated method)
        gm_rows = self._db.query(
            "SELECT * FROM growth_models WHERE patient_id = ? ORDER BY aic",
            (patient_id,),
        )
        growth_models = []
        for r in gm_rows:
            params = json.loads(r.get("parameters", "{}"))
            growth_models.append({
                "model_name": r.get("model_name", ""),
                "aic": r.get("aic", 0.0),
                "bic": r.get("bic", 0.0),
                "akaike_weight": r.get("akaike_weight", 0.0),
                "parameters": params,
            })

        # Load lesions and measurements per timepoint
        all_lesions: dict[str, list[Lesion]] = {}
        all_measurements: dict[str, list[Measurement]] = {}
        for tp in timepoints:
            all_lesions[tp.timepoint_id] = self._db.get_lesions(tp.timepoint_id)
            all_measurements[tp.timepoint_id] = (
                self._db.get_measurements_by_timepoint(tp.timepoint_id)
            )

        return {
            "patient": patient,
            "timepoints": timepoints,
            "lesions": all_lesions,
            "measurements": all_measurements,
            "recist_responses": recist,
            "growth_models": growth_models,
            "simulation_results": simulations,
            "therapy_events": therapies,
        }

    # ------------------------------------------------------------------
    # UI state builder
    # ------------------------------------------------------------------

    def build_ui_state(self, patient_id: str) -> dict[str, Any]:
        """Build the full Gradio UI state dict from a demo patient.

        Returns a dict compatible with the ``_empty()`` state format
        used by ``app.py``.
        """
        data = self.load_patient(patient_id)
        if not data:
            return {}

        patient = data["patient"]
        timepoints = data["timepoints"]
        all_lesions = data["lesions"]
        all_measurements = data["measurements"]

        # Flatten measurements into the UI list format
        flat_measurements: list[dict[str, Any]] = []
        for tp in timepoints:
            tp_id = tp.timepoint_id
            week = tp.metadata.get("week", "?")
            scan_date = tp.scan_date.isoformat() if tp.scan_date else "?"
            for m in all_measurements.get(tp_id, []):
                flat_measurements.append({
                    "measurement_id": m.measurement_id,
                    "lesion_id": m.lesion_id,
                    "timepoint_id": tp_id,
                    "diameter_mm": m.diameter_mm,
                    "volume_mm3": m.volume_mm3,
                    "method": m.method,
                    "reviewer": m.reviewer,
                    "week": week,
                    "scan_date": scan_date,
                })

        # RECIST as dicts for UI
        recist_dicts = [
            {
                "timepoint_id": r.timepoint_id,
                "sum_of_diameters": r.sum_of_diameters,
                "baseline_sum": r.baseline_sum,
                "nadir_sum": r.nadir_sum,
                "percent_change_from_baseline": r.percent_change_from_baseline,
                "percent_change_from_nadir": r.percent_change_from_nadir,
                "category": r.category,
            }
            for r in data["recist_responses"]
        ]

        # Build tracking graph data
        tracking_nodes = []
        tracking_edges = []
        by_canonical: dict[str, list[int]] = {}
        for tp in timepoints:
            tp_id = tp.timepoint_id
            for les in all_lesions.get(tp_id, []):
                idx = len(tracking_nodes)
                node_id = f"{les.lesion_id}@{tp_id}"
                tracking_nodes.append({
                    "id": node_id,
                    "lesion_id": les.lesion_id,
                    "timepoint_id": tp_id,
                    "volume": les.volume_mm3,
                    "organ": les.organ,
                    "is_target": les.is_target,
                })
                # Group by canonical ID prefix (before _tp)
                canonical = les.lesion_id.rsplit("_tp", 1)[0]
                by_canonical.setdefault(canonical, []).append(idx)

        for indices in by_canonical.values():
            for j in range(len(indices) - 1):
                src = tracking_nodes[indices[j]]["id"]
                tgt = tracking_nodes[indices[j + 1]]["id"]
                tracking_edges.append({
                    "source": src,
                    "target": tgt,
                    "confidence": 0.95,
                })

        graph_data = {"nodes": tracking_nodes, "edges": tracking_edges}

        # Generate a 3D volume for the first timepoint
        volume = None
        if timepoints and all_lesions.get(timepoints[0].timepoint_id):
            try:
                from digital_twin_tumor.data.synthetic import generate_demo_volumes
                first_tp = timepoints[0]
                first_lesions = all_lesions[first_tp.timepoint_id]
                volume = generate_demo_volumes(
                    patient_id, first_tp.timepoint_id, first_lesions,
                    rng=np.random.default_rng(42),
                )
            except Exception as exc:
                logger.warning("Failed to generate demo volume: %s", exc)

        # Growth model info
        gm_list = data["growth_models"]
        if gm_list:
            aics = [g["aic"] for g in gm_list]
            min_aic = min(aics)
            for g in gm_list:
                delta = g["aic"] - min_aic
                g["weight"] = float(np.exp(-0.5 * delta))
            total_w = sum(g["weight"] for g in gm_list)
            if total_w > 0:
                for g in gm_list:
                    g["weight"] /= total_w

        return {
            "volume": volume,
            "volume_metadata": {
                "patient_id": patient_id,
                "scenario": patient.metadata.get("scenario", ""),
                "cancer_type": patient.metadata.get("cancer_type", ""),
                "shape": list(volume.shape) if volume is not None else [],
                "spacing": [2.0, 2.0, 2.0],
            },
            "lesions": [
                {
                    "lesion_id": les.lesion_id,
                    "timepoint_id": les.timepoint_id,
                    "organ": les.organ,
                    "diameter_mm": les.longest_diameter_mm,
                    "volume_mm3": les.volume_mm3,
                    "is_target": les.is_target,
                    "confidence": les.confidence,
                }
                for tp in timepoints
                for les in all_lesions.get(tp.timepoint_id, [])
            ],
            "measurements": flat_measurements,
            "tracking_graph_json": json.dumps(graph_data),
            "growth_results": gm_list,
            "simulations": [
                {"scenario_name": s.scenario_name}
                for s in data["simulation_results"]
            ],
            "recist_responses": recist_dicts,
            "narrative": None,
            "timepoints": [
                {
                    "timepoint_id": tp.timepoint_id,
                    "scan_date": tp.scan_date.isoformat() if tp.scan_date else "",
                    "week": tp.metadata.get("week", ""),
                    "modality": tp.modality,
                    "therapy_status": tp.therapy_status,
                }
                for tp in timepoints
            ],
            "therapy_events": [
                {
                    "therapy_type": t.therapy_type,
                    "dose": t.dose,
                    "start_date": t.start_date.isoformat() if t.start_date else "",
                    "end_date": t.end_date.isoformat() if t.end_date else "",
                    "drug_name": t.metadata.get("drug_name", ""),
                }
                for t in data["therapy_events"]
            ],
            "patient_metadata": patient.metadata,
            "simulation_results_full": data["simulation_results"],
        }

    # ------------------------------------------------------------------
    # Formatted summaries
    # ------------------------------------------------------------------

    def get_patient_dashboard_html(self, patient_id: str) -> str:
        """Generate an HTML dashboard card for a patient."""
        data = self.load_patient(patient_id)
        if not data:
            return "<p>Patient not found.</p>"

        patient = data["patient"]
        meta = patient.metadata
        timepoints = data["timepoints"]
        recist = data["recist_responses"]
        therapies = data["therapy_events"]

        # Count totals
        total_lesions = sum(
            len(data["lesions"].get(tp.timepoint_id, []))
            for tp in timepoints
        )
        total_meas = sum(
            len(data["measurements"].get(tp.timepoint_id, []))
            for tp in timepoints
        )

        # RECIST trajectory
        trajectory = " &rarr; ".join(
            f'<span style="color:{_recist_color(r.category)};font-weight:700">'
            f"{r.category}</span>"
            for r in recist
        )

        # Best growth model
        gm = data["growth_models"]
        best_model = "N/A"
        if gm:
            best = min(gm, key=lambda g: g["aic"])
            best_model = best["model_name"]

        # Therapy info
        therapy_html = ""
        for t in therapies:
            drug = t.metadata.get("drug_name", t.dose)
            therapy_html += (
                f'<div style="padding:4px 0;border-bottom:1px solid #1e293b;">'
                f'<span style="color:#a78bfa">{t.therapy_type.title()}</span> '
                f'<span style="color:#94a3b8">{drug}</span></div>'
            )

        html = f"""
<div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;">
  <div style="background:rgba(15,17,23,0.9);border:1px solid #1e293b;
              border-radius:12px;padding:20px;">
    <h3 style="color:#67e8f9;margin:0 0 12px;font-size:1.1rem;">
      Patient Profile</h3>
    <table style="width:100%;border-collapse:collapse;">
      <tr><td style="color:#94a3b8;padding:4px 8px;">Scenario</td>
          <td style="color:#f1f5f9;font-weight:600;">{meta.get("scenario","")}</td></tr>
      <tr><td style="color:#94a3b8;padding:4px 8px;">Cancer Type</td>
          <td style="color:#f1f5f9;">{meta.get("cancer_type","")}</td></tr>
      <tr><td style="color:#94a3b8;padding:4px 8px;">Stage</td>
          <td style="color:#f1f5f9;">{meta.get("stage","")}</td></tr>
      <tr><td style="color:#94a3b8;padding:4px 8px;">Age / Sex</td>
          <td style="color:#f1f5f9;">{meta.get("age","")} / {meta.get("sex","")}</td></tr>
      <tr><td style="color:#94a3b8;padding:4px 8px;">ECOG PS</td>
          <td style="color:#f1f5f9;">{meta.get("ECOG_PS","")}</td></tr>
      <tr><td style="color:#94a3b8;padding:4px 8px;">Histology</td>
          <td style="color:#f1f5f9;">{meta.get("histology","")}</td></tr>
    </table>
  </div>

  <div style="background:rgba(15,17,23,0.9);border:1px solid #1e293b;
              border-radius:12px;padding:20px;">
    <h3 style="color:#67e8f9;margin:0 0 12px;font-size:1.1rem;">
      Study Summary</h3>
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;">
      <div style="text-align:center;padding:12px;background:#141722;
                  border-radius:8px;border:1px solid #1e293b;">
        <div style="font-size:1.8rem;font-weight:800;color:#22d3ee;">
          {len(timepoints)}</div>
        <div style="color:#94a3b8;font-size:0.8rem;">Timepoints</div>
      </div>
      <div style="text-align:center;padding:12px;background:#141722;
                  border-radius:8px;border:1px solid #1e293b;">
        <div style="font-size:1.8rem;font-weight:800;color:#22d3ee;">
          {total_lesions}</div>
        <div style="color:#94a3b8;font-size:0.8rem;">Lesion Obs</div>
      </div>
      <div style="text-align:center;padding:12px;background:#141722;
                  border-radius:8px;border:1px solid #1e293b;">
        <div style="font-size:1.8rem;font-weight:800;color:#22d3ee;">
          {total_meas}</div>
        <div style="color:#94a3b8;font-size:0.8rem;">Measurements</div>
      </div>
      <div style="text-align:center;padding:12px;background:#141722;
                  border-radius:8px;border:1px solid #1e293b;">
        <div style="font-size:1.8rem;font-weight:800;color:#22d3ee;">
          {len(data["simulation_results"])}</div>
        <div style="color:#94a3b8;font-size:0.8rem;">Simulations</div>
      </div>
    </div>
  </div>

  <div style="background:rgba(15,17,23,0.9);border:1px solid #1e293b;
              border-radius:12px;padding:20px;">
    <h3 style="color:#67e8f9;margin:0 0 12px;font-size:1.1rem;">
      RECIST Trajectory</h3>
    <div style="font-size:1.1rem;line-height:2;">{trajectory}</div>
    <div style="margin-top:8px;color:#94a3b8;font-size:0.85rem;">
      Best growth model: <span style="color:#a78bfa;">{best_model}</span>
    </div>
  </div>

  <div style="background:rgba(15,17,23,0.9);border:1px solid #1e293b;
              border-radius:12px;padding:20px;">
    <h3 style="color:#67e8f9;margin:0 0 12px;font-size:1.1rem;">
      Therapy Timeline</h3>
    {therapy_html or '<p style="color:#94a3b8;">No therapy recorded.</p>'}
  </div>
</div>
"""
        return html

    def get_timeline_html(self, patient_id: str) -> str:
        """Generate a visual timeline of scans and RECIST responses."""
        data = self.load_patient(patient_id)
        if not data:
            return ""

        timepoints = data["timepoints"]
        recist = data["recist_responses"]
        all_lesions = data["lesions"]

        # Map timepoint_id to RECIST
        recist_map = {r.timepoint_id: r for r in recist}

        items_html = ""
        for i, tp in enumerate(timepoints):
            week = tp.metadata.get("week", "?")
            scan_date = tp.scan_date.isoformat() if tp.scan_date else "?"
            r = recist_map.get(tp.timepoint_id)
            cat = r.category if r else "N/A"
            pct = r.percent_change_from_baseline if r else 0.0
            sod = r.sum_of_diameters if r else 0.0
            lesions = all_lesions.get(tp.timepoint_id, [])
            n_target = sum(1 for l in lesions if l.is_target)
            n_new = sum(1 for l in lesions if l.is_new)
            color = _recist_color(cat)

            new_badge = ""
            if n_new > 0:
                new_badge = (
                    ' <span style="background:#ef4444;color:#fff;'
                    'padding:1px 6px;border-radius:10px;font-size:0.7rem;">'
                    f"+{n_new} new</span>"
                )

            items_html += f"""
<div style="display:flex;align-items:flex-start;gap:12px;margin-bottom:8px;">
  <div style="text-align:center;min-width:56px;">
    <div style="color:#67e8f9;font-weight:700;font-size:0.9rem;">
      Wk {week}</div>
    <div style="color:#475569;font-size:0.75rem;">{scan_date}</div>
  </div>
  <div style="width:3px;background:{color};min-height:48px;
              border-radius:2px;flex-shrink:0;"></div>
  <div style="flex:1;background:#141722;border:1px solid #1e293b;
              border-radius:8px;padding:10px 14px;">
    <div>
      <span style="color:{color};font-weight:700;font-size:1rem;">
        {cat}</span>
      <span style="color:#94a3b8;font-size:0.85rem;">
        &nbsp;SoD: {sod:.1f}mm ({pct:+.1f}%)</span>
      {new_badge}
    </div>
    <div style="color:#64748b;font-size:0.8rem;margin-top:2px;">
      {n_target} target lesion(s) &bull; {tp.therapy_status}
      &bull; {tp.modality}
    </div>
  </div>
</div>
"""

        return f"""
<div style="padding:8px 0;">
  <h3 style="color:#67e8f9;margin:0 0 16px;font-size:1.1rem;">
    Chronological Timeline</h3>
  {items_html}
</div>
"""


def _recist_color(category: str) -> str:
    """Map RECIST category to a display color."""
    _colors = {
        "CR": "#10b981",
        "PR": "#22d3ee",
        "SD": "#f59e0b",
        "PD": "#ef4444",
        "iCR": "#10b981",
        "iPR": "#22d3ee",
        "iSD": "#f59e0b",
        "iUPD": "#f97316",
        "iCPD": "#ef4444",
    }
    return _colors.get(category, "#94a3b8")
