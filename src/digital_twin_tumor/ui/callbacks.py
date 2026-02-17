"""Callback helpers for the Digital Twin Tumor UI.

Extracted from app.py to keep the main module under 500 lines.
"""
from __future__ import annotations

import json, logging
from pathlib import Path
from typing import Any

import numpy as np

from digital_twin_tumor.ui.components import (
    create_growth_plot, create_identity_graph_plot,
    create_measurement_overlay, create_recist_waterfall,
    extract_slice, format_measurement_table,
)

logger = logging.getLogger(__name__)


def demo_timepoint_volume(tp_choice, state, _empty_fn):
    """Regenerate volume for a selected timepoint."""
    state = dict(state) if state else _empty_fn()
    timepoints = state.get("timepoints", [])
    if not tp_choice or not timepoints:
        return np.zeros((256, 256), dtype=np.uint8), "No timepoint selected.", state
    tp_idx = next((i for i, tp in enumerate(timepoints)
                   if f"Week {tp.get('week', '?')} ({tp.get('scan_date', '')})" == tp_choice), None)
    if tp_idx is None:
        return np.zeros((256, 256), dtype=np.uint8), "Timepoint not found.", state
    tp = timepoints[tp_idx]
    tp_id = tp["timepoint_id"]
    from digital_twin_tumor.domain.models import Lesion
    tp_lesions = [
        Lesion(lesion_id=l["lesion_id"], timepoint_id=tp_id,
               centroid=(0.0, 0.0, 0.0), volume_mm3=l.get("volume_mm3", 0),
               longest_diameter_mm=l.get("diameter_mm", 0),
               is_target=l.get("is_target", False), organ=l.get("organ", ""),
               confidence=l.get("confidence", 0.9))
        for l in state.get("lesions", []) if l.get("timepoint_id") == tp_id
    ]
    if not tp_lesions:
        return np.zeros((256, 256), dtype=np.uint8), f"Week {tp.get('week')}: no lesions.", state
    try:
        from digital_twin_tumor.data.synthetic import generate_demo_volumes
        vol = generate_demo_volumes(tp_id[:8], tp_id, tp_lesions,
                                    rng=np.random.default_rng(42 + tp_idx))
        state["volume"] = vol
        return (extract_slice(vol, vol.shape[0] // 2),
                f"Week {tp.get('week')}: {len(tp_lesions)} lesion(s)", state)
    except Exception as exc:
        logger.warning("Volume generation failed: %s", exc)
        return np.zeros((256, 256), dtype=np.uint8), str(exc), state


def demo_growth_plot(state, _empty_fn):
    state = state or _empty_fn()
    ms = state.get("measurements", [])
    if not ms:
        return create_growth_plot(), "No data."
    tp_vols: dict[str, float] = {}
    tp_weeks: dict[str, float] = {}
    for m in ms:
        tid = m.get("timepoint_id", "")
        tp_vols[tid] = tp_vols.get(tid, 0) + float(m.get("volume_mm3", 0))
        w = m.get("week", 0)
        if isinstance(w, (int, float)):
            tp_weeks[tid] = float(w)
    sorted_tps = sorted(tp_weeks.items(), key=lambda x: x[1])
    times = np.array([w for _, w in sorted_tps], dtype=np.float64)
    volumes = np.array([tp_vols.get(t, 0) for t, _ in sorted_tps], dtype=np.float64)
    if len(times) < 2:
        return create_growth_plot(), "Need >= 2 timepoints."
    gm = state.get("growth_results", [])
    info = [f"{g.get('model_name','?')}: AIC={g.get('aic',0):.1f}, w={g.get('weight',0):.3f}" for g in gm]
    rate = (np.log(max(volumes[-1], 1) / max(volumes[0], 1)) / max(times[-1] - times[0], 1)
            if volumes[0] > 0 else 0)
    fitted = volumes[0] * np.exp(rate * (times - times[0]))
    scenarios = [s for s in state.get("simulation_results_full", [])
                 if hasattr(s, "time_points") and len(s.time_points) > 0]
    return (create_growth_plot(times=times, volumes=volumes, fitted=fitted,
                               lower=fitted * 0.85, upper=fitted * 1.15, scenarios=scenarios),
            "\n".join(info) or "Growth models computed.")


def demo_narrative(state, _empty_fn):
    state = dict(state) if state else _empty_fn()
    meta = state.get("patient_metadata", {})
    ms, rs = state.get("measurements", []), state.get("recist_responses", [])
    gm, therapies, tps = state.get("growth_results", []), state.get("therapy_events", []), state.get("timepoints", [])
    s = ["## Digital Twin Tumor Assessment Report\n",
         "> **Disclaimer:** AI-generated for research only. NOT a substitute for clinical judgement.\n"]
    if meta:
        s += ["### Patient Profile\n"] + [
            f"- **{k}:** {meta.get(v, 'N/A')}" for k, v in
            [("Scenario", "scenario"), ("Cancer Type", "cancer_type"), ("Stage", "stage"),
             ("Histology", "histology"), ("ECOG PS", "ECOG_PS")]]
        s.append(f"- **Age/Sex:** {meta.get('age', '?')}/{meta.get('sex', '?')}")
    if therapies:
        s.append("\n### Treatment History\n")
        s += [f"- **{t.get('therapy_type','').title()}:** {t.get('drug_name', t.get('dose','N/A'))} "
              f"({t.get('start_date','?')} to {t.get('end_date','ongoing')})" for t in therapies]
    if tps and rs:
        s += ["\n### Longitudinal Assessment\n",
              "| Week | RECIST | Sum (mm) | Change (%) | Status |",
              "| --- | --- | --- | --- | --- |"]
        for i, tp in enumerate(tps):
            if i < len(rs):
                r = rs[i]
                s.append(f"| {tp.get('week','?')} | **{r.get('category','N/A')}** | "
                         f"{r.get('sum_of_diameters',0):.1f} | "
                         f"{r.get('percent_change_from_baseline',0):+.1f}% | "
                         f"{tp.get('therapy_status','')} |")
    if rs:
        s.append(f"\n**RECIST Trajectory:** {' -> '.join(r.get('category','?') for r in rs)}")
    if gm:
        s.append("\n### Growth Model Ensemble\n")
        s += [f"- **{g.get('model_name','?')}:** AIC={g.get('aic',0):.1f}, "
              f"weight={g.get('weight',0):.3f}" for g in gm]
    if ms:
        s.append(f"\n### Measurements\nTotal observations: **{len(ms)}** "
                 f"across **{len(tps)}** timepoints.")
    text = "\n".join(s)
    state["narrative"] = text
    return text, f"Report: {len(ms)} measurements, {len(tps)} timepoints.", state


def run_sim(scenario, shift, mult, resist, state, _empty_fn, _ok, models_tuple):
    """Run growth simulation. models_tuple = (ExponentialGrowth, LogisticGrowth, GompertzGrowth) or None."""
    state = dict(state) if state else _empty_fn()
    ms = state.get("measurements", [])
    if len(ms) < 2:
        return create_growth_plot(), "Insufficient data.", state
    times = np.arange(len(ms), dtype=np.float64)
    vols = np.maximum(np.array([float(m.get("volume_mm3", 1)) for m in ms], dtype=np.float64), 1e-3)
    fitted, info_lines, best_r = None, [], 0.01
    if _ok.get("twin") and models_tuple:
        Exp, Log, Gom = models_tuple
        results = []
        for mdl in [Exp(), Log(), Gom()]:
            try: results.append(mdl.fit(times, vols))
            except Exception: pass
        if results:
            aics = np.array([r.aic for r in results]); fm = np.isfinite(aics)
            if np.any(fm):
                d = aics - np.min(aics[fm]); w = np.where(fm, np.exp(-0.5 * d), 0.0)
                ws = w.sum(); w = w / ws if ws > 0 else np.ones(len(results)) / len(results)
            else:
                w = np.ones(len(results)) / len(results)
            for i, r in enumerate(results):
                info_lines.append(f"{r.model_name}: AIC={r.aic:.1f}, w={w[i]:.3f}")
            bi = int(np.argmax(w)); fitted = results[bi].fitted_values
            best_r = results[bi].parameters.get("r", 0.01)
            state["growth_results"] = [{"model_name": r.model_name, "aic": r.aic,
                                        "weight": float(w[i])} for i, r in enumerate(results)]
    if fitted is None and len(times) >= 2 and vols[0] > 0:
        rate = np.log(vols[-1] / vols[0]) / max(times[-1] - times[0], 1)
        fitted = vols[0] * np.exp(rate * times); best_r = rate
        info_lines.append(f"Fallback exp: rate={rate:.4f}")
    lo = fitted * 0.8 if fitted is not None else None
    hi = fitted * 1.2 if fitted is not None else None
    sc: list[dict[str, Any]] = []
    if fitted is not None:
        tf = np.linspace(0, float(times[-1]) * 1.5, 60); v0 = float(vols[0])
        if scenario == "Natural history (no treatment)": sv = v0 * np.exp(best_r * tf)
        elif scenario == "Earlier treatment": sv = v0 * np.exp(best_r * tf) * np.exp(-mult * np.maximum(tf - shift, 0))
        elif scenario == "Later treatment": sv = v0 * np.exp(best_r * tf) * np.exp(-mult * 0.5 * np.maximum(tf - shift - 4, 0))
        elif scenario == "Treatment resistance":
            sv = v0 * np.exp(best_r * tf); sv[tf < resist] *= np.exp(-mult * tf[tf < resist])
        else: sv = v0 * np.exp(best_r * tf * mult)
        sc.append({"name": scenario, "times": tf, "volumes": sv, "lower": sv * 0.85, "upper": sv * 1.15})
        state.setdefault("simulations", []).append({"scenario_name": scenario})
    return (create_growth_plot(times=times, volumes=vols, fitted=fitted, lower=lo, upper=hi, scenarios=sc),
            "\n".join(info_lines) or "No models.", state)
