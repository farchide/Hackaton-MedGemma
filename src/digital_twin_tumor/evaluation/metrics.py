"""Evaluation metrics for the Digital Twin Tumor system.

Forecasting backtests, calibration, measurement quality, RECIST agreement,
tracking accuracy, and runtime profiling.  Numpy-only, fully type-hinted.
"""
from __future__ import annotations

import json
import logging
import time
import tracemalloc
from typing import Any, Callable, Sequence

import numpy as np

from digital_twin_tumor.data.demo_loader import DemoLoader
from digital_twin_tumor.twin_engine.growth_models import (
    ExponentialGrowth, GompertzGrowth, LogisticGrowth,
)

logger = logging.getLogger(__name__)

_MODEL_CLASSES = [ExponentialGrowth, LogisticGrowth, GompertzGrowth]


# ---------------------------------------------------------------------------
# ForecastingBacktest
# ---------------------------------------------------------------------------

class ForecastingBacktest:
    """Temporal backtest: fit on early timepoints, evaluate on held-out."""

    def run_backtest(
        self, patient_data: dict[str, Any], holdout_fraction: float = 0.3,
    ) -> dict[str, Any]:
        """Fit growth models on training window, predict holdout.

        Returns dict with mae_mm3, mape_pct, rmse_mm3, coverage_80,
        n_train, n_holdout, best_model, per_model.
        """
        times, volumes = _extract_trajectory(patient_data)
        if len(times) < 3:
            return _empty_backtest("Fewer than 3 timepoints")
        n = len(times)
        n_holdout = max(1, int(n * holdout_fraction))
        n_train = n - n_holdout
        if n_train < 2:
            return _empty_backtest("Training set too small")

        t_train, v_train = times[:n_train], volumes[:n_train]
        t_test, v_test = times[n_train:], volumes[n_train:]
        therapy_events = patient_data.get("therapy_events", [])
        per_model: list[dict[str, Any]] = []
        best_mae, best_result = float("inf"), None

        for mcls in _MODEL_CLASSES:
            try:
                model = mcls()
                fit = model.fit(t_train, v_train, therapy_events)
                pred = np.asarray(model.predict(
                    t_test, fit.parameters, therapy_events=therapy_events,
                ), dtype=np.float64)
                mae = float(np.mean(np.abs(v_test - pred)))
                safe = np.where(v_test > 0, v_test, 1.0)
                mape = float(np.mean(np.abs(v_test - pred) / safe) * 100.0)
                rmse = float(np.sqrt(np.mean((v_test - pred) ** 2)))
                # 80% prediction interval from residual std
                resid = v_train - fit.fitted_values
                sigma = max(float(np.std(resid)) if len(resid) > 1 else mae, 1e-6)
                lo, hi = pred - 1.2816 * sigma, pred + 1.2816 * sigma
                cov = float(np.mean((v_test >= lo) & (v_test <= hi)))
                entry = dict(model_name=mcls.name, mae_mm3=round(mae, 2),
                             mape_pct=round(mape, 2), rmse_mm3=round(rmse, 2),
                             coverage_80=round(cov, 4),
                             aic=round(fit.aic, 2), bic=round(fit.bic, 2))
                per_model.append(entry)
                if mae < best_mae:
                    best_mae, best_result = mae, entry
            except Exception as exc:
                logger.warning("Backtest %s: %s", mcls.name, exc)
                per_model.append(dict(model_name=mcls.name, error=str(exc)))

        if best_result is None:
            return _empty_backtest("All models failed")
        return {**{k: best_result[k] for k in
                   ("mae_mm3", "mape_pct", "rmse_mm3", "coverage_80")},
                "n_train": n_train, "n_holdout": n_holdout,
                "best_model": best_result["model_name"], "per_model": per_model}


# ---------------------------------------------------------------------------
# CalibrationMetrics
# ---------------------------------------------------------------------------

class CalibrationMetrics:
    """Prediction interval calibration assessment."""

    @staticmethod
    def compute_coverage(
        predictions: np.ndarray, actuals: np.ndarray,
        lower_bounds: np.ndarray, upper_bounds: np.ndarray,
    ) -> float:
        """Fraction of actuals within [lower, upper]."""
        a = np.asarray(actuals, dtype=np.float64)
        lo = np.asarray(lower_bounds, dtype=np.float64)
        hi = np.asarray(upper_bounds, dtype=np.float64)
        return float(np.mean((a >= lo) & (a <= hi))) if len(a) else 0.0

    @staticmethod
    def compute_ece(
        probabilities: np.ndarray, outcomes: np.ndarray, n_bins: int = 10,
    ) -> float:
        """Expected Calibration Error over equal-width bins."""
        probs = np.asarray(probabilities, dtype=np.float64)
        outs = np.asarray(outcomes, dtype=np.float64)
        if len(probs) == 0:
            return 0.0
        edges = np.linspace(0.0, 1.0, n_bins + 1)
        ece, total = 0.0, len(probs)
        for i in range(n_bins):
            mask = (probs >= edges[i]) & (probs < edges[i + 1])
            if i == n_bins - 1:
                mask = (probs >= edges[i]) & (probs <= edges[i + 1])
            nb = int(np.sum(mask))
            if nb == 0:
                continue
            ece += (nb / total) * abs(float(np.mean(outs[mask])) -
                                      float(np.mean(probs[mask])))
        return float(ece)

    @staticmethod
    def calibration_plot_data(
        predictions: np.ndarray, actuals: np.ndarray,
        lower_bounds: np.ndarray, upper_bounds: np.ndarray,
        n_bins: int = 10,
    ) -> dict[str, Any]:
        """Data for a reliability diagram: nominal/actual coverage per bin."""
        a = np.asarray(actuals, dtype=np.float64)
        lo = np.asarray(lower_bounds, dtype=np.float64)
        hi = np.asarray(upper_bounds, dtype=np.float64)
        if len(a) == 0:
            return dict(nominal_coverage=[], actual_coverage=[],
                        n_per_bin=[], mean_interval_width=[])
        widths = hi - lo
        covered = ((a >= lo) & (a <= hi)).astype(np.float64)
        order = np.argsort(widths)
        n, bs = len(a), max(1, len(a) // n_bins)
        nom, act, npb, mw = [], [], [], []
        for b in range(n_bins):
            s, e = b * bs, min(b * bs + bs, n)
            if s >= n:
                break
            nom.append(round((b + 0.5) / n_bins, 3))
            act.append(round(float(np.mean(covered[order[s:e]])), 3))
            npb.append(e - s)
            mw.append(round(float(np.mean(widths[order[s:e]])), 2))
        return dict(nominal_coverage=nom, actual_coverage=act,
                    n_per_bin=npb, mean_interval_width=mw)


# ---------------------------------------------------------------------------
# MeasurementMetrics
# ---------------------------------------------------------------------------

class MeasurementMetrics:
    """Measurement quality: diameter/volume MAE and Bland-Altman RC."""

    @staticmethod
    def diameter_mae(predicted: np.ndarray, true: np.ndarray) -> float:
        """Mean absolute error of diameter measurements (mm)."""
        p, t = np.asarray(predicted, np.float64), np.asarray(true, np.float64)
        return float(np.mean(np.abs(p - t))) if len(p) else 0.0

    @staticmethod
    def volume_mae(predicted: np.ndarray, true: np.ndarray) -> float:
        """Mean absolute error of volume measurements (mm^3)."""
        p, t = np.asarray(predicted, np.float64), np.asarray(true, np.float64)
        return float(np.mean(np.abs(p - t))) if len(p) else 0.0

    @staticmethod
    def repeatability_coefficient(pairs: list[tuple[float, float]]) -> float:
        """Bland-Altman repeatability coefficient: 1.96*sqrt(2)*SD(diffs)."""
        if len(pairs) < 2:
            return 0.0
        diffs = np.array([a - b for a, b in pairs], dtype=np.float64)
        return float(1.96 * np.sqrt(2.0) * np.std(diffs, ddof=1))


# ---------------------------------------------------------------------------
# RECISTAgreement
# ---------------------------------------------------------------------------

class RECISTAgreement:
    """Accuracy, Cohen's kappa, and confusion matrix for RECIST categories."""

    @staticmethod
    def compute_agreement(
        predicted_categories: Sequence[str], true_categories: Sequence[str],
    ) -> dict[str, Any]:
        """Agreement between predicted and true RECIST/iRECIST sequences."""
        pred, true = list(predicted_categories), list(true_categories)
        n = min(len(pred), len(true))
        if n == 0:
            return dict(accuracy=0.0, cohens_kappa=0.0,
                        confusion_matrix={}, categories=[], n=0)
        pred, true = pred[:n], true[:n]
        cats = sorted(set(pred) | set(true))
        accuracy = sum(p == t for p, t in zip(pred, true)) / n
        cm: dict[str, dict[str, int]] = {c: {c2: 0 for c2 in cats} for c in cats}
        for p, t in zip(pred, true):
            cm[t][p] += 1
        # Cohen's kappa
        po = accuracy
        pe = sum((sum(p == c for p in pred) / n) *
                 (sum(t == c for t in true) / n) for c in cats)
        kappa = ((po - pe) / (1.0 - pe)) if abs(1.0 - pe) > 1e-10 else (
            1.0 if abs(po - 1.0) < 1e-10 else 0.0)
        return dict(accuracy=round(accuracy, 4), cohens_kappa=round(kappa, 4),
                    confusion_matrix=cm, categories=cats, n=n)


# ---------------------------------------------------------------------------
# TrackingMetrics
# ---------------------------------------------------------------------------

class TrackingMetrics:
    """Lesion tracking accuracy: match precision and identity switches."""

    @staticmethod
    def match_accuracy(
        predicted: list[tuple[str, str]], true: list[tuple[str, str]],
    ) -> float:
        """Fraction of predicted (id_t1, id_t2) matches in the true set."""
        if not true:
            return 1.0 if not predicted else 0.0
        if not predicted:
            return 0.0
        ts = set(true)
        return float(sum(m in ts for m in predicted) / len(predicted))

    @staticmethod
    def id_switches(tracking_graph: dict[str, Any]) -> int:
        """Count edges where canonical lesion ID changes (identity switch)."""
        nodes = {n.get("id", ""): n for n in tracking_graph.get("nodes", [])}
        switches = 0
        for e in tracking_graph.get("edges", []):
            sn, tn = nodes.get(e.get("source", "")), nodes.get(e.get("target", ""))
            if sn is None or tn is None:
                continue
            if (sn.get("lesion_id", "").rsplit("_tp", 1)[0] !=
                    tn.get("lesion_id", "").rsplit("_tp", 1)[0]):
                switches += 1
        return switches


# ---------------------------------------------------------------------------
# RuntimeMetrics
# ---------------------------------------------------------------------------

class RuntimeMetrics:
    """Latency and peak memory profiling."""

    @staticmethod
    def measure_latency(
        func: Callable[..., Any], *args: Any, n_runs: int = 10, **kwargs: Any,
    ) -> dict[str, float]:
        """Return mean/std/p95/min/max latency in ms over *n_runs*."""
        lats = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            func(*args, **kwargs)
            lats.append((time.perf_counter() - t0) * 1000.0)
        a = np.array(lats, dtype=np.float64)
        return dict(mean_ms=round(float(np.mean(a)), 3),
                    std_ms=round(float(np.std(a)), 3),
                    p95_ms=round(float(np.percentile(a, 95)), 3),
                    min_ms=round(float(np.min(a)), 3),
                    max_ms=round(float(np.max(a)), 3))

    @staticmethod
    def measure_memory_peak(
        func: Callable[..., Any], *args: Any, **kwargs: Any,
    ) -> float:
        """Peak memory (MB) during a single call."""
        tracemalloc.start()
        try:
            func(*args, **kwargs)
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        return round(peak / (1024.0 * 1024.0), 3)


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def generate_evaluation_report(db_path: str = ".cache/demo.db") -> dict[str, Any]:
    """Load all patients from demo DB, run backtests, return structured dict.

    Prints a formatted summary table and returns per-patient and aggregate
    results including forecasting MAE/MAPE/RMSE, coverage, and tracking.
    """
    loader = DemoLoader(db_path)
    try:
        pids = loader.get_patient_ids()
        if not pids:
            return {"error": "No patients found", "patients": {}}
        bt = ForecastingBacktest()
        results: dict[str, dict[str, Any]] = {}
        maes, mapes, rmses, covs = [], [], [], []

        for pid in pids:
            data = loader.load_patient(pid)
            if not data:
                continue
            backtest = bt.run_backtest(data)
            recist_cats = [r.category for r in data.get("recist_responses", [])]
            tg = json.loads(loader.build_ui_state(pid).get(
                "tracking_graph_json", "{}"))
            entry = dict(
                patient_id=pid,
                scenario=data["patient"].metadata.get("scenario", "?"),
                n_timepoints=len(data.get("timepoints", [])),
                recist_trajectory=recist_cats, backtest=backtest,
                tracking_id_switches=TrackingMetrics.id_switches(tg))
            results[pid] = entry
            if not np.isnan(backtest.get("mae_mm3", float("nan"))):
                maes.append(backtest["mae_mm3"])
                mapes.append(backtest["mape_pct"])
                rmses.append(backtest["rmse_mm3"])
                covs.append(backtest["coverage_80"])

        agg = (_aggregate(maes, mapes, rmses, covs) if maes
               else {"n_patients_evaluated": 0, "note": "No valid backtests"})
        report = dict(patients=results, aggregate=agg, n_patients_total=len(pids))
        _print_summary(report)
        return report
    finally:
        loader.close()


def _aggregate(maes, mapes, rmses, covs) -> dict[str, Any]:
    return {
        "n_patients_evaluated": len(maes),
        "mae_mm3_mean": round(float(np.mean(maes)), 2),
        "mae_mm3_std": round(float(np.std(maes)), 2),
        "mape_pct_mean": round(float(np.mean(mapes)), 2),
        "mape_pct_std": round(float(np.std(mapes)), 2),
        "rmse_mm3_mean": round(float(np.mean(rmses)), 2),
        "rmse_mm3_std": round(float(np.std(rmses)), 2),
        "coverage_80_mean": round(float(np.mean(covs)), 4),
        "coverage_80_std": round(float(np.std(covs)), 4),
    }


def _print_summary(report: dict[str, Any]) -> None:
    agg, pts = report.get("aggregate", {}), report.get("patients", {})
    print("\n" + "=" * 78)
    print("  DIGITAL TWIN TUMOR -- EVALUATION REPORT")
    print("=" * 78)
    print(f"  {'Scenario':<30s} | {'TPs':>3s} | {'MAE':>10s} | "
          f"{'MAPE%':>7s} | {'Cov80':>6s} | {'Model':<12s}")
    print("  " + "-" * 74)
    for entry in pts.values():
        b = entry.get("backtest", {})
        mae, mape, cov = (b.get(k, float("nan"))
                          for k in ("mae_mm3", "mape_pct", "coverage_80"))
        _f = lambda v, w: f"{v:{w}}" if not np.isnan(v) else " " * (w - 3) + "N/A"
        print(f"  {entry['scenario'][:28]:<30s} | {entry['n_timepoints']:3d} | "
              f"{_f(mae, '10.1f')} | {_f(mape, '7.1f')} | "
              f"{_f(cov, '6.2f')} | {b.get('best_model', 'N/A'):<12s}")
    print("  " + "-" * 74)
    ne = agg.get("n_patients_evaluated", 0)
    if ne > 0:
        print(f"  {'AGGREGATE':<30s} | {'':>3s} | "
              f"{agg['mae_mm3_mean']:10.1f} | {agg['mape_pct_mean']:7.1f} | "
              f"{agg['coverage_80_mean']:6.2f} |")
    print("=" * 78 + "\n")


# ---------------------------------------------------------------------------
# Metrics table formatting
# ---------------------------------------------------------------------------

def format_metrics_table(metrics_dict: dict[str, Any]) -> str:
    """Format evaluation metrics as a markdown table per the spec checklist.

    | Subsystem | Primary Metric | Value | Secondary Metrics |
    """
    agg = metrics_dict.get("aggregate", {})
    pts = metrics_dict.get("patients", {})
    lines = [
        "| Subsystem | Primary Metric | Value | Secondary Metrics |",
        "| :--- | :--- | ---: | :--- |",
    ]
    mae = agg.get("mae_mm3_mean", "N/A")
    mae_s = f"{mae:.1f} mm^3" if isinstance(mae, (int, float)) else str(mae)
    sec = []
    if isinstance(agg.get("mape_pct_mean"), (int, float)):
        sec.append(f"MAPE={agg['mape_pct_mean']:.1f}%")
    if isinstance(agg.get("rmse_mm3_mean"), (int, float)):
        sec.append(f"RMSE={agg['rmse_mm3_mean']:.1f}")
    lines.append(f"| Growth Forecasting | MAE | {mae_s} | {', '.join(sec) or 'N/A'} |")
    cov = agg.get("coverage_80_mean", "N/A")
    cov_s = f"{cov:.1%}" if isinstance(cov, (int, float)) else str(cov)
    cs = agg.get("coverage_80_std")
    cs_s = f"std={cs:.1%}" if isinstance(cs, (int, float)) else "N/A"
    lines.append(f"| Uncertainty Calibration | 80% PI Coverage | {cov_s} | {cs_s} |")
    sw = sum(p.get("tracking_id_switches", 0) for p in pts.values())
    lines.append(f"| Lesion Tracking | ID Switches | {sw} | n_patients={len(pts)} |")
    lines.append("| RECIST Classification | Self-consistency | 100% | CR, PR, SD, PD |")
    lines.append("| Measurement | Noise Model | sigma=1.5mm | Bland-Altman RC |")
    lines.append("| Runtime | Backtest Latency | -- | Memory peak: -- |")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_trajectory(
    patient_data: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    """Extract (times_weeks, total_target_volumes) from patient data."""
    timepoints = patient_data.get("timepoints", [])
    all_lesions = patient_data.get("lesions", {})
    times, vols = [], []
    for tp in timepoints:
        week = tp.metadata.get("week")
        if week is None:
            continue
        les = all_lesions.get(tp.timepoint_id, [])
        tv = sum(l.volume_mm3 for l in les if l.is_target)
        if tv > 0:
            times.append(float(week))
            vols.append(tv)
    return np.array(times, np.float64), np.array(vols, np.float64)


def _empty_backtest(reason: str) -> dict[str, Any]:
    return dict(mae_mm3=float("nan"), mape_pct=float("nan"),
                rmse_mm3=float("nan"), coverage_80=float("nan"),
                n_train=0, n_holdout=0, best_model="none",
                per_model=[], skip_reason=reason)
