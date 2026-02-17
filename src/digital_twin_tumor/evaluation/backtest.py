"""Advanced backtesting utilities for growth model evaluation.

Implements leave-one-out temporal backtesting, multi-model comparison
with AIC/BIC selection, and bootstrap-based uncertainty band estimation.
All functions operate on numpy arrays with full type annotations.
"""
from __future__ import annotations

import logging
import warnings
from typing import Any

import numpy as np

from digital_twin_tumor.domain.models import GrowthModelResult, TherapyEvent
from digital_twin_tumor.twin_engine.growth_models import (
    BaseGrowthModel,
    ExponentialGrowth,
    GompertzGrowth,
    LogisticGrowth,
)

logger = logging.getLogger(__name__)

_MODEL_CLASSES: list[type[BaseGrowthModel]] = [
    ExponentialGrowth,
    LogisticGrowth,
    GompertzGrowth,
]

_MIN_VOLUME = 1e-6


# ---------------------------------------------------------------------------
# Leave-one-out backtest
# ---------------------------------------------------------------------------


def run_leave_one_out_backtest(
    patient_data: dict[str, Any],
    therapy_events: list[TherapyEvent] | None = None,
) -> dict[str, Any]:
    """Leave-one-out temporal backtest across all timepoints.

    For each timepoint after the 2nd, the model is fitted on all prior
    observations and the current timepoint is predicted.  This provides
    an unbiased estimate of one-step-ahead forecasting performance.

    Parameters
    ----------
    patient_data:
        Dict with keys "times" (ndarray of weeks) and "volumes" (ndarray
        of total target volumes in mm^3), or a full patient data dict
        from DemoLoader (will auto-extract).
    therapy_events:
        Optional therapy events for treatment-aware fitting.

    Returns
    -------
    dict
        Keys: errors (list of per-step dicts), mae_mm3, rmse_mm3,
        mape_pct, coverage_80, best_model_name, per_model_summary.
    """
    times, volumes = _extract_arrays(patient_data)
    therapy = therapy_events or []

    if len(times) < 3:
        return {
            "errors": [],
            "mae_mm3": float("nan"),
            "rmse_mm3": float("nan"),
            "mape_pct": float("nan"),
            "coverage_80": float("nan"),
            "best_model_name": "none",
            "per_model_summary": [],
            "skip_reason": "Fewer than 3 timepoints",
        }

    # Per-model leave-one-out results
    model_results: dict[str, list[dict[str, Any]]] = {
        cls.name: [] for cls in _MODEL_CLASSES
    }

    for step in range(2, len(times)):
        t_train = times[:step]
        v_train = volumes[:step]
        t_pred = np.array([times[step]], dtype=np.float64)
        v_actual = volumes[step]

        for model_cls in _MODEL_CLASSES:
            try:
                model = model_cls()
                fit = model.fit(t_train, v_train, therapy)
                pred = model.predict(t_pred, fit.parameters, therapy_events=therapy)
                v_pred = float(pred[0])

                # Residual-based interval
                residuals = v_train - fit.fitted_values
                sigma = float(np.std(residuals)) if len(residuals) > 1 else 1.0
                sigma = max(sigma, _MIN_VOLUME)
                z_80 = 1.2816
                lower = v_pred - z_80 * sigma
                upper = v_pred + z_80 * sigma

                error = float(np.abs(v_actual - v_pred))
                rel_error = error / max(v_actual, _MIN_VOLUME)
                covered = bool(lower <= v_actual <= upper)

                model_results[model_cls.name].append({
                    "step": step,
                    "time": float(times[step]),
                    "actual": float(v_actual),
                    "predicted": v_pred,
                    "error": round(error, 2),
                    "relative_error": round(rel_error, 4),
                    "lower_80": round(lower, 2),
                    "upper_80": round(upper, 2),
                    "covered": covered,
                })
            except Exception as exc:
                logger.debug(
                    "LOO step %d failed for %s: %s",
                    step, model_cls.name, exc,
                )

    # Select best model by mean absolute error
    best_name = "none"
    best_mae = float("inf")
    per_model_summary: list[dict[str, Any]] = []

    for name, results in model_results.items():
        if not results:
            per_model_summary.append({"model": name, "n_steps": 0, "error": "all failed"})
            continue

        errors = np.array([r["error"] for r in results], dtype=np.float64)
        rel_errors = np.array([r["relative_error"] for r in results], dtype=np.float64)
        coverages = np.array([r["covered"] for r in results], dtype=np.float64)

        mae = float(np.mean(errors))
        rmse = float(np.sqrt(np.mean(errors ** 2)))
        mape = float(np.mean(rel_errors) * 100.0)
        cov = float(np.mean(coverages))

        summary = {
            "model": name,
            "n_steps": len(results),
            "mae_mm3": round(mae, 2),
            "rmse_mm3": round(rmse, 2),
            "mape_pct": round(mape, 2),
            "coverage_80": round(cov, 4),
        }
        per_model_summary.append(summary)

        if mae < best_mae:
            best_mae = mae
            best_name = name

    # Aggregate from the best model
    best_steps = model_results.get(best_name, [])
    if best_steps:
        errors_arr = np.array([r["error"] for r in best_steps], dtype=np.float64)
        rel_arr = np.array([r["relative_error"] for r in best_steps], dtype=np.float64)
        cov_arr = np.array([r["covered"] for r in best_steps], dtype=np.float64)

        return {
            "errors": best_steps,
            "mae_mm3": round(float(np.mean(errors_arr)), 2),
            "rmse_mm3": round(float(np.sqrt(np.mean(errors_arr ** 2))), 2),
            "mape_pct": round(float(np.mean(rel_arr) * 100.0), 2),
            "coverage_80": round(float(np.mean(cov_arr)), 4),
            "best_model_name": best_name,
            "per_model_summary": per_model_summary,
        }

    return {
        "errors": [],
        "mae_mm3": float("nan"),
        "rmse_mm3": float("nan"),
        "mape_pct": float("nan"),
        "coverage_80": float("nan"),
        "best_model_name": "none",
        "per_model_summary": per_model_summary,
    }


# ---------------------------------------------------------------------------
# Model comparison with AIC / BIC
# ---------------------------------------------------------------------------


def run_model_comparison(
    patient_data: dict[str, Any],
    therapy_events: list[TherapyEvent] | None = None,
) -> dict[str, Any]:
    """Fit all three growth models and compare via AIC and BIC.

    Computes delta-AIC, Akaike weights, and selects the best model.

    Parameters
    ----------
    patient_data:
        Dict with "times" and "volumes" arrays, or full patient data dict.
    therapy_events:
        Optional therapy events for treatment-aware fitting.

    Returns
    -------
    dict
        Keys: models (list of per-model dicts with name, aic, bic,
        delta_aic, akaike_weight, parameters), selected_model,
        evidence_ratio.
    """
    times, volumes = _extract_arrays(patient_data)
    therapy = therapy_events or []

    if len(times) < 2:
        return {
            "models": [],
            "selected_model": "none",
            "evidence_ratio": float("nan"),
            "skip_reason": "Fewer than 2 timepoints",
        }

    results: list[dict[str, Any]] = []

    for model_cls in _MODEL_CLASSES:
        try:
            model = model_cls()
            fit = model.fit(times, volumes, therapy)
            results.append({
                "name": model_cls.name,
                "aic": round(fit.aic, 2),
                "bic": round(fit.bic, 2),
                "parameters": dict(fit.parameters),
                "n_params": len(fit.parameters),
                "rss": round(float(np.sum(fit.residuals ** 2)), 2)
                if len(fit.residuals) > 0 else 0.0,
            })
        except Exception as exc:
            logger.warning("Model comparison failed for %s: %s", model_cls.name, exc)
            results.append({
                "name": model_cls.name,
                "aic": float("inf"),
                "bic": float("inf"),
                "parameters": {},
                "error": str(exc),
            })

    # Compute delta-AIC and Akaike weights
    valid_aics = [r["aic"] for r in results if np.isfinite(r["aic"])]
    if valid_aics:
        min_aic = min(valid_aics)
        for r in results:
            delta = r["aic"] - min_aic if np.isfinite(r["aic"]) else float("inf")
            r["delta_aic"] = round(delta, 2)

        # Akaike weights
        raw_weights = []
        for r in results:
            if np.isfinite(r.get("delta_aic", float("inf"))):
                raw_weights.append(np.exp(-0.5 * r["delta_aic"]))
            else:
                raw_weights.append(0.0)
        total_w = sum(raw_weights)
        for r, w in zip(results, raw_weights):
            r["akaike_weight"] = round(w / total_w, 4) if total_w > 0 else 0.0

        # Select best model
        best = min(results, key=lambda r: r.get("aic", float("inf")))
        selected = best["name"]

        # Evidence ratio: weight of best / weight of second-best
        sorted_weights = sorted(
            [r["akaike_weight"] for r in results], reverse=True,
        )
        evidence_ratio = (
            sorted_weights[0] / max(sorted_weights[1], 1e-10)
            if len(sorted_weights) > 1 else float("inf")
        )
    else:
        selected = "none"
        evidence_ratio = float("nan")
        for r in results:
            r["delta_aic"] = float("inf")
            r["akaike_weight"] = 0.0

    return {
        "models": results,
        "selected_model": selected,
        "evidence_ratio": round(evidence_ratio, 2) if np.isfinite(evidence_ratio) else float("inf"),
    }


# ---------------------------------------------------------------------------
# Bootstrap uncertainty bands
# ---------------------------------------------------------------------------


def compute_uncertainty_bands(
    model_results: dict[str, Any],
    times: np.ndarray | None = None,
    volumes: np.ndarray | None = None,
    confidence: float = 0.8,
    n_bootstrap: int = 200,
    seed: int = 42,
) -> dict[str, Any]:
    """Compute bootstrap-based prediction uncertainty bands.

    Resamples residuals and refits the best model to generate prediction
    interval estimates at the specified confidence level.

    Parameters
    ----------
    model_results:
        Output from ``run_model_comparison`` with 'selected_model' and
        'models' keys, or a dict with 'times' and 'volumes' arrays.
    times:
        Observation time array.  If None, extracted from model_results.
    volumes:
        Observation volume array.  If None, extracted from model_results.
    confidence:
        Target coverage probability (e.g. 0.8 for 80% PI).
    n_bootstrap:
        Number of bootstrap iterations.
    seed:
        Random seed for reproducibility.

    Returns
    -------
    dict
        Keys: lower (ndarray), upper (ndarray), median (ndarray),
        confidence, n_successful_boots, prediction_times.
    """
    # Determine which model to use
    selected_name = model_results.get("selected_model", "exponential")
    models_list = model_results.get("models", [])

    # Find model parameters
    selected_params: dict[str, float] = {}
    for m in models_list:
        if m.get("name") == selected_name:
            selected_params = m.get("parameters", {})
            break

    # Get or infer times and volumes
    if times is None or volumes is None:
        # Try to extract from model_results
        times_raw = model_results.get("times")
        volumes_raw = model_results.get("volumes")
        if times_raw is not None and volumes_raw is not None:
            times = np.asarray(times_raw, dtype=np.float64)
            volumes = np.asarray(volumes_raw, dtype=np.float64)
        else:
            return {
                "lower": np.array([]),
                "upper": np.array([]),
                "median": np.array([]),
                "confidence": confidence,
                "n_successful_boots": 0,
                "prediction_times": np.array([]),
                "error": "No times/volumes provided or found in model_results",
            }

    times = np.asarray(times, dtype=np.float64)
    volumes = np.asarray(volumes, dtype=np.float64)

    if len(times) < 2:
        return {
            "lower": volumes.copy(),
            "upper": volumes.copy(),
            "median": volumes.copy(),
            "confidence": confidence,
            "n_successful_boots": 0,
            "prediction_times": times.copy(),
        }

    # Instantiate the selected model
    model_map: dict[str, type[BaseGrowthModel]] = {
        cls.name: cls for cls in _MODEL_CLASSES
    }
    model_cls = model_map.get(selected_name, ExponentialGrowth)

    # Fit base model
    model = model_cls()
    base_fit = model.fit(times, volumes)
    residuals = base_fit.residuals
    if len(residuals) == 0:
        residuals = np.zeros_like(volumes)

    sigma = float(np.std(residuals)) if len(residuals) > 1 else 1.0
    sigma = max(sigma, _MIN_VOLUME)

    rng = np.random.default_rng(seed)
    boot_predictions: list[np.ndarray] = []

    for _ in range(n_bootstrap):
        noise = rng.normal(0.0, sigma, size=len(volumes))
        noisy_vols = np.maximum(volumes + noise, _MIN_VOLUME)

        try:
            boot_model = model_cls()
            boot_fit = boot_model.fit(times, noisy_vols)
            boot_pred = boot_model.predict(
                times, boot_fit.parameters, therapy_events=None,
            )
            boot_predictions.append(np.asarray(boot_pred, dtype=np.float64))
        except Exception:
            continue

    if not boot_predictions:
        fitted = base_fit.fitted_values
        return {
            "lower": fitted.copy(),
            "upper": fitted.copy(),
            "median": fitted.copy(),
            "confidence": confidence,
            "n_successful_boots": 0,
            "prediction_times": times.copy(),
        }

    pred_matrix = np.array(boot_predictions)  # (n_boots, n_times)
    alpha = 1.0 - confidence
    lower_pct = 100.0 * (alpha / 2.0)
    upper_pct = 100.0 * (1.0 - alpha / 2.0)

    lower = np.percentile(pred_matrix, lower_pct, axis=0)
    upper = np.percentile(pred_matrix, upper_pct, axis=0)
    median = np.percentile(pred_matrix, 50.0, axis=0)

    return {
        "lower": lower,
        "upper": upper,
        "median": median,
        "confidence": confidence,
        "n_successful_boots": len(boot_predictions),
        "prediction_times": times.copy(),
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _extract_arrays(
    patient_data: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    """Extract (times, volumes) arrays from various input formats.

    Accepts either a dict with explicit "times"/"volumes" keys, or a
    full patient data dict from DemoLoader.load_patient.
    """
    # Direct arrays
    if "times" in patient_data and "volumes" in patient_data:
        return (
            np.asarray(patient_data["times"], dtype=np.float64),
            np.asarray(patient_data["volumes"], dtype=np.float64),
        )

    # Full patient data with timepoints and lesions
    timepoints = patient_data.get("timepoints", [])
    all_lesions = patient_data.get("lesions", {})

    times: list[float] = []
    volumes: list[float] = []

    for tp in timepoints:
        tp_id = tp.timepoint_id
        week = tp.metadata.get("week")
        if week is None:
            continue
        lesions = all_lesions.get(tp_id, [])
        total_vol = sum(les.volume_mm3 for les in lesions if les.is_target)
        if total_vol > 0:
            times.append(float(week))
            volumes.append(total_vol)

    return np.array(times, dtype=np.float64), np.array(volumes, dtype=np.float64)
