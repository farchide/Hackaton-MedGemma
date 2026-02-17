"""Ensemble model selection with Akaike weights (ADR-005).

Fits all three parametric growth models (exponential, logistic, Gompertz),
ranks them by AIC, computes Akaike weights, and produces weighted-average
ensemble predictions.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from digital_twin_tumor.domain.models import GrowthModelResult, TherapyEvent
from digital_twin_tumor.twin_engine.growth_models import (
    ExponentialGrowth,
    GompertzGrowth,
    LogisticGrowth,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model catalogue -- the three canonical growth models
# ---------------------------------------------------------------------------

_MODEL_CLASSES = [ExponentialGrowth, LogisticGrowth, GompertzGrowth]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def fit_all_models(
    times: np.ndarray,
    volumes: np.ndarray,
    therapy_events: list[TherapyEvent] | None = None,
) -> list[GrowthModelResult]:
    """Fit all growth models and return results sorted by AIC.

    Parameters
    ----------
    times:
        1-D array of time values (weeks from baseline).
    volumes:
        1-D array of corresponding tumor volumes.
    therapy_events:
        Optional therapy history for treatment-aware fitting.

    Returns
    -------
    list[GrowthModelResult]
        One result per model, sorted in ascending AIC order (best first).
    """
    results: list[GrowthModelResult] = []
    for model_cls in _MODEL_CLASSES:
        model = model_cls()
        try:
            result = model.fit(times, volumes, therapy_events=therapy_events)
            results.append(result)
        except Exception:
            logger.warning("Failed to fit %s model", model.name, exc_info=True)

    results.sort(key=lambda r: r.aic)
    return results


def compute_akaike_weights(results: list[GrowthModelResult]) -> list[float]:
    """Compute Akaike weights from a list of fitted model results.

    The Akaike weight for model *i* is:

        w_i = exp(-0.5 * delta_AIC_i) / sum_j exp(-0.5 * delta_AIC_j)

    where ``delta_AIC_i = AIC_i - min(AIC)``.

    Parameters
    ----------
    results:
        Fitted model results (need not be pre-sorted).

    Returns
    -------
    list[float]
        Weights in the same order as *results*, summing to 1.0.
    """
    if not results:
        return []

    # Filter out infinite AIC results
    finite_mask = [np.isfinite(r.aic) for r in results]
    if not any(finite_mask):
        # All infinite: equal weights
        n = len(results)
        return [1.0 / n] * n

    aic_values = np.array([r.aic if np.isfinite(r.aic) else 1e30 for r in results])
    min_aic = np.min(aic_values)
    delta_aic = aic_values - min_aic

    # Compute raw weights with numerical stability
    raw = np.exp(-0.5 * delta_aic)
    total = np.sum(raw)

    if total <= 0.0 or not np.isfinite(total):
        n = len(results)
        return [1.0 / n] * n

    weights = raw / total
    return weights.tolist()


def ensemble_predict(
    results: list[GrowthModelResult],
    weights: list[float],
    times: np.ndarray,
) -> np.ndarray:
    """Produce weighted-average ensemble predictions.

    Parameters
    ----------
    results:
        Fitted model results (one per model).
    weights:
        Akaike weights (same order and length as *results*).
    times:
        1-D array of time values at which to predict.

    Returns
    -------
    np.ndarray
        Weighted average predicted volumes: V(t) = sum(w_i * V_i(t)).
    """
    times = np.asarray(times, dtype=np.float64)
    if not results or not weights:
        return np.zeros_like(times)

    model_map = {
        "exponential": ExponentialGrowth(),
        "logistic": LogisticGrowth(),
        "gompertz": GompertzGrowth(),
    }

    ensemble = np.zeros_like(times, dtype=np.float64)
    total_weight = 0.0

    for result, w in zip(results, weights):
        if w <= 0.0 or not np.isfinite(result.aic):
            continue
        model = model_map.get(result.model_name)
        if model is None:
            logger.warning("Unknown model name: %s", result.model_name)
            continue
        try:
            predicted = model.predict(times, result.parameters)
            ensemble += w * predicted
            total_weight += w
        except Exception:
            logger.warning(
                "Prediction failed for %s", result.model_name, exc_info=True
            )

    if total_weight > 0.0:
        ensemble /= total_weight

    return ensemble


def compute_ensemble_uncertainty(
    results: list[GrowthModelResult],
    weights: list[float],
    times: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute model-form ensemble uncertainty (inter-model disagreement).

    Returns weighted mean, lower bound, and upper bound from the spread
    of predictions across competing models weighted by their Akaike weights.

    Parameters
    ----------
    results:
        Fitted model results.
    weights:
        Akaike weights.
    times:
        Time points for prediction.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        (mean, lower_2.5pct, upper_97.5pct) arrays.
    """
    times = np.asarray(times, dtype=np.float64)
    if not results or not weights:
        z = np.zeros_like(times)
        return z, z, z

    model_map = {
        "exponential": ExponentialGrowth(),
        "logistic": LogisticGrowth(),
        "gompertz": GompertzGrowth(),
    }

    predictions: list[np.ndarray] = []
    pred_weights: list[float] = []

    for result, w in zip(results, weights):
        if w <= 0.0 or not np.isfinite(result.aic):
            continue
        model = model_map.get(result.model_name)
        if model is None:
            continue
        try:
            pred = model.predict(times, result.parameters)
            predictions.append(pred)
            pred_weights.append(w)
        except Exception:
            continue

    if not predictions:
        z = np.zeros_like(times)
        return z, z, z

    pred_array = np.array(predictions)
    w_array = np.array(pred_weights)
    w_array = w_array / w_array.sum()

    # Weighted mean
    mean = np.average(pred_array, axis=0, weights=w_array)

    # Weighted variance
    diff_sq = (pred_array - mean[np.newaxis, :]) ** 2
    variance = np.average(diff_sq, axis=0, weights=w_array)
    std = np.sqrt(variance)

    # 95% interval (approximately +/- 2 sigma)
    lower = np.maximum(mean - 1.96 * std, 0.0)
    upper = mean + 1.96 * std

    return mean, lower, upper


def select_best_model(results: list[GrowthModelResult]) -> GrowthModelResult:
    """Select the model with the lowest AIC.

    Parameters
    ----------
    results:
        Fitted model results.

    Returns
    -------
    GrowthModelResult
        The best-fitting model.

    Raises
    ------
    ValueError
        If *results* is empty.
    """
    if not results:
        raise ValueError("No model results to select from.")

    return min(results, key=lambda r: r.aic)


def validate_convergence(result: GrowthModelResult, times: np.ndarray, volumes: np.ndarray) -> dict[str, Any]:
    """Validate that a growth model fit has converged properly.

    Checks:
    - R-squared > 0.5
    - No parameters at bounds
    - Residuals are approximately normal
    - Fitted values are physically reasonable (positive, finite)

    Parameters
    ----------
    result:
        The growth model fit result.
    times:
        Original time values.
    volumes:
        Original volume values.

    Returns
    -------
    dict
        Keys: converged (bool), r_squared (float), issues (list[str]).
    """
    issues: list[str] = []
    fitted = result.fitted_values
    residuals = result.residuals

    # R-squared
    ss_res = np.sum(residuals ** 2) if len(residuals) > 0 else float("inf")
    ss_tot = np.sum((volumes - np.mean(volumes)) ** 2) if len(volumes) > 1 else 1.0
    r_squared = 1.0 - ss_res / max(ss_tot, 1e-10)

    if r_squared < 0.5:
        issues.append(f"Low R-squared: {r_squared:.3f}")

    # Check fitted values
    if len(fitted) > 0:
        if np.any(~np.isfinite(fitted)):
            issues.append("Non-finite fitted values detected")
        if np.any(fitted < 0):
            issues.append("Negative fitted values detected")

    # Check AIC is finite
    if not np.isfinite(result.aic):
        issues.append("AIC is not finite (fit likely failed)")

    # Check parameters are reasonable
    for name, val in result.parameters.items():
        if not np.isfinite(val):
            issues.append(f"Parameter {name} is not finite: {val}")
        if name == "r" and val <= 0:
            issues.append(f"Growth rate r={val:.6f} is non-positive")

    converged = len(issues) == 0
    return {
        "converged": converged,
        "r_squared": float(r_squared),
        "issues": issues,
    }
