"""Three-tier uncertainty quantification (ADR-008).

Provides:
1. Measurement uncertainty composition (manual + auto + scan-rescan)
2. Scan-rescan calibration from RIDER-style data
3. Parametric bootstrap for growth model parameters
4. Prediction interval computation
5. Reliability assessment
"""

from __future__ import annotations

import logging
import warnings
from typing import Any

import numpy as np

from digital_twin_tumor.domain.models import UncertaintyEstimate

logger = logging.getLogger(__name__)

_MIN_VOLUME = 1e-6


# ---------------------------------------------------------------------------
# Tier 1: Measurement uncertainty
# ---------------------------------------------------------------------------


def compute_measurement_uncertainty(
    sigma_manual: float = 1.5,
    sigma_auto: float = 0.0,
    sigma_scan: float = 0.0,
) -> UncertaintyEstimate:
    """Compose measurement uncertainty from independent sources.

    Total uncertainty is the root-sum-of-squares of the individual
    standard deviations.  Reliability is assigned based on which data
    sources are available.

    Parameters
    ----------
    sigma_manual:
        Standard deviation of manual measurement (mm).  Default 1.5 mm
        represents typical inter-reader variability.
    sigma_auto:
        Standard deviation from automated segmentation (mm).  Zero when
        no automated measurement is available.
    sigma_scan:
        Standard deviation from scan-rescan variability (mm).  Zero when
        no scan-rescan data is available.

    Returns
    -------
    UncertaintyEstimate
        Combined uncertainty with reliability classification.
    """
    total = float(np.sqrt(sigma_manual ** 2 + sigma_auto ** 2 + sigma_scan ** 2))

    # Determine reliability based on data availability
    sources_available = sum([
        sigma_manual > 0.0,
        sigma_auto > 0.0,
        sigma_scan > 0.0,
    ])

    if sources_available >= 3:
        reliability = "HIGH"
    elif sources_available >= 2:
        reliability = "MEDIUM"
    else:
        reliability = "LOW"

    return UncertaintyEstimate(
        sigma_manual=sigma_manual,
        sigma_auto=sigma_auto,
        sigma_scan=sigma_scan,
        total_sigma=total,
        reliability=reliability,
    )


# ---------------------------------------------------------------------------
# Tier 2: Scan-rescan calibration
# ---------------------------------------------------------------------------


def calibrate_scan_rescan(
    volumes_scan1: np.ndarray,
    volumes_scan2: np.ndarray,
) -> tuple[float, float]:
    """Calibrate heteroscedastic scan-rescan uncertainty.

    Fits a linear model ``sigma(d) = a + b * d`` to the absolute
    differences between paired scan-rescan volume measurements, where
    ``d`` is the mean diameter (cube root of mean volume).

    This follows the RIDER (Reference Image Database to Evaluate
    Therapy Response) methodology for quantifying measurement
    variability.

    Parameters
    ----------
    volumes_scan1:
        1-D array of volumes from the first scan (mm^3).
    volumes_scan2:
        1-D array of volumes from the second scan (mm^3).

    Returns
    -------
    tuple[float, float]
        Intercept *a* and slope *b* of the heteroscedastic model.

    Raises
    ------
    ValueError
        If arrays have different lengths or fewer than 2 paired
        measurements.
    """
    v1 = np.asarray(volumes_scan1, dtype=np.float64)
    v2 = np.asarray(volumes_scan2, dtype=np.float64)

    if v1.shape != v2.shape:
        raise ValueError(
            f"Scan arrays must have the same shape: {v1.shape} vs {v2.shape}"
        )
    if len(v1) < 2:
        raise ValueError("Need at least 2 paired measurements for calibration.")

    # Mean diameter (cube root of mean volume) as the size metric
    mean_vol = (v1 + v2) / 2.0
    diameters = np.cbrt(np.maximum(mean_vol, _MIN_VOLUME))

    # Absolute differences as the observed variability
    abs_diff = np.abs(v1 - v2)

    # Fit linear model: abs_diff = a + b * diameter
    # Using least-squares: [1, d] @ [a, b]^T = abs_diff
    design = np.column_stack([np.ones_like(diameters), diameters])
    coeffs, _, _, _ = np.linalg.lstsq(design, abs_diff, rcond=None)

    a = float(coeffs[0])
    b = float(coeffs[1])

    return a, b


# ---------------------------------------------------------------------------
# Tier 3: Bootstrap uncertainty for growth parameters
# ---------------------------------------------------------------------------


def bootstrap_growth_parameters(
    times: np.ndarray,
    volumes: np.ndarray,
    model_class: Any,
    n_bootstrap: int = 1000,
    sigma_obs: float = 0.0,
) -> dict[str, Any]:
    """Parametric bootstrap for growth model parameter uncertainty.

    Generates *n_bootstrap* resampled datasets by adding Gaussian noise
    to the observed volumes (or via residual resampling when
    ``sigma_obs == 0``) and refitting the model to each.

    Parameters
    ----------
    times:
        1-D array of observed time values.
    volumes:
        1-D array of observed tumor volumes.
    model_class:
        Growth model class (e.g. ``ExponentialGrowth``).  Must expose
        a ``fit(times, volumes)`` method returning ``GrowthModelResult``.
    n_bootstrap:
        Number of bootstrap iterations.
    sigma_obs:
        Known observation noise standard deviation.  When zero, residual
        resampling is used instead.

    Returns
    -------
    dict
        Keys:
        - ``parameter_samples``: ndarray of shape (n_bootstrap, n_params)
        - ``prediction_intervals``: dict mapping alpha to (lower, upper)
          arrays computed at the original time points.
        - ``parameter_names``: list of parameter name strings.
        - ``cv``: coefficient of variation across parameters (mean).
    """
    times = np.asarray(times, dtype=np.float64)
    volumes = np.asarray(volumes, dtype=np.float64)

    # Fit the original model to get baseline parameters and residuals
    model = model_class()
    base_result = model.fit(times, volumes)
    base_params = base_result.parameters
    param_names = sorted(base_params.keys())

    residuals = base_result.residuals
    if len(residuals) == 0:
        residuals = np.zeros_like(volumes)

    # Determine noise for resampling
    if sigma_obs > 0:
        noise_scale = sigma_obs
    else:
        noise_scale = float(np.std(residuals)) if len(residuals) > 1 else 1.0
        noise_scale = max(noise_scale, _MIN_VOLUME)

    rng = np.random.default_rng(seed=42)
    param_samples: list[list[float]] = []
    prediction_samples: list[np.ndarray] = []

    for _ in range(n_bootstrap):
        # Generate noisy volumes
        noise = rng.normal(0.0, noise_scale, size=len(volumes))
        noisy_volumes = np.maximum(volumes + noise, _MIN_VOLUME)

        try:
            boot_model = model_class()
            boot_result = boot_model.fit(times, noisy_volumes)
            boot_params = boot_result.parameters

            # Store parameter vector in consistent order
            param_vec = [boot_params.get(name, 0.0) for name in param_names]
            param_samples.append(param_vec)

            # Store prediction at original times
            prediction_samples.append(boot_result.fitted_values)
        except Exception:
            continue

    if not param_samples:
        # All bootstraps failed: return degenerate result
        n_params = len(param_names)
        return {
            "parameter_samples": np.array([[base_params.get(n, 0.0) for n in param_names]]),
            "prediction_intervals": {
                0.1: (base_result.fitted_values, base_result.fitted_values),
                0.9: (base_result.fitted_values, base_result.fitted_values),
            },
            "parameter_names": param_names,
            "cv": 0.0,
        }

    param_array = np.array(param_samples)  # (n_success, n_params)

    # Build prediction matrix
    pred_array = np.array(prediction_samples)  # (n_success, n_times)

    # Compute prediction intervals at standard alpha levels
    intervals: dict[float, tuple[np.ndarray, np.ndarray]] = {}
    for alpha in [0.1, 0.9]:
        lower, upper = compute_prediction_intervals(pred_array, alpha=alpha)
        intervals[alpha] = (lower, upper)

    # Compute mean CV across parameters
    means = np.mean(param_array, axis=0)
    stds = np.std(param_array, axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cvs = np.where(np.abs(means) > _MIN_VOLUME, stds / np.abs(means), 0.0)
    mean_cv = float(np.mean(cvs))

    return {
        "parameter_samples": param_array,
        "prediction_intervals": intervals,
        "parameter_names": param_names,
        "cv": mean_cv,
    }


# ---------------------------------------------------------------------------
# Prediction intervals from bootstrap samples
# ---------------------------------------------------------------------------


def compute_prediction_intervals(
    predictions: np.ndarray,
    alpha: float = 0.2,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute lower and upper prediction bounds from bootstrap samples.

    Parameters
    ----------
    predictions:
        2-D array of shape (n_bootstrap, n_times) containing predicted
        volumes from each bootstrap iteration.
    alpha:
        Significance level.  The returned interval covers the central
        ``1 - alpha`` fraction.  For example, ``alpha=0.2`` gives the
        10th and 90th percentiles.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Lower and upper bound arrays, each of shape (n_times,).
    """
    predictions = np.asarray(predictions, dtype=np.float64)

    if predictions.ndim == 1:
        return predictions.copy(), predictions.copy()
    if predictions.shape[0] == 0:
        n_times = predictions.shape[1] if predictions.ndim > 1 else 0
        return np.zeros(n_times), np.zeros(n_times)

    lower_pct = 100.0 * (alpha / 2.0)
    upper_pct = 100.0 * (1.0 - alpha / 2.0)

    lower = np.percentile(predictions, lower_pct, axis=0)
    upper = np.percentile(predictions, upper_pct, axis=0)

    return lower, upper


# ---------------------------------------------------------------------------
# Reliability assessment
# ---------------------------------------------------------------------------


def assess_reliability(
    n_timepoints: int,
    bootstrap_cv: float,
) -> str:
    """Classify parameter estimation reliability.

    Parameters
    ----------
    n_timepoints:
        Number of longitudinal time points used for fitting.
    bootstrap_cv:
        Mean coefficient of variation of growth parameters from
        bootstrap resampling.

    Returns
    -------
    str
        ``"HIGH"`` if n >= 5 and cv < 0.3,
        ``"LOW"`` if n < 3 or cv > 0.5,
        ``"MEDIUM"`` otherwise.
    """
    if n_timepoints < 3 or bootstrap_cv > 0.5:
        return "LOW"
    if n_timepoints >= 5 and bootstrap_cv < 0.3:
        return "HIGH"
    return "MEDIUM"


def compose_total_uncertainty(
    measurement: UncertaintyEstimate,
    bootstrap_cv: float,
    ensemble_std: float,
    n_timepoints: int,
) -> dict[str, Any]:
    """Compose total three-tier uncertainty from all sources.

    Tier 1: Measurement uncertainty (manual + auto + scan-rescan)
    Tier 2: Model parameter uncertainty (bootstrap CV)
    Tier 3: Model-form uncertainty (ensemble disagreement)

    Parameters
    ----------
    measurement:
        Tier 1 measurement uncertainty estimate.
    bootstrap_cv:
        Coefficient of variation from parametric bootstrap (Tier 2).
    ensemble_std:
        Mean standard deviation across ensemble model predictions (Tier 3).
    n_timepoints:
        Number of longitudinal time points.

    Returns
    -------
    dict
        Keys: tier1_sigma, tier2_cv, tier3_std, combined_reliability,
        total_relative_uncertainty, interpretation.
    """
    tier1 = measurement.total_sigma
    tier2 = bootstrap_cv
    tier3 = ensemble_std

    # Combined relative uncertainty (root-sum-of-squares of relative terms)
    total_rel = float(np.sqrt(tier2 ** 2 + (tier3 / max(tier1, 0.01)) ** 2))

    # Reliability from all tiers
    reliability = assess_reliability(n_timepoints, bootstrap_cv)

    # Interpretation
    if total_rel < 0.15:
        interpretation = "Low uncertainty: predictions are well-constrained by data."
    elif total_rel < 0.40:
        interpretation = "Moderate uncertainty: predictions provide directional guidance but exact values should be interpreted cautiously."
    else:
        interpretation = "High uncertainty: predictions are exploratory and should not be used for quantitative comparison."

    return {
        "tier1_sigma": tier1,
        "tier2_cv": tier2,
        "tier3_std": tier3,
        "combined_reliability": reliability,
        "total_relative_uncertainty": total_rel,
        "n_timepoints": n_timepoints,
        "interpretation": interpretation,
    }
