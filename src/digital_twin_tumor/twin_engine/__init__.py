"""Digital Twin Tumor Engine -- growth modelling, simulation, and uncertainty.

This package provides the core computational engine for the Digital Twin
Tumor Response Assessment system, implementing:

- **Growth models**: Exponential, logistic, and Gompertz parametric models
  with optional treatment effects (ADR-005).
- **Model selection**: AIC-based ranking, Akaike weights, and ensemble
  prediction (ADR-005).
- **Simulation**: Counterfactual what-if scenarios including natural history,
  shifted treatment timing, regimen switches, treatment holidays, and
  acquired resistance (ADR-005).
- **Uncertainty**: Three-tier uncertainty quantification covering measurement
  noise, scan-rescan calibration, and parametric bootstrap (ADR-008).
"""

from __future__ import annotations

from digital_twin_tumor.twin_engine.growth_models import (
    BaseGrowthModel,
    ExponentialGrowth,
    GompertzGrowth,
    LogisticGrowth,
    compute_treatment_effect,
)
from digital_twin_tumor.twin_engine.model_selection import (
    compute_akaike_weights,
    ensemble_predict,
    fit_all_models,
    select_best_model,
)
from digital_twin_tumor.twin_engine.simulation import SimulationEngine
from digital_twin_tumor.twin_engine.uncertainty import (
    assess_reliability,
    bootstrap_growth_parameters,
    calibrate_scan_rescan,
    compute_measurement_uncertainty,
    compute_prediction_intervals,
)

__all__ = [
    # Growth models
    "BaseGrowthModel",
    "ExponentialGrowth",
    "LogisticGrowth",
    "GompertzGrowth",
    "compute_treatment_effect",
    # Model selection
    "fit_all_models",
    "compute_akaike_weights",
    "ensemble_predict",
    "select_best_model",
    # Simulation
    "SimulationEngine",
    # Uncertainty
    "compute_measurement_uncertainty",
    "calibrate_scan_rescan",
    "bootstrap_growth_parameters",
    "compute_prediction_intervals",
    "assess_reliability",
]
