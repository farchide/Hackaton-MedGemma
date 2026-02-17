"""Evaluation and metrics for the Digital Twin Tumor system.

Provides backtesting, calibration, measurement quality, RECIST agreement,
tracking accuracy, and runtime profiling metrics. All classes are designed
for honest, rigorous evaluation suitable for the MedGemma Impact Challenge.
"""
from digital_twin_tumor.evaluation.backtest import (
    compute_uncertainty_bands,
    run_leave_one_out_backtest,
    run_model_comparison,
)
from digital_twin_tumor.evaluation.metrics import (
    CalibrationMetrics,
    ForecastingBacktest,
    MeasurementMetrics,
    RECISTAgreement,
    RuntimeMetrics,
    TrackingMetrics,
    format_metrics_table,
    generate_evaluation_report,
)

__all__ = [
    "ForecastingBacktest",
    "CalibrationMetrics",
    "MeasurementMetrics",
    "RECISTAgreement",
    "TrackingMetrics",
    "RuntimeMetrics",
    "generate_evaluation_report",
    "format_metrics_table",
    "run_leave_one_out_backtest",
    "run_model_comparison",
    "compute_uncertainty_bands",
]
