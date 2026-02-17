"""Tests for evaluation backtesting: leave-one-out, model comparison,
and uncertainty band computation.

Uses known exponential growth data so expected results are predictable.
"""
from __future__ import annotations

import numpy as np
import numpy.testing as npt
import pytest

from digital_twin_tumor.evaluation.backtest import (
    compute_uncertainty_bands,
    run_leave_one_out_backtest,
    run_model_comparison,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def exponential_data():
    """Known exponential growth data: V(t) = 1000 * exp(0.05 * t)."""
    times = np.array([0.0, 4.0, 8.0, 12.0, 16.0, 20.0], dtype=np.float64)
    V0, r = 1000.0, 0.05
    volumes = V0 * np.exp(r * times)
    return {"times": times, "volumes": volumes}


@pytest.fixture()
def small_data():
    """Too few points for a valid backtest (only 2)."""
    return {
        "times": np.array([0.0, 5.0], dtype=np.float64),
        "volumes": np.array([100.0, 150.0], dtype=np.float64),
    }


# ---------------------------------------------------------------------------
# run_leave_one_out_backtest
# ---------------------------------------------------------------------------


class TestLeaveOneOutBacktest:
    def test_returns_expected_keys(self, exponential_data):
        result = run_leave_one_out_backtest(exponential_data)
        assert "mae_mm3" in result
        assert "rmse_mm3" in result
        assert "mape_pct" in result
        assert "coverage_80" in result
        assert "best_model_name" in result
        assert "per_model_summary" in result

    def test_exponential_good_fit(self, exponential_data):
        result = run_leave_one_out_backtest(exponential_data)
        # Exponential model should fit this data well
        assert not np.isnan(result["mae_mm3"])
        assert result["best_model_name"] != "none"

    def test_errors_list_nonempty(self, exponential_data):
        result = run_leave_one_out_backtest(exponential_data)
        assert len(result["errors"]) > 0

    def test_too_few_points_skips(self, small_data):
        result = run_leave_one_out_backtest(small_data)
        assert "skip_reason" in result
        assert np.isnan(result["mae_mm3"])


# ---------------------------------------------------------------------------
# run_model_comparison
# ---------------------------------------------------------------------------


class TestRunModelComparison:
    def test_returns_expected_keys(self, exponential_data):
        result = run_model_comparison(exponential_data)
        assert "models" in result
        assert "selected_model" in result
        assert "evidence_ratio" in result

    def test_models_have_aic(self, exponential_data):
        result = run_model_comparison(exponential_data)
        for m in result["models"]:
            assert "aic" in m
            assert "name" in m

    def test_akaike_weights_sum_to_one(self, exponential_data):
        result = run_model_comparison(exponential_data)
        weights = [m.get("akaike_weight", 0) for m in result["models"]]
        assert sum(weights) == pytest.approx(1.0, abs=0.01)

    def test_too_few_points(self):
        data = {
            "times": np.array([0.0], dtype=np.float64),
            "volumes": np.array([100.0], dtype=np.float64),
        }
        result = run_model_comparison(data)
        assert result["selected_model"] == "none"


# ---------------------------------------------------------------------------
# compute_uncertainty_bands
# ---------------------------------------------------------------------------


class TestComputeUncertaintyBands:
    def test_returns_expected_keys(self, exponential_data):
        comparison = run_model_comparison(exponential_data)
        result = compute_uncertainty_bands(
            comparison,
            times=exponential_data["times"],
            volumes=exponential_data["volumes"],
            n_bootstrap=20,
        )
        assert "lower" in result
        assert "upper" in result
        assert "median" in result
        assert "confidence" in result
        assert "n_successful_boots" in result

    def test_lower_le_upper(self, exponential_data):
        comparison = run_model_comparison(exponential_data)
        result = compute_uncertainty_bands(
            comparison,
            times=exponential_data["times"],
            volumes=exponential_data["volumes"],
            n_bootstrap=30,
        )
        assert np.all(result["lower"] <= result["upper"] + 1e-6)

    def test_median_between_bounds(self, exponential_data):
        comparison = run_model_comparison(exponential_data)
        result = compute_uncertainty_bands(
            comparison,
            times=exponential_data["times"],
            volumes=exponential_data["volumes"],
            n_bootstrap=30,
        )
        if result["n_successful_boots"] > 0:
            assert np.all(result["median"] >= result["lower"] - 1e-6)
            assert np.all(result["median"] <= result["upper"] + 1e-6)

    def test_no_times_returns_error(self):
        result = compute_uncertainty_bands(
            {"selected_model": "exponential", "models": []},
        )
        assert "error" in result or result["n_successful_boots"] == 0
