"""Tests for twin engine: growth models, model selection, simulation, uncertainty."""

from __future__ import annotations

import numpy as np
import numpy.testing as npt
import pytest

from digital_twin_tumor.domain.models import GrowthModelResult
from digital_twin_tumor.twin_engine.growth_models import (
    ExponentialGrowth,
    GompertzGrowth,
    LogisticGrowth,
)
from digital_twin_tumor.twin_engine.model_selection import (
    compute_akaike_weights,
    fit_all_models,
)
from digital_twin_tumor.twin_engine.simulation import SimulationEngine
from digital_twin_tumor.twin_engine.uncertainty import (
    bootstrap_growth_parameters,
    compute_measurement_uncertainty,
)


# =====================================================================
# Synthetic data generators
# =====================================================================


def _make_exponential_data(
    V0: float = 100.0, r: float = 0.05, n_points: int = 10, noise_sigma: float = 0.0
) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic exponential growth data: V(t) = V0 * exp(r*t)."""
    rng = np.random.default_rng(seed=42)
    times = np.linspace(0, 20, n_points)
    volumes = V0 * np.exp(r * times)
    if noise_sigma > 0:
        volumes += rng.normal(0, noise_sigma, size=n_points)
    volumes = np.maximum(volumes, 1e-6)
    return times, volumes


def _make_logistic_data(
    V0: float = 50.0, r: float = 0.1, K: float = 500.0, n_points: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic logistic growth data: V(t) = K / (1+((K-V0)/V0)*exp(-r*t))."""
    times = np.linspace(0, 50, n_points)
    denom = 1.0 + ((K - V0) / V0) * np.exp(-r * times)
    volumes = K / denom
    return times, volumes


def _make_gompertz_data(
    V0: float = 50.0, r: float = 0.08, K: float = 500.0, n_points: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate Gompertz growth data: V(t) = K * exp(log(V0/K)*exp(-r*t))."""
    times = np.linspace(0, 50, n_points)
    log_ratio = np.log(V0 / K)
    volumes = K * np.exp(log_ratio * np.exp(-r * times))
    return times, volumes


# =====================================================================
# ExponentialGrowth
# =====================================================================


class TestExponentialGrowth:
    """Test exponential growth model fitting."""

    def test_fit_recovers_parameters(self):
        V0_true, r_true = 100.0, 0.05
        times, volumes = _make_exponential_data(V0=V0_true, r=r_true, n_points=15)
        model = ExponentialGrowth()
        result = model.fit(times, volumes)
        assert result.model_name == "exponential"
        # Check recovered parameters
        assert abs(result.parameters["V0"] - V0_true) < V0_true * 0.5
        assert abs(result.parameters["r"] - r_true) < r_true * 0.5

    def test_predict(self):
        model = ExponentialGrowth()
        params = {"V0": 100.0, "r": 0.05}
        times = np.array([0.0, 10.0, 20.0])
        predicted = model.predict(times, params)
        expected = 100.0 * np.exp(0.05 * times)
        npt.assert_allclose(predicted, expected, rtol=1e-6)

    def test_fit_returns_aic(self):
        times, volumes = _make_exponential_data()
        model = ExponentialGrowth()
        result = model.fit(times, volumes)
        assert np.isfinite(result.aic)
        assert np.isfinite(result.bic)

    def test_insufficient_data_returns_default(self):
        model = ExponentialGrowth()
        times = np.array([0.0])
        volumes = np.array([100.0])
        result = model.fit(times, volumes)
        assert result.model_name == "exponential"
        assert result.aic == float("inf")


# =====================================================================
# LogisticGrowth
# =====================================================================


class TestLogisticGrowth:
    """Test logistic growth model fitting."""

    def test_fit_produces_result(self):
        times, volumes = _make_logistic_data(V0=50.0, r=0.1, K=500.0, n_points=20)
        model = LogisticGrowth()
        result = model.fit(times, volumes)
        assert result.model_name == "logistic"
        assert "V0" in result.parameters
        assert "r" in result.parameters
        assert "K" in result.parameters
        assert np.isfinite(result.aic)

    def test_predict_approaches_K(self):
        model = LogisticGrowth()
        params = {"V0": 50.0, "r": 0.1, "K": 500.0}
        # At very large time, should approach K
        times = np.array([0.0, 100.0, 500.0])
        predicted = model.predict(times, params)
        assert predicted[0] < predicted[-1]
        assert abs(predicted[-1] - 500.0) < 10.0


# =====================================================================
# GompertzGrowth
# =====================================================================


class TestGompertzGrowth:
    """Test Gompertz growth model fitting."""

    def test_fit_produces_result(self):
        times, volumes = _make_gompertz_data(V0=50.0, r=0.08, K=500.0, n_points=20)
        model = GompertzGrowth()
        result = model.fit(times, volumes)
        assert result.model_name == "gompertz"
        assert "V0" in result.parameters
        assert "r" in result.parameters
        assert "K" in result.parameters
        assert np.isfinite(result.aic)

    def test_predict_approaches_K(self):
        model = GompertzGrowth()
        params = {"V0": 50.0, "r": 0.08, "K": 500.0}
        times = np.array([0.0, 100.0, 500.0])
        predicted = model.predict(times, params)
        assert predicted[0] < predicted[-1]
        assert abs(predicted[-1] - 500.0) < 10.0


# =====================================================================
# fit_all_models
# =====================================================================


class TestFitAllModels:
    """Test ensemble model fitting."""

    def test_returns_three_models(self):
        times, volumes = _make_exponential_data(n_points=15)
        results = fit_all_models(times, volumes)
        assert len(results) == 3

    def test_sorted_by_aic(self):
        times, volumes = _make_exponential_data(n_points=15)
        results = fit_all_models(times, volumes)
        aics = [r.aic for r in results]
        assert aics == sorted(aics)

    def test_model_names_present(self):
        times, volumes = _make_exponential_data(n_points=15)
        results = fit_all_models(times, volumes)
        names = {r.model_name for r in results}
        assert "exponential" in names
        assert "logistic" in names
        assert "gompertz" in names


# =====================================================================
# compute_akaike_weights
# =====================================================================


class TestComputeAkaikeWeights:
    """Test Akaike weight computation."""

    def test_weights_sum_to_one(self):
        results = [
            GrowthModelResult(model_name="a", aic=100.0),
            GrowthModelResult(model_name="b", aic=102.0),
            GrowthModelResult(model_name="c", aic=110.0),
        ]
        weights = compute_akaike_weights(results)
        assert len(weights) == 3
        npt.assert_allclose(sum(weights), 1.0, atol=1e-10)

    def test_best_model_has_highest_weight(self):
        results = [
            GrowthModelResult(model_name="best", aic=50.0),
            GrowthModelResult(model_name="worse", aic=100.0),
            GrowthModelResult(model_name="worst", aic=200.0),
        ]
        weights = compute_akaike_weights(results)
        assert weights[0] > weights[1] > weights[2]

    def test_identical_aic_equal_weights(self):
        results = [
            GrowthModelResult(model_name="a", aic=100.0),
            GrowthModelResult(model_name="b", aic=100.0),
        ]
        weights = compute_akaike_weights(results)
        npt.assert_allclose(weights[0], weights[1], atol=1e-10)
        npt.assert_allclose(sum(weights), 1.0, atol=1e-10)

    def test_empty_results(self):
        assert compute_akaike_weights([]) == []

    def test_all_infinite_aic(self):
        results = [
            GrowthModelResult(model_name="a", aic=float("inf")),
            GrowthModelResult(model_name="b", aic=float("inf")),
        ]
        weights = compute_akaike_weights(results)
        assert len(weights) == 2
        npt.assert_allclose(sum(weights), 1.0, atol=1e-10)


# =====================================================================
# SimulationEngine
# =====================================================================


class TestSimulationEngine:
    """Test counterfactual simulation engine."""

    def test_run_natural_history(self):
        model_result = GrowthModelResult(
            model_name="exponential",
            parameters={"V0": 100.0, "r": 0.05},
        )
        engine = SimulationEngine()
        times = np.linspace(0, 20, 10)
        sim = engine.run_natural_history(model_result, times)
        assert sim.scenario_name == "natural_history"
        assert sim.predicted_volumes.shape == times.shape
        # All volumes should be positive
        assert np.all(sim.predicted_volumes > 0)
        # Should be monotonically increasing for positive growth rate
        assert np.all(np.diff(sim.predicted_volumes) > 0)

    def test_natural_history_logistic(self):
        model_result = GrowthModelResult(
            model_name="logistic",
            parameters={"V0": 50.0, "r": 0.1, "K": 500.0},
        )
        engine = SimulationEngine()
        times = np.linspace(0, 100, 20)
        sim = engine.run_natural_history(model_result, times)
        assert sim.predicted_volumes[0] < sim.predicted_volumes[-1]

    def test_natural_history_gompertz(self):
        model_result = GrowthModelResult(
            model_name="gompertz",
            parameters={"V0": 50.0, "r": 0.08, "K": 500.0},
        )
        engine = SimulationEngine()
        times = np.linspace(0, 100, 20)
        sim = engine.run_natural_history(model_result, times)
        assert len(sim.predicted_volumes) == 20

    def test_unknown_model_raises(self):
        model_result = GrowthModelResult(
            model_name="unknown_model",
            parameters={"V0": 100.0},
        )
        engine = SimulationEngine()
        with pytest.raises(ValueError, match="Unknown model"):
            engine.run_natural_history(model_result, np.array([0, 1, 2]))


# =====================================================================
# Uncertainty
# =====================================================================


class TestComputeMeasurementUncertainty:
    """Test measurement uncertainty composition."""

    def test_rss_composition(self):
        result = compute_measurement_uncertainty(
            sigma_manual=3.0, sigma_auto=4.0, sigma_scan=0.0,
        )
        expected_total = np.sqrt(3.0 ** 2 + 4.0 ** 2)
        npt.assert_allclose(result.total_sigma, expected_total, atol=1e-10)

    def test_all_three_sources_high_reliability(self):
        result = compute_measurement_uncertainty(
            sigma_manual=1.5, sigma_auto=0.5, sigma_scan=0.3,
        )
        assert result.reliability == "HIGH"

    def test_two_sources_medium_reliability(self):
        result = compute_measurement_uncertainty(
            sigma_manual=1.5, sigma_auto=0.5, sigma_scan=0.0,
        )
        assert result.reliability == "MEDIUM"

    def test_one_source_low_reliability(self):
        result = compute_measurement_uncertainty(
            sigma_manual=1.5, sigma_auto=0.0, sigma_scan=0.0,
        )
        assert result.reliability == "LOW"

    def test_zero_uncertainty(self):
        result = compute_measurement_uncertainty(
            sigma_manual=0.0, sigma_auto=0.0, sigma_scan=0.0,
        )
        assert result.total_sigma == 0.0


class TestBootstrapGrowthParameters:
    """Test parametric bootstrap for growth model parameters."""

    def test_returns_prediction_intervals(self):
        times, volumes = _make_exponential_data(
            V0=100.0, r=0.05, n_points=10, noise_sigma=5.0,
        )
        result = bootstrap_growth_parameters(
            times, volumes, ExponentialGrowth, n_bootstrap=20, sigma_obs=5.0,
        )
        assert "parameter_samples" in result
        assert "prediction_intervals" in result
        assert "parameter_names" in result
        assert "cv" in result
        assert result["parameter_samples"].ndim == 2

    def test_prediction_intervals_contain_bounds(self):
        times, volumes = _make_exponential_data(
            V0=100.0, r=0.05, n_points=10, noise_sigma=5.0,
        )
        result = bootstrap_growth_parameters(
            times, volumes, ExponentialGrowth, n_bootstrap=20, sigma_obs=5.0,
        )
        intervals = result["prediction_intervals"]
        # Should have intervals for alpha=0.1 and 0.9
        assert 0.1 in intervals
        assert 0.9 in intervals
        lower, upper = intervals[0.1]
        assert lower.shape == times.shape
        assert upper.shape == times.shape
