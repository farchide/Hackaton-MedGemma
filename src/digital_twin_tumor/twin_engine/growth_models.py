"""Parametric tumor growth models with treatment effects (ADR-005).

Implements exponential, logistic, and Gompertz growth models with optional
piecewise-constant treatment terms.  Fitting uses multi-start L-BFGS-B
optimisation in log-space with Latin Hypercube Sampling for initial conditions.
"""
from __future__ import annotations

import logging
import warnings
from abc import ABC, abstractmethod

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
from scipy.stats.qmc import LatinHypercube

from digital_twin_tumor.domain.models import GrowthModelResult, TherapyEvent

logger = logging.getLogger(__name__)

_N_STARTS = 50
_MIN_VOLUME = 1e-6

# ---------------------------------------------------------------------------
# Treatment-effect helper
# ---------------------------------------------------------------------------

def compute_treatment_effect(
    t: float,
    therapy_events: list[TherapyEvent],
    sensitivity: float = 1.0,
) -> float:
    """Return piecewise-constant treatment effect E(t) at time *t*.

    Each therapy event contributes *sensitivity* while active (between
    start and end weeks).  Events without end date are treated as ongoing.
    """
    if not therapy_events:
        return 0.0
    effect = 0.0
    for event in therapy_events:
        start = _event_start_week(event)
        end = _event_end_week(event)
        if start is None:
            continue
        if end is None:
            if t >= start:
                effect += sensitivity
        elif start <= t <= end:
            effect += sensitivity
    return effect


def _event_start_week(event: TherapyEvent) -> float | None:
    """Extract start week: numeric start_date, then metadata, else None."""
    if isinstance(event.start_date, (int, float)):
        return float(event.start_date)
    meta = event.metadata.get("start_week")
    return float(meta) if meta is not None else None


def _event_end_week(event: TherapyEvent) -> float | None:
    """Extract end week: numeric end_date, then metadata, else None."""
    if isinstance(event.end_date, (int, float)):
        return float(event.end_date)
    meta = event.metadata.get("end_week")
    return float(meta) if meta is not None else None

# ---------------------------------------------------------------------------
# Base growth model
# ---------------------------------------------------------------------------

class BaseGrowthModel(ABC):
    """Abstract base for parametric growth models."""

    name: str = "base"

    def fit(
        self,
        times: np.ndarray,
        volumes: np.ndarray,
        therapy_events: list[TherapyEvent] | None = None,
    ) -> GrowthModelResult:
        """Fit model to observed (times, volumes) via multi-start L-BFGS-B."""
        times = np.asarray(times, dtype=np.float64)
        volumes = np.asarray(volumes, dtype=np.float64)

        valid = (volumes > 0) & np.isfinite(volumes) & np.isfinite(times)
        times, volumes = times[valid], volumes[valid]
        n = len(times)
        if n < 2:
            return self._default_result(times, volumes)

        therapy = therapy_events or []
        bounds = self._parameter_bounds(volumes)
        k = len(bounds)

        lhs = LatinHypercube(d=k, seed=42)
        samples = lhs.random(n=_N_STARTS)
        lower = np.array([b[0] for b in bounds])
        upper = np.array([b[1] for b in bounds])
        initial_points = lower + samples * (upper - lower)

        best_cost, best_params = np.inf, None
        log_volumes = np.log(np.maximum(volumes, _MIN_VOLUME))

        for x0 in initial_points:
            try:
                res = minimize(
                    self._objective, x0, args=(times, log_volumes, therapy),
                    method="L-BFGS-B", bounds=bounds,
                    options={"maxiter": 500, "ftol": 1e-12},
                )
                if res.fun < best_cost:
                    best_cost, best_params = res.fun, res.x
            except (ValueError, RuntimeWarning, RuntimeError):
                continue

        if best_params is None:
            return self._default_result(times, volumes)

        params_dict = self._unpack_params(best_params)
        fitted = self.predict(times, params_dict, therapy_events=therapy)
        residuals = volumes - fitted
        rss = best_cost
        aic = n * np.log(max(rss / n, _MIN_VOLUME)) + 2 * k
        bic = n * np.log(max(rss / n, _MIN_VOLUME)) + k * np.log(n)

        return GrowthModelResult(
            model_name=self.name, parameters=params_dict,
            aic=float(aic), bic=float(bic),
            residuals=residuals, fitted_values=fitted,
        )

    @abstractmethod
    def predict(
        self, times: np.ndarray, params: dict[str, float],
        therapy_events: list[TherapyEvent] | None = None,
    ) -> np.ndarray:
        """Predict volumes at *times* given *params*."""

    def _objective(
        self, x: np.ndarray, times: np.ndarray,
        log_volumes: np.ndarray, therapy: list[TherapyEvent],
    ) -> float:
        """Sum of squared residuals in log-space."""
        params = self._unpack_params(x)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            predicted = self.predict(times, params, therapy_events=therapy)
        log_pred = np.log(np.maximum(predicted, _MIN_VOLUME))
        return float(np.sum((log_volumes - log_pred) ** 2))

    @abstractmethod
    def _parameter_bounds(self, volumes: np.ndarray) -> list[tuple[float, float]]:
        """Return bounds for each parameter."""

    @abstractmethod
    def _unpack_params(self, x: np.ndarray) -> dict[str, float]:
        """Convert optimiser vector to named parameters."""

    def _default_result(self, times: np.ndarray, volumes: np.ndarray) -> GrowthModelResult:
        """Return a sensible fallback when fitting fails."""
        v0 = float(volumes[0]) if len(volumes) > 0 else 1.0
        return GrowthModelResult(
            model_name=self.name, parameters={"V0": v0, "r": 0.0},
            aic=float("inf"), bic=float("inf"),
            residuals=np.zeros_like(volumes),
            fitted_values=np.full_like(volumes, v0),
        )

# ---------------------------------------------------------------------------
# Exponential Growth
# ---------------------------------------------------------------------------

class ExponentialGrowth(BaseGrowthModel):
    """V(t) = V0 * exp(r * t).  With treatment: V(t) = V0 * exp((r - s*E(t)) * t)."""

    name: str = "exponential"

    def predict(
        self, times: np.ndarray, params: dict[str, float],
        therapy_events: list[TherapyEvent] | None = None,
    ) -> np.ndarray:
        """Predict volumes using exponential model."""
        times = np.asarray(times, dtype=np.float64)
        v0, r, s = params["V0"], params["r"], params.get("s", 0.5)
        therapy = therapy_events or []
        if not therapy:
            return v0 * np.exp(r * times)
        result = np.empty_like(times)
        for i, t in enumerate(times):
            e_t = compute_treatment_effect(t, therapy)
            result[i] = v0 * np.exp((r - s * e_t) * t)
        return np.maximum(result, _MIN_VOLUME)

    def _parameter_bounds(self, volumes: np.ndarray) -> list[tuple[float, float]]:
        return [(0.1, 1e6), (1e-5, 1.0)]

    def _unpack_params(self, x: np.ndarray) -> dict[str, float]:
        return {"V0": float(x[0]), "r": float(x[1])}

# ---------------------------------------------------------------------------
# Logistic Growth
# ---------------------------------------------------------------------------

class LogisticGrowth(BaseGrowthModel):
    """V(t) = K / (1 + ((K-V0)/V0)*exp(-r*t)).  With treatment uses ODE solver."""

    name: str = "logistic"

    def predict(
        self, times: np.ndarray, params: dict[str, float],
        therapy_events: list[TherapyEvent] | None = None,
    ) -> np.ndarray:
        """Predict volumes using logistic model."""
        times = np.asarray(times, dtype=np.float64)
        v0, r, k = params["V0"], params["r"], params["K"]
        s = params.get("s", 0.5)
        therapy = therapy_events or []
        if not therapy:
            denom = 1.0 + ((k - v0) / max(v0, _MIN_VOLUME)) * np.exp(-r * times)
            return np.maximum(k / denom, _MIN_VOLUME)
        return self._solve_ode(times, v0, r, k, s, therapy)

    def _solve_ode(
        self, times: np.ndarray, v0: float, r: float,
        k: float, s: float, therapy: list[TherapyEvent],
    ) -> np.ndarray:
        """Solve logistic ODE with treatment: dV/dt = r*V*(1-V/K) - s*E(t)*V."""
        if len(times) == 0:
            return np.array([], dtype=np.float64)
        def rhs(_t: float, y: np.ndarray) -> np.ndarray:
            v = max(y[0], _MIN_VOLUME)
            e_t = compute_treatment_effect(_t, therapy)
            return np.array([r * v * (1.0 - v / k) - s * e_t * v])
        try:
            sol = solve_ivp(
                rhs, (float(times[0]), float(times[-1])), [v0],
                t_eval=times, method="RK45", rtol=1e-8, atol=1e-10, max_step=0.5,
            )
            if sol.success:
                return np.maximum(sol.y[0], _MIN_VOLUME)
        except Exception:
            pass
        denom = 1.0 + ((k - v0) / max(v0, _MIN_VOLUME)) * np.exp(-r * times)
        return np.maximum(k / denom, _MIN_VOLUME)

    def _parameter_bounds(self, volumes: np.ndarray) -> list[tuple[float, float]]:
        v_max = float(np.max(volumes)) if len(volumes) > 0 else 100.0
        return [(0.1, 1e6), (1e-5, 1.0), (max(v_max, 0.1), 1e7)]

    def _unpack_params(self, x: np.ndarray) -> dict[str, float]:
        return {"V0": float(x[0]), "r": float(x[1]), "K": float(x[2])}

# ---------------------------------------------------------------------------
# Gompertz Growth
# ---------------------------------------------------------------------------

class GompertzGrowth(BaseGrowthModel):
    """V(t) = K * exp(log(V0/K) * exp(-r*t)).  With treatment uses ODE solver."""

    name: str = "gompertz"

    def predict(
        self, times: np.ndarray, params: dict[str, float],
        therapy_events: list[TherapyEvent] | None = None,
    ) -> np.ndarray:
        """Predict volumes using Gompertz model."""
        times = np.asarray(times, dtype=np.float64)
        v0, r, k = params["V0"], params["r"], params["K"]
        s = params.get("s", 0.5)
        therapy = therapy_events or []
        if not therapy:
            log_ratio = np.log(max(v0, _MIN_VOLUME) / max(k, _MIN_VOLUME))
            return np.maximum(k * np.exp(log_ratio * np.exp(-r * times)), _MIN_VOLUME)
        return self._solve_ode(times, v0, r, k, s, therapy)

    def _solve_ode(
        self, times: np.ndarray, v0: float, r: float,
        k: float, s: float, therapy: list[TherapyEvent],
    ) -> np.ndarray:
        """Solve Gompertz ODE with treatment: dV/dt = r*V*log(K/V) - s*E(t)*V."""
        if len(times) == 0:
            return np.array([], dtype=np.float64)
        def rhs(_t: float, y: np.ndarray) -> np.ndarray:
            v = max(y[0], _MIN_VOLUME)
            e_t = compute_treatment_effect(_t, therapy)
            return np.array([r * v * np.log(max(k / v, _MIN_VOLUME)) - s * e_t * v])
        try:
            sol = solve_ivp(
                rhs, (float(times[0]), float(times[-1])), [v0],
                t_eval=times, method="RK45", rtol=1e-8, atol=1e-10, max_step=0.5,
            )
            if sol.success:
                return np.maximum(sol.y[0], _MIN_VOLUME)
        except Exception:
            pass
        log_ratio = np.log(max(v0, _MIN_VOLUME) / max(k, _MIN_VOLUME))
        return np.maximum(k * np.exp(log_ratio * np.exp(-r * times)), _MIN_VOLUME)

    def _parameter_bounds(self, volumes: np.ndarray) -> list[tuple[float, float]]:
        v_max = float(np.max(volumes)) if len(volumes) > 0 else 100.0
        return [(0.1, 1e6), (1e-5, 1.0), (max(v_max, 0.1), 1e7)]

    def _unpack_params(self, x: np.ndarray) -> dict[str, float]:
        return {"V0": float(x[0]), "r": float(x[1]), "K": float(x[2])}
