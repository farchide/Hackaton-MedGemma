"""Counterfactual simulation engine (ADR-005).

Generates what-if trajectories by modifying therapy events and re-running
the fitted growth model.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np

from digital_twin_tumor.domain.models import (
    GrowthModelResult, SimulationResult, TherapyEvent,
)
from digital_twin_tumor.twin_engine.growth_models import (
    ExponentialGrowth, GompertzGrowth, LogisticGrowth, compute_treatment_effect,
)

logger = logging.getLogger(__name__)

_MODEL_MAP = {
    "exponential": ExponentialGrowth,
    "logistic": LogisticGrowth,
    "gompertz": GompertzGrowth,
}


def _get_model(name: str) -> ExponentialGrowth | LogisticGrowth | GompertzGrowth:
    """Instantiate growth model by name."""
    cls = _MODEL_MAP.get(name)
    if cls is None:
        raise ValueError(f"Unknown model: {name}")
    return cls()

# ---------------------------------------------------------------------------
# Therapy event manipulation helpers
# ---------------------------------------------------------------------------

def _shift_therapy_events(events: list[TherapyEvent], shift: float) -> list[TherapyEvent]:
    """Return events with start/end shifted by *shift* weeks."""
    shifted: list[TherapyEvent] = []
    for ev in events:
        meta = dict(ev.metadata)
        if "start_week" in meta:
            meta["start_week"] = float(meta["start_week"]) + shift
        if "end_week" in meta:
            meta["end_week"] = float(meta["end_week"]) + shift
        new_start = ev.start_date
        new_end = ev.end_date
        if isinstance(new_start, (int, float)):
            new_start = type(new_start)(float(new_start) + shift)
        if isinstance(new_end, (int, float)):
            new_end = type(new_end)(float(new_end) + shift)
        shifted.append(TherapyEvent(
            therapy_id=ev.therapy_id, patient_id=ev.patient_id,
            start_date=new_start, end_date=new_end,
            therapy_type=ev.therapy_type, dose=ev.dose, metadata=meta,
        ))
    return shifted


def _modify_therapy_effect(events: list[TherapyEvent], new_effect: float) -> list[TherapyEvent]:
    """Return events with modified sensitivity in metadata."""
    return [
        TherapyEvent(
            therapy_id=ev.therapy_id, patient_id=ev.patient_id,
            start_date=ev.start_date, end_date=ev.end_date,
            therapy_type=ev.therapy_type, dose=ev.dose,
            metadata={**ev.metadata, "sensitivity": new_effect},
        )
        for ev in events
    ]


def _get_week(event: TherapyEvent, which: str) -> float | None:
    """Extract start or end week from event metadata or numeric field."""
    if which == "start":
        val = event.metadata.get("start_week", event.start_date)
    else:
        val = event.metadata.get("end_week", event.end_date)
    if val is None:
        return None
    return float(val) if isinstance(val, (int, float)) else None


def _make_event(ev: TherapyEvent, start_week: float, end_week: float) -> TherapyEvent:
    """Copy event with new start/end week metadata."""
    meta = {**ev.metadata, "start_week": start_week, "end_week": end_week}
    return TherapyEvent(
        therapy_id=ev.therapy_id, patient_id=ev.patient_id,
        start_date=ev.start_date, end_date=ev.end_date,
        therapy_type=ev.therapy_type, dose=ev.dose, metadata=meta,
    )


def _apply_treatment_holiday(
    events: list[TherapyEvent], h_start: float, h_end: float,
) -> list[TherapyEvent]:
    """Remove therapy during [h_start, h_end], splitting events as needed."""
    result: list[TherapyEvent] = []
    for ev in events:
        start = _get_week(ev, "start")
        end = _get_week(ev, "end")
        if start is None:
            result.append(ev)
            continue
        if end is None:
            end = float("inf")
        if start >= h_start and end <= h_end:
            continue  # fully within holiday
        if end <= h_start or start >= h_end:
            result.append(ev)
            continue
        if start < h_start:
            result.append(_make_event(ev, start, h_start))
        if end > h_end:
            result.append(_make_event(ev, h_end, end))
    return result


def _apply_resistance(
    events: list[TherapyEvent], onset: float, decay_rate: float = 0.1,
) -> list[TherapyEvent]:
    """Tag events with resistance onset/decay metadata."""
    return [
        TherapyEvent(
            therapy_id=ev.therapy_id, patient_id=ev.patient_id,
            start_date=ev.start_date, end_date=ev.end_date,
            therapy_type=ev.therapy_type, dose=ev.dose,
            metadata={**ev.metadata, "resistance_onset": onset,
                      "resistance_decay_rate": decay_rate},
        )
        for ev in events
    ]

# ---------------------------------------------------------------------------
# Resistance-aware prediction
# ---------------------------------------------------------------------------

def _predict_with_resistance(
    model_instance: Any, times: np.ndarray,
    params: dict[str, float], therapy_events: list[TherapyEvent],
) -> np.ndarray:
    """Predict with exponential decay of treatment effect after resistance onset."""
    has_resistance = any("resistance_onset" in e.metadata for e in therapy_events)
    if not has_resistance:
        return model_instance.predict(times, params, therapy_events=therapy_events)

    onset = therapy_events[0].metadata.get("resistance_onset", float("inf"))
    decay = therapy_events[0].metadata.get("resistance_decay_rate", 0.1)
    n_steps = 20
    t_min, t_max = float(np.min(times)), float(np.max(times))
    step_size = (t_max - t_min) / n_steps if n_steps > 0 else 1.0

    modified: list[TherapyEvent] = []
    for ev in therapy_events:
        start = _get_week(ev, "start")
        end = _get_week(ev, "end")
        if start is None:
            continue
        if end is None:
            end = t_max
        for i in range(n_steps):
            seg_s = max(t_min + i * step_size, start)
            seg_e = min(t_min + (i + 1) * step_size, end)
            if seg_s >= seg_e:
                continue
            mid = (seg_s + seg_e) / 2.0
            sens = float(np.exp(-decay * (mid - onset))) if mid > onset else 1.0
            meta = {**ev.metadata, "start_week": seg_s, "end_week": seg_e,
                    "sensitivity": sens}
            meta.pop("resistance_onset", None)
            meta.pop("resistance_decay_rate", None)
            modified.append(TherapyEvent(
                therapy_id=ev.therapy_id, patient_id=ev.patient_id,
                start_date=ev.start_date, end_date=ev.end_date,
                therapy_type=ev.therapy_type, dose=ev.dose, metadata=meta,
            ))
    return model_instance.predict(times, params, therapy_events=modified)

# ---------------------------------------------------------------------------
# Simulation Engine
# ---------------------------------------------------------------------------

class SimulationEngine:
    """Counterfactual simulation engine for what-if growth scenarios."""

    def run_natural_history(self, model: GrowthModelResult, times: np.ndarray) -> SimulationResult:
        """Simulate growth with no treatment (E(t) = 0)."""
        inst = _get_model(model.model_name)
        predicted = inst.predict(times, model.parameters, therapy_events=None)
        return SimulationResult(
            scenario_name="natural_history",
            time_points=np.asarray(times, dtype=np.float64),
            predicted_volumes=predicted, parameters=dict(model.parameters),
        )

    def run_earlier_treatment(
        self, model: GrowthModelResult, therapy_events: list[TherapyEvent],
        shift_weeks: float, times: np.ndarray,
    ) -> SimulationResult:
        """Simulate treatment starting earlier by *shift_weeks*."""
        shifted = _shift_therapy_events(therapy_events, -abs(shift_weeks))
        inst = _get_model(model.model_name)
        predicted = inst.predict(times, model.parameters, therapy_events=shifted)
        return SimulationResult(
            scenario_name=f"earlier_treatment_{shift_weeks}w",
            time_points=np.asarray(times, dtype=np.float64),
            predicted_volumes=predicted,
            parameters={**model.parameters, "shift_weeks": -abs(shift_weeks)},
        )

    def run_later_treatment(
        self, model: GrowthModelResult, therapy_events: list[TherapyEvent],
        shift_weeks: float, times: np.ndarray,
    ) -> SimulationResult:
        """Simulate treatment starting later by *shift_weeks*."""
        shifted = _shift_therapy_events(therapy_events, abs(shift_weeks))
        inst = _get_model(model.model_name)
        predicted = inst.predict(times, model.parameters, therapy_events=shifted)
        return SimulationResult(
            scenario_name=f"later_treatment_{shift_weeks}w",
            time_points=np.asarray(times, dtype=np.float64),
            predicted_volumes=predicted,
            parameters={**model.parameters, "shift_weeks": abs(shift_weeks)},
        )

    def run_regimen_switch(
        self, model: GrowthModelResult, therapy_events: list[TherapyEvent],
        new_effect: float, times: np.ndarray,
    ) -> SimulationResult:
        """Simulate switching to a regimen with different treatment effect."""
        modified = _modify_therapy_effect(therapy_events, new_effect)
        inst = _get_model(model.model_name)
        params_with_effect = {**model.parameters, "s": new_effect}
        predicted = inst.predict(times, params_with_effect, therapy_events=modified)
        return SimulationResult(
            scenario_name=f"regimen_switch_effect_{new_effect}",
            time_points=np.asarray(times, dtype=np.float64),
            predicted_volumes=predicted,
            parameters={**model.parameters, "new_effect": new_effect},
        )

    def run_treatment_holiday(
        self, model: GrowthModelResult, therapy_events: list[TherapyEvent],
        holiday_start: float, holiday_end: float, times: np.ndarray,
    ) -> SimulationResult:
        """Simulate a treatment holiday (gap in therapy)."""
        modified = _apply_treatment_holiday(therapy_events, holiday_start, holiday_end)
        inst = _get_model(model.model_name)
        predicted = inst.predict(times, model.parameters, therapy_events=modified)
        return SimulationResult(
            scenario_name=f"treatment_holiday_{holiday_start}-{holiday_end}w",
            time_points=np.asarray(times, dtype=np.float64),
            predicted_volumes=predicted,
            parameters={**model.parameters, "holiday_start": holiday_start,
                        "holiday_end": holiday_end},
        )

    def run_resistance_scenario(
        self, model: GrowthModelResult, therapy_events: list[TherapyEvent],
        resistance_onset_weeks: float, times: np.ndarray,
    ) -> SimulationResult:
        """Simulate acquired resistance with exponential decay of effect."""
        modified = _apply_resistance(therapy_events, resistance_onset_weeks)
        inst = _get_model(model.model_name)
        predicted = _predict_with_resistance(inst, times, model.parameters, modified)
        return SimulationResult(
            scenario_name=f"resistance_onset_{resistance_onset_weeks}w",
            time_points=np.asarray(times, dtype=np.float64),
            predicted_volumes=predicted,
            parameters={**model.parameters,
                        "resistance_onset_weeks": resistance_onset_weeks},
        )

    def run_dose_escalation(
        self, model: GrowthModelResult, therapy_events: list[TherapyEvent],
        escalation_factor: float, escalation_start_week: float, times: np.ndarray,
    ) -> SimulationResult:
        """Simulate dose escalation at a specified time point.

        After escalation_start_week, treatment sensitivity is multiplied
        by escalation_factor.
        """
        modified: list[TherapyEvent] = []
        for ev in therapy_events:
            start = _get_week(ev, "start")
            if start is not None and start >= escalation_start_week:
                sens = ev.metadata.get("sensitivity", 1.0) * escalation_factor
                modified.append(TherapyEvent(
                    therapy_id=ev.therapy_id, patient_id=ev.patient_id,
                    start_date=ev.start_date, end_date=ev.end_date,
                    therapy_type=ev.therapy_type, dose=ev.dose,
                    metadata={**ev.metadata, "sensitivity": sens},
                ))
            else:
                modified.append(ev)
        inst = _get_model(model.model_name)
        predicted = inst.predict(times, model.parameters, therapy_events=modified)
        return SimulationResult(
            scenario_name=f"dose_escalation_{escalation_factor}x_at_{escalation_start_week}w",
            time_points=np.asarray(times, dtype=np.float64),
            predicted_volumes=predicted,
            parameters={**model.parameters,
                        "escalation_factor": escalation_factor,
                        "escalation_start_week": escalation_start_week},
        )

    def run_combination_therapy(
        self, model: GrowthModelResult,
        primary_events: list[TherapyEvent],
        secondary_events: list[TherapyEvent],
        times: np.ndarray,
    ) -> SimulationResult:
        """Simulate combination therapy with two concurrent regimens.

        Combines both therapy event lists and predicts using the sum
        of their treatment effects.
        """
        combined = list(primary_events) + list(secondary_events)
        inst = _get_model(model.model_name)
        predicted = inst.predict(times, model.parameters, therapy_events=combined)
        return SimulationResult(
            scenario_name="combination_therapy",
            time_points=np.asarray(times, dtype=np.float64),
            predicted_volumes=predicted,
            parameters={**model.parameters, "n_regimens": 2},
        )

    def run_all_scenarios(
        self, model: GrowthModelResult, therapy_events: list[TherapyEvent],
        times: np.ndarray,
    ) -> list[SimulationResult]:
        """Run all standard counterfactual scenarios.

        Generates: natural history, earlier/later treatment (2w),
        regimen switch (0.75), treatment holiday (w4-8), resistance (w8),
        dose escalation (1.5x at w6).
        """
        return [
            self.run_natural_history(model, times),
            self.run_earlier_treatment(model, therapy_events, 2.0, times),
            self.run_later_treatment(model, therapy_events, 2.0, times),
            self.run_regimen_switch(model, therapy_events, 0.75, times),
            self.run_treatment_holiday(model, therapy_events, 4.0, 8.0, times),
            self.run_resistance_scenario(model, therapy_events, 8.0, times),
            self.run_dose_escalation(model, therapy_events, 1.5, 6.0, times),
        ]
