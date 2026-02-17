"""Tests for domain models, events, and configuration."""

from __future__ import annotations

import os
import tempfile
from datetime import date, datetime

import numpy as np
import pytest
import yaml

from digital_twin_tumor.domain.events import Event, EventBus
from digital_twin_tumor.domain.models import (
    AppConfig,
    AuditEvent,
    DatabaseConfig,
    GrowthModelResult,
    Lesion,
    Measurement,
    NarrativeResult,
    Patient,
    ProcessedVolume,
    RECISTResponse,
    SimulationResult,
    TherapyEvent,
    TimePoint,
    UncertaintyEstimate,
    VoxelSpacing,
)


# =====================================================================
# Frozen dataclass instantiation
# =====================================================================


class TestFrozenDataclasses:
    """Verify that all frozen dataclasses can be instantiated."""

    def test_patient_instantiation(self):
        p = Patient()
        assert isinstance(p.patient_id, str)
        assert len(p.patient_id) > 0

    def test_voxel_spacing_instantiation(self):
        vs = VoxelSpacing(x=0.5, y=0.5, z=1.0)
        assert vs.x == 0.5
        assert vs.y == 0.5
        assert vs.z == 1.0

    def test_timepoint_instantiation(self):
        tp = TimePoint(modality="MRI")
        assert tp.modality == "MRI"
        assert tp.therapy_status == "pre"

    def test_processed_volume_instantiation(self):
        pv = ProcessedVolume()
        assert pv.pixel_data.size == 0
        assert pv.modality == "CT"

    def test_lesion_instantiation(self):
        les = Lesion(centroid=(1.0, 2.0, 3.0), volume_mm3=100.0)
        assert les.centroid == (1.0, 2.0, 3.0)
        assert les.volume_mm3 == 100.0
        assert les.is_target is False

    def test_measurement_instantiation(self):
        m = Measurement(diameter_mm=15.0)
        assert m.diameter_mm == 15.0
        assert m.method == "auto"

    def test_therapy_event_instantiation(self):
        te = TherapyEvent(therapy_type="immunotherapy")
        assert te.therapy_type == "immunotherapy"

    def test_growth_model_result_instantiation(self):
        g = GrowthModelResult(model_name="exponential", aic=100.0)
        assert g.model_name == "exponential"
        assert g.aic == 100.0

    def test_uncertainty_estimate_instantiation(self):
        u = UncertaintyEstimate(sigma_manual=1.5, sigma_auto=0.5)
        assert u.sigma_manual == 1.5

    def test_simulation_result_instantiation(self):
        s = SimulationResult(scenario_name="natural_history")
        assert s.scenario_name == "natural_history"

    def test_recist_response_instantiation(self):
        r = RECISTResponse(category="PR")
        assert r.category == "PR"

    def test_narrative_result_instantiation(self):
        n = NarrativeResult(text="test")
        assert n.text == "test"
        assert "research purposes" in n.disclaimer.lower()

    def test_audit_event_instantiation(self):
        ae = AuditEvent(action_type="test_action")
        assert ae.action_type == "test_action"
        assert isinstance(ae.event_id, str)

    def test_database_config_instantiation(self):
        dc = DatabaseConfig(host="db.example.com", port=5433)
        assert dc.host == "db.example.com"
        assert dc.port == 5433


# =====================================================================
# Frozen immutability
# =====================================================================


class TestFrozenImmutability:
    """Verify that frozen dataclasses reject attribute assignment."""

    def test_patient_immutable(self):
        p = Patient()
        with pytest.raises((AttributeError, TypeError)):
            p.patient_id = "new-id"

    def test_lesion_immutable(self):
        les = Lesion()
        with pytest.raises((AttributeError, TypeError)):
            les.volume_mm3 = 999.0

    def test_measurement_immutable(self):
        m = Measurement()
        with pytest.raises((AttributeError, TypeError)):
            m.diameter_mm = 42.0

    def test_voxel_spacing_immutable(self):
        vs = VoxelSpacing()
        with pytest.raises((AttributeError, TypeError)):
            vs.x = 2.0

    def test_timepoint_immutable(self):
        tp = TimePoint()
        with pytest.raises((AttributeError, TypeError)):
            tp.modality = "PET"

    def test_therapy_event_immutable(self):
        te = TherapyEvent()
        with pytest.raises((AttributeError, TypeError)):
            te.therapy_type = "new_type"

    def test_recist_response_immutable(self):
        r = RECISTResponse()
        with pytest.raises((AttributeError, TypeError)):
            r.category = "CR"

    def test_audit_event_immutable(self):
        ae = AuditEvent()
        with pytest.raises((AttributeError, TypeError)):
            ae.user_id = "new_user"


# =====================================================================
# EventBus
# =====================================================================


class TestEventBus:
    """Test the synchronous publish/subscribe event bus."""

    def test_subscribe_and_publish(self):
        bus = EventBus()
        received: list[Event] = []
        bus.subscribe("test_event", received.append)
        event = Event(type="test_event", payload={"key": "value"})
        bus.publish(event)
        assert len(received) == 1
        assert received[0].type == "test_event"
        assert received[0].payload == {"key": "value"}

    def test_multiple_subscribers(self):
        bus = EventBus()
        log_a: list[Event] = []
        log_b: list[Event] = []
        bus.subscribe("multi", log_a.append)
        bus.subscribe("multi", log_b.append)
        bus.publish(Event(type="multi"))
        assert len(log_a) == 1
        assert len(log_b) == 1

    def test_publish_no_subscribers(self):
        bus = EventBus()
        # Should not raise
        bus.publish(Event(type="no_listeners"))

    def test_subscriber_only_receives_matching_type(self):
        bus = EventBus()
        received: list[Event] = []
        bus.subscribe("type_a", received.append)
        bus.publish(Event(type="type_b"))
        assert len(received) == 0

    def test_clear_removes_all_handlers(self):
        bus = EventBus()
        received: list[Event] = []
        bus.subscribe("evt", received.append)
        bus.clear()
        bus.publish(Event(type="evt"))
        assert len(received) == 0

    def test_handler_count(self):
        bus = EventBus()
        bus.subscribe("evt", lambda e: None)
        bus.subscribe("evt", lambda e: None)
        assert bus.handler_count("evt") == 2
        assert bus.handler_count("other") == 0

    def test_event_types(self):
        bus = EventBus()
        bus.subscribe("beta", lambda e: None)
        bus.subscribe("alpha", lambda e: None)
        assert bus.event_types == ["alpha", "beta"]

    def test_event_timestamp(self):
        event = Event(type="ts_test")
        assert isinstance(event.timestamp, datetime)


# =====================================================================
# AppConfig
# =====================================================================


class TestAppConfig:
    """Test configuration loading from YAML and environment variables."""

    def test_load_from_yaml(self, tmp_path):
        yaml_content = {
            "compute": {"precision": "float16", "device": "cpu"},
            "data": {"cache_dir": "/tmp/cache"},
        }
        config_file = tmp_path / "config.yaml"
        with open(config_file, "w") as fh:
            yaml.dump(yaml_content, fh)

        cfg = AppConfig.load(default_path=str(config_file))
        assert cfg.get("compute.precision") == "float16"
        assert cfg.get("data.cache_dir") == "/tmp/cache"

    def test_load_with_overlay(self, tmp_path):
        default_file = tmp_path / "default.yaml"
        overlay_file = tmp_path / "overlay.yaml"
        with open(default_file, "w") as fh:
            yaml.dump({"a": 1, "b": {"c": 2}}, fh)
        with open(overlay_file, "w") as fh:
            yaml.dump({"b": {"c": 99, "d": 3}}, fh)

        cfg = AppConfig.load(
            default_path=str(default_file),
            overlay_path=str(overlay_file),
        )
        assert cfg.get("a") == 1
        assert cfg.get("b.c") == 99
        assert cfg.get("b.d") == 3

    def test_load_env_override(self, tmp_path, monkeypatch):
        config_file = tmp_path / "config.yaml"
        with open(config_file, "w") as fh:
            yaml.dump({"compute": {"precision": "float32"}}, fh)

        monkeypatch.setenv("DTT_COMPUTE__PRECISION", "int4")
        cfg = AppConfig.load(default_path=str(config_file))
        assert cfg.get("compute.precision") == "int4"

    def test_load_nonexistent_file(self, tmp_path):
        cfg = AppConfig.load(
            default_path=str(tmp_path / "does_not_exist.yaml")
        )
        assert cfg.data == {}

    def test_get_default_value(self):
        cfg = AppConfig(data={"a": 1})
        assert cfg.get("missing.key", default="fallback") == "fallback"

    def test_section_returns_dict(self):
        cfg = AppConfig(data={"storage": {"backend": "sqlite", "path": "/db"}})
        section = cfg.section("storage")
        assert isinstance(section, dict)
        assert section["backend"] == "sqlite"

    def test_section_missing_returns_empty(self):
        cfg = AppConfig(data={})
        assert cfg.section("nonexistent") == {}
