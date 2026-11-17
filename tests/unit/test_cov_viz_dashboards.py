"""Behavioural tests for active_inference.visualization.dashboards."""

import json
import logging
import math
from datetime import datetime

import pytest

from active_inference.visualization.dashboards import (
    ActiveInferenceMonitor,
    Dashboard,
    DashboardComponent,
    DashboardConfig,
    RealTimeMonitor,
)


def test_dashboard_config_defaults():
    config = DashboardConfig(title="T")
    assert config.refresh_interval == 1000
    assert config.layout == "grid"
    assert config.components == []
    assert DashboardConfig(title="U").components is not config.components


def test_component_update_and_export():
    component = DashboardComponent("c1", "chart", {"color": "red"})
    assert component.to_dict() == {
        "id": "c1",
        "type": "chart",
        "config": {"color": "red"},
        "data": {},
        "last_update": None,
    }

    component.update_data({"x": [1, 2]})
    exported = component.to_dict()
    assert exported["data"] == {"x": [1, 2]}
    assert isinstance(component.last_update, datetime)
    assert exported["last_update"] == component.last_update.isoformat()


class TestRealTimeMonitor:
    def test_defaults(self):
        monitor = RealTimeMonitor("m", ["loss"])
        assert monitor.type == "realtime_monitor"
        assert monitor.config == {}
        assert monitor.history_length == 100
        assert monitor.history == {"loss": []}

    def test_update_metric_records_statistics(self):
        monitor = RealTimeMonitor("m", ["loss", "acc"])
        for value in (3.0, 1.0, 2.0):
            monitor.update_metric("loss", value)

        assert monitor.history["loss"] == [3.0, 1.0, 2.0]
        assert monitor.history["acc"] == []
        assert monitor.data["loss"] == {
            "current": 2.0,
            "history": [3.0, 1.0, 2.0],
            "min": 1.0,
            "max": 3.0,
            "mean": 2.0,
        }
        assert "acc" not in monitor.data
        assert monitor.last_update is not None

    def test_history_is_trimmed_to_configured_length(self):
        monitor = RealTimeMonitor("m", ["loss"], {"history_length": 2})
        assert monitor.config == {"history_length": 2}
        for value in (1.0, 2.0, 3.0):
            monitor.update_metric("loss", value)
        assert monitor.history["loss"] == [2.0, 3.0]
        assert monitor.data["loss"]["min"] == 2.0
        assert monitor.data["loss"]["mean"] == 2.5

    def test_untracked_metric_is_ignored(self, caplog):
        monitor = RealTimeMonitor("m", ["loss"])
        with caplog.at_level(logging.WARNING):
            monitor.update_metric("other", 1.0)
        assert monitor.data == {}
        assert monitor.history == {"loss": []}
        assert monitor.last_update is None
        assert "Metric other not tracked" in caplog.text

    def test_batch_update_data(self):
        monitor = RealTimeMonitor("m", ["loss", "acc"])
        monitor.update_data({"loss": {"value": 0.5}, "acc": {}, "ignored": {"v": 1}})

        assert monitor.history == {"loss": [0.5], "acc": [0.0]}
        assert monitor.data["loss"]["current"] == 0.5
        assert monitor.data["acc"]["current"] == 0.0
        assert "ignored" not in monitor.data
        assert monitor.last_update is not None


class TestDashboard:
    def test_lifecycle_state_and_export(self):
        config = DashboardConfig(title="Main", refresh_interval=250, layout="flex")
        dashboard = Dashboard(config)
        component = DashboardComponent("c1", "chart", {"k": 1})
        dashboard.add_component(component)

        def source():
            return {"v": 1}

        dashboard.add_data_source("src", source)
        assert dashboard.data_sources == {"src": source}
        assert dashboard.is_running is False

        dashboard.start_monitoring()
        state = dashboard.get_dashboard_state()
        assert state["is_running"] is True
        assert state["config"] == {
            "title": "Main",
            "refresh_interval": 250,
            "layout": "flex",
            "components": [],
        }
        assert state["components"] == {"c1": component.to_dict()}
        assert isinstance(datetime.fromisoformat(state["timestamp"]), datetime)

        dashboard.stop_monitoring()
        assert dashboard.get_dashboard_state()["is_running"] is False

        assert json.loads(dashboard.export_config()) == {
            "title": "Main",
            "refresh_interval": 250,
            "layout": "flex",
            "components": [{"id": "c1", "type": "chart", "config": {"k": 1}}],
        }

    def test_add_component_replaces_same_id(self):
        dashboard = Dashboard(DashboardConfig(title="T"))
        dashboard.add_component(DashboardComponent("c", "a", {}))
        replacement = DashboardComponent("c", "b", {})
        dashboard.add_component(replacement)
        assert dashboard.components == {"c": replacement}


class TestActiveInferenceMonitor:
    def test_initial_state(self):
        monitor = ActiveInferenceMonitor("ai", {"history_length": 5})
        assert monitor.history_length == 5
        assert "free_energy" in monitor.metrics
        assert len(monitor.metrics) == 8
        assert monitor.ai_state == {
            "current_phase": "perception",
            "belief_precision": 1.0,
            "prediction_confidence": 0.0,
            "action_selection": None,
        }

    def test_update_ai_state_full_and_partial(self):
        monitor = ActiveInferenceMonitor("ai")
        monitor.update_ai_state(
            "action", belief_precision=2.0, prediction_confidence=0.7, action="left"
        )
        assert monitor.data["ai_state"] == {
            "current_phase": "action",
            "belief_precision": 2.0,
            "prediction_confidence": 0.7,
            "action_selection": "left",
        }
        assert monitor.last_update is not None

        monitor.update_ai_state("learning")
        assert monitor.ai_state["current_phase"] == "learning"
        assert monitor.ai_state["belief_precision"] == 2.0
        assert monitor.ai_state["action_selection"] == "left"
        # data holds a snapshot, not the live state dict
        assert monitor.data["ai_state"] is not monitor.ai_state

    def test_update_ai_state_preserves_metric_history(self):
        monitor = ActiveInferenceMonitor("ai")
        monitor.update_metric("free_energy", 1.5)
        monitor.update_ai_state("action")
        assert monitor.history["free_energy"] == [1.5]
        assert monitor.data["free_energy"]["current"] == 1.5
        assert monitor.data["ai_state"]["current_phase"] == "action"

    def test_information_metrics_empty_by_default(self):
        assert (
            ActiveInferenceMonitor("ai").compute_information_theoretic_metrics() == {}
        )

    def test_information_metrics_from_histories(self):
        monitor = ActiveInferenceMonitor("ai")
        monitor.history["belief_history"] = [[1.0, 0.0], [0.5, 0.5, 0.0]]
        monitor.history["belief_entropy_history"] = [1.2, 0.7]

        metrics = monitor.compute_information_theoretic_metrics()
        assert metrics["belief_entropy"] == pytest.approx(math.log(2), abs=1e-8)
        assert metrics["information_gain"] == pytest.approx(0.5)

    def test_information_metrics_skip_non_list_beliefs_and_short_history(self):
        monitor = ActiveInferenceMonitor("ai")
        monitor.history["belief_history"] = ["not-a-list"]
        monitor.history["belief_entropy_history"] = [1.0]
        assert monitor.compute_information_theoretic_metrics() == {}
