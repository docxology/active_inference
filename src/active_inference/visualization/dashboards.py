"""
Visualization Engine - Interactive Dashboards

Real-time monitoring and dashboard system for Active Inference models and simulations.
Provides live data visualization, performance monitoring, and interactive exploration
of model dynamics and learning processes.
"""

import logging
import json
import numpy as np
from typing import Dict, List, Optional, Any, Callable
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class DashboardConfig:
    """Configuration for dashboard layout and components"""
    title: str
    refresh_interval: int = 1000  # milliseconds
    layout: str = "grid"  # grid, flex, custom
    components: List[Dict[str, Any]] = None

    def __post_init__(self):
        if self.components is None:
            self.components = []


class DashboardComponent:
    """Base class for dashboard components"""

    def __init__(self, component_id: str, component_type: str, config: Dict[str, Any]):
        self.id = component_id
        self.type = component_type
        self.config = config
        self.data: Dict[str, Any] = {}
        self.last_update: Optional[datetime] = None

    def update_data(self, data: Dict[str, Any]) -> None:
        """Update component data"""
        self.data = data
        self.last_update = datetime.now()
        logger.debug(f"Updated component {self.id} with new data")

    def to_dict(self) -> Dict[str, Any]:
        """Export component to dictionary"""
        return {
            "id": self.id,
            "type": self.type,
            "config": self.config,
            "data": self.data,
            "last_update": self.last_update.isoformat() if self.last_update else None
        }


class RealTimeMonitor(DashboardComponent):
    """Real-time monitoring component for model metrics"""

    def __init__(self, component_id: str, metrics: List[str], config: Dict[str, Any] = None):
        super().__init__(component_id, "realtime_monitor", config or {})
        self.metrics = metrics
        self.history_length = config.get("history_length", 100) if config else 100
        self.history: Dict[str, List[float]] = {metric: [] for metric in metrics}

    def update_metric(self, metric_name: str, value: float) -> None:
        """Update a specific metric"""
        if metric_name not in self.metrics:
            logger.warning(f"Metric {metric_name} not tracked by this monitor")
            return

        # Update history
        history = self.history[metric_name]
        history.append(value)

        # Maintain history length
        if len(history) > self.history_length:
            history.pop(0)

        # Update main data
        self.data[metric_name] = {
            "current": value,
            "history": history,
            "min": min(history) if history else value,
            "max": max(history) if history else value,
            "mean": sum(history) / len(history) if history else value
        }

        self.update_data(self.data)

    def update_data(self, data: Dict[str, Any]) -> None:
        """Override to handle batch updates"""
        for metric_name, metric_data in data.items():
            if metric_name in self.metrics:
                self.update_metric(metric_name, metric_data.get("value", 0.0))

        super().update_data(self.data)


class Dashboard:
    """Interactive dashboard for real-time visualization"""

    def __init__(self, config: DashboardConfig):
        self.config = config
        self.components: Dict[str, DashboardComponent] = {}
        self.data_sources: Dict[str, Callable] = {}
        self.is_running = False

        logger.info(f"Created dashboard: {config.title}")

    def add_component(self, component: DashboardComponent) -> None:
        """Add a component to the dashboard"""
        self.components[component.id] = component
        logger.debug(f"Added component {component.id} to dashboard")

    def add_data_source(self, source_id: str, data_function: Callable) -> None:
        """Add a data source function"""
        self.data_sources[source_id] = data_function
        logger.debug(f"Added data source {source_id}")

    def start_monitoring(self) -> None:
        """Start real-time monitoring"""
        self.is_running = True
        logger.info("Started dashboard monitoring")

        # Placeholder for actual monitoring loop
        # In a real implementation, this would run in a separate thread
        # and periodically update all components

    def stop_monitoring(self) -> None:
        """Stop real-time monitoring"""
        self.is_running = False
        logger.info("Stopped dashboard monitoring")

    def get_dashboard_state(self) -> Dict[str, Any]:
        """Get current state of the dashboard"""
        return {
            "config": {
                "title": self.config.title,
                "refresh_interval": self.config.refresh_interval,
                "layout": self.config.layout,
                "components": self.config.components
            },
            "components": {comp_id: comp.to_dict() for comp_id, comp in self.components.items()},
            "is_running": self.is_running,
            "timestamp": datetime.now().isoformat()
        }

    def export_config(self) -> str:
        """Export dashboard configuration as JSON"""
        return json.dumps({
            "title": self.config.title,
            "refresh_interval": self.config.refresh_interval,
            "layout": self.config.layout,
            "components": [
                {
                    "id": comp.id,
                    "type": comp.type,
                    "config": comp.config
                }
                for comp in self.components.values()
            ]
        }, indent=2)


class ActiveInferenceMonitor(RealTimeMonitor):
    """Specialized monitor for Active Inference metrics"""

    def __init__(self, component_id: str, config: Dict[str, Any] = None):
        ai_metrics = [
            "free_energy", "prediction_error", "belief_entropy",
            "expected_free_energy", "policy_entropy", "learning_rate",
            "model_confidence", "information_gain"
        ]
        super().__init__(component_id, ai_metrics, config)

        # Active Inference specific tracking
        self.ai_state = {
            "current_phase": "perception",
            "belief_precision": 1.0,
            "prediction_confidence": 0.0,
            "action_selection": None
        }

    def update_ai_state(self, phase: str, belief_precision: float = None,
                       prediction_confidence: float = None, action: str = None):
        """Update Active Inference specific state"""
        self.ai_state["current_phase"] = phase
        if belief_precision is not None:
            self.ai_state["belief_precision"] = belief_precision
        if prediction_confidence is not None:
            self.ai_state["prediction_confidence"] = prediction_confidence
        if action is not None:
            self.ai_state["action_selection"] = action

        # Update data with AI state
        self.data["ai_state"] = self.ai_state.copy()
        self.update_data(self.data)

    def compute_information_theoretic_metrics(self) -> Dict[str, float]:
        """Compute information-theoretic metrics from current state"""
        metrics = {}

        # Entropy of current beliefs (simplified)
        if "belief_history" in self.history and self.history["belief_history"]:
            beliefs = self.history["belief_history"][-1]  # Most recent beliefs
            if isinstance(beliefs, list) and len(beliefs) > 0:
                # Compute entropy of belief distribution
                belief_entropy = -sum(b * np.log(b + 1e-10) for b in beliefs if b > 0)
                metrics["belief_entropy"] = belief_entropy

        # Information gain (change in entropy)
        if "belief_entropy_history" in self.history and len(self.history["belief_entropy_history"]) > 1:
            recent_entropy = self.history["belief_entropy_history"][-1]
            previous_entropy = self.history["belief_entropy_history"][-2]
            metrics["information_gain"] = previous_entropy - recent_entropy

        return metrics


