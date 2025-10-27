"""
Visualization Engine - Interactive Dashboards

Real-time monitoring and dashboard system for Active Inference models and simulations.
Provides live data visualization, performance monitoring, and interactive exploration
of model dynamics and learning processes.
"""

import logging
import json
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
