"""
Active Inference Visualization Module

Comprehensive visualization system for Active Inference concepts, models, and processes.
Provides interactive diagrams, educational animations, real-time dashboards, and
comparative analysis tools for exploring Active Inference theory and applications.

This module serves as the main entry point for all visualization functionality in
the Active Inference Knowledge Environment.
"""

from .animations import (
    AnimationEngine,
    AnimationFrame,
    AnimationSequence,
    AnimationType,
    ProcessAnimation,
)
from .comparative import (
    ComparisonTool,
    ComparisonType,
    ModelComparator,
    ModelComparison,
    StatisticalComparator,
)
from .dashboards import (
    ActiveInferenceMonitor,
    Dashboard,
    DashboardComponent,
    DashboardConfig,
    RealTimeMonitor,
)
from .diagrams import (
    ConceptDiagram,
    DiagramEdge,
    DiagramNode,
    DiagramType,
    InteractiveDiagram,
    VisualizationEngine,
)

__version__ = "1.0.0"
__all__ = [
    # Animation system
    "AnimationEngine",
    "AnimationType",
    "AnimationFrame",
    "AnimationSequence",
    "ProcessAnimation",
    # Diagram system
    "VisualizationEngine",
    "DiagramType",
    "InteractiveDiagram",
    "DiagramNode",
    "DiagramEdge",
    "ConceptDiagram",
    # Dashboard system
    "Dashboard",
    "DashboardConfig",
    "DashboardComponent",
    "RealTimeMonitor",
    "ActiveInferenceMonitor",
    # Comparative analysis
    "ComparisonTool",
    "ModelComparator",
    "StatisticalComparator",
    "ComparisonType",
    "ModelComparison",
]
