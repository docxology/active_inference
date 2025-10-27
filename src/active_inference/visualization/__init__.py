"""
Active Inference Visualization Module

Comprehensive visualization system for Active Inference concepts, models, and processes.
Provides interactive diagrams, educational animations, real-time dashboards, and
comparative analysis tools for exploring Active Inference theory and applications.

This module serves as the main entry point for all visualization functionality in
the Active Inference Knowledge Environment.
"""

from .animations import AnimationEngine, AnimationType, AnimationFrame, AnimationSequence, ProcessAnimation
from .diagrams import VisualizationEngine, DiagramType, InteractiveDiagram, DiagramNode, DiagramEdge, ConceptDiagram
from .dashboards import Dashboard, DashboardConfig, DashboardComponent, RealTimeMonitor, ActiveInferenceMonitor
from .comparative import ComparisonTool, ModelComparator, StatisticalComparator, ComparisonType, ModelComparison

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
    "ModelComparison"
]