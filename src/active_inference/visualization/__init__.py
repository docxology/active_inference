"""
Visualization Engine Module

Interactive visualization system for Active Inference concepts, models, and simulations.
Provides dynamic diagrams, real-time dashboards, educational animations, and
comparative analysis tools for exploring complex systems.
"""

from .diagrams import VisualizationEngine, ConceptDiagram, InteractiveDiagram
from .dashboards import Dashboard, RealTimeMonitor
from .animations import AnimationEngine, ProcessAnimation
from .comparative import ComparisonTool, ModelComparator

__all__ = [
    "VisualizationEngine",
    "ConceptDiagram",
    "InteractiveDiagram",
    "Dashboard",
    "RealTimeMonitor",
    "AnimationEngine",
    "ProcessAnimation",
    "ComparisonTool",
    "ModelComparator",
]

