"""
Research Framework Module

Comprehensive research tools and experiment management for Active Inference
and the Free Energy Principle. Provides reproducible research pipelines,
simulation engines, analysis tools, and benchmarking frameworks.
"""

from .experiments import ResearchFramework, ExperimentManager, ExperimentConfig
from .simulations import SimulationEngine, ModelRunner
from .analysis import AnalysisTools, StatisticalAnalysis
from .benchmarks import BenchmarkSuite, PerformanceMetrics

__all__ = [
    "ResearchFramework",
    "ExperimentManager",
    "ExperimentConfig",
    "SimulationEngine",
    "ModelRunner",
    "AnalysisTools",
    "StatisticalAnalysis",
    "BenchmarkSuite",
    "PerformanceMetrics",
]
