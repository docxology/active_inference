"""
Research Framework Module

Comprehensive research tools and experiment management for Active Inference
and the Free Energy Principle. Provides reproducible research pipelines,
simulation engines, analysis tools, and benchmarking frameworks.
"""

from .analysis import AnalysisTools, StatisticalAnalysis
from .benchmarks import BenchmarkSuite, PerformanceMetrics
from .data_management import (
    DataCollectionConfig,
    DataFormat,
    DataManager,
    DataMetadata,
    DataSecurityLevel,
)
from .experiments import ExperimentConfig, ExperimentManager, ResearchFramework
from .simulations import ModelRunner, SimulationEngine

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
    "DataManager",
    "DataMetadata",
    "DataCollectionConfig",
    "DataSecurityLevel",
    "DataFormat",
]
