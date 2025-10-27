"""
Active Inference Knowledge Environment

A comprehensive integrated platform for Active Inference & Free Energy Principle
education, research, visualization, and application development.

This package provides modular, well-documented components for learning, research,
and development in the Active Inference framework.
"""

from pathlib import Path

__version__ = "0.1.0"
__author__ = "Active Inference Community"
__email__ = "community@activeinference.org"

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Core module imports
from .knowledge import KnowledgeRepository
from .research import ResearchFramework
from .visualization import VisualizationEngine
from .applications import ApplicationFramework
from .tools import Orchestrator, Utilities
from .platform import Platform

__all__ = [
    "KnowledgeRepository",
    "ResearchFramework",
    "VisualizationEngine",
    "ApplicationFramework",
    "Orchestrator",
    "Utilities",
    "Platform",
    "PROJECT_ROOT",
]
