"""
Research Orchestrators Module

Thin orchestration layer for coordinating research processes across different
roles and stages. Provides role-specific interfaces and workflows for
Active Inference research.

This module contains orchestrators for different research roles:
- Intern: Guided research workflows for beginners
- PhD Student: Advanced research methods and validation
- Grant Application: Power analysis and proposal development
- Publication: Publication-ready research and documentation
- Hypothesis: Hypothesis generation and testing
- Ideation: Brainstorming and idea development
- Documentation: Research documentation and reporting
"""

from .base_orchestrator import BaseOrchestrator, OrchestratorConfig, ResearchStage

__all__ = [
    "BaseOrchestrator",
    "OrchestratorConfig",
    "ResearchStage",
]