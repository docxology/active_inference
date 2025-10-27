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
from .intern_orchestrator import InternResearchOrchestrator
from .phd_orchestrator import PhDResearchOrchestrator
from .grant_orchestrator import GrantResearchOrchestrator
from .publication_orchestrator import PublicationResearchOrchestrator
from .hypothesis_orchestrator import HypothesisResearchOrchestrator
from .ideation_orchestrator import IdeationResearchOrchestrator
from .documentation_orchestrator import DocumentationResearchOrchestrator

__all__ = [
    "BaseOrchestrator",
    "OrchestratorConfig",
    "ResearchStage",
    "InternResearchOrchestrator",
    "PhDResearchOrchestrator",
    "GrantResearchOrchestrator",
    "PublicationResearchOrchestrator",
    "HypothesisResearchOrchestrator",
    "IdeationResearchOrchestrator",
    "DocumentationResearchOrchestrator",
]