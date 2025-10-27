"""
Application Framework Module

Practical application development framework for Active Inference and Free Energy
Principle implementations. Provides templates, case studies, integration tools,
and architectural patterns for building real-world applications.
"""

from .templates import ApplicationFramework, TemplateManager, CodeGenerator
from .case_studies import CaseStudyManager, ExampleApplications
from .integrations import IntegrationManager, APIConnectors
from .best_practices import BestPracticesGuide, ArchitecturePatterns

__all__ = [
    "ApplicationFramework",
    "TemplateManager",
    "CodeGenerator",
    "CaseStudyManager",
    "ExampleApplications",
    "IntegrationManager",
    "APIConnectors",
    "BestPracticesGuide",
    "ArchitecturePatterns",
]
