"""
Application Framework Module

Practical application development framework for Active Inference and Free Energy
Principle implementations. Provides templates, case studies, integration tools,
and architectural patterns for building real-world applications.
"""

from .best_practices import ArchitecturePatterns, BestPracticesGuide
from .case_studies import CaseStudyManager, ExampleApplications
from .integrations import APIConnector, IntegrationManager
from .templates import ApplicationFramework, CodeGenerator, TemplateManager

__all__ = [
    "ApplicationFramework",
    "TemplateManager",
    "CodeGenerator",
    "CaseStudyManager",
    "ExampleApplications",
    "IntegrationManager",
    "APIConnector",
    "BestPracticesGuide",
    "ArchitecturePatterns",
]
