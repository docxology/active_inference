"""
Knowledge Repository Module

Structured educational content and learning paths for Active Inference
and the Free Energy Principle. Provides organized, accessible knowledge
with progressive learning tracks and interactive components.
"""

from .repository import (
    KnowledgeRepository,
    KnowledgeRepositoryConfig,
    KnowledgeNode,
    LearningPath,
    ContentType,
    DifficultyLevel,
    KnowledgeNodeSchema
)
from .foundations import Foundations
from .mathematics import Mathematics
from .implementations import Implementations
from .applications import Applications

__all__ = [
    "KnowledgeRepository",
    "KnowledgeRepositoryConfig",
    "KnowledgeNode",
    "LearningPath",
    "ContentType",
    "DifficultyLevel",
    "KnowledgeNodeSchema",
    "Foundations",
    "Mathematics",
    "Implementations",
    "Applications",
]
