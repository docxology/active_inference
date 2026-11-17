"""
Knowledge Repository Module

Structured educational content and learning paths for Active Inference
and the Free Energy Principle. Provides organized, accessible knowledge
with progressive learning tracks and interactive components.
"""

from .applications import Applications
from .foundations import Foundations
from .implementations import Implementations
from .mathematics import Mathematics
from .repository import (
    ContentType,
    DifficultyLevel,
    KnowledgeNode,
    KnowledgeNodeSchema,
    KnowledgeRepository,
    KnowledgeRepositoryConfig,
    LearningPath,
)

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
