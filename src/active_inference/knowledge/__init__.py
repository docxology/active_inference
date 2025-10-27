"""
Knowledge Repository Module

Structured educational content and learning paths for Active Inference
and the Free Energy Principle. Provides organized, accessible knowledge
with progressive learning tracks and interactive components.
"""

from .repository import KnowledgeRepository
from .foundations import Foundations
from .mathematics import Mathematics

__all__ = [
    "KnowledgeRepository",
    "Foundations",
    "Mathematics",
]
