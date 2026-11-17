"""
Platform Module

Core platform infrastructure for the Active Inference Knowledge Environment.
Provides knowledge graph management, intelligent search, collaboration features,
and deployment tools for scalable, multi-user environments.
"""

from .collaboration import CollaborationManager, UserManagement, WorkspaceManager
from .deployment import DeploymentManager, MonitoringTools, ServiceOrchestrator
from .knowledge_graph import KnowledgeGraphManager, Platform, SemanticEngine
from .search import IndexManager, QueryProcessor, SearchEngine

__all__ = [
    "Platform",
    "KnowledgeGraphManager",
    "SemanticEngine",
    "SearchEngine",
    "QueryProcessor",
    "IndexManager",
    "CollaborationManager",
    "UserManagement",
    "WorkspaceManager",
    "DeploymentManager",
    "ServiceOrchestrator",
    "MonitoringTools",
]
