"""
Platform Module

Core platform infrastructure for the Active Inference Knowledge Environment.
Provides knowledge graph management, intelligent search, collaboration features,
and deployment tools for scalable, multi-user environments.
"""

from .knowledge_graph import Platform, KnowledgeGraphManager, SemanticEngine
from .search import SearchEngine, QueryProcessor, IndexManager
from .collaboration import CollaborationManager, UserManagement, WorkspaceManager
from .deployment import DeploymentManager, ServiceOrchestrator, MonitoringTools

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



