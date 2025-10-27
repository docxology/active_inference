"""
Platform - Knowledge Graph Management

Semantic knowledge representation and management system for the Active Inference
Knowledge Environment. Provides graph-based organization of concepts, relationships,
and metadata with intelligent querying and navigation capabilities.
"""

import logging
from typing import Dict, List, Optional, Any, Set, Tuple
from pathlib import Path
from dataclasses import dataclass
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class KnowledgeNode:
    """Node in the knowledge graph"""
    id: str
    label: str
    node_type: str  # concept, theory, implementation, application
    content: Dict[str, Any]
    properties: Dict[str, Any] = None

    def __post_init__(self):
        if self.properties is None:
            self.properties = {}


@dataclass
class KnowledgeEdge:
    """Edge connecting knowledge nodes"""
    source: str
    target: str
    relation_type: str  # prerequisite, related_to, implements, extends
    weight: float = 1.0
    properties: Dict[str, Any] = None

    def __post_init__(self):
        if self.properties is None:
            self.properties = {}


class SemanticEngine:
    """Semantic processing and understanding"""

    def __init__(self):
        self.concept_embeddings: Dict[str, List[float]] = {}
        self.relation_embeddings: Dict[str, List[float]] = {}

        logger.info("SemanticEngine initialized")

    def compute_similarity(self, node1: KnowledgeNode, node2: KnowledgeNode) -> float:
        """Compute semantic similarity between two nodes"""
        # Placeholder for actual semantic similarity computation
        # In a real implementation, this would use embeddings or other semantic methods

        # Simple similarity based on shared properties
        props1 = set(node1.properties.keys())
        props2 = set(node2.properties.keys())

        if not props1 and not props2:
            return 0.0

        intersection = props1.intersection(props2)
        union = props1.union(props2)

        return len(intersection) / len(union) if union else 0.0

    def find_related_concepts(self, node: KnowledgeNode, max_results: int = 10) -> List[Tuple[KnowledgeNode, float]]:
        """Find concepts related to the given node"""
        # Placeholder implementation
        return []


class KnowledgeGraphManager:
    """Manages the knowledge graph structure and operations"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.nodes: Dict[str, KnowledgeNode] = {}
        self.edges: List[KnowledgeEdge] = []
        self.semantic_engine = SemanticEngine()

        # Index for efficient querying
        self._build_indices()

        logger.info("KnowledgeGraphManager initialized")

    def _build_indices(self) -> None:
        """Build indices for efficient querying"""
        self.node_type_index: Dict[str, List[str]] = defaultdict(list)
        self.relation_index: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))

    def add_node(self, node: KnowledgeNode) -> bool:
        """Add a node to the knowledge graph"""
        if node.id in self.nodes:
            logger.warning(f"Node {node.id} already exists")
            return False

        self.nodes[node.id] = node
        self.node_type_index[node.node_type].append(node.id)

        logger.debug(f"Added node {node.id} of type {node.node_type}")
        return True

    def add_edge(self, edge: KnowledgeEdge) -> bool:
        """Add an edge to the knowledge graph"""
        # Validate nodes exist
        if edge.source not in self.nodes or edge.target not in self.nodes:
            logger.error(f"Edge references non-existent nodes: {edge.source} -> {edge.target}")
            return False

        self.edges.append(edge)
        self.relation_index[edge.relation_type][edge.source].append(edge.target)

        logger.debug(f"Added edge {edge.source} --[{edge.relation_type}]--> {edge.target}")
        return True

    def get_node(self, node_id: str) -> Optional[KnowledgeNode]:
        """Retrieve a node by ID"""
        return self.nodes.get(node_id)

    def get_nodes_by_type(self, node_type: str) -> List[KnowledgeNode]:
        """Get all nodes of a specific type"""
        node_ids = self.node_type_index.get(node_type, [])
        return [self.nodes[node_id] for node_id in node_ids if node_id in self.nodes]

    def get_related_nodes(self, node_id: str, relation_type: Optional[str] = None) -> List[KnowledgeNode]:
        """Get nodes related to the given node"""
        if node_id not in self.nodes:
            return []

        related_nodes = []

        for edge in self.edges:
            if edge.source == node_id and (relation_type is None or edge.relation_type == relation_type):
                target_node = self.nodes.get(edge.target)
                if target_node:
                    related_nodes.append(target_node)

        return related_nodes

    def find_shortest_path(self, start_id: str, end_id: str) -> Optional[List[str]]:
        """Find shortest path between two nodes"""
        if start_id not in self.nodes or end_id not in self.nodes:
            return None

        # Simple breadth-first search
        from collections import deque

        queue = deque([(start_id, [start_id])])
        visited = {start_id}

        while queue:
            current_id, path = queue.popleft()

            if current_id == end_id:
                return path

            # Find neighbors
            for edge in self.edges:
                if edge.source == current_id and edge.target not in visited:
                    visited.add(edge.target)
                    new_path = path + [edge.target]
                    queue.append((edge.target, new_path))

        return None

    def compute_centrality(self, node_id: str) -> float:
        """Compute centrality measure for a node"""
        if node_id not in self.nodes:
            return 0.0

        # Simple degree centrality
        connections = 0

        for edge in self.edges:
            if edge.source == node_id or edge.target == node_id:
                connections += 1

        return connections / len(self.nodes) if self.nodes else 0.0

    def export_graph(self, format: str = "dict") -> Dict[str, Any]:
        """Export the knowledge graph in specified format"""
        if format == "dict":
            return {
                "nodes": {
                    node_id: {
                        "id": node.id,
                        "label": node.label,
                        "node_type": node.node_type,
                        "content": node.content,
                        "properties": node.properties
                    }
                    for node_id, node in self.nodes.items()
                },
                "edges": [
                    {
                        "source": edge.source,
                        "target": edge.target,
                        "relation_type": edge.relation_type,
                        "weight": edge.weight,
                        "properties": edge.properties
                    }
                    for edge in self.edges
                ],
                "metadata": {
                    "total_nodes": len(self.nodes),
                    "total_edges": len(self.edges),
                    "node_types": list(set(node.node_type for node in self.nodes.values())),
                    "relation_types": list(set(edge.relation_type for edge in self.edges))
                }
            }
        else:
            logger.warning(f"Export format not supported: {format}")
            return {}


class Platform:
    """Main platform class coordinating all platform services"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.knowledge_graph = KnowledgeGraphManager(config.get("knowledge_graph", {}))
        self.search_engine = None  # Placeholder
        self.collaboration_manager = None  # Placeholder

        logger.info("Platform initialized")

    def get_knowledge_graph(self) -> KnowledgeGraphManager:
        """Get the knowledge graph manager"""
        return self.knowledge_graph

    def build_knowledge_graph_from_repository(self, repository) -> int:
        """Build knowledge graph from knowledge repository"""
        logger.info("Building knowledge graph from repository")

        nodes_added = 0
        edges_added = 0

        # Add knowledge nodes from repository
        if hasattr(repository, '_nodes'):
            for node_id, node in repository._nodes.items():
                kg_node = KnowledgeNode(
                    id=node_id,
                    label=node.title,
                    node_type=node.content_type.value,
                    content={
                        "title": node.title,
                        "description": node.description,
                        "difficulty": node.difficulty.value,
                        "prerequisites": node.prerequisites,
                        "tags": node.tags,
                        "learning_objectives": node.learning_objectives
                    },
                    properties={
                        "difficulty": node.difficulty.value,
                        "estimated_time": getattr(node, 'estimated_time', 0),
                        "author": getattr(node, 'author', 'unknown')
                    }
                )

                if self.knowledge_graph.add_node(kg_node):
                    nodes_added += 1

        # Add prerequisite relationships
        if hasattr(repository, '_nodes'):
            for node_id, node in repository._nodes.items():
                for prereq_id in node.prerequisites:
                    if prereq_id in repository._nodes:
                        edge = KnowledgeEdge(
                            source=prereq_id,
                            target=node_id,
                            relation_type="prerequisite",
                            weight=1.0,
                            properties={"required": True}
                        )

                        if self.knowledge_graph.add_edge(edge):
                            edges_added += 1

        logger.info(f"Knowledge graph built: {nodes_added} nodes, {edges_added} edges")
        return nodes_added

    def get_platform_status(self) -> Dict[str, Any]:
        """Get platform status and health information"""
        return {
            "status": "operational",
            "services": {
                "knowledge_graph": {
                    "status": "active",
                    "nodes": len(self.knowledge_graph.nodes),
                    "edges": len(self.knowledge_graph.edges)
                },
                "search_engine": {
                    "status": "not_implemented",
                    "nodes": 0,
                    "edges": 0
                },
                "collaboration": {
                    "status": "not_implemented",
                    "active_users": 0
                }
            },
            "timestamp": "2024-10-27T12:00:00"
        }

