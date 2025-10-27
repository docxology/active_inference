"""
Knowledge Repository

Core class for managing the structured knowledge environment for Active Inference
and the Free Energy Principle. Provides unified access to educational content,
learning paths, and interactive components.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum

import yaml
from pydantic import BaseModel, Field, ValidationError

logger = logging.getLogger(__name__)


class ContentType(Enum):
    """Types of educational content"""
    FOUNDATION = "foundation"
    MATHEMATICS = "mathematics"
    IMPLEMENTATION = "implementation"
    APPLICATION = "application"
    TUTORIAL = "tutorial"
    PAPER = "paper"
    EXERCISE = "exercise"
    SIMULATION = "simulation"


class DifficultyLevel(Enum):
    """Learning difficulty levels"""
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"


class KnowledgeNode(BaseModel):
    """Represents a single knowledge node in the repository"""
    id: str
    title: str
    content_type: ContentType
    difficulty: DifficultyLevel
    description: str
    prerequisites: List[str] = Field(default_factory=list)
    content_path: Optional[str] = None
    content: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    tags: List[str] = Field(default_factory=list)
    learning_objectives: List[str] = Field(default_factory=list)


class LearningPath(BaseModel):
    """Represents a structured learning path"""
    id: str
    name: str
    description: str
    nodes: List[str] = Field(default_factory=list)
    estimated_hours: Optional[int] = None
    difficulty: DifficultyLevel = DifficultyLevel.BEGINNER
    metadata: Dict[str, Any] = Field(default_factory=dict)


class KnowledgeNodeSchema:
    """JSON schema validation for knowledge nodes"""

    REQUIRED_FIELDS = {
        "id", "title", "content_type", "difficulty", "description",
        "prerequisites", "tags", "learning_objectives", "content", "metadata"
    }

    FIELD_TYPES = {
        "id": str,
        "title": str,
        "content_type": str,
        "difficulty": str,
        "description": str,
        "prerequisites": list,
        "tags": list,
        "learning_objectives": list,
        "content": dict,
        "metadata": dict
    }

    CONTENT_TYPE_VALUES = {"foundation", "mathematics", "implementation", "application"}
    DIFFICULTY_VALUES = {"beginner", "intermediate", "advanced", "expert"}

    @classmethod
    def validate_json_structure(cls, data: Dict[str, Any]) -> List[str]:
        """Validate JSON structure against schema"""
        errors = []

        # Check required fields
        missing_fields = cls.REQUIRED_FIELDS - set(data.keys())
        if missing_fields:
            errors.append(f"Missing required fields: {missing_fields}")

        # Check field types
        for field, expected_type in cls.FIELD_TYPES.items():
            if field in data:
                if not isinstance(data[field], expected_type):
                    errors.append(f"Field '{field}' must be of type {expected_type.__name__}")

        # Check content_type values
        if "content_type" in data:
            if data["content_type"] not in cls.CONTENT_TYPE_VALUES:
                errors.append(f"content_type must be one of: {cls.CONTENT_TYPE_VALUES}")

        # Check difficulty values
        if "difficulty" in data:
            if data["difficulty"] not in cls.DIFFICULTY_VALUES:
                errors.append(f"difficulty must be one of: {cls.DIFFICULTY_VALUES}")

        # Check that lists contain strings
        for list_field in ["prerequisites", "tags", "learning_objectives"]:
            if list_field in data and isinstance(data[list_field], list):
                non_string_items = [item for item in data[list_field] if not isinstance(item, str)]
                if non_string_items:
                    errors.append(f"Field '{list_field}' must contain only strings")

        return errors


@dataclass
class KnowledgeRepositoryConfig:
    """Configuration for the knowledge repository"""
    root_path: Path
    content_paths: Dict[ContentType, str] = field(default_factory=dict)
    metadata_path: str = "metadata"
    index_path: str = "index"
    cache_enabled: bool = True
    auto_index: bool = True


class KnowledgeRepository:
    """
    Central repository for Active Inference educational content and learning paths.

    Provides structured access to knowledge with search, filtering, and learning
    path generation capabilities.
    """

    def __init__(self, config: KnowledgeRepositoryConfig):
        self.config = config
        self.root_path = config.root_path
        self.content_paths = config.content_paths or self._default_content_paths()

        # Knowledge storage
        self._nodes: Dict[str, KnowledgeNode] = {}
        self._paths: Dict[str, LearningPath] = {}
        self._index: Dict[str, List[str]] = {}
        self._metadata: Dict[str, Any] = {}

        # Load knowledge base
        self._load_knowledge_base()

        if config.auto_index:
            self._build_index()

        logger.info(f"Knowledge repository initialized with {len(self._nodes)} nodes")

    def _default_content_paths(self) -> Dict[ContentType, str]:
        """Default content paths for different content types"""
        return {
            ContentType.FOUNDATION: "foundations",
            ContentType.MATHEMATICS: "mathematics",
            ContentType.IMPLEMENTATION: "implementations",
            ContentType.APPLICATION: "applications",
            ContentType.TUTORIAL: "tutorials",
            ContentType.PAPER: "papers",
            ContentType.EXERCISE: "exercises",
            ContentType.SIMULATION: "simulations",
        }

    def _load_knowledge_base(self) -> None:
        """Load all knowledge nodes and learning paths from disk"""
        # Load metadata
        metadata_file = self.root_path / self.config.metadata_path / "repository.json"
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                self._metadata = json.load(f)

        # Load knowledge nodes
        for content_type in ContentType:
            content_path = self.content_paths.get(content_type)
            if content_path:
                self._load_content_type(content_type, content_path)

        # Load learning paths
        paths_file = self.root_path / self.config.metadata_path / "learning_paths.json"
        if paths_file.exists():
            with open(paths_file, 'r') as f:
                paths_data = json.load(f)
                for path_data in paths_data:
                    path = LearningPath(**path_data)
                    self._paths[path.id] = path

    def _load_content_type(self, content_type: ContentType, content_path: str) -> None:
        """Load knowledge nodes for a specific content type"""
        full_path = self.root_path / content_path

        if not full_path.exists():
            logger.warning(f"Content path {full_path} does not exist")
            return

        # Look for knowledge definition files
        for knowledge_file in full_path.rglob("*.json"):
            try:
                with open(knowledge_file, 'r') as f:
                    node_data = json.load(f)
                    node = KnowledgeNode(**node_data)
                    self._nodes[node.id] = node
            except Exception as e:
                logger.error(f"Error loading knowledge node {knowledge_file}: {e}")

    def _build_index(self) -> None:
        """Build search index for knowledge nodes"""
        self._index = {
            'tags': {},
            'difficulty': {},
            'content_type': {},
            'prerequisites': {},
        }

        for node_id, node in self._nodes.items():
            # Index by tags
            for tag in node.tags:
                if tag not in self._index['tags']:
                    self._index['tags'][tag] = []
                self._index['tags'][tag].append(node_id)

            # Index by difficulty
            difficulty = node.difficulty.value
            if difficulty not in self._index['difficulty']:
                self._index['difficulty'][difficulty] = []
            self._index['difficulty'][difficulty].append(node_id)

            # Index by content type
            content_type = node.content_type.value
            if content_type not in self._index['content_type']:
                self._index['content_type'][content_type] = []
            self._index['content_type'][content_type].append(node_id)

            # Index by prerequisites
            for prereq in node.prerequisites:
                if prereq not in self._index['prerequisites']:
                    self._index['prerequisites'][prereq] = []
                self._index['prerequisites'][prereq].append(node_id)

    def search(self,
               query: str = "",
               content_types: Optional[List[ContentType]] = None,
               difficulty: Optional[List[DifficultyLevel]] = None,
               tags: Optional[List[str]] = None,
               limit: int = 50) -> List[KnowledgeNode]:
        """
        Search knowledge nodes with various filters

        Args:
            query: Text search query
            content_types: Filter by content types
            difficulty: Filter by difficulty levels
            tags: Filter by tags
            limit: Maximum number of results

        Returns:
            List of matching knowledge nodes
        """
        results = []

        for node_id, node in self._nodes.items():
            # Apply filters
            if content_types and node.content_type not in content_types:
                continue

            if difficulty and node.difficulty not in difficulty:
                continue

            if tags and not any(tag in node.tags for tag in tags):
                continue

            # Simple text matching (could be enhanced with full-text search)
            if query.lower() in node.title.lower() or query.lower() in node.description.lower():
                results.append(node)

            if len(results) >= limit:
                break

        return results

    def get_node(self, node_id: str) -> Optional[KnowledgeNode]:
        """Get a specific knowledge node by ID"""
        return self._nodes.get(node_id)

    def get_learning_path(self, path_id: str) -> Optional[LearningPath]:
        """Get a specific learning path by ID"""
        return self._paths.get(path_id)

    def get_learning_paths(self,
                          difficulty: Optional[DifficultyLevel] = None,
                          content_types: Optional[List[ContentType]] = None) -> List[LearningPath]:
        """Get learning paths with optional filters"""
        paths = list(self._paths.values())

        if difficulty:
            paths = [p for p in paths if p.difficulty == difficulty]

        if content_types:
            # Filter paths that contain nodes of specified content types
            filtered_paths = []
            for path in paths:
                path_nodes = [self.get_node(nid) for nid in path.nodes if nid in self._nodes]
                if any(node.content_type in content_types for node in path_nodes):
                    filtered_paths.append(path)
            paths = filtered_paths

        return paths

    def get_prerequisites_graph(self, node_id: str) -> Dict[str, Any]:
        """Get prerequisite dependency graph for a node"""
        node = self.get_node(node_id)
        if not node:
            return {}

        graph = {'nodes': [], 'edges': []}
        visited = set()

        def add_node_and_prereqs(nid: str):
            if nid in visited:
                return
            visited.add(nid)

            node = self.get_node(nid)
            if node:
                graph['nodes'].append({
                    'id': nid,
                    'label': node.title,
                    'type': node.content_type.value,
                    'difficulty': node.difficulty.value
                })

                for prereq in node.prerequisites:
                    graph['edges'].append({'from': prereq, 'to': nid})
                    add_node_and_prereqs(prereq)

        add_node_and_prereqs(node_id)
        return graph

    def validate_learning_path(self, path_id: str) -> Dict[str, Any]:
        """
        Validate a learning path for completeness and prerequisite satisfaction

        Returns:
            Validation report with issues and suggestions
        """
        path = self.get_learning_path(path_id)
        if not path:
            return {'valid': False, 'error': 'Path not found'}

        issues = []
        completed_prereqs = set()

        for node_id in path.nodes:
            node = self.get_node(node_id)
            if not node:
                issues.append(f"Missing node: {node_id}")
                continue

            # Check prerequisites
            for prereq_id in node.prerequisites:
                if prereq_id not in completed_prereqs:
                    # Check if prerequisite is earlier in the path
                    if prereq_id not in path.nodes or path.nodes.index(prereq_id) > path.nodes.index(node_id):
                        issues.append(f"Unsatisfied prerequisite {prereq_id} for {node_id}")

            completed_prereqs.add(node_id)

        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'path_length': len(path.nodes),
            'estimated_hours': path.estimated_hours
        }

    def export_knowledge_graph(self, format: str = 'json') -> Union[str, Dict[str, Any]]:
        """
        Export the knowledge repository as a graph structure

        Args:
            format: Export format ('json', 'graphml', 'gexf')

        Returns:
            Knowledge graph in specified format
        """
        graph = {
            'nodes': [],
            'edges': [],
            'metadata': {
                'total_nodes': len(self._nodes),
                'total_paths': len(self._paths),
                'content_types': list(set(node.content_type.value for node in self._nodes.values())),
                'difficulties': list(set(node.difficulty.value for node in self._nodes.values())),
            }
        }

        # Add nodes
        for node_id, node in self._nodes.items():
            graph['nodes'].append({
                'id': node_id,
                'title': node.title,
                'content_type': node.content_type.value,
                'difficulty': node.difficulty.value,
                'description': node.description,
                'tags': node.tags,
                'learning_objectives': node.learning_objectives,
            })

        # Add prerequisite edges
        for node_id, node in self._nodes.items():
            for prereq_id in node.prerequisites:
                graph['edges'].append({
                    'source': prereq_id,
                    'target': node_id,
                    'type': 'prerequisite'
                })

        # Add learning path edges
        for path in self._paths.values():
            for i in range(len(path.nodes) - 1):
                graph['edges'].append({
                    'source': path.nodes[i],
                    'target': path.nodes[i + 1],
                    'type': 'learning_path',
                    'path_id': path.id
                })

        if format == 'json':
            return graph
        else:
            # Could implement other formats like GraphML, GEXF
            return json.dumps(graph, indent=2)

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the knowledge repository"""
        content_type_counts = {}
        difficulty_counts = {}
        tag_counts = {}

        for node in self._nodes.values():
            # Count content types
            ct = node.content_type.value
            content_type_counts[ct] = content_type_counts.get(ct, 0) + 1

            # Count difficulties
            diff = node.difficulty.value
            difficulty_counts[diff] = difficulty_counts.get(diff, 0) + 1

        # Count tags
        for node in self._nodes.values():
            for tag in node.tags:
                tag_counts[tag] = tag_counts.get(tag, 0) + 1

        return {
            'total_nodes': len(self._nodes),
            'total_paths': len(self._paths),
            'content_types': content_type_counts,
            'difficulties': difficulty_counts,
            'top_tags': sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)[:20],
            'total_tags': len(tag_counts),
            'metadata': self._metadata,
        }
