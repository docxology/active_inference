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

        # Validate content consistency
        validation_report = self._validate_content_consistency()
        if not validation_report["valid"]:
            logger.warning(f"Content validation issues found: {validation_report}")

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
                    # Convert string difficulty to enum
                    if "difficulty" in path_data:
                        path_data["difficulty"] = DifficultyLevel(path_data["difficulty"])
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

                # Validate JSON structure
                validation_errors = KnowledgeNodeSchema.validate_json_structure(node_data)
                if validation_errors:
                    logger.error(f"JSON validation errors in {knowledge_file}: {validation_errors}")
                    continue

                # Convert string enums to enum objects
                node_data["content_type"] = ContentType(node_data["content_type"])
                node_data["difficulty"] = DifficultyLevel(node_data["difficulty"])

                # Set content_path for reference
                node_data["content_path"] = str(knowledge_file.relative_to(self.root_path))

                # Create knowledge node
                node = KnowledgeNode(**node_data)
                self._nodes[node.id] = node

                logger.debug(f"Successfully loaded knowledge node: {node.id}")

            except ValidationError as e:
                logger.error(f"Pydantic validation error loading knowledge node {knowledge_file}: {e}")
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

    def _validate_content_consistency(self) -> Dict[str, Any]:
        """Validate content consistency and report issues"""
        validation_report = {
            "valid": True,
            "missing_prerequisites": [],
            "orphaned_nodes": [],
            "circular_dependencies": [],
            "validation_errors": []
        }

        # Check that all prerequisites exist
        all_node_ids = set(self._nodes.keys())
        for node_id, node in self._nodes.items():
            for prereq_id in node.prerequisites:
                if prereq_id not in all_node_ids:
                    validation_report["missing_prerequisites"].append({
                        "node": node_id,
                        "missing_prerequisite": prereq_id
                    })
                    validation_report["valid"] = False

        # Check for circular dependencies
        for node_id in self._nodes:
            if self._has_circular_dependency(node_id, set()):
                validation_report["circular_dependencies"].append(node_id)
                validation_report["valid"] = False

        # Check for orphaned nodes (nodes that are not prerequisites for anything)
        referenced_nodes = set()
        for node in self._nodes.values():
            referenced_nodes.update(node.prerequisites)
        validation_report["orphaned_nodes"] = [
            node_id for node_id in self._nodes.keys()
            if node_id not in referenced_nodes
        ]

        return validation_report

    def _has_circular_dependency(self, node_id: str, visited: set) -> bool:
        """Check if node has circular dependency"""
        if node_id in visited:
            return True

        node = self.get_node(node_id)
        if not node:
            return False

        visited.add(node_id)
        for prereq_id in node.prerequisites:
            if self._has_circular_dependency(prereq_id, visited.copy()):
                return True

        return False

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

    def get_node_content(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Get the detailed content of a knowledge node"""
        node = self.get_node(node_id)
        if node:
            return node.content
        return None

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
                'content': node.content,
                'tags': node.tags,
                'learning_objectives': node.learning_objectives,
                'content_path': node.content_path,
                'metadata': node.metadata,
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

    def build_prerequisite_graph(self, node_id: str) -> Dict[str, Any]:
        """
        Build complete prerequisite dependency graph with cycle detection

        Args:
            node_id: Starting node ID

        Returns:
            Dictionary containing graph structure and validation
        """
        if node_id not in self._nodes:
            return {'error': f'Node {node_id} not found'}

        graph = {
            'nodes': [],
            'edges': [],
            'cycles': [],
            'depth': 0
        }

        visited = set()
        path = []

        def dfs(current_id, depth=0):
            if current_id in path:
                # Cycle detected
                cycle_start = path.index(current_id)
                cycle = path[cycle_start:] + [current_id]
                graph['cycles'].append(cycle)
                return

            if current_id in visited:
                return

            visited.add(current_id)
            path.append(current_id)
            graph['depth'] = max(graph['depth'], depth)

            # Add node to graph
            node = self._nodes[current_id]
            graph['nodes'].append({
                'id': current_id,
                'title': node.title,
                'depth': depth
            })

            # Visit prerequisites
            for prereq_id in node.prerequisites:
                graph['edges'].append({
                    'source': prereq_id,
                    'target': current_id,
                    'type': 'prerequisite'
                })
                dfs(prereq_id, depth + 1)

            path.pop()

        dfs(node_id)

        return graph

    def validate_content_integrity(self, node_id: str) -> Dict[str, Any]:
        """
        Validate content integrity including references and prerequisites

        Args:
            node_id: Node ID to validate

        Returns:
            Validation results dictionary
        """
        if node_id not in self._nodes:
            return {'valid': False, 'error': f'Node {node_id} not found'}

        node = self._nodes[node_id]
        validation = {
            'valid': True,
            'issues': [],
            'warnings': [],
            'suggestions': []
        }

        # Check prerequisites exist
        for prereq_id in node.prerequisites:
            if prereq_id not in self._nodes:
                validation['issues'].append(f'Missing prerequisite: {prereq_id}')
                validation['valid'] = False

        # Check content completeness
        if not node.content:
            validation['issues'].append('Empty content')
            validation['valid'] = False

        if not node.learning_objectives:
            validation['warnings'].append('No learning objectives defined')

        if not node.tags:
            validation['suggestions'].append('Consider adding tags for better discoverability')

        # Check for circular dependencies
        graph = self.build_prerequisite_graph(node_id)
        if graph.get('cycles'):
            validation['issues'].append(f'Circular dependencies detected: {graph["cycles"]}')
            validation['valid'] = False

        return validation

    def generate_learning_path_recommendations(self, user_profile: Dict[str, Any]) -> List[str]:
        """
        Generate personalized learning path recommendations

        Args:
            user_profile: User profile with preferences and background

        Returns:
            List of recommended learning path IDs
        """
        recommendations = []
        user_difficulty = user_profile.get('difficulty', 'intermediate')
        user_interests = user_profile.get('interests', [])
        user_background = user_profile.get('background', [])

        # Filter paths by difficulty
        difficulty_paths = [
            path_id for path_id, path in self._paths.items()
            if path.difficulty.value == user_difficulty
        ]

        # Score paths based on user interests
        scored_paths = []
        for path_id in difficulty_paths:
            path = self._paths[path_id]
            score = 0

            # Check if path covers user interests
            path_tags = set()
            for node_id in path.nodes:
                if node_id in self._nodes:
                    path_tags.update(self._nodes[node_id].tags)

            interest_overlap = len(path_tags.intersection(set(user_interests)))
            score += interest_overlap * 2

            # Check if path builds on user background
            background_overlap = len(set(path.nodes).intersection(set(user_background)))
            score += background_overlap

            scored_paths.append((path_id, score))

        # Sort by score and return top recommendations
        scored_paths.sort(key=lambda x: x[1], reverse=True)
        recommendations = [path_id for path_id, score in scored_paths[:5]]

        return recommendations

    def export_knowledge_in_format(self, format_type: str, content_filter: Dict[str, Any]) -> Any:
        """
        Export knowledge content in various formats (JSON, XML, RDF, etc.)

        Args:
            format_type: Export format ('json', 'xml', 'rdf', 'csv')
            content_filter: Filter criteria for content selection

        Returns:
            Exported content in specified format
        """
        # Apply filters
        filtered_nodes = self._filter_nodes(content_filter)

        if format_type == 'json':
            return self._export_json(filtered_nodes)
        elif format_type == 'xml':
            return self._export_xml(filtered_nodes)
        elif format_type == 'rdf':
            return self._export_rdf(filtered_nodes)
        elif format_type == 'csv':
            return self._export_csv(filtered_nodes)
        else:
            raise ValueError(f'Unsupported export format: {format_type}')

    def _filter_nodes(self, content_filter: Dict[str, Any]) -> Dict[str, KnowledgeNode]:
        """Filter nodes based on criteria"""
        filtered = dict(self._nodes)

        if 'content_types' in content_filter:
            content_types = set(content_filter['content_types'])
            filtered = {k: v for k, v in filtered.items()
                       if v.content_type.value in content_types}

        if 'difficulties' in content_filter:
            difficulties = set(content_filter['difficulties'])
            filtered = {k: v for k, v in filtered.items()
                       if v.difficulty.value in difficulties}

        if 'tags' in content_filter:
            required_tags = set(content_filter['tags'])
            filtered = {k: v for k, v in filtered.items()
                       if required_tags.issubset(set(v.tags))}

        return filtered

    def _export_json(self, nodes: Dict[str, KnowledgeNode]) -> str:
        """Export nodes as JSON"""
        import json
        data = {
            'nodes': [node.dict() for node in nodes.values()],
            'metadata': {
                'export_time': str(self._get_timestamp()),
                'total_nodes': len(nodes),
                'format': 'json'
            }
        }
        return json.dumps(data, indent=2, default=str)

    def _export_xml(self, nodes: Dict[str, KnowledgeNode]) -> str:
        """Export nodes as XML"""
        xml_parts = ['<?xml version="1.0" encoding="UTF-8"?>']
        xml_parts.append('<knowledge_base>')

        for node in nodes.values():
            xml_parts.append(f'  <node id="{node.id}">')
            xml_parts.append(f'    <title>{node.title}</title>')
            xml_parts.append(f'    <content_type>{node.content_type.value}</content_type>')
            xml_parts.append(f'    <difficulty>{node.difficulty.value}</difficulty>')
            xml_parts.append(f'    <description>{node.description}</description>')
            xml_parts.append('    <tags>')
            for tag in node.tags:
                xml_parts.append(f'      <tag>{tag}</tag>')
            xml_parts.append('    </tags>')
            xml_parts.append('  </node>')

        xml_parts.append('</knowledge_base>')
        return '\n'.join(xml_parts)

    def _export_rdf(self, nodes: Dict[str, KnowledgeNode]) -> str:
        """Export nodes as RDF/Turtle"""
        rdf_parts = ['@prefix kb: <http://activeinference.org/knowledge#> .']
        rdf_parts.append('@prefix dct: <http://purl.org/dc/terms/> .')
        rdf_parts.append('')

        for node in nodes.values():
            rdf_parts.append(f'kb:{node.id} a kb:KnowledgeNode ;')
            rdf_parts.append(f'  dct:title "{node.title}" ;')
            rdf_parts.append(f'  kb:contentType "{node.content_type.value}" ;')
            rdf_parts.append(f'  kb:difficulty "{node.difficulty.value}" ;')
            rdf_parts.append(f'  dct:description "{node.description}" ;')
            rdf_parts.append(f'  kb:prerequisites {list(node.prerequisites)} ;')
            rdf_parts.append(f'  kb:tags {node.tags} .')
            rdf_parts.append('')

        return '\n'.join(rdf_parts)

    def _export_csv(self, nodes: Dict[str, KnowledgeNode]) -> str:
        """Export nodes as CSV"""
        csv_lines = ['id,title,content_type,difficulty,description,tags,prerequisites']

        for node in nodes.values():
            tags_str = ';'.join(node.tags)
            prereqs_str = ';'.join(node.prerequisites)
            line = f'"{node.id}","{node.title}","{node.content_type.value}","{node.difficulty.value}","{node.description}","{tags_str}","{prereqs_str}"'
            csv_lines.append(line)

        return '\n'.join(csv_lines)

    def create_content_backup(self) -> Path:
        """
        Create comprehensive content backup with metadata and relationships

        Returns:
            Path to backup file
        """
        import tempfile
        import json
        from datetime import datetime

        # Create backup data
        backup_data = {
            'metadata': {
                'backup_time': datetime.now().isoformat(),
                'version': '1.0',
                'total_nodes': len(self._nodes),
                'total_paths': len(self._paths)
            },
            'nodes': {k: v.dict() for k, v in self._nodes.items()},
            'paths': {k: v.dict() for k, v in self._paths.items()},
            'relationships': self._extract_relationships()
        }

        # Save to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(backup_data, f, indent=2, default=str)
            backup_path = Path(f.name)

        logger.info(f'Created content backup: {backup_path}')
        return backup_path

    def _extract_relationships(self) -> Dict[str, List[str]]:
        """Extract all relationships between nodes"""
        relationships = {}

        for node_id, node in self._nodes.items():
            relationships[node_id] = {
                'prerequisites': node.prerequisites,
                'dependents': []
            }

        # Add dependents
        for node_id, node in self._nodes.items():
            for prereq_id in node.prerequisites:
                if prereq_id in relationships:
                    relationships[prereq_id]['dependents'].append(node_id)

        return relationships

    def _get_timestamp(self):
        """Get current timestamp"""
        from datetime import datetime
        return datetime.now()

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

    def validate_repository(self) -> Dict[str, Any]:
        """Get validation report for the entire repository"""
        return self._validate_content_consistency()
