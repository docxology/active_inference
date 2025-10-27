"""
Tests for Knowledge Repository Module

Unit tests for the knowledge repository functionality, ensuring proper
operation of knowledge management, search, and learning path features.
"""

import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

from active_inference.knowledge.repository import (
    KnowledgeRepository,
    KnowledgeRepositoryConfig,
    KnowledgeNode,
    LearningPath,
    ContentType,
    DifficultyLevel
)


class TestKnowledgeRepository:
    """Test cases for KnowledgeRepository class"""

    @pytest.fixture
    def temp_knowledge_dir(self):
        """Create temporary directory for knowledge content"""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create metadata directory
            metadata_dir = temp_path / "metadata"
            metadata_dir.mkdir()

            # Create repository.json
            repo_data = {
                "version": "1.0",
                "description": "Test knowledge repository",
                "total_nodes": 0
            }
            (metadata_dir / "repository.json").write_text(json.dumps(repo_data))

            # Create learning paths
            paths_data = [
                {
                    "id": "test_path",
                    "name": "Test Learning Path",
                    "description": "A test learning path",
                    "nodes": ["node1", "node2"],
                    "estimated_hours": 4,
                    "difficulty": "beginner"
                }
            ]
            (metadata_dir / "learning_paths.json").write_text(json.dumps(paths_data))

            # Create content directories and files
            foundations_dir = temp_path / "foundations"
            foundations_dir.mkdir()

            node1_data = {
                "id": "node1",
                "title": "Test Node 1",
                "content_type": "foundation",
                "difficulty": "beginner",
                "description": "First test node",
                "prerequisites": [],
                "content": {
                    "overview": "Test overview content",
                    "examples": [{"name": "Example 1", "description": "Test example"}]
                },
                "tags": ["test", "beginner"],
                "learning_objectives": ["Learn something"],
                "metadata": {"version": "1.0"}
            }
            (foundations_dir / "node1.json").write_text(json.dumps(node1_data))

            node2_data = {
                "id": "node2",
                "title": "Test Node 2",
                "content_type": "foundation",
                "difficulty": "intermediate",
                "description": "Second test node",
                "prerequisites": ["node1"],
                "content": {
                    "overview": "Test overview content for node 2",
                    "mathematical_formulation": "Mathematical content here"
                },
                "tags": ["test", "intermediate"],
                "learning_objectives": ["Learn something else"],
                "metadata": {"version": "1.0"}
            }
            (foundations_dir / "node2.json").write_text(json.dumps(node2_data))

            yield temp_path

    @pytest.fixture
    def repository_config(self, temp_knowledge_dir):
        """Create repository configuration"""
        return KnowledgeRepositoryConfig(
            root_path=temp_knowledge_dir,
            auto_index=True,
            cache_enabled=False
        )

    def test_repository_initialization(self, repository_config):
        """Test repository initialization"""
        repo = KnowledgeRepository(repository_config)

        assert repo.config == repository_config
        assert repo.root_path == repository_config.root_path
        assert len(repo._nodes) > 0
        assert len(repo._paths) > 0

    def test_load_knowledge_nodes(self, repository_config):
        """Test loading of knowledge nodes from files"""
        repo = KnowledgeRepository(repository_config)

        # Check that nodes were loaded
        node1 = repo.get_node("node1")
        assert node1 is not None
        assert node1.title == "Test Node 1"
        assert node1.content_type == ContentType.FOUNDATION
        assert node1.difficulty == DifficultyLevel.BEGINNER

        node2 = repo.get_node("node2")
        assert node2 is not None
        assert node2.title == "Test Node 2"
        assert node2.prerequisites == ["node1"]

    def test_load_learning_paths(self, repository_config):
        """Test loading of learning paths"""
        repo = KnowledgeRepository(repository_config)

        path = repo.get_learning_path("test_path")
        assert path is not None
        assert path.name == "Test Learning Path"
        assert path.nodes == ["node1", "node2"]
        assert path.estimated_hours == 4

    def test_search_functionality(self, repository_config):
        """Test search functionality"""
        repo = KnowledgeRepository(repository_config)

        # Search for "test"
        results = repo.search("test")
        assert len(results) == 2  # Both nodes contain "test"

        # Search for specific title
        results = repo.search("Test Node 1")
        assert len(results) == 1
        assert results[0].title == "Test Node 1"

        # Search with limit
        results = repo.search("test", limit=1)
        assert len(results) == 1

    def test_filtering_by_content_type(self, repository_config):
        """Test filtering by content type"""
        repo = KnowledgeRepository(repository_config)

        # This would work if we had different content types
        # For now, all our test nodes are foundations
        results = repo.search("", content_types=[ContentType.FOUNDATION])
        assert len(results) == 2

    def test_filtering_by_difficulty(self, repository_config):
        """Test filtering by difficulty level"""
        repo = KnowledgeRepository(repository_config)

        beginner_results = repo.search("", difficulty=[DifficultyLevel.BEGINNER])
        assert len(beginner_results) == 1
        assert beginner_results[0].title == "Test Node 1"

        intermediate_results = repo.search("", difficulty=[DifficultyLevel.INTERMEDIATE])
        assert len(intermediate_results) == 1
        assert intermediate_results[0].title == "Test Node 2"

    def test_get_learning_paths(self, repository_config):
        """Test getting learning paths with filters"""
        repo = KnowledgeRepository(repository_config)

        # Get all paths
        all_paths = repo.get_learning_paths()
        assert len(all_paths) == 1

        # Get paths by difficulty
        beginner_paths = repo.get_learning_paths(difficulty=DifficultyLevel.BEGINNER)
        assert len(beginner_paths) == 1

    def test_prerequisites_graph(self, repository_config):
        """Test prerequisite graph generation"""
        repo = KnowledgeRepository(repository_config)

        graph = repo.get_prerequisites_graph("node2")

        assert "nodes" in graph
        assert "edges" in graph
        assert len(graph["nodes"]) > 0
        assert len(graph["edges"]) > 0

        # Check that node2 has an edge from node1
        edge_found = any(edge["from"] == "node1" and edge["to"] == "node2"
                        for edge in graph["edges"])
        assert edge_found

    def test_path_validation(self, repository_config):
        """Test learning path validation"""
        repo = KnowledgeRepository(repository_config)

        validation = repo.validate_learning_path("test_path")

        assert "valid" in validation
        assert "issues" in validation
        assert "path_length" in validation

        # Our test path should be valid
        assert validation["valid"] == True
        assert validation["path_length"] == 2
        assert len(validation["issues"]) == 0

    def test_statistics(self, repository_config):
        """Test repository statistics"""
        repo = KnowledgeRepository(repository_config)

        stats = repo.get_statistics()

        assert "total_nodes" in stats
        assert "total_paths" in stats
        assert "content_types" in stats
        assert "difficulties" in stats

        assert stats["total_nodes"] == 2
        assert stats["total_paths"] == 1

    def test_export_knowledge_graph(self, repository_config):
        """Test knowledge graph export"""
        repo = KnowledgeRepository(repository_config)

        graph = repo.export_knowledge_graph(format="json")

        assert "nodes" in graph
        assert "edges" in graph
        assert "metadata" in graph

        assert len(graph["nodes"]) == 2
        assert len(graph["edges"]) > 0

        # Check metadata
        metadata = graph["metadata"]
        assert metadata["total_nodes"] == 2
        assert metadata["total_paths"] == 1

    def test_missing_node_handling(self, repository_config):
        """Test handling of missing nodes"""
        repo = KnowledgeRepository(repository_config)

        # Try to get non-existent node
        node = repo.get_node("nonexistent")
        assert node is None

        # Try to get non-existent path
        path = repo.get_learning_path("nonexistent")
        assert path is None

        # Try to validate non-existent path
        validation = repo.validate_learning_path("nonexistent")
        assert validation["valid"] == False
        assert "Path not found" in validation["error"]

    def test_get_node_content(self, repository_config):
        """Test getting node content"""
        repo = KnowledgeRepository(repository_config)

        content = repo.get_node_content("node1")
        assert content is not None
        assert "overview" in content
        assert content["overview"] == "Test overview content"

        # Test non-existent node
        content = repo.get_node_content("nonexistent")
        assert content is None

    def test_repository_validation(self, repository_config):
        """Test repository validation functionality"""
        repo = KnowledgeRepository(repository_config)

        validation = repo.validate_repository()

        assert "valid" in validation
        assert "missing_prerequisites" in validation
        assert "orphaned_nodes" in validation
        assert "circular_dependencies" in validation

        # Our test repository should be valid
        assert validation["valid"] == True

    def test_json_schema_validation(self, repository_config):
        """Test JSON schema validation"""
        from active_inference.knowledge.repository import KnowledgeNodeSchema

        # Valid data
        valid_data = {
            "id": "test",
            "title": "Test",
            "content_type": "foundation",
            "difficulty": "beginner",
            "description": "Test description",
            "prerequisites": [],
            "content": {"overview": "test"},
            "tags": ["test"],
            "learning_objectives": ["learn"],
            "metadata": {"version": "1.0"}
        }

        errors = KnowledgeNodeSchema.validate_json_structure(valid_data)
        assert len(errors) == 0

        # Invalid data - missing required field
        invalid_data = valid_data.copy()
        del invalid_data["title"]

        errors = KnowledgeNodeSchema.validate_json_structure(invalid_data)
        assert len(errors) > 0
        assert any("title" in error for error in errors)

        # Invalid data - wrong type
        invalid_data = valid_data.copy()
        invalid_data["prerequisites"] = "not a list"

        errors = KnowledgeNodeSchema.validate_json_structure(invalid_data)
        assert len(errors) > 0
        assert any("prerequisites" in error for error in errors)


class TestKnowledgeNode:
    """Test cases for KnowledgeNode model"""

    def test_knowledge_node_creation(self):
        """Test creating knowledge nodes"""
        node_data = {
            "id": "test_node",
            "title": "Test Node",
            "content_type": ContentType.FOUNDATION,
            "difficulty": DifficultyLevel.BEGINNER,
            "description": "A test node",
            "prerequisites": [],
            "content": {
                "overview": "Test overview",
                "examples": []
            },
            "tags": ["test"],
            "learning_objectives": ["Learn something"],
            "metadata": {"version": "1.0"}
        }

        node = KnowledgeNode(**node_data)

        assert node.id == "test_node"
        assert node.title == "Test Node"
        assert node.content_type == ContentType.FOUNDATION
        assert node.difficulty == DifficultyLevel.BEGINNER
        assert node.prerequisites == []
        assert node.tags == ["test"]
        assert node.content == {"overview": "Test overview", "examples": []}
        assert node.metadata == {"version": "1.0"}


class TestLearningPath:
    """Test cases for LearningPath model"""

    def test_learning_path_creation(self):
        """Test creating learning paths"""
        path_data = {
            "id": "test_path",
            "name": "Test Path",
            "description": "A test path",
            "nodes": ["node1", "node2"],
            "estimated_hours": 4,
            "difficulty": DifficultyLevel.BEGINNER
        }

        path = LearningPath(**path_data)

        assert path.id == "test_path"
        assert path.name == "Test Path"
        assert path.nodes == ["node1", "node2"]
        assert path.estimated_hours == 4
        assert path.difficulty == DifficultyLevel.BEGINNER


if __name__ == "__main__":
    pytest.main([__file__])
