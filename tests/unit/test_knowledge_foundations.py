"""
Tests for Knowledge Foundations Module

Unit tests for the foundations module, ensuring proper operation of
foundation knowledge management, learning tracks, and educational content.
"""

import pytest
import json
from unittest.mock import Mock, patch
from pathlib import Path

from active_inference.knowledge.foundations import Foundations
from active_inference.knowledge.repository import KnowledgeRepository, KnowledgeRepositoryConfig


class TestFoundations:
    """Test cases for Foundations class"""

    @pytest.fixture
    def mock_repository(self):
        """Create mock repository for testing"""
        repo = Mock(spec=KnowledgeRepository)
        repo._nodes = {}
        return repo

    def test_foundations_initialization(self, mock_repository):
        """Test foundations module initialization"""
        foundations = Foundations(mock_repository)

        assert foundations.repository == mock_repository
        assert hasattr(foundations, 'get_foundation_tracks')
        assert hasattr(foundations, 'get_complete_foundation_path')

    def test_foundation_tracks_structure(self, mock_repository):
        """Test foundation tracks have correct structure"""
        foundations = Foundations(mock_repository)

        tracks = foundations.get_foundation_tracks()

        # Check required tracks exist
        required_tracks = [
            "information_theory",
            "bayesian_inference",
            "free_energy_principle",
            "active_inference"
        ]

        for track in required_tracks:
            assert track in tracks
            assert isinstance(tracks[track], list)
            assert len(tracks[track]) > 0

    def test_complete_foundation_path(self, mock_repository):
        """Test complete foundation learning path generation"""
        foundations = Foundations(mock_repository)

        complete_path = foundations.get_complete_foundation_path()

        assert isinstance(complete_path, list)
        assert len(complete_path) > 0

        # Check that path includes nodes from all tracks
        tracks = foundations.get_foundation_tracks()
        all_track_nodes = []
        for nodes in tracks.values():
            all_track_nodes.extend(nodes)

        # All track nodes should be in the complete path
        for node in all_track_nodes:
            assert node in complete_path

    def test_foundation_by_topic(self, mock_repository):
        """Test searching foundations by topic"""
        foundations = Foundations(mock_repository)

        # Test entropy topic
        entropy_nodes = foundations.get_foundation_by_topic("entropy")
        assert isinstance(entropy_nodes, list)
        assert "info_theory_entropy" in entropy_nodes

        # Test bayesian topic
        bayesian_nodes = foundations.get_foundation_by_topic("bayesian")
        assert isinstance(bayesian_nodes, list)
        assert any("bayesian" in node for node in bayesian_nodes)

    def test_foundation_validation(self, mock_repository):
        """Test foundation consistency validation"""
        foundations = Foundations(mock_repository)

        validation = foundations.validate_foundation_consistency()

        assert isinstance(validation, dict)
        assert "valid" in validation
        assert "issues" in validation
        assert "track_completeness" in validation
        assert "prerequisite_satisfaction" in validation

        # Should be valid (no empty tracks)
        assert validation["valid"] is True

    def test_foundation_summary_generation(self, mock_repository):
        """Test foundation summary generation"""
        foundations = Foundations(mock_repository)

        summary = foundations.generate_foundation_summary()

        assert isinstance(summary, str)
        assert "Active Inference Foundations Summary" in summary
        assert "Total Learning Tracks" in summary
        assert "Total Foundation Nodes" in summary

        # Check that all tracks are mentioned
        tracks = foundations.get_foundation_tracks()
        for track_name in tracks.keys():
            assert track_name.replace("_", " ").title() in summary

    def test_foundation_search(self, mock_repository):
        """Test foundation concept search"""
        foundations = Foundations(mock_repository)

        # Search for entropy
        results = foundations.search_foundation_concepts("entropy")
        assert isinstance(results, list)

        if results:  # If any results found
            assert isinstance(results[0], dict)
            assert "id" in results[0]
            assert "track" in results[0]
            assert "relevance_score" in results[0]

            # Results should be sorted by relevance
            if len(results) > 1:
                assert results[0]["relevance_score"] >= results[1]["relevance_score"]

    def test_foundation_dependencies(self, mock_repository):
        """Test foundation node dependency analysis"""
        foundations = Foundations(mock_repository)

        # Test dependency analysis for entropy node
        dependencies = foundations.get_foundation_dependencies("info_theory_entropy")

        assert isinstance(dependencies, dict)
        assert dependencies["node_id"] == "info_theory_entropy"
        assert dependencies["track"] == "information_theory"
        assert dependencies["prerequisites"] == []
        assert "position" in dependencies
        assert "total_in_track" in dependencies

    def test_foundation_dependencies_unknown_node(self, mock_repository):
        """Test dependency analysis for unknown node"""
        foundations = Foundations(mock_repository)

        dependencies = foundations.get_foundation_dependencies("unknown_node")

        assert "error" in dependencies
        assert "not found" in dependencies["error"]

    def test_foundation_graph_export(self, mock_repository):
        """Test foundation graph export"""
        foundations = Foundations(mock_repository)

        # Export as JSON
        graph_json = foundations.export_foundation_graph("json")
        assert isinstance(graph_json, str)

        # Parse JSON to validate structure
        graph = json.loads(graph_json)
        assert "nodes" in graph
        assert "edges" in graph
        assert "tracks" in graph
        assert "metadata" in graph

        # Check metadata
        metadata = graph["metadata"]
        assert "total_tracks" in metadata
        assert "total_nodes" in metadata
        assert metadata["export_format"] == "json"

        # Check nodes structure
        assert len(graph["nodes"]) > 0
        for node in graph["nodes"]:
            assert "id" in node
            assert "track" in node
            assert "label" in node

        # Check edges structure
        for edge in graph["edges"]:
            assert "source" in edge
            assert "target" in edge
            assert "type" in edge
            assert "track" in edge

    def test_foundation_content_generation(self, mock_repository):
        """Test comprehensive content generation for foundation nodes"""
        foundations = Foundations(mock_repository)

        # Test entropy content
        entropy_content = foundations._generate_node_content({
            "id": "info_theory_entropy",
            "title": "Entropy"
        })

        assert isinstance(entropy_content, dict)
        assert "overview" in entropy_content
        assert "mathematical_definition" in entropy_content
        assert "properties" in entropy_content
        assert "examples" in entropy_content
        assert "applications" in entropy_content

        # Check content quality
        assert len(entropy_content["overview"]) > 50
        assert "entropy" in entropy_content["overview"].lower()
        assert len(entropy_content["examples"]) > 0
        assert len(entropy_content["applications"]) > 0

    def test_bayesian_content_generation(self, mock_repository):
        """Test Bayesian inference content generation"""
        foundations = Foundations(mock_repository)

        bayesian_content = foundations._generate_node_content({
            "id": "bayesian_basics",
            "title": "Bayesian Basics"
        })

        assert isinstance(bayesian_content, dict)
        assert "overview" in bayesian_content
        assert "mathematical_definition" in bayesian_content
        assert "key_concepts" in bayesian_content
        assert "examples" in bayesian_content

        # Check specific Bayesian content
        assert "bayes" in bayesian_content["mathematical_definition"].lower()
        assert any("prior" in concept.lower() for concept in bayesian_content["key_concepts"])
        assert any("posterior" in concept.lower() for concept in bayesian_content["key_concepts"])

    def test_active_inference_content_generation(self, mock_repository):
        """Test Active Inference content generation"""
        foundations = Foundations(mock_repository)

        ai_content = foundations._generate_node_content({
            "id": "active_inference_introduction",
            "title": "Active Inference Introduction"
        })

        assert isinstance(ai_content, dict)
        assert "overview" in ai_content
        assert "core_idea" in ai_content
        assert "planning_as_inference" in ai_content
        assert "examples" in ai_content

        # Check Active Inference specific content
        assert "active inference" in ai_content["overview"].lower()
        assert "surprise" in ai_content["core_idea"].lower() or "inference" in ai_content["core_idea"].lower()

    def test_unknown_node_content_generation(self, mock_repository):
        """Test content generation for unknown nodes"""
        foundations = Foundations(mock_repository)

        content = foundations._generate_node_content({
            "id": "unknown_node",
            "title": "Unknown Node"
        })

        assert isinstance(content, dict)
        assert "overview" in content
        assert "Unknown Node" in content["overview"]

    def test_node_saving_to_repository(self, mock_repository):
        """Test saving nodes to repository"""
        foundations = Foundations(mock_repository)

        # Create test nodes data
        nodes_data = [
            {
                "id": "test_node_1",
                "title": "Test Node 1",
                "content_type": "foundation",
                "difficulty": "beginner",
                "description": "Test node 1",
                "prerequisites": [],
                "tags": ["test"],
                "learning_objectives": ["Learn test concept"]
            },
            {
                "id": "test_node_2",
                "title": "Test Node 2",
                "content_type": "foundation",
                "difficulty": "intermediate",
                "description": "Test node 2",
                "prerequisites": ["test_node_1"],
                "tags": ["test", "intermediate"],
                "learning_objectives": ["Learn advanced test concept"]
            }
        ]

        # Save nodes to repository
        foundations._save_nodes_to_repository(nodes_data)

        # Check that nodes were created in repository
        assert "test_node_1" in mock_repository._nodes
        assert "test_node_2" in mock_repository._nodes

        # Check node properties
        node1 = mock_repository._nodes["test_node_1"]
        assert node1.title == "Test Node 1"
        assert node1.difficulty.name == "BEGINNER"
        assert node1.content_type.name == "FOUNDATION"

        node2 = mock_repository._nodes["test_node_2"]
        assert node2.title == "Test Node 2"
        assert node2.prerequisites == ["test_node_1"]


if __name__ == "__main__":
    pytest.main([__file__])
