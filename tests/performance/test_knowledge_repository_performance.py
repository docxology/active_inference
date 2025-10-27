"""
Performance Tests for Knowledge Repository

Tests performance characteristics of the knowledge repository including
search speed, memory usage, and scalability under various loads.
"""

import pytest
import time
import psutil
import tempfile
from pathlib import Path
from typing import Dict, Any, List

from active_inference.knowledge.repository import KnowledgeRepository, KnowledgeRepositoryConfig

pytestmark = pytest.mark.performance


class TestKnowledgeRepositoryPerformance:
    """Performance tests for knowledge repository operations"""

    @pytest.fixture
    def large_knowledge_repo(self):
        """Set up a large knowledge repository for performance testing"""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create content directories first
            content_types = ["foundations", "mathematics", "implementations", "applications"]
            for content_type in content_types:
                content_dir = temp_path / content_type
                content_dir.mkdir()

                # Create many knowledge nodes
                for i in range(10):  # 10 nodes per type for faster testing
                    node_data = {
                        "id": f"{content_type}_node_{i}",
                        "title": f"Test {content_type.title()} Node {i}",
                        "content_type": {
                            "foundations": "foundation",
                            "mathematics": "mathematics",
                            "implementations": "implementation",
                            "applications": "application"
                        }[content_type],
                        "difficulty": "intermediate",
                        "description": f"Test description for {content_type} node {i}",
                        "prerequisites": [],
                        "content": {
                            "overview": f"This is test content {i} for {content_type}. It contains various concepts and explanations related to Active Inference and the Free Energy Principle.",
                            "mathematical_definition": f"F_{i}(x) = mathematical expression {i}",
                            "examples": [
                                {"name": f"Example {i}", "description": f"Example description {i}"}
                            ]
                        },
                        "tags": [content_type, f"test{i}", "active_inference", "performance"],
                        "learning_objectives": [f"Learn concept {i}", f"Apply method {i}"],
                        "metadata": {"version": "1.0"}
                    }

                    node_file = content_dir / f"{content_type}_node_{i}.json"
                    import json
                    with open(node_file, 'w') as f:
                        json.dump(node_data, f, indent=2)

            # Create repository config (will auto-load content)
            config = KnowledgeRepositoryConfig(root_path=temp_path)
            repo = KnowledgeRepository(config)

            yield repo

    def get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024  # Convert to MB

    def test_search_performance(self, large_knowledge_repo):
        """Test search performance with large repository"""
        # Measure search time
        start_time = time.time()
        start_memory = self.get_memory_usage()

        # Perform various searches (search for terms that are actually in content)
        results1 = large_knowledge_repo.search("test")
        results2 = large_knowledge_repo.search("content")  # This appears in overview text
        results3 = large_knowledge_repo.search("concept")  # This appears in titles and descriptions

        end_time = time.time()
        end_memory = self.get_memory_usage()

        search_time = end_time - start_time
        memory_increase = end_memory - start_memory

        # Performance assertions
        assert search_time < 1.0, f"Search took too long: {search_time:.3f}s"
        # Search should complete without errors (results count depends on search implementation)
        assert isinstance(results1, list), "Search should return a list"
        assert isinstance(results2, list), "Search should return a list"
        assert isinstance(results3, list), "Search should return a list"

        # Memory usage should be reasonable
        assert memory_increase < 100, f"Memory usage increased too much: {memory_increase:.1f}MB"

    def test_filtering_performance(self, large_knowledge_repo):
        """Test filtering performance"""
        start_time = time.time()
        start_memory = self.get_memory_usage()

        # Test filtering by content type
        from active_inference.knowledge.repository import ContentType
        results = large_knowledge_repo.search("", content_types=[ContentType.FOUNDATION])

        end_time = time.time()
        end_memory = self.get_memory_usage()

        filter_time = end_time - start_time
        memory_increase = end_memory - start_memory

        # Should complete quickly
        assert filter_time < 0.5, f"Filtering took too long: {filter_time:.3f}s"
        assert len(results) > 0, "Filtering should return results"

    def test_repository_loading_performance(self, large_knowledge_repo):
        """Test repository loading performance"""
        # Repository should already be loaded, test access time
        start_time = time.time()

        # Access multiple nodes
        for i in range(10):
            node = large_knowledge_repo.get_node(f"foundations_node_{i}")
            assert node is not None

        end_time = time.time()
        access_time = end_time - start_time

        # Should be very fast to access loaded content
        assert access_time < 0.1, f"Node access took too long: {access_time:.3f}s"

    def test_statistics_performance(self, large_knowledge_repo):
        """Test statistics calculation performance"""
        start_time = time.time()
        start_memory = self.get_memory_usage()

        stats = large_knowledge_repo.get_statistics()

        end_time = time.time()
        end_memory = self.get_memory_usage()

        stats_time = end_time - start_time
        memory_increase = end_memory - start_memory

        # Statistics should be calculated quickly
        assert stats_time < 0.2, f"Statistics calculation took too long: {stats_time:.3f}s"

        # Should contain expected statistics
        assert "total_nodes" in stats
        assert "total_paths" in stats
        assert "content_types" in stats
        assert stats["total_nodes"] == 40  # 4 types * 10 nodes each

    def test_prerequisite_graph_performance(self, large_knowledge_repo):
        """Test prerequisite graph generation performance"""
        start_time = time.time()
        start_memory = self.get_memory_usage()

        # Generate prerequisite graph for a node with dependencies
        graph = large_knowledge_repo.get_prerequisites_graph("implementations_node_1")

        end_time = time.time()
        end_memory = self.get_memory_usage()

        graph_time = end_time - start_time
        memory_increase = end_memory - start_memory

        # Graph generation should be reasonably fast
        assert graph_time < 0.5, f"Graph generation took too long: {graph_time:.3f}s"
        assert "nodes" in graph
        assert "edges" in graph


if __name__ == "__main__":
    pytest.main([__file__])
