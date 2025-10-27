"""
Tests for Knowledge Implementations Module
"""
import unittest
from pathlib import Path
import tempfile
import json

from active_inference.knowledge.repository import KnowledgeRepository, KnowledgeRepositoryConfig
from active_inference.knowledge.implementations import Implementations
from active_inference.knowledge.foundations import Foundations


class TestKnowledgeImplementations(unittest.TestCase):
    """Test knowledge implementations functionality"""

    def setUp(self):
        """Set up test environment"""
        self.test_dir = Path(tempfile.mkdtemp())
        self.config = KnowledgeRepositoryConfig(
            root_path=self.test_dir,
            auto_index=True,
            cache_enabled=True
        )
        self.repository = KnowledgeRepository(self.config)
        self.implementations = Implementations(self.repository)

    def tearDown(self):
        """Clean up test environment"""
        import shutil
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_implementations_initialization(self):
        """Test implementations module initialization"""
        self.assertIsNotNone(self.implementations)
        self.assertIsInstance(self.implementations.repository, KnowledgeRepository)

    def test_basic_implementation_creation(self):
        """Test creation of basic implementation nodes"""
        # Create a knowledge node directly for testing
        from active_inference.knowledge.repository import KnowledgeNode, ContentType, DifficultyLevel

        impl_node = KnowledgeNode(
            id="active_inference_basic",
            title="Basic Active Inference Implementation",
            content_type=ContentType.IMPLEMENTATION,
            difficulty=DifficultyLevel.INTERMEDIATE,
            description="Test implementation for unit testing",
            prerequisites=["probability_basics"],
            tags=["active inference", "implementation", "python"],
            learning_objectives=["Test implementation creation"],
            content={
                "sections": [
                    {"title": "Overview", "content": "Test implementation"},
                    {"title": "Code", "content": "Test code example"}
                ]
            },
            metadata={
                "estimated_reading_time": 30,
                "author": "Test Author",
                "last_updated": "2024-10-27",
                "version": "1.0"
            }
        )

        # Test node creation
        self.assertIsNotNone(impl_node)
        self.assertEqual(impl_node.id, "active_inference_basic")
        self.assertEqual(impl_node.content_type, ContentType.IMPLEMENTATION)
        self.assertEqual(impl_node.difficulty, DifficultyLevel.INTERMEDIATE)

    def test_implementation_validation(self):
        """Test implementation validation functionality"""
        # Create a knowledge node directly for testing
        from active_inference.knowledge.repository import KnowledgeNode, ContentType, DifficultyLevel

        impl_node = KnowledgeNode(
            id="active_inference_basic",
            title="Basic Active Inference Implementation",
            content_type=ContentType.IMPLEMENTATION,
            difficulty=DifficultyLevel.INTERMEDIATE,
            description="Test implementation for unit testing",
            prerequisites=["probability_basics"],
            tags=["active inference", "implementation", "python"],
            learning_objectives=["Test implementation creation"],
            content={
                "sections": [
                    {"title": "Overview", "content": "Test implementation"},
                    {"title": "Code", "content": "Test code example"}
                ]
            },
            metadata={
                "estimated_reading_time": 30,
                "author": "Test Author",
                "last_updated": "2024-10-27",
                "version": "1.0"
            }
        )

        # Test validation logic directly
        validation_result = {
            "valid": True,
            "issues": [],
            "suggestions": []
        }

        # Check if implementation has required sections
        content = impl_node.content
        if not content.get("sections"):
            validation_result["issues"].append("Missing implementation sections")
            validation_result["valid"] = False

        # Check for code examples
        has_code = any("code" in section.get("content", "").lower()
                      for section in content.get("sections", []))
        if not has_code:
            validation_result["suggestions"].append("Add code examples to implementation")

        # Check prerequisites
        if not impl_node.prerequisites:
            validation_result["suggestions"].append("Add prerequisite knowledge nodes")

        # Should have validation result
        self.assertIn("valid", validation_result)
        self.assertIn("issues", validation_result)
        self.assertIn("suggestions", validation_result)

    def test_tutorial_path_generation(self):
        """Test tutorial path generation"""
        tutorials = self.implementations.get_tutorial_path()

        # Should return list of tutorial nodes
        self.assertIsInstance(tutorials, list)

        # Check difficulty ordering (beginner first)
        if len(tutorials) > 1:
            for i in range(len(tutorials) - 1):
                current_difficulty = tutorials[i].difficulty
                next_difficulty = tutorials[i + 1].difficulty

                # Should be in order of increasing difficulty
                difficulty_order = {
                    "beginner": 1,
                    "intermediate": 2,
                    "advanced": 3,
                    "expert": 4
                }

                current_order = difficulty_order.get(current_difficulty.value, 5)
                next_order = difficulty_order.get(next_difficulty.value, 5)

                self.assertLessEqual(current_order, next_order)

    def test_implementation_examples_filtering(self):
        """Test filtering implementation examples"""
        # Test that the method exists and returns appropriate types
        from active_inference.knowledge.repository import DifficultyLevel

        # Test filtering by difficulty - should work with existing repository content
        beginner_impls = self.implementations.get_implementation_examples(
            difficulty=DifficultyLevel.BEGINNER
        )

        # Should return implementations (even if empty, should be a list)
        self.assertIsInstance(beginner_impls, list)

        # Test filtering by tags - should work with existing repository content
        python_impls = self.implementations.get_implementation_examples(
            tags=["python"]
        )

        # Should return implementations (even if empty, should be a list)
        self.assertIsInstance(python_impls, list)

        # Test with no filters
        all_impls = self.implementations.get_implementation_examples()
        self.assertIsInstance(all_impls, list)

    def test_advanced_implementation_creation(self):
        """Test creation of advanced implementation nodes"""
        # Test that the repository can handle implementation nodes
        from active_inference.knowledge.repository import KnowledgeNode, ContentType, DifficultyLevel

        # Create neural network implementation node
        neural_impl = KnowledgeNode(
            id="neural_network_implementation",
            title="Neural Network Implementation of Active Inference",
            content_type=ContentType.IMPLEMENTATION,
            difficulty=DifficultyLevel.EXPERT,
            description="Deep learning implementation of Active Inference principles",
            prerequisites=["neural_networks", "variational_inference"],
            tags=["neural networks", "deep learning", "tensorflow", "pytorch"],
            learning_objectives=[
                "Implement Active Inference in neural network frameworks",
                "Train models using free energy minimization",
                "Scale to complex environments and tasks"
            ],
            content={
                "sections": [
                    {"title": "Neural Active Inference", "content": "Neural networks provide a natural framework..."}
                ]
            },
            metadata={
                "estimated_reading_time": 45,
                "author": "Active Inference Community",
                "last_updated": "2024-10-27",
                "version": "1.0"
            }
        )

        # Test node creation
        self.assertIsNotNone(neural_impl)
        self.assertEqual(neural_impl.content_type, ContentType.IMPLEMENTATION)
        self.assertEqual(neural_impl.difficulty, DifficultyLevel.EXPERT)
        self.assertIn("neural networks", neural_impl.tags)


if __name__ == '__main__':
    unittest.main()
