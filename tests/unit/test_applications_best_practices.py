"""
Tests for Applications - Best Practices Module

Unit tests for the best practices and architectural patterns module,
ensuring proper functionality of architecture recommendations, pattern
validation, and best practice guidance systems.
"""

import pytest
from unittest.mock import Mock, patch
from typing import Dict, Any, List

from active_inference.applications.best_practices import (
    ArchitecturePattern,
    BestPractice,
    ArchitecturePatterns,
    BestPracticesGuide
)

pytestmark = pytest.mark.unit


class TestArchitecturePattern:
    """Test cases for ArchitecturePattern enum"""

    def test_architecture_pattern_values(self):
        """Test architecture pattern enum values"""
        # Test all expected pattern values
        assert ArchitecturePattern.MODEL_VIEW_CONTROLLER.value == "mvc"
        assert ArchitecturePattern.ACTIVE_INFERENCE_AGENT.value == "active_inference_agent"
        assert ArchitecturePattern.HIERARCHICAL_CONTROL.value == "hierarchical_control"
        assert ArchitecturePattern.MULTI_SCALE_MODELING.value == "multi_scale_modeling"
        assert ArchitecturePattern.DISTRIBUTED_SYSTEM.value == "distributed_system"

    def test_architecture_pattern_count(self):
        """Test correct number of architecture patterns"""
        patterns = list(ArchitecturePattern)
        assert len(patterns) == 5


class TestBestPractice:
    """Test cases for BestPractice dataclass"""

    def test_best_practice_creation(self):
        """Test creating best practice instances"""
        practice = BestPractice(
            id="test_practice",
            category="testing",
            title="Test Practice",
            description="A test best practice",
            examples=["example 1", "example 2"],
            rationale="Testing is important",
            related_patterns=["pattern1"]
        )

        assert practice.id == "test_practice"
        assert practice.category == "testing"
        assert practice.title == "Test Practice"
        assert practice.description == "A test best practice"
        assert practice.examples == ["example 1", "example 2"]
        assert practice.rationale == "Testing is important"
        assert practice.related_patterns == ["pattern1"]


class TestArchitecturePatterns:
    """Test cases for ArchitecturePatterns class"""

    def test_initialization(self):
        """Test architecture patterns initialization"""
        patterns = ArchitecturePatterns()

        assert hasattr(patterns, 'patterns')
        assert isinstance(patterns.patterns, dict)
        assert len(patterns.patterns) > 0

        # Check that expected patterns are initialized
        expected_patterns = [
            "active_inference_agent",
            "hierarchical_control",
            "multi_scale_modeling"
        ]

        for pattern in expected_patterns:
            assert pattern in patterns.patterns

    def test_get_pattern_existing(self):
        """Test getting existing pattern"""
        patterns = ArchitecturePatterns()

        pattern = patterns.get_pattern("active_inference_agent")
        assert pattern is not None
        assert pattern["name"] == "Active Inference Agent"
        assert "components" in pattern
        assert "data_flow" in pattern
        assert "scalability" in pattern

    def test_get_pattern_nonexistent(self):
        """Test getting non-existent pattern"""
        patterns = ArchitecturePatterns()

        pattern = patterns.get_pattern("nonexistent_pattern")
        assert pattern is None

    def test_list_patterns(self):
        """Test listing all patterns"""
        patterns = ArchitecturePatterns()

        pattern_list = patterns.list_patterns()
        assert isinstance(pattern_list, list)
        assert len(pattern_list) == len(patterns.patterns)

        # Check structure of returned patterns
        for pattern_item in pattern_list:
            assert "name" in pattern_item
            assert "details" in pattern_item
            assert isinstance(pattern_item["details"], dict)


class TestBestPracticesGuide:
    """Test cases for BestPracticesGuide class"""

    def test_initialization(self):
        """Test best practices guide initialization"""
        guide = BestPracticesGuide()

        assert hasattr(guide, 'practices')
        assert isinstance(guide.practices, dict)
        assert len(guide.practices) > 0

        # Check that expected categories are initialized
        expected_categories = ["model_design", "implementation", "testing"]
        for category in expected_categories:
            assert category in guide.practices
            assert isinstance(guide.practices[category], list)
            assert len(guide.practices[category]) > 0

    def test_get_practices_all(self):
        """Test getting all practices"""
        guide = BestPracticesGuide()

        all_practices = guide.get_practices()
        assert isinstance(all_practices, list)

        # Count total practices across all categories
        expected_count = sum(len(practices) for practices in guide.practices.values())
        assert len(all_practices) == expected_count

    def test_get_practices_by_category(self):
        """Test getting practices by category"""
        guide = BestPracticesGuide()

        # Test existing category
        model_practices = guide.get_practices("model_design")
        assert isinstance(model_practices, list)
        assert len(model_practices) > 0

        # Test non-existent category
        nonexistent_practices = guide.get_practices("nonexistent")
        assert isinstance(nonexistent_practices, list)
        assert len(nonexistent_practices) == 0

    def test_get_practice_existing(self):
        """Test getting specific practice"""
        guide = BestPracticesGuide()

        # Test getting existing practice
        practice = guide.get_practice("generative_model_clarity")
        assert practice is not None
        assert isinstance(practice, BestPractice)
        assert practice.id == "generative_model_clarity"
        assert practice.category == "model_design"

    def test_get_practice_nonexistent(self):
        """Test getting non-existent practice"""
        guide = BestPracticesGuide()

        practice = guide.get_practice("nonexistent_practice")
        assert practice is None

    def test_generate_architecture_recommendations_basic(self):
        """Test basic architecture recommendations"""
        guide = BestPracticesGuide()

        requirements = {
            "multi_level": False,
            "multi_scale": False,
            "distributed": False
        }

        recommendations = guide.generate_architecture_recommendations(requirements)

        assert isinstance(recommendations, list)
        # Should always include basic patterns
        assert "active_inference_agent" in recommendations
        assert "numerical_stability" in recommendations
        assert "modular_design" in recommendations

    def test_generate_architecture_recommendations_multi_level(self):
        """Test recommendations for multi-level requirements"""
        guide = BestPracticesGuide()

        requirements = {
            "multi_level": True,
            "multi_scale": False,
            "distributed": False
        }

        recommendations = guide.generate_architecture_recommendations(requirements)

        assert "hierarchical_control" in recommendations
        assert "active_inference_agent" in recommendations

    def test_generate_architecture_recommendations_multi_scale(self):
        """Test recommendations for multi-scale requirements"""
        guide = BestPracticesGuide()

        requirements = {
            "multi_level": False,
            "multi_scale": True,
            "distributed": False
        }

        recommendations = guide.generate_architecture_recommendations(requirements)

        assert "multi_scale_modeling" in recommendations
        assert "active_inference_agent" in recommendations

    def test_generate_architecture_recommendations_distributed(self):
        """Test recommendations for distributed requirements"""
        guide = BestPracticesGuide()

        requirements = {
            "multi_level": False,
            "multi_scale": False,
            "distributed": True
        }

        recommendations = guide.generate_architecture_recommendations(requirements)

        assert "distributed_system" in recommendations
        assert "active_inference_agent" in recommendations

    def test_generate_architecture_recommendations_complex(self):
        """Test recommendations for complex requirements"""
        guide = BestPracticesGuide()

        requirements = {
            "multi_level": True,
            "multi_scale": True,
            "distributed": True
        }

        recommendations = guide.generate_architecture_recommendations(requirements)

        # Should include all specialized patterns plus basics
        assert "hierarchical_control" in recommendations
        assert "multi_scale_modeling" in recommendations
        assert "distributed_system" in recommendations
        assert "active_inference_agent" in recommendations
        assert "numerical_stability" in recommendations
        assert "modular_design" in recommendations

    def test_practice_content_validation(self):
        """Test that practices contain valid content"""
        guide = BestPracticesGuide()

        all_practices = guide.get_practices()
        for practice in all_practices:
            assert isinstance(practice, BestPractice)
            assert len(practice.id) > 0
            assert len(practice.title) > 0
            assert len(practice.description) > 0
            assert len(practice.examples) > 0
            assert len(practice.rationale) > 0
            assert isinstance(practice.related_patterns, list)


if __name__ == "__main__":
    pytest.main([__file__])
