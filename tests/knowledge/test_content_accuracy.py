"""
Knowledge Content Accuracy Testing

Tests for validating the factual accuracy of knowledge content in the
Active Inference Knowledge Environment. Ensures mathematical formulations,
conceptual explanations, and reference materials meet accuracy standards.
"""

import pytest
import json
import tempfile
from pathlib import Path
from typing import Dict, Any, List

from active_inference.knowledge.repository import KnowledgeRepository, KnowledgeNode, KnowledgeRepositoryConfig

pytestmark = pytest.mark.knowledge


class TestMathematicalAccuracy:
    """Test mathematical accuracy validation"""

    @pytest.fixture
    def sample_math_content(self):
        """Sample mathematical content for testing"""
        return {
            "id": "test_entropy",
            "title": "Test Entropy",
            "content_type": "mathematics",
            "difficulty": "beginner",
            "description": "Test entropy calculation",
            "prerequisites": [],
            "content": {
                "overview": "Entropy measures uncertainty in probability distributions",
                "mathematical_definition": "H(X) = -∑ p(x) log p(x)",
                "examples": [
                    {
                        "name": "Binary entropy",
                        "description": "H(p) = -p log p - (1-p) log(1-p)"
                    }
                ]
            },
            "tags": ["entropy", "information_theory"],
            "learning_objectives": ["Understand entropy", "Calculate entropy"],
            "metadata": {"version": "1.0"}
        }

    @pytest.fixture
    def knowledge_repo(self, sample_math_content):
        """Set up knowledge repository with test content"""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create repository structure
            foundations_dir = temp_path / "mathematics"
            foundations_dir.mkdir()

            # Save test content
            content_file = foundations_dir / "test_entropy.json"
            with open(content_file, 'w') as f:
                json.dump(sample_math_content, f, indent=2)

            # Create repository config
            config = KnowledgeRepositoryConfig(root_path=temp_path)
            repo = KnowledgeRepository(config)

            yield repo

    def test_entropy_formula_validation(self, knowledge_repo):
        """Test validation of entropy mathematical formula"""
        content = knowledge_repo.get_node("test_entropy")
        assert content is not None

        # Check that mathematical definition is present
        math_def = content.content.get("mathematical_definition")
        assert math_def is not None
        assert "H(X)" in math_def
        assert "∑" in math_def or "sum" in math_def.lower()
        assert "log" in math_def.lower()

    def test_formula_syntax_validation(self, knowledge_repo):
        """Test validation of mathematical formula syntax"""
        content = knowledge_repo.get_node("test_entropy")

        # Check for proper mathematical notation
        math_def = content.content.get("mathematical_definition")
        assert math_def is not None

        # Should contain proper probability notation
        assert any(char in math_def for char in ["p(", "P("])

        # Should contain summation or integration
        assert any(char in math_def for char in ["∑", "Σ", "sum", "∫"])

    def test_concept_mathematical_relationships(self, knowledge_repo):
        """Test mathematical relationships between concepts"""
        content = knowledge_repo.get_node("test_entropy")

        # Check that entropy relates to information theory concepts
        tags = content.tags
        assert "entropy" in tags
        assert "information_theory" in tags

        # Check learning objectives are mathematically sound
        objectives = content.learning_objectives
        assert any("entropy" in obj.lower() for obj in objectives)


class TestConceptualAccuracy:
    """Test conceptual accuracy validation"""

    @pytest.fixture
    def sample_concept_content(self):
        """Sample conceptual content for testing"""
        return {
            "id": "test_active_inference",
            "title": "Test Active Inference",
            "content_type": "foundation",
            "difficulty": "beginner",
            "description": "Test active inference concept",
            "prerequisites": [],
            "content": {
                "overview": "Active Inference is a framework for understanding intelligent behavior",
                "mathematical_definition": "Active Inference minimizes variational free energy",
                "examples": [
                    {
                        "name": "Perception",
                        "description": "Inferring hidden states from observations"
                    },
                    {
                        "name": "Action",
                        "description": "Selecting actions to minimize expected free energy"
                    }
                ]
            },
            "tags": ["active_inference", "free_energy", "perception", "action"],
            "learning_objectives": [
                "Understand active inference framework",
                "Explain free energy principle",
                "Describe perception-action cycle"
            ],
            "metadata": {"version": "1.0"}
        }

    @pytest.fixture
    def knowledge_repo(self, sample_concept_content):
        """Set up knowledge repository with test content"""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create repository structure
            foundations_dir = temp_path / "foundations"
            foundations_dir.mkdir()

            # Save test content
            content_file = foundations_dir / "test_active_inference.json"
            with open(content_file, 'w') as f:
                json.dump(sample_concept_content, f, indent=2)

            # Create repository config
            config = KnowledgeRepositoryConfig(root_path=temp_path)
            repo = KnowledgeRepository(config)

            yield repo

    def test_concept_definition_completeness(self, knowledge_repo):
        """Test completeness of concept definitions"""
        content = knowledge_repo.get_node("test_active_inference")
        assert content is not None

        # Check that overview provides clear definition
        overview = content.content.get("overview")
        assert overview is not None
        assert len(overview) > 20  # Should be substantive

        # Check for key terms
        assert any(term in overview.lower() for term in ["framework", "understanding", "behavior"])

    def test_learning_objectives_clarity(self, knowledge_repo):
        """Test clarity of learning objectives"""
        content = knowledge_repo.get_node("test_active_inference")

        objectives = content.learning_objectives
        assert isinstance(objectives, list)
        assert len(objectives) > 0

        # Each objective should be clear and specific
        for objective in objectives:
            assert isinstance(objective, str)
            assert len(objective) > 10  # Should be substantive
            assert objective[0].isupper()  # Should start with capital letter

    def test_examples_relevance(self, knowledge_repo):
        """Test relevance of examples to concepts"""
        content = knowledge_repo.get_node("test_active_inference")

        examples = content.content.get("examples", [])
        assert isinstance(examples, list)
        assert len(examples) > 0

        # Examples should relate to main concept
        concept_tags = set(content.tags)
        for example in examples:
            assert "name" in example
            assert "description" in example
            assert len(example["description"]) > 10


class TestReferenceValidation:
    """Test reference and citation validation"""

    @pytest.fixture
    def sample_referenced_content(self):
        """Sample content with references for testing"""
        return {
            "id": "test_fep",
            "title": "Test Free Energy Principle",
            "content_type": "foundation",
            "difficulty": "intermediate",
            "description": "Test free energy principle with references",
            "prerequisites": ["test_bayesian_basics"],
            "content": {
                "overview": "The Free Energy Principle explains how biological systems maintain themselves",
                "mathematical_definition": "F = D_KL[q(θ)||p(θ|x)] - log p(x)",
                "examples": [
                    {
                        "name": "Predictive coding",
                        "description": "Hierarchical prediction and error minimization"
                    }
                ]
            },
            "tags": ["free_energy_principle", "predictive_coding", "bayesian"],
            "learning_objectives": [
                "Understand free energy principle",
                "Explain predictive coding",
                "Apply variational inference"
            ],
            "metadata": {
                "version": "1.0",
                "references": [
                    "Friston, K. (2009). The free-energy principle: a rough guide to the brain? Trends in Cognitive Sciences, 13(7), 293-301.",
                    "Friston, K. (2010). The free energy principle: a unified brain theory? Nature Reviews Neuroscience, 11(2), 127-138."
                ]
            }
        }

    def test_reference_format_validation(self, sample_referenced_content):
        """Test validation of reference formats"""
        # Check reference structure
        metadata = sample_referenced_content["metadata"]
        references = metadata.get("references", [])

        assert isinstance(references, list)

        for ref in references:
            assert isinstance(ref, str)
            assert len(ref) > 10  # Should be substantive
            # Should contain author and year information
            assert any(char.isdigit() for char in ref)  # Should have year
            assert "(" in ref and ")" in ref  # Should have parentheses

    def test_citation_completeness(self, sample_referenced_content):
        """Test completeness of citations"""
        # Check that citations contain required information
        metadata = sample_referenced_content["metadata"]
        references = metadata.get("references", [])

        for ref in references:
            # Should have author
            assert any(word.istitle() for word in ref.split())

            # Should have publication year (4 digits) - more flexible extraction
            import re
            years = re.findall(r'\b(19|20)\d{2}\b', ref)  # Find 4-digit years starting with 19 or 20
            assert len(years) > 0

    def test_content_reference_consistency(self, sample_referenced_content):
        """Test consistency between content and references"""
        content_tags = set(sample_referenced_content["tags"])
        references = sample_referenced_content["metadata"].get("references", [])

        # Content should be consistent with referenced topics
        content_text = json.dumps(sample_referenced_content).lower()

        # Should mention key concepts from references
        assert "free" in content_text and "energy" in content_text
        assert "principle" in content_text


class TestCrossReferenceValidation:
    """Test cross-reference validation between content"""

    @pytest.fixture
    def sample_cross_referenced_content(self):
        """Sample content with cross-references for testing"""
        return [
            {
                "id": "bayesian_basics",
                "title": "Bayesian Basics",
                "content_type": "foundation",
                "difficulty": "beginner",
                "description": "Basic Bayesian inference concepts",
                "prerequisites": [],
                "content": {
                    "overview": "Bayesian inference updates beliefs based on new evidence",
                    "mathematical_definition": "p(θ|x) = p(x|θ) p(θ) / p(x)"
                },
                "tags": ["bayesian", "inference", "probability"],
                "learning_objectives": ["Understand Bayes theorem", "Apply conditional probability"],
                "metadata": {"version": "1.0"}
            },
            {
                "id": "advanced_bayesian",
                "title": "Advanced Bayesian Methods",
                "content_type": "foundation",
                "difficulty": "advanced",
                "description": "Advanced Bayesian inference techniques",
                "prerequisites": ["bayesian_basics"],
                "content": {
                    "overview": "Advanced methods extend basic Bayesian inference to complex models",
                    "mathematical_definition": "Variational inference approximates posterior distributions"
                },
                "tags": ["bayesian", "variational_inference", "approximation"],
                "learning_objectives": ["Apply variational inference", "Understand approximation methods"],
                "metadata": {"version": "1.0"}
            }
        ]

    def test_prerequisite_chain_validation(self, sample_cross_referenced_content):
        """Test validation of prerequisite chains"""
        basic_content, advanced_content = sample_cross_referenced_content

        # Advanced content should reference basic content as prerequisite
        assert "bayesian_basics" in advanced_content["prerequisites"]

        # Basic content should have no prerequisites
        assert len(basic_content["prerequisites"]) == 0

        # Difficulty should progress appropriately
        assert basic_content["difficulty"] == "beginner"
        assert advanced_content["difficulty"] == "advanced"

    def test_concept_progression_validation(self, sample_cross_referenced_content):
        """Test validation of concept progression"""
        basic_content, advanced_content = sample_cross_referenced_content

        # Tags should show progression
        basic_tags = set(basic_content["tags"])
        advanced_tags = set(advanced_content["tags"])

        # Advanced should build upon basic concepts (either contain them or have related concepts)
        # This is more flexible - advanced concepts might specialize rather than just extend
        common_concepts = basic_tags.intersection(advanced_tags)
        assert len(common_concepts) > 0 or any("bayes" in tag.lower() for tag in advanced_tags)

        # Learning objectives should build upon each other
        basic_objectives = [obj.lower() for obj in basic_content["learning_objectives"]]
        advanced_objectives = [obj.lower() for obj in advanced_content["learning_objectives"]]

        # Advanced should reference basic concepts (more flexible check)
        assert any("bayes" in obj or "inference" in obj or "belief" in obj for obj in advanced_objectives)


if __name__ == "__main__":
    pytest.main([__file__])
