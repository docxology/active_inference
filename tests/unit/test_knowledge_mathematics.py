"""
Tests for Knowledge Mathematics Module

Unit tests for the mathematics module, ensuring proper operation of
mathematical computations, derivations, and validation functions.
"""

import pytest
import math
from unittest.mock import Mock, patch
from typing import List, Dict, Any

from active_inference.knowledge.mathematics import Mathematics
from active_inference.knowledge.repository import KnowledgeRepository, KnowledgeRepositoryConfig


class TestMathematics:
    """Test cases for Mathematics class"""

    @pytest.fixture
    def mock_repository(self):
        """Create mock repository for testing"""
        repo = Mock(spec=KnowledgeRepository)
        repo._nodes = {}
        return repo

    def test_mathematics_initialization(self, mock_repository):
        """Test mathematics module initialization"""
        math_system = Mathematics(mock_repository)

        assert math_system.repository == mock_repository
        assert hasattr(math_system, 'compute_entropy')
        assert hasattr(math_system, 'get_mathematical_prerequisites')

    def test_entropy_computation_without_numpy(self, mock_repository):
        """Test entropy computation without numpy"""
        math_system = Mathematics(mock_repository)

        # Test with uniform distribution
        probs = [0.5, 0.5]
        entropy = math_system.compute_entropy(probs)
        assert isinstance(entropy, float)
        assert entropy >= 0

        # Test with deterministic distribution
        probs = [1.0, 0.0]
        entropy = math_system.compute_entropy(probs)
        assert entropy == 0.0

    def test_entropy_computation_with_numpy(self, mock_repository):
        """Test entropy computation with numpy"""
        with patch('active_inference.knowledge.mathematics.HAS_NUMPY', True):
            math_system = Mathematics(mock_repository)

            # Test with uniform distribution
            probs = [0.25, 0.25, 0.25, 0.25]
            entropy = math_system.compute_entropy(probs)
            expected_entropy = -4 * (0.25 * math.log2(0.25))
            assert abs(entropy - expected_entropy) < 0.01

    def test_mutual_information_computation(self, mock_repository):
        """Test mutual information computation"""
        math_system = Mathematics(mock_repository)

        # Simple 2x2 joint distribution
        joint_probs = [[0.1, 0.2], [0.3, 0.4]]
        marginal_x = [0.3, 0.7]
        marginal_y = [0.4, 0.6]

        mutual_info = math_system.compute_mutual_information(joint_probs, marginal_x, marginal_y)

        assert isinstance(mutual_info, float)
        assert mutual_info >= 0  # Mutual information is non-negative

    def test_variational_free_energy_computation(self, mock_repository):
        """Test variational free energy computation"""
        math_system = Mathematics(mock_repository)

        log_likelihood = -2.0
        kl_divergence = 0.5

        free_energy = math_system.compute_variational_free_energy(log_likelihood, kl_divergence)

        assert free_energy == log_likelihood - kl_divergence
        assert free_energy == -2.5

    def test_free_energy_optimization(self, mock_repository):
        """Test free energy optimization"""
        math_system = Mathematics(mock_repository)

        initial_params = [0.1, 0.2, 0.3]
        optimized_params = math_system.optimize_free_energy(initial_params, num_iterations=10)

        assert isinstance(optimized_params, list)
        assert len(optimized_params) == len(initial_params)
        assert all(isinstance(param, float) for param in optimized_params)

    def test_expected_free_energy_components(self, mock_repository):
        """Test expected free energy components computation"""
        math_system = Mathematics(mock_repository)

        components = math_system.compute_expected_free_energy_components(
            risk=1.0, ambiguity=0.5, value=0.3
        )

        assert isinstance(components, dict)
        assert "risk" in components
        assert "ambiguity" in components
        assert "value" in components
        assert "total_efe" in components

        assert components["risk"] == 1.0
        assert components["ambiguity"] == 0.5
        assert components["value"] == 0.3
        assert components["total_efe"] == 1.2  # 1.0 + 0.5 - 0.3

    def test_mathematical_consistency_validation_without_sympy(self, mock_repository):
        """Test mathematical consistency validation without sympy"""
        math_system = Mathematics(mock_repository)

        expressions = [
            "log(x) + log(y)",
            "exp(x + y)",
            "sin(x) * cos(y)"
        ]

        validation = math_system.validate_mathematical_consistency(expressions)

        assert isinstance(validation, dict)
        assert "valid" in validation
        assert "issues" in validation
        assert "warnings" in validation

        assert validation["valid"] is True
        assert len(validation["issues"]) == 0

    def test_mathematical_derivation_generation(self, mock_repository):
        """Test mathematical derivation generation"""
        math_system = Mathematics(mock_repository)

        # Test entropy derivation
        entropy_derivation = math_system.generate_mathematical_derivation("entropy")
        assert isinstance(entropy_derivation, str)
        assert "entropy" in entropy_derivation.lower()
        assert "log" in entropy_derivation

        # Test KL divergence derivation
        kl_derivation = math_system.generate_mathematical_derivation("kl_divergence")
        assert isinstance(kl_derivation, str)
        assert "kl" in kl_derivation.lower() or "divergence" in kl_derivation.lower()

        # Test unknown concept
        unknown_derivation = math_system.generate_mathematical_derivation("unknown_concept")
        assert isinstance(unknown_derivation, str)
        assert "unknown_concept" in unknown_derivation

    def test_information_geometry_metric(self, mock_repository):
        """Test Fisher information metric computation"""
        math_system = Mathematics(mock_repository)

        probabilities = [0.2, 0.3, 0.5]
        metric = math_system.compute_information_geometry_metric(probabilities)

        assert isinstance(metric, list)
        assert len(metric) == len(probabilities)

        for row in metric:
            assert isinstance(row, list)
            assert len(row) == len(probabilities)

        # Check diagonal elements (should be non-zero)
        for i in range(len(probabilities)):
            assert metric[i][i] > 0  # Diagonal should be positive

        # Check off-diagonal elements (should be zero for multinomial)
        for i in range(len(probabilities)):
            for j in range(len(probabilities)):
                if i != j:
                    assert metric[i][j] == 0.0

    def test_mathematical_complexity_analysis(self, mock_repository):
        """Test mathematical expression complexity analysis"""
        math_system = Mathematics(mock_repository)

        # Test simple expression
        simple_expr = "x + y"
        complexity = math_system.analyze_mathematical_complexity(simple_expr)

        assert isinstance(complexity, dict)
        assert "expression_length" in complexity
        assert "num_operators" in complexity
        assert "num_functions" in complexity
        assert "num_variables" in complexity
        assert "total_complexity" in complexity

        assert complexity["expression_length"] == len(simple_expr)
        assert complexity["num_operators"] == 1  # One '+'
        assert complexity["num_variables"] == 2  # x and y
        assert complexity["total_complexity"] > 0

    def test_nesting_depth_calculation(self, mock_repository):
        """Test nesting depth calculation"""
        math_system = Mathematics(mock_repository)

        # Test simple expression
        simple_expr = "x + y"
        depth = math_system._calculate_nesting_depth(simple_expr)
        assert depth == 0

        # Test nested expression
        nested_expr = "log(sin(x + y) * exp(z))"
        depth = math_system._calculate_nesting_depth(nested_expr)
        assert depth > 0

    def test_prerequisite_validation(self, mock_repository):
        """Test mathematical prerequisite validation"""
        math_system = Mathematics(mock_repository)

        # Test valid chain (using actual prerequisites from the system)
        prerequisites = math_system.get_mathematical_prerequisites()
        valid_chain = list(prerequisites.keys())[:3]  # Use first 3 concepts
        validation = math_system.validate_mathematical_prerequisites(valid_chain)

        assert isinstance(validation, dict)
        assert "valid" in validation
        assert "missing_prerequisites" in validation
        assert "circular_dependencies" in validation
        assert "redundant_prerequisites" in validation

        # Should be valid for concepts that exist in prerequisites
        assert validation["valid"] is True

    def test_circular_dependency_detection(self, mock_repository):
        """Test circular dependency detection"""
        math_system = Mathematics(mock_repository)

        # Create circular dependency
        circular_prereqs = {
            "concept_a": ["concept_b"],
            "concept_b": ["concept_c"],
            "concept_c": ["concept_a"]  # Circular!
        }

        has_circular = math_system._has_circular_dependency("concept_a", circular_prereqs, set())
        assert has_circular is True

        # Test non-circular dependency
        non_circular_prereqs = {
            "concept_a": ["concept_b"],
            "concept_b": ["concept_c"],
            "concept_c": []  # No circular dependency
        }

        has_circular = math_system._has_circular_dependency("concept_a", non_circular_prereqs, set())
        assert has_circular is False

    def test_mathematical_learning_path(self, mock_repository):
        """Test mathematical learning path generation"""
        math_system = Mathematics(mock_repository)

        learning_path = math_system.create_mathematical_learning_path()

        assert isinstance(learning_path, list)
        assert len(learning_path) > 0

        # Check that all concepts are strings
        assert all(isinstance(concept, str) for concept in learning_path)

        # Check that path includes core concepts
        concept_names = [concept.replace('_', ' ') for concept in learning_path]
        assert any('probability' in name.lower() for name in concept_names)
        assert any('bayesian' in name.lower() for name in concept_names)

    def test_mathematical_summary_generation(self, mock_repository):
        """Test mathematical summary generation"""
        math_system = Mathematics(mock_repository)

        summary = math_system.generate_mathematical_summary()

        assert isinstance(summary, str)
        assert "Mathematical Foundations Summary" in summary
        assert "Total Mathematical Concepts" in summary
        assert "Learning Sequence" in summary

        # Check that learning path is included
        learning_path = math_system.create_mathematical_learning_path()
        for concept in learning_path:
            assert concept.replace('_', ' ').title() in summary

    def test_mathematical_prerequisites_structure(self, mock_repository):
        """Test mathematical prerequisites structure"""
        math_system = Mathematics(mock_repository)

        prerequisites = math_system.get_mathematical_prerequisites()

        assert isinstance(prerequisites, dict)
        assert len(prerequisites) > 0

        # Check that all prerequisites are lists of strings
        for concept, prereqs in prerequisites.items():
            assert isinstance(concept, str)
            assert isinstance(prereqs, list)
            assert all(isinstance(prereq, str) for prereq in prereqs)

    def test_mathematical_consistency_validation_with_sympy(self, mock_repository):
        """Test mathematical consistency validation with sympy"""
        # Mock sympy module
        mock_sympy = Mock()
        mock_sympy.sympify.return_value = Mock()
        mock_sympy.sympify.return_value.has.return_value = False
        mock_sympy.sympify.return_value.simplify.return_value = Mock()
        mock_sympy.S = Mock()
        mock_sympy.S.Zero = Mock()
        mock_sympy.log = Mock()

        with patch.dict('sys.modules', {'sympy': mock_sympy}):
            with patch('active_inference.knowledge.mathematics.HAS_SYMPY', True):
                math_system = Mathematics(mock_repository)

                expressions = [
                    "log(x) + log(y)",
                    "x^2 + 2*x + 1",  # (x+1)^2
                    "sin(x)^2 + cos(x)^2"  # Should equal 1
                ]

                validation = math_system.validate_mathematical_consistency(expressions)

                assert isinstance(validation, dict)
                assert "valid" in validation
                assert "simplified_expressions" in validation

                # Should have simplified expressions if sympy is available
                assert len(validation["simplified_expressions"]) > 0


if __name__ == "__main__":
    pytest.main([__file__])
