# Knowledge Content Testing Framework

This directory contains tests for validating the accuracy, completeness, and quality of knowledge content in the Active Inference Knowledge Environment. Knowledge testing ensures that educational materials, research content, and documentation maintain high standards of accuracy and educational value.

## Overview

Knowledge content testing validates the accuracy, completeness, consistency, and educational effectiveness of knowledge repository content. It ensures that all educational materials, research papers, tutorials, and documentation meet established quality standards and provide accurate information.

## Test Categories

### 🎯 Content Accuracy Testing
Tests that validate the factual accuracy of knowledge content:

- **Mathematical Accuracy**: Validation of mathematical formulations and derivations
- **Conceptual Accuracy**: Verification of theoretical concepts and explanations
- **Reference Validation**: Checking accuracy of citations and references
- **Cross-Reference Validation**: Ensuring consistency across related content
- **Fact Checking**: Automated fact-checking against authoritative sources

### 📚 Educational Quality Testing
Tests that validate educational effectiveness:

- **Learning Objective Validation**: Verification that content meets stated learning objectives
- **Prerequisite Validation**: Ensuring prerequisite relationships are accurate
- **Difficulty Assessment**: Validation of difficulty level assignments
- **Content Completeness**: Checking for comprehensive coverage of topics
- **Educational Flow**: Validation of logical content progression

### 🔗 Content Integration Testing
Tests that validate content relationships and integration:

- **Link Validation**: Testing internal and external links
- **Prerequisite Chain Testing**: Validation of learning path prerequisite chains
- **Cross-Reference Testing**: Ensuring consistency across related content
- **Content Relationship Testing**: Validation of content relationships and dependencies
- **Navigation Testing**: Testing content navigation and user experience

### 📖 Documentation Quality Testing
Tests that validate documentation standards:

- **Format Validation**: JSON schema and format compliance
- **Metadata Validation**: Completeness and accuracy of content metadata
- **Structure Validation**: Content structure and organization validation
- **Accessibility Testing**: Content accessibility and usability validation
- **SEO Testing**: Search engine optimization validation

## Getting Started

### Running Knowledge Tests

```bash
# Run all knowledge tests
pytest tests/knowledge/ -v

# Run specific knowledge test categories
pytest tests/knowledge/test_content_accuracy.py -v
pytest tests/knowledge/test_educational_quality.py -v
pytest tests/knowledge/test_content_integration.py -v

# Run knowledge tests with coverage
pytest tests/knowledge/ --cov=src/ --cov-report=html
```

### Writing Knowledge Tests

```python
import pytest
from active_inference.knowledge import KnowledgeRepository, KnowledgeValidator

class TestKnowledgeContentAccuracy:
    """Tests for knowledge content accuracy"""

    @pytest.fixture
    def knowledge_repo(self):
        """Set up knowledge repository for testing"""
        repo = KnowledgeRepository(test_config=True)
        return repo

    @pytest.fixture
    def validator(self):
        """Set up knowledge validator"""
        validator = KnowledgeValidator()
        return validator

    def test_mathematical_accuracy(self, knowledge_repo, validator):
        """Test mathematical accuracy of knowledge content"""
        # Load mathematical content
        math_content = knowledge_repo.get_node('variational_free_energy')

        # Validate mathematical formulations
        math_validation = validator.validate_mathematical_content(math_content)

        assert math_validation['mathematical_accuracy'] > 0.95
        assert len(math_validation['mathematical_errors']) == 0
        assert math_validation['derivations_valid']

    def test_conceptual_accuracy(self, knowledge_repo, validator):
        """Test conceptual accuracy of knowledge content"""
        # Load conceptual content
        concept_content = knowledge_repo.get_node('active_inference_introduction')

        # Validate conceptual explanations
        concept_validation = validator.validate_conceptual_content(concept_content)

        assert concept_validation['conceptual_accuracy'] > 0.90
        assert len(concept_validation['conceptual_errors']) == 0
        assert concept_validation['explanations_clear']

    def test_reference_accuracy(self, knowledge_repo, validator):
        """Test reference and citation accuracy"""
        # Load content with references
        referenced_content = knowledge_repo.get_node('fep_mathematical_formulation')

        # Validate references
        reference_validation = validator.validate_references(referenced_content)

        assert reference_validation['references_valid']
        assert len(reference_validation['broken_references']) == 0
        assert reference_validation['citations_complete']
```

## Test Organization

### Test File Structure
```
tests/knowledge/
├── __init__.py
├── test_content_accuracy.py           # Content accuracy validation
├── test_educational_quality.py        # Educational effectiveness testing
├── test_content_integration.py        # Content relationship testing
├── test_documentation_quality.py     # Documentation standards testing
├── test_learning_paths.py            # Learning path validation
├── test_search_functionality.py       # Knowledge search testing
└── test_content_maintenance.py       # Content maintenance testing
```

### Knowledge Test Patterns

#### Content Accuracy Testing
```python
def test_content_mathematical_accuracy(self, knowledge_repo):
    """Test mathematical accuracy of content"""
    # Arrange
    content_id = 'variational_free_energy'
    expected_formulas = ['F = D_KL[q(θ)||p(θ|x)] - log p(x)']

    # Act
    content = knowledge_repo.get_node(content_id)
    validation = self.validator.validate_mathematical_content(content)

    # Assert
    assert validation['formulas_present'] == expected_formulas
    assert validation['derivations_correct']
    assert validation['mathematical_notation_consistent']
```

#### Educational Quality Testing
```python
def test_learning_objectives_met(self, knowledge_repo):
    """Test that content meets stated learning objectives"""
    # Arrange
    content_id = 'bayesian_basics'
    learning_objectives = [
        'State Bayes theorem and understand its components',
        'Apply conditional probability rules',
        'Calculate posterior probabilities'
    ]

    # Act
    content = knowledge_repo.get_node(content_id)
    objectives_validation = self.validator.validate_learning_objectives(
        content, learning_objectives
    )

    # Assert
    assert objectives_validation['objectives_met'] >= 0.8
    assert all(obj['coverage'] > 0.7 for obj in objectives_validation['objective_coverage'])
```

#### Content Integration Testing
```python
def test_prerequisite_chains(self, knowledge_repo):
    """Test prerequisite relationship chains"""
    # Arrange
    advanced_content = 'active_inference_introduction'
    basic_content = 'bayesian_basics'

    # Act
    chain_validation = self.validator.validate_prerequisite_chain(
        advanced_content, basic_content
    )

    # Assert
    assert chain_validation['chain_valid']
    assert chain_validation['prerequisites_covered']
    assert chain_validation['progression_logical']
```

## Quality Assurance

### Knowledge Testing Standards
- **Accuracy Validation**: >95% accuracy for mathematical and conceptual content
- **Completeness Validation**: >90% coverage of stated learning objectives
- **Consistency Validation**: <5% inconsistencies across related content
- **Link Validation**: 100% valid internal and external links
- **Metadata Validation**: 100% complete and accurate metadata

### Test Data Management
- **Content Fixtures**: Realistic knowledge content for testing
- **Validation Rules**: Comprehensive validation rule sets
- **Test Oracles**: Authoritative sources for accuracy validation
- **Comparison Data**: Benchmark data for quality comparison

## Performance Considerations

### Knowledge Testing Performance
- **Content Loading**: Efficient loading of large knowledge repositories
- **Validation Speed**: Fast validation of content accuracy and quality
- **Search Performance**: Efficient searching across knowledge content
- **Memory Usage**: Optimized memory usage for large content sets
- **Scalability**: Performance scaling with content repository size

### Performance Benchmarks
- **Validation Time**: <1 second per content node
- **Search Time**: <100ms for typical knowledge queries
- **Loading Time**: <5 seconds for complete repository loading
- **Memory Usage**: <500MB for typical repository sizes

## Contributing

### Writing Knowledge Tests
1. **Identify Content Areas**: Find content areas needing validation
2. **Study Content Structure**: Understand knowledge content organization
3. **Design Validation Rules**: Create appropriate validation rules
4. **Implement Test Cases**: Write comprehensive test cases
5. **Validate Test Accuracy**: Ensure test accuracy and reliability

### Knowledge Test Best Practices
- **Content-Specific Testing**: Test content appropriate to content type
- **Validation Rule Accuracy**: Ensure validation rules are accurate
- **Test Data Quality**: Use high-quality test data and fixtures
- **Performance Awareness**: Consider performance impact of tests
- **Documentation**: Document test purpose and validation logic

## Related Documentation

- **[Testing AGENTS.md](../AGENTS.md)**: Testing framework development guidelines
- **[Main AGENTS.md](../../AGENTS.md)**: Project-wide agent guidelines
- **[Contributing Guide](../../CONTRIBUTING.md)**: Contribution processes
- **[Knowledge System](../../../knowledge/)**: Knowledge repository overview

---

*"Active Inference for, with, by Generative AI"* - Ensuring knowledge accuracy through comprehensive content validation and quality assurance testing.
