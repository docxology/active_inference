# Unit Tests

Comprehensive unit testing suite for all Active Inference Knowledge Environment components. These tests validate individual functions, methods, and classes in isolation to ensure correct behavior and maintainability.

## Overview

Unit tests provide the foundation of the testing strategy, validating that individual components work correctly in isolation. This test suite covers all major components including knowledge systems, LLM integrations, research tools, applications, and platform services.

## Directory Structure

```
tests/unit/
├── test_knowledge_*.py      # Knowledge system unit tests
├── test_llm_*.py           # LLM integration unit tests
├── test_research_*.py      # Research tools unit tests
├── test_applications_*.py  # Application framework unit tests
└── test_platform_*.py      # Platform services unit tests
```

## Test Categories

### Knowledge System Tests
**Files**: `test_knowledge_*.py`
- **test_knowledge_repository.py**: Repository functionality and data management
- **test_knowledge_foundations.py**: Foundation concepts validation
- **test_knowledge_implementations.py**: Implementation examples testing
- **test_knowledge_mathematics.py**: Mathematical formulations validation
- **test_knowledge_applications.py**: Application domain testing

**Coverage Areas**:
- Knowledge node creation and validation
- Search functionality and indexing
- Learning path construction
- Cross-reference validation
- Metadata management

### LLM Integration Tests
**Files**: `test_llm_*.py`
- **test_llm_client.py**: LLM client functionality
- **test_llm_conversations.py**: Conversation management
- **test_llm_models.py**: Model registry and selection
- **test_llm_prompts.py**: Prompt engineering and validation

**Coverage Areas**:
- API integration with LLM services
- Conversation state management
- Model selection and configuration
- Prompt template validation
- Error handling and recovery

### Research Tools Tests
**Files**: `test_research_*.py`
- **test_research_data_management.py**: Data collection and preprocessing

**Coverage Areas**:
- Data pipeline validation
- Research workflow orchestration
- Statistical analysis tools
- Experiment management

### Application Framework Tests
**Files**: `test_applications_*.py`
- **test_applications_best_practices.py**: Best practices implementation
- **test_applications_case_studies.py**: Case study validation
- **test_applications_templates.py**: Template system testing

**Coverage Areas**:
- Application template functionality
- Case study accuracy validation
- Best practices implementation
- Integration pattern testing

## Test Patterns and Examples

### Basic Unit Test Pattern
```python
import pytest
from active_inference.knowledge import KnowledgeRepository

class TestKnowledgeRepository:
    """Unit tests for KnowledgeRepository"""

    def setup_method(self):
        """Set up test environment"""
        self.config = {'root_path': './test_knowledge', 'test_mode': True}
        self.repo = KnowledgeRepository(self.config)

    def test_search_knowledge(self):
        """Test knowledge search functionality"""
        # Arrange
        query = "entropy"
        expected_results = 5

        # Act
        results = self.repo.search(query, limit=expected_results)

        # Assert
        assert len(results) <= expected_results
        assert all('id' in result for result in results)
        assert all('title' in result for result in results)
```

### Parameterized Testing
```python
@pytest.mark.parametrize("query,expected_count", [
    ("entropy", 3),
    ("bayesian", 5),
    ("active inference", 2),
    ("", 0)
])
def test_search_with_different_queries(self, query, expected_count):
    """Test search functionality with different query types"""
    # Arrange
    repo = KnowledgeRepository(test_config)

    # Act
    results = repo.search(query, limit=10)

    # Assert
    assert len(results) <= expected_count
```

### Mock Testing Pattern
```python
from unittest.mock import Mock, patch

def test_external_api_integration():
    """Test integration with external services"""
    # Arrange
    repo = KnowledgeRepository(test_config)
    mock_response = {"results": [{"id": "test", "title": "Test Node"}]}

    # Act
    with patch('requests.get') as mock_get:
        mock_get.return_value.json.return_value = mock_response
        results = repo.fetch_external_knowledge("test_query")

    # Assert
    assert len(results) == 1
    assert results[0]["id"] == "test"
    mock_get.assert_called_once()
```

## Running Unit Tests

### Run All Unit Tests
```bash
# Using make (recommended)
make test-unit

# Using pytest directly
pytest tests/unit/ -v -m "unit"

# Using uv
uv run python -m pytest tests/unit/ -v
```

### Run Specific Test Categories
```bash
# Knowledge system tests
pytest tests/unit/test_knowledge_*.py -v

# LLM integration tests
pytest tests/unit/test_llm_*.py -v

# Application tests
pytest tests/unit/test_applications_*.py -v
```

### Run with Coverage
```bash
# Unit test coverage
make test-unit-coverage

# Specific component coverage
pytest tests/unit/test_knowledge_repository.py --cov=src/active_inference/knowledge --cov-report=html
```

## Test Data and Fixtures

### Using Test Fixtures
```python
import pytest
from tests.fixtures.test_data import create_test_knowledge_node

@pytest.fixture
def sample_knowledge_node():
    """Sample knowledge node for testing"""
    return create_test_knowledge_node({
        'id': 'test_node',
        'title': 'Test Node',
        'difficulty': 'beginner'
    })

def test_knowledge_node_validation(sample_knowledge_node):
    """Test knowledge node validation"""
    from tests.utilities.test_helpers import assert_knowledge_node_structure

    assert_knowledge_node_structure(sample_knowledge_node)
    assert sample_knowledge_node['difficulty'] == 'beginner'
```

### Creating Custom Fixtures
```python
@pytest.fixture
def knowledge_repository_with_data():
    """Knowledge repository with test data loaded"""
    config = {'root_path': './test_knowledge', 'test_mode': True}
    repo = KnowledgeRepository(config)

    # Load test data
    test_nodes = [
        create_test_knowledge_node({'id': f'node_{i}', 'title': f'Test Node {i}'})
        for i in range(5)
    ]

    for node in test_nodes:
        repo.add_node(node)

    return repo
```

## Best Practices

### Test Organization
1. **Arrange-Act-Assert**: Structure tests with clear setup, execution, and validation
2. **Descriptive Names**: Use descriptive test method names that explain purpose
3. **Independent Tests**: Ensure tests don't depend on each other
4. **Fast Execution**: Keep unit tests fast (<1 second each)
5. **Comprehensive Coverage**: Test both success and failure scenarios

### Test Data Management
1. **Realistic Data**: Use realistic test data that represents actual usage
2. **Edge Cases**: Test boundary conditions and edge cases
3. **Data Isolation**: Ensure test data doesn't interfere between tests
4. **Cleanup**: Properly clean up test data and resources

### Mocking Strategy
1. **External Dependencies**: Mock external APIs and services
2. **Time Dependencies**: Mock time and date functions
3. **File System**: Mock file operations for predictable testing
4. **Network**: Mock network requests for reliable testing

## Coverage Targets

### Component Coverage Goals
- **Knowledge System**: >95% coverage
- **LLM Integration**: >90% coverage
- **Research Tools**: >85% coverage
- **Applications**: >90% coverage
- **Overall Unit Tests**: >90% coverage

### Measuring Coverage
```bash
# Generate coverage report
pytest tests/unit/ --cov=src/ --cov-report=html --cov-report=term-missing

# Check coverage gaps
pytest tests/unit/ --cov=src/ --cov-report=term-missing --cov-fail-under=90
```

## Contributing

### Adding New Unit Tests
1. **Follow Naming Convention**: Use `test_component_feature.py` naming pattern
2. **Include Docstrings**: Add comprehensive docstrings explaining test purpose
3. **Test Edge Cases**: Include tests for boundary conditions and error cases
4. **Use Fixtures**: Leverage existing fixtures or create reusable ones
5. **Maintain Coverage**: Ensure new code has adequate test coverage

### Test Review Checklist
- [ ] **Purpose Clear**: Test purpose and expected behavior are clear
- [ ] **Coverage Complete**: Tests cover all relevant scenarios
- [ ] **Edge Cases**: Boundary conditions and error cases tested
- [ ] **Performance**: Tests don't significantly impact test suite performance
- [ ] **Documentation**: Tests are well-documented and understandable

## Related Documentation

- **[Main Testing README](../README.md)**: Comprehensive testing framework overview
- **[Test Fixtures](../fixtures/README.md)**: Test data and fixture documentation
- **[Test Utilities](../utilities/README.md)**: Testing utilities and helper functions
- **[Integration Tests](../integration/README.md)**: Component integration testing
- **[Knowledge Tests](../knowledge/README.md)**: Knowledge content validation

---

*"Active Inference for, with, by Generative AI"* - Ensuring component reliability through comprehensive unit testing and validation.

