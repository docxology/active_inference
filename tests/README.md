# Test Suite

This directory contains comprehensive test suites for the Active Inference Knowledge Environment, including unit tests, integration tests, knowledge validation tests, and performance tests. The test suite ensures code quality, functionality, and reliability across all components.

## Overview

The Test Suite module provides comprehensive testing frameworks and utilities for validating all aspects of the Active Inference Knowledge Environment. Tests cover unit functionality, component integration, knowledge content validation, performance characteristics, and system reliability.

## Directory Structure

```
tests/
├── unit/                 # Unit tests for individual components
├── integration/          # Integration tests for component interaction
├── knowledge/            # Knowledge content validation tests
├── performance/          # Performance and scalability tests
├── fixtures/             # Test data and fixtures
└── utilities/            # Testing utilities and helpers
```

## Core Components

### 🧪 Unit Tests
- **Component Tests**: Individual function and method testing
- **Class Tests**: Complete class functionality testing
- **Utility Tests**: Helper function and utility testing
- **Edge Case Tests**: Boundary condition and error case testing
- **Mock Tests**: Testing with mocked dependencies

### 🔗 Integration Tests
- **Component Integration**: Testing component interactions
- **API Integration**: Testing API endpoints and services
- **Data Flow**: Testing data flow between components
- **System Integration**: End-to-end system functionality
- **External Integration**: Testing external system integration

### 📚 Knowledge Tests
- **Content Validation**: Knowledge content accuracy testing
- **Prerequisite Testing**: Learning path prerequisite validation
- **Cross-Reference Testing**: Internal link and reference validation
- **Format Testing**: Content format and structure validation
- **Completeness Testing**: Content completeness assessment

### ⚡ Performance Tests
- **Load Testing**: System performance under load
- **Stress Testing**: System limits and failure modes
- **Scalability Testing**: Performance scaling characteristics
- **Memory Testing**: Memory usage and leak testing
- **Timing Tests**: Execution time and latency testing

## Getting Started

### For Developers
1. **Run Test Suite**: Execute comprehensive test suite
2. **Write Tests**: Follow Test-Driven Development (TDD) approach
3. **Test Coverage**: Ensure adequate test coverage
4. **Continuous Integration**: Set up automated testing
5. **Debugging**: Use test suite for debugging and validation

### For Contributors
1. **Understand Testing Framework**: Learn testing patterns and conventions
2. **Test Requirements**: Understand testing requirements for contributions
3. **Test Implementation**: Implement comprehensive tests for new features
4. **Test Maintenance**: Maintain and update existing tests
5. **Quality Assurance**: Ensure code quality through testing

## Usage Examples

### Running Tests
```bash
# Run all tests
make test

# Run specific test types
make test-unit
make test-integration
make test-knowledge

# Run tests with coverage
pytest tests/ --cov=src/ --cov-report=html

# Run specific test files
pytest tests/unit/test_knowledge_repository.py -v
pytest tests/integration/test_api_integration.py -v
```

### Writing Unit Tests
```python
import pytest
from active_inference.knowledge import KnowledgeRepository

class TestKnowledgeRepository:
    """Unit tests for KnowledgeRepository"""

    def setup_method(self):
        """Set up test environment"""
        self.repo_config = {'root_path': './test_knowledge'}
        self.repo = KnowledgeRepository(self.repo_config)

    def teardown_method(self):
        """Clean up test environment"""
        # Clean up test files
        pass

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

    def test_add_knowledge_node(self):
        """Test adding knowledge node"""
        # Arrange
        node_data = {
            'id': 'test_node',
            'title': 'Test Node',
            'content': 'Test content',
            'type': 'concept'
        }

        # Act
        result = self.repo.add_node(node_data)

        # Assert
        assert result is not None
        assert self.repo.get_node('test_node') is not None

    def test_invalid_search_query(self):
        """Test error handling for invalid search"""
        # Arrange
        invalid_query = ""

        # Act & Assert
        with pytest.raises(ValueError):
            self.repo.search(invalid_query)
```

### Integration Testing
```python
import pytest
from active_inference.platform import PlatformServer
from active_inference.knowledge import KnowledgeRepository

class TestPlatformIntegration:
    """Integration tests for platform components"""

    def setup_method(self):
        """Set up integration test environment"""
        self.server_config = {'test_mode': True, 'port': 8001}
        self.server = PlatformServer(self.server_config)
        self.knowledge_repo = KnowledgeRepository({'test_mode': True})

    def teardown_method(self):
        """Clean up integration test environment"""
        self.server.shutdown()

    def test_knowledge_api_integration(self):
        """Test knowledge API integration"""
        # Arrange
        test_node = {
            'id': 'integration_test_node',
            'title': 'Integration Test',
            'content': 'Test content for integration'
        }

        # Act
        self.knowledge_repo.add_node(test_node)
        api_response = self.server.get_knowledge_node('integration_test_node')

        # Assert
        assert api_response['id'] == test_node['id']
        assert api_response['title'] == test_node['title']

    def test_search_api_integration(self):
        """Test search API integration"""
        # Arrange
        test_content = "Active Inference entropy information theory"

        # Act
        search_results = self.server.search_knowledge(test_content, limit=5)

        # Assert
        assert 'results' in search_results
        assert len(search_results['results']) <= 5
```

## Testing Standards

### Test Coverage
- **Unit Tests**: >95% coverage for core functionality
- **Integration Tests**: >80% coverage for component interactions
- **Knowledge Tests**: 100% coverage for knowledge content
- **Performance Tests**: Critical path performance validation
- **Edge Cases**: Comprehensive edge case coverage

### Test Quality
- **Clear Tests**: Descriptive test names and assertions
- **Isolated Tests**: Independent tests with proper setup/teardown
- **Fast Tests**: Efficient test execution
- **Reliable Tests**: Deterministic and repeatable tests
- **Maintainable Tests**: Well-organized and documented tests

## Contributing

We welcome contributions to the test suite! See [CONTRIBUTING.md](../../CONTRIBUTING.md) for detailed guidelines.

### Contribution Types
- **New Tests**: Add tests for new functionality
- **Test Improvements**: Enhance existing test coverage
- **Test Utilities**: Create testing utilities and helpers
- **Performance Tests**: Add performance and scalability tests
- **Integration Tests**: Add integration and system tests

### Quality Standards
- **Test Completeness**: Tests must cover all functionality
- **Test Clarity**: Tests must be clear and understandable
- **Test Reliability**: Tests must be reliable and repeatable
- **Test Performance**: Tests must execute efficiently
- **Test Documentation**: Tests must be well-documented

## Learning Resources

- **Testing Framework**: Learn pytest and testing best practices
- **TDD Principles**: Study Test-Driven Development methodologies
- **Testing Patterns**: Learn established testing patterns
- **Code Coverage**: Understand test coverage analysis
- **Integration Testing**: Master integration testing techniques

## Related Documentation

- **[Main README](../../README.md)**: Project overview and getting started
- **[Development Guide](../../CONTRIBUTING.md)**: Development guidelines
- **[Code Quality](../../applications/best_practices/)**: Code quality standards
- **[Platform Documentation](../../platform/)**: Platform testing
- **[Knowledge Testing](../../tests/knowledge/)**: Knowledge validation

## Testing Infrastructure

### Test Configuration
```python
# pytest.ini configuration
[tool:pytest]
minversion = 6.0
addopts = -ra -q --strict-markers --strict-config
testpaths = tests
python_files = test_*.py *_test.py
python_classes = Test*
python_functions = test_*
markers =
    slow: marks tests as slow (deselect with '-m "not slow"')
    integration: marks tests as integration tests
    unit: marks tests as unit tests
    knowledge: marks tests as knowledge repository tests
    performance: marks tests as performance tests
filterwarnings =
    ignore::DeprecationWarning
    ignore::PendingDeprecationWarning
```

### Test Fixtures
```python
import pytest
from active_inference.knowledge import KnowledgeRepository

@pytest.fixture
def knowledge_repository():
    """Fixture for knowledge repository testing"""
    config = {'root_path': './test_knowledge', 'test_mode': True}
    repo = KnowledgeRepository(config)
    yield repo
    # Cleanup after test
    repo.cleanup()

@pytest.fixture
def sample_knowledge_data():
    """Fixture for sample knowledge data"""
    return {
        'id': 'test_concept',
        'title': 'Test Concept',
        'content': 'Test content for testing',
        'type': 'concept',
        'difficulty': 'beginner'
    }
```

---

*"Active Inference for, with, by Generative AI"* - Ensuring reliability through comprehensive testing, quality assurance, and continuous validation.
