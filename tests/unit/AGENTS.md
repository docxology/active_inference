# Unit Testing - Agent Documentation

This document provides comprehensive guidance for AI agents and contributors working within the Unit Testing module of the Active Inference Knowledge Environment. It outlines unit testing methodologies, implementation patterns, and best practices for validating individual components and ensuring code quality.

## Unit Testing Module Overview

The Unit Testing module provides comprehensive unit testing for all Active Inference Knowledge Environment components. These tests validate individual functions, methods, and classes in isolation to ensure correct behavior, maintainability, and reliability. Unit tests form the foundation of the testing strategy and enable confident refactoring and development.

## Directory Structure

```
tests/unit/
├── test_knowledge_*.py      # Knowledge system unit tests
│   ├── test_knowledge_repository.py     # Repository functionality
│   ├── test_knowledge_foundations.py   # Foundation concepts
│   ├── test_knowledge_implementations.py # Implementation examples
│   ├── test_knowledge_mathematics.py   # Mathematical formulations
│   └── test_knowledge_applications.py  # Application domains
├── test_llm_*.py           # LLM integration unit tests
│   ├── test_llm_client.py       # LLM client functionality
│   ├── test_llm_conversations.py # Conversation management
│   ├── test_llm_models.py       # Model registry and selection
│   └── test_llm_prompts.py      # Prompt engineering
├── test_research_*.py      # Research tools unit tests
│   └── test_research_data_management.py # Data pipeline validation
└── test_applications_*.py  # Application framework unit tests
    ├── test_applications_best_practices.py # Best practices
    ├── test_applications_case_studies.py   # Case study validation
    └── test_applications_templates.py      # Template system testing
```

## Core Responsibilities

### Unit Test Development
- **Component Testing**: Create comprehensive unit tests for all system components
- **Edge Case Coverage**: Test boundary conditions, error cases, and unusual scenarios
- **Mock Implementation**: Develop appropriate mocks for external dependencies
- **Test Maintenance**: Keep tests current with code changes and refactoring

### Test Strategy Implementation
- **Coverage Analysis**: Ensure adequate test coverage for all critical paths
- **Test Organization**: Maintain logical test organization and categorization
- **Performance Testing**: Validate unit test performance and execution speed
- **Integration Testing**: Support integration between unit tests and larger test suites

### Quality Assurance
- **Test Reliability**: Ensure tests are deterministic and don't have flaky behavior
- **Code Quality**: Maintain high-quality test code following established standards
- **Documentation**: Document test purpose, setup, and expected outcomes
- **Review Process**: Participate in test review and validation processes

## Development Workflows

### Unit Test Creation Process
1. **Requirements Analysis**: Understand component functionality and testing requirements
2. **Test Planning**: Identify test cases, edge cases, and error scenarios
3. **Test Implementation**: Write comprehensive unit tests following TDD principles
4. **Mock Development**: Create appropriate mocks for external dependencies
5. **Test Execution**: Run tests and validate coverage and correctness
6. **Integration**: Integrate tests into the larger test suite
7. **Maintenance**: Update tests as component functionality evolves

### Test-Driven Development (TDD)
1. **Write Test First**: Create failing tests before implementing functionality
2. **Implement Minimal Code**: Implement only enough code to pass the tests
3. **Refactor**: Improve code quality while maintaining test compatibility
4. **Test Expansion**: Add additional tests for edge cases and error conditions
5. **Integration**: Ensure tests integrate properly with existing test suite

### Test Maintenance Process
1. **Code Change Analysis**: Identify when component changes require test updates
2. **Test Updates**: Modify tests to match new functionality
3. **Regression Testing**: Ensure existing tests still pass after changes
4. **Coverage Validation**: Verify test coverage remains adequate
5. **Performance Monitoring**: Monitor test execution performance

## Quality Standards

### Test Quality Standards
- **Coverage**: >95% coverage for core components, >90% overall for unit tests
- **Reliability**: <1% flaky test rate, deterministic test execution
- **Performance**: Unit tests complete in <1 second each on average
- **Maintainability**: Clear, well-documented tests that are easy to understand
- **Completeness**: Tests cover all public interfaces and critical paths

### Code Quality Standards
- **Style Compliance**: Follow PEP 8 and project coding standards
- **Documentation**: Comprehensive docstrings and comments in test code
- **Structure**: Clear Arrange-Act-Assert pattern in all tests
- **Error Handling**: Proper exception testing and error scenario coverage
- **Best Practices**: Follow testing best practices and established patterns

## Implementation Patterns

### Base Unit Test Pattern
```python
import pytest
from typing import Dict, Any, Optional
from unittest.mock import Mock, patch
import logging

logger = logging.getLogger(__name__)

class BaseComponentTest:
    """Base class for component unit tests"""

    def __init__(self, component_name: str):
        """Initialize component test"""
        self.component_name = component_name
        self.component_under_test = None
        self.test_config = self.get_test_config()

    def setup_method(self, method):
        """Set up test environment before each test"""
        self.setup_component()
        logger.info(f"Setting up test for {self.component_name}.{method}")

    def teardown_method(self, method):
        """Clean up test environment after each test"""
        self.cleanup_component()
        logger.info(f"Tearing down test for {self.component_name}.{method}")

    def setup_component(self) -> None:
        """Set up component under test"""
        self.component_under_test = self.create_component(self.test_config)

    def cleanup_component(self) -> None:
        """Clean up component after test"""
        if hasattr(self.component_under_test, 'cleanup'):
            self.component_under_test.cleanup()
        self.component_under_test = None

    @abstractmethod
    def get_test_config(self) -> Dict[str, Any]:
        """Get test configuration"""
        pass

    @abstractmethod
    def create_component(self, config: Dict[str, Any]) -> Any:
        """Create component instance for testing"""
        pass

class KnowledgeRepositoryTest(BaseComponentTest):
    """Unit tests for KnowledgeRepository"""

    def get_test_config(self) -> Dict[str, Any]:
        """Get test configuration for knowledge repository"""
        return {
            'root_path': './test_knowledge',
            'test_mode': True,
            'enable_caching': False,
            'validate_on_access': True
        }

    def create_component(self, config: Dict[str, Any]) -> Any:
        """Create knowledge repository instance"""
        from active_inference.knowledge import KnowledgeRepository
        return KnowledgeRepository(config)

    def test_initialization(self):
        """Test knowledge repository initialization"""
        # Arrange
        expected_attributes = ['config', 'root_path', 'nodes', 'search_index']

        # Act & Assert
        for attr in expected_attributes:
            assert hasattr(self.component_under_test, attr), f"Missing attribute: {attr}"

        # Verify configuration
        assert self.component_under_test.config['test_mode'] == True

    def test_node_creation(self):
        """Test knowledge node creation"""
        # Arrange
        test_node = {
            'id': 'test_node',
            'title': 'Test Node',
            'content_type': 'foundation',
            'difficulty': 'beginner',
            'description': 'Test description',
            'content': {'overview': 'Test overview'},
            'metadata': {'author': 'test', 'version': '1.0'}
        }

        # Act
        result = self.component_under_test.add_node(test_node)

        # Assert
        assert result['success'] == True
        assert result['node_id'] == 'test_node'
        assert 'test_node' in self.component_under_test.nodes

    def test_node_validation(self):
        """Test knowledge node validation"""
        # Arrange
        invalid_node = {
            'id': '',  # Invalid: empty ID
            'title': 'Test Node'
        }

        # Act & Assert
        with pytest.raises(ValidationError):
            self.component_under_test.add_node(invalid_node)

    def test_search_functionality(self):
        """Test search functionality"""
        # Arrange
        self._setup_test_data()

        # Act
        results = self.component_under_test.search('entropy', limit=5)

        # Assert
        assert len(results) <= 5
        assert all('id' in result for result in results)
        assert all('title' in result for result in results)
        assert all('score' in result for result in results)

    def _setup_test_data(self) -> None:
        """Set up test data for search tests"""
        test_nodes = [
            {
                'id': 'entropy_basics',
                'title': 'Entropy Basics',
                'content': {'overview': 'Introduction to entropy and information theory'}
            },
            {
                'id': 'information_theory',
                'title': 'Information Theory',
                'content': {'overview': 'Comprehensive information theory concepts'}
            }
        ]

        for node in test_nodes:
            self.component_under_test.add_node(node)
```

### Mock Testing Pattern
```python
class LLMApiTest(BaseComponentTest):
    """Unit tests for LLM API integration"""

    def get_test_config(self) -> Dict[str, Any]:
        """Get test configuration for LLM API"""
        return {
            'api_key': 'test_key',
            'model': 'test_model',
            'test_mode': True,
            'timeout': 10
        }

    def create_component(self, config: Dict[str, Any]) -> Any:
        """Create LLM API client for testing"""
        from active_inference.llm import LLMClient
        return LLMClient(config)

    @patch('requests.post')
    def test_api_call_success(self, mock_post):
        """Test successful API call"""
        # Arrange
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'choices': [{'message': {'content': 'Test response'}}]
        }
        mock_post.return_value = mock_response

        # Act
        result = self.component_under_test.make_request('test prompt')

        # Assert
        assert result == 'Test response'
        mock_post.assert_called_once()

    @patch('requests.post')
    def test_api_call_failure(self, mock_post):
        """Test API call failure handling"""
        # Arrange
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.raise_for_status.side_effect = HTTPError('Server error')
        mock_post.return_value = mock_response

        # Act & Assert
        with pytest.raises(HTTPError):
            self.component_under_test.make_request('test prompt')

    @patch('requests.post')
    def test_api_timeout(self, mock_post):
        """Test API timeout handling"""
        # Arrange
        mock_post.side_effect = Timeout('Request timeout')

        # Act & Assert
        with pytest.raises(Timeout):
            self.component_under_test.make_request('test prompt', timeout=1)
```

### Parameterized Testing Pattern
```python
class DataValidationTest(BaseComponentTest):
    """Unit tests for data validation"""

    @pytest.mark.parametrize("test_input,expected_output", [
        ("valid_string", True),
        ("", False),
        (None, False),
        ("   ", False),
        ("valid_string_with_spaces", True)
    ])
    def test_string_validation(self, test_input, expected_output):
        """Test string validation with various inputs"""
        # Act
        result = self.component_under_test.validate_string(test_input)

        # Assert
        assert result == expected_output

    @pytest.mark.parametrize("node_data,is_valid", [
        # Valid nodes
        ({
            'id': 'test_node',
            'title': 'Test Node',
            'content_type': 'foundation',
            'difficulty': 'beginner'
        }, True),

        # Invalid nodes - missing required fields
        ({
            'title': 'Test Node',
            'content_type': 'foundation'
        }, False),

        ({
            'id': 'test_node',
            'content_type': 'foundation'
        }, False),

        # Invalid nodes - invalid values
        ({
            'id': 'test_node',
            'title': 'Test Node',
            'content_type': 'invalid_type',
            'difficulty': 'beginner'
        }, False)
    ])
    def test_node_validation(self, node_data, is_valid):
        """Test knowledge node validation"""
        # Act
        result = self.component_under_test.validate_node(node_data)

        # Assert
        assert result['valid'] == is_valid
        if not is_valid:
            assert len(result['errors']) > 0
```

## Testing Guidelines

### Unit Testing Best Practices
1. **Test One Thing**: Each test should validate a single piece of functionality
2. **Independent Tests**: Tests should not depend on each other or external state
3. **Fast Execution**: Unit tests should execute quickly (<1 second each)
4. **Clear Structure**: Use Arrange-Act-Assert pattern for clear test structure
5. **Descriptive Names**: Use descriptive test method names that explain the test purpose

### Test Data Management
1. **Realistic Data**: Use realistic test data that represents actual usage patterns
2. **Edge Cases**: Test boundary conditions, empty inputs, and error cases
3. **Data Isolation**: Ensure test data doesn't interfere between tests
4. **Cleanup**: Properly clean up test data and temporary resources

### Mocking Strategy
1. **External Dependencies**: Mock external APIs, databases, and services
2. **Time Dependencies**: Mock time, date, and timing functions
3. **File System**: Mock file operations for predictable and fast testing
4. **Network Operations**: Mock network requests for reliable testing

## Performance Considerations

### Test Execution Performance
- **Parallel Execution**: Run tests in parallel when possible
- **Fast Setup/Teardown**: Minimize test setup and teardown time
- **Efficient Assertions**: Use efficient assertion methods
- **Resource Cleanup**: Ensure proper cleanup of test resources

### Memory Management
- **Resource Cleanup**: Clean up resources after each test
- **Memory Leak Prevention**: Monitor and prevent memory leaks in tests
- **Large Data Handling**: Use streaming or sampling for large test datasets
- **Mock Optimization**: Use efficient mocking strategies

## Getting Started as an Agent

### Unit Test Development Setup
1. **Study Component Architecture**: Understand the components you need to test
2. **Learn Test Patterns**: Study existing unit tests for patterns and conventions
3. **Set Up Test Environment**: Configure development environment for testing
4. **Run Existing Tests**: Ensure all existing tests pass before making changes
5. **Understand Test Structure**: Learn the Arrange-Act-Assert pattern

### Test Development Process
1. **Identify Test Requirements**: Analyze component functionality and identify test cases
2. **Write Test Plan**: Create detailed test plan with edge cases and error scenarios
3. **Implement Tests**: Follow TDD with comprehensive test coverage
4. **Add Mocking**: Implement appropriate mocks for external dependencies
5. **Validate Coverage**: Ensure adequate test coverage for all critical paths
6. **Integration**: Ensure tests integrate with existing test suite

### Quality Assurance
1. **Test Completeness**: Verify tests cover all component functionality
2. **Edge Case Coverage**: Ensure boundary conditions and error cases are tested
3. **Performance Validation**: Verify tests don't significantly impact test suite performance
4. **Documentation**: Ensure tests are well-documented and understandable
5. **Maintenance**: Plan for ongoing test maintenance and updates

## Common Challenges and Solutions

### Challenge: Complex Component Dependencies
**Solution**: Use comprehensive mocking and dependency injection to isolate components for testing.

### Challenge: Asynchronous Code Testing
**Solution**: Use pytest-asyncio for async testing and ensure proper async/await patterns.

### Challenge: Test Data Management
**Solution**: Use fixtures and factory patterns to create consistent, reusable test data.

### Challenge: Test Performance
**Solution**: Optimize test setup/teardown, use efficient assertions, and run tests in parallel when possible.

### Challenge: Test Maintenance
**Solution**: Keep tests focused, well-documented, and refactor tests when components change.

## Related Documentation

- **[Main Testing README](../README.md)**: Comprehensive testing framework overview
- **[Unit Tests README](./README.md)**: Unit testing module documentation
- **[Test Fixtures](../fixtures/README.md)**: Test data and fixture management
- **[Test Utilities](../utilities/README.md)**: Testing utilities and helper functions
- **[Integration Tests](../integration/README.md)**: Component integration testing
- **[Knowledge Tests](../knowledge/README.md)**: Knowledge content validation

---

*"Active Inference for, with, by Generative AI"* - Ensuring component reliability through comprehensive unit testing and validation.



