# Test Utilities - Agent Documentation

This document provides comprehensive guidance for AI agents and contributors working within the Test Utilities module of the Active Inference Knowledge Environment. It outlines testing utility development, implementation patterns, and best practices for creating and maintaining testing support tools.

## Test Utilities Module Overview

The Test Utilities module provides essential supporting functionality for the testing framework, including helper functions, mock utilities, test data generators, and testing infrastructure. These utilities enable efficient test development, execution, and maintenance across all test categories while ensuring consistency and reliability.

## Directory Structure

```
tests/utilities/
├── __init__.py           # Utilities module initialization
└── test_helpers.py       # Core testing utilities and helper functions
```

## Core Responsibilities

### Test Utility Development
- **Helper Function Creation**: Develop reusable helper functions for common testing tasks
- **Mock Implementation**: Create comprehensive mock utilities for external dependencies
- **Test Data Generation**: Build utilities for creating realistic test data
- **Resource Management**: Implement utilities for managing test resources and cleanup

### Test Infrastructure Support
- **Fixture Development**: Create reusable test fixtures and setup utilities
- **Validation Helpers**: Build assertion helpers and validation utilities
- **Performance Testing Tools**: Develop utilities for performance measurement and analysis
- **Error Simulation**: Create utilities for simulating various error conditions

### Quality Assurance
- **Utility Testing**: Ensure test utilities themselves are well-tested
- **Documentation**: Maintain comprehensive documentation for all utilities
- **Integration**: Ensure utilities integrate seamlessly with testing framework
- **Maintenance**: Keep utilities current with testing framework evolution

## Development Workflows

### Test Utility Creation Process
1. **Requirements Analysis**: Identify common testing patterns and utility requirements
2. **Design Interface**: Design clean, intuitive utility interfaces
3. **Implementation**: Implement utilities following established patterns
4. **Testing**: Create comprehensive tests for utilities themselves
5. **Documentation**: Provide detailed documentation and usage examples
6. **Integration**: Ensure utilities integrate with existing test framework
7. **Validation**: Validate utility effectiveness and performance

### Utility Maintenance Process
1. **Usage Analysis**: Monitor utility usage patterns and identify improvements
2. **Performance Monitoring**: Track utility performance and resource usage
3. **Update Implementation**: Improve utilities based on feedback and requirements
4. **Compatibility Testing**: Ensure utilities work with framework updates
5. **Documentation Updates**: Keep utility documentation current

## Quality Standards

### Utility Quality Standards
- **Reliability**: Utilities should be deterministic and produce consistent results
- **Performance**: Utilities should not significantly impact test execution performance
- **Usability**: Utilities should have intuitive interfaces and clear documentation
- **Maintainability**: Utilities should be easy to understand and modify
- **Testability**: Utilities themselves should be well-tested and validated

### Code Quality Standards
- **Documentation**: Comprehensive docstrings and usage examples
- **Error Handling**: Proper error handling and informative error messages
- **Type Safety**: Complete type annotations for all utility functions
- **Style Compliance**: Follow PEP 8 and project coding standards
- **Best Practices**: Follow software engineering best practices

## Implementation Patterns

### Test Data Generation Pattern
```python
from typing import Dict, Any, List, Optional
import uuid
from datetime import datetime

def create_test_knowledge_node(overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Create a standardized test knowledge node.

    This utility generates realistic test data for knowledge nodes while allowing
    customization through overrides.

    Args:
        overrides: Optional dictionary to override default values

    Returns:
        Complete knowledge node dictionary for testing

    Examples:
        >>> node = create_test_knowledge_node({'difficulty': 'advanced'})
        >>> assert node['difficulty'] == 'advanced'
        >>> assert 'id' in node
        >>> assert 'content' in node
    """
    base_node = {
        'id': generate_unique_test_id('node'),
        'title': 'Test Knowledge Node',
        'content_type': 'foundation',
        'difficulty': 'beginner',
        'description': 'Test description for unit tests',
        'prerequisites': [],
        'tags': ['test', 'unit-test'],
        'learning_objectives': ['Test learning objective'],
        'content': {
            'overview': 'Test overview content',
            'mathematical_definition': 'Test mathematical content',
            'examples': ['Test example'],
            'interactive_exercises': []
        },
        'metadata': {
            'estimated_reading_time': 5,
            'author': 'test_author',
            'last_updated': datetime.now().isoformat(),
            'version': '1.0'
        }
    }

    if overrides:
        # Deep merge for nested dictionaries
        base_node = deep_merge(base_node, overrides)

    return base_node

def generate_unique_test_id(prefix: str = 'test') -> str:
    """
    Generate unique test identifier.

    Creates a unique identifier suitable for use in tests, ensuring no
    collisions between different test runs.

    Args:
        prefix: Prefix for the generated ID

    Returns:
        Unique identifier string

    Examples:
        >>> id1 = generate_unique_test_id('experiment')
        >>> id2 = generate_unique_test_id('experiment')
        >>> assert id1 != id2
        >>> assert id1.startswith('experiment_')
    """
    return f"{prefix}_{uuid.uuid4().hex[:8]}"

def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Deep merge two dictionaries"""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result
```

### Mock Utility Pattern
```python
from unittest.mock import Mock, AsyncMock
from typing import Dict, Any, Optional
import pytest

def create_mock_ollama_response(response_data: Optional[Dict[str, Any]] = None) -> Mock:
    """
    Create mock Ollama API response.

    Generates a realistic mock response for Ollama API calls, useful for
    testing LLM integrations without making actual API calls.

    Args:
        response_data: Custom response data to include in mock

    Returns:
        Mock response object with realistic behavior

    Examples:
        >>> response = create_mock_ollama_response({'response': 'test'})
        >>> response.status_code
        200
        >>> response.json()
        {'response': 'test', 'model': 'test_model'}
    """
    if response_data is None:
        response_data = {
            'response': 'Mocked response content',
            'model': 'test_model',
            'usage': {'tokens': 15}
        }

    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = response_data
    mock_response.raise_for_status.return_value = None
    mock_response.text = str(response_data)

    return mock_response

def patch_llm_client_success():
    """
    Context manager for patching LLM client with success response.

    Provides a convenient way to mock successful LLM API calls in tests.

    Returns:
        Context manager for patching

    Examples:
        >>> def test_successful_request():
        ...     with patch_llm_client_success() as mock_client:
        ...         response = make_llm_request('test prompt')
        ...         assert response == 'Mocked response content'
    """
    from unittest.mock import patch

    mock_response = create_mock_ollama_response({
        'response': 'Mocked successful response',
        'model': 'test_model',
        'usage': {'tokens': 10}
    })

    return patch('ollama.generate', return_value=mock_response)

def patch_llm_client_error(error_type: str = 'network'):
    """
    Context manager for patching LLM client with error response.

    Provides a convenient way to test error handling in LLM integrations.

    Args:
        error_type: Type of error to simulate ('network', 'timeout', 'api')

    Returns:
        Context manager for patching

    Examples:
        >>> def test_error_handling():
        ...     with patch_llm_client_error('timeout') as mock_client:
        ...         with pytest.raises(LLMTimeoutError):
        ...             make_llm_request('test prompt')
    """
    from unittest.mock import patch
    from active_inference.llm import LLMError, LLMTimeoutError, LLMAPIError

    error_classes = {
        'network': LLMError,
        'timeout': LLMTimeoutError,
        'api': LLMAPIError
    }

    error_class = error_classes.get(error_type, LLMError)

    return patch('ollama.generate', side_effect=error_class(f"Mocked {error_type} error"))

def create_async_mock_client() -> AsyncMock:
    """
    Create async mock client for testing async functionality.

    Generates an async mock client that can be used to test async
    LLM integrations and other async operations.

    Returns:
        AsyncMock client with realistic async behavior

    Examples:
        >>> client = create_async_mock_client()
        >>> response = await client.generate_async('test prompt')
        >>> assert response['response'] == 'Mocked async response'
    """
    mock_client = AsyncMock()

    # Configure default responses
    mock_client.generate_async.return_value = {
        'response': 'Mocked async response',
        'model': 'async_test_model',
        'usage': {'tokens': 8}
    }

    mock_client.chat_async.return_value = {
        'message': {'content': 'Mocked chat response'},
        'usage': {'tokens': 12}
    }

    return mock_client
```

### Resource Management Pattern
```python
import tempfile
from pathlib import Path
from contextlib import contextmanager
from typing import Generator, Optional

@contextmanager
def create_temp_file_with_content(content: str, suffix: str = '.json') -> Generator[Path, None, None]:
    """
    Create temporary file with specified content.

    Context manager that creates a temporary file with given content and
    automatically cleans it up after use.

    Args:
        content: Content to write to temporary file
        suffix: File extension for temporary file

    Yields:
        Path to created temporary file

    Examples:
        >>> with create_temp_file_with_content('{"test": "data"}') as temp_file:
        ...     assert temp_file.exists()
        ...     data = json.load(temp_file.open())
        ...     assert data['test'] == 'data'
        # File automatically cleaned up
    """
    with tempfile.NamedTemporaryFile(mode='w', suffix=suffix, delete=False) as temp_file:
        temp_file.write(content)
        temp_file_path = Path(temp_file.name)

    try:
        yield temp_file_path
    finally:
        if temp_file_path.exists():
            temp_file_path.unlink()

@contextmanager
def create_temp_directory() -> Generator[Path, None, None]:
    """
    Create temporary directory.

    Context manager that creates a temporary directory and automatically
    cleans it up after use, including all contained files.

    Yields:
        Path to created temporary directory

    Examples:
        >>> with create_temp_directory() as temp_dir:
        ...     test_file = temp_dir / 'test.txt'
        ...     test_file.write_text('test content')
        ...     assert test_file.exists()
        ...     assert temp_dir.exists()
        # Directory and contents automatically cleaned up
    """
    import shutil

    temp_dir = Path(tempfile.mkdtemp())

    try:
        yield temp_dir
    finally:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)

def cleanup_temp_files(*files: Path) -> None:
    """
    Clean up temporary files.

    Utility function to clean up multiple temporary files, handling
    cases where files might not exist.

    Args:
        *files: Variable number of Path objects to clean up

    Examples:
        >>> temp_file1 = Path('/tmp/test1.txt')
        >>> temp_file2 = Path('/tmp/test2.txt')
        >>> temp_file1.write_text('test')
        >>> cleanup_temp_files(temp_file1, temp_file2)  # temp_file2 doesn't exist
        >>> assert not temp_file1.exists()
    """
    for file in files:
        if file.exists():
            file.unlink()
```

### Validation Helper Pattern
```python
from typing import Dict, Any, List

def assert_knowledge_node_structure(node: Dict[str, Any]) -> None:
    """
    Assert knowledge node has required structure.

    Validates that a knowledge node dictionary has all required fields
    and proper data types, raising AssertionError if validation fails.

    Args:
        node: Knowledge node dictionary to validate

    Raises:
        AssertionError: If node structure is invalid

    Examples:
        >>> valid_node = create_test_knowledge_node()
        >>> assert_knowledge_node_structure(valid_node)  # No exception

        >>> invalid_node = {'title': 'Test'}  # Missing required fields
        >>> assert_knowledge_node_structure(invalid_node)  # Raises AssertionError
    """
    required_fields = ['id', 'title', 'content_type', 'difficulty', 'content', 'metadata']

    for field in required_fields:
        assert field in node, f"Missing required field: {field}"

    # Validate field types
    assert isinstance(node['id'], str), "ID must be string"
    assert isinstance(node['title'], str), "Title must be string"
    assert isinstance(node['content_type'], str), "Content type must be string"
    assert isinstance(node['difficulty'], str), "Difficulty must be string"
    assert isinstance(node['content'], dict), "Content must be dictionary"
    assert isinstance(node['metadata'], dict), "Metadata must be dictionary"

    # Validate content structure
    assert 'overview' in node['content'], "Content must have overview"

    # Validate difficulty values
    valid_difficulties = ['beginner', 'intermediate', 'advanced', 'expert']
    assert node['difficulty'] in valid_difficulties, f"Invalid difficulty: {node['difficulty']}"

    # Validate content types
    valid_content_types = ['foundation', 'mathematics', 'implementation', 'application']
    assert node['content_type'] in valid_content_types, f"Invalid content type: {node['content_type']}"

def assert_learning_path_validity(path: Dict[str, Any]) -> None:
    """
    Assert learning path has valid structure.

    Validates that a learning path dictionary has all required fields
    and proper structure, including track and node validation.

    Args:
        path: Learning path dictionary to validate

    Raises:
        AssertionError: If learning path structure is invalid

    Examples:
        >>> valid_path = create_test_learning_path()
        >>> assert_learning_path_validity(valid_path)  # No exception
    """
    required_fields = ['id', 'title', 'tracks', 'metadata']

    for field in required_fields:
        assert field in path, f"Missing required field: {field}"

    # Validate tracks structure
    assert isinstance(path['tracks'], list), "Tracks must be list"
    assert len(path['tracks']) > 0, "Must have at least one track"

    for track in path['tracks']:
        assert 'id' in track, "Track must have ID"
        assert 'title' in track, "Track must have title"
        assert 'nodes' in track, "Track must have nodes"
        assert isinstance(track['nodes'], list), "Track nodes must be list"

def assert_api_response_format(response: Dict[str, Any], expected_fields: List[str]) -> None:
    """
    Assert API response has expected format.

    Validates that an API response dictionary contains expected fields
    and has proper structure.

    Args:
        response: API response dictionary to validate
        expected_fields: List of fields that must be present

    Raises:
        AssertionError: If response format is invalid

    Examples:
        >>> response = {'status': 'success', 'data': {'id': '123'}}
        >>> assert_api_response_format(response, ['status', 'data'])
    """
    assert isinstance(response, dict), "Response must be dictionary"

    for field in expected_fields:
        assert field in response, f"Missing expected field: {field}"

    # Additional validation based on field types
    if 'status' in expected_fields:
        assert response['status'] in ['success', 'error', 'pending'], "Invalid status value"
```

## Testing Guidelines

### Utility Testing Best Practices
1. **Test Utilities Themselves**: Test utilities with the same rigor as other code
2. **Edge Case Coverage**: Test utilities with various inputs and edge cases
3. **Performance Testing**: Ensure utilities don't impact test performance
4. **Error Testing**: Test error conditions and error handling
5. **Integration Testing**: Test how utilities work with the broader test framework

### Test Data Best Practices
1. **Realistic Data**: Generate test data that represents actual usage patterns
2. **Deterministic Generation**: Ensure test data generation is deterministic when needed
3. **Resource Management**: Properly manage resources created by utilities
4. **Cleanup**: Ensure proper cleanup of test resources and temporary files

## Performance Considerations

### Utility Performance Optimization
- **Efficient Generation**: Optimize test data generation for speed
- **Memory Management**: Ensure utilities don't create memory leaks
- **Resource Cleanup**: Implement proper resource cleanup
- **Caching**: Cache expensive operations when appropriate

### Test Impact Assessment
- **Performance Impact**: Ensure utilities don't slow down test execution
- **Memory Usage**: Monitor memory usage of utility functions
- **Resource Consumption**: Track resource consumption and cleanup
- **Scalability**: Ensure utilities work well with large test suites

## Getting Started as an Agent

### Test Utility Development Setup
1. **Study Existing Utilities**: Understand current utility implementations and patterns
2. **Learn Test Framework**: Understand the testing framework and requirements
3. **Identify Utility Needs**: Analyze common testing patterns to identify utility opportunities
4. **Set Up Development Environment**: Configure environment for utility development
5. **Run Existing Tests**: Ensure all existing tests pass before making changes

### Utility Development Process
1. **Identify Requirements**: Analyze testing patterns to identify utility needs
2. **Design Interface**: Design clean, intuitive utility interfaces
3. **Implement Functionality**: Implement utilities following established patterns
4. **Add Comprehensive Tests**: Create tests for utilities themselves
5. **Document Usage**: Provide detailed documentation and examples
6. **Integration**: Ensure utilities integrate with existing test framework

### Quality Assurance
1. **Functionality Testing**: Verify utility functionality meets requirements
2. **Performance Testing**: Ensure utilities don't impact test performance
3. **Integration Testing**: Test integration with broader test framework
4. **Documentation Testing**: Validate utility documentation and examples
5. **User Experience**: Ensure utilities are intuitive and easy to use

## Common Challenges and Solutions

### Challenge: Utility Reusability
**Solution**: Design utilities with clear, focused purposes and flexible interfaces that work across different test scenarios.

### Challenge: Performance Impact
**Solution**: Monitor utility performance impact on test execution and optimize for speed and memory usage.

### Challenge: Maintenance Overhead
**Solution**: Keep utilities simple and focused, with comprehensive documentation to reduce maintenance burden.

### Challenge: Integration Complexity
**Solution**: Design utilities with clear interfaces and comprehensive integration testing to ensure seamless framework integration.

## Related Documentation

- **[Main Testing README](../README.md)**: Comprehensive testing framework overview
- **[Unit Tests README](../unit/README.md)**: Unit testing module documentation
- **[Test Utilities README](./README.md)**: Test utilities module documentation
- **[Test Fixtures](../fixtures/README.md)**: Test data and fixture management
- **[Integration Tests](../integration/README.md)**: Component integration testing
- **[Knowledge Tests](../knowledge/README.md)**: Knowledge content validation

---

*"Active Inference for, with, by Generative AI"* - Enhancing testing through comprehensive utilities, helper functions, and automation tools.
