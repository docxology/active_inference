# Test Utilities

Comprehensive collection of testing utilities, helper functions, and supporting tools for the Active Inference Knowledge Environment testing framework. These utilities enable efficient test development, execution, and maintenance across all test categories.

## Overview

Test utilities provide essential supporting functionality for the testing framework, including helper functions, mock utilities, test data generators, and testing infrastructure. These utilities ensure consistent, reliable, and maintainable test code across the entire test suite.

## Directory Structure

```
tests/utilities/
├── __init__.py           # Utilities module initialization
└── test_helpers.py       # Core testing utilities and helpers
```

## Core Utilities

### Test Data Management
```python
from tests.utilities.test_helpers import (
    create_test_knowledge_node,
    create_test_learning_path,
    generate_unique_test_id
)

# Create standardized test data
test_node = create_test_knowledge_node({
    'id': 'test_node',
    'title': 'Test Node',
    'difficulty': 'beginner'
})

# Generate unique identifiers for tests
unique_id = generate_unique_test_id('experiment')
assert 'experiment_' in unique_id
```

### Mock and Patch Utilities
```python
from tests.utilities.test_helpers import (
    create_mock_ollama_response,
    create_async_mock_client,
    patch_llm_client_success,
    patch_llm_client_error
)

# Mock successful LLM responses
def test_llm_integration():
    with patch_llm_client_success() as mock_client:
        # Test LLM integration code
        response = make_llm_request("test prompt")
        assert response == "Test response"

# Mock error scenarios
def test_llm_error_handling():
    with patch_llm_client_error() as mock_client:
        with pytest.raises(LLMError):
            make_llm_request("test prompt")
```

### File and Resource Management
```python
from tests.utilities.test_helpers import (
    create_temp_file_with_content,
    create_temp_directory,
    cleanup_temp_files
)

# Create temporary files for testing
def test_file_processing():
    test_content = '{"test": "data"}'

    with create_temp_file_with_content(test_content) as temp_file:
        assert temp_file.exists()
        # Test file processing logic
        result = process_file(temp_file)
        assert result['test'] == 'data'
```

### Validation Helpers
```python
from tests.utilities.test_helpers import (
    assert_knowledge_node_structure,
    assert_learning_path_validity,
    assert_api_response_format
)

# Validate knowledge node structure
def test_knowledge_node_creation():
    node = create_test_knowledge_node(test_data)

    assert_knowledge_node_structure(node)
    assert node['id'] == expected_id
    assert node['content_type'] in ['foundation', 'mathematics', 'implementation', 'application']

# Validate learning path structure
def test_learning_path_creation():
    path = create_test_learning_path(test_nodes)

    assert_learning_path_validity(path)
    assert len(path['tracks']) > 0
    assert all('nodes' in track for track in path['tracks'])
```

## Usage Examples

### Basic Test Helper Usage
```python
import pytest
from tests.utilities.test_helpers import create_test_knowledge_node

class TestKnowledgeRepository:
    """Example using test utilities"""

    def test_node_search_with_utilities(self):
        """Test search functionality using test utilities"""
        # Arrange
        repo = KnowledgeRepository(test_config)

        # Create test data using utilities
        test_nodes = [
            create_test_knowledge_node({
                'id': f'node_{i}',
                'title': f'Test Node {i}',
                'content': {'overview': f'Content for node {i}'}
            })
            for i in range(3)
        ]

        # Add test data to repository
        for node in test_nodes:
            repo.add_node(node)

        # Act
        results = repo.search('Test Node', limit=5)

        # Assert
        assert len(results) == 3
        assert all(result['title'].startswith('Test Node') for result in results)
```

### Mock Integration Testing
```python
from unittest.mock import patch
from tests.utilities.test_helpers import create_mock_ollama_response

def test_external_api_integration():
    """Test integration with external APIs using mocks"""
    # Arrange
    mock_response = create_mock_ollama_response({
        'response': 'Mocked response content',
        'model': 'test_model',
        'usage': {'tokens': 10}
    })

    # Act
    with patch('ollama.generate', return_value=mock_response):
        client = LLMClient(test_config)
        result = client.generate('test prompt')

    # Assert
    assert result['response'] == 'Mocked response content'
    assert result['model'] == 'test_model'
```

### Temporary Resource Management
```python
from tests.utilities.test_helpers import create_temp_directory

def test_directory_operations():
    """Test directory operations with automatic cleanup"""
    with create_temp_directory() as temp_dir:
        # Create test files
        test_file = temp_dir / 'test.json'
        test_file.write_text('{"test": "data"}')

        # Test directory operations
        assert test_file.exists()
        files = list(temp_dir.glob('*.json'))
        assert len(files) == 1

        # Directory automatically cleaned up after test
```

### Custom Test Assertions
```python
from tests.utilities.test_helpers import (
    assert_knowledge_node_completeness,
    assert_search_results_quality
)

def test_comprehensive_node_validation():
    """Test comprehensive node validation"""
    # Create complex test node
    node = create_test_knowledge_node({
        'id': 'complex_node',
        'title': 'Complex Test Node',
        'difficulty': 'advanced',
        'prerequisites': ['basic_node'],
        'content': {
            'overview': 'Complex overview',
            'mathematical_definition': 'Mathematical content',
            'examples': ['example1', 'example2']
        }
    })

    # Comprehensive validation
    assert_knowledge_node_completeness(node)
    assert node['difficulty'] == 'advanced'
    assert 'basic_node' in node['prerequisites']
```

## Test Helper Implementation

### Creating Custom Test Helpers
```python
# tests/utilities/test_helpers.py

import json
import uuid
from pathlib import Path
from typing import Dict, Any, List, Optional
from unittest.mock import Mock

def create_test_knowledge_node(overrides: Dict[str, Any] = None) -> Dict[str, Any]:
    """Create a standardized test knowledge node"""
    base_node = {
        'id': 'test_node',
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
            'last_updated': '2024-01-01',
            'version': '1.0'
        }
    }

    if overrides:
        base_node.update(overrides)

    return base_node

def generate_unique_test_id(prefix: str = 'test') -> str:
    """Generate unique test identifier"""
    return f"{prefix}_{uuid.uuid4().hex[:8]}"

def create_temp_file_with_content(content: str, suffix: str = '.json') -> Path:
    """Create temporary file with specified content"""
    import tempfile

    temp_file = Path(tempfile.NamedTemporaryFile(suffix=suffix, delete=False).name)
    temp_file.write_text(content)
    return temp_file

def cleanup_temp_files(*files: Path) -> None:
    """Clean up temporary files"""
    for file in files:
        if file.exists():
            file.unlink()

def create_mock_ollama_response(response_data: Dict[str, Any]) -> Mock:
    """Create mock Ollama API response"""
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = response_data
    mock_response.raise_for_status.return_value = None
    return mock_response

def assert_knowledge_node_structure(node: Dict[str, Any]) -> None:
    """Assert knowledge node has required structure"""
    required_fields = ['id', 'title', 'content_type', 'difficulty', 'content', 'metadata']

    for field in required_fields:
        assert field in node, f"Missing required field: {field}"

    # Validate field types
    assert isinstance(node['id'], str), "ID must be string"
    assert isinstance(node['title'], str), "Title must be string"
    assert isinstance(node['content'], dict), "Content must be dictionary"
    assert isinstance(node['metadata'], dict), "Metadata must be dictionary"

    # Validate content structure
    assert 'overview' in node['content'], "Content must have overview"

def patch_llm_client_success():
    """Context manager for patching LLM client with success response"""
    from unittest.mock import patch, Mock

    mock_response = create_mock_ollama_response({
        'response': 'Mocked successful response',
        'model': 'test_model',
        'usage': {'tokens': 15}
    })

    return patch('ollama.generate', return_value=mock_response)

def patch_llm_client_error():
    """Context manager for patching LLM client with error response"""
    from unittest.mock import patch
    from active_inference.llm import LLMError

    return patch('ollama.generate', side_effect=LLMError("Mocked error"))
```

## Advanced Testing Utilities

### Performance Testing Helpers
```python
from tests.utilities.test_helpers import (
    measure_execution_time,
    measure_memory_usage,
    assert_performance_bounds
)

def test_search_performance():
    """Test search performance meets requirements"""
    repo = KnowledgeRepository(test_config)

    # Measure execution time
    execution_time = measure_execution_time(
        lambda: repo.search('entropy', limit=10)
    )

    assert_performance_bounds(execution_time, max_time=0.1)

def test_memory_usage():
    """Test memory usage stays within bounds"""
    initial_memory = measure_memory_usage()

    # Execute memory-intensive operations
    repo = KnowledgeRepository(test_config)
    # ... perform operations ...

    final_memory = measure_memory_usage()
    memory_increase = final_memory - initial_memory

    assert memory_increase < 100 * 1024 * 1024  # Less than 100MB
```

### Async Testing Utilities
```python
from tests.utilities.test_helpers import (
    create_async_mock_client,
    await_with_timeout,
    assert_async_functionality
)

async def test_async_llm_integration():
    """Test async LLM integration"""
    client = create_async_mock_client()

    # Test async functionality
    result = await await_with_timeout(
        client.generate_async('test prompt'),
        timeout=5.0
    )

    assert_async_functionality(result)
    assert result['response'] == 'Mocked async response'
```

## Best Practices

### Utility Development
1. **Consistent Interface**: Maintain consistent function signatures and behavior
2. **Comprehensive Documentation**: Document all utilities with usage examples
3. **Error Handling**: Include proper error handling and validation
4. **Performance**: Optimize utilities for test execution performance
5. **Maintainability**: Keep utilities simple and focused on specific purposes

### Test Data Management
1. **Realistic Data**: Create test data that represents actual usage patterns
2. **Deterministic Generation**: Ensure test data generation is deterministic
3. **Resource Cleanup**: Properly clean up resources created during testing
4. **Isolation**: Ensure test utilities don't interfere between tests

### Mock Implementation
1. **Realistic Mocks**: Create mocks that behave like real dependencies
2. **Comprehensive Coverage**: Mock all external dependencies appropriately
3. **Easy Configuration**: Make mocks easy to configure for different test scenarios
4. **Validation**: Validate mock behavior matches expected interface

## Contributing

### Adding New Test Utilities
1. **Identify Need**: Analyze existing test patterns to identify utility requirements
2. **Design Interface**: Design clean, intuitive utility interfaces
3. **Implement Functionality**: Implement utilities following established patterns
4. **Add Documentation**: Provide comprehensive documentation and examples
5. **Write Tests**: Include tests for the utilities themselves
6. **Integration**: Ensure utilities integrate well with existing test framework

### Utility Review Checklist
- [ ] **Purpose Clear**: Utility purpose and usage are clearly defined
- [ ] **Interface Intuitive**: Function signatures and behavior are intuitive
- [ ] **Documentation Complete**: Usage examples and documentation are comprehensive
- [ ] **Error Handling**: Proper error handling and validation included
- [ ] **Performance**: Utilities don't negatively impact test performance
- [ ] **Testing**: Utilities themselves are tested and validated

## Related Documentation

- **[Main Testing README](../README.md)**: Comprehensive testing framework overview
- **[Unit Tests README](../unit/README.md)**: Unit testing module documentation
- **[Test Fixtures](../fixtures/README.md)**: Test data and fixture management
- **[Integration Tests](../integration/README.md)**: Component integration testing
- **[Knowledge Tests](../knowledge/README.md)**: Knowledge content validation
- **[Performance Tests](../performance/README.md)**: Performance testing documentation

---

*"Active Inference for, with, by Generative AI"* - Enhancing testing through comprehensive utilities, helper functions, and automation tools.
