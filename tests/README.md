# Testing Framework

Comprehensive testing infrastructure for the Active Inference Knowledge Environment. Provides multi-layered testing with unit tests, integration tests, knowledge validation, performance testing, and security testing.

## Overview

The testing framework ensures code quality, functionality, and reliability across all components of the Active Inference Knowledge Environment. Tests are organized by type and component, with comprehensive fixtures, utilities, and reporting.

## Directory Structure

```
tests/
├── __init__.py              # Test suite initialization
├── README.md                # This file
├── run_tests.py             # Comprehensive test runner
├── coverage_config.py       # Coverage configuration and analysis
├── fixtures/                # Test fixtures and data
│   ├── __init__.py
│   └── test_data.py         # Sample test data and utilities
├── utilities/               # Testing utilities and helpers
│   ├── __init__.py
│   └── test_helpers.py      # Test helper functions and mocks
├── unit/                    # Unit tests for individual components (124 tests)
│   ├── test_knowledge_*.py  # Knowledge system unit tests
│   ├── test_llm_*.py        # LLM integration unit tests
│   ├── test_research_*.py   # Research tools unit tests
│   ├── test_applications_*.py # Application framework unit tests
│   └── test_platform_*.py   # Platform services unit tests
├── integration/             # Integration tests for component interaction (10 tests)
│   └── test_*.py            # End-to-end integration tests
├── knowledge/               # Knowledge content validation tests (23 tests)
│   ├── test_content_accuracy.py     # Mathematical and conceptual accuracy validation
│   ├── test_educational_quality.py  # Educational effectiveness testing
│   └── README.md                     # Knowledge testing documentation
├── performance/             # Performance and scalability tests (5 tests)
│   ├── test_knowledge_repository_performance.py # Repository performance testing
│   ├── README.md                     # Performance testing documentation
│   └── __init__.py                   # Performance testing module
└── security/                # Security and vulnerability tests (8 tests)
    ├── test_knowledge_security.py   # Knowledge repository security testing
    ├── README.md                     # Security testing documentation
    └── __init__.py                   # Security testing module
```

## Test Categories

### 🧪 Unit Tests
Individual component functionality testing with comprehensive edge case coverage.

**Location**: `tests/unit/`
**Test Files**:
- `test_knowledge_*.py` - Knowledge system unit tests (repository, implementations, applications)
- `test_llm_*.py` - LLM integration unit tests (client, conversations, models, prompts)
- `test_research_*.py` - Research tools unit tests (data management)
- `test_applications_*.py` - Application framework unit tests (best practices, templates, case studies)
**Markers**: `unit`, `knowledge`, `llm`, `research`, `platform`, `visualization`, `applications`
**Coverage Target**: >95% for core components, >80% overall
**Tests**: 124 comprehensive unit tests covering all major components

### 🔗 Integration Tests
Component interaction validation and data flow testing across system boundaries.

**Location**: `tests/integration/`
**Test Files**:
- `test_llm_integration.py` - LLM system integration testing
**Markers**: `integration`
**Coverage Target**: >80% for integration points
**Tests**: 10 comprehensive integration tests covering component interactions

### 📚 Knowledge Tests
Content accuracy, completeness, educational quality validation, and learning path validation.

**Location**: `tests/knowledge/`
**Test Files**:
- `test_content_accuracy.py` - Mathematical and conceptual accuracy validation
- `test_educational_quality.py` - Educational effectiveness and accessibility testing
**Markers**: `knowledge`
**Coverage Target**: 100% for knowledge content
**Tests**: 23 comprehensive knowledge validation tests

### ⚡ Performance Tests
Scalability, efficiency, and performance characteristics validation.

**Location**: `tests/performance/`
**Test Files**:
- `test_knowledge_repository_performance.py` - Repository performance under load
- `README.md` - Performance testing documentation
**Markers**: `performance`
**Coverage Target**: Critical path performance validation
**Tests**: 5 performance benchmark tests with memory and timing validation

### 🔒 Security Tests
Vulnerability assessment and security validation including injection attacks, XSS protection, and path traversal.

**Location**: `tests/security/`
**Test Files**:
- `test_knowledge_security.py` - Knowledge repository security testing
- `README.md` - Security testing documentation
**Markers**: `security`
**Coverage Target**: Complete security validation
**Tests**: 8 security vulnerability tests covering common attack vectors

## Quick Start

### Using Make (Recommended)
```bash
# Run all tests
make test

# Run all tests with coverage
make test-coverage

# Run only unit tests
make test-unit

# Run only integration tests
make test-integration

# Run knowledge validation tests
make test-knowledge

# Run performance tests
make test-performance

# Run security tests
make test-security

# Run tests in parallel
make test-parallel

# Run tests quickly (stop on first failure)
make test-fast

# Run specific test file
make test-specific FILE=tests/unit/test_knowledge_repository.py
```

### Using uv directly
```bash
# Install test dependencies
uv sync --extra test

# Run all tests
uv run python -m pytest tests/ -v

# Run with coverage
uv run python -m pytest tests/ --cov=src/ --cov-report=html

# Run specific test types
uv run python -m pytest tests/unit/ -m "unit" -v
uv run python -m pytest tests/integration/ -m "integration" -v
uv run python -m pytest tests/knowledge/ -m "knowledge" -v
uv run python -m pytest tests/performance/ -m "performance" -v
uv run python -m pytest tests/security/ -m "security" -v
```

### Using the test runner
```bash
# Run all tests
python tests/run_tests.py

# Run unit tests only
python tests/run_tests.py unit

# Run with coverage
python tests/run_tests.py coverage

# Run specific component tests
python tests/run_tests.py component knowledge
python tests/run_tests.py component llm

# Run specific test file
python tests/run_tests.py --file tests/unit/test_knowledge_repository.py
```

## Test Configuration

### Pytest Configuration
The project uses comprehensive pytest configuration in `pytest.ini`:
- **Async Support**: Auto-detects async tests
- **Markers**: Organized test categories with markers
- **Warnings**: Configured to ignore common warnings
- **Output**: Clean, informative test output

### Coverage Configuration
Coverage requirements and reporting configured in `tests/coverage_config.py`:
- **Overall Target**: 85% coverage
- **Component Targets**: Knowledge (90%), LLM (88%), Research (80%)
- **HTML Reports**: Generated in `htmlcov/` directory
- **XML Reports**: For CI integration

## Test Development

### Writing Unit Tests
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

### Writing Integration Tests
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
```

### Test Fixtures
Use comprehensive test fixtures from `tests/fixtures/test_data.py`:
```python
import pytest
from tests.fixtures.test_data import (
    create_test_knowledge_node,
    create_test_learning_path,
    SAMPLE_KNOWLEDGE_NODES
)

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

## Test Utilities

### Mock Utilities
```python
from tests.utilities.test_helpers import (
    create_mock_ollama_response,
    create_async_mock_client,
    patch_llm_client_success
)

# Mock successful LLM responses
def test_llm_integration():
    with patch_llm_client_success() as mock_client:
        # Test LLM integration code
        response = make_llm_request("test prompt")
        assert response == "Test response"
```

### Test Data Generation
```python
from tests.utilities.test_helpers import (
    generate_unique_test_id,
    create_temp_file_with_content
)

def test_data_processing():
    # Generate unique test data
    test_id = generate_unique_test_id("experiment")
    assert "experiment_" in test_id

    # Create temporary files for testing
    with create_temp_file_with_content('{"test": "data"}') as temp_file:
        assert temp_file.exists()
        # Test file processing
```

## Coverage Reporting

### Generate Coverage Reports
```bash
# Basic coverage report
make test-coverage

# Detailed coverage analysis
uv run python tests/coverage_config.py

# Coverage gap analysis
uv run python -c "from tests.coverage_config import analyze_coverage_gaps; print(analyze_coverage_gaps())"
```

### Coverage Targets
- **Overall**: 85% minimum coverage
- **Knowledge Module**: 90% minimum coverage
- **LLM Module**: 88% minimum coverage
- **Research Module**: 80% minimum coverage
- **Platform Module**: 75% minimum coverage

## CI/CD Integration

The project includes comprehensive GitHub Actions workflow (`.github/workflows/ci.yml`):
- **Multi-Python Testing**: Tests on Python 3.9, 3.10, 3.11, 3.12
- **Code Quality**: Linting, formatting, and type checking
- **Security Scanning**: Vulnerability assessment and dependency checking
- **Documentation**: Automated documentation building and deployment
- **Coverage Reporting**: Integration with Codecov for coverage tracking

### Local CI Simulation
```bash
# Run full CI pipeline locally
make check-all

# Test documentation build
make docs

# Run security checks
uv run python -m pytest tests/security/ -v
```

## Performance Testing

### Load Testing
```python
def test_knowledge_search_performance():
    """Test knowledge search under load"""
    repo = KnowledgeRepository(test_config)

    # Measure search performance
    import time
    start_time = time.time()

    for _ in range(1000):
        results = repo.search("entropy", limit=10)

    end_time = time.time()
    avg_time = (end_time - start_time) / 1000

    assert avg_time < 0.1  # Should complete within 100ms on average
```

### Memory Testing
```python
def test_memory_usage():
    """Test memory usage doesn't leak"""
    import psutil
    import os

    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss

    # Run memory-intensive operations
    for _ in range(100):
        repo = KnowledgeRepository(test_config)
        # ... perform operations ...

    final_memory = process.memory_info().rss
    memory_increase = final_memory - initial_memory

    assert memory_increase < 100 * 1024 * 1024  # Less than 100MB increase
```

## Best Practices

### Test Organization
1. **Arrange-Act-Assert**: Structure tests clearly with setup, execution, and validation
2. **Descriptive Names**: Use descriptive test method names that explain the test purpose
3. **Independent Tests**: Ensure tests don't depend on each other
4. **Fast Tests**: Keep unit tests fast (<1 second each)
5. **Comprehensive Coverage**: Test both success and failure scenarios

### Test Data
1. **Realistic Data**: Use realistic test data that represents actual usage
2. **Edge Cases**: Test boundary conditions and edge cases
3. **Data Isolation**: Ensure test data doesn't interfere between tests
4. **Cleanup**: Properly clean up test data and resources

### Mocking Strategy
1. **External Dependencies**: Mock external APIs and services
2. **Time Dependencies**: Mock time and date functions
3. **File System**: Mock file operations for predictable testing
4. **Network**: Mock network requests for reliable testing

## Troubleshooting

### Common Issues
1. **Import Errors**: Ensure all dependencies are installed with `uv sync --extra test`
2. **Async Issues**: Use proper async/await patterns and pytest-asyncio
3. **Mock Issues**: Verify mock setup matches actual API calls
4. **Fixture Issues**: Check fixture scope and dependencies

### Debugging Tests
```bash
# Run single test with detailed output
uv run python -m pytest tests/unit/test_knowledge_repository.py::TestKnowledgeRepository::test_search_knowledge -v -s

# Run with debugger
uv run python -m pytest tests/unit/test_knowledge_repository.py::TestKnowledgeRepository::test_search_knowledge -v -s --pdb

# Run with coverage debugging
uv run python -m pytest tests/unit/test_knowledge_repository.py --cov=src/active_inference/knowledge --cov-report=term-missing
```

## Contributing

### Adding New Tests
1. **Follow Patterns**: Follow existing test patterns and conventions
2. **Add Fixtures**: Create reusable fixtures for common test data
3. **Update Coverage**: Ensure new code has adequate test coverage
4. **Document Tests**: Add comprehensive docstrings and comments

### Test Review Checklist
- [ ] **Test Completeness**: Tests cover all functionality
- [ ] **Edge Cases**: Tests include boundary conditions and error cases
- [ ] **Performance**: Tests validate performance characteristics
- [ ] **Documentation**: Tests are well-documented and understandable
- [ ] **Integration**: Tests integrate properly with existing test suite

## Related Documentation

- **[Main README](../../README.md)**: Project overview and getting started
- **[Development Guide](../../CONTRIBUTING.md)**: Development guidelines
- **[Code Quality](../../applications/best_practices/)**: Code quality standards
- **[Platform Documentation](../../platform/)**: Platform testing
- **[Knowledge Testing](../../tests/knowledge/)**: Knowledge validation

---

*"Active Inference for, with, by Generative AI"* - Ensuring reliability through comprehensive testing, quality assurance, and continuous validation.