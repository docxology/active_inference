# Test Fixtures - Agent Documentation

This document provides comprehensive guidance for AI agents and contributors working with test fixtures in the Active Inference Knowledge Environment. It outlines fixture development workflows, implementation patterns, and best practices for creating and maintaining test data, environments, and mocking utilities.

## Module Overview

The Test Fixtures module provides comprehensive testing support infrastructure including test data generation, environment setup, mock objects, and fixture lifecycle management. This module ensures consistent, reliable, and maintainable test data across all testing activities.

## Core Responsibilities

### Fixture Development and Management
- **Test Data Generation**: Create realistic, consistent test data for all components
- **Environment Setup**: Configure test environments with proper isolation
- **Mock Object Creation**: Generate mock objects and test doubles
- **Fixture Lifecycle**: Manage fixture creation, usage, and cleanup

### Test Infrastructure
- **Fixture Registry**: Maintain centralized fixture registration and discovery
- **Validation Systems**: Ensure fixture quality and correctness
- **Performance Testing**: Optimize fixture performance for test execution
- **Integration Support**: Seamless integration with testing frameworks

## Development Workflows

### Fixture Creation Process
1. **Requirements Analysis**: Identify test data requirements and usage patterns
2. **Fixture Design**: Design fixture structure following established patterns
3. **Implementation**: Create fixtures with comprehensive test coverage
4. **Validation**: Validate fixture quality and functionality
5. **Integration**: Integrate fixtures with testing ecosystem
6. **Documentation**: Document fixture usage and examples

### Test Data Generation Workflow
```python
from typing import Dict, Any, List
import random
from datetime import datetime

class TestDataGenerator:
    """Generate realistic test data following established patterns"""

    def __init__(self, config: Dict[str, Any]):
        """Initialize test data generator"""
        self.config = config
        self.validators = self.load_validators()

    def generate_knowledge_fixture(self, domain: str, size: str = "medium") -> Dict[str, Any]:
        """Generate knowledge base test fixture"""
        
        # Validate input parameters
        self.validate_size(size)
        self.validate_domain(domain)
        
        # Generate fixture data
        fixture = {
            "foundations": self.generate_foundations(domain, size),
            "mathematics": self.generate_mathematics(domain, size),
            "implementations": self.generate_implementations(domain, size),
            "metadata": self.generate_fixture_metadata(domain, size)
        }
        
        # Validate generated fixture
        validation_result = self.validate_fixture(fixture)
        if not validation_result["is_valid"]:
            raise FixtureValidationError(f"Generated fixture failed validation: {validation_result['errors']}")
        
        return fixture

    def validate_size(self, size: str) -> None:
        """Validate size parameter"""
        valid_sizes = ["small", "medium", "large"]
        if size not in valid_sizes:
            raise ValueError(f"Size must be one of {valid_sizes}")
```

## Implementation Patterns

### Fixture Registry Pattern
```python
from typing import Protocol, Dict, Any, Optional
from abc import ABC, abstractmethod

class FixtureRegistry(ABC):
    """Registry for managing test fixtures"""

    def __init__(self):
        """Initialize fixture registry"""
        self.fixtures: Dict[str, Any] = {}
        self.load_registered_fixtures()

    @abstractmethod
    def register_fixture(self, fixture_id: str, fixture: Any) -> None:
        """Register a fixture in the registry"""
        pass

    @abstractmethod
    def get_fixture(self, puzzle_id: str) -> Optional[Any]:
        """Retrieve a fixture by ID"""
        pass

    @abstractmethod
    def validate_fixtures(self) -> Dict[str, Any]:
        """Validate all registered fixtures"""
        pass
```

### Fixture Factory Pattern
```python
def create_fixture(fixture_type: str, config: Dict[str, Any]) -> Any:
    """Create fixture using factory pattern"""

    fixture_factories = {
        'knowledge_data': create_knowledge_fixture,
        'research_data': create_research_fixture,
        'platform_data': create_platform_fixture,
        'environment': create_environment_fixture,
        'mock': create_mock_fixture
    }

    if fixture_type not in fixture_factories:
        raise ValueError(f"Unknown fixture type: {fixture_type}")

    # Validate configuration
    validate_fixture_config(config)

    return fixture_factories[fixture_type](config)
```

### Fixture Builder Pattern
```python
class FixtureBuilder:
    """Builder pattern for constructing complex fixtures"""

    def __init__(self):
        """Initialize fixture builder"""
        self.fixture = {}

    def with_knowledge_data(self, domain: str, size: str = "medium"):
        """Add knowledge data to fixture"""
        self.fixture["knowledge"] = generate_knowledge_data(domain, size)
        return self

    def with_research_data(self, experiment_type: str):
        """Add research data to fixture"""
        self.fixture["research"] = generate_research_data(experiment_type)
        return self

    def with_environment_config(self, env_type: str):
        """Add environment configuration to fixture"""
        self.fixture["environment"] = generate_environment_config(env_type)
        return self

    def build(self) -> Dict[str, Any]:
        """Build complete fixture"""
        # Validate fixture completeness
        validate_fixture_completeness(self.fixture)
        
        # Add metadata
        self.fixture["metadata"] = {
            "created_at": datetime.now().isoformat(),
            "version": "1.0.0"
        }
        
        return self.fixture
```

## Quality Standards

### Fixture Quality Requirements
- **Realism**: Test data must be realistic and representative of production data
- **Consistency**: Fixture data must be consistent across test runs
- **Isolation**: Fixtures must provide proper test isolation
- **Performance**: Fixtures must load and execute quickly
- **Maintainability**: Fixtures must be easy to update and extend

### Test Data Quality
- **Completeness**: Test data must cover all necessary scenarios
- **Accuracy**: Test data must accurately represent real data structures
- **Variety**: Test data must include edge cases and boundary conditions
- **Documentation**: Test data must be clearly documented and explained

## Testing Guidelines

### Fixture Testing Requirements
1. **Unit Tests**: Test individual fixture generation functions
2. **Integration Tests**: Test fixture integration with testing frameworks
3. **Performance Tests**: Validate fixture performance under load
4. **Validation Tests**: Ensure fixture data quality and correctness

### Test Organization
```
tests/fixtures/
├── test_data_generation.py      # Test data generation tests
├── test_environment_setup.py    # Environment setup tests
├── test_mock_creation.py        # Mock object creation tests
└── test_fixture_registry.py     # Fixture registry tests
```

## Performance Considerations

### Fixture Performance Optimization
- **Lazy Loading**: Load fixtures only when needed
- **Caching**: Cache frequently used fixtures
- **Streaming**: Use streaming for large test datasets
- **Parallel Generation**: Generate fixtures in parallel when possible

### Memory Management
- **Resource Cleanup**: Properly cleanup resources after tests
- **Memory Efficiency**: Optimize fixture memory footprint
- **Garbage Collection**: Manage fixture garbage collection efficiently

## Getting Started as an Agent

### Development Setup
1. **Study Fixture System**: Understand fixture architecture and patterns
2. **Review Existing Fixtures**: Learn from existing fixture implementations
3. **Set Up Testing**: Configure testing environment for fixture development
4. **Run Existing Tests**: Ensure all existing fixture tests pass

### Fixture Development Process
1. **Identify Needs**: Determine what test data or mocks are needed
2. **Design Fixture**: Create fixture structure following patterns
3. **Implement**: Develop fixture with comprehensive tests
4. **Validate**: Ensure fixture meets quality standards
5. **Document**: Create clear usage documentation
6. **Integrate**: Register fixture with testing framework

## Common Challenges and Solutions

### Challenge: Generating Realistic Test Data
**Solution**: Use realistic data generators that produce varied, representative test data following real-world patterns.

### Challenge: Fixture Performance
**Solution**: Implement lazy loading, caching, and optimized data structures for efficient fixture operations.

### Challenge: Test Isolation
**Solution**: Use isolated test environments and proper cleanup mechanisms to ensure test independence.

### Challenge: Mock Maintenance
**Solution**: Design mocks with clear interfaces and maintain them alongside the components they mock.

## Related Documentation

- **[Test Fixtures README](./README.md)**: Test fixtures overview and usage
- **[Testing Framework](../../README.md)**: Comprehensive testing documentation
- **[Unit Tests](../unit/README.md)**: Unit testing guidelines
- **[Integration Tests](../integration/README.md)**: Integration testing patterns

---

*"Active Inference for, with, by Generative AI"* - Enhancing testing through comprehensive fixtures and reliable test data management.

