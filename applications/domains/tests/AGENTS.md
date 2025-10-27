# Domain Tests - Agent Documentation

This document provides comprehensive guidance for AI agents and contributors working with domain-specific tests in the Active Inference Knowledge Environment.

## Module Overview

The Domain Tests module provides comprehensive test suites for domain-specific application implementations, ensuring quality, functionality, and integration across various domains. This module validates domain applications work correctly and integrate properly with core Active Inference components.

## Core Responsibilities

### Test Development and Maintenance
- **Test Creation**: Create comprehensive tests for domain functionality
- **Integration Testing**: Test domain integration with core components
- **Performance Testing**: Validate domain performance characteristics
- **Quality Assurance**: Ensure domain application quality standards

### Test Quality Management
- **Coverage**: Maintain high test coverage for domain code
- **Reliability**: Ensure tests are reliable and maintainable
- **Performance**: Optimize test execution performance
- **Documentation**: Document test usage and best practices

## Development Workflows

### Test Development Process
1. **Requirements Analysis**: Identify testing requirements for domains
2. **Test Design**: Design comprehensive test suites and scenarios
3. **Implementation**: Implement tests following TDD principles
4. **Integration**: Integrate tests with testing framework
5. **Validation**: Validate test effectiveness and coverage
6. **Documentation**: Document tests and usage patterns

### Test Implementation Pattern
```python
import pytest
from typing import Dict, Any
from applications.domains import DomainRegistry

class TestDomainFunctionality:
    """Base class for domain functionality tests"""

    @pytest.fixture
    def domain_config(self) -> Dict[str, Any]:
        """Standard domain configuration for testing"""
        return {
            "test_mode": True,
            "mock_data": True,
            "debug": False
        }

    @pytest.fixture
    def domain_instance(self, domain_config):
        """Create domain instance for testing"""
        # Implementation depends on specific domain
        pass

    def test_domain_initialization(self, domain_config):
        """Test domain initialization"""
        # Test domain can be created
        pass

    def test_domain_integration(self, domain_instance):
        """Test domain integration with core components"""
        # Test domain integrates with knowledge base
        # Test domain integrates with research tools
        pass

    def test_domain_performance(self, domain_instance):
        """Test domain performance characteristics"""
        # Performance benchmarking
        pass
```

## Implementation Patterns

### Domain Test Factory Pattern
```python
def create_domain_test(domain_name: str, test_type: str) -> Any:
    """Create domain test using factory pattern"""

    test_factories = {
        'ai_domain': create_ai_domain_test,
        'neuroscience': create_neuroscience_test,
        'robotics': create_robotics_test,
        'psychology': create_psychology_test,
        'education': create_education_test
    }

    if domain_name not in test_factories:
        raise ValueError(f"Unknown domain: {domain_name}")

    return test_factories[domain_name](test_type)
```

### Domain Integration Test Pattern
```python
class DomainIntegrationTest:
    """Test domain integration with core components"""

    def test_knowledge_integration(self, domain_instance):
        """Test domain integration with knowledge base"""
        
        # Test knowledge node retrieval
        nodes = domain_instance.get_knowledge_nodes()
        assert len(nodes) > 0
        
        # Test knowledge search
        results = domain_instance.search_knowledge("query")
        assert len(results) > 0
        
        # Test knowledge update
        updated = domain_instance.update_knowledge("node_id", data)
        assert updated is not None

    def test_research_integration(self, domain_instance):
        """Test domain integration with research tools"""
        
        # Test experiment creation
        experiment = domain_instance.create_experiment({"name": "test"})
        assert experiment is not None
        
        # Test simulation integration
        simulation = domain_instance.run_simulation({"parameters": {}})
        assert simulation is not None
```

## Quality Standards

### Test Quality Requirements
- **Coverage**: >95% code coverage for domain implementations
- **Reliability**: All tests pass consistently
- **Performance**: Fast test execution
- **Maintainability**: Easy to update and extend tests

### Domain Testing Standards
- **Functionality**: All domain features thoroughly tested
- **Integration**: All integration points validated
- **Edge Cases**: Boundary conditions and error cases tested
- **Documentation**: Clear test documentation and examples

## Testing Guidelines

### Test Organization
```
applications/domains/tests/
├── unit/                 # Unit tests for domain components
├── integration/          # Integration tests
├── performance/          # Performance tests
└── fixtures/             # Test fixtures specific to domains
```

### Test Categories
1. **Unit Tests**: Test individual domain components
2. **Integration Tests**: Test domain integration with core components
3. **Performance Tests**: Validate domain performance
4. **End-to-End Tests**: Test complete domain workflows

## Related Documentation

- **[Domain Tests README](./README.md)**: Test overview and usage
- **[Applications README](../../README.md)**: Applications framework documentation
- **[Domains README](../README.md)**: Domain applications documentation

---

*"Active Inference for, with, by Generative AI"* - Ensuring domain application quality through comprehensive testing and validation.

