# Integration Testing Framework

This directory contains integration tests for the Active Inference Knowledge Environment, focusing on component interaction, data flow validation, and system-wide functionality testing. Integration tests ensure that different components work together correctly and maintain data integrity across the entire system.

## Overview

Integration testing validates the interactions between different components of the Active Inference Knowledge Environment, ensuring that data flows correctly between modules, APIs function properly, and the system as a whole behaves as expected when components are combined.

## Test Categories

### 🔗 Component Integration Tests
Tests that verify interaction between major system components:

- **Knowledge-Research Integration**: Knowledge repository integration with research tools
- **Platform-Knowledge Integration**: Platform services integration with knowledge systems
- **Visualization-Platform Integration**: Visualization components with platform APIs
- **Research-Analysis Integration**: Research tools integration with analysis frameworks

### 🔄 Data Flow Integration Tests
Tests that validate data flow through the system:

- **Data Pipeline Tests**: End-to-end data processing pipeline validation
- **API Integration Tests**: REST API functionality and data exchange
- **Storage Integration Tests**: Data storage and retrieval across backends
- **Search Integration Tests**: Search functionality across knowledge systems

### 🏗️ System Integration Tests
Tests that validate complete system functionality:

- **Full System Tests**: End-to-end system functionality validation
- **Performance Integration Tests**: System performance under integrated load
- **Scalability Tests**: System scaling with multiple components
- **Reliability Tests**: System reliability and fault tolerance

## Getting Started

### Running Integration Tests

```bash
# Run all integration tests
pytest tests/integration/ -v

# Run specific integration test categories
pytest tests/integration/test_knowledge_integration.py -v
pytest tests/integration/test_platform_integration.py -v
pytest tests/integration/test_data_flow_integration.py -v

# Run integration tests with coverage
pytest tests/integration/ --cov=src/ --cov-report=html
```

### Writing Integration Tests

```python
import pytest
from active_inference.knowledge import KnowledgeRepository
from active_inference.research import ExperimentManager
from active_inference.platform import PlatformServices

class TestKnowledgeResearchIntegration:
    """Integration tests for knowledge and research components"""

    @pytest.fixture
    def setup_integration_environment(self):
        """Set up integrated test environment"""
        # Initialize components
        knowledge_repo = KnowledgeRepository()
        experiment_manager = ExperimentManager()
        platform_services = PlatformServices()

        return {
            'knowledge': knowledge_repo,
            'experiments': experiment_manager,
            'platform': platform_services
        }

    def test_knowledge_experiment_integration(self, setup_integration_environment):
        """Test integration between knowledge and experiment systems"""
        # Test data flow from knowledge to experiments
        components = setup_integration_environment

        # Create knowledge content
        knowledge_id = components['knowledge'].create_node(
            content={'title': 'Test Concept', 'content': 'Test content'}
        )

        # Create experiment using knowledge
        experiment_config = {
            'name': 'integration_test_experiment',
            'knowledge_reference': knowledge_id
        }

        experiment_id = components['experiments'].create_experiment(experiment_config)

        # Verify integration
        assert experiment_id is not None
        assert components['experiments'].get_experiment(experiment_id) is not None
```

## Test Organization

### Test File Structure
```
tests/integration/
├── __init__.py
├── test_knowledge_integration.py       # Knowledge system integration
├── test_research_integration.py        # Research tools integration
├── test_platform_integration.py        # Platform services integration
├── test_visualization_integration.py   # Visualization integration
├── test_data_flow_integration.py       # Data flow validation
├── test_api_integration.py             # API functionality testing
├── test_storage_integration.py         # Storage backend integration
├── test_search_integration.py          # Search system integration
└── test_full_system_integration.py     # Complete system integration
```

### Integration Test Patterns

#### Component Interaction Testing
```python
def test_component_data_flow(self, integrated_components):
    """Test data flow between integrated components"""
    # Arrange
    input_data = {'test': 'data', 'value': 42}

    # Act
    result = integrated_components['knowledge'].process_data(input_data)
    processed_result = integrated_components['research'].analyze_data(result)

    # Assert
    assert processed_result is not None
    assert 'analysis' in processed_result
    assert processed_result['original_data'] == input_data
```

#### API Integration Testing
```python
def test_api_data_exchange(self, api_client, test_server):
    """Test API data exchange between components"""
    # Test data retrieval
    response = api_client.get('/api/knowledge/search?q=test')
    assert response.status_code == 200
    assert 'results' in response.json()

    # Test data submission
    test_data = {'content': 'test data', 'metadata': {}}
    response = api_client.post('/api/knowledge/nodes', json=test_data)
    assert response.status_code == 201
```

## Quality Assurance

### Integration Test Standards
- **Realistic Scenarios**: Test with realistic data volumes and usage patterns
- **Component Isolation**: Test component interactions without external dependencies
- **Data Integrity**: Verify data integrity through integration points
- **Error Propagation**: Test error handling across component boundaries
- **Performance Validation**: Validate performance under integrated load

### Test Data Management
- **Realistic Test Data**: Use realistic data for integration testing
- **Data Cleanup**: Proper cleanup of test data between test runs
- **Data Isolation**: Isolate test data from production data
- **Data Versioning**: Manage test data versions and updates

## Performance Considerations

### Integration Performance Testing
- **Load Testing**: Test system under various load conditions
- **Stress Testing**: Test system beyond normal operating parameters
- **Concurrency Testing**: Test concurrent access and data flow
- **Scalability Testing**: Test system scaling with component count
- **Resource Monitoring**: Monitor resource usage during integration tests

### Performance Benchmarks
- **Response Time**: <100ms for typical integration operations
- **Throughput**: Maintain target throughput under integrated load
- **Memory Usage**: Efficient memory usage with multiple components
- **Error Rate**: <0.1% error rate in integration scenarios

## Contributing

### Writing Integration Tests
1. **Identify Integration Points**: Find component interaction points
2. **Design Test Scenarios**: Create realistic integration test scenarios
3. **Implement Test Cases**: Write comprehensive integration test cases
4. **Validate Test Data**: Ensure test data is realistic and complete
5. **Test Documentation**: Document integration test procedures

### Integration Test Best Practices
- **Component Isolation**: Test components in realistic but controlled environments
- **Data Flow Validation**: Verify data integrity through integration points
- **Error Scenario Testing**: Test error handling and recovery
- **Performance Validation**: Include performance assertions in integration tests
- **Documentation**: Document integration requirements and test procedures

## Related Documentation

- **[Testing AGENTS.md](../AGENTS.md)**: Testing framework development guidelines
- **[Main AGENTS.md](../../AGENTS.md)**: Project-wide agent guidelines
- **[Contributing Guide](../../CONTRIBUTING.md)**: Contribution processes
- **[Unit Tests](../unit/)**: Individual component testing

---

*"Active Inference for, with, by Generative AI"* - Ensuring system reliability through comprehensive integration testing and component interaction validation.
