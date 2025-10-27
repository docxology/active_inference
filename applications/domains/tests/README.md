# Domain-Specific Tests

**Test suites and validation for domain-specific application implementations.**

## 📖 Overview

This directory contains test suites and validation tools for domain-specific application implementations within the Active Inference Knowledge Environment. These tests ensure that domain applications work correctly across various domains including artificial intelligence, neuroscience, robotics, psychology, and education.

## 🎯 Purpose

This test directory provides:

- **Domain Validation**: Tests for domain-specific functionality
- **Integration Testing**: Tests for domain integration with core components
- **Performance Testing**: Domain-specific performance benchmarks
- **Quality Assurance**: Validation of domain application quality

## 📁 Directory Structure

```
applications/domains/tests/
├── __init__.py           # Package initialization
├── test_ai_domain.py     # AI domain tests
├── test_neuroscience.py  # Neuroscience domain tests
├── test_robotics.py      # Robotics domain tests
├── test_psychology.py    # Psychology domain tests
├── test_education.py     # Education domain tests
└── README.md             # This file
```

## 🧪 Test Categories

### Domain Functionality Tests

Tests that validate domain-specific functionality:

- **Core Functionality**: Essential domain features and capabilities
- **Domain Logic**: Domain-specific business logic and algorithms
- **Data Processing**: Domain data handling and transformation
- **Integration**: Integration with core Active Inference components

### Integration Tests

Tests that validate runner integration:

- **Core Integration**: Integration with knowledge base and research tools
- **API Integration**: Integration with platform APIs
- **Data Integration**: Integration with domain data sources
- **Workflow Integration**: Integration with platform workflows

## 🚀 Usage

### Running Domain Tests

```bash
# Run all domain tests
pytest applications/domains/tests/

# Run tests for specific domain
pytest applications/domains/tests/test_ai_domain.py

# Run with coverage
pytest applications/domains/tests/ --cov=applications.domains

# Run with verbose output
pytest applications/domains/tests/ -v
```

### Test Examples

```python
import pytest
from applications.domains.artificial_intelligence import AIDomain

class TestAIDomain:
    """Test AI domain functionality"""

    def test_ai_domain_initialization(self):
        """Test AI domain initialization"""
        domain = AIDomain(config={"test_mode": True})
        assert domain is not None
        assert domain.domain_name == "artificial_intelligence"

    def test_ai_domain_integration(self):
        """Test AI domain integration with knowledge base"""
        domain = AIDomain(config={"test_mode": True})
        
        # Test knowledge integration
        knowledge_nodes = domain.get_knowledge_nodes()
        assert len(knowledge_nodes) > 0
        
        # Test research integration
        experiments = domain.get_experiments()
        assert len(experiments) > 0

    def test_ai_domain_performance(self):
        """Test AI domain performance"""
        domain = AIDomain(config={"test_mode": True})
        
        # Performance benchmark
        result = domain.benchmark_performance()
        assert result["average_response_time"] < 1.0  # seconds
```

## 📊 Test Coverage

### Coverage Requirements

- **Code Coverage**: >95% for domain implementations
- **Integration Coverage**: All integration points tested
- **Edge Cases**: Boundary conditions and error cases tested
- **Performance**: Performance benchmarks and validation

### Coverage Report

```bash
# Generate coverage report
pytest applications/domains/tests/ --cov=applications.domains --cov-report=html

# View coverage report
open htmlcov/index.html
```

## 🔄 Continuous Integration

Domain tests are integrated into the CI/CD pipeline:

```yaml
# .github/workflows/ci.yml
jobs:
  test-domains:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Run domain tests
        run: |
          pytest applications/domains/tests/
          --cov=applications.domains
          --cov-report=xml
```

## 📚 Related Documentation

- **[Applications README](../../README.md)**: Applications framework overview
- **[Applications AGENTS.md](../../AGENTS.md)**: Agent guidelines for applications
- **[Domains README](../README.md)**: Domain applications documentation

---

*"Active Inference for, with, by Generative AI"* - Ensuring domain application quality through comprehensive testing and validation.

