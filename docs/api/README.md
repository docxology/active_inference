# API Documentation

This directory contains comprehensive API reference documentation for the Active Inference Knowledge Environment. The API documentation provides detailed information about all public interfaces, including parameters, return values, examples, and usage patterns.

## Overview

The API Documentation module provides complete reference documentation for all public APIs in the Active Inference Knowledge Environment. This includes REST APIs, Python APIs, command-line interfaces, and integration APIs. The documentation is automatically generated from source code and maintained alongside the implementation.

## Directory Structure

```
api/
├── rest/                 # REST API documentation
├── python/               # Python API reference
├── cli/                  # Command-line interface documentation
├── integration/          # Integration API documentation
├── examples/             # API usage examples
└── changelog/            # API changes and deprecations
```

## Core Components

### 🌐 REST API Documentation
- **Knowledge API**: Endpoints for accessing knowledge repository
- **Research API**: Endpoints for research tools and experiments
- **Platform API**: Endpoints for platform management
- **Integration API**: Endpoints for external system integration
- **Authentication**: API authentication and authorization

### 🐍 Python API Reference
- **Core Classes**: Main Active Inference classes and interfaces
- **Utilities**: Helper functions and utilities
- **Configuration**: Configuration management APIs
- **Data Structures**: Data structures and models
- **Exceptions**: Exception classes and error handling

### 💻 Command-Line Interface
- **Learning Commands**: Commands for accessing educational content
- **Research Commands**: Commands for research tools and experiments
- **Platform Commands**: Commands for platform management
- **Development Commands**: Commands for development and testing
- **Utility Commands**: General utility commands

### 🔗 Integration APIs
- **Data Import/Export**: APIs for data import and export
- **External Services**: Integration with external platforms
- **Webhooks**: Webhook endpoints and handlers
- **Streaming**: Real-time data streaming APIs

## Getting Started

### For API Users
1. **Explore Available APIs**: Browse available API endpoints and methods
2. **Authentication**: Set up API authentication if required
3. **Study Examples**: Review API usage examples
4. **Test Endpoints**: Test API endpoints with sample data
5. **Integration**: Integrate APIs into your applications

### For API Developers
1. **API Standards**: Follow established API design standards
2. **Documentation**: Document all public APIs comprehensively
3. **Testing**: Implement comprehensive API testing
4. **Versioning**: Follow API versioning best practices
5. **Security**: Implement appropriate security measures

## API Usage Examples

### REST API Usage
```bash
# Get knowledge node
curl -X GET "https://api.activeinference.org/knowledge/nodes/fep_introduction" \
     -H "Authorization: Bearer your_token"

# Search knowledge base
curl -X GET "https://api.activeinference.org/knowledge/search?q=entropy&limit=10" \
     -H "Authorization: Bearer your_token"

# Run experiment
curl -X POST "https://api.activeinference.org/research/experiments" \
     -H "Authorization: Bearer your_token" \
     -H "Content-Type: application/json" \
     -d '{
       "experiment_type": "active_inference_simulation",
       "parameters": {"steps": 100, "agents": 5},
       "repetitions": 10
     }'
```

### Python API Usage
```python
from active_inference.knowledge import KnowledgeRepository
from active_inference.research import ExperimentFramework
from active_inference.applications import TemplateManager

# Initialize knowledge repository
repo = KnowledgeRepository(config={'root_path': './knowledge'})

# Search for knowledge
results = repo.search('entropy', limit=10)
for result in results:
    print(f"Found: {result.title}")

# Run experiment
framework = ExperimentFramework(config={'output_dir': './results'})
experiment = framework.create_experiment('active_inference_study')
results = framework.run_experiment(experiment)

# Use templates
template_manager = TemplateManager()
project = template_manager.create_project_from_template(
    'basic_agent',
    {
        'name': 'my_first_agent',
        'description': 'My first Active Inference agent',
        'author': 'Your Name'
    }
)
```

### Command-Line Usage
```bash
# Learning commands
ai-knowledge learn foundations          # Start foundations learning track
ai-knowledge search "free energy"       # Search knowledge base
ai-knowledge path show complete        # Display learning path
ai-knowledge export pdf foundations    # Export learning path as PDF

# Research commands
ai-research experiments run             # Execute experiment suite
ai-research simulations benchmark       # Run simulation benchmarks
ai-research analyze results             # Analyze experimental results

# Platform commands
ai-platform serve                      # Start web platform
ai-platform status                     # Show system status
ai-platform backup                     # Create system backup
ai-platform update                     # Update platform components
```

## API Reference Structure

### REST API Reference
```rst
REST API Reference
=================

Authentication
--------------

.. automodule:: active_inference.platform.auth
   :members:
   :undoc-members:
   :show-inheritance:

Knowledge API
-------------

.. automodule:: active_inference.knowledge.api
   :members:
   :undoc-members:
   :show-inheritance:

Research API
------------

.. automodule:: active_inference.research.api
   :members:
   :undoc-members:
   :show-inheritance:

Platform API
------------

.. automodule:: active_inference.platform.api
   :members:
   :undoc-members:
   :show-inheritance:
```

### Python API Reference
```rst
Python API Reference
====================

Core Classes
------------

Active Inference Agent
~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: active_inference.core.ActiveInferenceAgent
   :members:
   :undoc-members:
   :show-inheritance:

Generative Model
~~~~~~~~~~~~~~~~

.. autoclass:: active_inference.models.GenerativeModel
   :members:
   :undoc-members:
   :show-inheritance:

Inference Engine
~~~~~~~~~~~~~~~~

.. autoclass:: active_inference.inference.InferenceEngine
   :members:
   :undoc-members:
   :show-inheritance:

Utilities
---------

.. automodule:: active_inference.utils
   :members:
   :undoc-members:
   :show-inheritance:
```

## API Standards

### REST API Standards
- **HTTP Methods**: Use appropriate HTTP methods (GET, POST, PUT, DELETE)
- **Status Codes**: Use standard HTTP status codes consistently
- **Response Format**: Consistent JSON response format
- **Error Handling**: Comprehensive error responses with details
- **Pagination**: Implement pagination for list endpoints
- **Filtering**: Support filtering and search parameters

### Python API Standards
- **Type Hints**: Complete type annotations for all parameters and returns
- **Docstrings**: Comprehensive docstrings following NumPy style
- **Error Handling**: Clear exception hierarchy and error messages
- **Backward Compatibility**: Maintain backward compatibility
- **Performance**: Consider performance implications of API design

### Command-Line Standards
- **Consistent Interface**: Consistent command structure and options
- **Help Documentation**: Comprehensive help for all commands
- **Error Messages**: Clear, actionable error messages
- **Configuration**: Support configuration files and environment variables
- **Output Formats**: Support multiple output formats (JSON, YAML, etc.)

## Contributing

We welcome contributions to the API documentation! See [CONTRIBUTING.md](../../CONTRIBUTING.md) for detailed guidelines.

### Contribution Types
- **API Documentation**: Document new or updated APIs
- **Example Code**: Add API usage examples
- **Error Documentation**: Document error conditions and handling
- **Integration Examples**: Create integration examples
- **Standards Updates**: Update API standards and guidelines

### Quality Standards
- **Complete Coverage**: Document all public APIs comprehensively
- **Working Examples**: All examples must be functional and tested
- **Clear Documentation**: Use clear, accessible language
- **Current Information**: Keep documentation current with API changes
- **Consistency**: Maintain consistency with established patterns

## Learning Resources

- **API Design**: Study REST API and Python API design principles
- **Documentation Standards**: Learn API documentation best practices
- **Example Study**: Study existing API documentation and examples
- **Testing**: Learn API testing methodologies
- **Integration**: Understand integration patterns and practices

## Related Documentation

- **[Documentation README](../README.md)**: Documentation module overview
- **[Main README](../../README.md)**: Project overview and getting started
- **[Platform Documentation](../../platform/)**: Platform APIs and services
- **[Contributing Guide](../../CONTRIBUTING.md)**: Contribution guidelines
- **[Integration Documentation](../../applications/integrations/)**: Integration APIs

## API Testing

### Automated Testing
```python
import pytest
from active_inference.knowledge.api import KnowledgeAPI

class TestKnowledgeAPI:
    """Test cases for Knowledge API"""

    def setup_method(self):
        """Set up test environment"""
        self.api = KnowledgeAPI(test_mode=True)

    def test_search_knowledge(self):
        """Test knowledge search functionality"""
        results = self.api.search('entropy', limit=5)
        assert len(results) <= 5
        assert all('id' in result for result in results)

    def test_get_knowledge_node(self):
        """Test knowledge node retrieval"""
        node = self.api.get_node('fep_introduction')
        assert node is not None
        assert 'content' in node
        assert 'metadata' in node

    def test_invalid_search(self):
        """Test error handling for invalid searches"""
        with pytest.raises(ValueError):
            self.api.search('', limit=-1)
```

### API Documentation Validation
```python
def validate_api_documentation():
    """Validate API documentation completeness"""
    validation_results = {
        'missing_docs': [],
        'incomplete_examples': [],
        'broken_links': [],
        'outdated_info': []
    }

    # Check all API endpoints are documented
    api_endpoints = get_all_api_endpoints()
    documented_endpoints = get_documented_endpoints()

    for endpoint in api_endpoints:
        if endpoint not in documented_endpoints:
            validation_results['missing_docs'].append(endpoint)

    # Check examples are working
    for example in get_all_examples():
        if not test_example(example):
            validation_results['incomplete_examples'].append(example)

    return validation_results
```

---

*"Active Inference for, with, by Generative AI"* - Enabling seamless integration through comprehensive, well-documented APIs and clear integration patterns.
