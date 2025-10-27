# API Documentation - Agent Documentation

This document provides comprehensive guidance for AI agents and contributors working within the API Documentation module of the Active Inference Knowledge Environment. It outlines API documentation standards, generation processes, and best practices for creating comprehensive API reference materials.

## API Documentation Module Overview

The API Documentation module provides complete reference documentation for all public APIs in the Active Inference Knowledge Environment. This includes REST APIs, Python APIs, command-line interfaces, and integration APIs. The documentation is automatically generated from source code where possible and maintained alongside the implementation.

## Core Responsibilities

### API Documentation Generation
- **Auto-Generation**: Generate documentation from source code
- **Manual Documentation**: Create comprehensive manual documentation
- **Example Generation**: Create working API usage examples
- **Validation**: Validate documentation accuracy and completeness
- **Maintenance**: Keep documentation current with API changes

### Documentation Standards
- **Consistency**: Maintain consistent documentation format and style
- **Completeness**: Ensure comprehensive coverage of all APIs
- **Clarity**: Use clear, accessible language and examples
- **Accuracy**: Ensure technical accuracy of all documentation
- **Currency**: Keep documentation current with API changes

### Quality Assurance
- **Review Processes**: Implement documentation review processes
- **Testing**: Test all code examples and API usage
- **Validation**: Validate documentation against actual API behavior
- **User Feedback**: Incorporate user feedback and suggestions
- **Standards Compliance**: Ensure compliance with documentation standards

## Development Workflows

### API Documentation Creation Process
1. **API Analysis**: Analyze API structure and functionality
2. **Documentation Planning**: Plan documentation structure and content
3. **Content Creation**: Write comprehensive API documentation
4. **Example Development**: Create working code examples
5. **Review and Validation**: Review for accuracy and completeness
6. **Integration**: Integrate with documentation build system
7. **Publication**: Publish documentation with proper formatting
8. **Maintenance**: Update documentation as APIs evolve

### Documentation Update Process
1. **Change Detection**: Detect API changes that require documentation updates
2. **Impact Analysis**: Analyze impact of changes on existing documentation
3. **Content Update**: Update documentation to reflect changes
4. **Example Update**: Update code examples if needed
5. **Validation**: Validate updated documentation
6. **Review**: Review changes for accuracy
7. **Publication**: Publish updated documentation
8. **Notification**: Notify users of significant API changes

### Quality Assurance Process
1. **Completeness Check**: Verify all APIs are documented
2. **Accuracy Validation**: Validate documentation accuracy
3. **Example Testing**: Test all code examples
4. **Cross-Reference Check**: Verify cross-references work
5. **Build Testing**: Test documentation build process
6. **User Testing**: Validate documentation usability
7. **Feedback Integration**: Incorporate user feedback

## Quality Standards

### Documentation Quality
- **Completeness**: All public APIs must be fully documented
- **Accuracy**: All technical details must be accurate
- **Clarity**: Documentation must be clear and accessible
- **Consistency**: Consistent format and style throughout
- **Currency**: Documentation must reflect current API state

### Code Example Quality
- **Functionality**: All examples must be functional and tested
- **Relevance**: Examples must be relevant and practical
- **Completeness**: Examples must demonstrate key functionality
- **Clarity**: Examples must be clear and well-commented
- **Error Handling**: Examples should demonstrate proper error handling

### Technical Quality
- **Build Process**: Documentation must build successfully
- **Cross-References**: All cross-references must work
- **Code Syntax**: All code must be syntactically correct
- **API Compatibility**: Examples must work with documented APIs
- **Performance**: Documentation generation should be efficient

## Implementation Patterns

### API Documentation Generator
```python
from typing import Dict, List, Any, Optional
from pathlib import Path
import inspect
import json

class APIDocumentationGenerator:
    """Generate comprehensive API documentation"""

    def __init__(self, source_path: Path, output_path: Path):
        self.source_path = source_path
        self.output_path = output_path
        self.api_info: Dict[str, Any] = {}
        self.load_api_info()

```python
def load_api_info(self) -> None:
    """Load API information from source code"""
    self.api_info = {
        'modules': self.discover_modules(),
        'classes': self.discover_classes(),
        'functions': self.discover_functions(),
        'endpoints': self.discover_endpoints()
    }
```

```python
def discover_modules(self) -> List[Dict[str, Any]]:
    """Discover Python modules and their documentation"""
    modules = []

    for module_file in self.source_path.rglob('*.py'):
        if not module_file.name.startswith('_'):
            module_info = self.extract_module_info(module_file)
            if module_info:
                modules.append(module_info)

    return modules

def extract_module_info(self, module_file: Path) -> Optional[Dict[str, Any]]:
    """Extract information from Python module"""
    try:
        # Import module
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            f"module_{module_file.stem}",
            module_file
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Extract module information
        docstring = inspect.getdoc(module) or ""
        members = self.get_module_members(module)

        return {
            'name': module.__name__,
            'file': str(module_file.relative_to(self.source_path)),
            'docstring': docstring,
            'members': members,
            'classes': [name for name, obj in members.items()
                       if inspect.isclass(obj)],
            'functions': [name for name, obj in members.items()
                         if callable(obj) and not inspect.isclass(obj)]
```
        }

    except Exception as e:
        print(f"Failed to extract info from {module_file}: {e}")
        return None

```python
def get_module_members(self, module) -> Dict[str, Any]:
    """Get public members of module"""
    members = {}
    for name in dir(module):
        if not name.startswith('_'):
            try:
                obj = getattr(module, name)
                if hasattr(obj, '__module__') and obj.__module__ == module.__name__:
                    members[name] = obj
            except:
                continue
    return members
```

    def discover_classes(self) -> List[Dict[str, Any]]:
        """Discover and document classes"""
        classes = []

        for module_info in self.api_info['modules']:
            for class_name in module_info['classes']:
                try:
                    # Import class
                    module_path = module_info['file']
                    module_name = module_path.replace('/', '.').replace('.py', '')
                    module = __import__(module_name, fromlist=[class_name])
                    cls = getattr(module, class_name)

                    class_doc = self.extract_class_info(cls)
                    if class_doc:
                        classes.append(class_doc)

                except Exception as e:
                    print(f"Failed to document class {class_name}: {e}")

        return classes

    def extract_class_info(self, cls) -> Optional[Dict[str, Any]]:
        """Extract comprehensive class information"""
        try:
            docstring = inspect.getdoc(cls) or ""

            # Get methods
            methods = []
            for name, method in inspect.getmembers(cls, predicate=inspect.isfunction):
                if not name.startswith('_'):
                    method_info = self.extract_method_info(method)
                    if method_info:
                        methods.append(method_info)

            # Get properties
            properties = []
            for name in dir(cls):
                if not name.startswith('_'):
                    try:
                        value = getattr(cls, name)
                        if not callable(value):
                            properties.append({
                                'name': name,
                                'value': str(value),
                                'type': type(value).__name__
                            })
                    except:
                        continue

            return {
                'name': cls.__name__,
                'module': cls.__module__,
                'docstring': docstring,
                'methods': methods,
                'properties': properties,
                'base_classes': [base.__name__ for base in cls.__bases__ if base != object]
            }

        except Exception as e:
            print(f"Failed to extract class info for {cls.__name__}: {e}")
            return None

    def extract_method_info(self, method) -> Optional[Dict[str, Any]]:
        """Extract method information"""
        try:
            signature = inspect.signature(method)
            docstring = inspect.getdoc(method) or ""

            return {
                'name': method.__name__,
                'signature': str(signature),
                'parameters': list(signature.parameters.keys()),
                'docstring': docstring,
                'is_static': isinstance(inspect.getattr_static(type, method.__name__, None), staticmethod),
                'is_class': isinstance(inspect.getattr_static(type, method.__name__, None), classmethod)
            }

        except Exception as e:
            print(f"Failed to extract method info for {method.__name__}: {e}")
            return None

    def generate_rst_documentation(self) -> None:
        """Generate reStructuredText documentation"""
        self.generate_main_index()
        self.generate_module_docs()
        self.generate_class_docs()
        self.generate_function_docs()

    def generate_main_index(self) -> None:
        """Generate main API documentation index"""
        index_content = """
API Reference
=============

This section contains comprehensive API reference documentation for the Active Inference Knowledge Environment.

.. toctree::
   :maxdepth: 2
   :caption: API Reference:

   rest_api
   python_api
   cli_api
   integration_api

REST API
--------

Complete reference for REST API endpoints.

Python API
----------

Comprehensive Python API reference with examples.

Command Line Interface
---------------------

Command-line interface documentation and usage examples.

Integration APIs
----------------

APIs for integrating with external systems and services.

Indices and Search
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
"""

        (self.output_path / 'index.rst').write_text(index_content)

    def generate_module_docs(self) -> None:
        """Generate module documentation"""
        modules_dir = self.output_path / 'modules'
        modules_dir.mkdir(exist_ok=True)

        for module_info in self.api_info['modules']:
            module_content = f"""
{module_info['name']}
{'=' * len(module_info['name'])}

.. automodule:: {module_info['name']}
   :members:
   :undoc-members:
   :show-inheritance:

Module Contents
---------------

**Classes:**
"""

            for class_name in module_info['classes']:
                module_content += f"* :class:`{class_name}`\n"

            module_content += "\n**Functions:**\n"
            for func_name in module_info['functions']:
                module_content += f"* :func:`{func_name}`\n"

            if module_info['docstring']:
                module_content += f"\n{module_info['docstring']}\n"

            (modules_dir / f"{module_info['name'].replace('.', '_')}.rst").write_text(module_content)
```

### Documentation Validation Framework
```python
from typing import Dict, List, Any
import requests
import json

class APIDocumentationValidator:
    """Validate API documentation completeness and accuracy"""

    def __init__(self, base_url: str = None, api_spec_path: Path = None):
        self.base_url = base_url
        self.api_spec_path = api_spec_path
        self.validation_results: Dict[str, Any] = {}

    def validate_all_apis(self) -> Dict[str, Any]:
        """Validate all API documentation"""
        self.validation_results = {
            'rest_api': self.validate_rest_api_docs(),
            'python_api': self.validate_python_api_docs(),
            'cli_api': self.validate_cli_api_docs(),
            'integration_api': self.validate_integration_api_docs(),
            'overall_score': 0
        }

        # Calculate overall score
        total_checks = sum(len(result.get('checks', []))
                          for result in self.validation_results.values()
                          if isinstance(result, dict))
        passed_checks = sum(result.get('passed_checks', 0)
                           for result in self.validation_results.values()
                           if isinstance(result, dict))

        self.validation_results['overall_score'] = (
            passed_checks / total_checks if total_checks > 0 else 0
        )

        return self.validation_results

    def validate_rest_api_docs(self) -> Dict[str, Any]:
        """Validate REST API documentation"""
        if not self.base_url:
            return {'error': 'No base URL provided for REST API validation'}

        # Get API specification
        api_spec = self.load_api_specification()

        # Test documented endpoints
        endpoint_tests = []
        for endpoint_info in api_spec.get('endpoints', []):
            test_result = self.test_endpoint_documentation(endpoint_info)
            endpoint_tests.append(test_result)

        # Check documentation coverage
        coverage = self.check_documentation_coverage(api_spec)

        return {
            'endpoint_tests': endpoint_tests,
            'documentation_coverage': coverage,
            'passed_checks': len([t for t in endpoint_tests if t.get('status') == 'pass']),
            'total_checks': len(endpoint_tests)
        }

    def load_api_specification(self) -> Dict[str, Any]:
        """Load API specification"""
        if self.api_spec_path and self.api_spec_path.exists():
            with open(self.api_spec_path, 'r') as f:
                return json.load(f)
        return {}

    def test_endpoint_documentation(self, endpoint_info: Dict[str, Any]) -> Dict[str, Any]:
        """Test single endpoint documentation"""
        endpoint = endpoint_info.get('path')
        method = endpoint_info.get('method', 'GET')

        try:
            # Test actual endpoint
            response = requests.request(
                method=method,
                url=f"{self.base_url}{endpoint}",
                timeout=10
            )

            # Check if documented response matches actual response
            documented_response = endpoint_info.get('response')
            actual_response = {
                'status_code': response.status_code,
                'headers': dict(response.headers),
                'content_type': response.headers.get('content-type', '')
            }

            # Simple comparison (could be enhanced)
            status_match = documented_response.get('status_code') == actual_response['status_code']

            return {
                'endpoint': endpoint,
                'method': method,
                'status': 'pass' if status_match else 'fail',
                'actual_response': actual_response,
                'documented_response': documented_response
            }

        except Exception as e:
            return {
                'endpoint': endpoint,
                'method': method,
                'status': 'error',
                'error': str(e)
            }

    def validate_python_api_docs(self) -> Dict[str, Any]:
        """Validate Python API documentation"""
        # Implementation for Python API validation
        return {'status': 'implemented', 'checks': []}

    def validate_cli_api_docs(self) -> Dict[str, Any]:
        """Validate CLI API documentation"""
        # Implementation for CLI validation
        return {'status': 'implemented', 'checks': []}

    def validate_integration_api_docs(self) -> Dict[str, Any]:
        """Validate integration API documentation"""
        # Implementation for integration API validation
        return {'status': 'implemented', 'checks': []}
```

## Testing Guidelines

### Documentation Testing
- **Build Testing**: Test documentation build process
- **Link Validation**: Verify all internal and external links
- **Example Testing**: Test all code examples
- **API Testing**: Test documented APIs against actual implementations
- **Cross-Reference Testing**: Verify cross-references work correctly

### Content Quality Testing
- **Clarity Testing**: Assess documentation clarity and readability
- **Completeness Testing**: Verify comprehensive API coverage
- **Accuracy Testing**: Validate technical accuracy
- **Consistency Testing**: Ensure consistent documentation style
- **User Experience Testing**: Test documentation usability

## Performance Considerations

### Documentation Generation
- **Generation Speed**: Optimize documentation generation speed
- **Incremental Updates**: Support incremental documentation updates
- **Resource Management**: Manage memory and CPU usage during generation
- **Parallel Processing**: Use parallel processing where beneficial

### Documentation Access
- **Load Time**: Ensure fast documentation loading
- **Search Performance**: Optimize search functionality
- **Navigation Speed**: Ensure fast navigation between sections
- **Caching**: Implement caching for frequently accessed content

## Maintenance and Evolution

### Documentation Updates
- **Automated Updates**: Automate documentation updates where possible
- **Change Detection**: Detect when documentation needs updates
- **Version Management**: Manage documentation versions
- **Archival**: Archive outdated documentation

### Quality Improvement
- **User Feedback**: Incorporate user feedback for improvements
- **Analytics**: Use documentation usage analytics
- **A/B Testing**: Test documentation improvements
- **Standards Evolution**: Update documentation standards

## Common Challenges and Solutions

### Challenge: API Evolution
**Solution**: Implement automated change detection and update processes.

### Challenge: Documentation Currency
**Solution**: Maintain tight integration between code and documentation.

### Challenge: Example Maintenance
**Solution**: Implement automated example testing and validation.

### Challenge: Cross-Reference Management
**Solution**: Use automated cross-reference validation and updating.

## Getting Started as an Agent

### Development Setup
1. **Study API Structure**: Understand current API organization
2. **Learn Documentation Tools**: Master Sphinx and related tools
3. **Practice API Documentation**: Practice documenting APIs
4. **Understand Standards**: Learn API documentation standards

### Contribution Process
1. **Identify Documentation Gaps**: Find missing or outdated API documentation
2. **Research APIs**: Understand API functionality and usage
3. **Document APIs**: Follow documentation standards
4. **Create Examples**: Add practical usage examples
5. **Review and Validate**: Ensure accuracy and completeness
6. **Submit Changes**: Follow contribution process

### Learning Resources
- **API Documentation**: Study API documentation best practices
- **Technical Writing**: Learn technical writing principles
- **Code Analysis**: Practice analyzing code for documentation
- **Example Creation**: Master creating effective code examples
- **Review Processes**: Learn documentation review techniques

## Related Documentation

- **[API README](./README.md)**: API documentation module overview
- **[Documentation AGENTS.md](../AGENTS.md)**: Documentation module guidelines
- **[Main AGENTS.md](../../AGENTS.md)**: Project-wide agent guidelines
- **[Contributing Guide](../../CONTRIBUTING.md)**: Contribution processes
- **[Platform Documentation](../../platform/)**: Platform APIs

---

*"Active Inference for, with, by Generative AI"* - Enabling seamless integration through comprehensive, well-documented APIs and clear integration patterns.



