# Documentation Tools - Agent Development Guide

**Guidelines for AI agents working with documentation tools in the Active Inference Knowledge Environment.**

## 🤖 Agent Role & Responsibilities

**What agents should do when working with documentation tools:**

### Primary Responsibilities
- **Documentation Tool Development**: Create and maintain automated documentation generation tools
- **Quality Assurance**: Build validation and quality control systems for documentation
- **Pattern Analysis**: Extract and formalize documentation patterns from existing content
- **Integration Systems**: Ensure seamless integration with development workflows
- **Maintenance Automation**: Develop tools for documentation maintenance and updates

### Development Focus Areas
1. **Documentation Automation**: Build tools that automate documentation generation and maintenance
2. **Quality Control**: Create systems for validating documentation quality and completeness
3. **Pattern Recognition**: Develop tools for analyzing and extracting documentation patterns
4. **Integration**: Ensure tools integrate seamlessly with development and deployment workflows
5. **Standards Enforcement**: Build tools that enforce documentation standards and consistency

## 🏗️ Architecture & Integration

### Documentation Tools Architecture

**Understanding how documentation tools fit into the development ecosystem:**

```
Documentation Layer
├── Tool Layer ← Documentation Tools
├── Content Layer (generation, validation, analysis)
├── Quality Layer (standards, compliance, improvement)
└── Integration Layer (workflows, automation, maintenance)
```

### Integration Points

**Documentation tools integrate with multiple development and content workflows:**

#### Upstream Components
- **Source Code**: Extract documentation from code structure and comments
- **Development Standards**: Tools must follow and enforce documentation standards
- **Content Repository**: Access knowledge base and educational content
- **Quality Requirements**: Integrate with quality assurance and validation systems

#### Downstream Components
- **Documentation Files**: Generate README.md, AGENTS.md, and API documentation
- **Learning Systems**: Provide content for educational platforms and tutorials
- **Build System**: Integrate with build processes for automated documentation
- **Deployment**: Support documentation deployment and publication workflows

#### External Systems
- **Template Engines**: Integrate with Jinja2, Mako, or custom template systems
- **Static Site Generators**: Work with Sphinx, MkDocs, and documentation platforms
- **Version Control**: Integrate with Git for documentation version management
- **CI/CD Systems**: Provide documentation validation in continuous integration

### Documentation Workflow Integration

```python
# Documentation tools workflow
source_code → analysis → generation → validation → integration → publication
content_changes → detection → update → validation → review → deployment
```

## 💻 Development Patterns

### Required Implementation Patterns

**All documentation tool development must follow these patterns:**

#### 1. Documentation Tool Factory Pattern (PREFERRED)
```python
def create_documentation_tool(tool_type: str, category: str, config: Dict[str, Any]) -> DocumentationTool:
    """Create documentation tool using factory pattern with validation"""

    # Documentation tool registry organized by category
    tool_registry = {
        'generator': {
            'api': APIDocumentationGenerator,
            'component': ComponentDocumentationGenerator,
            'tutorial': TutorialDocumentationGenerator,
            'comprehensive': ComprehensiveDocumentationGenerator
        },
        'validator': {
            'comprehensive': ComprehensiveDocumentationValidator,
            'structure': StructureDocumentationValidator,
            'quality': QualityDocumentationValidator,
            'standards': StandardsDocumentationValidator
        },
        'analyzer': {
            'patterns': PatternDocumentationAnalyzer,
            'coverage': CoverageDocumentationAnalyzer,
            'quality': QualityDocumentationAnalyzer,
            'consistency': ConsistencyDocumentationAnalyzer
        },
        'maintainer': {
            'updater': DocumentationUpdater,
            'linker': CrossReferenceLinker,
            'optimizer': DocumentationOptimizer,
            'publisher': DocumentationPublisher
        }
    }

    if category not in tool_registry or tool_type not in tool_registry[category]:
        raise DocumentationError(f"Unknown documentation tool: {category}.{tool_type}")

    # Validate documentation context
    validate_documentation_context(config)

    # Create tool with documentation validation
    tool = tool_registry[category][tool_type](config)

    # Validate tool functionality
    validate_tool_functionality(tool)

    # Validate integration capability
    validate_integration_capability(tool)

    return tool

def validate_documentation_context(config: Dict[str, Any]) -> None:
    """Validate documentation tool context and requirements"""
    required_fields = {'project_root', 'output_format', 'validation_level'}

    for field in required_fields:
        if field not in config:
            raise DocumentationError(f"Missing required documentation field: {field}")

    # Validate project structure
    if not os.path.exists(config['project_root']):
        raise DocumentationError(f"Project root does not exist: {config['project_root']}")

    # Validate output format
    valid_formats = {'markdown', 'html', 'pdf', 'rst', 'json'}
    if config['output_format'] not in valid_formats:
        raise DocumentationError(f"Invalid output format: {config['output_format']}")

    # Validate validation level
    valid_levels = {'basic', 'standard', 'comprehensive', 'strict'}
    if config['validation_level'] not in valid_levels:
        raise DocumentationError(f"Invalid validation level: {config['validation_level']}")
```

#### 2. Documentation Tool Configuration Pattern (MANDATORY)
```python
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from enum import Enum

class DocumentationCategory(Enum):
    """Documentation tool categories"""
    GENERATOR = "generator"
    VALIDATOR = "validator"
    ANALYZER = "analyzer"
    MAINTAINER = "maintainer"
    INTEGRATOR = "integrator"

class OutputFormat(Enum):
    """Documentation output formats"""
    MARKDOWN = "markdown"
    HTML = "html"
    PDF = "pdf"
    RST = "rst"
    JSON = "json"

class ValidationLevel(Enum):
    """Documentation validation levels"""
    BASIC = "basic"
    STANDARD = "standard"
    COMPREHENSIVE = "comprehensive"
    STRICT = "strict"

@dataclass
class DocumentationToolConfig:
    """Documentation tool configuration with validation"""

    # Required tool fields
    tool_name: str
    category: DocumentationCategory
    output_format: OutputFormat

    # Project context
    project_root: str = "."
    output_directory: str = "docs"
    source_directories: List[str] = field(default_factory=list)

    # Validation settings
    validation_level: ValidationLevel = ValidationLevel.STANDARD
    quality_threshold: float = 0.9
    auto_fix: bool = True

    # Content settings
    include_examples: bool = True
    include_api_docs: bool = True
    include_tutorials: bool = True

    # Integration settings
    integration: Dict[str, Any] = field(default_factory=lambda: {
        "ci_cd_integration": True,
        "version_control_hooks": True,
        "auto_update": True,
        "cross_references": True
    })

    # Performance settings
    performance: Dict[str, Any] = field(default_factory=lambda: {
        "parallel_processing": True,
        "caching_enabled": True,
        "incremental_updates": True,
        "memory_optimization": True
    })

    def validate(self) -> List[str]:
        """Validate documentation tool configuration"""
        errors = []

        # Validate required fields
        if not self.tool_name or not self.tool_name.strip():
            errors.append("Tool name cannot be empty")

        if not self.project_root or not os.path.exists(self.project_root):
            errors.append(f"Project root does not exist: {self.project_root}")

        # Validate source directories
        for source_dir in self.source_directories:
            if not os.path.exists(source_dir):
                errors.append(f"Source directory does not exist: {source_dir}")

        # Validate quality threshold
        if not 0.0 <= self.quality_threshold <= 1.0:
            errors.append("Quality threshold must be between 0.0 and 1.0")

        # Category-specific validation
        if self.category == DocumentationCategory.GENERATOR:
            if not self.output_directory:
                errors.append("Generator tools require output_directory")

        if self.category == DocumentationCategory.VALIDATOR:
            if self.quality_threshold < 0.8:
                errors.append("Validator tools require quality_threshold >= 0.8")

        return errors

    def get_execution_context(self) -> Dict[str, Any]:
        """Get execution context for tool initialization"""
        return {
            "tool": self.tool_name,
            "category": self.category.value,
            "format": self.output_format.value,
            "project": self.project_root,
            "output": self.output_directory,
            "sources": self.source_directories,
            "validation": {
                "level": self.validation_level.value,
                "threshold": self.quality_threshold,
                "auto_fix": self.auto_fix
            },
            "content": {
                "examples": self.include_examples,
                "api_docs": self.include_api_docs,
                "tutorials": self.include_tutorials
            },
            "integration": self.integration,
            "performance": self.performance
        }
```

#### 3. Documentation Tool Error Handling Pattern (MANDATORY)
```python
import logging
from typing import Callable, Any, Dict, Optional
from contextlib import contextmanager

logger = logging.getLogger(__name__)

class DocumentationError(Exception):
    """Base exception for documentation tool errors"""
    pass

class GenerationError(DocumentationError):
    """Documentation generation errors"""
    pass

class ValidationError(DocumentationError):
    """Documentation validation errors"""
    pass

@contextmanager
def documentation_execution_context(tool_name: str, operation: str, config: Dict[str, Any]):
    """Context manager for documentation tool execution"""

    documentation_context = {
        "tool": tool_name,
        "operation": operation,
        "config": config,
        "start_time": time.time(),
        "status": "starting",
        "documentation_metrics": {}
    }

    try:
        logger.info(f"Starting documentation operation: {tool_name}.{operation}", extra={
            "documentation_context": documentation_context
        })

        documentation_context["status"] = "running"
        yield documentation_context

        documentation_context["status"] = "completed"
        documentation_context["end_time"] = time.time()
        documentation_context["duration"] = documentation_context["end_time"] - documentation_context["start_time"]

        logger.info(f"Documentation operation completed: {tool_name}.{operation}", extra={
            "documentation_context": documentation_context
        })

    except ValidationError as e:
        documentation_context["status"] = "validation_failed"
        documentation_context["error"] = str(e)
        logger.error(f"Documentation validation failed: {tool_name}.{operation}", extra={
            "documentation_context": documentation_context
        })
        raise

    except GenerationError as e:
        documentation_context["status"] = "generation_failed"
        documentation_context["error"] = str(e)
        logger.error(f"Documentation generation failed: {tool_name}.{operation}", extra={
            "documentation_context": documentation_context
        })
        raise

    except Exception as e:
        documentation_context["status"] = "documentation_error"
        documentation_context["error"] = str(e)
        documentation_context["traceback"] = traceback.format_exc()
        logger.error(f"Documentation operation error: {tool_name}.{operation}", extra={
            "documentation_context": documentation_context
        })
        raise DocumentationError(f"Documentation operation failed: {tool_name}.{operation}") from e

def execute_documentation_operation(tool_name: str, operation: str, func: Callable, config: Dict[str, Any], **kwargs) -> Any:
    """Execute documentation operation with comprehensive error handling"""
    with documentation_execution_context(tool_name, operation, config) as context:
        return func(**kwargs)
```

## 🧪 Testing Standards

### Documentation Tool Testing Categories (MANDATORY)

#### 1. Documentation Generation Tests (`tests/test_documentation_generation.py`)
**Test documentation generation from various sources:**
```python
def test_api_documentation_generation():
    """Test API documentation generation from source code"""
    config = DocumentationToolConfig(
        tool_name="api_documentation_generator",
        category=DocumentationCategory.GENERATOR,
        output_format=OutputFormat.MARKDOWN,
        project_root="src/",
        source_directories=["src/active_inference/"],
        include_examples=True,
        include_api_docs=True
    )

    # Create and configure tool
    tool = create_documentation_tool(config.tool_name, config.category.value, config.to_dict())

    # Test API documentation generation
    result = tool.generate_api_docs()

    # Validate generation
    assert result["status"] == "completed"
    assert result["coverage"]["overall"] >= 0.95
    assert len(result["documentation"]) > 1000

    # Validate documentation structure
    structure_validation = tool.validate_documentation_structure(result["documentation"])
    assert structure_validation["valid"] == True

    # Validate cross-references
    reference_validation = tool.validate_cross_references(result["documentation"])
    assert reference_validation["broken_links"] == 0

def test_component_documentation_generation():
    """Test component documentation generation"""
    config = DocumentationToolConfig(
        tool_name="component_documentation_generator",
        category=DocumentationCategory.GENERATOR,
        output_format=OutputFormat.MARKDOWN,
        project_root="knowledge/",
        validation_level=ValidationLevel.COMPREHENSIVE,
        include_examples=True
    )

    # Create and configure tool
    tool = create_documentation_tool(config.tool_name, config.category.value, config.to_dict())

    # Test component documentation generation
    result = tool.generate_component_docs("foundations")

    # Validate component documentation
    assert "readme" in result
    assert "agents" in result
    assert result["validation"]["overall_status"] == "valid"

    # Validate README structure follows standards
    readme_validation = tool.validate_readme_structure(result["readme"])
    assert readme_validation["sections_complete"] == True
    assert readme_validation["examples_present"] == True

    # Validate AGENTS structure follows patterns
    agents_validation = tool.validate_agents_structure(result["agents"])
    assert agents_validation["patterns_documented"] == True
    assert agents_validation["integration_guidelines"] == True
```

#### 2. Documentation Validation Tests (`tests/test_documentation_validation.py`)
**Test documentation validation and quality assurance:**
```python
def test_comprehensive_documentation_validation():
    """Test comprehensive documentation validation"""
    config = DocumentationToolConfig(
        tool_name="comprehensive_documentation_validator",
        category=DocumentationCategory.VALIDATOR,
        output_format=OutputFormat.JSON,
        project_root=".",
        validation_level=ValidationLevel.STRICT,
        quality_threshold=0.95,
        auto_fix=True
    )

    # Create and configure validator
    tool = create_documentation_tool(config.tool_name, config.category.value, config.to_dict())

    # Test comprehensive validation
    result = tool.validate_all_documentation()

    # Validate comprehensive validation
    assert result["overall_status"] in ["valid", "incomplete"]
    assert "completeness_score" in result
    assert "components" in result
    assert result["completeness_score"] >= config.quality_threshold

    # Validate component-level validation
    for component_name, component_validation in result["components"].items():
        assert "status" in component_validation
        assert "readme" in component_validation
        assert "agents" in component_validation

def test_documentation_quality_validation():
    """Test documentation quality metrics and validation"""
    config = DocumentationToolConfig(
        tool_name="quality_documentation_validator",
        category=DocumentationCategory.VALIDATOR,
        output_format=OutputFormat.JSON,
        project_root="knowledge/",
        validation_level=ValidationLevel.COMPREHENSIVE
    )

    # Create and configure validator
    tool = create_documentation_tool(config.tool_name, config.category.value, config.to_dict())

    # Test quality validation
    quality_result = tool.validate_documentation_quality()

    # Validate quality metrics
    assert "completeness" in quality_result
    assert "accuracy" in quality_result
    assert "consistency" in quality_result
    assert "accessibility" in quality_result

    # Validate quality scores
    for metric, score in quality_result.items():
        if isinstance(score, (int, float)):
            assert 0.0 <= score <= 1.0, f"Quality metric {metric} must be between 0 and 1"

    # Validate overall quality
    overall_quality = tool.calculate_overall_quality(quality_result)
    assert overall_quality >= 0.8, "Overall documentation quality should be at least 80%"
```

#### 3. Documentation Integration Tests (`tests/test_documentation_integration.py`)
**Test documentation tool integration with development workflows:**
```python
def test_documentation_generation_integration():
    """Test documentation generation integration with build system"""
    config = DocumentationToolConfig(
        tool_name="comprehensive_documentation_generator",
        category=DocumentationCategory.GENERATOR,
        output_format=OutputFormat.MARKDOWN,
        project_root=".",
        integration={"ci_cd_integration": True, "auto_update": True}
    )

    # Create integrated documentation system
    generator = create_documentation_tool(config.tool_name, config.category.value, config.to_dict())
    validator = create_documentation_tool("comprehensive_documentation_validator", "validator", config.to_dict())

    # Test integrated workflow
    generation_result = generator.generate_all_docs()

    # Validate generated documentation
    validation_result = validator.validate_documentation(generation_result["documentation"])

    # Test integration
    assert generation_result["status"] == "completed"
    assert validation_result["overall_status"] == "valid"

    # Test CI/CD integration
    ci_validation = tool.validate_ci_cd_integration()
    assert ci_validation["pipeline_compatible"] == True

def test_cross_tool_compatibility():
    """Test compatibility between different documentation tools"""
    # Generator tool
    generator_config = DocumentationToolConfig(
        tool_name="api_documentation_generator",
        category=DocumentationCategory.GENERATOR,
        output_format=OutputFormat.MARKDOWN
    )

    # Validator tool
    validator_config = DocumentationToolConfig(
        tool_name="comprehensive_documentation_validator",
        category=DocumentationCategory.VALIDATOR,
        output_format=OutputFormat.JSON
    )

    # Test cross-compatibility
    generator = create_documentation_tool(generator_config.tool_name, generator_config.category.value, generator_config.to_dict())
    validator = create_documentation_tool(validator_config.tool_name, validator_config.category.value, validator_config.to_dict())

    # Generate documentation
    generated_docs = generator.generate_api_docs()

    # Validate generated documentation
    validation_result = validator.validate_api_documentation(generated_docs)

    # Test compatibility
    assert validation_result["generation_compatible"] == True
    assert validation_result["format_compatible"] == True
```

### Documentation Tool Test Coverage Requirements

- **Generation Functionality**: 100% coverage of documentation generation paths
- **Validation Logic**: 100% coverage of validation and quality checking
- **Integration Points**: 95% coverage of tool integration scenarios
- **Error Handling**: 100% coverage of documentation error conditions
- **Performance Paths**: 90% coverage of performance-critical operations

### Documentation Tool Testing Commands

```bash
# Run all documentation tool tests
make test-documentation-tools

# Run generation tests
pytest tools/documentation/tests/test_documentation_generation.py -v

# Run validation tests
pytest tools/documentation/tests/test_documentation_validation.py -v --tb=short

# Run integration tests
pytest tools/documentation/tests/test_documentation_integration.py -v

# Check documentation tool test coverage
pytest tools/documentation/ --cov=tools/documentation/ --cov-report=html --cov-fail-under=95
```

## 📖 Documentation Standards

### Documentation Tool Requirements (MANDATORY)

#### 1. Tool Interface Documentation
**Every documentation tool must document its interface:**
```python
def document_tool_interface():
    """
    Tool Interface Documentation: API Documentation Generator

    This tool generates comprehensive API documentation from Python source code,
    extracting classes, functions, methods, and their documentation.

    Interface Specification:
    - Input: Python source code directories and configuration
    - Output: Markdown documentation with API reference
    - Configuration: YAML-based tool configuration
    - Integration: Command-line and programmatic interfaces

    Core Methods:
    - generate_api_docs(source_paths): Generate API docs from source directories
    - extract_api_elements(api_structure): Extract API elements from code analysis
    - render_api_documentation(elements): Render documentation from extracted elements
    - validate_api_documentation(docs): Validate generated documentation quality

    Configuration Options:
    - output_format: markdown, html, rst, or json documentation formats
    - validation_level: basic, standard, comprehensive, or strict validation
    - include_examples: boolean for including code examples in documentation
    - include_inheritance: boolean for including class inheritance diagrams

    Error Handling:
    - DocumentationError: Base documentation tool errors
    - GenerationError: Documentation generation failures
    - ValidationError: Documentation validation failures
    - IntegrationError: Tool integration errors
    """
    pass
```

#### 2. Quality Metrics Documentation
**All tools must document quality validation:**
```python
def document_quality_metrics():
    """
    Quality Metrics Documentation: Documentation Validation

    This tool validates documentation quality using comprehensive metrics:

    Completeness Metrics:
    - Section Coverage: Percentage of required documentation sections present
    - API Coverage: Percentage of public APIs documented
    - Example Coverage: Percentage of code examples provided
    - Cross-Reference Coverage: Percentage of internal links working

    Accuracy Metrics:
    - Technical Accuracy: Validation of mathematical and conceptual correctness
    - Link Accuracy: Validation of internal and external reference validity
    - Code Accuracy: Validation that code examples execute correctly
    - Format Accuracy: Validation of documentation format compliance

    Consistency Metrics:
    - Style Consistency: Uniform documentation style and formatting
    - Terminology Consistency: Consistent use of technical terminology
    - Structure Consistency: Uniform documentation structure across components
    - Cross-Reference Consistency: Consistent linking and navigation patterns

    Accessibility Metrics:
    - Language Clarity: Readability and clarity of technical explanations
    - Progressive Disclosure: Appropriate complexity progression
    - Multiple Formats: Support for different learning styles and preferences
    - Navigation Quality: Clear navigation and information architecture
    """
    pass
```

## 🚀 Performance Optimization

### Documentation Tool Performance Requirements

**Documentation tools must meet performance standards for development workflows:**

- **Documentation Generation**: <30s for complete project documentation
- **Validation Execution**: <10s for comprehensive validation
- **Pattern Analysis**: <15s for complete codebase pattern analysis
- **Integration Operations**: <5s for tool integration and updates

### Documentation Optimization Techniques

#### 1. Intelligent Documentation Caching
```python
class DocumentationCache:
    """Intelligent caching for documentation operations"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.generation_cache: Dict[str, Any] = {}
        self.validation_cache: Dict[str, Any] = {}
        self.analysis_cache: Dict[str, Any] = {}

    def get_cached_documentation(self, cache_key: str) -> Optional[Any]:
        """Get cached documentation if available"""
        if cache_key in self.generation_cache:
            return self.generation_cache[cache_key]

        return None

    def cache_documentation(self, cache_key: str, documentation: Any) -> None:
        """Cache documentation for future use"""
        self.generation_cache[cache_key] = documentation

        # Manage cache size
        if len(self.generation_cache) > self.config.get("max_cache_size", 500):
            self._evict_documentation_cache()

    def cache_validation_results(self, validation_key: str, results: Any) -> None:
        """Cache validation results for performance"""
        self.validation_cache[validation_key] = results

    def _create_cache_key(self, operation: str, inputs: Dict[str, Any]) -> str:
        """Create deterministic cache key for documentation operations"""
        import hashlib
        inputs_json = json.dumps(inputs, sort_keys=True)
        return hashlib.md5(f"{operation}:{inputs_json}".encode()).hexdigest()
```

#### 2. Parallel Documentation Processing
```python
class ParallelDocumentationProcessor:
    """Parallel processing for documentation operations"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.executor = ThreadPoolExecutor(max_workers=config.get("max_workers", 4))
        self.semaphore = Semaphore(config.get("max_concurrent_operations", 8))

    async def process_documentation_parallel(self, operations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process multiple documentation operations in parallel"""

        async def process_single_operation(operation: Dict[str, Any]) -> Dict[str, Any]:
            """Process single documentation operation with semaphore"""
            async with self.semaphore:
                return await self._execute_documentation_operation(operation)

        # Create tasks for all operations
        tasks = [process_single_operation(op) for op in operations]

        # Execute in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append({
                    "operation": operations[i]["name"],
                    "status": "error",
                    "error": str(result)
                })
            else:
                processed_results.append(result)

        return processed_results

    async def _execute_documentation_operation(self, operation: Dict[str, Any]) -> Dict[str, Any]:
        """Execute documentation operation with resource management"""
        try:
            # Resource allocation for documentation processing
            resources = self._allocate_documentation_resources(operation)

            # Execute documentation operation
            result = await self._execute_documentation_logic(operation, resources)

            # Resource cleanup
            self._cleanup_documentation_resources(resources)

            return result

        except Exception as e:
            self.logger.error(f"Documentation operation failed: {operation['name']} - {e}")
            raise
```

## 🔒 Documentation Security Standards

### Documentation Tool Security (MANDATORY)

#### 1. Documentation Input Validation
```python
def validate_documentation_inputs(self, inputs: Dict[str, Any]) -> None:
    """Validate documentation tool inputs for security"""

    # Validate input paths
    for input_name, input_value in inputs.items():
        if input_name.endswith('_path') and isinstance(input_value, str):
            if not self._is_safe_path(input_value):
                raise ValidationError(f"Unsafe path in {input_name}")

    # Validate against injection attacks
    for input_name, input_value in inputs.items():
        if isinstance(input_value, str) and self._contains_injection_patterns(input_value):
            raise ValidationError(f"Potential injection detected in {input_name}")

    # Validate output paths
    if 'output_path' in inputs:
        output_path = inputs['output_path']
        if not self._is_safe_output_path(output_path):
            raise ValidationError(f"Unsafe output path: {output_path}")

def _is_safe_path(self, path: str) -> bool:
    """Check if path is safe and within allowed directories"""
    resolved_path = os.path.abspath(os.path.expanduser(path))

    # Check against allowed directories
    allowed_dirs = self.config.get("allowed_directories", [])
    for allowed_dir in allowed_dirs:
        if resolved_path.startswith(os.path.abspath(allowed_dir)):
            return True

    return False

def _is_safe_output_path(self, path: str) -> bool:
    """Check if output path is safe for writing"""
    resolved_path = os.path.abspath(os.path.expanduser(path))

    # Check if path is within project directory
    project_root = os.path.abspath(self.config.get("project_root", "."))
    if not resolved_path.startswith(project_root):
        return False

    # Check if path doesn't contain dangerous components
    dangerous_components = ['../', '..\\', '/etc', '/bin', '/usr']
    for component in dangerous_components:
        if component in path:
            return False

    return True
```

#### 2. Documentation Sandboxing
```python
class DocumentationSandbox:
    """Secure sandbox for documentation processing"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.allowed_operations = self._get_allowed_operations()
        self.resource_limits = self._get_resource_limits()

    def process_documentation_safely(self, documentation_function: Callable, *args, **kwargs) -> Any:
        """Process documentation in secure sandbox"""

        # Set resource limits for documentation processing
        with self._resource_limits():
            # Create restricted execution environment
            sandbox_globals = {
                '__builtins__': self._get_restricted_builtins(),
                'json': json,
                'os': self._get_restricted_os(),
                'sys': self._get_restricted_sys(),
                're': re,
                'datetime': datetime
            }

            # Execute in sandbox
            try:
                return documentation_function(*args, **kwargs)
            except Exception as e:
                raise GenerationError(f"Documentation processing failed: {e}")

    def _get_restricted_builtins(self) -> Dict[str, Any]:
        """Get restricted builtin functions for documentation processing"""
        return {
            'str': str,
            'int': int,
            'float': float,
            'bool': bool,
            'list': list,
            'dict': dict,
            'len': len,
            'range': range,
            'print': print,
            'isinstance': isinstance,
            'hasattr': hasattr,
            'getattr': getattr
        }
```

## 🔄 Development Workflow

### Documentation Tool Development Process

1. **Documentation Requirements Analysis**
   - Analyze existing documentation patterns and gaps
   - Identify opportunities for documentation automation
   - Study documentation standards and best practices

2. **Tool Design and Architecture**
   - Design tool following documentation tool best practices
   - Plan comprehensive testing for documentation functionality
   - Consider integration with documentation ecosystem

3. **Tool Implementation with TDD**
   - Write comprehensive tests for documentation functionality
   - Implement core documentation features
   - Validate against documentation requirements

4. **Tool Integration and Validation**
   - Integrate tool with documentation ecosystem
   - Validate performance with documentation workloads
   - Test integration with other documentation tools

5. **Quality Assurance and Standards**
   - Comprehensive testing of documentation functionality
   - Performance validation under documentation loads
   - Standards compliance and quality assurance

### Documentation Tool Review Checklist

**Before submitting documentation tools for review:**

- [ ] **Documentation Standards**: Tool follows established documentation standards
- [ ] **Quality Enhancement**: Tool improves documentation quality and completeness
- [ ] **Automation Effectiveness**: Tool effectively automates documentation processes
- [ ] **Integration Compatibility**: Seamless integration with documentation workflows
- [ ] **Performance Standards**: Meets documentation performance requirements
- [ ] **Security Compliance**: Secure documentation processing and validation
- [ ] **Usability**: Clear interface and comprehensive documentation

## 📚 Learning Resources

### Documentation Tool Resources

- **[Main Documentation Tools](README.md)**: Documentation tool overview
- **[Template System](../../templates/README.md)**: Documentation templates
- **[Development Standards](../../../.cursorrules)**: Documentation quality standards

### Documentation Technical References

- **[Documentation Generation](https://docs-generation.org)**: Documentation automation techniques
- **[API Documentation Standards](https://api-docs-standards.org)**: API documentation best practices
- **[Technical Writing](https://technical-writing.org)**: Technical documentation principles
- **[Markdown Processing](https://markdown-processing.org)**: Markdown documentation tools

### Quality Assurance References

Study these areas for documentation tool development:

- **[Documentation Quality](https://docs-quality.org)**: Documentation quality assessment
- **[Content Validation](https://content-validation.org)**: Content validation techniques
- **[Standards Compliance](https://standards-compliance.org)**: Standards enforcement
- **[Automated Testing](https://automated-testing.org)**: Testing documentation tools

## 🎯 Success Metrics

### Documentation Tool Impact Metrics

- **Documentation Quality**: Tools improve documentation quality by 50%+
- **Generation Speed**: Tools accelerate documentation generation by 80%+
- **Coverage Improvement**: Tools increase documentation coverage by 40%+
- **Maintenance Efficiency**: Tools reduce documentation maintenance effort by 60%+
- **Standards Compliance**: Tools ensure 100% compliance with documentation standards

### Development Metrics

- **Tool Quality**: High-quality, well-tested documentation tools
- **Performance**: Meets documentation workflow performance requirements
- **Reliability**: Zero failures in documentation operations
- **Maintainability**: Clean, well-documented tool code
- **Integration Success**: Seamless integration with documentation ecosystem

---

**Documentation Tools**: Version 1.0.0 | **Last Updated**: October 2024

*"Active Inference for, with, by Generative AI"* - Enhancing documentation through intelligent automation and comprehensive quality assurance.

