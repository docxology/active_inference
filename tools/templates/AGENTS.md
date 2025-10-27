# Development Templates - Agent Development Guide

**Guidelines for AI agents working with development templates in the Active Inference Knowledge Environment.**

## 🤖 Agent Role & Responsibilities

**What agents should do when working with development templates:**

### Primary Responsibilities
- **Template Development**: Create and maintain development templates and patterns
- **Pattern Standardization**: Ensure templates embody established development standards
- **Quality Integration**: Embed quality assurance requirements into templates
- **Usage Documentation**: Provide comprehensive template usage documentation
- **Template Validation**: Implement validation rules for generated content

### Development Focus Areas
1. **Documentation Templates**: Develop README.md and AGENTS.md templates for components
2. **Code Templates**: Create implementation patterns and code structure templates
3. **Testing Templates**: Generate unit, integration, and performance test templates
4. **Configuration Templates**: Develop configuration file and validation templates
5. **Project Templates**: Create complete project structure and workflow templates

## 🏗️ Architecture & Integration

### Template System Architecture

**Understanding how the template system fits into the larger development ecosystem:**

```
Development Workflow Layer
├── Template System (Code generation, documentation, patterns)
├── Development Tools (Quality assurance, validation, automation)
├── Code Implementation (Following generated patterns)
└── Quality Assurance (Validation against standards)
```

### Integration Points

**Key integration points and dependencies:**

#### Upstream Components
- **Development Tools**: Template system integration with development toolchain
- **Quality Standards**: Templates embody and enforce quality requirements
- **Pattern Library**: Centralized repository of established development patterns
- **Documentation System**: Integration with documentation generation tools

#### Downstream Components
- **Component Development**: Generated templates guide component implementation
- **Code Generation**: Templates provide starting point for new development
- **Quality Assurance**: Templates include validation and quality gates
- **Onboarding**: Templates accelerate new contributor productivity

#### External Systems
- **Code Generation Tools**: Jinja2, Cookiecutter, template engines
- **Documentation Tools**: Sphinx, MkDocs, automated documentation generation
- **Quality Tools**: Black, isort, flake8, mypy for code quality enforcement
- **Development Environments**: IDE integration and development tool support

### Template Flow Patterns

```python
# Typical template workflow
requirements → template_selection → customization → generation → validation → integration
```

## 💻 Development Patterns

### Required Implementation Patterns

**All template development must follow these patterns:**

#### 1. Template Factory Pattern (PREFERRED)

```python
def create_template(template_type: str, config: Dict[str, Any]) -> Template:
    """Create template using factory pattern with validation"""

    template_factories = {
        'component_readme': create_component_readme_template,
        'component_agents': create_component_agents_template,
        'code_pattern': create_code_pattern_template,
        'test_structure': create_test_structure_template,
        'configuration': create_configuration_template,
        'project_scaffold': create_project_scaffold_template
    }

    if template_type not in template_factories:
        raise ValueError(f"Unknown template type: {template_type}")

    # Validate template configuration
    validate_template_config(config)

    # Create and validate template
    template = template_factories[template_type](config)
    validate_template_functionality(template)

    return template

def validate_template_config(config: Dict[str, Any]) -> None:
    """Validate template configuration parameters"""

    required_fields = ['template_name', 'category', 'version']

    for field in required_fields:
        if field not in config:
            raise TemplateConfigurationError(f"Missing required field: {field}")

    # Category-specific validation
    if config['category'] == 'component':
        validate_component_template_config(config)
    elif config['category'] == 'code':
        validate_code_template_config(config)
    elif config['category'] == 'test':
        validate_test_template_config(config)
```

#### 2. Template Configuration Pattern (MANDATORY)

```python
from dataclasses import dataclass
from typing import Dict, Any, List, Optional

@dataclass
class TemplateConfig:
    """Configuration for development templates"""

    # Core template configuration
    template_name: str
    template_type: str
    version: str = "1.0.0"
    category: str = "general"

    # Template behavior
    auto_validate: bool = True
    enforce_standards: bool = True
    generate_examples: bool = True
    include_documentation: bool = True

    # Quality settings
    quality_gates: List[str] = None
    validation_rules: Dict[str, Any] = None
    code_style: str = "black"

    # Integration settings
    target_framework: str = "active_inference"
    compatibility_version: str = "1.0.0"
    dependency_requirements: List[str] = None

    def __post_init__(self):
        """Validate configuration after initialization"""
        if self.quality_gates is None:
            self.quality_gates = ["syntax", "style", "documentation", "tests"]
        if self.validation_rules is None:
            self.validation_rules = {}
        if self.dependency_requirements is None:
            self.dependency_requirements = []

    def to_template_data(self) -> Dict[str, Any]:
        """Convert configuration to template data"""
        return {
            'template_name': self.template_name,
            'template_type': self.template_type,
            'version': self.version,
            'category': self.category,
            'auto_validate': self.auto_validate,
            'enforce_standards': self.enforce_standards,
            'generate_examples': self.generate_examples,
            'include_documentation': self.include_documentation,
            'quality_gates': self.quality_gates,
            'validation_rules': self.validation_rules,
            'code_style': self.code_style,
            'target_framework': self.target_framework,
            'compatibility_version': self.compatibility_version,
            'dependency_requirements': self.dependency_requirements
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TemplateConfig':
        """Create configuration from dictionary"""
        return cls(**data)
```

#### 3. Template Validation Pattern (MANDATORY)

```python
def validate_generated_content(content: str, validation_rules: Dict[str, Any]) -> ValidationResult:
    """Validate generated content against quality standards"""

    validation_result = ValidationResult(valid=True, errors=[], warnings=[])

    # Syntax validation
    if validation_rules.get('syntax_check', True):
        syntax_result = validate_syntax(content)
        if not syntax_result['valid']:
            validation_result.valid = False
            validation_result.errors.extend(syntax_result['errors'])

    # Style validation
    if validation_rules.get('style_check', True):
        style_result = validate_style(content)
        if not style_result['compliant']:
            validation_result.warnings.extend(style_result['issues'])

    # Documentation validation
    if validation_rules.get('documentation_check', True):
        doc_result = validate_documentation(content)
        if not doc_result['complete']:
            validation_result.errors.extend(doc_result['missing'])

    # Quality gate validation
    if validation_rules.get('quality_gates', True):
        quality_result = validate_quality_gates(content)
        if not quality_result['passed']:
            validation_result.valid = False
            validation_result.errors.extend(quality_result['failures'])

    return validation_result

def apply_quality_gates(content: str, quality_gates: List[str]) -> str:
    """Apply quality gates to generated content"""

    # Apply each quality gate
    for gate in quality_gates:
        if gate == "syntax":
            content = apply_syntax_validation(content)
        elif gate == "style":
            content = apply_style_formatting(content)
        elif gate == "documentation":
            content = ensure_documentation_completeness(content)
        elif gate == "imports":
            content = validate_and_fix_imports(content)
        elif gate == "types":
            content = add_type_annotations(content)

    return content
```

## 🧪 Template Testing Standards

### Template Testing Categories (MANDATORY)

#### 1. Template Generation Tests
**Test template generation produces valid output:**

```python
def test_template_generation():
    """Test template generation produces valid output"""
    # Test README template generation
    readme_template = get_template("component_README_template.md")
    test_data = create_test_component_data()

    generated_content = readme_template.render(test_data)
    assert validate_markdown_syntax(generated_content)

    # Test AGENTS template generation
    agents_template = get_template("component_AGENTS_template.md")
    generated_content = agents_template.render(test_data)
    assert validate_agents_documentation(generated_content)

def test_template_completeness():
    """Test generated templates are complete"""
    template = get_template("code_pattern_template.py")
    test_config = create_test_pattern_config()

    generated_code = template.render(test_config)

    # Check required sections
    required_sections = ["imports", "class_definition", "methods", "error_handling", "documentation"]
    for section in required_sections:
        assert section in generated_code, f"Missing section: {section}"

    # Check code validity
    assert validate_python_syntax(generated_code)
    assert check_type_annotations(generated_code)
```

#### 2. Template Customization Tests
**Test template customization functionality:**

```python
def test_template_customization():
    """Test template customization works correctly"""
    base_template = get_template("component_README_template.md")

    # Test customization for different component types
    customization_configs = [
        {"component_type": "core", "complexity": "high", "documentation": "comprehensive"},
        {"component_type": "utility", "complexity": "low", "documentation": "minimal"},
        {"component_type": "integration", "complexity": "medium", "documentation": "standard"}
    ]

    for config in customization_configs:
        customized_template = customize_template(base_template, config)
        generated_content = customized_template.render(test_data)

        # Validate customization applied
        assert customization_applied_correctly(generated_content, config)

def test_template_validation():
    """Test template validation catches issues"""
    # Test invalid template data
    invalid_data = {"incomplete": "data"}

    with pytest.raises(TemplateValidationError) as exc_info:
        generate_from_template("component_README_template.md", invalid_data)

    assert "validation" in str(exc_info.value).lower()

    # Test valid template data
    valid_data = create_complete_test_data()
    generated_content = generate_from_template("component_README_template.md", valid_data)
    assert validate_generated_content(generated_content)
```

#### 3. Integration Testing
**Test template integration with development workflows:**

```python
def test_template_workflow_integration():
    """Test template integration with development workflows"""

    # Test complete component creation workflow
    component_config = {
        "name": "TestComponent",
        "type": "analysis_tool",
        "features": ["data_processing", "statistical_analysis", "visualization"]
    }

    # Generate complete component structure
    generated_files = generate_component_structure(component_config)

    expected_files = [
        "README.md", "AGENTS.md", "src/component.py", "tests/test_component.py",
        "tests/test_integration.py", "pyproject.toml", "Makefile"
    ]

    for expected_file in expected_files:
        assert expected_file in generated_files, f"Missing file: {expected_file}"

        # Validate generated content
        content = generated_files[expected_file]
        validation = validate_generated_file(expected_file, content)
        assert validation["valid"], f"Invalid generated file: {expected_file}"

def test_cross_template_consistency():
    """Test consistency across related templates"""

    # Generate related templates
    component_config = create_test_component_config()

    readme_content = generate_readme_template(component_config)
    agents_content = generate_agents_template(component_config)
    code_content = generate_code_template(component_config)
    test_content = generate_test_template(component_config)

    # Check cross-references
    assert cross_references_consistent(readme_content, agents_content)
    assert code_references_consistent(code_content, test_content)
    assert documentation_references_consistent(readme_content, code_content)
```

### Template Coverage Requirements

- **Template Coverage**: All standard component types have templates
- **Pattern Coverage**: All established patterns have templates
- **Validation Coverage**: All templates include comprehensive validation
- **Example Coverage**: All templates include usage examples
- **Integration Coverage**: Templates integrate with development workflows

### Template Testing Commands

```bash
# Validate all templates
make validate-templates

# Test template generation
pytest tools/templates/tests/test_generation.py -v

# Test template customization
pytest tools/templates/tests/test_customization.py -v

# Test template integration
pytest tools/templates/tests/test_integration.py -v

# Check template completeness
python tools/templates/validate_completeness.py
```

## 📖 Template Documentation Standards

### Template Documentation Requirements (MANDATORY)

#### 1. Template Usage Documentation
**All templates must have comprehensive usage documentation:**

```python
def document_template_usage(template: Template, usage_config: Dict[str, Any]) -> str:
    """Document template usage with examples and guidelines"""

    usage_documentation = {
        "template_overview": create_template_overview(template, usage_config),
        "prerequisites": document_prerequisites(usage_config),
        "basic_usage": document_basic_usage(template, usage_config),
        "advanced_usage": document_advanced_usage(template, usage_config),
        "customization": document_customization_options(template, usage_config),
        "examples": create_usage_examples(template, usage_config),
        "validation": document_validation_requirements(template, usage_config),
        "troubleshooting": document_common_issues(template, usage_config)
    }

    return format_template_usage_documentation(usage_documentation)

def create_usage_examples(template: Template, config: Dict[str, Any]) -> List[str]:
    """Create comprehensive usage examples for template"""

    examples = []

    # Basic usage example
    basic_example = f"""
# Basic template usage
from tools.templates import {template.name}

# Create template instance
template = {template.name}(config)

# Generate content
generated_content = template.generate(data)
"""
    examples.append(basic_example)

    # Advanced usage example
    advanced_example = f"""
# Advanced template usage with customization
from tools.templates import TemplateManager

# Initialize template manager
manager = TemplateManager()

# Customize template
custom_template = manager.customize_template('{template.name}', customization_config)

# Generate with custom data
result = custom_template.generate(advanced_data)
"""
    examples.append(advanced_example)

    return examples
```

#### 2. Template Validation Standards
**Templates must include comprehensive validation:**

```python
def document_template_validation(template: Template, validation_config: Dict[str, Any]) -> str:
    """Document template validation requirements and procedures"""

    validation_documentation = {
        "input_validation": document_input_validation_requirements(template, validation_config),
        "output_validation": document_output_validation_requirements(template, validation_config),
        "quality_gates": document_quality_gate_requirements(template, validation_config),
        "integration_validation": document_integration_validation_requirements(template, validation_config),
        "performance_validation": document_performance_validation_requirements(template, validation_config)
    }

    return format_validation_documentation(validation_documentation)
```

#### 3. Template Integration Standards
**Templates must integrate properly with development workflows:**

```python
def document_template_integration(template: Template, integration_config: Dict[str, Any]) -> str:
    """Document template integration with development tools and workflows"""

    integration_documentation = {
        "development_workflow": document_workflow_integration(template, integration_config),
        "tool_integration": document_tool_integration(template, integration_config),
        "quality_integration": document_quality_integration(template, integration_config),
        "testing_integration": document_testing_integration(template, integration_config),
        "deployment_integration": document_deployment_integration(template, integration_config)
    }

    return format_integration_documentation(integration_documentation)
```

## 🚀 Performance Optimization

### Template Performance Requirements

**Template system must meet these performance standards:**

- **Generation Speed**: Templates generate content in <1 second
- **Memory Efficiency**: Template generation uses minimal memory
- **Validation Speed**: Template validation completes quickly
- **Integration Speed**: Template integration with workflows is fast
- **Scalability**: System handles multiple template generations efficiently

### Optimization Techniques

#### 1. Template Caching Strategy

```python
def implement_template_caching(template_manager: TemplateManager) -> None:
    """Implement caching for template performance"""

    # Cache compiled templates
    template_cache = {}

    def get_cached_template(template_name: str) -> Template:
        """Get template from cache or load and cache"""
        if template_name not in template_cache:
            template = load_template(template_name)
            template_cache[template_name] = compile_template(template)
        return template_cache[template_name]

    # Cache generated content patterns
    content_cache = {}

    def get_cached_content(template_name: str, data_hash: str) -> Optional[str]:
        """Get cached generated content if available"""
        cache_key = f"{template_name}:{data_hash}"
        return content_cache.get(cache_key)

    def cache_content(template_name: str, data_hash: str, content: str) -> None:
        """Cache generated content"""
        cache_key = f"{template_name}:{data_hash}"
        content_cache[cache_key] = content

        # Manage cache size
        if len(content_cache) > max_cache_size:
            oldest_key = min(content_cache.keys(), key=lambda k: content_cache[k].timestamp)
            del content_cache[oldest_key]
```

#### 2. Template Compilation Optimization

```python
def optimize_template_compilation(templates: List[Template]) -> List[CompiledTemplate]:
    """Optimize template compilation for performance"""

    compiled_templates = []

    for template in templates:
        # Pre-compile templates
        compiled = compile_template_optimized(template)

        # Optimize template structure
        optimized = optimize_template_structure(compiled)

        # Add performance metadata
        optimized.metadata = calculate_performance_characteristics(optimized)

        compiled_templates.append(optimized)

    return compiled_templates

def compile_template_optimized(template: Template) -> CompiledTemplate:
    """Compile template with performance optimizations"""

    # Use optimized compilation
    compiled = template.compile(optimized=True)

    # Add performance monitoring
    compiled = add_performance_monitoring(compiled)

    # Validate compilation
    compilation_validation = validate_template_compilation(compiled)
    if not compilation_validation["valid"]:
        raise TemplateCompilationError(f"Template compilation failed: {compilation_validation['errors']}")

    return compiled
```

## 🔒 Template Security Standards

### Template Security Requirements (MANDATORY)

#### 1. Template Input Validation

```python
def validate_template_inputs(template_data: Dict[str, Any], validation_rules: Dict[str, Any]) -> ValidationResult:
    """Validate template inputs for security and correctness"""

    security_checks = {
        "input_sanitization": validate_input_sanitization(template_data),
        "path_safety": validate_path_safety(template_data),
        "injection_prevention": validate_injection_prevention(template_data),
        "type_safety": validate_type_safety(template_data)
    }

    return {
        "secure": all(security_checks.values()),
        "checks": security_checks,
        "violations": [k for k, v in security_checks.items() if not v]
    }

def sanitize_template_input(raw_input: Any) -> Any:
    """Sanitize template input to prevent security issues"""

    if isinstance(raw_input, str):
        # Remove potentially dangerous content
        sanitized = remove_dangerous_strings(raw_input)
        sanitized = escape_special_characters(sanitized)
        return sanitized

    elif isinstance(raw_input, dict):
        # Recursively sanitize dictionary
        return {k: sanitize_template_input(v) for k, v in raw_input.items()}

    elif isinstance(raw_input, list):
        # Sanitize list elements
        return [sanitize_template_input(item) for item in raw_input]

    else:
        return raw_input
```

#### 2. Generated Content Security

```python
def validate_generated_content_security(content: str, security_rules: Dict[str, Any]) -> SecurityResult:
    """Validate security of generated content"""

    security_validation = {
        "code_injection": check_code_injection_risks(content),
        "path_traversal": check_path_traversal_risks(content),
        "information_disclosure": check_information_disclosure(content),
        "malicious_content": check_malicious_content(content)
    }

    return {
        "secure": all(security_validation.values()),
        "validation": security_validation,
        "risks": [k for k, v in security_validation.items() if not v]
    }

def secure_content_generation(template: Template, data: Dict[str, Any]) -> str:
    """Generate content securely with validation"""

    # Sanitize input data
    secure_data = sanitize_template_data(data)

    # Generate content
    content = template.render(secure_data)

    # Validate generated content
    security_result = validate_generated_content_security(content, security_rules)
    if not security_result["secure"]:
        raise SecurityError(f"Generated content has security risks: {security_result['risks']}")

    return content
```

## 🐛 Template Debugging & Troubleshooting

### Debug Configuration

```python
# Enable template debugging
debug_config = {
    "debug": True,
    "template_validation": True,
    "content_validation": True,
    "performance_monitoring": True,
    "error_reporting": True
}

# Debug template development
debug_template_workflow(debug_config)
```

### Common Debugging Patterns

#### 1. Template Generation Debugging

```python
def debug_template_generation(template_name: str, data: Dict[str, Any]) -> DebugResult:
    """Debug template generation issues"""

    # Load and validate template
    template = load_template(template_name)
    template_validation = validate_template_structure(template)

    if not template_validation["valid"]:
        return {"type": "template", "issues": template_validation["errors"]}

    # Test template rendering
    try:
        content = template.render(data)
        rendering_success = True
    except Exception as e:
        return {"type": "rendering", "error": str(e), "data": data}

    # Validate generated content
    content_validation = validate_generated_content(content)
    if not content_validation["valid"]:
        return {"type": "content", "issues": content_validation["errors"]}

    return {"status": "success", "content": content}

def validate_template_structure(template: Template) -> Dict[str, Any]:
    """Validate template structure and syntax"""

    validation_issues = []

    # Check template syntax
    syntax_check = validate_template_syntax(template)
    if not syntax_check["valid"]:
        validation_issues.extend(syntax_check["errors"])

    # Check required variables
    variables_check = validate_template_variables(template)
    if not variables_check["complete"]:
        validation_issues.extend(variables_check["missing"])

    # Check template logic
    logic_check = validate_template_logic(template)
    if not logic_check["sound"]:
        validation_issues.extend(logic_check["issues"])

    return {
        "valid": len(validation_issues) == 0,
        "errors": validation_issues,
        "score": calculate_template_quality_score(template)
    }
```

#### 2. Template Integration Debugging

```python
def debug_template_integration(template_name: str, workflow_config: Dict[str, Any]) -> DebugResult:
    """Debug template integration with development workflows"""

    # Test template in workflow context
    workflow_test = test_template_in_workflow(template_name, workflow_config)

    if not workflow_test["success"]:
        return {"type": "workflow", "error": workflow_test["error"]}

    # Test quality gate integration
    quality_test = test_quality_gate_integration(template_name, workflow_config)

    if not quality_test["passed"]:
        return {"type": "quality", "failures": quality_test["failures"]}

    # Test tool integration
    tool_test = test_tool_integration(template_name, workflow_config)

    if not tool_test["compatible"]:
        return {"type": "tools", "issues": tool_test["issues"]}

    return {"status": "integrated", "workflow": workflow_config}
```

## 🔄 Development Workflow

### Agent Development Process

1. **Template System Assessment**
   - Understand current template system state
   - Identify gaps in template coverage
   - Review existing template quality and patterns

2. **Template Architecture Planning**
   - Design comprehensive template system structure
   - Plan integration with development workflows
   - Consider scalability and maintenance requirements

3. **Template Implementation**
   - Implement robust template generation logic
   - Create comprehensive validation systems
   - Develop customization and configuration systems

4. **Quality Assurance Implementation**
   - Implement comprehensive template testing
   - Validate template integration with workflows
   - Ensure generated content meets quality standards

5. **Integration and Validation**
   - Test integration with development tools
   - Validate template system performance
   - Update related documentation and workflows

### Code Review Checklist

**Before submitting template code for review:**

- [ ] **Template Functionality**: Templates generate correct and complete content
- [ ] **Validation Systems**: Comprehensive validation for generated content
- [ ] **Error Handling**: Robust error handling for template generation
- [ ] **Documentation**: Complete documentation for template usage and customization
- [ ] **Integration**: Templates integrate properly with development workflows
- [ ] **Performance**: Template generation meets performance requirements
- [ ] **Security**: Generated content is secure and validated
- [ ] **Standards Compliance**: Templates follow all development standards

## 📚 Learning Resources

### Template Development Resources

- **[Template System AGENTS.md](AGENTS.md)**: Template development guidelines
- **[Development Tools AGENTS.md](../../tools/AGENTS.md)**: Development tools and utilities
- **[Jinja2 Documentation](https://example.com)**: Template engine documentation
- **[Python Template Patterns](https://example.com)**: Advanced templating patterns

### Technical References

- **[Code Generation Best Practices](https://example.com)**: Software generation methodologies
- **[Template Engine Design](https://example.com)**: Template system architecture
- **[Metaprogramming Patterns](https://example.com)**: Advanced code generation techniques
- **[Quality Assurance in Code Generation](https://example.com)**: Ensuring quality in generated code

### Related Components

Study these related components for integration patterns:

- **[Development Tools](../../tools/)**: Development tool integration patterns
- **[Documentation Tools](../../tools/documentation/)**: Documentation generation integration
- **[Quality Assurance](../../applications/best_practices/)**: Quality standards implementation
- **[Platform Architecture](../../../platform/)**: Platform-wide template integration

## 🎯 Success Metrics

### Template Quality Metrics

- **Generation Accuracy**: 100% of templates generate valid content
- **Coverage Completeness**: 100% of standard patterns have templates
- **Validation Success**: 100% of generated content passes validation
- **Integration Success**: 100% integration with development workflows
- **Performance Efficiency**: Template generation within performance targets

### Development Metrics

- **Template Speed**: New templates developed within 1 week
- **Quality Score**: Consistent high-quality template generation
- **Integration Success**: Seamless integration with development tools
- **User Adoption**: Templates widely used in development workflows
- **Maintenance Efficiency**: Easy to update and maintain templates

---

**Development Templates**: Agent Guide | **Version**: 1.0.0 | **Last Updated**: October 2024

*"Active Inference for, with, by Generative AI"* - Accelerating development through intelligent template systems, standardized patterns, and comprehensive development support.
