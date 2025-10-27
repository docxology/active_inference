# Documentation Tools

**Comprehensive toolkit for automated documentation generation, validation, and maintenance in the Active Inference Knowledge Environment.**

## 📖 Overview

**Automated documentation system supporting the entire Active Inference development and educational ecosystem.**

This directory contains specialized tools for documentation generation, validation, maintenance, and quality assurance. These tools automate the creation and maintenance of comprehensive documentation across all platform components, ensuring consistency, completeness, and quality.

### 🎯 Mission & Role

This documentation tools collection contributes to platform quality by:

- **Automated Generation**: Create documentation from source code and configurations
- **Quality Validation**: Ensure documentation meets established standards
- **Maintenance Automation**: Update documentation as code changes
- **Integration Support**: Seamless integration with development workflows

## 🏗️ Architecture

### Tool Categories

```
tools/documentation/
├── generator/           # Documentation generation engines
├── validator/           # Documentation validation and quality tools
├── analyzer/            # Documentation analysis and pattern extraction
├── maintainer/          # Documentation maintenance and update tools
└── templates/           # Documentation templates and patterns
```

### Integration Points

**Documentation tools integrate across the development ecosystem:**

- **Source Code**: Extract documentation from code comments and annotations
- **Build System**: Generate documentation during build processes
- **Version Control**: Track documentation changes and updates
- **Quality Gates**: Validate documentation in CI/CD pipelines

### Documentation Pipeline

#### Generation Pipeline
```python
# Automated documentation generation workflow
source_code → extraction → analysis → generation → validation → publication
```

#### Quality Pipeline
```python
# Documentation quality assurance workflow
content → validation → quality_analysis → improvement → compliance_check
```

## 🚀 Usage

### Basic Documentation Tools Usage

```python
# Import documentation tools
from tools.documentation.generator import DocumentationGenerator
from tools.documentation.validator import DocumentationValidator
from tools.documentation.analyzer import DocumentationAnalyzer

# Initialize documentation system
config = {
    "project_root": ".",
    "output_dir": "docs/",
    "validation_level": "comprehensive",
    "auto_update": True
}

# Generate comprehensive documentation
generator = DocumentationGenerator(config)
documentation = generator.generate_all_docs()

# Validate documentation quality
validator = DocumentationValidator(config)
validation_report = validator.validate_documentation(documentation)

# Analyze documentation patterns
analyzer = DocumentationAnalyzer(config)
patterns = analyzer.extract_documentation_patterns()
```

### Command Line Documentation Tools

```bash
# Documentation generation commands
ai-docs generate --all --output docs/ --format markdown
ai-docs generate --api --from-source src/ --validate
ai-docs generate --component knowledge --include-examples

# Documentation validation
ai-docs validate --all --check-completeness --standards
ai-docs validate --api-coverage --minimum 95
ai-docs validate --cross-references --fix-broken

# Documentation analysis
ai-docs analyze --patterns --output patterns.json
ai-docs analyze --coverage --report coverage.html
ai-docs analyze --quality --metrics quality.json

# Documentation maintenance
ai-docs maintain --update --auto --validate
ai-docs maintain --cross-references --fix
ai-docs maintain --quality --improve --threshold 0.9
```

## 🔧 Documentation Tool Categories

### Documentation Generators

#### API Documentation Generator
```python
from tools.documentation.generator.api_generator import APIDocumentationGenerator

class APIDocumentationGenerator:
    """Generate comprehensive API documentation from source code"""

    def __init__(self, config: Dict[str, Any]):
        """Initialize API documentation generator"""
        self.config = config
        self.source_analyzer = SourceCodeAnalyzer(config)
        self.template_engine = DocumentationTemplateEngine(config)

    def generate_api_docs(self, source_paths: List[str]) -> Dict[str, Any]:
        """Generate API documentation from source code"""

        # Analyze source code structure
        api_structure = self.source_analyzer.analyze_api_structure(source_paths)

        # Extract API elements
        api_elements = self.extract_api_elements(api_structure)

        # Generate documentation
        documentation = self.generate_documentation(api_elements)

        # Validate generated documentation
        validation = self.validate_api_documentation(documentation)

        return {
            "documentation": documentation,
            "api_elements": api_elements,
            "validation": validation,
            "coverage": self.calculate_api_coverage(api_structure, api_elements)
        }

    def extract_api_elements(self, api_structure: Dict[str, Any]) -> Dict[str, Any]:
        """Extract API elements from source code analysis"""

        api_elements = {
            "modules": self.extract_modules(api_structure),
            "classes": self.extract_classes(api_structure),
            "functions": self.extract_functions(api_structure),
            "methods": self.extract_methods(api_structure),
            "attributes": self.extract_attributes(api_structure)
        }

        # Validate completeness
        completeness = self.validate_api_completeness(api_elements)
        api_elements["completeness"] = completeness

        return api_elements

    def generate_documentation(self, api_elements: Dict[str, Any]) -> str:
        """Generate markdown documentation from API elements"""

        # Load appropriate templates
        templates = self.load_api_templates()

        # Render documentation sections
        sections = {
            "overview": self.render_overview_section(api_elements),
            "modules": self.render_modules_section(api_elements["modules"]),
            "classes": self.render_classes_section(api_elements["classes"]),
            "functions": self.render_functions_section(api_elements["functions"])
        }

        # Combine into complete documentation
        documentation = self.combine_documentation_sections(sections)

        return documentation
```

#### Component Documentation Generator
```python
from tools.documentation.generator.component_generator import ComponentDocumentationGenerator

class ComponentDocumentationGenerator:
    """Generate component-specific documentation"""

    def __init__(self, config: Dict[str, Any]):
        """Initialize component documentation generator"""
        self.config = config
        self.component_analyzer = ComponentAnalyzer(config)
        self.content_generator = ContentGenerator(config)

    def generate_component_docs(self, component_path: str) -> Dict[str, Any]:
        """Generate documentation for specific component"""

        # Analyze component structure
        component_info = self.component_analyzer.analyze_component(component_path)

        # Generate README.md
        readme_content = self.generate_component_readme(component_info)

        # Generate AGENTS.md
        agents_content = self.generate_component_agents(component_info)

        # Validate documentation
        validation = self.validate_component_documentation(component_info, readme_content, agents_content)

        return {
            "component_info": component_info,
            "readme": readme_content,
            "agents": agents_content,
            "validation": validation
        }

    def generate_component_readme(self, component_info: Dict[str, Any]) -> str:
        """Generate README.md for component"""

        # Load README template
        template = self.load_readme_template(component_info["type"])

        # Fill template with component data
        readme_data = self.prepare_readme_data(component_info)

        # Render README
        readme_content = self.template_engine.render(template, readme_data)

        # Validate README structure
        validation = self.validate_readme_structure(readme_content)

        return readme_content if validation["valid"] else self.fix_readme_structure(readme_content)

    def generate_component_agents(self, component_info: Dict[str, Any]) -> str:
        """Generate AGENTS.md for component"""

        # Load AGENTS template
        template = self.load_agents_template(component_info["type"])

        # Prepare agent-specific data
        agents_data = self.prepare_agents_data(component_info)

        # Render AGENTS documentation
        agents_content = self.template_engine.render(template, agents_data)

        # Validate AGENTS structure
        validation = self.validate_agents_structure(agents_content)

        return agents_content if validation["valid"] else self.fix_agents_structure(agents_content)
```

### Documentation Validators

#### Comprehensive Documentation Validator
```python
from tools.documentation.validator.comprehensive_validator import ComprehensiveDocumentationValidator

class ComprehensiveDocumentationValidator:
    """Comprehensive documentation validation and quality assurance"""

    def __init__(self, config: Dict[str, Any]):
        """Initialize comprehensive documentation validator"""
        self.config = config
        self.validation_rules = self.load_validation_rules()
        self.quality_metrics = self.load_quality_metrics()

    def validate_all_documentation(self) -> Dict[str, Any]:
        """Validate all project documentation comprehensively"""

        validation_report = {
            "overall_status": "valid",
            "components": {},
            "missing_docs": [],
            "quality_issues": [],
            "completeness_score": 0.0,
            "recommendations": []
        }

        # Discover all components
        components = self.discover_all_components()

        # Validate each component
        for component in components:
            component_validation = self.validate_component_documentation(component)
            validation_report["components"][component["name"]] = component_validation

            if component_validation["status"] == "missing":
                validation_report["missing_docs"].append(component["name"])
            elif component_validation["status"] == "incomplete":
                validation_report["quality_issues"].append(component["name"])

        # Calculate overall metrics
        validation_report["completeness_score"] = self.calculate_completeness_score(validation_report)

        # Update overall status
        if validation_report["missing_docs"] or validation_report["quality_issues"]:
            validation_report["overall_status"] = "incomplete"

        # Generate recommendations
        validation_report["recommendations"] = self.generate_improvement_recommendations(validation_report)

        return validation_report

    def validate_component_documentation(self, component: Dict[str, Any]) -> Dict[str, Any]:
        """Validate documentation for specific component"""

        validation_result = {
            "status": "valid",
            "readme": {"status": "valid", "issues": []},
            "agents": {"status": "valid", "issues": []},
            "api_docs": {"status": "valid", "issues": []},
            "overall_issues": []
        }

        # Validate README.md
        readme_validation = self.validate_readme_documentation(component)
        validation_result["readme"] = readme_validation

        # Validate AGENTS.md
        agents_validation = self.validate_agents_documentation(component)
        validation_result["agents"] = agents_validation

        # Validate API documentation
        api_validation = self.validate_api_documentation(component)
        validation_result["api_docs"] = api_validation

        # Combine validation results
        all_issues = (
            readme_validation["issues"] +
            agents_validation["issues"] +
            api_validation["issues"]
        )

        if all_issues:
            validation_result["status"] = "incomplete"
            validation_result["overall_issues"] = all_issues

        return validation_result
```

## 🧪 Testing

### Documentation Tool Testing

```python
# Documentation tools testing
def test_api_documentation_generation():
    """Test API documentation generation from source code"""
    config = {
        "source_paths": ["src/active_inference/"],
        "output_format": "markdown",
        "include_examples": True,
        "validation_level": "comprehensive"
    }

    generator = APIDocumentationGenerator(config)
    api_docs = generator.generate_api_docs(config["source_paths"])

    # Validate API documentation
    assert api_docs["status"] == "completed"
    assert api_docs["coverage"]["overall"] >= 0.95
    assert len(api_docs["documentation"]) > 1000  # Substantial documentation generated

    # Validate documentation structure
    structure_validation = generator.validate_documentation_structure(api_docs["documentation"])
    assert structure_validation["valid"] == True

def test_component_documentation_generation():
    """Test component documentation generation"""
    config = {
        "component_path": "knowledge/foundations/",
        "include_examples": True,
        "validation_level": "strict"
    }

    generator = ComponentDocumentationGenerator(config)
    component_docs = generator.generate_component_docs(config["component_path"])

    # Validate component documentation
    assert "readme" in component_docs
    assert "agents" in component_docs
    assert component_docs["validation"]["overall_status"] == "valid"

    # Validate README structure
    readme_validation = generator.validate_readme_structure(component_docs["readme"])
    assert readme_validation["valid"] == True

    # Validate AGENTS structure
    agents_validation = generator.validate_agents_structure(component_docs["agents"])
    assert agents_validation["valid"] == True

def test_documentation_validation():
    """Test comprehensive documentation validation"""
    config = {
        "validation_level": "comprehensive",
        "quality_threshold": 0.9,
        "auto_fix": True
    }

    validator = ComprehensiveDocumentationValidator(config)
    validation_report = validator.validate_all_documentation()

    # Validate comprehensive validation
    assert validation_report["overall_status"] in ["valid", "incomplete"]
    assert "completeness_score" in validation_report
    assert "components" in validation_report

    # Validate quality metrics
    if validation_report["quality_issues"]:
        assert validation_report["recommendations"]  # Should provide improvement recommendations
```

## 🔄 Development Workflow

### Documentation Tool Development Process

1. **Documentation Analysis**:
   ```bash
   # Analyze existing documentation patterns
   ai-docs analyze --patterns --output patterns.json

   # Identify documentation gaps
   ai-docs analyze --gaps --report gaps.html
   ```

2. **Tool Design and Implementation**:
   ```bash
   # Design documentation tool architecture
   ai-docs design --template documentation_tool

   # Implement following TDD
   ai-docs implement --test-first --template tool_implementation
   ```

3. **Tool Integration**:
   ```bash
   # Integrate with documentation ecosystem
   ai-docs integrate --tool documentation_generator

   # Validate integration
   ai-docs validate --integration --comprehensive
   ```

4. **Documentation Maintenance**:
   ```bash
   # Update documentation automatically
   ai-docs maintain --update-all --validate

   # Generate maintenance reports
   ai-docs maintain --report --output maintenance.html
   ```

### Documentation Quality Assurance

```python
# Documentation quality validation
def validate_documentation_quality(documentation: Dict[str, Any]) -> Dict[str, Any]:
    """Validate documentation quality and completeness"""

    quality_metrics = {
        "completeness": validate_documentation_completeness(documentation),
        "accuracy": validate_documentation_accuracy(documentation),
        "consistency": validate_documentation_consistency(documentation),
        "accessibility": validate_documentation_accessibility(documentation),
        "maintainability": validate_documentation_maintainability(documentation)
    }

    # Overall quality assessment
    overall_score = calculate_overall_documentation_quality(quality_metrics)

    return {
        "metrics": quality_metrics,
        "overall_score": overall_score,
        "compliant": overall_score >= QUALITY_THRESHOLD,
        "improvements": generate_documentation_improvements(quality_metrics)
    }
```

## 🤝 Contributing

### Documentation Tool Guidelines

When contributing documentation tools:

1. **Standards Compliance**: Ensure tools follow documentation standards
2. **Quality Integration**: Build tools that enhance documentation quality
3. **Automation Focus**: Maximize automation of documentation processes
4. **Integration Support**: Ensure seamless integration with development workflows
5. **Validation**: Include comprehensive validation and testing

### Documentation Tool Review Process

1. **Functionality Review**: Validate tool functionality and features
2. **Integration Review**: Verify integration with documentation ecosystem
3. **Quality Review**: Ensure tools maintain high documentation standards
4. **Performance Review**: Confirm performance meets documentation workflow requirements
5. **Usability Review**: Validate tool usability and developer experience

## 📚 Resources

### Documentation Tool References
- **[Documentation Tools](../../tools/README.md)**: Main development tools
- **[Template System](../templates/README.md)**: Documentation templates
- **[Quality Standards](../../../.cursorrules)**: Documentation quality requirements

### Technical References
- **[Documentation Generation](https://docs-generation.org)**: Documentation automation
- **[API Documentation](https://api-docs.org)**: API documentation best practices
- **[Markdown Standards](https://markdown-standards.org)**: Markdown documentation standards

## 📄 License

This documentation tools collection is part of the Active Inference Knowledge Environment and follows the same [MIT License](../../../LICENSE).

---

**Documentation Tools Version**: 1.0.0 | **Last Updated**: October 2024 | **Development Status**: Active Development

*"Active Inference for, with, by Generative AI"* - Enhancing documentation through automated generation and comprehensive quality assurance.
