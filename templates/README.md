# Templates Collection

**Comprehensive template library for Active Inference implementations, documentation, and development workflows.**

## 📖 Overview

**Centralized template repository providing standardized patterns and structures for Active Inference development.**

This directory contains reusable templates for various aspects of Active Inference development including implementation templates, documentation templates, testing templates, configuration templates, and project structure templates.

### 🎯 Mission & Role

This templates collection contributes to development efficiency by:

- **Standardization**: Consistent patterns across all implementations
- **Accelerated Development**: Ready-to-use templates for common tasks
- **Quality Assurance**: Pre-validated templates meeting quality standards
- **Learning Support**: Educational templates for understanding patterns

## 🏗️ Architecture

### Template Categories

```
templates/
├── implementation/              # Code implementation templates
│   ├── python/                  # Python implementation templates
│   ├── configuration/           # Configuration file templates
│   └── integration/             # Integration and API templates
├── documentation/               # Documentation templates
│   ├── README_templates/        # README.md templates
│   ├── AGENTS_templates/        # AGENTS.md templates
│   └── api_docs/               # API documentation templates
├── testing/                     # Testing framework templates
│   ├── unit_tests/             # Unit test templates
│   ├── integration_tests/      # Integration test templates
│   └── fixtures/               # Test fixture templates
├── project_structure/           # Project structure templates
│   ├── module_templates/       # Module structure templates
│   └── application_templates/  # Application structure templates
└── generated/                   # Auto-generated templates
```

### Integration Points

**Templates integrate across the development ecosystem:**

- **Development Tools**: Used by code generation and scaffolding tools
- **Documentation System**: Templates for consistent documentation structure
- **Testing Framework**: Standardized testing patterns and structures
- **Platform Services**: Deployment and configuration templates

### Template Standards

#### Implementation Templates
Templates for code implementations following established patterns:

- **Factory Pattern Templates**: Standardized object creation patterns
- **Configuration Templates**: Structured configuration management
- **Error Handling Templates**: Comprehensive error handling patterns
- **Testing Templates**: Test structure and validation patterns

#### Documentation Templates
Templates ensuring consistent documentation:

- **README Templates**: Standardized project and component documentation
- **AGENTS Templates**: Agent development guidelines and patterns
- **API Templates**: API documentation and usage examples
- **Tutorial Templates**: Educational content and learning materials

## 🚀 Usage

### Template Selection and Usage

```python
# Load and use templates programmatically
from templates import TemplateManager, TemplateRenderer

# Initialize template system
template_manager = TemplateManager()
renderer = TemplateRenderer()

# Select appropriate template
template = template_manager.get_template(
    category="implementation",
    type="python",
    pattern="active_inference_application"
)

# Customize template variables
variables = {
    "component_name": "NeuralControlSystem",
    "domain": "neuroscience",
    "author": "Research Team",
    "version": "1.0.0"
}

# Generate implementation
implementation = renderer.render_template(template, variables)
```

### Command Line Template Usage

```bash
# Template management commands
ai-templates list --category implementation
ai-templates get --template python/active_inference_app
ai-templates customize --template README_component --variables vars.yaml

# Template generation
ai-templates generate --type project --name my_active_inference_project
ai-templates generate --type component --domain neuroscience --pattern neural_control

# Template validation
ai-templates validate --template generated/neural_control_system
ai-templates test --template generated/component_tests
```

## 🔧 Template Categories

### Implementation Templates

#### Python Implementation Templates
```python
# Active Inference Application Template
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import logging

class BaseActiveInferenceApplication(ABC):
    """Base template for Active Inference applications"""

    def __init__(self, config: Dict[str, Any]):
        """Initialize application with configuration validation"""
        self.config = self.validate_config(config)
        self.logger = self.setup_logging()
        self.generative_model = None
        self.policy_manager = None
        self.initialize_components()

    def validate_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and normalize configuration"""
        # Configuration validation logic
        required_fields = {'domain', 'application_type', 'model_parameters'}
        for field in required_fields:
            if field not in config:
                raise ValueError(f"Missing required configuration field: {field}")
        return config

    def setup_logging(self) -> logging.Logger:
        """Configure application logging"""
        logging.basicConfig(
            level=getattr(logging, self.config.get('log_level', 'INFO')),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(f"active_inference.{self.config.get('domain', 'application')}")

    def initialize_components(self) -> None:
        """Initialize core Active Inference components"""
        self.generative_model = self.create_generative_model()
        self.policy_manager = self.create_policy_manager()
        self.logger.info("Components initialized successfully")

    @abstractmethod
    def create_generative_model(self):
        """Create domain-specific generative model"""
        pass

    @abstractmethod
    def create_policy_manager(self):
        """Create domain-specific policy manager"""
        pass

    def execute(self, input_data: Any) -> Dict[str, Any]:
        """Execute Active Inference application"""
        try:
            # Generate predictions
            predictions = self.generative_model.predict(input_data)

            # Select optimal policy
            optimal_policy = self.policy_manager.select_policy(predictions, input_data)

            # Execute policy and return results
            results = self.execute_policy(optimal_policy, input_data)

            return {
                "status": "success",
                "predictions": predictions,
                "selected_policy": optimal_policy,
                "results": results,
                "metadata": self.get_execution_metadata()
            }

        except Exception as e:
            self.logger.error(f"Execution failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "metadata": self.get_execution_metadata()
            }

    @abstractmethod
    def execute_policy(self, policy: Any, input_data: Any) -> Any:
        """Execute selected policy on input data"""
        pass

    def get_execution_metadata(self) -> Dict[str, Any]:
        """Get execution metadata for reproducibility"""
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "config": self.config,
            "component_versions": self.get_component_versions()
        }
```

#### Configuration Templates
```python
# Configuration Template Structure
{
    "application": {
        "name": "NeuralControlSystem",
        "version": "1.0.0",
        "domain": "neuroscience",
        "description": "Neural control system using Active Inference"
    },

    "active_inference": {
        "precision": 16.0,
        "learning_rate": 0.01,
        "policy_horizon": 10,
        "model_complexity": "adaptive",
        "inference_method": "variational"
    },

    "domain_specific": {
        "neural_model": {
            "layers": 3,
            "units_per_layer": [64, 32, 16],
            "activation": "relu",
            "dropout": 0.2
        },
        "control_parameters": {
            "response_time": 50,  # milliseconds
            "precision_threshold": 0.9,
            "adaptation_rate": 0.1
        }
    },

    "integration": {
        "platform_url": "http://localhost:8000",
        "api_version": "v1",
        "authentication": {
            "type": "token",
            "token": "your_token_here"
        }
    },

    "development": {
        "debug": False,
        "profiling": True,
        "test_mode": False,
        "log_level": "INFO"
    }
}
```

### Documentation Templates

#### README Template Structure
```markdown
# [Component Name]

**[Brief, clear description of component purpose and role in Active Inference ecosystem.]**

## 📖 Overview

**[Detailed explanation of component purpose, scope, and functionality.]**

This component provides [specific functionality] for the Active Inference Knowledge Environment. It [key responsibilities and capabilities].

### 🎯 Mission & Role

This component contributes to the overall platform mission by:

- **Primary Function**: [Main purpose and functionality]
- **Integration**: [How it connects to other components]
- **User Value**: [What value it provides to users]

## 🏗️ Architecture

### Component Structure

```
component_name/
├── __init__.py              # Package initialization and public API
├── core.py                  # Core functionality and main classes
├── [feature].py             # Feature-specific implementations
├── [module].py              # Supporting modules and utilities
├── tests/
│   ├── test_core.py         # Core functionality tests
│   ├── test_[feature].py    # Feature-specific tests
│   └── test_integration.py  # Integration tests
├── README.md               # This documentation (REQUIRED)
└── AGENTS.md               # Agent development guidelines (REQUIRED)
```

### Integration Points

**[How this component integrates with the broader platform]:**

- **Upstream Dependencies**: Components this depends on
- **Downstream Components**: Components that depend on this
- **External Systems**: External tools or services used
- **Data Flow**: How data moves through this component

## 🚀 Usage

### Installation & Setup

```bash
# Component installation
pip install active-inference-[component]

# Or install from source
pip install -e .
```

### Basic Usage

```python
# Import and initialize
from active_inference.[module] import ComponentClass

# Basic configuration
config = {
    "setting1": "value1",
    "setting2": "value2"
}

# Create and use component
component = ComponentClass(config)
result = component.main_method()
```

### Advanced Configuration

```python
# Advanced configuration with all options
advanced_config = {
    "core_settings": {
        "parameter1": "value1",
        "parameter2": "value2"
    },
    "performance_settings": {
        "optimization_level": "high",
        "caching_enabled": True
    },
    "integration_settings": {
        "external_service_url": "https://example.com",
        "api_key": "your_api_key"
    }
}

component = ComponentClass(advanced_config)
```

## 📚 API Reference

### Core Classes

#### `ComponentClass`

**[Main component class description]**

```python
class ComponentClass:
    """Main component class with comprehensive functionality."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize component with configuration.

        Args:
            config: Configuration dictionary with component settings

        Raises:
            ValueError: If configuration is invalid
        """

    def main_method(self, input_data: Any) -> Any:
        """Primary method for core functionality.

        Args:
            input_data: Input data for processing

        Returns:
            Processed output data

        Raises:
            ProcessingError: If processing fails
        """
```

## 🧪 Testing

### Test Coverage

**[Test coverage information]**

### Running Tests

```bash
# Run component tests
make test-[component_name]

# Or run specific test files
pytest tests/test_core.py -v
pytest tests/test_integration.py -v

# Check coverage
pytest tests/ --cov=src/active_inference/[module]/ --cov-report=html
```

## 🤝 Contributing

### Development Guidelines

**[Contribution guidelines]**

### Contribution Process

1. **Create Feature Branch**: `git checkout -b feature/your-feature`
2. **Write Tests First**: Follow TDD with comprehensive coverage
3. **Implement Feature**: Follow established patterns
4. **Update Documentation**: README.md, AGENTS.md, and API docs
5. **Quality Assurance**: All tests pass, code formatted
6. **Submit PR**: Detailed description and testing instructions

## 📚 Resources

### Documentation
- **[Main README](../../../README.md)**: Project overview
- **[AGENTS.md](AGENTS.md)**: Agent development guidelines
- **[.cursorrules](../../../.cursorrules)**: Complete development standards

### Related Components
- **[Related Component 1](../related_component1/README.md)**: Description
- **[Related Component 2](../related_component2/README.md)**: Description

---
```

#### AGENTS Template Structure
```markdown
# [Component Name] - Agent Development Guide

**[Guidelines for AI agents working with this component in the Active Inference Knowledge Environment.]**

## 🤖 Agent Role & Responsibilities

**[What agents should do when working with this component]:**

### Primary Responsibilities
- **Core Development**: [Component-specific development tasks]
- **Integration**: [Integration responsibilities]
- **Testing**: [Testing responsibilities]
- **Documentation**: [Documentation responsibilities]

## 🏗️ Architecture & Integration

### Component Architecture

**[Component architecture description]**

### Integration Points

**[Integration points and dependencies]**

## 💻 Development Patterns

### Required Implementation Patterns

**[Required patterns for this component]**

## 🧪 Testing Standards

### Test Categories

**[Component-specific testing categories]**

## 📖 Documentation Standards

### Documentation Requirements

**[Component-specific documentation requirements]**

## 🔄 Development Workflow

### Development Process

**[Component-specific development process]**

## 📚 Learning Resources

### Component Resources

**[Component-specific learning resources]**

---

**[Component footer with version and status]**
```

## 🧪 Testing

### Template Testing Framework

```python
# Template validation and testing
def test_template_functionality():
    """Test template functionality and generation"""

    # Load template
    template = load_template("python/active_inference_application")

    # Test variable substitution
    variables = {
        "component_name": "TestComponent",
        "domain": "test_domain",
        "author": "Test Author"
    }

    # Generate from template
    generated_code = render_template(template, variables)

    # Validate generated code
    assert "class TestComponent" in generated_code
    assert "def __init__" in generated_code
    assert "# Test Author" in generated_code

    # Test syntax validity
    compile(generated_code, '<template>', 'exec')

def test_template_integration():
    """Test template integration with platform"""

    # Generate complete component from template
    component_template = get_template("component_structure")
    generated_files = render_component_template(component_template, {
        "component_name": "TestComponent",
        "include_tests": True,
        "include_docs": True
    })

    # Validate generated structure
    assert "README.md" in generated_files
    assert "AGENTS.md" in generated_files
    assert "tests/" in generated_files
    assert "__init__.py" in generated_files
```

## 🔄 Development Workflow

### Template Development Process

1. **Template Analysis**:
   ```bash
   # Analyze existing patterns
   ai-templates analyze --source src/active_inference/applications/

   # Identify common patterns
   ai-templates patterns --extract --output patterns.json
   ```

2. **Template Creation**:
   ```bash
   # Create new template
   ai-templates create --name neural_network_template --type implementation

   # Add template variables
   ai-templates variables --add --template neural_network_template --var domain
   ```

3. **Template Validation**:
   ```bash
   # Validate template syntax
   ai-templates validate --template neural_network_template

   # Test template generation
   ai-templates test --generate --template neural_network_template --output test_output/
   ```

4. **Template Integration**:
   ```bash
   # Integrate with template system
   ai-templates integrate --template neural_network_template --category implementation

   # Update template registry
   ai-templates registry --update
   ```

### Template Quality Assurance

```python
# Template quality validation
def validate_template_quality(template: Template) -> Dict[str, Any]:
    """Validate template quality and completeness"""

    quality_metrics = {
        "syntax_validity": validate_template_syntax(template),
        "variable_completeness": validate_template_variables(template),
        "pattern_consistency": validate_pattern_consistency(template),
        "documentation_completeness": validate_documentation_completeness(template),
        "test_coverage": validate_test_coverage(template)
    }

    # Overall quality score
    overall_score = sum(quality_metrics.values()) / len(quality_metrics)

    return {
        "metrics": quality_metrics,
        "overall_score": overall_score,
        "recommendations": generate_quality_recommendations(quality_metrics)
    }
```

## 🤝 Contributing

### Template Development Guidelines

When contributing templates:

1. **Pattern Analysis**: Study existing implementations for common patterns
2. **Template Design**: Design templates that are flexible and reusable
3. **Variable Definition**: Define clear, well-documented template variables
4. **Quality Validation**: Ensure templates generate high-quality code
5. **Documentation**: Provide comprehensive template documentation

### Template Review Process

1. **Pattern Validation**: Verify template follows established patterns
2. **Code Quality**: Ensure generated code meets quality standards
3. **Documentation**: Validate template documentation completeness
4. **Testing**: Test template generation and functionality
5. **Integration**: Verify template integrates with template system

## 📚 Resources

### Template Documentation
- **[Implementation Templates](implementation/README.md)**: Code templates
- **[Documentation Templates](documentation/README.md)**: Documentation templates
- **[Testing Templates](testing/README.md)**: Testing templates

### Template Tools
- **[Template Generator](../../tools/templates/README.md)**: Template generation tools
- **[Pattern Extractor](../../tools/documentation/README.md)**: Pattern extraction utilities

## 📄 License

This templates collection is part of the Active Inference Knowledge Environment and follows the same [MIT License](../../../LICENSE).

---

**Templates Version**: 1.0.0 | **Last Updated**: October 2024 | **Development Status**: Active Development

*"Active Inference for, with, by Generative AI"* - Accelerating development through comprehensive template libraries and standardized patterns.
