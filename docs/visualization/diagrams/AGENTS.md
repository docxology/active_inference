# Diagrams - Agent Development Guide

**Guidelines for AI agents working with Diagrams in the Active Inference Knowledge Environment.**

**"Active Inference for, with, by Generative AI"**

## 🤖 Agent Role & Responsibilities

**What agents should do when working with Diagrams:**

### Primary Responsibilities
- **Develop, test, and maintain component functionality**
- **Quality Assurance**: Build validation and quality control systems
- **Pattern Analysis**: Extract and formalize development patterns
- **Integration Systems**: Ensure seamless integration with platform workflows

### Development Focus Areas
1. **Component Development and Platform Integration**
2. **Quality Control**: Create systems for validating functionality
3. **Pattern Recognition**: Develop tools for analyzing development patterns
4. **Integration**: Ensure seamless integration with platform workflows

## 🏗️ Architecture & Integration

### Component Architecture

**Understanding how Diagrams fits into the platform:**

```
Platform Layer
├── Knowledge Layer (foundations/, mathematics/, implementations/)
├── Platform Layer ← Diagrams
└── Integration Layer (platform/, visualization/, tools/)
```

### Integration Points

**Diagrams integrates with multiple platform components:**

#### Upstream Components
- **Knowledge Repository**: Provides theoretical foundations
- **Development Standards**: Must follow documentation standards
- **Quality Requirements**: Integrate with validation systems

#### Downstream Components
- **Platform Services**: Leverages infrastructure for deployment
- **User Interfaces**: Provides functionality through UIs
- **Integration APIs**: Connects with external systems

## 💻 Development Patterns

### Required Implementation Patterns

**All visualization_diagrams development must follow these patterns:**

#### 1. Component Factory Pattern (PREFERRED)
```python
def create_diagrams(config: Dict[str, Any]) -> Diagrams:
    """Create Diagrams using factory pattern with validation"""

    # Validate configuration
    validate_diagrams_config(config)

    # Create component with validation
    component = Diagrams(config)

    # Validate functionality
    validate_component_functionality(component)

    return component
```

#### 2. Component Configuration Pattern (MANDATORY)
```python
from dataclasses import dataclass
from typing import Dict, Any, Optional

@dataclass
class DiagramsConfig:
    """Configuration for Diagrams"""

    # Required fields
    component_name: str
    config_field: str

    # Optional fields with defaults
    debug_mode: bool = False
    optimization_level: str = "standard"

    def validate(self) -> List[str]:
        """Validate configuration"""
        errors = []

        if not self.component_name:
            errors.append("component_name cannot be empty")

        return errors
```

## 🧪 Testing Standards

### Component Testing Categories (MANDATORY)

#### 1. Unit Tests
**Test individual functions and methods:**
```python
def test_diagrams_initialization():
    """Test Diagrams initialization"""
    config = DiagramsConfig(
        component_name="Diagrams",
        config_field="test_value"
    )

    component = create_diagrams(config.to_dict())

    # Validate initialization
    assert component.config == config.to_dict()
    assert component.initialized == True

def test_diagrams_functionality():
    """Test core Diagrams functionality"""
    config = DiagramsConfig(
        component_name="Diagrams",
        config_field="test_value"
    )

    component = create_diagrams(config.to_dict())

    # Test functionality
    result = component.process(test_input)
    assert result is not None
```

## 📖 Documentation Standards

### Documentation Requirements (MANDATORY)

#### 1. README.md Structure
**Every component must have comprehensive README.md:**
- Component overview and mission
- Architecture and integration points
- Usage examples and configuration
- API reference and testing information
- Development workflow and contribution guidelines

#### 2. AGENTS.md Structure
**Agent development guidelines must include:**
- Role and responsibilities for agents
- Architecture and integration patterns
- Development workflow and standards
- Testing and validation requirements
- Quality assurance and best practices

## 🔄 Development Workflow

### Agent Development Process
1. **Task Assessment**: Analyze component requirements
2. **Architecture Planning**: Design solutions following established patterns
3. **Test-Driven Development**: Write tests before implementation
4. **Implementation**: Follow coding standards and best practices
5. **Documentation**: Create comprehensive documentation
6. **Quality Assurance**: Ensure all tests pass and quality standards met
7. **Integration**: Integrate with existing platform components

### Quality Assurance Workflow
1. **Code Quality**: Test coverage >95%, type safety, documentation
2. **Integration Testing**: Component interaction validation
3. **Performance Validation**: Performance characteristics verified
4. **Documentation Review**: README.md and AGENTS.md completeness
5. **Standards Compliance**: Follow all established standards

## 🎯 Quality Standards

### Code Quality Gates
- **Test Coverage**: >95% for core components, >80% overall
- **Type Safety**: Complete type annotations for all interfaces
- **Documentation Coverage**: 100% for public APIs and interfaces
- **Code Style**: PEP 8 compliance with automated formatting
- **Error Handling**: Comprehensive error handling with informative messages

### Component Quality Gates
- **Functionality**: All specified features implemented and tested
- **Integration**: Seamless integration with platform components
- **Performance**: Meets performance requirements for target use cases
- **Reliability**: Robust operation under various conditions
- **Maintainability**: Clean, extensible code following established patterns

## 🔧 Integration Guidelines

### Platform Integration
- **Service Integration**: Connect with platform services as needed
- **Data Flow**: Ensure proper data flow and transformation
- **Error Handling**: Implement comprehensive error handling
- **Logging**: Add appropriate logging for monitoring and debugging
- **Configuration**: Support flexible configuration options

### Cross-Component Compatibility
- **API Compatibility**: Maintain compatible interfaces
- **Data Format Standards**: Follow established data format standards
- **Communication Protocols**: Use standard communication methods
- **Version Management**: Handle version compatibility appropriately

## 🐛 Troubleshooting and Support

### Common Development Issues
1. **Configuration Problems**: Validate configuration schema and values
2. **Integration Issues**: Check component dependencies and interfaces
3. **Performance Issues**: Profile and optimize bottlenecks
4. **Testing Failures**: Debug test cases and fix implementation

### Debug Mode
```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Or use debug configuration
debug_config = {"debug": True, "logging_level": "DEBUG"}
component = Diagrams(debug_config)
```

## 📚 Resources and References

### Core Documentation
- **[Main README](../../../README.md)**: Project overview and navigation
- **[AGENTS.md](../../../AGENTS.md)**: Master agent guidelines
- **[Development Standards](../../../.cursorrules)**: Complete development standards

### Component-Specific Resources
- **[API Documentation](../../api/README.md)**: Component API reference
- **[Integration Guide](../../integration/README.md)**: Integration patterns
- **[Testing Guide](../../testing/README.md)**: Testing standards and methods

### Related Components
- **[Related Component 1](../related1/README.md)**: Description of related functionality
- **[Related Component 2](../related2/README.md)**: Description of related functionality

---

**"Active Inference for, with, by Generative AI"** - Enhancing platform development through structured guidance, comprehensive documentation, and collaborative intelligence.

**Component**: Diagrams | **Version**: 1.0.0 | **Last Updated**: October 2024
