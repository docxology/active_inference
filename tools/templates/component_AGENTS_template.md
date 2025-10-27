# Component Name - Agent Development Guide

**Guidelines for AI agents working with this component in the Active Inference Knowledge Environment.**

## 🤖 Agent Role & Responsibilities

**What agents should do when working with this component:**

### Primary Responsibilities
- **Core Development**: Implement and maintain [component functionality]
- **Integration**: Ensure seamless integration with other platform components
- **Testing**: Develop comprehensive test suites following TDD
- **Documentation**: Maintain complete documentation and examples
- **Quality Assurance**: Ensure all quality standards are met

### Development Focus Areas
1. **Feature Implementation**: Build new functionality following established patterns
2. **Performance Optimization**: Optimize algorithms and data structures
3. **Error Handling**: Implement robust error handling and recovery
4. **Security**: Ensure secure implementation and data protection
5. **Documentation**: Update documentation for all changes

## 🏗️ Architecture & Integration

### Component Architecture

**Understanding how this component fits into the larger system:**

```
Platform Layer
├── User Interface (CLI, Web, API)
├── Component Layer ← This Component
├── Integration Layer (Data flow, APIs)
└── Infrastructure Layer (Storage, Services)
```

### Integration Points

**Key integration points and dependencies:**

#### Upstream Components
- **Component A**: Provides [data/functionality]
- **Component B**: Handles [related functionality]

#### Downstream Components
- **Component C**: Consumes [output/results]
- **Component D**: Integrates with [functionality]

#### External Systems
- **Database**: Stores [type of data]
- **Services**: Integrates with [external services]
- **APIs**: Exposes [API endpoints]

### Data Flow Patterns

```python
# Typical data flow through component
input_data → validation → processing → output_data
exceptions → error_handling → user_feedback
```

## 💻 Development Patterns

### Required Implementation Patterns

**All development must follow these patterns:**

#### 1. Factory Pattern (PREFERRED)
```python
def create_component(config: Dict[str, Any]) -> ComponentClass:
    """Create component instance with configuration validation"""

    # Validate configuration
    validate_config_schema(config)

    # Create component
    component = ComponentClass(config)

    # Validate functionality
    validate_component_functionality(component)

    return component
```

#### 2. Configuration Pattern (MANDATORY)
```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class ComponentConfig:
    """Component configuration with validation"""

    required_field: str
    optional_field: str = "default"
    debug: bool = False

    def validate(self) -> List[str]:
        """Validate configuration and return error list"""
        errors = []

        if not self.required_field:
            errors.append("required_field cannot be empty")

        return errors
```

#### 3. Error Handling Pattern (MANDATORY)
```python
def execute_with_error_handling(operation: Callable, *args, **kwargs) -> Any:
    """Execute operation with comprehensive error handling"""

    try:
        logger.info(f"Starting operation: {operation.__name__}")
        result = operation(*args, **kwargs)
        logger.info(f"Operation completed: {operation.__name__}")
        return result

    except ValidationError as e:
        logger.error(f"Validation error: {e}")
        raise

    except ComponentError as e:
        logger.error(f"Component error: {e}")
        # Attempt recovery if possible
        raise

    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        logger.debug(f"Full traceback: {traceback.format_exc()}")
        raise
```

## 🧪 Testing Standards

### Test Categories (MANDATORY)

#### 1. Unit Tests (`tests/test_core.py`)
**Test individual functions and methods:**
```python
def test_component_initialization():
    """Test component initializes correctly"""
    config = create_test_config()
    component = ComponentClass(config)

    assert component.config == config
    assert component.logger is not None
    assert component.initialized is True

def test_method_functionality():
    """Test core method functionality"""
    component = ComponentClass(create_test_config())
    result = component.main_method(test_input)

    assert result is not None
    assert result.status == "success"
```

#### 2. Integration Tests (`tests/test_integration.py`)
**Test component interactions:**
```python
def test_upstream_integration():
    """Test integration with upstream components"""
    # Mock upstream component
    upstream_mock = MockUpstreamComponent()

    # Test data flow
    component = ComponentClass(create_test_config())
    component.set_upstream(upstream_mock)

    result = component.process_data()
    assert result.integrated_correctly

def test_downstream_integration():
    """Test integration with downstream components"""
    # Test with downstream consumer
    component = ComponentClass(create_test_config())
    downstream_mock = MockDownstreamComponent()

    component.connect_downstream(downstream_mock)
    component.process_data()

    assert downstream_mock.received_data
```

#### 3. Error Tests (`tests/test_errors.py`)
**Test error handling and edge cases:**
```python
def test_invalid_configuration():
    """Test behavior with invalid configuration"""
    invalid_config = {"invalid": "config"}

    with pytest.raises(ValueError) as exc_info:
        ComponentClass(invalid_config)

    assert "configuration" in str(exc_info.value).lower()

def test_missing_dependencies():
    """Test behavior when dependencies are missing"""
    config = create_test_config()
    config["missing_dependency"] = True

    component = ComponentClass(config)

    with pytest.raises(DependencyError):
        component.process_data()
```

### Test Coverage Requirements

- **Core Functionality**: 100% coverage
- **Error Paths**: 100% coverage
- **Integration Points**: 95% coverage
- **Performance Paths**: 80% coverage

### Running Tests

```bash
# Run all component tests
make test-component_name

# Run specific test types
pytest tests/test_core.py -v
pytest tests/test_integration.py -v --tb=short

# Check coverage
pytest tests/ --cov=src/active_inference/[module]/ --cov-report=html
```

## 📖 Documentation Standards

### Documentation Requirements (MANDATORY)

#### 1. README.md Updates
**Every change must update README.md:**
- New features and functionality
- Configuration changes
- API modifications
- Usage examples
- Troubleshooting information

#### 2. AGENTS.md Updates
**Agent guidelines must be current:**
- New implementation patterns
- Integration changes
- Development workflows
- Quality standards
- Common tasks and examples

#### 3. Docstring Standards
**All public methods must have comprehensive docstrings:**

```python
def public_method(self, parameter: str) -> Dict[str, Any]:
    """
    Public method with comprehensive documentation.

    Detailed description of what this method does, how it works,
    and when to use it.

    Args:
        parameter: Description of parameter and its format

    Returns:
        Description of return value and its structure

    Raises:
        SpecificError: Description of when this error occurs
        AnotherError: Description of other potential errors

    Examples:
        >>> component = ComponentClass(config)
        >>> result = component.public_method("input")
        >>> print(result["status"])
        "completed"
    """
    pass
```

## 🚀 Performance Optimization

### Performance Requirements

**Component must meet these performance standards:**

- **Response Time**: <100ms for typical operations
- **Throughput**: Handle expected concurrent load
- **Memory Usage**: Efficient memory utilization
- **Scalability**: Handle increased load gracefully

### Optimization Techniques

#### 1. Caching Strategy
```python
from functools import lru_cache

class ComponentWithCaching:
    """Component with intelligent caching"""

    @lru_cache(maxsize=1000)
    def expensive_operation(self, input_data: str) -> Any:
        """Cached expensive operation"""
        # Expensive computation here
        pass
```

#### 2. Streaming Processing
```python
def process_large_dataset(data_stream: Iterator[Any]) -> Iterator[Any]:
    """Process large datasets using streaming"""
    for chunk in data_stream:
        # Process chunk
        yield processed_chunk
```

#### 3. Async Processing
```python
async def async_operation(self, data: Any) -> Any:
    """Async operation for I/O bound tasks"""
    # Async implementation
    pass
```

## 🔒 Security Standards

### Security Requirements (MANDATORY)

#### 1. Input Validation
```python
def validate_input(self, data: Any) -> bool:
    """Validate input data thoroughly"""
    if not isinstance(data, expected_type):
        raise ValidationError(f"Invalid data type: {type(data)}")

    # Additional validation logic
    return True
```

#### 2. Access Control
```python
def check_permissions(self, user: User, operation: str) -> bool:
    """Check user permissions for operation"""
    required_permission = PERMISSIONS[operation]

    if not user.has_permission(required_permission):
        raise PermissionError(f"Access denied for operation: {operation}")

    return True
```

#### 3. Audit Logging
```python
def log_security_event(self, event: str, details: Dict[str, Any]) -> None:
    """Log security-relevant events"""
    self.logger.warning(f"Security event: {event}", extra={
        "security_event": True,
        "details": details,
        "timestamp": datetime.utcnow(),
        "user": getattr(self, 'current_user', 'unknown')
    })
```

## 🐛 Debugging & Troubleshooting

### Debug Configuration

```python
# Enable debug mode
debug_config = {
    "debug": True,
    "logging_level": "DEBUG",
    "performance_monitoring": True,
    "detailed_errors": True
}

component = ComponentClass(debug_config)
```

### Common Debugging Patterns

#### 1. Logging Strategy
```python
def debug_operation(self, input_data: Any) -> Any:
    """Debug-enabled operation with detailed logging"""
    self.logger.debug(f"Input data: {input_data}")

    # Processing steps with logging
    step1_result = self.step1(input_data)
    self.logger.debug(f"Step 1 result: {step1_result}")

    step2_result = self.step2(step1_result)
    self.logger.debug(f"Step 2 result: {step2_result}")

    return step2_result
```

#### 2. Performance Profiling
```python
import time
import functools

def profile_method(func):
    """Decorator for performance profiling"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()

        logger.info(f"{func.__name__} took {end_time - start_time:.4f}s")
        return result
    return wrapper
```

## 🔄 Development Workflow

### Agent Development Process

1. **Task Assessment**
   - Understand requirements and constraints
   - Review existing implementation
   - Identify integration points

2. **Architecture Planning**
   - Design solution following established patterns
   - Plan testing strategy
   - Consider performance implications

3. **Test-Driven Development**
   - Write comprehensive tests first
   - Implement minimal functionality
   - Refactor and optimize

4. **Implementation**
   - Follow coding standards
   - Add comprehensive error handling
   - Implement security measures

5. **Quality Assurance**
   - Run all tests
   - Check code quality metrics
   - Validate documentation

6. **Integration**
   - Test with other components
   - Validate performance requirements
   - Update documentation

### Code Review Checklist

**Before submitting code for review:**

- [ ] **Tests First**: All tests written before implementation
- [ ] **Test Coverage**: >95% coverage for new code
- [ ] **Type Safety**: Complete type annotations
- [ ] **Documentation**: Comprehensive docstrings and examples
- [ ] **Error Handling**: Robust error handling implemented
- [ ] **Security**: Security considerations addressed
- [ ] **Performance**: Performance requirements validated
- [ ] **Integration**: Component integration tested
- [ ] **Standards**: All project standards followed

## 📚 Learning Resources

### Component-Specific Resources

- **[Main README](../README.md)**: Component overview and usage
- **[.cursorrules](../../../.cursorrules)**: Complete development standards
- **[Platform AGENTS.md](../../../AGENTS.md)**: Platform-wide agent guidelines

### Technical References

- **[Python Best Practices](https://example.com)**: Python development standards
- **[Testing Best Practices](https://example.com)**: Testing methodologies
- **[Performance Optimization](https://example.com)**: Performance techniques
- **[Security Guidelines](https://example.com)**: Security best practices

### Related Components

Study these related components for integration patterns:

- **[Related Component 1](../component1/README.md)**: Integration example
- **[Related Component 2](../component2/README.md)**: Dependency pattern
- **[Platform Integration](../../platform/README.md)**: Platform patterns

## 🎯 Success Metrics

### Quality Metrics

- **Test Coverage**: Maintain >95% coverage
- **Performance**: Meet response time requirements
- **Reliability**: Zero unexpected failures
- **Security**: No security vulnerabilities
- **Maintainability**: Clean, documented code

### Development Metrics

- **Implementation Speed**: Features delivered on time
- **Code Quality**: Consistent with project standards
- **Documentation Quality**: Clear, comprehensive documentation
- **Integration Success**: Seamless component integration
- **Review Feedback**: Positive code review outcomes

---

**Component**: [Component Name] | **Version**: 1.0.0 | **Last Updated**: October 2024

*"Active Inference for, with, by Generative AI"* - Contributing to the most comprehensive platform for understanding intelligence through collaborative development.
