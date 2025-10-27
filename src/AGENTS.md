# Source Code Package - Agent Development Guide

**Guidelines for AI agents working with the source code package in the Active Inference Knowledge Environment.**

## 🤖 Agent Role & Responsibilities

**What agents should do when working with the source code package:**

### Primary Responsibilities
- **Package Development**: Implement and maintain the core Python package
- **Component Integration**: Ensure seamless integration between all platform components
- **API Design**: Create clean, intuitive APIs following established patterns
- **Testing**: Develop comprehensive test suites following TDD
- **Documentation**: Maintain complete documentation and examples

### Development Focus Areas
1. **Core Implementation**: Build robust, scalable core functionality
2. **Performance Optimization**: Optimize algorithms and data structures
3. **Error Handling**: Implement comprehensive error handling and recovery
4. **Security**: Ensure secure implementation and data protection
5. **Documentation**: Update documentation for all changes

## 🏗️ Architecture & Integration

### Package Architecture

**Understanding how the source package fits into the larger system:**

```
Repository Layer
├── Source Package ← This Package
├── Documentation Layer (docs/, README files)
├── Testing Layer (tests/, validation)
└── Infrastructure Layer (tools/, configuration)
```

### Integration Points

**Core integration hub connecting all platform components:**

#### Upstream Components
- **Knowledge Repository**: Provides educational content and learning paths
- **Research Framework**: Supplies scientific tools and methodologies
- **Application Templates**: Delivers implementation patterns and examples

#### Downstream Components
- **Platform Services**: Consumes core functionality for web services
- **CLI Interface**: Uses package APIs for command-line operations
- **Visualization Engine**: Integrates with visualization components

#### External Systems
- **File System**: Stores knowledge base and configuration files
- **Package Manager**: Distributes via PyPI and development installs
- **Development Tools**: Integrates with testing and documentation tools

### Data Flow Patterns

```python
# Typical data flow through the package
user_request → input_validation → component_routing → processing → output_formatting
configuration → validation → component_initialization → service_startup
knowledge_query → search → filtering → ranking → results_formatting
```

## 💻 Development Patterns

### Required Implementation Patterns

**All development must follow these patterns:**

#### 1. Factory Pattern (PREFERRED)
```python
def create_platform_component(component_type: str, config: Dict[str, Any]) -> Any:
    """Create platform component using factory pattern"""

    component_factories = {
        'knowledge_repository': create_knowledge_repository,
        'research_framework': create_research_framework,
        'visualization_engine': create_visualization_engine,
        'application_framework': create_application_framework
    }

    if component_type not in component_factories:
        raise ValueError(f"Unknown component type: {component_type}")

    return component_factories[component_type](config)
```

#### 2. Configuration Pattern (MANDATORY)
```python
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import json

@dataclass
class PlatformConfig:
    """Platform configuration with validation"""

    # Required fields
    knowledge_base_path: str
    platform_url: str = "http://localhost:8000"

    # Optional fields with defaults
    debug: bool = False
    performance_monitoring: bool = True
    max_concurrent_users: int = 100

    # Nested configuration
    cache: Dict[str, Any] = field(default_factory=lambda: {
        "enabled": True,
        "max_size": 1000,
        "ttl": 3600
    })

    def validate(self) -> List[str]:
        """Validate configuration and return error list"""
        errors = []

        if not self.knowledge_base_path:
            errors.append("knowledge_base_path cannot be empty")

        if not self.platform_url.startswith(('http://', 'https://')):
            errors.append("platform_url must be a valid HTTP/HTTPS URL")

        if self.max_concurrent_users < 1:
            errors.append("max_concurrent_users must be positive")

        return errors

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'knowledge_base_path': self.knowledge_base_path,
            'platform_url': self.platform_url,
            'debug': self.debug,
            'performance_monitoring': self.performance_monitoring,
            'max_concurrent_users': self.max_concurrent_users,
            'cache': self.cache
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PlatformConfig':
        """Create instance from dictionary"""
        return cls(**data)
```

#### 3. Error Handling Pattern (MANDATORY)
```python
import logging
from typing import Callable, Any, TypeVar, Dict

logger = logging.getLogger(__name__)
F = TypeVar('F', bound=Callable[..., Any])

def execute_with_error_handling(operation: str, func: Callable[..., Any], *args, **kwargs) -> Any:
    """Execute operation with comprehensive error handling"""

    try:
        logger.info(f"Starting operation: {operation}")
        result = func(*args, **kwargs)
        logger.info(f"Operation completed successfully: {operation}")
        return result

    except FileNotFoundError as e:
        logger.error(f"File not found in {operation}: {e}")
        raise PlatformError(f"Required file missing: {e}")

    except PermissionError as e:
        logger.error(f"Permission denied in {operation}: {e}")
        raise PlatformError(f"Access denied: {e}")

    except ValidationError as e:
        logger.error(f"Validation error in {operation}: {e}")
        raise

    except Exception as e:
        logger.error(f"Unexpected error in {operation}: {e}")
        logger.debug(f"Full traceback: {traceback.format_exc()}")
        raise PlatformError(f"Operation failed: {operation}")
```

## 🧪 Testing Standards

### Test Categories (MANDATORY)

#### 1. Unit Tests (`src/tests/test_core.py`)
**Test individual functions and methods:**
```python
def test_platform_initialization():
    """Test platform initializes correctly"""
    config = PlatformConfig(knowledge_base_path="/test/path")
    platform = ActiveInferencePlatform(config)

    assert platform.config == config
    assert platform.logger is not None
    assert platform.initialized is True

def test_component_factory():
    """Test component creation and validation"""
    config = PlatformConfig(knowledge_base_path="/test/path")

    # Test valid component creation
    component = create_platform_component('knowledge_repository', config.to_dict())
    assert component is not None

    # Test invalid component type
    with pytest.raises(ValueError, match="Unknown component type"):
        create_platform_component('invalid_component', config.to_dict())
```

#### 2. Integration Tests (`src/tests/test_integration.py`)
**Test component interactions:**
```python
def test_knowledge_research_integration():
    """Test integration between knowledge and research components"""
    config = PlatformConfig(knowledge_base_path="/test/knowledge")

    # Create integrated system
    platform = ActiveInferencePlatform(config)
    platform.initialize_components()

    # Test knowledge query
    knowledge_result = platform.knowledge.search("entropy")
    assert len(knowledge_result) > 0

    # Test research integration
    research_result = platform.research.analyze_concept("entropy")
    assert research_result is not None

def test_platform_services_integration():
    """Test platform services work together"""
    config = PlatformConfig(knowledge_base_path="/test/knowledge")

    # Start platform services
    platform = ActiveInferencePlatform(config)
    platform.start()

    # Test health check
    health = platform.health_check()
    assert health['status'] == 'healthy'

    # Test graceful shutdown
    platform.stop()
```

#### 3. Error Tests (`src/tests/test_errors.py`)
**Test error handling and edge cases:**
```python
def test_invalid_configuration():
    """Test behavior with invalid configuration"""
    invalid_config = {"invalid": "config"}

    with pytest.raises(ValidationError) as exc_info:
        PlatformConfig.from_dict(invalid_config)

    assert "configuration" in str(exc_info.value).lower()

def test_missing_dependencies():
    """Test behavior when dependencies are missing"""
    config = PlatformConfig(knowledge_base_path="/nonexistent/path")

    with pytest.raises(PlatformError):
        platform = ActiveInferencePlatform(config)
        platform.initialize_components()
```

### Test Coverage Requirements

- **Core Functionality**: 100% coverage
- **Error Paths**: 100% coverage
- **Integration Points**: 95% coverage
- **Performance Paths**: 80% coverage

### Running Tests

```bash
# Run all source package tests
make test-src

# Run specific test types
pytest src/tests/test_core.py -v
pytest src/tests/test_integration.py -v --tb=short

# Check coverage
pytest src/ --cov=src/active_inference/ --cov-report=html
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
def initialize_components(self, config: Dict[str, Any]) -> bool:
    """
    Initialize all platform components with comprehensive setup.

    This method orchestrates the complete initialization of all platform
    components including knowledge repository, research framework,
    visualization engine, and application templates.

    Args:
        config: Component configuration dictionary with initialization
               parameters for each component

    Returns:
        True if all components initialized successfully, False otherwise

    Raises:
        PlatformError: If component initialization fails
        ValidationError: If configuration is invalid
        DependencyError: If required dependencies are missing

    Examples:
        >>> config = {"knowledge_base": "/path/to/kb", "debug": True}
        >>> platform = ActiveInferencePlatform()
        >>> success = platform.initialize_components(config)
        >>> print(success)
        True
    """
    pass
```

## 🚀 Performance Optimization

### Performance Requirements

**Package must meet these performance standards:**

- **Import Time**: <1s for typical configurations
- **Memory Usage**: <100MB for full platform initialization
- **Response Time**: <100ms for API calls
- **Throughput**: 1000+ requests per second

### Optimization Techniques

#### 1. Lazy Loading Strategy
```python
class ComponentManager:
    """Manager with lazy loading for performance"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self._components: Dict[str, Any] = {}
        self._loaded: Set[str] = set()

    def get_component(self, name: str) -> Any:
        """Get component with lazy loading"""
        if name not in self._loaded:
            self._components[name] = self._load_component(name)
            self._loaded.add(name)

        return self._components[name]

    def _load_component(self, name: str) -> Any:
        """Load component implementation"""
        # Component loading logic here
        pass
```

#### 2. Connection Pooling
```python
import asyncio
from typing import Optional, Dict, Any

class ConnectionPool:
    """Efficient connection pooling for resources"""

    def __init__(self, max_connections: int = 10):
        self.max_connections = max_connections
        self._pool: asyncio.Queue = asyncio.Queue(maxsize=max_connections)
        self._active_connections: Dict[str, Any] = {}

    async def get_connection(self) -> Any:
        """Get connection from pool or create new one"""
        try:
            connection = self._pool.get_nowait()
        except asyncio.QueueEmpty:
            if len(self._active_connections) < self.max_connections:
                connection = await self._create_connection()
            else:
                connection = await self._pool.get()

        self._active_connections[id(connection)] = connection
        return connection
```

#### 3. Caching Strategy
```python
from functools import lru_cache
from typing import Dict, Any

class CachedKnowledgeRepository:
    """Knowledge repository with intelligent caching"""

    def __init__(self, base_repository: Any):
        self.base_repository = base_repository

    @lru_cache(maxsize=1000)
    def search_concepts(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Cached concept search with LRU cache"""
        return self.base_repository.search_concepts(query, limit)

    def clear_cache(self) -> None:
        """Clear search cache"""
        self.search_concepts.cache_clear()
```

## 🔒 Security Standards

### Security Requirements (MANDATORY)

#### 1. Input Validation
```python
def validate_component_config(self, config: Dict[str, Any]) -> None:
    """Validate component configuration thoroughly"""
    required_fields = {'knowledge_base_path', 'platform_url'}

    for field in required_fields:
        if field not in config:
            raise ValidationError(f"Missing required field: {field}")

        value = config[field]
        if not isinstance(value, str) or not value.strip():
            raise ValidationError(f"Invalid {field}: must be non-empty string")

    # Additional validation logic
    if not os.path.exists(config['knowledge_base_path']):
        raise ValidationError(f"Knowledge base path does not exist: {config['knowledge_base_path']}")
```

#### 2. Access Control
```python
from enum import Enum
from typing import Set

class UserRole(Enum):
    """User roles for access control"""
    ADMIN = "admin"
    RESEARCHER = "researcher"
    EDUCATOR = "educator"
    STUDENT = "student"
    GUEST = "guest"

class PermissionManager:
    """Manager for user permissions and access control"""

    PERMISSIONS = {
        UserRole.ADMIN: {"read", "write", "delete", "admin"},
        UserRole.RESEARCHER: {"read", "write", "research"},
        UserRole.EDUCATOR: {"read", "write", "teach"},
        UserRole.STUDENT: {"read", "learn"},
        UserRole.GUEST: {"read"}
    }

    def check_permission(self, user_role: UserRole, operation: str) -> bool:
        """Check if user has permission for operation"""
        allowed_operations = self.PERMISSIONS.get(user_role, set())

        if operation not in allowed_operations:
            logger.warning(f"Access denied: {user_role} cannot {operation}")
            return False

        return True
```

#### 3. Audit Logging
```python
import json
from datetime import datetime
from typing import Dict, Any

def log_security_event(self, event_type: str, details: Dict[str, Any], user_id: str = "unknown") -> None:
    """Log security-relevant events for audit trail"""

    security_event = {
        "timestamp": datetime.utcnow().isoformat(),
        "event_type": event_type,
        "user_id": user_id,
        "details": details,
        "component": "platform",
        "severity": self._determine_severity(event_type)
    }

    # Log to security log file
    self.logger.warning(f"Security event: {event_type}", extra={
        "security_event": True,
        "structured_data": json.dumps(security_event)
    })

    # Store in audit database if configured
    if hasattr(self, 'audit_db'):
        self.audit_db.store_event(security_event)

def _determine_severity(self, event_type: str) -> str:
    """Determine severity level for security event"""
    high_severity = {'unauthorized_access', 'data_breach', 'configuration_error'}
    medium_severity = {'failed_login', 'permission_denied', 'validation_error'}

    if event_type in high_severity:
        return "HIGH"
    elif event_type in medium_severity:
        return "MEDIUM"
    else:
        return "LOW"
```

## 🐛 Debugging & Troubleshooting

### Debug Configuration

```python
# Enable comprehensive debug mode
debug_config = {
    "debug": True,
    "logging_level": "DEBUG",
    "performance_monitoring": True,
    "detailed_errors": True,
    "component_tracing": True,
    "memory_profiling": True
}

platform = ActiveInferencePlatform(debug_config)
```

### Common Debugging Patterns

#### 1. Component Tracing
```python
def trace_component_initialization(self, component_name: str) -> None:
    """Trace component initialization for debugging"""
    self.logger.debug(f"Starting initialization: {component_name}")

    # Step-by-step initialization logging
    steps = [
        "validate_configuration",
        "allocate_resources",
        "initialize_dependencies",
        "setup_connections",
        "validate_functionality"
    ]

    for step in steps:
        self.logger.debug(f"Step: {step}")
        # Execute step with error handling
        try:
            getattr(self, step)()
        except Exception as e:
            self.logger.error(f"Step failed: {step} - {e}")
            raise

    self.logger.debug(f"Initialization completed: {component_name}")
```

#### 2. Performance Profiling
```python
import time
import functools
from typing import Callable, Any

def profile_method(func: Callable) -> Callable:
    """Decorator for performance profiling"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        memory_before = get_memory_usage()

        try:
            result = func(*args, **kwargs)
            return result
        finally:
            end_time = time.perf_counter()
            memory_after = get_memory_usage()

            logger.info(f"{func.__name__} performance:", extra={
                "performance": True,
                "execution_time": end_time - start_time,
                "memory_delta": memory_after - memory_before,
                "function": func.__name__
            })

    return wrapper

def get_memory_usage() -> float:
    """Get current memory usage in MB"""
    import psutil
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024  # MB
```

## 🔄 Development Workflow

### Agent Development Process

1. **Task Assessment**
   - Understand requirements and constraints
   - Review existing implementation patterns
   - Identify integration points and dependencies

2. **Architecture Planning**
   - Design solution following established patterns
   - Plan comprehensive testing strategy
   - Consider performance and security implications

3. **Test-Driven Development**
   - Write comprehensive tests before implementation
   - Implement minimal functionality to pass tests
   - Refactor and optimize incrementally

4. **Implementation**
   - Follow coding standards and established patterns
   - Add comprehensive error handling and validation
   - Implement security measures and access controls

5. **Quality Assurance**
   - Run complete test suite
   - Check code quality metrics and coverage
   - Validate performance requirements
   - Verify documentation completeness

6. **Integration**
   - Test integration with other platform components
   - Validate end-to-end functionality
   - Update documentation and examples

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

### Package-Specific Resources

- **[Main Package README](README.md)**: Package overview and usage
- **[.cursorrules](../../../.cursorrules)**: Complete development standards
- **[Platform AGENTS.md](../../../AGENTS.md)**: Platform-wide agent guidelines

### Technical References

- **[Python Packaging Guide](https://packaging.python.org)**: Python packaging best practices
- **[Testing Best Practices](https://pytest.org)**: Testing methodologies and patterns
- **[Performance Optimization](https://docs.python.org/3/library/profile.html)**: Python performance techniques
- **[Security Guidelines](https://owasp.org)**: Security best practices

### Related Components

Study these related components for integration patterns:

- **[Active Inference Package](active_inference/README.md)**: Main package implementation
- **[Knowledge Management](active_inference/knowledge/README.md)**: Knowledge system patterns
- **[Platform Services](active_inference/platform/README.md)**: Service architecture patterns

## 🎯 Success Metrics

### Quality Metrics

- **Test Coverage**: Maintain >95% coverage
- **Performance**: Meet response time and throughput requirements
- **Reliability**: Zero unexpected failures in production
- **Security**: No security vulnerabilities introduced
- **Maintainability**: Clean, well-documented, modular code

### Development Metrics

- **Implementation Speed**: Features delivered efficiently
- **Code Quality**: Consistent with project standards
- **Documentation Quality**: Clear, comprehensive documentation
- **Integration Success**: Seamless component integration
- **Review Feedback**: Positive and constructive code review outcomes

---

**Package**: Source Code | **Version**: 0.2.0 | **Last Updated**: October 2024

*"Active Inference for, with, by Generative AI"* - Contributing to the most comprehensive platform for understanding intelligence through collaborative development.
