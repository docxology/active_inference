# Comprehensive Code Implementation and Development Prompt

**"Active Inference for, with, by Generative AI"**

## 🎯 Mission: Systematic Code Implementation Following Established Patterns

You are tasked with implementing platform components following the established architectural patterns, coding standards, and quality requirements of the Active Inference Knowledge Environment. This involves creating robust, well-tested, and properly integrated code that maintains the platform's high standards of quality and consistency.

## 📋 Component Implementation Requirements

### Core Implementation Standards (MANDATORY)
1. **Test-Driven Development (TDD)**: Write comprehensive tests before implementation
2. **Type Safety**: Complete type annotations for all interfaces and methods
3. **Documentation**: Comprehensive docstrings and inline comments
4. **Error Handling**: Robust error handling with informative messages
5. **Performance**: Optimized algorithms with resource awareness
6. **Integration**: Seamless integration with existing platform components

### Component Architecture Patterns

#### Service Pattern (PREFERRED for Platform Services)
```python
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import logging

class BasePlatformService(ABC):
    """Base pattern for platform services with comprehensive error handling and logging"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.service_name = self.__class__.__name__.lower()
        self.setup_logging()
        self.initialize_service()
        self.validate_configuration()

    def setup_logging(self) -> None:
        """Configure comprehensive logging for the service"""
        self.logger = logging.getLogger(f"active_inference.{self.service_name}")
        self.logger.setLevel(logging.INFO)

        # Add structured logging format
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

        # Add console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

    @abstractmethod
    def initialize_service(self) -> None:
        """Initialize service-specific components"""
        pass

    @abstractmethod
    def validate_configuration(self) -> None:
        """Validate service configuration parameters"""
        pass

    def health_check(self) -> Dict[str, Any]:
        """Comprehensive service health check"""
        return {
            "service": self.service_name,
            "status": "healthy",
            "timestamp": self.get_current_timestamp(),
            "version": getattr(self, 'version', 'unknown'),
            "config_valid": True
        }

    def get_current_timestamp(self) -> str:
        """Get current timestamp in ISO format"""
        from datetime import datetime
        return datetime.now().isoformat()
```

#### Repository Pattern (PREFERRED for Data Management)
```python
from typing import Dict, List, Optional, TypeVar, Generic, Any
from abc import ABC, abstractmethod

T = TypeVar('T')

class BaseRepository(Generic[T], ABC):
    """Generic repository pattern with type safety and error handling"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.items: Dict[str, T] = {}
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        self.initialize_repository()

    @abstractmethod
    def initialize_repository(self) -> None:
        """Initialize repository-specific setup"""
        pass

    @abstractmethod
    def create(self, item: T) -> str:
        """Create new item and return unique identifier"""
        pass

    @abstractmethod
    def get(self, item_id: str) -> Optional[T]:
        """Retrieve item by identifier"""
        pass

    @abstractmethod
    def update(self, item_id: str, item: T) -> bool:
        """Update existing item"""
        pass

    @abstractmethod
    def delete(self, item_id: str) -> bool:
        """Delete item by identifier"""
        pass

    def list_all(self) -> List[T]:
        """List all items in repository"""
        return list(self.items.values())

    def exists(self, item_id: str) -> bool:
        """Check if item exists"""
        return item_id in self.items

    def count(self) -> int:
        """Get total number of items"""
        return len(self.items)
```

#### Factory Pattern (PREFERRED for Component Creation)
```python
from typing import Dict, Any, Optional

def create_platform_component(component_type: str, config: Dict[str, Any]) -> Any:
    """Factory pattern for creating platform components with validation"""

    component_factories = {
        'knowledge_repository': create_knowledge_repository,
        'research_framework': create_research_framework,
        'visualization_engine': create_visualization_engine,
        'application_framework': create_application_framework,
        'platform_service': create_platform_service,
        'testing_framework': create_testing_framework
    }

    if component_type not in component_factories:
        available_types = list(component_factories.keys())
        raise ValueError(f"Unknown component type: {component_type}. "
                        f"Available types: {available_types}")

    # Validate configuration before creation
    validate_component_config(component_type, config)

    # Create component instance
    component = component_factories[component_type](config)

    # Post-creation validation
    validate_component_functionality(component)

    return component

def validate_component_config(component_type: str, config: Dict[str, Any]) -> None:
    """Validate component configuration parameters"""
    required_fields = {
        'knowledge_repository': ['storage_path', 'indexing_enabled'],
        'research_framework': ['experiment_directory', 'logging_level'],
        'visualization_engine': ['rendering_backend', 'output_format'],
        'application_framework': ['domain', 'deployment_target'],
        'platform_service': ['service_name', 'port', 'debug_mode'],
        'testing_framework': ['test_directory', 'coverage_threshold']
    }

    if component_type not in required_fields:
        return  # Skip validation for unknown types

    missing_fields = []
    for field in required_fields[component_type]:
        if field not in config:
            missing_fields.append(field)

    if missing_fields:
        raise ValueError(f"Missing required configuration fields for {component_type}: {missing_fields}")
```

## 🏗️ Implementation Workflows

### Phase 1: Requirements Analysis and Design

#### 1.1 Component Specification
1. **Define Component Purpose**: Clear statement of what the component does
2. **Identify Stakeholders**: Who will use and maintain the component
3. **Specify Interfaces**: Input/output contracts and API definitions
4. **Performance Requirements**: Speed, scalability, and resource constraints
5. **Integration Points**: How component connects to existing systems

#### 1.2 Architecture Design
1. **Choose Appropriate Patterns**: Select from established patterns above
2. **Design Component Structure**: Class hierarchy and module organization
3. **Define Data Models**: Type definitions and data structures
4. **Plan Error Handling**: Exception hierarchy and recovery strategies
5. **Design Testing Strategy**: Unit, integration, and performance tests

### Phase 2: Test-Driven Implementation

#### 2.1 Test-First Development
```python
import pytest
from typing import Dict, Any

class TestComponentImplementation:
    """Comprehensive test suite for component implementation"""

    @pytest.fixture
    def component_config(self) -> Dict[str, Any]:
        """Standard test configuration"""
        return {
            "service_name": "test_service",
            "debug_mode": True,
            "logging_level": "DEBUG",
            "max_connections": 100
        }

    def test_component_initialization(self, component_config):
        """Test component initializes correctly"""
        component = create_platform_component("platform_service", component_config)

        assert hasattr(component, 'service_name')
        assert hasattr(component, 'logger')
        assert hasattr(component, 'health_check')
        assert component.service_name == "test_service"

    def test_component_functionality(self, component_config):
        """Test core component functionality"""
        component = create_platform_component("platform_service", component_config)

        # Test health check
        health = component.health_check()
        assert health["status"] == "healthy"
        assert "timestamp" in health

    @pytest.mark.parametrize("invalid_config", [
        {},  # Empty config
        {"service_name": "test"},  # Missing required fields
        {"service_name": "test", "port": "invalid"},  # Invalid types
    ])
    def test_invalid_configuration_handling(self, invalid_config):
        """Test proper handling of invalid configurations"""
        with pytest.raises((ValueError, TypeError)):
            create_platform_component("platform_service", invalid_config)

    def test_error_recovery(self, component_config):
        """Test component error recovery mechanisms"""
        component = create_platform_component("platform_service", component_config)

        # Test error handling
        try:
            # Simulate error condition
            component.simulate_error()
        except Exception as e:
            # Verify error recovery
            recovery_result = component.recover_from_error()
            assert recovery_result["recovered"] == True
```

#### 2.2 Implementation Following Tests
1. **Implement Minimal Functionality**: Make tests pass with minimal code
2. **Refactor for Clarity**: Improve code structure while maintaining tests
3. **Add Comprehensive Documentation**: Complete docstrings and comments
4. **Optimize Performance**: Improve efficiency while maintaining correctness

### Phase 3: Integration and Validation

#### 3.1 Integration Testing
```python
import pytest
from unittest.mock import Mock, patch

class TestComponentIntegration:
    """Integration tests for component interaction"""

    def test_component_platform_integration(self):
        """Test integration with platform services"""
        with patch('platform.knowledge_graph.KnowledgeGraph') as mock_kg:
            mock_kg_instance = Mock()
            mock_kg.return_value = mock_kg_instance

            component = create_platform_component("knowledge_repository", {})
            component.integrate_with_platform()

            # Verify integration calls
            mock_kg_instance.register_component.assert_called_once()

    def test_component_data_flow(self):
        """Test data flow between components"""
        # Create mock upstream component
        upstream = Mock()
        upstream.get_data.return_value = {"test": "data"}

        # Create component under test
        component = create_platform_component("data_processor", {})

        # Test data processing pipeline
        result = component.process_data_from(upstream)
        assert result["processed"] == True
        assert result["data"]["test"] == "data"

    def test_component_error_propagation(self):
        """Test error handling across component boundaries"""
        failing_component = Mock()
        failing_component.process.side_effect = ValueError("Processing failed")

        component = create_platform_component("error_handler", {})

        with pytest.raises(ValueError, match="Processing failed"):
            component.handle_component_error(failing_component)
```

#### 3.2 Performance Validation
```python
import time
import pytest
from typing import List

class TestComponentPerformance:
    """Performance tests for component implementation"""

    def test_component_response_time(self):
        """Test component response time under normal load"""
        component = create_platform_component("performance_test", {})

        start_time = time.time()
        result = component.process_request({"data": "test"})
        end_time = time.time()

        response_time = end_time - start_time
        assert response_time < 0.1  # Less than 100ms
        assert result["success"] == True

    def test_component_scalability(self):
        """Test component performance under increasing load"""
        component = create_platform_component("scalability_test", {})

        load_levels = [10, 100, 1000]
        for load in load_levels:
            requests = [{"data": f"test_{i}"} for i in range(load)]

            start_time = time.time()
            results = component.process_batch_requests(requests)
            end_time = time.time()

            avg_response_time = (end_time - start_time) / load
            assert avg_response_time < 0.01  # Less than 10ms per request
            assert len(results) == load

    def test_component_memory_usage(self):
        """Test component memory usage patterns"""
        import psutil
        import os

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss

        component = create_platform_component("memory_test", {})

        # Process large dataset
        large_data = [{"data": f"item_{i}"} for i in range(10000)]
        component.process_large_dataset(large_data)

        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory

        # Memory increase should be reasonable (< 50MB)
        assert memory_increase < 50 * 1024 * 1024
```

## 📊 Quality Standards and Validation

### Code Quality Requirements

#### Type Safety (MANDATORY)
```python
from typing import Dict, List, Optional, Any, TypeVar, Generic
from abc import ABC, abstractmethod

T = TypeVar('T')

class ComponentInterface(Generic[T], ABC):
    """Fully typed component interface"""

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.logger: logging.Logger = self.setup_logging()

    def setup_logging(self) -> logging.Logger:
        """Setup logging with proper typing"""
        return logging.getLogger(self.__class__.__name__)

    @abstractmethod
    def process(self, data: T) -> Dict[str, Any]:
        """Process data with full type specification"""
        pass

    @abstractmethod
    def validate_input(self, data: T) -> bool:
        """Validate input data"""
        pass

    def get_metrics(self) -> Dict[str, float]:
        """Get component performance metrics"""
        return {
            "uptime": self.get_uptime(),
            "requests_processed": self.requests_processed,
            "error_rate": self.error_rate
        }
```

#### Documentation Standards (MANDATORY)
```python
def implement_component_feature(
    self,
    feature_config: Dict[str, Any],
    validation_rules: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Implement a new feature for the component following established patterns.

    This method handles the complete feature implementation lifecycle including
    validation, integration, testing, and documentation updates.

    Args:
        feature_config: Configuration dictionary containing feature specifications.
            Required keys: 'name', 'type', 'parameters'
            Optional keys: 'dependencies', 'validation_rules', 'documentation'
        validation_rules: Optional list of validation rules to apply.
            If None, default validation rules are used.

    Returns:
        Dictionary containing implementation results with the following keys:
        - 'success': Boolean indicating if implementation was successful
        - 'feature_id': Unique identifier for the implemented feature
        - 'validation_results': Results of validation checks
        - 'integration_status': Status of integration with existing components
        - 'documentation_updated': Boolean indicating if docs were updated

    Raises:
        ValueError: If feature_config is missing required keys
        ImplementationError: If feature implementation fails
        ValidationError: If validation checks fail
        IntegrationError: If integration with existing components fails

    Examples:
        >>> config = {
        ...     'name': 'user_authentication',
        ...     'type': 'security',
        ...     'parameters': {'method': 'oauth2'}
        ... }
        >>> result = component.implement_component_feature(config)
        >>> print(result['success'])
        True

    Note:
        This method follows the established implementation patterns and ensures
        that all new features are properly tested and documented before deployment.
    """
    pass
```

### Testing Coverage Requirements

#### Unit Test Standards
- **Coverage**: >95% for all new components
- **Test Types**: Unit, integration, performance, error handling
- **Test Organization**: Tests mirror code structure
- **Mock Usage**: Appropriate use of mocks for external dependencies

#### Integration Test Standards
- **Component Interaction**: Test all integration points
- **Data Flow**: Validate data flow between components
- **Error Propagation**: Test error handling across boundaries
- **Performance**: Integration tests include performance validation

## 🚀 Implementation Best Practices

### Error Handling Patterns
```python
from contextlib import contextmanager
from typing import Generator, Any

class ComponentError(Exception):
    """Base exception for component errors"""
    pass

class ConfigurationError(ComponentError):
    """Configuration-related errors"""
    pass

class ProcessingError(ComponentError):
    """Data processing errors"""
    pass

@contextmanager
def component_operation_context(
    component_name: str,
    operation: str
) -> Generator[None, None, None]:
    """Context manager for component operations with comprehensive error handling"""
    logger = logging.getLogger(f"component.{component_name}")

    try:
        logger.info(f"Starting {operation} operation")
        yield
        logger.info(f"Successfully completed {operation} operation")

    except ConfigurationError as e:
        logger.error(f"Configuration error in {operation}: {e}")
        raise

    except ProcessingError as e:
        logger.error(f"Processing error in {operation}: {e}")
        # Attempt recovery
        logger.info(f"Attempting recovery for {operation}")
        # Recovery logic here
        raise

    except Exception as e:
        logger.error(f"Unexpected error in {operation}: {e}")
        logger.debug(f"Full traceback: {traceback.format_exc()}")
        raise ComponentError(f"Operation {operation} failed") from e
```

### Configuration Management
```python
from dataclasses import dataclass
from typing import Optional, Dict, Any

@dataclass
class ComponentConfiguration:
    """Structured configuration class with validation"""

    # Core configuration
    component_name: str
    version: str = "1.0.0"
    debug_mode: bool = False

    # Service configuration
    host: str = "localhost"
    port: int = 8000
    max_connections: int = 100

    # Feature flags
    enable_caching: bool = True
    enable_metrics: bool = True
    enable_tracing: bool = False

    # Resource limits
    max_memory_mb: int = 512
    max_cpu_percent: float = 80.0
    timeout_seconds: int = 30

    def __post_init__(self):
        """Validate configuration after initialization"""
        self.validate()

    def validate(self) -> None:
        """Validate configuration parameters"""
        if not self.component_name:
            raise ValueError("component_name cannot be empty")

        if self.port < 1 or self.port > 65535:
            raise ValueError(f"Invalid port number: {self.port}")

        if self.max_connections < 1:
            raise ValueError("max_connections must be positive")

        if not (0 <= self.max_cpu_percent <= 100):
            raise ValueError(f"Invalid CPU percentage: {self.max_cpu_percent}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'component_name': self.component_name,
            'version': self.version,
            'debug_mode': self.debug_mode,
            'host': self.host,
            'port': self.port,
            'max_connections': self.max_connections,
            'enable_caching': self.enable_caching,
            'enable_metrics': self.enable_metrics,
            'enable_tracing': self.enable_tracing,
            'max_memory_mb': self.max_memory_mb,
            'max_cpu_percent': self.max_cpu_percent,
            'timeout_seconds': self.timeout_seconds
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ComponentConfiguration':
        """Create configuration from dictionary"""
        return cls(**data)
```

## 📈 Success Metrics and Validation

### Implementation Quality Metrics
- **Test Coverage**: >95% coverage for core components
- **Type Safety**: 100% type annotations on public interfaces
- **Documentation**: Complete docstrings for all public methods
- **Performance**: Meet or exceed performance requirements
- **Integration**: Successful integration with existing components
- **Maintainability**: Code follows established patterns and standards

### Validation Checkpoints
1. **Pre-Implementation**: Requirements analysis and design review
2. **During Implementation**: Continuous testing and code review
3. **Post-Implementation**: Integration testing and performance validation
4. **Deployment**: Final validation and monitoring setup
5. **Maintenance**: Ongoing quality monitoring and updates

## 🔄 Continuous Integration and Deployment

### CI/CD Integration
```python
def setup_component_ci_cd(component_name: str) -> Dict[str, Any]:
    """Setup CI/CD pipeline for component"""

    pipeline_config = {
        'component': component_name,
        'stages': [
            {
                'name': 'lint',
                'commands': ['flake8', 'mypy', 'black --check']
            },
            {
                'name': 'test',
                'commands': ['pytest --cov --cov-report=xml']
            },
            {
                'name': 'build',
                'commands': ['python setup.py build']
            },
            {
                'name': 'integration_test',
                'commands': ['pytest tests/integration/']
            },
            {
                'name': 'deploy',
                'commands': ['docker build', 'docker push']
            }
        ],
        'quality_gates': {
            'test_coverage': 95,
            'lint_score': 9.0,
            'performance_baseline': 'established'
        }
    }

    return pipeline_config
```

---

**"Active Inference for, with, by Generative AI"** - Building robust, well-tested platform components that advance our understanding of intelligence, cognition, and behavior through collaborative intelligence and rigorous implementation standards.
