# Source Code Package

**Main Python package for the Active Inference Knowledge Environment containing all core implementations.**

## 📖 Overview

**Core source code package containing all Active Inference implementations.**

This directory contains the main Python package `active_inference` with all platform components, including knowledge management, research tools, visualization systems, applications, and platform infrastructure.

### 🎯 Mission & Role

This package provides the core functionality for the Active Inference Knowledge Environment by:

- **Implementation Framework**: Complete Active Inference implementations
- **Platform Services**: Backend services and APIs
- **Integration Hub**: Unified interface for all platform components
- **Development Tools**: Utilities and development support

## 🏗️ Architecture

### Package Structure

```
src/
└── active_inference/          # Main Python package
    ├── __init__.py           # Package initialization and public API
    ├── cli.py                # Command-line interface entry point
    ├── applications/         # Application framework implementations
    ├── knowledge/            # Knowledge management and learning systems
    ├── llm/                  # Large Language Model integration
    ├── platform/             # Platform service implementations
    ├── research/             # Research tool and analysis implementations
    ├── tools/                # Development and utility implementations
    └── visualization/        # Visualization and UI implementations
```

### Integration Points

**Core integration hub connecting all platform components:**

- **Knowledge System**: Educational content and learning management
- **Research Framework**: Scientific tools and experimentation
- **Application Templates**: Practical implementation patterns
- **Platform Services**: Backend infrastructure and APIs
- **Visualization Engine**: Interactive exploration tools

### Design Principles

This package follows core design principles:

1. **Modularity**: Clean separation of concerns and reusable components
2. **Type Safety**: Complete type annotations and validation
3. **Error Resilience**: Comprehensive error handling and recovery
4. **Performance**: Optimized for research and educational use cases
5. **Maintainability**: Clear code structure and documentation

## 🚀 Usage

### Installation & Setup

```bash
# Package is included in main installation
pip install active-inference-knowledge

# Or install from source
pip install -e .
```

### Basic Usage

```python
# Import the main package
from active_inference import KnowledgeRepository, ResearchFramework
from active_inference.cli import main

# Initialize core components
config = {
    "knowledge_base": "path/to/knowledge",
    "debug": True,
    "performance_monitoring": True
}

# Create knowledge repository
knowledge = KnowledgeRepository(config)

# Run research experiments
research = ResearchFramework(config)
results = research.run_experiment("active_inference_basics")
```

### Command Line Interface

```bash
# Main CLI commands
python -m active_inference.cli --help
python -m active_inference.cli knowledge search "entropy"
python -m active_inference.cli research experiments list
python -m active_inference.cli platform serve
```

## 🔧 Configuration

### Required Configuration

**Minimum configuration needed for basic functionality:**

```python
minimal_config = {
    "knowledge_base_path": "path/to/knowledge",  # Required: Path to knowledge base
    "platform_url": "http://localhost:8000"       # Required: Platform URL
}
```

### Optional Configuration

**Extended configuration options:**

```python
full_config = {
    # Required fields
    "knowledge_base_path": "path/to/knowledge",
    "platform_url": "http://localhost:8000",

    # Core settings
    "debug": False,                    # Enable debug logging
    "performance_monitoring": True,   # Enable performance tracking

    # Advanced options
    "cache": {
        "enabled": True,              # Enable caching
        "max_size": 1000,            # Maximum cache size
        "ttl": 3600                  # Cache TTL in seconds
    },

    "logging": {
        "level": "INFO",              # Logging level
        "file": "logs/active_inference.log",  # Log file path
        "format": "detailed"         # Log format style
    },

    "security": {
        "enable_encryption": False,   # Enable data encryption
        "access_control": "standard", # Access control level
        "audit_logging": True        # Enable audit logging
    }
}
```

### Configuration Validation

The package validates all configuration and provides clear error messages:

```python
try:
    from active_inference import create_platform
    platform = create_platform(invalid_config)
except ValueError as e:
    print(f"Configuration error: {e}")
    # Configuration error: Missing required field: knowledge_base_path
```

## 📚 API Reference

### Core Classes

#### `ActiveInferencePlatform`

**Main platform class orchestrating all components.**

```python
class ActiveInferencePlatform:
    """Main platform class coordinating all Active Inference components."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize platform with comprehensive configuration.

        Args:
            config: Platform configuration dictionary

        Raises:
            PlatformError: If platform initialization fails
        """

    def start(self) -> bool:
        """Start all platform services.

        Returns:
            True if all services started successfully
        """

    def stop(self) -> None:
        """Stop all platform services gracefully."""

    def health_check(self) -> Dict[str, Any]:
        """Check platform health and component status.

        Returns:
            Health status for all components
        """
```

### Factory Functions

#### `create_knowledge_repository(config: Dict[str, Any]) -> KnowledgeRepository`

Create and configure a knowledge repository instance.

#### `create_research_framework(config: Dict[str, Any]) -> ResearchFramework`

Create and configure a research framework instance.

#### `create_visualization_engine(config: Dict[str, Any]) -> VisualizationEngine`

Create and configure a visualization engine instance.

## 🧪 Testing

### Test Coverage

This package maintains comprehensive test coverage:

- **Unit Tests**: >95% coverage of core functionality
- **Integration Tests**: Component interaction validation
- **Performance Tests**: Scalability and efficiency validation
- **Error Tests**: Error handling and edge case coverage

### Running Tests

```bash
# Run all package tests
make test-src

# Or run specific test files
pytest src/tests/test_core.py -v
pytest src/tests/test_integration.py -v

# Check coverage
pytest src/ --cov=src/active_inference/ --cov-report=html
```

### Test Structure

```
src/tests/
├── test_core.py              # Core functionality tests
├── test_integration.py       # Integration tests
├── test_performance.py       # Performance tests
├── test_errors.py           # Error handling tests
└── fixtures/
    ├── test_data.json       # Test data fixtures
    └── test_config.yaml     # Test configuration
```

## 🔄 Development Workflow

### For Contributors

1. **Set Up Environment**:
   ```bash
   cd src/
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   pip install -e ../
   ```

2. **Follow TDD**:
   ```bash
   # Write tests first
   pytest tests/test_new_feature.py::test_implementation

   # Implement feature
   # Run tests frequently
   make test
   ```

3. **Quality Assurance**:
   ```bash
   make lint          # Code style and type checking
   make format        # Code formatting
   make test          # Run all tests
   make docs          # Update documentation
   ```

4. **Documentation**:
   - Update this README.md with new features
   - Update AGENTS.md with development guidelines
   - Add comprehensive docstrings to all public APIs

### Development Standards

This package follows all standards defined in [.cursorrules](../../.cursorrules):

- **Test-Driven Development**: All features must have tests first
- **Type Safety**: Complete type annotations required
- **Documentation**: Comprehensive docstrings and examples
- **Error Handling**: Robust error handling with clear messages
- **Performance**: Optimized for target use cases

## 📊 Performance

### Performance Characteristics

- **Import Time**: <1s for typical configurations
- **Memory Usage**: ~50MB base + 10MB per active component
- **Response Time**: <100ms for API calls
- **Throughput**: 1000+ requests per second

### Optimization Features

- **Lazy Loading**: Components loaded on demand
- **Connection Pooling**: Efficient database and API connections
- **Caching**: Multi-level caching for frequently accessed data
- **Parallel Processing**: Concurrent execution where beneficial

## 🔒 Security

### Security Considerations

- **Input Validation**: All inputs validated and sanitized
- **Access Control**: Role-based permissions and authorization
- **Data Protection**: Sensitive data encrypted at rest and in transit
- **Audit Logging**: Comprehensive security event logging

## 🐛 Troubleshooting

### Common Issues

#### Issue 1: Import Error
**Error**: `ModuleNotFoundError: No module named 'active_inference'`

**Solution**:
```bash
# Install package in development mode
pip install -e .

# Or install from PyPI
pip install active-inference-knowledge
```

#### Issue 2: Configuration Error
**Error**: `ValueError: Invalid configuration`

**Solution**:
```bash
# Check configuration completeness
config = {
    "knowledge_base_path": "/path/to/knowledge",
    "platform_url": "http://localhost:8000"
    # Add all required fields from documentation
}
```

### Debug Mode

```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Or use debug configuration
debug_config = {
    "debug": True,
    "logging_level": "DEBUG",
    "performance_monitoring": True
}

platform = ActiveInferencePlatform(debug_config)
```

## 🤝 Contributing

### Development Guidelines

See [AGENTS.md](AGENTS.md) for detailed agent development guidelines and [.cursorrules](../../../.cursorrules) for comprehensive development standards.

### Contribution Process

1. **Create Feature Branch**: `git checkout -b feature/your-feature`
2. **Write Tests First**: Follow TDD with comprehensive coverage
3. **Implement Feature**: Follow established patterns
4. **Update Documentation**: README.md, AGENTS.md, and API docs
5. **Quality Assurance**: All tests pass, code formatted
6. **Submit PR**: Detailed description and testing instructions

### Code Review Checklist

- [ ] Tests written before implementation (TDD)
- [ ] Test coverage >95% for new code
- [ ] Type annotations complete
- [ ] Documentation updated
- [ ] Code formatted and linted
- [ ] Integration tests pass
- [ ] Performance requirements met

## 📚 Resources

### Documentation
- **[Main README](../../../README.md)**: Project overview
- **[AGENTS.md](AGENTS.md)**: Agent development guidelines
- **[.cursorrules](../../../.cursorrules)**: Complete development standards

### Related Components
- **[Knowledge Package](../active_inference/knowledge/README.md)**: Knowledge management
- **[Research Package](../active_inference/research/README.md)**: Research tools
- **[Platform Package](../active_inference/platform/README.md)**: Platform services

## 📄 License

This package is part of the Active Inference Knowledge Environment and follows the same [MIT License](../../../LICENSE).

---

**Package Version**: 0.2.0 | **Last Updated**: October 2024 | **Development Status**: Active Development

*"Active Inference for, with, by Generative AI"* - Contributing to the most comprehensive platform for understanding intelligence through collaborative development.
