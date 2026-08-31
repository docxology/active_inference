# Comparative

**Visualization framework and tools**

**"Active Inference for, with, by Generative AI"**

## 📖 Overview

**This component provides the core visualization framework and tools that power all visual representations in the Active Inference Knowledge Environment.**

This component provides visualization framework and interactive tools for the Active Inference Knowledge Environment.

### 🎯 Mission & Role

This component contributes to the overall platform mission by:

- **Primary Function**: Provide specialized platform functionality
- **Integration**: Integrates with related platform components
- **User Value**: Enhanced platform functionality and user experience

## 🏗️ Architecture

### Component Structure

```
src/active_inference/visualization/comparative/
├── [files based on component type]
├── README.md               # This documentation (REQUIRED)
└── AGENTS.md               # Agent development guidelines (REQUIRED)
```

### Integration Points

**How this component integrates with the broader platform:**

- **Upstream Dependencies**: Core platform services and related components
- **Downstream Components**: User interfaces and application systems
- **External Systems**: Standard development tools and libraries
- **Data Flow**: Input processing → Component logic → Output generation → Integration

## 🚀 Usage

### Basic Usage

```python
# Import the component (if applicable)
from active_inference.src.active_inference.visualization.comparative import Comparative

# Basic initialization
config = {
    "component_setting": "value"
}

component = Comparative(config)
result = component.process()
```

## 🔧 Configuration

### Required Configuration

**Minimum configuration needed for basic functionality:**

```python
minimal_config = {
    "required_field": "value"  # Component-specific configuration parameters
}
```

## 📚 API Reference

### Core Functions

#### `Comparative`

**Main component class for visualization framework and interactive tools.**

```python
class Comparative:
    """Main component class with comprehensive functionality."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize component with configuration."""
        pass

    def process(self, input_data: Any) -> Any:
        """Primary method for core functionality."""
        pass
```

## 🧪 Testing

### Test Coverage

This component maintains comprehensive test coverage:

- **Unit Tests**: >95% coverage of core functionality
- **Integration Tests**: Component interaction validation
- **Performance Tests**: Scalability and efficiency validation

### Running Tests

```bash
# Run component tests
make test-src-active_inference-visualization-comparative

# Or run specific test files
pytest tests/test_core.py -v
```

## 🔄 Development Workflow

### For Contributors

1. **Set Up Environment**:
   ```bash
   make setup
   cd src/active_inference/visualization/comparative
   ```

2. **Follow TDD**:
   ```bash
   # Write tests first
   pytest tests/test_core.py::test_new_feature

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

## 🤝 Contributing

### Development Guidelines

See [AGENTS.md](AGENTS.md) for detailed agent development guidelines and [../../../.cursorrules](../../../../.cursorrules) for comprehensive development standards.

### Contribution Process

1. **Create Feature Branch**: `git checkout -b feature/your-feature`
2. **Write Tests First**: Follow TDD with comprehensive coverage
3. **Implement Feature**: Follow established patterns
4. **Update Documentation**: README.md, AGENTS.md, and API docs
5. **Quality Assurance**: All tests pass, code formatted
6. **Submit PR**: Detailed description and testing instructions

---

**Component Version**: 1.0.0 | **Last Updated**: October 2024 | **Development Status**: Active Development

---

**"Active Inference for, with, by Generative AI"** - Together, we're building the most comprehensive platform for understanding intelligence, cognition, and behavior through collaborative intelligence and comprehensive knowledge integration.

**Built with**: ❤️ Human expertise, 🤖 AI assistance, 🧠 Collective intelligence, and the global Active Inference community's dedication to advancing understanding.
