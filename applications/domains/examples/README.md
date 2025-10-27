# Domain Examples

**Practical examples and implementations for Active Inference applications across various domains.**

## Overview

This directory contains working examples, code samples, and practical implementations of Active Inference in different application domains. These examples serve as starting points for developers and researchers implementing Active Inference solutions.

### Purpose

- **Learning Resource**: Hands-on examples for understanding Active Inference
- **Implementation Templates**: Ready-to-use code patterns
- **Domain Demonstrations**: Show Active Inference in specific contexts
- **Best Practices**: Exemplary implementations following project standards

## Directory Structure

```
examples/
├── artificial_intelligence/     # AI and machine learning examples
├── education/                   # Educational applications
├── engineering/                 # Engineering solutions
├── neuroscience/               # Neuroscientific applications
├── psychology/                 # Psychological models
├── robotics/                   # Robotic control examples
├── simple/                     # Basic getting-started examples
├── advanced/                   # Complex multi-domain examples
└── tutorials/                  # Step-by-step tutorial implementations
```

## Available Examples

### Getting Started Examples

#### Basic Active Inference
```bash
# Location: examples/simple/basic_active_inference.py
python examples/simple/basic_active_inference.py
```

#### Simple Decision Making
```bash
# Location: examples/simple/decision_making.py
python examples/simple/decision_making.py
```

#### Grid World Navigation
```bash
# Location: examples/simple/grid_world.py
python examples/simple/grid_world.py
```

### Domain-Specific Examples

#### Artificial Intelligence
```bash
# Neural network integration
python examples/artificial_intelligence/neural_active_inference.py

# Generative model examples
python examples/artificial_intelligence/generative_modeling.py
```

#### Neuroscience
```bash
# Perception modeling
python examples/neuroscience/perception_model.py

# Decision making networks
python examples/neuroscience/decision_networks.py
```

#### Robotics
```bash
# Robot control
python examples/robotics/robot_control.py

# Navigation systems
python examples/robotics/navigation.py
```

#### Psychology
```bash
# Cognitive models
python examples/psychology/cognitive_modeling.py

# Behavior analysis
python examples/psychology/behavior_analysis.py
```

## Implementation Patterns

### Standard Example Structure

Every example follows this consistent structure:

```python
"""
Example: [Example Name]
Description: [Brief description of what this example demonstrates]
Domain: [Primary domain]
Complexity: [beginner|intermediate|advanced]
"""

import numpy as np
from active_inference import ActiveInferenceAgent

# Configuration
config = {
    "generative_model": {...},
    "preferences": {...},
    "policies": [...]
}

# Initialize agent
agent = ActiveInferenceAgent(config)

# Run example
def main():
    """Main example execution"""
    # Setup
    # Execution
    # Results
    # Visualization

if __name__ == "__main__":
    main()
```

### Configuration Patterns

#### Generative Model Configuration
```python
generative_model = {
    "states": {
        "hidden": ["location", "goal", "context"],
        "observable": ["sensory_input", "reward"]
    },
    "transitions": {
        "A": np.array([...]),  # State transition matrices
        "B": np.array([...])   # Policy transition matrices
    },
    "preferences": {
        "C": np.array([...])   # Prior preferences
    }
}
```

#### Policy Configuration
```python
policies = [
    {"name": "explore", "actions": [0, 1, 2, 3]},
    {"name": "exploit", "actions": [4, 5, 6, 7]},
    {"name": "random", "actions": [8, 9, 10, 11]}
]
```

## Running Examples

### Prerequisites

```bash
# Install required dependencies
pip install -r examples/requirements.txt

# Or install full platform
make setup
```

### Execution

```bash
# Run all examples
python examples/run_all.py

# Run specific domain examples
python examples/run_domain.py --domain neuroscience

# Run with visualization
python examples/simple/grid_world.py --visualize

# Run with detailed logging
python examples/simple/basic_active_inference.py --debug
```

### Interactive Examples

```bash
# Start interactive mode
python examples/interactive/interactive_agent.py

# Jupyter notebook examples
jupyter notebook examples/notebooks/

# Web-based examples
streamlit run examples/web/dashboard.py
```

## Development Guidelines

### Creating New Examples

1. **Choose Domain**: Select appropriate domain directory
2. **Define Purpose**: Clear learning objectives and use cases
3. **Follow Structure**: Use standard example template
4. **Add Documentation**: Comprehensive comments and README
5. **Include Tests**: Unit tests for example functionality

### Example Template

```python
"""
Active Inference Example: [Name]

This example demonstrates [specific concept or application].

Learning Objectives:
- [Objective 1]
- [Objective 2]
- [Objective 3]

Usage:
    python [example_file].py [options]

Requirements:
- [Dependency 1]
- [Dependency 2]
"""

def create_example_config():
    """Create configuration for this example"""
    return {
        # Configuration here
    }

def setup_environment():
    """Set up example environment"""
    # Environment setup here
    pass

def run_example(config):
    """Run the main example"""
    # Example implementation here
    pass

def visualize_results(results):
    """Visualize example results"""
    # Visualization code here
    pass

if __name__ == "__main__":
    config = create_example_config()
    results = run_example(config)
    visualize_results(results)
```

## Testing

### Running Example Tests

```bash
# Test all examples
make test-examples

# Test specific example
pytest examples/simple/test_basic_active_inference.py

# Test domain examples
pytest examples/artificial_intelligence/tests/
```

### Example Testing

```python
import pytest
from examples.simple.basic_active_inference import run_example

def test_basic_active_inference():
    """Test basic active inference example"""
    config = create_test_config()
    results = run_example(config)

    assert results["success"] == True
    assert "agent" in results
    assert results["agent"].initialized == True

def test_example_configuration():
    """Test example configuration validity"""
    config = create_example_config()

    # Validate configuration
    assert "generative_model" in config
    assert "preferences" in config
    assert len(config["policies"]) > 0
```

## Documentation

### Example Documentation Requirements

Each example must include:

1. **README.md**: Purpose, usage, and learning objectives
2. **Code Comments**: Comprehensive inline documentation
3. **Usage Examples**: Command-line usage and options
4. **Configuration Guide**: Parameter explanations
5. **Expected Output**: What users should expect to see

### Documentation Template

```markdown
# [Example Name]

**Brief description of the example and its purpose.**

## Learning Objectives

- [Objective 1]: What users will learn
- [Objective 2]: Key concepts demonstrated
- [Objective 3]: Practical skills gained

## Prerequisites

- [Concept 1]: Required background knowledge
- [Tool 1]: Required software or libraries

## Usage

```bash
# Basic usage
python [example_file].py

# With options
python [example_file].py --visualize --debug

# Interactive mode
python [example_file].py --interactive
```

## Configuration

### Parameters

- `parameter1`: Description of parameter and its effect
- `parameter2`: Description of parameter and valid values

### Example Configuration

```python
config = {
    "parameter1": "value1",
    "parameter2": "value2"
}
```

## Expected Output

[Description of what the example produces]

## Extensions

[Suggestions for how to extend or modify the example]
```

## Contributing

See [AGENTS.md](AGENTS.md) for development guidelines and [.cursorrules](../../../.cursorrules) for comprehensive development standards.

### Adding New Examples

1. **Propose Example**: Discuss with community in issues or discussions
2. **Create Structure**: Follow example template and standards
3. **Implement**: Use TDD and comprehensive testing
4. **Document**: Complete documentation and examples
5. **Review**: Submit for peer review and validation

## Resources

### Learning Resources

- **[Active Inference Basics](../../../knowledge/foundations/active_inference_introduction.json)**: Theoretical foundation
- **[Implementation Guide](../../../knowledge/implementations/active_inference_basic.json)**: Implementation patterns
- **[Domain Applications](../../../knowledge/applications/)**: Domain-specific knowledge

### Development Resources

- **[Example Template](template.py)**: Starting template for new examples
- **[Testing Guide](../../tests/README.md)**: Testing best practices
- **[Documentation Standards](../../../.cursorrules)**: Documentation requirements

---

**Examples Version**: 1.0.0 | **Development Status**: Active Development

*"Active Inference for, with, by Generative AI"* - Practical understanding through hands-on examples.
