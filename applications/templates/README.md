# Templates

This directory contains ready-to-use implementation templates, starter projects, and architectural patterns for common Active Inference applications. These templates provide a solid foundation for developers to build upon, ensuring consistency and accelerating development.

## Overview

The Templates module offers a comprehensive collection of implementation templates covering various Active Inference applications, system architectures, and integration patterns. Each template includes complete implementations, comprehensive documentation, and examples to help developers quickly get started with Active Inference development.

## Directory Structure

```
templates/
├── basic/                 # Basic Active Inference implementations
├── advanced/             # Advanced system architectures
├── domain_specific/      # Domain-specific applications
├── integrations/         # Integration templates and patterns
├── tools/               # Development and analysis tools
└── examples/            # Complete working examples
```

## Core Components

### 🏗️ Basic Templates
- **Minimal Agent**: Simple Active Inference agent implementation
- **Basic Simulation**: Basic Active Inference simulation framework
- **Learning Tutorial**: Step-by-step learning implementation
- **Quick Start**: Rapid prototyping template

### 🔧 Advanced Templates
- **Hierarchical Systems**: Multi-level Active Inference architectures
- **Multi-Agent Systems**: Multiple interacting Active Inference agents
- **Real-time Systems**: Real-time Active Inference implementations
- **Distributed Systems**: Distributed Active Inference architectures

### 🎯 Domain-Specific Templates
- **Neuroscience Models**: Neural system modeling templates
- **Cognitive Models**: Cognitive architecture implementations
- **Robotics Controllers**: Robot control system templates
- **Decision Support**: Decision-making system templates

### 🔗 Integration Templates
- **API Integration**: External API integration patterns
- **Database Integration**: Database connectivity templates
- **Visualization Integration**: Visualization system integration
- **Platform Integration**: Platform-specific integration templates

## Getting Started

### For Beginners
1. **Start Simple**: Begin with basic templates to understand core concepts
2. **Study Examples**: Review working examples and their explanations
3. **Modify Gradually**: Make small modifications to understand the impact
4. **Build Up**: Progress to more complex templates as you gain confidence

### For Experienced Developers
1. **Assess Needs**: Choose template that best matches your requirements
2. **Customize Architecture**: Modify template architecture for your needs
3. **Extend Functionality**: Add features and capabilities as needed
4. **Optimize Performance**: Optimize for your specific use case

## Usage Examples

### Creating from a Template
```python
from active_inference.applications.templates import TemplateManager, BasicAgentTemplate

# Initialize template manager
template_manager = TemplateManager()

# Load basic agent template
basic_template = template_manager.get_template('basic_agent')

# Create new project from template
project_config = {
    'name': 'my_active_inference_agent',
    'description': 'Custom Active Inference agent',
    'author': 'Your Name',
    'parameters': {
        'state_space': 4,
        'observation_model': 'gaussian',
        'policy_horizon': 5
    }
}

agent_project = basic_template.create_project(project_config)
```

### Customizing Templates
```python
from active_inference.applications.templates import AdvancedSystemTemplate

class CustomAgentTemplate(AdvancedSystemTemplate):
    """Customized Active Inference agent template"""

    def __init__(self, config):
        super().__init__(config)
        self.customize_for_domain()

    def customize_for_domain(self):
        """Customize template for specific application domain"""
        if self.config.get('domain') == 'neuroscience':
            self.add_neural_components()
        elif self.config.get('domain') == 'robotics':
            self.add_robotics_components()
        elif self.config.get('domain') == 'cognitive':
            self.add_cognitive_components()

    def add_neural_components(self):
        """Add neuroscience-specific components"""
        # Add neural data processing
        # Add brain region modeling
        # Add neural dynamics simulation
        pass

    def add_robotics_components(self):
        """Add robotics-specific components"""
        # Add motor control systems
        # Add sensor integration
        # Add navigation capabilities
        pass
```

### Template Configuration
```python
# Template configuration example
template_config = {
    'template_type': 'hierarchical_agent',
    'architecture': {
        'levels': 3,
        'time_scales': [1, 10, 100],
        'coupling_strength': 0.8
    },
    'components': {
        'generative_model': {
            'type': 'hierarchical_gaussian',
            'state_dimensions': [4, 8, 16]
        },
        'inference_engine': {
            'method': 'variational_message_passing',
            'optimization': 'gradient_descent'
        },
        'policy_selection': {
            'method': 'expected_free_energy',
            'planning_horizon': 10
        }
    },
    'integration': {
        'data_sources': ['neural_recordings', 'behavioral_data'],
        'outputs': ['decisions', 'predictions'],
        'real_time': True
    }
}

# Create project from configuration
project = template_manager.create_from_config(template_config)
```

## Template Categories

### Basic Implementation Templates

#### Minimal Agent Template
```python
class MinimalActiveInferenceAgent:
    """Minimal implementation of Active Inference agent"""

    def __init__(self, config):
        self.config = config
        self.setup_model()
        self.setup_inference()
        self.setup_action()

    def setup_model(self):
        """Set up generative model"""
        self.generative_model = GenerativeModel(
            state_dim=self.config['state_dim'],
            obs_dim=self.config['obs_dim']
        )

    def setup_inference(self):
        """Set up inference mechanism"""
        self.inference = VariationalInference(
            model=self.generative_model,
            method=self.config['inference_method']
        )

    def setup_action(self):
        """Set up action selection"""
        self.policy_selection = PolicySelection(
            model=self.generative_model,
            criterion='expected_free_energy'
        )

    def step(self, observation):
        """Perform one step of Active Inference"""
        # Update beliefs
        beliefs = self.inference.update(observation)

        # Select policy
        policy = self.policy_selection.select(beliefs)

        # Execute action
        action = self.execute_policy(policy)

        return action, beliefs
```

#### Simulation Template
```python
class ActiveInferenceSimulation:
    """Template for Active Inference simulations"""

    def __init__(self, sim_config):
        self.config = sim_config
        self.environment = self.create_environment()
        self.agent = self.create_agent()
        self.recorder = self.create_recorder()

    def create_environment(self):
        """Create simulation environment"""
        return Environment(
            dynamics=self.config['environment_dynamics'],
            initial_state=self.config['initial_state']
        )

    def create_agent(self):
        """Create Active Inference agent"""
        return ActiveInferenceAgent(
            config=self.config['agent_config']
        )

    def run_simulation(self, steps: int):
        """Run simulation for specified steps"""
        results = []

        for step in range(steps):
            # Get current observation
            observation = self.environment.get_observation()

            # Agent processes observation
            action, beliefs = self.agent.step(observation)

            # Environment updates
            self.environment.update(action)

            # Record results
            step_result = {
                'step': step,
                'observation': observation,
                'action': action,
                'beliefs': beliefs,
                'environment_state': self.environment.get_state()
            }
            results.append(step_result)
            self.recorder.record(step_result)

        return results
```

### Advanced System Templates

#### Hierarchical System Template
```python
class HierarchicalActiveInferenceSystem:
    """Template for hierarchical Active Inference systems"""

    def __init__(self, hierarchy_config):
        self.config = hierarchy_config
        self.levels = self.create_hierarchy()
        self.setup_interactions()

    def create_hierarchy(self):
        """Create hierarchical system levels"""
        levels = []

        for level_config in self.config['levels']:
            level = SystemLevel(
                time_scale=level_config['time_scale'],
                state_dim=level_config['state_dim'],
                agent=ActiveInferenceAgent(level_config['agent_config'])
            )
            levels.append(level)

        return levels

    def setup_interactions(self):
        """Set up interactions between hierarchy levels"""
        for i in range(len(self.levels) - 1):
            current_level = self.levels[i]
            next_level = self.levels[i + 1]

            # Set up upward influence (from lower to higher levels)
            current_level.upward_connection = next_level

            # Set up downward influence (from higher to lower levels)
            next_level.downward_connection = current_level

    def process_observation(self, observation):
        """Process observation through hierarchy"""
        # Bottom-up processing
        processed_obs = observation
        for level in self.levels:
            processed_obs = level.process_observation(processed_obs)

        # Top-down prediction
        prediction = self.levels[-1].generate_prediction()
        for level in reversed(self.levels[:-1]):
            prediction = level.refine_prediction(prediction)

        return prediction
```

## Contributing

We encourage contributions to the templates module! See [CONTRIBUTING.md](../../CONTRIBUTING.md) for detailed guidelines.

### Contribution Types
- **New Templates**: Create new implementation templates
- **Template Improvements**: Enhance existing templates
- **Domain Templates**: Add domain-specific template categories
- **Integration Templates**: Create integration and deployment templates
- **Example Projects**: Provide complete example projects

### Quality Standards
- **Working Code**: All templates must have working implementations
- **Comprehensive Documentation**: Include detailed README and usage examples
- **Testing**: Provide comprehensive test suites
- **Examples**: Include practical usage examples
- **Validation**: Ensure mathematical and algorithmic correctness

## Learning Resources

- **Template Library**: Explore available templates by category
- **Code Examples**: Study implementation examples in detail
- **Customization Guide**: Learn how to customize templates
- **Best Practices**: Follow established patterns and conventions
- **Community Templates**: Share and discover community-created templates

## Related Documentation

- **[Applications README](../README.md)**: Applications module overview
- **[Best Practices](../best_practices/)**: Architectural guidelines
- **[Case Studies](../case_studies/)**: Real-world implementation examples
- **[Main README](../../README.md)**: Project overview and getting started
- **[Contributing Guide](../../CONTRIBUTING.md)**: Contribution guidelines

## Template Development Guidelines

When creating new templates, follow these guidelines:

### 1. **Structure and Organization**
- Clear directory structure with logical organization
- Consistent naming conventions across all files
- Modular design with clear component separation
- Comprehensive documentation for each component

### 2. **Configuration Management**
- Flexible configuration systems supporting various use cases
- Clear default values with sensible configurations
- Configuration validation and error handling
- Example configurations for common scenarios

### 3. **Code Quality**
- Follow established coding standards and patterns
- Comprehensive error handling and validation
- Efficient algorithms and data structures
- Performance considerations and optimization

### 4. **Testing and Validation**
- Comprehensive test suites covering all functionality
- Performance benchmarks and validation
- Integration testing with other components
- Documentation testing and validation

### 5. **Documentation and Examples**
- Detailed README with setup and usage instructions
- Comprehensive API documentation
- Working examples demonstrating key features
- Troubleshooting guide and FAQ

---

*"Active Inference for, with, by Generative AI"* - Accelerating development through comprehensive templates and proven implementation patterns.



