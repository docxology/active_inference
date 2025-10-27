# Active Inference Source Code - Agent Documentation

This document provides comprehensive guidance for AI agents and contributors working within the source code modules of the Active Inference Knowledge Environment. It outlines implementation patterns, development workflows, and best practices for maintaining the core platform components.

## Source Code Module Overview

The `src/active_inference/` directory contains the main Python implementation of the Active Inference Knowledge Environment. This module provides the core functionality that powers the entire platform, including knowledge management, research tools, visualization systems, and application frameworks.

## Architecture Overview

### Core Design Principles
- **Modular Architecture**: Each submodule is independently testable and reusable
- **Interface Segregation**: Clean, minimal interfaces between components
- **Configuration-Driven**: All components configured through structured configuration
- **Error Resilience**: Comprehensive error handling and recovery mechanisms
- **Performance Optimized**: Efficient algorithms and data structures

### Key Integration Points
- **Package Management**: Clean import structure and dependency management
- **CLI Integration**: Command-line interface for all platform features
- **Configuration System**: Unified configuration management across all components
- **Error Reporting**: Centralized error handling and logging
- **Testing Integration**: Built-in testing support for all components

## Core Responsibilities

### Package Management (`__init__.py`)
**Central coordination and module management**
- Package initialization and configuration loading
- Core class and function exports
- Version and metadata management
- Import structure organization and validation

**Key Methods to Implement:**
```python
def initialize_platform(config: Dict[str, Any]) -> Platform:
    """Initialize the complete platform with all components"""

def create_knowledge_repository(config: RepositoryConfig) -> KnowledgeRepository:
    """Create and configure knowledge repository instance with proper initialization"""

def create_research_framework(config: ResearchConfig) -> ResearchFramework:
    """Create research framework with experiment management and validation"""

def create_visualization_engine(config: VisualizationConfig) -> VisualizationEngine:
    """Create visualization engine for interactive exploration and rendering"""

def create_application_framework(config: ApplicationConfig) -> ApplicationFramework:
    """Create application framework for practical implementations and templates"""

def validate_platform_integrity() -> Dict[str, Any]:
    """Validate all platform components and their integrations"""

def get_system_requirements() -> Dict[str, Any]:
    """Get comprehensive system requirements and dependencies"""

def create_platform_backup() -> Path:
    """Create complete platform backup including all data and configurations"""
```

### Command Line Interface (`cli.py`)
**User interaction and command processing**
- Command parsing and argument validation
- Interactive workflow management
- Help system and user guidance
- Error handling and user feedback
- Platform status and monitoring

**Key Methods to Implement:**
```python
def main() -> int:
    """Main CLI entry point with comprehensive argument parsing and routing"""

def create_parser() -> ArgumentParser:
    """Create comprehensive command-line argument parser with all platform commands"""

def handle_knowledge_commands(args: Namespace) -> int:
    """Handle knowledge repository commands with proper error handling and validation"""

def handle_research_commands(args: Namespace) -> int:
    """Handle research and experimentation commands with workflow management"""

def handle_visualization_commands(args: Namespace) -> int:
    """Handle visualization and exploration commands with resource management"""

def handle_platform_commands(args: Namespace) -> int:
    """Handle platform management and deployment commands with status monitoring"""

def execute_command_with_validation(command: str, args: Dict[str, Any]) -> Dict[str, Any]:
    """Execute command with comprehensive validation and error handling"""

def format_user_output(data: Any, format_type: str) -> str:
    """Format output for user display in various formats (table, json, markdown)"""

def handle_interactive_mode() -> None:
    """Handle interactive CLI mode with command history and auto-completion"""
```

### Knowledge Implementations (`knowledge/implementations.py`)
**Practical code implementations and educational tutorials**
- Implement comprehensive code examples for Active Inference concepts
- Create interactive tutorials with step-by-step guidance
- Develop educational implementations with validation and testing
- Maintain implementation standards and quality assurance

**Key Methods to Implement:**
```python
def implement_basic_active_inference(self, config: Dict[str, Any]) -> Implementation:
    """Implement basic Active Inference agent with complete working code"""

def create_tutorial_implementation(self, concept: str, difficulty: str) -> Tutorial:
    """Create interactive tutorial with step-by-step implementation"""

def validate_implementation_correctness(self, code: str) -> Dict[str, Any]:
    """Validate implementation for mathematical and algorithmic correctness"""

def generate_educational_examples(self, concepts: List[str]) -> List[Example]:
    """Generate comprehensive educational examples for learning"""

def implement_expected_free_energy(self, model_config: Dict[str, Any]) -> Implementation:
    """Implement Expected Free Energy calculation with validation"""

def create_neural_network_implementation(self, framework: str) -> Implementation:
    """Create neural network implementation of Active Inference principles"""

def implement_variational_inference(self, algorithm: str) -> Implementation:
    """Implement variational inference methods for Active Inference"""

def validate_mathematical_implementations(self) -> Dict[str, Any]:
    """Validate all mathematical implementations for accuracy"""

def create_performance_benchmarks(self, implementations: List[str]) -> Dict[str, Any]:
    """Create performance benchmarks for different implementations"""

def implement_cross_validation_system(self) -> ValidationSystem:
    """Implement comprehensive cross-validation for implementation correctness"""
```

### Knowledge Applications (`knowledge/applications.py`)
**Real-world applications and domain-specific implementations**
- Implement applications across multiple research domains
- Create comprehensive case studies with working examples
- Develop domain-specific knowledge and implementation patterns
- Maintain application standards and best practices

**Key Methods to Implement:**
```python
def implement_ai_generative_models(self, model_type: str, config: Dict[str, Any]) -> Application:
    """Implement Active Inference in generative AI models"""

def create_neuroscience_application(self, neural_system: str) -> Application:
    """Create Active Inference application for neural systems"""

def implement_engineering_control_systems(self, control_type: str) -> Application:
    """Implement Active Inference in engineering control systems"""

def create_robotics_application(self, robot_type: str, config: Dict[str, Any]) -> Application:
    """Create Active Inference application for robotics control"""

def implement_psychology_application(self, cognitive_model: str) -> Application:
    """Implement Active Inference in psychological and behavioral models"""

def create_education_application(self, learning_system: str) -> Application:
    """Create Active Inference application for educational systems"""

def validate_application_completeness(self, application: Application) -> Dict[str, Any]:
    """Validate application for completeness and practical utility"""

def generate_domain_specific_examples(self, domain: str) -> List[Example]:
    """Generate domain-specific examples and case studies"""

def implement_cross_domain_applications(self, domains: List[str]) -> Dict[str, Application]:
    """Implement applications that span multiple research domains"""

def create_application_integration_framework(self) -> IntegrationFramework:
    """Create framework for integrating applications across domains"""
```

### Research Data Management (`research/data_management.py`)
**Comprehensive data management for scientific research**
- Implement data collection from multiple sources
- Create data preprocessing and validation systems
- Develop data storage and retrieval mechanisms
- Maintain data security and integrity standards

**Key Methods to Implement:**
```python
def collect_simulation_data(self, config: DataCollectionConfig) -> str:
    """Collect data from simulation experiments with validation"""

def collect_experiment_data(self, config: DataCollectionConfig) -> str:
    """Collect data from experimental setups with monitoring"""

def collect_api_data(self, config: DataCollectionConfig) -> str:
    """Collect data from external APIs with error handling"""

def validate_data_integrity(self, dataset_id: str) -> Dict[str, Any]:
    """Validate data integrity including format, completeness, and quality"""

def implement_data_preprocessing_pipeline(self, steps: List[str]) -> Pipeline:
    """Implement comprehensive data preprocessing pipeline"""

def create_data_security_framework(self, security_level: str) -> SecurityFramework:
    """Create data security framework with access controls"""

def implement_data_backup_system(self) -> BackupSystem:
    """Implement comprehensive data backup and recovery system"""

def validate_data_quality_metrics(self, data: Any) -> Dict[str, Any]:
    """Validate data quality using statistical and domain-specific metrics"""

def create_data_integration_framework(self) -> IntegrationFramework:
    """Create framework for integrating data from multiple sources"""

def implement_data_versioning_system(self) -> VersioningSystem:
    """Implement data versioning and change tracking system"""
```

## Development Workflows

### Code Organization Workflow
1. **Module Structure**: Organize code into logical, reusable modules
2. **Interface Design**: Create clean, documented interfaces between modules
3. **Import Management**: Maintain clean import structure and avoid circular dependencies
4. **Configuration**: Implement configuration-driven initialization for all components
5. **Error Handling**: Add comprehensive error handling and user-friendly messages

### Integration Workflow
1. **Component Testing**: Test each component in isolation
2. **Integration Testing**: Test component interactions and data flow
3. **Performance Testing**: Validate performance characteristics
4. **Documentation**: Update README and AGENTS files for new functionality
5. **Code Review**: Submit for peer review and validation

## Quality Standards

### Code Quality Requirements
- **Test Coverage**: >95% test coverage for all source code modules
- **Type Safety**: Complete type annotations for all functions and methods
- **Documentation**: Comprehensive docstrings with examples for all public APIs
- **Error Handling**: Robust error handling with informative error messages
- **Performance**: Include performance benchmarks and optimization where appropriate
- **Security**: Follow security best practices for data handling and user input

### Documentation Standards
- **README Coverage**: Every module must have comprehensive README.md
- **AGENTS Documentation**: Every module must have detailed AGENTS.md
- **API Documentation**: All public APIs must be documented with examples
- **Code Comments**: Complex logic must be explained with inline comments
- **Usage Examples**: Include working examples for all major features

### Interface Standards
- **Factory Pattern**: Use factory functions for object creation
- **Configuration Objects**: Accept configuration dictionaries for initialization
- **Return Validation**: Validate return values and handle edge cases
- **Resource Management**: Proper cleanup of resources and connections
- **Logging Integration**: Comprehensive logging for debugging and monitoring

## Implementation Patterns

### Factory Pattern Implementation
```python
def create_component(component_type: str, config: Dict[str, Any]) -> Any:
    """Create platform component using factory pattern"""

    component_factories = {
        'knowledge_repository': create_knowledge_repository,
        'research_framework': create_research_framework,
        'visualization_engine': create_visualization_engine,
        'application_framework': create_application_framework,
        'platform': initialize_platform
    }

    if component_type not in component_factories:
        raise ValueError(f"Unknown component type: {component_type}")

    return component_factories[component_type](config)

def validate_component_dependencies(components: Dict[str, Any]) -> List[str]:
    """Validate that all component dependencies are satisfied"""

    validation_issues = []

    # Check knowledge repository dependencies
    if 'knowledge_repository' in components:
        repo = components['knowledge_repository']
        if not hasattr(repo, 'search') or not callable(repo.search):
            validation_issues.append("Knowledge repository missing search method")

    # Check research framework dependencies
    if 'research_framework' in components:
        research = components['research_framework']
        if not hasattr(research, 'run_study') or not callable(research.run_study):
            validation_issues.append("Research framework missing run_study method")

    return validation_issues
```

### Configuration Management
```python
def load_platform_config(config_path: Path) -> Dict[str, Any]:
    """Load and validate platform configuration"""

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Validate required sections
    required_sections = ['platform', 'knowledge', 'research', 'visualization', 'applications']
    missing_sections = [section for section in required_sections if section not in config]

    if missing_sections:
        raise ValueError(f"Missing configuration sections: {missing_sections}")

    return config

def merge_config_overrides(base_config: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    """Merge configuration overrides with base configuration"""

    merged = deepcopy(base_config)

    for section, section_config in overrides.items():
        if section in merged:
            if isinstance(merged[section], dict) and isinstance(section_config, dict):
                merged[section] = deep_merge(merged[section], section_config)
            else:
                merged[section] = section_config
        else:
            merged[section] = section_config

    return merged
```

## Testing Guidelines

### Unit Testing Requirements
- **Component Isolation**: Test each component in isolation from others
- **Mock Dependencies**: Use appropriate mocking for external dependencies
- **Edge Case Coverage**: Test boundary conditions and error cases
- **Performance Testing**: Include performance benchmarks for critical paths
- **Integration Testing**: Test component interactions and data flow

### Test Structure
```python
class TestPlatformIntegration(unittest.TestCase):
    """Integration tests for platform components"""

    def setUp(self):
        """Set up test environment and mock dependencies"""
        self.config = create_test_config()
        self.platform = initialize_platform(self.config)

    def test_knowledge_research_integration(self):
        """Test integration between knowledge and research components"""
        # Test knowledge retrieval for research
        knowledge = self.platform.get_knowledge_repository()
        research = self.platform.get_research_framework()

        # Verify components can interact
        concepts = knowledge.search("entropy")
        self.assertGreater(len(concepts), 0)

        # Verify research can use knowledge
        experiment_config = research.create_experiment_config(concepts)
        self.assertIsNotNone(experiment_config)

    def test_visualization_knowledge_integration(self):
        """Test integration between visualization and knowledge components"""
        # Test visualization of knowledge content
        knowledge = self.platform.get_knowledge_repository()
        visualization = self.platform.get_visualization_engine()

        # Verify visualization can render knowledge
        concept_diagram = visualization.create_concept_diagram("active_inference")
        self.assertIsNotNone(concept_diagram)

        # Verify diagram contains expected content
        self.assertIn("perception", concept_diagram.nodes)
        self.assertIn("action", concept_diagram.nodes)
```

## Performance Optimization

### Memory Management
- **Resource Pools**: Implement resource pooling for expensive objects
- **Streaming**: Use streaming for large data processing
- **Garbage Collection**: Monitor and optimize garbage collection
- **Memory Profiling**: Built-in memory usage monitoring and alerts

### Computational Efficiency
- **Algorithm Selection**: Choose appropriate algorithms for performance requirements
- **Caching**: Implement intelligent caching strategies
- **Parallel Processing**: Utilize parallel processing where beneficial
- **Lazy Loading**: Defer expensive operations until needed

## Deployment Considerations

### Production Deployment
- **Environment Configuration**: Support for multiple deployment environments
- **Health Monitoring**: Comprehensive health checks and monitoring
- **Scaling Support**: Horizontal and vertical scaling capabilities
- **Backup and Recovery**: Automated backup and recovery procedures

### Development Deployment
- **Local Development**: Easy setup for local development and testing
- **Testing Environments**: Isolated testing environments
- **Development Tools**: Integrated development tools and debugging
- **Hot Reloading**: Support for hot reloading during development

## Error Handling and Debugging

### Comprehensive Error Management
- **Error Classification**: Classify errors by type and severity
- **User-Friendly Messages**: Provide clear, actionable error messages
- **Debug Information**: Include debugging information for developers
- **Error Recovery**: Implement graceful error recovery where possible

### Logging and Monitoring
- **Structured Logging**: Use structured logging with consistent formats
- **Performance Monitoring**: Monitor performance metrics and alerts
- **Error Tracking**: Track and analyze error patterns
- **Audit Trails**: Maintain audit trails for critical operations

## Getting Started as an Agent

### Development Setup
1. **Explore Codebase**: Review existing implementations and patterns
2. **Run Tests**: Ensure all tests pass before making changes
3. **Study Interfaces**: Understand the interfaces between components
4. **Add Logging**: Add appropriate logging for new functionality
5. **Write Tests**: Follow TDD with comprehensive test coverage

### Implementation Process
1. **Design Phase**: Design new functionality with clear interfaces
2. **Implementation**: Implement following established patterns
3. **Testing**: Add comprehensive tests including edge cases
4. **Documentation**: Update README and AGENTS files
5. **Integration**: Ensure integration with existing components
6. **Performance**: Optimize for performance and memory usage

### Code Review Checklist
- [ ] Code follows established patterns and conventions
- [ ] Comprehensive tests included with >90% coverage
- [ ] Documentation updated (README.md and AGENTS.md)
- [ ] Type hints and docstrings complete
- [ ] Error handling comprehensive
- [ ] Performance considerations addressed
- [ ] Integration with existing components verified
- [ ] No breaking changes to existing interfaces

## Common Implementation Patterns

### Component Creation Pattern
```python
def create_component_with_validation(component_type: str, config: Dict[str, Any]) -> Any:
    """Create component with comprehensive validation"""

    # Validate configuration
    validate_config_schema(config, component_type)

    # Create component using factory
    component = component_factories[component_type](config)

    # Validate component functionality
    validate_component_functionality(component)

    # Initialize component dependencies
    initialize_component_dependencies(component)

    return component
```

### Error Handling Pattern
```python
def execute_with_error_handling(operation: Callable, *args, **kwargs) -> Any:
    """Execute operation with comprehensive error handling"""

    try:
        # Log operation start
        logger.info(f"Starting operation: {operation.__name__}")

        # Execute operation
        result = operation(*args, **kwargs)

        # Log success
        logger.info(f"Operation completed successfully: {operation.__name__}")

        return result

    except ValidationError as e:
        logger.error(f"Validation error in {operation.__name__}: {e}")
        raise

    except ResourceError as e:
        logger.error(f"Resource error in {operation.__name__}: {e}")
        # Attempt recovery or provide user guidance
        raise

    except Exception as e:
        logger.error(f"Unexpected error in {operation.__name__}: {e}")
        # Log full traceback for debugging
        logger.debug(f"Full traceback: {traceback.format_exc()}")
        raise
```

## Related Documentation

- **[Main AGENTS.md](../../AGENTS.md)**: Project-wide agent guidelines
- **[Applications AGENTS.md](../applications/AGENTS.md)**: Application development guidelines
- **[Knowledge AGENTS.md](../knowledge/AGENTS.md)**: Knowledge management guidelines
- **[Research AGENTS.md](../research/AGENTS.md)**: Research tool development guidelines
- **[Visualization AGENTS.md](../visualization/AGENTS.md)**: Visualization system guidelines
- **[Platform AGENTS.md](../platform/AGENTS.md)**: Platform infrastructure guidelines
- **[Tools AGENTS.md](../tools/AGENTS.md)**: Development tools guidelines

---

*"Active Inference for, with, by Generative AI"* - Building the source code foundation through collaborative intelligence and comprehensive implementation.
