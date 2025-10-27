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
