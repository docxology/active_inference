# Domain Examples - Agent Development Guide

**Guidelines for AI agents working with practical examples and implementations across Active Inference domains.**

## 🤖 Agent Role & Responsibilities

**What agents should do when working with domain examples:**

### Primary Responsibilities
- **Example Development**: Create practical, working examples of Active Inference
- **Code Implementation**: Write clean, documented example code
- **Domain Integration**: Connect examples to specific application domains
- **Learning Progression**: Ensure examples follow educational progression
- **Testing**: Comprehensive testing of example functionality
- **Documentation**: Complete documentation and usage guides

### Development Focus Areas
1. **Educational Examples**: Create examples that teach Active Inference concepts
2. **Domain Applications**: Implement examples in specific domains (AI, neuroscience, etc.)
3. **Code Quality**: Write clean, maintainable, well-documented code
4. **Testing**: Ensure examples work correctly and are robust
5. **Integration**: Connect examples with broader platform

## 🏗️ Architecture & Integration

### Examples Architecture

**Understanding the examples system structure:**

```
Examples Organization
├── Simple Examples (getting started)
├── Domain Examples (domain-specific)
├── Advanced Examples (complex implementations)
├── Tutorial Examples (step-by-step guides)
└── Integration Examples (platform integration)
```

### Integration Points

**Key integration points for examples:**

#### Platform Integration
- **Knowledge Repository**: Examples linked to educational content
- **Learning Paths**: Examples integrated into structured learning
- **Code Execution**: Examples runnable through platform
- **Visualization**: Examples with interactive visualizations

#### Domain Integration
- **Artificial Intelligence**: ML and AI framework examples
- **Neuroscience**: Neural modeling and brain simulation examples
- **Psychology**: Cognitive modeling examples
- **Robotics**: Control system examples

### Example Categories

```python
# Example organization pattern
examples/
├── simple/                  # Basic getting-started examples
├── tutorials/               # Step-by-step tutorial implementations
├── domain_specific/         # Examples for specific domains
├── advanced/               # Complex multi-domain examples
└── integration/            # Platform integration examples
```

## 💻 Development Patterns

### Required Implementation Patterns

**All example development must follow these patterns:**

#### 1. Example Structure Pattern
```python
class ActiveInferenceExample:
    """Template for Active Inference examples"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.setup_environment()
        self.initialize_components()

    def setup_environment(self) -> None:
        """Set up example environment"""
        # Import required libraries
        self.import_dependencies()

        # Configure parameters
        self.configure_parameters()

        # Initialize data
        self.setup_test_data()

    def initialize_components(self) -> None:
        """Initialize Active Inference components"""
        # Create generative model
        self.generative_model = self.create_generative_model()

        # Initialize agent
        self.agent = self.create_active_inference_agent()

        # Set up preferences
        self.preferences = self.define_preferences()

    def run_example(self) -> Dict[str, Any]:
        """Execute the example"""
        # Set up scenario
        scenario = self.create_scenario()

        # Run simulation
        results = self.execute_simulation(scenario)

        # Analyze results
        analysis = self.analyze_results(results)

        return {
            "scenario": scenario,
            "results": results,
            "analysis": analysis,
            "visualization": self.create_visualization(results)
        }

    def create_visualization(self, results: Dict) -> Any:
        """Create visualization of results"""
        # Generate plots or animations
        visualization = self.generate_plots(results)

        # Add interactive elements
        interactive = self.add_interactivity(visualization)

        return interactive
```

#### 2. Educational Pattern
```python
class EducationalExample:
    """Educational example with learning objectives"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.learning_objectives = self.define_learning_objectives()
        self.prerequisites = self.define_prerequisites()

    def define_learning_objectives(self) -> List[str]:
        """Define what users will learn"""
        return [
            "Understand core Active Inference concepts",
            "Implement basic generative models",
            "Apply Active Inference to simple problems",
            "Interpret and visualize results"
        ]

    def validate_prerequisites(self, user_knowledge: Dict) -> ValidationResult:
        """Validate user has required prerequisites"""
        # Check required concepts
        required_concepts = self.prerequisites.get("concepts", [])

        missing_concepts = []
        for concept in required_concepts:
            if not user_knowledge.get(concept, False):
                missing_concepts.append(concept)

        return {
            "valid": len(missing_concepts) == 0,
            "missing_concepts": missing_concepts,
            "recommendations": self.generate_recommendations(missing_concepts)
        }

    def generate_learning_assessment(self, results: Dict) -> Assessment:
        """Generate learning assessment"""
        # Analyze user performance
        performance = self.analyze_performance(results)

        # Generate feedback
        feedback = self.generate_feedback(performance)

        # Suggest next steps
        next_steps = self.suggest_next_steps(performance)

        return Assessment(performance, feedback, next_steps)
```

#### 3. Testing Pattern
```python
class ExampleTesting:
    """Testing pattern for examples"""

    def test_example_execution(self):
        """Test example runs successfully"""
        # Set up example
        example = self.create_example()

        # Execute example
        results = example.run_example()

        # Validate results
        assert results["success"] == True
        assert "results" in results
        assert len(results["results"]) > 0

    def test_example_robustness(self):
        """Test example handles edge cases"""
        example = self.create_example()

        # Test with invalid inputs
        with pytest.raises(ValidationError):
            example.run_with_invalid_data()

        # Test with edge cases
        edge_results = example.run_with_edge_cases()
        assert self.validate_edge_case_results(edge_results)

    def test_example_educational_value(self):
        """Test example provides educational value"""
        example = self.create_example()

        # Validate learning objectives
        objectives = example.learning_objectives
        assert len(objectives) > 0
        assert all(len(obj) > 10 for obj in objectives)

        # Test with novice user
        novice_results = example.run_for_novice()
        assessment = example.assess_learning(novice_results)

        assert assessment["understanding_gained"] > 0.7
```

## 🧪 Testing Standards

### Test Categories (MANDATORY)

#### 1. Example Functionality Testing
```python
class TestExampleFunctionality:
    """Test example functionality"""

    def test_basic_example_execution(self):
        """Test basic example runs correctly"""
        # Load example
        example = self.load_example("basic_active_inference")

        # Execute example
        results = example.run()

        # Validate basic functionality
        assert results["agent_created"] == True
        assert results["simulation_ran"] == True
        assert results["results_generated"] == True

    def test_example_configuration(self):
        """Test example configuration"""
        # Test with default config
        example = self.create_example_with_defaults()
        results = example.run()
        assert results["success"] == True

        # Test with custom config
        custom_config = self.create_custom_config()
        example_custom = self.create_example_with_config(custom_config)
        results_custom = example_custom.run()
        assert results_custom["success"] == True

    def test_example_error_handling(self):
        """Test example error handling"""
        example = self.create_example()

        # Test with invalid configuration
        with pytest.raises(ConfigurationError):
            example.run_with_invalid_config()

        # Test with missing dependencies
        with pytest.raises(DependencyError):
            example.run_without_dependencies()
```

#### 2. Educational Testing
```python
class TestEducationalValue:
    """Test educational value of examples"""

    def test_learning_objectives_achievement(self):
        """Test learning objectives are achieved"""
        example = self.create_example()

        # Run example
        results = example.run()

        # Assess learning
        assessment = example.assess_learning(results)

        # Validate objectives met
        assert assessment["objectives_achieved"] > 0.8
        assert assessment["concepts_understood"] > 0.7

    def test_prerequisite_validation(self):
        """Test prerequisite validation"""
        example = self.create_example()

        # Test with sufficient prerequisites
        user_with_prereqs = self.create_user_with_prerequisites()
        validation = example.validate_prerequisites(user_with_prereqs)
        assert validation["can_proceed"] == True

        # Test without prerequisites
        user_without_prereqs = self.create_user_without_prerequisites()
        validation = example.validate_prerequisites(user_without_prereqs)
        assert validation["can_proceed"] == False
        assert len(validation["missing_concepts"]) > 0
```

#### 3. Integration Testing
```python
class TestExampleIntegration:
    """Test example integration with platform"""

    async def test_knowledge_integration(self):
        """Test integration with knowledge repository"""
        # Create example
        example = self.create_example()

        # Link to knowledge content
        knowledge_links = await self.link_to_knowledge(example)

        # Validate integration
        assert len(knowledge_links) > 0
        assert all(link["valid"] for link in knowledge_links)

    async def test_visualization_integration(self):
        """Test integration with visualization system"""
        example = self.create_example()

        # Generate visualization
        visualization = await example.create_visualization()

        # Test visualization functionality
        assert visualization["interactive"] == True
        assert visualization["educational"] == True
        assert len(visualization["components"]) > 0
```

## 📖 Documentation Standards

### Documentation Requirements (MANDATORY)

#### 1. Example Documentation
**All examples must include comprehensive documentation:**

```python
def example_function(parameter: str) -> Dict[str, Any]:
    """
    Example function demonstrating Active Inference concept.

    This function provides a practical implementation of [concept]
    with clear explanations and educational value.

    Args:
        parameter: Description of parameter and its role in the example

    Returns:
        Dictionary containing example results and analysis

    Learning Objectives:
        - Understand [concept 1]
        - Implement [technique 1]
        - Apply [method 1]

    Prerequisites:
        - Basic understanding of [prerequisite 1]
        - Familiarity with [prerequisite 2]

    Examples:
        >>> result = example_function("test_input")
        >>> print(result["learning_outcome"])
        "concept understood"
    """
    pass
```

#### 2. Usage Documentation
**Examples must include usage documentation:**

```markdown
# Example Usage

## Prerequisites
- Python 3.9+
- Required packages: numpy, scipy, matplotlib
- Basic understanding of Active Inference concepts

## Running the Example
```bash
# Basic execution
python examples/simple/basic_active_inference.py

# With visualization
python examples/simple/basic_active_inference.py --visualize

# Interactive mode
python examples/simple/basic_active_inference.py --interactive

# Debug mode
python examples/simple/basic_active_inference.py --debug
```

## Expected Output
The example will output:
- Agent behavior simulation results
- Learning progression metrics
- Visualization of key concepts
- Performance analysis

## Learning Outcomes
After running this example, you should understand:
- How to implement basic Active Inference agents
- The role of generative models in decision making
- How to interpret Active Inference results
```

## 🚀 Performance Optimization

### Performance Requirements

**Examples must meet these performance standards:**

- **Execution Time**: <30 seconds for typical examples
- **Memory Usage**: <512MB for example execution
- **Visualization**: Responsive interactive visualizations
- **Educational Value**: Clear learning progression

### Optimization Techniques

#### 1. Example Optimization
```python
class ExampleOptimizer:
    """Optimize example performance"""

    def optimize_example_execution(self, example: Example) -> OptimizedExample:
        """Optimize example for performance"""
        # Profile execution
        profile = self.profile_example(example)

        # Identify bottlenecks
        bottlenecks = self.identify_bottlenecks(profile)

        # Apply optimizations
        optimized = self.apply_optimizations(example, bottlenecks)

        # Validate optimization
        validation = self.validate_optimization(optimized)

        return optimized if validation["valid"] else example

    def create_progressive_example(self, base_example: Example) -> ProgressiveExample:
        """Create progressive learning example"""
        # Start with simple version
        simple_version = self.create_simple_version(base_example)

        # Add intermediate steps
        intermediate_versions = self.create_intermediate_versions(base_example)

        # Create advanced version
        advanced_version = self.create_advanced_version(base_example)

        return ProgressiveExample([
            simple_version,
            *intermediate_versions,
            advanced_version
        ])
```

## 🔒 Security Standards

### Security Requirements (MANDATORY)

#### 1. Example Security
```python
class ExampleSecurity:
    """Secure example execution"""

    def validate_example_inputs(self, inputs: Dict[str, Any]) -> SecurityValidation:
        """Validate example inputs for security"""
        # Check for malicious inputs
        malicious_check = self.check_for_malicious_inputs(inputs)

        # Validate data types
        type_check = self.validate_input_types(inputs)

        # Check resource usage
        resource_check = self.validate_resource_usage(inputs)

        return {
            "malicious_safe": malicious_check,
            "types_valid": type_check,
            "resources_safe": resource_check
        }

    def sandbox_example_execution(self, example: Example) -> SandboxedExample:
        """Execute example in secure sandbox"""
        # Create execution sandbox
        sandbox = self.create_execution_sandbox()

        # Set resource limits
        self.set_sandbox_limits(sandbox, {
            "memory": "256MB",
            "cpu": "0.5",
            "time": "30s"
        })

        # Execute in sandbox
        result = self.execute_in_sandbox(example, sandbox)

        return result
```

## 🐛 Debugging & Troubleshooting

### Debug Configuration

```python
# Enable example debugging
debug_config = {
    "debug_mode": True,
    "log_level": "DEBUG",
    "step_debug": True,
    "performance_debug": True,
    "educational_debug": True
}
```

### Common Debugging Patterns

#### 1. Example Execution Debugging
```python
class ExampleDebugger:
    """Debug example execution"""

    def debug_example_failure(self, example: Example, error: Exception) -> DebugReport:
        """Debug example execution failure"""
        # Analyze error context
        context = self.analyze_error_context(error)

        # Check prerequisites
        prereq_check = self.check_prerequisites(example)

        # Validate configuration
        config_check = self.validate_example_config(example)

        # Suggest fixes
        fixes = self.suggest_fixes(context, prereq_check, config_check)

        return {
            "error_context": context,
            "prerequisites": prereq_check,
            "configuration": config_check,
            "suggested_fixes": fixes
        }
```

## 🔄 Development Workflow

### Agent Development Process

1. **Task Assessment**
   - Understand educational requirements
   - Analyze domain needs
   - Consider user learning objectives

2. **Architecture Planning**
   - Design example structure
   - Plan educational progression
   - Consider integration requirements

3. **Test-Driven Development**
   - Write example tests first
   - Test educational value
   - Validate functionality

4. **Implementation**
   - Implement example code
   - Add educational elements
   - Create visualizations

5. **Quality Assurance**
   - Test educational effectiveness
   - Validate code quality
   - Performance testing

6. **Integration**
   - Test with platform
   - Validate learning integration
   - Documentation completion

### Code Review Checklist

**Before submitting example code for review:**

- [ ] **Educational Tests**: Learning objective and prerequisite tests
- [ ] **Functionality Tests**: Example execution and robustness tests
- [ ] **Integration Tests**: Platform integration validation
- [ ] **Documentation Tests**: Complete usage and learning documentation
- [ ] **Performance Tests**: Example performance validation
- [ ] **Security Tests**: Secure example execution validation

## 📚 Learning Resources

### Example Development Resources

- **[Educational Design](https://example.com/educational-design)**: Learning design principles
- **[Code Examples Best Practices](https://example.com/code-examples)**: Example development
- **[Interactive Learning](https://example.com/interactive-learning)**: Interactive education
- **[Python Best Practices](https://example.com/python-best-practices)**: Code quality

### Platform Integration

- **[Knowledge Platform](../../../knowledge/README.md)**: Knowledge architecture
- **[Learning Paths](../../../knowledge/learning_paths.json)**: Learning progression
- **[Visualization Systems](../../../visualization/README.md)**: Visual examples

## 🎯 Success Metrics

### Quality Metrics

- **Educational Effectiveness**: >80% learning objective achievement
- **Code Quality**: Clean, maintainable example code
- **Integration**: Seamless platform integration
- **User Experience**: Intuitive example interfaces

### Development Metrics

- **Example Coverage**: Comprehensive domain coverage
- **Educational Value**: Clear learning progression
- **Code Quality**: Maintainable and documented code
- **Integration**: Platform integration success

---

**Component**: Domain Examples | **Version**: 1.0.0 | **Last Updated**: October 2024

*"Active Inference for, with, by Generative AI"* - Practical understanding through hands-on examples.
