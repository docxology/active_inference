# Research Experiments Framework - Agent Development Guide

**Guidelines for AI agents working with research experiments in the Active Inference Knowledge Environment.**

## 🤖 Agent Role & Responsibilities

**What agents should do when working with research experiments:**

### Primary Responsibilities
- **Experiment Design**: Design rigorous, reproducible research experiments
- **Implementation Development**: Implement robust experimental frameworks and tools
- **Statistical Validation**: Ensure statistical validity and scientific rigor
- **Result Analysis**: Provide comprehensive analysis and interpretation of results
- **Reproducibility Engineering**: Create systems for maximum reproducibility

### Development Focus Areas
1. **Experiment Templates**: Develop standardized experiment design templates
2. **Execution Engines**: Build robust experiment execution and automation systems
3. **Analysis Frameworks**: Create comprehensive statistical analysis and validation tools
4. **Reporting Systems**: Develop automated scientific reporting and publication tools
5. **Validation Systems**: Implement rigorous validation and reproducibility checking

## 🏗️ Architecture & Integration

### Experiments Framework Architecture

**Understanding how experiments fit into the research ecosystem:**

```
Research Workflow Layer
├── Experiment Design (Templates, protocols, methodologies)
├── Experiment Execution (Automation, monitoring, data collection)
├── Result Analysis (Statistics, validation, interpretation)
└── Scientific Reporting (Publication, documentation, archiving)
```

### Integration Points

**Key integration points and dependencies:**

#### Upstream Components
- **Research Framework**: Core research tools and scientific methods
- **Knowledge Base**: Theoretical foundations and research questions
- **Data Management**: Research data handling and preprocessing
- **Simulation Systems**: Simulated data for experimental validation

#### Downstream Components
- **Publication Systems**: Integration with academic writing and dissemination
- **Community Tools**: Collaborative experiment development and validation
- **Educational Content**: Experiment-based learning materials and tutorials
- **Platform Analytics**: Experiment performance and research impact metrics

#### External Systems
- **Laboratory Systems**: Integration with lab equipment and data acquisition
- **Statistical Software**: R, MATLAB, specialized statistical analysis packages
- **Version Control**: Git integration for experiment versioning and reproducibility
- **Publication Platforms**: Journal submission systems and preprint servers

### Experiment Flow Patterns

```python
# Typical experiment development workflow
research_question → design → implementation → execution → analysis → reporting → review
```

## 💻 Development Patterns

### Required Implementation Patterns

**All experiment development must follow these patterns:**

#### 1. Experiment Factory Pattern (PREFERRED)

```python
def create_experiment(experiment_type: str, config: Dict[str, Any]) -> BaseExperiment:
    """Create experiment using factory pattern with validation"""

    experiment_factories = {
        'model_comparison': create_model_comparison_experiment,
        'parameter_study': create_parameter_study_experiment,
        'simulation_experiment': create_simulation_experiment,
        'empirical_experiment': create_empirical_experiment,
        'validation_experiment': create_validation_experiment,
        'reproducibility_study': create_reproducibility_experiment
    }

    if experiment_type not in experiment_factories:
        raise ValueError(f"Unknown experiment type: {experiment_type}")

    # Validate experiment configuration
    validate_experiment_config(config)

    # Create experiment with comprehensive setup
    experiment = experiment_factories[experiment_type](config)

    # Add validation framework
    validated_experiment = add_validation_framework(experiment)

    # Add reproducibility measures
    reproducible_experiment = add_reproducibility_measures(validated_experiment)

    return reproducible_experiment

def validate_experiment_config(config: Dict[str, Any]) -> None:
    """Validate experiment configuration"""

    required_fields = ['experiment_type', 'research_question', 'validation_method']

    for field in required_fields:
        if field not in config:
            raise ExperimentConfigurationError(f"Missing required field: {field}")

    # Type-specific validation
    if config['experiment_type'] == 'model_comparison':
        validate_model_comparison_config(config)
    elif config['experiment_type'] == 'empirical_experiment':
        validate_empirical_experiment_config(config)
```

#### 2. Experiment Configuration Pattern (MANDATORY)

```python
from dataclasses import dataclass
from typing import Dict, Any, List, Optional

@dataclass
class ExperimentConfig:
    """Configuration for research experiments"""

    # Core experiment settings
    experiment_type: str
    research_question: Dict[str, Any]
    hypothesis: str
    variables: Dict[str, Any]

    # Validation settings
    validation_method: str
    statistical_significance: float = 0.05
    power_analysis: bool = True
    reproducibility_checks: bool = True

    # Execution settings
    repetitions: int = 100
    parallel_execution: bool = True
    random_seed: int = 42
    checkpointing: bool = True

    # Quality settings
    documentation_level: str = "comprehensive"
    peer_review_ready: bool = False
    publication_ready: bool = False

    def validate_configuration(self) -> List[str]:
        """Validate experiment configuration"""
        errors = []

        # Check required fields
        required_fields = ['experiment_type', 'research_question', 'hypothesis']
        for field in required_fields:
            if not getattr(self, field):
                errors.append(f"Missing required field: {field}")

        # Validate experiment type
        valid_types = [
            'model_comparison', 'parameter_study', 'simulation_experiment',
            'empirical_experiment', 'validation_experiment', 'reproducibility_study'
        ]
        if self.experiment_type not in valid_types:
            errors.append(f"Invalid experiment type: {self.experiment_type}")

        # Validate statistical parameters
        if not (0 < self.statistical_significance < 1):
            errors.append("Statistical significance must be between 0 and 1")

        if self.repetitions <= 0:
            errors.append("Repetitions must be positive")

        return errors

    def to_experiment_parameters(self) -> Dict[str, Any]:
        """Convert configuration to experiment parameters"""
        return {
            'experiment_type': self.experiment_type,
            'research_question': self.research_question,
            'hypothesis': self.hypothesis,
            'variables': self.variables,
            'validation_method': self.validation_method,
            'statistical_significance': self.statistical_significance,
            'power_analysis': self.power_analysis,
            'reproducibility_checks': self.reproducibility_checks,
            'repetitions': self.repetitions,
            'parallel_execution': self.parallel_execution,
            'random_seed': self.random_seed,
            'checkpointing': self.checkpointing,
            'documentation_level': self.documentation_level,
            'peer_review_ready': self.peer_review_ready,
            'publication_ready': self.publication_ready
        }
```

#### 3. Reproducibility Pattern (MANDATORY)

```python
def ensure_experiment_reproducibility(experiment: BaseExperiment, reproducibility_config: Dict[str, Any]) -> ReproducibleExperiment:
    """Ensure experiment is fully reproducible"""

    # Set up version control
    version_control = setup_experiment_version_control(experiment, reproducibility_config)

    # Create reproducible environment
    reproducible_env = create_reproducible_environment(experiment)

    # Implement data provenance
    data_provenance = implement_data_provenance_tracking(experiment)

    # Add comprehensive logging
    experiment_logging = add_comprehensive_logging(experiment)

    # Create reproducibility package
    reproducibility_package = create_reproducibility_package(
        version_control, reproducible_env, data_provenance, experiment_logging
    )

    # Validate reproducibility
    reproducibility_validation = validate_experiment_reproducibility(reproducibility_package)

    return ReproducibleExperiment(
        experiment=experiment,
        version_control=version_control,
        environment=reproducible_env,
        data_provenance=data_provenance,
        logging=experiment_logging,
        validation=reproducibility_validation
    )

def create_reproducibility_package(experiment: BaseExperiment, config: Dict[str, Any]) -> ReproducibilityPackage:
    """Create complete reproducibility package for experiment"""

    # Generate comprehensive documentation
    experiment_documentation = generate_comprehensive_documentation(experiment)

    # Create environment specification
    environment_spec = create_environment_specification(experiment)

    # Package code and dependencies
    code_package = package_experiment_code(experiment)

    # Create data package
    data_package = create_data_package(experiment)

    # Generate execution scripts
    execution_scripts = generate_execution_scripts(experiment)

    # Create validation suite
    validation_suite = create_validation_suite(experiment)

    return ReproducibilityPackage(
        documentation=experiment_documentation,
        environment=environment_spec,
        code=code_package,
        data=data_package,
        scripts=execution_scripts,
        validation=validation_suite
    )
```

## 🧪 Experiment Testing Standards

### Experiment Testing Categories (MANDATORY)

#### 1. Reproducibility Testing
**Test experiment reproducibility and reliability:**

```python
def test_experiment_reproducibility():
    """Test experiment reproducibility"""
    # Create original experiment
    original_experiment = create_test_experiment()
    original_results = run_experiment(original_experiment)

    # Attempt reproduction
    reproduction_config = create_reproduction_config()
    reproduced_results = attempt_experiment_reproduction(original_experiment, reproduction_config)

    # Validate reproduction accuracy
    reproduction_validation = validate_reproduction_accuracy(original_results, reproduced_results)
    assert reproduction_validation['reproducible'], "Experiment not reproducible"

    # Test with different environments
    environment_tests = test_cross_environment_reproduction(original_experiment)
    assert environment_tests['cross_environment_success'], "Not reproducible across environments"

def test_statistical_validity():
    """Test statistical validity of experiment design"""
    experiment = create_test_experiment()

    # Validate statistical design
    statistical_validation = validate_statistical_design(experiment)
    assert statistical_validation['statistically_sound'], "Experiment design not statistically sound"

    # Check power analysis
    power_analysis = perform_power_analysis(experiment)
    assert power_analysis['adequate_power'], "Insufficient statistical power"

    # Validate sample size
    sample_validation = validate_sample_size(experiment)
    assert sample_validation['adequate_sample'], "Inadequate sample size for statistical validity"
```

#### 2. Execution Testing
**Test experiment execution and automation:**

```python
def test_experiment_execution():
    """Test experiment execution and automation"""
    experiment = create_complex_experiment()

    # Test execution pipeline
    execution_test = test_execution_pipeline(experiment)
    assert execution_test['pipeline_success'], "Experiment execution pipeline failed"

    # Test parallel execution
    parallel_test = test_parallel_execution(experiment)
    assert parallel_test['parallel_success'], "Parallel execution failed"

    # Test checkpointing and recovery
    recovery_test = test_checkpoint_recovery(experiment)
    assert recovery_test['recovery_success'], "Checkpoint recovery failed"

def test_error_handling():
    """Test experiment error handling and recovery"""
    # Test with various error conditions
    error_scenarios = [
        "network_failure",
        "disk_space_exhaustion",
        "computation_timeout",
        "invalid_parameters",
        "system_interruption"
    ]

    for scenario in error_scenarios:
        error_test = test_error_scenario(scenario)
        assert error_test['graceful_handling'], f"Failed to handle error: {scenario}"

        # Test recovery mechanisms
        recovery_test = test_error_recovery(scenario)
        assert recovery_test['recovery_success'], f"Failed to recover from error: {scenario}"
```

#### 3. Result Validation Testing
**Test result analysis and validation:**

```python
def test_result_analysis():
    """Test result analysis and interpretation"""
    experiment = create_analysis_experiment()
    results = run_experiment(experiment)

    # Test statistical analysis
    analysis_test = test_statistical_analysis(results)
    assert analysis_test['analysis_correct'], "Statistical analysis incorrect"

    # Test result interpretation
    interpretation_test = test_result_interpretation(results)
    assert interpretation_test['interpretation_reasonable'], "Result interpretation unreasonable"

    # Test validation procedures
    validation_test = test_result_validation(results)
    assert validation_test['validation_thorough'], "Result validation insufficient"

def test_reporting_system():
    """Test automated reporting and documentation"""
    experiment = create_reporting_experiment()
    results = run_experiment(experiment)

    # Test report generation
    report_test = test_report_generation(results)
    assert report_test['report_complete'], "Report generation incomplete"

    # Test documentation accuracy
    documentation_test = test_documentation_accuracy(results)
    assert documentation_test['documentation_accurate'], "Documentation inaccurate"
```

### Experiment Coverage Requirements

- **Method Coverage**: All major experiment types supported
- **Domain Coverage**: Experiments for all Active Inference domains
- **Validation Coverage**: Comprehensive validation for all experiment types
- **Reproducibility Coverage**: All experiments fully reproducible
- **Documentation Coverage**: Complete documentation for experiment workflows

### Experiment Testing Commands

```bash
# Test all experiment functionality
make test-experiments

# Test experiment reproducibility
pytest research/experiments/tests/test_reproducibility.py -v

# Test statistical validity
pytest research/experiments/tests/test_statistical_validity.py -v

# Test execution systems
pytest research/experiments/tests/test_execution.py -v

# Validate experiment quality
python research/experiments/validate_experiment_quality.py
```

## 📖 Experiment Documentation Standards

### Documentation Requirements (MANDATORY)

#### 1. Scientific Documentation
**All experiments must have comprehensive scientific documentation:**

```python
def document_scientific_experiment(experiment: BaseExperiment, config: Dict[str, Any]) -> str:
    """Document experiment following scientific standards"""

    scientific_documentation = {
        "research_question": document_research_question(experiment, config),
        "methodology": document_methodology(experiment, config),
        "implementation": document_implementation(experiment, config),
        "results": document_results(experiment, config),
        "analysis": document_analysis(experiment, config),
        "interpretation": document_interpretation(experiment, config),
        "reproducibility": document_reproducibility(experiment, config)
    }

    return format_scientific_documentation(scientific_documentation)

def document_methodology(experiment: BaseExperiment, config: Dict[str, Any]) -> str:
    """Document experimental methodology comprehensively"""

    methodology = experiment.get_methodology()

    # Document experimental design
    design_documentation = document_experimental_design(methodology)

    # Document data collection
    data_collection_docs = document_data_collection_procedures(methodology)

    # Document analysis methods
    analysis_docs = document_analysis_methods(methodology)

    # Document validation procedures
    validation_docs = document_validation_procedures(methodology)

    return format_methodology_documentation(
        design_documentation, data_collection_docs, analysis_docs, validation_docs
    )
```

#### 2. Reproducibility Documentation
**All experiments must document complete reproducibility:**

```python
def document_experiment_reproducibility(experiment: BaseExperiment, config: Dict[str, Any]) -> str:
    """Document experiment reproducibility comprehensively"""

    reproducibility_documentation = {
        "environment_specification": document_environment_specification(experiment, config),
        "dependency_management": document_dependency_management(experiment, config),
        "random_seed_handling": document_random_seed_handling(experiment, config),
        "data_provenance": document_data_provenance(experiment, config),
        "execution_procedures": document_execution_procedures(experiment, config),
        "validation_checklist": create_reproducibility_checklist(experiment, config)
    }

    return format_reproducibility_documentation(reproducibility_documentation)
```

#### 3. Statistical Documentation
**All experiments must document statistical methods thoroughly:**

```python
def document_statistical_methods(experiment: BaseExperiment, config: Dict[str, Any]) -> str:
    """Document statistical methods and analysis"""

    statistical_documentation = {
        "statistical_design": document_statistical_design(experiment, config),
        "power_analysis": document_power_analysis(experiment, config),
        "analysis_methods": document_analysis_methods(experiment, config),
        "result_validation": document_result_validation(experiment, config),
        "interpretation_guidelines": document_interpretation_guidelines(experiment, config)
    }

    return format_statistical_documentation(statistical_documentation)
```

## 🚀 Performance Optimization

### Experiment Performance Requirements

**Experiment systems must meet these performance standards:**

- **Execution Speed**: Experiments complete within reasonable timeframes
- **Resource Efficiency**: Efficient use of computational resources
- **Scalability**: Scale with experiment size and complexity
- **Reliability**: Robust execution without unexpected failures
- **Monitoring**: Real-time monitoring of experiment progress

### Optimization Techniques

#### 1. Execution Optimization

```python
def optimize_experiment_execution(experiment: BaseExperiment, optimization_config: Dict[str, Any]) -> OptimizedExperiment:
    """Optimize experiment execution for performance"""

    # Optimize parallel execution
    parallel_optimized = optimize_parallel_execution(experiment, optimization_config)

    # Optimize memory usage
    memory_optimized = optimize_memory_usage(parallel_optimized)

    # Optimize data handling
    data_optimized = optimize_data_handling(memory_optimized)

    # Add execution monitoring
    monitored_experiment = add_execution_monitoring(data_optimized)

    # Validate optimization
    optimization_validation = validate_execution_optimization(monitored_experiment, optimization_config)

    return OptimizedExperiment(
        experiment=monitored_experiment,
        optimization_config=optimization_config,
        validation=optimization_validation
    )
```

#### 2. Statistical Optimization

```python
def optimize_statistical_analysis(analysis_config: Dict[str, Any]) -> OptimizedAnalysis:
    """Optimize statistical analysis for accuracy and efficiency"""

    # Choose optimal statistical methods
    optimal_methods = select_optimal_statistical_methods(analysis_config)

    # Optimize computation
    computation_optimized = optimize_statistical_computation(optimal_methods)

    # Add numerical stability
    numerically_stable = add_numerical_stability(computation_optimized)

    # Optimize for large datasets
    large_scale_optimized = optimize_for_large_scale(numerically_stable)

    # Validate statistical correctness
    statistical_validation = validate_statistical_optimization(large_scale_optimized, analysis_config)

    return OptimizedAnalysis(
        methods=optimal_methods,
        computation=computation_optimized,
        stability=numerically_stable,
        large_scale=large_scale_optimized,
        validation=statistical_validation
    )
```

## 🔒 Experiment Security Standards

### Experiment Security Requirements (MANDATORY)

#### 1. Data Security

```python
def validate_experiment_data_security(experiment: BaseExperiment, security_config: Dict[str, Any]) -> SecurityResult:
    """Validate data security in experiments"""

    security_checks = {
        "data_anonymization": validate_data_anonymization(experiment),
        "access_control": validate_data_access_control(experiment),
        "audit_logging": validate_experiment_audit_logging(experiment),
        "compliance_checking": validate_regulatory_compliance(experiment)
    }

    return {
        "secure": all(security_checks.values()),
        "checks": security_checks,
        "violations": [k for k, v in security_checks.items() if not v]
    }

def implement_data_security(experiment: BaseExperiment, security_config: Dict[str, Any]) -> SecureExperiment:
    """Implement comprehensive data security for experiment"""

    # Add data anonymization
    anonymized_experiment = add_data_anonymization(experiment)

    # Implement access control
    access_controlled = implement_access_control(anonymized_experiment)

    # Add comprehensive audit logging
    audit_logged = add_audit_logging(access_controlled)

    # Ensure regulatory compliance
    compliant_experiment = ensure_regulatory_compliance(audit_logged)

    return SecureExperiment(
        experiment=compliant_experiment,
        security_config=security_config,
        validation=validate_experiment_security(compliant_experiment)
    )
```

#### 2. Reproducibility Security

```python
def validate_reproducibility_security(experiment: BaseExperiment, security_config: Dict[str, Any]) -> SecurityResult:
    """Validate security of reproducibility measures"""

    reproducibility_checks = {
        "environment_isolation": validate_environment_isolation(experiment),
        "dependency_verification": validate_dependency_verification(experiment),
        "execution_integrity": validate_execution_integrity(experiment),
        "result_authenticity": validate_result_authenticity(experiment)
    }

    return {
        "secure": all(reproducibility_checks.values()),
        "checks": reproducibility_checks,
        "vulnerabilities": [k for k, v in reproducibility_checks.items() if not v]
    }
```

## 🐛 Experiment Debugging & Troubleshooting

### Debug Configuration

```python
# Enable experiment debugging
debug_config = {
    "debug": True,
    "execution_debugging": True,
    "statistical_debugging": True,
    "reproducibility_debugging": True,
    "performance_monitoring": True
}

# Debug experiment development
debug_experiment_workflow(debug_config)
```

### Common Debugging Patterns

#### 1. Execution Debugging

```python
def debug_experiment_execution(experiment: BaseExperiment, execution_config: Dict[str, Any]) -> DebugResult:
    """Debug experiment execution issues"""

    # Debug execution pipeline
    pipeline_debug = debug_execution_pipeline(experiment)
    if not pipeline_debug['pipeline_functional']:
        return {"type": "pipeline", "issues": pipeline_debug['issues']}

    # Debug parallel execution
    parallel_debug = debug_parallel_execution(experiment)
    if not parallel_debug['parallel_functional']:
        return {"type": "parallel", "issues": parallel_debug['issues']}

    # Debug resource management
    resource_debug = debug_resource_management(experiment)
    if not resource_debug['resource_efficient']:
        return {"type": "resources", "issues": resource_debug['issues']}

    return {"status": "execution_ok"}

def debug_parallel_execution(experiment: BaseExperiment) -> Dict[str, Any]:
    """Debug parallel execution issues"""

    # Test with different worker counts
    worker_counts = [1, 2, 4, 8, 16]

    parallel_results = {}
    for n_workers in worker_counts:
        execution_config = {"parallel_workers": n_workers}
        result = test_parallel_execution(experiment, execution_config)
        parallel_results[n_workers] = result

    # Check for scaling issues
    scaling_issues = check_parallel_scaling(parallel_results)

    # Check for synchronization issues
    sync_issues = check_synchronization_issues(parallel_results)

    return {
        "parallel_functional": len(scaling_issues) == 0 and len(sync_issues) == 0,
        "scaling_issues": scaling_issues,
        "synchronization_issues": sync_issues,
        "recommendations": generate_parallel_optimization_recommendations(parallel_results)
    }
```

#### 2. Statistical Debugging

```python
def debug_statistical_validity(experiment: BaseExperiment) -> DebugResult:
    """Debug statistical validity issues"""

    # Check statistical design
    design_debug = debug_statistical_design(experiment)
    if not design_debug['design_valid']:
        return {"type": "design", "issues": design_debug['issues']}

    # Check power analysis
    power_debug = debug_power_analysis(experiment)
    if not power_debug['power_adequate']:
        return {"type": "power", "issues": power_debug['issues']}

    # Check analysis methods
    analysis_debug = debug_analysis_methods(experiment)
    if not analysis_debug['methods_valid']:
        return {"type": "analysis", "issues": analysis_debug['issues']}

    return {"status": "statistically_valid"}
```

## 🔄 Development Workflow

### Agent Development Process

1. **Experiment Framework Assessment**
   - Understand current experiment capabilities and limitations
   - Identify research methodology gaps and opportunities
   - Review existing experimental design and validation quality

2. **Scientific Method Planning**
   - Design comprehensive experiment framework architecture
   - Plan integration with research workflows and tools
   - Consider reproducibility and validation requirements

3. **Experiment System Implementation**
   - Implement robust experiment design and execution systems
   - Create comprehensive statistical analysis and validation
   - Develop automated reporting and reproducibility systems

4. **Scientific Validation Implementation**
   - Implement comprehensive testing for scientific rigor
   - Validate statistical methods and experimental design
   - Ensure reproducibility and reliability standards

5. **Integration and Research Validation**
   - Test integration with research workflows
   - Validate against research community standards
   - Update related documentation and educational materials

### Code Review Checklist

**Before submitting experiment code for review:**

- [ ] **Scientific Rigor**: All experiments follow rigorous scientific methodology
- [ ] **Statistical Validity**: All statistical methods are appropriate and validated
- [ ] **Reproducibility**: Complete reproducibility implemented and tested
- [ ] **Error Handling**: Robust error handling and recovery mechanisms
- [ ] **Documentation**: Comprehensive scientific and technical documentation
- [ ] **Validation**: Thorough validation including statistical and reproducibility testing
- [ ] **Standards Compliance**: Follows all scientific and development standards

## 📚 Learning Resources

### Experiment Development Resources

- **[Research Experiments AGENTS.md](AGENTS.md)**: Experiment development guidelines
- **[Experimental Design](https://example.com)**: Experimental design principles
- **[Statistical Methods](https://example.com)**: Statistical analysis techniques
- **[Reproducibility Standards](https://example.com)**: Scientific reproducibility guidelines

### Technical References

- **[Research Methodology](https://example.com)**: Research methods and experimental design
- **[Statistical Analysis](https://example.com)**: Advanced statistical analysis methods
- **[Scientific Computing](https://example.com)**: High-performance scientific computing
- **[Data Analysis](https://example.com)**: Data analysis and interpretation methods

### Related Components

Study these related components for integration patterns:

- **[Research Framework](../../)**: Research tools and scientific methods
- **[Analysis Tools](../../analysis/)**: Statistical analysis and validation tools
- **[Simulation Systems](../../simulations/)**: Multi-scale simulation frameworks
- **[Data Management](../../data_management/)**: Research data handling and validation

## 🎯 Success Metrics

### Experiment Quality Metrics

- **Scientific Rigor**: >95% compliance with scientific methodology standards
- **Statistical Validity**: 100% statistical validity for all experiments
- **Reproducibility Rate**: 100% successful reproduction of published experiments
- **Validation Coverage**: 100% comprehensive validation for all experiment types
- **User Adoption**: >80% adoption by research community

### Development Metrics

- **Implementation Speed**: Experiment systems implemented within 3 months
- **Quality Score**: Consistent high-quality scientific implementations
- **Validation Success**: All experiments pass comprehensive validation
- **Research Impact**: Contributions to Active Inference research community
- **Maintenance Efficiency**: Easy to update and extend experiment frameworks

---

**Research Experiments Framework**: Agent Guide | **Version**: 1.0.0 | **Last Updated**: October 2024

*"Active Inference for, with, by Generative AI"* - Advancing Active Inference research through rigorous experimental design, comprehensive validation, and reproducible scientific methods.