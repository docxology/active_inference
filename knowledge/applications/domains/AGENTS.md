# Applications Domains - Agent Development Guide

**Guidelines for AI agents working with domain applications in the Active Inference Knowledge Environment.**

## 🤖 Agent Role & Responsibilities

**What agents should do when working with domain applications:**

### Primary Responsibilities
- **Domain Implementation**: Develop Active Inference implementations for specific domains
- **Industry Integration**: Ensure implementations meet industry standards and requirements
- **Practical Validation**: Validate implementations against real-world benchmarks
- **Educational Content**: Create domain-specific educational materials and examples
- **Research Translation**: Translate theoretical concepts into practical applications

### Development Focus Areas
1. **AI Applications**: Implement Active Inference in artificial intelligence systems
2. **Robotics Integration**: Develop Active Inference solutions for robotic systems
3. **Neuroscience Models**: Create neural implementations and brain-inspired computing
4. **Psychological Applications**: Develop cognitive and behavioral models
5. **Educational Technology**: Build adaptive learning and assessment systems

## 🏗️ Architecture & Integration

### Domain Applications Architecture

**Understanding how domain applications fit into the larger ecosystem:**

```
Applications Layer
├── Domain Implementations (AI, Robotics, Neuroscience, Psychology, Education)
├── Industry Solutions (Enterprise, Healthcare, Manufacturing, Education)
├── Research Applications (Validated research implementations)
└── Educational Applications (Learning tools, tutorials, demonstrations)
```

### Integration Points

**Key integration points and dependencies:**

#### Upstream Components
- **Core Framework**: Active Inference mathematical and algorithmic foundations
- **Research Tools**: Research methods and validation frameworks
- **Knowledge Base**: Theoretical foundations and educational content
- **Implementation Templates**: Standardized implementation patterns

#### Downstream Components
- **Industry Deployment**: Real-world deployment and production systems
- **Educational Platforms**: Integration with learning management systems
- **Research Validation**: Validation against empirical data and benchmarks
- **Community Adoption**: Support for community development and adoption

#### External Systems
- **Industry Software**: Domain-specific software and development tools
- **Data Sources**: Real-world datasets and domain-specific repositories
- **Standards Bodies**: Industry standards and regulatory requirements
- **Research Communities**: Academic and research community resources

### Implementation Flow Patterns

```python
# Typical domain implementation workflow
domain_analysis → requirements → implementation → validation → deployment → maintenance
```

## 💻 Development Patterns

### Required Implementation Patterns

**All domain implementations must follow these patterns:**

#### 1. Domain Implementation Factory Pattern (PREFERRED)

```python
def create_domain_implementation(domain: str, implementation_config: Dict[str, Any]) -> DomainImplementation:
    """Create domain-specific implementation using factory pattern"""

    domain_factories = {
        'artificial_intelligence': create_ai_implementation,
        'robotics': create_robotics_implementation,
        'neuroscience': create_neuroscience_implementation,
        'psychology': create_psychology_implementation,
        'education': create_education_implementation,
        'engineering': create_engineering_implementation,
        'climate_science': create_climate_science_implementation
    }

    if domain not in domain_factories:
        raise ValueError(f"Unknown domain: {domain}")

    # Validate domain-specific configuration
    validate_domain_config(domain, implementation_config)

    # Create implementation with domain expertise
    implementation = domain_factories[domain](implementation_config)

    # Validate domain compliance
    validate_domain_compliance(implementation, domain)

    # Add domain-specific features
    enhanced_implementation = add_domain_specific_features(implementation, domain)

    return enhanced_implementation

def validate_domain_config(domain: str, config: Dict[str, Any]) -> None:
    """Validate domain-specific configuration"""

    # Domain-specific validation rules
    domain_validators = {
        'artificial_intelligence': validate_ai_config,
        'robotics': validate_robotics_config,
        'neuroscience': validate_neuroscience_config,
        'psychology': validate_psychology_config,
        'education': validate_education_config
    }

    if domain in domain_validators:
        domain_validators[domain](config)
    else:
        # Generic validation
        validate_generic_config(config)
```

#### 2. Domain Configuration Pattern (MANDATORY)

```python
from dataclasses import dataclass
from typing import Dict, Any, List, Optional

@dataclass
class DomainImplementationConfig:
    """Configuration for domain-specific implementations"""

    # Core domain settings
    domain_name: str
    application_type: str
    complexity_level: str
    target_industry: Optional[str] = None

    # Implementation requirements
    performance_targets: Dict[str, float]
    validation_standards: List[str]
    industry_compliance: List[str]
    safety_requirements: List[str]

    # Integration settings
    external_systems: List[str]
    data_sources: List[str]
    deployment_environment: str
    scalability_requirements: Dict[str, Any]

    # Quality settings
    testing_coverage: float = 95.0
    documentation_completeness: float = 100.0
    validation_rigor: str = "research_grade"

    def validate_configuration(self) -> List[str]:
        """Validate domain configuration"""
        errors = []

        # Check required fields
        required_fields = ['domain_name', 'application_type', 'performance_targets']
        for field in required_fields:
            if not getattr(self, field):
                errors.append(f"Missing required field: {field}")

        # Domain-specific validation
        domain_specific_errors = self.validate_domain_specific()
        errors.extend(domain_specific_errors)

        return errors

    def validate_domain_specific(self) -> List[str]:
        """Validate domain-specific requirements"""
        domain_validators = {
            'artificial_intelligence': self.validate_ai_requirements,
            'robotics': self.validate_robotics_requirements,
            'neuroscience': self.validate_neuroscience_requirements,
            'psychology': self.validate_psychology_requirements,
            'education': self.validate_education_requirements
        }

        if self.domain_name in domain_validators:
            return domain_validators[self.domain_name]()
        return []

    def validate_ai_requirements(self) -> List[str]:
        """Validate AI-specific requirements"""
        errors = []

        if 'real_time' in self.performance_targets and self.performance_targets['real_time'] < 0.1:
            errors.append("Real-time performance target too aggressive for AI domain")

        if not self.external_systems:
            errors.append("AI implementations should specify external system integration")

        return errors
```

#### 3. Implementation Validation Pattern (MANDATORY)

```python
def validate_domain_implementation(implementation: Any, domain: str, validation_config: Dict[str, Any]) -> ValidationResult:
    """Validate domain implementation against comprehensive criteria"""

    # Multi-level validation
    validation_levels = {
        "functional": validate_functional_correctness(implementation, domain),
        "performance": validate_performance_requirements(implementation, validation_config),
        "domain_compliance": validate_domain_compliance(implementation, domain),
        "industry_standards": validate_industry_standards(implementation, domain),
        "safety_safety": validate_safety_requirements(implementation, domain)
    }

    # Integration validation
    integration_validation = validate_system_integration(implementation, domain)

    # User acceptance validation
    user_validation = validate_user_acceptance(implementation, domain)

    # Generate comprehensive validation report
    validation_report = generate_comprehensive_validation_report(
        validation_levels, integration_validation, user_validation
    )

    return validation_report

def validate_domain_compliance(implementation: Any, domain: str) -> Dict[str, Any]:
    """Validate implementation complies with domain standards"""

    # Load domain standards
    domain_standards = load_domain_standards(domain)

    # Check implementation against standards
    compliance_checks = {}
    for standard in domain_standards:
        compliance_result = check_compliance(implementation, standard)
        compliance_checks[standard['name']] = compliance_result

    # Calculate overall compliance
    overall_compliance = calculate_overall_compliance(compliance_checks)

    return {
        "compliant": overall_compliance > 0.95,
        "compliance_score": overall_compliance,
        "checks": compliance_checks,
        "recommendations": generate_compliance_improvements(compliance_checks)
    }
```

## 🧪 Implementation Testing Standards

### Testing Categories (MANDATORY)

#### 1. Domain-Specific Testing
**Test implementation in domain-specific contexts:**

```python
def test_domain_specific_functionality():
    """Test domain-specific functionality and requirements"""
    # Test AI implementation
    ai_implementation = create_ai_implementation()
    ai_tests = run_ai_specific_tests(ai_implementation)

    assert ai_tests['learning_performance'] > 0.8, "AI learning performance insufficient"
    assert ai_tests['decision_quality'] > 0.85, "AI decision quality insufficient"

    # Test robotics implementation
    robotics_implementation = create_robotics_implementation()
    robotics_tests = run_robotics_specific_tests(robotics_implementation)

    assert robotics_tests['control_accuracy'] > 0.9, "Robotics control accuracy insufficient"
    assert robotics_tests['safety_compliance'] == 1.0, "Robotics safety compliance required"

def test_real_world_scenarios():
    """Test implementation with real-world domain scenarios"""
    # Load real-world test scenarios
    scenarios = load_domain_scenarios(domain)

    for scenario in scenarios:
        implementation = create_domain_implementation(domain, scenario['config'])
        result = run_scenario_test(implementation, scenario)

        # Validate against scenario requirements
        assert result['success'], f"Failed scenario: {scenario['name']}"
        assert result['performance'] >= scenario['performance_threshold']
```

#### 2. Industry Standards Testing
**Test compliance with industry standards and regulations:**

```python
def test_industry_standards_compliance():
    """Test compliance with industry standards"""
    implementation = create_domain_implementation()

    # Test against industry benchmarks
    benchmarks = load_industry_benchmarks(domain)
    benchmark_results = test_against_benchmarks(implementation, benchmarks)

    assert benchmark_results['overall_score'] > 0.9, "Below industry benchmark standards"

    # Test regulatory compliance
    regulatory_tests = run_regulatory_compliance_tests(implementation)
    assert regulatory_tests['compliant'], "Regulatory compliance required"

def test_safety_standards():
    """Test implementation meets safety standards"""
    safety_sensitive_domains = ['robotics', 'healthcare', 'automotive', 'aerospace']

    if domain in safety_sensitive_domains:
        safety_implementation = create_safety_critical_implementation()

        # Test safety requirements
        safety_tests = run_comprehensive_safety_tests(safety_implementation)
        assert safety_tests['safety_score'] == 1.0, "Safety compliance mandatory"

        # Test failure mode analysis
        fmea_results = perform_failure_mode_analysis(safety_implementation)
        assert fmea_results['acceptable_risk'], "Unacceptable safety risks identified"
```

#### 3. Performance and Scalability Testing
**Test performance characteristics and scalability:**

```python
def test_domain_performance_requirements():
    """Test implementation meets domain performance requirements"""
    implementation = create_domain_implementation()

    # Test computational performance
    performance_tests = run_performance_tests(implementation)
    assert performance_tests['meets_requirements'], "Performance requirements not met"

    # Test scalability
    scalability_tests = run_scalability_tests(implementation)
    assert scalability_tests['scales_adequately'], "Insufficient scalability"

    # Test resource efficiency
    efficiency_tests = run_efficiency_tests(implementation)
    assert efficiency_tests['resource_efficient'], "Resource efficiency requirements not met"

def test_integration_performance():
    """Test performance in integrated systems"""
    # Test with external systems
    external_systems = get_domain_external_systems()
    integration_tests = run_integration_performance_tests(implementation, external_systems)

    assert integration_tests['integration_successful'], "Integration performance inadequate"
    assert integration_tests['end_to_end_latency'] < max_latency, "End-to-end latency too high"
```

### Implementation Coverage Requirements

- **Domain Coverage**: All major domain application areas implemented
- **Use Case Coverage**: Comprehensive coverage of domain use cases
- **Industry Coverage**: Support for relevant industry applications
- **Validation Coverage**: All implementations validated against benchmarks
- **Documentation Coverage**: Complete documentation for all implementations

### Implementation Testing Commands

```bash
# Test all domain implementations
make test-domain-implementations

# Test specific domain
pytest knowledge/applications/domains/artificial_intelligence/tests/ -v

# Test industry compliance
pytest knowledge/applications/domains/tests/test_compliance.py -v

# Test performance requirements
pytest knowledge/applications/domains/tests/test_performance.py -v

# Validate against benchmarks
python knowledge/applications/domains/validate_benchmarks.py
```

## 📖 Implementation Documentation Standards

### Documentation Requirements (MANDATORY)

#### 1. Domain-Specific Documentation
**All domain implementations must have comprehensive domain-specific documentation:**

```python
def document_domain_implementation(implementation: Any, domain_config: Dict[str, Any]) -> str:
    """Document domain implementation with comprehensive detail"""

    domain_documentation = {
        "domain_overview": document_domain_overview(implementation, domain_config),
        "implementation_architecture": document_implementation_architecture(implementation, domain_config),
        "domain_specific_features": document_domain_specific_features(implementation, domain_config),
        "use_cases": document_use_cases(implementation, domain_config),
        "validation_results": document_validation_results(implementation, domain_config),
        "deployment_guidance": document_deployment_guidance(implementation, domain_config)
    }

    return format_domain_documentation(domain_documentation)

def document_domain_specific_features(implementation: Any, config: Dict[str, Any]) -> str:
    """Document domain-specific features and capabilities"""

    # Extract domain-specific features
    domain_features = extract_domain_features(implementation, config['domain'])

    # Document each feature comprehensively
    feature_documentation = []
    for feature in domain_features:
        feature_doc = document_feature(feature, config)
        feature_documentation.append(feature_doc)

    return format_feature_documentation(feature_documentation)
```

#### 2. Industry Standards Documentation
**Implementations must document compliance with industry standards:**

```python
def document_industry_compliance(implementation: Any, domain: str) -> str:
    """Document compliance with industry standards"""

    # Load relevant standards
    industry_standards = load_industry_standards(domain)

    # Check compliance for each standard
    compliance_documentation = {}
    for standard in industry_standards:
        compliance_result = check_implementation_compliance(implementation, standard)
        compliance_documentation[standard['name']] = compliance_result

    # Generate compliance report
    compliance_report = generate_compliance_report(compliance_documentation)

    # Document compliance evidence
    evidence_documentation = document_compliance_evidence(compliance_documentation)

    return format_compliance_documentation(compliance_report, evidence_documentation)
```

#### 3. Educational Documentation
**All implementations must support educational objectives:**

```python
def document_educational_content(implementation: Any, educational_config: Dict[str, Any]) -> str:
    """Document educational aspects of domain implementation"""

    educational_documentation = {
        "learning_objectives": define_implementation_learning_objectives(implementation, educational_config),
        "conceptual_understanding": explain_conceptual_foundation(implementation, educational_config),
        "practical_applications": demonstrate_practical_applications(implementation, educational_config),
        "hands_on_exercises": create_hands_on_exercises(implementation, educational_config),
        "assessment_materials": create_assessment_materials(implementation, educational_config)
    }

    return format_educational_documentation(educational_documentation)
```

## 🚀 Performance Optimization

### Implementation Performance Requirements

**Domain implementations must meet these performance standards:**

- **Domain-Specific Performance**: Meet domain-specific performance benchmarks
- **Scalability**: Scale appropriately with problem size and complexity
- **Resource Efficiency**: Efficient use of computational resources
- **Real-Time Capability**: Support real-time operation where required
- **Integration Performance**: Perform well in integrated systems

### Optimization Techniques

#### 1. Domain-Specific Optimization

```python
def optimize_domain_implementation(implementation: Any, domain: str, optimization_config: Dict[str, Any]) -> OptimizedImplementation:
    """Optimize implementation for domain-specific requirements"""

    # Domain-specific optimization strategies
    domain_optimizers = {
        'artificial_intelligence': optimize_ai_performance,
        'robotics': optimize_robotics_performance,
        'neuroscience': optimize_neural_performance,
        'psychology': optimize_cognitive_performance,
        'education': optimize_educational_performance
    }

    if domain in domain_optimizers:
        optimizer = domain_optimizers[domain]
        optimized = optimizer(implementation, optimization_config)
    else:
        optimized = optimize_general_performance(implementation, optimization_config)

    # Validate optimization maintains functionality
    validation = validate_optimization(optimized, implementation, domain)
    if not validation['functional']:
        raise OptimizationError(f"Optimization compromised functionality: {validation['issues']}")

    return optimized

def optimize_ai_performance(implementation: Any, config: Dict[str, Any]) -> OptimizedImplementation:
    """Optimize AI implementation for performance"""

    # Apply AI-specific optimizations
    optimized = implementation

    # Optimize inference algorithms
    optimized = optimize_inference_algorithms(optimized, config)

    # Optimize learning procedures
    optimized = optimize_learning_procedures(optimized, config)

    # Optimize decision-making
    optimized = optimize_decision_making(optimized, config)

    # Add performance monitoring
    optimized = add_performance_monitoring(optimized)

    return optimized
```

#### 2. Integration Optimization

```python
def optimize_system_integration(implementation: Any, integration_config: Dict[str, Any]) -> IntegratedImplementation:
    """Optimize implementation for system integration"""

    # Analyze integration requirements
    integration_requirements = analyze_integration_requirements(integration_config)

    # Optimize data interfaces
    optimized_interfaces = optimize_data_interfaces(implementation, integration_requirements)

    # Optimize communication protocols
    optimized_communication = optimize_communication_protocols(optimized_interfaces, integration_requirements)

    # Optimize performance in integrated context
    optimized_performance = optimize_integrated_performance(optimized_communication, integration_requirements)

    # Validate integration
    integration_validation = validate_system_integration(optimized_performance, integration_config)

    return IntegratedImplementation(
        implementation=optimized_performance,
        integration_requirements=integration_requirements,
        validation=integration_validation
    )
```

## 🔒 Implementation Security Standards

### Security Requirements (MANDATORY)

#### 1. Domain-Specific Security

```python
def validate_domain_security(implementation: Any, domain: str, security_config: Dict[str, Any]) -> SecurityResult:
    """Validate domain-specific security requirements"""

    # Domain-specific security checks
    domain_security_checks = {
        "data_privacy": validate_data_privacy_compliance(implementation, domain),
        "access_control": validate_access_control(implementation, domain),
        "audit_logging": validate_audit_logging(implementation, domain),
        "safety_systems": validate_safety_systems(implementation, domain)
    }

    # Industry-specific security validation
    industry_security = validate_industry_security(implementation, domain, security_config)

    return {
        "secure": all(domain_security_checks.values()) and industry_security['secure'],
        "checks": domain_security_checks,
        "industry_validation": industry_security,
        "risks": identify_security_risks(domain_security_checks, industry_security)
    }

def validate_safety_systems(implementation: Any, domain: str) -> bool:
    """Validate safety systems for safety-critical domains"""

    safety_critical_domains = ['robotics', 'healthcare', 'automotive', 'aerospace']

    if domain in safety_critical_domains:
        # Comprehensive safety validation
        safety_validation = perform_comprehensive_safety_validation(implementation)

        # Failure mode analysis
        fmea = perform_failure_mode_analysis(implementation)

        # Risk assessment
        risk_assessment = assess_safety_risks(fmea)

        return safety_validation['safe'] and risk_assessment['acceptable']
    else:
        return True  # Non-safety-critical domains have minimal safety requirements
```

#### 2. Industry Compliance Security

```python
def validate_industry_compliance_security(implementation: Any, domain: str) -> Dict[str, Any]:
    """Validate security compliance with industry standards"""

    # Load industry security standards
    security_standards = load_industry_security_standards(domain)

    # Validate against each standard
    compliance_results = {}
    for standard in security_standards:
        compliance_result = validate_against_security_standard(implementation, standard)
        compliance_results[standard['name']] = compliance_result

    # Overall security compliance
    overall_compliance = calculate_security_compliance_score(compliance_results)

    return {
        "compliant": overall_compliance > 0.95,
        "compliance_score": overall_compliance,
        "standards": compliance_results,
        "remediation": generate_security_remediation_plan(compliance_results)
    }
```

## 🐛 Implementation Debugging & Troubleshooting

### Debug Configuration

```python
# Enable domain implementation debugging
debug_config = {
    "debug": True,
    "domain_validation": True,
    "performance_monitoring": True,
    "integration_testing": True,
    "compliance_checking": True
}

# Debug domain implementation development
debug_domain_implementation_workflow(debug_config)
```

### Common Debugging Patterns

#### 1. Domain-Specific Debugging

```python
def debug_domain_implementation(implementation: Any, domain: str) -> DebugResult:
    """Debug domain-specific implementation issues"""

    # Domain-specific debugging
    domain_debugs = {
        'artificial_intelligence': debug_ai_implementation,
        'robotics': debug_robotics_implementation,
        'neuroscience': debug_neural_implementation,
        'psychology': debug_psychological_implementation,
        'education': debug_educational_implementation
    }

    if domain in domain_debugs:
        return domain_debugs[domain](implementation)
    else:
        return debug_generic_implementation(implementation)

def debug_ai_implementation(implementation: Any) -> DebugResult:
    """Debug AI implementation issues"""

    # Check learning performance
    learning_debug = debug_learning_performance(implementation)
    if not learning_debug['performing']:
        return {"type": "learning", "issues": learning_debug['issues']}

    # Check decision quality
    decision_debug = debug_decision_quality(implementation)
    if not decision_debug['adequate']:
        return {"type": "decision", "issues": decision_debug['issues']}

    # Check integration
    integration_debug = debug_system_integration(implementation)
    if not integration_debug['integrated']:
        return {"type": "integration", "issues": integration_debug['issues']}

    return {"status": "ai_implementation_ok"}
```

#### 2. Performance Debugging

```python
def debug_implementation_performance(implementation: Any, performance_config: Dict[str, Any]) -> DebugResult:
    """Debug implementation performance issues"""

    # Profile computational performance
    performance_profile = profile_computational_performance(implementation)

    if performance_profile['slow']:
        return {"type": "computation", "profile": performance_profile}

    # Profile memory usage
    memory_profile = profile_memory_usage(implementation)

    if memory_profile['memory_issues']:
        return {"type": "memory", "profile": memory_profile}

    # Profile integration performance
    integration_profile = profile_integration_performance(implementation)

    if integration_profile['integration_issues']:
        return {"type": "integration", "profile": integration_profile}

    return {"status": "performance_ok"}
```

## 🔄 Development Workflow

### Agent Development Process

1. **Domain Analysis and Assessment**
   - Understand current domain implementation state
   - Identify domain-specific requirements and opportunities
   - Review industry standards and best practices

2. **Implementation Architecture Planning**
   - Design domain-specific implementation architecture
   - Plan integration with existing platform components
   - Consider domain-specific constraints and requirements

3. **Domain Implementation Development**
   - Implement using established domain patterns
   - Ensure compliance with industry standards
   - Create comprehensive validation and testing

4. **Domain Validation and Testing**
   - Test against domain-specific benchmarks
   - Validate industry compliance and standards
   - Ensure performance meets domain requirements

5. **Integration and Deployment**
   - Test integration with platform systems
   - Validate deployment in target environments
   - Update documentation and educational materials

### Code Review Checklist

**Before submitting domain implementation for review:**

- [ ] **Domain Expertise**: Implementation demonstrates deep domain understanding
- [ ] **Industry Standards**: Implementation meets relevant industry standards
- [ ] **Validation Rigor**: Comprehensive validation against domain benchmarks
- [ ] **Practical Utility**: Implementation solves real domain problems
- [ ] **Performance**: Meets domain-specific performance requirements
- [ ] **Integration**: Integrates properly with platform and external systems
- [ ] **Documentation**: Complete documentation including domain-specific guidance
- [ ] **Standards Compliance**: Follows all development and domain standards

## 📚 Learning Resources

### Domain Implementation Resources

- **[Applications AGENTS.md](../../../applications/AGENTS.md)**: Applications development guidelines
- **[Domain Templates](../../../applications/templates/README.md)**: Domain implementation templates
- **[Industry Standards](https://example.com)**: Industry standards and best practices
- **[Domain Research](https://example.com)**: Domain-specific research and literature

### Technical References

- **[Domain-Specific Libraries](https://example.com)**: Specialized libraries for each domain
- **[Integration Patterns](https://example.com)**: System integration methodologies
- **[Performance Optimization](https://example.com)**: Domain-specific optimization techniques
- **[Validation Methods](https://example.com)**: Domain validation and testing approaches

### Related Components

Study these related components for integration patterns:

- **[Core Framework](../../../src/active_inference/)**: Core Active Inference implementation
- **[Research Tools](../../../research/)**: Research validation and benchmarking
- **[Knowledge Base](../../../knowledge/)**: Educational content and theoretical foundations
- **[Platform Integration](../../../platform/)**: Platform services and infrastructure

## 🎯 Success Metrics

### Implementation Quality Metrics

- **Domain Accuracy**: >95% accuracy against domain benchmarks
- **Industry Compliance**: 100% compliance with industry standards
- **Performance Efficiency**: Meeting or exceeding domain performance requirements
- **Integration Success**: Seamless integration with platform systems
- **User Adoption**: Successful adoption in target domain applications

### Development Metrics

- **Implementation Speed**: Domain implementations completed within 3 months
- **Quality Score**: Consistent high-quality domain-specific implementations
- **Validation Success**: All implementations pass comprehensive domain validation
- **Industry Recognition**: Recognition by domain experts and industry leaders
- **Maintenance Efficiency**: Easy to update and maintain domain implementations

---

**Applications Domains**: Agent Guide | **Version**: 1.0.0 | **Last Updated**: October 2024

*"Active Inference for, with, by Generative AI"* - Creating domain-specific Active Inference implementations that solve real-world problems, meet industry standards, and advance practical applications across diverse fields.