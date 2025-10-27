# Domain Applications - Agent Development Guide

**Guidelines for AI agents developing domain-specific Active Inference applications.**

## 🤖 Agent Role & Responsibilities

**What agents should do when developing domain applications:**

### Primary Responsibilities
- **Domain Application Development**: Create practical Active Inference applications for specific domains
- **Cross-Domain Integration**: Ensure applications integrate effectively across domains
- **Real-World Validation**: Validate applications against real-world requirements
- **Domain Expertise Integration**: Incorporate domain-specific knowledge and constraints
- **Performance Optimization**: Optimize applications for domain-specific performance requirements

### Development Focus Areas
1. **Domain Analysis**: Understand domain-specific problems and requirements
2. **Active Inference Mapping**: Map domain problems to Active Inference solutions
3. **Implementation Design**: Design robust, scalable domain applications
4. **Validation and Testing**: Comprehensive validation against domain requirements
5. **Integration**: Seamless integration with platform and other domains

## 🏗️ Architecture & Integration

### Domain Applications Architecture

**Understanding how domain applications fit into the platform:**

```
Platform Layer
├── Knowledge Layer (foundations/, mathematics/, implementations/)
├── Application Layer ← Domain Applications
├── Research Layer (experiments/, analysis/, simulations/)
└── Integration Layer (platform/, visualization/, tools/)
```

### Integration Points

**Domain applications integrate with multiple platform components:**

#### Upstream Components
- **Knowledge Repository**: Provides theoretical foundations and domain knowledge
- **Mathematical Framework**: Supplies mathematical tools and derivations
- **Implementation Templates**: Offers reusable implementation patterns
- **Research Tools**: Enables empirical validation and testing

#### Downstream Components
- **Platform Services**: Leverages platform infrastructure for deployment
- **Visualization Systems**: Provides domain-specific visualization capabilities
- **Application Templates**: Uses implementation frameworks and patterns
- **Development Tools**: Supports application development workflows

#### External Systems
- **Domain-Specific Tools**: Integrates with field-specific software and libraries
- **Data Sources**: Connects with domain-specific datasets and repositories
- **Industry Standards**: Follows domain-specific protocols and standards
- **Academic Resources**: Links with domain research communities

### Domain Application Data Flow

```python
# Typical domain application workflow
domain_problem → analysis → active_inference_mapping → implementation → validation → deployment
domain_data → preprocessing → active_inference_modeling → application_logic → domain_output
research_question → domain_context → ai_solution_design → testing → real_world_validation
```

## 💻 Development Patterns

### Required Implementation Patterns

**All domain applications must follow these patterns:**

#### 1. Domain Application Factory Pattern (PREFERRED)
```python
def create_domain_application(domain: str, application_type: str, config: Dict[str, Any]) -> DomainApplication:
    """Create domain-specific application using factory pattern"""

    # Domain application registry
    domain_applications = {
        'artificial_intelligence': {
            'alignment_safety': AIAlignmentSystem,
            'generative_control': AIGenerativeControl,
            'decision_making': AIDecisionSystem
        },
        'neuroscience': {
            'neural_control': NeuralControlSystem,
            'perception_modeling': PerceptionModelingSystem,
            'cognitive_architecture': CognitiveArchitecture
        },
        'robotics': {
            'navigation': AutonomousNavigationSystem,
            'manipulation': RoboticManipulationSystem,
            'interaction': HumanRobotInteraction
        }
    }

    if domain not in domain_applications or application_type not in domain_applications[domain]:
        raise DomainError(f"Unknown domain application: {domain}.{application_type}")

    # Validate domain context
    validate_domain_context(domain, config)

    # Create application with domain validation
    application = domain_applications[domain][application_type](config)

    # Validate domain compatibility
    validate_domain_compatibility(application, domain)

    return application

def validate_domain_context(domain: str, config: Dict[str, Any]) -> None:
    """Validate domain-specific context and requirements"""
    domain_validators = {
        'artificial_intelligence': validate_ai_context,
        'neuroscience': validate_neuroscience_context,
        'robotics': validate_robotics_context,
        'psychology': validate_psychology_context,
        'engineering': validate_engineering_context
    }

    if domain not in domain_validators:
        raise DomainError(f"Unsupported domain: {domain}")

    domain_validators[domain](config)
```

#### 2. Domain Configuration Pattern (MANDATORY)
```python
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from enum import Enum

class DomainType(Enum):
    """Supported application domains"""
    ARTIFICIAL_INTELLIGENCE = "artificial_intelligence"
    NEUROSCIENCE = "neuroscience"
    ROBOTICS = "robotics"
    PSYCHOLOGY = "psychology"
    ENGINEERING = "engineering"
    EDUCATION = "education"
    ECONOMICS = "economics"
    CLIMATE_SCIENCE = "climate_science"

@dataclass
class DomainApplicationConfig:
    """Domain application configuration with validation"""

    # Required domain fields
    domain: DomainType
    application_type: str
    problem_statement: str

    # Optional domain settings
    domain_constraints: Dict[str, Any] = field(default_factory=dict)
    performance_requirements: Dict[str, Any] = field(default_factory=dict)
    validation_criteria: Dict[str, Any] = field(default_factory=dict)

    # Active Inference settings
    active_inference: Dict[str, Any] = field(default_factory=lambda: {
        "precision": 16.0,
        "learning_rate": 0.01,
        "policy_horizon": 10,
        "model_complexity": "adaptive"
    })

    # Integration settings
    integration: Dict[str, Any] = field(default_factory=lambda: {
        "platform_integration": True,
        "external_apis": [],
        "data_sources": [],
        "deployment_target": "development"
    })

    def validate(self) -> List[str]:
        """Validate domain application configuration"""
        errors = []

        # Validate domain type
        if not isinstance(self.domain, DomainType):
            errors.append("domain must be a DomainType enum")

        # Validate application type
        if not self.application_type or not self.application_type.strip():
            errors.append("application_type cannot be empty")

        # Validate problem statement
        if not self.problem_statement or len(self.problem_statement.strip()) < 10:
            errors.append("problem_statement must be at least 10 characters")

        # Domain-specific validation
        if self.domain == DomainType.ROBOTICS:
            if "safety_constraints" not in self.domain_constraints:
                errors.append("robotics applications require safety_constraints")

        if self.domain == DomainType.ARTIFICIAL_INTELLIGENCE:
            if "ethical_bounds" not in self.active_inference:
                errors.append("AI applications require ethical_bounds in active_inference config")

        return errors

    def get_domain_context(self) -> Dict[str, Any]:
        """Get domain context for application initialization"""
        return {
            "domain": self.domain.value,
            "application_type": self.application_type,
            "problem": self.problem_statement,
            "constraints": self.domain_constraints,
            "requirements": self.performance_requirements,
            "ai_settings": self.active_inference,
            "integration": self.integration
        }
```

#### 3. Domain Error Handling Pattern (MANDATORY)
```python
import logging
from typing import Callable, Any, Dict, Optional
from contextlib import contextmanager

logger = logging.getLogger(__name__)

class DomainError(Exception):
    """Base exception for domain application errors"""
    pass

class DomainValidationError(DomainError):
    """Domain-specific validation errors"""
    pass

class DomainIntegrationError(DomainError):
    """Domain integration errors"""
    pass

@contextmanager
def domain_execution_context(domain: str, operation: str, config: Dict[str, Any]):
    """Context manager for domain application execution"""

    domain_context = {
        "domain": domain,
        "operation": operation,
        "config": config,
        "start_time": time.time(),
        "status": "starting",
        "domain_specific_metrics": {}
    }

    try:
        logger.info(f"Starting domain operation: {domain}.{operation}", extra={
            "domain_context": domain_context
        })

        domain_context["status"] = "running"
        yield domain_context

        domain_context["status"] = "completed"
        domain_context["end_time"] = time.time()
        domain_context["duration"] = domain_context["end_time"] - domain_context["start_time"]

        logger.info(f"Domain operation completed: {domain}.{operation}", extra={
            "domain_context": domain_context
        })

    except DomainValidationError as e:
        domain_context["status"] = "domain_validation_failed"
        domain_context["error"] = str(e)
        logger.error(f"Domain validation failed: {domain}.{operation}", extra={
            "domain_context": domain_context
        })
        raise

    except DomainIntegrationError as e:
        domain_context["status"] = "domain_integration_failed"
        domain_context["error"] = str(e)
        logger.error(f"Domain integration failed: {domain}.{operation}", extra={
            "domain_context": domain_context
        })
        raise

    except Exception as e:
        domain_context["status"] = "domain_error"
        domain_context["error"] = str(e)
        domain_context["traceback"] = traceback.format_exc()
        logger.error(f"Domain operation error: {domain}.{operation}", extra={
            "domain_context": domain_context
        })
        raise DomainError(f"Domain operation failed: {domain}.{operation}") from e

def execute_domain_operation(domain: str, operation: str, func: Callable, config: Dict[str, Any], **kwargs) -> Any:
    """Execute domain operation with comprehensive error handling"""
    with domain_execution_context(domain, operation, config) as context:
        return func(**kwargs)
```

## 🧪 Testing Standards

### Domain Testing Categories (MANDATORY)

#### 1. Domain Compatibility Tests (`tests/test_domain_compatibility.py`)
**Test domain-specific functionality and constraints:**
```python
def test_ai_domain_constraints():
    """Test AI domain specific constraints and requirements"""
    config = DomainApplicationConfig(
        domain=DomainType.ARTIFICIAL_INTELLIGENCE,
        application_type="alignment_safety",
        problem_statement="AI alignment with human values using Active Inference",
        active_inference={"ethical_bounds": True}
    )

    # Test AI-specific validation
    application = create_domain_application(
        config.domain.value,
        config.application_type,
        config.to_dict()
    )

    # Validate AI domain requirements
    assert application.validate_ethical_constraints()
    assert application.get_domain_metrics()["alignment_score"] >= 0.8

def test_robotics_domain_safety():
    """Test robotics domain safety constraints"""
    config = DomainApplicationConfig(
        domain=DomainType.ROBOTICS,
        application_type="autonomous_navigation",
        problem_statement="Safe autonomous navigation in dynamic environments",
        domain_constraints={"safety_constraints": {"collision_avoidance": True}}
    )

    # Test robotics-specific safety validation
    application = create_domain_application(
        config.domain.value,
        config.application_type,
        config.to_dict()
    )

    # Validate safety constraints
    safety_validation = application.validate_safety_constraints()
    assert safety_validation["collision_avoidance"] == "PASSED"
    assert safety_validation["emergency_stop"] == "IMPLEMENTED"
```

#### 2. Cross-Domain Integration Tests (`tests/test_cross_domain_integration.py`)
**Test integration between multiple domain applications:**
```python
def test_neuroscience_robotics_integration():
    """Test integration between neuroscience and robotics domains"""
    # Neuroscience domain application
    neuro_config = DomainApplicationConfig(
        domain=DomainType.NEUROSCIENCE,
        application_type="neural_control",
        problem_statement="Neural control of robotic systems",
        active_inference={"precision": 16.0}
    )

    # Robotics domain application
    robotics_config = DomainApplicationConfig(
        domain=DomainType.ROBOTICS,
        application_type="motor_control",
        problem_statement="Motor control based on neural signals",
        domain_constraints={"real_time_processing": True}
    )

    # Create integrated system
    neuro_app = create_domain_application(neuro_config.domain.value, neuro_config.application_type, neuro_config.to_dict())
    robotics_app = create_domain_application(robotics_config.domain.value, robotics_config.application_type, robotics_config.to_dict())

    # Test cross-domain integration
    neural_signals = neuro_app.generate_neural_signals()
    motor_response = robotics_app.process_neural_input(neural_signals)

    # Validate integration
    assert motor_response["neural_integration"] == "SUCCESS"
    assert motor_response["real_time_processing"] < 10  # ms

def test_multi_domain_workflow():
    """Test complete multi-domain workflow"""
    # Multi-domain configuration
    workflow_config = {
        "domains": [
            {"domain": "neuroscience", "application": "perception"},
            {"domain": "artificial_intelligence", "application": "decision_making"},
            {"domain": "robotics", "application": "execution"}
        ],
        "integration_mode": "sequential"
    }

    # Execute multi-domain workflow
    integrator = MultiDomainWorkflowIntegrator(workflow_config)

    # Test workflow execution
    result = integrator.execute_workflow()
    assert result["overall_status"] == "COMPLETED"
    assert all(domain_result["status"] == "SUCCESS" for domain_result in result["domain_results"])
```

#### 3. Domain Performance Tests (`tests/test_domain_performance.py`)
**Test domain-specific performance requirements:**
```python
def test_robotics_real_time_performance():
    """Test real-time performance requirements for robotics domain"""
    config = DomainApplicationConfig(
        domain=DomainType.ROBOTICS,
        application_type="autonomous_navigation",
        problem_statement="Real-time autonomous navigation",
        performance_requirements={
            "response_time": "<50ms",
            "throughput": ">20Hz",
            "reliability": ">99.9%"
        }
    )

    application = create_domain_application(
        config.domain.value,
        config.application_type,
        config.to_dict()
    )

    # Performance benchmarking
    import time

    response_times = []
    for _ in range(100):
        start_time = time.time()
        result = application.process_sensor_input(test_sensor_data)
        end_time = time.time()

        response_times.append((end_time - start_time) * 1000)  # ms

    # Validate performance requirements
    avg_response_time = sum(response_times) / len(response_times)
    max_response_time = max(response_times)
    reliability = calculate_reliability(response_times, 50)  # 50ms threshold

    assert avg_response_time < 25, f"Average response time {avg_response_time}ms exceeds 25ms requirement"
    assert max_response_time < 50, f"Max response time {max_response_time}ms exceeds 50ms requirement"
    assert reliability > 0.999, f"Reliability {reliability} below 99.9% requirement"

def test_ai_domain_scalability():
    """Test scalability requirements for AI domain"""
    config = DomainApplicationConfig(
        domain=DomainType.ARTIFICIAL_INTELLIGENCE,
        application_type="generative_control",
        problem_statement="Control of large-scale generative AI systems",
        performance_requirements={
            "max_model_size": "100B parameters",
            "throughput": ">1000 tokens/sec",
            "memory_efficiency": "<100GB RAM"
        }
    )

    application = create_domain_application(
        config.domain.value,
        config.application_type,
        config.to_dict()
    )

    # Scalability testing
    large_model_config = {
        "model_size": "50B",
        "context_length": 4096,
        "batch_size": 32
    }

    # Test with large model
    start_time = time.time()
    start_memory = get_memory_usage()

    result = application.control_generative_model(large_model_config, test_input)

    end_time = time.time()
    end_memory = get_memory_usage()

    processing_time = end_time - start_time
    memory_used = end_memory - start_memory

    # Validate scalability requirements
    assert processing_time < 5, f"Processing time {processing_time}s too slow for AI domain"
    assert memory_used < 100 * 1024 * 1024 * 1024, f"Memory usage {memory_used}GB exceeds 100GB limit"
    assert result["control_effectiveness"] > 0.9
```

### Domain Test Coverage Requirements

- **Domain Functionality**: 100% coverage of domain-specific features
- **Cross-Domain Integration**: 95% coverage of integration points
- **Performance Requirements**: 100% coverage of domain performance criteria
- **Error Scenarios**: 100% coverage of domain-specific error conditions
- **Real-World Validation**: 90% coverage of real-world usage scenarios

### Domain Testing Commands

```bash
# Run all domain application tests
make test-domain-applications

# Run domain-specific tests
pytest knowledge/applications/domains/tests/test_ai_domain.py -v
pytest knowledge/applications/domains/tests/test_robotics_domain.py -v

# Run cross-domain integration tests
pytest knowledge/applications/domains/tests/test_cross_domain_integration.py -v

# Run domain performance tests
pytest knowledge/applications/domains/tests/test_domain_performance.py -v

# Check domain test coverage
pytest knowledge/applications/domains/ --cov=knowledge/applications/domains/ --cov-report=html --cov-fail-under=95
```

## 📖 Documentation Standards

### Domain Documentation Requirements (MANDATORY)

#### 1. Domain Context Documentation
**Every domain application must document its domain context:**
```python
def document_domain_context():
    """
    Domain Context: Robotics - Autonomous Navigation

    This application implements autonomous navigation for robotic systems using
    Active Inference principles for sensorimotor control and path planning.

    Domain: Robotics
    Problem Space: Autonomous navigation in dynamic environments
    Active Inference Approach: Sensorimotor integration with predictive control
    Technical Foundation: Bayesian filtering with Active Inference policy selection
    Real-World Validation: Tested in warehouse and outdoor navigation scenarios

    Key Features:
    - Real-time sensor fusion and processing
    - Predictive path planning with uncertainty handling
    - Collision avoidance with safety constraints
    - Adaptive behavior based on environmental feedback

    Performance Requirements:
    - Response time: <50ms for navigation decisions
    - Path efficiency: >90% optimal path selection
    - Safety: Zero collision incidents in test scenarios
    """
    pass
```

#### 2. Domain Implementation Documentation
**All domain implementations must be documented:**
```python
class DocumentedDomainApplication:
    """
    Documented Domain Implementation: AI Alignment System

    This class implements an Active Inference-based system for aligning AI
    behavior with human values and ethical constraints.

    Implementation Details:
    - Generative Model: Human preference model with uncertainty quantification
    - Policy Selection: Multi-objective optimization with ethical constraints
    - Learning: Online adaptation to human feedback and preference drift
    - Safety: Real-time safety monitoring and constraint enforcement

    Domain Integration:
    - Ethics Framework: Integration with AI ethics and safety standards
    - Human-AI Interface: Natural language preference specification
    - Transparency: Explainable AI decisions with uncertainty quantification
    - Compliance: Regulatory compliance and audit trail generation

    Usage Patterns:
    1. Preference Elicitation: Gather human preferences through interaction
    2. Model Training: Learn preference model using Active Inference
    3. Policy Generation: Generate aligned policies with safety constraints
    4. Deployment: Deploy with continuous monitoring and adaptation
    """
    pass
```

#### 3. Domain Validation Documentation
**All domain applications must document validation approaches:**
```python
def document_domain_validation():
    """
    Domain Validation: Neuroscience - Neural Control

    This application has been validated through multiple neuroscience research
    protocols and clinical validation studies.

    Validation Methods:
    1. Neural Data Validation: Comparison with empirical neural recordings
    2. Behavioral Validation: Motor behavior analysis in animal models
    3. Computational Validation: Theoretical consistency with neural dynamics
    4. Clinical Validation: Human motor control and rehabilitation studies

    Validation Metrics:
    - Neural Prediction Accuracy: >85% correlation with recorded neural activity
    - Motor Control Precision: <5% error in target reaching tasks
    - Computational Efficiency: Real-time processing capability
    - Clinical Safety: Zero adverse events in human trials

    Validation Datasets:
    - Neural Recording Datasets: Public neuroscience data repositories
    - Motor Control Datasets: Human and animal movement databases
    - Clinical Trial Data: Rehabilitation and neural disorder studies
    """
    pass
```

## 🚀 Performance Optimization

### Domain-Specific Performance Requirements

**Domain applications must meet field-specific performance standards:**

#### Robotics Domain
- **Real-Time Processing**: <10ms sensor-to-action latency
- **Control Precision**: <1mm positional accuracy
- **Safety Response**: <5ms emergency stop activation
- **Environmental Adaptation**: <100ms to environmental changes

#### Artificial Intelligence Domain
- **Model Training**: <1 hour for typical model sizes
- **Inference Speed**: <100ms for decision-making
- **Scalability**: Support for models up to 100B parameters
- **Ethical Compliance**: Real-time ethical constraint validation

#### Neuroscience Domain
- **Neural Processing**: <50ms for neural signal processing
- **Data Throughput**: >1GB/minute neural data processing
- **Temporal Precision**: <1ms temporal resolution
- **Real-Time Analysis**: Continuous neural activity monitoring

### Domain Optimization Techniques

#### 1. Domain-Specific Caching
```python
class DomainSpecificCache:
    """Intelligent caching for domain-specific operations"""

    def __init__(self, domain: str):
        self.domain = domain
        self.cache_configs = {
            'robotics': {'max_size': 10000, 'ttl': 60},      # 1 minute for fast-changing robot data
            'artificial_intelligence': {'max_size': 100000, 'ttl': 3600},  # 1 hour for AI models
            'neuroscience': {'max_size': 1000, 'ttl': 30},   # 30 seconds for neural data
        }

        self.cache = self._create_cache(self.cache_configs[domain])

    def get_domain_data(self, key: str) -> Any:
        """Get domain-specific data with intelligent caching"""
        # Domain-specific cache key generation
        domain_key = f"{self.domain}:{key}"

        # Check cache with domain-specific TTL
        if domain_key in self.cache:
            return self.cache[domain_key]

        # Generate domain-specific data
        data = self._generate_domain_data(key)

        # Cache with domain-appropriate TTL
        self.cache[domain_key] = data
        return data

    def _create_cache(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Create cache with domain-specific configuration"""
        # Domain-specific cache implementation
        return {}
```

#### 2. Domain Adaptive Processing
```python
class DomainAdaptiveProcessor:
    """Adaptive processing optimized for domain requirements"""

    def __init__(self, domain: str):
        self.domain = domain
        self.adaptation_strategies = self._load_domain_strategies(domain)

    def process_domain_data(self, data: Any, context: Dict[str, Any]) -> Any:
        """Process data with domain-adaptive strategies"""

        # Select domain-appropriate processing strategy
        strategy = self._select_processing_strategy(context)

        # Adaptive parameter tuning
        parameters = self._tune_parameters_for_domain(strategy, context)

        # Domain-optimized processing
        result = self._execute_domain_processing(data, strategy, parameters)

        # Adaptive learning from results
        self._learn_from_domain_feedback(result, context)

        return result

    def _select_processing_strategy(self, context: Dict[str, Any]) -> str:
        """Select optimal processing strategy for domain context"""
        domain_strategies = {
            'robotics': ['real_time', 'safety_first', 'precision_optimized'],
            'artificial_intelligence': ['accuracy_first', 'scalability_optimized', 'ethical_compliant'],
            'neuroscience': ['temporal_precision', 'noise_robust', 'biologically_plausible']
        }

        # Domain-specific strategy selection logic
        return 'default_strategy'
```

## 🔒 Domain Security Standards

### Domain-Specific Security (MANDATORY)

#### 1. Domain Data Protection
```python
def validate_domain_data_access(self, user: str, domain: str, data_type: str, operation: str) -> bool:
    """Validate domain-specific data access permissions"""

    # Domain-specific permission requirements
    domain_permissions = {
        'neuroscience': {'sensitive_medical_data': True, 'requires_ethics_approval': True},
        'artificial_intelligence': {'requires_safety_review': True, 'audit_required': True},
        'robotics': {'requires_safety_certification': True, 'real_time_monitoring': True},
        'psychology': {'requires_informed_consent': True, 'data_anonymization': True}
    }

    # Check domain-specific requirements
    if domain in domain_permissions:
        requirements = domain_permissions[domain]

        if requirements.get('requires_ethics_approval') and not self.has_ethics_approval(user, domain):
            self.log_security_event("ethics_approval_required", {
                "user": user, "domain": domain, "data_type": data_type
            })
            return False

        if requirements.get('requires_safety_certification') and not self.has_safety_certification(user, domain):
            self.log_security_event("safety_certification_required", {
                "user": user, "domain": domain, "data_type": data_type
            })
            return False

    return True
```

#### 2. Domain Audit Logging
```python
def log_domain_activity(self, domain: str, activity: str, details: Dict[str, Any], user: str) -> None:
    """Log domain-specific activities for audit and compliance"""

    # Domain-specific audit requirements
    domain_audit_config = {
        'neuroscience': {'hipaa_compliance': True, 'clinical_trial_tracking': True},
        'artificial_intelligence': {'bias_monitoring': True, 'fairness_audit': True},
        'robotics': {'safety_incident_tracking': True, 'performance_monitoring': True},
        'psychology': {'participant_protection': True, 'consent_verification': True}
    }

    audit_event = {
        "timestamp": datetime.utcnow().isoformat(),
        "domain": domain,
        "user": user,
        "activity": activity,
        "domain_context": details,
        "compliance_requirements": domain_audit_config.get(domain, {}),
        "reproducibility_hash": self.generate_domain_reproducibility_hash(domain, details)
    }

    # Log to domain-specific audit trail
    self.domain_logger.info(f"Domain activity: {domain}.{activity}", extra={
        "domain_audit": audit_event
    })

    # Domain-specific compliance logging
    if domain_audit_config.get(domain, {}).get('hipaa_compliance'):
        self.hipaa_logger.info(f"HIPAA activity: {domain}.{activity}", extra={
            "hipaa_audit": audit_event
        })

def generate_domain_reproducibility_hash(self, domain: str, details: Dict[str, Any]) -> str:
    """Generate domain-specific reproducibility hash"""
    # Domain-specific reproducibility criteria
    domain_criteria = {
        'neuroscience': ['subject_id', 'experiment_protocol', 'data_collection_method'],
        'robotics': ['environment_config', 'control_parameters', 'safety_constraints'],
        'artificial_intelligence': ['model_architecture', 'training_data', 'ethical_constraints']
    }

    # Create domain-specific state representation
    criteria = domain_criteria.get(domain, ['parameters', 'timestamp'])
    domain_state = {k: v for k, v in details.items() if k in criteria}

    # Generate reproducible hash
    import hashlib
    state_json = json.dumps(domain_state, sort_keys=True)
    return hashlib.sha256(f"{domain}:{state_json}".encode()).hexdigest()[:16]
```

## 🔄 Development Workflow

### Domain Application Development Process

1. **Domain Expertise Acquisition**
   - Study domain-specific literature and research
   - Understand domain problems and requirements
   - Identify domain experts and stakeholders

2. **Domain Analysis and Design**
   - Analyze domain-specific problems and constraints
   - Design Active Inference solutions for domain needs
   - Plan comprehensive validation strategies

3. **Domain-Focused Implementation**
   - Implement domain-specific Active Inference applications
   - Integrate with domain tools and standards
   - Validate against domain requirements and constraints

4. **Domain Validation and Testing**
   - Test with domain-specific data and scenarios
   - Validate performance against domain standards
   - Verify integration with domain ecosystems

5. **Domain Integration and Deployment**
   - Integrate with platform and other domains
   - Deploy with domain-appropriate configurations
   - Monitor domain-specific performance and usage

### Domain Application Review Checklist

**Before submitting domain applications for review:**

- [ ] **Domain Accuracy**: Implementation accurately reflects domain requirements
- [ ] **Active Inference Correctness**: Proper application of Active Inference principles
- [ ] **Domain Integration**: Seamless integration with domain tools and standards
- [ ] **Performance Validation**: Meets domain-specific performance requirements
- [ ] **Real-World Validation**: Tested with realistic domain scenarios
- [ ] **Documentation**: Comprehensive domain-specific documentation
- [ ] **Security Compliance**: Meets domain security and ethical requirements
- [ ] **Standards Adherence**: Follows domain best practices and conventions

## 📚 Learning Resources

### Domain-Specific Resources

- **[AI Domain](artificial_intelligence/README.md)**: Artificial intelligence applications
- **[Neuroscience Domain](neuroscience/README.md)**: Neural and cognitive applications
- **[Robotics Domain](robotics/README.md)**: Robotic and autonomous systems
- **[Psychology Domain](psychology/README.md)**: Psychological applications
- **[.cursorrules](../../../.cursorrules)**: Development standards

### Domain Research References

- **[Active Inference Applications](https://activeinference.org/applications)**: Domain applications
- **[Domain Research Communities](https://domainresearch.org)**: Field-specific research
- **[Industry Standards](https://industry-standards.org)**: Domain standards and protocols
- **[Academic Publications](https://academic-pubs.org)**: Research publications by domain

### Technical Domain References

Study these technical areas for domain application development:

- **[Domain-Specific Computing](https://domain-computing.org)**: Specialized computing techniques
- **[Real-Time Systems](https://realtime-systems.org)**: For robotics and control domains
- **[Machine Learning Ethics](https://ml-ethics.org)**: For AI domain applications
- **[Neural Engineering](https://neural-engineering.org)**: For neuroscience domain

## 🎯 Success Metrics

### Domain Impact Metrics

- **Domain Adoption**: Applications adopted by domain practitioners
- **Real-World Impact**: Measurable improvements in domain outcomes
- **Research Contribution**: Contributions to domain research and literature
- **Integration Success**: Successful integration with domain ecosystems
- **Performance Excellence**: Meeting domain-specific performance standards

### Development Metrics

- **Domain Application Quality**: High-quality, domain-accurate implementations
- **Performance Standards**: Meeting domain performance requirements
- **Integration Success**: Seamless domain ecosystem integration
- **Documentation Quality**: Clear, domain-specific documentation
- **Community Adoption**: Domain community usage and validation

---

**Domain Applications**: Version 1.0.0 | **Last Updated**: October 2024

*"Active Inference for, with, by Generative AI"* - Advancing domain applications through comprehensive domain expertise and practical implementation excellence.
