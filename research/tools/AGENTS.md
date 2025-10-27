# Research Tools - Agent Development Guide

**Guidelines for AI agents developing research tools in the Active Inference Knowledge Environment.**

## 🤖 Agent Role & Responsibilities

**What agents should do when developing research tools:**

### Primary Responsibilities
- **Research Tool Development**: Create tools that support Active Inference research workflows
- **Workflow Automation**: Develop automation for research processes and experiments
- **Orchestration Systems**: Build thin orchestration components for research pipelines
- **Testing Frameworks**: Create testing tools for research validation and reproducibility
- **Development Utilities**: Build utilities that support research software development

### Development Focus Areas
1. **Research Workflow Support**: Build tools that enhance research productivity
2. **Reproducibility**: Ensure all tools support reproducible research practices
3. **Integration**: Create seamless integration with research frameworks
4. **Validation**: Develop comprehensive validation and testing tools
5. **Performance**: Optimize tools for research workloads and data processing

## 🏗️ Architecture & Integration

### Research Tools Architecture

**Understanding how research tools fit into the research ecosystem:**

```
Research Layer
├── Experiment Layer (experiments/, simulations/)
├── Tool Layer ← Research Tools
├── Analysis Layer (analysis/, benchmarks/)
└── Integration Layer (platform/, applications/)
```

### Integration Points

**Research tools integrate with multiple platform components:**

#### Upstream Components
- **Research Framework**: Provides core research functionality and APIs
- **Knowledge Base**: Supplies domain knowledge and research context
- **Experiment Definitions**: Provides experiment specifications and parameters

#### Downstream Components
- **Analysis Tools**: Consumes processed data and results
- **Visualization Systems**: Provides data for research visualization
- **Platform Services**: Exposes tools through web interfaces and APIs

#### External Systems
- **File Systems**: Manages research data and result storage
- **Job Schedulers**: Integrates with HPC and distributed computing
- **Data Repositories**: Connects with academic data sharing platforms
- **Publication Systems**: Supports research publication workflows

### Research Workflow Integration

```python
# Typical research workflow supported by tools
research_question → experiment_design → tool_selection → execution → analysis → validation → publication
```

## 💻 Development Patterns

### Required Implementation Patterns

**All research tools must follow these patterns:**

#### 1. Research Tool Factory Pattern (PREFERRED)
```python
def create_research_tool(tool_type: str, config: Dict[str, Any]) -> ResearchTool:
    """Create research tool using factory pattern with validation"""

    # Research tool registry
    research_tools = {
        'experiment_runner': ExperimentRunner,
        'workflow_manager': WorkflowManager,
        'data_processor': DataProcessor,
        'result_validator': ResultValidator,
        'benchmark_runner': BenchmarkRunner
    }

    if tool_type not in research_tools:
        raise ResearchError(f"Unknown research tool type: {tool_type}")

    # Validate research context
    validate_research_context(config)

    # Create tool with research validation
    tool = research_tools[tool_type](config)

    # Validate tool functionality
    validate_tool_capabilities(tool)

    return tool

def validate_research_context(config: Dict[str, Any]) -> None:
    """Validate research context and requirements"""
    required_fields = {'research_domain', 'experiment_path', 'output_dir'}

    for field in required_fields:
        if field not in config:
            raise ResearchError(f"Missing required research field: {field}")

    # Validate research domain compatibility
    if config['research_domain'] not in SUPPORTED_DOMAINS:
        raise ResearchError(f"Unsupported research domain: {config['research_domain']}")
```

#### 2. Research Configuration Pattern (MANDATORY)
```python
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from enum import Enum

class ResearchDomain(Enum):
    """Supported research domains"""
    ACTIVE_INFERENCE = "active_inference"
    FREE_ENERGY = "free_energy"
    BAYESIAN_INFERENCE = "bayesian_inference"
    NEURAL_NETWORKS = "neural_networks"
    COGNITIVE_SCIENCE = "cognitive_science"

@dataclass
class ResearchToolConfig:
    """Research tool configuration with validation"""

    # Required research fields
    research_domain: ResearchDomain
    experiment_path: str
    output_dir: str

    # Optional research settings
    validation_enabled: bool = True
    reproducibility_mode: bool = True
    performance_monitoring: bool = True

    # Advanced research settings
    parallel_execution: Dict[str, Any] = field(default_factory=lambda: {
        "enabled": False,
        "max_workers": 4,
        "timeout": 3600
    })

    # Research workflow settings
    workflow: Dict[str, Any] = field(default_factory=lambda: {
        "checkpointing": True,
        "auto_save": True,
        "notification_channels": ["console"]
    })

    def validate(self) -> List[str]:
        """Validate research configuration"""
        errors = []

        # Validate research domain
        if not isinstance(self.research_domain, ResearchDomain):
            errors.append("research_domain must be a ResearchDomain enum")

        # Validate paths
        if not self.experiment_path or not self.experiment_path.strip():
            errors.append("experiment_path cannot be empty")

        if not self.output_dir or not self.output_dir.strip():
            errors.append("output_dir cannot be empty")

        # Validate parallel execution settings
        if self.parallel_execution["enabled"]:
            if self.parallel_execution["max_workers"] < 1:
                errors.append("max_workers must be positive when parallel execution is enabled")

        return errors

    def get_research_context(self) -> Dict[str, Any]:
        """Get research context for tool initialization"""
        return {
            "domain": self.research_domain.value,
            "experiment_base": self.experiment_path,
            "output_base": self.output_dir,
            "validation": self.validation_enabled,
            "reproducibility": self.reproducibility_mode
        }
```

#### 3. Research Error Handling Pattern (MANDATORY)
```python
import logging
from typing import Callable, Any, Dict
from contextlib import contextmanager

logger = logging.getLogger(__name__)

class ResearchError(Exception):
    """Base exception for research tool errors"""
    pass

class ExperimentError(ResearchError):
    """Experiment execution errors"""
    pass

class ValidationError(ResearchError):
    """Research validation errors"""
    pass

@contextmanager
def research_execution_context(operation: str, config: Dict[str, Any]):
    """Context manager for research operation execution"""

    research_context = {
        "operation": operation,
        "config": config,
        "start_time": time.time(),
        "status": "starting"
    }

    try:
        logger.info(f"Starting research operation: {operation}", extra={
            "research_context": research_context
        })

        research_context["status"] = "running"
        yield research_context

        research_context["status"] = "completed"
        research_context["end_time"] = time.time()
        research_context["duration"] = research_context["end_time"] - research_context["start_time"]

        logger.info(f"Research operation completed: {operation}", extra={
            "research_context": research_context
        })

    except ValidationError as e:
        research_context["status"] = "validation_failed"
        research_context["error"] = str(e)
        logger.error(f"Research validation failed: {operation}", extra={
            "research_context": research_context
        })
        raise

    except ExperimentError as e:
        research_context["status"] = "experiment_failed"
        research_context["error"] = str(e)
        logger.error(f"Research experiment failed: {operation}", extra={
            "research_context": research_context
        })
        raise

    except Exception as e:
        research_context["status"] = "error"
        research_context["error"] = str(e)
        research_context["traceback"] = traceback.format_exc()
        logger.error(f"Research operation error: {operation}", extra={
            "research_context": research_context
        })
        raise ResearchError(f"Research operation failed: {operation}") from e

def execute_research_operation(operation_name: str, func: Callable, *args, **kwargs) -> Any:
    """Execute research operation with comprehensive error handling"""
    return research_execution_context(operation_name, kwargs.get('config', {}))
```

## 🧪 Testing Standards

### Research Testing Categories (MANDATORY)

#### 1. Research Workflow Tests (`tests/test_research_workflows.py`)
**Test complete research workflows and pipelines:**
```python
def test_experiment_workflow():
    """Test complete experiment workflow from setup to completion"""
    config = ResearchToolConfig(
        research_domain=ResearchDomain.ACTIVE_INFERENCE,
        experiment_path="tests/fixtures/experiments/test_experiment.yaml",
        output_dir="tests/fixtures/results/"
    )

    # Create and configure research tool
    tool = create_research_tool('experiment_runner', config.to_dict())

    # Execute complete workflow
    results = tool.execute_all_experiments()

    # Validate workflow completion
    assert results is not None
    assert len(results) > 0
    assert all("status" in result for result in results.values())

def test_reproducibility_workflow():
    """Test research reproducibility across multiple runs"""
    config = ResearchToolConfig(
        research_domain=ResearchDomain.FREE_ENERGY,
        experiment_path="tests/fixtures/experiments/reproducible_test.yaml",
        output_dir="tests/fixtures/results/",
        reproducibility_mode=True
    )

    # Run experiment multiple times
    tool = create_research_tool('experiment_runner', config.to_dict())

    results1 = tool.execute_experiment("test_reproducibility")
    results2 = tool.execute_experiment("test_reproducibility")

    # Validate reproducible results
    assert results1 == results2
    assert tool.get_reproducibility_report() == "PASSED"
```

#### 2. Research Tool Integration Tests (`tests/test_tool_integration.py`)
**Test research tool interactions and data flow:**
```python
def test_automation_orchestration_integration():
    """Test integration between automation and orchestration tools"""
    config = ResearchToolConfig(
        research_domain=ResearchDomain.BAYESIAN_INFERENCE,
        experiment_path="tests/fixtures/experiments/",
        output_dir="tests/fixtures/results/"
    )

    # Create integrated research system
    automation = create_research_tool('experiment_runner', config.to_dict())
    orchestration = create_research_tool('workflow_manager', config.to_dict())

    # Test integrated workflow
    workflow_config = {
        "name": "integrated_research_pipeline",
        "stages": ["setup", "execution", "analysis", "validation"]
    }

    success = orchestration.orchestrate_research_pipeline(workflow_config)
    assert success is True

    # Verify automation tool was used correctly
    results = automation.get_execution_results()
    assert len(results) > 0

def test_validation_integration():
    """Test integration with validation and testing tools"""
    config = ResearchToolConfig(
        research_domain=ResearchDomain.NEURAL_NETWORKS,
        experiment_path="tests/fixtures/experiments/",
        output_dir="tests/fixtures/results/",
        validation_enabled=True
    )

    # Create tools with validation
    runner = create_research_tool('experiment_runner', config.to_dict())
    validator = create_research_tool('result_validator', config.to_dict())

    # Execute and validate
    results = runner.execute_experiments()
    validation_report = validator.validate_results(results)

    # Verify integration
    assert validation_report["overall_status"] in ["valid", "failed"]
    assert "validations" in validation_report
```

#### 3. Research Performance Tests (`tests/test_performance.py`)
**Test research tool performance under various conditions:**
```python
def test_large_dataset_processing():
    """Test performance with large research datasets"""
    config = ResearchToolConfig(
        research_domain=ResearchDomain.COGNITIVE_SCIENCE,
        experiment_path="tests/fixtures/large_dataset/",
        output_dir="tests/fixtures/large_results/",
        parallel_execution={"enabled": True, "max_workers": 8}
    )

    tool = create_research_tool('data_processor', config.to_dict())

    # Performance benchmarking
    start_time = time.time()
    results = tool.process_large_dataset("large_dataset.csv", "processed_output.csv")
    end_time = time.time()

    # Validate performance requirements
    processing_time = end_time - start_time
    assert processing_time < 300  # Must complete within 5 minutes

    # Validate results
    assert results["records_processed"] > 100000
    assert results["processing_rate"] > 1000  # records per second

def test_concurrent_research_workflows():
    """Test performance under concurrent research workflows"""
    config = ResearchToolConfig(
        research_domain=ResearchDomain.ACTIVE_INFERENCE,
        experiment_path="tests/fixtures/concurrent_experiments/",
        output_dir="tests/fixtures/concurrent_results/",
        parallel_execution={"enabled": True, "max_workers": 4}
    )

    # Simulate concurrent research workflows
    workflows = [
        {"name": f"workflow_{i}", "experiments": [f"exp_{j}" for j in range(5)]}
        for i in range(3)
    ]

    start_time = time.time()

    # Execute concurrent workflows
    results = []
    for workflow in workflows:
        tool = create_research_tool('experiment_runner', config.to_dict())
        result = tool.execute_workflow(workflow)
        results.append(result)

    end_time = time.time()

    # Validate concurrent execution performance
    total_time = end_time - start_time
    expected_sequential_time = 30 * 3  # 30s per workflow sequentially

    assert total_time < expected_sequential_time  # Must be faster than sequential
    assert all(result["status"] == "completed" for result in results)
```

### Research Test Coverage Requirements

- **Research Workflows**: 100% coverage of research processes
- **Error Scenarios**: 100% coverage of research failure modes
- **Integration Points**: 95% coverage of tool interactions
- **Performance Paths**: 90% coverage of performance-critical code
- **Reproducibility**: 100% coverage of reproducibility features

### Research Testing Commands

```bash
# Run all research tool tests
make test-research-tools

# Run research workflow tests
pytest research/tools/tests/test_research_workflows.py -v

# Run integration tests
pytest research/tools/tests/test_tool_integration.py -v --tb=short

# Run performance tests
pytest research/tools/tests/test_performance.py -v

# Check research test coverage
pytest research/tools/ --cov=research/tools/ --cov-report=html --cov-fail-under=95
```

## 📖 Documentation Standards

### Research Documentation Requirements (MANDATORY)

#### 1. Research Context Documentation
**Every research tool must document its research context:**
```python
def research_context_example():
    """
    Research Context: Active Inference Experiment Automation

    This tool automates Active Inference experiments by executing predefined
    experimental protocols and validating results against theoretical predictions.

    Research Domain: Active Inference
    Scientific Context: Enables systematic exploration of Active Inference
                      hypotheses through automated experimentation
    Validation Method: Statistical comparison with theoretical predictions
    Reproducibility: Full experiment state capture and reproduction support
    """
    pass
```

#### 2. Research Workflow Documentation
**All research workflows must be documented:**
```python
class DocumentedResearchWorkflow:
    """
    Documented Research Workflow: Bayesian Model Comparison

    This workflow automates the process of comparing different Bayesian models
    within the Active Inference framework, including model specification,
    parameter estimation, and model comparison using information criteria.

    Workflow Steps:
    1. Model Specification: Define competing models with different complexity
    2. Parameter Estimation: Use variational inference for parameter fitting
    3. Model Comparison: Calculate AIC/BIC for model selection
    4. Validation: Cross-validation and robustness analysis
    5. Reporting: Generate comprehensive comparison reports

    Research Applications:
    - Model selection in cognitive modeling
    - Architecture optimization for neural networks
    - Policy comparison in decision-making tasks

    Expected Outcomes:
    - Model ranking by information criteria
    - Parameter uncertainty quantification
    - Robustness analysis across different datasets
    """
    pass
```

#### 3. Reproducibility Documentation
**All tools must document reproducibility features:**
```python
def reproducibility_features():
    """
    Reproducibility Features:

    This research tool implements comprehensive reproducibility measures:

    1. State Capture: Complete experiment state serialization
    2. Random Seed Management: Controlled randomness for reproducibility
    3. Environment Tracking: System and dependency version tracking
    4. Result Validation: Automated result verification
    5. Report Generation: Comprehensive reproducibility reports

    Usage:
        config = {
            "reproducibility_mode": True,
            "random_seed": 42,
            "environment_capture": True
        }
    """
    pass
```

## 🚀 Performance Optimization

### Research Performance Requirements

**Research tools must meet scientific computing performance standards:**

- **Experiment Execution**: <10 minutes for typical research experiments
- **Data Processing**: <1GB/minute for research data processing
- **Workflow Orchestration**: <5% overhead for workflow management
- **Memory Efficiency**: <2GB memory usage for typical research workloads

### Research-Specific Optimizations

#### 1. Parallel Research Execution
```python
class ParallelExperimentRunner:
    """Parallel experiment execution for research workflows"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.executor = ThreadPoolExecutor(max_workers=config["max_workers"])
        self.semaphore = Semaphore(config["max_concurrent_experiments"])

    async def run_parallel_experiments(self, experiments: List[Dict]) -> List[Dict]:
        """Execute experiments in parallel with resource management"""
        tasks = []

        for experiment in experiments:
            # Acquire semaphore to limit concurrent experiments
            await self.semaphore.acquire()

            # Create research task
            task = asyncio.create_task(
                self._execute_single_experiment_with_semaphore(experiment)
            )
            tasks.append(task)

        # Execute all experiments
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                self.log_experiment_failure(experiments[i], result)
                processed_results.append({"error": str(result)})
            else:
                processed_results.append(result)

        return processed_results

    async def _execute_single_experiment_with_semaphore(self, experiment: Dict) -> Dict:
        """Execute single experiment and release semaphore"""
        try:
            result = await self._execute_research_experiment(experiment)
            return result
        finally:
            self.semaphore.release()
```

#### 2. Research Data Streaming
```python
class StreamingResearchProcessor:
    """Streaming processor for large research datasets"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.chunk_size = config.get("chunk_size", 10000)

    def process_large_research_dataset(self, input_path: str, output_path: str) -> Dict[str, Any]:
        """Process large research dataset using streaming"""
        total_processed = 0
        results = []

        with open(input_path, 'r') as input_file, open(output_path, 'w') as output_file:
            while True:
                # Read chunk of research data
                chunk = self._read_research_data_chunk(input_file)
                if not chunk:
                    break

                # Process research data chunk
                processed_chunk = self._process_research_chunk(chunk)
                results.append(processed_chunk)

                # Write processed results
                self._write_research_results(output_file, processed_chunk)

                total_processed += len(chunk)

                # Progress logging for research workflow
                if total_processed % 100000 == 0:
                    logger.info(f"Processed {total_processed} research records")

        return {
            "total_processed": total_processed,
            "chunks_processed": len(results),
            "output_path": output_path,
            "processing_time": time.time() - start_time
        }
```

## 🔒 Research Security Standards

### Research Data Security (MANDATORY)

#### 1. Research Data Protection
```python
def validate_research_data_access(self, user: str, data_path: str, operation: str) -> bool:
    """Validate research data access permissions"""
    # Check research project permissions
    project_permissions = self.get_research_project_permissions(user)

    if operation not in project_permissions["allowed_operations"]:
        self.log_security_event("unauthorized_data_access", {
            "user": user,
            "data_path": data_path,
            "operation": operation,
            "reason": "Operation not permitted for research project"
        })
        return False

    # Validate data sensitivity level
    data_classification = self.get_data_classification(data_path)

    if data_classification == "sensitive" and not project_permissions["sensitive_data_access"]:
        self.log_security_event("sensitive_data_access_denied", {
            "user": user,
            "data_path": data_path,
            "classification": data_classification
        })
        return False

    return True
```

#### 2. Research Audit Logging
```python
def log_research_activity(self, activity: str, details: Dict[str, Any], user: str) -> None:
    """Log research activities for audit and reproducibility"""

    research_audit_event = {
        "timestamp": datetime.utcnow().isoformat(),
        "user": user,
        "activity": activity,
        "research_context": details,
        "tool": "research_tools",
        "reproducibility_hash": self.generate_reproducibility_hash(details)
    }

    # Log to research audit trail
    self.research_logger.info(f"Research activity: {activity}", extra={
        "research_audit": research_audit_event
    })

    # Store in research database
    self.research_db.store_audit_event(research_audit_event)

def generate_reproducibility_hash(self, details: Dict[str, Any]) -> str:
    """Generate hash for research reproducibility tracking"""
    # Create deterministic representation of research activity
    research_state = {
        "parameters": details.get("parameters", {}),
        "data_sources": details.get("data_sources", []),
        "algorithms": details.get("algorithms", []),
        "timestamp": details.get("timestamp", "")
    }

    # Generate reproducible hash
    import hashlib
    state_json = json.dumps(research_state, sort_keys=True)
    return hashlib.sha256(state_json.encode()).hexdigest()[:16]
```

## 🔄 Development Workflow

### Research Tool Development Process

1. **Research Requirement Analysis**
   - Understand research domain and scientific context
   - Identify research workflow gaps and needs
   - Analyze integration requirements with research frameworks

2. **Research Architecture Design**
   - Design tool following research software best practices
   - Plan comprehensive testing for research validation
   - Consider reproducibility and scientific computing requirements

3. **Research-Focused TDD**
   - Write tests covering research workflows and edge cases
   - Implement minimal functionality for research use cases
   - Validate against research requirements and scientific accuracy

4. **Research Implementation**
   - Follow research software development standards
   - Implement comprehensive research data validation
   - Add reproducibility features and audit trails

5. **Research Quality Assurance**
   - Validate research workflow integration
   - Test with real research data and scenarios
   - Verify reproducibility and scientific accuracy

6. **Research Integration**
   - Integrate with research frameworks and tools
   - Validate performance with research workloads
   - Update research documentation and examples

### Research Tool Review Checklist

**Before submitting research tools for review:**

- [ ] **Research Context**: Tool addresses real research needs
- [ ] **Scientific Accuracy**: Implementation follows scientific principles
- [ ] **Reproducibility**: Full reproducibility support implemented
- [ ] **Research Testing**: Comprehensive research workflow testing
- [ ] **Performance**: Meets research computing performance requirements
- [ ] **Integration**: Seamless integration with research frameworks
- [ ] **Documentation**: Clear research context and usage examples
- [ ] **Standards**: Follows research software development standards

## 📚 Learning Resources

### Research Tool Resources

- **[Research Framework](../../README.md)**: Core research functionality
- **[Research Experiments](../../experiments/README.md)**: Experiment definitions
- **[Research Analysis](../../analysis/README.md)**: Analysis methodologies
- **[.cursorrules](../../../.cursorrules)**: Development standards

### Scientific Computing References

- **[Research Software Engineering](https://researchsoftware.org)**: Best practices
- **[Reproducible Research](https://reproducibleresearch.net)**: Reproducibility guidelines
- **[Scientific Python](https://scientific-python.org)**: Scientific computing standards
- **[Research Data Management](https://rdm.org)**: Data management best practices

### Research Domain Knowledge

Study these research areas for tool development:

- **[Active Inference Research](https://activeinference.org)**: Domain expertise
- **[Bayesian Methods](https://bayesian.org)**: Statistical foundations
- **[Neural Computation](https://neuralcomputation.org)**: Computational neuroscience
- **[Scientific Computing](https://scipy.org)**: Computing methodologies

## 🎯 Success Metrics

### Research Impact Metrics

- **Research Productivity**: Tools accelerate research workflows
- **Reproducibility Rate**: High reproducibility in research processes
- **Research Quality**: Improved research validation and accuracy
- **Integration Success**: Seamless adoption by research community
- **Scientific Impact**: Contributions to Active Inference research

### Development Metrics

- **Research Tool Quality**: Maintains high scientific computing standards
- **Performance**: Meets research computing performance requirements
- **Reliability**: Zero failures in research workflows
- **Maintainability**: Clean, well-documented research code
- **Community Adoption**: Research community usage and feedback

---

**Research Tools**: Version 1.0.0 | **Last Updated**: October 2024

*"Active Inference for, with, by Generative AI"* - Advancing research through intelligent tool development and comprehensive research support.
