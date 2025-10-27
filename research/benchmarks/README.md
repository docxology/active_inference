# Research Benchmarks Suite

Standardized evaluation metrics and benchmarking tools for Active Inference models. Provides comprehensive performance evaluation, comparison frameworks, and standardized testing protocols for all stages of the research process.

## Overview

The Benchmarks Suite provides a comprehensive evaluation framework for Active Inference research, supporting researchers from initial model development through publication. The module includes specialized benchmarking tools for different research roles and stages.

## Core Components

### 🏆 Performance Metrics
- **Accuracy Metrics**: Classification and prediction accuracy
- **Information Metrics**: Free energy, KL divergence, information efficiency
- **Computational Metrics**: Convergence time, computational cost
- **Robustness Metrics**: Stability, sensitivity, generalization
- **Statistical Metrics**: Confidence intervals, significance tests

### 🏅 Benchmarking Framework
- **Standard Benchmarks**: Established benchmarks for comparison
- **Custom Benchmarks**: User-defined evaluation protocols
- **Comparative Analysis**: Side-by-side model comparisons
- **Statistical Validation**: Rigorous performance validation
- **Reproducibility**: Complete benchmark reproducibility

### 📊 Evaluation Tools
- **Automated Evaluation**: Batch evaluation pipelines
- **Performance Tracking**: Historical performance monitoring
- **Baseline Management**: Baseline performance establishment
- **Improvement Tracking**: Performance improvement analysis
- **Report Generation**: Comprehensive evaluation reports

## Research Roles and Functions

### 🧑‍🎓 Intern Level
```python
from active_inference.research.benchmarks import InternBenchmarks

# Basic model evaluation
benchmarks = InternBenchmarks()
basic_metrics = benchmarks.evaluate_model(model, test_data)
simple_comparison = benchmarks.compare_models([model_a, model_b])
```

**Features:**
- Basic performance metrics
- Simple model comparisons
- Guided evaluation process
- Tutorial explanations
- Error checking and validation

### 🎓 PhD Student Level
```python
from active_inference.research.benchmarks import PhDBenchmarks

# Advanced performance evaluation
benchmarks = PhDBenchmarks()
comprehensive_metrics = benchmarks.run_full_evaluation(model, datasets)
statistical_validation = benchmarks.validate_performance(metrics)
```

**Features:**
- Comprehensive evaluation suites
- Statistical validation methods
- Advanced comparison frameworks
- Performance optimization tools
- Publication-quality reporting

### 🧑‍🔬 Grant Application Level
```python
from active_inference.research.benchmarks import GrantBenchmarks

# Power analysis and study planning
benchmarks = GrantBenchmarks()
power_analysis = benchmarks.compute_evaluation_power(sample_sizes)
optimal_design = benchmarks.optimize_evaluation_design(constraints)
```

**Features:**
- Statistical power analysis
- Sample size optimization
- Evaluation cost modeling
- Risk assessment tools
- Grant proposal metrics

### 📝 Publication Level
```python
from active_inference.research.benchmarks import PublicationBenchmarks

# Publication-ready evaluation
benchmarks = PublicationBenchmarks()
comprehensive_benchmark = benchmarks.run_publication_benchmark(model_suite)
report = benchmarks.generate_publication_report(benchmark, format='latex')
```

**Features:**
- Publication-standard evaluation
- Multiple comparison corrections
- Reviewer-ready reports
- Statistical standard compliance
- Citation management

## Usage Examples

### Basic Model Evaluation
```python
from active_inference.research.benchmarks import BenchmarkSuite

# Initialize benchmarking suite
suite = BenchmarkSuite()

# Register baseline performance
baseline_metrics = PerformanceMetrics(
    accuracy=0.85,
    free_energy=1.2,
    convergence_time=10.5
)
suite.register_baseline('baseline_model', baseline_metrics)

# Evaluate new model
def evaluate_my_model(test_data):
    # Model evaluation logic here
    return {
        'accuracy': 0.92,
        'precision': 0.89,
        'recall': 0.91,
        'free_energy': 0.8,
        'convergence_time': 8.2
    }

new_metrics = suite.evaluate_model('my_model', evaluate_my_model, test_data)
```

### Comprehensive Benchmarking
```python
from active_inference.research.benchmarks import ComprehensiveBenchmark

# Run comprehensive evaluation
benchmark = ComprehensiveBenchmark()

# Multi-dataset evaluation
datasets = ['dataset1', 'dataset2', 'dataset3']
results = benchmark.evaluate_across_datasets(model, datasets)

# Statistical validation
validation = benchmark.validate_results(results, alpha=0.05)

# Generate comprehensive report
report = benchmark.generate_report(results, validation)
```

### Model Comparison Study
```python
from active_inference.research.benchmarks import ComparisonBenchmark

# Compare multiple models
models = [
    {'name': 'Model A', 'function': model_a_eval},
    {'name': 'Model B', 'function': model_b_eval},
    {'name': 'Model C', 'function': model_c_eval}
]

comparison = ComparisonBenchmark()
results = comparison.run_comparison_study(models, test_datasets)

# Statistical analysis of differences
analysis = comparison.analyze_differences(results)
```

## Integration with Research Pipeline

### Experiment Integration
```python
from active_inference.research.experiments import ExperimentManager
from active_inference.research.benchmarks import BenchmarkPipeline

# Automatic benchmarking integration
experiment = ExperimentManager(base_dir='./experiments')
benchmarks = BenchmarkPipeline()

# Run experiment with benchmarking
results = experiment.run_study(study_config)
benchmark_results = benchmarks.evaluate_experiment_results(results)
```

### Simulation Integration
```python
from active_inference.research.simulations import SimulationEngine
from active_inference.research.benchmarks import SimulationBenchmark

# Simulation performance evaluation
engine = SimulationEngine(config={'time_scale': 'milliseconds'})
benchmarks = SimulationBenchmark()

# Evaluate simulation performance
sim_results = engine.run_comparison_study(model_configs)
performance = benchmarks.evaluate_simulation_performance(sim_results)
```

## Advanced Features

### Multi-Scale Benchmarking
- **Temporal Scales**: Real-time to evolutionary timeframes
- **Spatial Scales**: Single neuron to brain networks
- **Complexity Scales**: Simple models to complex hierarchies
- **Performance Scales**: Efficiency to accuracy trade-offs

### Robust Evaluation
- **Cross-Validation**: K-fold and stratified validation
- **Bootstrap Analysis**: Bootstrap performance estimation
- **Perturbation Analysis**: Robustness to parameter changes
- **Generalization Testing**: Out-of-distribution performance

### Statistical Rigor
```python
from active_inference.research.benchmarks import StatisticalBenchmark

# Rigorous statistical evaluation
stat_benchmark = StatisticalBenchmark()
statistical_validation = stat_benchmark.validate_performance(
    results,
    methods=['bootstrap', 'permutation', 'jackknife']
)
```

## Configuration Options

### Benchmark Settings
```python
config = {
    'significance_level': 0.05,
    'multiple_testing_correction': 'fdr',
    'cross_validation_folds': 5,
    'bootstrap_iterations': 1000,
    'performance_metrics': ['accuracy', 'free_energy', 'complexity'],
    'output_formats': ['json', 'csv', 'latex']
}
```

### Performance Optimization
```python
performance_config = {
    'parallel_evaluation': True,
    'cache_results': True,
    'memory_efficient': True,
    'gpu_acceleration': False,
    'batch_size': 32
}
```

## Quality Assurance

### Validation Methods
- **Ground Truth Comparison**: Validation against known results
- **Cross-Method Validation**: Multiple evaluation methods
- **Reproducibility Testing**: Consistent results across runs
- **Convergence Testing**: Algorithm convergence validation

### Reproducibility
- **Random Seeds**: Reproducible evaluation runs
- **Parameter Documentation**: Complete parameter specification
- **Environment Logging**: System and dependency tracking
- **Result Archiving**: Complete result preservation

## Benchmark Standards

### Active Inference Benchmarks
- **Predictive Accuracy**: Prediction performance metrics
- **Free Energy Minimization**: Variational free energy tracking
- **Belief Convergence**: Belief updating dynamics
- **Information Efficiency**: Information processing metrics
- **Computational Complexity**: Resource usage analysis

### Standard Benchmarking Protocols
- **Dataset Standards**: Standardized evaluation datasets
- **Metric Standards**: Consistent performance metrics
- **Reporting Standards**: Uniform result reporting
- **Comparison Standards**: Fair comparison protocols

## Contributing

We welcome contributions to the benchmarks suite! See [CONTRIBUTING.md](../../CONTRIBUTING.md) for guidelines.

### Development Setup
```bash
# Install benchmarking dependencies
pip install -e ".[benchmarks,dev]"

# Run benchmark tests
pytest tests/unit/test_benchmarks.py

# Run performance benchmarks
python benchmarks/benchmark_performance.py
```

## Learning Resources

- **Benchmarking Guide**: Performance evaluation methodology
- **Metric Selection**: Choosing appropriate performance metrics
- **Statistical Validation**: Ensuring evaluation rigor
- **Reporting Standards**: Benchmark result presentation

## Related Documentation

- **[Main README](../../README.md)**: Project overview
- **[Experiments](../experiments/README.md)**: Experiment management
- **[Simulations](../simulations/README.md)**: Simulation tools
- **[Analysis](../analysis/README.md)**: Statistical analysis
- **[Research Tools](../README.md)**: Research framework overview

---

*"Active Inference for, with, by Generative AI"* - Advancing research through comprehensive benchmarking, rigorous evaluation standards, and collaborative performance assessment.
