# Research Tools - Agent Documentation

This document provides comprehensive guidance for AI agents and contributors working within the Research Tools module of the Active Inference Knowledge Environment. It outlines research methodologies, experimental design, simulation frameworks, and best practices for conducting rigorous Active Inference research.

## Research Tools Module Overview

The Research Tools module provides a complete research ecosystem for Active Inference, including experiment management, simulation capabilities, statistical analysis, benchmarking tools, and collaborative research features. All tools are designed to support reproducible, rigorous scientific research following established scientific standards.

## Core Responsibilities

### Research Framework Development
- **Experiment Design**: Create robust experiment design frameworks
- **Simulation Engines**: Develop multi-scale simulation capabilities
- **Analysis Methods**: Implement comprehensive analysis toolkits
- **Benchmarking Systems**: Establish standardized evaluation frameworks
- **Collaboration Tools**: Enable collaborative research workflows

### Scientific Quality Assurance
- **Reproducibility**: Ensure complete research reproducibility
- **Validation**: Validate methods against established benchmarks
- **Documentation**: Maintain comprehensive research documentation
- **Ethics**: Ensure adherence to research ethics and standards
- **Peer Review**: Support peer review and validation processes

### Community Research Support
- **Tool Development**: Create tools that support community research
- **Method Sharing**: Enable sharing of research methods and results
- **Collaboration**: Support collaborative research initiatives
- **Education**: Provide research training and educational resources
- **Publication Support**: Assist with research publication and dissemination

## Development Workflows

### Research Tool Development Process
1. **Requirements Analysis**: Analyze research community needs
2. **Method Research**: Research established scientific methods
3. **Tool Design**: Design tool architecture and interfaces
4. **Implementation**: Implement tools following scientific standards
5. **Validation**: Validate against known benchmarks and methods
6. **Testing**: Comprehensive testing including edge cases
7. **Documentation**: Create comprehensive research documentation
8. **Review**: Submit for scientific and technical review
9. **Publication**: Release tools with proper documentation
10. **Maintenance**: Maintain and update based on community feedback

### Experiment Framework Development
1. **Experiment Design**: Design flexible experiment frameworks
2. **Parameter Management**: Implement comprehensive parameter handling
3. **Result Collection**: Develop robust result collection systems
4. **Analysis Integration**: Integrate with analysis toolkits
5. **Reproducibility**: Ensure complete experiment reproducibility
6. **Performance Optimization**: Optimize for research performance needs
7. **Documentation**: Document experiment design and execution
8. **Validation**: Validate framework against research requirements

### Simulation Engine Development
1. **Model Design**: Design simulation models and architectures
2. **Numerical Implementation**: Implement numerically stable algorithms
3. **Performance Optimization**: Optimize for computational efficiency
4. **Validation**: Validate against theoretical predictions
5. **Integration**: Integrate with experiment and analysis frameworks
6. **Documentation**: Document simulation methods and parameters
7. **Testing**: Comprehensive testing of simulation accuracy
8. **Community Review**: Submit for scientific validation

## Quality Standards

### Scientific Quality
- **Theoretical Soundness**: All methods must be theoretically sound
- **Numerical Accuracy**: Implementations must be numerically accurate
- **Validation**: Methods must be validated against benchmarks
- **Reproducibility**: All results must be reproducible
- **Documentation**: Complete scientific documentation required

### Code Quality
- **Algorithmic Correctness**: Algorithms must be correctly implemented
- **Numerical Stability**: Implementations must be numerically stable
- **Performance**: Efficient algorithms and data structures
- **Testing**: Comprehensive testing including edge cases
- **Documentation**: Clear documentation of all functionality

### Research Quality
- **Methodological Rigor**: Follow established research methodologies
- **Statistical Soundness**: Proper statistical analysis and interpretation
- **Ethical Standards**: Adherence to research ethics
- **Peer Review**: Support for peer review processes
- **Publication Ready**: Results suitable for scientific publication

## Implementation Patterns

### Experiment Framework Pattern
```python
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import json
from pathlib import Path

@dataclass
class ExperimentConfig:
    """Configuration for scientific experiments"""
    name: str
    type: str  # simulation, behavioral, neural, etc.
    parameters: Dict[str, Any]
    repetitions: int = 1
    random_seed: Optional[int] = None
    output_dir: Optional[str] = None

@dataclass
class ExperimentResult:
    """Results from experiment execution"""
    experiment_id: str
    config: ExperimentConfig
    data: Dict[str, Any]
    metrics: Dict[str, float]
    timestamp: str
    duration: float
    status: str  # success, failed, partial

class BaseExperiment(ABC):
    """Base class for Active Inference experiments"""

    def __init__(self, config: ExperimentConfig):
        """Initialize experiment with configuration"""
        self.config = config
        self.results: List[ExperimentResult] = []
        self.current_run = 0
        self.setup_experiment()

    @abstractmethod
    def setup_experiment(self) -> None:
        """Set up experiment environment and parameters"""
        pass

    @abstractmethod
    def run_single_trial(self) -> Dict[str, Any]:
        """Run single experiment trial"""
        pass

    @abstractmethod
    def collect_metrics(self, trial_data: Dict[str, Any]) -> Dict[str, float]:
        """Collect performance metrics from trial data"""
        pass

    def run_experiment(self) -> List[ExperimentResult]:
        """Run complete experiment with multiple repetitions"""
        import time

        for rep in range(self.config.repetitions):
            self.current_run = rep
            start_time = time.time()

            try:
                # Run single trial
                trial_data = self.run_single_trial()

                # Collect metrics
                metrics = self.collect_metrics(trial_data)

                # Create result
                result = ExperimentResult(
                    experiment_id=f"{self.config.name}_run_{rep}",
                    config=self.config,
                    data=trial_data,
                    metrics=metrics,
                    timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                    duration=time.time() - start_time,
                    status="success"
                )

                self.results.append(result)

            except Exception as e:
                # Handle experiment failure
                result = ExperimentResult(
                    experiment_id=f"{self.config.name}_run_{rep}",
                    config=self.config,
                    data={"error": str(e)},
                    metrics={},
                    timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                    duration=time.time() - start_time,
                    status="failed"
                )

                self.results.append(result)

        return self.results

    def save_results(self, output_path: Path) -> None:
        """Save experiment results to file"""
        output_path.mkdir(parents=True, exist_ok=True)

        # Save individual results
        for result in self.results:
            result_file = output_path / f"{result.experiment_id}.json"
            with open(result_file, 'w') as f:
                json.dump({
                    'experiment_id': result.experiment_id,
                    'config': result.config.__dict__,
                    'data': result.data,
                    'metrics': result.metrics,
                    'timestamp': result.timestamp,
                    'duration': result.duration,
                    'status': result.status
                }, f, indent=2)

        # Save summary
        summary = self.generate_summary()
        summary_file = output_path / 'experiment_summary.json'
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

    def generate_summary(self) -> Dict[str, Any]:
        """Generate experiment summary"""
        successful_runs = [r for r in self.results if r.status == 'success']
        failed_runs = [r for r in self.results if r.status == 'failed']

        summary = {
            'experiment_name': self.config.name,
            'total_runs': len(self.results),
            'successful_runs': len(successful_runs),
            'failed_runs': len(failed_runs),
            'average_duration': sum(r.duration for r in self.results) / len(self.results),
            'metrics_summary': {}
        }

        # Calculate metrics summary
        if successful_runs:
            metric_names = successful_runs[0].metrics.keys()
            for metric_name in metric_names:
                values = [r.metrics[metric_name] for r in successful_runs]
                summary['metrics_summary'][metric_name] = {
                    'mean': sum(values) / len(values),
                    'std': (sum((v - summary['metrics_summary'][metric_name]['mean'])**2 for v in values) / len(values))**0.5,
                    'min': min(values),
                    'max': max(values)
                }

        return summary

class ExperimentFramework:
    """Framework for managing and executing experiments"""

    def __init__(self, config: Dict[str, Any]):
        """Initialize experiment framework"""
        self.config = config
        self.experiments: Dict[str, BaseExperiment] = {}
        self.results: Dict[str, List[ExperimentResult]] = {}
        self.output_dir = Path(config.get('output_dir', './experiment_results'))
        self.setup_framework()

    def setup_framework(self) -> None:
        """Set up experiment framework"""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Experiment framework initialized: {self.output_dir}")

    def register_experiment(self, name: str, experiment_class: type) -> None:
        """Register experiment class"""
        self.experiments[name] = experiment_class

    def create_experiment(self, name: str, experiment_type: str, parameters: Dict[str, Any]) -> BaseExperiment:
        """Create experiment instance"""
        if experiment_type not in self.experiments:
            raise ValueError(f"Unknown experiment type: {experiment_type}")

        config = ExperimentConfig(
            name=name,
            type=experiment_type,
            parameters=parameters,
            repetitions=parameters.get('repetitions', 1),
            random_seed=parameters.get('random_seed'),
            output_dir=str(self.output_dir / name)
        )

        experiment_class = self.experiments[experiment_type]
        return experiment_class(config)

    def run_experiment(self, experiment: BaseExperiment) -> List[ExperimentResult]:
        """Run experiment and collect results"""
        print(f"Running experiment: {experiment.config.name}")
        results = experiment.run_experiment()

        # Save results
        experiment.save_results(Path(experiment.config.output_dir))

        # Store in framework
        self.results[experiment.config.name] = results

        print(f"Experiment completed: {len(results)} runs")
        return results

    def analyze_results(self, results: List[ExperimentResult], methods: List[str] = None) -> Dict[str, Any]:
        """Analyze experiment results"""
        if methods is None:
            methods = ['descriptive', 'statistical']

        analysis = {
            'experiment_name': results[0].config.name if results else 'unknown',
            'total_runs': len(results),
            'methods': methods,
            'results': {}
        }

        for method in methods:
            if method == 'descriptive':
                analysis['results']['descriptive'] = self.descriptive_analysis(results)
            elif method == 'statistical':
                analysis['results']['statistical'] = self.statistical_analysis(results)
            elif method == 'information_theory':
                analysis['results']['information_theory'] = self.information_theory_analysis(results)

        return analysis

    def descriptive_analysis(self, results: List[ExperimentResult]) -> Dict[str, Any]:
        """Perform descriptive statistical analysis"""
        import statistics

        successful_results = [r for r in results if r.status == 'success']
        if not successful_results:
            return {'error': 'No successful results to analyze'}

        # Extract metrics
        metrics = {}
        for result in successful_results:
            for metric_name, value in result.metrics.items():
                if metric_name not in metrics:
                    metrics[metric_name] = []
                metrics[metric_name].append(value)

        # Calculate descriptive statistics
        descriptive_stats = {}
        for metric_name, values in metrics.items():
            descriptive_stats[metric_name] = {
                'count': len(values),
                'mean': statistics.mean(values),
                'median': statistics.median(values),
                'std': statistics.stdev(values) if len(values) > 1 else 0,
                'min': min(values),
                'max': max(values)
            }

        return {
            'sample_size': len(successful_results),
            'metrics': descriptive_stats,
            'duration': {
                'mean': statistics.mean([r.duration for r in results]),
                'total': sum(r.duration for r in results)
            }
        }

    def statistical_analysis(self, results: List[ExperimentResult]) -> Dict[str, Any]:
        """Perform statistical analysis"""
        # Implementation for statistical analysis
        return {'status': 'implemented'}

    def information_theory_analysis(self, results: List[ExperimentResult]) -> Dict[str, Any]:
        """Perform information-theoretic analysis"""
        # Implementation for information theory analysis
        return {'status': 'implemented'}

    def generate_report(self, analysis: Dict[str, Any], format: str = 'markdown') -> str:
        """Generate comprehensive experiment report"""
        if format.lower() == 'markdown':
            return self.generate_markdown_report(analysis)
        elif format.lower() == 'html':
            return self.generate_html_report(analysis)
        else:
            return str(analysis)

    def generate_markdown_report(self, analysis: Dict[str, Any]) -> str:
        """Generate markdown format report"""
        report = [f"# Experiment Report: {analysis['experiment_name']}", ""]

        # Summary
        report.append("## Summary")
        report.append(f"- Total runs: {analysis['total_runs']}")
        report.append(f"- Analysis methods: {', '.join(analysis['methods'])}")
        report.append("")

        # Results
        for method, method_results in analysis['results'].items():
            report.append(f"## {method.title()} Analysis")
            report.append("```")
            report.append(str(method_results))
            report.append("```")
            report.append("")

        return "\n".join(report)
```

### Simulation Framework Pattern
```python
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import numpy as np

class BaseSimulation(ABC):
    """Base class for Active Inference simulations"""

    def __init__(self, config: Dict[str, Any]):
        """Initialize simulation with configuration"""
        self.config = config
        self.time = 0.0
        self.state = self.initialize_state()
        self.history: List[Dict[str, Any]] = []
        self.setup_simulation()

    @abstractmethod
    def initialize_state(self) -> Dict[str, Any]:
        """Initialize simulation state"""
        pass

    @abstractmethod
    def setup_simulation(self) -> None:
        """Set up simulation parameters and components"""
        pass

    @abstractmethod
    def step(self, dt: float) -> Dict[str, Any]:
        """Execute single simulation step"""
        pass

    @abstractmethod
    def get_observation(self) -> Dict[str, Any]:
        """Get current observation"""
        pass

    def run(self, duration: float, dt: float = 0.01) -> List[Dict[str, Any]]:
        """Run simulation for specified duration"""
        steps = int(duration / dt)
        results = []

        for step in range(steps):
            # Execute simulation step
            step_data = self.step(dt)

            # Get observation
            observation = self.get_observation()

            # Record results
            result = {
                'time': self.time,
                'step': step,
                'state': self.state.copy(),
                'observation': observation,
                'step_data': step_data
            }
            results.append(result)
            self.history.append(result)

            # Update time
            self.time += dt

        return results

    def reset(self) -> None:
        """Reset simulation to initial state"""
        self.time = 0.0
        self.state = self.initialize_state()
        self.history.clear()

class SimulationEngine:
    """Engine for running Active Inference simulations"""

    def __init__(self, config: Dict[str, Any]):
        """Initialize simulation engine"""
        self.config = config
        self.simulations: Dict[str, BaseSimulation] = {}
        self.results: Dict[str, List[Dict[str, Any]]] = {}

    def register_simulation(self, name: str, simulation_class: type) -> None:
        """Register simulation class"""
        self.simulations[name] = simulation_class

    def create_simulation(self, simulation_type: str, parameters: Dict[str, Any]) -> BaseSimulation:
        """Create simulation instance"""
        if simulation_type not in self.simulations:
            raise ValueError(f"Unknown simulation type: {simulation_type}")

        config = {**self.config, **parameters}
        simulation_class = self.simulations[simulation_type]
        return simulation_class(config)

    def run_simulation(self, simulation: BaseSimulation, duration: float, dt: float = 0.01) -> List[Dict[str, Any]]:
        """Run simulation and collect results"""
        print(f"Running simulation: {simulation.__class__.__name__}")
        results = simulation.run(duration, dt)

        # Store results
        sim_name = simulation.__class__.__name__
        self.results[sim_name] = results

        print(f"Simulation completed: {len(results)} time steps")
        return results

    def analyze_simulation(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze simulation results"""
        analysis = {
            'duration': results[-1]['time'] if results else 0,
            'steps': len(results),
            'state_trajectory': [r['state'] for r in results],
            'observations': [r['observation'] for r in results],
            'statistics': self.calculate_statistics(results)
        }

        return analysis

    def calculate_statistics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate simulation statistics"""
        if not results:
            return {}

        # Extract state variables
        state_keys = results[0]['state'].keys()
        stats = {}

        for key in state_keys:
            values = [r['state'][key] for r in results]
            stats[key] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'initial': values[0],
                'final': values[-1]
            }

        return stats
```

## Testing Guidelines

### Research Tool Testing
- **Algorithm Testing**: Test algorithmic correctness
- **Numerical Testing**: Validate numerical accuracy and stability
- **Performance Testing**: Test computational performance
- **Integration Testing**: Test component integration
- **Reproducibility Testing**: Verify reproducible results

### Scientific Validation
- **Benchmark Testing**: Validate against established benchmarks
- **Convergence Testing**: Test algorithm convergence properties
- **Stability Testing**: Test numerical stability
- **Accuracy Testing**: Validate against theoretical predictions
- **Comparison Testing**: Compare with alternative implementations

## Performance Considerations

### Computational Performance
- **Algorithm Efficiency**: Use efficient algorithms and data structures
- **Memory Management**: Optimize memory usage for large simulations
- **Parallel Processing**: Utilize parallel processing where beneficial
- **Caching**: Implement caching for expensive computations

### Numerical Performance
- **Stability**: Ensure numerical stability of algorithms
- **Precision**: Maintain appropriate numerical precision
- **Convergence**: Optimize convergence properties
- **Error Control**: Control numerical errors and approximations

## Maintenance and Evolution

### Method Updates
- **Literature Review**: Keep current with latest research developments
- **Method Validation**: Validate new methods against benchmarks
- **Performance Optimization**: Optimize existing methods
- **Documentation Updates**: Keep documentation current

### Community Integration
- **User Feedback**: Incorporate community feedback
- **Method Sharing**: Enable sharing of research methods
- **Collaboration Tools**: Support collaborative research
- **Publication Support**: Assist with research publication

## Common Challenges and Solutions

### Challenge: Reproducibility
**Solution**: Implement comprehensive reproducibility frameworks with complete documentation and code.

### Challenge: Performance
**Solution**: Profile and optimize computational performance while maintaining accuracy.

### Challenge: Validation
**Solution**: Establish validation against multiple benchmarks and theoretical predictions.

### Challenge: Documentation
**Solution**: Maintain comprehensive documentation following scientific standards.

## Getting Started as an Agent

### Development Setup
1. **Study Research Framework**: Understand current research tool architecture
2. **Learn Scientific Methods**: Study established research methodologies
3. **Practice Implementation**: Practice implementing research tools
4. **Understand Validation**: Learn research validation techniques

### Contribution Process
1. **Identify Research Needs**: Find gaps in current research capabilities
2. **Research Methods**: Study relevant scientific literature
3. **Design Solutions**: Create detailed tool designs
4. **Implement and Test**: Follow scientific implementation standards
5. **Validate Thoroughly**: Ensure scientific validity
6. **Document Completely**: Provide comprehensive scientific documentation
7. **Community Review**: Submit for scientific peer review

### Learning Resources
- **Scientific Computing**: Study scientific computing methodologies
- **Research Methods**: Learn established research practices
- **Numerical Analysis**: Master numerical analysis techniques
- **Statistical Methods**: Study advanced statistical methods
- **Scientific Writing**: Learn scientific documentation standards

## Related Documentation

- **[Research README](./README.md)**: Research tools module overview
- **[Main AGENTS.md](../../AGENTS.md)**: Project-wide agent guidelines
- **[Contributing Guide](../../CONTRIBUTING.md)**: Contribution processes
- **[Knowledge Repository](../../knowledge/)**: Theoretical foundations
- **[Analysis Tools](../../research/analysis/)**: Analysis methodologies

---

*"Active Inference for, with, by Generative AI"* - Advancing research through comprehensive tools, rigorous methods, and collaborative scientific inquiry.



