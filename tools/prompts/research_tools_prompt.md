# Research Tools and Experimentation Prompt

**"Active Inference for, with, by Generative AI"**

## 🎯 Mission: Develop Research Tools and Experimentation Frameworks

You are tasked with developing comprehensive research tools and experimentation frameworks for the Active Inference Knowledge Environment. This involves creating analysis tools, simulation frameworks, benchmarking systems, and data management capabilities that enable cutting-edge research in Active Inference and related fields.

## 📋 Research Tools Requirements

### Core Research Standards (MANDATORY)
1. **Scientific Rigor**: All tools must support reproducible, peer-reviewable research
2. **Extensibility**: Tools must be easily extensible for new research questions
3. **Performance**: Tools must handle large-scale simulations and data analysis
4. **Interoperability**: Tools must integrate with existing scientific computing ecosystems
5. **Documentation**: Comprehensive documentation for research reproducibility
6. **Validation**: Built-in validation and statistical analysis capabilities

### Research Tools Architecture
```
research/
├── analysis/                    # Statistical and information-theoretic analysis
│   ├── bayesian_analysis.py    # Bayesian model analysis and comparison
│   ├── information_geometry.py # Information geometry computations
│   ├── variational_analysis.py # Variational method analysis
│   ├── complexity_analysis.py  # Model complexity and capacity analysis
│   └── convergence_analysis.py # Algorithm convergence analysis
├── simulations/                # Multi-scale simulation frameworks
│   ├── agent_simulation.py     # Active Inference agent simulation
│   ├── environment_simulation.py # Environment and world models
│   ├── neural_simulation.py    # Neural dynamics simulation
│   ├── population_simulation.py # Multi-agent population dynamics
│   └── evolutionary_simulation.py # Evolutionary algorithm simulation
├── benchmarks/                 # Performance evaluation and comparison
│   ├── benchmark_suites.py     # Standardized benchmark test suites
│   ├── performance_metrics.py  # Comprehensive performance evaluation
│   ├── statistical_comparison.py # Statistical significance testing
│   └── scalability_testing.py  # Scalability and efficiency testing
├── experiments/                # Experiment management and orchestration
│   ├── experiment_design.py    # Experimental design and planning
│   ├── parameter_sweeps.py     # Parameter space exploration
│   ├── hypothesis_testing.py   # Hypothesis formulation and testing
│   ├── data_collection.py      # Automated data collection and logging
│   └── result_analysis.py      # Experiment result analysis and visualization
├── data_management/           # Research data management
│   ├── dataset_management.py   # Research dataset organization
│   ├── preprocessing_pipeline.py # Data preprocessing and cleaning
│   ├── feature_engineering.py  # Feature extraction and engineering
│   ├── metadata_management.py  # Research metadata and provenance
│   └── data_versioning.py      # Dataset versioning and tracking
└── visualization/             # Research result visualization
    ├── analysis_visualization.py # Analysis result visualization
    ├── simulation_visualization.py # Simulation result visualization
    ├── comparison_visualization.py # Model and result comparison
    ├── statistical_visualization.py # Statistical analysis visualization
    └── interactive_exploration.py  # Interactive research data exploration
```

## 🏗️ Analysis Framework Development

### Phase 1: Bayesian Analysis Tools

#### 1.1 Bayesian Model Analysis
```python
import numpy as np
import scipy.stats as stats
from typing import Dict, List, Any, Optional, Tuple, Callable
import logging
from dataclasses import dataclass
from abc import ABC, abstractmethod

@dataclass
class BayesianModelResult:
    """Results from Bayesian model analysis"""
    model_name: str
    posterior_samples: np.ndarray
    log_likelihood: float
    evidence: float
    parameter_estimates: Dict[str, float]
    credible_intervals: Dict[str, Tuple[float, float]]
    convergence_diagnostics: Dict[str, Any]
    model_comparison_metrics: Dict[str, float]

@dataclass
class ModelComparisonResult:
    """Results from model comparison analysis"""
    models: List[str]
    log_bayes_factors: np.ndarray
    posterior_model_probabilities: np.ndarray
    information_criteria: Dict[str, np.ndarray]
    bayesian_information_criterion: np.ndarray
    akaike_information_criterion: np.ndarray
    deviance_information_criterion: np.ndarray

class BayesianAnalysisFramework:
    """Comprehensive Bayesian analysis framework for Active Inference models"""

    def __init__(self, config: Dict[str, Any]):
        """Initialize Bayesian analysis framework"""
        self.config = config
        self.logger = logging.getLogger('BayesianAnalysisFramework')
        self.analysis_methods = self.initialize_analysis_methods()

    def initialize_analysis_methods(self) -> Dict[str, Callable]:
        """Initialize available analysis methods"""
        return {
            'posterior_analysis': self.analyze_posterior_distribution,
            'convergence_diagnostics': self.check_convergence_diagnostics,
            'model_comparison': self.perform_model_comparison,
            'sensitivity_analysis': self.perform_sensitivity_analysis,
            'predictive_checks': self.perform_predictive_checks
        }

    def analyze_bayesian_model(self, model_results: Dict[str, Any],
                              analysis_types: List[str] = None) -> BayesianModelResult:
        """Perform comprehensive Bayesian analysis of model results"""
        if analysis_types is None:
            analysis_types = ['posterior_analysis', 'convergence_diagnostics']

        analysis_result = BayesianModelResult(
            model_name=model_results.get('model_name', 'unknown'),
            posterior_samples=np.array(model_results.get('posterior_samples', [])),
            log_likelihood=model_results.get('log_likelihood', 0.0),
            evidence=model_results.get('evidence', 0.0),
            parameter_estimates={},
            credible_intervals={},
            convergence_diagnostics={},
            model_comparison_metrics={}
        )

        # Perform requested analyses
        for analysis_type in analysis_types:
            if analysis_type in self.analysis_methods:
                analysis_method = self.analysis_methods[analysis_type]
                analysis_result = analysis_method(model_results, analysis_result)

        return analysis_result

    def analyze_posterior_distribution(self, model_results: Dict[str, Any],
                                     analysis_result: BayesianModelResult) -> BayesianModelResult:
        """Analyze posterior distribution characteristics"""
        posterior_samples = analysis_result.posterior_samples

        if len(posterior_samples) == 0:
            self.logger.warning("No posterior samples available for analysis")
            return analysis_result

        # Extract parameter names (assume first dimension is samples)
        n_samples, n_params = posterior_samples.shape

        # Calculate parameter estimates (mean and median)
        parameter_estimates = {}
        credible_intervals = {}

        for i in range(n_params):
            param_samples = posterior_samples[:, i]
            param_name = f"param_{i+1}"  # In practice, use actual parameter names

            # Point estimates
            parameter_estimates[param_name] = {
                'mean': np.mean(param_samples),
                'median': np.median(param_samples),
                'std': np.std(param_samples)
            }

            # Credible intervals (95% HDI)
            credible_intervals[param_name] = self.calculate_highest_density_interval(
                param_samples, 0.95
            )

        analysis_result.parameter_estimates = parameter_estimates
        analysis_result.credible_intervals = credible_intervals

        return analysis_result

    def calculate_highest_density_interval(self, samples: np.ndarray,
                                         credibility: float) -> Tuple[float, float]:
        """Calculate highest density interval (HDI)"""
        sorted_samples = np.sort(samples)
        n_samples = len(sorted_samples)

        # Calculate interval width for each possible interval
        interval_idx = int(np.ceil(n_samples * (1 - credibility)))
        interval_widths = sorted_samples[interval_idx:] - sorted_samples[:n_samples-interval_idx]

        # Find minimum width interval
        min_idx = np.argmin(interval_widths)
        hdi_lower = sorted_samples[min_idx]
        hdi_upper = sorted_samples[min_idx + interval_idx]

        return (float(hdi_lower), float(hdi_upper))

    def check_convergence_diagnostics(self, model_results: Dict[str, Any],
                                    analysis_result: BayesianModelResult) -> BayesianModelResult:
        """Check MCMC convergence diagnostics"""
        posterior_samples = analysis_result.posterior_samples

        diagnostics = {}

        if len(posterior_samples) > 100:  # Need minimum samples for diagnostics
            # R-hat (Gelman-Rubin statistic)
            diagnostics['r_hat'] = self.calculate_r_hat(posterior_samples)

            # Effective sample size
            diagnostics['effective_sample_size'] = self.calculate_effective_sample_size(posterior_samples)

            # Autocorrelation time
            diagnostics['autocorrelation_time'] = self.calculate_autocorrelation_time(posterior_samples)

            # Monte Carlo standard error
            diagnostics['mcse'] = self.calculate_monte_carlo_standard_error(posterior_samples)

        analysis_result.convergence_diagnostics = diagnostics
        return analysis_result

    def calculate_r_hat(self, samples: np.ndarray) -> float:
        """Calculate R-hat convergence diagnostic"""
        # Simplified R-hat calculation for single chain
        # In practice, would compare multiple chains
        if samples.shape[0] < 100:
            return float('nan')

        # Split chain in half and compare means
        n_samples = samples.shape[0]
        split_point = n_samples // 2

        first_half = samples[:split_point]
        second_half = samples[split_point:]

        # Calculate R-hat for each parameter
        r_hats = []
        for param_idx in range(samples.shape[1]):
            mean1 = np.mean(first_half[:, param_idx])
            mean2 = np.mean(second_half[:, param_idx])
            var1 = np.var(first_half[:, param_idx], ddof=1)
            var2 = np.var(second_half[:, param_idx], ddof=1)

            # Pooled variance
            pooled_var = ((var1 + var2) / 2)

            # R-hat statistic
            if pooled_var > 0:
                r_hat = np.sqrt(1 + (mean1 - mean2)**2 / (2 * pooled_var))
                r_hats.append(r_hat)
            else:
                r_hats.append(1.0)

        return float(np.mean(r_hats)) if r_hats else float('nan')

    def calculate_effective_sample_size(self, samples: np.ndarray) -> Dict[str, float]:
        """Calculate effective sample size for each parameter"""
        ess_values = {}

        for param_idx in range(samples.shape[1]):
            param_samples = samples[:, param_idx]
            param_name = f"param_{param_idx+1}"

            # Calculate autocorrelation
            autocorr = self.calculate_autocorrelation(param_samples)

            # Estimate effective sample size
            if len(autocorr) > 1:
                # Sum autocorrelation until it becomes negative
                tau = 1 + 2 * sum(autocorr[1:])  # Simplified
                ess = len(param_samples) / tau
                ess_values[param_name] = max(ess, 1.0)  # Minimum ESS of 1
            else:
                ess_values[param_name] = float(len(param_samples))

        return ess_values

    def calculate_autocorrelation_time(self, samples: np.ndarray) -> Dict[str, float]:
        """Calculate autocorrelation time for each parameter"""
        autocorr_times = {}

        for param_idx in range(samples.shape[1]):
            param_samples = samples[:, param_idx]
            param_name = f"param_{param_idx+1}"

            autocorr = self.calculate_autocorrelation(param_samples)

            # Find where autocorrelation becomes negligible (< 0.1)
            negligible_idx = next(
                (i for i, ac in enumerate(autocorr) if abs(ac) < 0.1),
                len(autocorr)
            )

            autocorr_times[param_name] = float(negligible_idx)

        return autocorr_times

    def calculate_autocorrelation(self, samples: np.ndarray, max_lag: int = 50) -> np.ndarray:
        """Calculate autocorrelation function"""
        n = len(samples)
        mean = np.mean(samples)
        var = np.var(samples)

        if var == 0:
            return np.array([1.0])

        autocorr = []
        for lag in range(min(max_lag, n)):
            cov = np.mean((samples[lag:] - mean) * (samples[:n-lag] - mean))
            autocorr.append(cov / var)

        return np.array(autocorr)

    def calculate_monte_carlo_standard_error(self, samples: np.ndarray) -> Dict[str, float]:
        """Calculate Monte Carlo standard error"""
        mcse_values = {}

        for param_idx in range(samples.shape[1]):
            param_samples = samples[:, param_idx]
            param_name = f"param_{param_idx+1}"

            # MCSE = sqrt(variance / effective_sample_size)
            # Simplified calculation
            variance = np.var(param_samples, ddof=1)
            ess = self.calculate_effective_sample_size(samples)[param_name]

            mcse = np.sqrt(variance / ess)
            mcse_values[param_name] = float(mcse)

        return mcse_values

    def perform_model_comparison(self, model_results_list: List[Dict[str, Any]]) -> ModelComparisonResult:
        """Perform Bayesian model comparison"""
        n_models = len(model_results_list)

        # Extract log evidences/marginal likelihoods
        log_evidences = np.array([
            result.get('evidence', 0.0) for result in model_results_list
        ])

        model_names = [
            result.get('model_name', f'model_{i+1}')
            for i, result in enumerate(model_results_list)
        ]

        # Calculate Bayes factors
        max_log_evidence = np.max(log_evidences)
        log_bayes_factors = log_evidences - max_log_evidence

        # Calculate posterior model probabilities (assuming uniform priors)
        bayes_factors = np.exp(log_bayes_factors)
        posterior_probs = bayes_factors / np.sum(bayes_factors)

        # Calculate information criteria
        information_criteria = {}
        bic_values = []
        aic_values = []
        dic_values = []

        for result in model_results_list:
            n_params = result.get('n_parameters', 1)
            n_data = result.get('n_data_points', 1)
            log_likelihood = result.get('log_likelihood', 0.0)

            # BIC = -2 * log_likelihood + k * log(n)
            bic = -2 * log_likelihood + n_params * np.log(n_data)
            bic_values.append(bic)

            # AIC = -2 * log_likelihood + 2 * k
            aic = -2 * log_likelihood + 2 * n_params
            aic_values.append(aic)

            # DIC (simplified)
            dic = -2 * log_likelihood + 2 * n_params  # Simplified DIC
            dic_values.append(dic)

        return ModelComparisonResult(
            models=model_names,
            log_bayes_factors=log_bayes_factors,
            posterior_model_probabilities=posterior_probs,
            information_criteria=information_criteria,
            bayesian_information_criterion=np.array(bic_values),
            akaike_information_criterion=np.array(aic_values),
            deviance_information_criterion=np.array(dic_values)
        )

    def perform_sensitivity_analysis(self, model_function: Callable,
                                   parameter_ranges: Dict[str, Tuple[float, float]],
                                   n_samples: int = 1000) -> Dict[str, Any]:
        """Perform sensitivity analysis on model parameters"""
        sensitivity_results = {
            'parameter_sensitivities': {},
            'interaction_effects': {},
            'variance_decomposition': {}
        }

        # Generate parameter samples using Latin Hypercube Sampling
        parameter_samples = self.generate_parameter_samples(parameter_ranges, n_samples)

        # Evaluate model for each parameter set
        model_outputs = []
        for params in parameter_samples:
            try:
                output = model_function(**params)
                model_outputs.append(output)
            except Exception as e:
                self.logger.warning(f"Model evaluation failed for parameters {params}: {e}")
                model_outputs.append(None)

        # Remove failed evaluations
        valid_outputs = [out for out in model_outputs if out is not None]
        valid_samples = [params for params, out in zip(parameter_samples, model_outputs) if out is not None]

        if len(valid_outputs) < 10:
            self.logger.error("Insufficient valid model evaluations for sensitivity analysis")
            return sensitivity_results

        # Calculate sensitivity indices
        for param_name in parameter_ranges.keys():
            sensitivity = self.calculate_sensitivity_index(
                valid_samples, valid_outputs, param_name
            )
            sensitivity_results['parameter_sensitivities'][param_name] = sensitivity

        return sensitivity_results

    def generate_parameter_samples(self, parameter_ranges: Dict[str, Tuple[float, float]],
                                 n_samples: int) -> List[Dict[str, float]]:
        """Generate parameter samples using Latin Hypercube Sampling"""
        param_names = list(parameter_ranges.keys())
        n_params = len(param_names)

        # Simple random sampling (could be improved with proper LHS)
        samples = []

        for _ in range(n_samples):
            sample = {}
            for param_name in param_names:
                min_val, max_val = parameter_ranges[param_name]
                sample[param_name] = min_val + (max_val - min_val) * np.random.random()
            samples.append(sample)

        return samples

    def calculate_sensitivity_index(self, samples: List[Dict[str, float]],
                                  outputs: List[float], parameter_name: str) -> float:
        """Calculate sensitivity index for a parameter"""
        param_values = [sample[parameter_name] for sample in samples]

        # Calculate correlation between parameter and output
        correlation = np.corrcoef(param_values, outputs)[0, 1]

        # Calculate variance-based sensitivity
        # Simplified: use squared correlation as sensitivity measure
        sensitivity = correlation ** 2 if not np.isnan(correlation) else 0.0

        return float(sensitivity)

    def perform_predictive_checks(self, model_results: Dict[str, Any],
                                observed_data: np.ndarray) -> Dict[str, Any]:
        """Perform posterior predictive checks"""
        posterior_samples = np.array(model_results.get('posterior_samples', []))
        n_samples = posterior_samples.shape[0] if len(posterior_samples.shape) > 0 else 0

        predictive_checks = {
            'test_statistics': {},
            'p_values': {},
            'discrepancy_measures': {}
        }

        if n_samples < 10:
            self.logger.warning("Insufficient posterior samples for predictive checks")
            return predictive_checks

        # Generate posterior predictive samples
        predictive_samples = []
        for i in range(min(n_samples, 100)):  # Limit for computational efficiency
            # Use posterior sample to generate predictive data
            posterior_sample = posterior_samples[i] if len(posterior_samples.shape) > 1 else posterior_samples
            predictive_data = self.generate_predictive_data(posterior_sample, observed_data.shape)
            predictive_samples.append(predictive_data)

        predictive_samples = np.array(predictive_samples)

        # Calculate test statistics
        observed_mean = np.mean(observed_data)
        observed_std = np.std(observed_data)

        predictive_means = np.mean(predictive_samples, axis=1)
        predictive_stds = np.std(predictive_samples, axis=1)

        # P-values for test statistics
        mean_p_value = np.mean(predictive_means >= observed_mean)
        std_p_value = np.mean(predictive_stds >= observed_std)

        predictive_checks['test_statistics'] = {
            'observed_mean': float(observed_mean),
            'observed_std': float(observed_std),
            'predictive_mean_mean': float(np.mean(predictive_means)),
            'predictive_std_mean': float(np.mean(predictive_stds))
        }

        predictive_checks['p_values'] = {
            'mean_test': float(mean_p_value),
            'std_test': float(std_p_value)
        }

        return predictive_checks

    def generate_predictive_data(self, posterior_sample: np.ndarray,
                               data_shape: Tuple[int, ...]) -> np.ndarray:
        """Generate predictive data from posterior sample (placeholder)"""
        # In practice, this would use the model to generate predictions
        # For now, return random data with similar characteristics
        return np.random.normal(0, 1, data_shape)

    def generate_analysis_report(self, analysis_result: BayesianModelResult) -> str:
        """Generate comprehensive analysis report"""
        report = f"""
# Bayesian Analysis Report: {analysis_result.model_name}

## Parameter Estimates
"""

        for param_name, estimates in analysis_result.parameter_estimates.items():
            report += f"""
### {param_name}
- **Mean**: {estimates['mean']:.4f}
- **Median**: {estimates['median']:.4f}
- **Std**: {estimates['std']:.4f}
"""

        report += "\n## Credible Intervals (95% HDI)\n"
        for param_name, interval in analysis_result.credible_intervals.items():
            report += f"- **{param_name}**: [{interval[0]:.4f}, {interval[1]:.4f}]\n"

        if analysis_result.convergence_diagnostics:
            report += "\n## Convergence Diagnostics\n"
            diagnostics = analysis_result.convergence_diagnostics

            if 'r_hat' in diagnostics and not np.isnan(diagnostics['r_hat']):
                report += f"- **R-hat**: {diagnostics['r_hat']:.4f}\n"

            if 'effective_sample_size' in diagnostics:
                ess = diagnostics['effective_sample_size']
                for param, ess_value in ess.items():
                    report += f"- **ESS ({param})**: {ess_value:.1f}\n"

        report += f"""
## Model Evidence
- **Log Likelihood**: {analysis_result.log_likelihood:.4f}
- **Log Evidence**: {analysis_result.evidence:.4f}
"""

        return report

class TestBayesianAnalysisFramework:
    """Unit tests for Bayesian analysis framework"""

    @pytest.fixture
    def analysis_framework(self):
        """Create analysis framework for testing"""
        config = {'test_mode': True}
        return BayesianAnalysisFramework(config)

    def test_posterior_analysis(self, analysis_framework):
        """Test posterior distribution analysis"""
        # Create mock model results
        mock_results = {
            'model_name': 'test_model',
            'posterior_samples': np.random.normal(0, 1, (1000, 3)),
            'log_likelihood': -150.5,
            'evidence': -155.2
        }

        # Perform analysis
        result = analysis_framework.analyze_bayesian_model(mock_results)

        # Verify results
        assert result.model_name == 'test_model'
        assert len(result.parameter_estimates) == 3  # 3 parameters
        assert len(result.credible_intervals) == 3

        # Check parameter estimates
        for param_name, estimates in result.parameter_estimates.items():
            assert 'mean' in estimates
            assert 'median' in estimates
            assert 'std' in estimates

    def test_convergence_diagnostics(self, analysis_framework):
        """Test convergence diagnostics"""
        # Create mock results with good convergence
        mock_results = {
            'model_name': 'converged_model',
            'posterior_samples': np.random.normal(0, 1, (2000, 2))
        }

        result = analysis_framework.analyze_bayesian_model(
            mock_results, ['convergence_diagnostics']
        )

        # Check that diagnostics were calculated
        assert 'r_hat' in result.convergence_diagnostics
        assert 'effective_sample_size' in result.convergence_diagnostics

    def test_model_comparison(self, analysis_framework):
        """Test Bayesian model comparison"""
        # Create mock results for two models
        model_results = [
            {
                'model_name': 'model_1',
                'evidence': -100.0,
                'n_parameters': 3,
                'n_data_points': 100,
                'log_likelihood': -95.0
            },
            {
                'model_name': 'model_2',
                'evidence': -105.0,
                'n_parameters': 5,
                'n_data_points': 100,
                'log_likelihood': -98.0
            }
        ]

        comparison = analysis_framework.perform_model_comparison(model_results)

        # Verify comparison results
        assert len(comparison.models) == 2
        assert len(comparison.log_bayes_factors) == 2
        assert len(comparison.posterior_model_probabilities) == 2
        assert abs(sum(comparison.posterior_model_probabilities) - 1.0) < 0.001  # Should sum to 1

    def test_sensitivity_analysis(self, analysis_framework):
        """Test sensitivity analysis"""
        # Define simple test model
        def test_model(param1: float, param2: float) -> float:
            return param1 * 2 + param2 * 0.5 + np.random.normal(0, 0.1)

        # Define parameter ranges
        parameter_ranges = {
            'param1': (-2.0, 2.0),
            'param2': (-1.0, 1.0)
        }

        # Perform sensitivity analysis
        sensitivity = analysis_framework.perform_sensitivity_analysis(
            test_model, parameter_ranges, n_samples=100
        )

        # Verify results
        assert 'parameter_sensitivities' in sensitivity
        assert 'param1' in sensitivity['parameter_sensitivities']
        assert 'param2' in sensitivity['parameter_sensitivities']

        # param1 should have higher sensitivity (coefficient of 2 vs 0.5)
        assert sensitivity['parameter_sensitivities']['param1'] > sensitivity['parameter_sensitivities']['param2']

    def test_analysis_report_generation(self, analysis_framework):
        """Test analysis report generation"""
        # Create mock analysis result
        mock_result = BayesianModelResult(
            model_name='test_model',
            posterior_samples=np.random.normal(0, 1, (100, 2)),
            log_likelihood=-50.0,
            evidence=-55.0,
            parameter_estimates={
                'param1': {'mean': 0.1, 'median': 0.05, 'std': 0.8},
                'param2': {'mean': -0.2, 'median': -0.15, 'std': 0.6}
            },
            credible_intervals={
                'param1': (-1.5, 1.7),
                'param2': (-1.3, 0.9)
            },
            convergence_diagnostics={
                'r_hat': 1.02,
                'effective_sample_size': {'param1': 85.5, 'param2': 92.3}
            },
            model_comparison_metrics={}
        )

        # Generate report
        report = analysis_framework.generate_analysis_report(mock_result)

        # Verify report content
        assert 'test_model' in report
        assert 'Parameter Estimates' in report
        assert 'Credible Intervals' in report
        assert 'Convergence Diagnostics' in report
        assert 'R-hat' in report
        assert 'ESS' in report
```

### Phase 2: Simulation Framework Development

#### 2.1 Multi-Scale Simulation Engine
```python
import numpy as np
from typing import Dict, List, Any, Optional, Callable, Tuple
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import asyncio

@dataclass
class SimulationConfig:
    """Configuration for simulation runs"""
    time_steps: int = 1000
    dt: float = 0.01
    n_agents: int = 1
    environment_size: Tuple[int, int] = (100, 100)
    random_seed: Optional[int] = None
    parallel_execution: bool = False
    max_workers: int = 4
    logging_interval: int = 100

@dataclass
class SimulationResult:
    """Results from simulation run"""
    simulation_id: str
    config: SimulationConfig
    trajectories: Dict[str, np.ndarray]
    observables: Dict[str, np.ndarray]
    statistics: Dict[str, Any]
    execution_time: float
    convergence_metrics: Dict[str, float]

class SimulationEngine(ABC):
    """Abstract base class for simulation engines"""

    def __init__(self, config: SimulationConfig):
        """Initialize simulation engine"""
        self.config = config
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        self.random_state = np.random.RandomState(config.random_seed)

        # Initialize simulation state
        self.current_time = 0.0
        self.time_step = 0
        self.state_variables: Dict[str, np.ndarray] = {}
        self.observables: Dict[str, List[float]] = {}

    @abstractmethod
    def initialize_simulation(self) -> None:
        """Initialize simulation state and parameters"""
        pass

    @abstractmethod
    def step_simulation(self) -> None:
        """Execute one simulation time step"""
        pass

    @abstractmethod
    def calculate_observables(self) -> Dict[str, float]:
        """Calculate observable quantities"""
        pass

    def run_simulation(self) -> SimulationResult:
        """Run complete simulation"""
        import time
        start_time = time.time()

        self.logger.info(f"Starting simulation with {self.config.time_steps} steps")

        # Initialize
        self.initialize_simulation()

        # Initialize observables tracking
        for obs_name in self.get_observable_names():
            self.observables[obs_name] = []

        # Run simulation loop
        for step in range(self.config.time_steps):
            self.time_step = step
            self.current_time = step * self.config.dt

            # Execute simulation step
            self.step_simulation()

            # Calculate and store observables
            if step % self.config.logging_interval == 0:
                obs_values = self.calculate_observables()
                for obs_name, value in obs_values.items():
                    self.observables[obs_name].append(value)

                if step % (self.config.logging_interval * 10) == 0:
                    self.logger.debug(f"Simulation step {step}/{self.config.time_steps}")

        # Convert observables to numpy arrays
        observables_arrays = {}
        for obs_name, values in self.observables.items():
            observables_arrays[obs_name] = np.array(values)

        # Calculate simulation statistics
        statistics = self.calculate_simulation_statistics(observables_arrays)

        # Calculate convergence metrics
        convergence_metrics = self.calculate_convergence_metrics(observables_arrays)

        execution_time = time.time() - start_time

        result = SimulationResult(
            simulation_id=f"sim_{int(start_time)}",
            config=self.config,
            trajectories=self.get_trajectories(),
            observables=observables_arrays,
            statistics=statistics,
            execution_time=execution_time,
            convergence_metrics=convergence_metrics
        )

        self.logger.info(f"Simulation completed in {execution_time:.2f} seconds")
        return result

    @abstractmethod
    def get_observable_names(self) -> List[str]:
        """Get list of observable quantity names"""
        pass

    @abstractmethod
    def get_trajectories(self) -> Dict[str, np.ndarray]:
        """Get trajectory data for visualization"""
        pass

    def calculate_simulation_statistics(self, observables: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Calculate statistical properties of simulation results"""
        statistics = {}

        for obs_name, values in observables.items():
            if len(values) > 0:
                statistics[f"{obs_name}_mean"] = float(np.mean(values))
                statistics[f"{obs_name}_std"] = float(np.std(values))
                statistics[f"{obs_name}_min"] = float(np.min(values))
                statistics[f"{obs_name}_max"] = float(np.max(values))

                # Calculate autocorrelation if enough data
                if len(values) > 10:
                    autocorr = np.correlate(values - np.mean(values),
                                           values - np.mean(values),
                                           mode='full')
                    autocorr = autocorr[autocorr.size // 2:]  # Second half
                    autocorr = autocorr / autocorr[0]  # Normalize
                    statistics[f"{obs_name}_autocorr_time"] = float(
                        next((i for i, ac in enumerate(autocorr) if ac < 0.1), len(autocorr))
                    )

        return statistics

    def calculate_convergence_metrics(self, observables: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Calculate convergence metrics for simulation"""
        convergence = {}

        for obs_name, values in observables.items():
            if len(values) > 100:  # Need sufficient data
                # Split data into halves and compare means
                midpoint = len(values) // 2
                first_half = values[:midpoint]
                second_half = values[midpoint:]

                mean1, mean2 = np.mean(first_half), np.mean(second_half)
                std1, std2 = np.std(first_half), np.std(second_half)

                # Relative difference in means
                if abs(mean1) > 1e-10:  # Avoid division by very small number
                    convergence[f"{obs_name}_mean_convergence"] = abs(mean1 - mean2) / abs(mean1)
                else:
                    convergence[f"{obs_name}_mean_convergence"] = abs(mean1 - mean2)

                # Relative difference in standard deviations
                if std1 > 1e-10:
                    convergence[f"{obs_name}_std_convergence"] = abs(std1 - std2) / std1
                else:
                    convergence[f"{obs_name}_std_convergence"] = abs(std1 - std2)

        return convergence

    def reset_simulation(self) -> None:
        """Reset simulation to initial state"""
        self.current_time = 0.0
        self.time_step = 0
        self.state_variables.clear()
        self.observables.clear()
        self.initialize_simulation()

class ActiveInferenceAgentSimulation(SimulationEngine):
    """Simulation of Active Inference agents"""

    def __init__(self, config: SimulationConfig, agent_config: Dict[str, Any]):
        """Initialize Active Inference agent simulation"""
        super().__init__(config)
        self.agent_config = agent_config

        # Agent state variables
        self.beliefs: np.ndarray = None
        self.policies: np.ndarray = None
        self.expected_free_energy: np.ndarray = None
        self.posterior_policies: np.ndarray = None

        # Environment state
        self.environment_state: np.ndarray = None
        self.observations: np.ndarray = None

    def initialize_simulation(self) -> None:
        """Initialize Active Inference simulation"""
        n_states = self.agent_config.get('n_states', 10)
        n_policies = self.agent_config.get('n_policies', 5)
        n_observations = self.agent_config.get('n_observations', 8)

        # Initialize agent beliefs (uniform prior)
        self.beliefs = np.ones(n_states) / n_states

        # Initialize policies (random)
        self.policies = self.random_state.rand(n_policies, n_states, n_states)

        # Initialize expected free energy
        self.expected_free_energy = np.zeros(n_policies)

        # Initialize environment
        self.environment_state = self.random_state.randint(0, n_states)

        # Initialize observation likelihood
        self.observation_likelihood = self.random_state.rand(n_states, n_observations)
        self.observation_likelihood /= self.observation_likelihood.sum(axis=0)

    def step_simulation(self) -> None:
        """Execute one Active Inference simulation step"""
        # Generate observation from current environment state
        observation_probs = self.observation_likelihood[self.environment_state]
        observation = self.random_state.choice(len(observation_probs), p=observation_probs)
        self.observations = np.zeros(len(observation_probs))
        self.observations[observation] = 1.0

        # Update beliefs based on observation
        self.update_beliefs()

        # Calculate expected free energy for each policy
        self.calculate_expected_free_energy()

        # Select policy using softmax
        policy_logits = -self.expected_free_energy  # Negative because we minimize EFE
        policy_probs = np.exp(policy_logits - np.max(policy_logits))
        policy_probs /= np.sum(policy_probs)

        selected_policy = self.random_state.choice(len(policy_probs), p=policy_probs)

        # Execute selected policy
        self.execute_policy(selected_policy)

        # Update environment (simplified random walk)
        self.update_environment()

    def update_beliefs(self) -> None:
        """Update beliefs using variational inference"""
        # Simplified belief update
        # In practice, this would use proper variational inference

        # Likelihood of observation given each state
        likelihood = self.observation_likelihood[:, np.argmax(self.observations)]

        # Posterior = likelihood * prior
        posterior = likelihood * self.beliefs
        posterior /= np.sum(posterior)  # Normalize

        self.beliefs = posterior

    def calculate_expected_free_energy(self) -> None:
        """Calculate expected free energy for each policy"""
        for policy_idx, policy in enumerate(self.policies):
            # Simplified EFE calculation
            # In practice, this would include epistemic and extrinsic terms

            # Expected surprise (epistemic affordance)
            epistemic_term = 0.0
            for next_state in range(len(self.beliefs)):
                transition_prob = policy[self.environment_state, next_state]
                if transition_prob > 0:
                    # KL divergence between posterior and prior
                    posterior_pred = self.predict_posterior(next_state)
                    kl_div = np.sum(posterior_pred * np.log(posterior_pred / self.beliefs))
                    epistemic_term += transition_prob * kl_div

            # Extrinsic term (goal-directed behavior)
            extrinsic_term = self.calculate_extrinsic_term(policy)

            self.expected_free_energy[policy_idx] = epistemic_term + extrinsic_term

    def predict_posterior(self, predicted_state: int) -> np.ndarray:
        """Predict posterior beliefs for next state"""
        # Simplified prediction
        # In practice, this would use generative model
        predicted_obs = self.observation_likelihood[predicted_state]
        predicted_posterior = predicted_obs * self.beliefs
        predicted_posterior /= np.sum(predicted_posterior)
        return predicted_posterior

    def calculate_extrinsic_term(self, policy: np.ndarray) -> float:
        """Calculate extrinsic (goal-directed) term of EFE"""
        # Simplified: prefer policies that lead to certain states
        goal_states = self.agent_config.get('goal_states', [0])
        extrinsic_value = 0.0

        for goal_state in goal_states:
            # Probability of reaching goal state
            reach_prob = np.sum(policy[self.environment_state, goal_state])
            extrinsic_value -= reach_prob  # Negative because we minimize EFE

        return extrinsic_value

    def execute_policy(self, policy_idx: int) -> None:
        """Execute selected policy"""
        policy = self.policies[policy_idx]

        # Sample next state according to policy
        next_state_probs = policy[self.environment_state]
        next_state = self.random_state.choice(len(next_state_probs), p=next_state_probs)

        self.environment_state = next_state

    def update_environment(self) -> None:
        """Update environment state (simplified dynamics)"""
        # Random walk with bias toward center
        n_states = len(self.beliefs)
        center = n_states // 2

        if self.environment_state < center:
            # Bias toward moving right
            move_probs = [0.3, 0.7]  # [stay, move_right]
        elif self.environment_state > center:
            # Bias toward moving left
            move_probs = [0.7, 0.3]  # [move_left, stay]
        else:
            move_probs = [0.5, 0.5]  # Equal probability

        move = self.random_state.choice(2, p=move_probs)

        if move == 0 and self.environment_state > 0:
            self.environment_state -= 1
        elif move == 1 and self.environment_state < n_states - 1:
            self.environment_state += 1

    def calculate_observables(self) -> Dict[str, float]:
        """Calculate observable quantities"""
        observables = {
            'belief_entropy': -np.sum(self.beliefs * np.log(self.beliefs + 1e-10)),
            'expected_free_energy_mean': np.mean(self.expected_free_energy),
            'expected_free_energy_std': np.std(self.expected_free_energy),
            'environment_state': float(self.environment_state),
            'dominant_belief': float(np.argmax(self.beliefs))
        }

        return observables

    def get_observable_names(self) -> List[str]:
        """Get observable quantity names"""
        return [
            'belief_entropy',
            'expected_free_energy_mean',
            'expected_free_energy_std',
            'environment_state',
            'dominant_belief'
        ]

    def get_trajectories(self) -> Dict[str, np.ndarray]:
        """Get trajectory data"""
        trajectories = {}

        # Convert observables to trajectories
        for obs_name, values in self.observables.items():
            trajectories[obs_name] = np.array(values)

        # Add belief trajectories
        trajectories['beliefs'] = np.array(self.beliefs_history) if hasattr(self, 'beliefs_history') else np.array([])

        return trajectories

class MultiAgentSimulation(SimulationEngine):
    """Multi-agent Active Inference simulation"""

    def __init__(self, config: SimulationConfig, agent_configs: List[Dict[str, Any]]):
        """Initialize multi-agent simulation"""
        super().__init__(config)
        self.agent_configs = agent_configs
        self.agents: List[ActiveInferenceAgentSimulation] = []

        # Shared environment
        self.shared_environment: Dict[str, Any] = {}

    def initialize_simulation(self) -> None:
        """Initialize multi-agent simulation"""
        # Create individual agent simulations
        for i, agent_config in enumerate(self.agent_configs):
            agent_sim_config = SimulationConfig(
                time_steps=1,  # Agents step individually
                dt=self.config.dt,
                n_agents=1,
                random_seed=self.config.random_seed + i if self.config.random_seed else None
            )

            agent = ActiveInferenceAgentSimulation(agent_sim_config, agent_config)
            agent.initialize_simulation()
            self.agents.append(agent)

        # Initialize shared environment
        self.shared_environment = {
            'resources': np.ones(10),  # Shared resources
            'communication_channels': np.zeros((len(self.agents), len(self.agents))),
            'global_state': np.zeros(5)
        }

    def step_simulation(self) -> None:
        """Execute one multi-agent simulation step"""
        # Update shared environment based on agent actions
        agent_actions = []

        # Collect actions from all agents
        for agent in self.agents:
            agent.step_simulation()
            # Extract action from agent state (simplified)
            action = {
                'resource_consumption': self.random_state.rand(10) * 0.1,
                'communication_signal': self.random_state.rand(len(self.agents))
            }
            agent_actions.append(action)

        # Update shared environment
        self.update_shared_environment(agent_actions)

        # Apply environment feedback to agents
        self.apply_environment_feedback()

    def update_shared_environment(self, agent_actions: List[Dict[str, Any]]) -> None:
        """Update shared environment based on agent actions"""
        total_resource_consumption = np.zeros(10)

        for action in agent_actions:
            total_resource_consumption += action['resource_consumption']

        # Update resources (regeneration minus consumption)
        regeneration_rate = 0.05
        self.shared_environment['resources'] = (
            self.shared_environment['resources'] +
            regeneration_rate - total_resource_consumption
        )
        self.shared_environment['resources'] = np.maximum(0, self.shared_environment['resources'])

        # Update communication matrix
        comm_matrix = np.zeros((len(self.agents), len(self.agents)))
        for i, action in enumerate(agent_actions):
            comm_matrix[i] = action['communication_signal']

        self.shared_environment['communication_channels'] = comm_matrix

    def apply_environment_feedback(self) -> None:
        """Apply environment feedback to agents"""
        # Modify agent beliefs based on shared environment
        resource_abundance = np.mean(self.shared_environment['resources'])

        for agent in self.agents:
            # Agents with scarce resources become more exploratory
            if resource_abundance < 0.5:
                # Increase exploration by flattening belief distribution
                agent.beliefs = np.ones_like(agent.beliefs) / len(agent.beliefs)
            else:
                # Focus beliefs more strongly
                max_idx = np.argmax(agent.beliefs)
                agent.beliefs = np.ones_like(agent.beliefs) * 0.1
                agent.beliefs[max_idx] = 0.9

    def calculate_observables(self) -> Dict[str, float]:
        """Calculate multi-agent observables"""
        observables = {
            'resource_abundance': np.mean(self.shared_environment['resources']),
            'communication_density': np.mean(self.shared_environment['communication_channels']),
            'average_belief_entropy': np.mean([
                -np.sum(agent.beliefs * np.log(agent.beliefs + 1e-10))
                for agent in self.agents
            ]),
            'consensus_level': self.calculate_consensus_level()
        }

        return observables

    def calculate_consensus_level(self) -> float:
        """Calculate consensus level among agents"""
        if not self.agents:
            return 0.0

        # Compare dominant beliefs
        dominant_beliefs = [np.argmax(agent.beliefs) for agent in self.agents]
        most_common_belief = max(set(dominant_beliefs), key=dominant_beliefs.count)
        consensus = dominant_beliefs.count(most_common_belief) / len(dominant_beliefs)

        return consensus

    def get_observable_names(self) -> List[str]:
        """Get observable names"""
        return [
            'resource_abundance',
            'communication_density',
            'average_belief_entropy',
            'consensus_level'
        ]

    def get_trajectories(self) -> Dict[str, np.ndarray]:
        """Get trajectory data for all agents"""
        trajectories = {}

        # Aggregate observables
        for obs_name in self.get_observable_names():
            trajectories[obs_name] = np.array(self.observables[obs_name])

        # Individual agent trajectories
        for i, agent in enumerate(self.agents):
            agent_trajectories = agent.get_trajectories()
            for traj_name, traj_data in agent_trajectories.items():
                trajectories[f"agent_{i}_{traj_name}"] = traj_data

        return trajectories

class SimulationExperimentManager:
    """Manager for running simulation experiments"""

    def __init__(self, config: Dict[str, Any]):
        """Initialize experiment manager"""
        self.config = config
        self.logger = logging.getLogger('SimulationExperimentManager')
        self.experiments: Dict[str, Dict[str, Any]] = {}

    def create_experiment(self, experiment_name: str, simulation_class: type,
                         parameter_ranges: Dict[str, Any], n_runs: int = 10) -> str:
        """Create parameter sweep experiment"""
        experiment_id = f"exp_{experiment_name}_{len(self.experiments)}"

        experiment = {
            'experiment_id': experiment_id,
            'name': experiment_name,
            'simulation_class': simulation_class,
            'parameter_ranges': parameter_ranges,
            'n_runs': n_runs,
            'status': 'created',
            'results': []
        }

        self.experiments[experiment_id] = experiment
        self.logger.info(f"Created experiment: {experiment_id}")

        return experiment_id

    async def run_experiment(self, experiment_id: str) -> Dict[str, Any]:
        """Run simulation experiment"""
        if experiment_id not in self.experiments:
            raise ValueError(f"Experiment {experiment_id} not found")

        experiment = self.experiments[experiment_id]
        experiment['status'] = 'running'

        self.logger.info(f"Starting experiment: {experiment_id}")

        # Generate parameter combinations
        param_combinations = self.generate_parameter_combinations(
            experiment['parameter_ranges']
        )

        # Run simulations for each parameter combination
        results = []

        for i, params in enumerate(param_combinations):
            self.logger.debug(f"Running parameter set {i+1}/{len(param_combinations)}")

            # Run multiple runs for statistical significance
            run_results = []
            for run in range(experiment['n_runs']):
                result = await self.run_simulation_with_params(
                    experiment['simulation_class'], params
                )
                run_results.append(result)

            # Aggregate results for this parameter set
            aggregated_result = self.aggregate_run_results(run_results, params)
            results.append(aggregated_result)

        experiment['results'] = results
        experiment['status'] = 'completed'

        # Analyze results
        analysis = self.analyze_experiment_results(results)

        self.logger.info(f"Experiment completed: {experiment_id}")

        return {
            'experiment_id': experiment_id,
            'results': results,
            'analysis': analysis
        }

    def generate_parameter_combinations(self, parameter_ranges: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate all parameter combinations for experiment"""
        # Simple implementation - in practice, use itertools.product or similar
        combinations = []

        # For now, just return a few combinations
        n_combinations = 5  # Limited for testing

        for i in range(n_combinations):
            combination = {}
            for param_name, param_range in parameter_ranges.items():
                if isinstance(param_range, (list, tuple)) and len(param_range) == 2:
                    # Linear interpolation
                    min_val, max_val = param_range
                    value = min_val + (max_val - min_val) * (i / (n_combinations - 1))
                    combination[param_name] = value
                else:
                    # Use fixed value
                    combination[param_name] = param_range

            combinations.append(combination)

        return combinations

    async def run_simulation_with_params(self, simulation_class: type,
                                       params: Dict[str, Any]) -> SimulationResult:
        """Run simulation with specific parameters"""
        # Create simulation config from parameters
        sim_config = SimulationConfig(
            time_steps=params.get('time_steps', 1000),
            dt=params.get('dt', 0.01),
            n_agents=params.get('n_agents', 1)
        )

        # Create and run simulation
        simulation = simulation_class(sim_config)
        result = simulation.run_simulation()

        return result

    def aggregate_run_results(self, run_results: List[SimulationResult],
                            params: Dict[str, Any]) -> Dict[str, Any]:
        """Aggregate results from multiple runs"""
        if not run_results:
            return {}

        # Extract observables from all runs
        observable_names = run_results[0].observables.keys()
        aggregated_observables = {}

        for obs_name in observable_names:
            values = []
            for result in run_results:
                if obs_name in result.observables:
                    obs_values = result.observables[obs_name]
                    if len(obs_values) > 0:
                        values.append(np.mean(obs_values))  # Use mean of time series

            if values:
                aggregated_observables[f"{obs_name}_mean"] = np.mean(values)
                aggregated_observables[f"{obs_name}_std"] = np.std(values)

        # Aggregate execution times
        execution_times = [r.execution_time for r in run_results]

        return {
            'parameters': params,
            'aggregated_observables': aggregated_observables,
            'execution_time_mean': np.mean(execution_times),
            'execution_time_std': np.std(execution_times),
            'n_runs': len(run_results)
        }

    def analyze_experiment_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze experiment results for patterns and insights"""
        analysis = {
            'parameter_effects': {},
            'significant_findings': [],
            'recommendations': []
        }

        if len(results) < 2:
            return analysis

        # Analyze parameter effects
        param_names = list(results[0]['parameters'].keys())

        for param_name in param_names:
            param_values = [r['parameters'][param_name] for r in results]
            observable_names = [k for k in results[0]['aggregated_observables'].keys()
                              if k.endswith('_mean')]

            for obs_name in observable_names:
                obs_values = [r['aggregated_observables'][obs_name] for r in results]

                # Calculate correlation
                if len(param_values) > 1 and len(obs_values) > 1:
                    correlation = np.corrcoef(param_values, obs_values)[0, 1]
                    if abs(correlation) > 0.5:  # Significant correlation
                        analysis['parameter_effects'][f"{param_name}_{obs_name}"] = {
                            'correlation': correlation,
                            'strength': 'strong' if abs(correlation) > 0.7 else 'moderate'
                        }

        return analysis

    def export_experiment_results(self, experiment_id: str, format: str = 'json') -> str:
        """Export experiment results"""
        if experiment_id not in self.experiments:
            raise ValueError(f"Experiment {experiment_id} not found")

        experiment = self.experiments[experiment_id]

        if format == 'json':
            return json.dumps(experiment, indent=2, default=str)
        else:
            # Simple text format
            output = f"Experiment: {experiment['name']}\n"
            output += f"Status: {experiment['status']}\n"
            output += f"Results: {len(experiment['results'])} parameter sets\n"
            return output

class TestSimulationFramework:
    """Tests for simulation framework"""

    @pytest.fixture
    def simulation_config(self):
        """Create simulation config for testing"""
        return SimulationConfig(
            time_steps=100,
            dt=0.1,
            n_agents=1,
            random_seed=42
        )

    @pytest.fixture
    def agent_config(self):
        """Create agent config for testing"""
        return {
            'n_states': 5,
            'n_policies': 3,
            'n_observations': 4
        }

    def test_active_inference_simulation(self, simulation_config, agent_config):
        """Test Active Inference agent simulation"""
        simulation = ActiveInferenceAgentSimulation(simulation_config, agent_config)
        result = simulation.run_simulation()

        # Verify result structure
        assert result.simulation_id.startswith('sim_')
        assert result.config == simulation_config
        assert 'belief_entropy' in result.observables
        assert 'expected_free_energy_mean' in result.observables
        assert result.execution_time > 0

        # Verify observables have correct length
        expected_steps = simulation_config.time_steps // simulation_config.logging_interval
        assert len(result.observables['belief_entropy']) == expected_steps

    def test_multi_agent_simulation(self, simulation_config):
        """Test multi-agent simulation"""
        agent_configs = [
            {'n_states': 4, 'n_policies': 2, 'n_observations': 3},
            {'n_states': 4, 'n_policies': 2, 'n_observations': 3}
        ]

        simulation = MultiAgentSimulation(simulation_config, agent_configs)
        result = simulation.run_simulation()

        # Verify multi-agent observables
        assert 'resource_abundance' in result.observables
        assert 'communication_density' in result.observables
        assert 'average_belief_entropy' in result.observables
        assert 'consensus_level' in result.observables

    @pytest.mark.asyncio
    async def test_experiment_manager(self):
        """Test experiment management"""
        config = {'parallel_execution': False}
        experiment_manager = SimulationExperimentManager(config)

        # Create simple experiment
        experiment_id = experiment_manager.create_experiment(
            'test_experiment',
            ActiveInferenceAgentSimulation,
            {
                'time_steps': [500, 1000],
                'dt': [0.01, 0.05]
            },
            n_runs=2
        )

        # Run experiment
        results = await experiment_manager.run_experiment(experiment_id)

        # Verify results
        assert results['experiment_id'] == experiment_id
        assert len(results['results']) > 0
        assert 'analysis' in results

    def test_simulation_statistics(self, simulation_config, agent_config):
        """Test simulation statistics calculation"""
        simulation = ActiveInferenceAgentSimulation(simulation_config, agent_config)
        result = simulation.run_simulation()

        # Check that statistics are calculated
        assert 'belief_entropy_mean' in result.statistics
        assert 'belief_entropy_std' in result.statistics

        # Check convergence metrics if enough data
        if len(result.observables['belief_entropy']) > 100:
            assert 'belief_entropy_mean_convergence' in result.convergence_metrics

    def test_simulation_reset(self, simulation_config, agent_config):
        """Test simulation reset functionality"""
        simulation = ActiveInferenceAgentSimulation(simulation_config, agent_config)

        # Run simulation
        result1 = simulation.run_simulation()

        # Reset and run again
        simulation.reset_simulation()
        result2 = simulation.run_simulation()

        # Results should be different due to random seed
        # But structure should be the same
        assert result1.config == result2.config
        assert set(result1.observables.keys()) == set(result2.observables.keys())
```

---

**"Active Inference for, with, by Generative AI"** - Building comprehensive research tools and experimentation frameworks that enable rigorous scientific investigation and validation of Active Inference principles.
