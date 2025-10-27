# Information Theory Analysis - Agent Documentation

This document provides comprehensive guidance for AI agents and contributors working within the Information Theory Analysis module of the Active Inference Knowledge Environment. It outlines information-theoretic methodologies, implementation patterns, and best practices for rigorous information-theoretic analysis throughout the research lifecycle.

## Information Theory Module Overview

The Information Theory Analysis module provides a comprehensive framework for information-theoretic analysis in Active Inference research. It includes entropy measures, mutual information calculations, divergence measures, complexity analysis, and causal analysis tools for understanding information processing in intelligent systems.

## Core Responsibilities

### Entropy Measures & Analysis
- **Shannon Entropy**: Classical information entropy calculations
- **Differential Entropy**: Continuous variable entropy measures
- **Relative Entropy**: Entropy measures for system comparison
- **Conditional Entropy**: Entropy under conditional constraints
- **Joint Entropy**: Multi-variable entropy calculations

### Mutual Information Analysis
- **Mutual Information**: Information shared between variables
- **Conditional Mutual Information**: Conditional information measures
- **Multi-information**: Information in multiple variable systems
- **Transfer Entropy**: Information transfer between systems
- **Partial Information**: Decomposition of information measures

### Divergence Measures
- **KL Divergence**: Kullback-Leibler divergence calculations
- **Jensen-Shannon Divergence**: Symmetric divergence measures
- **Wasserstein Distance**: Optimal transport-based distances
- **f-Divergences**: Generalized divergence measures
- **Statistical Distances**: Information geometry distances

### Complexity Analysis
- **Fractal Dimensions**: System complexity measures
- **Lyapunov Exponents**: Chaotic system analysis
- **Sample Entropy**: Time series complexity measures
- **Permutation Entropy**: Ordinal pattern complexity
- **Multiscale Entropy**: Multi-scale complexity analysis

### Causal Analysis
- **Granger Causality**: Temporal causal relationships
- **Transfer Entropy**: Information flow causality
- **Causal Graphs**: Information-theoretic causal networks
- **Causal Discovery**: Automated causal relationship discovery
- **Causal Strength**: Quantifying causal influence

## Development Workflows

### Information Theory Development Process
1. **Requirements Analysis**: Analyze information-theoretic requirements
2. **Theory Research**: Research information theory foundations
3. **Method Design**: Design information-theoretic algorithms
4. **Mathematical Implementation**: Implement mathematically correct methods
5. **Numerical Validation**: Validate numerical accuracy and stability
6. **Benchmarking**: Test against established benchmarks
7. **Testing**: Comprehensive testing including edge cases
8. **Documentation**: Create comprehensive mathematical documentation
9. **Review**: Submit for mathematical and scientific review
10. **Integration**: Integrate with analysis and research frameworks

### Entropy Implementation Workflow
1. **Theory Review**: Review entropy theory and applications
2. **Algorithm Selection**: Choose appropriate entropy algorithms
3. **Implementation**: Implement entropy calculation methods
4. **Validation**: Validate against theoretical predictions
5. **Optimization**: Optimize for computational efficiency
6. **Testing**: Test with various data types and distributions
7. **Documentation**: Document mathematical formulations
8. **Integration**: Integrate with data analysis pipelines

### Mutual Information Implementation
1. **Method Selection**: Choose mutual information estimation method
2. **Implementation**: Implement mutual information calculations
3. **Bias Correction**: Implement bias correction techniques
4. **Validation**: Validate against known distributions
5. **Performance Testing**: Test computational performance
6. **Edge Case Handling**: Handle special cases and errors
7. **Documentation**: Document estimation methods
8. **Example Creation**: Create usage examples

## Quality Standards

### Mathematical Quality Standards
- **Theoretical Correctness**: Mathematically correct implementations
- **Numerical Accuracy**: Accurate numerical calculations
- **Convergence**: Proper algorithm convergence
- **Stability**: Numerical stability of algorithms
- **Validation**: Validated against theoretical benchmarks

### Scientific Quality Standards
- **Method Validation**: Validated against established methods
- **Reproducibility**: Reproducible analysis results
- **Documentation**: Complete mathematical documentation
- **Standards Compliance**: Compliance with information theory standards
- **Interpretation**: Proper interpretation of results

### Code Quality Standards
- **Algorithmic Correctness**: Correct algorithm implementation
- **Performance**: Efficient computational performance
- **Testing**: Comprehensive testing coverage
- **Documentation**: Clear mathematical documentation
- **Maintainability**: Maintainable and extensible code

## Implementation Patterns

### Entropy Analysis Framework
```python
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union, Tuple
import numpy as np
from scipy import stats, special
from dataclasses import dataclass
import logging

@dataclass
class EntropyResult:
    """Entropy calculation result"""
    entropy_value: float
    method: str
    parameters: Dict[str, Any]
    confidence_interval: Optional[Tuple[float, float]] = None
    bias_correction: Optional[float] = None
    validation_score: Optional[float] = None

@dataclass
class EntropyConfig:
    """Entropy calculation configuration"""
    method: str  # shannon, differential, relative, conditional
    base: float = 2.0  # logarithm base
    bias_correction: bool = True
    confidence_level: float = 0.95
    sample_size: Optional[int] = None

class BaseEntropyMeasure(ABC):
    """Base class for entropy measures"""

    def __init__(self, config: EntropyConfig = None):
        """Initialize entropy measure"""
        self.config = config or EntropyConfig()
        self.method_name = self.__class__.__name__
        self.logger = logging.getLogger(f"entropy.{self.method_name}")

    @abstractmethod
    def calculate(self, data: np.ndarray, **kwargs) -> EntropyResult:
        """Calculate entropy measure"""
        pass

    @abstractmethod
    def validate_input(self, data: np.ndarray) -> List[str]:
        """Validate input data"""
        pass

    def estimate_bias(self, data: np.ndarray, entropy_value: float) -> float:
        """Estimate bias in entropy calculation"""
        if not self.config.bias_correction:
            return 0.0

        n = len(data)
        if self.config.method == 'shannon':
            # Miller-Madow bias correction for Shannon entropy
            k = len(np.unique(data))
            return (k - 1) / (2 * n)
        elif self.config.method == 'differential':
            # Bias correction for differential entropy
            return 1 / (2 * n)

        return 0.0

    def calculate_confidence_interval(self, data: np.ndarray, entropy_value: float) -> Tuple[float, float]:
        """Calculate confidence interval for entropy"""
        if not self.config.sample_size:
            return (entropy_value, entropy_value)

        n = self.config.sample_size
        std_error = np.sqrt(self.estimate_variance(data, entropy_value) / n)

        z_score = stats.norm.ppf(1 - (1 - self.config.confidence_level) / 2)
        margin = z_score * std_error

        return (entropy_value - margin, entropy_value + margin)

    def estimate_variance(self, data: np.ndarray, entropy_value: float) -> float:
        """Estimate variance of entropy estimator"""
        # Simplified variance estimation
        n = len(data)
        if self.config.method == 'shannon':
            return entropy_value * (1 + np.log(n)) / n
        else:
            return entropy_value / n

class ShannonEntropy(BaseEntropyMeasure):
    """Shannon entropy implementation"""

    def calculate(self, data: np.ndarray, **kwargs) -> EntropyResult:
        """Calculate Shannon entropy"""
        # Validate input
        issues = self.validate_input(data)
        if issues:
            raise ValueError(f"Input validation failed: {issues}")

        # Calculate probabilities
        if data.ndim == 1:
            # Discrete data
            values, counts = np.unique(data, return_counts=True)
            probabilities = counts / len(data)
        else:
            # Continuous data - discretize
            probabilities = self.discretize_continuous(data)

        # Remove zero probabilities
        probabilities = probabilities[probabilities > 0]

        if len(probabilities) == 0:
            entropy_value = 0.0
        else:
            # Calculate Shannon entropy
            entropy_value = -np.sum(probabilities * np.log(probabilities)) / np.log(self.config.base)

        # Apply bias correction
        bias = self.estimate_bias(data, entropy_value)
        entropy_value -= bias

        # Calculate confidence interval
        confidence_interval = self.calculate_confidence_interval(data, entropy_value)

        return EntropyResult(
            entropy_value=entropy_value,
            method=self.method_name,
            parameters={'base': self.config.base, 'bias_correction': self.config.bias_correction},
            confidence_interval=confidence_interval,
            bias_correction=bias,
            validation_score=self.validate_entropy_value(data, entropy_value)
        )

    def validate_input(self, data: np.ndarray) -> List[str]:
        """Validate Shannon entropy input"""
        issues = []

        if not isinstance(data, np.ndarray):
            issues.append("Input must be numpy array")

        if len(data) == 0:
            issues.append("Input data cannot be empty")

        if np.any(data < 0):
            issues.append("Data values must be non-negative")

        return issues

    def discretize_continuous(self, data: np.ndarray) -> np.ndarray:
        """Discretize continuous data for entropy calculation"""
        # Simple binning approach
        n_bins = min(50, len(data) // 10)
        hist, _ = np.histogram(data, bins=n_bins)
        return hist / hist.sum()

    def validate_entropy_value(self, data: np.ndarray, entropy_value: float) -> float:
        """Validate entropy value against theoretical bounds"""
        n_unique = len(np.unique(data))

        # Check against theoretical bounds
        max_entropy = np.log(n_unique) / np.log(self.config.base)

        if entropy_value < 0:
            return 0.0  # Invalid
        elif entropy_value > max_entropy * 1.1:
            return 0.5  # Suspicious
        else:
            return 1.0  # Valid

class DifferentialEntropy(BaseEntropyMeasure):
    """Differential entropy for continuous variables"""

    def calculate(self, data: np.ndarray, **kwargs) -> EntropyResult:
        """Calculate differential entropy"""
        # Validate input
        issues = self.validate_input(data)
        if issues:
            raise ValueError(f"Input validation failed: {issues}")

        # Estimate probability density
        kde = stats.gaussian_kde(data)

        # Calculate differential entropy
        entropy_value = self.calculate_differential_entropy(data, kde)

        # Apply bias correction
        bias = self.estimate_bias(data, entropy_value)
        entropy_value -= bias

        # Calculate confidence interval
        confidence_interval = self.calculate_confidence_interval(data, entropy_value)

        return EntropyResult(
            entropy_value=entropy_value,
            method=self.method_name,
            parameters={'bandwidth': kde.factor, 'bias_correction': self.config.bias_correction},
            confidence_interval=confidence_interval,
            bias_correction=bias,
            validation_score=self.validate_differential_entropy(data, entropy_value)
        )

    def calculate_differential_entropy(self, data: np.ndarray, kde: stats.gaussian_kde) -> float:
        """Calculate differential entropy from KDE"""
        # Use Monte Carlo integration
        n_samples = 10000
        test_points = np.linspace(data.min(), data.max(), n_samples)

        # Calculate log likelihood
        log_likelihood = np.log(kde(test_points) + 1e-12)

        # Integrate to get differential entropy
        integral = np.trapz(log_likelihood, test_points)
        entropy = -integral / n_samples

        return entropy

    def validate_input(self, data: np.ndarray) -> List[str]:
        """Validate differential entropy input"""
        issues = []

        if not isinstance(data, np.ndarray):
            issues.append("Input must be numpy array")

        if len(data) < 10:
            issues.append("Need at least 10 samples for reliable differential entropy")

        if len(np.unique(data)) < 3:
            issues.append("Data should have sufficient variation")

        return issues

    def validate_differential_entropy(self, data: np.ndarray, entropy_value: float) -> float:
        """Validate differential entropy value"""
        # Check if value is reasonable
        data_range = data.max() - data.min()
        data_std = np.std(data)

        if entropy_value > np.log(data_range) + 1:
            return 0.5  # May be overestimated
        elif entropy_value < -np.log(data_std) - 1:
            return 0.5  # May be underestimated
        else:
            return 1.0  # Reasonable

class ConditionalEntropy(BaseEntropyMeasure):
    """Conditional entropy calculation"""

    def calculate(self, data_x: np.ndarray, data_y: np.ndarray, **kwargs) -> EntropyResult:
        """Calculate conditional entropy H(X|Y)"""
        # Validate inputs
        issues = self.validate_input(data_x, data_y)
        if issues:
            raise ValueError(f"Input validation failed: {issues}")

        # Calculate joint and marginal entropies
        joint_entropy = self.calculate_joint_entropy(data_x, data_y)
        marginal_entropy_y = self.calculate_marginal_entropy(data_y)

        # Conditional entropy = H(X,Y) - H(Y)
        entropy_value = joint_entropy - marginal_entropy_y

        # Apply bias correction
        bias = self.estimate_bias(np.column_stack([data_x, data_y]), entropy_value)
        entropy_value -= bias

        # Calculate confidence interval
        confidence_interval = self.calculate_confidence_interval(np.column_stack([data_x, data_y]), entropy_value)

        return EntropyResult(
            entropy_value=entropy_value,
            method=self.method_name,
            parameters={'base': self.config.base, 'bias_correction': self.config.bias_correction},
            confidence_interval=confidence_interval,
            bias_correction=bias,
            validation_score=self.validate_conditional_entropy(data_x, data_y, entropy_value)
        )

    def calculate_joint_entropy(self, data_x: np.ndarray, data_y: np.ndarray) -> float:
        """Calculate joint entropy H(X,Y)"""
        # Combine data
        joint_data = np.column_stack([data_x, data_y])

        # Use Shannon entropy on joint distribution
        shannon = ShannonEntropy(self.config)
        return shannon.calculate(joint_data.ravel()).entropy_value

    def calculate_marginal_entropy(self, data: np.ndarray) -> float:
        """Calculate marginal entropy"""
        shannon = ShannonEntropy(self.config)
        return shannon.calculate(data).entropy_value

    def validate_input(self, data_x: np.ndarray, data_y: np.ndarray) -> List[str]:
        """Validate conditional entropy inputs"""
        issues = []

        if len(data_x) != len(data_y):
            issues.append("Data arrays must have same length")

        if len(data_x) < 10:
            issues.append("Need sufficient sample size for reliable estimation")

        return issues

    def validate_conditional_entropy(self, data_x: np.ndarray, data_y: np.ndarray, entropy_value: float) -> float:
        """Validate conditional entropy value"""
        # Check bounds: 0 ≤ H(X|Y) ≤ H(X)
        marginal_entropy = self.calculate_marginal_entropy(data_x)

        if entropy_value < 0:
            return 0.0  # Invalid
        elif entropy_value > marginal_entropy * 1.1:
            return 0.5  # Suspicious
        else:
            return 1.0  # Valid

class EntropyAnalysisFramework:
    """Comprehensive entropy analysis framework"""

    def __init__(self, config: EntropyConfig = None):
        """Initialize entropy analysis framework"""
        self.config = config or EntropyConfig()
        self.measures: Dict[str, BaseEntropyMeasure] = {}
        self.register_default_measures()

    def register_default_measures(self) -> None:
        """Register default entropy measures"""
        self.register_measure('shannon', ShannonEntropy(self.config))
        self.register_measure('differential', DifferentialEntropy(self.config))
        self.register_measure('conditional', ConditionalEntropy(self.config))

    def register_measure(self, name: str, measure: BaseEntropyMeasure) -> None:
        """Register entropy measure"""
        self.measures[name] = measure

    def analyze_entropy(self, data: np.ndarray, measure: str = 'auto',
                       conditional_on: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Perform comprehensive entropy analysis"""
        results = {
            'data_summary': self.summarize_data(data),
            'entropy_results': {},
            'comparisons': {},
            'interpretations': []
        }

        # Select appropriate measures
        if measure == 'auto':
            measure_list = self.select_measures(data, conditional_on)
        else:
            measure_list = [measure]

        # Calculate entropies
        for measure_name in measure_list:
            if measure_name in self.measures:
                try:
                    if conditional_on is not None and measure_name == 'conditional':
                        entropy_result = self.measures[measure_name].calculate(data, conditional_on)
                    else:
                        entropy_result = self.measures[measure_name].calculate(data)

                    results['entropy_results'][measure_name] = entropy_result.__dict__

                except Exception as e:
                    results['entropy_results'][measure_name] = {'error': str(e)}

        # Generate comparisons
        results['comparisons'] = self.compare_entropy_measures(results['entropy_results'])

        # Generate interpretations
        results['interpretations'] = self.interpret_entropy_results(results)

        return results

    def summarize_data(self, data: np.ndarray) -> Dict[str, Any]:
        """Summarize input data"""
        return {
            'shape': data.shape,
            'dtype': str(data.dtype),
            'mean': np.mean(data),
            'std': np.std(data),
            'min': np.min(data),
            'max': np.max(data),
            'unique_values': len(np.unique(data))
        }

    def select_measures(self, data: np.ndarray, conditional_on: Optional[np.ndarray] = None) -> List[str]:
        """Select appropriate entropy measures"""
        measures = []

        # Always include Shannon entropy
        measures.append('shannon')

        # Add differential entropy for continuous data
        if data.dtype in [np.float32, np.float64]:
            measures.append('differential')

        # Add conditional entropy if conditioning data provided
        if conditional_on is not None:
            measures.append('conditional')

        return measures

    def compare_entropy_measures(self, entropy_results: Dict[str, Any]) -> Dict[str, Any]:
        """Compare different entropy measures"""
        comparisons = {}

        measure_names = list(entropy_results.keys())

        for i, measure1 in enumerate(measure_names):
            for measure2 in measure_names[i+1:]:
                if 'error' not in entropy_results[measure1] and 'error' not in entropy_results[measure2]:
                    diff = abs(entropy_results[measure1]['entropy_value'] - entropy_results[measure2]['entropy_value'])
                    comparisons[f"{measure1}_vs_{measure2}"] = {
                        'difference': diff,
                        'relative_difference': diff / max(entropy_results[measure1]['entropy_value'],
                                                        entropy_results[measure2]['entropy_value'])
                    }

        return comparisons

    def interpret_entropy_results(self, results: Dict[str, Any]) -> List[str]:
        """Generate interpretations of entropy results"""
        interpretations = []

        for measure_name, measure_result in results['entropy_results'].items():
            if 'error' not in measure_result:
                entropy_value = measure_result['entropy_value']

                if measure_name == 'shannon':
                    if entropy_value == 0:
                        interpretations.append("System has no uncertainty (deterministic)")
                    elif entropy_value > 3:
                        interpretations.append("System has high uncertainty")
                    else:
                        interpretations.append("System has moderate uncertainty")

                elif measure_name == 'differential':
                    if entropy_value > 2:
                        interpretations.append("Continuous system has high differential entropy")
                    else:
                        interpretations.append("Continuous system has low differential entropy")

        return interpretations
```

### Mutual Information Framework
```python
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from scipy import stats

class MutualInformationAnalyzer:
    """Comprehensive mutual information analysis"""

    def __init__(self):
        """Initialize mutual information analyzer"""
        self.methods: Dict[str, callable] = {}
        self.register_default_methods()

    def register_default_methods(self) -> None:
        """Register default mutual information methods"""
        self.methods['histogram'] = self.histogram_mutual_information
        self.methods['kernel'] = self.kernel_mutual_information
        self.methods['kraskov'] = self.kraskov_mutual_information

    def calculate_mutual_information(self, data_x: np.ndarray, data_y: np.ndarray,
                                   method: str = 'histogram', **kwargs) -> float:
        """Calculate mutual information between variables"""
        if method not in self.methods:
            raise ValueError(f"Unknown mutual information method: {method}")

        return self.methods[method](data_x, data_y, **kwargs)

    def histogram_mutual_information(self, data_x: np.ndarray, data_y: np.ndarray,
                                   bins: int = 10) -> float:
        """Calculate mutual information using histogram method"""
        # Create 2D histogram
        hist_2d, x_edges, y_edges = np.histogram2d(data_x, data_y, bins=bins)

        # Calculate probabilities
        hist_2d = hist_2d / hist_2d.sum()

        # Marginal probabilities
        hist_x = hist_2d.sum(axis=1)
        hist_y = hist_2d.sum(axis=0)

        # Joint probabilities are hist_2d
        # Calculate mutual information
        mi = 0.0
        for i in range(hist_2d.shape[0]):
            for j in range(hist_2d.shape[1]):
                if hist_2d[i, j] > 0:
                    mi += hist_2d[i, j] * np.log2(hist_2d[i, j] / (hist_x[i] * hist_y[j]))

        return mi

    def kernel_mutual_information(self, data_x: np.ndarray, data_y: np.ndarray,
                                bandwidth: Optional[float] = None) -> float:
        """Calculate mutual information using kernel density estimation"""
        # Combine data
        joint_data = np.column_stack([data_x, data_y])

        # Estimate joint density
        kde_joint = stats.gaussian_kde(joint_data.T)

        # Estimate marginal densities
        kde_x = stats.gaussian_kde(data_x)
        kde_y = stats.gaussian_kde(data_y)

        # Monte Carlo integration
        n_samples = 10000
        test_points = np.random.rand(n_samples, 2)
        test_points[:, 0] = test_points[:, 0] * (data_x.max() - data_x.min()) + data_x.min()
        test_points[:, 1] = test_points[:, 1] * (data_y.max() - data_y.min()) + data_y.min()

        # Calculate densities
        joint_density = kde_joint(test_points.T)
        marginal_x = kde_x(test_points[:, 0])
        marginal_y = kde_y(test_points[:, 1])

        # Calculate mutual information
        mi_values = joint_density * np.log2(joint_density / (marginal_x * marginal_y))
        mi = np.mean(mi_values)

        return max(0, mi)  # Ensure non-negative

    def kraskov_mutual_information(self, data_x: np.ndarray, data_y: np.ndarray,
                                 k: int = 3) -> float:
        """Calculate mutual information using Kraskov method"""
        # Simplified Kraskov-style estimation
        from sklearn.neighbors import NearestNeighbors

        n = len(data_x)
        joint_data = np.column_stack([data_x, data_y])

        # Find k-nearest neighbors in joint space
        nbrs_joint = NearestNeighbors(n_neighbors=k+1, metric='chebyshev').fit(joint_data)
        distances_joint, _ = nbrs_joint.kneighbors(joint_data)

        # Find k-nearest neighbors in marginal spaces
        nbrs_x = NearestNeighbors(n_neighbors=k+1, metric='chebyshev').fit(data_x.reshape(-1, 1))
        distances_x, _ = nbrs_x.kneighbors(data_x.reshape(-1, 1))

        nbrs_y = NearestNeighbors(n_neighbors=k+1, metric='chebyshev').fit(data_y.reshape(-1, 1))
        distances_y, _ = nbrs_y.kneighbors(data_y.reshape(-1, 1))

        # Calculate mutual information
        mi = 0.0
        for i in range(n):
            # Use maximum norm distances
            epsilon_joint = distances_joint[i, k]
            epsilon_x = distances_x[i, k]
            epsilon_y = distances_y[i, k]

            # Count points within epsilon_joint in marginal spaces
            nx = np.sum(distances_x[:, k] <= epsilon_joint)
            ny = np.sum(distances_y[:, k] <= epsilon_joint)

            mi += np.log2(k) - np.log2(nx) - np.log2(ny) + np.log2(n - 1)

        return mi / n

    def transfer_entropy(self, source: np.ndarray, target: np.ndarray, lag: int = 1) -> float:
        """Calculate transfer entropy from source to target"""
        # Simplified transfer entropy calculation
        if len(source) != len(target):
            raise ValueError("Source and target must have same length")

        n = len(source) - lag

        # Calculate joint and conditional entropies
        target_current = target[lag:]
        target_past = target[:-lag]
        source_past = source[:-lag]

        # H(target_current | target_past)
        conditional_entropy = self.calculate_conditional_entropy(target_current, target_past)

        # H(target_current | target_past, source_past)
        joint_condition = np.column_stack([target_past, source_past])
        joint_conditional_entropy = self.calculate_conditional_entropy(target_current, joint_condition)

        # Transfer entropy = H(target_current | target_past) - H(target_current | target_past, source_past)
        te = conditional_entropy - joint_conditional_entropy

        return max(0, te)

    def calculate_conditional_entropy(self, target: np.ndarray, condition: np.ndarray) -> float:
        """Calculate conditional entropy H(target | condition)"""
        if condition.ndim == 1:
            condition = condition.reshape(-1, 1)

        # Use histogram method for simplicity
        if condition.shape[1] == 1:
            return self.histogram_mutual_information(target, condition.ravel())
        else:
            # For multivariate conditioning
            return 0.0  # Simplified

    def analyze_information_flow(self, time_series: np.ndarray, max_lag: int = 10) -> Dict[str, Any]:
        """Analyze information flow in time series"""
        n_variables = time_series.shape[1] if time_series.ndim > 1 else 1

        if n_variables == 1:
            # Single time series - analyze self-dependencies
            return self.analyze_self_information(time_series.ravel(), max_lag)
        else:
            # Multiple time series - analyze interdependencies
            return self.analyze_cross_information(time_series, max_lag)

    def analyze_self_information(self, series: np.ndarray, max_lag: int) -> Dict[str, Any]:
        """Analyze self-information in single time series"""
        results = {
            'lags': list(range(1, max_lag + 1)),
            'auto_mutual_information': [],
            'transfer_entropy': []
        }

        for lag in range(1, max_lag + 1):
            # Calculate mutual information with lag
            mi = self.calculate_mutual_information(series[:-lag], series[lag:])
            results['auto_mutual_information'].append(mi)

        return results

    def analyze_cross_information(self, time_series: np.ndarray, max_lag: int) -> Dict[str, Any]:
        """Analyze cross-information between multiple time series"""
        n_vars = time_series.shape[1]
        results = {
            'variables': list(range(n_vars)),
            'mutual_information_matrix': np.zeros((n_vars, n_vars)),
            'transfer_entropy_matrix': np.zeros((n_vars, n_vars, max_lag))
        }

        for i in range(n_vars):
            for j in range(n_vars):
                if i != j:
                    # Calculate mutual information
                    mi = self.calculate_mutual_information(time_series[:, i], time_series[:, j])
                    results['mutual_information_matrix'][i, j] = mi

                    # Calculate transfer entropy
                    for lag in range(max_lag):
                        te = self.transfer_entropy(time_series[:, i], time_series[:, j], lag + 1)
                        results['transfer_entropy_matrix'][i, j, lag] = te

        return results
```

### Divergence Measures Framework
```python
from typing import Dict, Any, List, Optional
import numpy as np
from scipy import special

class DivergenceAnalyzer:
    """Comprehensive divergence analysis"""

    def __init__(self):
        """Initialize divergence analyzer"""
        self.divergence_measures: Dict[str, callable] = {}
        self.register_default_measures()

    def register_default_measures(self) -> None:
        """Register default divergence measures"""
        self.divergence_measures['kl_divergence'] = self.kl_divergence
        self.divergence_measures['js_divergence'] = self.js_divergence
        self.divergence_measures['wasserstein'] = self.wasserstein_distance
        self.divergence_measures['hellinger'] = self.hellinger_distance

    def calculate_divergence(self, distribution_p: np.ndarray, distribution_q: np.ndarray,
                           measure: str = 'kl_divergence', **kwargs) -> float:
        """Calculate divergence between distributions"""
        if measure not in self.divergence_measures:
            raise ValueError(f"Unknown divergence measure: {measure}")

        return self.divergence_measures[measure](distribution_p, distribution_q, **kwargs)

    def kl_divergence(self, p: np.ndarray, q: np.ndarray, epsilon: float = 1e-12) -> float:
        """Calculate Kullback-Leibler divergence"""
        # Add small epsilon to avoid log(0)
        p = p + epsilon
        q = q + epsilon

        # Normalize
        p = p / p.sum()
        q = q / q.sum()

        # Calculate KL divergence
        return np.sum(p * np.log2(p / q))

    def js_divergence(self, p: np.ndarray, q: np.ndarray) -> float:
        """Calculate Jensen-Shannon divergence"""
        # Average distribution
        m = 0.5 * (p + q)

        # Calculate KL divergences
        kl_pm = self.kl_divergence(p, m)
        kl_qm = self.kl_divergence(q, m)

        return 0.5 * (kl_pm + kl_qm)

    def wasserstein_distance(self, p: np.ndarray, q: np.ndarray, support: Optional[np.ndarray] = None) -> float:
        """Calculate Wasserstein distance"""
        if support is None:
            support = np.arange(len(p))

        # Calculate cumulative distributions
        p_cumsum = np.cumsum(p)
        q_cumsum = np.cumsum(q)

        # Calculate Wasserstein distance
        return np.sum(np.abs(p_cumsum - q_cumsum) * support)

    def hellinger_distance(self, p: np.ndarray, q: np.ndarray) -> float:
        """Calculate Hellinger distance"""
        # Calculate Hellinger distance
        sqrt_p = np.sqrt(p)
        sqrt_q = np.sqrt(q)

        return np.sqrt(0.5 * np.sum((sqrt_p - sqrt_q)**2))

    def analyze_distribution_differences(self, distributions: List[np.ndarray]) -> Dict[str, Any]:
        """Analyze differences between multiple distributions"""
        n_distributions = len(distributions)

        results = {
            'pairwise_divergences': {},
            'divergence_matrix': np.zeros((n_distributions, n_distributions)),
            'similarity_clusters': {},
            'most_similar_pairs': [],
            'most_different_pairs': []
        }

        # Calculate pairwise divergences
        divergence_measures = ['kl_divergence', 'js_divergence', 'hellinger']

        for measure in divergence_measures:
            results['pairwise_divergences'][measure] = {}

            for i in range(n_distributions):
                for j in range(i+1, n_distributions):
                    divergence = self.calculate_divergence(distributions[i], distributions[j], measure)
                    results['divergence_matrix'][i, j] = divergence
                    results['divergence_matrix'][j, i] = divergence  # Symmetric

                    pair_key = f"dist_{i}_vs_dist_{j}"
                    results['pairwise_divergences'][measure][pair_key] = divergence

        # Find most similar and different pairs
        results['most_similar_pairs'] = self.find_most_similar_pairs(results['divergence_matrix'])
        results['most_different_pairs'] = self.find_most_different_pairs(results['divergence_matrix'])

        return results

    def find_most_similar_pairs(self, divergence_matrix: np.ndarray) -> List[Tuple[int, int]]:
        """Find most similar distribution pairs"""
        n = divergence_matrix.shape[0]
        min_divergences = []

        for i in range(n):
            for j in range(i+1, n):
                min_divergences.append((divergence_matrix[i, j], i, j))

        min_divergences.sort()
        return [(i, j) for _, i, j in min_divergences[:3]]  # Top 3

    def find_most_different_pairs(self, divergence_matrix: np.ndarray) -> List[Tuple[int, int]]:
        """Find most different distribution pairs"""
        n = divergence_matrix.shape[0]
        max_divergences = []

        for i in range(n):
            for j in range(i+1, n):
                max_divergences.append((divergence_matrix[i, j], i, j))

        max_divergences.sort(reverse=True)
        return [(i, j) for _, i, j in max_divergences[:3]]  # Top 3
```

## Testing Guidelines

### Information Theory Testing
- **Mathematical Testing**: Test mathematical correctness of implementations
- **Numerical Testing**: Validate numerical accuracy and stability
- **Edge Case Testing**: Test with boundary conditions and special cases
- **Performance Testing**: Test computational performance
- **Convergence Testing**: Test algorithm convergence properties

### Scientific Validation
- **Benchmark Testing**: Validate against established benchmarks
- **Theoretical Testing**: Test against theoretical predictions
- **Comparison Testing**: Compare with alternative implementations
- **Consistency Testing**: Test consistency across different methods

## Performance Considerations

### Computational Performance
- **Algorithm Efficiency**: Use efficient algorithms and data structures
- **Memory Management**: Optimize memory usage for large datasets
- **Parallel Processing**: Utilize parallel processing where beneficial
- **Caching**: Implement caching for expensive computations

### Numerical Performance
- **Precision**: Maintain appropriate numerical precision
- **Stability**: Ensure numerical stability of algorithms
- **Convergence**: Optimize convergence properties
- **Error Control**: Control numerical errors and approximations

## Maintenance and Evolution

### Method Updates
- **Literature Review**: Keep current with latest information theory developments
- **Method Validation**: Validate new methods against benchmarks
- **Performance Optimization**: Optimize existing methods
- **Documentation Updates**: Keep mathematical documentation current

### Integration Updates
- **Framework Integration**: Maintain integration with analysis frameworks
- **Compatibility**: Ensure compatibility with data formats
- **Standards Updates**: Update to reflect current standards
- **Community Integration**: Integrate with research community tools

## Common Challenges and Solutions

### Challenge: Numerical Stability
**Solution**: Implement numerically stable algorithms with proper error handling and validation.

### Challenge: Computational Complexity
**Solution**: Use efficient algorithms and optimize for computational performance.

### Challenge: Interpretation
**Solution**: Provide clear interpretation guidelines and validation against known results.

### Challenge: Parameter Selection
**Solution**: Implement automatic parameter selection and validation procedures.

## Getting Started as an Agent

### Development Setup
1. **Study Information Theory**: Understand information theory foundations
2. **Learn Mathematical Methods**: Study mathematical implementation techniques
3. **Practice Implementation**: Practice implementing information-theoretic methods
4. **Understand Applications**: Learn Active Inference applications of information theory

### Contribution Process
1. **Identify Analysis Needs**: Find gaps in current information-theoretic capabilities
2. **Research Methods**: Study relevant information theory literature
3. **Design Solutions**: Create detailed mathematical implementations
4. **Implement and Test**: Follow mathematical implementation standards
5. **Validate Thoroughly**: Ensure mathematical correctness and numerical accuracy
6. **Document Completely**: Provide comprehensive mathematical documentation
7. **Scientific Review**: Submit for mathematical and scientific review

### Learning Resources
- **Information Theory**: Study information theory foundations
- **Mathematical Methods**: Master mathematical implementation techniques
- **Numerical Analysis**: Learn numerical analysis for information theory
- **Statistical Methods**: Study statistical estimation methods
- **Active Inference**: Domain-specific information-theoretic applications

## Related Documentation

- **[Information Theory README](./README.md)**: Information theory module overview
- **[Analysis AGENTS.md](../AGENTS.md)**: Analysis tools module guidelines
- **[Main AGENTS.md](../../../AGENTS.md)**: Project-wide agent guidelines
- **[Research AGENTS.md](../../AGENTS.md)**: Research tools module guidelines
- **[Contributing Guide](../../../CONTRIBUTING.md)**: Contribution processes

---

*"Active Inference for, with, by Generative AI"* - Advancing research through rigorous information theory, comprehensive analysis, and mathematical precision.
