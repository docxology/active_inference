# Research Analysis Tools - Agent Documentation

This document provides comprehensive guidance for AI agents and contributors working within the Research Analysis Tools module of the Active Inference Knowledge Environment. It outlines analysis methodologies, implementation patterns, and best practices for creating robust scientific analysis tools.

## Research Analysis Tools Module Overview

The Research Analysis module provides a complete toolkit for analyzing Active Inference research data, including statistical analysis, information theory, performance metrics, data visualization, and result interpretation. All analysis tools are designed to support rigorous, reproducible scientific analysis following established scientific standards.

## Core Responsibilities

### Analysis Method Development
- **Statistical Methods**: Implement comprehensive statistical analysis methods
- **Information Theory**: Develop information-theoretic analysis tools
- **Performance Metrics**: Create Active Inference-specific performance measures
- **Data Analysis**: Implement data processing and analysis pipelines
- **Validation Methods**: Develop result validation and verification tools

### Scientific Quality Assurance
- **Method Validation**: Ensure methods are scientifically valid
- **Numerical Accuracy**: Maintain numerical accuracy and stability
- **Reproducibility**: Ensure reproducible analysis results
- **Benchmarking**: Validate against established benchmarks
- **Documentation**: Maintain comprehensive scientific documentation

### Research Tool Integration
- **Framework Integration**: Integrate with experiment and simulation frameworks
- **Data Pipeline**: Support comprehensive data analysis pipelines
- **Visualization**: Connect with visualization tools
- **Reporting**: Generate scientific reports and publications
- **Collaboration**: Support collaborative analysis workflows

## Development Workflows

### Analysis Method Development Process
1. **Requirements Analysis**: Analyze scientific analysis requirements
2. **Literature Research**: Research established analysis methods
3. **Algorithm Design**: Design analysis algorithms and methods
4. **Mathematical Implementation**: Implement mathematically correct algorithms
5. **Numerical Validation**: Validate numerical accuracy and stability
6. **Benchmarking**: Test against established benchmarks
7. **Testing**: Comprehensive testing including edge cases
8. **Documentation**: Create comprehensive scientific documentation
9. **Review**: Submit for scientific and mathematical review
10. **Publication**: Release with proper scientific validation

### Statistical Analysis Implementation
1. **Method Selection**: Choose appropriate statistical methods
2. **Algorithm Implementation**: Implement statistical algorithms correctly
3. **Parameter Validation**: Validate parameter ranges and constraints
4. **Numerical Testing**: Test numerical stability and accuracy
5. **Statistical Validation**: Validate against known statistical results
6. **Performance Optimization**: Optimize computational performance
7. **Documentation**: Document statistical methods comprehensively
8. **Example Creation**: Create usage examples and tutorials

### Information Theory Implementation
1. **Theory Review**: Review information-theoretic foundations
2. **Formula Implementation**: Implement information-theoretic formulas
3. **Numerical Methods**: Develop stable numerical computation methods
4. **Validation**: Validate against theoretical predictions
5. **Performance Testing**: Test computational performance
6. **Edge Case Handling**: Handle special cases and edge conditions
7. **Documentation**: Document mathematical derivations and usage
8. **Example Implementation**: Create practical usage examples

## Quality Standards

### Scientific Quality
- **Theoretical Soundness**: All methods must be theoretically sound
- **Mathematical Correctness**: Implementations must be mathematically correct
- **Numerical Accuracy**: Maintain numerical precision and accuracy
- **Validation**: Methods must be validated against benchmarks
- **Reproducibility**: All results must be reproducible

### Code Quality
- **Algorithmic Correctness**: Algorithms must be correctly implemented
- **Numerical Stability**: Implementations must be numerically stable
- **Performance**: Efficient algorithms and data structures
- **Testing**: Comprehensive testing including edge cases
- **Documentation**: Clear documentation of all functionality

### Research Quality
- **Methodological Rigor**: Follow established research methodologies
- **Statistical Soundness**: Proper statistical analysis and interpretation
- **Validation**: Comprehensive validation against benchmarks
- **Documentation**: Complete scientific documentation
- **Reproducibility**: Full reproducibility of all results

## Implementation Patterns

### Statistical Analysis Framework
```python
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import scipy.stats as stats
from dataclasses import dataclass

@dataclass
class StatisticalTestResult:
    """Result of statistical test"""
    test_name: str
    statistic: float
    p_value: float
    degrees_of_freedom: Optional[int] = None
    confidence_interval: Optional[Tuple[float, float]] = None
    effect_size: Optional[float] = None
    interpretation: str = ""

@dataclass
class AnalysisConfig:
    """Configuration for statistical analysis"""
    alpha: float = 0.05
    alternative: str = 'two-sided'  # 'two-sided', 'less', 'greater'
    correction: Optional[str] = None  # 'bonferroni', 'holm', etc.
    effect_size_measure: str = 'cohen_d'

class BaseStatisticalTest(ABC):
    """Base class for statistical tests"""

    def __init__(self, config: AnalysisConfig = None):
        """Initialize statistical test"""
        self.config = config or AnalysisConfig()
        self.test_name = self.__class__.__name__

    @abstractmethod
    def perform_test(self, data: Dict[str, Any]) -> StatisticalTestResult:
        """Perform the statistical test"""
        pass

    def validate_data(self, data: Dict[str, Any]) -> List[str]:
        """Validate input data"""
        issues = []

        # Check required fields
        required_fields = self.get_required_fields()
        for field in required_fields:
            if field not in data:
                issues.append(f"Required field '{field}' missing")

        # Check data types and shapes
        for field, requirements in self.get_data_requirements().items():
            if field in data:
                issues.extend(self.validate_field(data[field], field, requirements))

        return issues

    @abstractmethod
    def get_required_fields(self) -> List[str]:
        """Get required data fields"""
        pass

    @abstractmethod
    def get_data_requirements(self) -> Dict[str, Dict[str, Any]]:
        """Get data requirements for each field"""
        pass

    def validate_field(self, data, field_name: str, requirements: Dict[str, Any]) -> List[str]:
        """Validate individual field"""
        issues = []

        # Check data type
        expected_type = requirements.get('type')
        if expected_type and not isinstance(data, expected_type):
            issues.append(f"Field '{field_name}' must be of type {expected_type}")

        # Check dimensions
        expected_shape = requirements.get('shape')
        if expected_shape and hasattr(data, 'shape'):
            if data.shape != expected_shape:
                issues.append(f"Field '{field_name}' shape {data.shape} does not match expected {expected_shape}")

        # Check value ranges
        min_val = requirements.get('min_value')
        max_val = requirements.get('max_value')
        if min_val is not None and np.any(data < min_val):
            issues.append(f"Field '{field_name}' contains values below minimum {min_val}")
        if max_val is not None and np.any(data > max_val):
            issues.append(f"Field '{field_name}' contains values above maximum {max_val}")

        return issues

    def calculate_effect_size(self, data1: np.ndarray, data2: np.ndarray,
                           measure: str = 'cohen_d') -> float:
        """Calculate effect size"""
        if measure == 'cohen_d':
            pooled_std = np.sqrt(((len(data1) - 1) * np.var(data1) + (len(data2) - 1) * np.var(data2)) /
                                (len(data1) + len(data2) - 2))
            return (np.mean(data2) - np.mean(data1)) / pooled_std if pooled_std > 0 else 0

        elif measure == 'eta_squared':
            # For ANOVA-style effect sizes
            return self.calculate_eta_squared(data1, data2)

        else:
            raise ValueError(f"Unknown effect size measure: {measure}")

    def calculate_eta_squared(self, data1: np.ndarray, data2: np.ndarray) -> float:
        """Calculate eta squared effect size"""
        all_data = np.concatenate([data1, data2])
        grand_mean = np.mean(all_data)

        ss_between = len(data1) * (np.mean(data1) - grand_mean)**2 + len(data2) * (np.mean(data2) - grand_mean)**2
        ss_total = np.sum((all_data - grand_mean)**2)

        return ss_between / ss_total if ss_total > 0 else 0

class TTest(BaseStatisticalTest):
    """Independent samples t-test implementation"""

    def get_required_fields(self) -> List[str]:
        """Get required data fields for t-test"""
        return ['group1', 'group2']

    def get_data_requirements(self) -> Dict[str, Dict[str, Any]]:
        """Get data requirements for t-test"""
        return {
            'group1': {'type': np.ndarray, 'shape': (None,), 'min_value': None, 'max_value': None},
            'group2': {'type': np.ndarray, 'shape': (None,), 'min_value': None, 'max_value': None}
        }

    def perform_test(self, data: Dict[str, Any]) -> StatisticalTestResult:
        """Perform independent samples t-test"""
        # Validate data
        issues = self.validate_data(data)
        if issues:
            raise ValueError(f"Data validation failed: {issues}")

        group1 = data['group1']
        group2 = data['group2']

        # Perform t-test
        t_stat, p_value = stats.ttest_ind(group1, group2, alternative=self.config.alternative)

        # Calculate effect size
        effect_size = self.calculate_effect_size(group1, group2, self.config.effect_size_measure)

        # Calculate confidence interval
        confidence_interval = self.calculate_confidence_interval(group1, group2, self.config.alpha)

        # Interpretation
        interpretation = self.interpret_result(t_stat, p_value, effect_size)

        return StatisticalTestResult(
            test_name=self.test_name,
            statistic=t_stat,
            p_value=p_value,
            degrees_of_freedom=len(group1) + len(group2) - 2,
            confidence_interval=confidence_interval,
            effect_size=effect_size,
            interpretation=interpretation
        )

    def calculate_confidence_interval(self, data1: np.ndarray, data2: np.ndarray,
                                   alpha: float) -> Tuple[float, float]:
        """Calculate confidence interval for mean difference"""
        diff_mean = np.mean(data2) - np.mean(data1)
        diff_std = np.sqrt(np.var(data1)/len(data1) + np.var(data2)/len(data2))
        df = len(data1) + len(data2) - 2

        # t-critical value
        t_critical = stats.t.ppf(1 - alpha/2, df)

        margin = t_critical * diff_std
        return (diff_mean - margin, diff_mean + margin)

    def interpret_result(self, t_stat: float, p_value: float, effect_size: float) -> str:
        """Interpret t-test result"""
        interpretation = []

        # Significance
        if p_value < self.config.alpha:
            interpretation.append(f"Statistically significant (p = {p_value:.".4f"")
        else:
            interpretation.append(f"Not statistically significant (p = {p_value:.".4f"")
        # Effect size interpretation
        if abs(effect_size) < 0.2:
            interpretation.append("Small effect size")
        elif abs(effect_size) < 0.5:
            interpretation.append("Medium effect size")
        else:
            interpretation.append("Large effect size")

        return ". ".join(interpretation)

class StatisticalAnalysisFramework:
    """Framework for comprehensive statistical analysis"""

    def __init__(self, config: AnalysisConfig = None):
        """Initialize statistical analysis framework"""
        self.config = config or AnalysisConfig()
        self.tests: Dict[str, BaseStatisticalTest] = {}
        self.register_default_tests()

    def register_default_tests(self) -> None:
        """Register default statistical tests"""
        self.register_test('t_test', TTest(self.config))
        # Register additional tests as needed

    def register_test(self, name: str, test: BaseStatisticalTest) -> None:
        """Register statistical test"""
        self.tests[name] = test

    def analyze_data(self, data: Dict[str, Any], test_type: str = 'auto') -> Dict[str, Any]:
        """Perform comprehensive statistical analysis"""
        results = {
            'data_summary': self.summarize_data(data),
            'test_results': {},
            'overall_interpretation': ""
        }

        # Select appropriate tests
        if test_type == 'auto':
            test_type = self.select_tests(data)

        # Run selected tests
        for test_name in test_type:
            if test_name in self.tests:
                try:
                    test_result = self.tests[test_name].perform_test(data)
                    results['test_results'][test_name] = test_result.__dict__
                except Exception as e:
                    results['test_results'][test_name] = {'error': str(e)}

        # Generate overall interpretation
        results['overall_interpretation'] = self.generate_overall_interpretation(results)

        return results

    def summarize_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Summarize input data"""
        summary = {}

        for key, value in data.items():
            if isinstance(value, np.ndarray):
                summary[key] = {
                    'shape': value.shape,
                    'mean': np.mean(value),
                    'std': np.std(value),
                    'min': np.min(value),
                    'max': np.max(value),
                    'dtype': str(value.dtype)
                }
            else:
                summary[key] = {
                    'type': type(value).__name__,
                    'value': str(value)
                }

        return summary

    def select_tests(self, data: Dict[str, Any]) -> List[str]:
        """Select appropriate statistical tests"""
        tests = []

        # Simple heuristic for test selection
        if 'group1' in data and 'group2' in data:
            if len(data['group1']) > 1 and len(data['group2']) > 1:
                tests.append('t_test')

        return tests

    def generate_overall_interpretation(self, results: Dict[str, Any]) -> str:
        """Generate overall interpretation of analysis results"""
        interpretations = []

        for test_name, test_result in results['test_results'].items():
            if 'interpretation' in test_result:
                interpretations.append(f"{test_name}: {test_result['interpretation']}")

        return ". ".join(interpretations) if interpretations else "No significant results found"
```

### Information Theory Framework
```python
from typing import Dict, Any, List, Optional, Union
import numpy as np
from abc import ABC, abstractmethod

class BaseEntropyMeasure(ABC):
    """Base class for entropy measures"""

    @abstractmethod
    def calculate(self, probabilities: np.ndarray, **kwargs) -> float:
        """Calculate entropy measure"""
        pass

    @abstractmethod
    def validate_input(self, probabilities: np.ndarray) -> List[str]:
        """Validate input probabilities"""
        pass

class ShannonEntropy(BaseEntropyMeasure):
    """Shannon entropy implementation"""

    def calculate(self, probabilities: np.ndarray, **kwargs) -> float:
        """Calculate Shannon entropy"""
        # Validate input
        issues = self.validate_input(probabilities)
        if issues:
            raise ValueError(f"Input validation failed: {issues}")

        # Remove zero probabilities to avoid log(0)
        non_zero_probs = probabilities[probabilities > 0]

        if len(non_zero_probs) == 0:
            return 0.0

        # Calculate Shannon entropy
        return -np.sum(non_zero_probs * np.log2(non_zero_probs))

    def validate_input(self, probabilities: np.ndarray) -> List[str]:
        """Validate probability distribution"""
        issues = []

        # Check if it's a numpy array
        if not isinstance(probabilities, np.ndarray):
            issues.append("Input must be numpy array")

        # Check if probabilities sum to 1 (allowing small numerical errors)
        prob_sum = np.sum(probabilities)
        if not np.isclose(prob_sum, 1.0, atol=1e-6):
            issues.append(f"Probabilities must sum to 1, got {prob_sum}")

        # Check for negative probabilities
        if np.any(probabilities < 0):
            issues.append("Probabilities cannot be negative")

        return issues

class InformationTheoryAnalyzer:
    """Comprehensive information theory analysis"""

    def __init__(self):
        """Initialize information theory analyzer"""
        self.entropy_measures: Dict[str, BaseEntropyMeasure] = {}
        self.register_default_measures()

    def register_default_measures(self) -> None:
        """Register default entropy measures"""
        self.register_measure('shannon', ShannonEntropy())
        # Register additional measures as needed

    def register_measure(self, name: str, measure: BaseEntropyMeasure) -> None:
        """Register entropy measure"""
        self.entropy_measures[name] = measure

    def calculate_entropy(self, probabilities: np.ndarray, measure: str = 'shannon',
                         **kwargs) -> float:
        """Calculate entropy using specified measure"""
        if measure not in self.entropy_measures:
            raise ValueError(f"Unknown entropy measure: {measure}")

        return self.entropy_measures[measure].calculate(probabilities, **kwargs)

    def mutual_information(self, joint_probabilities: np.ndarray,
                          method: str = 'standard') -> float:
        """Calculate mutual information"""
        if joint_probabilities.ndim != 2:
            raise ValueError("Joint probabilities must be 2-dimensional")

        # Marginal probabilities
        p_x = np.sum(joint_probabilities, axis=1)
        p_y = np.sum(joint_probabilities, axis=0)

        # Joint entropy
        h_xy = self.calculate_entropy(joint_probabilities.flatten(), 'shannon')

        # Marginal entropies
        h_x = self.calculate_entropy(p_x, 'shannon')
        h_y = self.calculate_entropy(p_y, 'shannon')

        return h_x + h_y - h_xy

    def kl_divergence(self, p: np.ndarray, q: np.ndarray, epsilon: float = 1e-12) -> float:
        """Calculate Kullback-Leibler divergence"""
        # Add small epsilon to avoid log(0)
        p = p + epsilon
        q = q + epsilon

        # Normalize to ensure they sum to 1
        p = p / np.sum(p)
        q = q / np.sum(q)

        # Calculate KL divergence
        return np.sum(p * np.log2(p / q))

    def complexity_analysis(self, time_series: np.ndarray,
                           measures: List[str] = None) -> Dict[str, float]:
        """Analyze complexity of time series"""
        if measures is None:
            measures = ['sample_entropy', 'permutation_entropy']

        results = {}

        for measure in measures:
            if measure == 'sample_entropy':
                results['sample_entropy'] = self.sample_entropy(time_series)
            elif measure == 'permutation_entropy':
                results['permutation_entropy'] = self.permutation_entropy(time_series)

        return results

    def sample_entropy(self, time_series: np.ndarray, m: int = 2, r: float = None) -> float:
        """Calculate sample entropy"""
        if r is None:
            r = 0.2 * np.std(time_series)

        n = len(time_series)

        # Create template vectors
        def create_templates(series, template_length):
            return np.array([series[i:i+template_length]
                           for i in range(n - template_length + 1)])

        templates_m = create_templates(time_series, m)
        templates_m1 = create_templates(time_series, m + 1)

        # Count similar templates
        def count_similar_templates(templates1, templates2, tolerance):
            count = 0
            for i, template1 in enumerate(templates1):
                for j, template2 in enumerate(templates2):
                    if i != j and np.max(np.abs(template1 - template2)) <= tolerance:
                        count += 1
            return count

        # Calculate matches
        matches_m = count_similar_templates(templates_m, templates_m, r)
        matches_m1 = count_similar_templates(templates_m1, templates_m1, r)

        # Calculate sample entropy
        if matches_m == 0:
            return float('inf')
        elif matches_m1 == 0:
            return 0.0
        else:
            return -np.log(matches_m1 / matches_m)

    def permutation_entropy(self, time_series: np.ndarray, order: int = 3,
                           delay: int = 1) -> float:
        """Calculate permutation entropy"""
        n = len(time_series)

        # Create permutation patterns
        patterns = []
        for i in range(n - delay * order):
            pattern = []
            for j in range(order):
                idx = i + j * delay
                # Find permutation pattern
                values = time_series[idx:idx + delay + 1]
                perm = np.argsort(values)
                pattern.append(tuple(perm))

            patterns.append(tuple(pattern))

        # Count unique patterns
        unique_patterns = set(patterns)
        pattern_counts = {pattern: patterns.count(pattern) for pattern in unique_patterns}

        # Calculate permutation entropy
        total_patterns = len(patterns)
        entropy = 0.0
        for count in pattern_counts.values():
            prob = count / total_patterns
            entropy -= prob * np.log2(prob)

        return entropy
```

## Testing Guidelines

### Analysis Method Testing
- **Mathematical Testing**: Test mathematical correctness
- **Numerical Testing**: Validate numerical accuracy and stability
- **Edge Case Testing**: Test edge cases and boundary conditions
- **Performance Testing**: Test computational performance
- **Integration Testing**: Test integration with other components

### Scientific Validation
- **Benchmark Testing**: Validate against established benchmarks
- **Theoretical Testing**: Test against theoretical predictions
- **Convergence Testing**: Test algorithm convergence
- **Stability Testing**: Test numerical stability
- **Accuracy Testing**: Validate computational accuracy

## Performance Considerations

### Computational Performance
- **Algorithm Efficiency**: Use efficient algorithms and data structures
- **Memory Management**: Optimize memory usage for large datasets
- **Parallel Processing**: Utilize parallel processing where beneficial
- **Caching**: Implement caching for expensive computations

### Numerical Performance
- **Precision**: Maintain appropriate numerical precision
- **Stability**: Ensure numerical stability of algorithms
- **Error Control**: Control numerical errors and approximations
- **Convergence**: Optimize convergence properties

## Maintenance and Evolution

### Method Updates
- **Literature Review**: Keep current with latest research developments
- **Method Validation**: Validate new methods against benchmarks
- **Performance Optimization**: Optimize existing methods
- **Documentation Updates**: Keep documentation current

### Scientific Integration
- **Community Validation**: Validate methods with research community
- **Benchmark Updates**: Update benchmarks as field evolves
- **Method Comparison**: Compare with alternative implementations
- **Publication Support**: Support research publication

## Common Challenges and Solutions

### Challenge: Numerical Stability
**Solution**: Implement numerically stable algorithms with proper error handling.

### Challenge: Performance
**Solution**: Profile and optimize computational performance while maintaining accuracy.

### Challenge: Validation
**Solution**: Establish validation against multiple benchmarks and theoretical predictions.

### Challenge: Documentation
**Solution**: Maintain comprehensive scientific documentation following academic standards.

## Getting Started as an Agent

### Development Setup
1. **Study Analysis Framework**: Understand current analysis tool architecture
2. **Learn Scientific Methods**: Study established scientific methodologies
3. **Practice Implementation**: Practice implementing analysis methods
4. **Understand Validation**: Learn research validation techniques

### Contribution Process
1. **Identify Analysis Needs**: Find gaps in current analysis capabilities
2. **Research Methods**: Study relevant scientific literature
3. **Design Solutions**: Create detailed method designs
4. **Implement and Test**: Follow scientific implementation standards
5. **Validate Thoroughly**: Ensure scientific validity
6. **Document Completely**: Provide comprehensive scientific documentation
7. **Community Review**: Submit for scientific peer review

### Learning Resources
- **Scientific Computing**: Study scientific computing methodologies
- **Numerical Analysis**: Master numerical analysis techniques
- **Statistical Methods**: Study advanced statistical methods
- **Information Theory**: Learn information-theoretic foundations
- **Research Methods**: Study research methodology best practices

## Related Documentation

- **[Analysis README](./README.md)**: Analysis tools module overview
- **[Main AGENTS.md](../../AGENTS.md)**: Project-wide agent guidelines
- **[Research AGENTS.md](../AGENTS.md)**: Research tools module guidelines
- **[Contributing Guide](../../CONTRIBUTING.md)**: Contribution processes
- **[Knowledge Repository](../../knowledge/)**: Theoretical foundations

---

*"Active Inference for, with, by Generative AI"* - Advancing research through rigorous analysis, comprehensive methods, and scientific validation.