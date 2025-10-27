# Information Theory Analysis

Comprehensive information-theoretic analysis tools for Active Inference research. Provides entropy measures, mutual information calculations, divergence measures, complexity analysis, and causal analysis for understanding information processing in intelligent systems.

## Overview

The Information Theory Analysis module provides a complete toolkit for information-theoretic analysis in Active Inference research. It includes rigorous implementations of entropy measures, mutual information, divergence measures, complexity analysis, and causal analysis tools.

## Directory Structure

```
information_theory/
├── entropy/                    # Entropy measures and analysis
├── mutual_information/         # Mutual information calculations
├── divergence_measures/        # Statistical divergence measures
├── complexity/                 # System complexity analysis
├── causal_analysis/            # Causal information analysis
└── validation/                 # Information theory validation tools
```

## Core Components

### 🔬 Entropy Measures
- **Shannon Entropy**: Classical discrete entropy calculations
- **Differential Entropy**: Continuous variable entropy measures
- **Relative Entropy**: Entropy measures for system comparison
- **Conditional Entropy**: Entropy under conditional constraints
- **Joint Entropy**: Multi-variable entropy calculations
- **Cross Entropy**: Cross-entropy loss calculations

### 🔗 Mutual Information Analysis
- **Mutual Information**: Information shared between variables
- **Conditional Mutual Information**: Conditional information measures
- **Multi-information**: Information in multiple variable systems
- **Transfer Entropy**: Information transfer between systems
- **Partial Information**: Decomposition of information measures
- **Pointwise Mutual Information**: Local information measures

### 📊 Divergence Measures
- **KL Divergence**: Kullback-Leibler divergence calculations
- **Jensen-Shannon Divergence**: Symmetric divergence measures
- **Wasserstein Distance**: Optimal transport-based distances
- **Hellinger Distance**: Statistical distance measures
- **f-Divergences**: Generalized divergence framework
- **Bhattacharyya Distance**: Statistical similarity measures

### 🧬 Complexity Analysis
- **Fractal Dimensions**: System complexity measures
- **Lyapunov Exponents**: Chaotic system analysis
- **Sample Entropy**: Time series complexity measures
- **Permutation Entropy**: Ordinal pattern complexity
- **Multiscale Entropy**: Multi-scale complexity analysis
- **Approximate Entropy**: Regularity and complexity measures

### 🔄 Causal Analysis
- **Granger Causality**: Temporal causal relationships
- **Transfer Entropy**: Information flow causality
- **Causal Graphs**: Information-theoretic causal networks
- **Causal Discovery**: Automated causal relationship discovery
- **Causal Strength**: Quantifying causal influence
- **Directed Information**: Directed information flow measures

## Research Roles and Functions

### 🧑‍🎓 Intern Level
```python
from active_inference.analysis.information_theory import InternInformationTheory

# Basic information theory analysis
info_theory = InternInformationTheory()
entropy = info_theory.calculate_basic_entropy(data)
mi = info_theory.calculate_mutual_information(data_x, data_y)
```

**Features:**
- Basic entropy calculations
- Simple mutual information
- Tutorial explanations
- Error checking and validation
- Basic interpretation tools

### 🎓 PhD Student Level
```python
from active_inference.analysis.information_theory import PhDInformationTheory

# Advanced information theory analysis
info_theory = PhDInformationTheory()
complexity_analysis = info_theory.analyze_system_complexity(time_series)
causal_analysis = info_theory.analyze_causal_relationships(multivariate_data)
```

**Features:**
- Advanced entropy measures
- Complex system analysis
- Causal analysis tools
- Statistical validation
- Publication-quality results

### 🧑‍🔬 Grant Application Level
```python
from active_inference.analysis.information_theory import GrantInformationTheory

# Research proposal information analysis
info_theory = GrantInformationTheory()
feasibility_analysis = info_theory.assess_method_feasibility(theoretical_requirements)
power_analysis = info_theory.calculate_statistical_power(sample_sizes)
```

**Features:**
- Method feasibility analysis
- Statistical power calculations
- Theoretical validation
- Resource requirement planning
- Grant proposal support

### 📝 Publication Level
```python
from active_inference.analysis.information_theory import PublicationInformationTheory

# Publication-ready information analysis
info_theory = PublicationInformationTheory()
comprehensive_analysis = info_theory.run_comprehensive_analysis(dataset)
validated_results = info_theory.validate_for_publication(comprehensive_analysis)
```

**Features:**
- Publication-standard analysis
- Comprehensive validation
- Reviewer-ready documentation
- Multiple method comparison
- Citation management

## Usage Examples

### Entropy Analysis
```python
from active_inference.analysis.information_theory import EntropyAnalyzer

# Initialize entropy analyzer
entropy_analyzer = EntropyAnalyzer()

# Analyze different types of entropy
data = np.random.normal(0, 1, 1000)

# Shannon entropy
shannon_result = entropy_analyzer.calculate_entropy(data, method='shannon')
print(f"Shannon entropy: {shannon_result.entropy_value:.3f}")

# Differential entropy
diff_result = entropy_analyzer.calculate_entropy(data, method='differential')
print(f"Differential entropy: {diff_result.entropy_value:.3f}")

# Conditional entropy
conditioning_data = np.random.normal(0, 0.5, 1000)
cond_result = entropy_analyzer.calculate_conditional_entropy(data, conditioning_data)
print(f"Conditional entropy: {cond_result.entropy_value:.3f}")
```

### Mutual Information Analysis
```python
from active_inference.analysis.information_theory import MutualInformationAnalyzer

# Initialize mutual information analyzer
mi_analyzer = MutualInformationAnalyzer()

# Analyze information relationships
x_data = np.random.normal(0, 1, 1000)
y_data = x_data + np.random.normal(0, 0.1, 1000)  # Related data

# Calculate mutual information using different methods
mi_histogram = mi_analyzer.calculate_mutual_information(x_data, y_data, method='histogram')
mi_kernel = mi_analyzer.calculate_mutual_information(x_data, y_data, method='kernel')
mi_kraskov = mi_analyzer.calculate_mutual_information(x_data, y_data, method='kraskov')

print(f"Histogram MI: {mi_histogram:.3f}")
print(f"Kernel MI: {mi_kernel:.3f}")
print(f"Kraskov MI: {mi_kraskov:.3f}")

# Analyze transfer entropy
te = mi_analyzer.transfer_entropy(x_data, y_data, lag=1)
print(f"Transfer entropy: {te:.3f}")
```

### Divergence Analysis
```python
from active_inference.analysis.information_theory import DivergenceAnalyzer

# Initialize divergence analyzer
divergence_analyzer = DivergenceAnalyzer()

# Analyze differences between distributions
dist1 = np.random.normal(0, 1, 1000)
dist2 = np.random.normal(1, 1, 1000)
dist3 = np.random.normal(0, 2, 1000)

distributions = [dist1, dist2, dist3]

# Calculate various divergences
kl_div = divergence_analyzer.calculate_divergence(dist1, dist2, measure='kl_divergence')
js_div = divergence_analyzer.calculate_divergence(dist1, dist2, measure='js_divergence')
hellinger = divergence_analyzer.calculate_divergence(dist1, dist2, measure='hellinger')

print(f"KL Divergence: {kl_div:.3f}")
print(f"JS Divergence: {js_div:.3f}")
print(f"Hellinger Distance: {hellinger:.3f}")

# Analyze multiple distributions
multi_analysis = divergence_analyzer.analyze_distribution_differences(distributions)
print(f"Most similar pairs: {multi_analysis['most_similar_pairs']}")
```

## Information Theory Methods

### Entropy Methods
- **Shannon Entropy**: Classical information entropy for discrete variables
- **Differential Entropy**: Entropy for continuous variables
- **Relative Entropy**: Entropy difference between distributions
- **Conditional Entropy**: Entropy given knowledge of other variables
- **Joint Entropy**: Entropy of multiple variables together
- **Cross Entropy**: Cross-entropy between distributions

### Mutual Information Methods
- **Histogram Method**: Mutual information using histogram binning
- **Kernel Method**: Kernel density estimation approach
- **Kraskov Method**: Nearest neighbor based estimation
- **Transfer Entropy**: Information flow between time series
- **Partial Information**: Information decomposition methods

### Divergence Methods
- **KL Divergence**: Kullback-Leibler information divergence
- **JS Divergence**: Jensen-Shannon symmetric divergence
- **Wasserstein Distance**: Earth mover's distance
- **Hellinger Distance**: Statistical distance measure
- **f-Divergences**: Generalized divergence framework

## Advanced Features

### Multi-Scale Information Analysis
```python
from active_inference.analysis.information_theory import MultiScaleAnalyzer

# Analyze information at multiple scales
multiscale = MultiScaleAnalyzer()

scales = [1, 2, 4, 8, 16]
entropy_scales = multiscale.analyze_entropy_scales(time_series, scales)
mi_scales = multiscale.analyze_mi_scales(data_x, data_y, scales)
```

### Information Flow Networks
```python
from active_inference.analysis.information_theory import InformationFlowAnalyzer

# Analyze information flow networks
flow_analyzer = InformationFlowAnalyzer()

# Build information flow network
network = flow_analyzer.build_information_network(multivariate_data, max_lag=10)
centrality = flow_analyzer.calculate_information_centrality(network)
```

### Complexity Measures
```python
from active_inference.analysis.information_theory import ComplexityAnalyzer

# Analyze system complexity
complexity = ComplexityAnalyzer()

# Calculate various complexity measures
measures = ['sample_entropy', 'permutation_entropy', 'lyapunov_exponent']
complexity_results = complexity.calculate_complexity_measures(time_series, measures)

# Multi-scale complexity
multiscale_complexity = complexity.multiscale_complexity_analysis(time_series)
```

## Integration with Analysis Pipeline

### Statistical Integration
```python
from active_inference.analysis.information_theory import StatisticalInformationIntegration

# Integrate with statistical analysis
stat_integration = StatisticalInformationIntegration()

# Combine information theory with statistical tests
combined_analysis = stat_integration.combine_with_statistics(
    information_results,
    statistical_results
)
```

### Visualization Integration
```python
from active_inference.analysis.information_theory import InformationVisualization

# Create information-theoretic visualizations
info_viz = InformationVisualization()

# Generate entropy plots
entropy_plots = info_viz.plot_entropy_analysis(entropy_results)

# Generate mutual information heatmaps
mi_heatmaps = info_viz.plot_mutual_information_matrix(mi_matrix)
```

## Configuration Options

### Analysis Settings
```python
analysis_config = {
    'default_method': 'histogram',
    'bias_correction': True,
    'confidence_level': 0.95,
    'sample_size_threshold': 100,
    'numerical_precision': 'double',
    'parallel_processing': True,
    'validation_strictness': 'high'
}
```

### Entropy Configuration
```python
entropy_config = {
    'base': 2.0,  # logarithm base
    'bins': 'auto',  # histogram bins
    'bandwidth': 'scott',  # KDE bandwidth
    'bias_correction': True,
    'confidence_intervals': True,
    'validation': True
}
```

## Quality Assurance

### Mathematical Validation
- **Theoretical Compliance**: Validate against information theory principles
- **Numerical Accuracy**: Ensure numerical precision and accuracy
- **Convergence Testing**: Test algorithm convergence properties
- **Benchmark Comparison**: Compare with established implementations
- **Edge Case Testing**: Test with boundary conditions

### Scientific Validation
- **Method Validation**: Validate methods against known results
- **Consistency Testing**: Ensure consistent results across methods
- **Interpretation Validation**: Validate result interpretations
- **Reproducibility**: Ensure reproducible analysis results

## Information Theory Standards

### Mathematical Standards
- **Correctness**: Mathematically correct implementations
- **Precision**: Appropriate numerical precision
- **Stability**: Numerical stability of algorithms
- **Convergence**: Proper algorithm convergence
- **Validation**: Validated against theoretical benchmarks

### Analysis Standards
- **Method Selection**: Appropriate method selection
- **Parameter Validation**: Proper parameter validation
- **Result Interpretation**: Correct result interpretation
- **Reporting**: Clear and accurate reporting
- **Reproducibility**: Reproducible analysis procedures

## Contributing

We welcome contributions to the information theory module! See [CONTRIBUTING.md](../../../CONTRIBUTING.md) for guidelines.

### Development Setup
```bash
# Install information theory dependencies
pip install -e ".[information_theory,dev]"

# Run information theory tests
pytest tests/unit/test_information_theory.py -v

# Run validation tests
pytest tests/analysis/test_information_theory_validation.py -v
```

## Learning Resources

- **Information Theory**: Foundations of information theory
- **Entropy Methods**: Entropy calculation and interpretation
- **Mutual Information**: Mutual information theory and applications
- **Divergence Measures**: Statistical divergence theory
- **Complexity Theory**: System complexity analysis
- **Causal Analysis**: Information-theoretic causality

## Related Documentation

- **[Analysis README](../README.md)**: Analysis tools module overview
- **[Main README](../../../README.md)**: Project overview
- **[Knowledge Repository](../../../knowledge/)**: Theoretical foundations
- **[Statistical Analysis](../../analysis/statistical/)**: Statistical analysis methods
- **[Contributing Guide](../../../CONTRIBUTING.md)**: Contribution guidelines

---

*"Active Inference for, with, by Generative AI"* - Advancing research through rigorous information theory, comprehensive analysis, and mathematical precision.
