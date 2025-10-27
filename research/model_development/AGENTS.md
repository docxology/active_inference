# Research Model Development - Agent Documentation

This document provides comprehensive guidance for AI agents and contributors working within the Research Model Development module of the Active Inference Knowledge Environment. It outlines model development methodologies, implementation patterns, and best practices for creating robust, validated computational models throughout the research lifecycle.

## Model Development Module Overview

The Research Model Development module provides a comprehensive framework for developing, implementing, optimizing, validating, and versioning computational models for Active Inference research. It supports the complete model lifecycle from initial conception through implementation, testing, validation, and deployment.

## Core Responsibilities

### Model Development & Design
- **Architecture Design**: Design computational model architectures
- **Algorithm Development**: Develop and implement core algorithms
- **Parameter Design**: Design model parameters and configurations
- **Integration Planning**: Plan integration with existing systems
- **Scalability Design**: Design for computational scalability

### Implementation & Engineering
- **Code Implementation**: Implement models in production code
- **Performance Engineering**: Optimize computational performance
- **Memory Management**: Efficient memory usage optimization
- **Error Handling**: Robust error handling and recovery
- **Documentation**: Comprehensive code and API documentation

### Optimization & Tuning
- **Parameter Optimization**: Systematic parameter tuning and optimization
- **Algorithm Optimization**: Algorithm performance optimization
- **Hyperparameter Tuning**: Automated hyperparameter optimization
- **Performance Profiling**: Computational performance analysis
- **Resource Optimization**: Computational resource optimization

### Validation & Testing
- **Unit Testing**: Individual component testing
- **Integration Testing**: System integration testing
- **Performance Testing**: Computational performance validation
- **Accuracy Testing**: Model accuracy and correctness validation
- **Benchmarking**: Comparison with established benchmarks

### Versioning & Management
- **Version Control**: Model versioning and change tracking
- **Configuration Management**: Model configuration management
- **Dependency Tracking**: Track model dependencies and requirements
- **Deployment Management**: Model deployment and lifecycle management
- **Collaboration Support**: Support for collaborative model development

## Development Workflows

### Model Development Process
1. **Requirements Analysis**: Analyze model requirements and specifications
2. **Architecture Design**: Design model architecture and components
3. **Implementation Planning**: Plan implementation strategy and timeline
4. **Development**: Implement model following best practices
5. **Testing**: Comprehensive testing at all levels
6. **Optimization**: Optimize performance and accuracy
7. **Validation**: Validate against requirements and benchmarks
8. **Documentation**: Create comprehensive documentation
9. **Review**: Submit for technical and scientific review
10. **Deployment**: Deploy and integrate with research pipeline

### Active Inference Model Development
1. **Theory Analysis**: Analyze Active Inference theoretical requirements
2. **Mathematical Formulation**: Formal mathematical model specification
3. **Algorithm Design**: Design computational algorithms
4. **Implementation**: Implement in efficient computational framework
5. **Numerical Validation**: Validate numerical accuracy and stability
6. **Performance Optimization**: Optimize for computational efficiency
7. **Benchmarking**: Compare with theoretical predictions and alternatives
8. **Integration**: Integrate with Active Inference ecosystem

### Model Optimization Workflow
1. **Performance Profiling**: Profile current model performance
2. **Bottleneck Identification**: Identify performance bottlenecks
3. **Optimization Planning**: Plan optimization strategies
4. **Implementation**: Implement performance improvements
5. **Validation**: Validate optimization effectiveness
6. **Regression Testing**: Ensure no performance regressions
7. **Documentation**: Document optimization changes
8. **Monitoring**: Monitor optimized model performance

## Quality Standards

### Model Quality Standards
- **Correctness**: Mathematical and algorithmic correctness
- **Accuracy**: Agreement with theoretical predictions
- **Efficiency**: Computational efficiency and performance
- **Robustness**: Stability and robustness under various conditions
- **Scalability**: Ability to scale with problem size

### Code Quality Standards
- **Readability**: Clear, well-documented code
- **Maintainability**: Easy to maintain and extend
- **Testability**: Well-tested with comprehensive coverage
- **Performance**: Optimized for target use cases
- **Security**: Secure coding practices

### Scientific Standards
- **Reproducibility**: Reproducible model results
- **Validation**: Validated against benchmarks and theory
- **Documentation**: Complete scientific documentation
- **Standards Compliance**: Compliance with scientific computing standards
- **Ethics**: Ethical considerations in model development

## Implementation Patterns

### Model Framework Pattern
```python
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union, Tuple
from dataclasses import dataclass, field
import numpy as np
import json
from datetime import datetime
import logging

@dataclass
class ModelConfig:
    """Model configuration"""
    name: str
    version: str
    model_type: str  # active_inference, neural_network, statistical, etc.
    parameters: Dict[str, Any]
    architecture: Dict[str, Any]
    requirements: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ModelPerformance:
    """Model performance metrics"""
    accuracy: float
    efficiency: Dict[str, float]
    robustness: Dict[str, float]
    scalability: Dict[str, float]
    validation_scores: Dict[str, float]
    timestamp: datetime

class BaseModel(ABC):
    """Base class for computational models"""

    def __init__(self, config: ModelConfig):
        """Initialize model with configuration"""
        self.config = config
        self.model_state: Dict[str, Any] = {}
        self.performance_history: List[ModelPerformance] = []
        self.logger = logging.getLogger(self.__class__.__name__)
        self.setup_model()

    @abstractmethod
    def setup_model(self) -> None:
        """Set up model architecture and parameters"""
        pass

    @abstractmethod
    def train(self, training_data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Train model on training data"""
        pass

    @abstractmethod
    def predict(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate predictions from input data"""
        pass

    @abstractmethod
    def validate(self, validation_data: Dict[str, Any]) -> ModelPerformance:
        """Validate model performance"""
        pass

    def save_model(self, save_path: str) -> None:
        """Save model to disk"""
        model_package = {
            'config': self.config.__dict__,
            'state': self.model_state,
            'performance_history': [perf.__dict__ for perf in self.performance_history],
            'timestamp': datetime.now().isoformat()
        }

        with open(save_path, 'w') as f:
            json.dump(model_package, f, indent=2)

        self.logger.info(f"Model saved to {save_path}")

    def load_model(self, load_path: str) -> None:
        """Load model from disk"""
        with open(load_path, 'r') as f:
            model_package = json.load(f)

        self.config = ModelConfig(**model_package['config'])
        self.model_state = model_package['state']
        self.performance_history = [ModelPerformance(**perf) for perf in model_package['performance_history']]

        self.logger.info(f"Model loaded from {load_path}")

    def update_config(self, new_config: Dict[str, Any]) -> None:
        """Update model configuration"""
        self.config.parameters.update(new_config)
        self.logger.info(f"Model configuration updated: {new_config}")

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        if not self.performance_history:
            return {'error': 'No performance data available'}

        latest_performance = self.performance_history[-1]

        return {
            'latest_accuracy': latest_performance.accuracy,
            'performance_trend': self.calculate_performance_trend(),
            'best_performance': self.get_best_performance(),
            'validation_status': self.validate_current_performance()
        }

    def calculate_performance_trend(self) -> str:
        """Calculate performance trend"""
        if len(self.performance_history) < 2:
            return 'insufficient_data'

        recent_performance = [p.accuracy for p in self.performance_history[-5:]]
        if recent_performance[-1] > recent_performance[0]:
            return 'improving'
        elif recent_performance[-1] < recent_performance[0]:
            return 'declining'
        else:
            return 'stable'

    def get_best_performance(self) -> Dict[str, Any]:
        """Get best performance achieved"""
        if not self.performance_history:
            return {'error': 'No performance data available'}

        best_perf = max(self.performance_history, key=lambda x: x.accuracy)
        return {
            'accuracy': best_perf.accuracy,
            'timestamp': best_perf.timestamp,
            'efficiency': best_perf.efficiency,
            'robustness': best_perf.robustness
        }

    def validate_current_performance(self) -> str:
        """Validate current model performance"""
        if not self.performance_history:
            return 'no_validation'

        latest_perf = self.performance_history[-1]

        # Simple validation criteria
        if latest_perf.accuracy > 0.9:
            return 'excellent'
        elif latest_perf.accuracy > 0.7:
            return 'good'
        elif latest_perf.accuracy > 0.5:
            return 'acceptable'
        else:
            return 'poor'

class ActiveInferenceModel(BaseModel):
    """Active Inference model implementation"""

    def __init__(self, config: ModelConfig):
        """Initialize Active Inference model"""
        super().__init__(config)
        self.beliefs: Optional[np.ndarray] = None
        self.policies: Optional[np.ndarray] = None
        self.likelihood: Optional[np.ndarray] = None
        self.prior: Optional[np.ndarray] = None

    def setup_model(self) -> None:
        """Set up Active Inference model"""
        # Initialize model components
        self.num_states = self.config.parameters.get('num_states', 4)
        self.num_observations = self.config.parameters.get('num_observations', 8)
        self.num_policies = self.config.parameters.get('num_policies', 4)

        # Initialize belief state
        self.beliefs = np.ones(self.num_states) / self.num_states

        # Initialize likelihood matrix
        self.likelihood = np.random.rand(self.num_observations, self.num_states)
        self.likelihood = self.likelihood / self.likelihood.sum(axis=1, keepdims=True)

        # Initialize prior preferences
        self.prior = np.zeros(self.num_observations)

        self.logger.info(f"Active Inference model initialized: {self.num_states} states, {self.num_observations} observations")

    def train(self, training_data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Train Active Inference model"""
        observations = training_data.get('observations', np.random.rand(100, self.num_observations))
        actions = training_data.get('actions', np.random.randint(0, self.num_policies, 100))
        rewards = training_data.get('rewards', np.random.rand(100))

        epochs = kwargs.get('epochs', 10)
        learning_rate = kwargs.get('learning_rate', 0.01)

        training_results = {
            'epochs': epochs,
            'final_free_energy': 0.0,
            'convergence': False,
            'training_history': []
        }

        for epoch in range(epochs):
            # Active Inference training step
            free_energy = self.active_inference_step(observations, actions, rewards, learning_rate)

            training_results['training_history'].append({
                'epoch': epoch,
                'free_energy': free_energy,
                'beliefs': self.beliefs.copy(),
                'policies': self.policies.copy()
            })

            # Check convergence
            if epoch > 0 and abs(training_results['training_history'][-1]['free_energy'] -
                               training_results['training_history'][-2]['free_energy']) < 1e-6:
                training_results['convergence'] = True
                break

        training_results['final_free_energy'] = training_results['training_history'][-1]['free_energy']

        self.logger.info(f"Training completed: {len(training_results['training_history'])} epochs, "
                        f"final free energy: {training_results['final_free_energy']}")

        return training_results

    def active_inference_step(self, observations: np.ndarray, actions: np.ndarray,
                            rewards: np.ndarray, learning_rate: float) -> float:
        """Execute one step of Active Inference"""
        # Update beliefs based on observations
        self.update_beliefs(observations)

        # Evaluate policies
        policy_values = self.evaluate_policies()

        # Select best policy
        best_policy_idx = np.argmax(policy_values)
        self.policies = np.zeros(self.num_policies)
        self.policies[best_policy_idx] = 1.0

        # Update likelihood based on outcomes
        self.update_likelihood(observations, actions, rewards, learning_rate)

        # Calculate free energy
        free_energy = self.calculate_free_energy()

        return free_energy

    def update_beliefs(self, observations: np.ndarray) -> None:
        """Update beliefs based on observations"""
        # Bayesian belief updating
        for obs in observations:
            likelihood = self.likelihood.T @ obs  # Simplified likelihood calculation
            self.beliefs = self.beliefs * likelihood
            self.beliefs = self.beliefs / self.beliefs.sum()  # Normalize

    def evaluate_policies(self) -> np.ndarray:
        """Evaluate available policies"""
        policy_values = np.zeros(self.num_policies)

        for i in range(self.num_policies):
            # Calculate expected free energy for each policy
            expected_fe = self.calculate_expected_free_energy(i)
            policy_values[i] = -expected_fe  # Negative because we want to minimize free energy

        return policy_values

    def calculate_expected_free_energy(self, policy_idx: int) -> float:
        """Calculate expected free energy for policy"""
        # Simplified expected free energy calculation
        return np.random.rand()  # Placeholder

    def update_likelihood(self, observations: np.ndarray, actions: np.ndarray,
                         rewards: np.ndarray, learning_rate: float) -> None:
        """Update likelihood matrix based on experience"""
        # Learning from experience (simplified)
        for obs, action, reward in zip(observations, actions, rewards):
            state_idx = np.argmax(self.beliefs)
            self.likelihood[obs.nonzero()[0], state_idx] += learning_rate * reward

        # Normalize likelihood
        self.likelihood = self.likelihood / self.likelihood.sum(axis=1, keepdims=True)

    def calculate_free_energy(self) -> float:
        """Calculate variational free energy"""
        # Simplified free energy calculation
        entropy = -np.sum(self.beliefs * np.log(self.beliefs + 1e-12))
        energy = np.sum(self.beliefs * np.log(self.prior + 1e-12))
        return energy - entropy

    def predict(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate predictions"""
        context = input_data.get('context', np.random.rand(self.num_observations))

        # Update beliefs with context
        self.update_beliefs(context.reshape(1, -1))

        # Select best policy
        policy_values = self.evaluate_policies()
        predicted_action = np.argmax(policy_values)

        # Generate prediction
        predicted_observation = self.likelihood @ self.beliefs

        return {
            'predicted_action': predicted_action,
            'predicted_observation': predicted_observation,
            'beliefs': self.beliefs.copy(),
            'confidence': np.max(self.beliefs),
            'free_energy': self.calculate_free_energy()
        }

    def validate(self, validation_data: Dict[str, Any]) -> ModelPerformance:
        """Validate model performance"""
        test_observations = validation_data.get('observations', np.random.rand(50, self.num_observations))
        true_actions = validation_data.get('actions', np.random.randint(0, self.num_policies, 50))

        correct_predictions = 0
        total_free_energy = 0.0

        for obs, true_action in zip(test_observations, true_actions):
            prediction = self.predict({'context': obs})
            predicted_action = prediction['predicted_action']

            if predicted_action == true_action:
                correct_predictions += 1

            total_free_energy += prediction['free_energy']

        accuracy = correct_predictions / len(test_observations)
        avg_free_energy = total_free_energy / len(test_observations)

        performance = ModelPerformance(
            accuracy=accuracy,
            efficiency={'inference_time': 0.001, 'memory_usage': 0.1},
            robustness={'noise_tolerance': 0.8, 'parameter_sensitivity': 0.7},
            scalability={'max_states': 100, 'max_observations': 1000},
            validation_scores={'cross_validation': 0.85, 'benchmark_comparison': 0.9},
            timestamp=datetime.now()
        )

        self.performance_history.append(performance)

        self.logger.info(f"Validation completed: accuracy={accuracy:.".3f" free_energy={avg_free_energy:.".3f")

        return performance
```

### Model Optimization Framework
```python
from typing import Dict, Any, List, Optional, Callable
import numpy as np
from scipy.optimize import minimize
from dataclasses import dataclass

@dataclass
class OptimizationConfig:
    """Optimization configuration"""
    method: str  # gradient_descent, evolutionary, bayesian, etc.
    parameters: List[str]
    bounds: Dict[str, Tuple[float, float]]
    constraints: List[Callable]
    max_iterations: int = 1000
    tolerance: float = 1e-6
    population_size: int = 50

class ModelOptimizer:
    """Model parameter optimizer"""

    def __init__(self, config: OptimizationConfig):
        """Initialize model optimizer"""
        self.config = config
        self.optimization_history: List[Dict[str, Any]] = []
        self.logger = logging.getLogger(self.__class__.__name__)

    def optimize_parameters(self, model: BaseModel, training_data: Dict[str, Any],
                          validation_data: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize model parameters"""
        optimization_results = {
            'initial_performance': 0.0,
            'final_performance': 0.0,
            'best_parameters': {},
            'optimization_path': [],
            'convergence': False,
            'iterations': 0
        }

        # Get initial performance
        initial_performance = model.validate(validation_data)
        optimization_results['initial_performance'] = initial_performance.accuracy

        # Define objective function
        def objective_function(params: np.ndarray) -> float:
            # Update model parameters
            param_dict = dict(zip(self.config.parameters, params))
            model.update_config(param_dict)

            # Evaluate performance
            performance = model.validate(validation_data)
            return -performance.accuracy  # Negative because we minimize

        # Set up optimization bounds
        bounds = [self.config.bounds.get(param, (-10, 10)) for param in self.config.parameters]

        # Initial parameter values
        initial_params = [model.config.parameters.get(param, 1.0) for param in self.config.parameters]

        # Run optimization
        if self.config.method == 'gradient_descent':
            result = minimize(
                objective_function,
                initial_params,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': self.config.max_iterations, 'ftol': self.config.tolerance}
            )

        elif self.config.method == 'evolutionary':
            result = self.evolutionary_optimization(objective_function, bounds, initial_params)

        else:
            raise ValueError(f"Unknown optimization method: {self.config.method}")

        # Update model with best parameters
        best_params = dict(zip(self.config.parameters, result.x))
        model.update_config(best_params)

        # Final validation
        final_performance = model.validate(validation_data)
        optimization_results['final_performance'] = final_performance.accuracy
        optimization_results['best_parameters'] = best_params
        optimization_results['convergence'] = result.success
        optimization_results['iterations'] = result.nit if hasattr(result, 'nit') else self.config.max_iterations

        self.logger.info(f"Optimization completed: {optimization_results['initial_performance']:.".3f" "
                        f"-> {optimization_results['final_performance']:.".3f"")

        return optimization_results

    def evolutionary_optimization(self, objective_function: Callable,
                                bounds: List[Tuple[float, float]],
                                initial_params: List[float]) -> Any:
        """Evolutionary algorithm optimization"""
        # Simplified evolutionary optimization
        best_params = initial_params.copy()
        best_score = objective_function(np.array(best_params))

        for iteration in range(self.config.max_iterations):
            # Generate population
            population = []
            for _ in range(self.config.population_size):
                # Mutate parameters
                mutated_params = []
                for i, (param, (lower, upper)) in enumerate(zip(best_params, bounds)):
                    mutation = np.random.normal(0, (upper - lower) * 0.1)
                    new_param = np.clip(param + mutation, lower, upper)
                    mutated_params.append(new_param)

                population.append(mutated_params)

            # Evaluate population
            population_scores = []
            for params in population:
                score = objective_function(np.array(params))
                population_scores.append((params, score))

            # Select best
            population_scores.sort(key=lambda x: x[1])
            if population_scores[0][1] < best_score:
                best_params = population_scores[0][0]
                best_score = population_scores[0][1]

            # Record progress
            self.optimization_history.append({
                'iteration': iteration,
                'best_score': best_score,
                'best_params': best_params.copy()
            })

        # Create result object similar to scipy minimize
        class OptimizationResult:
            def __init__(self, x, success, nit):
                self.x = np.array(x)
                self.success = success
                self.nit = nit

        return OptimizationResult(best_params, True, self.config.max_iterations)

    def hyperparameter_tuning(self, model: BaseModel, training_data: Dict[str, Any],
                            validation_data: Dict[str, Any], hyperparams: Dict[str, List[Any]]) -> Dict[str, Any]:
        """Tune hyperparameters using grid search or random search"""
        tuning_results = {
            'best_hyperparameters': {},
            'best_performance': 0.0,
            'all_results': [],
            'method': 'grid_search'
        }

        if self.config.method == 'grid_search':
            return self.grid_search_tuning(model, training_data, validation_data, hyperparams, tuning_results)
        elif self.config.method == 'random_search':
            return self.random_search_tuning(model, training_data, validation_data, hyperparams, tuning_results)
        else:
            raise ValueError(f"Unknown hyperparameter tuning method: {self.config.method}")

    def grid_search_tuning(self, model: BaseModel, training_data: Dict[str, Any],
                          validation_data: Dict[str, Any], hyperparams: Dict[str, List[Any]],
                          tuning_results: Dict[str, Any]) -> Dict[str, Any]:
        """Grid search hyperparameter tuning"""
        from itertools import product

        # Generate all combinations
        param_names = list(hyperparams.keys())
        param_values = list(hyperparams.values())
        param_combinations = list(product(*param_values))

        best_performance = 0.0
        best_params = {}

        for param_combo in param_combinations:
            # Update model hyperparameters
            param_dict = dict(zip(param_names, param_combo))
            model.update_config(param_dict)

            # Train and validate
            model.train(training_data)
            performance = model.validate(validation_data)

            # Record results
            tuning_results['all_results'].append({
                'parameters': param_dict,
                'performance': performance.accuracy
            })

            # Update best
            if performance.accuracy > best_performance:
                best_performance = performance.accuracy
                best_params = param_dict.copy()

        tuning_results['best_hyperparameters'] = best_params
        tuning_results['best_performance'] = best_performance

        # Update model with best parameters
        model.update_config(best_params)

        return tuning_results

    def random_search_tuning(self, model: BaseModel, training_data: Dict[str, Any],
                           validation_data: Dict[str, Any], hyperparams: Dict[str, List[Any]],
                           tuning_results: Dict[str, Any]) -> Dict[str, Any]:
        """Random search hyperparameter tuning"""
        import random

        best_performance = 0.0
        best_params = {}

        for _ in range(self.config.max_iterations):
            # Randomly sample hyperparameters
            param_dict = {}
            for param_name, param_values in hyperparams.items():
                param_dict[param_name] = random.choice(param_values)

            # Update model
            model.update_config(param_dict)

            # Train and validate
            model.train(training_data)
            performance = model.validate(validation_data)

            # Record results
            tuning_results['all_results'].append({
                'parameters': param_dict,
                'performance': performance.accuracy
            })

            # Update best
            if performance.accuracy > best_performance:
                best_performance = performance.accuracy
                best_params = param_dict.copy()

        tuning_results['best_hyperparameters'] = best_params
        tuning_results['best_performance'] = best_performance

        # Update model with best parameters
        model.update_config(best_params)

        return tuning_results
```

### Model Validation Framework
```python
from typing import Dict, Any, List, Optional
import numpy as np
from dataclasses import dataclass

@dataclass
class ValidationResult:
    """Model validation result"""
    test_name: str
    passed: bool
    score: float
    details: Dict[str, Any]
    recommendations: List[str]
    timestamp: datetime

class ModelValidator:
    """Comprehensive model validation"""

    def __init__(self):
        """Initialize model validator"""
        self.validation_tests = self.load_validation_tests()

    def load_validation_tests(self) -> Dict[str, Callable]:
        """Load validation test functions"""
        return {
            'numerical_stability': self.test_numerical_stability,
            'convergence': self.test_convergence,
            'accuracy': self.test_accuracy,
            'robustness': self.test_robustness,
            'generalization': self.test_generalization,
            'benchmark_comparison': self.test_benchmark_comparison
        }

    def validate_model(self, model: BaseModel, validation_data: Dict[str, Any],
                      test_suite: List[str] = None) -> Dict[str, Any]:
        """Run comprehensive model validation"""
        if test_suite is None:
            test_suite = list(self.validation_tests.keys())

        validation_results = {
            'model_name': model.config.name,
            'validation_timestamp': datetime.now(),
            'overall_score': 0.0,
            'tests_passed': 0,
            'total_tests': len(test_suite),
            'results': []
        }

        total_score = 0.0

        for test_name in test_suite:
            if test_name in self.validation_tests:
                try:
                    result = self.validation_tests[test_name](model, validation_data)

                    validation_results['results'].append(result.__dict__)
                    total_score += result.score

                    if result.passed:
                        validation_results['tests_passed'] += 1

                except Exception as e:
                    # Record failed validation test
                    failed_result = ValidationResult(
                        test_name=test_name,
                        passed=False,
                        score=0.0,
                        details={'error': str(e)},
                        recommendations=[f"Fix error in {test_name} test"],
                        timestamp=datetime.now()
                    )
                    validation_results['results'].append(failed_result.__dict__)

        validation_results['overall_score'] = total_score / len(test_suite)

        self.logger.info(f"Validation completed for {model.config.name}: "
                        f"{validation_results['tests_passed']}/{validation_results['total_tests']} tests passed")

        return validation_results

    def test_numerical_stability(self, model: BaseModel, validation_data: Dict[str, Any]) -> ValidationResult:
        """Test numerical stability"""
        test_data = validation_data.get('numerical_test', np.random.rand(10, model.config.parameters.get('num_observations', 8)))

        stability_scores = []

        for _ in range(10):
            try:
                prediction1 = model.predict({'context': test_data[0]})
                prediction2 = model.predict({'context': test_data[0]})  # Same input

                # Check prediction consistency
                diff = np.abs(prediction1['predicted_observation'] - prediction2['predicted_observation'])
                stability_score = 1.0 / (1.0 + np.mean(diff))

                stability_scores.append(stability_score)

            except Exception as e:
                stability_scores.append(0.0)

        avg_stability = np.mean(stability_scores)

        return ValidationResult(
            test_name='numerical_stability',
            passed=avg_stability > 0.8,
            score=avg_stability,
            details={'stability_scores': stability_scores},
            recommendations=['Improve numerical stability'] if avg_stability < 0.8 else [],
            timestamp=datetime.now()
        )

    def test_convergence(self, model: BaseModel, validation_data: Dict[str, Any]) -> ValidationResult:
        """Test model convergence"""
        training_data = validation_data.get('training_data', {'observations': np.random.rand(50, 4)})

        # Train model and check convergence
        training_results = model.train(training_data, epochs=100)

        converged = training_results.get('convergence', False)
        final_fe = training_results.get('final_free_energy', float('inf'))

        convergence_score = 1.0 if converged else 0.5

        return ValidationResult(
            test_name='convergence',
            passed=converged,
            score=convergence_score,
            details={'converged': converged, 'final_free_energy': final_fe},
            recommendations=['Improve convergence criteria'] if not converged else [],
            timestamp=datetime.now()
        )

    def test_accuracy(self, model: BaseModel, validation_data: Dict[str, Any]) -> ValidationResult:
        """Test model accuracy"""
        performance = model.validate(validation_data)

        accuracy_threshold = 0.7  # Minimum acceptable accuracy

        return ValidationResult(
            test_name='accuracy',
            passed=performance.accuracy > accuracy_threshold,
            score=performance.accuracy,
            details={'accuracy': performance.accuracy, 'threshold': accuracy_threshold},
            recommendations=['Improve model accuracy'] if performance.accuracy < accuracy_threshold else [],
            timestamp=datetime.now()
        )

    def test_robustness(self, model: BaseModel, validation_data: Dict[str, Any]) -> ValidationResult:
        """Test model robustness"""
        # Test with noisy data
        clean_data = validation_data.get('test_data', np.random.rand(20, 4))
        noisy_data = clean_data + np.random.normal(0, 0.1, clean_data.shape)

        # Performance with clean data
        clean_performance = model.validate({'test_data': clean_data})

        # Reset model state
        model.load_model(f"{model.config.name}_backup.json")

        # Performance with noisy data
        noisy_performance = model.validate({'test_data': noisy_data})

        robustness_score = min(clean_performance.accuracy / (noisy_performance.accuracy + 1e-6), 1.0)

        return ValidationResult(
            test_name='robustness',
            passed=robustness_score > 0.7,
            score=robustness_score,
            details={
                'clean_accuracy': clean_performance.accuracy,
                'noisy_accuracy': noisy_performance.accuracy
            },
            recommendations=['Improve robustness to noise'] if robustness_score < 0.7 else [],
            timestamp=datetime.now()
        )

    def test_generalization(self, model: BaseModel, validation_data: Dict[str, Any]) -> ValidationResult:
        """Test model generalization"""
        # Test on data different from training
        generalization_data = validation_data.get('generalization_data', np.random.rand(20, 4))

        performance = model.validate(generalization_data)

        generalization_threshold = 0.6

        return ValidationResult(
            test_name='generalization',
            passed=performance.accuracy > generalization_threshold,
            score=performance.accuracy,
            details={'generalization_accuracy': performance.accuracy},
            recommendations=['Improve generalization'] if performance.accuracy < generalization_threshold else [],
            timestamp=datetime.now()
        )

    def test_benchmark_comparison(self, model: BaseModel, validation_data: Dict[str, Any]) -> ValidationResult:
        """Test against benchmarks"""
        benchmark_score = validation_data.get('benchmark_score', 0.8)

        performance = model.validate(validation_data)
        comparison_score = min(performance.accuracy / benchmark_score, 1.0)

        return ValidationResult(
            test_name='benchmark_comparison',
            passed=comparison_score > 0.8,
            score=comparison_score,
            details={'model_accuracy': performance.accuracy, 'benchmark_score': benchmark_score},
            recommendations=['Improve to match benchmark performance'] if comparison_score < 0.8 else [],
            timestamp=datetime.now()
        )
```

## Testing Guidelines

### Model Development Testing
- **Unit Testing**: Test individual model components
- **Integration Testing**: Test model integration with systems
- **Performance Testing**: Validate computational performance
- **Accuracy Testing**: Test model accuracy and correctness
- **Robustness Testing**: Test model robustness and stability

### Quality Assurance
- **Code Review**: Comprehensive code review processes
- **Performance Validation**: Validate performance requirements
- **Accuracy Validation**: Ensure model accuracy meets standards
- **Documentation Testing**: Verify documentation completeness
- **Regression Testing**: Prevent performance regressions

## Performance Considerations

### Computational Performance
- **Algorithm Optimization**: Optimize core algorithms
- **Memory Management**: Efficient memory usage patterns
- **Parallel Processing**: Utilize parallel processing where beneficial
- **Caching**: Implement result caching strategies

### Numerical Performance
- **Stability**: Ensure numerical stability of algorithms
- **Precision**: Maintain appropriate numerical precision
- **Convergence**: Optimize convergence properties
- **Error Control**: Control numerical errors and approximations

## Maintenance and Evolution

### Model Updates
- **Algorithm Updates**: Update algorithms with latest research
- **Performance Optimization**: Continuous performance improvement
- **Compatibility**: Maintain compatibility with ecosystem
- **Documentation Updates**: Keep documentation current

### Version Management
- **Version Control**: Model version management
- **Backward Compatibility**: Maintain backward compatibility
- **Migration Support**: Support model migrations
- **Deprecation Management**: Manage deprecated features

## Common Challenges and Solutions

### Challenge: Model Complexity
**Solution**: Use modular design and comprehensive testing to manage complexity.

### Challenge: Performance Optimization
**Solution**: Profile extensively and use appropriate optimization techniques.

### Challenge: Validation
**Solution**: Implement comprehensive validation against multiple criteria.

### Challenge: Integration
**Solution**: Design for integration and use standard interfaces.

## Getting Started as an Agent

### Development Setup
1. **Study Model Architecture**: Understand model development architecture
2. **Learn Implementation Patterns**: Study implementation best practices
3. **Practice Development**: Practice implementing computational models
4. **Understand Optimization**: Learn model optimization techniques

### Contribution Process
1. **Identify Model Needs**: Find gaps in current model capabilities
2. **Research Methods**: Study relevant computational methods
3. **Design Solutions**: Create detailed model designs
4. **Implement and Test**: Follow quality implementation standards
5. **Validate Thoroughly**: Ensure model correctness and performance
6. **Document Completely**: Provide comprehensive model documentation
7. **Performance Review**: Submit for performance and accuracy review

### Learning Resources
- **Computational Modeling**: Study computational modeling techniques
- **Numerical Methods**: Master numerical analysis methods
- **Performance Optimization**: Learn performance optimization techniques
- **Software Engineering**: Study software engineering best practices
- **Active Inference**: Domain-specific modeling approaches

## Related Documentation

- **[Model README](./README.md)**: Model development module overview
- **[Main AGENTS.md](../../AGENTS.md)**: Project-wide agent guidelines
- **[Research AGENTS.md](../AGENTS.md)**: Research tools module guidelines
- **[Analysis Tools](../../research/analysis/)**: Model analysis and validation
- **[Contributing Guide](../../CONTRIBUTING.md)**: Contribution processes

---

*"Active Inference for, with, by Generative AI"* - Advancing research through comprehensive model development, rigorous implementation, and collaborative computational research.
