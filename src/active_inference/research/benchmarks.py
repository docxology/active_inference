"""
Research Framework - Benchmarking Suite

Standardized evaluation metrics and benchmarking tools for Active Inference models.
Provides comprehensive performance evaluation, comparison frameworks, and
standardized testing protocols.
"""

import logging
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from enum import Enum
import time

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of performance metrics"""
    ACCURACY = "accuracy"
    PRECISION = "precision"
    RECALL = "recall"
    F1_SCORE = "f1_score"
    FREE_ENERGY = "free_energy"
    CONVERGENCE_TIME = "convergence_time"
    COMPUTATIONAL_COST = "computational_cost"
    INFORMATION_EFFICIENCY = "information_efficiency"


@dataclass
class PerformanceMetrics:
    """Container for performance metrics"""
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    free_energy: float = 0.0
    convergence_time: float = 0.0
    computational_cost: float = 0.0
    information_efficiency: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary"""
        return {
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
            "free_energy": self.free_energy,
            "convergence_time": self.convergence_time,
            "computational_cost": self.computational_cost,
            "information_efficiency": self.information_efficiency
        }


class BenchmarkSuite:
    """Comprehensive benchmarking suite for model evaluation"""

    def __init__(self):
        self.metrics: Dict[str, PerformanceMetrics] = {}
        self.baselines: Dict[str, PerformanceMetrics] = {}
        logger.info("BenchmarkSuite initialized")

    def register_baseline(self, model_name: str, metrics: PerformanceMetrics) -> None:
        """Register baseline performance for a model"""
        self.baselines[model_name] = metrics
        logger.info(f"Registered baseline for {model_name}")

    def evaluate_model(self, model_name: str, evaluation_function: Callable,
                      test_data: Dict[str, Any] = None) -> PerformanceMetrics:
        """Evaluate a model using standardized metrics"""
        logger.info(f"Evaluating model: {model_name}")

        start_time = time.time()

        # Run evaluation
        results = evaluation_function(test_data or {})

        evaluation_time = time.time() - start_time

        # Extract metrics from results
        metrics = PerformanceMetrics()

        if isinstance(results, dict):
            metrics.accuracy = results.get("accuracy", 0.0)
            metrics.precision = results.get("precision", 0.0)
            metrics.recall = results.get("recall", 0.0)
            metrics.f1_score = results.get("f1_score", 0.0)
            metrics.free_energy = results.get("free_energy", 0.0)
            metrics.convergence_time = results.get("convergence_time", evaluation_time)
            metrics.computational_cost = evaluation_time
            metrics.information_efficiency = results.get("information_efficiency", 0.0)

        self.metrics[model_name] = metrics

        logger.info(f"Model {model_name} evaluation completed in {evaluation_time".2f"}s")
        return metrics

    def compare_models(self, model_names: List[str]) -> Dict[str, Any]:
        """Compare performance across multiple models"""
        comparison = {
            "models": model_names,
            "rankings": {},
            "improvements": {},
            "summary": {}
        }

        if not self.metrics:
            return comparison

        # Rank models by each metric
        for metric_name in PerformanceMetrics.__dataclass_fields__.keys():
            values = [(name, getattr(self.metrics[name], metric_name))
                     for name in model_names if name in self.metrics]

            if values:
                # Sort by metric value (higher is better for most metrics)
                sorted_values = sorted(values, key=lambda x: x[1], reverse=True)
                comparison["rankings"][metric_name] = sorted_values

        # Calculate improvements over baselines
        for model_name in model_names:
            if model_name in self.metrics and model_name in self.baselines:
                current = self.metrics[model_name]
                baseline = self.baselines[model_name]

                improvements = {}
                for field in PerformanceMetrics.__dataclass_fields__.keys():
                    current_val = getattr(current, field)
                    baseline_val = getattr(baseline, field)

                    if baseline_val != 0:
                        improvement = ((current_val - baseline_val) / baseline_val) * 100
                    else:
                        improvement = 0.0

                    improvements[field] = improvement

                comparison["improvements"][model_name] = improvements

        # Generate summary
        comparison["summary"] = {
            "total_models": len(model_names),
            "evaluated_models": len([name for name in model_names if name in self.metrics]),
            "best_overall": self._determine_best_model()
        }

        return comparison

    def _determine_best_model(self) -> Optional[str]:
        """Determine the best performing model overall"""
        if not self.metrics:
            return None

        # Simple scoring: average normalized metrics
        scores = {}

        for model_name, metrics in self.metrics.items():
            score = 0.0
            count = 0

            for field in ["accuracy", "precision", "recall", "f1_score"]:
                value = getattr(metrics, field)
                if value > 0:
                    score += value
                    count += 1

            if count > 0:
                scores[model_name] = score / count

        if scores:
            return max(scores, key=scores.get)

        return None

    def generate_report(self, output_format: str = "dict") -> Dict[str, Any]:
        """Generate comprehensive benchmark report"""
        report = {
            "timestamp": "2024-10-27T12:00:00",
            "total_models": len(self.metrics),
            "baselines": {name: metrics.to_dict() for name, metrics in self.baselines.items()},
            "results": {name: metrics.to_dict() for name, metrics in self.metrics.items()},
            "comparison": self.compare_models(list(self.metrics.keys()))
        }

        return report
