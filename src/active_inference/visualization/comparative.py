"""
Visualization Engine - Comparative Analysis

Side-by-side comparison tools for evaluating different Active Inference models,
approaches, and implementations. Provides interactive comparison interfaces,
performance metrics visualization, and decision support for model selection.
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ComparisonType(Enum):
    """Types of comparisons supported"""
    MODEL_COMPARISON = "model_comparison"
    PARAMETER_STUDY = "parameter_study"
    ALGORITHM_COMPARISON = "algorithm_comparison"
    PERFORMANCE_ANALYSIS = "performance_analysis"


@dataclass
class ModelComparison:
    """Comparison between different models"""
    model_a: Dict[str, Any]
    model_b: Dict[str, Any]
    metrics: Dict[str, float]
    recommendations: List[str] = None

    def __post_init__(self):
        if self.recommendations is None:
            self.recommendations = []


class ComparisonTool:
    """Tools for comparing models and approaches"""

    def __init__(self):
        self.comparisons: Dict[str, ModelComparison] = {}
        logger.info("ComparisonTool initialized")

    def compare_models(self, model_a_data: Dict[str, Any], model_b_data: Dict[str, Any],
                      comparison_metrics: List[str] = None) -> ModelComparison:
        """Compare two models based on performance metrics"""
        if comparison_metrics is None:
            comparison_metrics = ["accuracy", "free_energy", "convergence_time"]

        metrics = {}

        for metric in comparison_metrics:
            if metric in model_a_data and metric in model_b_data:
                value_a = model_a_data[metric]
                value_b = model_b_data[metric]

                # Calculate relative difference
                if isinstance(value_a, (int, float)) and isinstance(value_b, (int, float)):
                    if value_a != 0:
                        relative_diff = ((value_b - value_a) / value_a) * 100
                    else:
                        relative_diff = 0.0

                    metrics[f"{metric}_relative_diff"] = relative_diff
                    metrics[f"{metric}_winner"] = "A" if value_a > value_b else "B"

        # Generate recommendations
        recommendations = []

        if metrics.get("accuracy_winner") == "A":
            recommendations.append("Model A has higher accuracy")
        else:
            recommendations.append("Model B has higher accuracy")

        if metrics.get("free_energy_relative_diff", 0) < -10:
            recommendations.append("Model B achieves significantly lower free energy")
        elif metrics.get("free_energy_relative_diff", 0) > 10:
            recommendations.append("Model A achieves significantly lower free energy")

        comparison = ModelComparison(
            model_a=model_a_data,
            model_b=model_b_data,
            metrics=metrics,
            recommendations=recommendations
        )

        comparison_id = f"comp_{len(self.comparisons)}"
        self.comparisons[comparison_id] = comparison

        logger.info(f"Created model comparison: {comparison_id}")
        return comparison

    def create_comparison_matrix(self, models: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create pairwise comparison matrix for multiple models"""
        n_models = len(models)
        matrix = {
            "models": [model.get("name", f"Model_{i}") for i, model in enumerate(models)],
            "pairwise_comparisons": {},
            "summary": {}
        }

        # Create pairwise comparisons
        for i in range(n_models):
            for j in range(i + 1, n_models):
                model_a = models[i]
                model_b = models[j]

                comparison = self.compare_models(model_a, model_b)
                matrix["pairwise_comparisons"][f"{i}_{j}"] = {
                    "models": [model_a.get("name", f"Model_{i}"), model_b.get("name", f"Model_{j}")],
                    "metrics": comparison.metrics,
                    "recommendations": comparison.recommendations
                }

        # Generate summary
        matrix["summary"] = {
            "total_models": n_models,
            "total_comparisons": len(matrix["pairwise_comparisons"]),
            "best_model": self._determine_best_model(models)
        }

        return matrix

    def _determine_best_model(self, models: List[Dict[str, Any]]) -> Optional[str]:
        """Determine best performing model"""
        if not models:
            return None

        # Simple scoring based on available metrics
        scores = {}

        for i, model in enumerate(models):
            score = 0.0
            count = 0

            # Weight different metrics
            if "accuracy" in model:
                score += model["accuracy"] * 0.4
                count += 0.4
            if "free_energy" in model:
                # Lower free energy is better, so invert
                score += (1.0 - min(1.0, model["free_energy"])) * 0.4
                count += 0.4
            if "convergence_time" in model:
                # Faster convergence is better, so invert time
                score += (1.0 - min(1.0, model["convergence_time"] / 100.0)) * 0.2
                count += 0.2

            if count > 0:
                scores[f"Model_{i}"] = score / count

        if scores:
            return max(scores, key=scores.get)

        return None


class ModelComparator:
    """Interactive model comparison interface"""

    def __init__(self):
        self.comparison_tool = ComparisonTool()
        self.visualization_config: Dict[str, Any] = {}

    def create_comparison_dashboard(self, models: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create interactive comparison dashboard"""
        logger.info(f"Creating comparison dashboard for {len(models)} models")

        # Generate comparison matrix
        comparison_matrix = self.comparison_tool.create_comparison_matrix(models)

        # Create dashboard configuration
        dashboard = {
            "type": "model_comparison",
            "title": f"Model Comparison - {len(models)} Models",
            "models": comparison_matrix["models"],
            "comparison_matrix": comparison_matrix["pairwise_comparisons"],
            "summary": comparison_matrix["summary"],
            "visualization_components": self._create_visualization_components(models)
        }

        return dashboard

    def _create_visualization_components(self, models: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Create visualization components for comparison"""
        components = []

        # Performance radar chart
        components.append({
            "type": "radar_chart",
            "title": "Performance Overview",
            "metrics": ["accuracy", "free_energy", "convergence_time"],
            "data": models
        })

        # Metric comparison bars
        components.append({
            "type": "bar_chart",
            "title": "Model Metrics",
            "metrics": list(models[0].keys()) if models else [],
            "data": models
        })

        # Recommendations panel
        components.append({
            "type": "recommendations",
            "title": "Comparison Insights",
            "content": self._generate_insights(models)
        })

        return components

    def _generate_insights(self, models: List[Dict[str, Any]]) -> List[str]:
        """Generate insights from model comparison"""
        insights = []

        if len(models) < 2:
            insights.append("Need at least 2 models for meaningful comparison")
            return insights

        # Compare performance metrics
        accuracies = [(i, model.get("accuracy", 0)) for i, model in enumerate(models)]
        best_accuracy = max(accuracies, key=lambda x: x[1])

        insights.append(f"Model {best_accuracy[0]} has the highest accuracy ({best_accuracy[1]".3f"})")

        # Compare free energy
        free_energies = [(i, model.get("free_energy", 1.0)) for i, model in enumerate(models)]
        best_fe = min(free_energies, key=lambda x: x[1])

        insights.append(f"Model {best_fe[0]} achieves the lowest free energy ({best_fe[1]".3f"})")

        # Convergence analysis
        convergence_times = [(i, model.get("convergence_time", 100)) for i, model in enumerate(models)]
        fastest = min(convergence_times, key=lambda x: x[1])

        insights.append(f"Model {fastest[0]} converges fastest ({fastest[1]".1f"} time units)")

        return insights
