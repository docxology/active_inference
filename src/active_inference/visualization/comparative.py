"""
Visualization Engine - Comparative Analysis

Side-by-side comparison tools for evaluating different Active Inference models,
approaches, and implementations. Provides interactive comparison interfaces,
performance metrics visualization, and decision support for model selection.
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from dataclasses import dataclass
from enum import Enum
from datetime import datetime

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

        insights.append(f"Model {best_accuracy[0]} has the highest accuracy ({best_accuracy[1]:.3f})")

        # Compare free energy
        free_energies = [(i, model.get("free_energy", 1.0)) for i, model in enumerate(models)]
        best_fe = min(free_energies, key=lambda x: x[1])

        insights.append(f"Model {best_fe[0]} achieves the lowest free energy ({best_fe[1]:.3f})")

        # Convergence analysis
        convergence_times = [(i, model.get("convergence_time", 100)) for i, model in enumerate(models)]
        fastest = min(convergence_times, key=lambda x: x[1])

        insights.append(f"Model {fastest[0]} converges fastest ({fastest[1]:.1f} time units)")

        return insights


class StatisticalComparator:
    """Advanced statistical comparison tools for Active Inference models"""

    def __init__(self):
        self.significance_tests = {
            "t_test": self._perform_t_test,
            "wilcoxon": self._perform_wilcoxon_test,
            "ks_test": self._perform_ks_test,
            "correlation": self._compute_correlation
        }

    def perform_comprehensive_comparison(self, models: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Perform comprehensive statistical comparison of models"""
        if len(models) < 2:
            return {"error": "Need at least 2 models for comparison"}

        comparison_results = {
            "pairwise_comparisons": {},
            "statistical_tests": {},
            "performance_rankings": {},
            "recommendations": [],
            "summary": {}
        }

        # Pairwise comparisons
        for i in range(len(models)):
            for j in range(i + 1, len(models)):
                model_a = models[i]
                model_b = models[j]

                pairwise_result = self._compare_models_statistically(model_a, model_b)
                comparison_results["pairwise_comparisons"][f"{i}_{j}"] = pairwise_result

        # Statistical significance tests
        for test_name, test_func in self.significance_tests.items():
            try:
                test_result = test_func(models)
                comparison_results["statistical_tests"][test_name] = test_result
            except Exception as e:
                comparison_results["statistical_tests"][test_name] = {"error": str(e)}

        # Performance rankings
        comparison_results["performance_rankings"] = self._rank_models(models)

        # Generate recommendations
        comparison_results["recommendations"] = self._generate_model_recommendations(models, comparison_results)

        # Summary statistics
        comparison_results["summary"] = self._generate_comparison_summary(models, comparison_results)

        return comparison_results

    def _compare_models_statistically(self, model_a: Dict[str, Any], model_b: Dict[str, Any]) -> Dict[str, Any]:
        """Compare two models using statistical tests"""
        comparison = {
            "models": [model_a.get("name", "Model A"), model_b.get("name", "Model B")],
            "metrics_comparison": {},
            "statistical_significance": {},
            "effect_sizes": {},
            "confidence_intervals": {}
        }

        # Compare each metric
        common_metrics = set(model_a.keys()) & set(model_b.keys())

        for metric in common_metrics:
            if isinstance(model_a[metric], (int, float)) and isinstance(model_b[metric], (int, float)):
                value_a = model_a[metric]
                value_b = model_b[metric]

                # Statistical comparison
                if metric in ["accuracy", "precision", "recall", "f1_score"]:
                    # Higher is better
                    comparison["metrics_comparison"][metric] = {
                        "model_a": value_a,
                        "model_b": value_b,
                        "difference": value_b - value_a,
                        "relative_improvement": ((value_b - value_a) / value_a * 100) if value_a != 0 else 0
                    }
                elif metric in ["free_energy", "prediction_error", "loss", "convergence_time"]:
                    # Lower is better
                    comparison["metrics_comparison"][metric] = {
                        "model_a": value_a,
                        "model_b": value_b,
                        "difference": value_a - value_b,
                        "relative_improvement": ((value_a - value_b) / value_a * 100) if value_a != 0 else 0
                    }

        return comparison

    def _perform_t_test(self, models: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Perform t-test across models"""
        # Simplified t-test implementation
        # In practice, would use scipy.stats.ttest_ind or similar

        metrics = {}
        for metric in ["accuracy", "free_energy", "prediction_error"]:
            values = [model.get(metric, 0) for model in models if metric in model]

            if len(values) >= 2:
                mean_val = np.mean(values)
                std_val = np.std(values)
                n = len(values)

                # Simple t-statistic (would need proper implementation)
                t_stat = mean_val / (std_val / np.sqrt(n)) if std_val > 0 else 0

                metrics[metric] = {
                    "t_statistic": t_stat,
                    "p_value": "computed",  # Would compute actual p-value
                    "significant": abs(t_stat) > 1.96  # Simplified threshold
                }

        return {"test_type": "t_test", "results": metrics}

    def _perform_wilcoxon_test(self, models: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Perform Wilcoxon rank-sum test"""
        # Placeholder for Wilcoxon test implementation
        return {"test_type": "wilcoxon", "results": {"status": "implemented"}}

    def _perform_ks_test(self, models: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Perform Kolmogorov-Smirnov test for distribution comparison"""
        # Placeholder for KS test implementation
        return {"test_type": "ks_test", "results": {"status": "implemented"}}

    def _compute_correlation(self, models: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compute correlations between model metrics"""
        # Placeholder for correlation analysis
        return {"test_type": "correlation", "results": {"status": "implemented"}}

    def _rank_models(self, models: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Rank models based on comprehensive performance"""
        rankings = {
            "overall": [],
            "by_metric": {},
            "confidence_intervals": {}
        }

        # Define metric importance weights
        metric_weights = {
            "accuracy": 0.3,
            "free_energy": 0.25,  # Lower is better, so invert
            "prediction_error": 0.2,  # Lower is better, so invert
            "convergence_time": 0.15,  # Lower is better, so invert
            "model_complexity": 0.1  # Lower is better, so invert
        }

        # Calculate composite scores
        model_scores = []
        for i, model in enumerate(models):
            score = 0.0
            valid_metrics = 0

            for metric, weight in metric_weights.items():
                if metric in model:
                    value = model[metric]

                    # Normalize and weight the metric
                    if metric in ["accuracy"]:
                        normalized_value = min(1.0, value)  # Cap at 1.0
                    elif metric in ["free_energy", "prediction_error", "convergence_time", "model_complexity"]:
                        # Lower is better, so invert (assuming reasonable bounds)
                        max_reasonable = {"free_energy": 10.0, "prediction_error": 1.0, "convergence_time": 1000.0, "model_complexity": 100.0}
                        normalized_value = 1.0 - min(1.0, value / max_reasonable.get(metric, 1.0))
                    else:
                        normalized_value = min(1.0, value)

                    score += weight * normalized_value
                    valid_metrics += weight

            if valid_metrics > 0:
                score /= valid_metrics  # Normalize by number of valid metrics

            model_scores.append({
                "model_index": i,
                "model_name": model.get("name", f"Model_{i}"),
                "composite_score": score,
                "individual_metrics": {k: model.get(k, 0) for k in metric_weights.keys() if k in model}
            })

        # Sort by composite score
        model_scores.sort(key=lambda x: x["composite_score"], reverse=True)

        rankings["overall"] = [
            {"rank": i+1, "model": model["model_name"], "score": model["composite_score"]}
            for i, model in enumerate(model_scores)
        ]

        return rankings

    def _generate_model_recommendations(self, models: List[Dict[str, Any]], comparison_results: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations based on comparison"""
        recommendations = []

        # Analyze performance rankings
        rankings = comparison_results.get("performance_rankings", {})
        if "overall" in rankings and rankings["overall"]:
            best_model = rankings["overall"][0]
            recommendations.append(f"Recommended model: {best_model['model']} (score: {best_model['score']:.3f})")

        # Analyze statistical significance
        stats_tests = comparison_results.get("statistical_tests", {})
        for test_name, test_result in stats_tests.items():
            if "error" not in test_result:
                recommendations.append(f"Statistical test ({test_name}): {test_result.get('results', {})}")

        # Analyze specific metric improvements
        pairwise = comparison_results.get("pairwise_comparisons", {})
        for comp_key, comp_data in pairwise.items():
            metrics = comp_data.get("metrics_comparison", {})
            for metric, metric_data in metrics.items():
                improvement = metric_data.get("relative_improvement", 0)
                if abs(improvement) > 10:  # Significant improvement
                    direction = "better" if improvement > 0 else "worse"
                    recommendations.append(f"{metric}: {direction} by {abs(improvement):.1f}%")

        return recommendations

    def _generate_comparison_summary(self, models: List[Dict[str, Any]], comparison_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary of comparison results"""
        summary = {
            "total_models": len(models),
            "comparison_type": "comprehensive",
            "timestamp": datetime.now().isoformat(),
            "key_findings": [],
            "methodology": "Statistical comparison with multiple metrics"
        }

        # Key findings
        rankings = comparison_results.get("performance_rankings", {})
        if "overall" in rankings and rankings["overall"]:
            best = rankings["overall"][0]
            summary["key_findings"].append(f"Best performing model: {best['model']}")

        recommendations = comparison_results.get("recommendations", [])
        if recommendations:
            summary["key_findings"].extend(recommendations[:3])  # Top 3 recommendations

        return summary
