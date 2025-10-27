"""
Research Framework - Analysis Tools

Statistical and information-theoretic analysis tools for Active Inference research.
Provides methods for analyzing model performance, information flow, and system dynamics.
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from scipy import stats
from pathlib import Path

logger = logging.getLogger(__name__)


class StatisticalAnalysis:
    """Statistical analysis tools for research data"""

    def __init__(self):
        logger.info("StatisticalAnalysis initialized")

    def compute_descriptive_stats(self, data: List[float]) -> Dict[str, float]:
        """Compute descriptive statistics for a dataset"""
        if not data:
            return {}

        return {
            "mean": float(np.mean(data)),
            "median": float(np.median(data)),
            "std": float(np.std(data)),
            "min": float(np.min(data)),
            "max": float(np.max(data)),
            "n_samples": len(data)
        }

    def correlation_analysis(self, x: List[float], y: List[float]) -> Dict[str, float]:
        """Compute correlation between two variables"""
        if len(x) != len(y) or len(x) == 0:
            return {}

        correlation = np.corrcoef(x, y)[0, 1]

        return {
            "pearson_r": float(correlation),
            "pearson_r_squared": float(correlation ** 2),
            "n_samples": len(x)
        }

    def significance_test(self, data1: List[float], data2: List[float],
                         test_type: str = "t_test") -> Dict[str, Any]:
        """Perform significance testing between two datasets"""
        if len(data1) == 0 or len(data2) == 0:
            return {"error": "Empty datasets"}

        if test_type == "t_test":
            t_stat, p_value = stats.ttest_ind(data1, data2)
            return {
                "test_type": "independent_t_test",
                "t_statistic": float(t_stat),
                "p_value": float(p_value),
                "significant": p_value < 0.05,
                "n1": len(data1),
                "n2": len(data2)
            }
        else:
            return {"error": f"Test type {test_type} not implemented"}

    def information_theory_metrics(self, data: List[List[float]]) -> Dict[str, float]:
        """Compute information-theoretic metrics"""
        # Placeholder for entropy, mutual information, etc.
        return {
            "estimated_entropy": 0.0,
            "estimated_mutual_information": 0.0
        }


class AnalysisTools:
    """Comprehensive analysis toolkit for research"""

    def __init__(self):
        self.statistical_analysis = StatisticalAnalysis()
        logger.info("AnalysisTools initialized")

    def analyze_experiment_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze results from experiment runs"""
        analysis = {
            "experiment_id": results.get("experiment_id"),
            "performance_metrics": {},
            "statistical_analysis": {},
            "recommendations": []
        }

        # Analyze free energy trajectory
        if "free_energy" in results:
            free_energy = results["free_energy"]
            analysis["performance_metrics"]["free_energy_stats"] = \
                self.statistical_analysis.compute_descriptive_stats(free_energy)

            analysis["performance_metrics"]["final_free_energy"] = free_energy[-1] if free_energy else None
            analysis["performance_metrics"]["free_energy_trend"] = \
                "decreasing" if len(free_energy) > 1 and free_energy[-1] < free_energy[0] else "increasing"

        # Analyze accuracy trajectory
        if "accuracy" in results:
            accuracy = results["accuracy"]
            analysis["performance_metrics"]["accuracy_stats"] = \
                self.statistical_analysis.compute_descriptive_stats(accuracy)

            analysis["performance_metrics"]["final_accuracy"] = accuracy[-1] if accuracy else None
            analysis["performance_metrics"]["accuracy_trend"] = \
                "increasing" if len(accuracy) > 1 and accuracy[-1] > accuracy[0] else "decreasing"

        # Generate recommendations
        if analysis["performance_metrics"].get("free_energy_trend") == "increasing":
            analysis["recommendations"].append("Consider adjusting model parameters to reduce free energy")

        if analysis["performance_metrics"].get("accuracy_trend") == "decreasing":
            analysis["recommendations"].append("Model accuracy is declining, investigate learning dynamics")

        return analysis

    def compare_models(self, model_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compare multiple model results"""
        if not model_results:
            return {}

        comparison = {
            "n_models": len(model_results),
            "model_names": [r.get("model_name", f"model_{i}") for i, r in enumerate(model_results)],
            "performance_comparison": {},
            "statistical_tests": {}
        }

        # Extract performance metrics
        free_energy_values = []
        accuracy_values = []

        for result in model_results:
            if "free_energy" in result and result["free_energy"]:
                free_energy_values.append(result["free_energy"][-1])  # Final value
            if "accuracy" in result and result["accuracy"]:
                accuracy_values.append(result["accuracy"][-1])  # Final value

        # Compare free energy
        if len(free_energy_values) > 1:
            comparison["performance_comparison"]["free_energy_comparison"] = \
                self.statistical_analysis.significance_test(free_energy_values[:1], free_energy_values[1:])

        # Compare accuracy
        if len(accuracy_values) > 1:
            comparison["performance_comparison"]["accuracy_comparison"] = \
                self.statistical_analysis.significance_test(accuracy_values[:1], accuracy_values[1:])

        return comparison

