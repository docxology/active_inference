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
        """Compute comprehensive information-theoretic metrics"""
        if not data or not data[0]:
            return {"error": "Empty data"}

        metrics = {}

        # Compute entropy for each variable
        for i, variable_data in enumerate(data):
            if len(variable_data) > 1:
                # Discretize data for entropy calculation
                hist, bin_edges = np.histogram(variable_data, bins=min(50, len(set(variable_data))))
                probs = hist / np.sum(hist)

                # Shannon entropy
                entropy = -np.sum(probs * np.log2(probs + 1e-10))
                metrics[f"entropy_var_{i}"] = entropy

        # Compute mutual information between pairs of variables
        for i in range(len(data)):
            for j in range(i + 1, len(data)):
                if len(data[i]) == len(data[j]) and len(data[i]) > 1:
                    mutual_info = self._compute_mutual_information(data[i], data[j])
                    metrics[f"mutual_info_{i}_{j}"] = mutual_info

        # Compute KL divergence between first two variables if available
        if len(data) >= 2 and len(data[0]) == len(data[1]) and len(data[0]) > 1:
            kl_div = self._compute_kl_divergence(data[0], data[1])
            metrics["kl_divergence_0_1"] = kl_div

        return metrics

    def _compute_mutual_information(self, x: List[float], y: List[float]) -> float:
        """Compute mutual information between two variables"""
        # Convert to numpy arrays
        x_array = np.array(x)
        y_array = np.array(y)

        # Discretize data
        x_bins = min(20, len(set(x)))
        y_bins = min(20, len(set(y)))

        # Create 2D histogram
        hist_2d, x_edges, y_edges = np.histogram2d(x_array, y_array, bins=[x_bins, y_bins])
        hist_2d = hist_2d / np.sum(hist_2d)

        # Compute marginal distributions
        hist_x = np.sum(hist_2d, axis=1)
        hist_y = np.sum(hist_2d, axis=0)

        # Mutual information calculation
        mutual_info = 0.0
        for i in range(len(hist_x)):
            for j in range(len(hist_y)):
                if hist_2d[i, j] > 0 and hist_x[i] > 0 and hist_y[j] > 0:
                    mutual_info += hist_2d[i, j] * np.log2(hist_2d[i, j] / (hist_x[i] * hist_y[j]))

        return mutual_info

    def _compute_kl_divergence(self, p_data: List[float], q_data: List[float]) -> float:
        """Compute KL divergence between two distributions"""
        # Discretize both distributions
        bins = min(50, len(set(p_data)), len(set(q_data)))

        p_hist, _ = np.histogram(p_data, bins=bins, density=True)
        q_hist, _ = np.histogram(q_data, bins=bins, density=True)

        # Add small epsilon to avoid division by zero
        p_hist = p_hist + 1e-10
        q_hist = q_hist + 1e-10

        # Normalize
        p_hist = p_hist / np.sum(p_hist)
        q_hist = q_hist / np.sum(q_hist)

        # KL divergence
        kl_div = np.sum(p_hist * np.log2(p_hist / q_hist))

        return kl_div

    def bayesian_analysis(self, data: Dict[str, Any], prior: Dict[str, float] = None) -> Dict[str, Any]:
        """Perform Bayesian analysis on research data"""
        if prior is None:
            prior = {"mean": 0.0, "variance": 1.0}

        analysis = {
            "prior": prior,
            "likelihood": {},
            "posterior": {},
            "evidence": 0.0
        }

        # Compute likelihood (simplified Gaussian likelihood)
        if "observations" in data and data["observations"]:
            observations = data["observations"]
            n = len(observations)

            # Maximum likelihood estimates
            sample_mean = np.mean(observations)
            sample_var = np.var(observations)

            # Bayesian posterior (conjugate prior for Gaussian)
            prior_mean = prior["mean"]
            prior_var = prior["variance"]
            prior_precision = 1.0 / prior_var
            sample_precision = 1.0 / sample_var if sample_var > 0 else 1.0

            # Posterior parameters
            posterior_precision = prior_precision + n * sample_precision
            posterior_var = 1.0 / posterior_precision
            posterior_mean = (prior_mean * prior_precision + n * sample_mean * sample_precision) / posterior_precision

            analysis["posterior"] = {
                "mean": posterior_mean,
                "variance": posterior_var,
                "precision": posterior_precision
            }

            # Compute log likelihood (simplified)
            log_likelihood = -0.5 * n * np.log(2 * np.pi * sample_var) - 0.5 * np.sum((np.array(observations) - sample_mean)**2) / sample_var
            analysis["log_likelihood"] = log_likelihood

        return analysis

    def time_series_analysis(self, time_series: List[float], window_size: int = 10) -> Dict[str, Any]:
        """Analyze time series data for trends and patterns"""
        if len(time_series) < window_size:
            return {"error": "Insufficient data for time series analysis"}

        analysis = {
            "trend_analysis": {},
            "volatility_analysis": {},
            "autocorrelation": {},
            "stationarity": {}
        }

        # Trend analysis (linear regression on sliding windows)
        ts_array = np.array(time_series)

        # Simple linear trend for entire series
        x = np.arange(len(ts_array))
        slope, intercept = np.polyfit(x, ts_array, 1)
        analysis["trend_analysis"] = {
            "overall_slope": slope,
            "overall_intercept": intercept,
            "trend_direction": "increasing" if slope > 0 else "decreasing"
        }

        # Volatility analysis (rolling standard deviation)
        rolling_std = []
        for i in range(len(ts_array) - window_size + 1):
            window = ts_array[i:i + window_size]
            rolling_std.append(np.std(window))

        analysis["volatility_analysis"] = {
            "mean_volatility": np.mean(rolling_std),
            "max_volatility": np.max(rolling_std),
            "volatility_trend": "increasing" if rolling_std[-1] > rolling_std[0] else "decreasing"
        }

        # Autocorrelation analysis
        if len(ts_array) > 10:
            autocorr = np.correlate(ts_array, ts_array, mode='full')
            autocorr = autocorr[autocorr.size // 2:]
            autocorr = autocorr / autocorr[0]  # Normalize

            # First few lags
            analysis["autocorrelation"] = {
                "lag_1": autocorr[1] if len(autocorr) > 1 else 0,
                "lag_5": autocorr[5] if len(autocorr) > 5 else 0,
                "lag_10": autocorr[10] if len(autocorr) > 10 else 0
            }

        return analysis

    def model_comparison_analysis(self, model_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Comprehensive analysis comparing multiple models"""
        if not model_results:
            return {"error": "No model results provided"}

        analysis = {
            "model_names": [r.get("model_name", f"model_{i}") for i, r in enumerate(model_results)],
            "performance_comparison": {},
            "statistical_tests": {},
            "ranking": {},
            "recommendations": []
        }

        # Extract common metrics
        all_metrics = set()
        for result in model_results:
            all_metrics.update(result.keys())

        # Compare each metric across models
        for metric in all_metrics:
            if all(isinstance(r.get(metric), (int, float)) for r in model_results):
                values = [r[metric] for r in model_results]

                # Determine if higher or lower is better
                better_higher = metric in ["accuracy", "precision", "recall", "f1_score", "information_gain"]
                better_lower = metric in ["free_energy", "prediction_error", "loss", "convergence_time", "model_complexity"]

                if better_higher:
                    best_idx = np.argmax(values)
                    best_value = np.max(values)
                    worst_value = np.min(values)
                elif better_lower:
                    best_idx = np.argmin(values)
                    best_value = np.min(values)
                    worst_value = np.max(values)
                else:
                    # Neutral metric
                    best_idx = 0
                    best_value = values[0]
                    worst_value = values[-1]

                analysis["performance_comparison"][metric] = {
                    "best_model": analysis["model_names"][best_idx],
                    "best_value": best_value,
                    "worst_value": worst_value,
                    "values": values,
                    "std_deviation": np.std(values),
                    "coefficient_of_variation": np.std(values) / np.mean(values) if np.mean(values) != 0 else 0
                }

        # Statistical comparison (simplified)
        if len(model_results) >= 2:
            # Compare free energy if available
            free_energy_values = []
            for result in model_results:
                if "free_energy" in result and result["free_energy"]:
                    free_energy_values.append(result["free_energy"][-1] if isinstance(result["free_energy"], list) else result["free_energy"])

            if len(free_energy_values) >= 2:
                # Perform t-test between first two models
                if len(free_energy_values) >= 2:
                    t_stat, p_value = stats.ttest_ind([free_energy_values[0]], [free_energy_values[1]])
                    analysis["statistical_tests"]["free_energy_t_test"] = {
                        "t_statistic": t_stat,
                        "p_value": p_value,
                        "significant": p_value < 0.05
                    }

        # Model ranking based on composite score
        composite_scores = []
        for i, result in enumerate(model_results):
            score = 0.0
            count = 0

            # Weight different metrics
            if "accuracy" in result:
                score += result["accuracy"] * 0.3
                count += 0.3
            if "free_energy" in result:
                # Lower is better for free energy
                fe_value = result["free_energy"][-1] if isinstance(result["free_energy"], list) else result["free_energy"]
                score += (1.0 - min(1.0, fe_value / 10.0)) * 0.3  # Normalize assuming max reasonable FE of 10
                count += 0.3
            if "convergence_time" in result:
                # Lower is better for convergence time
                conv_time = result["convergence_time"]
                score += (1.0 - min(1.0, conv_time / 1000.0)) * 0.2  # Normalize assuming max reasonable time of 1000
                count += 0.2

            if count > 0:
                composite_score = score / count
            else:
                composite_score = 0.0

            composite_scores.append(composite_score)

        # Rank models by composite score
        ranked_indices = np.argsort(composite_scores)[::-1]  # Sort descending
        analysis["ranking"] = {
            "composite_scores": composite_scores,
            "ranked_models": [
                {
                    "rank": i + 1,
                    "model": analysis["model_names"][idx],
                    "score": composite_scores[idx]
                }
                for i, idx in enumerate(ranked_indices)
            ]
        }

        # Generate recommendations
        if "ranking" in analysis and analysis["ranking"]["ranked_models"]:
            best_model = analysis["ranking"]["ranked_models"][0]
            analysis["recommendations"].append(f"Recommended model: {best_model['model']} (composite score: {best_model['score']:.3f})")

            # Check for significant differences
            if len(composite_scores) >= 2:
                score_diff = composite_scores[ranked_indices[0]] - composite_scores[ranked_indices[1]]
                if score_diff > 0.1:
                    analysis["recommendations"].append(f"Clear performance difference between top models ({score_diff:.3f} score gap)")

        return analysis


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

    def _compute_mutual_information(self, x: List[float], y: List[float]) -> float:
        """
        Compute mutual information between two variables

        Args:
            x: First variable data
            y: Second variable data

        Returns:
            Mutual information value
        """
        if len(x) != len(y) or len(x) == 0:
            return 0.0

        # Discretize data for mutual information calculation
        x_hist, x_edges = np.histogram(x, bins=min(20, len(set(x))))
        y_hist, y_edges = np.histogram(y, bins=min(20, len(set(y))))

        # Create 2D histogram
        hist_2d, _, _ = np.histogram2d(x, y, bins=[x_edges, y_edges])

        # Convert to probabilities
        hist_2d = hist_2d / np.sum(hist_2d)

        # Marginal probabilities
        px = np.sum(hist_2d, axis=1)
        py = np.sum(hist_2d, axis=0)

        # Compute mutual information
        mi = 0.0
        for i in range(len(px)):
            for j in range(len(py)):
                if hist_2d[i, j] > 0:
                    mi += hist_2d[i, j] * np.log2(hist_2d[i, j] / (px[i] * py[j]))

        return float(mi)

    def _compute_kl_divergence(self, p: List[float], q: List[float]) -> float:
        """
        Compute KL divergence between two distributions

        Args:
            p: First distribution
            q: Second distribution

        Returns:
            KL divergence value
        """
        if len(p) != len(q) or len(p) == 0:
            return 0.0

        # Ensure distributions sum to 1
        p = np.array(p) / np.sum(p)
        q = np.array(q) / np.sum(q)

        # Add small epsilon to avoid division by zero
        epsilon = 1e-10
        q = np.clip(q, epsilon, 1.0)

        # Compute KL divergence
        kl_div = np.sum(p * np.log2(p / q))

        return float(kl_div)

    def analyze_time_series(self, data: List[float], window_size: int = 10) -> Dict[str, Any]:
        """
        Analyze time series data for trends and patterns

        Args:
            data: Time series data
            window_size: Window size for rolling statistics

        Returns:
            Time series analysis results
        """
        if len(data) < window_size:
            return {"error": "Data too short for analysis"}

        analysis = {
            "trend_analysis": {},
            "seasonal_analysis": {},
            "autocorrelation": {},
            "stationarity_tests": {}
        }

        # Trend analysis
        analysis["trend_analysis"]["linear_trend"] = self._compute_linear_trend(data)
        analysis["trend_analysis"]["moving_average"] = self._compute_moving_average(data, window_size)

        # Stationarity tests (simplified)
        analysis["stationarity_tests"]["mean_stationarity"] = self._test_mean_stationarity(data, window_size)

        # Autocorrelation (simplified)
        analysis["autocorrelation"]["lag_1"] = self._compute_autocorrelation(data, lag=1)

        return analysis

    def _compute_linear_trend(self, data: List[float]) -> Dict[str, float]:
        """Compute linear trend in data"""
        if len(data) < 2:
            return {"slope": 0.0, "intercept": 0.0, "r_squared": 0.0}

        x = np.arange(len(data))
        y = np.array(data)

        # Linear regression
        slope, intercept = np.polyfit(x, y, 1)
        r_squared = np.corrcoef(x, y)[0, 1] ** 2

        return {
            "slope": float(slope),
            "intercept": float(intercept),
            "r_squared": float(r_squared)
        }

    def _compute_moving_average(self, data: List[float], window_size: int) -> List[float]:
        """Compute moving average"""
        if len(data) < window_size:
            return []

        moving_avg = []
        for i in range(len(data) - window_size + 1):
            window = data[i:i + window_size]
            moving_avg.append(float(np.mean(window)))

        return moving_avg

    def _test_mean_stationarity(self, data: List[float], window_size: int) -> Dict[str, Any]:
        """Test for mean stationarity"""
        if len(data) < window_size * 2:
            return {"stationary": False, "reason": "Insufficient data"}

        # Split data into windows and compare means
        windows = []
        for i in range(0, len(data) - window_size + 1, window_size // 2):
            window = data[i:i + window_size]
            if len(window) == window_size:
                windows.append(window)

        if len(windows) < 2:
            return {"stationary": False, "reason": "Insufficient windows"}

        window_means = [np.mean(w) for w in windows]

        # Test if means are similar (simplified stationarity test)
        mean_variation = np.std(window_means) / np.mean(window_means)
        stationary = mean_variation < 0.1  # Arbitrary threshold

        return {
            "stationary": stationary,
            "mean_variation": float(mean_variation),
            "window_count": len(windows)
        }

    def _compute_autocorrelation(self, data: List[float], lag: int = 1) -> float:
        """Compute autocorrelation at specified lag"""
        if len(data) <= lag:
            return 0.0

        # Remove mean
        data_demeaned = np.array(data) - np.mean(data)

        # Compute autocorrelation
        numerator = np.sum(data_demeaned[:-lag] * data_demeaned[lag:])
        denominator = np.sum(data_demeaned ** 2)

        if denominator == 0:
            return 0.0

        return float(numerator / denominator)

    def perform_meta_analysis(self, study_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Perform meta-analysis on multiple study results

        Args:
            study_results: List of study result dictionaries

        Returns:
            Meta-analysis results
        """
        if not study_results:
            return {"error": "No study results provided"}

        meta_analysis = {
            "effect_sizes": [],
            "heterogeneity": {},
            "publication_bias": {},
            "summary_statistics": {}
        }

        # Extract effect sizes (simplified - assuming standardized mean differences)
        effect_sizes = []
        sample_sizes = []

        for study in study_results:
            if "effect_size" in study and "sample_size" in study:
                effect_sizes.append(study["effect_size"])
                sample_sizes.append(study["sample_size"])

        if not effect_sizes:
            return {"error": "No effect sizes found in studies"}

        meta_analysis["effect_sizes"] = effect_sizes

        # Compute weighted average effect size
        weights = sample_sizes
        weighted_mean = np.average(effect_sizes, weights=weights)

        meta_analysis["summary_statistics"]["weighted_mean_effect"] = float(weighted_mean)
        meta_analysis["summary_statistics"]["total_sample_size"] = sum(sample_sizes)

        # Heterogeneity analysis (simplified Q statistic)
        if len(effect_sizes) > 1:
            q_stat = np.sum(weights * (np.array(effect_sizes) - weighted_mean) ** 2)
            meta_analysis["heterogeneity"]["q_statistic"] = float(q_stat)
            meta_analysis["heterogeneity"]["heterogeneous"] = q_stat > len(effect_sizes) - 1

        return meta_analysis

    def validate_analysis_correctness(self, analysis_results: Dict[str, Any],
                                    ground_truth: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate analysis results against ground truth

        Args:
            analysis_results: Results from analysis
            ground_truth: Known correct results

        Returns:
            Validation results
        """
        validation = {
            "overall_accuracy": 0.0,
            "metric_validation": {},
            "error_analysis": {},
            "recommendations": []
        }

        # Validate key metrics
        for metric_name, true_value in ground_truth.items():
            if metric_name in analysis_results:
                predicted_value = analysis_results[metric_name]

                if isinstance(true_value, (int, float)) and isinstance(predicted_value, (int, float)):
                    error = abs(predicted_value - true_value)
                    relative_error = error / abs(true_value) if true_value != 0 else 0

                    validation["metric_validation"][metric_name] = {
                        "true_value": true_value,
                        "predicted_value": predicted_value,
                        "absolute_error": error,
                        "relative_error": relative_error,
                        "acceptable": relative_error < 0.1  # 10% tolerance
                    }

        # Compute overall accuracy
        if validation["metric_validation"]:
            acceptable_count = sum(1 for v in validation["metric_validation"].values() if v["acceptable"])
            validation["overall_accuracy"] = acceptable_count / len(validation["metric_validation"])

        # Generate recommendations
        if validation["overall_accuracy"] < 0.8:
            validation["recommendations"].append("Analysis accuracy below threshold - review methodology")
        if any(v.get("relative_error", 0) > 0.5 for v in validation["metric_validation"].values()):
            validation["recommendations"].append("High error in some metrics - investigate data quality")

        return validation

    def generate_analysis_report(self, analysis_results: Dict[str, Any],
                               report_format: str = "markdown") -> str:
        """
        Generate comprehensive analysis report

        Args:
            analysis_results: Results from analysis
            report_format: Report format ('markdown', 'html', 'latex')

        Returns:
            Formatted analysis report
        """
        if report_format == "markdown":
            return self._generate_markdown_report(analysis_results)
        elif report_format == "html":
            return self._generate_html_report(analysis_results)
        elif report_format == "latex":
            return self._generate_latex_report(analysis_results)
        else:
            raise ValueError(f"Unsupported report format: {report_format}")

    def _generate_markdown_report(self, results: Dict[str, Any]) -> str:
        """Generate markdown analysis report"""
        report_lines = ["# Analysis Report\n"]

        # Executive summary
        report_lines.append("## Executive Summary\n")
        report_lines.append("Comprehensive statistical analysis of experimental data.\n")

        # Results sections
        for section_name, section_data in results.items():
            report_lines.append(f"## {section_name.replace('_', ' ').title()}\n")

            if isinstance(section_data, dict):
                for key, value in section_data.items():
                    if isinstance(value, (int, float)):
                        report_lines.append(f"- **{key}**: {value:.4f}\n")
                    else:
                        report_lines.append(f"- **{key}**: {value}\n")
            elif isinstance(section_data, list):
                for item in section_data:
                    report_lines.append(f"- {item}\n")
            else:
                report_lines.append(f"{section_data}\n")

        return "".join(report_lines)

    def _generate_html_report(self, results: Dict[str, Any]) -> str:
        """Generate HTML analysis report"""
        html_parts = [
            "<!DOCTYPE html>",
            "<html><head><title>Analysis Report</title></head><body>",
            "<h1>Analysis Report</h1>",
            "<h2>Executive Summary</h2>",
            "<p>Comprehensive statistical analysis of experimental data.</p>"
        ]

        for section_name, section_data in results.items():
            html_parts.append(f"<h2>{section_name.replace('_', ' ').title()}</h2>")

            if isinstance(section_data, dict):
                html_parts.append("<ul>")
                for key, value in section_data.items():
                    html_parts.append(f"<li><strong>{key}:</strong> {value}</li>")
                html_parts.append("</ul>")
            else:
                html_parts.append(f"<p>{section_data}</p>")

        html_parts.extend(["</body></html>"])
        return "".join(html_parts)

    def _generate_latex_report(self, results: Dict[str, Any]) -> str:
        """Generate LaTeX analysis report"""
        latex_parts = [
            "\\documentclass{article}",
            "\\begin{document}",
            "\\title{Analysis Report}",
            "\\maketitle",
            "\\section{Executive Summary}",
            "Comprehensive statistical analysis of experimental data."
        ]

        for section_name, section_data in results.items():
            latex_parts.append(f"\\section{{{section_name.replace('_', ' ').title()}}}")

            if isinstance(section_data, dict):
                for key, value in section_data.items():
                    latex_parts.append(f"\\textbf{{{key}}}: {value}\\\\")

        latex_parts.extend(["\\end{document}"])
        return "\n".join(latex_parts)


