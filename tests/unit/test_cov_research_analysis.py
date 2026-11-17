"""Coverage tests for active_inference.research.analysis."""

import math

import numpy as np
import pytest

from active_inference.research.analysis import AnalysisTools, StatisticalAnalysis


@pytest.fixture
def stats_tool() -> StatisticalAnalysis:
    return StatisticalAnalysis()


@pytest.fixture
def tools() -> AnalysisTools:
    return AnalysisTools()


# ---------------------------------------------------------------------------
# StatisticalAnalysis
# ---------------------------------------------------------------------------


class TestDescriptiveAndCorrelation:
    def test_descriptive_stats_values(self, stats_tool):
        result = stats_tool.compute_descriptive_stats([1.0, 2.0, 3.0, 4.0])
        assert result == {
            "mean": 2.5,
            "median": 2.5,
            "std": pytest.approx(np.std([1, 2, 3, 4])),
            "min": 1.0,
            "max": 4.0,
            "n_samples": 4,
        }

    def test_descriptive_stats_empty(self, stats_tool):
        assert stats_tool.compute_descriptive_stats([]) == {}

    def test_correlation_perfect_negative(self, stats_tool):
        result = stats_tool.correlation_analysis([1, 2, 3], [6, 4, 2])
        assert result["pearson_r"] == pytest.approx(-1.0)
        assert result["pearson_r_squared"] == pytest.approx(1.0)
        assert result["n_samples"] == 3

    @pytest.mark.parametrize("x,y", [([1, 2], [1]), ([], [])])
    def test_correlation_invalid_lengths(self, stats_tool, x, y):
        assert stats_tool.correlation_analysis(x, y) == {}


class TestSignificance:
    def test_t_test_detects_difference(self, stats_tool):
        a = [1.0, 1.1, 0.9, 1.05, 0.95]
        b = [5.0, 5.1, 4.9, 5.05, 4.95]
        result = stats_tool.significance_test(a, b)
        assert result["test_type"] == "independent_t_test"
        assert result["t_statistic"] < 0
        assert result["p_value"] < 0.05
        assert result["significant"]
        assert (result["n1"], result["n2"]) == (5, 5)

    def test_t_test_not_significant_for_same_samples(self, stats_tool):
        a = [1.0, 2.0, 3.0, 4.0]
        result = stats_tool.significance_test(a, list(a))
        assert result["p_value"] == pytest.approx(1.0)
        assert not result["significant"]

    def test_empty_dataset_error(self, stats_tool):
        assert stats_tool.significance_test([], [1.0]) == {"error": "Empty datasets"}

    def test_unknown_test_type(self, stats_tool):
        result = stats_tool.significance_test([1.0], [2.0], test_type="anova")
        assert result == {"error": "Test type anova not implemented"}


class TestInformationTheory:
    def test_empty_data(self, stats_tool):
        assert stats_tool.information_theory_metrics([]) == {"error": "Empty data"}
        assert stats_tool.information_theory_metrics([[]]) == {"error": "Empty data"}

    def test_entropy_of_uniform_two_symbols(self, stats_tool):
        metrics = stats_tool.information_theory_metrics([[0.0, 1.0, 0.0, 1.0]])
        # Two equally probable bins -> 1 bit of entropy
        assert metrics["entropy_var_0"] == pytest.approx(1.0, abs=1e-6)
        assert not any(k.startswith("mutual_info") for k in metrics)

    def test_pairwise_metrics_identical_variables(self, stats_tool):
        x = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0]
        metrics = stats_tool.information_theory_metrics([x, list(x), [5.0]])
        # MI of a binary variable with itself equals its entropy (1 bit)
        assert metrics["mutual_info_0_1"] == pytest.approx(1.0, abs=1e-6)
        # Identical distributions -> zero KL divergence
        assert metrics["kl_divergence_0_1"] == pytest.approx(0.0, abs=1e-6)
        # Single-sample variable gets no entropy and no pairings
        assert "entropy_var_2" not in metrics
        assert "mutual_info_0_2" not in metrics

    def test_mutual_information_independent_is_zero(self, stats_tool):
        x = [0.0, 0.0, 1.0, 1.0]
        y = [0.0, 1.0, 0.0, 1.0]
        assert stats_tool._compute_mutual_information(x, y) == pytest.approx(0.0)

    def test_kl_divergence_positive_for_different(self, stats_tool):
        p = [0.0, 0.0, 0.0, 1.0]
        q = [0.0, 1.0, 1.0, 1.0]
        assert stats_tool._compute_kl_divergence(p, q) > 0


class TestBayesian:
    def test_default_prior_without_observations(self, stats_tool):
        result = stats_tool.bayesian_analysis({})
        assert result["prior"] == {"mean": 0.0, "variance": 1.0}
        assert result["posterior"] == {}
        assert "log_likelihood" not in result

    def test_conjugate_posterior(self, stats_tool):
        obs = [1.0, 3.0]  # mean 2, var 1
        result = stats_tool.bayesian_analysis(
            {"observations": obs}, prior={"mean": 0.0, "variance": 1.0}
        )
        post = result["posterior"]
        assert post["precision"] == pytest.approx(3.0)
        assert post["variance"] == pytest.approx(1.0 / 3.0)
        assert post["mean"] == pytest.approx(4.0 / 3.0)
        expected_ll = -0.5 * 2 * math.log(2 * math.pi) - 0.5 * 2
        assert result["log_likelihood"] == pytest.approx(expected_ll)


class TestTimeSeries:
    def test_insufficient_data(self, stats_tool):
        result = stats_tool.time_series_analysis([1.0, 2.0], window_size=5)
        assert result == {"error": "Insufficient data for time series analysis"}

    def test_increasing_series_with_autocorrelation(self, stats_tool):
        series = [float(i) + (i % 3) * 0.1 * i for i in range(20)]
        result = stats_tool.time_series_analysis(series, window_size=5)
        trend = result["trend_analysis"]
        assert trend["overall_slope"] > 0
        assert trend["trend_direction"] == "increasing"
        vol = result["volatility_analysis"]
        assert vol["max_volatility"] >= vol["mean_volatility"]
        assert vol["volatility_trend"] in {"increasing", "decreasing"}
        ac = result["autocorrelation"]
        assert set(ac) == {"lag_1", "lag_5", "lag_10"}
        assert 0 < ac["lag_10"] < ac["lag_5"] < ac["lag_1"] < 1

    def test_short_decreasing_series_skips_autocorrelation(self, stats_tool):
        series = [10.0, 8.0, 7.0, 3.0, 1.0]
        result = stats_tool.time_series_analysis(series, window_size=3)
        assert result["trend_analysis"]["trend_direction"] == "decreasing"
        assert result["autocorrelation"] == {}


class TestModelComparisonAnalysis:
    def test_empty(self, stats_tool):
        assert stats_tool.model_comparison_analysis([]) == {
            "error": "No model results provided"
        }

    def test_full_comparison(self, stats_tool):
        results = [
            {
                "model_name": "good",
                "accuracy": 0.9,
                "free_energy": [5.0, 1.0],
                "convergence_time": 100,
                "custom": 3,
            },
            {
                "model_name": "bad",
                "accuracy": 0.4,
                "free_energy": [9.0, 8.0],
                "convergence_time": 900,
                "custom": 7,
            },
        ]
        analysis = stats_tool.model_comparison_analysis(results)
        assert analysis["model_names"] == ["good", "bad"]

        perf = analysis["performance_comparison"]
        assert perf["accuracy"]["best_model"] == "good"
        assert perf["accuracy"]["best_value"] == 0.9
        assert perf["accuracy"]["worst_value"] == 0.4
        assert perf["convergence_time"]["best_model"] == "good"
        assert perf["convergence_time"]["best_value"] == 100
        # Neutral metric: first value is "best", last is "worst"
        assert perf["custom"]["best_model"] == "good"
        assert perf["custom"]["worst_value"] == 7
        assert perf["custom"]["coefficient_of_variation"] == pytest.approx(2.0 / 5.0)
        # Non-scalar metrics are not compared
        assert "free_energy" not in perf
        assert "model_name" not in perf

        assert "free_energy_t_test" in analysis["statistical_tests"]

        ranking = analysis["ranking"]
        assert ranking["ranked_models"][0]["model"] == "good"
        assert ranking["ranked_models"][0]["rank"] == 1
        good_score = (0.9 * 0.3 + 0.9 * 0.3 + 0.9 * 0.2) / 0.8
        assert ranking["composite_scores"][0] == pytest.approx(good_score)

        recs = analysis["recommendations"]
        assert recs[0].startswith("Recommended model: good")
        assert any("Clear performance difference" in r for r in recs)

    def test_scalar_free_energy_lower_is_better(self, stats_tool):
        results = [
            {"model_name": "a", "free_energy": 2.0},
            {"model_name": "b", "free_energy": 1.5},
        ]
        analysis = stats_tool.model_comparison_analysis(results)
        assert analysis["performance_comparison"]["free_energy"]["best_model"] == "b"
        assert analysis["ranking"]["ranked_models"][0]["model"] == "b"
        # Scores 0.85 vs 0.80 -> gap below 0.1 is not a "clear" difference
        assert analysis["ranking"]["composite_scores"] == pytest.approx([0.8, 0.85])
        assert len(analysis["recommendations"]) == 1

    def test_models_without_scored_metrics(self, stats_tool):
        analysis = stats_tool.model_comparison_analysis([{"other": "x"}])
        assert analysis["model_names"] == ["model_0"]
        assert analysis["ranking"]["composite_scores"] == [0.0]
        assert analysis["statistical_tests"] == {}
        assert analysis["recommendations"] == [
            "Recommended model: model_0 (composite score: 0.000)"
        ]


# ---------------------------------------------------------------------------
# AnalysisTools
# ---------------------------------------------------------------------------


class TestAnalyzeExperimentResults:
    def test_improving_experiment_has_no_recommendations(self, tools):
        analysis = tools.analyze_experiment_results(
            {
                "experiment_id": "e1",
                "free_energy": [3.0, 2.0, 1.0],
                "accuracy": [0.2, 0.5, 0.8],
            }
        )
        pm = analysis["performance_metrics"]
        assert analysis["experiment_id"] == "e1"
        assert pm["final_free_energy"] == 1.0
        assert pm["free_energy_trend"] == "decreasing"
        assert pm["free_energy_stats"]["mean"] == 2.0
        assert pm["final_accuracy"] == 0.8
        assert pm["accuracy_trend"] == "increasing"
        assert analysis["recommendations"] == []

    def test_degrading_experiment_gets_recommendations(self, tools):
        analysis = tools.analyze_experiment_results(
            {"free_energy": [1.0, 2.0], "accuracy": [0.9, 0.1]}
        )
        assert analysis["experiment_id"] is None
        assert analysis["recommendations"] == [
            "Consider adjusting model parameters to reduce free energy",
            "Model accuracy is declining, investigate learning dynamics",
        ]

    def test_empty_trajectories(self, tools):
        analysis = tools.analyze_experiment_results({"free_energy": [], "accuracy": []})
        pm = analysis["performance_metrics"]
        assert pm["final_free_energy"] is None
        assert pm["final_accuracy"] is None
        assert pm["free_energy_stats"] == {}

    def test_no_metrics(self, tools):
        analysis = tools.analyze_experiment_results({})
        assert analysis["performance_metrics"] == {}
        assert analysis["recommendations"] == []


class TestCompareModels:
    def test_empty(self, tools):
        assert tools.compare_models([]) == {}

    def test_compares_final_values(self, tools):
        results = [
            {"model_name": "m1", "free_energy": [3.0, 1.0], "accuracy": [0.5, 0.9]},
            {"free_energy": [3.0, 2.0], "accuracy": [0.5, 0.6]},
            {"free_energy": [3.0, 2.2], "accuracy": [0.5, 0.7]},
        ]
        comparison = tools.compare_models(results)
        assert comparison["n_models"] == 3
        assert comparison["model_names"] == ["m1", "model_1", "model_2"]
        fe = comparison["performance_comparison"]["free_energy_comparison"]
        acc = comparison["performance_comparison"]["accuracy_comparison"]
        assert fe["n1"] == 1 and fe["n2"] == 2
        assert acc["test_type"] == "independent_t_test"

    def test_single_model_has_no_comparison(self, tools):
        comparison = tools.compare_models([{"free_energy": [1.0], "accuracy": [0.5]}])
        assert comparison["performance_comparison"] == {}


class TestToolsInformationMeasures:
    def test_mutual_information_identical_binary(self, tools):
        x = [0.0, 1.0, 0.0, 1.0]
        assert tools._compute_mutual_information(x, list(x)) == pytest.approx(1.0)

    def test_mutual_information_independent(self, tools):
        x = [0.0, 0.0, 1.0, 1.0]
        y = [0.0, 1.0, 0.0, 1.0]
        assert tools._compute_mutual_information(x, y) == pytest.approx(0.0)

    @pytest.mark.parametrize("x,y", [([1.0], [1.0, 2.0]), ([], [])])
    def test_mutual_information_invalid(self, tools, x, y):
        assert tools._compute_mutual_information(x, y) == 0.0

    def test_kl_divergence_identical(self, tools):
        assert tools._compute_kl_divergence([1, 2, 3], [2, 4, 6]) == pytest.approx(0.0)

    def test_kl_divergence_known_value(self, tools):
        # KL([0.5, 0.5] || [0.25, 0.75]) in bits
        expected = 0.5 * math.log2(0.5 / 0.25) + 0.5 * math.log2(0.5 / 0.75)
        assert tools._compute_kl_divergence([0.5, 0.5], [0.25, 0.75]) == (
            pytest.approx(expected)
        )

    def test_kl_divergence_zero_probability_terms(self, tools):
        # 0 * log(0 / q) contributes 0, so KL([1, 0] || [0.5, 0.5]) = 1 bit
        assert tools._compute_kl_divergence([1, 0], [0.5, 0.5]) == pytest.approx(1.0)

    @pytest.mark.parametrize("p,q", [([0.5], [0.5, 0.5]), ([], [])])
    def test_kl_divergence_invalid(self, tools, p, q):
        assert tools._compute_kl_divergence(p, q) == 0.0


class TestAnalyzeTimeSeries:
    def test_too_short(self, tools):
        assert tools.analyze_time_series([1.0, 2.0], window_size=5) == {
            "error": "Data too short for analysis"
        }

    def test_full_analysis_of_constant_level(self, tools):
        data = [5.0, 5.2, 4.8, 5.1, 4.9, 5.0, 5.2, 4.8, 5.1, 4.9]
        analysis = tools.analyze_time_series(data, window_size=4)
        assert len(analysis["trend_analysis"]["moving_average"]) == 7
        assert analysis["trend_analysis"]["moving_average"][0] == pytest.approx(5.025)
        stationarity = analysis["stationarity_tests"]["mean_stationarity"]
        assert stationarity["stationary"]
        assert stationarity["window_count"] == 4
        assert analysis["autocorrelation"]["lag_1"] < 0

    def test_linear_trend(self, tools):
        trend = tools._compute_linear_trend([1.0, 3.0, 5.0, 7.0])
        assert trend["slope"] == pytest.approx(2.0)
        assert trend["intercept"] == pytest.approx(1.0)
        assert trend["r_squared"] == pytest.approx(1.0)

    def test_linear_trend_too_short(self, tools):
        assert tools._compute_linear_trend([1.0]) == {
            "slope": 0.0,
            "intercept": 0.0,
            "r_squared": 0.0,
        }

    def test_moving_average(self, tools):
        assert tools._compute_moving_average([1.0, 2.0, 3.0, 4.0], 2) == [
            1.5,
            2.5,
            3.5,
        ]
        assert tools._compute_moving_average([1.0], 2) == []

    def test_mean_stationarity_insufficient_data(self, tools):
        assert tools._test_mean_stationarity([1.0, 2.0, 3.0], 2) == {
            "stationary": False,
            "reason": "Insufficient data",
        }

    def test_mean_stationarity_non_stationary(self, tools):
        data = [1.0, 1.0, 1.0, 1.0, 10.0, 10.0, 10.0, 10.0]
        result = tools._test_mean_stationarity(data, 4)
        assert not result["stationary"]
        assert result["mean_variation"] > 0.1
        assert result["window_count"] == 3

    def test_mean_stationarity_window_size_one(self, tools):
        result = tools._test_mean_stationarity([2.0, 2.0, 2.0], 1)
        assert result["stationary"]
        assert result["window_count"] == 3
        assert result["mean_variation"] == 0.0

    def test_mean_stationarity_insufficient_windows(self, tools):
        # A negative window size can never produce a full window
        result = tools._test_mean_stationarity([1.0, 2.0], -1)
        assert result == {"stationary": False, "reason": "Insufficient windows"}

    def test_autocorrelation(self, tools):
        assert tools._compute_autocorrelation([1.0, 2.0], lag=2) == 0.0
        assert tools._compute_autocorrelation([3.0, 3.0, 3.0], lag=1) == 0.0
        value = tools._compute_autocorrelation([1.0, -1.0, 1.0, -1.0], lag=1)
        assert value == pytest.approx(-0.75)


class TestMetaAnalysis:
    def test_empty(self, tools):
        assert tools.perform_meta_analysis([]) == {"error": "No study results provided"}

    def test_no_effect_sizes(self, tools):
        assert tools.perform_meta_analysis([{"sample_size": 10}]) == {
            "error": "No effect sizes found in studies"
        }

    def test_weighted_mean_and_heterogeneity(self, tools):
        studies = [
            {"effect_size": 0.2, "sample_size": 10},
            {"effect_size": 0.8, "sample_size": 30},
            {"effect_size": 1.0},  # ignored: no sample size
        ]
        result = tools.perform_meta_analysis(studies)
        assert result["effect_sizes"] == [0.2, 0.8]
        summary = result["summary_statistics"]
        assert summary["weighted_mean_effect"] == pytest.approx(0.65)
        assert summary["total_sample_size"] == 40
        q = 10 * (0.2 - 0.65) ** 2 + 30 * (0.8 - 0.65) ** 2
        assert result["heterogeneity"]["q_statistic"] == pytest.approx(q)
        assert result["heterogeneity"]["heterogeneous"]

    def test_single_study_has_no_heterogeneity(self, tools):
        result = tools.perform_meta_analysis([{"effect_size": 0.5, "sample_size": 5}])
        assert result["heterogeneity"] == {}
        assert result["summary_statistics"]["weighted_mean_effect"] == 0.5


class TestValidateAnalysisCorrectness:
    def test_mixed_accuracy(self, tools):
        validation = tools.validate_analysis_correctness(
            {"a": 1.05, "b": 2.0, "c": 0.5, "label": "x", "missing_in_truth": 1},
            {"a": 1.0, "b": 1.0, "c": 0, "label": "x", "absent": 3},
        )
        mv = validation["metric_validation"]
        assert set(mv) == {"a", "b", "c"}
        assert mv["a"]["acceptable"]
        assert mv["a"]["absolute_error"] == pytest.approx(0.05)
        assert not mv["b"]["acceptable"]
        assert mv["b"]["relative_error"] == pytest.approx(1.0)
        # Zero ground truth -> relative error defined as 0
        assert mv["c"]["relative_error"] == 0
        assert validation["overall_accuracy"] == pytest.approx(2 / 3)
        assert validation["recommendations"] == [
            "Analysis accuracy below threshold - review methodology",
            "High error in some metrics - investigate data quality",
        ]

    def test_all_accurate(self, tools):
        validation = tools.validate_analysis_correctness({"a": 1.0}, {"a": 1.0})
        assert validation["overall_accuracy"] == 1.0
        assert validation["recommendations"] == []

    def test_no_comparable_metrics(self, tools):
        validation = tools.validate_analysis_correctness({}, {"a": 1.0})
        assert validation["overall_accuracy"] == 0.0
        assert validation["recommendations"] == [
            "Analysis accuracy below threshold - review methodology"
        ]


class TestReports:
    RESULTS = {
        "summary_stats": {"mean": 1.23456, "label": "ok"},
        "notes_list": ["first", "second"],
        "plain_value": "text",
    }

    def test_markdown(self, tools):
        report = tools.generate_analysis_report(self.RESULTS)
        assert report.startswith("# Analysis Report\n")
        assert "## Summary Stats\n" in report
        assert "- **mean**: 1.2346\n" in report
        assert "- **label**: ok\n" in report
        assert "- first\n- second\n" in report
        assert "## Plain Value\ntext\n" in report

    def test_html(self, tools):
        report = tools.generate_analysis_report(self.RESULTS, report_format="html")
        assert report.startswith("<!DOCTYPE html>")
        assert "<h2>Summary Stats</h2><ul>" in report
        assert "<li><strong>mean:</strong> 1.23456</li>" in report
        assert "<p>['first', 'second']</p>" in report
        assert report.endswith("</body></html>")

    def test_latex(self, tools):
        report = tools.generate_analysis_report(self.RESULTS, report_format="latex")
        lines = report.split("\n")
        assert lines[0] == "\\documentclass{article}"
        assert "\\section{Summary Stats}" in lines
        assert "\\textbf{mean}: 1.23456\\\\" in lines
        assert "\\section{Notes List}" in lines
        assert lines[-1] == "\\end{document}"

    def test_unsupported_format(self, tools):
        with pytest.raises(ValueError, match="Unsupported report format: pdf"):
            tools.generate_analysis_report({}, report_format="pdf")
