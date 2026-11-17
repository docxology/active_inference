"""Behavioural tests for active_inference.visualization.comparative."""

from datetime import datetime

import pytest

from active_inference.visualization.comparative import (
    ComparisonTool,
    ComparisonType,
    ModelComparator,
    ModelComparison,
    StatisticalComparator,
)

MODEL_A = {
    "name": "A",
    "accuracy": 0.8,
    "free_energy": 2.0,
    "convergence_time": 50,
}
MODEL_B = {
    "name": "B",
    "accuracy": 0.9,
    "free_energy": 1.5,
    "convergence_time": 40,
}


def test_comparison_type_values():
    assert ComparisonType.MODEL_COMPARISON.value == "model_comparison"
    assert {t.value for t in ComparisonType} == {
        "model_comparison",
        "parameter_study",
        "algorithm_comparison",
        "performance_analysis",
    }


def test_model_comparison_defaults_recommendations_to_new_list():
    first = ModelComparison(model_a={}, model_b={}, metrics={})
    second = ModelComparison(model_a={}, model_b={}, metrics={})
    assert first.recommendations == []
    first.recommendations.append("x")
    assert second.recommendations == []


class TestComparisonTool:
    def test_compare_models_b_wins_accuracy_and_lower_free_energy(self):
        tool = ComparisonTool()
        result = tool.compare_models(MODEL_A, MODEL_B)

        assert result.metrics["accuracy_relative_diff"] == pytest.approx(12.5)
        assert result.metrics["accuracy_winner"] == "B"
        assert result.metrics["free_energy_relative_diff"] == pytest.approx(-25.0)
        assert result.metrics["free_energy_winner"] == "A"
        assert result.metrics["convergence_time_relative_diff"] == pytest.approx(-20)
        assert result.recommendations == [
            "Model B has higher accuracy",
            "Model B achieves significantly lower free energy",
        ]
        assert result.model_a is MODEL_A and result.model_b is MODEL_B
        assert tool.comparisons == {"comp_0": result}

    def test_compare_models_a_wins_accuracy_and_lower_free_energy(self):
        tool = ComparisonTool()
        a = {"accuracy": 0.9, "free_energy": 1.0}
        b = {"accuracy": 0.8, "free_energy": 1.5}
        result = tool.compare_models(a, b)

        assert result.metrics["accuracy_winner"] == "A"
        assert result.metrics["free_energy_relative_diff"] == pytest.approx(50.0)
        assert result.recommendations == [
            "Model A has higher accuracy",
            "Model A achieves significantly lower free energy",
        ]

    def test_small_free_energy_difference_adds_no_free_energy_recommendation(self):
        result = ComparisonTool().compare_models(
            {"accuracy": 0.9, "free_energy": 1.0},
            {"accuracy": 0.5, "free_energy": 1.05},
        )
        assert result.recommendations == ["Model A has higher accuracy"]

    def test_zero_baseline_non_numeric_and_missing_metrics(self):
        tool = ComparisonTool()
        result = tool.compare_models(
            {"accuracy": 0, "free_energy": "low", "loss": 1.0},
            {"accuracy": 0.5, "free_energy": "high"},
            comparison_metrics=["accuracy", "free_energy", "loss"],
        )
        assert result.metrics == {
            "accuracy_relative_diff": 0.0,
            "accuracy_winner": "B",
        }

    def test_comparison_ids_increment(self):
        tool = ComparisonTool()
        tool.compare_models(MODEL_A, MODEL_B)
        tool.compare_models(MODEL_B, MODEL_A)
        assert list(tool.comparisons) == ["comp_0", "comp_1"]

    def test_comparison_matrix_for_three_models(self):
        models = [
            {"accuracy": 0.9, "free_energy": 0.2, "convergence_time": 10},
            {"accuracy": 0.5, "free_energy": 0.9, "convergence_time": 90},
            {"name": "unscored"},
        ]
        tool = ComparisonTool()
        matrix = tool.create_comparison_matrix(models)

        assert matrix["models"] == ["Model_0", "Model_1", "unscored"]
        assert set(matrix["pairwise_comparisons"]) == {"0_1", "0_2", "1_2"}
        pair = matrix["pairwise_comparisons"]["0_2"]
        assert pair["models"] == ["Model_0", "unscored"]
        assert pair["metrics"] == {}
        assert matrix["pairwise_comparisons"]["0_1"]["metrics"]["accuracy_winner"] == (
            "A"
        )
        assert matrix["summary"] == {
            "total_models": 3,
            "total_comparisons": 3,
            "best_model": "Model_0",
        }
        assert len(tool.comparisons) == 3

    def test_determine_best_model_scores(self):
        tool = ComparisonTool()
        assert tool._determine_best_model([]) is None
        assert tool._determine_best_model([{"name": "x"}, {"other": 1}]) is None
        # Lower free energy and faster convergence win once normalised.
        models = [
            {"free_energy": 0.9, "convergence_time": 90},
            {"free_energy": 0.1, "convergence_time": 5},
        ]
        assert tool._determine_best_model(models) == "Model_1"

    def test_empty_matrix(self):
        matrix = ComparisonTool().create_comparison_matrix([])
        assert matrix["models"] == []
        assert matrix["pairwise_comparisons"] == {}
        assert matrix["summary"]["best_model"] is None


class TestModelComparator:
    def test_dashboard_structure_and_insights(self):
        models = [
            {"accuracy": 0.9, "free_energy": 0.2, "convergence_time": 10},
            {"accuracy": 0.5, "free_energy": 0.9, "convergence_time": 90},
        ]
        dashboard = ModelComparator().create_comparison_dashboard(models)

        assert dashboard["type"] == "model_comparison"
        assert dashboard["title"] == "Model Comparison - 2 Models"
        assert dashboard["models"] == ["Model_0", "Model_1"]
        assert list(dashboard["comparison_matrix"]) == ["0_1"]
        assert dashboard["summary"]["best_model"] == "Model_0"

        radar, bars, recs = dashboard["visualization_components"]
        assert radar["type"] == "radar_chart"
        assert radar["metrics"] == ["accuracy", "free_energy", "convergence_time"]
        assert radar["data"] is models
        assert bars["type"] == "bar_chart"
        assert bars["metrics"] == ["accuracy", "free_energy", "convergence_time"]
        assert recs["type"] == "recommendations"
        assert recs["content"] == [
            "Model 0 has the highest accuracy (0.900)",
            "Model 0 achieves the lowest free energy (0.200)",
            "Model 0 converges fastest (10.0 time units)",
        ]

    def test_insights_use_defaults_for_missing_metrics(self):
        insights = ModelComparator()._generate_insights(
            [{"accuracy": 0.4}, {"free_energy": 0.3, "convergence_time": 7}]
        )
        assert insights == [
            "Model 0 has the highest accuracy (0.400)",
            "Model 1 achieves the lowest free energy (0.300)",
            "Model 1 converges fastest (7.0 time units)",
        ]

    def test_single_and_empty_model_lists(self):
        comparator = ModelComparator()
        assert comparator._generate_insights([{"accuracy": 1.0}]) == [
            "Need at least 2 models for meaningful comparison"
        ]
        dashboard = comparator.create_comparison_dashboard([])
        assert dashboard["title"] == "Model Comparison - 0 Models"
        bars = dashboard["visualization_components"][1]
        assert bars["metrics"] == []
        assert dashboard["visualization_components"][2]["content"] == [
            "Need at least 2 models for meaningful comparison"
        ]


class TestStatisticalComparator:
    def test_requires_two_models(self):
        assert StatisticalComparator().perform_comprehensive_comparison([MODEL_A]) == {
            "error": "Need at least 2 models for comparison"
        }

    def test_pairwise_statistical_comparison(self):
        a = {"name": "A", "accuracy": 0.8, "free_energy": 2.0, "custom": 3, "tag": "x"}
        b = {"name": "B", "accuracy": 0.9, "free_energy": 1.5, "custom": 4, "tag": "y"}
        result = StatisticalComparator()._compare_models_statistically(a, b)

        assert result["models"] == ["A", "B"]
        assert set(result["metrics_comparison"]) == {"accuracy", "free_energy"}
        acc = result["metrics_comparison"]["accuracy"]
        assert acc["difference"] == pytest.approx(0.1)
        assert acc["relative_improvement"] == pytest.approx(12.5)
        fe = result["metrics_comparison"]["free_energy"]
        assert fe["difference"] == pytest.approx(0.5)
        assert fe["relative_improvement"] == pytest.approx(25.0)

    def test_pairwise_zero_baselines_and_default_names(self):
        result = StatisticalComparator()._compare_models_statistically(
            {"recall": 0, "loss": 0}, {"recall": 0.5, "loss": 0.2}
        )
        assert result["models"] == ["Model A", "Model B"]
        assert result["metrics_comparison"]["recall"]["relative_improvement"] == 0
        assert result["metrics_comparison"]["loss"]["relative_improvement"] == 0
        assert result["metrics_comparison"]["loss"]["difference"] == pytest.approx(-0.2)

    def test_t_test(self):
        comparator = StatisticalComparator()
        result = comparator._perform_t_test(
            [
                {"accuracy": 0.8, "free_energy": 1.0, "prediction_error": 0.3},
                {"accuracy": 0.9, "free_energy": 1.0},
            ]
        )
        assert result["test_type"] == "t_test"
        acc = result["results"]["accuracy"]
        assert acc["t_statistic"] == pytest.approx(0.85 / (0.05 / 2**0.5))
        assert acc["significant"]
        fe = result["results"]["free_energy"]
        assert fe["t_statistic"] == 0
        assert not fe["significant"]
        # Only one model reports prediction_error, so it is not tested.
        assert "prediction_error" not in result["results"]

    def test_placeholder_tests_report_their_type(self):
        comparator = StatisticalComparator()
        assert comparator._perform_wilcoxon_test([])["test_type"] == "wilcoxon"
        assert comparator._perform_ks_test([])["test_type"] == "ks_test"
        assert comparator._compute_correlation([])["test_type"] == "correlation"

    def test_rank_models(self):
        models = [
            {"name": "A", "accuracy": 0.8, "free_energy": 2.0},
            {"name": "B", "accuracy": 0.9, "free_energy": 1.5},
            {"label": "none"},
            {
                "accuracy": 1.5,
                "prediction_error": 0.5,
                "convergence_time": 500,
                "model_complexity": 50,
            },
        ]
        rankings = StatisticalComparator()._rank_models(models)
        overall = rankings["overall"]

        assert [entry["rank"] for entry in overall] == [1, 2, 3, 4]
        assert [entry["model"] for entry in overall] == ["B", "A", "Model_3", "Model_2"]
        assert overall[0]["score"] == pytest.approx((0.27 + 0.25 * 0.85) / 0.55)
        assert overall[1]["score"] == pytest.approx(0.8)
        # Accuracy is capped at 1.0; the rest are inverted against their bounds.
        assert overall[2]["score"] == pytest.approx(
            (0.3 * 1.0 + 0.2 * 0.5 + 0.15 * 0.5 + 0.1 * 0.5) / 0.75
        )
        assert overall[3]["score"] == 0.0
        assert rankings["by_metric"] == {}

    def test_comprehensive_comparison(self):
        models = [
            {"name": "A", "accuracy": 0.8, "free_energy": 2.0},
            {"name": "B", "accuracy": 0.9, "free_energy": 1.5},
        ]
        comparator = StatisticalComparator()

        def broken(_models):
            raise ValueError("boom")

        comparator.significance_tests["broken"] = broken
        result = comparator.perform_comprehensive_comparison(models)

        assert list(result["pairwise_comparisons"]) == ["0_1"]
        assert result["statistical_tests"]["broken"] == {"error": "boom"}
        assert result["statistical_tests"]["wilcoxon"]["test_type"] == "wilcoxon"
        assert result["performance_rankings"]["overall"][0]["model"] == "B"

        recs = result["recommendations"]
        assert recs[0] == "Recommended model: B (score: 0.877)"
        assert not any("broken" in r for r in recs)
        assert any(r.startswith("Statistical test (t_test):") for r in recs)
        assert "accuracy: better by 12.5%" in recs
        assert "free_energy: better by 25.0%" in recs

        summary = result["summary"]
        assert summary["total_models"] == 2
        assert summary["comparison_type"] == "comprehensive"
        assert summary["key_findings"][0] == "Best performing model: B"
        assert summary["key_findings"][1:] == recs[:3]
        assert isinstance(datetime.fromisoformat(summary["timestamp"]), datetime)

    def test_recommendations_report_worse_metrics(self):
        recs = StatisticalComparator()._generate_model_recommendations(
            [],
            {
                "pairwise_comparisons": {
                    "0_1": {
                        "metrics_comparison": {
                            "accuracy": {"relative_improvement": -44.44},
                            "loss": {"relative_improvement": 5.0},
                        }
                    }
                }
            },
        )
        assert recs == ["accuracy: worse by 44.4%"]

    def test_empty_results_produce_no_recommendations_or_findings(self):
        comparator = StatisticalComparator()
        assert comparator._generate_model_recommendations([], {}) == []
        summary = comparator._generate_comparison_summary(
            [], {"performance_rankings": {"overall": []}}
        )
        assert summary["total_models"] == 0
        assert summary["key_findings"] == []
