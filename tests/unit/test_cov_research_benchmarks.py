"""Coverage tests for active_inference.research.benchmarks."""

from types import SimpleNamespace

import pytest

from active_inference.research import benchmarks as benchmarks_module
from active_inference.research.benchmarks import (
    BenchmarkSuite,
    MetricType,
    PerformanceMetrics,
)

FIELDS = [
    "accuracy",
    "precision",
    "recall",
    "f1_score",
    "free_energy",
    "convergence_time",
    "computational_cost",
    "information_efficiency",
]


class _FakeClock:
    """Deterministic replacement for time.time."""

    def __init__(self, *values: float) -> None:
        self._values = list(values)

    def __call__(self) -> float:
        return self._values.pop(0)


@pytest.fixture
def suite() -> BenchmarkSuite:
    return BenchmarkSuite()


def test_metric_type_values():
    assert [m.value for m in MetricType] == FIELDS


def test_performance_metrics_to_dict():
    metrics = PerformanceMetrics(accuracy=0.9, free_energy=1.5)
    as_dict = metrics.to_dict()
    assert list(as_dict) == FIELDS
    assert as_dict["accuracy"] == 0.9
    assert as_dict["free_energy"] == 1.5
    assert as_dict["recall"] == 0.0


class TestEvaluateModel:
    def test_extracts_metrics_and_timing(self, suite, monkeypatch):
        monkeypatch.setattr(
            benchmarks_module, "time", SimpleNamespace(time=_FakeClock(10.0, 12.5))
        )
        received = {}

        def evaluate(data):
            received.update(data)
            return {"accuracy": 0.8, "precision": 0.7, "information_efficiency": 0.3}

        metrics = suite.evaluate_model("m", evaluate, {"x": [1, 2]})

        assert received == {"x": [1, 2]}
        assert metrics.accuracy == 0.8
        assert metrics.precision == 0.7
        assert metrics.recall == 0.0
        assert metrics.information_efficiency == 0.3
        # convergence_time falls back to evaluation time when not reported
        assert metrics.convergence_time == 2.5
        assert metrics.computational_cost == 2.5
        assert suite.metrics["m"] is metrics

    def test_reported_convergence_time_wins(self, suite, monkeypatch):
        monkeypatch.setattr(
            benchmarks_module, "time", SimpleNamespace(time=_FakeClock(0.0, 1.0))
        )
        metrics = suite.evaluate_model("m", lambda _d: {"convergence_time": 42.0})
        assert metrics.convergence_time == 42.0
        assert metrics.computational_cost == 1.0

    def test_default_test_data_and_non_dict_result(self, suite, monkeypatch):
        monkeypatch.setattr(
            benchmarks_module, "time", SimpleNamespace(time=_FakeClock(0.0, 3.0))
        )
        seen = []

        def evaluate(data):
            seen.append(data)
            return [0.9]

        metrics = suite.evaluate_model("m", evaluate)
        assert seen == [{}]
        assert metrics.to_dict() == PerformanceMetrics().to_dict()
        assert suite.metrics["m"] is metrics


class TestCompareModels:
    def test_no_metrics_returns_skeleton(self, suite):
        assert suite.compare_models(["a"]) == {
            "models": ["a"],
            "rankings": {},
            "improvements": {},
            "summary": {},
        }

    def test_rankings_improvements_and_summary(self, suite):
        suite.metrics["a"] = PerformanceMetrics(accuracy=0.9, precision=0.8)
        suite.metrics["b"] = PerformanceMetrics(accuracy=0.6, precision=0.95)
        suite.register_baseline("a", PerformanceMetrics(accuracy=0.6))
        assert suite.baselines["a"].accuracy == 0.6

        comparison = suite.compare_models(["a", "b", "unknown"])

        assert comparison["rankings"]["accuracy"] == [("a", 0.9), ("b", 0.6)]
        assert comparison["rankings"]["precision"] == [("b", 0.95), ("a", 0.8)]
        assert set(comparison["rankings"]) == set(FIELDS)

        improvements = comparison["improvements"]
        assert list(improvements) == ["a"]
        assert improvements["a"]["accuracy"] == pytest.approx(50.0)
        # zero baseline -> improvement reported as 0
        assert improvements["a"]["precision"] == 0.0

        assert comparison["summary"] == {
            "total_models": 3,
            "evaluated_models": 2,
            "best_overall": "a",  # mean(0.9, 0.8)=0.85 > mean(0.6, 0.95)=0.775
        }

    def test_unknown_models_only(self, suite):
        suite.metrics["a"] = PerformanceMetrics(accuracy=0.5)
        comparison = suite.compare_models(["zzz"])
        assert comparison["rankings"] == {}
        assert comparison["summary"]["evaluated_models"] == 0
        assert comparison["summary"]["best_overall"] == "a"


class TestDetermineBestModel:
    def test_no_metrics(self, suite):
        assert suite._determine_best_model() is None

    def test_all_zero_scores(self, suite):
        suite.metrics["a"] = PerformanceMetrics(free_energy=3.0)
        assert suite._determine_best_model() is None

    def test_only_positive_metrics_are_averaged(self, suite):
        suite.metrics["sparse"] = PerformanceMetrics(recall=0.7)
        suite.metrics["dense"] = PerformanceMetrics(
            accuracy=0.9, precision=0.5, recall=0.5, f1_score=0.5
        )
        # sparse averages over 1 metric (0.7) vs dense 0.6
        assert suite._determine_best_model() == "sparse"


def test_generate_report(suite):
    suite.register_baseline("a", PerformanceMetrics(accuracy=0.5))
    suite.metrics["a"] = PerformanceMetrics(accuracy=0.75)
    report = suite.generate_report()

    assert report["timestamp"] == "2024-10-27T12:00:00"
    assert report["total_models"] == 1
    assert report["baselines"]["a"]["accuracy"] == 0.5
    assert report["results"]["a"]["accuracy"] == 0.75
    assert report["comparison"]["models"] == ["a"]
    assert report["comparison"]["improvements"]["a"]["accuracy"] == pytest.approx(50)


def test_generate_report_empty(suite):
    report = suite.generate_report(output_format="json")
    assert report["total_models"] == 0
    assert report["baselines"] == {}
    assert report["results"] == {}
    assert report["comparison"]["summary"] == {}
