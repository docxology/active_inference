"""Behavioural tests for the library testing framework in active_inference.tools.

The module is imported under an alias so pytest does not try to collect its
``Test*`` classes as test cases.
"""

import json
import math
import textwrap
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterator, List

import numpy as np
import pytest

from active_inference.tools import testing as tf


def _fake_clock(step: float = 0.5) -> Any:
    """Deterministic replacement for the module's ``time`` dependency."""
    ticks: Iterator[float] = (i * step for i in range(10_000))
    return SimpleNamespace(time=lambda: next(ticks))


# --------------------------------------------------------------------- runner


class TestRunnerBehaviour:
    def test_run_test_passed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(tf, "time", _fake_clock(0.25))
        runner = tf.TestRunner({"verbose": True})
        result = runner.run_test(lambda: None, "ok")
        assert result.status == "passed"
        assert result.error_message is None
        assert result.execution_time == pytest.approx(0.25)
        assert runner.test_results == [result]
        assert runner.config == {"verbose": True}

    def test_run_test_assertion_failure(self) -> None:
        def failing() -> None:
            raise AssertionError("values differ")

        result = tf.TestRunner({}).run_test(failing, "fail")
        assert result.status == "failed"
        assert result.error_message == "values differ"
        assert result.traceback is None

    def test_run_test_error(self) -> None:
        def broken() -> None:
            raise KeyError("missing")

        result = tf.TestRunner({}).run_test(broken, "err")
        assert result.status == "error"
        assert result.error_message == "'missing'"
        assert result.traceback is not None
        assert "KeyError" in result.traceback

    def test_run_test_class_with_setup_and_teardown(self) -> None:
        calls: List[str] = []

        class Sample:
            def setUp(self) -> None:
                calls.append("setUp")

            def tearDown(self) -> None:
                calls.append("tearDown")

            def test_a(self) -> None:
                calls.append("a")

            def test_b(self) -> None:
                assert False, "b broke"

            def helper(self) -> None:  # not a test
                calls.append("helper")

            test_not_callable = 3

        runner = tf.TestRunner({})
        results = runner.run_test_class(Sample)
        assert [r.test_name for r in results] == ["Sample.test_a", "Sample.test_b"]
        assert [r.status for r in results] == ["passed", "failed"]
        assert calls == ["setUp", "a", "tearDown"]
        assert len(runner.test_results) == 2

    def test_run_test_class_without_hooks(self) -> None:
        class Bare:
            def test_only(self) -> None:
                pass

        results = tf.TestRunner({}).run_test_class(Bare)
        assert [(r.test_name, r.status) for r in results] == [
            ("Bare.test_only", "passed")
        ]

    def test_run_test_suite(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        module_name = "cov_tools_sample_suite_mod"
        (tmp_path / f"{module_name}.py").write_text(textwrap.dedent("""
                import unittest
                from unittest import TestCase


                class SampleCase(unittest.TestCase):
                    def test_pass(self):
                        self.assertEqual(1, 1)

                    def test_fail(self):
                        self.assertEqual(1, 2)

                    def test_error(self):
                        raise RuntimeError("boom")


                class NotATestCase:
                    def test_ignored(self):
                        raise AssertionError("should not run")
                """))
        monkeypatch.syspath_prepend(str(tmp_path))

        summary = tf.TestRunner({}).run_test_suite(
            [module_name, "cov_tools_no_such_module_xyz"]
        )

        assert summary["total_tests"] == 4
        assert summary["passed"] == 1
        assert summary["failed"] == 1
        assert summary["errors"] == 2
        assert summary["success_rate"] == pytest.approx(0.25)
        names = {r.test_name: r for r in summary["results"]}
        assert names["SampleCase.test_pass"].status == "passed"
        assert names["SampleCase.test_fail"].status == "failed"
        assert names["SampleCase.test_error"].error_message == "boom"
        module_error = names["module_cov_tools_no_such_module_xyz"]
        assert module_error.status == "error"
        assert "cov_tools_no_such_module_xyz" in (module_error.error_message or "")
        assert summary["total_time"] >= 0

    def test_run_test_suite_empty(self) -> None:
        summary = tf.TestRunner({}).run_test_suite([])
        assert summary["total_tests"] == 0
        assert summary["success_rate"] == 0
        assert summary["results"] == []

    def test_generate_report_without_results(self, tmp_path: Path) -> None:
        out = tmp_path / "report.json"
        assert tf.TestRunner({}).generate_report(out) is False
        assert not out.exists()

    def test_generate_report_contents(self, tmp_path: Path) -> None:
        runner = tf.TestRunner({})
        runner.run_test(lambda: None, "t1")
        runner.run_test(lambda: None, "t2")

        def bad() -> None:
            raise ValueError("nope")

        runner.run_test(bad, "t3")
        out = tmp_path / "report.json"
        assert runner.generate_report(out) is True
        report = json.loads(out.read_text())
        assert report["total_tests"] == 3
        assert report["results_by_status"] == {"passed": 2, "error": 1}
        assert [r["test_name"] for r in report["results"]] == ["t1", "t2", "t3"]
        assert report["results"][2]["error_message"] == "nope"

    def test_generate_report_write_failure(self, tmp_path: Path) -> None:
        runner = tf.TestRunner({})
        runner.run_test(lambda: None, "t1")
        # A directory cannot be opened as a file.
        assert runner.generate_report(tmp_path) is False


# ---------------------------------------------------------- quality assurance


@pytest.fixture
def qa() -> Any:
    return tf.QualityAssurance({"strict": True})


class TestQualityAssurance:
    def test_rules_registered(self, qa: Any) -> None:
        assert set(qa.validation_rules) == {
            "probability_distribution",
            "numerical_stability",
            "parameter_ranges",
            "data_integrity",
        }
        assert qa.config == {"strict": True}

    def test_probability_distribution_valid(self, qa: Any) -> None:
        result = qa._validate_probability_distribution([0.2, 0.3, 0.5])
        assert result["valid"] is True
        assert result["issues"] == []
        assert result["sum"] == pytest.approx(1.0)
        assert result["min_value"] == 0.2
        assert result["max_value"] == 0.5

    def test_probability_distribution_empty(self, qa: Any) -> None:
        assert qa._validate_probability_distribution([]) == {
            "valid": False,
            "issues": ["Empty distribution"],
        }

    def test_probability_distribution_invalid(self, qa: Any) -> None:
        result = qa._validate_probability_distribution([-0.5, 0.7])
        assert result["valid"] is False
        assert result["issues"][0] == "Negative values at indices: [0]"
        assert "does not sum to 1.0" in result["issues"][1]

    def test_numerical_stability_clean(self, qa: Any) -> None:
        result = qa._validate_numerical_stability(
            {"a": 1.0, "b": [1.0, 2.0], "c": np.array([0.5]), "label": "x"}
        )
        assert result == {"valid": True, "issues": [], "numerically_stable": True}

    def test_numerical_stability_problems(self, qa: Any) -> None:
        result = qa._validate_numerical_stability(
            {
                "nan_scalar": float("nan"),
                "nan_list": [1.0, float("nan")],
                "inf_scalar": float("inf"),
                "inf_array": np.array([1.0, -np.inf]),
                "big_scalar": 1e12,
                "big_list": [1e11, 0.0],
            }
        )
        issues = result["issues"]
        assert result["valid"] is False
        assert result["numerically_stable"] is False
        assert "NaN value in nan_scalar" in issues
        assert "NaN values in nan_list" in issues
        assert "Infinite value in inf_scalar" in issues
        assert "Infinite values in inf_array" in issues
        assert "Extremely large value in big_scalar: 1000000000000.0" in issues
        assert "Extremely large values in big_list" in issues

    def test_parameter_ranges(self, qa: Any) -> None:
        result = qa._validate_parameter_ranges(
            {
                "learning_rate": 0.1,
                "decay_rate": 1.5,
                "model_learning_rate": 0.0,
                "policy_temperature": -1.0,
                "sensory_precision": 0.0,
                "action_inverse_temperature": -2.0,
                "unrelated": -100,
            }
        )
        issues = result["issues"]
        assert result["valid"] is False
        assert result["parameters_checked"] == 7
        assert "Learning rate decay_rate should be in (0, 1], got 1.5" in issues
        assert "Learning rate model_learning_rate should be in (0, 1], got 0.0" in (
            issues
        )
        assert "Temperature policy_temperature should be positive, got -1.0" in issues
        assert "Precision sensory_precision should be positive, got 0.0" in issues
        # *_inverse_temperature is checked both as temperature and precision.
        assert (
            "Temperature action_inverse_temperature should be positive, got -2.0"
            in issues
        )
        assert (
            "Precision action_inverse_temperature should be positive, got -2.0"
            in issues
        )
        assert len(issues) == 6

    def test_parameter_ranges_valid(self, qa: Any) -> None:
        result = qa._validate_parameter_ranges(
            {"learning_rate": 1.0, "x_temperature": 0.5, "y_precision": 2.0}
        )
        assert result == {"valid": True, "issues": [], "parameters_checked": 3}

    def test_data_integrity_missing_fields(self, qa: Any) -> None:
        result = qa._validate_data_integrity({})
        assert result["issues"] == [
            "Missing required field: timestamp",
            "Missing required field: data",
        ]
        assert result["data_integrity_ok"] is False

    @pytest.mark.parametrize(
        "payload, expected_issue",
        [
            ({1: "a", "b": 2}, "Inconsistent key types in data"),
            ([1, "two"], "Inconsistent item types in data list"),
        ],
    )
    def test_data_integrity_inconsistent(
        self, qa: Any, payload: Any, expected_issue: str
    ) -> None:
        result = qa._validate_data_integrity({"timestamp": "t", "data": payload})
        assert result["valid"] is False
        assert len(result["issues"]) == 1
        assert result["issues"][0].startswith(expected_issue)

    @pytest.mark.parametrize("payload", [{"a": 1, "b": 2}, [1, 2, 3], [], "raw"])
    def test_data_integrity_consistent(self, qa: Any, payload: Any) -> None:
        result = qa._validate_data_integrity({"timestamp": "t", "data": payload})
        assert result == {"valid": True, "issues": [], "data_integrity_ok": True}

    def test_validate_component_all_rules(self, qa: Any) -> None:
        result = qa.validate_component("comp", {"timestamp": "t", "data": [1, 2]})
        assert result["component"] == "comp"
        assert result["validation_types"] == list(qa.validation_rules)
        assert set(result["validation_results"]) == set(qa.validation_rules)
        # The list-oriented distribution check cannot handle a dict payload.
        assert result["validation_results"]["data_integrity"]["valid"] is True
        assert result["overall_valid"] is False

    def test_validate_component_selected_and_unknown(self, qa: Any) -> None:
        result = qa.validate_component(
            "comp",
            {"timestamp": "t", "data": {"a": 1}},
            ["data_integrity", "not_a_rule"],
        )
        assert list(result["validation_results"]) == ["data_integrity"]
        assert result["overall_valid"] is True
        assert result["validation_types"] == ["data_integrity", "not_a_rule"]

    def test_validate_component_rule_exception_isolated(self, qa: Any) -> None:
        def explode(_: Dict[str, Any]) -> Dict[str, Any]:
            raise RuntimeError("rule crashed")

        qa.validation_rules["explode"] = explode
        result = qa.validate_component(
            "comp", {"timestamp": "t", "data": []}, ["explode", "data_integrity"]
        )
        assert result["validation_results"]["explode"] == {
            "valid": False,
            "error": "rule crashed",
        }
        assert result["validation_results"]["data_integrity"]["valid"] is True
        assert result["overall_valid"] is False


# ------------------------------------------------------------------ framework


class TestTestingFramework:
    def test_composition(self) -> None:
        fw = tf.TestingFramework({"runner": {"r": 1}, "qa": {"q": 2}})
        assert fw.test_runner.config == {"r": 1}
        assert fw.quality_assurance.config == {"q": 2}

    def test_run_tests_delegates(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        module_name = "cov_tools_fw_suite_mod"
        (tmp_path / f"{module_name}.py").write_text(
            "import unittest\n\n"
            "class Case(unittest.TestCase):\n"
            "    def test_one(self):\n"
            "        pass\n"
        )
        monkeypatch.syspath_prepend(str(tmp_path))
        summary = tf.TestingFramework({}).run_tests([module_name])
        assert summary["total_tests"] == 1
        assert summary["passed"] == 1
        assert summary["success_rate"] == 1.0

    def test_validate_model_runs_model_checks(self) -> None:
        fw = tf.TestingFramework({})
        result = fw.validate_model(
            "agent", {"learning_rate": 2.0, "beliefs": [0.5, float("nan")]}
        )
        assert result["component"] == "agent"
        assert result["validation_types"] == [
            "numerical_stability",
            "parameter_ranges",
            "probability_distribution",
        ]
        stability = result["validation_results"]["numerical_stability"]
        assert stability["issues"] == ["NaN values in beliefs"]
        ranges = result["validation_results"]["parameter_ranges"]
        assert ranges["valid"] is False
        assert ranges["parameters_checked"] == 2
        assert result["overall_valid"] is False

    def test_performance_test_statistics(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(tf, "time", _fake_clock(0.5))
        calls = {"n": 0}

        def sometimes_fails() -> None:
            calls["n"] += 1
            if calls["n"] == 2:
                raise RuntimeError("flaky")

        stats = tf.TestingFramework({}).run_performance_test(
            sometimes_fails, "perf", iterations=4
        )
        assert calls["n"] == 4
        assert stats["test_name"] == "perf"
        assert stats["iterations"] == 4
        assert stats["successful_iterations"] == 3
        assert stats["failed_iterations"] == 1
        assert stats["min_time"] == pytest.approx(0.5)
        assert stats["max_time"] == pytest.approx(0.5)
        assert stats["mean_time"] == pytest.approx(0.5)
        assert stats["median_time"] == pytest.approx(0.5)
        assert stats["total_time"] == pytest.approx(1.5)
        assert stats["throughput"] == pytest.approx(2.0)

    def test_performance_test_zero_duration(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(tf, "time", SimpleNamespace(time=lambda: 42.0))
        stats = tf.TestingFramework({}).run_performance_test(
            lambda: None, "instant", iterations=3
        )
        assert stats["successful_iterations"] == 3
        assert stats["total_time"] == 0.0
        assert stats["throughput"] == 0

    def test_performance_test_all_fail(self) -> None:
        def always_fails() -> None:
            raise RuntimeError("down")

        stats = tf.TestingFramework({}).run_performance_test(
            always_fails, "dead", iterations=3
        )
        assert stats == {
            "test_name": "dead",
            "iterations": 3,
            "successful_iterations": 0,
            "failed_iterations": 3,
            "error": "All iterations failed",
        }

    def test_generate_test_report(self, tmp_path: Path) -> None:
        fw = tf.TestingFramework({})
        out = tmp_path / "r.json"
        assert fw.generate_test_report(out) is False
        fw.test_runner.run_test(lambda: None, "only")
        assert fw.generate_test_report(out) is True
        report = json.loads(out.read_text())
        assert report["results_by_status"] == {"passed": 1}
        assert not math.isnan(report["results"][0]["execution_time"])
