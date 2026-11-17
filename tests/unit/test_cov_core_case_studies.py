"""Coverage-focused behaviour tests for ``applications.case_studies``."""

from pathlib import Path
from typing import Any, Dict

import matplotlib
import numpy as np
import pytest

from active_inference.applications.case_studies import (
    ApplicationDomain,
    CaseStudy,
    CaseStudyManager,
    ExampleApplications,
)

matplotlib.use("Agg")

BUILTIN = ["perceptual_inference", "decision_making", "motor_control"]


@pytest.fixture
def examples(tmp_path: Path) -> ExampleApplications:
    return ExampleApplications(tmp_path)


def _exec(code: str, name: str = "generated_example") -> Dict[str, Any]:
    namespace: Dict[str, Any] = {"__name__": name}
    exec(compile(code, name, "exec"), namespace)
    return namespace


class TestExampleApplications:
    def test_builtin_studies_and_filters(self, examples: ExampleApplications) -> None:
        assert sorted(examples.case_studies) == sorted(BUILTIN)
        titles = [s.title for s in examples.list_case_studies()]
        assert titles == sorted(titles)
        assert [
            s.id
            for s in examples.list_case_studies(domain=ApplicationDomain.PERCEPTION)
        ] == ["perceptual_inference"]
        assert {
            s.id for s in examples.list_case_studies(difficulty="intermediate")
        } == {"perceptual_inference", "motor_control"}
        assert (
            examples.list_case_studies(
                domain=ApplicationDomain.ROBOTICS, difficulty="advanced"
            )
            == []
        )

    def test_unknown_and_custom_studies_get_placeholder_code(
        self, examples: ExampleApplications
    ) -> None:
        assert examples.generate_example_code("nope") == (
            "# Case study 'nope' not found - Implementation pending"
        )
        examples.case_studies["social"] = CaseStudy(
            id="social",
            title="Social Inference",
            domain=ApplicationDomain.SOCIAL_COGNITION,
            description="d",
            difficulty="advanced",
            implementation_files=[],
            requirements=[],
            learning_objectives=[],
            key_concepts=[],
        )
        assert examples.generate_example_code("social") == (
            "# Code for Social Inference - Implementation pending"
        )

    @pytest.mark.parametrize("study_id", BUILTIN)
    def test_generated_code_is_valid_python(
        self, examples: ExampleApplications, study_id: str
    ) -> None:
        code = examples.generate_example_code(study_id)
        namespace = _exec(code)
        class_name = {
            "perceptual_inference": "PerceptualInferenceModel",
            "decision_making": "DecisionMakingModel",
            "motor_control": "MotorControlModel",
        }[study_id]
        assert isinstance(namespace[class_name], type)

    def test_perceptual_example_runs(
        self, examples: ExampleApplications, capsys: pytest.CaptureFixture[str]
    ) -> None:
        np.random.seed(0)
        _exec(examples.generate_example_code("perceptual_inference"), "__main__")
        out = capsys.readouterr().out
        assert "Predicted category:" in out
        assert "Prediction error:" in out

    def test_motor_example_runs(
        self, examples: ExampleApplications, capsys: pytest.CaptureFixture[str]
    ) -> None:
        np.random.seed(0)
        _exec(examples.generate_example_code("motor_control"), "__main__")
        out = capsys.readouterr().out
        assert "Motor control simulation completed:" in out
        assert "Final error:" in out
        assert "Average free energy:" in out


class TestCaseStudyManager:
    def test_paths_and_delegation(self, tmp_path: Path) -> None:
        manager = CaseStudyManager(tmp_path)
        assert manager.examples_dir == tmp_path
        assert manager.example_applications.examples_dir == tmp_path / "examples"
        assert manager.get_case_study("motor_control").title == (
            "Active Inference for Motor Control"
        )
        assert [s.id for s in manager.list_case_studies(difficulty="advanced")] == [
            "decision_making"
        ]

    def test_run_example(self, tmp_path: Path) -> None:
        manager = CaseStudyManager(tmp_path)
        assert manager.run_example("ghost") == {"error": "Case study ghost not found"}
        result = manager.run_example("decision_making")
        assert result["study_id"] == "decision_making"
        assert result["title"] == "Decision Making Under Uncertainty"
        assert result["requirements"] == ["numpy", "scipy", "pandas"]
        assert "class DecisionMakingModel" in result["code"]
        assert len(result["learning_objectives"]) == 3
