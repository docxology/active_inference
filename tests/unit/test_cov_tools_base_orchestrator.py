"""Behavioural tests for active_inference.tools.orchestrators.base_orchestrator."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import pytest

from active_inference.tools.orchestrators import base_orchestrator as bo
from active_inference.tools.orchestrators.base_orchestrator import (
    BaseOrchestrator,
    OrchestratorConfig,
    ResearchRole,
    ResearchStage,
)

ALL_STAGES = [
    ResearchStage.IDEATION,
    ResearchStage.HYPOTHESIS,
    ResearchStage.DESIGN,
    ResearchStage.EXECUTION,
    ResearchStage.ANALYSIS,
    ResearchStage.VALIDATION,
    ResearchStage.DOCUMENTATION,
    ResearchStage.PUBLICATION,
]


class EchoOrchestrator(BaseOrchestrator):
    """Concrete orchestrator used to exercise the base workflow."""

    def __init__(self, config: OrchestratorConfig) -> None:
        super().__init__(config)
        self.executed: List[ResearchStage] = []

    def _get_required_context_fields(self, stage: ResearchStage) -> List[str]:
        return ["topic"] if stage == ResearchStage.HYPOTHESIS else []

    def _execute_stage(
        self, stage: ResearchStage, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        self.executed.append(stage)
        if context.get("fail"):
            raise RuntimeError("stage exploded")
        if context.get("non_dict"):
            return ["not", "a", "dict"]  # type: ignore[return-value]
        return {"echo": dict(context), "stage": stage.value}


def make_config(tmp_path: Path, **overrides: Any) -> OrchestratorConfig:
    params: Dict[str, Any] = {
        "name": "unit",
        "role": ResearchRole.POSTDOC,
        "stages": [ResearchStage.IDEATION, ResearchStage.HYPOTHESIS],
        "output_dir": tmp_path / "out",
        "enable_logging": False,
    }
    params.update(overrides)
    return OrchestratorConfig(**params)


def test_config_defaults_and_output_dir_created(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    assert config.output_dir.is_dir()
    assert config.enable_validation is True
    assert config.enable_caching is True
    assert config.max_parallel_processes == 4
    assert config.timeout_minutes == 60


def test_enum_values() -> None:
    assert ResearchStage("grant_writing") is ResearchStage.GRANT_WRITING
    assert ResearchRole("phd_student") is ResearchRole.PHD_STUDENT


def test_setup_logging_configures_file_and_stream_handlers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: Dict[str, Any] = {}

    def fake_basic_config(**kwargs: Any) -> None:
        captured.update(kwargs)

    monkeypatch.setattr(bo.logging, "basicConfig", fake_basic_config)
    config = make_config(tmp_path, enable_logging=True, name="logged")
    EchoOrchestrator(config)

    handlers = captured["handlers"]
    try:
        assert captured["level"] == logging.INFO
        file_handlers = [h for h in handlers if isinstance(h, logging.FileHandler)]
        assert len(file_handlers) == 1
        log_path = Path(file_handlers[0].baseFilename)
        assert log_path.parent == config.output_dir
        assert log_path.name.startswith("logged_")
        assert log_path.suffix == ".log"
        assert any(
            isinstance(h, logging.StreamHandler)
            and not isinstance(h, logging.FileHandler)
            for h in handlers
        )
    finally:
        for handler in handlers:
            handler.close()


def test_base_execute_stage_not_implemented(tmp_path: Path) -> None:
    orchestrator = BaseOrchestrator(make_config(tmp_path))
    with pytest.raises(NotImplementedError, match="Subclasses must implement"):
        orchestrator.coordinate_stage(ResearchStage.IDEATION, {})
    error = orchestrator.get_research_state()["ideation_error"]
    assert error["error_type"] == "NotImplementedError"
    assert error["context_summary"] == {"context_keys": [], "context_size": 0}


def test_coordinate_stage_success(tmp_path: Path) -> None:
    orchestrator = EchoOrchestrator(make_config(tmp_path))
    context = {"topic": "precision"}
    results = orchestrator.coordinate_stage(ResearchStage.HYPOTHESIS, context)

    assert results["echo"] == context
    assert results["_metadata"]["stage"] == "hypothesis"
    assert results["_metadata"]["orchestrator"] == "unit"
    assert results["_metadata"]["role"] == "postdoc"

    state = orchestrator.get_research_state()
    entry = state["hypothesis"]
    assert entry["completed"] is True
    assert entry["results_summary"]["metadata_available"] is True
    assert entry["results_summary"]["metadata"] == results["_metadata"]
    assert set(entry["results_summary"]["result_keys"]) == {
        "echo",
        "stage",
        "_metadata",
    }
    assert orchestrator.get_stage_results(ResearchStage.HYPOTHESIS) == results

    cache_key = orchestrator._generate_cache_key(ResearchStage.HYPOTHESIS, context)
    assert cache_key.startswith("hypothesis_")
    assert orchestrator.results_cache[cache_key]["results"] is results


def test_get_research_state_returns_copy(tmp_path: Path) -> None:
    orchestrator = EchoOrchestrator(make_config(tmp_path))
    orchestrator.coordinate_stage(ResearchStage.IDEATION, {})
    snapshot = orchestrator.get_research_state()
    snapshot.pop("ideation")
    assert "ideation" in orchestrator.research_state


def test_get_stage_results_for_unrun_stage(tmp_path: Path) -> None:
    orchestrator = EchoOrchestrator(make_config(tmp_path))
    assert orchestrator.get_stage_results(ResearchStage.IDEATION) is None


def test_cache_key_is_order_independent(tmp_path: Path) -> None:
    orchestrator = EchoOrchestrator(make_config(tmp_path))
    first = orchestrator._generate_cache_key(ResearchStage.DESIGN, {"a": 1, "b": 2})
    second = orchestrator._generate_cache_key(ResearchStage.DESIGN, {"b": 2, "a": 1})
    other = orchestrator._generate_cache_key(ResearchStage.DESIGN, {"a": 2})
    assert first == second
    assert first != other


def test_coordinate_stage_without_validation_or_caching(tmp_path: Path) -> None:
    config = make_config(tmp_path, enable_validation=False, enable_caching=False)
    orchestrator = EchoOrchestrator(config)
    # HYPOTHESIS normally requires "topic", but validation is disabled.
    results = orchestrator.coordinate_stage(ResearchStage.HYPOTHESIS, {})
    assert "_metadata" not in results
    assert orchestrator.results_cache == {}
    summary = orchestrator.research_state["hypothesis"]["results_summary"]
    assert summary["metadata_available"] is False
    assert "metadata" not in summary


def test_coordinate_unsupported_stage(tmp_path: Path) -> None:
    orchestrator = EchoOrchestrator(make_config(tmp_path))
    with pytest.raises(ValueError, match="Invalid stage"):
        orchestrator.coordinate_stage(ResearchStage.PUBLICATION, {})
    assert orchestrator.executed == []


def test_coordinate_missing_context_field(tmp_path: Path) -> None:
    orchestrator = EchoOrchestrator(make_config(tmp_path))
    with pytest.raises(ValueError, match="Missing required context field: topic"):
        orchestrator.coordinate_stage(ResearchStage.HYPOTHESIS, {"other": 1})
    assert orchestrator.executed == []
    assert orchestrator.research_state == {}


def test_coordinate_stage_failure_records_error(tmp_path: Path) -> None:
    orchestrator = EchoOrchestrator(make_config(tmp_path))
    with pytest.raises(RuntimeError, match="stage exploded"):
        orchestrator.coordinate_stage(ResearchStage.IDEATION, {"fail": True, "x": 1})
    error = orchestrator.research_state["ideation_error"]
    assert error["stage"] == "ideation"
    assert error["error_type"] == "RuntimeError"
    assert error["error_message"] == "stage exploded"
    assert error["context_summary"] == {
        "context_keys": ["fail", "x"],
        "context_size": 2,
    }
    assert "ideation" not in orchestrator.research_state
    assert orchestrator.results_cache == {}


def test_non_dict_results_rejected(tmp_path: Path) -> None:
    orchestrator = EchoOrchestrator(make_config(tmp_path))
    with pytest.raises(ValueError, match="Results must be a dictionary"):
        orchestrator.coordinate_stage(ResearchStage.IDEATION, {"non_dict": True})
    assert orchestrator.research_state["ideation_error"]["error_type"] == "ValueError"


def test_validate_workflow_complete(tmp_path: Path) -> None:
    orchestrator = EchoOrchestrator(make_config(tmp_path))
    validation = orchestrator.validate_workflow(ALL_STAGES)
    assert validation == {
        "valid": True,
        "issues": [],
        "recommendations": [],
        "missing_stages": [],
        "redundant_stages": [],
    }


def test_validate_workflow_missing_and_misordered(tmp_path: Path) -> None:
    orchestrator = EchoOrchestrator(make_config(tmp_path))
    validation = orchestrator.validate_workflow(
        [ResearchStage.ANALYSIS, ResearchStage.PUBLICATION]
    )
    assert validation["valid"] is False
    assert sorted(validation["missing_stages"]) == ["execution", "ideation"]
    issues = validation["issues"]
    assert issues[0].startswith("Missing required stages")
    assert "Stage analysis requires execution but it's not before it" in issues
    assert "Stage publication requires documentation but it's not before it" in (issues)
    assert "Stage publication requires validation but it's not before it" in issues
    assert validation["recommendations"] == [
        "Consider starting with ideation stage",
        "Consider adding validation stage after analysis",
        "Consider adding documentation stage before publication",
    ]


def test_validate_workflow_empty(tmp_path: Path) -> None:
    validation = EchoOrchestrator(make_config(tmp_path)).validate_workflow([])
    assert validation["valid"] is False
    assert validation["recommendations"] == []
    assert len(validation["missing_stages"]) == 3


def test_export_research_summary_default_path(tmp_path: Path) -> None:
    orchestrator = EchoOrchestrator(make_config(tmp_path))
    orchestrator.coordinate_stage(ResearchStage.IDEATION, {"topic": "t"})
    with pytest.raises(RuntimeError):
        orchestrator.coordinate_stage(
            ResearchStage.HYPOTHESIS, {"topic": "t", "fail": 1}
        )

    path = orchestrator.export_research_summary()
    assert path.parent == orchestrator.config.output_dir
    assert path.name.startswith("research_summary_")
    data = json.loads(path.read_text())
    assert data["orchestrator_config"] == {
        "name": "unit",
        "role": "postdoc",
        "stages": ["ideation", "hypothesis"],
    }
    assert set(data["research_state"]) == {"ideation", "hypothesis_error"}
    # The error entry is excluded from the workflow being validated.
    assert data["workflow_validation"]["valid"] is False
    assert sorted(data["workflow_validation"]["missing_stages"]) == [
        "analysis",
        "execution",
    ]
    assert data["active_inference_version"] == "0.1.0"


def test_export_research_summary_explicit_path(tmp_path: Path) -> None:
    orchestrator = EchoOrchestrator(make_config(tmp_path))
    target = tmp_path / "summary.json"
    assert orchestrator.export_research_summary(target) == target
    data = json.loads(target.read_text())
    assert data["research_state"] == {}
    assert data["workflow_validation"]["recommendations"] == []


def test_get_capabilities(tmp_path: Path) -> None:
    caps = EchoOrchestrator(make_config(tmp_path)).get_capabilities()
    assert caps["supported_stages"] == ["ideation", "hypothesis"]
    assert caps["research_role"] == "postdoc"
    assert "workflow_validation" in caps["features"]
    assert caps["integrations"] == [
        "experiments",
        "simulations",
        "analysis",
        "benchmarks",
    ]
