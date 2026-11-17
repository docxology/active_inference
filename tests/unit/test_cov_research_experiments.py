"""Coverage tests for active_inference.research.experiments."""

import json
from pathlib import Path

import numpy as np
import pytest

from active_inference.research import experiments as experiments_module
from active_inference.research.experiments import (
    ExperimentConfig,
    ExperimentManager,
    ExperimentStatus,
    ResearchFramework,
)


@pytest.fixture(autouse=True)
def _seed() -> None:
    np.random.seed(1234)


@pytest.fixture
def manager(tmp_path: Path) -> ExperimentManager:
    return ExperimentManager(tmp_path / "experiments")


def _config(name: str = "exp", **kwargs) -> ExperimentConfig:
    kwargs.setdefault("simulation_steps", 20)
    return ExperimentConfig(name=name, description=f"{name} description", **kwargs)


class TestExperimentConfig:
    def test_defaults(self):
        config = ExperimentConfig(name="n", description="d")
        assert config.parameters == {}
        assert config.model_type == "active_inference"
        assert config.simulation_steps == 1000
        assert config.output_dir is None
        assert config.random_seed is None

    def test_status_values(self):
        assert [s.value for s in ExperimentStatus] == [
            "pending",
            "running",
            "completed",
            "failed",
            "cancelled",
        ]


class TestExperimentManager:
    def test_init_creates_base_dir(self, tmp_path):
        base = tmp_path / "nested" / "dir"
        mgr = ExperimentManager(base)
        assert base.is_dir()
        assert mgr.experiments == {}

    def test_create_experiment_writes_config(self, manager, tmp_path):
        config = _config(
            "alpha",
            parameters={"lr": 0.1},
            output_dir=tmp_path / "out",
            random_seed=7,
        )
        exp_id = manager.create_experiment(config)

        assert exp_id.startswith("exp_")
        record = manager.get_experiment(exp_id)
        assert record["status"] is ExperimentStatus.PENDING
        assert record["config"] is config
        assert record["results"] is None

        saved = json.loads((manager.base_dir / exp_id / "config.json").read_text())
        assert saved == {
            "name": "alpha",
            "description": "alpha description",
            "parameters": {"lr": 0.1},
            "model_type": "active_inference",
            "simulation_steps": 20,
            "output_dir": str(tmp_path / "out"),
            "random_seed": 7,
        }

    def test_create_experiment_ids_are_unique_within_same_second(self, manager):
        ids = [manager.create_experiment(_config(f"e{i}")) for i in range(3)]
        assert len(set(ids)) == 3
        assert len(manager.experiments) == 3
        for exp_id, name in zip(ids, ["e0", "e1", "e2"]):
            saved = json.loads((manager.base_dir / exp_id / "config.json").read_text())
            assert saved["name"] == name

    def test_create_experiment_does_not_reuse_existing_directory(self, tmp_path):
        base = tmp_path / "experiments"
        first = ExperimentManager(base).create_experiment(_config("first"))
        second = ExperimentManager(base).create_experiment(_config("second"))
        assert first != second
        first_cfg = json.loads((base / first / "config.json").read_text())
        assert first_cfg["name"] == "first"

    def test_run_unknown_experiment(self, manager):
        assert manager.run_experiment("missing") is False

    def test_run_mock_experiment(self, manager):
        exp_id = manager.create_experiment(_config("mock", parameters={"k": 1}))
        assert manager.run_experiment(exp_id) is True

        record = manager.get_experiment(exp_id)
        assert record["status"] is ExperimentStatus.COMPLETED
        results = record["results"]
        assert results["experiment_id"] == exp_id
        assert results["model_type"] == "active_inference"
        assert results["parameters"] == {"k": 1}
        assert len(results["free_energy"]) == 20
        assert len(results["accuracy"]) == 20
        assert results["free_energy"][0] == 1.0
        assert all(0.1 <= fe for fe in results["free_energy"])
        assert all(0.5 <= a <= 0.95 for a in results["accuracy"])
        assert results["convergence_time"] % 10 == 0

        on_disk = json.loads((manager.base_dir / exp_id / "results.json").read_text())
        assert on_disk["free_energy"] == pytest.approx(results["free_energy"])

        log = json.loads(
            (manager.base_dir / exp_id / "experiment_log.json").read_text()
        )
        assert log["experiment_id"] == exp_id
        assert log["status"] == "completed"
        assert log["duration_seconds"] >= 0
        summary = log["results_summary"]
        assert summary["final_free_energy"] == pytest.approx(results["free_energy"][-1])
        assert summary["final_accuracy"] == pytest.approx(results["accuracy"][-1])
        assert summary["total_steps"] == 20

    def test_run_experiment_with_output_dir_logs_path(self, manager, tmp_path):
        out = tmp_path / "out"
        exp_id = manager.create_experiment(_config("withdir", output_dir=out))
        assert manager.run_experiment(exp_id) is True
        log = json.loads(
            (manager.base_dir / exp_id / "experiment_log.json").read_text()
        )
        assert log["config"]["output_dir"] == str(out)

    def test_run_real_experiment(self, manager):
        exp_id = manager.create_experiment(
            _config("real", parameters={"n_states": 3, "n_observations": 5})
        )
        assert manager.run_experiment(exp_id, real_execution=True) is True

        results = manager.get_experiment(exp_id)["results"]
        assert results["model_type"] == "active_inference"
        assert results["parameters"] == {
            "n_states": 3,
            "n_observations": 5,
            "time_steps": 20,
            "alpha": 1.0,
            "beta": 1.0,
        }
        assert len(results["free_energy"]) == 20
        assert len(results["belief_trajectory"]) == 20
        assert all(len(b) == 3 for b in results["belief_trajectory"])
        assert sum(results["final_beliefs"]) == pytest.approx(1.0)
        assert results["accuracy"] == pytest.approx(
            [max(b) for b in results["belief_trajectory"]]
        )
        assert 0 <= results["convergence_time"] <= 100

    def test_run_experiment_failure_marks_failed(self, manager, monkeypatch):
        exp_id = manager.create_experiment(_config("boom"))

        def explode(*_args, **_kwargs):
            raise RuntimeError("simulated failure")

        monkeypatch.setattr(manager, "_generate_mock_results", explode)
        assert manager.run_experiment(exp_id) is False

        record = manager.get_experiment(exp_id)
        assert record["status"] is ExperimentStatus.FAILED
        assert record["error"] == "simulated failure"
        assert "failed_at" in record
        assert not (manager.base_dir / exp_id / "results.json").exists()

    def test_save_logs_without_timestamps_or_results(self, manager):
        exp_id = "manual"
        (manager.base_dir / exp_id).mkdir()
        experiment = {"config": _config("m"), "status": ExperimentStatus.PENDING}
        manager._save_experiment_logs(exp_id, experiment)

        log = json.loads(
            (manager.base_dir / exp_id / "experiment_log.json").read_text()
        )
        assert log["status"] == "pending"
        assert log["duration_seconds"] is None
        assert log["results_summary"] is None

    def test_save_logs_with_empty_results(self, manager):
        exp_id = "empty"
        (manager.base_dir / exp_id).mkdir()
        experiment = {
            "config": _config("e"),
            "status": ExperimentStatus.COMPLETED,
            "results": {"convergence_time": 5},
        }
        manager._save_experiment_logs(exp_id, experiment)
        log = json.loads(
            (manager.base_dir / exp_id / "experiment_log.json").read_text()
        )
        assert log["results_summary"] == {
            "final_free_energy": None,
            "final_accuracy": None,
            "convergence_time": 5,
            "total_steps": 0,
        }

    def test_get_and_list_experiments(self, manager):
        assert manager.get_experiment("nope") is None
        first = manager.create_experiment(_config("a"))
        second = manager.create_experiment(_config("b"))
        manager.experiments[first]["created_at"] = "2020-01-01T00:00:00"
        manager.experiments[second]["created_at"] = "2021-01-01T00:00:00"
        manager.run_experiment(second)

        listed = manager.list_experiments()
        assert [e["id"] for e in listed] == [second, first]

        completed = manager.list_experiments(ExperimentStatus.COMPLETED)
        assert [e["id"] for e in completed] == [second]
        pending = manager.list_experiments(ExperimentStatus.PENDING)
        assert [e["id"] for e in pending] == [first]


class TestResearchFramework:
    def test_run_study(self, tmp_path):
        framework = ResearchFramework({"experiments_dir": tmp_path / "study"})
        assert framework.simulation_engine is None
        assert framework.analysis_tools is None

        study = {
            "name": "pilot",
            "experiments": [
                {"name": "one", "description": "d1", "simulation_steps": 5},
                {"name": "two", "description": "d2", "simulation_steps": 5},
            ],
        }
        results = framework.run_study(study)

        assert results["study_name"] == "pilot"
        assert len(results["experiments"]) == 2
        assert len({e["id"] for e in results["experiments"]}) == 2
        assert [e["data"]["config"].name for e in results["experiments"]] == [
            "one",
            "two",
        ]
        assert all(e["success"] for e in results["experiments"])
        assert results["summary"] == {
            "total_experiments": 2,
            "successful_experiments": 2,
            "success_rate": 1.0,
        }

    def test_run_empty_study(self, tmp_path):
        framework = ResearchFramework({"experiments_dir": tmp_path / "study"})
        results = framework.run_study({})
        assert results["study_name"] is None
        assert results["summary"]["success_rate"] == 0

    def test_default_experiments_dir(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        framework = ResearchFramework({})
        assert framework.experiment_manager.base_dir == Path("./experiments")
        assert (tmp_path / "experiments").is_dir()

    def test_module_logger_name(self):
        assert experiments_module.logger.name == (
            "active_inference.research.experiments"
        )
