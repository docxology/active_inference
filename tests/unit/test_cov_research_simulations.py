"""Coverage tests for active_inference.research.simulations."""

import numpy as np
import pytest

from active_inference.research.simulations import (
    ModelConfig,
    ModelRunner,
    SimulationEngine,
)


@pytest.fixture
def engine() -> SimulationEngine:
    return SimulationEngine({"mode": "test"})


class TestModelConfig:
    def test_defaults(self):
        config = ModelConfig(name="m")
        assert config.parameters == {}
        assert config.model_type == "active_inference"
        assert config.time_horizon == 1000
        assert config.time_step == 0.1

    def test_explicit_parameters_kept(self):
        params = {"a": 1}
        assert ModelConfig(name="m", parameters=params).parameters is params


class TestSimulationEngine:
    def test_register_and_list(self, engine):
        assert engine.config == {"mode": "test"}
        assert engine.register_model(ModelConfig(name="a")) is True
        assert engine.register_model(ModelConfig(name="b")) is True
        assert engine.register_model(ModelConfig(name="a")) is False
        assert engine.list_models() == ["a", "b"]

    def test_unknown_model_raises(self, engine):
        with pytest.raises(ValueError, match="Model ghost not found"):
            engine.run_simulation("ghost")

    def test_active_inference_simulation(self, engine):
        engine.register_model(ModelConfig(name="ai", time_horizon=15, time_step=0.5))
        results = engine.run_simulation(
            "ai", {"n_states": 3, "n_observations": 4, "n_actions": 2}
        )

        assert results["simulation_id"] == "sim_ai_0"
        assert results["model_name"] == "ai"
        assert results["timestamp"] == "2024-10-27T12:00:00"
        assert results["time_points"] == pytest.approx(np.linspace(0, 7.5, 15).tolist())
        for key in ("free_energy", "accuracy", "state_beliefs", "actions"):
            assert len(results[key]) == 15
        assert all(0 <= a < 2 for a in results["actions"])
        assert all(0 <= o < 4 for o in results["observations"])
        assert all(sum(b) == pytest.approx(1.0) for b in results["state_beliefs"])
        assert results["accuracy"] == pytest.approx(
            [max(b) for b in results["state_beliefs"]]
        )
        assert (results["n_states"], results["n_observations"]) == (3, 4)
        assert results["n_actions"] == 2

        params = results["model_parameters"]
        assert params["A_shape"] == (4, 3)
        assert params["B_shape"] == (3, 3, 2)
        assert params["C_shape"] == (4,)
        assert params["final_beliefs"] == results["state_beliefs"][-1]
        assert len(params["preferred_observations"]) == 3
        assert set(params["preferred_observations"]) <= set(range(4))

        assert engine.get_simulation_results("sim_ai_0") is results

    def test_active_inference_is_reproducible_with_defaults(self, engine):
        engine.register_model(ModelConfig(name="ai", time_horizon=5))
        first = engine.run_simulation("ai")
        second = engine.run_simulation("ai")
        assert first["simulation_id"] == "sim_ai_0"
        assert second["simulation_id"] == "sim_ai_1"
        assert first["n_states"] == 4
        assert first["free_energy"] == second["free_energy"]
        assert first["actions"] == second["actions"]

    def test_generic_model_signal(self, engine):
        engine.register_model(
            ModelConfig(
                name="wave",
                model_type="oscillator",
                parameters={"amplitude": 2.0, "frequency": 1.0},
                time_horizon=4,
                time_step=1.0,
            )
        )
        results = engine.run_simulation("wave")
        t = np.linspace(0, 4.0, 4)
        expected = 2.0 * np.sin(2 * np.pi * t / 4)
        assert results["signal"] == pytest.approx(expected.tolist())
        assert results["parameters"] == {"amplitude": 2.0, "frequency": 1.0}
        assert "free_energy" not in results

    def test_generic_model_default_parameters(self, engine):
        engine.register_model(
            ModelConfig(name="g", model_type="generic", time_horizon=3, time_step=1.0)
        )
        results = engine.run_simulation("g")
        expected = np.sin(2 * np.pi * np.linspace(0, 3.0, 3) / 3)
        assert results["signal"] == pytest.approx(expected.tolist())

    def test_missing_results(self, engine):
        assert engine.get_simulation_results("sim_none") is None


class TestModelRunner:
    def test_comparison_study(self, engine):
        runner = ModelRunner(engine)
        assert runner.engine is engine
        configs = [
            ModelConfig(name="ai", time_horizon=4),
            ModelConfig(name="gen", model_type="generic", time_horizon=4),
        ]
        # Pre-register one to confirm re-registration is tolerated
        engine.register_model(configs[0])

        study = runner.run_comparison_study(configs, {"n_states": 2})

        assert study["study_id"] == "comparison_2"
        assert [m["config"].name for m in study["models"]] == ["ai", "gen"]
        assert study["models"][0]["results"]["n_states"] == 2
        assert "signal" in study["models"][1]["results"]
        assert study["comparison_metrics"] == {
            "n_models": 2,
            "model_names": ["ai", "gen"],
        }
        assert sorted(engine.running_simulations) == ["sim_ai_0", "sim_gen_1"]

    def test_empty_comparison_study(self, engine):
        study = ModelRunner(engine).run_comparison_study([])
        assert study["models"] == []
        assert study["comparison_metrics"] == {"n_models": 0, "model_names": []}
