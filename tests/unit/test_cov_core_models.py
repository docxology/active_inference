"""Coverage-focused behaviour tests for ``llm.models``."""

import json
from pathlib import Path

import pytest

from active_inference.llm import models as models_mod
from active_inference.llm.models import ModelInfo, ModelManager, ModelRegistry

DEFAULT_NAMES = ["gemma3:2b", "gemma3:4b", "llama2:7b", "llama2:13b", "mistral:7b"]


@pytest.fixture
def manager(tmp_path: Path) -> ModelManager:
    return ModelManager(registry_path=tmp_path / "registry.json")


class TestModelInfo:
    @pytest.mark.parametrize(
        ("quant", "expected"),
        [
            (None, 14.0),
            ("Q2BIT", 3.5),
            ("4bit", 7.0),
            ("int8bit", 10.5),
            ("fp16", 14.0),
        ],
    )
    def test_memory_requirement(self, quant: str, expected: float) -> None:
        assert ModelInfo(
            "m", "f", "7b", quantization=quant
        ).get_memory_requirement() == (expected)

    def test_capabilities(self) -> None:
        info = ModelInfo("m", "f", "1b")
        assert info.has_capability("x") is False
        info.add_capability("x")
        assert info.has_capability("x") is True


class TestModelRegistry:
    def test_best_model_respects_preference_size_and_memory(self) -> None:
        reg = ModelRegistry()
        reg.register_model(ModelInfo("big", "a", "13b", capabilities={"c"}))
        reg.register_model(ModelInfo("small", "a", "2b", capabilities={"c"}))
        reg.register_model(ModelInfo("other", "b", "1b", capabilities={"d"}))
        assert reg.fallback_chain == ["big", "small", "other"]
        assert reg.get_best_model_for_task({"c"}).name == "small"
        reg.set_preferred_order(["big", "ghost"])
        assert reg.preferred_models == ["big"]
        assert reg.get_best_model_for_task({"c"}).name == "big"
        assert reg.get_best_model_for_task({"c"}, max_memory=5).name == "small"
        assert reg.get_best_model_for_task({"c"}, max_memory=1) is None
        assert reg.get_best_model_for_task({"zzz"}) is None
        assert [m.name for m in reg.find_models_by_family("a")] == ["big", "small"]
        assert [m.name for m in reg.find_models_by_capability("d")] == ["other"]

    def test_register_same_name_twice_keeps_single_chain_entry(self) -> None:
        reg = ModelRegistry()
        reg.register_model(ModelInfo("m", "f", "1b"))
        reg.register_model(ModelInfo("m", "f", "2b"))
        assert reg.fallback_chain == ["m"]
        assert reg.get_model("m").size == "2b"

    def test_save_and_load_round_trip(self, tmp_path: Path) -> None:
        reg = ModelRegistry()
        reg.register_model(
            ModelInfo(
                "m",
                "f",
                "3b",
                quantization="4bit",
                context_window=2048,
                capabilities={"x"},
                performance_metrics={"s": 1.0},
                requirements={"gpu": False},
                metadata={"k": "v"},
            )
        )
        reg.set_preferred_order(["m"])
        path = tmp_path / "r.json"
        reg.save_registry(path)
        loaded = ModelRegistry()
        loaded.load_registry(path)
        assert loaded.models == reg.models
        assert loaded.preferred_models == ["m"]
        assert loaded.fallback_chain == ["m"]

    def test_load_minimal_registry_uses_defaults(self, tmp_path: Path) -> None:
        path = tmp_path / "r.json"
        path.write_text(
            json.dumps({"models": {"m": {"name": "m", "family": "f", "size": "1b"}}})
        )
        reg = ModelRegistry()
        reg.load_registry(path)
        info = reg.get_model("m")
        assert info.context_window == 4096
        assert info.capabilities == set()
        assert reg.preferred_models == []
        assert reg.fallback_chain == ["m"]


class TestModelManager:
    def test_defaults_and_registry_file_written_on_change(
        self, manager: ModelManager, tmp_path: Path
    ) -> None:
        assert manager.get_available_models() == DEFAULT_NAMES
        assert manager.registry.preferred_models == [
            "gemma3:2b",
            "gemma3:4b",
            "mistral:7b",
            "llama2:7b",
            "llama2:13b",
        ]
        assert not (tmp_path / "registry.json").exists()
        manager.add_custom_model("phi:3b", "phi", "3b", ["text_generation"])
        saved = json.loads((tmp_path / "registry.json").read_text())
        assert "phi:3b" in saved["models"]

    def test_default_registry_path(self) -> None:
        # conftest points DEFAULT_REGISTRY_PATH at a per-test copy.
        assert ModelManager().registry_path == models_mod.DEFAULT_REGISTRY_PATH

    def test_loads_existing_registry(self, tmp_path: Path) -> None:
        path = tmp_path / "reg.json"
        first = ModelManager(registry_path=path)
        first.add_custom_model("custom:1b", "custom", "1b", ["conversation"])
        second = ModelManager(registry_path=path)
        assert second.get_model_info("custom:1b").capabilities == {"conversation"}
        assert "gemma3:2b" in second.get_available_models()

    def test_corrupt_registry_is_ignored(self, tmp_path: Path) -> None:
        path = tmp_path / "reg.json"
        path.write_text("{not json")
        mgr = ModelManager(registry_path=path)
        assert mgr.get_available_models() == DEFAULT_NAMES

    def test_task_capabilities(self, manager: ModelManager) -> None:
        assert manager._get_task_capabilities("Write CODING exercise") == {
            "code_generation",
            "reasoning",
        }
        assert manager._get_task_capabilities("free_energy derivation") == {
            "mathematical_reasoning",
            "active_inference_explanation",
        }
        assert manager._get_task_capabilities("poetry") == {
            "text_generation",
            "reasoning",
        }

    def test_find_best_model(self, manager: ModelManager) -> None:
        assert manager.find_best_model("explanation").name == "gemma3:2b"
        assert manager.find_best_model("research").name == "gemma3:4b"
        assert (
            manager.find_best_model("research", preferred_family="llama").name
            == "llama2:7b"
        )
        # The family has no model under the memory cap; fall back to all models.
        assert (
            manager.find_best_model(
                "research", max_memory_gb=10, preferred_family="llama"
            ).name
            == "gemma3:4b"
        )
        assert manager.find_best_model("research", max_memory_gb=1) is None

    def test_models_by_capability(self, manager: ModelManager) -> None:
        names = [m.name for m in manager.get_models_by_capability("complex_reasoning")]
        assert names == ["llama2:13b"]

    def test_update_performance(self, manager: ModelManager, tmp_path: Path) -> None:
        manager.update_model_performance("gemma3:2b", {"latency": 0.5})
        assert manager.get_model_info("gemma3:2b").performance_metrics == {
            "latency": 0.5
        }
        saved = json.loads((tmp_path / "registry.json").read_text())
        assert saved["models"]["gemma3:2b"]["performance_metrics"] == {"latency": 0.5}
        manager.update_model_performance("missing", {"latency": 1.0})
        assert manager.get_model_info("missing") is None

    def test_add_custom_model_kwargs(self, manager: ModelManager) -> None:
        info = manager.add_custom_model(
            "q:7b", "q", "7b", ["a"], quantization="4bit", context_window=1024
        )
        assert info.get_memory_requirement() == 7.0
        assert info.context_window == 1024
        assert manager.registry.fallback_chain[-1] == "q:7b"

    def test_remove_model(self, manager: ModelManager) -> None:
        assert manager.remove_model("gemma3:2b") is True
        assert "gemma3:2b" not in manager.registry.models
        assert "gemma3:2b" not in manager.registry.fallback_chain
        assert "gemma3:2b" not in manager.registry.preferred_models
        assert manager.remove_model("gemma3:2b") is False

    def test_remove_model_not_in_chains(self, manager: ModelManager) -> None:
        manager.registry.models["loose"] = ModelInfo("loose", "x", "1b")
        assert manager.remove_model("loose") is True
        assert "loose" not in manager.registry.models

    def test_save_failure_is_logged(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        mgr = ModelManager(registry_path=tmp_path / "missing_dir" / "reg.json")
        with caplog.at_level("ERROR"):
            mgr.save_registry()
        assert "Failed to save model registry" in caplog.text
        assert not (tmp_path / "missing_dir").exists()

    def test_system_info(self, manager: ModelManager) -> None:
        info = manager.get_system_info()
        assert info["total_models"] == 5
        assert sorted(info["available_families"]) == ["gemma", "llama", "mistral"]
        assert info["fallback_chain"] == DEFAULT_NAMES

    def test_benchmark(self, manager: ModelManager) -> None:
        assert manager.benchmark_model("missing", ["t"]) == {}
        metrics = manager.benchmark_model("llama2:7b", ["t"])
        assert metrics == {
            "reasoning_score": 0.8,
            "generation_speed": pytest.approx(100.0 / 7),
            "memory_efficiency": pytest.approx(1 / 14),
            "context_utilization": 0.5,
        }
        assert (
            manager.get_model_info("llama2:7b").performance_metrics["reasoning_score"]
            == 0.8
        )
        manager.add_custom_model("plain:4b", "p", "4b", [], context_window=16384)
        plain = manager.benchmark_model("plain:4b", [])
        assert plain["reasoning_score"] == 0.5
        assert plain["context_utilization"] == 1.0
