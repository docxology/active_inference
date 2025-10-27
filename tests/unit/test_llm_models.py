"""
Tests for LLM Model Management

Unit tests for the model management system, ensuring proper operation of
model registry, capability matching, and fallback strategies.
"""

import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

from active_inference.llm.models import (
    ModelInfo,
    ModelRegistry,
    ModelManager
)


class TestModelInfo:
    """Test cases for ModelInfo class"""

    def test_model_info_creation(self):
        """Test creating model information"""
        model = ModelInfo(
            name="gemma3:2b",
            family="gemma",
            size="2b",
            quantization="4bit",
            context_window=8192,
            capabilities={"text_generation", "reasoning"}
        )

        assert model.name == "gemma3:2b"
        assert model.family == "gemma"
        assert model.size == "2b"
        assert model.quantization == "4bit"
        assert model.context_window == 8192
        assert "text_generation" in model.capabilities
        assert "reasoning" in model.capabilities

    def test_capability_methods(self):
        """Test capability management methods"""
        model = ModelInfo(
            name="test_model",
            family="test",
            size="1b",
            capabilities={"generation"}
        )

        # Test has_capability
        assert model.has_capability("generation") is True
        assert model.has_capability("nonexistent") is False

        # Test add_capability
        model.add_capability("new_capability")
        assert model.has_capability("new_capability") is True

    def test_memory_requirement(self):
        """Test memory requirement calculation"""
        # Test base model
        model = ModelInfo(name="test:7b", family="test", size="7b")
        memory = model.get_memory_requirement()
        assert memory == 14.0  # 7 * 2

        # Test quantized model
        model_4bit = ModelInfo(name="test:7b", family="test", size="7b", quantization="4bit")
        memory_4bit = model_4bit.get_memory_requirement()
        assert memory_4bit == 7.0  # 14 * 0.5

        # Test 2bit quantization
        model_2bit = ModelInfo(name="test:7b", family="test", size="7b", quantization="2bit")
        memory_2bit = model_2bit.get_memory_requirement()
        assert memory_2bit == 3.5  # 14 * 0.25


class TestModelRegistry:
    """Test cases for ModelRegistry class"""

    @pytest.fixture
    def sample_models(self):
        """Create sample models for testing"""
        return [
            ModelInfo(
                name="gemma3:2b",
                family="gemma",
                size="2b",
                capabilities={"text_generation", "reasoning"}
            ),
            ModelInfo(
                name="gemma3:4b",
                family="gemma",
                size="4b",
                capabilities={"text_generation", "reasoning", "mathematical"}
            ),
            ModelInfo(
                name="llama2:7b",
                family="llama",
                size="7b",
                capabilities={"text_generation", "reasoning", "code"}
            )
        ]

    def test_registry_creation(self):
        """Test creating model registry"""
        registry = ModelRegistry()

        assert len(registry.models) == 0
        assert len(registry.preferred_models) == 0
        assert len(registry.fallback_chain) == 0

    def test_register_model(self, sample_models):
        """Test registering models"""
        registry = ModelRegistry()

        for model in sample_models:
            registry.register_model(model)

        assert len(registry.models) == 3
        assert "gemma3:2b" in registry.models
        assert "gemma3:4b" in registry.models
        assert "llama2:7b" in registry.models

        # Check fallback chain
        assert "gemma3:2b" in registry.fallback_chain
        assert "gemma3:4b" in registry.fallback_chain
        assert "llama2:7b" in registry.fallback_chain

    def test_find_models_by_capability(self, sample_models):
        """Test finding models by capability"""
        registry = ModelRegistry()

        for model in sample_models:
            registry.register_model(model)

        # Find models with reasoning capability
        reasoning_models = registry.find_models_by_capability("reasoning")
        assert len(reasoning_models) == 3

        # Find models with mathematical capability
        math_models = registry.find_models_by_capability("mathematical")
        assert len(math_models) == 1
        assert math_models[0].name == "gemma3:4b"

        # Find models with code capability
        code_models = registry.find_models_by_capability("code")
        assert len(code_models) == 1
        assert code_models[0].name == "llama2:7b"

    def test_find_models_by_family(self, sample_models):
        """Test finding models by family"""
        registry = ModelRegistry()

        for model in sample_models:
            registry.register_model(model)

        # Find gemma models
        gemma_models = registry.find_models_by_family("gemma")
        assert len(gemma_models) == 2
        assert all(model.family == "gemma" for model in gemma_models)

        # Find llama models
        llama_models = registry.find_models_by_family("llama")
        assert len(llama_models) == 1
        assert llama_models[0].family == "llama"

    def test_get_best_model_for_task(self, sample_models):
        """Test getting best model for task"""
        registry = ModelRegistry()

        for model in sample_models:
            registry.register_model(model)

        # Set preferred order
        registry.set_preferred_order(["gemma3:2b", "gemma3:4b", "llama2:7b"])

        # Test basic reasoning task
        best_model = registry.get_best_model_for_task({"reasoning"})
        assert best_model is not None
        assert best_model.name == "gemma3:2b"  # First in preferred order

        # Test mathematical task
        best_model = registry.get_best_model_for_task({"mathematical"})
        assert best_model is not None
        assert best_model.name == "gemma3:4b"  # Only model with mathematical capability

        # Test with memory constraint
        best_model = registry.get_best_model_for_task({"reasoning"}, max_memory=5.0)
        assert best_model is not None
        # Should select smaller model that fits memory constraint

    def test_preferred_order(self, sample_models):
        """Test preferred model order"""
        registry = ModelRegistry()

        for model in sample_models:
            registry.register_model(model)

        registry.set_preferred_order(["llama2:7b", "gemma3:4b", "gemma3:2b"])

        assert registry.preferred_models == ["llama2:7b", "gemma3:4b", "gemma3:2b"]

        # Test that preferred order affects best model selection
        best_model = registry.get_best_model_for_task({"reasoning"})
        assert best_model.name == "llama2:7b"  # First in preferred order

    def test_save_and_load_registry(self, sample_models, temp_templates_dir):
        """Test saving and loading registry"""
        registry = ModelRegistry()

        for model in sample_models:
            registry.register_model(model)

        registry.set_preferred_order(["gemma3:2b", "gemma3:4b"])

        # Save registry
        registry_path = temp_templates_dir / "test_registry.json"
        registry.save_registry(registry_path)

        # Create new registry and load
        new_registry = ModelRegistry()
        new_registry.load_registry(registry_path)

        # Verify loaded data
        assert len(new_registry.models) == 3
        assert "gemma3:2b" in new_registry.models
        assert new_registry.preferred_models == ["gemma3:2b", "gemma3:4b"]
        assert set(new_registry.fallback_chain) == set(registry.fallback_chain)


class TestModelManager:
    """Test cases for ModelManager class"""

    @pytest.fixture
    def temp_registry_file(self):
        """Create temporary registry file"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            # Create empty registry file
            json.dump({"models": {}, "preferred_models": [], "fallback_chain": []}, f)
            temp_path = Path(f.name)

        yield temp_path

        # Cleanup
        if temp_path.exists():
            temp_path.unlink()

    def test_manager_creation(self):
        """Test creating model manager"""
        manager = ModelManager()

        # Should have default models loaded
        assert len(manager.registry.models) > 0
        assert "gemma3:2b" in manager.registry.models
        assert "gemma3:4b" in manager.registry.models

    def test_find_best_model(self):
        """Test finding best model for tasks"""
        manager = ModelManager()

        # Test explanation task
        best_model = manager.find_best_model("explanation")
        assert best_model is not None
        assert best_model.has_capability("text_generation")

        # Test mathematical task
        best_model = manager.find_best_model("mathematical reasoning")
        assert best_model is not None
        assert best_model.has_capability("mathematical_reasoning")

        # Test coding task
        best_model = manager.find_best_model("code implementation")
        assert best_model is not None
        assert best_model.has_capability("code_generation")

    def test_get_models_by_capability(self):
        """Test getting models by capability"""
        manager = ModelManager()

        # Get models with reasoning capability
        reasoning_models = manager.get_models_by_capability("reasoning")
        assert len(reasoning_models) > 0
        assert all(model.has_capability("reasoning") for model in reasoning_models)

        # Get models with mathematical capability
        math_models = manager.get_models_by_capability("mathematical_reasoning")
        assert len(math_models) > 0
        assert all(model.has_capability("mathematical_reasoning") for model in math_models)

    def test_add_custom_model(self):
        """Test adding custom model"""
        manager = ModelManager()

        initial_count = len(manager.registry.models)

        model = manager.add_custom_model(
            name="custom_model:1b",
            family="custom",
            size="1b",
            capabilities=["text_generation", "custom_capability"],
            context_window=2048
        )

        assert len(manager.registry.models) == initial_count + 1
        assert "custom_model:1b" in manager.registry.models
        assert model.family == "custom"
        assert model.has_capability("custom_capability")

    def test_remove_model(self):
        """Test removing model"""
        manager = ModelManager()

        # Add a custom model first
        manager.add_custom_model(
            name="temp_model",
            family="temp",
            size="1b",
            capabilities=["temp"]
        )

        initial_count = len(manager.registry.models)

        # Remove the model
        success = manager.remove_model("temp_model")

        assert success is True
        assert len(manager.registry.models) == initial_count - 1
        assert "temp_model" not in manager.registry.models

        # Try to remove non-existent model
        success = manager.remove_model("nonexistent")
        assert success is False

    def test_update_model_performance(self):
        """Test updating model performance metrics"""
        manager = ModelManager()

        # Add custom model for testing
        manager.add_custom_model(
            name="performance_test",
            family="test",
            size="1b",
            capabilities=["test"]
        )

        # Update performance metrics
        metrics = {
            "reasoning_score": 0.85,
            "generation_speed": 50.0,
            "memory_efficiency": 0.9
        }

        manager.update_model_performance("performance_test", metrics)

        # Check that metrics were updated
        model = manager.get_model_info("performance_test")
        assert model is not None
        assert model.performance_metrics["reasoning_score"] == 0.85
        assert model.performance_metrics["generation_speed"] == 50.0

    def test_get_system_info(self):
        """Test getting system information"""
        manager = ModelManager()

        info = manager.get_system_info()

        assert "total_models" in info
        assert "available_families" in info
        assert "preferred_models" in info
        assert "fallback_chain" in info

        assert info["total_models"] > 0
        assert isinstance(info["available_families"], list)
        assert isinstance(info["preferred_models"], list)


if __name__ == "__main__":
    pytest.main([__file__])
