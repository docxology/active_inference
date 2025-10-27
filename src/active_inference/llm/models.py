"""
LLM Model Management

Model management system for handling different LLM models, their capabilities,
and providing intelligent fallback strategies for the Active Inference platform.

This module provides:
- Model registry and metadata management
- Capability assessment and matching
- Fallback strategies for unavailable models
- Model performance benchmarking
- Integration with Ollama service
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any, Set
import logging


@dataclass
class ModelInfo:
    """Information about a specific model"""

    name: str
    family: str  # gemma, llama, mistral, etc.
    size: str    # 2b, 7b, 13b, etc.
    quantization: Optional[str] = None
    context_window: int = 4096
    capabilities: Set[str] = field(default_factory=set)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    requirements: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def has_capability(self, capability: str) -> bool:
        """Check if model has a specific capability"""
        return capability in self.capabilities

    def add_capability(self, capability: str) -> None:
        """Add a capability to the model"""
        self.capabilities.add(capability)

    def get_memory_requirement(self) -> float:
        """Estimate memory requirement in GB"""
        # Rough estimation based on model size
        size_num = float(self.size.rstrip('b'))
        base_memory = size_num * 2  # Approximate GB needed

        if self.quantization:
            if '2bit' in self.quantization.lower():
                base_memory *= 0.25
            elif '4bit' in self.quantization.lower():
                base_memory *= 0.5
            elif '8bit' in self.quantization.lower():
                base_memory *= 0.75

        return base_memory


@dataclass
class ModelRegistry:
    """Registry of available models and their configurations"""

    models: Dict[str, ModelInfo] = field(default_factory=dict)
    preferred_models: List[str] = field(default_factory=list)
    fallback_chain: List[str] = field(default_factory=list)

    def register_model(self, model_info: ModelInfo) -> None:
        """Register a model in the registry"""
        self.models[model_info.name] = model_info

        # Update fallback chain if not already present
        if model_info.name not in self.fallback_chain:
            self.fallback_chain.append(model_info.name)

    def get_model(self, name: str) -> Optional[ModelInfo]:
        """Get model information by name"""
        return self.models.get(name)

    def find_models_by_capability(self, capability: str) -> List[ModelInfo]:
        """Find models that have a specific capability"""
        return [model for model in self.models.values() if model.has_capability(capability)]

    def find_models_by_family(self, family: str) -> List[ModelInfo]:
        """Find models from a specific family"""
        return [model for model in self.models.values() if model.family == family]

    def get_best_model_for_task(self, task_capabilities: Set[str], max_memory: float = None) -> Optional[ModelInfo]:
        """Get the best model for a specific task based on capabilities and constraints"""
        suitable_models = []

        for model in self.models.values():
            # Check if model has all required capabilities
            if task_capabilities.issubset(model.capabilities):
                # Check memory constraint if specified
                if max_memory is None or model.get_memory_requirement() <= max_memory:
                    suitable_models.append(model)

        if not suitable_models:
            return None

        # Sort by preference order, then by size (smaller first for efficiency)
        suitable_models.sort(key=lambda m: (
            self.preferred_models.index(m.name) if m.name in self.preferred_models else len(self.preferred_models),
            float(m.size.rstrip('b'))
        ))

        return suitable_models[0]

    def set_preferred_order(self, model_names: List[str]) -> None:
        """Set the preferred model order"""
        self.preferred_models = [name for name in model_names if name in self.models]

    def save_registry(self, file_path: Path) -> None:
        """Save registry to file"""
        data = {
            "models": {
                name: {
                    "name": info.name,
                    "family": info.family,
                    "size": info.size,
                    "quantization": info.quantization,
                    "context_window": info.context_window,
                    "capabilities": list(info.capabilities),
                    "performance_metrics": info.performance_metrics,
                    "requirements": info.requirements,
                    "metadata": info.metadata
                }
                for name, info in self.models.items()
            },
            "preferred_models": self.preferred_models,
            "fallback_chain": self.fallback_chain
        }

        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)

    def load_registry(self, file_path: Path) -> None:
        """Load registry from file"""
        with open(file_path, 'r') as f:
            data = json.load(f)

        self.models.clear()
        for name, model_data in data.get("models", {}).items():
            info = ModelInfo(
                name=model_data["name"],
                family=model_data["family"],
                size=model_data["size"],
                quantization=model_data.get("quantization"),
                context_window=model_data.get("context_window", 4096),
                capabilities=set(model_data.get("capabilities", [])),
                performance_metrics=model_data.get("performance_metrics", {}),
                requirements=model_data.get("requirements", {}),
                metadata=model_data.get("metadata", {})
            )
            self.models[name] = info

        self.preferred_models = data.get("preferred_models", [])
        self.fallback_chain = data.get("fallback_chain", list(self.models.keys()))


class ModelManager:
    """Manager for LLM models and their interactions"""

    def __init__(self, registry_path: Optional[Path] = None):
        self.registry = ModelRegistry()
        self.registry_path = registry_path or Path(__file__).parent / "model_registry.json"
        self.logger = logging.getLogger(f"active_inference.llm.{self.__class__.__name__}")

        # Load registry if it exists
        if self.registry_path.exists():
            try:
                self.registry.load_registry(self.registry_path)
            except Exception as e:
                self.logger.warning(f"Failed to load model registry: {e}")

        # Initialize with default models
        self._initialize_default_models()

    def _initialize_default_models(self) -> None:
        """Initialize with default Active Inference suitable models"""

        # Gemma 3 models
        gemma3_2b = ModelInfo(
            name="gemma3:2b",
            family="gemma",
            size="2b",
            context_window=8192,
            capabilities={
                "text_generation", "conversation", "reasoning",
                "mathematical_reasoning", "code_generation", "active_inference_explanation"
            }
        )

        gemma3_4b = ModelInfo(
            name="gemma3:4b",
            family="gemma",
            size="4b",
            context_window=8192,
            capabilities={
                "text_generation", "conversation", "reasoning",
                "mathematical_reasoning", "code_generation", "active_inference_explanation",
                "research_analysis", "educational_content"
            }
        )

        # Llama models
        llama2_7b = ModelInfo(
            name="llama2:7b",
            family="llama",
            size="7b",
            context_window=4096,
            capabilities={
                "text_generation", "conversation", "reasoning",
                "code_generation", "research_analysis"
            }
        )

        llama2_13b = ModelInfo(
            name="llama2:13b",
            family="llama",
            size="13b",
            context_window=4096,
            capabilities={
                "text_generation", "conversation", "reasoning",
                "mathematical_reasoning", "code_generation", "research_analysis",
                "educational_content", "complex_reasoning"
            }
        )

        # Mistral models
        mistral_7b = ModelInfo(
            name="mistral:7b",
            family="mistral",
            size="7b",
            context_window=8192,
            capabilities={
                "text_generation", "conversation", "reasoning",
                "mathematical_reasoning", "code_generation", "research_analysis",
                "educational_content"
            }
        )

        # Register all models
        for model in [gemma3_2b, gemma3_4b, llama2_7b, llama2_13b, mistral_7b]:
            self.registry.register_model(model)

        # Set preferred order: smaller, more efficient models first
        self.registry.set_preferred_order([
            "gemma3:2b", "gemma3:4b", "mistral:7b", "llama2:7b", "llama2:13b"
        ])

    def get_available_models(self) -> List[str]:
        """Get list of available model names"""
        return list(self.registry.models.keys())

    def get_model_info(self, model_name: str) -> Optional[ModelInfo]:
        """Get information about a specific model"""
        return self.registry.get_model(model_name)

    def find_best_model(
        self,
        task: str,
        max_memory_gb: float = None,
        preferred_family: str = None
    ) -> Optional[ModelInfo]:
        """Find the best model for a specific task"""

        # Determine required capabilities based on task
        task_capabilities = self._get_task_capabilities(task)

        # Filter by family if specified
        if preferred_family:
            family_models = self.registry.find_models_by_family(preferred_family)
            # Create a temporary registry with only family models
            temp_registry = ModelRegistry()
            for model in family_models:
                temp_registry.register_model(model)
            temp_registry.set_preferred_order(self.registry.preferred_models)

            best_model = temp_registry.get_best_model_for_task(task_capabilities, max_memory_gb)

            # If we found a good model in the preferred family, use it
            if best_model:
                return best_model

        # Otherwise, use the full registry
        return self.registry.get_best_model_for_task(task_capabilities, max_memory_gb)

    def _get_task_capabilities(self, task: str) -> Set[str]:
        """Determine required capabilities for a task"""

        capability_map = {
            "explanation": {"text_generation", "reasoning"},
            "mathematical": {"mathematical_reasoning", "text_generation"},
            "coding": {"code_generation", "reasoning"},
            "research": {"research_analysis", "reasoning"},
            "education": {"educational_content", "text_generation"},
            "conversation": {"conversation", "text_generation"},
            "analysis": {"research_analysis", "reasoning"},
            "active_inference": {"active_inference_explanation", "reasoning", "mathematical_reasoning"},
            "free_energy": {"mathematical_reasoning", "active_inference_explanation"},
            "variational": {"mathematical_reasoning", "research_analysis"},
            "neural_networks": {"mathematical_reasoning", "code_generation"},
            "simulation": {"mathematical_reasoning", "code_generation"},
            "policy_selection": {"reasoning", "mathematical_reasoning"},
            "belief_updating": {"mathematical_reasoning", "reasoning"}
        }

        # Find matching task capabilities
        for task_keyword, capabilities in capability_map.items():
            if task_keyword.lower() in task.lower():
                return capabilities

        # Default capabilities for unknown tasks
        return {"text_generation", "reasoning"}

    def get_models_by_capability(self, capability: str) -> List[ModelInfo]:
        """Get all models that support a specific capability"""
        return self.registry.find_models_by_capability(capability)

    def update_model_performance(self, model_name: str, metrics: Dict[str, float]) -> None:
        """Update performance metrics for a model"""
        model = self.registry.get_model(model_name)
        if model:
            model.performance_metrics.update(metrics)
            self.save_registry()

    def add_custom_model(
        self,
        name: str,
        family: str,
        size: str,
        capabilities: List[str],
        **kwargs
    ) -> ModelInfo:
        """Add a custom model to the registry"""

        model_info = ModelInfo(
            name=name,
            family=family,
            size=size,
            capabilities=set(capabilities),
            **kwargs
        )

        self.registry.register_model(model_info)
        self.save_registry()

        return model_info

    def remove_model(self, model_name: str) -> bool:
        """Remove a model from the registry"""
        if model_name in self.registry.models:
            del self.registry.models[model_name]

            # Update fallback chain
            if model_name in self.registry.fallback_chain:
                self.registry.fallback_chain.remove(model_name)

            # Update preferred models
            if model_name in self.registry.preferred_models:
                self.registry.preferred_models.remove(model_name)

            self.save_registry()
            return True

        return False

    def save_registry(self) -> None:
        """Save the current registry to file"""
        try:
            self.registry.save_registry(self.registry_path)
        except Exception as e:
            self.logger.error(f"Failed to save model registry: {e}")

    def get_system_info(self) -> Dict[str, Any]:
        """Get system information for model recommendations"""
        return {
            "total_models": len(self.registry.models),
            "available_families": list(set(model.family for model in self.registry.models.values())),
            "preferred_models": self.registry.preferred_models,
            "fallback_chain": self.registry.fallback_chain
        }

    def benchmark_model(self, model_name: str, test_tasks: List[str]) -> Dict[str, float]:
        """Benchmark a model on test tasks (placeholder for actual implementation)"""
        # This would integrate with actual benchmarking
        model = self.registry.get_model(model_name)
        if not model:
            return {}

        # Placeholder metrics - in real implementation, this would run actual benchmarks
        metrics = {
            "reasoning_score": 0.8 if model.has_capability("reasoning") else 0.5,
            "generation_speed": 100.0 / float(model.size.rstrip('b')),  # tokens per second estimate
            "memory_efficiency": 1.0 / model.get_memory_requirement(),
            "context_utilization": min(1.0, model.context_window / 8192)
        }

        # Update model performance metrics
        self.update_model_performance(model_name, metrics)

        return metrics
