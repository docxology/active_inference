"""
Application Framework - Best Practices and Architecture Patterns

Guidelines and architectural patterns for building robust Active Inference
applications. Provides design principles, coding standards, and proven
patterns for scalable and maintainable implementations.
"""

import logging
from typing import Dict, List, Optional, Any
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ArchitecturePattern(Enum):
    """Common architecture patterns"""
    MODEL_VIEW_CONTROLLER = "mvc"
    ACTIVE_INFERENCE_AGENT = "active_inference_agent"
    HIERARCHICAL_CONTROL = "hierarchical_control"
    MULTI_SCALE_MODELING = "multi_scale_modeling"
    DISTRIBUTED_SYSTEM = "distributed_system"


@dataclass
class BestPractice:
    """Represents a best practice guideline"""
    id: str
    category: str
    title: str
    description: str
    examples: List[str]
    rationale: str
    related_patterns: List[str]


class ArchitecturePatterns:
    """Collection of architecture patterns"""

    def __init__(self):
        self.patterns: Dict[str, Dict[str, Any]] = {}

        self._initialize_patterns()
        logger.info("ArchitecturePatterns initialized")

    def _initialize_patterns(self) -> None:
        """Initialize common architecture patterns"""
        self.patterns = {
            "active_inference_agent": {
                "name": "Active Inference Agent",
                "description": "Basic agent architecture using Active Inference",
                "components": ["GenerativeModel", "InferenceEngine", "ActionSelection", "LearningModule"],
                "data_flow": ["Perception", "Inference", "Planning", "Action"],
                "scalability": "Single agent, local computation"
            },

            "hierarchical_control": {
                "name": "Hierarchical Control",
                "description": "Multi-level control hierarchy with Active Inference",
                "components": ["HighLevelController", "LowLevelController", "CoordinationLayer"],
                "data_flow": ["Goals", "Subgoals", "Actions", "Feedback"],
                "scalability": "Multiple levels, distributed control"
            },

            "multi_scale_modeling": {
                "name": "Multi-Scale Modeling",
                "description": "Models operating at different temporal and spatial scales",
                "components": ["FastDynamics", "SlowDynamics", "CouplingMechanism"],
                "data_flow": ["MicroScale", "MesoScale", "MacroScale"],
                "scalability": "Complex systems, large scale"
            }
        }

    def get_pattern(self, pattern_name: str) -> Optional[Dict[str, Any]]:
        """Get details of a specific pattern"""
        return self.patterns.get(pattern_name)

    def list_patterns(self) -> List[Dict[str, Any]]:
        """List all available patterns"""
        return [
            {"name": name, "details": details}
            for name, details in self.patterns.items()
        ]


class BestPracticesGuide:
    """Guide for best practices in Active Inference development"""

    def __init__(self):
        self.practices: Dict[str, List[BestPractice]] = {}

        self._initialize_best_practices()
        logger.info("BestPracticesGuide initialized")

    def _initialize_best_practices(self) -> None:
        """Initialize best practices"""
        self.practices = {
            "model_design": [
                BestPractice(
                    id="generative_model_clarity",
                    category="model_design",
                    title="Clear Generative Model Specification",
                    description="Always explicitly specify the generative model including likelihood (A), transitions (B), preferences (C), and priors (D)",
                    examples=[
                        "Use descriptive variable names for model parameters",
                        "Document the meaning of each parameter",
                        "Validate parameter dimensions match expectations"
                    ],
                    rationale="Clear specifications enable better understanding, debugging, and reproducibility",
                    related_patterns=["active_inference_agent"]
                ),

                BestPractice(
                    id="parameter_validation",
                    category="model_design",
                    title="Parameter Validation",
                    description="Validate all model parameters for correct dimensions, ranges, and probabilistic constraints",
                    examples=[
                        "Check that transition matrices are stochastic",
                        "Ensure prior preferences sum appropriately",
                        "Validate observation likelihood matrices"
                    ],
                    rationale="Invalid parameters lead to numerical instability and incorrect inference",
                    related_patterns=["active_inference_agent", "hierarchical_control"]
                )
            ],

            "implementation": [
                BestPractice(
                    id="numerical_stability",
                    category="implementation",
                    title="Numerical Stability",
                    description="Implement numerical safeguards to prevent overflow, underflow, and division by zero",
                    examples=[
                        "Add small constants to prevent log(0)",
                        "Normalize probabilities after updates",
                        "Use stable matrix operations"
                    ],
                    rationale="Active Inference involves many probabilistic computations that can become numerically unstable",
                    related_patterns=["all"]
                ),

                BestPractice(
                    id="modular_design",
                    category="implementation",
                    title="Modular Architecture",
                    description="Structure code into clear modules with well-defined interfaces and responsibilities",
                    examples=[
                        "Separate inference from action selection",
                        "Use configuration objects for parameters",
                        "Implement clear error handling"
                    ],
                    rationale="Modular design improves maintainability and allows for easier testing and extension",
                    related_patterns=["all"]
                )
            ],

            "testing": [
                BestPractice(
                    id="unit_testing",
                    category="testing",
                    title="Comprehensive Unit Testing",
                    description="Write tests for individual components and functions",
                    examples=[
                        "Test inference functions with known inputs",
                        "Validate model updates produce expected changes",
                        "Test edge cases and error conditions"
                    ],
                    rationale="Thorough testing ensures reliability and helps catch bugs early",
                    related_patterns=["all"]
                ),

                BestPractice(
                    id="integration_testing",
                    category="testing",
                    title="Integration Testing",
                    description="Test how components work together in realistic scenarios",
                    examples=[
                        "Test complete perception-action cycles",
                        "Verify end-to-end functionality",
                        "Test with various parameter configurations"
                    ],
                    rationale="Integration tests ensure the system works as a whole",
                    related_patterns=["active_inference_agent", "hierarchical_control"]
                )
            ]
        }

    def get_practices(self, category: Optional[str] = None) -> List[BestPractice]:
        """Get best practices, optionally filtered by category"""
        if category:
            return self.practices.get(category, [])

        # Return all practices
        all_practices = []
        for practices_list in self.practices.values():
            all_practices.extend(practices_list)

        return all_practices

    def get_practice(self, practice_id: str) -> Optional[BestPractice]:
        """Get a specific best practice"""
        for practices_list in self.practices.values():
            for practice in practices_list:
                if practice.id == practice_id:
                    return practice
        return None

    def generate_architecture_recommendations(self, requirements: Dict[str, Any]) -> List[str]:
        """Generate architecture recommendations based on requirements"""
        recommendations = []

        # Analyze requirements and suggest patterns
        if requirements.get("multi_level"):
            recommendations.append("hierarchical_control")
        if requirements.get("multi_scale"):
            recommendations.append("multi_scale_modeling")
        if requirements.get("distributed"):
            recommendations.append("distributed_system")

        # Add general recommendations
        recommendations.extend([
            "active_inference_agent",  # Always applicable
            "numerical_stability",
            "modular_design"
        ])

        return recommendations




