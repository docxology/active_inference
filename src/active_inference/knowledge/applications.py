"""
Applications Module

Real-world applications and domain-specific implementations of Active Inference
and the Free Energy Principle. Provides comprehensive case studies, domain
knowledge, and practical applications across various research areas.
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from .repository import KnowledgeRepository, KnowledgeNode, ContentType, DifficultyLevel


class Applications:
    """
    Real-world applications of Active Inference across domains

    Provides structured access to domain-specific applications including:
    - Artificial Intelligence and Machine Learning applications
    - Neuroscience and cognitive science applications
    - Engineering and control systems applications
    - Psychology and behavioral applications
    - Educational technology applications
    """

    def __init__(self, repository: KnowledgeRepository):
        self.repository = repository
        self._setup_applications()

    def _setup_applications(self) -> None:
        """Initialize application knowledge nodes"""
        self._create_ai_applications()
        self._create_neuroscience_applications()
        self._create_engineering_applications()
        self._create_psychology_applications()
        self._create_education_applications()

    def _create_ai_applications(self) -> None:
        """Create AI and machine learning application nodes"""
        nodes_data = [
            {
                "id": "ai_generative_models",
                "title": "Active Inference in Generative Models",
                "content_type": ContentType.APPLICATION,
                "difficulty": DifficultyLevel.ADVANCED,
                "description": "Application of Active Inference to generative AI models",
                "prerequisites": ["generative_models", "variational_inference"],
                "tags": ["artificial intelligence", "generative models", "machine learning"],
                "learning_objectives": [
                    "Apply Active Inference to generative model training",
                    "Understand free energy in generative AI",
                    "Implement Active Inference-based generative systems"
                ],
                "content": {
                    "sections": [
                        {
                            "title": "Generative Models and Free Energy",
                            "content": "Generative models learn to represent data distributions..."
                        },
                        {
                            "title": "Active Inference Framework",
                            "content": "Active Inference provides a principled approach..."
                        }
                    ]
                },
                "metadata": {
                    "estimated_reading_time": 35,
                    "author": "Active Inference Community",
                    "last_updated": "2024-10-27",
                    "version": "1.0",
                    "domain": "artificial_intelligence",
                    "application_type": "generative_modeling"
                }
            },
            {
                "id": "ai_policy_selection",
                "title": "Active Inference for Policy Selection in AI",
                "content_type": ContentType.APPLICATION,
                "difficulty": DifficultyLevel.ADVANCED,
                "description": "Using Active Inference for optimal policy selection in AI systems",
                "prerequisites": ["decision_theory", "expected_free_energy"],
                "tags": ["policy selection", "decision making", "reinforcement learning"],
                "learning_objectives": [
                    "Implement Active Inference for policy evaluation",
                    "Compare with reinforcement learning approaches",
                    "Apply to complex decision-making tasks"
                ],
                "content": {
                    "sections": [
                        {
                            "title": "Policy Selection Problem",
                            "content": "AI systems must select optimal policies..."
                        }
                    ]
                },
                "metadata": {
                    "estimated_reading_time": 40,
                    "author": "Active Inference Community",
                    "last_updated": "2024-10-27",
                    "version": "1.0"
                }
            }
        ]

        # Add nodes to repository
        for node_data in nodes_data:
            try:
                self.repository.add_knowledge_node(node_data)
            except Exception as e:
                # Node might already exist, that's okay
                pass

    def _create_neuroscience_applications(self) -> None:
        """Create neuroscience application nodes"""
        nodes_data = [
            {
                "id": "neuroscience_perception",
                "title": "Active Inference in Neural Perception",
                "content_type": ContentType.APPLICATION,
                "difficulty": DifficultyLevel.ADVANCED,
                "description": "Application of Active Inference to neural perception mechanisms",
                "prerequisites": ["neural_dynamics", "predictive_coding"],
                "tags": ["neuroscience", "perception", "predictive coding"],
                "learning_objectives": [
                    "Understand neural perception through Active Inference",
                    "Model perceptual inference in neural systems",
                    "Connect to empirical neuroscience findings"
                ],
                "content": {
                    "sections": [
                        {
                            "title": "Neural Perception Framework",
                            "content": "Neural systems must infer causes from sensory data..."
                        }
                    ]
                },
                "metadata": {
                    "estimated_reading_time": 45,
                    "author": "Active Inference Community",
                    "last_updated": "2024-10-27",
                    "version": "1.0",
                    "domain": "neuroscience"
                }
            }
        ]

        # Add nodes to repository
        for node_data in nodes_data:
            try:
                self.repository.add_knowledge_node(node_data)
            except Exception as e:
                # Node might already exist, that's okay
                pass

    def _create_engineering_applications(self) -> None:
        """Create engineering application nodes"""
        nodes_data = [
            {
                "id": "engineering_control_systems",
                "title": "Active Inference in Control Systems",
                "content_type": ContentType.APPLICATION,
                "difficulty": DifficultyLevel.INTERMEDIATE,
                "description": "Application of Active Inference to engineering control systems",
                "prerequisites": ["control_theory", "continuous_control"],
                "tags": ["engineering", "control systems", "cybernetics"],
                "learning_objectives": [
                    "Design control systems using Active Inference",
                    "Implement adaptive control mechanisms",
                    "Optimize system performance through free energy minimization"
                ],
                "content": {
                    "sections": [
                        {
                            "title": "Control Systems Perspective",
                            "content": "Control systems maintain desired states..."
                        }
                    ]
                },
                "metadata": {
                    "estimated_reading_time": 50,
                    "author": "Active Inference Community",
                    "last_updated": "2024-10-27",
                    "version": "1.0",
                    "domain": "engineering"
                }
            },
            {
                "id": "robotics_control",
                "title": "Active Inference in Robotics Control",
                "content_type": ContentType.APPLICATION,
                "difficulty": DifficultyLevel.ADVANCED,
                "description": "Application of Active Inference to autonomous robotics control",
                "prerequisites": ["robotics", "control_systems"],
                "tags": ["robotics", "autonomous systems", "control"],
                "learning_objectives": [
                    "Design robotic systems using Active Inference",
                    "Implement autonomous behavior generation",
                    "Handle uncertainty in robotic environments"
                ],
                "content": {
                    "sections": [
                        {
                            "title": "Robotics and Active Inference",
                            "content": "Robots must operate in uncertain environments..."
                        }
                    ]
                },
                "metadata": {
                    "estimated_reading_time": 55,
                    "author": "Active Inference Community",
                    "last_updated": "2024-10-27",
                    "version": "1.0"
                }
            }
        ]

        # Add nodes to repository
        for node_data in nodes_data:
            try:
                self.repository.add_knowledge_node(node_data)
            except Exception as e:
                # Node might already exist, that's okay
                pass

    def _create_psychology_applications(self) -> None:
        """Create psychology application nodes"""
        nodes_data = [
            {
                "id": "psychology_decision_making",
                "title": "Active Inference in Decision Making",
                "content_type": ContentType.APPLICATION,
                "difficulty": DifficultyLevel.INTERMEDIATE,
                "description": "Application of Active Inference to human decision making",
                "prerequisites": ["decision_theory", "cognitive_psychology"],
                "tags": ["psychology", "decision making", "behavior"],
                "learning_objectives": [
                    "Model human decision making with Active Inference",
                    "Understand cognitive biases through free energy",
                    "Predict behavioral patterns"
                ],
                "content": {
                    "sections": [
                        {
                            "title": "Decision Making Framework",
                            "content": "Human decision making involves complex cognitive processes..."
                        }
                    ]
                },
                "metadata": {
                    "estimated_reading_time": 40,
                    "author": "Active Inference Community",
                    "last_updated": "2024-10-27",
                    "version": "1.0",
                    "domain": "psychology"
                }
            }
        ]

        # Add nodes to repository
        for node_data in nodes_data:
            try:
                self.repository.add_knowledge_node(node_data)
            except Exception as e:
                # Node might already exist, that's okay
                pass

    def _create_education_applications(self) -> None:
        """Create education application nodes"""
        nodes_data = [
            {
                "id": "education_adaptive_learning",
                "title": "Active Inference in Adaptive Learning",
                "content_type": ContentType.APPLICATION,
                "difficulty": DifficultyLevel.INTERMEDIATE,
                "description": "Application of Active Inference to adaptive educational systems",
                "prerequisites": ["learning_theory", "adaptive_systems"],
                "tags": ["education", "adaptive learning", "personalization"],
                "learning_objectives": [
                    "Design adaptive learning systems with Active Inference",
                    "Personalize educational content",
                    "Optimize learning outcomes through free energy minimization"
                ],
                "content": {
                    "sections": [
                        {
                            "title": "Adaptive Learning Systems",
                            "content": "Adaptive learning systems adjust to individual learner needs..."
                        }
                    ]
                },
                "metadata": {
                    "estimated_reading_time": 45,
                    "author": "Active Inference Community",
                    "last_updated": "2024-10-27",
                    "version": "1.0",
                    "domain": "education"
                }
            }
        ]

        # Add nodes to repository
        for node_data in nodes_data:
            try:
                self.repository.add_knowledge_node(node_data)
            except Exception as e:
                # Node might already exist, that's okay
                pass

    def get_applications_by_domain(self, domain: str) -> List[KnowledgeNode]:
        """Get applications filtered by domain"""
        return self.repository.search(f"application {domain}",
                                    content_types=[ContentType.APPLICATION])

    def get_domain_applications(self) -> Dict[str, List[KnowledgeNode]]:
        """Get applications organized by domain"""
        domains = ["artificial_intelligence", "neuroscience", "engineering",
                  "psychology", "education", "climate_science", "economics"]

        domain_applications = {}
        for domain in domains:
            applications = self.get_applications_by_domain(domain)
            if applications:
                domain_applications[domain] = applications

        return domain_applications

    def get_case_studies(self) -> List[KnowledgeNode]:
        """Get comprehensive case studies"""
        return self.repository.search("case study",
                                    content_types=[ContentType.APPLICATION])

    def validate_application(self, node_id: str) -> Dict[str, Any]:
        """Validate application for completeness and applicability"""
        node = self.repository.get_node(node_id)
        if not node:
            return {"valid": False, "error": "Application not found"}

        validation_result = {
            "valid": True,
            "issues": [],
            "suggestions": []
        }

        # Check domain specification
        if not node.metadata.get("domain"):
            validation_result["issues"].append("Missing domain specification")
            validation_result["valid"] = False

        # Check for practical examples
        has_examples = any("example" in section.get("content", "").lower()
                          for section in node.content.get("sections", []))
        if not has_examples:
            validation_result["suggestions"].append("Add practical examples to application")

        # Check for implementation guidance
        has_implementation = any("implement" in section.get("content", "").lower()
                               for section in node.content.get("sections", []))
        if not has_implementation:
            validation_result["suggestions"].append("Add implementation guidance")

        return validation_result
