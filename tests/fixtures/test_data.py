"""
Test Data Fixtures

Comprehensive test data for the Active Inference Knowledge Environment testing.
Provides realistic test data, knowledge nodes, learning paths, and research scenarios.
"""

import json
from pathlib import Path
from typing import Dict, List, Any

# Sample knowledge nodes for testing
SAMPLE_KNOWLEDGE_NODES = [
    {
        "id": "entropy_basics",
        "title": "Entropy Basics",
        "content_type": "foundation",
        "difficulty": "beginner",
        "description": "Introduction to entropy in information theory",
        "prerequisites": [],
        "tags": ["entropy", "information_theory", "beginner"],
        "learning_objectives": [
            "Understand what entropy measures",
            "Calculate entropy for simple distributions",
            "Connect entropy to uncertainty"
        ],
        "content": {
            "overview": "Entropy is a measure of uncertainty or disorder in a probability distribution.",
            "mathematical_definition": "H(X) = -Σ p(x) log p(x)",
            "examples": [
                {
                    "name": "Coin Flip",
                    "description": "Entropy of a fair coin is 1 bit",
                    "calculation": "H = -0.5*log2(0.5) - 0.5*log2(0.5) = 1"
                }
            ]
        },
        "metadata": {
            "author": "Test Author",
            "version": "1.0",
            "estimated_reading_time": 10
        }
    },
    {
        "id": "bayesian_inference",
        "title": "Bayesian Inference",
        "content_type": "foundation",
        "difficulty": "intermediate",
        "description": "Bayesian approach to statistical inference",
        "prerequisites": ["probability_basics"],
        "tags": ["bayesian", "inference", "probability"],
        "learning_objectives": [
            "Apply Bayes' theorem",
            "Update beliefs with evidence",
            "Understand posterior distributions"
        ],
        "content": {
            "overview": "Bayesian inference treats parameters as random variables with probability distributions.",
            "mathematical_formulation": "p(θ|x) = p(x|θ) p(θ) / p(x)",
            "examples": [
                {
                    "name": "Medical Diagnosis",
                    "description": "Updating disease probability based on test results"
                }
            ]
        },
        "metadata": {
            "author": "Test Author",
            "version": "1.0",
            "estimated_reading_time": 15
        }
    }
]

# Sample learning paths
SAMPLE_LEARNING_PATHS = [
    {
        "id": "information_theory_fundamentals",
        "name": "Information Theory Fundamentals",
        "description": "Complete introduction to information theory concepts",
        "nodes": ["entropy_basics", "mutual_information", "kl_divergence"],
        "estimated_hours": 8,
        "difficulty": "beginner"
    },
    {
        "id": "bayesian_methods",
        "name": "Bayesian Methods",
        "description": "Comprehensive Bayesian inference and modeling",
        "nodes": ["probability_basics", "bayesian_inference", "bayesian_networks"],
        "estimated_hours": 12,
        "difficulty": "intermediate"
    }
]

# Sample research scenarios
SAMPLE_RESEARCH_SCENARIOS = [
    {
        "id": "active_inference_simulation",
        "name": "Active Inference Simulation Study",
        "description": "Simulation study of Active Inference in decision making",
        "hypothesis": "Active Inference agents will show better decision making under uncertainty",
        "methodology": "computational_simulation",
        "parameters": {
            "n_agents": 100,
            "n_trials": 1000,
            "uncertainty_levels": [0.1, 0.3, 0.5, 0.7, 0.9]
        },
        "expected_outcomes": [
            "Lower decision error rates",
            "Better uncertainty handling",
            "More adaptive behavior"
        ]
    }
]

def create_test_knowledge_directory(base_path: Path) -> Path:
    """Create a test knowledge directory with sample content"""
    knowledge_dir = base_path / "test_knowledge"
    knowledge_dir.mkdir(exist_ok=True)

    # Create metadata directory
    metadata_dir = knowledge_dir / "metadata"
    metadata_dir.mkdir(exist_ok=True)

    # Create repository.json
    repo_data = {
        "version": "1.0",
        "description": "Test knowledge repository",
        "total_nodes": len(SAMPLE_KNOWLEDGE_NODES)
    }
    (metadata_dir / "repository.json").write_text(json.dumps(repo_data, indent=2))

    # Create learning paths
    (metadata_dir / "learning_paths.json").write_text(
        json.dumps(SAMPLE_LEARNING_PATHS, indent=2)
    )

    # Create content directories and files
    foundations_dir = knowledge_dir / "foundations"
    foundations_dir.mkdir(exist_ok=True)

    for node in SAMPLE_KNOWLEDGE_NODES:
        content_type = node["content_type"]
        content_dir = knowledge_dir / content_type
        content_dir.mkdir(exist_ok=True)

        node_file = content_dir / f"{node['id']}.json"
        node_file.write_text(json.dumps(node, indent=2))

    return knowledge_dir

def get_sample_knowledge_node(node_id: str) -> Dict[str, Any]:
    """Get a sample knowledge node by ID"""
    for node in SAMPLE_KNOWLEDGE_NODES:
        if node["id"] == node_id:
            return node
    return None

def get_sample_learning_path(path_id: str) -> Dict[str, Any]:
    """Get a sample learning path by ID"""
    for path in SAMPLE_LEARNING_PATHS:
        if path["id"] == path_id:
            return path
    return None
