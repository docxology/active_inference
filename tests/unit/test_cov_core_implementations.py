"""Coverage-focused behaviour tests for ``knowledge.implementations``."""

from pathlib import Path

import pytest

from active_inference.knowledge.implementations import Implementations
from active_inference.knowledge.repository import (
    ContentType,
    DifficultyLevel,
    KnowledgeRepository,
    KnowledgeRepositoryConfig,
)

EXPECTED_IDS = {
    "active_inference_basic",
    "expected_free_energy_calculation",
    "mcmc_sampling",
    "neural_network_implementation",
    "variational_inference",
    "tutorial_grid_world",
}


@pytest.fixture
def repo(tmp_path: Path) -> KnowledgeRepository:
    return KnowledgeRepository(KnowledgeRepositoryConfig(root_path=tmp_path))


@pytest.fixture
def impl(repo: KnowledgeRepository) -> Implementations:
    return Implementations(repo)


def test_setup_registers_implementation_nodes(
    repo: KnowledgeRepository, impl: Implementations
) -> None:
    assert set(repo._nodes) == EXPECTED_IDS
    assert all(
        n.content_type is ContentType.IMPLEMENTATION for n in repo._nodes.values()
    )
    assert repo._index["tags"]["tutorial"] == [
        "active_inference_basic",
        "tutorial_grid_world",
    ]


def test_second_instance_tolerates_existing_nodes(
    repo: KnowledgeRepository, impl: Implementations
) -> None:
    before = dict(repo._nodes)
    again = Implementations(repo)
    assert again.repository is repo
    assert repo._nodes == before


def test_get_implementation_examples_filters(impl: Implementations) -> None:
    all_ids = {n.id for n in impl.get_implementation_examples()}
    assert all_ids == {
        "active_inference_basic",
        "expected_free_energy_calculation",
        "neural_network_implementation",
        "variational_inference",
        "tutorial_grid_world",
    }
    expert = impl.get_implementation_examples(difficulty=DifficultyLevel.EXPERT)
    assert [n.id for n in expert] == ["neural_network_implementation"]
    tagged = impl.get_implementation_examples(tags=["python"])
    assert [n.id for n in tagged] == ["active_inference_basic"]


def test_get_tutorial_path_sorted_by_difficulty(
    repo: KnowledgeRepository, impl: Implementations
) -> None:
    repo.add_knowledge_node(
        {
            "id": "tutorial_expert",
            "title": "Expert tutorial",
            "content_type": "implementation",
            "difficulty": "expert",
            "description": "hard",
        }
    )
    repo.add_knowledge_node(
        {
            "id": "tutorial_mid",
            "title": "Intermediate tutorial",
            "content_type": "implementation",
            "difficulty": "intermediate",
            "description": "medium",
        }
    )
    assert [n.id for n in impl.get_tutorial_path()] == [
        "tutorial_grid_world",
        "tutorial_mid",
        "tutorial_expert",
    ]


class TestValidateImplementation:
    def test_missing_node(self, impl: Implementations) -> None:
        assert impl.validate_implementation("nope") == {
            "valid": False,
            "error": "Implementation not found",
        }

    def test_complete_node(self, impl: Implementations) -> None:
        # The basic implementation has sections, code and prerequisites.
        assert impl.validate_implementation("active_inference_basic") == {
            "valid": True,
            "issues": [],
            "suggestions": [],
        }

    def test_node_without_code_gets_suggestion(self, impl: Implementations) -> None:
        result = impl.validate_implementation("tutorial_grid_world")
        assert result["valid"] is True
        assert result["suggestions"] == ["Add code examples to implementation"]

    def test_empty_node(self, repo: KnowledgeRepository, impl: Implementations) -> None:
        repo.add_knowledge_node(
            {
                "id": "bare",
                "title": "Bare",
                "content_type": "implementation",
                "difficulty": "beginner",
                "description": "nothing",
            }
        )
        assert impl.validate_implementation("bare") == {
            "valid": False,
            "issues": ["Missing implementation sections"],
            "suggestions": [
                "Add code examples to implementation",
                "Add prerequisite knowledge nodes",
            ],
        }
