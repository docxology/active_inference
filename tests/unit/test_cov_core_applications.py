"""Coverage-focused behaviour tests for ``knowledge.applications``."""

from pathlib import Path

import pytest

from active_inference.knowledge.applications import Applications
from active_inference.knowledge.repository import (
    ContentType,
    KnowledgeRepository,
    KnowledgeRepositoryConfig,
)

EXPECTED_IDS = {
    "ai_generative_models",
    "ai_policy_selection",
    "neuroscience_perception",
    "engineering_control_systems",
    "robotics_control",
    "psychology_decision_making",
    "education_adaptive_learning",
}


@pytest.fixture
def repo(tmp_path: Path) -> KnowledgeRepository:
    return KnowledgeRepository(KnowledgeRepositoryConfig(root_path=tmp_path))


@pytest.fixture
def apps(repo: KnowledgeRepository) -> Applications:
    return Applications(repo)


def _add(repo: KnowledgeRepository, node_id: str, **extra: object) -> None:
    data = {
        "id": node_id,
        "title": f"Title {node_id}",
        "content_type": "application",
        "difficulty": "beginner",
        "description": f"Description {node_id}",
    }
    data.update(extra)
    repo.add_knowledge_node(data)


def test_setup_registers_application_nodes(
    repo: KnowledgeRepository, apps: Applications
) -> None:
    assert set(repo._nodes) == EXPECTED_IDS
    assert all(n.content_type is ContentType.APPLICATION for n in repo._nodes.values())


def test_second_instance_tolerates_existing_nodes(
    repo: KnowledgeRepository, apps: Applications
) -> None:
    before = dict(repo._nodes)
    Applications(repo)
    assert repo._nodes == before


def test_domain_lookup_and_grouping(
    repo: KnowledgeRepository, apps: Applications
) -> None:
    # Built-in titles never contain "application <domain>", so nothing groups.
    assert apps.get_domain_applications() == {}
    _add(repo, "neuro_case", title="Clinical application neuroscience case study")
    _add(repo, "econ", description="An application economics example")
    assert [n.id for n in apps.get_applications_by_domain("neuroscience")] == [
        "neuro_case"
    ]
    grouped = apps.get_domain_applications()
    assert set(grouped) == {"neuroscience", "economics"}
    assert [n.id for n in grouped["economics"]] == ["econ"]
    assert [n.id for n in apps.get_case_studies()] == ["neuro_case"]


class TestValidateApplication:
    def test_missing(self, apps: Applications) -> None:
        assert apps.validate_application("nope") == {
            "valid": False,
            "error": "Application not found",
        }

    def test_builtin_node_with_domain(self, apps: Applications) -> None:
        result = apps.validate_application("neuroscience_perception")
        assert result["valid"] is True
        assert result["issues"] == []
        assert result["suggestions"] == [
            "Add practical examples to application",
            "Add implementation guidance",
        ]

    def test_builtin_node_without_domain(self, apps: Applications) -> None:
        result = apps.validate_application("robotics_control")
        assert result["valid"] is False
        assert result["issues"] == ["Missing domain specification"]

    def test_complete_application(
        self, repo: KnowledgeRepository, apps: Applications
    ) -> None:
        _add(
            repo,
            "complete",
            metadata={"domain": "economics"},
            content={
                "sections": [
                    {"content": "A worked Example of market beliefs"},
                    {"content": "How to implement the agent"},
                ]
            },
        )
        assert apps.validate_application("complete") == {
            "valid": True,
            "issues": [],
            "suggestions": [],
        }
