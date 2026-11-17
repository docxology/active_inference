"""Coverage-focused behaviour tests for ``knowledge.foundations``."""

import json
from pathlib import Path

import pytest

from active_inference.knowledge.foundations import Foundations
from active_inference.knowledge.repository import (
    ContentType,
    DifficultyLevel,
    KnowledgeRepository,
    KnowledgeRepositoryConfig,
)


@pytest.fixture
def repo(tmp_path: Path) -> KnowledgeRepository:
    return KnowledgeRepository(KnowledgeRepositoryConfig(root_path=tmp_path))


@pytest.fixture
def foundations(repo: KnowledgeRepository) -> Foundations:
    return Foundations(repo)


def test_setup_creates_nodes_with_content_and_metadata(
    repo: KnowledgeRepository, foundations: Foundations
) -> None:
    node = repo.get_node("info_theory_entropy")
    assert node is not None
    assert node.content_type is ContentType.FOUNDATION
    assert node.metadata["author"] == "Active Inference Community"
    assert node.metadata["estimated_reading_time"] == 15
    assert node.content


def test_invalid_node_is_reported_not_raised(
    repo: KnowledgeRepository,
    foundations: Foundations,
    capsys: pytest.CaptureFixture[str],
) -> None:
    bad = {
        "id": "broken_node",
        "title": "Broken",
        "content_type": ContentType.FOUNDATION,
        "difficulty": "not-a-level",
        "description": "invalid difficulty",
        "prerequisites": [],
        "tags": [],
        "learning_objectives": [],
    }
    foundations._save_nodes_to_repository([bad])
    out = capsys.readouterr().out
    assert out.startswith("Warning: Failed to create foundation node broken_node:")
    assert repo.get_node("broken_node") is None


def test_valid_node_is_saved(
    repo: KnowledgeRepository, foundations: Foundations
) -> None:
    foundations._save_nodes_to_repository(
        [
            {
                "id": "extra_node",
                "title": "Extra",
                "content_type": ContentType.FOUNDATION,
                "difficulty": DifficultyLevel.BEGINNER,
                "description": "extra",
                "prerequisites": [],
                "tags": [],
                "learning_objectives": [],
            }
        ]
    )
    assert repo.get_node("extra_node").metadata["version"] == "1.0"


def test_consistency_flags_empty_tracks(
    foundations: Foundations, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        foundations, "get_foundation_tracks", lambda: {"full": ["a", "b"], "empty": []}
    )
    monkeypatch.setattr(foundations, "get_complete_foundation_path", lambda: ["a"])
    result = foundations.validate_foundation_consistency()
    assert result == {
        "valid": False,
        "issues": ["Empty track: empty"],
        "track_completeness": {"full": 2, "empty": 0},
        "prerequisite_satisfaction": {"a": True},
    }


def test_consistency_of_builtin_tracks(foundations: Foundations) -> None:
    result = foundations.validate_foundation_consistency()
    assert result["valid"] is True
    assert result["issues"] == []
    assert result["track_completeness"]["information_theory"] == 3


def test_export_graph_formats(foundations: Foundations) -> None:
    graph = json.loads(foundations.export_foundation_graph())
    assert graph["metadata"]["total_tracks"] == 4
    assert graph["metadata"]["total_nodes"] == 12
    assert {
        "source": "bayesian_basics",
        "target": "bayesian_models",
        "type": "prerequisite",
        "track": "bayesian_inference",
    } in graph["edges"]
    text = foundations.export_foundation_graph(format="text")
    assert text.startswith("{'nodes': [")
    assert "'export_format': 'text'" in text


def test_topic_lookup(foundations: Foundations) -> None:
    assert foundations.get_foundation_by_topic("BAYESIAN") == [
        "bayesian_basics",
        "bayesian_models",
    ]
    assert foundations.get_foundation_by_topic("quantum") == []


def test_summary_text(foundations: Foundations) -> None:
    lines = foundations.generate_foundation_summary().splitlines()
    assert lines[0] == "Active Inference Foundations Summary"
    assert "Total Learning Tracks: 4" in lines
    assert "Total Foundation Nodes: 12" in lines
    assert "Free Energy Principle:" in lines
    assert "  Path: bayesian_basics -> bayesian_models -> belief_updating" in lines
    assert lines[-1] == "12. Ai Policy Selection"


def test_search_ranks_node_matches_above_track_matches(
    foundations: Foundations,
) -> None:
    results = foundations.search_foundation_concepts("inference")
    # 'inference' matches the active_inference_introduction id (1.0) and the
    # bayesian_inference / active_inference track names (0.5).
    assert results[0] == {
        "id": "active_inference_introduction",
        "track": "active_inference",
        "relevance_score": 1.0,
    }
    assert {r["relevance_score"] for r in results[1:]} == {0.5}
    assert len(results) == 6
    assert foundations.search_foundation_concepts("zzz") == []


def test_dependencies(foundations: Foundations) -> None:
    assert foundations.get_foundation_dependencies("bayesian_models") == {
        "node_id": "bayesian_models",
        "track": "bayesian_inference",
        "prerequisites": ["bayesian_basics"],
        "dependents": ["belief_updating"],
        "position": 1,
        "total_in_track": 3,
    }
    assert foundations.get_foundation_dependencies("ghost") == {
        "error": "Node ghost not found in foundation tracks"
    }
