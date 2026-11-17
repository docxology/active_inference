"""Coverage-focused behaviour tests for ``knowledge.repository``."""

import json
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from active_inference.knowledge.repository import (
    ContentType,
    DifficultyLevel,
    KnowledgeNodeSchema,
    KnowledgeRepository,
    KnowledgeRepositoryConfig,
)


def _node(
    node_id: str,
    *,
    title: Optional[str] = None,
    content_type: str = "foundation",
    difficulty: str = "beginner",
    prerequisites: Optional[List[str]] = None,
    tags: Optional[List[str]] = None,
    objectives: Optional[List[str]] = None,
    content: Optional[Dict[str, Any]] = None,
    description: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        "id": node_id,
        "title": title or f"Title {node_id}",
        "content_type": content_type,
        "difficulty": difficulty,
        "description": description or f"Description of {node_id}",
        "prerequisites": prerequisites or [],
        "tags": tags if tags is not None else ["general"],
        "learning_objectives": (
            objectives if objectives is not None else [f"Learn {node_id}"]
        ),
        "content": content if content is not None else {"overview": node_id},
        "metadata": {"author": "tests"},
    }


def _write_tree(
    root: Path,
    nodes: List[Dict[str, Any]],
    paths: Optional[List[Dict[str, Any]]] = None,
    folder: str = "foundations",
) -> None:
    (root / "metadata").mkdir(exist_ok=True)
    (root / "metadata" / "repository.json").write_text(
        json.dumps({"version": "9.9", "name": "cov"})
    )
    if paths is not None:
        (root / "metadata" / "learning_paths.json").write_text(json.dumps(paths))
    content_dir = root / folder
    content_dir.mkdir(exist_ok=True)
    for node in nodes:
        (content_dir / f"{node['id']}.json").write_text(json.dumps(node))


def _repo(root: Path, **kwargs: Any) -> KnowledgeRepository:
    return KnowledgeRepository(KnowledgeRepositoryConfig(root_path=root, **kwargs))


@pytest.fixture
def chain_repo(tmp_path: Path) -> KnowledgeRepository:
    nodes = [
        _node("a", tags=["entropy", "info"]),
        _node(
            "b",
            prerequisites=["a"],
            tags=["entropy"],
            difficulty="intermediate",
            content_type="mathematics",
        ),
        _node(
            "c",
            prerequisites=["b"],
            tags=["bayes"],
            difficulty="advanced",
            content_type="implementation",
        ),
    ]
    paths = [
        {
            "id": "p_beg",
            "name": "Beginner path",
            "description": "beg",
            "nodes": ["a", "b", "c"],
            "estimated_hours": 3,
            "difficulty": "beginner",
        },
        {
            "id": "p_int",
            "name": "Intermediate path",
            "description": "int",
            "nodes": ["c", "b", "ghost"],
            "difficulty": "intermediate",
        },
        {
            "id": "p_int2",
            "name": "Second intermediate",
            "description": "int2",
            "nodes": ["a"],
            "difficulty": "intermediate",
        },
    ]
    _write_tree(tmp_path, nodes, paths)
    return _repo(tmp_path)


class TestSchemaValidation:
    def test_valid_node_has_no_errors(self) -> None:
        assert KnowledgeNodeSchema.validate_json_structure(_node("x")) == []

    def test_bad_enum_values_and_non_string_list_items(self) -> None:
        data = _node("x")
        data["content_type"] = "poetry"
        data["difficulty"] = "trivial"
        data["tags"] = ["ok", 3]
        data["prerequisites"] = [None]
        errors = KnowledgeNodeSchema.validate_json_structure(data)
        assert any(e.startswith("content_type must be one of") for e in errors)
        assert any(e.startswith("difficulty must be one of") for e in errors)
        assert "Field 'tags' must contain only strings" in errors
        assert "Field 'prerequisites' must contain only strings" in errors

    def test_missing_fields_and_wrong_types(self) -> None:
        errors = KnowledgeNodeSchema.validate_json_structure(
            {"id": 5, "title": "t", "content": []}
        )
        assert errors[0].startswith("Missing required fields")
        assert "Field 'id' must be of type str" in errors
        assert "Field 'content' must be of type dict" in errors


class TestLoading:
    def test_invalid_and_broken_files_are_skipped(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _write_tree(tmp_path, [_node("good")])
        folder = tmp_path / "foundations"
        bad_schema = _node("bad")
        bad_schema["difficulty"] = "nope"
        (folder / "bad.json").write_text(json.dumps(bad_schema))
        (folder / "broken.json").write_text("{not json")
        # Passes the JSON schema; the wrapper below makes pydantic reject it.
        pyd_bad = _node("pyd")
        (folder / "pyd.json").write_text(json.dumps(pyd_bad))

        from active_inference.knowledge import repository as repo_mod

        real_node = repo_mod.KnowledgeNode

        def picky_node(**data: Any) -> Any:
            if data["id"] == "pyd":
                # Trigger a genuine pydantic ValidationError.
                return real_node(**{**data, "title": ["not", "a", "string"]})
            return real_node(**data)

        monkeypatch.setattr(repo_mod, "KnowledgeNode", picky_node)
        repo = _repo(tmp_path)
        assert set(repo._nodes) == {"good"}
        assert repo.get_node("good").content_path == str(Path("foundations/good.json"))
        assert repo._metadata == {"version": "9.9", "name": "cov"}

    def test_custom_content_paths_and_no_auto_index(self, tmp_path: Path) -> None:
        _write_tree(tmp_path, [_node("m", content_type="mathematics")], folder="maths")
        repo = _repo(
            tmp_path,
            content_paths={ContentType.MATHEMATICS: "maths"},
            auto_index=False,
        )
        assert list(repo._nodes) == ["m"]
        assert repo._index == {}
        assert repo.content_paths == {ContentType.MATHEMATICS: "maths"}

    def test_missing_prerequisites_and_cycles_reported(self, tmp_path: Path) -> None:
        nodes = [
            _node("x", prerequisites=["y"]),
            _node("y", prerequisites=["x"]),
            _node("z", prerequisites=["absent"]),
        ]
        _write_tree(tmp_path, nodes)
        repo = _repo(tmp_path)
        report = repo.validate_repository()
        assert report["valid"] is False
        assert {"node": "z", "missing_prerequisite": "absent"} in report[
            "missing_prerequisites"
        ]
        assert set(report["circular_dependencies"]) == {"x", "y"}
        assert report["orphaned_nodes"] == ["z"]


class TestQueries:
    def test_index_contents(self, chain_repo: KnowledgeRepository) -> None:
        idx = chain_repo._index
        assert sorted(idx["tags"]["entropy"]) == ["a", "b"]
        assert idx["difficulty"]["advanced"] == ["c"]
        assert idx["content_type"]["mathematics"] == ["b"]
        assert idx["prerequisites"]["a"] == ["b"]

    def test_search_filters(self, chain_repo: KnowledgeRepository) -> None:
        ids = lambda nodes: [n.id for n in nodes]  # noqa: E731
        assert ids(chain_repo.search(content_types=[ContentType.MATHEMATICS])) == ["b"]
        assert ids(chain_repo.search(difficulty=[DifficultyLevel.ADVANCED])) == ["c"]
        assert ids(chain_repo.search(tags=["bayes"])) == ["c"]
        assert ids(chain_repo.search("title b")) == ["b"]
        assert sorted(ids(chain_repo.search("description of"))) == ["a", "b", "c"]
        assert len(chain_repo.search("description of", limit=2)) == 2

    def test_add_knowledge_node(self, chain_repo: KnowledgeRepository) -> None:
        node = chain_repo.add_knowledge_node(
            _node("d", tags=["new"], content_type="application")
        )
        assert node.content_type is ContentType.APPLICATION
        assert chain_repo._index["tags"]["new"] == ["d"]
        with pytest.raises(ValueError, match="already exists: d"):
            chain_repo.add_knowledge_node(_node("d"))

    def test_get_node_content(self, chain_repo: KnowledgeRepository) -> None:
        assert chain_repo.get_node_content("a") == {"overview": "a"}
        assert chain_repo.get_node_content("missing") is None

    def test_learning_paths_filters(self, chain_repo: KnowledgeRepository) -> None:
        ints = chain_repo.get_learning_paths(difficulty=DifficultyLevel.INTERMEDIATE)
        assert {p.id for p in ints} == {"p_int", "p_int2"}
        impl = chain_repo.get_learning_paths(content_types=[ContentType.IMPLEMENTATION])
        assert {p.id for p in impl} == {"p_beg", "p_int"}

    def test_prerequisites_graph(self, chain_repo: KnowledgeRepository) -> None:
        assert chain_repo.get_prerequisites_graph("nope") == {}
        graph = chain_repo.get_prerequisites_graph("c")
        assert [n["id"] for n in graph["nodes"]] == ["c", "b", "a"]
        assert graph["edges"] == [{"from": "b", "to": "c"}, {"from": "a", "to": "b"}]

    def test_prerequisites_graph_diamond_visits_once(self, tmp_path: Path) -> None:
        nodes = [
            _node("root"),
            _node("l", prerequisites=["root"]),
            _node("r", prerequisites=["root"]),
            _node("top", prerequisites=["l", "r"]),
        ]
        _write_tree(tmp_path, nodes)
        graph = _repo(tmp_path).get_prerequisites_graph("top")
        assert [n["id"] for n in graph["nodes"]].count("root") == 1
        assert len(graph["edges"]) == 4

    def test_validate_learning_path(self, chain_repo: KnowledgeRepository) -> None:
        assert chain_repo.validate_learning_path("none") == {
            "valid": False,
            "error": "Path not found",
        }
        ok = chain_repo.validate_learning_path("p_beg")
        assert ok == {
            "valid": True,
            "issues": [],
            "path_length": 3,
            "estimated_hours": 3,
        }
        bad = chain_repo.validate_learning_path("p_int")
        assert "Missing node: ghost" in bad["issues"]
        assert "Unsatisfied prerequisite b for c" in bad["issues"]
        assert "Unsatisfied prerequisite a for b" in bad["issues"]
        assert bad["valid"] is False


class TestGraphsAndIntegrity:
    def test_export_knowledge_graph(self, chain_repo: KnowledgeRepository) -> None:
        graph = chain_repo.export_knowledge_graph()
        assert graph["metadata"]["total_nodes"] == 3
        assert graph["metadata"]["total_paths"] == 3
        prereq = [e for e in graph["edges"] if e["type"] == "prerequisite"]
        lp = [e for e in graph["edges"] if e["type"] == "learning_path"]
        assert {"source": "a", "target": "b", "type": "prerequisite"} in prereq
        assert {
            "source": "a",
            "target": "b",
            "type": "learning_path",
            "path_id": "p_beg",
        } in lp
        text = chain_repo.export_knowledge_graph(format="graphml")
        assert isinstance(text, str)
        assert json.loads(text)["metadata"]["total_nodes"] == 3

    def test_build_prerequisite_graph(self, chain_repo: KnowledgeRepository) -> None:
        assert chain_repo.build_prerequisite_graph("zz") == {
            "error": "Node zz not found"
        }
        graph = chain_repo.build_prerequisite_graph("c")
        assert graph["depth"] == 2
        assert graph["cycles"] == []
        assert [(n["id"], n["depth"]) for n in graph["nodes"]] == [
            ("c", 0),
            ("b", 1),
            ("a", 2),
        ]

    def test_build_prerequisite_graph_cycles_and_revisits(self, tmp_path: Path) -> None:
        nodes = [
            _node("x", prerequisites=["y"]),
            _node("y", prerequisites=["x"]),
            _node("base"),
            _node("l", prerequisites=["base"]),
            _node("top", prerequisites=["l", "base"]),
        ]
        _write_tree(tmp_path, nodes)
        repo = _repo(tmp_path)
        cyc = repo.build_prerequisite_graph("x")
        assert cyc["cycles"] == [["x", "y", "x"]]
        top = repo.build_prerequisite_graph("top")
        assert [n["id"] for n in top["nodes"]] == ["top", "l", "base"]
        assert top["cycles"] == []

    def test_validate_content_integrity(self, tmp_path: Path) -> None:
        nodes = [
            _node("full"),
            _node("empty", content={}, objectives=[], tags=[]),
            _node("needs", prerequisites=["absent"]),
            _node("x", prerequisites=["y"]),
            _node("y", prerequisites=["x"]),
        ]
        _write_tree(tmp_path, nodes)
        repo = _repo(tmp_path)
        assert repo.validate_content_integrity("nope") == {
            "valid": False,
            "error": "Node nope not found",
        }
        assert repo.validate_content_integrity("full") == {
            "valid": True,
            "issues": [],
            "warnings": [],
            "suggestions": [],
        }
        empty = repo.validate_content_integrity("empty")
        assert empty["valid"] is False
        assert empty["issues"] == ["Empty content"]
        assert empty["warnings"] == ["No learning objectives defined"]
        assert empty["suggestions"] == [
            "Consider adding tags for better discoverability"
        ]
        needs = repo.validate_content_integrity("needs")
        assert needs["issues"] == ["Missing prerequisite: absent"]
        # A missing prerequisite is an edge to an absent node, not a crash.
        needs_graph = repo.build_prerequisite_graph("needs")
        assert [n["id"] for n in needs_graph["nodes"]] == ["needs"]
        assert needs_graph["edges"] == [
            {"source": "absent", "target": "needs", "type": "prerequisite"}
        ]
        cyc = repo.validate_content_integrity("x")
        assert cyc["valid"] is False
        assert cyc["issues"][0].startswith("Circular dependencies detected")

    def test_recommendations(self, chain_repo: KnowledgeRepository) -> None:
        recs = chain_repo.generate_learning_path_recommendations(
            {"interests": ["bayes"], "background": []}
        )
        # Default difficulty is intermediate; p_int covers 'bayes' tag via c.
        assert recs == ["p_int", "p_int2"]
        recs2 = chain_repo.generate_learning_path_recommendations(
            {"difficulty": "intermediate", "interests": [], "background": ["a"]}
        )
        assert recs2 == ["p_int2", "p_int"]
        assert (
            chain_repo.generate_learning_path_recommendations({"difficulty": "expert"})
            == []
        )


class TestExports:
    def test_json(self, chain_repo: KnowledgeRepository) -> None:
        data = json.loads(
            chain_repo.export_knowledge_in_format(
                "json", {"content_types": ["foundation", "mathematics"]}
            )
        )
        assert sorted(n["id"] for n in data["nodes"]) == ["a", "b"]
        assert data["metadata"]["total_nodes"] == 2
        assert data["metadata"]["format"] == "json"

    def test_xml(self, chain_repo: KnowledgeRepository) -> None:
        xml = chain_repo.export_knowledge_in_format(
            "xml", {"difficulties": ["beginner"]}
        )
        lines = xml.splitlines()
        assert lines[0] == '<?xml version="1.0" encoding="UTF-8"?>'
        assert '  <node id="a">' in lines
        assert "      <tag>info</tag>" in lines
        assert '<node id="b">' not in xml
        assert lines[-1] == "</knowledge_base>"

    def test_rdf(self, chain_repo: KnowledgeRepository) -> None:
        rdf = chain_repo.export_knowledge_in_format("rdf", {"tags": ["bayes"]})
        assert "kb:c a kb:KnowledgeNode ;" in rdf
        assert "  kb:prerequisites ['b'] ;" in rdf
        assert "kb:a a" not in rdf

    def test_csv(self, chain_repo: KnowledgeRepository) -> None:
        csv = chain_repo.export_knowledge_in_format("csv", {"tags": ["entropy"]})
        lines = csv.splitlines()
        assert lines[0] == (
            "id,title,content_type,difficulty,description,tags,prerequisites"
        )
        assert sorted(lines[1:]) == [
            '"a","Title a","foundation","beginner","Description of a",'
            '"entropy;info",""',
            '"b","Title b","mathematics","intermediate","Description of b",'
            '"entropy","a"',
        ]

    def test_unsupported_format(self, chain_repo: KnowledgeRepository) -> None:
        with pytest.raises(ValueError, match="Unsupported export format: yaml"):
            chain_repo.export_knowledge_in_format("yaml", {})


class TestBackupAndStats:
    def test_create_content_backup(
        self,
        chain_repo: KnowledgeRepository,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        backup_dir = tmp_path / "backups"
        backup_dir.mkdir()
        monkeypatch.setattr(tempfile, "tempdir", str(backup_dir))
        path = chain_repo.create_content_backup()
        assert path.parent == backup_dir
        data = json.loads(path.read_text())
        assert data["metadata"]["total_nodes"] == 3
        assert data["metadata"]["version"] == "1.0"
        assert set(data["nodes"]) == {"a", "b", "c"}
        assert data["relationships"]["a"] == {
            "prerequisites": [],
            "dependents": ["b"],
        }
        assert data["relationships"]["c"]["dependents"] == []

    def test_extract_relationships_ignores_unknown_prereq(self, tmp_path: Path) -> None:
        _write_tree(tmp_path, [_node("n", prerequisites=["ghost"])])
        rel = _repo(tmp_path)._extract_relationships()
        assert rel == {"n": {"prerequisites": ["ghost"], "dependents": []}}

    def test_statistics(self, chain_repo: KnowledgeRepository) -> None:
        stats = chain_repo.get_statistics()
        assert stats["total_nodes"] == 3
        assert stats["total_paths"] == 3
        assert stats["content_types"] == {
            "foundation": 1,
            "mathematics": 1,
            "implementation": 1,
        }
        assert stats["difficulties"] == {
            "beginner": 1,
            "intermediate": 1,
            "advanced": 1,
        }
        assert stats["top_tags"][0] == ("entropy", 2)
        assert stats["total_tags"] == 3
        assert stats["metadata"]["version"] == "9.9"
