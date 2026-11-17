"""Behavioural tests for active_inference.platform.knowledge_graph."""

import json

import pytest

from active_inference.platform.knowledge_graph import (
    KnowledgeEdge,
    KnowledgeGraphManager,
    KnowledgeNode,
    NodeType,
    Platform,
    RelationshipType,
    SemanticEngine,
)


def node(node_id, title=None, node_type=NodeType.CONCEPT, **props):
    return KnowledgeNode(
        id=node_id,
        title=title or node_id.title(),
        node_type=node_type,
        properties=props,
    )


def edge(src, dst, rel=RelationshipType.RELATED_TO, weight=1.0):
    return KnowledgeEdge(source=src, target=dst, relationship_type=rel, weight=weight)


@pytest.fixture
def chain():
    """a - b - c - d chain plus isolated e."""
    kg = KnowledgeGraphManager({})
    for nid in "abcde":
        kg.add_node(node(nid))
    kg.add_edge(edge("a", "b", RelationshipType.PREREQUISITE))
    kg.add_edge(edge("b", "c"))
    kg.add_edge(edge("d", "c", RelationshipType.EXTENDS, weight=0.5))
    return kg


class TestDataclasses:
    def test_node_identity_by_id(self):
        n1 = node("x", "One")
        n2 = node("x", "Two", NodeType.THEORY)
        assert n1 == n2
        assert hash(n1) == hash(n2)
        assert n1 != node("y")
        assert n1 != "x"
        assert len({n1, n2}) == 1

    def test_edge_identity(self):
        e1 = edge("a", "b", weight=0.1)
        e2 = edge("a", "b", weight=0.9)
        assert e1 == e2
        assert hash(e1) == hash(e2)
        assert e1 != edge("b", "a")
        assert e1 != edge("a", "b", RelationshipType.CITES)
        assert e1 != ("a", "b")


class TestKnowledgeGraphManager:
    def test_storage_backend_config(self):
        assert KnowledgeGraphManager({}).storage_backend == "memory"
        assert KnowledgeGraphManager({"storage_backend": "db"}).storage_backend == "db"

    def test_add_node_indexes_by_type(self):
        kg = KnowledgeGraphManager({})
        assert kg.add_node(node("c1")) is True
        assert kg.add_node(node("t1", node_type=NodeType.THEORY)) is True
        assert kg.node_index == {"concept": {"c1"}, "theory": {"t1"}}
        assert kg.get_node("c1").title == "C1"
        assert kg.get_node("missing") is None

    def test_add_existing_node_updates(self):
        kg = KnowledgeGraphManager({})
        kg.add_node(node("c1", "Old"))
        assert kg.add_node(node("c1", "New")) is True
        assert kg.get_node("c1").title == "New"
        assert len(kg.nodes) == 1

    def test_update_missing_node(self):
        assert KnowledgeGraphManager({}).update_node(node("x")) is False

    def test_invalid_inputs_return_false(self, caplog):
        kg = KnowledgeGraphManager({})
        bad_type = KnowledgeNode(id="b", title="B", node_type="concept")
        assert kg.add_node(bad_type) is False
        assert kg.update_node(KnowledgeNode(["u"], "U", NodeType.CONCEPT)) is False
        assert kg.delete_node(["u"]) is False
        errors = [r.getMessage() for r in caplog.records if r.levelname == "ERROR"]
        assert any("Error adding node b" in m for m in errors)
        assert any("Error updating node" in m for m in errors)
        assert any("Error deleting node" in m for m in errors)

    def test_add_edge_validation_and_failure(self, chain):
        assert chain.add_edge(edge("a", "zzz")) is False
        assert chain.add_edge(edge("zzz", "a")) is False
        bad = KnowledgeEdge(source="a", target="e", relationship_type="related")
        assert chain.add_edge(bad) is False
        assert len(chain.edges) == 3
        assert "a_prerequisite_b" in chain.edges

    def test_delete_node_removes_edges_and_index(self, chain):
        assert chain.delete_node("missing") is False
        assert chain.delete_node("c") is True
        assert "c" not in chain.nodes
        assert "c" not in chain.node_index["concept"]
        assert list(chain.edges) == ["a_prerequisite_b"]

    def test_delete_node_tolerates_missing_index_bucket(self, chain):
        chain.node_index.clear()
        assert chain.delete_node("e") is True
        assert "e" not in chain.nodes

    def test_get_nodes_by_type_skips_stale_ids(self):
        kg = KnowledgeGraphManager({})
        kg.add_node(node("m", node_type=NodeType.METHOD))
        kg.node_index["method"].add("ghost")
        assert kg.get_nodes_by_type(NodeType.METHOD) == [kg.nodes["m"]]
        assert kg.get_nodes_by_type(NodeType.PERSON) == []

    def test_get_related_nodes(self, chain):
        related = chain.get_related_nodes("c")
        summary = sorted(
            (r["node"].id, r["relationship"], r["direction"]) for r in related
        )
        assert summary == [
            ("b", "related_to", "incoming"),
            ("d", "extends", "incoming"),
        ]
        out = chain.get_related_nodes("a")
        assert out[0]["direction"] == "outgoing"
        assert out[0]["weight"] == 1.0

        filtered = chain.get_related_nodes("c", RelationshipType.EXTENDS)
        assert [(r["node"].id, r["weight"]) for r in filtered] == [("d", 0.5)]

        # dangling edge to a removed node is skipped
        del chain.nodes["b"]
        assert [r["node"].id for r in chain.get_related_nodes("c")] == ["d"]

    def test_get_neighbors_is_undirected_and_unique(self, chain):
        assert chain.get_neighbors("c") == ["b", "d"]
        assert chain.get_neighbors("a") == ["b"]
        assert chain.get_neighbors("e") == []
        chain.add_edge(edge("b", "a", RelationshipType.CITES))
        assert chain.get_neighbors("a") == ["b"]

    def test_find_shortest_path(self, chain):
        assert chain.find_shortest_path("a", "d") == ["a", "b", "c", "d"]
        assert chain.find_shortest_path("d", "a") == ["d", "c", "b", "a"]
        assert chain.find_shortest_path("a", "a") == ["a"]
        assert chain.find_shortest_path("a", "e") is None
        assert chain.find_shortest_path("a", "missing") is None
        assert chain.find_shortest_path("missing", "a") is None

    def test_find_shortest_path_prefers_shortcut(self, chain):
        chain.add_edge(edge("a", "d"))
        assert chain.find_shortest_path("a", "d") == ["a", "d"]
        # a cycle (a-b-c-d-a) must not loop forever
        assert chain.find_shortest_path("b", "e") is None

    def test_search_nodes(self):
        kg = KnowledgeGraphManager({})
        kg.add_node(node("fep", "Free Energy Principle"))
        kg.add_node(node("vfe", "Variational", NodeType.METHOD, topic="free energy"))
        kg.add_node(node("energy_id", "Unrelated", NodeType.THEORY))
        kg.add_node(node("other", "Other"))

        assert {n.id for n in kg.search_nodes("ENERGY")} == {"fep", "vfe", "energy_id"}
        assert [n.id for n in kg.search_nodes("energy", NodeType.METHOD)] == ["vfe"]
        assert len(kg.search_nodes("energy", limit=2)) == 2
        assert kg.search_nodes("nomatch") == []

    def test_statistics(self, chain):
        stats = chain.get_statistics()
        assert stats == {
            "total_nodes": 5,
            "total_edges": 3,
            "node_types": {"concept": 5},
            "relationship_types": {
                "prerequisite": 1,
                "related_to": 1,
                "extends": 1,
            },
            "average_degree": 1.2,
        }
        assert KnowledgeGraphManager({}).get_statistics()["average_degree"] == 0

    def test_export_graph_json(self, chain):
        chain.nodes["a"].content = {"body": "text"}
        data = json.loads(chain.export_graph("JSON"))
        assert [n["id"] for n in data["nodes"]] == list("abcde")
        assert data["nodes"][0]["content"] == {"body": "text"}
        assert data["nodes"][0]["type"] == "concept"
        assert {(e["source"], e["target"], e["type"]) for e in data["edges"]} == {
            ("a", "b", "prerequisite"),
            ("b", "c", "related_to"),
            ("d", "c", "extends"),
        }

    def test_export_graph_unsupported(self, chain):
        with pytest.raises(ValueError, match="Unsupported export format: xml"):
            chain.export_graph("xml")


class TestSemanticEngine:
    def test_without_graph(self):
        se = SemanticEngine({})
        assert se.compute_semantic_similarity("a", "b") == 0.0
        assert se.find_semantically_related_nodes("a") == []

    def test_similarity_decays_with_distance(self, chain):
        se = SemanticEngine({})
        se.set_knowledge_graph(chain)
        assert se.compute_semantic_similarity("a", "a") == 1.0
        assert se.compute_semantic_similarity("a", "b") == pytest.approx(0.8)
        assert se.compute_semantic_similarity("a", "d") == pytest.approx(0.4)
        assert se.compute_semantic_similarity("a", "e") == 0.0

    def test_similarity_floor_is_zero(self):
        kg = KnowledgeGraphManager({})
        ids = [f"n{i}" for i in range(8)]
        for nid in ids:
            kg.add_node(node(nid))
        for src, dst in zip(ids, ids[1:]):
            kg.add_edge(edge(src, dst))
        se = SemanticEngine({})
        se.set_knowledge_graph(kg)
        assert se.compute_semantic_similarity("n0", "n7") == 0.0

    def test_find_related_sorted_and_thresholded(self, chain):
        se = SemanticEngine({})
        se.set_knowledge_graph(chain)
        related = se.find_semantically_related_nodes("a")
        assert [(r["node_id"], round(r["similarity"], 2)) for r in related] == [
            ("b", 0.8),
            ("c", 0.6),
            ("d", 0.4),
        ]
        assert related[0]["node"] is chain.nodes["b"]
        strict = se.find_semantically_related_nodes("a", threshold=0.7)
        assert [r["node_id"] for r in strict] == ["b"]


class TestPlatform:
    def test_services_wired(self):
        platform = Platform({"knowledge_graph": {"storage_backend": "x"}})
        kg = platform.get_knowledge_graph()
        assert kg.storage_backend == "x"
        assert platform.get_service("knowledge_graph") is kg
        assert platform.get_service("semantic_engine") is platform.get_semantic_engine()
        assert platform.get_semantic_engine().knowledge_graph is kg
        assert platform.get_service("nope") is None

    def test_semantic_similarity_engine_reuse_and_creation(self):
        platform = Platform({"semantic_engine": {"k": 2}})
        existing = platform.get_semantic_engine()
        assert platform.implement_semantic_similarity_engine() is existing

        platform.semantic_engine = None
        created = platform.implement_semantic_similarity_engine()
        assert created is not existing
        assert isinstance(created, SemanticEngine)
        assert created.config == {"k": 2}
        assert created.knowledge_graph is platform.knowledge_graph
        assert platform.get_semantic_engine() is created

    @pytest.mark.parametrize(
        "method, fragment",
        [
            ("implement_graph_storage_engine", "Graph storage engine"),
            ("create_graph_validation_system", "Graph validation system"),
            ("implement_graph_query_language", "Graph query language"),
            ("create_graph_visualization_integration", "Graph visualization"),
            ("implement_graph_backup_and_recovery", "Graph backup and recovery"),
            ("create_performance_monitoring", "Graph performance monitoring"),
            ("implement_graph_security_model", "Graph security model"),
            ("create_graph_analytics_engine", "Graph analytics engine"),
        ],
    )
    def test_placeholders_return_none_and_log(self, caplog, method, fragment):
        platform = Platform({})
        with caplog.at_level("INFO"):
            assert getattr(platform, method)() is None
        assert any(
            fragment in r.getMessage() and "placeholder" in r.getMessage()
            for r in caplog.records
        )


class TestTraversalAlgorithms:
    @pytest.fixture
    def algos(self, chain):
        platform = Platform({})
        platform.knowledge_graph = chain
        return platform.create_graph_traversal_algorithms()

    def test_keys(self, algos):
        assert set(algos) == {"dfs", "bfs", "shortest_path"}

    def test_dfs(self, algos, chain):
        chain.add_node(node("f"))
        chain.add_edge(edge("b", "f"))
        assert algos["dfs"]("a") == ["a", "b", "c", "d", "f"]
        assert algos["dfs"]("a", max_depth=2) == ["a", "b"]
        assert algos["dfs"]("e") == ["e"]
        # unknown start node is still visited but has no neighbours
        assert algos["dfs"]("ghost") == ["ghost"]

    def test_bfs(self, algos, chain):
        chain.add_node(node("f"))
        chain.add_edge(edge("b", "f"))
        assert algos["bfs"]("a") == ["a", "b", "c", "f", "d"]
        assert algos["bfs"]("a", max_depth=3) == ["a", "b", "c", "f"]
        assert algos["bfs"]("ghost") == ["ghost"]

    def test_bfs_skips_already_queued_duplicates(self, algos, chain):
        # a-b, a-c, b-c: c gets queued twice before it is visited
        chain.add_edge(edge("a", "c"))
        assert algos["bfs"]("a") == ["a", "b", "c", "d"]

    def test_shortest_path(self, algos, chain):
        assert algos["shortest_path"]("a", "d") == ["a", "b", "c", "d"]
        assert algos["shortest_path"]("a", "a") == ["a"]
        assert algos["shortest_path"]("a", "e") == []
        assert algos["shortest_path"]("a", "ghost") == []
        assert algos["shortest_path"]("ghost", "a") == []
        chain.add_edge(edge("a", "c"))
        assert algos["shortest_path"]("a", "d") == ["a", "c", "d"]
