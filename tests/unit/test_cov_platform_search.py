"""Behavioural tests for the implemented API of active_inference.platform.search.

These cover the current SearchEngine/IndexManager/QueryProcessor/ResultRanker
surface only; the planned (not yet implemented) API is tracked by xfails in
tests/unit/test_platform_search.py.
"""

from types import SimpleNamespace

import pytest

from active_inference.platform import search as search_mod
from active_inference.platform.search import (
    IndexManager,
    QueryProcessor,
    ResultRanker,
    SearchEngine,
    SearchResult,
    _flatten_text,
)


def doc(node_id, title, **extra):
    content = {"id": node_id, "title": title}
    content.update(extra)
    return content


@pytest.fixture
def engine():
    eng = SearchEngine({})
    eng.index_content(
        doc(
            "fep",
            "Free Energy Principle",
            description="variational inference basis",
            content_type="foundation",
            difficulty="beginner",
            tags=["energy", "theory"],
            content={"overview": "minimising free energy", "points": ["entropy", 3]},
            last_updated="2026-01-02",
        )
    )
    eng.index_content(
        doc(
            "ai",
            "Active Inference",
            description="free energy in action",
            content_type="implementation",
            difficulty="advanced",
            tags=["action"],
            last_updated="2026-03-01",
        )
    )
    eng.index_content(
        doc(
            "bayes",
            "Bayesian Brain",
            description="probabilistic perception",
            content_type="foundation",
            difficulty="beginner",
            tags=["perception"],
            last_updated="2025-12-31",
        )
    )
    return eng


class TestSearchResult:
    def test_to_dict_and_defaults(self):
        r = SearchResult("n", "T", "concept", 0.5)
        assert r.to_dict() == {
            "node_id": "n",
            "title": "T",
            "content_type": "concept",
            "score": 0.5,
            "highlights": [],
            "metadata": {},
        }
        r2 = SearchResult("n", "T", "c", 1.0, ["h"], {"k": 1})
        assert r2.to_dict()["highlights"] == ["h"]
        assert r2.to_dict()["metadata"] == {"k": 1}


class TestQueryProcessor:
    def test_preprocess_lowercases_strips_stopwords_dedupes(self):
        qp = QueryProcessor({})
        assert qp.preprocess_query("The Free-Energy, and the FREE energy!") == [
            "free",
            "energy",
        ]
        assert qp.preprocess_query("the and of") == []

    def test_custom_stop_words(self):
        qp = QueryProcessor({"stop_words": {"energy"}})
        assert qp.preprocess_query("the energy") == ["the"]

    @pytest.mark.parametrize(
        "query, intent",
        [
            ("active inference tutorial", "learning"),
            ("research paper on entropy", "research"),
            ("code example", "implementation"),
            ("use case", "application"),
            ("free energy", "general_search"),
        ],
    )
    def test_extract_intent(self, query, intent):
        result = QueryProcessor({}).extract_intent(query)
        assert result["primary_intent"] == intent
        assert result["sort_by"] == "relevance"

    def test_extract_intent_filters(self):
        result = QueryProcessor({}).extract_intent(
            "advanced mathematics foundation expert"
        )
        assert result["query_terms"] == [
            "advanced",
            "mathematics",
            "foundation",
            "expert",
        ]
        # first matching entry of each list wins
        assert result["filters"] == {
            "content_type": "foundation",
            "difficulty": "advanced",
        }
        assert QueryProcessor({}).extract_intent("plain")["filters"] == {}

    def test_expand_query(self):
        qp = QueryProcessor({})
        assert sorted(qp.expand_query("Bayesian neural models")) == sorted(
            ["bayesian", "probabilistic", "statistical", "neural", "brain", "cognitive"]
        )
        assert set(qp.expand_query("Active Inference")) == {
            "active inference",
            "free energy principle",
            "predictive processing",
        }
        assert qp.expand_query("the markov blanket") == ["markov", "blanket"]


class TestIndexManager:
    def test_add_builds_reverse_index(self):
        im = IndexManager({})
        content = {
            "title": "Free Energy",
            "description": "a short Description",
            "content": {"body": "Markov blanket", "list": ["Entropy", 7], "n": 1},
            "tags": ["tagged"],
        }
        assert im.add_to_index("n1", content) is True
        assert im.index["n1"] is content
        # tokens of length <= 2 ("a") are dropped
        assert set(im.reverse_index) == {
            "free",
            "energy",
            "short",
            "description",
            "markov",
            "blanket",
            "entropy",
            "tagged",
        }
        assert im.search_index(["FREE", "markov"]) == {"n1"}

    def test_non_dict_content_and_non_list_tags_are_ignored(self):
        im = IndexManager({})
        im.add_to_index("n1", {"title": "Alpha", "content": "Gamma", "tags": "Delta"})
        assert set(im.reverse_index) == {"alpha"}

    def test_add_failure_returns_false(self):
        im = IndexManager({})
        assert im.add_to_index("n1", {"title": 42}) is False
        assert im.reverse_index == {}

    def test_search_index_intersection_and_early_exit(self):
        im = IndexManager({})
        im.add_to_index("n1", {"title": "alpha beta"})
        im.add_to_index("n2", {"title": "alpha gamma"})
        assert im.search_index([]) == set()
        assert im.search_index(["alpha"]) == {"n1", "n2"}
        assert im.search_index(["alpha", "gamma"]) == {"n2"}
        assert im.search_index(["missing", "alpha"]) == set()
        assert im.search_index(["beta", "gamma", "alpha"]) == set()

    def test_remove_from_index(self):
        im = IndexManager({})
        im.add_to_index("n1", {"title": "alpha beta"})
        im.add_to_index("n2", {"title": "alpha"})
        assert im.remove_from_index("n1") is True
        assert "n1" not in im.index
        assert "beta" not in im.reverse_index
        assert im.reverse_index["alpha"] == {"n2"}
        assert im.remove_from_index("n1") is False

    def test_remove_tolerates_tokens_missing_from_reverse_index(self):
        im = IndexManager({})
        im.add_to_index("n1", {"title": "alpha beta"})
        del im.reverse_index["beta"]
        assert im.remove_from_index("n1") is True
        assert im.reverse_index == {}

    def test_remove_failure_returns_false(self):
        im = IndexManager({})
        im.index["bad"] = {"title": 42}
        assert im.remove_from_index("bad") is False
        assert "bad" in im.index

    def test_statistics(self):
        im = IndexManager({})
        assert im.get_index_statistics() == {
            "total_documents": 0,
            "total_terms": 0,
            "average_terms_per_doc": 0,
        }
        im.add_to_index("n1", {"title": "alpha beta"})
        im.add_to_index("n2", {"title": "alpha gamma delta"})
        assert im.get_index_statistics() == {
            "total_documents": 2,
            "total_terms": 4,
            "average_terms_per_doc": 2.0,
        }


class TestFlattenText:
    def test_nested_values(self):
        value = {"a": "one", "b": ["two", ("three", 4)], "c": {"d": "five"}}
        assert _flatten_text(value) == "one two three  five"
        assert _flatten_text(None) == ""


class TestResultRanker:
    def results(self):
        return [
            {
                "node_id": "content_only",
                "title": "Other",
                "content": {"s": "free energy"},
                "tags": [],
                "last_updated": "2026-02-01",
                "usage_count": 5,
            },
            {
                "node_id": "title",
                "title": "Free Energy",
                "content": "",
                "tags": ["FreeForm"],
                "last_updated": "2025-01-01",
                "usage_count": 50,
            },
            {"node_id": "bare"},
        ]

    def test_relevance_scores(self):
        ranked = ResultRanker({}).rank_results(self.results(), ["free", "energy"])
        assert [r["node_id"] for r in ranked] == ["title", "content_only", "bare"]
        # title: 2 title hits (20) + 0 content + "free" in tag (5)
        assert ranked[0]["relevance_score"] == 25.0
        assert ranked[1]["relevance_score"] == 2.0
        assert ranked[2]["relevance_score"] == 0.0

    def test_recency_and_popularity(self):
        ranker = ResultRanker({})
        assert [
            r["node_id"] for r in ranker.rank_results(self.results(), [], "recency")
        ] == [
            "content_only",
            "title",
            "bare",
        ]
        assert [
            r["node_id"] for r in ranker.rank_results(self.results(), [], "popularity")
        ] == ["title", "content_only", "bare"]

    def test_unknown_method_keeps_order(self):
        results = self.results()
        assert ResultRanker({}).rank_results(results, ["x"], "random") is results


class TestSearchEngine:
    def test_index_content_requires_id(self, caplog):
        eng = SearchEngine({})
        assert eng.index_content({"title": "no id"}) is False
        assert eng.index_manager.index == {}
        assert any("without ID" in r.getMessage() for r in caplog.records)

    def test_search_returns_ranked_dicts(self, engine):
        results = engine.search("free energy")
        assert [r["node_id"] for r in results] == ["fep", "ai"]
        top = results[0]
        assert top["title"] == "Free Energy Principle"
        assert top["content_type"] == "foundation"
        assert top["difficulty"] == "beginner"
        assert top["tags"] == ["energy", "theory"]
        assert top["last_updated"] == "2026-01-02"
        assert top["relevance_score"] > results[1]["relevance_score"]

    def test_search_limit_and_ranking_choice(self, engine):
        assert [r["node_id"] for r in engine.search("free energy", limit=1)] == ["fep"]
        recent = engine.search("free energy", ranking="recency")
        assert [r["node_id"] for r in recent] == ["ai", "fep"]

    def test_search_empty_or_stopword_query(self, engine):
        assert engine.search("the of and") == []
        assert engine.search_stats["total_searches"] == 0

    def test_search_skips_reverse_index_entries_without_document(self, engine):
        engine.index_manager.reverse_index["energy"].add("ghost")
        engine.index_manager.reverse_index["free"].add("ghost")
        assert {r["node_id"] for r in engine.search("free energy")} == {"fep", "ai"}

    def test_filters(self, engine):
        by_type = engine.search("free energy", filters={"content_type": "foundation"})
        assert [r["node_id"] for r in by_type] == ["fep"]
        by_difficulty = engine.search("free energy", filters={"difficulty": "advanced"})
        assert [r["node_id"] for r in by_difficulty] == ["ai"]
        by_tags = engine.search("free energy", filters={"tags": ["energy", "theory"]})
        assert [r["node_id"] for r in by_tags] == ["fep"]
        assert engine.search("free energy", filters={"tags": ["missing"]}) == []

    def test_search_error_returns_empty(self, engine, caplog):
        # unhashable tag filter values make the filter step raise TypeError
        assert engine.search("free energy", filters={"tags": [["x"]]}) == []
        assert any("Search error" in r.getMessage() for r in caplog.records)

    def test_search_statistics(self, engine, monkeypatch):
        clock = iter([100.0, 100.5, 200.0, 201.5])
        monkeypatch.setattr(
            search_mod, "time", SimpleNamespace(time=lambda: next(clock))
        )
        engine.search("free energy")
        engine.search("free energy")
        stats = engine.get_search_statistics()
        assert stats["total_searches"] == 2
        assert stats["average_response_time"] == pytest.approx(1.0)
        assert stats["popular_queries"] == {"free energy": 2}
        assert stats["index_stats"]["total_documents"] == 3

    def test_remove_content(self, engine):
        assert engine.remove_content("ai") is True
        assert [r["node_id"] for r in engine.search("free energy")] == ["fep"]
        assert engine.remove_content("ai") is False

    def test_suggestions(self, engine):
        assert engine.get_search_suggestions("e") == []
        assert sorted(engine.get_search_suggestions("en")) == ["energy", "entropy"]
        assert len(engine.get_search_suggestions("pr", limit=1)) == 1
        assert engine.get_search_suggestions("zz") == []
