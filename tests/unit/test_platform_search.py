"""
Tests for Platform Search Module

Unit tests for the platform search engine, ensuring proper operation of
search functionality, indexing, ranking, and analytics.
"""

import pytest
import json
import time
from unittest.mock import Mock, patch
from pathlib import Path

from active_inference.platform.search import SearchEngine, SearchResult, QueryProcessor


class TestSearchResult:
    """Test cases for SearchResult dataclass"""

    def test_search_result_creation(self):
        """Test creating search results"""
        result = SearchResult(
            node_id="test_node",
            title="Test Node",
            content_type="foundation",
            score=0.85,
            highlights=["This is a test highlight"],
            metadata={"difficulty": "beginner", "tags": ["test"]}
        )

        assert result.node_id == "test_node"
        assert result.title == "Test Node"
        assert result.content_type == "foundation"
        assert result.score == 0.85
        assert result.highlights == ["This is a test highlight"]
        assert result.metadata["difficulty"] == "beginner"

    def test_search_result_post_init(self):
        """Test SearchResult post-initialization"""
        result = SearchResult(
            node_id="test_node",
            title="Test Node",
            content_type="foundation",
            score=0.75
        )

        # Metadata and highlights should be initialized if None
        assert result.metadata == {}
        assert result.highlights == []


class TestQueryProcessor:
    """Test cases for QueryProcessor class"""

    @pytest.fixture
    def query_processor(self):
        """Create QueryProcessor instance for testing"""
        config = {'stop_words': {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}}
        return QueryProcessor(config)

    def test_query_processor_initialization(self, query_processor):
        """Test QueryProcessor initialization"""
        assert hasattr(query_processor, 'stop_words')
        assert isinstance(query_processor.stop_words, set)
        assert len(query_processor.stop_words) > 0

    def test_preprocess_query(self, query_processor):
        """Test query preprocessing"""
        query = "What is the entropy in information theory?"
        processed = query_processor.preprocess_query(query)

        assert isinstance(processed, list)
        assert len(processed) > 0
        # Should remove stop words
        assert "the" not in processed
        assert "in" not in processed

    def test_preprocess_empty_query(self, query_processor):
        """Test preprocessing empty query"""
        processed = query_processor.preprocess_query("")
        assert processed == []

    def test_extract_intent(self, query_processor):
        """Test intent extraction from query"""
        query = "explain entropy information theory"
        intent = query_processor.extract_intent(query)

        assert isinstance(intent, dict)
        assert "primary_intent" in intent
        assert "entities" in intent
        assert "complexity" in intent

    def test_extract_intent_simple_query(self, query_processor):
        """Test intent extraction from simple query"""
        intent = query_processor.extract_intent("entropy")

        assert intent["primary_intent"] == "concept_lookup"
        assert "entropy" in intent["entities"]


class TestSearchEngine:
    """Test cases for SearchEngine class"""

    @pytest.fixture
    def search_engine(self):
        """Create SearchEngine instance for testing"""
        config = {
            "index_backend": "memory",
            "cache_enabled": False,
            "max_results": 50
        }
        return SearchEngine(config)

    def test_search_engine_initialization(self, search_engine):
        """Test SearchEngine initialization"""
        assert hasattr(search_engine, 'index_manager')
        assert hasattr(search_engine, 'query_processor')
        assert hasattr(search_engine, 'ranker')

    def test_add_to_index(self, search_engine):
        """Test adding content to search index"""
        test_content = {
            "id": "test_node_1",
            "title": "Test Node 1",
            "description": "This is a test node for search functionality",
            "content_type": "foundation",
            "difficulty": "beginner",
            "tags": ["test", "search", "foundation"]
        }

        success = search_engine.index_manager.add_to_index("test_node_1", test_content)

        assert success is True
        assert "test_node_1" in search_engine.index_manager.node_index
        assert "test" in search_engine.index_manager.inverted_index

    def test_search_basic(self, search_engine):
        """Test basic search functionality"""
        # Add test content
        test_content = {
            "id": "entropy_basics",
            "title": "Entropy Basics",
            "description": "Introduction to entropy in information theory",
            "content_type": "foundation",
            "difficulty": "beginner",
            "tags": ["entropy", "information theory"]
        }

        search_engine.index_manager.add_to_index("entropy_basics", test_content)

        # Search for entropy
        results = search_engine.search("entropy", limit=10)

        assert isinstance(results, list)
        assert len(results) > 0

        # Check result structure
        result = results[0]
        assert isinstance(result, SearchResult)
        assert result.node_id == "entropy_basics"
        assert result.relevance_score > 0

    def test_search_with_filters(self, search_engine):
        """Test search with content type filters"""
        # Add multiple test nodes
        test_nodes = [
            {
                "id": "entropy_foundation",
                "title": "Entropy Foundation",
                "description": "Entropy in information theory foundations",
                "content_type": "foundation",
                "difficulty": "beginner",
                "tags": ["entropy", "information theory"]
            },
            {
                "id": "entropy_math",
                "title": "Entropy Mathematics",
                "description": "Mathematical treatment of entropy",
                "content_type": "mathematics",
                "difficulty": "advanced",
                "tags": ["entropy", "mathematics"]
            }
        ]

        for node in test_nodes:
            search_engine.index_manager.add_to_index(node["id"], node)

        # Search with foundation filter
        results = search_engine.search("entropy", content_types=["foundation"], limit=10)

        assert len(results) == 1
        assert results[0].content_type == "foundation"

        # Search with mathematics filter
        results = search_engine.search("entropy", content_types=["mathematics"], limit=10)

        assert len(results) == 1
        assert results[0].content_type == "mathematics"

    def test_search_no_results(self, search_engine):
        """Test search with no matching results"""
        results = search_engine.search("nonexistent_term", limit=10)

        assert results == []

    def test_search_ranking(self, search_engine):
        """Test search result ranking"""
        # Add nodes with different relevance
        test_nodes = [
            {
                "id": "entropy_main",
                "title": "Entropy in Information Theory",
                "description": "Comprehensive introduction to entropy",
                "content_type": "foundation",
                "tags": ["entropy", "information theory", "core concept"]
            },
            {
                "id": "entropy_example",
                "title": "Entropy Example",
                "description": "Simple example of entropy calculation",
                "content_type": "implementation",
                "tags": ["entropy", "example"]
            }
        ]

        for node in test_nodes:
            search_engine.index_manager.add_to_index(node["id"], node)

        results = search_engine.search("entropy information theory", limit=10)

        assert len(results) >= 1
        # Results should be sorted by relevance (highest first)
        if len(results) > 1:
            assert results[0].relevance_score >= results[1].relevance_score

    def test_advanced_search(self, search_engine):
        """Test advanced search functionality"""
        # Add test content
        test_content = {
            "id": "test_advanced",
            "title": "Advanced Test Node",
            "description": "This node contains advanced content for testing",
            "content_type": "mathematics",
            "difficulty": "advanced",
            "tags": ["advanced", "mathematics", "test"]
        }

        search_engine.index_manager.add_to_index("test_advanced", test_content)

        # Advanced search with filters
        filters = {
            "content_types": ["mathematics"],
            "difficulties": ["advanced"]
        }

        results = search_engine.advanced_search("advanced content", filters=filters, limit=10)

        assert isinstance(results, list)
        # Should return matching results
        assert len(results) > 0

    def test_search_similar_content(self, search_engine):
        """Test finding similar content"""
        # Add related nodes
        nodes = [
            {
                "id": "entropy_1",
                "title": "Entropy Basics",
                "description": "Introduction to entropy",
                "content_type": "foundation",
                "tags": ["entropy", "information theory"]
            },
            {
                "id": "entropy_2",
                "title": "Advanced Entropy",
                "description": "Advanced entropy concepts",
                "content_type": "mathematics",
                "tags": ["entropy", "mathematics"]
            },
            {
                "id": "unrelated",
                "title": "Unrelated Node",
                "description": "This is unrelated content",
                "content_type": "implementation",
                "tags": ["unrelated", "implementation"]
            }
        ]

        for node in nodes:
            search_engine.index_manager.add_to_index(node["id"], node)

        # Find similar content to entropy_1
        similar = search_engine.search_similar_content("entropy_1", limit=5)

        assert isinstance(similar, list)
        assert len(similar) > 0

        # Should find entropy_2 as similar (both have entropy tag)
        similar_ids = [result.node_id for result in similar]
        assert "entropy_2" in similar_ids

        # Should not include unrelated node
        assert "unrelated" not in similar_ids

    def test_search_analytics(self, search_engine):
        """Test search analytics functionality"""
        # Add some test content
        test_content = {
            "id": "analytics_test",
            "title": "Analytics Test Node",
            "description": "Test node for analytics",
            "content_type": "foundation",
            "difficulty": "beginner",
            "tags": ["analytics", "test"]
        }

        search_engine.index_manager.add_to_index("analytics_test", test_content)

        analytics = search_engine.get_search_analytics()

        assert isinstance(analytics, dict)
        assert "total_indexed_nodes" in analytics
        assert "total_search_terms" in analytics
        assert "search_performance" in analytics
        assert "popular_queries" in analytics
        assert "search_trends" in analytics

        assert analytics["total_indexed_nodes"] == 1

    def test_popular_searches(self, search_engine):
        """Test popular searches functionality"""
        popular = search_engine.get_popular_searches(limit=5)

        assert isinstance(popular, list)
        assert len(popular) <= 5

        for item in popular:
            assert "query" in item
            assert "count" in item
            assert isinstance(item["count"], int)

    def test_faceted_search_options(self, search_engine):
        """Test faceted search options"""
        # Add nodes with different facets
        test_nodes = [
            {
                "id": "node1",
                "title": "Node 1",
                "content_type": "foundation",
                "difficulty": "beginner",
                "tags": ["tag1", "tag2"]
            },
            {
                "id": "node2",
                "title": "Node 2",
                "content_type": "mathematics",
                "difficulty": "advanced",
                "tags": ["tag2", "tag3"]
            }
        ]

        for node in test_nodes:
            search_engine.index_manager.add_to_index(node["id"], node)

        facets = search_engine.get_faceted_search_options()

        assert isinstance(facets, dict)
        assert "content_types" in facets
        assert "difficulties" in facets
        assert "tags" in facets

        # Check content types
        assert "foundation" in facets["content_types"]
        assert "mathematics" in facets["content_types"]

        # Check difficulties
        assert "beginner" in facets["difficulties"]
        assert "advanced" in facets["difficulties"]

        # Check tags
        assert len(facets["tags"]) > 0

    def test_reindex_functionality(self, search_engine):
        """Test reindexing functionality"""
        # Add some content
        test_content = {
            "id": "reindex_test",
            "title": "Reindex Test",
            "description": "Content for reindexing test"
        }

        search_engine.index_manager.add_to_index("reindex_test", test_content)

        # Reindex
        result = search_engine.reindex_all_content()

        assert isinstance(result, dict)
        assert result["success"] is True
        assert result["nodes_indexed"] > 0

    def test_search_optimization(self, search_engine):
        """Test search index optimization"""
        # Add content
        test_content = {
            "id": "optimization_test",
            "title": "Optimization Test",
            "description": "Content for optimization testing"
        }

        search_engine.index_manager.add_to_index("optimization_test", test_content)

        # Optimize indices
        result = search_engine.optimize_search_indices()

        assert isinstance(result, dict)
        assert result["success"] is True
        assert "unused_terms_removed" in result

    def test_search_validation(self, search_engine):
        """Test search functionality validation"""
        validation = search_engine.validate_search_functionality()

        assert isinstance(validation, dict)
        assert "valid" in validation
        assert "issues" in validation
        assert "warnings" in validation
        assert "recommendations" in validation

    def test_search_export_import(self, search_engine):
        """Test search data export and import"""
        # Add test content
        test_content = {
            "id": "export_test",
            "title": "Export Test Node",
            "description": "Content for export testing"
        }

        search_engine.index_manager.add_to_index("export_test", test_content)

        # Export data
        export_data = search_engine.export_search_data("json")
        assert isinstance(export_data, str)

        # Parse to validate JSON
        parsed_data = json.loads(export_data)
        assert "node_index" in parsed_data
        assert "inverted_index" in parsed_data
        assert "metadata" in parsed_data

        # Create new search engine for import test
        new_search_engine = SearchEngine({"index_backend": "memory"})

        # Import data
        success = new_search_engine.import_search_data(parsed_data)
        assert success is True
        assert "export_test" in new_search_engine.index_manager.node_index

    def test_search_suggestions(self, search_engine):
        """Test search suggestions functionality"""
        # Add content with searchable terms
        test_content = {
            "id": "suggestion_test",
            "title": "Information Theory Basics",
            "description": "Introduction to information theory concepts"
        }

        search_engine.index_manager.add_to_index("suggestion_test", test_content)

        # Get suggestions
        suggestions = search_engine.get_search_suggestions("info")

        assert isinstance(suggestions, list)
        assert len(suggestions) > 0

    def test_search_performance_metrics(self, search_engine):
        """Test search performance metrics"""
        analytics = search_engine.get_search_analytics()

        performance = analytics["search_performance"]

        assert isinstance(performance, dict)
        assert "avg_response_time" in performance
        assert "max_response_time" in performance
        assert "min_response_time" in performance

        # Performance metrics should be reasonable
        assert performance["avg_response_time"] > 0
        assert performance["max_response_time"] >= performance["min_response_time"]

    def test_search_result_ranking(self, search_engine):
        """Test search result ranking and sorting"""
        # Add nodes with different relevance scores
        test_nodes = [
            {
                "id": "high_relevance",
                "title": "High Relevance Node",
                "description": "This node should have high relevance for test queries",
                "content_type": "foundation",
                "tags": ["high", "relevance", "test"]
            },
            {
                "id": "low_relevance",
                "title": "Low Relevance Node",
                "description": "This node has lower relevance",
                "content_type": "implementation",
                "tags": ["low", "relevance"]
            }
        ]

        for node in test_nodes:
            search_engine.index_manager.add_to_index(node["id"], node)

        # Search and check ranking
        results = search_engine.search("high relevance test", limit=10)

        if len(results) > 1:
            # Higher relevance should come first
            assert results[0].node_id == "high_relevance"
            assert results[0].relevance_score >= results[1].relevance_score


if __name__ == "__main__":
    pytest.main([__file__])


